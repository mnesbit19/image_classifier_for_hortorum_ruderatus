"""We verify the accepted image files and flag exact duplicate candidates offline.

Raw inputs, labels and split assignments are never modified. Full cohorts belong
in a CPU batch job; small local fixtures exercise the same checks.
"""

import argparse
import collections
import csv
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import socket
import sys
import uuid
import warnings

import PIL
from PIL import Image, ImageFile, ImageOps, features
from report_download import analyse

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 25 * 1024**2
MAX_PIXELS = 40_000_000
MAX_FRAMES = 100
MAX_FRAME_PIXELS = 100_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
ImageFile.LOAD_TRUNCATED_IMAGES = False


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def csv_rows(raw):
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def inside(root, relative, prefix):
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or "\\" in relative or parts.parts[:len(prefix)] != prefix:
        raise ValueError("Unexpected input path: " + relative)
    path = (root / relative).resolve()
    if not path.is_relative_to((root / Path(*prefix)).resolve()):
        raise ValueError("Input resolves outside the expected directory: " + relative)
    return path


def load_sources(root, cohorts, output):
    if cohorts.get("schema_version") != 1 or not cohorts.get("sources"):
        raise ValueError("Unsupported or empty cohort configuration.")
    all_rows, source_reports = [], []
    seen_names = set()
    for source in cohorts["sources"]:
        for field in ("name", "dataset", "attempt"):
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,99}", source[field]):
                raise ValueError("Invalid source identifier: " + field)
        if source["name"] in seen_names:
            raise ValueError("Repeated source name.")
        seen_names.add(source["name"])
        relative = f"data/manifests/{source['dataset']}/{source['attempt']}"
        attempt = inside(root, relative, ("data", "manifests"))
        names = ("summary.json", "photos.csv", "excluded.csv", "config.json", "code_snapshot.py")
        raw = {name: (attempt / name).read_bytes() for name in names}
        plan_path = f"data/raw/metadata/{source['dataset']}/plan.json"
        raw["plan.json"] = inside(root, plan_path, ("data", "raw", "metadata")).read_bytes()
        summary, config, plan = (json.loads(raw[name]) for name in ("summary.json", "config.json", "plan.json"))
        if summary["dataset"] != source["dataset"] or summary["attempt"] != source["attempt"]:
            raise ValueError("Attempt identity differs from the accepted cohort.")
        if sha(raw["plan.json"]) != source["plan_sha256"] or summary["plan_sha256"] != source["plan_sha256"]:
            raise ValueError("Accepted source plan hash differs.")
        if sha(raw["code_snapshot.py"]) != source["download_code_sha256"] or summary["code_sha256"] != source["download_code_sha256"]:
            raise ValueError("Accepted download code hash differs.")
        if config["classes"] != cohorts["classes"] or config["classes"] != plan["class_mapping"] or config["data"] != plan["data_policy"]:
            raise ValueError("Source configuration differs from its plan or class mapping.")
        rows = csv_rows(raw["photos.csv"])
        report, _ = analyse(summary, rows, csv_rows(raw["excluded.csv"]), config,
                            source["job_id"], source["accept_budget_limited"])
        if set(r["species"] for r in rows) != {source["species"]}:
            raise ValueError("Unexpected source species.")
        if len(plan["observation_ids"]) != summary["selected_observations"] or not {int(r["observation_id"]) for r in rows} <= set(plan["observation_ids"]):
            raise ValueError("Photo observations differ from the saved plan.")
        for key, expected in source["expected_counts"].items():
            if report[key] != expected:
                raise ValueError(f"Accepted source count differs: {source['name']} {key}")
        saved = output / "sources" / source["name"]
        saved.mkdir(parents=True, exist_ok=False)
        for name, content in raw.items():
            (saved / name).write_bytes(content)
        report["source_sha256"] = {name: sha(content) for name, content in raw.items()}
        source_reports.append(report)
        all_rows.extend({**r, "source_dataset": source["dataset"], "source_attempt": source["attempt"]} for r in rows)
    return all_rows, source_reports


def unique_photos(rows):
    photos, observations = {}, collections.defaultdict(set)
    pairs = set()
    for row in rows:
        pair = (row["observation_id"], row["photo_id"])
        if pair in pairs:
            raise ValueError("Repeated observation/photo link across accepted sources.")
        pairs.add(pair)
        observations[row["observation_id"]].add(row["species"])
        previous = photos.setdefault(row["photo_id"], row)
        if any(previous[k] != row[k] for k in ("file_path", "sha256", "bytes")):
            raise ValueError("Conflicting saved file for a shared photo ID.")
    if any(len(labels) > 1 for labels in observations.values()):
        raise ValueError("One observation has conflicting species labels across cohorts.")
    return photos


def check_photo(root, row):
    result = {"photo_id": row["photo_id"], "file_path": row["file_path"],
              "status": "unchecked", "error": "", "actual_bytes": "", "actual_sha256": "",
              "format": "", "mode": "", "width": "", "height": "", "frames": "",
              "display_width": "", "display_height": "", "pixel_sha256": "", "review_flags": ""}
    try:
        path = inside(root, row["file_path"], ("data", "raw", "images"))
        result["actual_bytes"] = path.stat().st_size
        if result["actual_bytes"] > MAX_BYTES:
            result.update(status="review_limits", error="File exceeds our 25-MiB read limit.")
            return result
        raw = path.read_bytes()
        result["actual_sha256"] = sha(raw)
        if len(raw) != int(row["bytes"]) or result["actual_sha256"] != row["sha256"]:
            result.update(status="integrity_error", error="Saved size/SHA-256 does not match file bytes.")
            return result
    except (OSError, ValueError) as error:
        result.update(status="file_error", error=str(error))
        return result
    flags = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as im:
                im.verify()
            # Pillow.verify does not decode pixels. We reopen and load all frames.
            with Image.open(io.BytesIO(raw)) as im:
                frames = getattr(im, "n_frames", 1)
                result.update(format=im.format, mode=im.mode, width=im.width, height=im.height, frames=frames)
                if frames > MAX_FRAMES:
                    result.update(status="review_limits", error="Frame count exceeds our review limit.")
                    return result
                frame_pixels = 0
                for index in range(frames):
                    im.seek(index)
                    frame_pixels += im.width * im.height
                    if im.width * im.height > MAX_PIXELS or frame_pixels > MAX_FRAME_PIXELS:
                        result.update(status="review_limits", error="Decoded pixel count exceeds our review limit.")
                        return result
                    im.load()
                if frames == 1:
                    # We hash decoded RGBA after EXIF orientation, including dimensions.
                    # ICC colour profiles are not applied; matches remain review candidates.
                    oriented = ImageOps.exif_transpose(im)
                    with oriented.convert("RGBA") as pixels:
                        result.update(display_width=pixels.width, display_height=pixels.height)
                        size_bytes = pixels.width.to_bytes(8, "big") + pixels.height.to_bytes(8, "big")
                        result["pixel_sha256"] = sha(size_bytes + pixels.tobytes())
                        for minimum in (224, 384, 448):
                            if min(pixels.size) < minimum:
                                flags.append(f"whole_image_short_side_below_{minimum}")
                    oriented.close()
                else:
                    flags.append("multiple_frames_need_review")
            flags.extend("decoder_warning:" + str(w.message) for w in caught)
        result.update(status="pass", review_flags=";".join(flags))
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        result.update(status="review_limits", error=str(error))
    except Exception as error:
        result.update(status="decode_error", error=f"{type(error).__name__}: {error}")
    return result


def duplicate_groups(rows, checked):
    groups = []
    for kind, field in (("file_bytes", "actual_sha256"), ("oriented_rgba_pixels", "pixel_sha256")):
        by_hash = collections.defaultdict(list)
        for row in rows:
            result = checked[row["photo_id"]]
            # Only confirmed stored bytes participate; failed decodes can still
            # share exact bytes, but cannot contribute decoded-pixel matches.
            if result["actual_sha256"] == row["sha256"] and result.get(field):
                by_hash[result[field]].append(row)
        for digest, members in sorted(by_hash.items()):
            photos = sorted({r["photo_id"] for r in members}, key=int)
            observations = sorted({r["observation_id"] for r in members}, key=int)
            if len(photos) > 1 or len(observations) > 1:
                species = sorted({r["species"] for r in members})
                groups.append({"kind": kind, "sha256": digest, "photo_ids": photos,
                    "observation_ids": observations, "species": species,
                    "cross_observation": len(observations) > 1,
                    "conflicting_species_labels": len(species) > 1})
    return groups


def run(root, cohort_path):
    root = root.resolve()
    raw_cohorts = cohort_path.read_bytes()
    cohorts = json.loads(raw_cohorts)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / "data/derived/quality" / f"qc_{stamp}_{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    print("QC_OUTPUT_DIRECTORY=" + str(output), flush=True)
    (output / "cohorts.json").write_bytes(raw_cohorts)
    for name in ("check_images.py", "report_download.py"):
        (output / name).write_bytes((Path(__file__).parent / name).read_bytes())
    checked = {}
    try:
        rows, sources = load_sources(root, cohorts, output)
        photos = unique_photos(rows)
        with (output / "photo_qc.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = None
            for photo_id, row in sorted(photos.items(), key=lambda item: int(item[0])):
                result = check_photo(root, row)
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(result), lineterminator="\n")
                    writer.writeheader()
                writer.writerow(result)
                checked[photo_id] = result
                if len(checked) % 250 == 0 or len(checked) == len(photos):
                    stream.flush()
                    print(f"CHECKED_PHOTOS={len(checked)}/{len(photos)}", flush=True)
        groups = duplicate_groups(rows, checked)
        write_json(output / "duplicate_candidates.json", groups)
        fields = list(dict.fromkeys(key for row in rows for key in row)) + ["image_qc_status"]
        with (output / "observation_photos.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows({**row, "image_qc_status": checked[row["photo_id"]]["status"]} for row in rows)
        counts = collections.Counter(r["status"] for r in checked.values())
        issues = sum(count for status, count in counts.items() if status != "pass")
        summary = {"status": "IMAGE_QC_COMPLETE_WITH_ISSUES" if issues else "IMAGE_QC_COMPLETE",
            "cohort_id": cohorts["cohort_id"], "cohorts_sha256": sha(raw_cohorts),
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "host": socket.gethostname(),
            "pbs_job_id": os.getenv("PBS_JOBID"), "python": sys.version, "platform": platform.platform(),
            "pillow": PIL.__version__, "decoder_support": {name: features.check(name) for name in ("jpg", "zlib", "webp")},
            "source_reports": sources, "unique_photo_ids": len(photos), "photo_observation_links": len(rows),
            "observations": len({r["observation_id"] for r in rows}), "photo_status_counts": dict(counts),
            "files_needing_review": issues,
            "passed_files_with_review_flags": sum(r["status"] == "pass" and bool(r["review_flags"]) for r in checked.values()),
            "duplicate_candidate_counts": dict(collections.Counter(g["kind"] for g in groups)),
            "cross_species_candidate_counts": dict(collections.Counter(g["kind"] for g in groups if g["conflicting_species_labels"])),
            "limits": {"max_file_bytes": MAX_BYTES, "max_frame_pixels": MAX_PIXELS, "max_frames": MAX_FRAMES, "max_total_frame_pixels": MAX_FRAME_PIXELS},
            "scope": "Technical image checks and exact byte/pixel duplicate candidates only; no bee visibility, biological label validation, near-duplicate search, crops or splits.",
            "raw_files_modified": False}
        write_json(output / "summary.json", summary)
        # We keep console output compact; full source provenance is saved in JSON.
        print(json.dumps({k: v for k, v in summary.items() if k != "source_reports"}, indent=2), flush=True)
        print("QC_SUMMARY_FILE=" + str(output / "summary.json"), flush=True)
        return 2 if issues else 0
    except (Exception, KeyboardInterrupt) as error:
        write_json(output / "failure.json", {"status": "IMAGE_QC_FAILED", "error": f"{type(error).__name__}: {error}", "checked_photos": len(checked)})
        print("IMAGE_QC_FAILED: " + str(error), file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cohorts", type=Path, default=ROOT / "config/cohorts_v1.json")
    args = parser.parse_args()
    return run(args.root, args.cohorts)


if __name__ == "__main__":
    sys.exit(main())
