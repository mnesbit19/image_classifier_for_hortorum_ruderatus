"""We propose near-image matches for review without changing labels, files or splits."""

import argparse
import collections
import hashlib
import io
import json
import math
import os
from pathlib import Path
import socket
import statistics
import sys
import warnings

import PIL
from PIL import Image, ImageOps, ImageStat

from check_images import MAX_BYTES, MAX_PIXELS, inside, write_json
from prepare_curation import ROOT, load_qc, new_output, read_json, sha, write_csv

ALGORITHM = "dct63_dhash64_v1"
COSINES = tuple(tuple(math.cos(math.pi * (2 * x + 1) * k / 64) for x in range(32)) for k in range(8))
TRANSFORMS = (("identity", None), ("rotate90", Image.Transpose.ROTATE_90),
              ("rotate180", Image.Transpose.ROTATE_180), ("rotate270", Image.Transpose.ROTATE_270),
              ("flip_lr", Image.Transpose.FLIP_LEFT_RIGHT), ("flip_tb", Image.Transpose.FLIP_TOP_BOTTOM),
              ("transpose", Image.Transpose.TRANSPOSE), ("transverse", Image.Transpose.TRANSVERSE))
PAIR_FIELDS = ("candidate_id", "photo_id_1", "photo_id_2", "observations_1", "observations_2",
               "species_1", "species_2", "cross_observation", "cross_species", "same_observation_overlap",
               "phash_distance", "dhash_distance", "view_1", "view_2", "review_status", "review_decision", "reviewer_notes")


def hashes(gray):
    """We use 63 AC DCT bits and 64 horizontal-difference bits with versioned definitions."""
    pixels = list(gray.tobytes())
    horizontal = [[sum(pixels[y * 32 + x] * c[x] for x in range(32)) for c in COSINES] for y in range(32)]
    coefficients = [sum(horizontal[y][u] * COSINES[v][y] for y in range(32))
                    for v in range(8) for u in range(8) if (v, u) != (0, 0)]
    median = statistics.median(coefficients)
    perceptual = sum((value > median) << bit for bit, value in enumerate(coefficients))
    with gray.resize((9, 8), Image.Resampling.LANCZOS) as small:
        values = small.tobytes()
    difference = sum((values[y * 9 + x + 1] > values[y * 9 + x]) << (y * 8 + x)
                     for y in range(8) for x in range(8))
    return perceptual, difference


def image_features(image, settings):
    with ImageOps.exif_transpose(image) as oriented:
        with oriented.convert("RGBA") as rgba:
            with Image.new("RGBA", rgba.size, "white") as matte:
                matte.alpha_composite(rgba)
                rgb = matte.convert("RGB")
        features = []
        try:
            for view, fraction in (("full", 1.0), ("center075", settings["center_crop_fraction"])):
                width, height = rgb.size
                cw, ch = max(1, round(width * fraction)), max(1, round(height * fraction))
                left, top = (width - cw) // 2, (height - ch) // 2
                with rgb.crop((left, top, left + cw, top + ch)) as crop:
                    with ImageOps.grayscale(crop) as gray:
                        thumb = gray.resize((32, 32), Image.Resampling.LANCZOS)
                try:
                    contrast = ImageStat.Stat(thumb).stddev[0]
                    if contrast < settings["min_luma_stddev"]:
                        continue
                    for transform, method in TRANSFORMS:
                        variant = thumb.copy() if method is None else thumb.transpose(method)
                        with variant:
                            phash, dhash = hashes(variant)
                        features.append({"view": view + "/" + transform, "phash": phash, "dhash": dhash,
                                         "luma_stddev": round(contrast, 8)})
                finally:
                    thumb.close()
        finally:
            rgb.close()
        return features


def features_for_photo(root, row, checked, settings):
    path = inside(root, row["file_path"], ("data", "raw", "images"))
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("File exceeds the QC byte limit.")
    raw = path.read_bytes()
    if len(raw) != int(row["bytes"]) or hashlib.sha256(raw).hexdigest() != row["sha256"]:
        raise ValueError("Raw bytes differ from the accepted QC file.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(io.BytesIO(raw)) as image:
            if image.width * image.height > MAX_PIXELS or getattr(image, "n_frames", 1) != int(checked["frames"]):
                raise ValueError("Image size/frame count differs from accepted QC limits.")
            if getattr(image, "n_frames", 1) != 1:
                return [], "held_animation"
            image.load()
            result = image_features(image, settings)
    return result, "hashed" if result else "held_low_contrast"


class HashIndex:
    """Our disjoint-bit index retrieves every stored 63-bit hash within the chosen radius."""
    def __init__(self, radius):
        if not 0 <= radius <= 10:
            raise ValueError("Perceptual radius must be between 0 and 10.")
        self.radius = radius
        self.slices = [(63 * i // (radius + 1), 63 * (i + 1) // (radius + 1)) for i in range(radius + 1)]
        self.buckets = collections.defaultdict(list)
        self.entries = []

    def keys(self, value):
        for i, (start, end) in enumerate(self.slices):
            yield i, (value >> start) & ((1 << (end - start)) - 1)

    def add(self, entry):
        index = len(self.entries)
        self.entries.append(entry)
        for key in self.keys(entry["phash"]):
            self.buckets[key].append(index)

    def query(self, value):
        hits = set()
        for key in self.keys(value):
            hits.update(self.buckets.get(key, ()))
        for index in sorted(hits):
            entry = self.entries[index]
            distance = (entry["phash"] ^ value).bit_count()
            if distance <= self.radius:
                yield entry, distance


def propose(features, pid, index, photos, checked, by_photo, pairs, settings):
    for feature in features:
        for previous, distance in index.query(feature["phash"]):
            other = previous["photo_id"]
            if checked[pid]["actual_sha256"] == checked[other]["actual_sha256"] or (
                checked[pid]["pixel_sha256"] and checked[pid]["pixel_sha256"] == checked[other]["pixel_sha256"]):
                continue
            dhash_distance = (previous["dhash"] ^ feature["dhash"]).bit_count()
            if dhash_distance > settings["dhash_radius"]:
                continue
            key = other, pid
            score = (distance, dhash_distance, previous["view"], feature["view"])
            if key in pairs and score >= pairs[key][0]:
                continue
            if key not in pairs and len(pairs) >= settings["max_candidate_pairs"]:
                raise RuntimeError("Candidate-pair limit reached; retain the partial run and review settings before a new scan.")
            a, b = by_photo[other], by_photo[pid]
            pairs[key] = (score, {
                "candidate_id": "near_" + hashlib.sha256((other + "|" + pid).encode()).hexdigest()[:20],
                "photo_id_1": other, "photo_id_2": pid,
                "observations_1": ";".join(sorted(a, key=int)), "observations_2": ";".join(sorted(b, key=int)),
                "species_1": photos[other]["species"], "species_2": photos[pid]["species"],
                "cross_observation": a != b or len(a | b) > 1,
                "cross_species": photos[other]["species"] != photos[pid]["species"],
                "same_observation_overlap": bool(a & b), "phash_distance": distance, "dhash_distance": dhash_distance,
                "view_1": previous["view"], "view_2": feature["view"],
                "review_status": "pending", "review_decision": "", "reviewer_notes": ""})
    for feature in features:
        if feature["view"].endswith("/identity"):
            index.add({**feature, "photo_id": pid})


def run(root, config_path, photo_ids=None):
    config = read_json(config_path)
    settings = config["near_duplicates"]
    if settings["algorithm"] != ALGORITHM or settings["center_crop_fraction"] != 0.75 or not 0 <= settings["dhash_radius"] <= 64 or settings["max_candidate_pairs"] < 1:
        raise ValueError("Unsupported near-duplicate settings.")
    _, qc, rows, photos, checked, exact, inputs = load_qc(root, config)
    selected = set(photos) if photo_ids is None else set(photo_ids)
    if not selected or selected - set(photos):
        raise ValueError("Requested trial photo IDs differ from the accepted cohort.")
    output = new_output(root, "data/derived/quality", "near")
    print("NEAR_OUTPUT_DIRECTORY=" + str(output), flush=True)
    (output / "review_config.json").write_bytes(config_path.read_bytes())
    for name in ("scan_near_duplicates.py", "prepare_curation.py", "check_images.py", "report_download.py"):
        (output / name).write_bytes((Path(__file__).parent / name).read_bytes())
    write_json(output / "qc_input_sha256.json", inputs)
    write_json(output / "requested_photo_ids.json", sorted(selected, key=int))
    write_json(output / "existing_exact_candidates.json", exact)
    by_photo = collections.defaultdict(set)
    for row in rows:
        by_photo[row["photo_id"]].add(row["observation_id"])
    statuses, pairs = [], {}
    index = HashIndex(settings["phash_radius"])
    try:
        import csv
        with (output / "features.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("photo_id", "view", "phash63", "dhash64", "luma_stddev"), lineterminator="\n")
            writer.writeheader()
            for pid in sorted(selected, key=int):
                try:
                    features, state = features_for_photo(root, photos[pid], checked[pid], settings)
                    status = {"photo_id": pid, "status": state, "feature_views": len(features), "error": ""}
                except Exception as error:
                    features = []
                    status = {"photo_id": pid, "status": "input_error", "feature_views": 0, "error": f"{type(error).__name__}: {error}"}
                for f in features:
                    writer.writerow({"photo_id": pid, "view": f["view"], "phash63": f"{f['phash']:016x}",
                                     "dhash64": f"{f['dhash']:016x}", "luma_stddev": f["luma_stddev"]})
                propose(features, pid, index, photos, checked, by_photo, pairs, settings)
                statuses.append(status)
                if len(statuses) % 250 == 0 or len(statuses) == len(selected):
                    stream.flush()
                    print(f"NEAR_PROCESSED={len(statuses)}/{len(selected)}; CANDIDATE_PAIRS={len(pairs)}", flush=True)
        candidates = [value[1] for _, value in sorted(pairs.items(), key=lambda item: tuple(map(int, item[0])))]
        write_csv(output / "candidate_pairs.csv", candidates, PAIR_FIELDS)
        write_csv(output / "photo_scan_status.csv", statuses)
        counts = dict(collections.Counter(r["status"] for r in statuses))
        errors = counts.get("input_error", 0)
        summary = {"status": "NEAR_SCAN_COMPLETE_WITH_ISSUES" if errors else "NEAR_SCAN_COMPLETE",
            "scope": "full_cohort" if photo_ids is None else "explicit_local_subset_trial",
            "cohort_id": qc["cohort_id"], "qc_job_id": qc["pbs_job_id"], "config_sha256": sha(config_path),
            "pbs_job_id": os.environ.get("PBS_JOBID"), "host": socket.gethostname(),
            "requested_photos": len(selected), "processed_photos": len(statuses), "photo_status_counts": counts,
            "candidate_pairs": len(candidates), "cross_species_pairs": sum(r["cross_species"] for r in candidates),
            "cross_observation_pairs": sum(r["cross_observation"] for r in candidates),
            "algorithm": ALGORITHM, "settings": settings, "pillow": PIL.__version__, "python": sys.version,
            "raw_files_modified": False, "labels_revised": False, "new_individual_merges": 0, "splits_assigned": False,
            "limits": "Candidate retrieval only: arbitrary crops, local subjects, appearance changes and all related individuals are not guaranteed. Review every proposed pair."}
        write_json(output / "summary.json", summary)
        write_json(output / "OUTPUT_SHA256.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})
        print(json.dumps(summary, indent=2), flush=True)
        return 2 if errors else 0, output
    except (Exception, KeyboardInterrupt) as error:
        write_csv(output / "partial_photo_scan_status.csv", statuses, ("photo_id", "status", "feature_views", "error"))
        write_csv(output / "partial_candidate_pairs.csv", [value[1] for value in pairs.values()], PAIR_FIELDS)
        write_json(output / "failure.json", {"status": "NEAR_SCAN_FAILED", "processed_photos": len(statuses),
                   "candidate_pairs_so_far": len(pairs), "error": f"{type(error).__name__}: {error}"})
        print("NEAR_SCAN_FAILED: " + str(error), file=sys.stderr, flush=True)
        return 1, output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/review_v1.json")
    parser.add_argument("--photo-ids", type=Path, help="Explicit local trial IDs, one per line; omitted for full-cohort scan")
    args = parser.parse_args()
    try:
        selected = None if args.photo_ids is None else args.photo_ids.read_text().splitlines()
        code, _ = run(ROOT, args.config.resolve(), selected)
    except Exception as error:
        print("NEAR_SCAN_FAILED: " + str(error), file=sys.stderr)
        code = 1
    raise SystemExit(code)
