"""We audit saved download tables without reading image pixels.

This lightweight, offline report uses Python's standard library. It does not
make training splits, change labels, merge observations or modify raw inputs.
"""

import argparse
import collections
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
BUDGET_STOP = "BudgetReached: Project media budget reached; resume after 24 hours."


def analyse(summary, rows, excluded, config, expected_job=None, accept_budget_limited=False):
    complete = summary.get("status") == "DOWNLOAD_COMPLETE" and not summary.get("error") and not summary.get("http_error")
    budget_limited = (summary.get("status") == "INCOMPLETE" and
                     summary.get("error") == BUDGET_STOP and not summary.get("http_error"))
    if not complete and not (accept_budget_limited and budget_limited):
        raise ValueError("Requires DOWNLOAD_COMPLETE or explicitly accepted project-budget stop.")
    if not complete and not expected_job:
        raise ValueError("Accepting a budget-limited subset requires an expected PBS job ID.")
    if expected_job and summary.get("pbs_job_id") != expected_job:
        raise ValueError("The saved attempt belongs to a different PBS job.")
    if not 0 < summary["processed_observations"] <= summary["selected_observations"]:
        raise ValueError("Invalid processed/selected observation counts.")
    if complete and summary["processed_observations"] != summary["selected_observations"]:
        raise ValueError("The attempt did not process all selected observations.")
    if not rows:
        raise ValueError("Saved photo manifest is empty.")
    checks = {
        "photo_status_counts": collections.Counter(r["status"] for r in rows),
        "eligible_species_counts": collections.Counter(r["species"] for r in rows),
        "eligible_photo_license_counts": collections.Counter(r["photo_license"] for r in rows),
        "exclusion_counts": collections.Counter(r["reason"] for r in excluded),
    }
    for key, actual in checks.items():
        if dict(actual) != summary.get(key):
            raise ValueError("Summary and saved tables disagree: " + key)
    if set(checks["photo_status_counts"]) - {"downloaded", "cached"}:
        raise ValueError("The manifest contains unfinished/failed photo downloads.")
    observations, photos, files = {}, {}, {}
    pairs = set()
    hashes = collections.defaultdict(list)
    for row in rows:
        for field in ("observation_id", "photo_id"):
            if not re.fullmatch(r"[1-9][0-9]*", row[field]):
                raise ValueError("Invalid " + field)
        pair = (row["observation_id"], row["photo_id"])
        if pair in pairs:
            raise ValueError("Repeated observation/photo link: " + str(pair))
        pairs.add(pair)
        if config["classes"].get(row["species"]) != int(row["class_id"]):
            raise ValueError("Photo class mapping differs from the saved configuration.")
        if row["photo_license"] not in config["data"]["initial_photo_license_allowlist"]:
            raise ValueError("Photo licence differs from the saved allowlist.")
        digest = row["sha256"]
        size = int(row["bytes"])
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or size <= 0:
            raise ValueError("Invalid saved image hash or size.")
        path = PurePosixPath(row["file_path"])
        if path.is_absolute() or ".." in path.parts or "\\" in row["file_path"] or path.parts[:3] != ("data", "raw", "images"):
            raise ValueError("Image path is outside the expected raw-image layout.")
        identity = (row["file_path"], digest, size)
        old_photo = photos.setdefault(row["photo_id"], identity)
        old_file = files.setdefault(row["file_path"], (digest, size))
        if old_photo != identity or old_file != (digest, size):
            raise ValueError("Conflicting file records for the same photo/path.")
        obs_fields = {k: row.get(k, "") for k in ("species", "observer_id", "observer_login", "export_country", "current_scientific_name")}
        old_observation = observations.setdefault(row["observation_id"], obs_fields)
        if old_observation != obs_fields:
            raise ValueError("Conflicting metadata within one observation.")
        hashes[digest].append(row)
    if len(observations) > summary["processed_observations"]:
        raise ValueError("Manifest has more observations than the processed count.")

    # We flag matching saved byte hashes for review. We do not infer that two
    # sightings are the same individual, or automatically assign split groups.
    duplicate_candidates = []
    for digest, members in hashes.items():
        photo_ids = sorted({r["photo_id"] for r in members}, key=int)
        obs_ids = sorted({r["observation_id"] for r in members}, key=int)
        if len(photo_ids) > 1 or len(obs_ids) > 1:
            species = sorted({r["species"] for r in members})
            duplicate_candidates.append({"sha256": digest, "photo_ids": photo_ids,
                "observation_ids": obs_ids, "cross_observation": len(obs_ids) > 1,
                "species": species, "conflicting_species_labels": len(species) > 1})
    country_obs = collections.Counter(r["export_country"] or "Unknown" for r in observations.values())
    country_photos = collections.defaultdict(set)
    for row in rows:
        country_photos[row.get("export_country") or "Unknown"].add(row["photo_id"])
    observer_counts = collections.Counter(r["observer_id"] for r in observations.values() if r["observer_id"])
    known_names = {r["observer_id"]: r["observer_login"] for r in observations.values() if r["observer_id"]}
    report = {
        "status": "MANIFEST_REPORT_COMPLETE",
        "scope": "Saved metadata only; image existence, hashes and pixels were not rechecked.",
        "dataset": summary["dataset"], "download_attempt": summary["attempt"],
        "download_pbs_job_id": summary.get("pbs_job_id"),
        "source_completion_status": summary["status"],
        "source_error": summary.get("error"),
        "source_acceptance": "completed_download" if complete else "budget_limited_saved_photos_for_curation",
        "processed_observations": summary["processed_observations"],
        "unprocessed_export_observations": summary["selected_observations"] - summary["processed_observations"],
        "selected_observations": summary["selected_observations"],
        "observations_with_downloads": len(observations),
        "observations_without_downloads": summary["selected_observations"] - len(observations),
        "photo_observation_links": len(rows), "unique_photo_ids": len(photos),
        "unique_saved_paths": len(files), "unique_saved_hashes": len(hashes),
        "reported_image_bytes": sum(size for _, size in files.values()),
        "observers_with_ids": len(observer_counts),
        "observations_without_observer_id": sum(not r["observer_id"] for r in observations.values()),
        "species_observations": dict(collections.Counter(r["species"] for r in observations.values())),
        "current_taxa_observations": dict(collections.Counter(r["current_scientific_name"] for r in observations.values())),
        "photo_license_link_counts": dict(checks["eligible_photo_license_counts"]),
        "exclusion_reason_counts": dict(checks["exclusion_counts"]),
        "countries": [{"country_from_export": country, "observations": count,
                       "unique_photos": len(country_photos[country])} for country, count in country_obs.most_common()],
        "top_observers": [{"observer_id": oid, "observer_login": known_names[oid],
                           "observations": count} for oid, count in observer_counts.most_common(10)],
        "largest_observer_share": max(observer_counts.values(), default=0) / len(observations),
        "matching_hash_candidate_groups": len(duplicate_candidates),
        "cross_observation_hash_groups": sum(g["cross_observation"] for g in duplicate_candidates),
        "hash_groups_with_conflicting_species": sum(g["conflicting_species_labels"] for g in duplicate_candidates),
    }
    return report, duplicate_candidates


def render(report):
    def safe(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    lines = ["# Saved download: manifest report", "", f"Dataset: {report['dataset']}",
        f"Download attempt: {report['download_attempt']}", f"Download job: {report['download_pbs_job_id']}", "",
        f"Source status: {report['source_completion_status']}; acceptance: {report['source_acceptance']}.", "",
        f"- Observations with photos: {report['observations_with_downloads']:,} of {report['selected_observations']:,}.",
        f"- Unique photo IDs: {report['unique_photo_ids']:,}; observation/photo links: {report['photo_observation_links']:,}.",
        f"- Reported image size: {report['reported_image_bytes'] / 1024**3:.3f} GiB (from saved sizes).",
        f"- Observers with IDs: {report['observers_with_ids']:,}; largest observer share: {report['largest_observer_share']:.1%} of retained observations.",
        f"- Matching-hash groups: {report['matching_hash_candidate_groups']}; crossing observations: {report['cross_observation_hash_groups']}.",
        f"- Matching-hash groups with conflicting species labels: {report['hash_groups_with_conflicting_species']}.", "",
        "Countries use the original export labels. Photos with obscured locations remain obscured.", "",
        "| Country in export | Retained observations | Unique photos |", "| --- | ---: | ---: |"]
    lines.extend(f"| {safe(r['country_from_export'])} | {r['observations']} | {r['unique_photos']} |" for r in report["countries"])
    if report["source_completion_status"] == "INCOMPLETE":
        lines.extend(["", "We accept only the saved downloaded/cached records for curation.",
            "The source run remains INCOMPLETE. Export order and the budget stop selected",
            "this subset; it is not a random or geographically balanced sample. The last",
            "processed observation may have unfinished views. Counts without saved photos",
            "include unvisited observations, not just licence exclusions."])
    lines.extend(["", "This is a metadata report. It does not decode images, recompute image hashes,",
        "verify biological labels, find near duplicates, or create training splits.",
        "Matching hashes are review candidates; no inputs or labels have been changed.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", type=Path, required=True, help="Directory containing summary.json and photos.csv")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--expected-job")
    parser.add_argument("--accept-budget-limited", action="store_true",
                        help="Accept saved photos from a known project-budget stop for curation; requires --expected-job")
    args = parser.parse_args()
    root = args.root.resolve()
    attempt = args.attempt.resolve()
    names = ("summary.json", "photos.csv", "excluded.csv", "config.json")
    try:
        raw = {name: (attempt / name).read_bytes() for name in names}
        summary = json.loads(raw["summary.json"])
        config = json.loads(raw["config.json"])
        rows = list(csv.DictReader(io.StringIO(raw["photos.csv"].decode("utf-8-sig"))))
        excluded = list(csv.DictReader(io.StringIO(raw["excluded.csv"].decode("utf-8-sig"))))
        report, candidates = analyse(summary, rows, excluded, config, args.expected_job, args.accept_budget_limited)
        dataset = report["dataset"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", dataset):
            raise ValueError("Unexpected dataset identifier.")
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = root / "reports" / f"manifest_{dataset}_{stamp}_{uuid.uuid4().hex[:8]}"
        output.mkdir(parents=True, exist_ok=False)
        report["created_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        report["source_attempt_directory"] = str(attempt)
        report["source_sha256"] = {name: hashlib.sha256(content).hexdigest() for name, content in raw.items()}
        report["report_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        (output / "source").mkdir()
        for name, content in raw.items():
            (output / "source" / name).write_bytes(content)
        for name, value in (("report.json", report), ("matching_hash_candidates.json", candidates)):
            (output / name).write_bytes((json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
        text = render(report)
        (output / "report.md").write_bytes(text.encode("utf-8"))
        print(text)
        print("MANIFEST_REPORT=PASS")
        print("REPORT_DIRECTORY=" + str(output))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print("MANIFEST_REPORT=FAIL: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
