"""We validate our completed QC tables and create a versioned curation seed offline."""

import argparse
import collections
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import uuid

from check_images import csv_rows, duplicate_groups, inside, unique_photos, write_json
from report_download import analyse

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path, rows, fields=None):
    import csv
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def new_output(root, relative, prefix):
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = root / relative / f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    return output


def load_qc(root, config):
    """We bind every downstream row to the fixed successful QC run and source snapshots."""
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported review configuration.")
    bound = config["qc"]
    folder = inside(root, bound["directory"], ("data", "derived", "quality"))
    for name, digest in bound["files_sha256"].items():
        path = folder / name
        if path.resolve().is_relative_to(folder.resolve()) is False or sha(path) != digest:
            raise ValueError("Pinned QC input hash differs: " + name)
    if sha(folder / "summary.json") != bound["summary_sha256"]:
        raise ValueError("QC summary hash differs from the accepted run.")
    summary = read_json(folder / "summary.json")
    if summary["status"] != "IMAGE_QC_COMPLETE" or summary["pbs_job_id"] != bound["job_id"]:
        raise ValueError("QC is not the accepted complete successful run.")
    cohort_path = inside(root, config["cohorts_file"], ("config",))
    cohorts = read_json(folder / "cohorts.json")
    if sha(folder / "cohorts.json") != summary["cohorts_sha256"] or sha(cohort_path) != summary["cohorts_sha256"]:
        raise ValueError("Accepted cohort configuration changed.")
    checked_rows = csv_rows((folder / "photo_qc.csv").read_bytes())
    checked = {r["photo_id"]: r for r in checked_rows}
    rows = csv_rows((folder / "observation_photos.csv").read_bytes())
    photos = unique_photos(rows)
    if len(checked) != len(checked_rows) or set(checked) != set(photos):
        raise ValueError("QC photo IDs are duplicated or differ from observation links.")
    expected = []
    if len(cohorts["sources"]) != len(summary["source_reports"]):
        raise ValueError("QC source reports differ from the cohort configuration.")
    for source, report in zip(cohorts["sources"], summary["source_reports"]):
        src = folder / "sources" / source["name"]
        for name, digest in report["source_sha256"].items():
            if name not in ("summary.json", "photos.csv", "excluded.csv", "config.json", "code_snapshot.py", "plan.json") or sha(src / name) != digest:
                raise ValueError("QC source snapshot hash differs: " + source["name"] + "/" + name)
        if sha(src / "plan.json") != source["plan_sha256"] or sha(src / "code_snapshot.py") != source["download_code_sha256"]:
            raise ValueError("Source plan/downloader differs from the accepted cohort.")
        saved = csv_rows((src / "photos.csv").read_bytes())
        reproduced, _ = analyse(read_json(src / "summary.json"), saved,
            csv_rows((src / "excluded.csv").read_bytes()), read_json(src / "config.json"),
            source["job_id"], source["accept_budget_limited"])
        if reproduced != {k: v for k, v in report.items() if k != "source_sha256"}:
            raise ValueError("QC source report does not reproduce.")
        for key, count in source["expected_counts"].items():
            if reproduced[key] != count:
                raise ValueError("Accepted source counts differ.")
        expected.extend({**r, "source_dataset": source["dataset"], "source_attempt": source["attempt"],
                         "image_qc_status": checked[r["photo_id"]]["status"]} for r in saved)
    if rows != expected:
        raise ValueError("Observation/photo table differs from the source snapshots.")
    for pid, row in photos.items():
        result = checked[pid]
        if result["status"] != "pass" or result["actual_sha256"] != row["sha256"] or int(result["actual_bytes"]) != int(row["bytes"]) or result["file_path"] != row["file_path"]:
            raise ValueError("Photo QC differs from its accepted file record: " + pid)
    counts = {"unique_photo_ids": len(photos), "photo_observation_links": len(rows),
              "observations": len({r["observation_id"] for r in rows})}
    if any(summary[k] != count for k, count in counts.items()):
        raise ValueError("QC counts differ from its tables.")
    exact = read_json(folder / "duplicate_candidates.json")
    if duplicate_groups(rows, checked) != exact:
        raise ValueError("Exact-match groups do not reproduce.")
    if dict(collections.Counter(g["kind"] for g in exact)) != summary["duplicate_candidate_counts"]:
        raise ValueError("Exact-match counts differ.")
    inputs = {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob("*")) if p.is_file()}
    return folder, summary, rows, photos, checked, exact, inputs


def observation_groups(rows, exact):
    parent = {r["observation_id"]: r["observation_id"] for r in rows}

    def find(o):
        while parent[o] != o:
            parent[o] = parent[parent[o]]
            o = parent[o]
        return o

    for group in exact:
        observations = group["observation_ids"]
        for other in observations[1:]:
            a, b = sorted((find(observations[0]), find(other)), key=int)
            parent[b] = a
    components = collections.defaultdict(list)
    for obs in sorted(parent, key=int):
        components[find(obs)].append(obs)
    mapping, records = {}, []
    for members in components.values():
        ident = "obsgrp_" + hashlib.sha256(";".join(members).encode()).hexdigest()[:16]
        for obs in members:
            mapping[obs] = ident
        records.append({"observation_group_id": ident, "observation_ids": ";".join(members),
                        "observation_count": len(members), "basis": "observation_id_and_exact_or_shared_photo",
                        "individual_identity_established": False, "split": ""})
    return mapping, records


def build_manifest(rows, checked, exact, triage=None, source_reports=()):
    groups, constraints = observation_groups(rows, exact)
    photo_groups = collections.defaultdict(set)
    for g in exact:
        ident = "exact_" + hashlib.sha256((";".join(g["photo_ids"]) + "|" + ";".join(g["observation_ids"])).encode()).hexdigest()[:16]
        for pid in g["photo_ids"]:
            photo_groups[pid].add(ident)
    triage = triage or {}
    sources = {r["dataset"]: r for r in source_reports}
    output = []
    manual = ("target_id", "bbox_left", "bbox_top", "bbox_right", "bbox_bottom", "selected_frame_zero_based",
              "native_bee_width", "native_bee_height", "visibility", "focus_detail", "view", "caste",
              "multiple_subjects", "target_label_correspondence", "reviewed_species", "label_evidence_type",
              "label_evidence_notes", "resolved_country", "country_evidence", "reviewer", "reviewed_utc", "decision_reason")
    for row in rows:
        q = checked[row["photo_id"]]
        screened = triage.get(row["photo_id"], {})
        source = sources.get(row.get("source_dataset"), {})
        record = {**row, "curation_record_id": "obs_" + row["observation_id"] + "_photo_" + row["photo_id"],
            "source_species": row["species"], "source_class_id": row["class_id"],
            "source_download_status": source.get("source_completion_status", "see_bound_QC_source_reports"),
            "source_acceptance": source.get("source_acceptance", "see_bound_QC_source_reports"),
            "observation_group_id": groups[row["observation_id"]],
            "exact_match_group_ids": ";".join(sorted(photo_groups.get(row["photo_id"], ()))),
            "image_width": q["display_width"], "image_height": q["display_height"], "frames": q["frames"],
            "technical_review_flags": q["review_flags"], "visual_triage_status": screened.get("screening_status", "not_reviewed"),
            "visual_triage_notes": screened.get("screening_notes", ""),
            "provisional_review_route": screened.get("primary_review_route", "pending_crop_and_label_review"),
            "frame_policy_status": "pending" if int(q["frames"]) > 1 else "single_frame",
            "bbox_coordinate_system": "EXIF_oriented_pixels_top_left_exclusive_right_bottom",
            "label_review_status": "pending", "curation_decision": "pending", "country_derivation_status": "not_derived",
            **{field: "" for field in manual}, "split": ""}
        output.append(record)
    return output, constraints


def run(root, config_path, use_triage=True):
    config = read_json(config_path)
    folder, qc, rows, photos, checked, exact, inputs = load_qc(root, config)
    triage, triage_hash = {}, None
    if use_triage:
        bound = config["local_triage"]
        path = inside(root, bound["file"], ("reports",))
        triage_hash = sha(path)
        if triage_hash != bound["sha256"]:
            raise ValueError("Local visual triage changed; version the input before use.")
        for row in csv_rows(path.read_bytes()):
            pid = row["photo_id"]
            if pid in triage or pid not in photos or row["expected_sha256"] != checked[pid]["actual_sha256"] or row["species"] != photos[pid]["species"]:
                raise ValueError("Visual triage differs from the accepted photo records.")
            triage[pid] = row
    manifest, constraints = build_manifest(rows, checked, exact, triage, qc["source_reports"])
    output = new_output(root, "data/review", "curation_seed")
    (output / "review_config.json").write_bytes(config_path.read_bytes())
    (output / "prepare_curation.py").write_bytes(Path(__file__).read_bytes())
    (output / "qc_summary.json").write_bytes((folder / "summary.json").read_bytes())
    (output / "accepted_cohorts.json").write_bytes((folder / "cohorts.json").read_bytes())
    write_csv(output / "curation_manifest.csv", manifest)
    write_csv(output / "observation_constraints.csv", constraints)
    summary = {"status": "CURATION_SEED_PREPARED", "schema_version": 1, "qc_job_id": qc["pbs_job_id"],
        "cohort_id": qc["cohort_id"], "records": len(manifest), "photos": len(photos),
        "observations": qc["observations"], "observation_groups": len(constraints),
        "linked_components": sum(r["observation_count"] > 1 for r in constraints),
        "triaged_photos": len(triage), "triage_sha256": triage_hash,
        "source_download_statuses": {r["dataset"]: r["source_completion_status"] for r in qc["source_reports"]},
        "qc_input_sha256": inputs, "config_sha256": sha(config_path),
        "raw_images_opened": False, "labels_revised": False, "decisions": {"pending": len(manifest)},
        "scope": "Versioned editable seed only. No accepted training images, bee boxes, crops or splits."}
    write_json(output / "summary.json", summary)
    write_json(output / "OUTPUT_SHA256.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    print("CURATION_SEED_DIRECTORY=" + str(output))
    print(json.dumps({k: v for k, v in summary.items() if k != "qc_input_sha256"}, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/review_v1.json")
    parser.add_argument("--without-local-triage", action="store_true")
    args = parser.parse_args()
    try:
        run(ROOT, args.config.resolve(), not args.without_local_triage)
    except Exception as error:
        print("CURATION_SEED_FAILED: " + str(error), file=sys.stderr)
        raise SystemExit(1)
