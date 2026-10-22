"""We evaluate our frozen first checkpoint once, with observer-cluster uncertainty."""

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from baseline_data import (CLASSES, PREPROCESS, binary_metrics, evaluate_predictions,
                           preprocess_image, project_path, read_csv, sha256,
                           verify_dataset, write_csv, write_json)


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def cluster_bootstrap(groups, replicates=2000, seed=42):
    """We resample split components, retaining all their evaluation groups together."""
    if replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required.")
    clusters = defaultdict(list)
    for group in groups:
        clusters[group["component_id"]].append(group)
    if len(clusters) < 2:
        raise ValueError("At least two independent split components are required.")
    if {int(g["class_id"]) for g in groups} != {0, 1}:
        raise ValueError("Both species are required for uncertainty estimates.")
    rng = random.Random(seed)
    keys = sorted(clusters)
    draws = defaultdict(list)
    skipped = 0
    for _ in range(replicates):
        sample = [group for key in rng.choices(keys, k=len(keys)) for group in clusters[key]]
        metrics = binary_metrics([int(g["class_id"]) for g in sample], [float(g["p_ruderatus"]) for g in sample])
        if metrics["balanced_accuracy"] is None:
            skipped += 1
            continue
        for key in ("balanced_accuracy", "accuracy", "log_loss"):
            draws[key].append(metrics[key])
        for species, label in (("hortorum", "0"), ("ruderatus", "1")):
            for metric in ("precision", "recall", "f1"):
                draws[species + "_" + metric].append(metrics["per_class"][label][metric])
    if not draws or replicates - skipped < replicates * 0.9:
        raise ValueError("Too few bootstrap samples contain both species.")
    return {"method": "unstratified_split_component_percentile_bootstrap", "level": 0.95,
            "replicates_requested": replicates, "replicates_used": replicates - skipped,
            "replicates_missing_one_species": skipped, "seed": seed, "clusters": len(keys),
            "intervals": {name: {"lower": percentile(values, .025), "upper": percentile(values, .975)}
                          for name, values in draws.items()},
            "zero_predicted_class_precision": "defined_as_zero_as_in_point_metrics"}


def verify_plan(root, plan):
    if plan["schema_version"] != 1 or plan["classes"] != CLASSES or plan["preprocess"] != PREPROCESS:
        raise ValueError("Unsupported evaluation mapping or preprocessing.")
    if plan["decision_threshold"] != .5 or plan["device"] != "cpu":
        raise ValueError("Our released protocol fixes CPU inference and the original decision rule.")
    training = project_path(root, plan["training_run_dir"])
    for relative, digest in plan["training_files_sha256"].items():
        if sha256(project_path(training, relative)) != digest:
            raise ValueError("Training artifact changed: " + relative)
    if (training / "job_id.txt").read_text().strip() != plan["training_job_id"]:
        raise ValueError("Training job identity changed.")
    summary = json.loads((training / "artifacts/summary.json").read_text())
    if summary["status"] != "BASELINE_TRAINING_COMPLETE" or summary["test_model_evaluation"] != "NOT_RUN":
        raise ValueError("Unexpected training/test status.")
    if summary["best_epoch"] != plan["best_epoch"] or summary["best_checkpoint_sha256"] != plan["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed.")
    source = json.loads((training / "artifacts/source_sha256.json").read_text())
    for name, digest in source.items():
        if sha256(Path(root) / "code" / name) != digest:
            raise ValueError("Training/prediction source changed: " + name)
    config = json.loads((training / "artifacts/config.json").read_text())
    if sha256(Path(root) / "config/baseline_v1.json") != plan["baseline_config_sha256"]:
        raise ValueError("Original training configuration changed.")
    if config != json.loads((Path(root) / "config/baseline_v1.json").read_text()):
        raise ValueError("Training and current baseline configurations disagree.")
    rows = verify_dataset(root, config)
    test = [r for r in rows if r["split"] == "test"]
    counts = {"photos": len(test), "evaluation_groups": len({r["evaluation_group_id"] for r in test}),
              "components": len({r["component_id"] for r in test}),
              "observations": len({obs for r in test for obs in r["observation_ids"].split("|")}),
              "species_photos": {species: sum(r["species"] == species for r in test) for species in CLASSES}}
    if counts != plan["expected_test_counts"]:
        raise ValueError("Held-out set counts changed.")
    return training, config, rows, test


def verify_checkpoint_metadata(checkpoint, config, plan):
    if checkpoint["epoch"] != plan["best_epoch"] or checkpoint["phase"] != "classifier":
        raise ValueError("Unexpected checkpoint epoch/phase.")
    if checkpoint["config"] != config or checkpoint["config_sha256"] != plan["baseline_config_sha256"]:
        raise ValueError("Checkpoint/config binding changed.")
    if checkpoint["test_model_evaluation"] != "NOT_RUN":
        raise ValueError("Unexpected checkpoint test status.")
    if checkpoint["classes"] != CLASSES or checkpoint["preprocess"] != PREPROCESS:
        raise ValueError("Checkpoint class mapping/preprocessing changed.")


def release_once(marker, receipt):
    try:
        marker.mkdir()
    except FileExistsError as error:
        raise RuntimeError("Final test already released. Inspect its existing record; no automatic repeat.") from error
    write_json(marker / "release.json", receipt)


def runtime_score_comparison(actual, reference, tolerance):
    """We report numerical agreement separately from the fixed species decision."""
    if (not math.isfinite(tolerance) or tolerance <= 0 or
            any(not math.isfinite(value) or not 0 <= value <= 1 for value in (actual, reference))):
        raise ValueError("Runtime comparison requires finite probabilities and a positive finite tolerance.")
    difference = abs(actual - reference)
    return {"actual": actual, "reference": reference, "absolute_difference": difference,
            "absolute_tolerance": tolerance, "within_tolerance": difference <= tolerance,
            "same_argmax_class": (actual > .5) == (reference > .5)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/test_baseline_v1.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--release-final-test", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("PBS_JOBID") or not args.release_final_test:
        raise SystemExit("STOP: use the separate 10 CPU evaluation submission.")
    root = Path(__file__).resolve().parents[1]
    if root.as_posix() != "/rds/general/user/mln19/home/ml_rudvhort":
        raise SystemExit("STOP: unexpected HPC project root.")
    plan = json.loads(args.config.read_text())
    marker = project_path(root, plan["release_marker_dir"])
    if marker.exists():
        raise SystemExit("STOP: final test already released; inspect the existing evaluation.")
    training, config, rows, test = verify_plan(root, plan)
    output = args.output_dir.resolve()
    if not output.is_relative_to(root / "runs/evaluation"):
        raise SystemExit("STOP: evaluation output must be under runs/evaluation.")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", plan)
    write_json(output / "source_sha256.json", {name: sha256(root / "code" / name) for name in
                                               ("evaluate_baseline.py", "baseline_data.py", "train_baseline.py", "predict_photo.py")})

    import numpy as np
    import PIL
    import torch
    import torchvision
    from train_baseline import PhotoDataset, model_from_checkpoint

    versions = {"torch": str(torch.__version__).split("+")[0], "torchvision": str(torchvision.__version__).split("+")[0],
                "Pillow": PIL.__version__, "numpy": np.__version__}
    if any(versions[key] != config["software"][key] for key in versions):
        raise RuntimeError("Evaluation software differs from the trained baseline.")
    torch.set_num_threads(1)
    torch.manual_seed(plan["seed"])
    torch.use_deterministic_algorithms(True)
    checkpoint_path = training / "artifacts/best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    verify_checkpoint_metadata(checkpoint, config, plan)
    model = model_from_checkpoint(checkpoint, "cpu")
    expected_validation = {r["photo_id"]: r for r in read_csv(training / "artifacts/validation_photo_predictions.csv")}
    by_id = {r["photo_id"]: r for r in rows}
    parity = []
    # We test the CPU inference path on fixed development examples before
    # releasing any held-out photo. These checks do not change the model.
    for photo_id in plan["validation_parity_photo_ids"]:
        row = by_id[photo_id]
        if row["split"] != "validation":
            raise ValueError("Runtime parity check must use validation photos only.")
        path = project_path(root, row["file_path"])
        if sha256(path) != row["sha256"]:
            raise ValueError("Validation-check photo changed.")
        with torch.inference_mode():
            probability = float(model(torch.from_numpy(preprocess_image(path)).unsqueeze(0)).softmax(dim=1)[0, 1])
        expected = float(expected_validation[photo_id]["p_ruderatus"])
        comparison = runtime_score_comparison(probability, expected, plan["validation_parity_absolute_tolerance"])
        parity.append({"photo_id": photo_id, "p_ruderatus_cpu": probability,
                       "p_ruderatus_saved": expected, **comparison})
        print(f"VALIDATION_PARITY={len(parity)}/{len(plan['validation_parity_photo_ids'])}; "
              f"PHOTO_ID={photo_id}; CPU={probability:.10f}; GPU_REFERENCE={expected:.10f}; "
              f"ABS_DIFF={comparison['absolute_difference']:.10f}; "
              f"TOLERANCE={comparison['absolute_tolerance']}", flush=True)
    passed = all(row["within_tolerance"] for row in parity)
    write_json(output / "validation_parity.json", {"status": "PASS" if passed else "FAIL",
               "validation_parity": parity, "test_images_opened": 0})
    if not passed:
        raise RuntimeError("CPU/reference score check failed; inspect validation_parity.json. Test remains sealed.")
    first = by_id[plan["validation_parity_photo_ids"][0]]
    # We exercise the standalone prediction tool on a development photo as well.
    try:
        subprocess.run([sys.executable, str(root / "code/predict_photo.py"), "--checkpoint", str(checkpoint_path),
                        "--image", str(project_path(root, first["file_path"])), "--output", str(output / "predictor_check.json")],
                       check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as error:
        print(error.stdout or "", error.stderr or "", flush=True)
        raise
    cli_prediction = json.loads((output / "predictor_check.json").read_text())
    cli_probability = float(cli_prediction["scores"]["Bombus ruderatus"])
    cli_checks = {"versus_saved_gpu": runtime_score_comparison(
                    cli_probability, parity[0]["p_ruderatus_saved"], plan["validation_parity_absolute_tolerance"]),
                  "versus_direct_cpu": runtime_score_comparison(
                    cli_probability, parity[0]["p_ruderatus_cpu"],
                    plan.get("predictor_cpu_absolute_tolerance", plan["validation_parity_absolute_tolerance"]))}
    write_json(output / "predictor_runtime_parity.json", cli_checks)
    if not all(check["within_tolerance"] for check in cli_checks.values()):
        raise RuntimeError("Standalone predictor/reference disagreement; inspect predictor_runtime_parity.json.")
    write_json(output / "runtime_check.json", {"status": "PASS", "versions": versions, "device": "cpu",
                                               "validation_parity": parity, "predictor_cli": "PASS",
                                               "predictor_comparisons": cli_checks, "test_images_opened": 0})
    print("CPU_RUNTIME_AND_PREDICTOR_CHECK=PASS; FINAL_TEST_RELEASE_NEXT", flush=True)

    release_once(marker, {"utc": datetime.now(timezone.utc).isoformat(), "pbs_job_id": os.environ["PBS_JOBID"],
                          "output_directory": str(output), "checkpoint_sha256": plan["checkpoint_sha256"],
                          "protocol_sha256": sha256(args.config), "test_model_evaluation": "RELEASED_ONCE"})
    started = time.monotonic()
    try:
        for i, row in enumerate(test, 1):
            path = project_path(root, row["file_path"])
            if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
                raise RuntimeError("Test photo checksum mismatch: " + row["photo_id"])
            if i % 100 == 0 or i == len(test):
                print(f"TEST_IMAGE_PREFLIGHT={i}/{len(test)}", flush=True)
        loader = torch.utils.data.DataLoader(PhotoDataset(root, test), batch_size=plan["batch_size"],
                                              num_workers=plan["num_workers"], shuffle=False)
        probabilities = []
        partial = output / "test_photo_predictions.partial.csv"
        with partial.open("x", encoding="utf-8", newline="") as handle:
            fields = ("photo_id", "class_id", "component_id", "evaluation_group_id", "p_ruderatus")
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            with torch.inference_mode():
                for images, _ in loader:
                    batch = model(images).softmax(dim=1)[:, 1].tolist()
                    offset = len(probabilities)
                    for row, probability in zip(test[offset:offset + len(batch)], batch, strict=True):
                        writer.writerow({**{key: row[key] for key in fields[:-1]}, "p_ruderatus": probability})
                    probabilities.extend(batch)
                    handle.flush()
                    print(f"TEST_INFERENCE={len(probabilities)}/{len(test)}", flush=True)
        partial.rename(output / "test_photo_predictions.csv")
        metrics, groups = evaluate_predictions(test, probabilities)
        write_csv(output / "test_group_predictions.csv", groups)
        uncertainty = cluster_bootstrap(groups, plan["bootstrap_replicates"], plan["seed"])
        if sha256(checkpoint_path) != plan["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint changed during evaluation.")
        summary = {"status": "FINAL_TEST_COMPLETE", "training_job_id": plan["training_job_id"],
                   "pbs_job_id": os.environ["PBS_JOBID"], "best_epoch": plan["best_epoch"],
                   "checkpoint_sha256": plan["checkpoint_sha256"], "decision_threshold": .5,
                   "test": metrics, "group_uncertainty": uncertainty, "test_photos": len(test),
                   "model_update": "NONE", "threshold_selection": "NONE", "device": "cpu",
                   "elapsed_seconds_after_release": time.monotonic() - started,
                   "interpretation": "Held-out agreement with source community labels; no DNA-ground-truth or calibrated-certainty claim."}
        write_json(output / "summary.json", summary)
        write_json(output / "OUTPUT_SHA256.json", {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
        write_json(marker / "completion.json", {"status": "FINAL_TEST_COMPLETE", "output_directory": str(output),
                                                "summary_sha256": sha256(output / "summary.json")})
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as error:
        write_json(marker / "failure.json", {"status": "STOPPED_AFTER_TEST_RELEASE", "error": str(error),
                                             "output_directory": str(output), "automatic_retry": "DISABLED"})
        raise


if __name__ == "__main__":
    main()
