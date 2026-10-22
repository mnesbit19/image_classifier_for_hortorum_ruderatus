"""We measure CPU/reference differences on development photos; we never release the test."""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

from baseline_data import preprocess_image, project_path, read_csv, sha256, write_csv, write_json
from evaluate_baseline import verify_checkpoint_metadata, verify_plan
from train_baseline import model_from_checkpoint


def validation_context_batches(rows, photo_ids, batch_size):
    """We reproduce the original validation batch context without selecting test images."""
    if batch_size < 1 or not photo_ids or len(set(photo_ids)) != len(photo_ids):
        raise ValueError("Require a positive batch size and distinct development photo IDs.")
    validation = [row for row in rows if row["split"] == "validation"]
    positions = {row["photo_id"]: i for i, row in enumerate(validation)}
    if len(positions) != len(validation) or any(photo_id not in positions for photo_id in photo_ids):
        raise ValueError("Every diagnostic target must be a unique validation photo.")
    starts = sorted({positions[photo_id] // batch_size * batch_size for photo_id in photo_ids})
    return [validation[start:start + batch_size] for start in starts]


def score_comparison(actual, reference, tolerance):
    if (not math.isfinite(tolerance) or tolerance <= 0 or
            any(not math.isfinite(p) or not 0 <= p <= 1 for p in (actual, reference))):
        raise ValueError("Require finite probabilities and a positive finite tolerance.")
    difference = abs(actual - reference)
    return {"absolute_difference": difference, "within_original_tolerance": difference <= tolerance,
            "same_argmax_class": (actual > .5) == (reference > .5)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/test_baseline_v1.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not os.environ.get("PBS_JOBID") or root.as_posix() != "/rds/general/user/mln19/home/ml_rudvhort":
        raise SystemExit("STOP: use our 11 validation-runtime diagnostic on HPC compute.")
    plan = json.loads(args.config.read_text())
    training, config, rows, _ = verify_plan(root, plan)
    marker = project_path(root, plan["release_marker_dir"])
    if marker.exists():
        raise SystemExit("STOP: newer test-release evidence exists; inspect it before further work.")
    ids = plan["validation_parity_photo_ids"]
    contexts = validation_context_batches(rows, ids, config["batch_size"])
    by_id = {row["photo_id"]: row for context in contexts for row in context}
    output = args.output_dir.resolve()
    if not output.is_relative_to(root / "runs/evaluation"):
        raise SystemExit("STOP: diagnostic output must be under runs/evaluation.")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", plan)
    write_json(output / "source_sha256.json", {name: sha256(root / "code" / name) for name in
               ("diagnose_validation_runtime.py", "evaluate_baseline.py", "baseline_data.py",
                "train_baseline.py", "predict_photo.py")})

    import numpy as np
    import PIL
    import torch
    import torchvision

    versions = {"torch": str(torch.__version__).split("+")[0],
                "torchvision": str(torchvision.__version__).split("+")[0],
                "Pillow": PIL.__version__, "numpy": np.__version__}
    if any(versions[key] != config["software"][key] for key in versions):
        raise RuntimeError("Diagnostic software differs from the trained baseline.")
    torch.set_num_threads(1)
    torch.manual_seed(plan["seed"])
    torch.use_deterministic_algorithms(True)
    checkpoint_path = training / "artifacts/best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    verify_checkpoint_metadata(checkpoint, config, plan)
    model = model_from_checkpoint(checkpoint, "cpu")
    write_json(output / "runtime.json", {
        "pbs_job_id": os.environ["PBS_JOBID"], "host": platform.node(),
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": versions, "torch_full_version": str(torch.__version__),
        "torch_configuration": torch.__config__.show(), "threads": torch.get_num_threads(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "saved_gpu_runtime": json.loads((training / "artifacts/runtime.json").read_text()),
        "device": "cpu", "test_images_opened": 0,
    })
    # We verify bytes for validation context photos only. Metadata verification
    # above reads the frozen split table, but never opens a test image.
    for row in by_id.values():
        path = project_path(root, row["file_path"])
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            raise RuntimeError("Validation photo changed: " + row["photo_id"])
    expected = {row["photo_id"]: float(row["p_ruderatus"]) for row in
                read_csv(training / "artifacts/validation_photo_predictions.csv")}
    single, tensors = {}, {}
    tolerance = plan["validation_parity_absolute_tolerance"]
    with torch.inference_mode():
        for photo_id in ids:
            array = preprocess_image(project_path(root, by_id[photo_id]["file_path"]))
            tensors[photo_id] = hashlib.sha256(array.tobytes()).hexdigest()
            probability = float(model(torch.from_numpy(array).unsqueeze(0)).softmax(dim=1)[0, 1])
            single[photo_id] = probability
            difference = score_comparison(probability, expected[photo_id], tolerance)
            print(f"VALIDATION_ONLY_SINGLE PHOTO={photo_id} CPU={probability:.10f} "
                  f"SAVED_GPU={expected[photo_id]:.10f} "
                  f"ABS_DIFF={difference['absolute_difference']:.10f}", flush=True)
        batched, context_sizes = {}, {}
        for context in contexts:
            images = torch.stack([torch.from_numpy(preprocess_image(project_path(root, row["file_path"])))
                                  for row in context])
            probabilities = model(images).softmax(dim=1)[:, 1].tolist()
            for row, probability in zip(context, probabilities, strict=True):
                if row["photo_id"] in single:
                    batched[row["photo_id"]] = probability
                    context_sizes[row["photo_id"]] = len(context)
            print(f"VALIDATION_ONLY_CONTEXT_BATCH={len(context)}", flush=True)
    comparisons = []
    for photo_id in ids:
        entry = {"photo_id": photo_id, "saved_gpu": expected[photo_id], "cpu_single": single[photo_id],
                 "cpu_original_batch": batched[photo_id], "original_batch_size": context_sizes[photo_id],
                 "preprocessed_tensor_sha256": tensors[photo_id]}
        for name, actual, reference in (("single_vs_gpu", single[photo_id], expected[photo_id]),
                                        ("batch_vs_gpu", batched[photo_id], expected[photo_id]),
                                        ("single_vs_batch", single[photo_id], batched[photo_id])):
            entry.update({name + "_" + key: value for key, value in score_comparison(actual, reference, tolerance).items()})
        comparisons.append(entry)
    write_csv(output / "validation_scores.csv", comparisons)

    # We exercise the public prediction tool on the first two fixed development
    # photos, including the photo that stopped job 4428754. We record failures
    # without opening the final test or changing the original tolerance.
    cli_checks = []
    for photo_id in ids[:2]:
        destination = output / ("predictor_" + photo_id + ".json")
        command = [sys.executable, str(root / "code/predict_photo.py"), "--checkpoint", str(checkpoint_path),
                   "--image", str(project_path(root, by_id[photo_id]["file_path"])), "--output", str(destination)]
        result = subprocess.run(command, capture_output=True, text=True)
        entry = {"photo_id": photo_id, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        if result.returncode == 0:
            prediction = json.loads(destination.read_text())
            probability = float(prediction["scores"]["Bombus ruderatus"])
            entry.update({"p_ruderatus": probability,
                          "versus_direct_cpu": score_comparison(probability, single[photo_id], tolerance),
                          "versus_saved_gpu": score_comparison(probability, expected[photo_id], tolerance)})
        cli_checks.append(entry)
    write_json(output / "cli_checks.json", cli_checks)
    if sha256(checkpoint_path) != plan["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed during diagnostics.")
    summary = {"status": "VALIDATION_RUNTIME_DIAGNOSTIC_COMPLETE", "pbs_job_id": os.environ["PBS_JOBID"],
               "targets": ids, "validation_context_photos": len(by_id), "context_batches": len(contexts),
               "original_absolute_tolerance": tolerance, "test_images_opened": 0,
               "test_release": "NO_ACTION", "model_update": "NONE", "threshold_update": "NONE",
               "checkpoint_sha256": plan["checkpoint_sha256"],
               "comparisons": {name: {
                   "max_absolute_difference": max(row[name + "_absolute_difference"] for row in comparisons),
                   "all_within_original_tolerance": all(row[name + "_within_original_tolerance"] for row in comparisons),
                   "all_same_argmax_class": all(row[name + "_same_argmax_class"] for row in comparisons)}
                   for name in ("single_vs_gpu", "batch_vs_gpu", "single_vs_batch")},
               "cli_direct_cpu_agreement": all(entry["returncode"] == 0 and
                     entry["versus_direct_cpu"]["within_original_tolerance"] for entry in cli_checks),
               "interpretation": "Diagnostic completion is not a parity pass or permission to release the test."}
    write_json(output / "summary.json", summary)
    write_json(output / "OUTPUT_SHA256.json", {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
