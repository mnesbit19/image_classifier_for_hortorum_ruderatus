"""We train one fixed EfficientNet baseline on an allocated GPU; the test stays sealed."""

import argparse
import json
import os
import platform
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from baseline_data import (CLASSES, PREPROCESS, evaluate_predictions, preprocess_image,
                           project_path, sampling_weights, sha256, verify_dataset,
                           write_csv, write_json)


class PhotoDataset:
    def __init__(self, root, rows, seed=42, augment=False):
        self.root, self.rows, self.seed, self.augment = root, rows, seed, augment
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        rng = random.Random(f"{self.seed}|{self.epoch}|{row['photo_id']}") if self.augment else None
        return preprocess_image(project_path(self.root, row["file_path"]), rng), int(row["class_id"])


def infer(model, loader, device):
    import torch
    probabilities = []
    model.eval()
    with torch.inference_mode():
        for images, _ in loader:
            probabilities.extend(model(images.to(device, non_blocking=True)).softmax(dim=1)[:, 1].cpu().tolist())
    return probabilities


def model_from_checkpoint(checkpoint, device="cpu"):
    import torch
    from torchvision.models import efficientnet_b0
    if checkpoint.get("schema_version") != 1 or checkpoint.get("architecture") != "efficientnet_b0":
        raise ValueError("Unsupported checkpoint.")
    if checkpoint["classes"] != CLASSES or checkpoint["preprocess"] != PREPROCESS or checkpoint.get("synthetic_only", True):
        raise ValueError("Checkpoint mapping, preprocessing or training status is invalid.")
    # We restore our saved model without downloading ImageNet weights again.
    model = efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/baseline_v1.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("PBS_JOBID"):
        raise SystemExit("STOP: training requires the submitted HPC GPU job.")
    root = Path(__file__).resolve().parents[1]
    if root.as_posix() != "/rds/general/user/mln19/home/ml_rudvhort":
        raise SystemExit("STOP: unexpected HPC project root.")

    import numpy as np
    import PIL
    import torch
    import torchvision
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise SystemExit("STOP: CUDA GPU unavailable; no CPU fallback for training.")
    expected = config["software"]
    if torch.__version__.split("+")[0] != expected["torch"] or torchvision.__version__.split("+")[0] != expected["torchvision"]:
        raise SystemExit("STOP: unexpected torch/torchvision versions.")
    if PIL.__version__ != expected["Pillow"] or np.__version__ != expected["numpy"]:
        raise SystemExit("STOP: unexpected image/numerical package versions.")
    if config["test_policy"] != "sealed_until_separate_evaluation" or config["architecture"] != "efficientnet_b0":
        raise SystemExit("STOP: unsupported baseline protocol.")
    rows = verify_dataset(root, config)
    train = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    output = args.output_dir.resolve()
    if not output.is_relative_to(root / "runs" / "baseline"):
        raise SystemExit("STOP: output must be a new directory under runs/baseline.")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    write_json(output / "source_sha256.json", {name: sha256(root / "code" / name)
                                               for name in ("train_baseline.py", "baseline_data.py", "predict_photo.py")})
    shutil.copyfile(root / config["dataset_dir"] / "summary.json", output / "split_summary.json")
    started = time.monotonic()
    # Our fit and development sets are verified again on the compute node.
    # We do not open any test photo in this job.
    for i, row in enumerate(train + validation, 1):
        path = project_path(root, row["file_path"])
        if path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            raise RuntimeError("Raw image checksum mismatch: " + row["photo_id"])
        if i % 1000 == 0:
            print(f"IMAGE_PREFLIGHT={i}/{len(train) + len(validation)}", flush=True)
    write_json(output / "image_preflight.json", {"status": "PASS", "checked_photos": len(train) + len(validation),
                                                "test_images_opened": 0})
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = "cuda"
    if config["pretrained_weights"] != "EfficientNet_B0_Weights.IMAGENET1K_V1":
        raise ValueError("Unsupported initial weights.")
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model = model.to(device)
    weights_file = Path(torch.hub.get_dir()) / "checkpoints" / "efficientnet_b0_rwightman-7f5810bc.pth"
    write_json(output / "runtime.json", {
        "utc": datetime.now(timezone.utc).isoformat(), "pbs_job_id": os.environ["PBS_JOBID"],
        "python": platform.python_version(), "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__), "Pillow": PIL.__version__, "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "pretrained_file": weights_file.name, "pretrained_sha256": sha256(weights_file),
        "precision": "float32", "deterministic_algorithms": True,
    })
    train_data = PhotoDataset(root, train, seed, augment=True)
    val_data = PhotoDataset(root, validation)
    sampler = torch.utils.data.WeightedRandomSampler(
        sampling_weights(train), num_samples=len(train), replacement=True,
        generator=torch.Generator().manual_seed(seed))
    loader_options = {"batch_size": config["batch_size"], "num_workers": config["num_workers"],
                      "pin_memory": True, "persistent_workers": False}
    train_loader = torch.utils.data.DataLoader(train_data, sampler=sampler, **loader_options)
    val_loader = torch.utils.data.DataLoader(val_data, shuffle=False, **loader_options)
    history = []
    best_score, best_loss = -1.0, float("inf")
    best_epoch = None
    epoch = 0
    for phase in config["phases"]:
        head_only = phase["name"] == "classifier"
        for parameter in model.features.parameters():
            parameter.requires_grad = not head_only
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                     lr=phase["learning_rate"], weight_decay=config["weight_decay"])
        for _ in range(phase["epochs"]):
            epoch += 1
            train_data.epoch = epoch
            model.train()
            if head_only:
                model.features.eval()
            total_loss, samples = 0.0, 0
            for images, labels in train_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss.")
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(labels)
                samples += len(labels)
            probabilities = infer(model, val_loader, device)
            metrics, groups = evaluate_predictions(validation, probabilities)
            primary = metrics["observation_exact_group"]
            score, val_loss = primary["balanced_accuracy"], primary["log_loss"]
            improved = score > best_score or (score == best_score and val_loss < best_loss)
            record = {"epoch": epoch, "phase": phase["name"], "training_loss": total_loss / samples,
                      "validation": metrics, "selected_best": improved, "elapsed_seconds": time.monotonic() - started}
            history.append(record)
            with (output / "epochs.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            checkpoint = {
                "schema_version": 1, "synthetic_only": False, "architecture": config["architecture"],
                "classes": CLASSES, "preprocess": PREPROCESS, "state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch, "phase": phase["name"],
                "config": config, "config_sha256": sha256(args.config),
                "validation_metrics": metrics, "test_model_evaluation": "NOT_RUN",
                "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all(),
                "sampler_rng_state": sampler.generator.get_state(), "python_rng_state": random.getstate(),
            }
            # Atomic replacement is confined to this fresh run's own checkpoints.
            torch.save(checkpoint, output / "last.pt.tmp")
            (output / "last.pt.tmp").replace(output / "last.pt")
            if improved:
                best_score, best_loss, best_epoch = score, val_loss, epoch
                shutil.copyfile(output / "last.pt", output / "best.pt.tmp")
                (output / "best.pt.tmp").replace(output / "best.pt")
            print(f"EPOCH={epoch}; PHASE={phase['name']}; TRAIN_LOSS={total_loss / samples:.5f}; "
                  f"VAL_GROUP_BALANCED_ACCURACY={score:.5f}; BEST={improved}", flush=True)
    restored = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model = model_from_checkpoint(restored, device)
    probabilities = infer(model, val_loader, device)
    metrics, groups = evaluate_predictions(validation, probabilities)
    if abs(metrics["observation_exact_group"]["balanced_accuracy"] - best_score) > 1e-12:
        raise RuntimeError("Best-checkpoint validation round trip disagrees.")
    write_csv(output / "validation_group_predictions.csv", groups)
    write_csv(output / "validation_photo_predictions.csv", [
        {"photo_id": row["photo_id"], "class_id": row["class_id"], "component_id": row["component_id"],
         "evaluation_group_id": row["evaluation_group_id"], "p_ruderatus": probability}
        for row, probability in zip(validation, probabilities, strict=True)])
    final = {"status": "BASELINE_TRAINING_COMPLETE", "best_epoch": best_epoch,
             "epochs": epoch, "validation": metrics, "checkpoint_round_trip": "PASS",
             "best_checkpoint_sha256": sha256(output / "best.pt"),
             "test_model_evaluation": "NOT_RUN", "training_photos": len(train), "validation_photos": len(validation),
             "elapsed_seconds": time.monotonic() - started,
             "next_stage": "Separate one-time held-out test evaluation of the frozen best checkpoint."}
    write_json(output / "summary.json", final)
    write_json(output / "OUTPUT_SHA256.json", {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
