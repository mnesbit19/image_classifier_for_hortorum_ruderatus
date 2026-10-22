"""We test CUDA, a two-output EfficientNet and checkpoint I/O on synthetic data."""

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get("PBS_JOBID"):
        raise SystemExit("Run through hpc/pbs/02_gpu_smoke.pbs, not on a login node.")

    import torch
    import torchvision
    from torch import nn
    from torchvision.models import efficientnet_b0

    if not torch.cuda.is_available():
        raise SystemExit("FAIL: PyTorch cannot see an allocated CUDA GPU.")
    if not args.output_dir.is_dir():
        raise SystemExit("The job wrapper must create a fresh output directory.")

    torch.manual_seed(42)
    torch.set_num_threads(4)
    # We deliberately avoid internet access and pretrained weight downloads.
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model = model.to("cuda").train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    images = torch.rand(2, 3, 224, 224, device="cuda")
    targets = torch.tensor([0, 1], device="cuda")
    before = model.classifier[1].weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    if tuple(logits.shape) != (2, 2):
        raise RuntimeError("Unexpected classifier output shape.")
    loss = nn.CrossEntropyLoss()(logits, targets)
    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite synthetic loss.")
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    if torch.equal(before, model.classifier[1].weight.detach()):
        raise RuntimeError("Classifier weights did not update.")

    checkpoint = args.output_dir / "synthetic_checkpoint.pt"
    # We never overwrite a previous run's checkpoint.
    with checkpoint.open("xb") as handle:
        torch.save({"state_dict": model.state_dict(), "synthetic_only": True}, handle)
    restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not restored["synthetic_only"]:
        raise RuntimeError("Synthetic checkpoint marker missing.")
    expected = model.classifier[1].weight.detach().cpu()
    if not torch.equal(restored["state_dict"]["classifier.1.weight"], expected):
        raise RuntimeError("Checkpoint round-trip mismatch.")

    report = {
        "status": "PASS",
        "scope": "synthetic GPU and checkpoint test; NOT species training",
        "utc": datetime.now(timezone.utc).isoformat(),
        "pbs_job_id": os.environ["PBS_JOBID"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "loss_on_random_data": float(loss.detach().cpu()),
        "checkpoint_round_trip": "PASS",
        "classifier_update": "PASS",
    }
    with (args.output_dir / "gpu_smoke_report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
