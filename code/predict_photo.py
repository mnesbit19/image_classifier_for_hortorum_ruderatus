"""We predict one new photo using our saved two-species checkpoint, without downloads."""

import argparse
from pathlib import Path

from baseline_data import CLASSES, preprocess_image, sha256, write_json
from train_baseline import model_from_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("STOP: choose a new prediction output filename.")
    import torch
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = model_from_checkpoint(checkpoint, args.device)
    tensor = torch.from_numpy(preprocess_image(args.image)).unsqueeze(0).to(args.device)
    with torch.inference_mode():
        scores = model(tensor).softmax(dim=1)[0].cpu().tolist()
    result = {
        "image": str(args.image), "image_sha256": sha256(args.image),
        "checkpoint_sha256": sha256(args.checkpoint),
        "prediction": max(CLASSES, key=lambda species: scores[CLASSES[species]]),
        "scores": {species: scores[label] for species, label in CLASSES.items()},
        "scope": "Assumes a bee from this species pair; scores are not calibrated identification certainty.",
        "reference_labels": "iNaturalist community labels, not DNA ground truth",
    }
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
