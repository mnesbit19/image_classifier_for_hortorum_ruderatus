"""We bind our first classifier to fixed data, full-frame inputs and grouped metrics."""

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = {"Bombus hortorum": 0, "Bombus ruderatus": 1}
SPLITS = ("train", "validation", "test")
PREPROCESS = {
    "name": "exif_rgb_full_frame_pad_v1", "size": 224,
    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    "interpolation": "bicubic", "padding_rgb": [124, 116, 104],
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields=None):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def project_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Path escapes our project: " + str(relative))
    return path


def verify_dataset(root, config):
    directory = project_path(root, config["dataset_dir"])
    expected = config["dataset_sha256"]
    for name, digest in expected.items():
        if sha256(project_path(directory, name)) != digest:
            raise ValueError("Dataset checksum mismatch: " + name)
    rows = read_csv(directory / "photos.csv")
    if len({r["photo_id"] for r in rows}) != len(rows):
        raise ValueError("Training table repeats a photo ID.")
    if config["classes"] != CLASSES or config["preprocess"] != PREPROCESS:
        raise ValueError("Unsupported class mapping or preprocessing.")
    for row in rows:
        if CLASSES.get(row["species"]) != int(row["class_id"]) or row["split"] not in SPLITS:
            raise ValueError("Invalid species or split.")
        project_path(root, row["file_path"])
    return rows


def preprocess_image(path, augment_rng=None):
    # We keep the whole frame: cropping could remove a bee near the edge.
    import numpy as np
    from PIL import Image, ImageEnhance, ImageOps

    with Image.open(path) as opened:
        if getattr(opened, "n_frames", 1) != 1:
            raise ValueError("Our first baseline accepts single-frame images only.")
        image = ImageOps.exif_transpose(opened).convert("RGBA")
        background = Image.new("RGBA", image.size, tuple(PREPROCESS["padding_rgb"]) + (255,))
        image = Image.alpha_composite(background, image).convert("RGB")
        image = ImageOps.contain(image, (224, 224), Image.Resampling.BICUBIC)
        if augment_rng is not None:
            if augment_rng.random() < 0.5:
                image = ImageOps.mirror(image)
            image = ImageEnhance.Brightness(image).enhance(augment_rng.uniform(0.9, 1.1))
            image = ImageEnhance.Contrast(image).enhance(augment_rng.uniform(0.9, 1.1))
            image = ImageEnhance.Color(image).enhance(augment_rng.uniform(0.9, 1.1))
        canvas = Image.new("RGB", (224, 224), tuple(PREPROCESS["padding_rgb"]))
        canvas.paste(image, ((224 - image.width) // 2, (224 - image.height) // 2))
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    array = (array - np.asarray(PREPROCESS["mean"], dtype=np.float32)) / np.asarray(PREPROCESS["std"], dtype=np.float32)
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def sampling_weights(rows):
    # Each class has equal expected weight; each observation/exact group within
    # that class also has equal weight regardless of its number of views.
    groups = defaultdict(set)
    group_sizes = Counter()
    for row in rows:
        groups[int(row["class_id"])].add(row["evaluation_group_id"])
        group_sizes[(int(row["class_id"]), row["evaluation_group_id"])] += 1
    if set(groups) != {0, 1}:
        raise ValueError("Both classes are required in training.")
    return [1.0 / (len(groups[int(r["class_id"])]) * group_sizes[(int(r["class_id"]), r["evaluation_group_id"])]) for r in rows]


def binary_metrics(labels, probabilities):
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("Metrics require equally sized nonempty inputs.")
    matrix = [[0, 0], [0, 0]]
    losses = []
    for label, probability in zip(labels, probabilities):
        if label not in (0, 1) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid prediction.")
        # A tie uses class 0, matching two-output argmax.
        prediction = int(probability > 0.5)
        matrix[label][prediction] += 1
        p = min(max(probability, 1e-12), 1 - 1e-12)
        losses.append(-math.log(p if label else 1 - p))
    per_class = {}
    for label in (0, 1):
        true_positive = matrix[label][label]
        support = sum(matrix[label])
        predicted = sum(matrix[k][label] for k in (0, 1))
        recall = true_positive / support if support else None
        precision = true_positive / predicted if predicted else 0.0
        f1 = 2 * true_positive / (support + predicted) if support + predicted else 0.0
        per_class[str(label)] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
    recalls = [v["recall"] for v in per_class.values()]
    return {
        "n": len(labels), "confusion_true_rows_predicted_columns": matrix,
        "accuracy": (matrix[0][0] + matrix[1][1]) / len(labels),
        "balanced_accuracy": sum(recalls) / 2 if all(v is not None for v in recalls) else None,
        "log_loss": sum(losses) / len(losses), "per_class": per_class,
    }


def grouped_predictions(rows, probabilities, field="evaluation_group_id"):
    grouped = defaultdict(list)
    for row, probability in zip(rows, probabilities, strict=True):
        grouped[row[field]].append((row, probability))
    results = []
    for group_id, values in sorted(grouped.items()):
        labels = {int(row["class_id"]) for row, _ in values}
        components = {row["component_id"] for row, _ in values}
        if len(labels) != 1 or len(components) != 1:
            raise ValueError("Mixed labels or split components in an evaluation group.")
        results.append({"group_id": group_id, "component_id": next(iter(components)),
                        "class_id": next(iter(labels)), "n_photos": len(values),
                        "p_ruderatus": sum(p for _, p in values) / len(values)})
    return results


def evaluate_predictions(rows, probabilities):
    groups = grouped_predictions(rows, probabilities)
    metrics = {
        "photo": binary_metrics([int(r["class_id"]) for r in rows], probabilities),
        "observation_exact_group": binary_metrics([r["class_id"] for r in groups], [r["p_ruderatus"] for r in groups]),
    }
    return metrics, groups
