"""We reproduce the published metrics from saved scores, without running the model."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from baseline_data import evaluate_predictions, read_csv
from evaluate_baseline import cluster_bootstrap


def main():
    rows = [r for r in read_csv(ROOT / "data/photo_splits.csv") if r["split"] == "test"]
    saved = read_csv(ROOT / "results/test_photo_predictions.csv")
    if len(rows) != len(saved) or any(
        any(a[key] != b[key] for key in ("photo_id", "class_id", "component_id", "evaluation_group_id"))
        for a, b in zip(rows, saved)
    ):
        raise ValueError("Saved predictions do not match the published test split.")
    metrics, groups = evaluate_predictions(rows, [float(r["p_ruderatus"]) for r in saved])
    print(json.dumps({"test": metrics, "group_uncertainty": cluster_bootstrap(groups, 2000, 42)}, indent=2))


if __name__ == "__main__":
    main()
