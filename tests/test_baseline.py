"""We check split isolation, source preservation, whole-frame inputs and group scoring."""

import math
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from baseline_data import (binary_metrics, evaluate_predictions, preprocess_image, project_path,
                           sampling_weights)
from prepare_baseline import assign_splits, unique_photos, validate_constraints


def fixture():
    rows = []
    for c in (0, 1):
        species = ("Bombus hortorum", "Bombus ruderatus")[c]
        for i in range(30):
            obs = str(c * 100 + i)
            rows.append({"observation_id": obs, "observer_id": obs, "observation_group_id": "group" + obs,
                         "photo_id": obs, "sha256": "hash" + obs, "species": species, "class_id": str(c),
                         "source_species": species, "source_class_id": str(c), "exact_match_group_ids": "",
                         "file_path": "data/" + obs, "bytes": "1", "source_dataset": "source",
                         "source_attempt": "attempt", "source_download_status": "INCOMPLETE" if c == 0 else "DOWNLOAD_COMPLETE",
                         "source_acceptance": "saved", "photo_license": "cc-by-nc", "attribution": "Author"})
    return rows


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_root = Path(__file__).resolve().parents[1] / "tmp" / "baseline_tests"
        cls.temp_root.mkdir(parents=True, exist_ok=True)

    def test_transitive_constraints_and_order_independence(self):
        rows = fixture()
        rows[1]["observer_id"] = rows[0]["observer_id"]
        rows[2]["observation_group_id"] = rows[1]["observation_group_id"]
        rows[3]["sha256"] = rows[2]["sha256"]
        rows[3]["exact_match_group_ids"] = "pixelA;pixelB"
        rows[34]["exact_match_group_ids"] = "pixelB"
        fractions = {"train": .8, "validation": .1, "test": .1}
        links = assign_splits(rows, 42, fractions)
        reverse = assign_splits(list(reversed(rows)), 42, fractions)
        mapping = {r["observation_id"]: (r["split"], r["component_id"]) for r in links}
        self.assertEqual(mapping, {r["observation_id"]: (r["split"], r["component_id"]) for r in reverse})
        self.assertEqual(len({mapping[rows[i]["observation_id"]] for i in (0, 1, 2, 3, 34)}), 1)
        for original, assigned in zip(rows, links):
            self.assertTrue(all(assigned[k] == v for k, v in original.items()))

    def test_rejects_missing_observer_and_relabel(self):
        for key, value in (("observer_id", ""), ("source_class_id", "8")):
            rows = fixture()
            rows[0][key] = value
            with self.assertRaises(ValueError):
                assign_splits(rows, 42, {"train": .8, "validation": .1, "test": .1})

    def test_shared_photo_used_once_and_source_status_unchanged(self):
        rows = fixture()
        duplicate = dict(rows[0], observation_id="extra")
        rows.append(duplicate)
        links = assign_splits(rows, 42, {"train": .8, "validation": .1, "test": .1})
        photos = unique_photos(links)
        self.assertEqual(len(photos), 60)
        photo = next(r for r in photos if r["photo_id"] == "0")
        self.assertEqual(photo["observation_ids"], "0|extra")
        self.assertEqual(photo["source_download_status"], "INCOMPLETE")
        broken = [dict(r) for r in links]
        broken[-1]["split"] = "test" if broken[0]["split"] != "test" else "train"
        with self.assertRaises(ValueError):
            validate_constraints(broken)

    def test_class_and_group_sampling_mass(self):
        rows = [{"class_id": str(c), "evaluation_group_id": group}
                for c, group, n in ((0, "a", 10), (0, "b", 1), (1, "c", 2)) for _ in range(n)]
        masses = defaultdict(float)
        for row, weight in zip(rows, sampling_weights(rows)):
            masses[(row["class_id"], row["evaluation_group_id"])] += weight
        self.assertAlmostEqual(masses[("0", "a")], .5)
        self.assertAlmostEqual(masses[("0", "b")], .5)
        self.assertAlmostEqual(masses[("1", "c")], 1)

    def test_metrics_use_groups_and_fixed_threshold(self):
        rows = [{"class_id": str(c), "evaluation_group_id": g, "component_id": g}
                for c, g in ((0, "a"), (0, "a"), (1, "b"), (1, "c"))]
        metrics, groups = evaluate_predictions(rows, [.1, .3, .9, .5])
        self.assertEqual(len(groups), 3)
        self.assertEqual(metrics["observation_exact_group"]["confusion_true_rows_predicted_columns"], [[1, 0], [1, 1]])
        self.assertEqual(metrics["observation_exact_group"]["balanced_accuracy"], .75)
        self.assertEqual(binary_metrics([0, 1], [0, 0])["balanced_accuracy"], .5)
        self.assertTrue(math.isfinite(binary_metrics([0, 1], [0, 1])["log_loss"]))
        with self.assertRaises(ValueError):
            evaluate_predictions(rows, [.1])

    def test_full_frame_preprocessing_keeps_edges_and_matches_padding(self):
        import numpy as np
        from PIL import Image, ImageDraw
        with tempfile.TemporaryDirectory(dir=self.temp_root) as tmp:
            path = Path(tmp) / "wide.png"
            image = Image.new("RGB", (800, 200), (255, 255, 255))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 150, 199), fill="red")
            draw.rectangle((650, 0, 799, 199), fill="blue")
            image.save(path)
            tensor = preprocess_image(path)
            self.assertEqual(tensor.shape, (3, 224, 224))
            self.assertEqual(tensor.dtype, np.float32)
            self.assertGreater(float(tensor[0, 112, 0]), 2)
            self.assertGreater(float(tensor[2, 112, 223]), 2)
            self.assertLess(abs(float(tensor[0, 0, 0])), .02)
            self.assertTrue(np.array_equal(tensor, preprocess_image(path)))

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as tmp:
            with self.assertRaises(ValueError):
                project_path(tmp, "../outside")


if __name__ == "__main__":
    unittest.main()
