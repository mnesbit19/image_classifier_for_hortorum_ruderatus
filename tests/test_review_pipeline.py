"""We test candidate retrieval, label preservation, source binding and explicit holds."""

import contextlib
import csv
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "tests"))
import prepare_curation as c
import scan_near_duplicates as n
import check_images as q
import test_check_images as fixtures


def settings():
    return {"algorithm": n.ALGORITHM, "phash_radius": 6, "dhash_radius": 10,
            "center_crop_fraction": 0.75, "min_luma_stddev": 2.0, "max_candidate_pairs": 100000}


def pattern():
    image = Image.new("RGB", (240, 160), "#385d29")
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 20, 110, 60), fill="#ffdd11")
    draw.ellipse((100, 50, 210, 135), fill="#aa2244")
    draw.polygon([(20, 140), (80, 75), (130, 150)], fill="#2288ff")
    draw.line((0, 90, 240, 15), fill="white", width=5)
    return image


class ReviewPipeline(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="review_test_", dir=ROOT / "tmp")
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.assertTrue(self.root.resolve().is_relative_to((ROOT / "tmp").resolve()))
        self.tmp.cleanup()

    def fixture(self):
        path, rows = fixtures.ImageChecks.cohort_fixture(self)
        configdir = self.root / "config"; configdir.mkdir()
        accepted = configdir / "cohorts.json"; accepted.write_bytes(path.read_bytes())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(q.run(self.root, accepted), 0)
        folder = next((self.root / "data/derived/quality").iterdir())
        config = {"schema_version": 1, "cohorts_file": "config/cohorts.json",
            "qc": {"directory": folder.relative_to(self.root).as_posix(), "job_id": None,
                "summary_sha256": c.sha(folder / "summary.json"),
                "files_sha256": {p.relative_to(folder).as_posix(): c.sha(p) for p in folder.rglob("*") if p.is_file()}},
            "near_duplicates": settings()}
        target = configdir / "review.json"; q.write_json(target, config)
        return target, config, folder, rows

    def test_index_matches_brute_force_with_boundary_neighbours(self):
        rng = random.Random(83)
        values = [rng.getrandbits(63) for _ in range(300)]
        index = n.HashIndex(6)
        for i, value in enumerate(values):
            index.add({"phash": value, "id": i})
        queries = [values[i] ^ sum(1 << bit for bit in rng.sample(range(63), 6)) for i in range(20)]
        for query in queries:
            actual = {entry["id"] for entry, _ in index.query(query)}
            expected = {i for i, value in enumerate(values) if (query ^ value).bit_count() <= 6}
            self.assertEqual(actual, expected)

    def test_resized_recompressed_rotated_and_mirrored_images_are_candidates(self):
        with pattern() as image:
            originals = n.image_features(image, settings())
            buf = io.BytesIO()
            image.resize((180, 120)).save(buf, "JPEG", quality=86)
            with Image.open(io.BytesIO(buf.getvalue())) as other:
                altered = n.image_features(other, settings())
            for method in (Image.Transpose.ROTATE_90, Image.Transpose.FLIP_LEFT_RIGHT):
                with image.transpose(method) as other:
                    transformed = n.image_features(other, settings())
                self.assertTrue(any((a["phash"] ^ b["phash"]).bit_count() <= 6 and
                    (a["dhash"] ^ b["dhash"]).bit_count() <= 10 for a in originals if a["view"] == "full/identity" for b in transformed))
            self.assertTrue(any((a["phash"] ^ b["phash"]).bit_count() <= 6 and
                (a["dhash"] ^ b["dhash"]).bit_count() <= 10 for a in originals if a["view"] == "full/identity" for b in altered))

    def test_uniform_view_is_held_instead_of_creating_similarity_evidence(self):
        with Image.new("RGB", (100, 100), "white") as image:
            self.assertEqual(n.image_features(image, settings()), [])

    def test_qc_table_tampering_is_rejected(self):
        _, config, folder, _ = self.fixture()
        path = folder / "photo_qc.csv"
        path.write_bytes(path.read_bytes().replace(b"pass", b"fail", 1))
        with self.assertRaisesRegex(ValueError, "Pinned QC input hash"):
            c.load_qc(self.root, config)

    def test_manifest_preserves_all_links_labels_and_incomplete_history(self):
        config, _, _, rows = self.fixture()
        with contextlib.redirect_stdout(io.StringIO()):
            output = c.run(self.root, config, False)
        manifest = q.csv_rows((output / "curation_manifest.csv").read_bytes())
        self.assertEqual(len(manifest), len(rows))
        for row, original in zip(manifest, rows):
            self.assertEqual(row["source_species"], original["species"])
            self.assertEqual(row["curation_decision"], "pending")
            self.assertEqual(row["reviewed_species"], "")
            self.assertEqual(row["bbox_left"], "")
            self.assertEqual(row["split"], "")
        self.assertEqual(len({r["observation_group_id"] for r in manifest}), 1)
        self.assertTrue(all(r["source_download_status"] == "INCOMPLETE" for r in manifest if r["species"] == "Bombus hortorum"))
        source = next((self.root / "data/derived/quality").iterdir()) / "sources/hor/summary.json"
        self.assertEqual(c.read_json(source)["status"], "INCOMPLETE")

    def test_exact_transitive_constraints_do_not_assign_individuals_or_splits(self):
        rows = [{"observation_id": str(i)} for i in range(1, 5)]
        exact = [{"observation_ids": ["1", "2"]}, {"observation_ids": ["2", "3"]}]
        mapping, components = c.observation_groups(rows, exact)
        self.assertEqual(mapping["1"], mapping["3"])
        self.assertNotEqual(mapping["3"], mapping["4"])
        self.assertTrue(all(not g["individual_identity_established"] and g["split"] == "" for g in components))

    def test_animation_is_held_and_changed_bytes_are_not_hashed(self):
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), "red").save(buf, "GIF", save_all=True,
            append_images=[Image.new("RGB", (8, 8), "blue")])
        row = fixtures.row_for(self.root, 8, 9, raw=buf.getvalue())
        checked = q.check_photo(self.root, row)
        features, status = n.features_for_photo(self.root, row, checked, settings())
        self.assertEqual((features, status), ([], "held_animation"))
        (self.root / row["file_path"]).write_bytes(buf.getvalue() + b"changed")
        with self.assertRaisesRegex(ValueError, "Raw bytes differ"):
            n.features_for_photo(self.root, row, checked, settings())

    def test_scan_input_error_is_complete_with_issues_not_a_success(self):
        config, _, _, rows = self.fixture()
        raw = self.root / rows[0]["file_path"]
        raw.write_bytes(raw.read_bytes() + b"changed")
        with contextlib.redirect_stdout(io.StringIO()):
            code, output = n.run(self.root, config)
        self.assertEqual(code, 2)
        self.assertEqual(c.read_json(output / "summary.json")["status"], "NEAR_SCAN_COMPLETE_WITH_ISSUES")
        self.assertEqual(c.read_json(output / "summary.json")["photo_status_counts"]["input_error"], 1)

    def test_candidate_overflow_has_failure_receipt_and_no_complete_summary(self):
        config, _, _, _ = self.fixture()
        with patch.object(n, "propose", side_effect=RuntimeError("Candidate-pair limit reached")):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                code, output = n.run(self.root, config)
        self.assertEqual(code, 1)
        self.assertTrue((output / "failure.json").exists())
        self.assertFalse((output / "summary.json").exists())
        self.assertTrue((output / "partial_candidate_pairs.csv").exists())

    def test_cross_species_matches_remain_pending_candidates(self):
        index = n.HashIndex(6)
        photos = {"1": {"species": "Bombus hortorum"}, "2": {"species": "Bombus ruderatus"}}
        checked = {p: {"actual_sha256": p, "pixel_sha256": p} for p in photos}
        byphoto = {"1": {"10"}, "2": {"20"}}
        features = [{"view": "full/identity", "phash": 123456, "dhash": 8888, "luma_stddev": 10}]
        pairs = {}
        n.propose(features, "1", index, photos, checked, byphoto, pairs, settings())
        n.propose(features, "2", index, photos, checked, byphoto, pairs, settings())
        candidate = pairs[("1", "2")][1]
        self.assertTrue(candidate["cross_species"])
        self.assertEqual(candidate["review_status"], "pending")
        self.assertEqual(candidate["review_decision"], "")
        self.assertEqual(photos["2"]["species"], "Bombus ruderatus")

    def test_real_pair_limit_is_not_silently_truncated(self):
        index = n.HashIndex(6)
        photos = {p: {"species": "Bombus hortorum"} for p in ("1", "2", "3")}
        checked = {p: {"actual_sha256": p, "pixel_sha256": p} for p in photos}
        observations = {p: {p} for p in photos}
        features = [{"view": "full/identity", "phash": 123, "dhash": 456, "luma_stddev": 10}]
        pairs = {}; limited = {**settings(), "max_candidate_pairs": 1}
        n.propose(features, "1", index, photos, checked, observations, pairs, limited)
        n.propose(features, "2", index, photos, checked, observations, pairs, limited)
        with self.assertRaisesRegex(RuntimeError, "Candidate-pair limit reached"):
            n.propose(features, "3", index, photos, checked, observations, pairs, limited)
        self.assertEqual(len(pairs), 1)


if __name__ == "__main__":
    unittest.main()
