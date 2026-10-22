"""We test corrupt files, orientation, duplicate links and accepted-source binding."""

import collections
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, PngImagePlugin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import check_images as q
from report_download import BUDGET_STOP


def image_bytes(note="", orientation=None):
    im = Image.new("RGB", (12, 8), "green")
    im.putpixel((0, 0), (255, 0, 0))
    info = PngImagePlugin.PngInfo()
    info.add_text("test_note", note)
    options = {"pnginfo": info}
    if orientation:
        exif = im.getexif(); exif[274] = orientation
        options["exif"] = exif
    stream = io.BytesIO(); im.save(stream, "PNG", **options)
    return stream.getvalue()


def row_for(root, photo, obs, species="Bombus ruderatus", raw=None):
    raw = image_bytes() if raw is None else raw
    digest = q.sha(raw)
    relative = f"data/raw/images/{photo}/original/{digest}.png"
    path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
    return {"photo_id": str(photo), "observation_id": str(obs), "file_path": relative,
        "sha256": digest, "bytes": str(len(raw)), "species": species,
        "class_id": "0" if species == "Bombus hortorum" else "1", "status": "downloaded",
        "photo_license": "cc-by", "observer_id": str(obs), "observer_login": "test",
        "export_country": "", "current_scientific_name": species}


def make_source(root, name, rows, partial=False):
    classes = {"Bombus hortorum": 0, "Bombus ruderatus": 1}
    config = {"classes": classes, "data": {"initial_photo_license_allowlist": ["cc-by"]}}
    ids = sorted({int(r["observation_id"]) for r in rows})
    plan = {"observation_ids": ids + ([999] if partial else []),
            "class_mapping": classes, "data_policy": config["data"]}
    plan_raw = json.dumps(plan).encode(); code_raw = b"synthetic test download source\n"
    source = {"name": name, "dataset": name + "_test", "attempt": "attempt_v1", "job_id": name + ".pbs",
        "species": rows[0]["species"], "accept_budget_limited": partial,
        "plan_sha256": q.sha(plan_raw), "download_code_sha256": q.sha(code_raw),
        "expected_counts": {"observations_with_downloads": len(ids),
                            "unique_photo_ids": len({r["photo_id"] for r in rows}), "photo_observation_links": len(rows)}}
    summary = {"dataset": source["dataset"], "attempt": source["attempt"], "pbs_job_id": source["job_id"],
        "plan_sha256": source["plan_sha256"], "code_sha256": source["download_code_sha256"],
        "status": "INCOMPLETE" if partial else "DOWNLOAD_COMPLETE", "error": BUDGET_STOP if partial else None,
        "selected_observations": len(plan["observation_ids"]), "processed_observations": len(ids),
        "photo_status_counts": dict(collections.Counter(r["status"] for r in rows)),
        "eligible_species_counts": dict(collections.Counter(r["species"] for r in rows)),
        "eligible_photo_license_counts": {"cc-by": len(rows)}, "exclusion_counts": {}}
    attempt = root / "data/manifests" / source["dataset"] / source["attempt"]
    attempt.mkdir(parents=True)
    for filename, value in (("summary.json", summary), ("config.json", config)):
        q.write_json(attempt / filename, value)
    stream = io.StringIO(); writer = q.csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    (attempt / "photos.csv").write_text(stream.getvalue(), encoding="utf-8")
    (attempt / "excluded.csv").write_text("reason\n", encoding="utf-8")
    (attempt / "code_snapshot.py").write_bytes(code_raw)
    cache = root / "data/raw/metadata" / source["dataset"]; cache.mkdir(parents=True)
    (cache / "plan.json").write_bytes(plan_raw)
    return source


class ImageChecks(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="qc_test_", dir=ROOT / "tmp")
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_decode_hash_and_corruption_are_distinct(self):
        row = row_for(self.root, 11, 1)
        before = (self.root / row["file_path"]).read_bytes()
        result = q.check_photo(self.root, row)
        self.assertEqual(result["status"], "pass")
        self.assertEqual((result["width"], result["height"]), (12, 8))
        self.assertEqual((self.root / row["file_path"]).read_bytes(), before)
        (self.root / row["file_path"]).write_bytes(before + b"changed")
        self.assertEqual(q.check_photo(self.root, row)["status"], "integrity_error")
        corrupt = row_for(self.root, 12, 2, raw=b"not an image")
        self.assertEqual(q.check_photo(self.root, corrupt)["status"], "decode_error")
        (self.root / row["file_path"]).unlink()
        self.assertEqual(q.check_photo(self.root, row)["status"], "file_error")

    def test_orientation_and_alpha_are_preserved_in_pixel_hash(self):
        oriented = row_for(self.root, 11, 1, raw=image_bytes(orientation=6))
        result = q.check_photo(self.root, oriented)
        self.assertEqual((result["display_width"], result["display_height"]), (8, 12))
        with Image.open(io.BytesIO(image_bytes())) as im:
            rotated = im.transpose(Image.Transpose.ROTATE_270)
            buf = io.BytesIO(); rotated.save(buf, "PNG")
        other = row_for(self.root, 12, 2, raw=buf.getvalue())
        self.assertEqual(result["pixel_sha256"], q.check_photo(self.root, other)["pixel_sha256"])
        hashes = []
        for alpha in (0, 255):
            buf = io.BytesIO(); Image.new("RGBA", (12, 8), (0, 0, 0, alpha)).save(buf, "PNG")
            row = row_for(self.root, 20 + alpha, 3, raw=buf.getvalue())
            hashes.append(q.check_photo(self.root, row)["pixel_sha256"])
        self.assertNotEqual(*hashes)

    def test_multiple_frames_and_limits_do_not_masquerade_as_still_images(self):
        buf = io.BytesIO()
        Image.new("RGB", (4, 4), "red").save(buf, "GIF", save_all=True,
            append_images=[Image.new("RGB", (4, 4), "blue")])
        row = row_for(self.root, 11, 1, raw=buf.getvalue())
        result = q.check_photo(self.root, row)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["frames"], 2)
        self.assertEqual(result["pixel_sha256"], "")
        self.assertIn("multiple_frames", result["review_flags"])
        with patch.object(q, "MAX_FRAMES", 1):
            self.assertEqual(q.check_photo(self.root, row)["status"], "review_limits")
        row["file_path"] = "data/raw/images/../../outside.png"
        self.assertEqual(q.check_photo(self.root, row)["status"], "file_error")

    def cohort_fixture(self):
        a = row_for(self.root, 11, 1, raw=image_bytes("first"))
        b = row_for(self.root, 12, 2, "Bombus hortorum", image_bytes("second"))
        c = {**b, "observation_id": "3", "observer_id": "3", "status": "cached"}
        sources = [make_source(self.root, "rud", [a]), make_source(self.root, "hor", [b, c], partial=True)]
        cohorts = {"schema_version": 1, "cohort_id": "synthetic_test",
                   "classes": {"Bombus hortorum": 0, "Bombus ruderatus": 1}, "sources": sources}
        path = self.root / "cohorts.json"; q.write_json(path, cohorts)
        return path, [a, b, c]

    def test_cross_species_and_shared_photo_duplicates(self):
        _, rows = self.cohort_fixture()
        photos = q.unique_photos(rows)
        self.assertEqual(len(photos), 2)
        checked = {pid: q.check_photo(self.root, row) for pid, row in photos.items()}
        groups = q.duplicate_groups(rows, checked)
        pixel = next(g for g in groups if g["kind"] == "oriented_rgba_pixels")
        self.assertTrue(pixel["conflicting_species_labels"])
        self.assertEqual(pixel["observation_ids"], ["1", "2", "3"])
        byte = next(g for g in groups if g["kind"] == "file_bytes")
        self.assertEqual(byte["photo_ids"], ["12"])

    def test_pipeline_keeps_provenance_and_raw_images(self):
        path, rows = self.cohort_fixture()
        originals = {r["file_path"]: (self.root / r["file_path"]).read_bytes() for r in rows}
        self.assertEqual(q.run(self.root, path), 0)
        output = next((self.root / "data/derived/quality").iterdir())
        summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(summary["unique_photo_ids"], 2)
        self.assertEqual(summary["observations"], 3)
        self.assertEqual(summary["source_reports"][1]["source_completion_status"], "INCOMPLETE")
        self.assertEqual(len(q.csv_rows((output / "observation_photos.csv").read_bytes())), 3)
        self.assertTrue((output / "sources/hor/plan.json").exists())
        for name, raw in originals.items():
            self.assertEqual((self.root / name).read_bytes(), raw)

    def test_changed_plan_is_rejected_before_image_processing(self):
        path, _ = self.cohort_fixture()
        (self.root / "data/raw/metadata/hor_test/plan.json").write_bytes(b"{}")
        self.assertEqual(q.run(self.root, path), 1)
        output = next((self.root / "data/derived/quality").iterdir())
        failure = json.loads((output / "failure.json").read_text())
        self.assertIn("plan hash", failure["error"])
        self.assertEqual(failure["checked_photos"], 0)


if __name__ == "__main__":
    unittest.main()
