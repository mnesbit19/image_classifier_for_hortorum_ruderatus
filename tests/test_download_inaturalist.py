"""We test licence boundaries, provenance, interrupted downloads and resume logic."""

import argparse
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import urllib.error
import urllib.parse

SCRIPT = Path(__file__).resolve().parents[1] / "code/download_inaturalist.py"
SPEC = importlib.util.spec_from_file_location("downloader", SCRIPT)
d = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d)
CONFIG = json.loads((SCRIPT.parents[1] / "config/project.json").read_text())
TEST_TMP = SCRIPT.parents[1] / "tmp/downloader_tests"
TEST_TMP.mkdir(parents=True, exist_ok=True)
ROW = {"id": "10", "scientific_name": "Bombus ruderatus", "taxon_id": "133704",
       "quality_grade": "research", "license": "CC-BY", "place_country_name": "United Kingdom"}
PHOTO = {"id": 20, "license_code": "cc-by-nc", "attribution": "(c) Example (CC BY-NC)",
         "url": "https://inaturalist-open-data.s3.amazonaws.com/photos/20/square."}
OBS = {"id": 10, "quality_grade": "research", "taxon": {"id": 133704, "name": "Bombus ruderatus",
       "ancestor_ids": [133704]}, "photos": [PHOTO]}
JPEG = b"\xff\xd8\xff" + b"fixture-signature-only" + b"\xff\xd9"


class FakeClient:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def fetch(self, url, media=False):
        self.calls.append((url, media))
        if media:
            if self.fail:
                raise OSError("interrupted transfer")
            return JPEG, "image/jpeg", url
        return json.dumps({"total_results": 1, "results": [OBS]}).encode(), "application/json", url


class DownloaderTests(unittest.TestCase):
    def records(self, observation=OBS, row=ROW):
        return d.eligible_records(row, observation, "snapshot.json", CONFIG, "original")

    def test_photo_licence_overrides_observation_licence(self):
        obs = copy.deepcopy(OBS)
        obs["photos"].append({**PHOTO, "id": 21, "license_code": "cc-by-nc-nd"})
        accepted, rejected = self.records(obs)
        self.assertEqual([p["photo_id"] for p in accepted], [20])
        self.assertEqual(rejected[0]["reason"], "photo_license_not_in_allowlist")
        obs["photos"][0]["license_code"] = None
        self.assertEqual(len(self.records(obs)[0]), 0)

    def test_subspecies_ancestry_is_preserved(self):
        obs = copy.deepcopy(OBS)
        obs["taxon"] = {"id": 999, "ancestor_ids": [133704], "name": "Bombus ruderatus test"}
        accepted, _ = self.records(obs)
        self.assertEqual(accepted[0]["species"], "Bombus ruderatus")
        self.assertEqual(accepted[0]["current_taxon_id"], 999)
        self.assertTrue(accepted[0]["taxon_changed_since_export"])

    def test_changed_species_is_not_silently_relabelled(self):
        obs = copy.deepcopy(OBS)
        obs["taxon"] = {"id": 121989, "ancestor_ids": [121989], "name": "Bombus hortorum"}
        accepted, rejected = self.records(obs)
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "species_changed_since_export_review_required")

    def test_missing_and_downgraded_observations(self):
        self.assertEqual(self.records(None)[1][0]["reason"], "observation_missing_from_api")
        self.assertEqual(self.records({**OBS, "quality_grade": "needs_id"})[1][0]["reason"],
                         "current_quality_grade_not_research")

    def test_all_views_and_old_extensionless_urls(self):
        obs = copy.deepcopy(OBS)
        obs["photos"].append({**PHOTO, "id": 21,
            "url": "https://inaturalist-open-data.s3.amazonaws.com/photos/21/square.JPG"})
        accepted, _ = self.records(obs)
        self.assertEqual(len(accepted), 2)
        self.assertTrue(accepted[0]["download_url"].endswith("/original."))
        self.assertTrue(accepted[1]["download_url"].endswith("/original.JPG"))

    def test_html_and_truncation_rejected(self):
        for raw, mime in [(b"<html>oops</html>", "text/html"), (JPEG[:-2], "image/jpeg")]:
            with self.assertRaises(ValueError):
                d.image_extension(raw, mime)

    def test_resume_reuses_file_and_detects_corruption(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            root = Path(directory)
            client = FakeClient()
            item = self.records()[0][0]
            first = d.download_photo(item, root, root / "cache", client)
            second = d.download_photo(item, root, root / "cache", client)
            self.assertEqual(second["status"], "cached")
            self.assertEqual(len(client.calls), 1)
            (root / first["file_path"]).write_bytes(b"damaged")
            with self.assertRaises(ValueError):
                d.download_photo(item, root, root / "cache", client)
            self.assertEqual(len(client.calls), 1)

    def test_failure_creates_no_image_or_receipt(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            root = Path(directory)
            with self.assertRaises(OSError):
                d.download_photo(self.records()[0][0], root, root / "cache", FakeClient(fail=True))
            self.assertFalse(list(root.rglob("*.jpg")))
            self.assertFalse(list(root.rglob("*.json")))

    def test_existing_snapshot_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            path = Path(directory) / "plan.json"
            d.save_once(path, b"original")
            with self.assertRaises(ValueError):
                d.save_once(path, b"changed")
            self.assertEqual(path.read_bytes(), b"original")

    def test_end_to_end_failure_then_resume(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            root = Path(directory)
            (root / "config").mkdir()
            (root / "config/project.json").write_text(json.dumps(CONFIG))
            source = root / "export.csv"
            source.write_text("id,scientific_name,taxon_id,quality_grade\n10,Bombus ruderatus,133704,research\n")
            args = argparse.Namespace(root=root, export=source, dataset="test_v1", size="original",
                audit_only=False, metadata_only=False, limit_observations=None, max_downloads=None)
            self.assertEqual(d.run(args, FakeClient(fail=True)), 2)
            resumed = FakeClient()
            self.assertEqual(d.run(args, resumed), 0)
            self.assertEqual(len(resumed.calls), 1)
            self.assertTrue(resumed.calls[0][1])
            cached = FakeClient()
            self.assertEqual(d.run(args, cached), 0)
            self.assertEqual(cached.calls, [])
            self.assertEqual(len(list((root / "data/manifests/test_v1").glob("*/summary.json"))), 3)

    def test_full_batches_and_remainder_set_page_size_and_resume(self):
        class PageSizeClient:
            def __init__(self):
                self.batches = []

            def fetch(self, url, media=False):
                self_assert.assertFalse(media)
                parts = urllib.parse.urlsplit(url)
                ids = [int(i) for i in parts.path.rsplit("/", 1)[-1].split(",")]
                page_size = int(urllib.parse.parse_qs(parts.query).get("per_page", [30])[0])
                if len(ids) > page_size:
                    raise d.HTTPFailure(url, 422, "Unprocessable Entity", '{"error":"Too many IDs"}')
                self.batches.append(ids)
                results = [{**OBS, "id": i} for i in ids]
                return json.dumps({"total_results": len(ids), "results": results}).encode(), "application/json", url

        self_assert = self
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            root = Path(directory)
            (root / "config").mkdir()
            (root / "config/project.json").write_text(json.dumps(CONFIG))
            source = root / "export.csv"
            source.write_text("id,scientific_name,taxon_id,quality_grade\n" + "".join(
                f"{i},Bombus ruderatus,133704,research\n" for i in range(1, 206)))
            args = argparse.Namespace(root=root, export=source, dataset="batch_test_v1", size="original",
                audit_only=False, metadata_only=True, limit_observations=None, max_downloads=None)
            client = PageSizeClient()
            with mock.patch("builtins.print"):
                self.assertEqual(d.run(args, client), 0)
            self.assertEqual([len(batch) for batch in client.batches], [100, 100, 5])
            self.assertEqual([i for batch in client.batches for i in batch], list(range(1, 206)))
            saved = {p: p.read_bytes() for p in (root / "data/raw/metadata/batch_test_v1").rglob("*") if p.is_file()}
            resumed = PageSizeClient()
            with mock.patch("builtins.print"):
                self.assertEqual(d.run(args, resumed), 0)
            self.assertEqual(resumed.batches, [])
            self.assertTrue(all(p.read_bytes() == raw for p, raw in saved.items()))

    def test_http_error_retains_url_status_and_server_explanation(self):
        url = d.API + "/observations/10?per_page=1"
        error = urllib.error.HTTPError(url, 422, "Unprocessable Entity", {},
            io.BytesIO(b'{"error":"Too many IDs","status":422}'))
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            with mock.patch.object(d.urllib.request, "urlopen", side_effect=error) as opener:
                with self.assertRaises(d.HTTPFailure) as raised:
                    d.Client(Path(directory)).fetch(url)
            self.assertEqual(opener.call_count, 1)
            self.assertEqual(raised.exception.details["status"], 422)
            self.assertEqual(raised.exception.details["url"], url)
            self.assertIn("Too many IDs", raised.exception.details["response_body_excerpt"])

    def test_failed_metadata_request_has_a_report_and_can_resume(self):
        class RejectingClient:
            def fetch(self, url, media=False):
                raise d.HTTPFailure(url, 422, "Unprocessable Entity", '{"error":"Too many IDs"}')

        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            root = Path(directory)
            (root / "config").mkdir()
            (root / "config/project.json").write_text(json.dumps(CONFIG))
            source = root / "export.csv"
            source.write_text("id,scientific_name,taxon_id,quality_grade\n10,Bombus ruderatus,133704,research\n")
            args = argparse.Namespace(root=root, export=source, dataset="failed_api_v1", size="original",
                audit_only=False, metadata_only=False, limit_observations=None, max_downloads=None)
            with mock.patch("builtins.print"):
                self.assertEqual(d.run(args, RejectingClient()), 2)
            report_path = next((root / "data/manifests/failed_api_v1").glob("*/summary.json"))
            before = report_path.read_bytes()
            report = json.loads(before)
            self.assertEqual(report["processed_observations"], 0)
            self.assertEqual(report["http_error"]["status"], 422)
            self.assertIn("Too many IDs", report["http_error"]["response_body_excerpt"])
            plan_path = root / "data/raw/metadata/failed_api_v1/plan.json"
            plan_bytes = plan_path.read_bytes()
            with mock.patch("builtins.print"):
                self.assertEqual(d.run(args, FakeClient()), 0)
            self.assertEqual(plan_path.read_bytes(), plan_bytes)
            self.assertEqual(report_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
