"""We check observation counting, manifest integrity and duplicate warnings."""

import copy
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("report_download", Path(__file__).resolve().parents[1] / "code/report_download.py")
r = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r)


def fixture():
    rows = []
    for obs_id, photo_id, digest in (("1", "11", "a" * 64), ("1", "12", "b" * 64), ("2", "13", "a" * 64)):
        rows.append({"observation_id": obs_id, "photo_id": photo_id, "status": "downloaded",
            "species": "Bombus ruderatus", "class_id": "1", "photo_license": "cc-by", "sha256": digest,
            "bytes": "100", "file_path": f"data/raw/images/{photo_id}/original/{digest}.jpg",
            "observer_id": "21", "observer_login": "example", "export_country": "United Kingdom",
            "current_scientific_name": "Bombus ruderatus"})
    summary = {"status": "DOWNLOAD_COMPLETE", "selected_observations": 3, "processed_observations": 3,
        "photo_status_counts": {"downloaded": 3}, "eligible_species_counts": {"Bombus ruderatus": 3},
        "eligible_photo_license_counts": {"cc-by": 3}, "exclusion_counts": {},
        "pbs_job_id": "123.pbs", "dataset": "test", "attempt": "test_run"}
    config = {"classes": {"Bombus ruderatus": 1}, "data": {"initial_photo_license_allowlist": ["cc-by"]}}
    return summary, rows, [], config


class ReportTests(unittest.TestCase):
    def test_budget_subset_requires_opt_in_and_preserves_incomplete_source(self):
        args = fixture()
        args[0].update(status="INCOMPLETE", error=r.BUDGET_STOP, processed_observations=2)
        original = copy.deepcopy(args)
        with self.assertRaisesRegex(ValueError, "explicitly accepted"):
            r.analyse(*args, expected_job="123.pbs")
        with self.assertRaisesRegex(ValueError, "expected PBS"):
            r.analyse(*args, accept_budget_limited=True)
        report, _ = r.analyse(*args, expected_job="123.pbs", accept_budget_limited=True)
        self.assertEqual(report["source_completion_status"], "INCOMPLETE")
        self.assertEqual(report["source_acceptance"], "budget_limited_saved_photos_for_curation")
        self.assertEqual(report["unprocessed_export_observations"], 1)
        self.assertEqual(report["observations_with_downloads"], 2)
        self.assertEqual(args, original)
        self.assertIn("not a random", r.render(report))

    def test_budget_opt_in_does_not_accept_other_errors_or_failed_photos(self):
        for error in ("HTTPFailure: error", "BudgetReached: Server requests a long pause; resume later.", None):
            args = fixture()
            args[0].update(status="INCOMPLETE", error=error)
            with self.assertRaisesRegex(ValueError, "explicitly accepted"):
                r.analyse(*args, expected_job="123.pbs", accept_budget_limited=True)
        args = fixture()
        args[0].update(status="INCOMPLETE", error=r.BUDGET_STOP)
        args[1][0]["status"] = "failed"
        args[0]["photo_status_counts"] = {"downloaded": 2, "failed": 1}
        with self.assertRaisesRegex(ValueError, "unfinished/failed"):
            r.analyse(*args, expected_job="123.pbs", accept_budget_limited=True)

    def test_counts_observations_and_flags_cross_observation_hashes(self):
        report, candidates = r.analyse(*fixture(), expected_job="123.pbs")
        self.assertEqual(report["observations_with_downloads"], 2)
        self.assertEqual(report["observations_without_downloads"], 1)
        self.assertEqual(report["unique_photo_ids"], 3)
        self.assertEqual(report["unique_saved_hashes"], 2)
        self.assertEqual(report["reported_image_bytes"], 300)
        self.assertEqual(candidates[0]["observation_ids"], ["1", "2"])
        self.assertEqual(report["countries"][0]["observations"], 2)

    def test_shared_photo_is_not_counted_twice_as_a_file(self):
        summary, rows, excluded, config = fixture()
        rows[2].update({k: rows[0][k] for k in ("photo_id", "file_path", "sha256")})
        report, _ = r.analyse(summary, rows, excluded, config)
        self.assertEqual(report["photo_observation_links"], 3)
        self.assertEqual(report["unique_photo_ids"], 2)
        self.assertEqual(report["reported_image_bytes"], 200)

    def test_duplicate_links_and_summary_mismatch_fail(self):
        summary, rows, excluded, config = fixture()
        rows[2] = copy.deepcopy(rows[0])
        with self.assertRaisesRegex(ValueError, "Repeated observation/photo"):
            r.analyse(summary, rows, excluded, config)
        summary, rows, excluded, config = fixture()
        summary["photo_status_counts"]["downloaded"] = 4
        with self.assertRaisesRegex(ValueError, "disagree"):
            r.analyse(summary, rows, excluded, config)

    def test_incomplete_attempt_or_wrong_job_fails(self):
        with self.assertRaisesRegex(ValueError, "different PBS"):
            r.analyse(*fixture(), expected_job="999.pbs")
        args = fixture()
        args[0]["status"] = "METADATA_COMPLETE"
        with self.assertRaisesRegex(ValueError, "DOWNLOAD_COMPLETE"):
            r.analyse(*args)

    def test_unsafe_path_and_conflicting_photo_record_fail(self):
        args = fixture()
        args[1][0]["file_path"] = "data/raw/images/../../other.jpg"
        with self.assertRaisesRegex(ValueError, "outside"):
            r.analyse(*args)
        args = fixture()
        args[1][2]["photo_id"] = "11"
        with self.assertRaisesRegex(ValueError, "Conflicting file"):
            r.analyse(*args)

    def test_changed_licence_or_class_mapping_fails(self):
        args = fixture()
        args[3]["classes"]["Bombus ruderatus"] = 0
        with self.assertRaisesRegex(ValueError, "class mapping"):
            r.analyse(*args)
        args = fixture()
        args[3]["data"]["initial_photo_license_allowlist"] = ["cc0"]
        with self.assertRaisesRegex(ValueError, "licence"):
            r.analyse(*args)


if __name__ == "__main__":
    unittest.main()
