"""Offline progress tests; no web server, Modal client, or GPU is started."""

import unittest

from backend.status import pending_progress, progress_record


class PendingProgressTests(unittest.TestCase):
    def test_job_progress_wins_over_pool_activity(self):
        for phase in ("generating", "encoding"):
            progress = progress_record(phase)
            self.assertEqual(pending_progress(progress, {"gpu": "warming", "backlog": 2}), progress)

    def test_completed_worker_is_not_a_completed_result(self):
        progress = progress_record("complete")
        result = pending_progress(progress, {})
        self.assertEqual(result["phase"], "processing")
        self.assertEqual(result["updated_at"], progress["updated_at"])
        self.assertEqual(progress["phase"], "complete")

    def test_cold_start_precedes_queue(self):
        self.assertEqual(pending_progress(None, {"gpu": "warming", "backlog": 1})["phase"], "warming")

    def test_queue_requires_backlog(self):
        self.assertEqual(pending_progress(None, {"gpu": "running", "backlog": 1})["phase"], "queued")
        self.assertEqual(pending_progress(None, {"gpu": "running", "running_inputs": 1})["phase"], "processing")

    def test_missing_metadata_does_not_claim_gpu_is_ready(self):
        self.assertEqual(pending_progress(None, {"gpu": "unknown"})["phase"], "processing")
        self.assertEqual(pending_progress("invalid", {})["phase"], "processing")


if __name__ == "__main__":
    unittest.main()
