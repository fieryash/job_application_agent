import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import src.app as app_module
from src.models import AutoApplyResult, Constraints, Contact, Identity, Job, Profile


class TestAutoApplyStartEndpoint(unittest.TestCase):
    def test_auto_apply_start_returns_run_id_and_status_endpoint(self):
        profile = Profile(
            tag="p_apply",
            identity=Identity(
                name="Apply User",
                targets=["machine_learning_engineer"],
                industries=[],
                notes=None,
                seniority="mid",
                contact=Contact(email="apply@example.com"),
            ),
            constraints=Constraints(location=["us_any"], remote_policy=["remote"]),
            raw_resume_text="Python ML",
        )
        job = Job(id="j_apply", company="Acme", title="MLE")
        fake_result = AutoApplyResult(
            job_id=job.id,
            profile_tag=profile.tag,
            status="ready_to_submit",
            submitted=False,
            apply_url="https://example.com/apply",
            ats=None,
            message="Application form prepared.",
            steps=["Opened page", "Filled fields"],
        )

        with patch.object(app_module, "_prepare_auto_apply_context", return_value=(profile, job, Path("resume.txt"))), patch.object(
            app_module, "auto_apply_job", return_value=fake_result
        ), patch.object(app_module, "set_job_status"):
            app_module.API_KEY = None
            client = TestClient(app_module.app)
            start = client.post(
                f"/jobs/{job.id}/auto-apply/start?profile_tag={profile.tag}",
                json={"auto_submit": False, "headless": True},
            )
            self.assertEqual(start.status_code, 200, start.text)
            data = start.json()
            self.assertIn("run_id", data)

            run = None
            for _ in range(20):
                status = client.get(f"/jobs/auto-apply/{data['run_id']}")
                self.assertEqual(status.status_code, 200, status.text)
                run = status.json()
                if run.get("status") in {"completed", "failed"}:
                    break
                time.sleep(0.05)

            self.assertIsNotNone(run)
            self.assertEqual(run.get("status"), "completed", run)
            self.assertEqual((run.get("result") or {}).get("status"), "ready_to_submit")
            self.assertEqual(run.get("run_id"), data["run_id"])


if __name__ == "__main__":
    unittest.main()
