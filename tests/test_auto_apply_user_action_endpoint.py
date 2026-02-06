import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import src.app as app_module
from src.models import AutoApplyResult, Constraints, Contact, Identity, Job, Profile


class TestAutoApplyUserActionEndpoint(unittest.TestCase):
    def test_user_action_close_and_mark_applied(self):
        profile = Profile(
            tag="p_user_action",
            identity=Identity(
                name="Action User",
                targets=["data_scientist"],
                industries=[],
                notes=None,
                seniority="mid",
                contact=Contact(email="action@example.com"),
            ),
            constraints=Constraints(location=["us_any"], remote_policy=["remote"]),
        )
        job = Job(id="j_user_action", company="Acme", title="Data Scientist")

        def fake_auto_apply_job(**kwargs):
            pause_cb = kwargs.get("pause_cb")
            wait_for_close_cb = kwargs.get("wait_for_close_cb")
            if pause_cb:
                pause_cb(
                    {
                        "reason": "missing_required_fields",
                        "prompt": "Fill missing required fields and OTP.",
                        "missing_fields": ["otp_code"],
                        "url": "https://example.com/apply",
                    }
                )
            if wait_for_close_cb:
                wait_for_close_cb()
            return AutoApplyResult(
                job_id=job.id,
                profile_tag=profile.tag,
                status="ready_to_submit",
                submitted=False,
                apply_url="https://example.com/apply",
                ats="workday",
                message="Verify details, complete OTP if required, and submit manually.",
                steps=["Paused for user"],
            )

        with patch.object(app_module, "_prepare_auto_apply_context", return_value=(profile, job, Path("resume.txt"))), patch.object(
            app_module, "auto_apply_job", side_effect=fake_auto_apply_job
        ), patch.object(app_module, "set_job_status"):
            app_module.API_KEY = None
            client = TestClient(app_module.app)

            start = client.post(
                f"/jobs/{job.id}/auto-apply/start?profile_tag={profile.tag}",
                json={"auto_submit": False, "headless": False},
            )
            self.assertEqual(start.status_code, 200, start.text)
            run_id = start.json()["run_id"]

            waiting = None
            for _ in range(25):
                status = client.get(f"/jobs/auto-apply/{run_id}")
                self.assertEqual(status.status_code, 200, status.text)
                payload = status.json()
                if payload.get("status") == "waiting_for_user":
                    waiting = payload
                    break
                time.sleep(0.05)

            self.assertIsNotNone(waiting)
            self.assertTrue(waiting.get("user_action_required"))
            self.assertIn("otp_code", waiting.get("missing_fields", []))

            action = client.post(
                f"/jobs/auto-apply/{run_id}/action",
                json={"action": "close_and_mark_applied"},
            )
            self.assertEqual(action.status_code, 200, action.text)

            completed = None
            for _ in range(30):
                status = client.get(f"/jobs/auto-apply/{run_id}")
                payload = status.json()
                if payload.get("status") in {"completed", "failed"}:
                    completed = payload
                    break
                time.sleep(0.05)

            self.assertIsNotNone(completed)
            self.assertEqual(completed.get("status"), "completed", completed)
            result = completed.get("result") or {}
            self.assertEqual(result.get("status"), "submitted")
            self.assertTrue(result.get("submitted"))


if __name__ == "__main__":
    unittest.main()

