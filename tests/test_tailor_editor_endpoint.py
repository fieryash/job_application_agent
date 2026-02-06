import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import src.app as app_module
from src.models import (
    Constraints,
    Contact,
    Identity,
    Job,
    JobRequirements,
    Profile,
    TailoredEditorDraft,
    TailoredResume,
)


class TestTailorEditorEndpoint(unittest.TestCase):
    def test_editor_endpoint_returns_live_score_and_draft(self):
        profile = Profile(
            tag="p_editor",
            identity=Identity(
                name="Editor User",
                targets=["data_scientist"],
                industries=[],
                notes="Data scientist with ML experience.",
                seniority="mid",
                contact=Contact(email="editor@example.com"),
            ),
            constraints=Constraints(location=["us_any"], remote_policy=["remote"]),
            raw_resume_text="Python SQL ML",
        )

        job = Job(
            id="j_editor",
            company="Acme",
            title="Data Scientist",
            requirements=JobRequirements(must_have=["Python", "SQL"]),
            stack=["PyTorch"],
            responsibilities=["Build ML models"],
            raw_text="Python SQL and PyTorch required",
        )

        tailored = TailoredResume(
            job_id=job.id,
            editor_draft=TailoredEditorDraft(
                summary="Initial summary",
                skills=["Python", "SQL"],
                bullets=["Built ML systems."],
            ),
        )

        with patch.object(app_module, "load_profile_by_tag", return_value=profile), patch.object(
            app_module, "get_job", return_value=job
        ), patch.object(app_module, "_ensure_tailored_resume", return_value=tailored):
            app_module.API_KEY = None
            client = TestClient(app_module.app)
            resp = client.get("/tailor/p_editor/j_editor/editor")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("fit_score", data)
            self.assertIn("draft", data)
            self.assertEqual(data["draft"]["summary"], "Initial summary")


if __name__ == "__main__":
    unittest.main()
