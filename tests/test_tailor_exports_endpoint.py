import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
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
    TailoredBullet,
    TailoredResume,
)


class TestTailorExportsEndpoint(unittest.TestCase):
    def test_tailor_endpoint_returns_export_links(self):
        profile = Profile(
            tag="p_tailor",
            identity=Identity(
                name="Test User",
                targets=["data_scientist"],
                industries=[],
                notes=None,
                seniority="mid",
                contact=Contact(email="test@example.com"),
            ),
            constraints=Constraints(location=["us_any"], remote_policy=["remote"]),
        )

        job = Job(
            id="job_tailor_1",
            company="Acme",
            title="Data Scientist",
            requirements=JobRequirements(must_have=["Python"]),
            stack=["SQL"],
            responsibilities=["Analyze data"],
        )

        fake_tailored = TailoredResume(
            job_id=job.id,
            selected_bullets=["b1"],
            rewritten_bullets=[
                TailoredBullet(
                    source_id="b1",
                    source_text="Built ETL pipelines.",
                    rewritten_text="Built production ETL pipelines for analytics.",
                    preview_html="Built <mark>production</mark> ETL pipelines for analytics.",
                )
            ],
            preview_html="<div>preview</div>",
        )

        with TemporaryDirectory() as tmp_dir:
            with patch.object(app_module, "EXPORTS_DIR", Path(tmp_dir)), patch.object(
                app_module, "save_tailored_resume"
            ), patch.object(app_module, "load_profile_by_tag", return_value=profile), patch.object(
                app_module, "tailor_resume", return_value=fake_tailored
            ):
                app_module.API_KEY = None
                client = TestClient(app_module.app)
                resp = client.post(
                    "/tailor?profile_tag=p_tailor",
                    json=job.model_dump(),
                )
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                exports = data.get("exports", {})
                self.assertIn("download_txt_url", exports)
                self.assertIn("download_html_url", exports)
                self.assertIn("txt_path", exports)
                self.assertTrue(Path(exports["txt_path"]).exists())


if __name__ == "__main__":
    unittest.main()
