import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import src.app as app_module
from src.models import Constraints, Contact, Identity, Job, JobRequirements, Profile, Skill


class TestMatchBatchEndpoint(unittest.TestCase):
    def test_match_batch(self):
        profile = Profile(
            tag="p1",
            identity=Identity(
                name="Test User",
                targets=["data_scientist"],
                industries=[],
                notes=None,
                seniority="mid",
                contact=Contact(email="test@example.com"),
            ),
            constraints=Constraints(location=["us_any"], remote_policy=["remote"]),
            skills=[Skill(name="Python"), Skill(name="SQL")],
            raw_resume_text="Python SQL",
        )

        jobs = [
            Job(
                id="j1",
                company="Acme",
                title="Data Scientist",
                requirements=JobRequirements(must_have=["Python"]),
                stack=["SQL"],
                responsibilities=["Analyze data"],
            ),
            Job(
                id="j2",
                company="Beta",
                title="Data Scientist",
                requirements=JobRequirements(must_have=["Python", "SQL"]),
                responsibilities=["Analyze data"],
            ),
        ]

        with patch.object(app_module, "load_profile_by_tag", return_value=profile):
            app_module.API_KEY = None
            client = TestClient(app_module.app)
            resp = client.post(
                "/match/batch?profile_tag=p1",
                json=[j.model_dump() for j in jobs],
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual({d["job_id"] for d in data}, {"j1", "j2"})
            self.assertTrue(all("fit_score" in d for d in data))
            self.assertTrue(all("exact_match" in d for d in data))


if __name__ == "__main__":
    unittest.main()

