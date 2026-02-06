import unittest

from src.matcher import score_job
from src.models import Constraints, Contact, Identity, Job, JobRequirements, Profile, Skill


class TestScoringExplanations(unittest.TestCase):
    def test_keyword_coverage_and_components_are_present(self):
        profile = Profile(
            tag="p_explain",
            identity=Identity(
                name="Test User",
                targets=["machine_learning_engineer"],
                industries=[],
                notes=None,
                seniority="mid",
                contact=Contact(email="test@example.com"),
            ),
            constraints=Constraints(location=["us_any"], remote_policy=["remote"]),
            skills=[Skill(name="Python"), Skill(name="SQL"), Skill(name="Docker")],
            raw_resume_text="Built Python SQL services and deployed with Docker.",
        )

        job = Job(
            id="j_explain",
            company="Acme",
            title="Machine Learning Engineer",
            requirements=JobRequirements(
                must_have=["Python", "SQL", "Spark"],
                nice_to_have=["Kubernetes"],
            ),
            stack=["PyTorch", "Docker"],
            responsibilities=["Build ML systems"],
            raw_text="Need Python SQL Spark and Docker, with PyTorch in production.",
        )

        result = score_job(profile, job)

        self.assertTrue(result.score_components)
        self.assertTrue(any(c.key == "skills" for c in result.score_components))
        self.assertIn("Spark", result.keyword_coverage.must_have_missing)
        self.assertIn("PyTorch", result.keyword_coverage.stack_missing)
        self.assertTrue(any("missing must-have keywords" in gap for gap in result.gaps))


if __name__ == "__main__":
    unittest.main()
