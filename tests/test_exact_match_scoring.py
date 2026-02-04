import unittest

from src.matcher import score_job
from src.models import Constraints, Contact, Identity, Job, JobRequirements, Profile, Skill


class TestExactMatchScoring(unittest.TestCase):
    def test_exact_match_score_uses_resume_text(self):
        profile = Profile(
            tag="p1",
            identity=Identity(
                name="Test User",
                targets=["machine_learning_engineer"],
                industries=[],
                notes=None,
                seniority="mid",
                contact=Contact(email="test@example.com"),
            ),
            constraints=Constraints(location=["us_any"], remote_policy=["remote"]),
            skills=[Skill(name="Python"), Skill(name="SQL"), Skill(name="PyTorch")],
            raw_resume_text="Built ML pipelines in Python and SQL. Trained models with PyTorch.",
        )

        job = Job(
            id="j1",
            company="Acme",
            title="Machine Learning Engineer",
            requirements=JobRequirements(
                must_have=["Python", "SQL"],
                nice_to_have=["Spark"],
            ),
            stack=["PyTorch", "AWS"],
            responsibilities=["Build ML systems"],
            raw_text="We need Python/SQL and PyTorch on AWS. Spark is a plus.",
        )

        result = score_job(profile, job)
        self.assertAlmostEqual(result.exact_match.score, 0.8, places=3)
        self.assertIn("Python", result.exact_match.matched)
        self.assertIn("SQL", result.exact_match.matched)
        self.assertIn("PyTorch", result.exact_match.matched)
        self.assertIn("AWS", result.exact_match.missing)


if __name__ == "__main__":
    unittest.main()

