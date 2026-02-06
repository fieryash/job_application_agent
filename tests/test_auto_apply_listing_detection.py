import unittest
from pathlib import Path

from src.auto_apply import _looks_like_listing_page, auto_apply_job
from src.models import Constraints, Contact, Identity, Job, JobSource, Profile


class TestAutoApplyListingDetection(unittest.TestCase):
    def test_listing_url_is_unsupported_before_browser_run(self):
        profile = Profile(
            tag="p_apply_list",
            identity=Identity(
                name="Test User",
                targets=["machine_learning_engineer"],
                industries=[],
                notes=None,
                seniority="mid",
                contact=Contact(email="test@example.com"),
            ),
            constraints=Constraints(location=["us_any"], remote_policy=["remote"]),
        )
        job = Job(
            id="j_listing",
            company="Dice",
            title="Unknown",
            source=JobSource(url="https://www.dice.com/jobs/q-machine+learning-l-remote-jobs"),
        )

        result = auto_apply_job(
            profile=profile,
            job=job,
            resume_path=Path("nonexistent_resume.txt"),
            auto_submit=True,
            headless=True,
        )
        self.assertEqual(result.status, "unsupported")
        self.assertIn("listing/search page", result.message.lower())

    def test_heuristic_detects_listing_but_not_direct_posting(self):
        self.assertTrue(_looks_like_listing_page("https://www.dice.com/jobs/q-machine+learning-l-remote-jobs"))
        self.assertFalse(_looks_like_listing_page("https://www.indeed.com/viewjob?jk=1234567890abcdef"))


if __name__ == "__main__":
    unittest.main()

