import unittest
from unittest.mock import patch

import src.resume_ingest as resume_ingest


class TestResumeIngest(unittest.TestCase):
    def test_build_profile_from_resume_text_uses_parsed_bullets(self):
        parsed = resume_ingest.ParsedResume(
            name="Jane Doe",
            email="jane@example.com",
            seniority="mid",
            target_roles=["data_scientist"],
            work_authorization="US Citizen",
            skills=[
                resume_ingest.ParsedResumeSkill(
                    name="Python",
                    category="programming",
                    level="expert",
                )
            ],
            experience=[
                resume_ingest.ParsedResumeExperience(
                    company="Acme",
                    role="Data Scientist",
                    bullets=["Built ML pipelines in Python for fraud detection."],
                )
            ],
        )

        with patch.object(resume_ingest, "parse_resume_with_gpt", return_value=parsed):
            profile = resume_ingest.build_profile_from_resume_text(
                "Jane Doe\nExperience\n- Built ML pipelines in Python for fraud detection.",
                explicit_tag="profile_test",
                source_resume="resume.pdf",
            )

        self.assertEqual(profile.tag, "profile_test")
        self.assertEqual(profile.identity.name, "Jane Doe")
        self.assertEqual(profile.identity.targets, ["data_scientist"])
        self.assertEqual(profile.identity.contact.email, "jane@example.com")
        self.assertEqual(profile.constraints.work_authorization, "US Citizen")

        self.assertTrue(profile.skills)
        self.assertEqual(profile.skills[0].name, "Python")

        # Bullet bank should be derived from parsed experience bullets (not purely regex fallback).
        self.assertTrue(profile.bullet_bank)
        self.assertTrue(profile.bullet_bank[0].id.startswith("exp_bullet_"))
        self.assertIn("Python", profile.bullet_bank[0].skills)


if __name__ == "__main__":
    unittest.main()

