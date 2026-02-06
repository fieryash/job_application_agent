import unittest
from unittest.mock import patch

from src.job_parser import _extract_location_heuristic, parse_job_with_openai


class TestJobParserLocationCleanup(unittest.TestCase):
    @patch("src.job_parser.OpenAI")
    def test_noise_location_is_rejected_and_title_recovered(self, mock_openai):
        mock_openai.return_value.chat.completions.create.return_value.choices = [
            type(
                "Choice",
                (),
                {
                    "message": type(
                        "Message",
                        (),
                        {
                            "content": (
                                '{"title":"Unknown","company":"Apple",'
                                '"location":"We seek to promote equal opportunity for all applicants without regard to race, co"}'
                            )
                        },
                    )()
                },
            )
        ]

        text = """
        Page Title: AI/ML Engineer: System RF Data Ecosystem | Careers at Apple
        We seek to promote equal opportunity for all applicants without regard to race, co
        """
        job = parse_job_with_openai(text=text, url="https://jobs.apple.com/en-us/details/200635475-3337")
        self.assertNotEqual((job.title or "").lower(), "unknown")
        self.assertIn("engineer", (job.title or "").lower())
        self.assertIsNone(job.location)

    def test_city_state_country_line_is_detected(self):
        text = """
        AI/ML Engineer: System RF Data Ecosystem
        Cupertino, California, United States
        Hardware
        """
        loc = _extract_location_heuristic(text)
        self.assertEqual(loc, "Cupertino, California, United States")

    def test_non_location_sentence_is_not_detected(self):
        text = "We seek to promote equal opportunity for all applicants without regard to race, co"
        loc = _extract_location_heuristic(text)
        self.assertIsNone(loc)


if __name__ == "__main__":
    unittest.main()

