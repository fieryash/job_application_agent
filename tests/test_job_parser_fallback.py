import unittest
from unittest.mock import patch

from src.job_parser import parse_job_with_openai


class TestJobParserFallback(unittest.TestCase):
    @patch("src.job_parser.OpenAI")
    def test_heuristic_title_used_when_model_output_is_weak(self, mock_openai):
        # Simulate parser model returning an unusable/unknown title.
        mock_openai.return_value.chat.completions.create.return_value.choices = [
            type("Choice", (), {"message": type("Message", (), {"content": '{"title":"Unknown","company":"Unknown"}'})()})
        ]

        text = """
        Page Title: Remote Data Scientist - Example Corp Careers
        About the role
        Build predictive models and analytics systems.
        """
        job = parse_job_with_openai(text=text, url="https://careers.example.com/jobs/12345")

        self.assertNotEqual(job.title.lower(), "unknown")
        self.assertIn("scientist", job.title.lower())

    @patch("src.job_parser.OpenAI")
    def test_generic_title_fallback_is_not_unknown(self, mock_openai):
        mock_openai.return_value.chat.completions.create.side_effect = Exception("model error")
        text = "We are hiring. Responsibilities include coding and deployment."
        job = parse_job_with_openai(text=text, url="https://example.com/careers/job/42")

        self.assertNotEqual(job.title.lower(), "unknown")
        self.assertTrue(len(job.title) > 0)


if __name__ == "__main__":
    unittest.main()
