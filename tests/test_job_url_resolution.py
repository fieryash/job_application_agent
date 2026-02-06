import unittest

from src.job_ingest import resolve_job_url


class TestJobUrlResolution(unittest.TestCase):
    def test_resolve_direct_url_from_aggregator_query_parameter(self):
        url = (
            "https://www.indeed.com/rc/clk?"
            "jk=12345&url=https%3A%2F%2Fjobs.lever.co%2Facme%2Fabc123&utm_source=test"
        )
        resolved = resolve_job_url(url)
        self.assertIsNotNone(resolved)
        self.assertIn("jobs.lever.co/acme/abc123", resolved)


if __name__ == "__main__":
    unittest.main()
