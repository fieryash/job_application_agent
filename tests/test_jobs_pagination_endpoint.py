import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import src.app as app_module


class TestJobsPaginationEndpoint(unittest.TestCase):
    def test_jobs_offset_is_forwarded(self):
        with patch.object(app_module, "list_jobs", return_value=[]) as mocked:
            app_module.API_KEY = None
            client = TestClient(app_module.app)
            resp = client.get("/jobs?limit=10&offset=20")
            self.assertEqual(resp.status_code, 200, resp.text)
            mocked.assert_called_once_with(limit=10, offset=20)

    def test_saved_and_applied_offset_are_forwarded(self):
        with patch.object(app_module, "list_jobs_by_status", return_value=[]) as mocked:
            app_module.API_KEY = None
            client = TestClient(app_module.app)

            saved = client.get("/jobs/saved?profile_tag=p1&limit=7&offset=14")
            self.assertEqual(saved.status_code, 200, saved.text)
            mocked.assert_any_call("p1", "saved", 7, 14)

            applied = client.get("/jobs/applied?profile_tag=p1&limit=5&offset=10")
            self.assertEqual(applied.status_code, 200, applied.text)
            mocked.assert_any_call("p1", "applied", 5, 10)


if __name__ == "__main__":
    unittest.main()

