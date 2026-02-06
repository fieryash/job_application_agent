import unittest

from src.auto_apply import APPLY_ENTRY_SELECTORS, _looks_like_listing_page


class TestAutoApplySelectors(unittest.TestCase):
    def test_submit_resume_and_apply_selectors_present(self):
        joined = " | ".join(APPLY_ENTRY_SELECTORS).lower()
        self.assertIn("submit resume", joined)
        self.assertIn("apply", joined)

    def test_apple_detail_url_is_not_listing(self):
        self.assertFalse(_looks_like_listing_page("https://jobs.apple.com/en-us/details/200635475-3337"))


if __name__ == "__main__":
    unittest.main()

