"""parse_due: ISO dates parse as that date (10:00 when no time is given), and the
existing natural-language phrasings are unchanged.

Run from backend/:   venv\\Scripts\\python -m unittest discover -s tests -v
"""

import unittest
from datetime import datetime

from tools.reminders import parse_due

_NOW = datetime(2026, 10, 19, 10, 0)  # a Monday


class IsoDateTest(unittest.TestCase):

    def test_date_only_defaults_to_10am(self):
        self.assertEqual(parse_due("2026-10-15", now=_NOW), datetime(2026, 10, 15, 10, 0))

    def test_past_and_today_dates_are_not_rolled_forward(self):
        # The old bug: "2026-10-15" came back as tomorrow 10:00.
        self.assertEqual(parse_due("2026-10-19", now=_NOW), datetime(2026, 10, 19, 10, 0))
        self.assertEqual(parse_due("2025-01-02", now=_NOW), datetime(2025, 1, 2, 10, 0))

    def test_with_time(self):
        self.assertEqual(parse_due("2026-10-15 14:30", now=_NOW),
                         datetime(2026, 10, 15, 14, 30))
        self.assertEqual(parse_due("2026-10-15T14:30:45", now=_NOW),
                         datetime(2026, 10, 15, 14, 30, 45))
        self.assertEqual(parse_due("2026-10-15 9:05", now=_NOW),
                         datetime(2026, 10, 15, 9, 5))

    def test_with_clock_phrase(self):
        self.assertEqual(parse_due("on 2026-10-15 at 6pm", now=_NOW),
                         datetime(2026, 10, 15, 18, 0))
        self.assertEqual(parse_due("2026-10-15 morning", now=_NOW),
                         datetime(2026, 10, 15, 9, 0))

    def test_impossible_date_raises(self):
        with self.assertRaises(ValueError):
            parse_due("2026-13-40", now=_NOW)


class NaturalLanguageUnchangedTest(unittest.TestCase):

    def test_existing_phrasings(self):
        self.assertEqual(parse_due("tomorrow", now=_NOW), datetime(2026, 10, 20, 9, 0))
        self.assertEqual(parse_due("friday", now=_NOW), datetime(2026, 10, 23, 9, 0))
        self.assertEqual(parse_due("in 3 days", now=_NOW), datetime(2026, 10, 22, 10, 0))
        self.assertEqual(parse_due("tomorrow at 6pm", now=_NOW), datetime(2026, 10, 20, 18, 0))
        self.assertEqual(parse_due("25 Oct", now=_NOW).date(), datetime(2026, 10, 25).date())

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            parse_due("", now=_NOW)


if __name__ == "__main__":
    unittest.main()
