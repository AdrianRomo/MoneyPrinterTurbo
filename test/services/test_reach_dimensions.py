"""Reach broken down by something other than the hashtag set.

The loop scored exactly one variable — which hashtag set was used — the one the
module's own docstring calls a weak ranking signal since 2024. These cover the
axes that plausibly matter more, and the guard that stops a breakdown over two
posts being read as a result.
"""

import unittest

from app.services import hashtags


def _sample(media_id, kind, hour, subject, reach, saved=0, shares=0):
    return {
        "set_id": "hope",
        "media_id": media_id,
        "kind": kind,
        "local_hour": hour,
        "variant": {"subject": subject},
        "metrics": {"reach": reach, "likes": 1, "comments": 0,
                    "saved": saved, "shares": shares},
    }


SAMPLES = [
    _sample("a", "reel", 20, "oceans", 140, saved=3, shares=1),
    _sample("b", "reel", 20, "forests", 120, saved=2),
    _sample("c", "reel", 13, "mist", 90, saved=1),
    _sample("d", "carousel", 18, "galaxies", 60, saved=4, shares=2),
    _sample("e", "post", 7, "none", 20),
]


class TestReachBy(unittest.TestCase):
    def test_groups_by_ledger_field(self):
        rows = hashtags.reach_by("kind", SAMPLES)
        self.assertEqual(rows["reel"]["samples"], 3)
        self.assertEqual(rows["reel"]["saves"], 6)
        self.assertAlmostEqual(rows["reel"]["mean_reach"], 116.67, places=1)

    def test_groups_by_variant_field(self):
        rows = hashtags.reach_by("subject", SAMPLES)
        self.assertIn("oceans", rows)
        self.assertEqual(rows["oceans"]["samples"], 1)

    def test_sorted_by_score_descending(self):
        scores = [r["mean_score"] for r in hashtags.reach_by("kind", SAMPLES).values()]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_thin_groups_are_marked_unreadable(self):
        """A breakdown over one post is a story, not a measurement."""
        rows = hashtags.reach_by("kind", SAMPLES)
        self.assertTrue(rows["reel"]["readable"])       # 3 samples
        self.assertFalse(rows["carousel"]["readable"])  # 1 sample

    def test_samples_missing_the_dimension_are_skipped_not_guessed(self):
        rows = hashtags.reach_by("local_hour", [{"set_id": "hope", "media_id": "z",
                                                 "metrics": {"reach": 5}}])
        self.assertEqual(rows, {})

    def test_report_is_empty_rather_than_invented_without_data(self):
        report = hashtags.dimension_report([])
        self.assertEqual(report["dimensions"], {})


class TestRecordSample(unittest.TestCase):
    def test_new_fields_are_optional(self):
        """Posts published before these axes existed must still record."""
        import tempfile

        original = hashtags.STORAGE
        try:
            hashtags.STORAGE = tempfile.mkdtemp()
            hashtags.record_sample("hope", "m1", {"reach": 3})
            hashtags.record_sample("hope", "m2", {"reach": 4},
                                   kind="reel", local_hour=20)
            rows = hashtags._load("samples.json", [])
            self.assertEqual(len(rows), 2)
            self.assertNotIn("kind", rows[0])
            self.assertEqual(rows[1]["kind"], "reel")
            self.assertEqual(rows[1]["local_hour"], 20)
        finally:
            hashtags.STORAGE = original


if __name__ == "__main__":
    unittest.main()
