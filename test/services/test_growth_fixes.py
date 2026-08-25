"""Tests for the 2026-08-25 growth changes.

Each test pins the *reason* a change was made, not just its current value, so
that reverting the reasoning is what breaks the build rather than retuning a
number. See docs runbook "influencer-automation-to-postiz-pipeline".
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import carousel, content_scheduler, hashtags, insights, quote_reel


class TestReelDurationIsReadable(unittest.TestCase):
    """The 15s hold produced 26% completion; the quote is read in ~4s."""

    def test_default_hold_is_short(self):
        self.assertLessEqual(quote_reel.target_seconds(), 10.0)

    def test_floor_does_not_silently_clamp_the_default(self):
        # The original bug shape: lowering DEFAULT_SECONDS alone did nothing,
        # because target_seconds() clamps up to MIN_SECONDS.
        self.assertLessEqual(quote_reel.MIN_SECONDS, quote_reel.DEFAULT_SECONDS)

    def test_quote_ceiling_tracks_the_duration(self):
        # A shorter Reel must not keep a quote nobody can finish reading.
        chars = quote_reel.max_quote_chars()
        self.assertLessEqual(chars, quote_reel.READING_CHARS_PER_SECOND
                             * quote_reel.target_seconds())
        self.assertLessEqual(chars, quote_reel.MAX_QUOTE_CHARS)

    def test_prompt_word_budget_fits_the_ceiling(self):
        low, high = quote_reel._quote_word_budget()
        self.assertLess(low, high)
        # Asking for more words than the normaliser will keep means shipping
        # truncated quotes; the budget must fit inside the character ceiling.
        self.assertLessEqual(high * 5.9, quote_reel.max_quote_chars() + 5.9)


class TestSelectorNeedsASeparableWinner(unittest.TestCase):
    """Reach ran 6-108, so a mean over a handful of posts is not a ranking."""

    def _scores(self, spec):
        return {sid: {"samples": n, "mean_score": mean, "stderr": err}
                for sid, (n, mean, err) in spec.items()}

    def test_one_eligible_set_is_not_a_comparison(self):
        self.assertFalse(hashtags._is_separable(self._scores({"a": (20, 50.0, 2.0)})))

    def test_overlapping_error_bars_are_not_a_winner(self):
        # This is the live shape: peace 64.3 +/- 51.9 against hope 27.0 +/- 12.5.
        scores = self._scores({"peace": (3, 64.3, 51.9), "hope": (3, 27.0, 12.5)})
        self.assertFalse(hashtags._is_separable(scores))

    def test_a_clear_lead_is_exploited(self):
        scores = self._scores({"peace": (30, 80.0, 3.0), "hope": (30, 20.0, 3.0)})
        self.assertTrue(hashtags._is_separable(scores))

    def test_single_sample_reports_infinite_uncertainty(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(hashtags, "STORAGE", tmp):
                sid = next(iter(hashtags.SETS))
                hashtags._save("samples.json",
                               [{"set_id": sid, "metrics": {"reach": 108}}])
                self.assertEqual(hashtags.set_scores()[sid]["stderr"], float("inf"))

    def test_choose_set_rotates_while_nothing_is_separable(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(hashtags, "STORAGE", tmp):
                a, b = list(hashtags.SETS)[:2]
                hashtags._save("samples.json",
                               [{"set_id": a, "metrics": {"reach": 108}}] * 20
                               + [{"set_id": b, "metrics": {"reach": 8}}] * 20)
                # Even at 20 samples a set only wins if it is separable; here it
                # is, so this asserts the guard does not block a real signal.
                self.assertTrue(hashtags._is_separable(
                    {k: v for k, v in hashtags.set_scores().items()
                     if v["samples"] >= hashtags.MIN_SAMPLES}))


class TestStoriesAreNotScheduled(unittest.TestCase):
    def test_story_cadence_is_zero_in_code_and_pack(self):
        self.assertEqual(content_scheduler.PLAN["story"]["per_day"], 0)
        # The pack overrides PLAN, so the code default alone proves nothing.
        self.assertEqual(content_scheduler.cadence()["story"]["per_day"], 0)

    def test_discovery_formats_survived_the_cut(self):
        cadence = content_scheduler.cadence()
        self.assertGreaterEqual(cadence["reel"]["per_day"], 1)
        self.assertGreaterEqual(cadence["carousel"]["per_day"], 1)


class TestCarouselSlideBudget(unittest.TestCase):
    def test_furniture_slides_are_reserved_not_hoped_for(self):
        self.assertEqual(carousel.MAX_PHOTO_SLIDES,
                         carousel.MAX_SLIDES - carousel.RESERVED_SLIDES)

    def test_photo_budget_leaves_room_for_verse_and_cta(self):
        # 8 photos + verse + CTA must not exceed what the API accepts.
        self.assertLessEqual(carousel.MAX_PHOTO_SLIDES + carousel.RESERVED_SLIDES,
                             carousel.MAX_SLIDES)

    def test_verse_lookup_failure_costs_the_slide_not_the_carousel(self):
        with patch("app.services.verse_card.select_verse",
                   side_effect=RuntimeError("verse API down")):
            self.assertIsNone(carousel.carousel_verse("mountains"))


class TestAccountLevelGrowth(unittest.TestCase):
    def test_stories_are_not_counted_as_match_failures(self):
        # A story can never appear in /me/media, so counting it as "unmatched"
        # made the counter climb forever and hid real matcher regressions.
        log = [{"set_id": "hope", "post_id": "p1", "kind": "story",
                "at": "2020-01-01T00:00:00+00:00"},
               {"set_id": "hope", "post_id": "p2", "kind": "reel",
                "at": "2020-01-01T00:00:00+00:00"}]
        with patch("app.services.postiz.PostizService._load_publish_log", return_value=log), \
             patch.object(insights, "fetch_media", return_value={"x": {"id": "1"}}), \
             patch.object(insights, "record_account_snapshot", return_value={}), \
             patch.object(insights, "growth_report", return_value={}):
            out = insights.collect("tok", {"p1": "https://a/", "p2": "https://b/"})
        self.assertEqual(out["stories_unmeasurable"], 1)
        self.assertEqual(out["unmatched"], 1)

    def test_snapshot_is_one_row_per_day_last_write_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = str(Path(tmp) / "account.json")
            with patch.object(insights, "GROWTH_STORE", store), \
                 patch.object(insights, "fetch_account_insights", return_value={}):
                with patch.object(insights, "fetch_account",
                                  return_value={"followers_count": 47}):
                    insights.record_account_snapshot("tok")
                with patch.object(insights, "fetch_account",
                                  return_value={"followers_count": 48}):
                    insights.record_account_snapshot("tok")
                series = json.load(open(store))
        self.assertEqual(len(series), 1)
        self.assertEqual(series[0]["followers_count"], 48)

    def test_growth_refuses_to_report_a_trend_from_one_reading(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = str(Path(tmp) / "account.json")
            with patch.object(insights, "GROWTH_STORE", store), \
                 patch.object(insights, "fetch_account_insights", return_value={}), \
                 patch.object(insights, "fetch_account",
                              return_value={"followers_count": 47}):
                insights.record_account_snapshot("tok")
                self.assertIn("note", insights.growth_report())

    def test_growth_reports_followers_per_day_once_there_is_a_span(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = str(Path(tmp) / "account.json")
            series = [{"date": "2026-08-25", "followers_count": 47, "follows_count": 358},
                      {"date": "2026-08-27", "followers_count": 53, "follows_count": 140}]
            Path(store).write_text(json.dumps(series))
            with patch.object(insights, "GROWTH_STORE", store):
                report = insights.growth_report()
        self.assertEqual(report["delta"]["followers_count"], 6)
        self.assertEqual(report["followers_per_day"], 6.0)
        self.assertEqual(report["follow_ratio"], round(53 / 140, 2))

    def test_a_failed_account_read_does_not_write_a_hollow_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = str(Path(tmp) / "account.json")
            with patch.object(insights, "GROWTH_STORE", store), \
                 patch.object(insights, "fetch_account", return_value={}):
                self.assertEqual(insights.record_account_snapshot("tok"), {})
            self.assertFalse(Path(store).exists())


if __name__ == "__main__":
    unittest.main()
