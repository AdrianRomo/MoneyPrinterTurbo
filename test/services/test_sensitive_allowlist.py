"""The per-account sensitive-category allowlist, and the shorter article Reel.

A blanket "any sensitive category gates" is unusable on an account whose whole
subject is one of those categories: 68% of the first 200 clusters were flagged,
84 of them for the single word "religion", and nothing ever auto-published.
Turning the check OFF would have been the wrong fix — these tests pin that the
hazard categories still gate.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.article import AutomationMode, AutomationSettings, RiskLevel, StoryAssessment
from app.services import article_pipeline, reel_script

FAITH_ALLOWLIST = ["religion", "religious", "theology", "faith", "christianity",
                   "spirituality", "scripture", "bible", "prayer", "church"]


def _settings(allowlist=None, **kw):
    return AutomationSettings(
        mode=AutomationMode.autonomous,
        auto_publish_enabled=True,
        require_review_for_sensitive_topics=True,
        maximum_risk_for_auto_publish=RiskLevel.low,
        sensitive_category_allowlist=list(allowlist or []),
        **kw,
    )


class TestGatingCategories(unittest.TestCase):
    def test_empty_allowlist_keeps_the_original_behaviour(self):
        # Default must not silently loosen an existing deployment.
        settings = _settings()
        self.assertEqual(
            article_pipeline.gating_categories(["religion", "grief"], settings),
            ["religion", "grief"],
        )

    def test_the_accounts_own_subject_stops_gating(self):
        settings = _settings(FAITH_ALLOWLIST)
        self.assertEqual(
            article_pipeline.gating_categories(["religion", "theology"], settings), []
        )

    def test_hazard_categories_still_gate_alongside_allowed_ones(self):
        # The whole point of allowlisting instead of switching the check off.
        settings = _settings(FAITH_ALLOWLIST)
        for hazard in ("grief", "politics", "mental health", "abortion",
                       "marriage", "family dynamics", "health"):
            with self.subTest(hazard=hazard):
                self.assertEqual(
                    article_pipeline.gating_categories(["religion", hazard], settings),
                    [hazard],
                )

    def test_matching_ignores_case_and_separators(self):
        # Real assessor output contains both "mental health" and "mental_health".
        settings = _settings(["mental health"])
        self.assertEqual(article_pipeline.gating_categories(["Mental_Health"], settings), [])
        self.assertEqual(article_pipeline.gating_categories(["MENTAL HEALTH"], settings), [])

    def test_no_categories_means_nothing_to_gate(self):
        self.assertEqual(article_pipeline.gating_categories([], _settings(FAITH_ALLOWLIST)), [])
        self.assertEqual(article_pipeline.gating_categories(None, _settings()), [])


class TestPublishGate(unittest.TestCase):
    def _assessment(self, categories, risk=RiskLevel.low):
        return StoryAssessment(risk_level=risk, sensitive_categories=list(categories))

    def test_an_allowlisted_only_story_now_clears_the_publish_gate(self):
        settings = _settings(FAITH_ALLOWLIST)
        gating = article_pipeline.gating_categories(["religion"], settings)
        ok, reason = article_pipeline.should_publish(
            self._assessment(["religion"]), settings,
            AutomationMode.autonomous, sensitive=bool(gating))
        self.assertTrue(ok, reason)

    def test_a_hazard_story_is_still_held_for_review(self):
        settings = _settings(FAITH_ALLOWLIST)
        gating = article_pipeline.gating_categories(["religion", "grief"], settings)
        ok, reason = article_pipeline.should_publish(
            self._assessment(["religion", "grief"]), settings,
            AutomationMode.autonomous, sensitive=bool(gating))
        self.assertFalse(ok)
        self.assertIn("sensitive", reason)

    def test_the_risk_gate_still_applies_on_top_of_the_allowlist(self):
        # Allowlisting sensitivity must not become a way past the risk ceiling.
        settings = _settings(FAITH_ALLOWLIST)
        ok, reason = article_pipeline.should_publish(
            self._assessment(["religion"], risk=RiskLevel.high), settings,
            AutomationMode.autonomous, sensitive=False)
        self.assertFalse(ok)
        self.assertIn("risk", reason)


class TestSettingsParsing(unittest.TestCase):
    def _load(self, value):
        with patch.dict(article_pipeline.config.app,
                        {"article_sensitive_category_allowlist": value}, clear=False):
            return article_pipeline.load_automation_settings().sensitive_category_allowlist

    def test_a_list_is_read_as_is(self):
        self.assertEqual(self._load(["religion", "faith"]), ["religion", "faith"])

    def test_a_comma_separated_string_is_accepted(self):
        self.assertEqual(self._load("religion, faith"), ["religion", "faith"])

    def test_blank_entries_are_dropped(self):
        # An empty string must not become an allowlist entry that matches "".
        self.assertEqual(self._load(["religion", "", "  "]), ["religion"])

    def test_a_nonsense_value_falls_back_to_gating_everything(self):
        self.assertEqual(self._load(42), [])


class TestArticleReelLength(unittest.TestCase):
    def test_article_reels_are_shorter_than_the_measured_failure(self):
        # The one narrated Reel published ran 15.3s at 16.3% completion.
        self.assertLess(reel_script.target_seconds(), 15.0)

    def test_the_script_budget_follows_the_duration(self):
        budget = reel_script.char_budget()
        expected = reel_script.target_seconds() * reel_script.DEFAULT_CHARS_PER_SECOND
        self.assertAlmostEqual(budget, int(expected), delta=1)

    def test_the_budget_floor_still_protects_against_a_fragment(self):
        self.assertGreaterEqual(reel_script.char_budget(), reel_script.MIN_CHAR_BUDGET)

    def test_a_shorter_reel_gets_fewer_shots_not_the_same_cuts_compressed(self):
        self.assertLessEqual(reel_script.scene_target(), 4)
        self.assertGreaterEqual(reel_script.scene_target(), 3)


if __name__ == "__main__":
    unittest.main()
