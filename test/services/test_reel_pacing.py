import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.models.article import Scene
from app.services import article_llm, brand_footage, reel_script


class PacingTestCase(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        config.app["script_style"] = "brand"
        config.app["script_target_seconds"] = 20
        config.app["script_chars_per_second"] = 18

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)


class TestSceneTarget(PacingTestCase):
    def test_twenty_seconds_wants_a_handful_of_shots(self):
        """基线那条 70 秒片子有 15 个镜头；快切看起来像屏保，不像用心拍的。"""
        self.assertEqual(reel_script.scene_target(), 4)

    def test_target_is_clamped_at_both_ends(self):
        config.app["script_target_seconds"] = 3
        self.assertEqual(reel_script.scene_target(), 3)
        config.app["script_target_seconds"] = 120
        self.assertEqual(reel_script.scene_target(), 5)


class TestLimitScenes(PacingTestCase):
    def scenes(self, *lengths):
        return [Scene(narration="x" * n, duration_weight=1.0) for n in lengths]

    def test_scenes_within_the_limit_are_untouched(self):
        scenes = self.scenes(50, 50, 50)
        self.assertEqual(len(article_llm._limit_scenes(scenes, 4)), 3)

    def test_extra_scenes_are_merged_down(self):
        self.assertEqual(len(article_llm._limit_scenes(self.scenes(*([40] * 9)), 4)), 4)

    def test_narration_survives_the_merge(self):
        """镜头数是画面问题，旁白是脚本本身，不能被删掉。"""
        scenes = [Scene(narration="one."), Scene(narration="two."), Scene(narration="three.")]
        merged = article_llm._limit_scenes(scenes, 1)
        self.assertEqual(len(merged), 1)
        for word in ("one.", "two.", "three."):
            self.assertIn(word, merged[0].narration)

    def test_the_shortest_pair_merges_first(self):
        scenes = [Scene(narration="a" * 200), Scene(narration="b" * 5), Scene(narration="c" * 5)]
        merged = article_llm._limit_scenes(scenes, 2)
        self.assertEqual(merged[0].narration, "a" * 200)

    def test_duration_weight_is_carried_over(self):
        scenes = [Scene(narration="a", duration_weight=1.5), Scene(narration="b", duration_weight=2.0)]
        merged = article_llm._limit_scenes(scenes, 1)
        self.assertAlmostEqual(merged[0].duration_weight, 3.5)

    def test_empty_and_degenerate_input(self):
        self.assertEqual(article_llm._limit_scenes([], 4), [])
        self.assertEqual(len(article_llm._limit_scenes(self.scenes(10, 10), 0)), 2)


class TestArticleModeInheritsTheDiscipline(PacingTestCase):
    def test_guidance_reaches_the_article_prompt_without_the_verse(self):
        """Article Mode 的契约是每句话都能追溯到来源，塞经文会破坏这一点。"""
        block = reel_script.guidance_block("mornings", include_anchor=False)
        self.assertIn("HARD LIMIT", block)
        self.assertIn("Do not quote scripture", block)
        self.assertNotIn("Quote it EXACTLY", block)

    def test_the_generic_path_still_gets_the_verse(self):
        with patch("app.services.reel_script.fetch_anchor",
                   return_value=("Psalm 23:1", "The LORD is my shepherd")):
            block = reel_script.guidance_block("trust", include_anchor=True)
        self.assertIn("The LORD is my shepherd", block)


class TestOverLengthFeedsTheRewriteLoop(PacingTestCase):
    class FakeReview:
        def __init__(self):
            self.issues = []
            self.approved = True
            self.confidence = 0.9

    def script_of(self, narration):
        from app.models.article import GeneratedScript
        return GeneratedScript(narration=narration)

    def test_an_over_long_script_becomes_a_review_issue(self):
        review = self.FakeReview()
        article_llm._note_length(self.script_of("x" * 900), review)
        self.assertFalse(review.approved)
        self.assertTrue(any("over the 360-character limit" in i for i in review.issues))

    def test_a_short_script_is_left_approved(self):
        review = self.FakeReview()
        article_llm._note_length(self.script_of("Short enough. Really it is."), review)
        self.assertTrue(review.approved)
        self.assertEqual(review.issues, [])

    def test_nothing_happens_when_the_flag_is_off(self):
        config.app["script_style"] = "default"
        review = self.FakeReview()
        article_llm._note_length(self.script_of("x" * 900), review)
        self.assertTrue(review.approved)


class TestFootageRepetition(unittest.TestCase):
    def test_different_terms_can_collide_without_avoid(self):
        a = brand_footage.subject_for("quiet morning kitchen", 0)
        b = brand_footage.subject_for("hope at first light", 3)
        self.assertEqual(a, b)  # the defect this fix exists for

    def test_avoid_walks_on_to_a_free_subject(self):
        a = brand_footage.subject_for("quiet morning kitchen", 0)
        b = brand_footage.subject_for("hope at first light", 3, avoid={a})
        self.assertNotEqual(a, b)

    def test_a_whole_reel_comes_out_distinct(self):
        terms = ["quiet morning kitchen", "ordinary daily work",
                 "waiting in stillness", "hope at first light"]
        used = set()
        for index, term in enumerate(terms):
            used.add(brand_footage.subject_for(term, index, avoid=used))
        self.assertEqual(len(used), len(terms))

    def test_selection_is_still_deterministic(self):
        self.assertEqual(brand_footage.subject_for("hope at first light", 3),
                         brand_footage.subject_for("hope at first light", 3))

    def test_an_exhausted_mood_still_returns_something(self):
        subjects = set(brand_footage.MOODS["hope"]["subjects"])
        self.assertIn(brand_footage.subject_for("hope dawn new light", 0, avoid=subjects),
                      subjects)


if __name__ == "__main__":
    unittest.main()
