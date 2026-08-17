import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.services import llm, reel_script


class ReelScriptTestCase(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        config.app["script_style"] = "brand"
        config.app["script_verse_anchor"] = False
        config.app["script_target_seconds"] = 20

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)


class TestFlagAndBudget(ReelScriptTestCase):
    def test_disabled_unless_explicitly_selected(self):
        """默认必须关闭：整套 reel 规则通过配置开关生效，回退只需改一个键。"""
        config.app["script_style"] = "default"
        self.assertFalse(reel_script.enabled())
        config.app.pop("script_style")
        self.assertFalse(reel_script.enabled())

    def test_budget_follows_target_duration(self):
        config.app["script_chars_per_second"] = 18
        config.app["script_target_seconds"] = 20
        self.assertEqual(reel_script.char_budget(), 360)
        config.app["script_target_seconds"] = 30
        self.assertEqual(reel_script.char_budget(), 540)

    def test_unusable_config_falls_back_instead_of_raising(self):
        # A typo in config.toml must not take the script path down with it.
        config.app["script_target_seconds"] = "twenty"
        config.app["script_chars_per_second"] = 18
        self.assertEqual(reel_script.char_budget(), 360)
        config.app["script_target_seconds"] = -5
        self.assertEqual(reel_script.char_budget(), 360)

    def test_budget_never_collapses_to_a_fragment(self):
        config.app["script_target_seconds"] = 0.5
        self.assertEqual(reel_script.char_budget(), reel_script.MIN_CHAR_BUDGET)


class TestSentenceSplitting(ReelScriptTestCase):
    def test_splits_plain_sentences(self):
        text = "One thing. Then another. Then the last."
        self.assertEqual(len(reel_script.sentences(text)), 3)

    def test_period_inside_a_closing_quote_is_a_boundary(self):
        """经文以 `…me.” Coffee…` 结尾，不能被当成一个 20 词的长句。"""
        text = 'Mornings are quiet. “I can do all things through Christ.” Coffee drips.'
        parts = reel_script.sentences(text)
        self.assertEqual(len(parts), 3)
        self.assertTrue(parts[1].endswith("”"))

    def test_kjv_punctuation_inside_the_quote_is_still_a_boundary(self):
        text = 'Your quiet labor matters. “…as to the Lord, and not unto men;” Your quiet labor matters.'
        self.assertEqual(len(reel_script.sentences(text)), 3)

    def test_quote_mid_sentence_is_not_a_boundary(self):
        text = 'A quote like "this one" should not split the line.'
        self.assertEqual(len(reel_script.sentences(text)), 1)

    def test_splitting_is_lossless(self):
        text = 'Mornings are quiet. “I can do all things.” Coffee drips into the mug.'
        self.assertEqual(" ".join(reel_script.sentences(text)), text)

    def test_first_line_is_the_opening_sentence(self):
        text = "The dishes can wait. God is not counting them. The dishes can wait."
        self.assertEqual(reel_script.first_line(text), "The dishes can wait.")


class TestProblems(ReelScriptTestCase):
    def test_a_clean_script_has_no_problems(self):
        text = ("The dishes can wait. God is not counting them. "
                "He is already in the kitchen. The dishes can wait.")
        self.assertEqual(reel_script.problems(text), [])

    def test_over_budget_is_reported_with_the_overshoot(self):
        text = "Sentence one. Sentence two. Sentence three. " + ("Filler words here. " * 40)
        issues = reel_script.problems(text)
        self.assertTrue(any("over the hard limit" in issue for issue in issues))

    def test_a_single_run_on_sentence_is_reported(self):
        """模型常把整篇脚本写成一句话，这样既没有 hook，字幕也无处断句。"""
        text = ("Steam rises from your mug as you scroll past the sunrise and a calm "
                "voice reminds you that the day still belongs to God who made it.")
        issues = reel_script.problems(text)
        self.assertTrue(any("sentence(s)" in issue for issue in issues))
        self.assertTrue(any("words" in issue for issue in issues))

    def test_long_opening_sentence_is_reported_as_a_hook_failure(self):
        text = ("Every single morning when the coffee steams and the kettle clatters "
                "you are being invited. Short one. Short two.")
        issues = reel_script.problems(text)
        self.assertTrue(any("opening sentence" in issue for issue in issues))

    def test_correction_note_lists_every_issue(self):
        note = reel_script.correction_note(["it is 900 characters", "only 1 sentence"])
        self.assertIn("- it is 900 characters", note)
        self.assertIn("- only 1 sentence", note)


class TestTrimToBudget(ReelScriptTestCase):
    def test_trims_on_a_sentence_boundary(self):
        text = " ".join(["This is a complete sentence."] * 40)
        trimmed = reel_script.trim_to_budget(text, 100)
        self.assertLessEqual(len(trimmed), 100)
        self.assertTrue(trimmed.endswith("."))

    def test_never_cuts_mid_sentence(self):
        """宁可略微超长，也不要把句子从中间截断。"""
        text = ("One enormous opening clause that will not fit inside the budget "
                "no matter how it is measured.")
        self.assertEqual(reel_script.trim_to_budget(text, 20), text)

    def test_leaves_a_script_within_budget_untouched(self):
        text = "Short enough. Really."
        self.assertEqual(reel_script.trim_to_budget(text, 400), text)


class TestVerseAnchor(ReelScriptTestCase):
    def test_anchor_uses_api_text_not_model_text(self):
        verse = type("V", (), {"reference": "Psalm 23:1", "text": "The LORD is my shepherd",
                               "translation": "KJV"})()
        with patch("app.services.verse_card.pick_reference", return_value="Psalm 23:1"), \
                patch("app.services.verse_card.fetch_verse", return_value=verse) as fetch:
            anchor = reel_script.fetch_anchor("trust")
        fetch.assert_called_once_with("Psalm 23:1")
        self.assertEqual(anchor, ("Psalm 23:1", "The LORD is my shepherd"))

    def test_unverifiable_reference_is_dropped_not_invented(self):
        """API 不认的经文引用一律丢弃，绝不让模型凭记忆补全。"""
        with patch("app.services.verse_card.pick_reference", return_value="Hesitations 4:12"), \
                patch("app.services.verse_card.fetch_verse", return_value=None):
            self.assertIsNone(reel_script.fetch_anchor("trust"))

    def test_a_failing_lookup_never_breaks_script_generation(self):
        with patch("app.services.verse_card.pick_reference", side_effect=RuntimeError("boom")):
            self.assertIsNone(reel_script.fetch_anchor("trust"))
        with patch("app.services.verse_card.pick_reference", return_value="Psalm 23:1"), \
                patch("app.services.verse_card.fetch_verse", side_effect=RuntimeError("network")):
            self.assertIsNone(reel_script.fetch_anchor("trust"))

    def test_guidance_forbids_quoting_when_no_anchor_is_available(self):
        config.app["script_verse_anchor"] = True
        with patch("app.services.reel_script.fetch_anchor", return_value=None):
            block = reel_script.guidance_block("trust")
        self.assertIn("Do not quote scripture", block)

    def test_guidance_carries_the_verified_text_verbatim(self):
        config.app["script_verse_anchor"] = True
        with patch("app.services.reel_script.fetch_anchor",
                   return_value=("Psalm 23:1", "The LORD is my shepherd")):
            block = reel_script.guidance_block("trust")
        self.assertIn("The LORD is my shepherd", block)
        self.assertIn("Quote it EXACTLY", block)


class TestPromptWiring(ReelScriptTestCase):
    def test_discipline_is_absent_when_the_flag_is_off(self):
        config.app["script_style"] = "default"
        self.assertNotIn("Reel Discipline", llm.build_script_prompt("mornings"))

    def test_discipline_is_appended_last_so_it_outranks_user_requirements(self):
        prompt = llm.build_script_prompt("mornings", video_script_prompt="make it long and detailed")
        self.assertIn("Reel Discipline", prompt)
        self.assertGreater(prompt.index("Reel Discipline"), prompt.index("make it long and detailed"))

    def test_budget_appears_in_the_prompt(self):
        config.app["script_chars_per_second"] = 18
        self.assertIn("360 characters", llm.build_script_prompt("mornings"))


class TestShaping(ReelScriptTestCase):
    def test_a_clean_script_is_returned_untouched_without_a_retry(self):
        text = ("The dishes can wait. God is not counting them. "
                "He is already in the kitchen. The dishes can wait.")
        with patch("app.services.llm._generate_response") as gen:
            self.assertEqual(llm._shape_reel_script(text, "prompt"), text)
        gen.assert_not_called()

    def test_a_broken_script_earns_exactly_one_retry(self):
        bad = "One very long run on sentence that has no hook and never stops going on and on"
        good = "The dishes can wait. God is not counting them. The dishes can wait."
        with patch("app.services.llm._generate_response", return_value=good) as gen:
            result = llm._shape_reel_script(bad, "prompt")
        self.assertEqual(gen.call_count, 1)
        self.assertEqual(result, good)

    def test_a_worse_retry_is_rejected(self):
        """重写没有变好就不采用，避免修了 hook 又超长。"""
        bad = "One very long run on sentence that has no hook and never stops going on and on"
        worse = "Even longer single sentence " * 30
        with patch("app.services.llm._generate_response", return_value=worse):
            result = llm._shape_reel_script(bad, "prompt")
        self.assertNotIn("Even longer single sentence Even longer", result)

    def test_a_failing_retry_still_returns_a_trimmed_script(self):
        bad = " ".join(["This sentence is filler."] * 60)
        with patch("app.services.llm._generate_response", side_effect=RuntimeError("api down")):
            result = llm._shape_reel_script(bad, "prompt")
        self.assertLessEqual(len(result), reel_script.char_budget())

    def test_generate_script_leaves_long_scripts_alone_when_disabled(self):
        """关闭时行为必须和改动前完全一致：不重写、不裁剪、只调用一次模型。"""
        config.app["script_style"] = "default"
        long_script = " ".join(["Filler sentence here."] * 60)
        with patch("app.services.llm._generate_response", return_value=long_script) as gen:
            result = llm.generate_script(video_subject="mornings")
        self.assertEqual(gen.call_count, 1)
        self.assertEqual(result, long_script)
        self.assertGreater(len(result), reel_script.char_budget())

    def test_generate_script_shapes_the_script_when_enabled(self):
        long_script = " ".join(["Filler sentence here."] * 60)
        good = "The dishes can wait. God is not counting them. The dishes can wait."
        with patch("app.services.llm._generate_response", side_effect=[long_script, good]):
            result = llm.generate_script(video_subject="mornings")
        self.assertEqual(result, good)


if __name__ == "__main__":
    unittest.main()
