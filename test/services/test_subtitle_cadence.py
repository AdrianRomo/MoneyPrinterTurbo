import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.services import subtitle_cadence


def words(*spec):
    """(text, start, end) triples from a compact "text@start-end" spec."""
    out = []
    for item in spec:
        text, times = item.rsplit("@", 1)
        start, end = times.split("-")
        out.append((text, float(start), float(end)))
    return out


def evenly(text: str, start: float = 0.0, step: float = 0.4):
    out = []
    for index, word in enumerate(text.split()):
        begin = start + index * step
        out.append((word, begin, begin + step * 0.9))
    return out


class CadenceTestCase(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        config.app["subtitle_cadence"] = "words"

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)


class TestFlag(CadenceTestCase):
    def test_disabled_unless_selected(self):
        """默认仍按标点分句，保持改动前行为。"""
        config.app["subtitle_cadence"] = "punctuation"
        self.assertFalse(subtitle_cadence.enabled())
        config.app.pop("subtitle_cadence")
        self.assertFalse(subtitle_cadence.enabled())

    def test_limits_are_configurable_and_fall_back(self):
        config.app["subtitle_cue_max_words"] = 3
        config.app["subtitle_cue_max_seconds"] = "nonsense"
        max_words, _, _, max_seconds = subtitle_cadence.limits()
        self.assertEqual(max_words, 3)
        self.assertEqual(max_seconds, subtitle_cadence.MAX_SECONDS)


class TestGrouping(CadenceTestCase):
    def test_empty_input(self):
        self.assertEqual(subtitle_cadence.group([]), [])

    def test_accepts_objects_as_well_as_tuples(self):
        word = type("W", (), {"word": "hello", "start": 0.0, "end": 0.5})()
        self.assertEqual(subtitle_cadence.group([word])[0]["msg"], "hello")

    def test_cues_stay_within_the_word_ceiling(self):
        cues = subtitle_cadence.group(evenly("one two three four five six seven eight"))
        for cue in cues:
            self.assertLessEqual(len(cue["msg"].split()), subtitle_cadence.MAX_WORDS + 1)

    def test_cues_stay_within_the_character_ceiling(self):
        cues = subtitle_cadence.group(evenly(
            "extraordinarily complicated vocabulary appears here repeatedly today"))
        for cue in cues:
            self.assertLessEqual(len(cue["msg"]), subtitle_cadence.MAX_CHARS)

    def test_no_cue_outlives_the_ceiling(self):
        """成片里不能有超过上限的字幕，这是这一阶段唯一可验证的硬指标。"""
        cues = subtitle_cadence.group(evenly("one two three four five six", step=0.9))
        for cue in cues:
            self.assertLessEqual(cue["end_time"] - cue["start_time"],
                                 subtitle_cadence.MAX_SECONDS + 0.001)

    def test_a_pause_is_a_cut(self):
        cues = subtitle_cadence.group(words(
            "the@0.0-0.3", "dishes@0.3-0.7",
            "can@2.0-2.3", "wait@2.3-2.6",
        ))
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0]["msg"], "the dishes")
        self.assertEqual(cues[1]["msg"], "can wait")

    def test_a_pause_is_preserved_not_papered_over(self):
        cues = subtitle_cadence.group(words(
            "the@0.0-0.3", "dishes@0.3-0.7",
            "can@2.0-2.3", "wait@2.3-2.6",
        ))
        self.assertLess(cues[0]["end_time"], cues[1]["start_time"])

    def test_timings_come_from_the_words(self):
        cues = subtitle_cadence.group(words("the@1.5-1.8", "dishes@1.8-2.4"))
        self.assertEqual(cues[0]["start_time"], 1.5)


class TestOrphans(CadenceTestCase):
    def test_a_stranded_word_merges_backwards(self):
        """单词字幕像卡顿；`in the rustle of / leaves,` 必须合成一条。"""
        cues = subtitle_cadence.group(words(
            "in@0.0-0.2", "the@0.2-0.4", "rustle@0.4-0.6", "of@0.6-0.8",
            "leaves@0.8-1.0",
        ))
        self.assertEqual(len(cues), 1)
        self.assertEqual(cues[0]["msg"], "in the rustle of leaves")

    def test_a_leading_orphan_merges_forwards(self):
        cues = subtitle_cadence.group(words(
            "and@0.0-0.2", "then@1.0-1.2", "the@1.2-1.4", "kettle@1.4-1.6",
        ))
        self.assertTrue(cues[0]["msg"].startswith("and"))

    def test_an_orphan_that_cannot_fit_is_left_alone(self):
        # Merging across a real pause would desynchronise the caption from the
        # audio, which is worse than a short cue.
        cues = subtitle_cadence.group(words(
            "one@0.0-0.4", "two@0.4-0.8", "three@0.8-1.2", "four@1.2-1.6",
            "gratitude@6.0-6.6",
        ))
        self.assertEqual(cues[-1]["msg"], "gratitude")


class TestHolding(CadenceTestCase):
    def test_short_gaps_are_closed_so_captions_do_not_blink(self):
        # Six words split on the word ceiling, with a 0.2s gap at the seam —
        # too short to be a breath, so the first cue holds until the second.
        cues = subtitle_cadence.group(words(
            "one@0.0-0.2", "two@0.2-0.4", "three@0.4-0.6", "four@0.6-0.8",
            "five@1.0-1.2", "six@1.2-1.4",
        ))
        self.assertEqual(len(cues), 2)
        self.assertEqual(cues[0]["end_time"], cues[1]["start_time"])

    def test_a_very_short_cue_is_lifted_to_a_readable_minimum(self):
        cues = subtitle_cadence.group(words("hello@0.0-0.2"))
        self.assertGreaterEqual(cues[0]["end_time"] - cues[0]["start_time"],
                                subtitle_cadence.MIN_SECONDS)

    def test_holding_never_overlaps_the_next_cue(self):
        cues = subtitle_cadence.group(words(
            "hi@0.0-0.1", "there@2.0-2.4", "friend@2.4-2.8",
        ))
        self.assertLessEqual(cues[0]["end_time"], cues[1]["start_time"])

    def test_holding_never_breaks_the_ceiling(self):
        cues = subtitle_cadence.group(words(
            "one@0.0-0.5", "two@0.5-1.0", "three@1.0-1.5",
            "next@1.9-2.3", "cue@2.3-2.7",
        ))
        for cue in cues:
            self.assertLessEqual(cue["end_time"] - cue["start_time"],
                                 subtitle_cadence.MAX_SECONDS + 0.001)


class TestReport(CadenceTestCase):
    def test_report_counts_breaches(self):
        cues = subtitle_cadence.group(evenly("one two three four five six"))
        report = subtitle_cadence.report(cues)
        self.assertIn("cues", report)
        self.assertIn("0 over", report)

    def test_report_handles_nothing(self):
        self.assertIn("no cues", subtitle_cadence.report([]))


if __name__ == "__main__":
    unittest.main()
