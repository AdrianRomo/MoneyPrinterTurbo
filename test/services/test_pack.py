"""Niche packs.

The property that matters most is the boring one: with the shipped pack loaded,
every value the pipeline reads is identical to the module constant it replaced.
This lands on a pipeline that publishes unattended to a live account, so the
failure mode of a missing or broken pack must be "the default applies".
"""

import os
import tempfile
import unittest

from app.services import carousel, content_scheduler, hashtags, pack, series, verse_card


class TestPackIsANoOp(unittest.TestCase):
    """The shipped pack must reproduce the constants it was generated from."""

    def test_hashtag_sets_match(self):
        self.assertEqual(hashtags._sets(), hashtags.SETS)

    def test_caption_banks_match(self):
        self.assertEqual(pack.typed("captions.questions", hashtags.QUESTIONS),
                         hashtags.QUESTIONS)
        self.assertEqual(pack.typed("captions.save_asks", hashtags.SAVE_ASKS),
                         hashtags.SAVE_ASKS)
        self.assertEqual(
            pack.typed("captions.keyword_leads", hashtags.KEYWORD_LEADS),
            hashtags.KEYWORD_LEADS)

    def test_carousel_subjects_match(self):
        self.assertEqual(carousel.subjects(), carousel.SUBJECTS)

    def test_carousel_copy_matches(self):
        self.assertEqual(pack.typed("carousel.cover_variants", carousel.COVER_VARIANTS),
                         carousel.COVER_VARIANTS)
        self.assertEqual(pack.typed("carousel.questions", carousel.QUESTIONS),
                         carousel.QUESTIONS)

    def test_verse_card_vocabulary_matches(self):
        self.assertEqual(verse_card.background_subjects(),
                         verse_card.BACKGROUND_SUBJECTS)
        self.assertEqual(verse_card.style_suffix(), verse_card.STYLE_SUFFIX)
        self.assertEqual(verse_card.negative_prompt(), verse_card.NEGATIVE_PROMPT)

    def test_series_match(self):
        self.assertEqual(series.all_series(), series.SERIES)

    def test_cadence_matches(self):
        self.assertEqual(content_scheduler.cadence(), content_scheduler.PLAN)


class TestPackDegradesSafely(unittest.TestCase):
    def setUp(self):
        self._dir = pack.PACKS_DIR
        self._cache = dict(pack._cache)

    def tearDown(self):
        pack.PACKS_DIR = self._dir
        pack._cache.clear()
        pack._cache.update(self._cache)

    def _use(self, contents=None, name="probe"):
        pack.PACKS_DIR = tempfile.mkdtemp()
        if contents is not None:
            os.makedirs(os.path.join(pack.PACKS_DIR, name))
            with open(os.path.join(pack.PACKS_DIR, name, "pack.yaml"), "w") as fh:
                fh.write(contents)
        pack._cache.clear()
        return name

    def test_missing_pack_is_an_empty_pack(self):
        self._use(None)
        pack._cache.clear()
        self.assertEqual(pack.load("nope"), {})
        self.assertEqual(pack.value("anything.at.all", "fallback"), "fallback")

    def test_malformed_yaml_does_not_raise(self):
        name = self._use("carousel: [unclosed\n")
        self.assertEqual(pack.load(name), {})

    def test_pack_that_is_not_a_mapping_is_ignored(self):
        name = self._use("- just\n- a\n- list\n")
        self.assertEqual(pack.load(name), {})

    def test_empty_value_falls_back(self):
        """An emptied bank is far more likely an edit slip than an intention."""
        name = self._use("carousel:\n  questions: []\n")
        pack._cache[pack.pack_name()] = pack.load(name)
        self.assertEqual(pack.value("carousel.questions", ["a"]), ["a"])

    def test_wrong_type_falls_back(self):
        name = self._use("carousel:\n  questions: not-a-list\n")
        pack._cache[pack.pack_name()] = pack.load(name)
        self.assertEqual(pack.typed("carousel.questions", ["a"]), ["a"])

    def test_missing_key_falls_back(self):
        name = self._use("brand:\n  wordmark: probe\n")
        pack._cache[pack.pack_name()] = pack.load(name)
        self.assertEqual(pack.value("carousel.questions", ["a"]), ["a"])
        self.assertEqual(pack.value("brand.wordmark", "x"), "probe")


class TestPackOverrides(unittest.TestCase):
    """A pack that DOES define something must actually be used."""

    def setUp(self):
        self._dir, self._cache = pack.PACKS_DIR, dict(pack._cache)
        pack.PACKS_DIR = tempfile.mkdtemp()
        os.makedirs(os.path.join(pack.PACKS_DIR, pack.pack_name()))
        with open(os.path.join(pack.PACKS_DIR, pack.pack_name(), "pack.yaml"), "w") as fh:
            fh.write(
                "carousel:\n"
                "  subjects:\n"
                "    lighthouses:\n"
                "      noun: LIGHTHOUSES\n"
                "      query: lighthouse coast\n"
                "      label_locations: true\n"
                "    broken_one:\n"
                "      noun: NO QUERY\n"
                "cadence:\n"
                "  reel: {per_day: 4}\n"
                "  nonsense: {per_day: 9}\n"
            )
        pack._cache.clear()

    def tearDown(self):
        pack.PACKS_DIR = self._dir
        pack._cache.clear()
        pack._cache.update(self._cache)

    def test_subjects_come_from_the_pack(self):
        got = carousel.subjects()
        self.assertIn("lighthouses", got)
        self.assertEqual(got["lighthouses"],
                         ("LIGHTHOUSES", "lighthouse coast", None, True))

    def test_malformed_subject_is_skipped_not_fatal(self):
        self.assertNotIn("broken_one", carousel.subjects())

    def test_cadence_comes_from_the_pack(self):
        self.assertEqual(content_scheduler.cadence()["reel"], {"per_day": 4})

    def test_cadence_ignores_unknown_formats(self):
        self.assertNotIn("nonsense", content_scheduler.cadence())


if __name__ == "__main__":
    unittest.main()
