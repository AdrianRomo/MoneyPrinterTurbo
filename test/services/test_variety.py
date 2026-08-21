"""Feed-level variety.

Every other gate in quality.py measures one image and asks "is this good?".
These cover the gate that asks the question none of them did: "is this
DIFFERENT from the last few?" — the gap that let 41 individually-passing cards
add up to one repetitive feed.
"""

import os
import tempfile
import unittest

from PIL import Image

from app.services import quality, rotation, verse_card


def _solid(rgb, size=(64, 64)):
    return Image.new("RGB", size, rgb)


class TestPaletteBucket(unittest.TestCase):
    def test_dark_muted_amber_is_warm_dark_muted(self):
        # Near-grey with a warm bias. A SATURATED amber like (70, 55, 35) reads
        # "moderate", which is right: a real photograph averages far closer to
        # grey than a flat swatch of its dominant hue does.
        self.assertEqual(quality.palette_bucket(_solid((70, 64, 58))),
                         "warm/dark/muted")

    def test_bright_saturated_blue_is_not_muted_or_dark(self):
        family, band, colour = quality.palette_bucket(_solid((40, 120, 245))).split("/")
        self.assertEqual(family, "blue")
        self.assertNotEqual(colour, "muted")

    def test_near_white_reads_as_light(self):
        self.assertIn("/light/", quality.palette_bucket(_solid((242, 240, 236))))

    def test_the_published_range_was_one_corner_of_the_space(self):
        # Measured over the 41 cards published 2026-08-14..27: 40 of 41 landed
        # "muted", none landed "light", and 27 sat in just two buckets
        # (warm/dark/muted and blue/dark/muted). These fixtures are averages
        # from that corner — the gate has to be able to tell them apart from
        # anything the new looks produce.
        for rgb in ((60, 58, 54), (45, 50, 58), (58, 62, 56)):
            self.assertIn("/muted", quality.palette_bucket(_solid(rgb)))
            self.assertIn("/dark/", quality.palette_bucket(_solid(rgb)))


class TestCheckVariety(unittest.TestCase):
    def _reg(self, cls="water", look="muted-warm", palette="warm/dark/muted"):
        return {"subject_class": cls, "look": look, "palette": palette}

    def test_nothing_published_yet_passes(self):
        ok, _ = quality.check_variety(self._reg(), [])
        self.assertTrue(ok)

    def test_an_identical_register_is_rejected(self):
        history = [self._reg()] + [self._reg(cls=c, look=l) for c, l in
                                   (("sky", "blue-hour"), ("forest", "clear-vivid"))]
        ok, why = quality.check_variety(self._reg(), history)
        self.assertFalse(ok)
        self.assertIn("identical register", why)

    def test_the_same_subject_class_too_soon_is_rejected(self):
        history = [self._reg(cls="water", look="blue-hour", palette="blue/dark/muted")]
        ok, why = quality.check_variety(self._reg(cls="water"), history)
        self.assertFalse(ok)
        self.assertIn("subject class", why)

    def test_the_same_look_too_soon_is_rejected(self):
        history = [self._reg(cls="sky", look="muted-warm", palette="blue/mid/muted")]
        ok, why = quality.check_variety(self._reg(cls="forest", look="muted-warm"), history)
        self.assertFalse(ok)
        self.assertIn("look", why)

    def test_a_genuinely_different_card_passes(self):
        history = [self._reg(cls="water", look="muted-warm")]
        ok, _ = quality.check_variety(
            self._reg(cls="architecture", look="high-key-airy",
                      palette="warm/light/moderate"), history)
        self.assertTrue(ok)

    def test_a_class_falls_out_of_the_window_and_becomes_usable_again(self):
        history = [self._reg(cls="water", look="muted-warm")]
        history += [self._reg(cls=c, look=l, palette=p) for c, l, p in
                    (("sky", "blue-hour", "blue/dark/muted"),
                     ("forest", "clear-vivid", "green/mid/vivid"),
                     ("desert", "backlit-golden", "warm/mid/moderate"))]
        ok, _ = quality.check_variety(
            self._reg(cls="water", look="overcast-cool", palette="blue/mid/muted"),
            history)
        self.assertTrue(ok, "the window must forget, or the pool runs dry")

    def test_only_the_recent_window_is_considered(self):
        old = [self._reg()] + [self._reg(cls=f"c{i}", look=f"l{i}", palette=f"p{i}")
                               for i in range(quality.VARIETY_WINDOW)]
        ok, _ = quality.check_variety(self._reg(), old)
        self.assertTrue(ok)

    def test_a_malformed_history_entry_does_not_raise(self):
        ok, _ = quality.check_variety(self._reg(), ["not a dict", None])
        self.assertTrue(ok)

    def test_an_empty_register_passes(self):
        ok, _ = quality.check_variety({}, [self._reg()])
        self.assertTrue(ok)


class TestRegisterOf(unittest.TestCase):
    def test_palette_is_measured_from_pixels_not_taken_from_the_prompt(self):
        # A prompt asking for airy light and a model returning something dark is
        # exactly the disagreement worth catching.
        reg = quality.register_of(_solid((20, 20, 24)),
                                  {"subject_class": "sky", "look": "high-key-airy"})
        self.assertEqual(reg["look"], "high-key-airy")
        self.assertIn("/dark/", reg["palette"])

    def test_missing_meta_degrades_to_palette_only(self):
        reg = quality.register_of(_solid((70, 55, 35)))
        self.assertEqual(reg["subject_class"], "")
        self.assertTrue(reg["palette"])


class TestVerseCardRotation(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = verse_card._state_path
        verse_card._state_path = lambda: os.path.join(self._tmp.name, "used_references.json")

    def tearDown(self):
        verse_card._state_path = self._orig
        self._tmp.cleanup()

    def test_backgrounds_exhaust_the_pool_before_repeating(self):
        pool = verse_card.background_subjects()
        picked = [verse_card.choose_background_subject() for _ in range(len(pool))]
        self.assertEqual(len(set(picked)), len(pool))

    def test_looks_rotate_through_every_look(self):
        names = {l["name"] for l in verse_card.STYLE_LOOKS}
        picked = {verse_card.choose_look().get("name") for _ in names}
        self.assertEqual(picked, names)

    def test_choosing_without_remembering_writes_no_state(self):
        # Not "returns the same answer twice": never-used candidates are
        # shuffled among themselves by design, so a fresh pool is deliberately
        # non-deterministic. What must hold is that nothing was recorded.
        verse_card.choose_background_subject(remember_it=False)
        self.assertEqual(
            rotation.load_history(verse_card._rotation_path("used_backgrounds.json")),
            [])


class TestStyleComposition(unittest.TestCase):
    def test_a_look_contributes_its_light_and_the_core(self):
        style = verse_card.style_for({"name": "x", "suffix": "low-key, 85mm"})
        self.assertIn("low-key", style)
        self.assertIn("negative space", style)

    def test_a_missing_look_falls_back_to_the_shipped_string(self):
        # "no looks configured" must degrade to the behaviour that shipped, not
        # to a background with no style direction at all.
        self.assertEqual(verse_card.style_for({}), verse_card.style_suffix())
        self.assertEqual(verse_card.style_for(None), verse_card.style_suffix())

    def test_every_look_is_distinct(self):
        suffixes = [l["suffix"] for l in verse_card.STYLE_LOOKS]
        self.assertEqual(len(set(suffixes)), len(suffixes))

    def test_the_incumbent_look_is_still_in_rotation(self):
        names = [l["name"] for l in verse_card.STYLE_LOOKS]
        self.assertIn("muted-warm", names)


class TestSubjectClass(unittest.TestCase):
    def test_every_shipped_subject_classifies(self):
        unclassified = [s for s in verse_card.BACKGROUND_SUBJECTS
                        if verse_card.subject_class(s) == "other"]
        self.assertEqual(unclassified, [])

    def test_scale_cues_beat_the_scene_nouns_they_contain(self):
        self.assertEqual(
            verse_card.subject_class("close detail of lichen on grey stone"),
            "texture")

    def test_a_landscape_that_mentions_sky_is_not_a_sky_frame(self):
        self.assertNotEqual(
            verse_card.subject_class("rolling hills under a wide pastel sky"),
            "sky")

    def test_a_frame_with_no_terrain_is_a_sky_frame(self):
        self.assertEqual(
            verse_card.subject_class("high cirrus streaks across a pale sky"),
            "sky")

    def test_the_pool_spans_many_registers(self):
        classes = {verse_card.subject_class(s) for s in verse_card.BACKGROUND_SUBJECTS}
        self.assertGreaterEqual(len(classes), 8)

    def test_no_class_dominates_the_pool(self):
        import collections
        counts = collections.Counter(verse_card.subject_class(s)
                                     for s in verse_card.BACKGROUND_SUBJECTS)
        worst = counts.most_common(1)[0][1]
        self.assertLess(worst / len(verse_card.BACKGROUND_SUBJECTS), 0.30)

    def test_an_unknown_subject_is_other_rather_than_a_wildcard(self):
        self.assertEqual(verse_card.subject_class("zzz nothing at all"), "other")


if __name__ == "__main__":
    unittest.main()


class TestInkPolarity(unittest.TestCase):
    """Dark type on a lightened ground — the half of the fix that widens light.

    Rotating the prompt vocabulary alone cannot produce a light card: the scrim
    darkens adaptively to protect WHITE type, so a high-key background composes
    to a dark card anyway. These cover the maths that lets the other polarity
    exist, which is not symmetric with the original.
    """

    def test_the_alpha_actually_reaches_the_target(self):
        for bg in (0.05, 0.20, 0.40, 0.60, 0.90):
            alpha = quality.alpha_for_target(bg, ink="dark") / 255
            final = bg + (1.0 - bg) * alpha      # white overlay
            ok, why = quality.check_card(final, "dark")
            self.assertTrue(ok, f"bg={bg}: {why}")

    def test_the_light_ink_alpha_still_reaches_its_target(self):
        for bg in (0.05, 0.20, 0.40, 0.60, 0.90):
            alpha = quality.alpha_for_target(bg) / 255
            final = bg * (1 - alpha)             # black overlay
            ok, why = quality.check_card(final)
            self.assertTrue(ok, f"bg={bg}: {why}")

    def test_an_already_light_ground_needs_no_veil(self):
        self.assertEqual(quality.alpha_for_target(0.85, ink="dark"), 0)

    def test_an_already_dark_ground_needs_no_veil_for_white_type(self):
        self.assertEqual(quality.alpha_for_target(0.05), 0)

    def test_the_ratio_is_not_symmetric(self):
        # A mid-grey ground is poor for white type and good for dark type. A
        # single threshold for both would pass cards that are merely grey.
        self.assertLess(quality.contrast_ratio(0.5), quality.contrast_ratio(0.5, "dark"))

    def test_dark_ink_has_a_stricter_target_so_the_card_is_really_light(self):
        self.assertGreater(quality.target_ratio_for("dark"), quality.target_ratio_for("light"))
        self.assertGreater(quality.min_ratio_for("dark"), quality.min_ratio_for("light"))

    def test_a_muddy_mid_ground_is_rejected_for_dark_type(self):
        ok, _ = quality.check_card(0.20, "dark")
        self.assertFalse(ok)

    def test_an_unknown_ink_behaves_like_the_shipped_one(self):
        self.assertEqual(quality.contrast_ratio(0.3, "nonsense"),
                         quality.contrast_ratio(0.3, "light"))

    def test_every_look_declares_an_ink(self):
        for look in verse_card.STYLE_LOOKS:
            self.assertIn(look.get("ink"), ("light", "dark"), look["name"])

    def test_both_polarities_are_in_rotation(self):
        inks = {look["ink"] for look in verse_card.STYLE_LOOKS}
        self.assertEqual(inks, {"light", "dark"})

    def test_the_register_carries_the_ink_for_the_story_twin(self):
        reg = quality.register_of(_solid((200, 200, 200)), {"ink": "dark"})
        self.assertEqual(reg["ink"], "dark")

    def test_variety_ignores_ink_because_the_look_already_implies_it(self):
        a = {"subject_class": "sky", "look": "x", "palette": "p", "ink": "dark"}
        b = dict(a, ink="light")
        ok, why = quality.check_variety(b, [a])
        self.assertFalse(ok, "differing only by ink must not read as varied")


class TestComposedPolarity(unittest.TestCase):
    """Integration: the same background must compose lighter in dark ink."""

    def setUp(self):
        self.verse = verse_card.Verse(
            reference="Psalm 23:1",
            text="The LORD is my shepherd; I shall not want.",
            translation="WEBBE", verses=[])
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def _compose(self, bg, ink):
        out = os.path.join(self._tmp.name, f"{ink}.jpg")
        return verse_card.compose_card(bg, self.verse, kind="post",
                                       out_path=out, ink=ink)

    def test_dark_ink_composes_a_lighter_card_than_light_ink(self):
        bg = Image.new("RGB", (896, 1152), (150, 148, 144))
        light = self._compose(bg, "light")
        dark = self._compose(bg, "dark")
        self.assertTrue(light and dark, "both polarities must produce a card")
        self.assertGreater(
            quality.aesthetics(Image.open(dark))["brightness"],
            quality.aesthetics(Image.open(light))["brightness"],
            "the whole point is that a light card can exist at all")

    def test_the_light_band_is_reachable_which_it_was_not_before(self):
        # No card the account has ever published measured "light"; the scrim
        # made it unreachable. This is the regression test for that.
        bg = Image.new("RGB", (896, 1152), (225, 223, 219))
        path = self._compose(bg, "dark")
        self.assertTrue(path)
        self.assertIn("/light/", quality.palette_bucket(Image.open(path)))


class TestRejectedCardsAreCleanedUp(unittest.TestCase):
    """A variety-rejected candidate must not be left on disk.

    compose_card writes its JPEG before returning, so the variety check — which
    can only run on the composed card — always fires after the file exists. Left
    behind, those files would corrupt feed_variety_report.py, which measures the
    output directory to decide what the account has actually been publishing.
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def _card(self, name):
        p = os.path.join(self._tmp.name, name)
        Image.new("RGB", (8, 8), (0, 0, 0)).save(p, "JPEG")
        return p

    def test_discard_removes_every_path(self):
        paths = [self._card("a.jpg"), self._card("b.jpg")]
        verse_card._discard(paths)
        self.assertFalse(any(os.path.exists(p) for p in paths))

    def test_discard_tolerates_a_missing_file(self):
        paths = [self._card("a.jpg"), os.path.join(self._tmp.name, "gone.jpg")]
        verse_card._discard(paths)          # must not raise
        self.assertFalse(os.path.exists(paths[0]))

    def test_discard_of_nothing_is_harmless(self):
        verse_card._discard([])
