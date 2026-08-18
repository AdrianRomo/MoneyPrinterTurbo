"""The aesthetic gate, and attribution that survives being set on a slide.

Every other carousel gate measures pixels. These cover the two things that were
still getting through: photographs with no light in them, and credits that are
free text rather than a name.
"""

import unittest

from PIL import Image

from app.services import quality, wikimedia


def _solid(colour, size=(240, 300)):
    return Image.new("RGB", size, colour)


def _gradient(size=(240, 300)):
    """A frame with real colour in it."""
    img = Image.new("RGB", size)
    px = img.load()
    for y in range(size[1]):
        for x in range(size[0]):
            px[x, y] = (255 * x // size[0], 40, 255 * y // size[1])
    return img


class TestAestheticGate(unittest.TestCase):
    def test_flat_and_bright_is_rejected(self):
        """An overcast whiteout: no colour, plenty of light."""
        ok, why = quality.check_slide_aesthetics(_solid((200, 202, 205)))
        self.assertFalse(ok)
        self.assertIn("washed out", why)

    def test_flat_and_dark_is_kept(self):
        """A night sky is near-monochrome on purpose.

        This is the case a bare colourfulness floor gets wrong, and the reason
        the gate tests the conjunction: the account's best aurora and its
        deep-space cover both score below the floor and are both excellent.
        """
        ok, _ = quality.check_slide_aesthetics(_solid((18, 20, 26)))
        self.assertTrue(ok)

    def test_colourful_is_kept(self):
        ok, _ = quality.check_slide_aesthetics(_gradient())
        self.assertTrue(ok)

    def test_metrics_are_reported(self):
        m = quality.aesthetics(_gradient())
        self.assertGreater(m["colour"], quality.FLAT_COLOUR)
        for key in ("colour", "contrast", "brightness"):
            self.assertIsInstance(m[key], float)


class TestCleanAuthor(unittest.TestCase):
    def test_strips_licence_sermon(self):
        self.assertEqual(
            wikimedia.clean_author(
                "This Photo was taken by Wolfgang Moroder. Feel free to use my "
                "photos, but please mention me as the author"),
            "Wolfgang Moroder")

    def test_keeps_a_plain_name_untouched(self):
        for name in ("Milan Bališin", "Walter Siegmund (talk)", "Basile Morin"):
            self.assertEqual(wikimedia.clean_author(name), name)

    def test_keeps_both_names_on_a_derivative_work(self):
        self.assertEqual(
            wikimedia.clean_author(
                "Original: Anubhav Agarwal Derivative work: UnpetitproleX"),
            "Anubhav Agarwal, UnpetitproleX")

    def test_long_author_list_becomes_et_al(self):
        self.assertEqual(
            wikimedia.clean_author(
                "NASA, ESA, D. Lennon and E. Sabbi (ESA/STScI), J. Anderson, "
                "S. E. de Mink, R. van der Marel (STScI)"),
            "NASA et al.")

    def test_noisy_parenthetical_does_not_cost_the_name(self):
        """The name is in front of the chatter, and CC BY still requires it."""
        self.assertEqual(
            wikimedia.clean_author(
                "Jérémie Silvestro (If you can improve this photo "
                "development, ask me the RAW.)"),
            "Jérémie Silvestro")

    def test_length_alone_never_drops_an_attribution(self):
        """A single over-long name is truncated, never discarded.

        Returning "unknown" for a credited CC BY work would be a licence
        breach, so only recognisable boilerplate may become unknown.
        """
        got = wikimedia.clean_author("Wolfgangus " * 12)
        self.assertTrue(got)
        self.assertLessEqual(len(got), wikimedia.MAX_AUTHOR_CHARS + 2)

    def test_pure_boilerplate_is_unknown(self):
        self.assertEqual(wikimedia.clean_author("Please credit this image"), "")


if __name__ == "__main__":
    unittest.main()
