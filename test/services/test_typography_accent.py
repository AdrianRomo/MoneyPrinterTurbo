"""Mixed-weight line setting, shared by the quote overlay and the captions.

Both reel types set one clause of a line in a heavier weight; this is the layer
that lets them do it the same way, so the two read as one account.
"""

import sys
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import typography


def _fonts(size: int = 42) -> dict:
    return {
        False: typography.font(typography.SERIF, size, "Medium"),
        True: typography.font(typography.SERIF, size, "Bold"),
    }


class TestAccentParsing(unittest.TestCase):
    def test_marked_clause_becomes_its_own_run(self):
        self.assertEqual(
            typography.parse_accent("a window *onto the eternal*."),
            [("a window ", False), ("onto the eternal", True), (".", False)],
        )

    def test_unmarked_text_is_a_single_plain_run(self):
        self.assertEqual(
            typography.parse_accent("no emphasis here"), [("no emphasis here", False)]
        )

    def test_empty_text_still_yields_one_run(self):
        self.assertEqual(typography.parse_accent(""), [("", False)])

    def test_strip_accent_leaves_plain_prose(self):
        self.assertEqual(
            typography.strip_accent("a window *onto the eternal*."),
            "a window onto the eternal.",
        )

    def test_strip_accent_is_what_tts_and_captions_should_receive(self):
        self.assertNotIn("*", typography.strip_accent("hold it *gently* now"))


class TestTokens(unittest.TestCase):
    def test_punctuation_stays_attached_to_the_accented_word(self):
        # Splitting per run tore "*eternal*." into "eternal" and ".", which the
        # renderer then drew a full word-space apart.
        tokens = typography.tokens(typography.parse_accent("a window *onto the eternal*."))
        self.assertEqual(tokens[-1], ("eternal.", True))
        self.assertNotIn(".", [word for word, _ in tokens])

    def test_a_word_takes_the_weight_most_of_its_letters_carry(self):
        tokens = typography.tokens(typography.parse_accent("*gent*le"))
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0][0], "gentle")


class TestWrapping(unittest.TestCase):
    def setUp(self):
        self.draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
        self.fonts = _fonts()

    def test_wraps_on_pixel_width(self):
        tokens = typography.tokens(typography.parse_accent("one two three four five six"))
        lines = typography.wrap_tokens(self.draw, tokens, self.fonts, 120.0, 0.015)
        self.assertGreater(len(lines), 1)

    def test_word_cap_breaks_a_line_that_would_still_fit(self):
        tokens = typography.tokens(
            typography.parse_accent("one two three four five six seven eight")
        )
        lines = typography.wrap_tokens(
            self.draw, tokens, self.fonts, 10_000.0, 0.015, max_words=3
        )
        self.assertEqual([len(line) for line in lines], [3, 3, 2])

    def test_no_word_cap_keeps_one_line_when_it_fits(self):
        tokens = typography.tokens(typography.parse_accent("one two three four"))
        lines = typography.wrap_tokens(self.draw, tokens, self.fonts, 10_000.0, 0.015)
        self.assertEqual(len(lines), 1)

    def test_line_width_counts_both_weights_and_the_spaces_between(self):
        line = [("one", False), ("two", True)]
        combined = typography.line_width(self.draw, line, self.fonts, 0.015)
        parts = sum(
            typography.width(self.draw, word, self.fonts[bold], 0.015)
            for word, bold in line
        )
        self.assertGreater(combined, parts)

    def test_empty_line_has_no_width(self):
        self.assertEqual(typography.line_width(self.draw, [], self.fonts), 0.0)


class TestDrawing(unittest.TestCase):
    def test_accented_line_puts_ink_on_the_canvas(self):
        canvas = Image.new("L", (400, 80), 0)
        draw = ImageDraw.Draw(canvas)
        fonts = _fonts(36)
        line = typography.tokens(typography.parse_accent("hold it *gently*"))
        typography.draw_line_centered(draw, 400, 10, line, fonts, 255, 0.015)
        self.assertGreater(sum(canvas.getdata()), 0)


if __name__ == "__main__":
    unittest.main()
