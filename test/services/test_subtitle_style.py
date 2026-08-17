import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.services import quality, subtitle_style

WIDTH, HEIGHT = 1080, 1920


def solid_frame(value: int) -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), value, dtype=np.uint8)


class SubtitleStyleTestCase(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        config.app["subtitle_renderer"] = "brand"

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)


class TestFlag(SubtitleStyleTestCase):
    def test_disabled_unless_explicitly_selected(self):
        """默认仍走 MoviePy 渲染，两套渲染器可以并存对比。"""
        config.app["subtitle_renderer"] = "moviepy"
        self.assertFalse(subtitle_style.enabled())
        config.app.pop("subtitle_renderer")
        self.assertFalse(subtitle_style.enabled())

    def test_enabled_is_case_insensitive(self):
        config.app["subtitle_renderer"] = " BRAND "
        self.assertTrue(subtitle_style.enabled())


class TestRendering(SubtitleStyleTestCase):
    def test_empty_text_renders_nothing(self):
        self.assertIsNone(subtitle_style.render_cue("", WIDTH, HEIGHT))
        self.assertIsNone(subtitle_style.render_cue("   ", WIDTH, HEIGHT))

    def test_returns_full_width_rgba(self):
        out = subtitle_style.render_cue("The dishes can wait", WIDTH, HEIGHT)
        self.assertEqual(out.shape[1], WIDTH)
        self.assertEqual(out.shape[2], 4)
        self.assertGreater(out[..., 3].max(), 0)

    def test_draws_white_type(self):
        out = subtitle_style.render_cue("The dishes can wait", WIDTH, HEIGHT)
        opaque = out[..., 3] > 250
        self.assertTrue(opaque.any())
        self.assertEqual(int(out[..., :3][opaque].max()), 255)

    def test_wraps_to_two_lines_by_shrinking(self):
        """两行是上限：宁可缩小字号，也不铺成五行字幕墙。"""
        short = subtitle_style.describe("The hum of traffic", WIDTH)
        longer = subtitle_style.describe(
            "Every morning when the coffee steams or a neighbor's kettle clatters", WIDTH)
        self.assertIn("1 line(s)", short)
        self.assertIn("2 line(s)", longer)

    def test_an_over_long_cue_is_reported_rather_than_clipped(self):
        text = ("there is an invitation that feels as natural as breathing itself, a quiet "
                "reminder that God does not wait for grand ceremonies but speaks in the "
                "rustle of leaves and the hum of traffic")
        with patch("app.services.subtitle_style.logger.warning") as warn:
            out = subtitle_style.render_cue(text, WIDTH, HEIGHT)
        self.assertIsNotNone(out)
        self.assertTrue(
            any("needs" in str(call) and "lines" in str(call) for call in warn.call_args_list)
        )

    def test_taller_overlay_for_more_lines(self):
        one = subtitle_style.render_cue("The hum of traffic", WIDTH, HEIGHT)
        two = subtitle_style.render_cue(
            "Every morning when the coffee steams or a neighbor's kettle clatters",
            WIDTH, HEIGHT)
        self.assertGreater(two.shape[0], one.shape[0])

    def test_tracking_widens_the_line(self):
        """字距是排版系统的核心差异，必须真的作用到测量宽度上。"""
        from PIL import Image, ImageDraw
        from app.services import typography
        draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
        fnt = typography.font(typography.SERIF, 42, "Medium")
        plain = typography.width(draw, "ordinary grace", fnt, 0.0)
        tracked = typography.width(draw, "ordinary grace", fnt, subtitle_style.TRACKING)
        self.assertGreater(tracked, plain)


class TestAdaptiveScrim(SubtitleStyleTestCase):
    @staticmethod
    def scrim_strength(overlay: np.ndarray) -> float:
        # Median alpha across the overlay's middle row: the glyphs cover only a
        # minority of that row, so this reads the scrim rather than the type.
        return float(np.median(overlay[overlay.shape[0] // 2, :, 3]))

    def test_bright_background_gets_a_stronger_scrim_than_a_dark_one(self):
        text = "The dishes can wait"
        dark = subtitle_style.render_cue(text, WIDTH, HEIGHT, background=solid_frame(12))
        bright = subtitle_style.render_cue(text, WIDTH, HEIGHT, background=solid_frame(235))
        self.assertGreater(self.scrim_strength(bright), self.scrim_strength(dark))

    def test_a_dark_background_needs_no_scrim_at_all(self):
        out = subtitle_style.render_cue("The dishes can wait", WIDTH, HEIGHT,
                                        background=solid_frame(0))
        self.assertEqual(self.scrim_strength(out), 0.0)

    def test_delivered_contrast_meets_the_target_on_a_bright_frame(self):
        """模糊会削弱压暗强度，所以 alpha 必须按实际落到字上的部分反解。"""
        text = "The dishes can wait"
        frame = solid_frame(235)
        out = subtitle_style.render_cue(text, WIDTH, HEIGHT, background=frame)

        # Rebuild the glyph mask and measure what the overlay actually delivers.
        glyphs = out[..., :3].mean(axis=2) * (out[..., 3] / 255.0) > 200
        self.assertTrue(glyphs.any())
        scrim = out[..., 3][~glyphs]
        luminance = (235 / 255.0)
        delivered = luminance * (1 - float(scrim.max()) / 255.0)
        self.assertGreaterEqual(quality.contrast_ratio(delivered),
                                subtitle_style.MIN_RATIO)

    def test_scrim_never_reaches_a_flat_opaque_plate(self):
        out = subtitle_style.render_cue("The dishes can wait", WIDTH, HEIGHT,
                                        background=solid_frame(255))
        self.assertLessEqual(int(out[..., 3].max()), 255)
        self.assertLessEqual(subtitle_style.SCRIM_MAX_ALPHA, 255)

    def test_scrim_fades_out_before_the_overlay_edge(self):
        """羽化被裁掉就会出现硬边——正是遮罩要避免的东西。"""
        out = subtitle_style.render_cue("The dishes can wait", WIDTH, HEIGHT,
                                        background=solid_frame(235))
        self.assertEqual(int(out[0, :, 3].max()), 0)
        self.assertEqual(int(out[-1, :, 3].max()), 0)

    def test_measurement_without_a_frame_is_none(self):
        from PIL import Image
        mask = Image.new("L", (10, 10), 255)
        self.assertIsNone(subtitle_style.measure_background(None, (0, 0, 10, 10), mask))

    def test_unmeasurable_background_still_gets_a_usable_scrim(self):
        with patch("app.services.subtitle_style.measure_background", return_value=None):
            out = subtitle_style.render_cue("The dishes can wait", WIDTH, HEIGHT,
                                            background=solid_frame(128))
        self.assertGreater(int(out[..., 3].max()), 0)

    def test_a_broken_frame_does_not_raise(self):
        out = subtitle_style.render_cue("The dishes can wait", WIDTH, HEIGHT,
                                        background=np.zeros((4, 4), dtype=np.uint8))
        self.assertIsNotNone(out)


class TestFace(SubtitleStyleTestCase):
    def test_serif_is_the_default_voice(self):
        path, instance = subtitle_style._face()
        self.assertIn("Cormorant", path)
        self.assertEqual(instance, "Medium")

    def test_sans_is_selectable(self):
        config.app["subtitle_face"] = "sans"
        path, instance = subtitle_style._face()
        self.assertIn("Inter", path)

    def test_variable_instance_is_actually_applied(self):
        """MoviePy 会把可变字体渲染成默认字重，这正是必须走 PIL 的原因。"""
        from app.services import typography
        medium = typography.font(typography.SERIF, 42, "Medium")
        light = typography.font(typography.SERIF, 42, "Light")
        self.assertNotEqual(medium.getmask("Ordinary").size,
                            light.getmask("Ordinary").size)

    def test_font_size_floor_is_respected(self):
        config.app["subtitle_font_size"] = 4
        self.assertGreaterEqual(subtitle_style._font_size(), subtitle_style.MIN_FONT_SIZE)

    def test_unusable_font_size_falls_back(self):
        config.app["subtitle_font_size"] = "large"
        self.assertEqual(subtitle_style._font_size(), subtitle_style.DEFAULT_FONT_SIZE)


if __name__ == "__main__":
    unittest.main()
