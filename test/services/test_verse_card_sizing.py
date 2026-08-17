"""Verse card / story sizing — the cards must never be enlarged to fill the frame.

Backgrounds are generated at an SDXL bucket, which is smaller than the output
canvas, and `_cover` was enlarging them: 1.21x for feed posts and 1.43x for
stories. Generating at the canvas outright is not the fix — off-bucket sizes are
what produce duplicated subjects, which is why the buckets exist — so a second
low-denoise pass re-samples the composition at output size instead.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import verse_card as vc


class TestHiresSize(unittest.TestCase):
    def test_no_kind_is_ever_upscaled(self):
        for kind, canvas in vc.ASPECTS.items():
            bucket = vc.SDXL_BUCKET[kind]
            hires = vc.hires_size(bucket, canvas)
            scale = max(canvas[0] / hires[0], canvas[1] / hires[1])
            self.assertLessEqual(scale, 1.0, f"{kind} would upscale {scale:.2f}x")

    def test_bucket_aspect_is_preserved(self):
        # Upscaling straight to the canvas ratio would stretch the image: a 4:5
        # canvas and a 896x1152 bucket are not the same shape.
        for kind in vc.ASPECTS:
            bucket = vc.SDXL_BUCKET[kind]
            hires = vc.hires_size(bucket, vc.ASPECTS[kind])
            self.assertAlmostEqual(hires[0] / hires[1], bucket[0] / bucket[1], places=2,
                                   msg=f"{kind} aspect drifted")

    def test_dimensions_are_multiples_of_eight(self):
        for kind in vc.ASPECTS:
            hires = vc.hires_size(vc.SDXL_BUCKET[kind], vc.ASPECTS[kind])
            self.assertEqual((hires[0] % 8, hires[1] % 8), (0, 0), f"{kind} not /8")

    def test_a_bucket_that_already_covers_is_left_alone(self):
        self.assertEqual(vc.hires_size((2000, 2500), (1080, 1350)), (2000, 2500))

    def test_degenerate_bucket_does_not_raise(self):
        self.assertEqual(vc.hires_size((0, 0), (1080, 1350)), (1080, 1350))


class TestWorkflow(unittest.TestCase):
    def _flow(self, hires=None):
        return vc._workflow("a lake", 896, 1152, seed=7, ckpt="x.safetensors", hires=hires)

    def test_without_hires_the_workflow_is_a_single_pass(self):
        flow = self._flow()
        self.assertEqual(flow["8"]["inputs"]["samples"], ["3", 0])
        self.assertNotIn("11", flow)

    def test_hires_pass_is_wired_between_sampler_and_decode(self):
        flow = self._flow((1440, 1848))
        self.assertEqual(flow["10"]["inputs"]["samples"], ["3", 0])      # base latent
        self.assertEqual(flow["11"]["inputs"]["latent_image"], ["10", 0])  # upscaled
        self.assertEqual(flow["8"]["inputs"]["samples"], ["11", 0])      # decode 2nd pass
        self.assertEqual((flow["10"]["inputs"]["width"], flow["10"]["inputs"]["height"]),
                         (1440, 1848))

    def test_second_pass_keeps_the_negative_prompt(self):
        # The whole guarantee about what is NOT in frame lives in the negative
        # prompt; a second pass that dropped it could reintroduce figures.
        flow = self._flow((1440, 1848))
        self.assertEqual(flow["11"]["inputs"]["negative"], flow["3"]["inputs"]["negative"])
        self.assertEqual(flow["11"]["inputs"]["positive"], flow["3"]["inputs"]["positive"])

    def test_second_pass_denoise_preserves_composition(self):
        # High enough to resolve detail, low enough not to redraw the scene.
        denoise = self._flow((1440, 1848))["11"]["inputs"]["denoise"]
        self.assertGreater(denoise, 0.2)
        self.assertLess(denoise, 0.6)

    def test_hires_equal_to_bucket_adds_no_second_pass(self):
        flow = self._flow((896, 1152))
        self.assertNotIn("11", flow)
        self.assertEqual(flow["8"]["inputs"]["samples"], ["3", 0])


class TestCanvas(unittest.TestCase):
    def test_feed_card_matches_the_carousel_frame(self):
        from app.services import carousel
        self.assertEqual(vc.ASPECTS["post"], (carousel.WIDTH, carousel.HEIGHT))

    def test_story_stays_at_the_story_surface(self):
        # 1080x1920 IS the story surface; there is nothing above it to serve.
        self.assertEqual(vc.ASPECTS["story"], (1080, 1920))


if __name__ == "__main__":
    unittest.main()
