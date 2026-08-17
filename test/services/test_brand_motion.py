import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.models.schema import VideoParams
from app.services import brand_motion, quote_reel


class MotionTestCase(unittest.TestCase):
    """Base that keeps every test off the production storage tree.

    `app/services/series.py` hardcodes its state path, which is how a full test
    run once advanced the live Reel series counter. The pool is addressed
    through `pool_dir()`, so patching that one function is enough to keep the
    clips, and the misleading confidence of a pool that looks full, inside a
    temp directory.
    """

    def setUp(self):
        self.original_app_config = dict(config.app)
        self._tmp = tempfile.TemporaryDirectory()
        self.pool = self._tmp.name
        patcher = patch.object(brand_motion, "pool_dir", return_value=self.pool)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        self._tmp.cleanup()

    def _place(self, subject, variant=0, size=1024):
        """Put a plausible clip in the pool and return its path."""
        path = brand_motion.clip_path(subject, variant)
        with open(path, "wb") as fh:
            fh.write(b"\0" * size)
        return path


class TestPoolAddressing(MotionTestCase):
    def test_clip_path_is_stable_and_distinct_per_variant(self):
        subject = "calm lake at dawn with gentle reflections"
        self.assertEqual(brand_motion.clip_path(subject, 0),
                         brand_motion.clip_path(subject, 0))
        self.assertNotEqual(brand_motion.clip_path(subject, 0),
                            brand_motion.clip_path(subject, 1))
        self.assertNotEqual(brand_motion.clip_path(subject, 0),
                            brand_motion.clip_path("desert dunes under a soft dawn sky", 0))

    def test_zero_byte_clips_are_not_poolable(self):
        """A half-written clip must never count toward the target."""
        self._place("calm lake at dawn with gentle reflections", 0, size=0)
        self.assertEqual(brand_motion.pool_clips(), [])
        self.assertEqual(brand_motion.pool_status()["clips"], 0)

    def test_planned_slots_fill_breadth_before_depth(self):
        config.app["motion_pool_variants_per_subject"] = 3
        slots = brand_motion._planned_slots()
        # Only the animatable subset — the pool does not generate clips for
        # subjects Wan cannot bring to life.
        subjects = brand_motion.moving_subjects()

        self.assertEqual(len(slots), len(subjects) * 3)
        # The first pass must give every subject one clip before any subject
        # gets a second: a Reel needs three DIFFERENT subjects, so breadth is
        # what makes the pool usable at all, and depth only adds variety.
        first_pass = slots[:len(subjects)]
        self.assertEqual({variant for _, variant in first_pass}, {0})
        self.assertEqual(len({subject for subject, _ in first_pass}), len(subjects))

    def test_pool_status_reports_shortfall(self):
        config.app["motion_pool_target_clips"] = 4
        self._place("calm lake at dawn with gentle reflections", 0)
        self._place("desert dunes under a soft dawn sky", 0)
        status = brand_motion.pool_status()
        self.assertEqual(status["clips"], 2)
        self.assertEqual(status["target"], 4)
        self.assertEqual(status["missing"], 2)


class TestDrawingFromPool(MotionTestCase):
    def test_clip_for_subject_returns_none_when_pool_is_empty(self):
        """A miss must be a miss — never an inline generation on the GPU."""
        self.assertIsNone(
            brand_motion.clip_for_subject("calm lake at dawn with gentle reflections")
        )

    def test_clip_for_subject_walks_past_an_already_used_variant(self):
        subject = "calm lake at dawn with gentle reflections"
        config.app["motion_pool_variants_per_subject"] = 3
        first = self._place(subject, 0)
        second = self._place(subject, 1)

        self.assertEqual(brand_motion.clip_for_subject(subject), first)
        self.assertEqual(
            brand_motion.clip_for_subject(subject, avoid={first}), second
        )
        self.assertIsNone(
            brand_motion.clip_for_subject(subject, avoid={first, second})
        )


class TestDeadline(MotionTestCase):
    def test_until_rolls_to_tomorrow_when_the_time_has_passed(self):
        import time as _time

        # 00:00 has always just passed by the time a 01:00 run starts.
        deadline = brand_motion._deadline_from("00:00")
        self.assertIsNotNone(deadline)
        self.assertGreater(deadline, _time.time())

    def test_unparseable_until_means_no_deadline_rather_than_a_crash(self):
        self.assertIsNone(brand_motion._deadline_from("half past five"))


class TestMotionQualityGate(MotionTestCase):
    """The gate that catches the failure nothing else can see.

    A frozen clip passes every file-level check — duration, resolution, codec —
    and is indistinguishable from a good one in any single frame. Only a
    temporal measurement separates them.
    """

    def test_verdicts_use_real_measurements_from_the_3090(self):
        """Every case here is a clip that was actually generated and watched."""
        cases = [
            # ocean surf: the reference for "clearly moving, publishable"
            ({"drift": 17.06, "delta": 5.17}, "calm"),
            # open fields: it DOES change, but reads as a near-still on a phone.
            # This is the case a delta-only gate wrongly passed.
            ({"drift": 4.82, "delta": 1.31}, "frozen"),
            # pastel hills: static
            ({"drift": 2.90, "delta": 0.76}, "frozen"),
            # LTXV 13B on a plain KSampler: under-denoised, effectively a still
            ({"drift": 3.72, "delta": 0.40}, "frozen"),
            # LTXV 2B on sand: goes somewhere, but shimmers getting there
            ({"drift": 17.75, "delta": 9.56}, "restless"),
        ]
        for motion, expected in cases:
            self.assertEqual(brand_motion._motion_verdict(motion), expected,
                             f"wrong verdict for {motion}")
        self.assertEqual(brand_motion._motion_verdict(None), "unmeasured")
        self.assertEqual(brand_motion._motion_verdict({}), "unmeasured")

    def test_band_edges_are_inclusive_of_calm(self):
        edge = {"drift": brand_motion.MOTION_MIN_DRIFT,
                "delta": brand_motion.MOTION_MAX_DELTA}
        self.assertEqual(brand_motion._motion_verdict(edge), "calm")

    def test_drift_is_the_floor_and_delta_only_the_ceiling(self):
        """A smooth clip with tiny per-frame change is fine IF it travels.

        This is the whole point of the two-metric gate: texture-poor imagery
        produces a low delta even while the camera moves, so delta must never be
        able to reject a clip on its own.
        """
        self.assertEqual(
            brand_motion._motion_verdict({"drift": 12.0, "delta": 0.79}), "calm")

    def test_wan_negative_prompt_keeps_the_models_own_vocabulary(self):
        """Translating Wan's negative prompt weakens it; assert we did not."""
        negative = brand_motion.MOTION_NEGATIVE
        # The three terms that suppress a frozen clip, in the model's language.
        for term in ("静态", "静止不动的画面", "画面"):
            self.assertIn(term, negative)
        # Our own additions still ride along.
        self.assertIn("people", negative)
        self.assertIn("watermark", negative)

    def test_motion_is_chosen_from_what_the_scene_can_actually_do(self):
        cases = {
            "gentle ocean waves on an empty shore at first light": "waves",
            "rain on a window with soft daylight behind": "raindrops",
            "a mountain stream over smooth stones": "flowing",
            "wheat field swaying in warm evening light": "swaying",
            "snowfall over silent evergreen trees": "snowflakes",
            # "sky" appears here too, but sand is the better answer — the
            # generic cloud rule must not steal it.
            "desert dunes under a soft dawn sky": "sand",
        }
        for subject, expected in cases.items():
            self.assertIn(expected, brand_motion.motion_for(subject),
                          f"wrong motion chosen for {subject!r}")

    def test_a_scene_with_nothing_moving_still_gets_motion(self):
        """Hills do not move. The clip must not therefore be frozen."""
        motion = brand_motion.motion_for("an empty wooden bench beside still water")
        self.assertTrue(motion.strip())
        # And the prompt always asks for a camera move, which is the only
        # guaranteed source of motion in a genuinely static scene.
        self.assertIn("camera push", brand_motion.MOTION_HINT)

    def test_unmeasurable_clip_is_accepted_rather_than_discarded(self):
        """If ffmpeg or numpy cannot measure, we must not throw away good work."""
        self.assertEqual(brand_motion._motion_verdict(None), "unmeasured")

    def test_measure_motion_on_a_nonvideo_returns_none(self):
        junk = os.path.join(self.pool, "not-a-video.mp4")
        with open(junk, "wb") as fh:
            fh.write(b"definitely not an mp4")
        self.assertIsNone(brand_motion.measure_motion(junk))

    def test_retry_scratch_files_never_count_as_pooled_clips(self):
        """A `.tryN` candidate must not be mistaken for a finished clip."""
        subject = "calm lake at dawn with gentle reflections"
        dest = brand_motion.clip_path(subject, 0)
        with open(f"{dest}.try2", "wb") as fh:
            fh.write(b"\0" * 1024)
        self.assertEqual(brand_motion.pool_clips(), [])


class TestAutoPublishRequiresMotion(MotionTestCase):
    """Sign-off was given on motion Reels, so only motion Reels auto-publish.

    The renderer falls back to stills when the pool is short, which is right for
    rendering and wrong for publishing: it would silently substitute a format
    that was never approved.
    """

    def _qc(self):
        return {"passed": True, "publishable": True}

    def _asset(self, kind):
        return quote_reel.QuoteReelAsset(
            path=f"/tmp/x.{'mp4' if kind == 'video' else 'jpg'}",
            kind=kind, provider="comfyui_motion", label="x", source_info={},
        )

    def test_all_motion_clips_are_allowed_through_the_gate(self):
        config.app["quote_reel_auto_schedule_enabled"] = True
        config.app["quote_reel_media_source"] = "comfyui_motion"
        result = quote_reel._schedule_if_enabled(
            "/tmp/final.mp4", "caption", "quote", self._qc(),
            assets=[self._asset("video")] * 3,
        )
        # Passes this gate and moves on to the Postiz checks below it.
        self.assertNotIn("fell back to stills", str(result.get("error", "")))

    def test_a_stills_fallback_is_not_auto_published(self):
        config.app["quote_reel_auto_schedule_enabled"] = True
        config.app["quote_reel_media_source"] = "comfyui_motion"
        result = quote_reel._schedule_if_enabled(
            "/tmp/final.mp4", "caption", "quote", self._qc(),
            assets=[self._asset("video"), self._asset("image"), self._asset("image")],
        )
        self.assertFalse(result["success"])
        self.assertTrue(result["skipped"])
        self.assertIn("fell back to stills", result["error"])

    def test_an_empty_asset_list_is_not_auto_published(self):
        config.app["quote_reel_auto_schedule_enabled"] = True
        config.app["quote_reel_media_source"] = "comfyui_motion"
        result = quote_reel._schedule_if_enabled(
            "/tmp/final.mp4", "caption", "quote", self._qc(), assets=[])
        self.assertFalse(result["success"])

    def test_the_guard_does_not_apply_to_other_media_sources(self):
        """Someone running stock or curated footage has approved that format."""
        config.app["quote_reel_auto_schedule_enabled"] = True
        config.app["quote_reel_media_source"] = "curated"
        result = quote_reel._schedule_if_enabled(
            "/tmp/final.mp4", "caption", "quote", self._qc(),
            assets=[self._asset("image")] * 3,
        )
        self.assertNotIn("fell back to stills", str(result.get("error", "")))


class TestQuoteReelMotionSelection(MotionTestCase):
    """The dedupe the Reel actually depends on.

    brand_footage caches by subject, so several near-synonym search terms
    collapse onto one subject and would otherwise hand the same file back three
    times — cutting from a shot straight back to that same shot.
    """

    def _params(self):
        return VideoParams(
            video_subject="hope at first light",
            video_script="A quiet line.",
            content_mode=quote_reel.CONTENT_MODE,
        )

    def test_identical_subjects_do_not_yield_the_same_clip_twice(self):
        with patch.object(quote_reel, "_quote_search_terms",
                          return_value=["hope", "hope", "hope"]):
            with patch("app.services.brand_footage.subject_for",
                       return_value="calm lake at dawn with gentle reflections"):
                self._place("calm lake at dawn with gentle reflections", 0)
                assets = quote_reel._comfyui_motion_assets(self._params(), count=3)

        # One subject, one clip: better a short list the caller can fall back
        # from than three copies of one shot.
        self.assertEqual(len(assets), 1)
        self.assertEqual(len({asset.path for asset in assets}), 1)

    def test_distinct_subjects_yield_distinct_clips(self):
        subjects = [
            "calm lake at dawn with gentle reflections",
            "desert dunes under a soft dawn sky",
            "wheat field swaying in warm evening light",
        ]
        for subject in subjects:
            self._place(subject, 0)

        with patch.object(quote_reel, "_quote_search_terms",
                          return_value=["a", "b", "c"]):
            with patch("app.services.brand_footage.subject_for",
                       side_effect=lambda term, i, avoid=None: subjects[i]):
                assets = quote_reel._comfyui_motion_assets(self._params(), count=3)

        self.assertEqual(len(assets), 3)
        self.assertEqual(len({asset.path for asset in assets}), 3)
        for asset in assets:
            self.assertEqual(asset.kind, "video")
            self.assertEqual(asset.provider, "comfyui_motion")
            # Locally generated from a vetted seed frame, so the release and
            # embedded-text gates have nothing to flag.
            self.assertTrue(asset.source_info["raw_text_free"])
            self.assertFalse(asset.source_info["contains_people"])

    def test_an_empty_pool_selects_nothing_rather_than_generating(self):
        with patch.object(quote_reel, "_quote_search_terms",
                          return_value=["a", "b", "c"]):
            assets = quote_reel._comfyui_motion_assets(self._params(), count=3)
        self.assertEqual(assets, [])


if __name__ == "__main__":
    unittest.main()
