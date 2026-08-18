import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.services import bgm as bgm_service
from app.services import brand_music, quote_reel, video

# Six prompts is what the module actually ships; the tests pin their own so a
# pack edit cannot quietly change what the arithmetic here is asserting.
PROMPTS = ["calm pad", "felt piano", "low strings"]


class MusicTestCase(unittest.TestCase):
    """Base that keeps every test off the production pool.

    The pool holds paid generations. A test that wrote into the live directory
    would either burn credit or, worse, leave a plausible-looking file that a
    real Reel then published — so `pool_dir` is redirected before anything else
    can resolve it.
    """

    def setUp(self):
        self.original_app_config = dict(config.app)
        self._tmp = tempfile.TemporaryDirectory()
        self.pool = self._tmp.name
        for target in ("pool_dir",):
            patcher = patch.object(brand_music, target, return_value=self.pool)
            patcher.start()
            self.addCleanup(patcher.stop)
        prompts = patch.object(brand_music, "prompts", return_value=list(PROMPTS))
        prompts.start()
        self.addCleanup(prompts.stop)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        self._tmp.cleanup()

    def _place(self, prompt, variant=0, size=None):
        """Put a plausible track in the pool and return its path."""
        path = brand_music._track_path(prompt, variant)
        with open(path, "wb") as fh:
            fh.write(b"\0" * (brand_music.MIN_TRACK_BYTES if size is None else size))
        return path


class TestPoolAddressing(MusicTestCase):
    def test_track_path_is_stable_and_distinct_per_variant(self):
        self.assertEqual(brand_music._track_path("calm pad", 0),
                         brand_music._track_path("calm pad", 0))
        self.assertNotEqual(brand_music._track_path("calm pad", 0),
                            brand_music._track_path("calm pad", 1))
        self.assertNotEqual(brand_music._track_path("calm pad", 0),
                            brand_music._track_path("felt piano", 0))

    def test_target_is_capped_at_the_reachable_slot_count(self):
        """A target above (prompts x variants) is a shortfall that never clears.

        The motion pool hit this first: a job reporting `missing` forever is a
        job the operator learns to ignore, and then a real shortfall goes unread
        too.
        """
        config.app["music_pool_variants_per_prompt"] = 2
        config.app["music_pool_target"] = 500
        self.assertEqual(brand_music.target(), len(PROMPTS) * 2)

    def test_target_survives_a_non_numeric_config_value(self):
        config.app["music_pool_target"] = "twelve"
        self.assertEqual(brand_music.target(),
                         min(brand_music.DEFAULT_TARGET, len(PROMPTS) * 2))

    def test_status_reports_the_shortfall(self):
        config.app["music_pool_target"] = 3
        config.app["music_pool_variants_per_prompt"] = 1
        self._place("calm pad")
        status = brand_music.status()
        self.assertEqual((status["tracks"], status["target"], status["missing"]),
                         (1, 3, 2))
        self.assertFalse(status["bundled_songs_allowed"])


class TestSelection(MusicTestCase):
    def test_an_empty_pool_selects_silence(self):
        self.assertEqual(brand_music.select_track(), "")

    def test_consecutive_draws_do_not_repeat_a_track(self):
        """Variety across consecutive Reels is the whole point of the pool.

        Plain random.choice over twelve tracks repeats often enough to be
        noticeable on an account posting six times a day.
        """
        for i, prompt in enumerate(PROMPTS):
            self._place(prompt, 0)
        picks = [os.path.basename(brand_music.select_track())
                 for _ in range(len(PROMPTS))]
        self.assertEqual(len(set(picks)), len(PROMPTS))

    def test_selection_recycles_once_every_track_is_recent(self):
        """Exhausting the pool must return a track, never silence."""
        self._place("calm pad", 0)
        first = brand_music.select_track()
        self.assertTrue(first)
        self.assertEqual(brand_music.select_track(), first)

    def test_a_corrupt_recent_file_does_not_break_selection(self):
        self._place("calm pad", 0)
        with open(os.path.join(self.pool, "recent.json"), "w") as fh:
            fh.write("{not json")
        self.assertTrue(brand_music.select_track())

    def test_recent_history_is_bounded(self):
        for variant in range(3):
            for prompt in PROMPTS:
                self._place(prompt, variant)
        for _ in range(20):
            brand_music.select_track()
        with open(os.path.join(self.pool, "recent.json")) as fh:
            self.assertLessEqual(len(json.load(fh)), 8)


class TestTheFallbackIsSilenceNotTheBundledSongs(MusicTestCase):
    """The reason this module exists, asserted directly.

    `resource/songs/` ships 29 tracks with no licence file. Every earlier Reel
    carried one. Instagram fingerprints audio, so the failure mode is a post
    muted or region-blocked long after publication — which is why an empty pool
    must render silent rather than reach for them.
    """

    def test_random_resolves_to_the_licensed_pool(self):
        placed = self._place("calm pad", 0)
        self.assertEqual(video.get_bgm_file(bgm_type="random"), placed)

    def test_an_empty_pool_never_reaches_the_bundled_songs(self):
        def boom():
            raise AssertionError("fell through to the unlicensed bundled songs")

        with patch.object(bgm_service, "list_bgm_files", side_effect=boom):
            self.assertEqual(video.get_bgm_file(bgm_type="random"), "")

    def test_the_bundled_songs_can_be_restored_deliberately(self):
        """The escape hatch has to work, or someone edits the guard out."""
        config.app["music_allow_bundled_songs"] = True
        with patch.object(bgm_service, "list_bgm_files",
                          return_value=["/songs/one.mp3"]):
            self.assertEqual(video.get_bgm_file(bgm_type="random"),
                             "/songs/one.mp3")

    def test_the_opt_in_accepts_the_string_forms_config_actually_holds(self):
        """config.toml round-trips booleans as strings often enough to matter."""
        for truthy in ("true", "True", "1", "yes", "on"):
            config.app["music_allow_bundled_songs"] = truthy
            with patch.object(bgm_service, "list_bgm_files",
                              return_value=["/songs/one.mp3"]):
                self.assertEqual(video.get_bgm_file(bgm_type="random"),
                                 "/songs/one.mp3", f"{truthy!r} should opt in")
        for falsy in ("false", "0", "no", "off", ""):
            config.app["music_allow_bundled_songs"] = falsy
            with patch.object(bgm_service, "list_bgm_files",
                              side_effect=AssertionError("opted in on " + repr(falsy))):
                self.assertEqual(video.get_bgm_file(bgm_type="random"), "")

    def test_an_explicit_operator_file_still_wins(self):
        """Choosing a specific bed by hand is a deliberate act; honour it.

        This went through `get_bgm_file(bgm_type="")`, which returns "" before
        it ever looks at bgm_file — so a configured `quote_reel_music_file` was
        silently dropped and the Reel rendered silent. Harmless while nobody
        had set the key, and indistinguishable from the empty-pool fallback the
        moment somebody did.
        """
        config.app["quote_reel_music_enabled"] = True
        config.app["quote_reel_music_file"] = "chosen.mp3"
        chosen = self._place("operator's own bed", 0)
        # The pool must not even be consulted: the operator named a file.
        with patch.object(bgm_service, "resolve_bgm_file", return_value=chosen) as resolve, \
             patch.object(brand_music, "select_track",
                          side_effect=AssertionError("reached for the pool instead")):
            self.assertEqual(
                video.get_bgm_file(bgm_type="random", bgm_file="chosen.mp3"), chosen)
            # ...and through the path the Reel actually uses. The clip itself is
            # stubbed: what is under test is which file gets opened, not moviepy.
            with patch.object(quote_reel, "AudioFileClip") as clip:
                self.assertIsNotNone(quote_reel._music_bed(4.0))
        clip.assert_called_once_with(chosen)
        resolve.assert_called_with("chosen.mp3")


class TestTheLoudnessGate(MusicTestCase):
    """A dead track passes every structural check.

    Three of the first generations came back as valid 32.04s 192kbps stereo
    mp3s of identical size, and one of them measured peak -30.8 dB — inaudible,
    then multiplied by 0.22 on top. Duration, codec and byte count cannot tell
    the two apart, so loudness has to.
    """

    def _generation(self, *peaks):
        """Fake generations returning the given peak dB in order."""
        measurements = iter(peaks)

        def _generate_once(prompt, out_path, seconds):
            with open(out_path, "wb") as fh:
                fh.write(b"\0" * brand_music.MIN_TRACK_BYTES)
            return True

        def measure(path):
            return {"peak_db": next(measurements), "mean_db": -25.0}

        return _generate_once, measure

    def test_a_silent_generation_is_rerolled_and_the_retry_kept(self):
        gen, measure = self._generation(-40.0, -3.0)
        path = os.path.join(self.pool, "t.mp3")
        with patch.object(brand_music, "_generate_once", side_effect=gen), \
             patch.object(brand_music, "measure", side_effect=measure), \
             patch.object(brand_music, "_normalise", return_value=False):
            self.assertTrue(brand_music.generate_track("calm pad", path))
        self.assertTrue(os.path.exists(path))

    def test_giving_up_leaves_no_silent_track_behind(self):
        """The worst outcome is a dead bed sitting in the pool looking fine."""
        gen, measure = self._generation(-40.0, -45.0)
        path = os.path.join(self.pool, "t.mp3")
        with patch.object(brand_music, "_generate_once", side_effect=gen), \
             patch.object(brand_music, "measure", side_effect=measure):
            self.assertFalse(brand_music.generate_track("calm pad", path))
        self.assertFalse(os.path.exists(path))

    def test_an_unmeasurable_track_is_kept_rather_than_discarded(self):
        """Losing a paid generation to a broken ffmpeg would be worse."""
        gen, _ = self._generation()
        path = os.path.join(self.pool, "t.mp3")
        with patch.object(brand_music, "_generate_once", side_effect=gen), \
             patch.object(brand_music, "measure", return_value={}), \
             patch.object(brand_music, "_normalise", return_value=False):
            self.assertTrue(brand_music.generate_track("calm pad", path))

    def test_a_failed_normalisation_still_yields_a_usable_track(self):
        """An un-normalised but audible bed is still a licensed bed."""
        gen, measure = self._generation(-3.0)
        path = os.path.join(self.pool, "t.mp3")
        with patch.object(brand_music, "_generate_once", side_effect=gen), \
             patch.object(brand_music, "measure", side_effect=measure), \
             patch.object(brand_music, "_normalise", return_value=False):
            self.assertTrue(brand_music.generate_track("calm pad", path))
        self.assertTrue(os.path.exists(path))

    def test_the_peak_floor_sits_above_the_measured_dead_track(self):
        """Guards the constant against being loosened past the real failure."""
        self.assertGreater(brand_music.MIN_SOURCE_PEAK_DB, -30.8)
        # ...and below the quietest generation that was actually usable.
        self.assertLess(brand_music.MIN_SOURCE_PEAK_DB, -10.7)


class TestAuditAndPrune(MusicTestCase):
    def _measured(self, **peaks):
        def measure(path):
            return {"peak_db": peaks[os.path.basename(path)], "mean_db": -25.0}
        return measure

    def test_audit_names_the_dead_tracks(self):
        good = os.path.basename(self._place("calm pad", 0))
        dead = os.path.basename(self._place("felt piano", 0))
        with patch.object(brand_music, "measure",
                          side_effect=self._measured(**{good: -4.0, dead: -35.0})):
            report = brand_music.audit()
        self.assertEqual(report["dead"], [dead])
        self.assertEqual(len(report["tracks"]), 2)

    def test_prune_is_a_dry_run_without_apply(self):
        dead = self._place("felt piano", 0)
        with patch.object(brand_music, "measure",
                          side_effect=self._measured(**{os.path.basename(dead): -35.0})):
            result = brand_music.prune()
        self.assertEqual(result["rejected"], [os.path.basename(dead)])
        self.assertFalse(result["applied"])
        self.assertTrue(os.path.exists(dead))

    def test_prune_moves_rather_than_deletes(self):
        """Each track cost a paid generation; a too-strict threshold must be
        recoverable rather than having destroyed the evidence."""
        dead = self._place("felt piano", 0)
        with patch.object(brand_music, "measure",
                          side_effect=self._measured(**{os.path.basename(dead): -35.0})):
            result = brand_music.prune(apply=True)
        self.assertFalse(os.path.exists(dead))
        self.assertTrue(os.path.exists(
            os.path.join(result["dir"], os.path.basename(dead))))

    def test_rejected_tracks_no_longer_count_as_pooled(self):
        """A moved track must leave the pool, or the target never re-fills."""
        dead = self._place("felt piano", 0)
        with patch.object(brand_music, "measure",
                          side_effect=self._measured(**{os.path.basename(dead): -35.0})):
            brand_music.prune(apply=True)
        self.assertEqual(brand_music.tracks(), [])


class TestTopUp(MusicTestCase):
    def test_a_full_pool_is_a_no_op(self):
        """The timer fires repeatedly; a full pool must cost nothing."""
        config.app["music_pool_variants_per_prompt"] = 1
        config.app["music_pool_target"] = len(PROMPTS)
        for prompt in PROMPTS:
            self._place(prompt, 0)
        with patch.object(brand_music, "generate_track",
                          side_effect=AssertionError("regenerated a full pool")):
            result = brand_music.top_up()
        self.assertEqual(result["made"], 0)

    def test_fill_order_is_breadth_before_depth(self):
        """After one pass every prompt has a track, which is what variety across
        consecutive Reels needs. Depth only adds more of the same mood."""
        config.app["music_pool_variants_per_prompt"] = 2
        config.app["music_pool_target"] = len(PROMPTS) * 2
        seen = []

        def generate(prompt, path, seconds=brand_music.TRACK_SECONDS):
            seen.append(prompt)
            with open(path, "wb") as fh:
                fh.write(b"\0" * brand_music.MIN_TRACK_BYTES)
            return True

        with patch.object(brand_music, "generate_track", side_effect=generate):
            brand_music.top_up()
        self.assertEqual(seen[:len(PROMPTS)], PROMPTS)
        self.assertEqual(len(seen), len(PROMPTS) * 2)

    def test_limit_stops_the_run_early(self):
        config.app["music_pool_variants_per_prompt"] = 2
        config.app["music_pool_target"] = len(PROMPTS) * 2

        def generate(prompt, path, seconds=brand_music.TRACK_SECONDS):
            with open(path, "wb") as fh:
                fh.write(b"\0" * brand_music.MIN_TRACK_BYTES)
            return True

        with patch.object(brand_music, "generate_track", side_effect=generate):
            result = brand_music.top_up(limit=2)
        self.assertEqual(result["made"], 2)
        self.assertEqual(result["reason"], "limit reached")

    def test_a_failing_prompt_does_not_stop_the_others(self):
        config.app["music_pool_variants_per_prompt"] = 1
        config.app["music_pool_target"] = len(PROMPTS)

        def generate(prompt, path, seconds=brand_music.TRACK_SECONDS):
            if prompt == PROMPTS[0]:
                return False
            with open(path, "wb") as fh:
                fh.write(b"\0" * brand_music.MIN_TRACK_BYTES)
            return True

        with patch.object(brand_music, "generate_track", side_effect=generate):
            result = brand_music.top_up()
        self.assertEqual(result["made"], len(PROMPTS) - 1)
        self.assertEqual(result["failed"], 1)


class TestTheReelBed(MusicTestCase):
    def test_music_can_be_turned_off_entirely(self):
        config.app["quote_reel_music_enabled"] = False
        with patch.object(brand_music, "select_track",
                          side_effect=AssertionError("selected while disabled")):
            self.assertIsNone(quote_reel._music_bed(12.0))

    def test_an_empty_pool_renders_a_silent_reel_rather_than_failing(self):
        config.app["quote_reel_music_enabled"] = True
        with patch.object(bgm_service, "list_bgm_files", return_value=[]):
            self.assertIsNone(quote_reel._music_bed(12.0))

    def test_a_selection_failure_is_survivable(self):
        """Music is a nice-to-have; it must never cost the post."""
        config.app["quote_reel_music_enabled"] = True
        with patch.object(video, "get_bgm_file", side_effect=RuntimeError("boom")):
            self.assertIsNone(quote_reel._music_bed(12.0))


if __name__ == "__main__":
    unittest.main()
