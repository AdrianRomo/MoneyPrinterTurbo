"""Carousel image sizing — the rules that keep slides sharp.

Carousels published soft for weeks because every check in the pipeline judged
the *file* while the slide is built from the *crop*. These tests pin the
distinction down at each of the three places it is now enforced: the search
filter, the download, and the quality gate.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import carousel, quality, wikimedia


def _response(payload):
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status = MagicMock()
    return response


class TestCropGeometry(unittest.TestCase):
    def test_landscape_loses_its_width_to_the_crop(self):
        # The bug in one assertion: a 1920px-wide 16:9 thumbnail looks generous
        # and leaves 864 usable pixels for a 1440-wide frame.
        self.assertEqual(wikimedia.crop_dimensions(1920, 1080), (864, 1080))

    def test_portrait_taller_than_the_frame_keeps_its_width(self):
        self.assertEqual(wikimedia.crop_dimensions(1920, 2560), (1920, 2400))

    def test_exactly_four_five_is_unchanged(self):
        self.assertEqual(wikimedia.crop_dimensions(1440, 1800), (1440, 1800))

    def test_degenerate_sizes_do_not_raise(self):
        self.assertEqual(wikimedia.crop_dimensions(0, 0), (0, 0))


class TestFetchWidth(unittest.TestCase):
    def test_landscape_is_sized_by_height(self):
        # 16:9: filling 1800px of height needs 3200px of width, not 1440.
        want = wikimedia.fetch_width_for(1920, 1080)
        self.assertGreater(want, 3000)
        crop_w, crop_h = wikimedia.crop_dimensions(want, int(want * 1080 / 1920))
        self.assertGreaterEqual(crop_w, wikimedia.TARGET_W)
        self.assertGreaterEqual(crop_h, wikimedia.TARGET_H)

    def test_portrait_only_needs_the_frame_width(self):
        self.assertEqual(wikimedia.fetch_width_for(1200, 2000), wikimedia.TARGET_W)

    def test_result_always_covers_the_frame(self):
        for w, h in [(3000, 2000), (4000, 2250), (2000, 3000), (1500, 1500)]:
            want = wikimedia.fetch_width_for(w, h)
            crop_w, crop_h = wikimedia.crop_dimensions(want, int(want * h / w))
            self.assertGreaterEqual(crop_w, wikimedia.TARGET_W, f"{w}x{h}")
            self.assertGreaterEqual(crop_h, wikimedia.TARGET_H, f"{w}x{h}")


def _search_payload(width, height, title="File:Test peak, Norway.jpg",
                    description="A mountain peak", categories=None):
    return {
        "query": {
            "pages": {
                "1": {
                    "title": title,
                    "categories": [{"title": c} for c in (categories or [])],
                    "imageinfo": [{
                        "width": width, "height": height,
                        "thumburl": "https://upload.example.test/1600px-x.jpg",
                        "url": "https://upload.example.test/x.jpg",
                        "descriptionurl": "https://commons.example.test/x",
                        "extmetadata": {
                            "LicenseShortName": {"value": "CC BY-SA 4.0"},
                            "Artist": {"value": "A Photographer"},
                            "ObjectName": {"value": "Test peak, Norway"},
                            "ImageDescription": {"value": description},
                        },
                    }],
                }
            }
        }
    }


class TestSearchFilter(unittest.TestCase):
    @patch("app.services.wikimedia.requests.get")
    def test_panorama_is_rejected(self, mock_get):
        # 12500x2332 is a huge file and a 286px-wide slide.
        mock_get.return_value = _response(_search_payload(12500, 2332))
        self.assertEqual(wikimedia.search("mountains", limit=4), [])

    @patch("app.services.wikimedia.requests.get")
    def test_large_file_with_a_thin_crop_is_rejected(self, mock_get):
        # Comfortably past the old MIN_WIDTH/MIN_HEIGHT, still unusable: the
        # 4:5 crop of a 3000x1200 frame is 960px wide.
        mock_get.return_value = _response(_search_payload(3000, 1200))
        self.assertEqual(wikimedia.search("mountains", limit=4), [])

    @patch("app.services.wikimedia.requests.get")
    def test_ordinary_landscape_is_accepted(self, mock_get):
        mock_get.return_value = _response(_search_payload(4000, 2667))
        found = wikimedia.search("mountains", limit=4)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].location, "norway")


class TestRelevanceGuard(unittest.TestCase):
    """The OR queries that refilled the pool also widened its tail: an OR clause
    only has to match one term to rank, so results must be checked back against
    what the file says about itself."""

    def test_terms_ignore_the_or_operator_and_short_words(self):
        pat = wikimedia._term_pattern("(leopard OR tiger OR panthera)")
        self.assertTrue(pat.search("Panthera pardus in Kruger"))
        self.assertTrue(pat.search("A tiger at dusk"))
        self.assertFalse(pat.search("Australian fur seal"))

    def test_match_starts_at_a_word_boundary(self):
        pat = wikimedia._term_pattern("(milky OR starry)")
        self.assertTrue(pat.search("The Milky Way over the Alps"))
        self.assertTrue(pat.search("milkyway panorama"))     # prefix, so plurals work
        self.assertFalse(pat.search("an airborne mission"))  # the old substring bug

    @patch("app.services.wikimedia.requests.get")
    def test_unrelated_result_is_dropped(self, mock_get):
        mock_get.return_value = _response(_search_payload(
            4000, 2667, title="File:Ribblehead Viaduct.jpg",
            description="A railway bridge in Yorkshire"))
        self.assertEqual(wikimedia.search("(penguin OR penguins)", limit=4), [])

    @patch("app.services.wikimedia.requests.get")
    def test_description_match_is_enough(self, mock_get):
        # Commons searches prose too, so the guard must read it — otherwise it
        # drops results the search legitimately found.
        mock_get.return_value = _response(_search_payload(
            4000, 2667, title="File:Chrysaora hysoscella.jpg",
            description="A jellyfish drifting in open water"))
        self.assertEqual(len(wikimedia.search("(jellyfish OR medusa)", limit=4)), 1)

    @patch("app.services.wikimedia.requests.get")
    def test_category_match_is_enough(self, mock_get):
        mock_get.return_value = _response(_search_payload(
            4000, 2667, title="File:PIA12345.jpg", description="Deep field",
            categories=["Category:Spiral galaxies"]))
        self.assertEqual(len(wikimedia.search("(galaxy OR galaxies)", limit=4)), 1)

    def test_plural_query_term_matches_singular_prose(self):
        pat = wikimedia._term_pattern("(mountains OR peaks)")
        self.assertTrue(pat.search("A mountain peak in Norway"))

    def test_irregular_plurals_are_not_guessed(self):
        # "galaxy" does not match "galaxies"; the query must spell both out.
        self.assertFalse(wikimedia._term_pattern("galaxy").search("Spiral galaxies"))
        self.assertTrue(
            wikimedia._term_pattern("(galaxy OR galaxies)").search("Spiral galaxies"))

    @patch("app.services.wikimedia.requests.get")
    def test_artists_impressions_are_rejected(self, mock_get):
        # The format promises real photographs; the astronomy pools are full of
        # renderings.
        mock_get.return_value = _response(_search_payload(
            4000, 2667, title="File:Artist's impression of an exoplanet.jpg",
            description="planet illustration"))
        self.assertEqual(wikimedia.search("(saturn OR planet)", limit=4), [])

    @patch("app.services.wikimedia.requests.get")
    def test_prose_mentioning_a_map_is_not_disqualified(self, mock_get):
        # NOT_PHOTOGRAPHS reads title and categories only, never free prose.
        mock_get.return_value = _response(_search_payload(
            4000, 2667, title="File:Mount Cook.jpg",
            description="A peak; see the map of the area for the route"))
        self.assertEqual(len(wikimedia.search("(peak OR mountain)", limit=4)), 1)


class TestDownloadSizing(unittest.TestCase):
    @patch("app.services.wikimedia.Image.open")
    @patch("app.services.wikimedia.requests.get")
    def test_landscape_is_re_asked_at_a_bigger_rendition(self, mock_get, mock_open):
        photo = wikimedia.Photo(
            title="File:Peak.jpg", url="https://upload.example.test/1600px-Peak.jpg",
            width=9429, height=5304, author="A", licence="CC BY-SA 4.0",
            location="norway", descriptionurl="",
        )
        big = "https://upload.example.test/3840px-Peak.jpg"
        mock_get.side_effect = [
            _response({"query": {"pages": {"1": {"imageinfo": [{"thumburl": big}]}}}}),
            MagicMock(content=b"jpegbytes", raise_for_status=MagicMock()),
        ]
        mock_open.return_value.convert.return_value = "image"

        self.assertEqual(wikimedia.download(photo), "image")
        asked = mock_get.call_args_list[0].kwargs["params"]["iiurlwidth"]
        self.assertGreater(asked, wikimedia.SEARCH_THUMB_WIDTH)
        self.assertLessEqual(asked, wikimedia.COMMONS_MAX_THUMB)
        self.assertEqual(mock_get.call_args_list[1].args[0], big)

    @patch("app.services.wikimedia.Image.open")
    @patch("app.services.wikimedia.requests.get")
    def test_portrait_uses_the_search_thumbnail_unchanged(self, mock_get, mock_open):
        photo = wikimedia.Photo(
            title="File:Falls.jpg", url="https://upload.example.test/1600px-Falls.jpg",
            width=1920, height=2560, author="A", licence="CC0",
            location=None, descriptionurl="",
        )
        mock_get.return_value = MagicMock(content=b"jpegbytes",
                                          raise_for_status=MagicMock())
        mock_open.return_value.convert.return_value = "image"

        wikimedia.download(photo)
        self.assertEqual(mock_get.call_count, 1)  # no second API round-trip
        self.assertEqual(mock_get.call_args.args[0], photo.url)


class TestSubjectRotation(unittest.TestCase):
    """The account published "the creativity of God in mountains" three times
    while 45 subjects had never run once. The scheduler was shuffling its own
    hardcoded list and never reading the rotation history at all."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        patcher = patch.object(carousel, "OUT_DIR", self.tmp)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _history(self, subjects):
        with open(Path(self.tmp) / "recent_subjects.json", "w", encoding="utf-8") as fh:
            json.dump(subjects, fh)

    def test_never_used_subjects_come_first(self):
        self._history(["mountains", "deserts"])
        ranked = carousel.rank_subjects(["mountains", "deserts", "whales"])
        self.assertEqual(ranked[0], "whales")

    def test_used_subjects_are_ordered_oldest_first(self):
        self._history(["mountains", "deserts", "waterfalls"])
        ranked = carousel.rank_subjects(["waterfalls", "deserts", "mountains"])
        self.assertEqual(ranked, ["mountains", "deserts", "waterfalls"])

    def test_history_is_not_truncated_below_the_pool_size(self):
        # A short window silently repeats topics while others wait their turn.
        self._history([f"s{i}" for i in range(200)])
        self.assertEqual(len(carousel._recent_subjects()), 200)

    def test_whole_pool_is_worked_through_before_repeating(self):
        pool = list(carousel.SUBJECTS)[:6]   # rank_subjects only knows real subjects
        history = []
        for _ in range(len(pool)):
            self._history(history)
            pick = carousel.rank_subjects(pool)[0]
            self.assertNotIn(pick, history, "repeated before the pool was exhausted")
            history.append(pick)
        self.assertCountEqual(history, pool)

    def test_scheduler_defaults_to_every_subject_not_a_hardcoded_list(self):
        from app.services import content_scheduler
        self.assertEqual(content_scheduler.CAROUSEL_SUBJECTS, [])


class TestQualityGate(unittest.TestCase):
    def _car(self, scales):
        return {
            "paths": [f"s{i}.jpg" for i in range(len(scales) + 1)],
            "photos": [MagicMock(url=f"u{i}") for i in range(len(scales))],
            "scales": scales,
        }

    def test_upscaled_slide_is_rejected(self):
        ok, reason = quality.check_carousel(self._car([0.6, 0.8, 1.25, 0.7]))
        self.assertFalse(ok)
        self.assertIn("1.25", reason)

    def test_downscaled_set_passes(self):
        ok, reason = quality.check_carousel(self._car([0.6, 0.88, 0.53, 0.7]))
        self.assertTrue(ok, reason)
        self.assertIn("0.88", reason)

    def test_rounding_is_tolerated(self):
        ok, _ = quality.check_carousel(self._car([1.0, 1.01, 0.9, 0.9]))
        self.assertTrue(ok)

    def test_missing_scales_still_checks_the_old_rules(self):
        car = self._car([0.5, 0.5, 0.5, 0.5])
        car.pop("scales")
        self.assertTrue(quality.check_carousel(car)[0])
        car["photos"] = [MagicMock(url="same"), MagicMock(url="same")]
        self.assertFalse(quality.check_carousel(car)[0])


if __name__ == "__main__":
    unittest.main()
