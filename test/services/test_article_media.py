"""Image providers and media rendering tests.

Provider HTTP calls are mocked. Rendering uses the ffmpeg bundled with
imageio-ffmpeg, small generated fixtures, and silent audio — no network.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PIL import Image

from app.models.article import AutomationSettings, GeneratedScript, MediaAsset, Scene
from app.models.schema import VideoAspect, VideoParams
from app.services import article_pipeline, material, video
from app.utils import utils


class _FakeResponse:
    def __init__(self, payload, headers=None, status_code=200, text=""):
        self._payload = payload
        self.headers = headers or {"content-type": "application/json"}
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._payload


def _fixture_image(path, size=(1200, 800), color=(180, 60, 60)):
    Image.new("RGB", size, color).save(path)
    return path


def _silent_audio(path, seconds=4):
    subprocess.run(
        [
            utils.get_ffmpeg_binary(), "-y", "-f", "lavfi",
            "-i", "anullsrc=r=44100:cl=stereo", "-t", str(seconds), path,
        ],
        check=True, capture_output=True,
    )
    return path


class TestImageProviders(unittest.TestCase):
    def test_pexels_image_search(self):
        payload = {
            "photos": [
                {
                    "id": 123, "width": 1080, "height": 1920,
                    "photographer": "Jane Doe",
                    "url": "https://www.pexels.com/photo/123/?utm_source=x",
                    "src": {"large2x": "https://images.pexels.com/123?auto=compress"},
                }
            ]
        }
        with patch.object(material, "get_api_key", return_value="k"), patch.object(
            material.requests, "get", return_value=_FakeResponse(payload)
        ):
            assets = material.search_images_pexels("city skyline", VideoAspect.portrait)
        self.assertEqual(len(assets), 1)
        asset = assets[0]
        self.assertEqual(asset.provider, "pexels")
        self.assertEqual(asset.creator, "Jane Doe")
        self.assertEqual(asset.license_name, "Pexels License")
        # source page URL is stripped of query params (no tracking / secrets)
        self.assertEqual(asset.source_page_url, "https://www.pexels.com/photo/123/")

    def test_pixabay_image_search(self):
        payload = {
            "hits": [
                {
                    "id": 7, "imageWidth": 1920, "imageHeight": 1280, "user": "Bob",
                    "pageURL": "https://pixabay.com/photos/x-7/",
                    "largeImageURL": "https://pixabay.com/get/7.jpg?token=abc",
                }
            ]
        }
        headers = {"content-type": "application/json"}
        with patch.object(material, "get_api_key", return_value="k"), patch.object(
            material.requests, "get", return_value=_FakeResponse(payload, headers=headers)
        ):
            assets = material.search_images_pixabay("mountain", VideoAspect.landscape)
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0].provider, "pixabay")
        self.assertEqual(assets[0].license_name, "Pixabay Content License")


class TestImageValidation(unittest.TestCase):
    def test_valid_image_passes(self):
        with tempfile.TemporaryDirectory() as d:
            path = _fixture_image(os.path.join(d, "a.png"), (800, 800))
            width, height = material.validate_image_file(path)
            self.assertEqual((width, height), (800, 800))

    def test_too_small_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = _fixture_image(os.path.join(d, "a.png"), (100, 100))
            with self.assertRaises(ValueError):
                material.validate_image_file(path)

    def test_non_image_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.png")
            with open(path, "wb") as fp:
                fp.write(b"this is definitely not an image file")
            with self.assertRaises(ValueError):
                material.validate_image_file(path)


class TestLicenseMetadataPersistence(unittest.TestCase):
    def test_manifest_entry_has_license_no_secrets(self):
        asset = MediaAsset(
            provider="pexels", asset_id="1", license_name="Pexels License",
            license_url="https://www.pexels.com/license/",
            attribution_text="Photo by X on Pexels",
            source_page_url="https://www.pexels.com/photo/1/?utm_source=x",
            url="https://images.pexels.com/1?token=SECRET",
            local_path="/tasks/t/article-scene-1.mp4",
        )
        entry = asset.manifest_entry()
        self.assertEqual(entry["license_name"], "Pexels License")
        self.assertEqual(entry["source_page_url"], "https://www.pexels.com/photo/1/")
        self.assertNotIn("token", str(entry))  # signed download URL never persisted
        self.assertEqual(entry["local_file"], "article-scene-1.mp4")


class TestMediaRendering(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.task_patch = patch.object(
            article_pipeline.utils, "task_dir", return_value=self.dir
        )
        self.task_patch.start()

    def tearDown(self):
        self.task_patch.stop()

    def _script(self, n=3):
        return GeneratedScript(
            title="Story",
            scenes=[
                Scene(narration=f"scene {i}", visual_queries=[f"query {i}"], duration_weight=1.0)
                for i in range(n)
            ],
        )

    def test_scene_durations_cover_audio(self):
        durations = article_pipeline.scene_durations([1.0, 2.0, 1.0], audio_duration=8.0)
        self.assertEqual(len(durations), 3)
        self.assertGreaterEqual(sum(durations), 8.0)
        self.assertTrue(all(d >= 1.6 for d in durations))

    def test_images_only_timeline_in_order_with_fallback(self):
        img = _fixture_image(os.path.join(self.dir, "img.png"))

        def searcher(query):
            # scene 1 (index 1) returns nothing -> fallback background
            if query == "query 1":
                return []
            return [MediaAsset(media_type="image", provider="pexels", asset_id=query, url="u", width=1080, height=1080)]

        settings = AutomationSettings()
        clip_paths, assets = article_pipeline.build_visual_timeline(
            "task", self._script(3), audio_duration=6.0,
            video_aspect=VideoAspect.portrait, provider="pexels", settings=settings,
            searcher=searcher, downloader=lambda url: img,
        )
        # One clip per scene, in order.
        self.assertEqual(len(clip_paths), 3)
        self.assertEqual(
            [os.path.basename(p) for p in clip_paths],
            ["article-scene-1.mp4", "article-scene-2.mp4", "article-scene-3.mp4"],
        )
        for path in clip_paths:
            self.assertTrue(os.path.exists(path))
        # scene 1 had no asset -> only 2 assets kept, and beat indices preserved
        self.assertEqual(sorted(a.beat_index for a in assets), [0, 2])

    def test_images_only_full_render_9_16(self):
        img = _fixture_image(os.path.join(self.dir, "img.png"))
        settings = AutomationSettings()
        clip_paths, _ = article_pipeline.build_visual_timeline(
            "task", self._script(2), audio_duration=4.0,
            video_aspect=VideoAspect.portrait, provider="pexels", settings=settings,
            searcher=lambda q: [MediaAsset(media_type="image", provider="pexels", asset_id=q, url="u", width=1080, height=1080)],
            downloader=lambda url: img,
        )
        combined = os.path.join(self.dir, "combined-1.mp4")
        video.combine_article_clips(clip_paths, combined, audio_duration=4.0)
        audio = _silent_audio(os.path.join(self.dir, "audio.wav"), 4)
        final = os.path.join(self.dir, "final-1.mp4")
        params = VideoParams(video_subject="x", video_aspect="9:16", subtitle_enabled=False, bgm_type="")
        video.generate_video(combined, audio, "", final, params)
        self.assertTrue(os.path.exists(final) and os.path.getsize(final) > 0)
        probe = video._open_video_clip_quietly(final)
        self.assertEqual(list(probe.size), [1080, 1920])
        video.close_clip(probe)

    def test_mixed_media_render(self):
        img = _fixture_image(os.path.join(self.dir, "img.png"))
        # A small mp4 stands in for a downloaded "video" asset source.
        vid_src = video.image_to_video_clip(
            _fixture_image(os.path.join(self.dir, "v.png"), (1280, 720), (30, 120, 60)),
            os.path.join(self.dir, "vsrc.mp4"), 3.0, VideoAspect.portrait,
        )

        def searcher(query):
            if query == "query 0":  # first scene -> a video asset
                return [MediaAsset(media_type="video", provider="pexels", asset_id="v", url="vid://x", width=1280, height=720)]
            return [MediaAsset(media_type="image", provider="pexels", asset_id=query, url="img://x", width=1080, height=1080)]

        def downloader(url):
            return vid_src if url.startswith("vid://") else img

        settings = AutomationSettings()
        clip_paths, assets = article_pipeline.build_visual_timeline(
            "task", self._script(2), audio_duration=5.0,
            video_aspect=VideoAspect.portrait, provider="pexels", settings=settings,
            searcher=searcher, downloader=downloader,
        )
        self.assertEqual(len(clip_paths), 2)
        for path in clip_paths:
            self.assertTrue(os.path.exists(path))
        # Both a video and an image asset were normalized and kept, in order.
        self.assertEqual([a.media_type for a in assets], ["video", "image"])
        combined = os.path.join(self.dir, "combined-1.mp4")
        video.combine_article_clips(clip_paths, combined, audio_duration=5.0)
        self.assertTrue(os.path.exists(combined))


if __name__ == "__main__":
    unittest.main()
