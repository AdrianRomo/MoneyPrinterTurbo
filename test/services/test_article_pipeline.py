"""Automation pipeline tests: thresholds, gates and the render path."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PIL import Image

from app.models.article import (
    ArticleRecord,
    ArticleSource,
    AutomationMode,
    AutomationSettings,
    GeneratedScript,
    MediaAsset,
    RecommendedAction,
    RiskLevel,
    Scene,
    StoryAssessment,
)
from app.models.schema import VideoParams
from app.services import article_pipeline, material
from app.services import task as task_service
from app.services import voice
from app.utils import utils


def _silent_audio(path, seconds=4):
    subprocess.run(
        [utils.get_ffmpeg_binary(), "-y", "-f", "lavfi",
         "-i", "anullsrc=r=44100:cl=stereo", "-t", str(seconds), path],
        check=True, capture_output=True,
    )
    return path


class TestThresholdGates(unittest.TestCase):
    def setUp(self):
        self.settings = AutomationSettings(
            minimum_story_score=0.6, minimum_confidence_score=0.6
        )

    def test_should_generate_passes_and_fails(self):
        good = StoryAssessment(
            story_score=0.8,
            confidence=0.7,
            visual_potential=0.8,
            recommended_action=RecommendedAction.generate,
        )
        ok, _ = article_pipeline.should_generate(good, self.settings)
        self.assertTrue(ok)

        low = StoryAssessment(story_score=0.4, confidence=0.7, visual_potential=0.8)
        ok, reason = article_pipeline.should_generate(low, self.settings)
        self.assertFalse(ok)
        self.assertIn("story_score", reason)

        skip = StoryAssessment(
            story_score=0.9,
            confidence=0.9,
            visual_potential=0.8,
            recommended_action=RecommendedAction.skip,
        )
        ok, _ = article_pipeline.should_generate(skip, self.settings)
        self.assertFalse(ok)

    def test_should_render_by_mode(self):
        self.assertFalse(article_pipeline.should_render(self.settings, AutomationMode.assisted))
        self.assertTrue(article_pipeline.should_render(self.settings, AutomationMode.automated))
        self.assertTrue(article_pipeline.should_render(self.settings, AutomationMode.autonomous))

    def test_auto_publish_disabled_by_default(self):
        defaults = AutomationSettings()
        self.assertFalse(defaults.auto_publish_enabled)
        assessment = StoryAssessment(story_score=0.9, confidence=0.9, risk_level=RiskLevel.low)
        # Even autonomous mode will not publish while the flag is off.
        ok, reason = article_pipeline.should_publish(
            assessment, defaults, AutomationMode.autonomous
        )
        self.assertFalse(ok)
        self.assertIn("auto_publish_enabled", reason)

    def test_publish_requires_low_risk_and_non_sensitive(self):
        settings = AutomationSettings(auto_publish_enabled=True, maximum_risk_for_auto_publish=RiskLevel.low)
        low = StoryAssessment(risk_level=RiskLevel.low)
        ok, _ = article_pipeline.should_publish(low, settings, AutomationMode.autonomous)
        self.assertTrue(ok)

        high = StoryAssessment(risk_level=RiskLevel.high)
        ok, reason = article_pipeline.should_publish(high, settings, AutomationMode.autonomous)
        self.assertFalse(ok)
        self.assertIn("risk", reason)

        ok, reason = article_pipeline.should_publish(low, settings, AutomationMode.autonomous, sensitive=True)
        self.assertFalse(ok)
        self.assertIn("sensitive", reason)


class TestSceneDurations(unittest.TestCase):
    def test_covers_audio_no_zero_length(self):
        durations = article_pipeline.scene_durations([1, 1, 1, 1], audio_duration=10.0)
        self.assertGreaterEqual(sum(durations), 10.0)
        self.assertTrue(all(d > 0 for d in durations))

    def test_handles_empty(self):
        self.assertEqual(article_pipeline.scene_durations([], 10.0), [])


class TestRenderPath(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.img = os.path.join(self.dir, "img.png")
        Image.new("RGB", (1080, 1080), (60, 90, 160)).save(self.img)
        self.audio = _silent_audio(os.path.join(self.dir, "audio.wav"), 4)
        self.task_patch = patch("app.utils.utils.task_dir", return_value=self.dir)
        self.task_patch.start()

    def tearDown(self):
        self.task_patch.stop()

    def _script(self, narration_ok=True):
        scenes = [
            Scene(narration="The event happened on Tuesday." if narration_ok else "", visual_queries=["q0"]),
            Scene(narration="Officials responded quickly." if narration_ok else "", visual_queries=["q1"]),
        ]
        return GeneratedScript(
            title="Story",
            scenes=scenes,
            sources=[ArticleSource(publisher="Reuters", domain="reuters.com", canonical_url="https://reuters.com/x")],
            confidence=0.8,
        )

    def _params(self, script):
        return VideoParams(
            video_subject="Story", content_mode="article_feed", media_mode="images_only",
            image_source="pexels", video_aspect="9:16", subtitle_enabled=False, bgm_type="",
            article_script=script.model_dump(mode="json"),
        )

    def test_images_only_render_end_to_end(self):
        asset = MediaAsset(
            media_type="image", provider="pexels", asset_id="a", url="u",
            width=1080, height=1080, license_name="Pexels License",
            license_url="https://www.pexels.com/license/",
            source_page_url="https://www.pexels.com/photo/1/?utm=x",
            attribution_text="Photo by X on Pexels",
        )
        params = self._params(self._script())
        with patch.object(task_service, "generate_audio", return_value=(self.audio, 4, None)), \
             patch.object(task_service, "generate_subtitle", return_value=""), \
             patch.object(material, "search_images", return_value=[asset]), \
             patch.object(material, "save_image", return_value=self.img):
            result = article_pipeline.render_article_video("tid", params)
        self.assertIn("videos", result)
        self.assertTrue(os.path.exists(result["videos"][0]))
        self.assertEqual(result["article_sources"], ["Reuters"])
        self.assertEqual(result["provenance_label"], "Generated from the listed sources.")
        # Provenance artifacts written, secret-free.
        manifest = json.loads((Path(self.dir) / "media_manifest.json").read_text())
        self.assertTrue(manifest)
        self.assertEqual(manifest[0]["license_name"], "Pexels License")
        self.assertNotIn("token", (Path(self.dir) / "media_manifest.json").read_text())
        self.assertTrue((Path(self.dir) / "sources.json").exists())
        self.assertTrue((Path(self.dir) / "provenance.json").exists())

    def test_terminates_before_tts_when_no_narration(self):
        params = self._params(self._script(narration_ok=False))
        with patch.object(task_service, "generate_audio") as audio_mock, \
             patch.object(task_service, "generate_subtitle") as sub_mock:
            result = article_pipeline.render_article_video("tid", params)
        # Must fail at research and never reach TTS / subtitles.
        self.assertEqual(result.get("failed_stage"), "research")
        audio_mock.assert_not_called()
        sub_mock.assert_not_called()

    def test_script_entity_terms_uses_repo_article_subjects(self):
        article = ArticleRecord(
            id="article-1",
            cluster_id="cluster-1",
            title="NASA Mars update",
            entities=["NASA", "Mars"],
            keywords=["launch"],
        )

        class Repo:
            def list_articles(self, **_kwargs):
                return [article]

            def get_article(self, _article_id):
                return article

        script = GeneratedScript(
            cluster_id="cluster-1",
            primary_article_id="article-1",
            title="NASA Mars update",
            scenes=[Scene(narration="a")],
        )
        with patch(
            "app.services.article_repository.get_repository",
            return_value=Repo(),
        ):
            terms = article_pipeline._script_entity_terms(script)

        self.assertEqual(terms[:3], ["NASA", "Mars", "launch"])


class TestResolveScript(unittest.TestCase):
    def test_resolve_from_article_script(self):
        script = GeneratedScript(title="T", scenes=[Scene(narration="a")])
        params = VideoParams(video_subject="T", content_mode="article_feed", article_script=script.model_dump(mode="json"))
        resolved = article_pipeline.resolve_script(params)
        self.assertEqual(resolved.title, "T")

    def test_resolve_requires_source(self):
        params = VideoParams(video_subject="T", content_mode="article_feed")
        with self.assertRaises(ValueError):
            article_pipeline.resolve_script(params)

    def test_build_render_params_has_safe_worker_voice_default(self):
        script = GeneratedScript(title="T", scenes=[Scene(narration="a")])
        with patch.dict(article_pipeline.config.app, {"article_voice_name": ""}), patch.dict(
            article_pipeline.config.ui, {"voice_name": ""}
        ):
            params = article_pipeline.build_render_params(script)
        self.assertEqual(params.voice_name, voice.NO_VOICE_NAME)


if __name__ == "__main__":
    unittest.main()
