"""Backward-compatibility regression tests for the existing topic workflow."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.schema import MaterialInfo, VideoParams
from app.services import task as task_service


class TestSchemaDefaultsUnchanged(unittest.TestCase):
    def test_videoparams_defaults_preserve_topic_behaviour(self):
        params = VideoParams(video_subject="anything")
        # New fields must default so old payloads behave exactly as before.
        self.assertEqual(params.content_mode, "topic")
        self.assertEqual(params.media_mode, "videos_only")
        self.assertEqual(params.image_source, "pexels")
        self.assertIsNone(params.article_url)
        self.assertIsNone(params.article_id)
        self.assertIsNone(params.article_script)
        # Core legacy fields intact.
        self.assertEqual(params.video_source, "pexels")
        self.assertEqual(params.video_aspect, "9:16")

    def test_old_payload_without_new_fields_validates(self):
        legacy_payload = {
            "video_subject": "spring",
            "video_script": "",
            "video_source": "pexels",
            "voice_name": "zh-CN-XiaoxiaoNeural-Female",
        }
        params = VideoParams(**legacy_payload)
        self.assertEqual(params.content_mode, "topic")

    def test_material_info_fields_intact(self):
        material = MaterialInfo(provider="local", url="a.mp4", duration=0)
        self.assertEqual(material.provider, "local")
        self.assertEqual(material.url, "a.mp4")


class TestTaskRouting(unittest.TestCase):
    def test_topic_routes_to_legacy_pipeline(self):
        params = VideoParams(video_subject="x")  # content_mode defaults to topic
        with patch.object(task_service, "_run_pipeline", return_value={"pipeline": "topic"}) as legacy:
            result = task_service.start("t", params, stop_at="video")
        legacy.assert_called_once()
        self.assertEqual(result, {"pipeline": "topic"})

    def test_article_routes_to_article_pipeline(self):
        params = VideoParams(video_subject="x", content_mode="article_feed")
        with patch("app.services.article_pipeline.render_article_video", return_value={"pipeline": "article"}) as article, \
             patch.object(task_service, "_run_pipeline") as legacy:
            result = task_service.start("t", params, stop_at="video")
        article.assert_called_once()
        legacy.assert_not_called()
        self.assertEqual(result, {"pipeline": "article"})


class TestLocalImageWorkflowRegression(unittest.TestCase):
    def test_preprocess_video_empty_list(self):
        # The existing local-material preprocessing must remain callable and
        # behave as before (empty in -> empty out).
        from app.services import video

        self.assertEqual(video.preprocess_video([]), [])
        # The compatibility function still exists with its original name.
        self.assertTrue(hasattr(video, "preprocess_video"))


if __name__ == "__main__":
    unittest.main()
