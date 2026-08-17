import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.postiz import (
    PostizService,
    _local_date,
    _local_midnight_utc,
    redact_postiz_text,
)


_CONFIG_BASE = {
    "postiz_enabled": True,
    "postiz_base_url": "https://postiz.example.test/public/v1",
    "postiz_api_key": "postiz-secret-key",
    "postiz_integration_id": "ig-1",
    "postiz_provider_type": "instagram",
    "postiz_auto_schedule_enabled": True,
    "postiz_schedule_interval_hours": 2,
    "postiz_schedule_jitter_minutes": 0,
    "postiz_daily_post_cap": 8,
    "postiz_post_type": "post",
}


def _response(payload):
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status = MagicMock()
    return response


class TestPostizService(unittest.TestCase):
    @patch("app.services.postiz.config.app", {**_CONFIG_BASE, "postiz_enabled": False})
    @patch("app.services.postiz.requests.get")
    def test_disabled_service_skips_integration_lookup(self, mock_get):
        result = PostizService().list_integrations()

        self.assertFalse(result["success"])
        self.assertIn("not configured", result["error"])
        mock_get.assert_not_called()

    @patch("app.services.postiz.config.app", _CONFIG_BASE)
    @patch("app.services.postiz.requests.get")
    def test_get_configured_integration_requires_matching_enabled_instagram(self, mock_get):
        mock_get.return_value = _response([
            {
                "id": "ig-1",
                "identifier": "instagram",
                "name": "Daily Bible Tips",
                "disabled": False,
            }
        ])

        result = PostizService().get_configured_integration()

        self.assertTrue(result["success"])
        self.assertEqual(result["integration"]["id"], "ig-1")

        mock_get.return_value = _response([
            {"id": "ig-1", "identifier": "instagram", "disabled": True}
        ])
        disabled = PostizService().get_configured_integration()
        self.assertFalse(disabled["success"])
        self.assertIn("disabled", disabled["error"])

    @patch("app.services.postiz.config.app", _CONFIG_BASE)
    @patch("app.services.postiz.requests.post")
    def test_missing_video_file_skips_upload_request(self, mock_post):
        result = PostizService().upload_media("/missing/video.mp4")

        self.assertFalse(result["success"])
        self.assertIn("not found", result["error"])
        mock_post.assert_not_called()

    @patch("app.services.postiz.config.app", _CONFIG_BASE)
    @patch("app.services.postiz.requests.post")
    def test_upload_media_sends_multipart_file(self, mock_post):
        mock_post.return_value = _response(
            {"id": "media-1", "path": "https://uploads.postiz.example/reel.mp4"}
        )
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
            handle.write(b"fake mp4")
            video_path = handle.name
        try:
            result = PostizService().upload_media(video_path)
        finally:
            os.unlink(video_path)

        self.assertTrue(result["success"])
        self.assertEqual(result["media"]["id"], "media-1")
        self.assertTrue(mock_post.call_args[0][0].endswith("/upload"))
        self.assertIn("file", mock_post.call_args[1]["files"])

    @patch("app.services.postiz.config.app", _CONFIG_BASE)
    @patch("app.services.postiz.requests.post")
    def test_schedule_post_payload_uses_instagram_settings(self, mock_post):
        mock_post.return_value = _response({"postId": "post-1", "integration": "ig-1"})
        publish_at = datetime(2026, 7, 29, 20, 0, tzinfo=timezone.utc)

        result = PostizService().schedule_post(
            {"id": "media-1", "path": "https://uploads.postiz.example/reel.mp4?sig=abc"},
            "One Bible-based tip for today. #Bible",
            publish_at,
            integration={"id": "ig-1"},
        )

        self.assertTrue(result["success"])
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["type"], "schedule")
        self.assertEqual(payload["date"], "2026-07-29T20:00:00.000Z")
        post = payload["posts"][0]
        self.assertEqual(post["integration"]["id"], "ig-1")
        self.assertEqual(post["settings"]["__type"], "instagram")
        self.assertEqual(post["settings"]["post_type"], "post")
        self.assertEqual(post["value"][0]["image"][0]["id"], "media-1")

    @patch("app.services.postiz.config.app", _CONFIG_BASE)
    def test_select_publish_at_honors_daily_cap(self):
        service = PostizService()
        now = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)
        same_day_post = {
            "publishDate": "2026-07-29T18:00:00.000Z",
            "integration": {"id": "ig-1"},
        }
        service.daily_post_cap = 1

        with (
            patch.object(
                service,
                "get_configured_integration",
                return_value={"success": True, "integration": {"id": "ig-1"}},
            ),
            patch.object(
                service,
                "find_available_slot",
                return_value={
                    "success": True,
                    "date": now + timedelta(hours=2),
                },
            ),
            patch.object(
                service,
                "list_posts",
                return_value={"success": True, "posts": [same_day_post]},
            ),
        ):
            result = service.select_publish_at(now=now)

        self.assertTrue(result["success"])
        self.assertEqual(result["publish_at"].date().isoformat(), "2026-07-30")

    @patch("app.services.postiz.config.app", _CONFIG_BASE)
    @patch("app.services.postiz.requests.post")
    @patch("app.services.postiz.requests.get")
    def test_schedule_video_contract_calls(self, mock_get, mock_post):
        def get_response(url, **_kwargs):
            if url.endswith("/integrations"):
                return _response([
                    {"id": "ig-1", "identifier": "instagram", "disabled": False}
                ])
            if url.endswith("/find-slot/ig-1"):
                return _response({"date": "2026-07-29T20:00:00.000Z"})
            if url.endswith("/posts"):
                return _response({"posts": []})
            raise AssertionError(f"unexpected GET {url}")

        def post_response(url, **_kwargs):
            if url.endswith("/upload"):
                return _response(
                    {"id": "media-1", "path": "https://uploads.postiz.example/reel.mp4"}
                )
            if url.endswith("/posts"):
                return _response({"postId": "post-1", "integration": "ig-1"})
            raise AssertionError(f"unexpected POST {url}")

        mock_get.side_effect = get_response
        mock_post.side_effect = post_response
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as handle:
            handle.write(b"fake mp4")
            video_path = handle.name
        variant = {"caption_style": "saveable_contemplative"}
        try:
            with patch.object(PostizService, "_record_publish") as record_publish:
                result = PostizService().schedule_video(
                    video_path,
                    "One Bible-based tip for today. #Bible",
                    now=datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc),
                    variant=variant,
                    set_id="peace",
                )
        finally:
            os.unlink(video_path)

        self.assertTrue(result["success"])
        self.assertEqual(result["post_id"], "post-1")
        record_publish.assert_called_once()
        self.assertEqual(record_publish.call_args.kwargs["set_id"], "peace")
        self.assertEqual(record_publish.call_args.kwargs["variant"], variant)
        self.assertEqual(mock_get.call_count, 4)  # integration verified twice, slot, posts
        self.assertEqual(mock_post.call_count, 2)  # upload, create post

    def test_redaction_removes_api_key_and_signed_url_query(self):
        text = (
            "failed with postiz-secret-key at "
            "https://uploads.postiz.example/reel.mp4?token=abc&sig=def"
        )

        redacted = redact_postiz_text(text, "postiz-secret-key")

        self.assertNotIn("postiz-secret-key", redacted)
        self.assertNotIn("token=abc", redacted)
        self.assertIn("https://uploads.postiz.example/reel.mp4", redacted)

    @patch("app.services.postiz.config.app", _CONFIG_BASE)
    @patch("app.services.postiz.requests.get")
    def test_request_errors_are_returned_as_safe_failures(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout(
            "timeout for https://postiz.example.test/public/v1/integrations?api_key=postiz-secret-key"
        )

        result = PostizService().list_integrations()

        self.assertFalse(result["success"])
        self.assertNotIn("postiz-secret-key", result["error"])
        self.assertNotIn("api_key=", result["error"])


_TZ_CONFIG = dict(_CONFIG_BASE, content_timezone="America/Mexico_City")


class LocalQuotaDayTest(unittest.TestCase):
    """The quota day is the audience's civil day, not UTC.

    Mexico City sits at -6, so the UTC day rolls over at 18:00 local. Counting
    quota in UTC therefore handed every evening slot a fresh, empty quota and
    split a single local day's posts across two ledger days.
    """

    @patch("app.services.postiz.config.app", _TZ_CONFIG)
    def test_evening_and_morning_share_one_local_day(self):
        morning = datetime(2026, 8, 16, 14, 7, tzinfo=timezone.utc)   # 08:07 local
        evening = datetime(2026, 8, 17, 2, 5, tzinfo=timezone.utc)    # 20:05 local

        # They differ in UTC — that is precisely the trap being closed.
        self.assertNotEqual(morning.date(), evening.date())
        self.assertEqual(_local_date(morning), _local_date(evening))

    @patch("app.services.postiz.config.app", _TZ_CONFIG)
    def test_local_midnight_maps_to_0600_utc(self):
        midnight = _local_midnight_utc(_local_date(
            datetime(2026, 8, 16, 14, 7, tzinfo=timezone.utc)))

        self.assertEqual(midnight, datetime(2026, 8, 16, 6, 0, tzinfo=timezone.utc))

    @patch("app.services.postiz.config.app",
           dict(_CONFIG_BASE, content_timezone="Not/AZone",
                content_utc_offset_hours=-6))
    def test_unusable_timezone_falls_back_to_fixed_offset(self):
        # A missing tzdata or a typo must degrade to the old behaviour rather
        # than raising in the middle of a publish.
        evening = datetime(2026, 8, 17, 2, 5, tzinfo=timezone.utc)

        self.assertEqual(_local_date(evening), datetime(2026, 8, 16).date())


if __name__ == "__main__":
    unittest.main()
