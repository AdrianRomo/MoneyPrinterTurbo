import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.services import content_scheduler
from app.services.postiz import PostizService


class TestContentScheduler(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    @staticmethod
    def _service(**quotas):
        def select_publish_at(now, kind):
            return {
                "success": True,
                "publish_at": (now + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
            }

        return SimpleNamespace(
            type_quotas={
                "post": quotas.get("post", 1),
                "carousel": quotas.get("carousel", 1),
                "reel": quotas.get("reel", 1),
                "story": quotas.get("story", 3),
            },
            select_publish_at=select_publish_at,
        )

    def test_reel_due_requires_quote_reel_auto_schedule(self):
        now = datetime(2026, 8, 17, 16, 0, tzinfo=timezone.utc)
        with (
            patch.dict(
                content_scheduler.config.app,
                {"quote_reel_auto_schedule_enabled": False},
                clear=False,
            ),
            patch.object(PostizService, "_count_kind_on", return_value=0),
        ):
            is_due, reason = content_scheduler.due("reel", now, self._service())

        self.assertFalse(is_due)
        self.assertIn("disabled", reason)

    def test_reel_due_uses_existing_reel_quota(self):
        now = datetime(2026, 8, 17, 16, 0, tzinfo=timezone.utc)
        with (
            patch.dict(
                content_scheduler.config.app,
                {"quote_reel_auto_schedule_enabled": True},
                clear=False,
            ),
            patch.object(PostizService, "_count_kind_on", return_value=1),
        ):
            is_due, reason = content_scheduler.due("reel", now, self._service(reel=1))

        self.assertFalse(is_due)
        self.assertIn("already scheduled 1 reel", reason)

    def test_reel_due_when_enabled_and_quota_is_open(self):
        now = datetime(2026, 8, 17, 16, 0, tzinfo=timezone.utc)
        with (
            patch.dict(
                content_scheduler.config.app,
                {"quote_reel_auto_schedule_enabled": True},
                clear=False,
            ),
            patch.object(PostizService, "_count_kind_on", return_value=0),
        ):
            is_due, reason = content_scheduler.due("reel", now, self._service(reel=1))

        self.assertTrue(is_due)
        self.assertEqual(reason, "due")

    def test_due_skips_when_next_slot_is_outside_horizon(self):
        now = datetime(2026, 8, 17, 16, 0, tzinfo=timezone.utc)
        service = self._service(reel=1)
        service.select_publish_at = lambda now, kind: {
            "success": True,
            "publish_at": (now + timedelta(days=3)).isoformat().replace("+00:00", "Z"),
        }

        with (
            patch.dict(
                content_scheduler.config.app,
                {
                    "quote_reel_auto_schedule_enabled": True,
                    "content_scheduler_schedule_days_ahead": 1,
                },
                clear=False,
            ),
            patch.object(PostizService, "_count_kind_on", return_value=1),
        ):
            is_due, reason = content_scheduler.due("reel", now, service)

        self.assertFalse(is_due)
        self.assertIn("outside 1d scheduler horizon", reason)

    def test_produce_reel_renders_and_returns_publish_result(self):
        render_result = {
            "videos": ["/tmp/final-quote-reel.mp4"],
            "quote_reel_artifact": "/tmp/quote_reel.json",
            "publish_result": {
                "success": True,
                "post_id": "post-1",
                "publish_at": "2026-08-18T02:00:00.000Z",
            },
        }

        with patch(
            "app.services.quote_reel.render_quote_reel",
            return_value=render_result,
        ) as render:
            outcome = content_scheduler.produce("reel")

        self.assertTrue(outcome["success"])
        self.assertEqual(outcome["post_id"], "post-1")
        self.assertEqual(outcome["video"], "/tmp/final-quote-reel.mp4")
        params = render.call_args.args[1]
        self.assertEqual(params.content_mode, "quiet_quote_reel")
        self.assertEqual(params.video_aspect, "9:16")

    def test_produce_carousel_retries_reliable_subjects(self):
        built = {"paths": ["/tmp/1.jpg"], "subject": "sunsets"}

        def build(*, subject, slides):
            return built if subject == "sunsets" else None

        with (
            patch.dict(
                content_scheduler.config.app,
                {
                    "content_scheduler_carousel_subjects": "whales,sunsets",
                    "content_scheduler_carousel_attempts": 2,
                },
                clear=False,
            ),
            patch("app.services.carousel.build", side_effect=build) as build_mock,
            patch(
                "app.services.carousel.publish",
                return_value={"success": True, "post_id": "carousel-1"},
            ) as publish,
            patch("app.services.content_scheduler.random.shuffle", lambda items: None),
        ):
            outcome = content_scheduler.produce("carousel")

        self.assertTrue(outcome["success"])
        self.assertEqual(outcome["post_id"], "carousel-1")
        self.assertEqual(
            [call.kwargs["subject"] for call in build_mock.call_args_list],
            ["whales", "sunsets"],
        )
        publish.assert_called_once_with(built)

    def test_run_once_can_target_reel_only(self):
        fake_service = SimpleNamespace(
            is_configured=lambda: True,
            type_quotas={"post": 1, "carousel": 1, "reel": 1, "story": 3},
        )
        with (
            patch(
                "app.services.content_scheduler.PostizService",
                return_value=fake_service,
            ),
            patch(
                "app.services.content_scheduler.due",
                return_value=(True, "due"),
            ) as due,
            patch(
                "app.services.content_scheduler.produce",
                return_value={
                    "success": True,
                    "post_id": "post-1",
                    "publish_at": "2026-08-18T02:00:00.000Z",
                },
            ) as produce,
        ):
            result = content_scheduler.run_once(only="reel")

        due.assert_called_once()
        produce.assert_called_once_with("reel")
        self.assertEqual(result["reel"]["action"], "published")
        self.assertEqual(result["reel"]["post_id"], "post-1")

    def test_run_once_force_bypasses_due_check_for_selected_kind(self):
        fake_service = SimpleNamespace(
            is_configured=lambda: True,
            type_quotas={"post": 1, "carousel": 1, "reel": 1, "story": 3},
        )
        with (
            patch(
                "app.services.content_scheduler.PostizService",
                return_value=fake_service,
            ),
            patch("app.services.content_scheduler.due") as due,
            patch(
                "app.services.content_scheduler.produce",
                return_value={
                    "success": True,
                    "post_id": "post-1",
                    "publish_at": "2026-08-18T02:00:00.000Z",
                },
            ) as produce,
        ):
            result = content_scheduler.run_once(only="reel", force=True)

        due.assert_not_called()
        produce.assert_called_once_with("reel")
        self.assertEqual(result["reel"]["action"], "published")


if __name__ == "__main__":
    unittest.main()
