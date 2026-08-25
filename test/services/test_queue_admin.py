"""Tests for the Postiz queue flush tool.

The important one is that PUBLISHED posts can never be deleted: a date-ranged
GET /posts returns them alongside queued posts, so the naive flush would delete
the account's entire live history.
"""

import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import queue_admin


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code != 200:
            raise AssertionError(f"http {self.status_code}")


def _posts():
    return [
        {"id": "pub1", "state": "PUBLISHED", "publishDate": "2026-08-20T00:00:00Z"},
        {"id": "pub2", "state": "PUBLISHED", "publishDate": "2026-08-21T00:00:00Z"},
        {"id": "q1", "state": "QUEUE", "publishDate": "2026-08-30T00:00:00Z"},
        {"id": "d1", "state": "DRAFT", "publishDate": "2026-08-31T00:00:00Z"},
        {"id": "e1", "state": "ERROR", "publishDate": "2026-08-19T00:00:00Z"},
    ]


class FakeService:
    api_key = "secret-key"

    def _endpoint(self, path):
        return f"http://postiz.invalid/api/public/v1/{path}"

    def _headers(self):
        return {"Authorization": self.api_key}


class TestOnlyQueuedPostsAreFlushable(unittest.TestCase):
    def test_published_posts_are_never_flushable(self):
        ids = {p["id"] for p in queue_admin.flushable(_posts())}
        self.assertEqual(ids, {"q1", "d1"})

    def test_errored_posts_are_protected(self):
        # They pin their R2 media and are worth a human look first.
        self.assertNotIn("ERROR", queue_admin.FLUSHABLE_STATES)

    def test_published_is_not_in_the_flushable_set(self):
        self.assertNotIn("PUBLISHED", queue_admin.FLUSHABLE_STATES)

    def test_state_matching_is_case_insensitive(self):
        posts = [{"id": "q", "state": "queue"}, {"id": "p", "state": "published"}]
        self.assertEqual([p["id"] for p in queue_admin.flushable(posts)], ["q"])

    def test_unknown_state_is_protected_rather_than_assumed_safe(self):
        posts = [{"id": "x", "state": "SOMETHING_NEW"}, {"id": "y"}]
        self.assertEqual(queue_admin.flushable(posts), [])


class TestFlushIsDryRunByDefault(unittest.TestCase):
    def _run(self, apply, tmp):
        with patch.object(queue_admin, "BACKUP_DIR", tmp), \
             patch.object(queue_admin, "list_posts", return_value=_posts()), \
             patch.object(queue_admin, "delete_post", return_value=(True, "deleted")) as deleter, \
             patch.object(queue_admin.time, "sleep"):
            result = queue_admin.flush(FakeService(),
                                       datetime(2026, 8, 1, tzinfo=timezone.utc),
                                       datetime(2026, 9, 1, tzinfo=timezone.utc),
                                       apply=apply, reason="test")
        return result, deleter

    def test_dry_run_deletes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, deleter = self._run(False, tmp)
        deleter.assert_not_called()
        self.assertFalse(result["applied"])
        self.assertEqual(result["flushable"], 2)
        self.assertEqual(result["protected"], 3)

    def test_dry_run_still_writes_a_backup_to_read_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _ = self._run(False, tmp)
            saved = json.load(open(result["backup"]))
        self.assertEqual(saved["count"], 2)
        self.assertEqual({p["id"] for p in saved["posts"]}, {"q1", "d1"})

    def test_apply_deletes_only_the_flushable_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, deleter = self._run(True, tmp)
        deleted = {call.args[1] for call in deleter.call_args_list}
        self.assertEqual(deleted, {"q1", "d1"})
        self.assertEqual(result["deleted"], 2)

    def test_backup_is_written_before_any_delete(self):
        order = []
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(queue_admin, "BACKUP_DIR", tmp), \
                 patch.object(queue_admin, "list_posts", return_value=_posts()), \
                 patch.object(queue_admin, "write_backup",
                              side_effect=lambda p, r: order.append("backup") or "/x.json"), \
                 patch.object(queue_admin, "delete_post",
                              side_effect=lambda s, i: order.append("delete") or (True, "ok")), \
                 patch.object(queue_admin.time, "sleep"):
                queue_admin.flush(FakeService(),
                                  datetime(2026, 8, 1, tzinfo=timezone.utc),
                                  datetime(2026, 9, 1, tzinfo=timezone.utc),
                                  apply=True, reason="test")
        self.assertEqual(order[0], "backup")

    def test_remaining_is_re_queried_not_inferred(self):
        # A delete that returns 200 without sticking must still be visible.
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(queue_admin, "BACKUP_DIR", tmp), \
                 patch.object(queue_admin, "list_posts", return_value=_posts()), \
                 patch.object(queue_admin, "delete_post", return_value=(True, "deleted")), \
                 patch.object(queue_admin.time, "sleep"):
                result = queue_admin.flush(FakeService(),
                                           datetime(2026, 8, 1, tzinfo=timezone.utc),
                                           datetime(2026, 9, 1, tzinfo=timezone.utc),
                                           apply=True, reason="test")
        # list_posts still reports them, so remaining is 2 despite 2 "successes".
        self.assertEqual(result["remaining"], 2)


class TestThrottleHandling(unittest.TestCase):
    def test_429_is_retried_with_backoff_then_succeeds(self):
        responses = [FakeResponse(429), FakeResponse(429), FakeResponse(200)]
        with patch.object(queue_admin.requests, "delete", side_effect=responses), \
             patch.object(queue_admin.time, "sleep") as slept:
            ok, detail = queue_admin.delete_post(FakeService(), "q1")
        self.assertTrue(ok)
        self.assertEqual(detail, "deleted")
        # Backoff must grow, not poll at a fixed interval.
        waits = [c.args[0] for c in slept.call_args_list]
        self.assertEqual(waits, sorted(waits))
        self.assertGreater(waits[-1], waits[0])

    def test_persistent_throttle_reports_failure_rather_than_success(self):
        with patch.object(queue_admin.requests, "delete", return_value=FakeResponse(429)), \
             patch.object(queue_admin.time, "sleep"):
            ok, detail = queue_admin.delete_post(FakeService(), "q1")
        self.assertFalse(ok)
        self.assertIn("throttled", detail)

    def test_non_throttle_error_is_not_retried(self):
        with patch.object(queue_admin.requests, "delete",
                          return_value=FakeResponse(404, text="nope")) as deleter, \
             patch.object(queue_admin.time, "sleep"):
            ok, _ = queue_admin.delete_post(FakeService(), "q1")
        self.assertFalse(ok)
        self.assertEqual(deleter.call_count, 1)

    def test_the_api_key_is_never_returned_in_an_error(self):
        with patch.object(queue_admin.requests, "delete",
                          return_value=FakeResponse(500, text="boom secret-key leaked")), \
             patch.object(queue_admin.time, "sleep"):
            _, detail = queue_admin.delete_post(FakeService(), "q1")
        self.assertNotIn("secret-key", detail)


if __name__ == "__main__":
    unittest.main()
