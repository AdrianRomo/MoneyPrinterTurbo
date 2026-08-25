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


class TestLedgerReconciliation(unittest.TestCase):
    """Deleting from Postiz is only half a flush.

    Per-type quota is counted from the LOCAL publish log, not from Postiz, so a
    deleted post's ledger row keeps holding its slot. After the first real flush
    the scheduler reported "next post slot is 2026-09-01" against a completely
    empty queue and would have published nothing for a week.
    """

    def _log(self):
        return [
            {"kind": "post", "date": "2026-08-26", "post_id": "q1"},
            {"kind": "reel", "date": "2026-08-27", "post_id": "q2"},
            {"kind": "reel", "date": "2026-08-20", "post_id": "live1"},
            {"kind": "story", "date": "2026-08-19"},  # predates id tracking
        ]

    def _prune(self, ids, tmp):
        path = str(Path(tmp) / "publish_log.json")
        Path(path).write_text(json.dumps(self._log()))
        with patch("app.services.postiz.PostizService._publish_log_path",
                   staticmethod(lambda: path)):
            result = queue_admin.prune_publish_log(ids)
        return result, json.loads(Path(path).read_text())

    def test_only_deleted_posts_lose_their_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, kept = self._prune({"q1", "q2"}, tmp)
        self.assertEqual(result["pruned"], 2)
        self.assertEqual({e.get("post_id") for e in kept}, {"live1", None})

    def test_a_still_live_post_keeps_holding_its_slot(self):
        # Pruning a post that is still in Postiz would let the scheduler
        # double-book the day.
        with tempfile.TemporaryDirectory() as tmp:
            _, kept = self._prune({"q1"}, tmp)
        self.assertIn("q2", {e.get("post_id") for e in kept})

    def test_entries_without_an_id_are_never_pruned(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, kept = self._prune({"q1", "q2", "live1"}, tmp)
        self.assertEqual(len(kept), 1)
        self.assertIsNone(kept[0].get("post_id"))

    def test_the_log_is_backed_up_before_it_is_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _ = self._prune({"q1"}, tmp)
            saved = json.loads(Path(result["backup"]).read_text())
        self.assertEqual(len(saved), 4)

    def test_flush_prunes_only_what_it_actually_deleted(self):
        posts = [{"id": "q1", "state": "QUEUE"}, {"id": "q2", "state": "QUEUE"}]

        def delete(_svc, post_id):
            return (post_id == "q1", "deleted" if post_id == "q1" else "http 500")

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(queue_admin, "BACKUP_DIR", tmp), \
                 patch.object(queue_admin, "list_posts", return_value=posts), \
                 patch.object(queue_admin, "delete_post", side_effect=delete), \
                 patch.object(queue_admin, "prune_publish_log",
                              return_value={"pruned": 1}) as pruner, \
                 patch.object(queue_admin.time, "sleep"):
                queue_admin.flush(FakeService(),
                                  datetime(2026, 8, 1, tzinfo=timezone.utc),
                                  datetime(2026, 9, 1, tzinfo=timezone.utc),
                                  apply=True, reason="test")
        pruner.assert_called_once_with({"q1"})

    def test_nothing_deleted_means_the_ledger_is_untouched(self):
        posts = [{"id": "q1", "state": "QUEUE"}]
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(queue_admin, "BACKUP_DIR", tmp), \
                 patch.object(queue_admin, "list_posts", return_value=posts), \
                 patch.object(queue_admin, "delete_post", return_value=(False, "http 429")), \
                 patch.object(queue_admin, "prune_publish_log") as pruner, \
                 patch.object(queue_admin.time, "sleep"):
                queue_admin.flush(FakeService(),
                                  datetime(2026, 8, 1, tzinfo=timezone.utc),
                                  datetime(2026, 9, 1, tzinfo=timezone.utc),
                                  apply=True, reason="test")
        pruner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
