"""Least-recently-used rotation.

The property under test is the one the account actually needed: a caller that
keeps picking must work through the WHOLE pool before anything comes round
again. `random.choice` satisfies none of this, which is why the feed read as one
photograph taken 41 times.
"""

import json
import os
import tempfile
import unittest

from app.services import rotation


class TestRank(unittest.TestCase):
    def test_never_used_come_first(self):
        ranked = rotation.rank(["a", "b", "c"], ["a", "b"])
        self.assertEqual(ranked[0], "c")

    def test_used_are_ordered_oldest_first(self):
        ranked = rotation.rank(["a", "b", "c"], ["c", "a", "b"])
        self.assertEqual(ranked, ["c", "a", "b"])

    def test_unknown_history_entries_are_ignored(self):
        self.assertEqual(rotation.rank(["a"], ["zzz", "a"]), ["a"])

    def test_empty_pool_ranks_to_nothing(self):
        self.assertEqual(rotation.rank([], ["a"]), [])

    def test_rank_does_not_touch_disk(self):
        # Pure, so a test can rank against a hypothetical history.
        self.assertEqual(rotation.rank(["a", "b"], ["a"]), ["b", "a"])


class TestState(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "state", "used.json")

    def tearDown(self):
        self._tmp.cleanup()

    def test_missing_file_is_an_empty_history(self):
        self.assertEqual(rotation.load_history(self.path), [])

    def test_corrupt_file_is_an_empty_history(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        self.assertEqual(rotation.load_history(self.path), [])

    def test_remember_creates_the_directory(self):
        rotation.remember(self.path, "a")
        self.assertEqual(rotation.load_history(self.path), ["a"])

    def test_remember_moves_a_repeat_to_the_end(self):
        for item in ("a", "b", "a"):
            rotation.remember(self.path, item)
        self.assertEqual(rotation.load_history(self.path), ["b", "a"])

    def test_remember_truncates_to_keep(self):
        for item in "abcde":
            rotation.remember(self.path, item, keep=3)
        self.assertEqual(rotation.load_history(self.path), ["c", "d", "e"])

    def test_limit_returns_the_most_recent(self):
        for item in "abcde":
            rotation.remember(self.path, item)
        self.assertEqual(rotation.load_history(self.path, 2), ["d", "e"])


class TestChoose(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "used.json")

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_whole_pool_is_used_before_anything_repeats(self):
        pool = list("abcdefgh")
        picked = [rotation.choose(pool, self.path) for _ in pool]
        self.assertEqual(sorted(picked), sorted(pool),
                         "a full cycle must cover the pool exactly once")

    def test_the_second_cycle_repeats_in_the_same_order(self):
        pool = list("abcd")
        first = [rotation.choose(pool, self.path) for _ in pool]
        second = [rotation.choose(pool, self.path) for _ in pool]
        self.assertEqual(first, second)

    def test_empty_pool_returns_none_rather_than_raising(self):
        self.assertIsNone(rotation.choose([], self.path))

    def test_remember_it_false_leaves_no_trace(self):
        rotation.choose(["a", "b"], self.path, remember_it=False)
        self.assertEqual(rotation.load_history(self.path), [])

    def test_a_grown_pool_prefers_the_new_entries(self):
        # Adding subjects to a pack must not wait a full cycle to take effect.
        for item in ("a", "b"):
            rotation.remember(self.path, item)
        self.assertEqual(rotation.choose(["a", "b", "c"], self.path), "c")


if __name__ == "__main__":
    unittest.main()
