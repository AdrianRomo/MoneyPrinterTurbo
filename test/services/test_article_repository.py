"""SQLite article repository tests."""

import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.article import (
    ArticleCluster,
    ArticleRecord,
    ArticleStatus,
    PollRun,
    StoryAssessment,
    TopicSubscription,
)
from app.services.article_repository import SqliteArticleRepository


class TestSqliteRepository(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.repo = SqliteArticleRepository(os.path.join(self.dir, "a.db"))

    def test_wal_mode_enabled(self):
        con = sqlite3.connect(os.path.join(self.dir, "a.db"))
        mode = con.execute("PRAGMA journal_mode").fetchone()[0]
        con.close()
        self.assertEqual(mode.lower(), "wal")

    def test_subscription_crud(self):
        sub = TopicSubscription(name="AI")
        self.repo.upsert_subscription(sub)
        sub.name = "AI News"
        self.repo.upsert_subscription(sub)
        fetched = self.repo.get_subscription(sub.id)
        self.assertEqual(fetched.name, "AI News")
        self.assertEqual(len(self.repo.list_subscriptions()), 1)
        self.assertTrue(self.repo.delete_subscription(sub.id))
        self.assertIsNone(self.repo.get_subscription(sub.id))

    def test_article_dedupe_lookups(self):
        art = ArticleRecord(
            subscription_id="s", domain="ex.com",
            canonical_url="https://ex.com/a", text_hash="h1",
            title="T", status=ArticleStatus.extracted,
        )
        self.repo.save_article(art)
        self.assertEqual(self.repo.find_article_by_hash("h1").id, art.id)
        self.assertEqual(self.repo.find_article_by_canonical("https://ex.com/a").id, art.id)
        self.assertIsNone(self.repo.find_article_by_hash("nope"))
        self.assertEqual(len(self.repo.list_articles(status="extracted")), 1)
        self.assertEqual(len(self.repo.list_articles(domain="ex.com")), 1)

    def test_clusters_and_assessments(self):
        cluster = ArticleCluster(subscription_id="s", normalized_title="t", domains=["a.com", "b.com"])
        self.repo.save_cluster(cluster)
        self.assertEqual(cluster.independent_domain_count, 2)
        self.repo.save_assessment(cluster.id, StoryAssessment(story_score=0.9, confidence=0.8))
        self.assertAlmostEqual(self.repo.get_assessment(cluster.id).story_score, 0.9)
        self.assertEqual(len(self.repo.list_clusters(subscription_id="s")), 1)

    def test_generation_bookkeeping(self):
        self.assertFalse(self.repo.has_generated_cluster("c1"))
        self.repo.record_generation("c1", "task-1", published=False)
        self.repo.record_generation("c1", "task-1:pub", published=True)
        self.assertTrue(self.repo.has_generated_cluster("c1"))
        self.assertEqual(self.repo.count_generations(), 2)
        self.assertEqual(self.repo.count_publications(), 1)

    def test_poll_run_persistence(self):
        run = PollRun(subscription_id="s", feeds_polled=2, articles_found=3)
        self.repo.save_poll_run(run)  # should not raise


if __name__ == "__main__":
    unittest.main()
