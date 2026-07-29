"""Article worker tests: poll, process, automation modes, limits (mocked)."""

import json
import os
import sys
import tempfile
import unittest
from email.utils import format_datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.models.article import AutomationMode, AutomationSettings, TopicSubscription, utcnow
from app.services import article_worker
from app.services.article_repository import SqliteArticleRepository

_RSS = (
    b"<rss version='2.0'><channel>"
    b"<item><title>Central bank raises rates</title><link>https://reuters.com/x</link>"
    b"<pubDate>{published}</pubDate></item>"
    b"</channel></rss>"
)
_BODY = "The central bank raised interest rates by 50 basis points on Tuesday to fight inflation. " * 25


def _feed_fetch(url):
    published = format_datetime(utcnow(), usegmt=True).encode("utf-8")
    return (url, _RSS.replace(b"{published}", published), "application/rss+xml")


def _article_fetch(url):
    html = f"<html><head><title>Central bank raises rates</title></head><body><p>{_BODY}</p></body></html>"
    return (url, html.encode(), "text/html")


def _mock_llm(prompt):
    if "Story Scorer" in prompt:
        return json.dumps({
            "story_score": 0.9,
            "confidence": 0.9,
            "visual_potential": 0.9,
            "risk_level": "low",
            "recommended_action": "generate",
        })
    if "Editorial Reviewer" in prompt:
        return json.dumps({"approved": True, "confidence": 0.9, "issues": []})
    return json.dumps({
        "title": "Rates up", "narration": "The central bank raised rates.",
        "scenes": [{"narration": "Central bank building", "visual_queries": ["central bank"], "visual_type": "image"}],
        "social_metadata": {"hashtags": ["#news"]},
    })


class TestWorker(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.repo = SqliteArticleRepository(os.path.join(self.dir, "a.db"))
        self.sub = TopicSubscription(name="AI", query="ai", rss_urls=["https://feeds.example.com/rss"])
        self.repo.upsert_subscription(self.sub)

    def test_poll_persists_articles(self):
        run, articles, clusters = article_worker.poll_subscription(
            self.repo, self.sub, feed_fetcher=_feed_fetch, article_fetcher=_article_fetch
        )
        self.assertEqual(len(articles), 1)
        self.assertEqual(run.articles_found, 1)
        self.assertIsNotNone(self.repo.get_subscription(self.sub.id).last_polled_at)
        # A second poll finds nothing new (dedupe against stored canonical/hash).
        _run2, articles2, _ = article_worker.poll_subscription(
            self.repo, self.sub, feed_fetcher=_feed_fetch, article_fetcher=_article_fetch
        )
        self.assertEqual(len(articles2), 0)

    def test_assisted_generates_without_render(self):
        _run, articles, clusters = article_worker.poll_subscription(
            self.repo, self.sub, feed_fetcher=_feed_fetch, article_fetcher=_article_fetch
        )
        settings = AutomationSettings(minimum_story_score=0.5, minimum_confidence_score=0.5)
        with patch("app.services.article_llm.llm._generate_response", side_effect=_mock_llm):
            results = article_worker.process_clusters(
                self.repo, self.sub, articles, clusters, settings, AutomationMode.assisted
            )
        self.assertEqual(results[0]["decision"], "generate")
        self.assertFalse(results[0].get("rendered"))

    def test_automated_renders_and_records_generation(self):
        _run, articles, clusters = article_worker.poll_subscription(
            self.repo, self.sub, feed_fetcher=_feed_fetch, article_fetcher=_article_fetch
        )
        settings = AutomationSettings(minimum_story_score=0.5, minimum_confidence_score=0.5)
        fake_task = MagicMock()
        with patch("app.services.article_llm.llm._generate_response", side_effect=_mock_llm), \
             patch("app.services.task.start", fake_task):
            results = article_worker.process_clusters(
                self.repo, self.sub, articles, clusters, settings, AutomationMode.automated
            )
        self.assertTrue(results[0].get("rendered"))
        self.assertFalse(results[0].get("published"))  # auto-publish disabled
        fake_task.assert_called_once()
        # Generation recorded; a re-run skips the already-generated cluster.
        self.assertTrue(self.repo.has_generated_cluster(clusters[0].id))
        with patch("app.services.article_llm.llm._generate_response", side_effect=_mock_llm):
            again = article_worker.process_clusters(
                self.repo, self.sub, articles, clusters, settings, AutomationMode.automated
            )
        self.assertEqual(again, [])

    def test_daily_generation_limit(self):
        _run, articles, clusters = article_worker.poll_subscription(
            self.repo, self.sub, feed_fetcher=_feed_fetch, article_fetcher=_article_fetch
        )
        settings = AutomationSettings(max_generations_per_day=0)
        with patch("app.services.article_llm.llm._generate_response", side_effect=_mock_llm):
            results = article_worker.process_clusters(
                self.repo, self.sub, articles, clusters, settings, AutomationMode.automated
            )
        self.assertEqual(results, [])  # limit reached before any generation

    def test_main_once_invokes_run_once(self):
        with patch.object(article_worker, "run_once", return_value=[]) as run_once:
            code = article_worker.main(["--once"])
        self.assertEqual(code, 0)
        run_once.assert_called_once()

    def test_main_autonomous_flag(self):
        captured = {}

        def fake_run_once(**kwargs):
            captured.update(kwargs)
            return []

        with patch.object(article_worker, "run_once", side_effect=fake_run_once):
            article_worker.main(["--once", "--autonomous"])
        self.assertEqual(captured["mode_override"], AutomationMode.autonomous)

    def test_main_handles_keyboard_interrupt(self):
        with patch.object(article_worker, "run_once", side_effect=KeyboardInterrupt):
            code = article_worker.main(["--interval", "1"])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
