import os
import tempfile
from datetime import date, datetime, timezone

import pytest

from app.models.article import (
    ArticleCluster,
    ArticleRecord,
    ArticleStatus,
    PollRun,
    StoryAssessment,
)
from app.services import article_ui
from app.services.article_repository import SqliteArticleRepository


def test_lines_from_text_splits_deduplicates_and_trims():
    assert article_ui.lines_from_text(" https://a.test/rss\nhttps://b.test/rss, https://a.test/rss ") == [
        "https://a.test/rss",
        "https://b.test/rss",
    ]


def test_format_poll_summary_reports_empty_and_errors():
    assert article_ui.format_poll_summary(None)["last_poll_result"] == "Never polled"
    run = PollRun(subscription_id="s", articles_accepted=2, articles_rejected=1, errors=["bad feed"])
    summary = article_ui.format_poll_summary(run)
    assert "2 accepted" in summary["last_poll_result"]
    assert summary["errors"] == ["bad feed"]


def test_build_subscription_validates_name_and_normalizes_lists():
    with pytest.raises(ValueError):
        article_ui.build_subscription(name=" ")
    sub = article_ui.build_subscription(
        name=" News ",
        rss_urls=["https://example.com/rss"],
        trusted_domains=["example.com"],
        enabled=False,
    )
    assert sub.name == "News"
    assert sub.rss_urls == ["https://example.com/rss"]
    assert not sub.enabled


def test_article_rows_include_assessment_and_filters():
    with tempfile.TemporaryDirectory() as temp_dir:
        repo = SqliteArticleRepository(os.path.join(temp_dir, "a.db"))
        cluster = ArticleCluster(subscription_id="s", article_ids=[], domains=["example.com"])
        repo.save_cluster(cluster)
        article = ArticleRecord(
            subscription_id="s",
            cluster_id=cluster.id,
            domain="example.com",
            canonical_url="https://example.com/a",
            title="Story",
            text_hash="h",
            published_at=datetime(2026, 7, 28, tzinfo=timezone.utc),
            status=ArticleStatus.scored,
        )
        cluster.article_ids = [article.id]
        repo.save_cluster(cluster)
        repo.save_article(article)
        repo.save_assessment(
            cluster.id,
            StoryAssessment(story_score=0.8, confidence=0.7, visual_potential=0.6),
        )

        rows = article_ui.article_rows(
            repo,
            subscription_id="s",
            date_from=date(2026, 7, 1),
            minimum_score=0.5,
        )

    assert len(rows) == 1
    assert rows[0]["source_count"] == 1
    assert rows[0]["story_score"] == 0.8
    assert rows[0]["generation_status"] == "scored"

