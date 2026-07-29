"""Thin service helpers for the Streamlit Article Mode UI.

The Streamlit page should stay a view/controller layer: forms collect input,
then call these helpers for repository access, filtering, and render task
submission. None of these helpers performs direct widget work.
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Iterable, Optional

from app.models.article import (
    ArticleRecord,
    GeneratedScript,
    MediaMode,
    PollRun,
    Scene,
    TopicSubscription,
)
from app.services import article_ingestion, article_pipeline, article_worker, webui_task
from app.services.article_repository import ArticleRepository, get_repository
from app.services import state as sm
from app.utils import utils


def lines_from_text(value: str) -> list[str]:
    """Normalize newline/comma separated form text into unique non-empty lines."""
    seen: set[str] = set()
    result: list[str] = []
    for raw in str(value or "").replace(",", "\n").splitlines():
        item = raw.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def repository() -> ArticleRepository:
    return get_repository()


def format_poll_summary(run: Optional[PollRun]) -> dict:
    if run is None:
        return {
            "last_poll_time": "-",
            "last_poll_result": "Never polled",
            "errors": [],
        }
    finished = run.finished_at or run.started_at
    return {
        "last_poll_time": finished.astimezone(timezone.utc).strftime(
            "%Y-%m-%d %H:%M UTC"
        ),
        "last_poll_result": (
            f"{run.articles_accepted} accepted, "
            f"{run.articles_rejected} rejected, {len(run.errors)} errors"
        ),
        "errors": run.errors,
    }


def latest_poll_run(
    repo: ArticleRepository, subscription_id: str
) -> Optional[PollRun]:
    runs = repo.list_poll_runs(subscription_id=subscription_id, limit=1)
    return runs[0] if runs else None


def subscription_rows(repo: Optional[ArticleRepository] = None) -> list[dict]:
    repo = repo or get_repository()
    rows: list[dict] = []
    for sub in repo.list_subscriptions():
        latest = latest_poll_run(repo, sub.id)
        rows.append(
            {
                "subscription": sub,
                "poll": latest,
                "summary": format_poll_summary(latest),
            }
        )
    return rows


def build_subscription(
    *,
    existing: Optional[TopicSubscription] = None,
    name: str,
    query: str = "",
    language: str = "",
    rss_urls: Iterable[str] = (),
    trusted_domains: Iterable[str] = (),
    blocked_domains: Iterable[str] = (),
    freshness_hours: int = 72,
    poll_interval_minutes: int = 60,
    minimum_independent_sources: int = 1,
    automation_mode: str = "",
    enabled: bool = True,
) -> TopicSubscription:
    if not name.strip():
        raise ValueError("Subscription name is required")
    payload = {
        "name": name.strip(),
        "query": query.strip(),
        "language": language.strip(),
        "rss_urls": list(rss_urls),
        "trusted_domains": list(trusted_domains),
        "blocked_domains": list(blocked_domains),
        "freshness_hours": int(freshness_hours),
        "poll_interval_minutes": int(poll_interval_minutes),
        "minimum_independent_sources": int(minimum_independent_sources),
        "automation_mode": automation_mode or None,
        "enabled": bool(enabled),
    }
    if existing is not None:
        return existing.model_copy(update=payload)
    return TopicSubscription(**payload)


def save_subscription(
    repo: ArticleRepository,
    subscription: TopicSubscription,
) -> TopicSubscription:
    subscription.rss_urls = [
        article_ingestion.validate_url(url) for url in subscription.rss_urls
    ]
    return repo.upsert_subscription(subscription)


def poll_now(repo: ArticleRepository, subscription_id: str) -> tuple[PollRun, int, int]:
    subscription = repo.get_subscription(subscription_id)
    if subscription is None:
        raise ValueError("Subscription not found")
    run, articles, clusters = article_worker.poll_subscription(repo, subscription)
    return run, len(articles), len(clusters)


def ingest_direct_url(
    repo: ArticleRepository,
    url: str,
    subscription_id: str = "",
) -> tuple[ArticleRecord, bool]:
    subscription = repo.get_subscription(subscription_id) if subscription_id else None
    article = article_ingestion.ingest_url(url, subscription=subscription)
    existing = (
        repo.find_article_by_canonical(article.canonical_url)
        or repo.find_article_by_hash(article.text_hash)
    )
    if existing:
        return existing, True
    repo.save_article(article)
    return article, False


def _date_floor(value: Optional[date]) -> Optional[datetime]:
    if value is None:
        return None
    return datetime.combine(value, time.min, tzinfo=timezone.utc)


def _date_ceiling(value: Optional[date]) -> Optional[datetime]:
    if value is None:
        return None
    return datetime.combine(value, time.max, tzinfo=timezone.utc)


def article_rows(
    repo: ArticleRepository,
    *,
    subscription_id: str = "",
    status: str = "",
    source: str = "",
    cluster_id: str = "",
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    minimum_score: float = 0.0,
    limit: int = 200,
) -> list[dict]:
    articles = repo.list_articles(
        subscription_id=subscription_id or None,
        status=status or None,
        domain=source or None,
        cluster_id=cluster_id or None,
        limit=limit,
    )
    start = _date_floor(date_from)
    end = _date_ceiling(date_to)
    rows: list[dict] = []
    clusters = {cluster.id: cluster for cluster in repo.list_clusters(subscription_id or None)}
    for article in articles:
        timestamp = article.published_at or article.retrieved_at
        if start and timestamp < start:
            continue
        if end and timestamp > end:
            continue
        assessment = repo.get_assessment(article.cluster_id) if article.cluster_id else None
        score = assessment.story_score if assessment else 0.0
        if minimum_score and score < minimum_score:
            continue
        cluster = clusters.get(article.cluster_id)
        rows.append(
            {
                "article": article,
                "cluster": cluster,
                "assessment": assessment,
                "source_count": len(cluster.article_ids) if cluster else 1,
                "story_score": score,
                "confidence": assessment.confidence if assessment else 0.0,
                "visual_score": assessment.visual_potential if assessment else 0.0,
                "risk": assessment.risk_level.value if assessment else "-",
                "uncertainties": assessment.uncertainties if assessment else [],
                "generation_status": article.status.value,
            }
        )
    return rows


def cluster_rows(
    repo: ArticleRepository, subscription_id: str = ""
) -> list[dict]:
    rows: list[dict] = []
    for cluster in repo.list_clusters(subscription_id or None):
        assessment = repo.get_assessment(cluster.id)
        rows.append(
            {
                "cluster": cluster,
                "assessment": assessment,
                "source_count": len(cluster.article_ids),
                "story_score": assessment.story_score if assessment else 0.0,
                "confidence": assessment.confidence if assessment else 0.0,
                "visual_score": assessment.visual_potential if assessment else 0.0,
                "risk": assessment.risk_level.value if assessment else "-",
            }
        )
    return rows


def assess_article(repo: ArticleRepository, article_id: str):
    article = repo.get_article(article_id)
    if article is None:
        raise ValueError("Article not found")
    from app.services import article_llm

    articles = repo.list_articles(cluster_id=article.cluster_id) if article.cluster_id else [article]
    assessment = article_llm.assess_story(articles or [article], query=article.title)
    if article.cluster_id:
        repo.save_assessment(article.cluster_id, assessment)
    return assessment


def generate_script(
    repo: ArticleRepository,
    article_id: str,
    *,
    media_mode: str = "images_only",
) -> GeneratedScript:
    article = repo.get_article(article_id)
    if article is None:
        raise ValueError("Article not found")
    articles = repo.list_articles(cluster_id=article.cluster_id) if article.cluster_id else [article]
    subscription = repo.get_subscription(article.subscription_id) if article.subscription_id else None
    settings = article_pipeline.load_automation_settings()
    outcome = article_pipeline.assess_and_generate(
        articles or [article],
        settings=settings,
        subscription=subscription,
        media_mode=MediaMode(media_mode),
    )
    if outcome["decision"] != "generate" or not outcome["script"]:
        raise ValueError(str(outcome.get("reason") or "Article did not clear thresholds"))
    if article.cluster_id:
        repo.save_assessment(article.cluster_id, outcome["assessment"])
    return outcome["script"]


def update_script_narration(script_payload: dict, narration: str) -> dict:
    script = GeneratedScript.model_validate(script_payload)
    script.narration = str(narration or "").strip()
    if not script.scenes and script.narration:
        script.scenes = [Scene(narration=script.narration)]
    return script.model_dump(mode="json")


def update_scene_queries(
    script_payload: dict, scene_index: int, queries: Iterable[str]
) -> dict:
    script = GeneratedScript.model_validate(script_payload)
    if scene_index < 0 or scene_index >= len(script.scenes):
        raise ValueError("Scene not found")
    script.scenes[scene_index].visual_queries = list(queries)[:4]
    return script.model_dump(mode="json")


def review_script_payload(
    repo: ArticleRepository, article_id: str, script_payload: dict
) -> dict:
    article = repo.get_article(article_id)
    if article is None:
        raise ValueError("Article not found")
    articles = repo.list_articles(cluster_id=article.cluster_id) if article.cluster_id else [article]
    from app.services import article_llm

    script = GeneratedScript.model_validate(script_payload)
    script.review = article_llm.review_script(script, articles or [article])
    return script.model_dump(mode="json")


def render_task_key(action: str, target_id: str) -> str:
    return f"article:{action}:{target_id}"


def submit_render_task(
    script_payload: dict,
    *,
    media_mode: str,
    image_source: str,
    video_aspect: str,
    voice_name: str = "",
) -> str:
    script = GeneratedScript.model_validate(script_payload)
    params = article_pipeline.build_render_params(
        script,
        media_mode=MediaMode(media_mode),
        image_source=image_source,
        video_aspect=video_aspect,
        voice_name=voice_name,
    )
    task_id = utils.get_uuid()
    sm.state.update_task(task_id)
    webui_task.submit_generation(task_id=task_id, params=params)
    return task_id
