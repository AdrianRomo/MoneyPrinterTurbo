"""Article Mode API endpoints (v1).

Subscriptions CRUD, manual polling, article listing, AI scoring and generation.
All request fields are validated with Pydantic; RSS URLs are SSRF-checked so a
subscription can never be used to reach internal-network addresses.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import Path, Query, Request
from loguru import logger
from pydantic import BaseModel, Field

from app.config import config
from app.controllers import base
from app.controllers.v1.base import new_router
from app.models.article import MediaMode, TopicSubscription
from app.models.exception import HttpException
from app.services import article_ingestion, article_pipeline, article_worker
from app.services.article_repository import get_repository
from app.utils import utils

router = new_router()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class SubscriptionRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    query: str = Field(default="", max_length=500)
    language: str = Field(default="", max_length=32)
    rss_urls: List[str] = Field(default_factory=list)
    trusted_domains: List[str] = Field(default_factory=list)
    blocked_domains: List[str] = Field(default_factory=list)
    freshness_hours: int = Field(default=72, ge=1, le=8760)
    minimum_independent_sources: int = Field(default=1, ge=1, le=10)
    poll_interval_minutes: int = Field(default=60, ge=5, le=10080)
    audience: str = Field(default="", max_length=200)
    tone: str = Field(default="", max_length=200)
    platform: str = Field(default="tiktok", max_length=64)
    brand_preset: str = Field(default="", max_length=200)
    sensitive: bool = False
    automation_mode: Optional[str] = None
    enabled: bool = True


class GenerateRequest(BaseModel):
    media_mode: str = Field(default="images_only", max_length=32)
    image_source: str = Field(default="pexels", max_length=32)
    video_aspect: str = Field(default="9:16", max_length=16)
    voice_name: str = Field(default="", max_length=128)
    render: bool = False  # if false, only prepare the script (assisted mode)


class IngestUrlRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2000)
    subscription_id: Optional[str] = None


def _cfg_int(key: str, default: int) -> int:
    try:
        return int(config.app.get(key, default))
    except (TypeError, ValueError):
        return default


def _cfg_list(key: str) -> List[str]:
    value = config.app.get(key, [])
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(item).strip() for item in value or [] if str(item).strip()]


def _subscription_from_request(body: SubscriptionRequest) -> TopicSubscription:
    payload = body.model_dump()
    fields_set = body.model_fields_set
    if "freshness_hours" not in fields_set:
        payload["freshness_hours"] = _cfg_int("article_default_freshness_hours", 72)
    if "minimum_independent_sources" not in fields_set:
        payload["minimum_independent_sources"] = _cfg_int(
            "article_default_min_independent_sources", 1
        )
    if "poll_interval_minutes" not in fields_set:
        payload["poll_interval_minutes"] = _cfg_int("article_poll_interval_minutes", 60)
    if "trusted_domains" not in fields_set:
        payload["trusted_domains"] = _cfg_list("article_trusted_domains")
    if "blocked_domains" not in fields_set:
        payload["blocked_domains"] = _cfg_list("article_blocked_domains")
    return TopicSubscription(**payload)


def _validate_rss_urls(request_id: str, urls: List[str]) -> List[str]:
    """Reject non-http(s)/internal URLs before persisting a subscription."""
    validated: List[str] = []
    for url in urls:
        try:
            validated.append(article_ingestion.validate_url(url))
        except article_ingestion.SecurityError as exc:
            raise HttpException(
                task_id=request_id,
                status_code=400,
                message=f"{request_id}: invalid rss url ({exc})",
            )
    return validated


# ---------------------------------------------------------------------------
# Subscriptions
# ---------------------------------------------------------------------------


@router.post("/article-subscriptions", summary="Create a topic subscription")
async def create_subscription(request: Request, body: SubscriptionRequest):
    request_id = base.get_task_id(request)
    rss_urls = _validate_rss_urls(request_id, body.rss_urls)
    subscription = _subscription_from_request(body).model_copy(
        update={"rss_urls": rss_urls}
    )
    get_repository().upsert_subscription(subscription)
    return utils.get_response(200, subscription.model_dump(mode="json"))


@router.get("/article-subscriptions", summary="List topic subscriptions")
async def list_subscriptions(request: Request, enabled_only: bool = Query(False)):
    subscriptions = get_repository().list_subscriptions(enabled_only=enabled_only)
    return utils.get_response(
        200, {"subscriptions": [s.model_dump(mode="json") for s in subscriptions]}
    )


@router.get("/article-subscriptions/{subscription_id}", summary="Get a subscription")
async def get_subscription(request: Request, subscription_id: str = Path(...)):
    request_id = base.get_task_id(request)
    subscription = get_repository().get_subscription(subscription_id)
    if not subscription:
        raise HttpException(subscription_id, 404, f"{request_id}: subscription not found")
    return utils.get_response(200, subscription.model_dump(mode="json"))


@router.put("/article-subscriptions/{subscription_id}", summary="Update a subscription")
async def update_subscription(
    request: Request, body: SubscriptionRequest, subscription_id: str = Path(...)
):
    request_id = base.get_task_id(request)
    repo = get_repository()
    existing = repo.get_subscription(subscription_id)
    if not existing:
        raise HttpException(subscription_id, 404, f"{request_id}: subscription not found")
    rss_urls = _validate_rss_urls(request_id, body.rss_urls)
    updated = existing.model_copy(
        update={**body.model_dump(), "rss_urls": rss_urls, "id": subscription_id}
    )
    repo.upsert_subscription(updated)
    return utils.get_response(200, updated.model_dump(mode="json"))


@router.delete("/article-subscriptions/{subscription_id}", summary="Delete a subscription")
async def delete_subscription(request: Request, subscription_id: str = Path(...)):
    request_id = base.get_task_id(request)
    if not get_repository().delete_subscription(subscription_id):
        raise HttpException(subscription_id, 404, f"{request_id}: subscription not found")
    return utils.get_response(200)


@router.post("/article-subscriptions/{subscription_id}/poll", summary="Poll a subscription now")
async def poll_subscription(request: Request, subscription_id: str = Path(...)):
    request_id = base.get_task_id(request)
    repo = get_repository()
    subscription = repo.get_subscription(subscription_id)
    if not subscription:
        raise HttpException(subscription_id, 404, f"{request_id}: subscription not found")
    run, articles, clusters = article_worker.poll_subscription(repo, subscription)
    return utils.get_response(
        200,
        {
            "poll": run.model_dump(mode="json"),
            "articles": [a.public_dict() for a in articles],
            "clusters": [c.model_dump(mode="json") for c in clusters],
        },
    )


@router.get("/article-subscriptions/{subscription_id}/poll-runs", summary="List poll runs")
async def list_poll_runs(
    request: Request,
    subscription_id: str = Path(...),
    limit: int = Query(20, ge=1, le=100),
):
    request_id = base.get_task_id(request)
    repo = get_repository()
    if not repo.get_subscription(subscription_id):
        raise HttpException(subscription_id, 404, f"{request_id}: subscription not found")
    runs = repo.list_poll_runs(subscription_id=subscription_id, limit=limit)
    return utils.get_response(
        200, {"poll_runs": [run.model_dump(mode="json") for run in runs]}
    )


# ---------------------------------------------------------------------------
# Articles
# ---------------------------------------------------------------------------


@router.get("/articles", summary="List article candidates")
async def list_articles(
    request: Request,
    subscription_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    domain: Optional[str] = Query(None),
    cluster_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
):
    articles = get_repository().list_articles(
        subscription_id=subscription_id,
        status=status,
        domain=domain,
        cluster_id=cluster_id,
        limit=limit,
    )
    return utils.get_response(200, {"articles": [a.public_dict() for a in articles]})


@router.post("/articles/ingest-url", summary="Ingest a direct article URL")
async def ingest_article_url(request: Request, body: IngestUrlRequest):
    request_id = base.get_task_id(request)
    repo = get_repository()
    subscription = repo.get_subscription(body.subscription_id) if body.subscription_id else None
    try:
        article = article_ingestion.ingest_url(body.url, subscription=subscription)
    except article_ingestion.SecurityError as exc:
        raise HttpException(request_id, 400, f"{request_id}: {exc}")
    except Exception as exc:
        logger.warning(f"failed to ingest article url: {exc}")
        raise HttpException(request_id, 502, f"{request_id}: {exc}")

    existing = (
        repo.find_article_by_canonical(article.canonical_url)
        or repo.find_article_by_hash(article.text_hash)
    )
    if existing:
        return utils.get_response(200, {"article": existing.public_dict(), "duplicate": True})
    repo.save_article(article)
    return utils.get_response(200, {"article": article.public_dict(), "duplicate": False})


@router.get("/article-clusters", summary="List article clusters")
async def list_clusters(
    request: Request,
    subscription_id: Optional[str] = Query(None),
):
    clusters = get_repository().list_clusters(subscription_id=subscription_id)
    return utils.get_response(
        200, {"clusters": [cluster.model_dump(mode="json") for cluster in clusters]}
    )


@router.get("/articles/{article_id}", summary="Get an article candidate")
async def get_article(request: Request, article_id: str = Path(...)):
    request_id = base.get_task_id(request)
    article = get_repository().get_article(article_id)
    if not article:
        raise HttpException(article_id, 404, f"{request_id}: article not found")
    return utils.get_response(200, article.public_dict())


@router.post("/articles/{article_id}/assess", summary="AI-score an article's story")
async def assess_article(request: Request, article_id: str = Path(...)):
    request_id = base.get_task_id(request)
    repo = get_repository()
    article = repo.get_article(article_id)
    if not article:
        raise HttpException(article_id, 404, f"{request_id}: article not found")
    from app.services import article_llm

    cluster_articles = (
        repo.list_articles(cluster_id=article.cluster_id) if article.cluster_id else [article]
    ) or [article]
    assessment = article_llm.assess_story(cluster_articles, query=article.title)
    if article.cluster_id:
        repo.save_assessment(article.cluster_id, assessment)
    return utils.get_response(200, assessment.public_dict())


@router.post("/articles/{article_id}/generate", summary="Generate a video from an article")
async def generate_article(request: Request, body: GenerateRequest, article_id: str = Path(...)):
    request_id = base.get_task_id(request)
    repo = get_repository()
    article = repo.get_article(article_id)
    if not article:
        raise HttpException(article_id, 404, f"{request_id}: article not found")
    try:
        media_mode = MediaMode(body.media_mode)
    except ValueError:
        raise HttpException(article_id, 400, f"{request_id}: invalid media_mode")

    settings = article_pipeline.load_automation_settings()
    cluster_articles = (
        repo.list_articles(cluster_id=article.cluster_id) if article.cluster_id else [article]
    ) or [article]
    subscription = (
        repo.get_subscription(article.subscription_id) if article.subscription_id else None
    )
    outcome = article_pipeline.assess_and_generate(
        cluster_articles,
        settings=settings,
        subscription=subscription,
        media_mode=media_mode,
    )
    if outcome["decision"] != "generate" or not outcome["script"]:
        return utils.get_response(
            200,
            {
                "decision": "skip",
                "reason": outcome.get("reason"),
                "assessment": outcome["assessment"].public_dict(),
            },
        )

    script = outcome["script"]
    if not body.render:
        # Assisted mode: return the prepared script for review.
        return utils.get_response(
            200,
            {
                "decision": "generate",
                "rendered": False,
                "assessment": outcome["assessment"].public_dict(),
                "script": script.model_dump(mode="json"),
            },
        )

    # Automated/manual render: enqueue a render task via the standard pipeline.
    from app.controllers.v1 import video as video_controller

    params = article_pipeline.build_render_params(
        script,
        media_mode=media_mode,
        image_source=body.image_source,
        video_aspect=body.video_aspect,
        voice_name=body.voice_name,
    )
    task_id = utils.get_uuid()
    from app.services import state as sm
    from app.services import task as tm

    sm.state.update_task(task_id)
    try:
        video_controller.task_manager.add_task(
            tm.start, task_id=task_id, params=params, stop_at="video"
        )
    except Exception as exc:
        sm.state.delete_task(task_id)
        logger.warning(f"failed to enqueue article render: {exc}")
        raise HttpException(article_id, 429, f"{request_id}: {exc}")
    repo.record_generation(article.cluster_id or article.id, task_id)
    return utils.get_response(
        200,
        {
            "decision": "generate",
            "rendered": True,
            "task_id": task_id,
            "assessment": outcome["assessment"].public_dict(),
            "script_title": script.title,
        },
    )
