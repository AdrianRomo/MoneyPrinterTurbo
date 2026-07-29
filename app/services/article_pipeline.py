"""Automation-first orchestration for Article Mode.

This ties the pieces together end-to-end with as much or as little autonomy as
configured:

    ingest → AI assess → threshold gate → generate script → AI review/rewrite
    → build visual timeline → render → (optionally) publish

Human review is optional. Thresholds in :class:`AutomationSettings` decide how
far a story proceeds. Editorial judgement is the LLM's job; the deterministic
checks here are only technical/safety (valid audio/video durations, valid
paths, secret-free artifacts).
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

from loguru import logger

from app.config import config
from app.models import const
from app.models.article import (
    ArticleRecord,
    AutomationMode,
    AutomationSettings,
    GeneratedScript,
    MediaAsset,
    MediaMode,
    RecommendedAction,
    RiskLevel,
    StoryAssessment,
    TopicSubscription,
    safe_public_page,
)
from app.models.schema import VideoAspect, VideoParams
from app.services import article_artifacts, article_ingestion, article_llm, material, video, voice
from app.services import state as sm
from app.utils import utils

_MIN_SCENE_SECONDS = 1.6
_DEFAULT_TARGET_SECONDS = 45


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _cfg(key: str, default):
    value = config.app.get(key, default)
    return default if value is None else value


def load_automation_settings() -> AutomationSettings:
    """Read the typed automation thresholds from the ``[app]`` config section."""

    def as_float(key, default):
        try:
            return float(_cfg(key, default))
        except (TypeError, ValueError):
            return default

    def as_bool(key, default):
        value = _cfg(key, default)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def as_int(key, default):
        try:
            return int(_cfg(key, default))
        except (TypeError, ValueError):
            return default

    try:
        mode = AutomationMode(str(_cfg("article_automation_mode", "assisted")).lower())
    except ValueError:
        mode = AutomationMode.assisted
    try:
        max_risk = RiskLevel(
            str(_cfg("article_maximum_risk_for_auto_publish", "low")).lower()
        )
    except ValueError:
        max_risk = RiskLevel.low

    return AutomationSettings(
        mode=mode,
        minimum_story_score=as_float("article_minimum_story_score", 0.6),
        minimum_confidence_score=as_float("article_minimum_confidence_score", 0.6),
        minimum_visual_score=as_float("article_minimum_visual_score", 0.4),
        maximum_risk_for_auto_publish=max_risk,
        auto_generate_enabled=as_bool("article_auto_generate_enabled", True),
        auto_render_enabled=as_bool("article_auto_render_enabled", False),
        auto_publish_enabled=as_bool("article_auto_publish_enabled", False),
        auto_rewrite_attempts=as_int("article_auto_rewrite_attempts", 1),
        allow_single_source_stories=as_bool("article_allow_single_source_stories", True),
        allow_unverified_developing_stories=as_bool(
            "article_allow_unverified_developing_stories", True
        ),
        require_review_for_sensitive_topics=as_bool(
            "article_require_review_for_sensitive_topics", True
        ),
        add_illustrative_label=as_bool("article_add_illustrative_label", True),
        max_generations_per_day=as_int("article_max_generations_per_day", 20),
        max_publications_per_day=as_int("article_max_publications_per_day", 10),
    )


# ---------------------------------------------------------------------------
# Gate decisions
# ---------------------------------------------------------------------------

_RISK_ORDER = {RiskLevel.low: 0, RiskLevel.medium: 1, RiskLevel.high: 2}


def should_generate(
    assessment: StoryAssessment,
    settings: AutomationSettings,
    sensitive: bool = False,
    *,
    independent_sources: int = 1,
    minimum_independent_sources: int = 1,
    has_unverified_dates: bool = False,
    has_fresh_source: bool = True,
) -> Tuple[bool, str]:
    """Decide whether a story clears the *generation* thresholds."""
    if assessment.recommended_action == RecommendedAction.skip:
        return False, "LLM recommended skipping this story"
    if assessment.story_score < settings.minimum_story_score:
        return False, (
            f"story_score {assessment.story_score:.2f} < "
            f"{settings.minimum_story_score:.2f}"
        )
    if assessment.confidence < settings.minimum_confidence_score:
        return False, (
            f"confidence {assessment.confidence:.2f} < "
            f"{settings.minimum_confidence_score:.2f}"
        )
    if assessment.visual_potential < settings.minimum_visual_score:
        return False, (
            f"visual_potential {assessment.visual_potential:.2f} < "
            f"{settings.minimum_visual_score:.2f}"
        )
    if (
        not settings.allow_single_source_stories
        and independent_sources < minimum_independent_sources
    ):
        return False, (
            f"independent_sources {independent_sources} < "
            f"{minimum_independent_sources}"
        )
    if not settings.allow_unverified_developing_stories and has_unverified_dates:
        return False, "story has sources without publication dates"
    if not has_fresh_source:
        return False, "story has no source inside the freshness window"
    if sensitive and settings.require_review_for_sensitive_topics:
        # Sensitive stories may still generate; the review requirement only
        # blocks *auto-publish* (handled in should_publish).
        return True, "sensitive topic: generation allowed, review required before publish"
    return True, "clears generation thresholds"


def should_render(settings: AutomationSettings, mode: AutomationMode) -> bool:
    if mode == AutomationMode.assisted:
        return False
    return settings.auto_render_enabled or mode in (
        AutomationMode.automated,
        AutomationMode.autonomous,
    )


def should_publish(
    assessment: StoryAssessment,
    settings: AutomationSettings,
    mode: AutomationMode,
    sensitive: bool = False,
) -> Tuple[bool, str]:
    if mode != AutomationMode.autonomous:
        return False, "auto-publish only in autonomous mode"
    if not settings.auto_publish_enabled:
        return False, "auto_publish_enabled is false"
    if sensitive and settings.require_review_for_sensitive_topics:
        return False, "sensitive topic requires human approval before publishing"
    if _RISK_ORDER[assessment.risk_level] > _RISK_ORDER[settings.maximum_risk_for_auto_publish]:
        return False, (
            f"risk {assessment.risk_level.value} exceeds max "
            f"{settings.maximum_risk_for_auto_publish.value}"
        )
    return True, "clears publish thresholds"


# ---------------------------------------------------------------------------
# Visual timeline
# ---------------------------------------------------------------------------


def scene_durations(
    weights: List[float], audio_duration: float, minimum: float = _MIN_SCENE_SECONDS
) -> List[float]:
    """Split ``audio_duration`` across scenes proportional to their weights,
    with a per-scene minimum, so the visual timeline always covers the
    voice-over and no zero-length clip is ever produced."""
    if not weights:
        return []
    safe_weights = [max(0.01, float(w)) for w in weights]
    total_weight = sum(safe_weights)
    target = max(audio_duration, minimum * len(weights))
    durations = [max(minimum, target * (w / total_weight)) for w in safe_weights]
    # Ensure the sum covers the audio (add a small margin to the last scene).
    shortfall = (audio_duration + 0.2) - sum(durations)
    if shortfall > 0:
        durations[-1] += shortfall
    return [round(d, 3) for d in durations]


def build_visual_timeline(
    task_id: str,
    script: GeneratedScript,
    audio_duration: float,
    *,
    video_aspect: VideoAspect,
    provider: str,
    settings: AutomationSettings,
    entities: Optional[List[str]] = None,
    media_mode: MediaMode = MediaMode.images_only,
    searcher=None,
    downloader=None,
) -> Tuple[List[str], List[MediaAsset]]:
    """Select, download and normalize one visual per scene (in order).

    Returns ``(clip_paths, chosen_assets)``. A scene that yields no usable asset
    falls back to a branded background so the render never blocks. ``searcher``
    and ``downloader`` are injectable for tests."""
    task_directory = utils.task_dir(task_id)
    chosen = material.select_scene_assets(
        provider,
        script.scenes,
        video_aspect=video_aspect,
        entities=entities,
        media_mode=media_mode.value,
        searcher=searcher,
    )
    assets_by_index = {asset.beat_index: asset for asset in chosen}
    durations = scene_durations(
        [scene.duration_weight for scene in script.scenes], audio_duration
    )

    clip_paths: List[str] = []
    kept_assets: List[MediaAsset] = []
    for index, scene in enumerate(script.scenes):
        clip_path = os.path.join(task_directory, f"article-scene-{index + 1}.mp4")
        duration = durations[index] if index < len(durations) else _MIN_SCENE_SECONDS
        asset = assets_by_index.get(index)
        rendered = False
        if asset is not None:
            if downloader:
                try:
                    local_path = downloader(asset.url, asset)
                except TypeError:
                    local_path = downloader(asset.url)
            elif asset.media_type == "video":
                local_path = material.save_video(asset.url)
            else:
                local_path = material.save_image(asset.url)
            if local_path:
                try:
                    if asset.media_type == "video":
                        video.video_to_normalized_clip(
                            local_path, clip_path, duration, video_aspect
                        )
                    else:
                        video.image_to_video_clip(
                            local_path, clip_path, duration, video_aspect
                        )
                    asset.local_path = clip_path
                    asset.duration = duration
                    asset.beat_index = index
                    if settings.add_illustrative_label:
                        asset.illustrative = bool(scene.is_contextual_visual)
                    kept_assets.append(asset)
                    rendered = True
                except Exception as exc:
                    logger.warning(f"failed to normalize asset for scene {index + 1}: {exc}")
        if not rendered:
            video.color_background_clip(clip_path, duration, video_aspect)
        clip_paths.append(clip_path)
    return clip_paths, kept_assets


# ---------------------------------------------------------------------------
# Script resolution
# ---------------------------------------------------------------------------


def resolve_script(params: VideoParams) -> GeneratedScript:
    """Obtain a :class:`GeneratedScript` for a render request.

    Order of preference:
      1. ``params.article_script`` (already produced by the pipeline/worker).
      2. ``params.article_url`` — ingest + assess + generate on the fly.
    Raises ValueError only for hard technical problems (nothing to work with)."""
    if params.article_script:
        return GeneratedScript.model_validate(params.article_script)

    settings = load_automation_settings()
    if params.article_id:
        from app.services.article_repository import get_repository

        repo = get_repository()
        article = repo.get_article(params.article_id)
        if not article:
            raise ValueError(f"article not found: {params.article_id}")
        cluster_articles = (
            repo.list_articles(cluster_id=article.cluster_id)
            if article.cluster_id
            else [article]
        ) or [article]
        subscription = (
            repo.get_subscription(article.subscription_id)
            if article.subscription_id
            else None
        )
        outcome = assess_and_generate(
            cluster_articles,
            settings=settings,
            subscription=subscription,
            media_mode=_media_mode(params),
        )
        if outcome["decision"] != "generate" or not outcome["script"]:
            raise ValueError(f"article did not clear thresholds: {outcome.get('reason')}")
        return outcome["script"]

    if params.article_url:
        article = article_ingestion.ingest_url(params.article_url)
        from app.services.article_repository import get_repository

        repo = get_repository()
        existing = (
            repo.find_article_by_canonical(article.canonical_url)
            or repo.find_article_by_hash(article.text_hash)
        )
        if existing:
            article = existing
        else:
            from app.models.article import ArticleCluster, ArticleStatus

            cluster = ArticleCluster(
                subscription_id=article.subscription_id,
                normalized_title=article_ingestion.normalize_title(article.title),
                article_ids=[article.id],
                domains=[article.domain] if article.domain else [],
            )
            article.cluster_id = cluster.id
            article.status = ArticleStatus.clustered
            repo.save_cluster(cluster)
            repo.save_article(article)
        outcome = assess_and_generate(
            [article], settings=settings, media_mode=_media_mode(params)
        )
        if outcome["decision"] != "generate" or not outcome["script"]:
            raise ValueError(f"article did not clear thresholds: {outcome.get('reason')}")
        return outcome["script"]

    raise ValueError("article mode requires article_script, article_id or article_url")


def _media_mode(params: VideoParams) -> MediaMode:
    try:
        return MediaMode(str(params.media_mode or "images_only"))
    except ValueError:
        return MediaMode.images_only


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def render_article_video(task_id: str, params: VideoParams, stop_at: str = "video") -> dict:
    """Render an article video, reusing the existing TTS/subtitle/compositor.

    Only the *materials* step differs from the topic pipeline: instead of
    downloading stock videos by keyword, it builds an ordered per-scene visual
    timeline (images and/or videos) from the grounded script."""
    # Imported lazily to avoid an import cycle (task imports this module).
    from app.services import task as task_service

    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=5)

    try:
        script = resolve_script(params)
    except Exception as exc:
        return task_service._mark_task_failed(task_id, "research", str(exc))

    narration = script.narration_text()
    if not narration.strip():
        return task_service._mark_task_failed(
            task_id, "research", "grounded script produced no narration"
        )

    # Feed the narration into the shared audio/subtitle path.
    params.video_script = narration
    sm.state.update_task(task_id, progress=15)

    audio_file, audio_duration, sub_maker = task_service.generate_audio(
        task_id, params, narration
    )
    if not audio_file:
        return task_service._mark_task_failed(
            task_id, "audio", "failed to prepare narration audio"
        )
    sm.state.update_task(task_id, progress=35)

    subtitle_path = task_service.generate_subtitle(
        task_id, params, narration, sub_maker, audio_file
    )

    provider = (params.image_source or "pexels").strip().lower()
    aspect = VideoAspect(params.video_aspect)
    settings = load_automation_settings()
    media_mode = _media_mode(params)
    entities: List[str] = []

    clip_paths, chosen_assets = build_visual_timeline(
        task_id,
        script,
        audio_duration,
        video_aspect=aspect,
        provider=provider,
        settings=settings,
        entities=entities,
        media_mode=media_mode,
    )
    if not clip_paths:
        return task_service._mark_task_failed(
            task_id, "materials", "failed to build article visual timeline"
        )
    sm.state.update_task(task_id, progress=55)

    task_directory = utils.task_dir(task_id)
    combined_path = os.path.join(task_directory, "combined-1.mp4")
    try:
        video.combine_article_clips(
            clip_paths, combined_path, audio_duration, threads=params.n_threads or 2
        )
    except Exception as exc:
        return task_service._mark_task_failed(task_id, "video", f"failed to combine clips: {exc}")
    sm.state.update_task(task_id, progress=75)

    final_path = os.path.join(task_directory, "final-1.mp4")
    try:
        video.generate_video(
            video_path=combined_path,
            audio_path=audio_file,
            subtitle_path=subtitle_path,
            output_file=final_path,
            params=params,
        )
    except Exception as exc:
        return task_service._mark_task_failed(task_id, "video", f"failed to render video: {exc}")

    # Persist provenance artifacts (secret-free).
    article_artifacts.write_all(
        task_id,
        script=script,
        assessment=script.assessment,
        assets=chosen_assets,
    )

    source_names = script.source_names()
    result = {
        "videos": [final_path],
        "combined_videos": [combined_path],
        "script": narration,
        "audio_file": audio_file,
        "audio_duration": audio_duration,
        "subtitle_path": subtitle_path,
        "content_mode": params.content_mode,
        "media_mode": params.media_mode,
        "article_title": script.title,
        "article_sources": source_names,
        "article_confidence": script.confidence,
        "article_uncertainties": script.uncertainties,
        "media_attribution": [a.attribution_text for a in chosen_assets if a.attribution_text],
        "provenance_label": "Generated from the listed sources.",
    }
    sm.state.update_task(
        task_id, state=const.TASK_STATE_COMPLETE, progress=100, **result
    )
    logger.success(
        f"article video finished, task_id: {task_id}, sources: {', '.join(source_names)}"
    )
    return result


# ---------------------------------------------------------------------------
# Story → script (no render): used by the worker / assisted mode
# ---------------------------------------------------------------------------


def assess_and_generate(
    articles: List[ArticleRecord],
    *,
    settings: AutomationSettings,
    subscription: Optional[TopicSubscription] = None,
    media_mode: MediaMode = MediaMode.images_only,
) -> dict:
    """Run assess → gate → generate → review for a cluster of articles.

    Returns a dict with ``assessment``, ``decision`` and (when generated)
    ``script``. Never raises for editorial reasons — a below-threshold story is
    reported as skipped, not failed."""
    language = subscription.language if subscription else ""
    query = subscription.query if subscription else ""
    audience = subscription.audience if subscription else ""
    tone = subscription.tone if subscription else ""
    platform = subscription.platform if subscription else "tiktok"
    sensitive = bool(subscription.sensitive) if subscription else False

    assessment = article_llm.assess_story(
        articles, query=query, audience=audience, language=language
    )
    sensitive = sensitive or bool(assessment.sensitive_categories)
    independent_sources = article_ingestion.independent_domain_count(articles)
    minimum_sources = subscription.minimum_independent_sources if subscription else 1
    freshness_results = [
        article_ingestion.article_is_fresh(
            article, subscription.freshness_hours if subscription else 10 * 365 * 24
        )
        for article in articles
    ]
    has_unverified_dates = any(result is None for result in freshness_results)
    known_freshness = [result for result in freshness_results if result is not None]
    has_fresh_source = any(known_freshness) if known_freshness else True
    ok, reason = should_generate(
        assessment,
        settings,
        sensitive=sensitive,
        independent_sources=independent_sources,
        minimum_independent_sources=minimum_sources,
        has_unverified_dates=has_unverified_dates,
        has_fresh_source=has_fresh_source,
    )
    if not ok:
        return {"assessment": assessment, "decision": "skip", "reason": reason, "script": None}

    script = article_llm.generate_reviewed_script(
        articles,
        auto_rewrite_attempts=settings.auto_rewrite_attempts,
        minimum_confidence=settings.minimum_confidence_score,
        language=language,
        tone=tone,
        audience=audience,
        platform=platform,
        media_mode=media_mode,
        brand_preset=(subscription.brand_preset if subscription else ""),
        assessment=assessment,
    )
    script.assessment = assessment
    return {
        "assessment": assessment,
        "decision": "generate",
        "reason": reason,
        "script": script,
        "sensitive": sensitive,
    }


def build_render_params(
    script: GeneratedScript,
    *,
    base: Optional[VideoParams] = None,
    media_mode: MediaMode = MediaMode.images_only,
    image_source: str = "pexels",
    video_aspect: str = "9:16",
    voice_name: str = "",
) -> VideoParams:
    """Create a VideoParams that renders ``script`` through the article path."""
    params = base or VideoParams(video_subject=script.title or "article")
    params.content_mode = "article_feed"
    params.media_mode = media_mode.value
    params.image_source = image_source
    params.video_aspect = video_aspect
    params.video_script = script.narration_text()
    params.article_script = script.model_dump(mode="json")
    configured_voice = (
        voice_name
        or str(config.app.get("article_voice_name", "") or "").strip()
        or str(config.ui.get("voice_name", "") or "").strip()
        or voice.NO_VOICE_NAME
    )
    params.voice_name = configured_voice
    return params


def provenance_summary(script: GeneratedScript, assets: List[MediaAsset]) -> dict:
    """Human-inspectable, secret-free provenance for API/task details."""
    return {
        "title": script.title,
        "confidence": script.confidence,
        "sources": [
            {
                "publisher": s.publisher,
                "title": s.title,
                "canonical_url": safe_public_page(s.canonical_url),
                "published_at": s.published_at.isoformat() if s.published_at else None,
            }
            for s in script.sources
        ],
        "uncertainties": script.uncertainties,
        "media": [a.manifest_entry() for a in assets],
        "label": "Generated from the listed sources.",
    }
