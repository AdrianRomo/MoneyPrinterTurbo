"""Standalone Article Mode polling/generation worker.

Runs independently of Streamlit. Depending on the configured automation mode it
polls feeds, scores stories, generates scripts and (optionally) renders and
publishes videos — pausing only on repeated technical failure or very high
AI-assessed risk.

Usage::

    python -m app.services.article_worker            # loop on the configured interval
    python -m app.services.article_worker --once     # a single poll+process pass
    python -m app.services.article_worker --autonomous   # force autonomous mode
    python -m app.services.article_worker --subscription <id>   # poll one subscription
    python -m app.services.article_worker --interval 900        # loop every 900s
"""

from __future__ import annotations

import argparse
import time
from typing import List, Optional

from loguru import logger

from app.models.article import (
    ArticleCluster,
    ArticleRecord,
    ArticleStatus,
    AutomationMode,
    AutomationSettings,
    MediaMode,
    PollRun,
    TopicSubscription,
)
from app.services import article_ingestion, article_pipeline
from app.services import article_llm
from app.services.article_repository import ArticleRepository, get_repository
from app.utils import utils


def _existing_keys(repo: ArticleRepository, subscription_id: str) -> tuple[set, set]:
    hashes: set = set()
    canonicals: set = set()
    for article in repo.list_articles(subscription_id=subscription_id, limit=500):
        if article.text_hash:
            hashes.add(article.text_hash)
        if article.canonical_url:
            canonicals.add(article.canonical_url)
    return hashes, canonicals


def poll_subscription(
    repo: ArticleRepository,
    subscription: TopicSubscription,
    *,
    feed_fetcher=None,
    article_fetcher=None,
) -> tuple[PollRun, List[ArticleRecord], List[ArticleCluster]]:
    """Poll one subscription, persist results, and return new articles/clusters.

    One failing feed or article never aborts the others; errors are recorded on
    the poll run."""
    run = PollRun(subscription_id=subscription.id, feeds_polled=len(subscription.rss_urls))
    hashes, canonicals = _existing_keys(repo, subscription.id)
    try:
        articles, clusters, errors = article_ingestion.ingest_subscription(
            subscription,
            feed_fetcher=feed_fetcher,
            article_fetcher=article_fetcher,
            existing_hashes=hashes,
            existing_canonicals=canonicals,
        )
    except Exception as exc:  # defensive: never let a poll crash the worker loop
        from app.models.article import utcnow

        logger.exception(f"subscription poll crashed: {subscription.id}")
        run.errors.append(str(exc))
        run.finished_at = utcnow()
        repo.save_poll_run(run)
        return run, [], []

    for article in articles:
        repo.save_article(article)
    for cluster in clusters:
        repo.save_cluster(cluster)

    run.articles_found = len(articles)
    run.articles_accepted = len(articles)
    run.errors = errors
    from app.models.article import utcnow

    run.finished_at = utcnow()
    repo.save_poll_run(run)
    repo.upsert_subscription(subscription)  # persists last_polled_at
    logger.info(
        f"polled {subscription.name}: {len(articles)} new articles, "
        f"{len(clusters)} clusters, {len(errors)} errors"
    )
    return run, articles, clusters


def _cluster_articles(
    articles: List[ArticleRecord], cluster: ArticleCluster
) -> List[ArticleRecord]:
    by_id = {article.id: article for article in articles}
    ordered = [by_id[a] for a in cluster.article_ids if a in by_id]
    # Prefer primary sources first, then newest.
    ordered.sort(
        key=lambda a: (
            not a.is_authoritative_primary,
            -(a.published_at.timestamp() if a.published_at else 0),
        )
    )
    return ordered


def process_clusters(
    repo: ArticleRepository,
    subscription: TopicSubscription,
    articles: List[ArticleRecord],
    clusters: List[ArticleCluster],
    settings: AutomationSettings,
    mode: AutomationMode,
) -> List[dict]:
    """Assess, generate and (per mode) render each new cluster."""
    results: List[dict] = []
    media_mode = _media_mode(subscription)
    for cluster in clusters:
        if repo.has_generated_cluster(cluster.id):
            continue
        if repo.count_generations() >= settings.max_generations_per_day:
            logger.warning("daily generation limit reached; stopping generation")
            break
        cluster_articles = _cluster_articles(articles, cluster)
        if not cluster_articles:
            continue
        if not settings.auto_generate_enabled:
            try:
                assessment = article_llm.assess_story(
                    cluster_articles,
                    query=subscription.query,
                    audience=subscription.audience,
                    language=subscription.language,
                )
                repo.save_assessment(cluster.id, assessment)
                cluster.summary = assessment.reasoning_summary
                repo.save_cluster(cluster)
                for article in cluster_articles:
                    article.status = ArticleStatus.scored
                    repo.save_article(article)
                results.append(
                    {
                        "cluster_id": cluster.id,
                        "decision": "assess_only",
                        "reason": "auto_generate_enabled is false",
                    }
                )
            except Exception as exc:
                logger.warning(f"assessment failed for cluster {cluster.id}: {exc}")
                for article in cluster_articles:
                    article.status = ArticleStatus.failed
                    article.rejection_reasons.append(str(exc))
                    repo.save_article(article)
                results.append({"cluster_id": cluster.id, "decision": "error", "error": str(exc)})
            continue
        try:
            outcome = article_pipeline.assess_and_generate(
                cluster_articles,
                settings=settings,
                subscription=subscription,
                media_mode=media_mode,
            )
        except Exception as exc:
            logger.warning(f"assessment/generation failed for cluster {cluster.id}: {exc}")
            for article in cluster_articles:
                article.status = ArticleStatus.failed
                article.rejection_reasons.append(str(exc))
                repo.save_article(article)
            results.append({"cluster_id": cluster.id, "decision": "error", "error": str(exc)})
            continue

        repo.save_assessment(cluster.id, outcome["assessment"])
        cluster.summary = outcome["assessment"].reasoning_summary
        repo.save_cluster(cluster)
        for article in cluster_articles:
            article.status = ArticleStatus.scored
            repo.save_article(article)

        if outcome["decision"] != "generate" or not outcome["script"]:
            for article in cluster_articles:
                article.status = ArticleStatus.skipped
                article.rejection_reasons.append(str(outcome.get("reason") or "below thresholds"))
                repo.save_article(article)
            results.append(
                {"cluster_id": cluster.id, "decision": "skip", "reason": outcome.get("reason")}
            )
            continue

        result = {"cluster_id": cluster.id, "decision": "generate", "rendered": False, "published": False}
        if article_pipeline.should_render(settings, mode):
            for article in cluster_articles:
                article.status = ArticleStatus.rendering
                repo.save_article(article)
            rendered = _render_cluster(
                repo,
                subscription,
                outcome,
                settings,
                mode,
                media_mode,
                cluster,
            )
            result.update(rendered)
        else:
            task_id = outcome["script"].id
            repo.record_generation(cluster.id, task_id, published=False)
            for article in cluster_articles:
                article.status = ArticleStatus.generated
                article.generated_task_ids.append(task_id)
                repo.save_article(article)
        results.append(result)
    return results


def _render_cluster(
    repo: ArticleRepository,
    subscription: TopicSubscription,
    outcome: dict,
    settings: AutomationSettings,
    mode: AutomationMode,
    media_mode: MediaMode,
    cluster: ArticleCluster,
) -> dict:
    script = outcome["script"]
    assessment = outcome["assessment"]
    task_id = utils.get_uuid()
    params = article_pipeline.build_render_params(
        script,
        media_mode=media_mode,
        image_source=str(article_pipeline.config.app.get("article_image_provider", "pexels")),
        video_aspect="9:16",
    )
    try:
        from app.services import task as task_service

        task_service.start(task_id, params, stop_at="video")
    except Exception as exc:
        logger.warning(f"render failed for cluster {cluster.id}: {exc}")
        for article_id in cluster.article_ids:
            article = repo.get_article(article_id)
            if article:
                article.status = ArticleStatus.failed
                article.generated_task_ids.append(task_id)
                article.rejection_reasons.append(str(exc))
                repo.save_article(article)
        return {"rendered": False, "task_id": task_id, "error": str(exc)}

    repo.record_generation(cluster.id, task_id, published=False)
    for article_id in cluster.article_ids:
        article = repo.get_article(article_id)
        if article:
            article.status = ArticleStatus.rendered
            article.generated_task_ids.append(task_id)
            repo.save_article(article)

    published = False
    publish_result: dict = {}
    can_publish, publish_reason = article_pipeline.should_publish(
        assessment, settings, mode, sensitive=bool(outcome.get("sensitive"))
    )
    if can_publish and repo.count_publications() < settings.max_publications_per_day:
        publish_result = _publish(task_id, script, cluster)
        published = bool(publish_result.get("success"))
        if published:
            provider = publish_result.get("provider") or "publisher"
            publish_id = publish_result.get("post_id") or publish_result.get("request_id") or "ok"
            repo.record_generation(
                cluster.id,
                f"{task_id}:publish:{provider}:{publish_id}",
                published=True,
            )
        else:
            logger.warning(
                f"auto-publish failed for cluster {cluster.id}: "
                f"{publish_result.get('error', 'unknown error')}"
            )
    else:
        logger.info(f"not auto-publishing cluster {cluster.id}: {publish_reason}")

    result = {"rendered": True, "task_id": task_id, "published": published}
    if publish_result:
        result["publish_result"] = publish_result
    return result


def _with_series_line(caption: str, part: Optional[dict]) -> str:
    """Lead the caption with the series line, e.g. 'Ordinary Grace, no. 4'.

    First line, because Instagram indexes caption text for search and shows the
    first line before the fold — it is the only place a viewer reliably reads.
    """
    from app.services import series

    line = series.reel_label(part)
    if not line or not caption.strip():
        return caption
    if caption.lstrip().lower().startswith(line.lower()):
        return caption
    return f"{line}\n\n{caption}".strip()[:2200]


def _caption_for_script(script) -> str:
    metadata = getattr(script, "social_metadata", None)
    candidates = [
        getattr(metadata, "instagram_caption", "") if metadata else "",
        getattr(metadata, "tiktok_caption", "") if metadata else "",
        getattr(metadata, "youtube_description", "") if metadata else "",
        getattr(script, "summary", ""),
        getattr(script, "hook", ""),
        getattr(script, "title", ""),
    ]
    caption = next((str(value).strip() for value in candidates if str(value).strip()), "")
    hashtags = []
    if metadata:
        for tag in getattr(metadata, "hashtags", []) or []:
            cleaned = str(tag).strip()
            if not cleaned:
                continue
            cleaned = cleaned if cleaned.startswith("#") else f"#{cleaned}"
            if cleaned.lower() not in {existing.lower() for existing in hashtags}:
                hashtags.append(cleaned)
    if hashtags:
        existing_caption = caption.lower()
        missing = [tag for tag in hashtags[:6] if tag.lower() not in existing_caption]
        if missing:
            caption = f"{caption}\n\n{' '.join(missing)}".strip()
    return caption[:2200].strip()


def _video_seconds(path: str) -> Optional[float]:
    try:
        from moviepy import VideoFileClip

        with VideoFileClip(path) as clip:
            return round(float(clip.duration), 2)
    except Exception as exc:  # noqa: BLE001 - a missing duration is not fatal
        logger.warning(f"could not read duration of {path}: {exc}")
        return None


def _variant_of(video_path: str, narration: str, hook: str) -> dict:
    """What treatment produced this reel, recorded at publish time.

    Watch time two days later is unreadable without it: eight seconds watched is
    superb on a 9-second reel and dismal on a 70-second one, and "which of these
    changes helped" is unanswerable if nobody wrote down which ones were on.
    """
    from app.config import config

    return {
        "video_seconds": _video_seconds(video_path),
        "script_chars": len(narration or ""),
        "hook": (hook or "")[:80],
        "script_style": str(config.app.get("script_style", "default")),
        "subtitle_renderer": str(config.app.get("subtitle_renderer", "moviepy")),
        "subtitle_cadence": str(config.app.get("subtitle_cadence", "punctuation")),
        "voice": str(config.app.get("article_voice_name", "")).split(":")[-1][:40],
    }


def _publish(task_id: str, script, cluster: Optional[ArticleCluster] = None) -> dict:
    """Auto-publish or schedule a rendered Article Mode video."""
    try:
        from app.services import state as sm
        from app.services import upload_post
        from app.services import postiz

        task = sm.state.get_task(task_id) or {}
        videos = task.get("videos") or []
        if not videos:
            return {"success": False, "error": "rendered video not found"}

        caption = _caption_for_script(script)
        if not caption:
            return {"success": False, "error": "caption is empty"}

        if postiz.postiz_service.enabled or postiz.postiz_service.auto_schedule_enabled:
            if not postiz.postiz_service.is_auto_schedule_configured():
                return {"success": False, "error": "Postiz auto-scheduling is not configured"}
            # Burn a hook into the opening seconds. Falls back to the
            # untouched render on any failure, so this can never cost a post.
            from app.services import reel_hook

            # The subject comes from the script, then the cluster. `script` is a
            # GeneratedScript, not a string, and ArticleCluster has no `title` —
            # reading either as one raised before schedule_video was ever called,
            # which silently disabled auto-publish entirely (see runbook).
            subject = (
                str(getattr(script, "title", "") or "").strip()
                or str(getattr(cluster, "normalized_title", "") or "").strip()
            )
            narration = script.narration_text() if hasattr(script, "narration_text") else str(script)
            hook_text = reel_hook.generate_hook(subject, narration)
            video_for_post = reel_hook.add_hook(videos[0], hook_text)

            from app.services import series

            part = series.reel_current()
            variant = _variant_of(video_for_post, narration, hook_text)
            if part:
                variant["series"] = series.reel_label(part)
            result = postiz.schedule_video(
                video_for_post, _with_series_line(caption, part), variant=variant,
            )
            result["provider"] = "postiz"
            # Count the number only once the post really exists, for the same
            # reason the card series does: a failed publish must not burn a
            # number and leave a gap in the run.
            if result.get("success"):
                series.reel_advance()
            return result

        if not (
            upload_post.upload_post_service.is_configured()
            and upload_post.upload_post_service.platforms
        ):
            return {"success": False, "error": "no publishing integration configured"}

        successes = []
        failures = []
        for video_path in videos:
            result = upload_post.cross_post_video(
                video_path=video_path,
                title=script.social_metadata.youtube_title or script.title,
                platforms=list(upload_post.upload_post_service.platforms),
                youtube_extra={
                    "youtube_title": script.social_metadata.youtube_title or script.title,
                    "youtube_description": script.social_metadata.youtube_description,
                    "tags": script.social_metadata.hashtags,
                    "containsSyntheticMedia": True,
                },
            )
            if result.get("success"):
                successes.append(result)
            else:
                failures.append(result)
        if successes:
            return {
                "success": True,
                "provider": "upload_post",
                "request_id": successes[0].get("request_id", "ok"),
                "results": successes + failures,
            }
        error = "Upload-Post failed"
        if failures:
            error = failures[0].get("error") or failures[0].get("message") or error
        return {
            "success": False,
            "provider": "upload_post",
            "error": error,
            "results": failures,
        }
    except Exception as exc:
        logger.warning(f"auto-publish failed for task {task_id}: {exc}")
        return {"success": False, "error": str(exc)}


def _media_mode(subscription: Optional[TopicSubscription]) -> MediaMode:
    value = str(article_pipeline.config.app.get("article_media_mode", "images_only"))
    try:
        return MediaMode(value)
    except ValueError:
        return MediaMode.images_only


def run_once(
    repo: Optional[ArticleRepository] = None,
    settings: Optional[AutomationSettings] = None,
    mode_override: Optional[AutomationMode] = None,
    subscription_id: Optional[str] = None,
) -> List[dict]:
    """A single poll + process pass over due subscriptions."""
    repo = repo or get_repository()
    settings = settings or article_pipeline.load_automation_settings()
    all_results: List[dict] = []
    if subscription_id:
        subscription = repo.get_subscription(subscription_id)
        subscriptions = [subscription] if subscription else []
    else:
        subscriptions = [s for s in repo.list_subscriptions(enabled_only=True) if s.is_due()]

    for subscription in subscriptions:
        mode = mode_override or settings.resolve_mode(subscription.automation_mode)
        run, articles, clusters = poll_subscription(repo, subscription)
        results = process_clusters(repo, subscription, articles, clusters, settings, mode)
        all_results.append(
            {
                "subscription": subscription.name,
                "poll": run.model_dump(mode="json"),
                "clusters": results,
            }
        )
    return all_results


def run_forever(interval_seconds: int, mode_override: Optional[AutomationMode] = None) -> None:
    logger.info(f"article worker started, interval={interval_seconds}s")
    try:
        while True:
            try:
                run_once(mode_override=mode_override)
            except Exception:
                logger.exception("article worker pass failed; continuing")
            time.sleep(max(30, interval_seconds))
    except KeyboardInterrupt:
        logger.info("article worker interrupted; shutting down")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Influencer-Automation 2.0 Article Mode worker")
    parser.add_argument("--once", action="store_true", help="run a single pass and exit")
    parser.add_argument("--autonomous", action="store_true", help="force autonomous mode")
    parser.add_argument("--automated", action="store_true", help="force automated mode")
    parser.add_argument("--assisted", action="store_true", help="force assisted mode")
    parser.add_argument("--subscription", help="poll only this subscription id")
    parser.add_argument("--interval", type=int, help="loop interval in seconds")
    args = parser.parse_args(argv)

    mode_override: Optional[AutomationMode] = None
    if args.autonomous:
        mode_override = AutomationMode.autonomous
    elif args.automated:
        mode_override = AutomationMode.automated
    elif args.assisted:
        mode_override = AutomationMode.assisted

    settings = article_pipeline.load_automation_settings()
    interval = args.interval or int(
        article_pipeline.config.app.get("article_poll_interval_minutes", 60)
    ) * 60

    if args.once or args.subscription:
        results = run_once(
            settings=settings,
            mode_override=mode_override,
            subscription_id=args.subscription,
        )
        logger.info(f"article worker pass complete: {len(results)} subscription(s) processed")
        return 0

    run_forever(interval, mode_override=mode_override)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
