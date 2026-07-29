#!/usr/bin/env python3
"""Reproducible Article Mode release-candidate smoke validation.

This command uses temporary storage, disables publishing, and replaces external
network providers with deterministic local fixtures while still exercising the
production repository, worker, task routing, no-voice TTS path, subtitle
generation, image/video normalization, renderer, and provenance artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from contextlib import ExitStack
from datetime import timezone
from email.utils import format_datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTICLE_URL = "https://example.com/news/article-mode-smoke?utm_source=test"
ARTICLE_CANONICAL = "https://example.com/news/article-mode-smoke"
FEED_URL = "https://feeds.example.com/article-mode-smoke.xml"
TIMEOUT_FEED_URL = "https://feeds.example.com/timeout.xml"
TEST_SECRET = "SMOKE_TEST_SECRET_TOKEN"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        help="directory for temporary config, storage, database and artifacts",
    )
    parser.add_argument(
        "--keep-output",
        action="store_true",
        help="keep the generated temporary directory; default already keeps it",
    )
    return parser.parse_args()


def _prepare_environment(output_dir: str | None) -> Path:
    root = Path(output_dir or tempfile.mkdtemp(prefix="ia2-article-smoke-")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.toml"
    if not config_path.exists():
        shutil.copyfile(PROJECT_ROOT / "config.example.toml", config_path)
    storage_dir = root / "storage"
    storage_dir.mkdir(parents=True, exist_ok=True)
    os.environ["IA2_CONFIG_FILE"] = str(config_path)
    os.environ["IA2_STORAGE_DIR"] = str(storage_dir)
    return root


def _load_runtime() -> dict[str, Any]:
    from app.config import config
    from app.models import const
    from app.models.article import (
        AutomationMode,
        AutomationSettings,
        MediaAsset,
        RiskLevel,
        TopicSubscription,
        utcnow,
    )
    from app.models.schema import VideoAspect, VideoParams
    from app.services import (
        article_ingestion,
        article_llm,
        article_repository,
        article_worker,
        material,
        state,
        task,
        video,
        voice,
    )
    from app.services.article_repository import SqliteArticleRepository
    from app.utils import utils
    from moviepy.video.io.VideoFileClip import VideoFileClip
    from PIL import Image, ImageDraw

    return {
        "config": config,
        "const": const,
        "AutomationMode": AutomationMode,
        "AutomationSettings": AutomationSettings,
        "MediaAsset": MediaAsset,
        "RiskLevel": RiskLevel,
        "TopicSubscription": TopicSubscription,
        "utcnow": utcnow,
        "VideoAspect": VideoAspect,
        "VideoParams": VideoParams,
        "article_ingestion": article_ingestion,
        "article_llm": article_llm,
        "article_repository": article_repository,
        "article_worker": article_worker,
        "material": material,
        "state": state,
        "task": task,
        "video": video,
        "voice": voice,
        "SqliteArticleRepository": SqliteArticleRepository,
        "utils": utils,
        "VideoFileClip": VideoFileClip,
        "Image": Image,
        "ImageDraw": ImageDraw,
    }


def _configure_app(rt: dict[str, Any]) -> None:
    config = rt["config"]
    config.app["enable_redis"] = False
    config.app["article_auto_publish_enabled"] = False
    config.app["article_auto_generate_enabled"] = True
    config.app["article_auto_render_enabled"] = True
    config.app["article_auto_rewrite_attempts"] = 1
    config.app["article_minimum_story_score"] = 0.5
    config.app["article_minimum_confidence_score"] = 0.5
    config.app["article_minimum_visual_score"] = 0.4
    config.app["article_media_mode"] = "mixed"
    config.app["article_image_provider"] = "pexels"
    config.app["article_voice_name"] = "no-voice"
    config.app["subtitle_provider"] = "edge"


def _article_html(title: str) -> bytes:
    paragraphs = [
        "Example News reports that engineers completed a controlled article mode release test on Tuesday.",
        "The test follows direct article extraction, RSS polling, AI story scoring, script review, media selection, text to speech, subtitles, and video rendering.",
        "Operators said the validation uses temporary storage and keeps automatic publishing disabled by default.",
        "The release candidate keeps source provenance and media licensing information with the generated artifacts.",
    ]
    body = "\n".join(f"<p>{line}</p>" for line in paragraphs * 3)
    return (
        "<html><head>"
        f"<title>{title}</title>"
        "<meta name='author' content='Example Desk'>"
        "</head><body><article>"
        f"{body}"
        "</article></body></html>"
    ).encode("utf-8")


def _feed_xml(rt: dict[str, Any]) -> bytes:
    published = format_datetime(rt["utcnow"]().astimezone(timezone.utc), usegmt=True)
    return f"""<?xml version="1.0"?>
<rss version="2.0"><channel>
  <title>Article Mode Smoke Feed</title>
  <item>
    <title>Article mode smoke release candidate</title>
    <link>{ARTICLE_URL}</link>
    <pubDate>{published}</pubDate>
  </item>
</channel></rss>""".encode("utf-8")


def _fixture_article_fetch(url: str) -> tuple[str, bytes, str]:
    if "empty" in url:
        return (url, b"<html><body></body></html>", "text/html")
    return (ARTICLE_CANONICAL, _article_html("Article mode smoke release candidate"), "text/html")


def _fixture_feed_fetch(rt: dict[str, Any], url: str) -> tuple[str, bytes, str]:
    if "timeout" in url:
        raise TimeoutError("fixture feed timeout")
    if "malformed" in url:
        return (url, b"not an rss feed", "application/rss+xml")
    return (url, _feed_xml(rt), "application/rss+xml")


class FixtureProviders:
    def __init__(self, rt: dict[str, Any], output_root: Path):
        self.rt = rt
        self.asset_dir = output_root / "fixtures"
        self.asset_dir.mkdir(parents=True, exist_ok=True)
        self.images = self._make_images()
        self.videos = self._make_videos()
        self.review_calls = 0

    def _make_images(self) -> list[Path]:
        image_cls = self.rt["Image"]
        draw_cls = self.rt["ImageDraw"]
        images: list[Path] = []
        colors = [(37, 99, 235), (5, 150, 105), (202, 138, 4)]
        for index, color in enumerate(colors, start=1):
            path = self.asset_dir / f"scene-{index}.png"
            image = image_cls.new("RGB", (1400, 1000), color)
            draw = draw_cls.Draw(image)
            draw.rectangle((80, 80, 1320, 920), outline=(255, 255, 255), width=8)
            draw.text((120, 130), f"Article smoke scene {index}", fill=(255, 255, 255))
            image.save(path)
            images.append(path)
        return images

    def _make_videos(self) -> list[Path]:
        video = self.rt["video"]
        aspect = self.rt["VideoAspect"].landscape
        videos: list[Path] = []
        for index, color in enumerate([(80, 40, 120), (20, 100, 100)], start=1):
            path = self.asset_dir / f"fixture-video-{index}.mp4"
            video.color_background_clip(str(path), 2.0, aspect, color=color)
            videos.append(path)
        return videos

    def llm_response(self, prompt: str) -> str:
        if "Story Scorer" in prompt:
            return json.dumps(
                {
                    "story_score": 0.91,
                    "confidence": 0.86,
                    "source_quality": 0.84,
                    "relevance": 0.88,
                    "viral_potential": 0.7,
                    "visual_potential": 0.82,
                    "freshness": 0.9,
                    "audience_fit": 0.8,
                    "duplicate_story": 0.1,
                    "risk_level": "low",
                    "sensitive_categories": [],
                    "recommended_action": "generate",
                    "reasoning_summary": "Release-candidate validation is timely and concrete.",
                    "uncertainties": ["This is fixture source material for local validation."],
                }
            )
        if "Editorial Reviewer" in prompt:
            self.review_calls += 1
            if self.review_calls % 2 == 1:
                return json.dumps(
                    {
                        "approved": False,
                        "confidence": 0.42,
                        "issues": ["Clarify that this is generated from available sources."],
                        "revised_script": None,
                    }
                )
            return json.dumps({"approved": True, "confidence": 0.91, "issues": []})
        return json.dumps(
            {
                "title": "Article Mode Smoke Release Candidate",
                "hook": "A release candidate just walked through the article pipeline.",
                "summary": "A source-grounded Article Mode smoke test rendered a short video.",
                "confidence": 0.88,
                "narration": "Article Mode is being validated from a public article and a feed.",
                "scenes": [
                    {
                        "narration": "A public article is extracted and persisted for the release candidate.",
                        "visual_queries": ["news article page", "editorial workflow"],
                        "visual_type": "image_or_video",
                        "duration_weight": 1.0,
                        "is_contextual_visual": True,
                    },
                    {
                        "narration": "The automated reviewer asks for clearer wording before approving the script.",
                        "visual_queries": ["script review checklist", "newsroom editing"],
                        "visual_type": "image_or_video",
                        "duration_weight": 1.0,
                        "is_contextual_visual": True,
                    },
                    {
                        "narration": "The final video stores source and license provenance without publishing.",
                        "visual_queries": ["video render timeline", "media license metadata"],
                        "visual_type": "image_or_video",
                        "duration_weight": 1.0,
                        "is_contextual_visual": True,
                    },
                ],
                "uncertainties": ["Local fixtures are not live provider evidence."],
                "source_ids": [],
                "social_metadata": {
                    "youtube_title": "Article Mode Smoke RC",
                    "youtube_description": "Generated from available sources for validation.",
                    "tiktok_caption": "Article Mode smoke validation",
                    "instagram_caption": "Article Mode smoke validation",
                    "hashtags": ["#ArticleMode", "#SmokeTest"],
                },
            }
        )

    def search_images(self, provider, query, video_aspect):
        del provider, video_aspect
        assets = []
        for index, image_path in enumerate(self.images):
            assets.append(
                self.rt["MediaAsset"](
                    media_type="image",
                    provider="fixture",
                    url=f"fixture://image/{index}",
                    width=1400,
                    height=1000,
                    asset_id=f"image-{index}",
                    creator="Fixture Creator",
                    license_name="Fixture License",
                    license_url=f"https://licenses.example.com/fixture?token={TEST_SECRET}",
                    attribution_text="Fixture image by Fixture Creator",
                    source_page_url=f"https://media.example.com/image/{index}?signature={TEST_SECRET}",
                    search_query=query,
                )
            )
        return assets

    def search_videos(self, provider, query, video_aspect):
        del provider, video_aspect
        if "script review" not in query and "newsroom" not in query:
            return []
        assets = []
        for index, video_path in enumerate(self.videos):
            assets.append(
                self.rt["MediaAsset"](
                    media_type="video",
                    provider="fixture",
                    url=f"fixture://video/{index}",
                    width=720,
                    height=1280,
                    duration=2.0,
                    asset_id=f"video-{index}",
                    creator="Fixture Creator",
                    license_name="Fixture Video License",
                    license_url=f"https://licenses.example.com/video?token={TEST_SECRET}",
                    attribution_text="Fixture video by Fixture Creator",
                    source_page_url=f"https://media.example.com/video/{index}?signature={TEST_SECRET}",
                    search_query=query,
                )
            )
        return assets

    def save_image(self, url: str, save_dir: str = "") -> str:
        del save_dir
        index = int(url.rsplit("/", 1)[-1])
        return str(self.images[index % len(self.images)])

    def save_video(self, url: str, save_dir: str = "") -> str:
        del save_dir
        index = int(url.rsplit("/", 1)[-1])
        return str(self.videos[index % len(self.videos)])


def _stage(name: str) -> None:
    print(f"[article-smoke] {name}", flush=True)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _task_dir(rt: dict[str, Any], task_id: str) -> Path:
    return Path(rt["utils"].task_dir(task_id))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _assert_no_secret(task_directory: Path) -> None:
    for artifact in task_directory.glob("*.json"):
        text = artifact.read_text(encoding="utf-8")
        _assert(TEST_SECRET not in text, f"secret leaked into {artifact}")
        _assert("signature=" not in text, f"signed URL query leaked into {artifact}")


def _video_duration(rt: dict[str, Any], video_path: str) -> float:
    clip_cls = rt["VideoFileClip"]
    with clip_cls(video_path, audio=True) as clip:
        _assert((clip.duration or 0) > 0, f"video has no duration: {video_path}")
        _assert(bool(clip.size), f"video has no frame size: {video_path}")
        return float(clip.duration)


def _validate_render(rt: dict[str, Any], task_id: str, result: dict[str, Any]) -> dict[str, Any]:
    const = rt["const"]
    task_state = rt["state"].state.get_task(task_id)
    _assert(task_state and task_state.get("state") == const.TASK_STATE_COMPLETE, "task did not complete")
    videos = result.get("videos") or []
    _assert(videos, "render returned no videos")
    output_path = videos[0]
    _assert(Path(output_path).is_file(), f"missing output mp4: {output_path}")
    audio_duration = float(result.get("audio_duration") or 0)
    actual_duration = _video_duration(rt, output_path)
    _assert(actual_duration + 0.3 >= audio_duration, "video duration does not cover narration")
    subtitle_path = result.get("subtitle_path") or ""
    _assert(subtitle_path and Path(subtitle_path).is_file(), "subtitle file was not generated")
    task_directory = _task_dir(rt, task_id)
    manifest = _read_json(task_directory / "media_manifest.json")
    _assert(manifest, "media manifest is empty")
    beat_indexes = [item["beat_index"] for item in manifest]
    _assert(beat_indexes == sorted(beat_indexes), "media manifest is not in scene order")
    _assert((task_directory / "sources.json").is_file(), "sources artifact missing")
    _assert((task_directory / "provenance.json").is_file(), "provenance artifact missing")
    _assert_no_secret(task_directory)
    return {
        "task_id": task_id,
        "output": output_path,
        "task_dir": str(task_directory),
        "duration": round(actual_duration, 3),
        "audio_duration": audio_duration,
        "media_types": [item["media_type"] for item in manifest],
        "beat_indexes": beat_indexes,
    }


def _render_task(rt: dict[str, Any], task_id: str, params) -> dict[str, Any]:
    result = rt["task"].start(task_id, params, stop_at="video")
    _assert(isinstance(result, dict), "task did not return a result dict")
    _assert(result.get("videos"), f"task failed: {result}")
    return result


def _direct_url_flow(rt: dict[str, Any]) -> dict[str, Any]:
    _stage("direct URL images-only flow")
    params = rt["VideoParams"](
        video_subject="Article Mode smoke direct URL",
        content_mode="article_url",
        media_mode="images_only",
        article_url=ARTICLE_URL,
        image_source="pexels",
        video_aspect="9:16",
        voice_name=rt["voice"].NO_VOICE_NAME,
        subtitle_enabled=True,
        bgm_type="",
        n_threads=2,
    )
    task_id = "article-smoke-direct"
    result = _render_task(rt, task_id, params)
    repo = rt["article_repository"].get_repository()
    articles = repo.list_articles(limit=10)
    _assert(any(a.canonical_url == ARTICLE_CANONICAL for a in articles), "direct article not persisted")
    return _validate_render(rt, task_id, result)


def _rss_worker_flow(rt: dict[str, Any]) -> dict[str, Any]:
    _stage("RSS worker flow with one valid feed and one timeout")
    repo = rt["article_repository"].get_repository()
    subscription = rt["TopicSubscription"](
        name="Smoke subscription",
        query="article mode",
        language="en",
        rss_urls=[FEED_URL, TIMEOUT_FEED_URL],
        trusted_domains=["example.com"],
        freshness_hours=24,
        poll_interval_minutes=0,
        automation_mode="automated",
    )
    repo.upsert_subscription(subscription)
    settings = rt["AutomationSettings"](
        minimum_story_score=0.5,
        minimum_confidence_score=0.5,
        minimum_visual_score=0.4,
        auto_generate_enabled=True,
        auto_render_enabled=True,
        auto_publish_enabled=False,
        maximum_risk_for_auto_publish=rt["RiskLevel"].low,
    )
    first = rt["article_worker"].run_once(
        repo=repo,
        settings=settings,
        mode_override=rt["AutomationMode"].automated,
        subscription_id=subscription.id,
    )
    _assert(first, "first worker pass did not process subscription")
    poll = first[0]["poll"]
    _assert(poll["articles_found"] == 1, f"expected one new RSS article, got {poll}")
    _assert(poll["errors"], "timeout feed error was not persisted")
    _assert(first[0]["clusters"] and first[0]["clusters"][0].get("rendered"), "worker did not render cluster")
    rendered_task_id = first[0]["clusters"][0]["task_id"]
    rendered_state = rt["state"].state.get_task(rendered_task_id)
    rendered = _validate_render(rt, rendered_task_id, rendered_state)

    article_count = len(repo.list_articles(subscription_id=subscription.id, limit=100))
    cluster_count = len(repo.list_clusters(subscription_id=subscription.id))
    second = rt["article_worker"].run_once(
        repo=repo,
        settings=settings,
        mode_override=rt["AutomationMode"].automated,
        subscription_id=subscription.id,
    )
    second_poll = second[0]["poll"] if second else {}
    _assert(second_poll.get("articles_found") == 0, f"second pass inserted articles: {second}")
    _assert(len(repo.list_articles(subscription_id=subscription.id, limit=100)) == article_count, "duplicate RSS article inserted")
    _assert(len(repo.list_clusters(subscription_id=subscription.id)) == cluster_count, "duplicate cluster inserted")
    _assert(repo.count_generations() == 1, "duplicate generation recorded")
    return {"first": first, "second": second, "render": rendered}


def _mixed_media_flow(rt: dict[str, Any]) -> dict[str, Any]:
    _stage("mixed-media render flow")
    repo = rt["article_repository"].get_repository()
    article = repo.find_article_by_canonical(ARTICLE_CANONICAL)
    _assert(article is not None, "direct article missing before mixed-media render")
    params = rt["VideoParams"](
        video_subject="Article Mode smoke mixed media",
        content_mode="article_feed",
        media_mode="mixed",
        article_id=article.id,
        image_source="pexels",
        video_aspect="9:16",
        voice_name=rt["voice"].NO_VOICE_NAME,
        subtitle_enabled=True,
        bgm_type="",
        n_threads=2,
    )
    task_id = "article-smoke-mixed"
    result = _render_task(rt, task_id, params)
    validated = _validate_render(rt, task_id, result)
    _assert("video" in validated["media_types"], "mixed-media manifest contains no video assets")
    return validated


def _negative_checks(rt: dict[str, Any]) -> dict[str, Any]:
    _stage("negative and recovery checks")
    ingestion = rt["article_ingestion"]
    failures: dict[str, str] = {}
    for name, fn in {
        "localhost_url": lambda: ingestion.validate_url("http://127.0.0.1/private"),
        "private_network_url": lambda: ingestion.validate_url("http://192.168.1.10/private"),
        "malformed_rss": lambda: ingestion.parse_feed(b"not an rss feed"),
        "empty_article": lambda: ingestion.build_article("https://example.com/empty", b"<html></html>"),
    }.items():
        try:
            fn()
        except Exception as exc:
            failures[name] = str(exc)
        else:
            raise AssertionError(f"negative check unexpectedly passed: {name}")
    return failures


def main() -> int:
    args = _parse_args()
    output_root = _prepare_environment(args.output_dir)
    sys.path.insert(0, str(PROJECT_ROOT))
    rt = _load_runtime()
    _configure_app(rt)
    db_path = output_root / "article-smoke.db"
    repo = rt["SqliteArticleRepository"](str(db_path))
    rt["article_repository"].set_repository(repo)
    providers = FixtureProviders(rt, output_root)

    with ExitStack() as stack:
        stack.enter_context(patch.object(rt["article_ingestion"], "_resolve_addresses", return_value=["93.184.216.34"]))
        stack.enter_context(patch.object(rt["article_ingestion"], "_default_fetcher", side_effect=_fixture_article_fetch))
        stack.enter_context(patch.object(rt["article_ingestion"], "_feed_fetcher", side_effect=lambda url: _fixture_feed_fetch(rt, url)))
        stack.enter_context(patch.object(rt["article_llm"].llm, "_generate_response", side_effect=providers.llm_response))
        stack.enter_context(patch.object(rt["material"], "search_images", side_effect=providers.search_images))
        stack.enter_context(patch.object(rt["material"], "search_video_assets", side_effect=providers.search_videos))
        stack.enter_context(patch.object(rt["material"], "save_image", side_effect=providers.save_image))
        stack.enter_context(patch.object(rt["material"], "save_video", side_effect=providers.save_video))

        direct = _direct_url_flow(rt)
        rss = _rss_worker_flow(rt)
        mixed = _mixed_media_flow(rt)
        negative = _negative_checks(rt)

    summary = {
        "validation": "local_integration",
        "output_root": str(output_root),
        "config": os.environ["IA2_CONFIG_FILE"],
        "storage": os.environ["IA2_STORAGE_DIR"],
        "database": str(db_path),
        "direct_url": direct,
        "rss_worker": rss,
        "mixed_media": mixed,
        "negative_checks": negative,
        "real_tts": False,
        "tts_path": "no-voice local TTS integration",
        "publishing_enabled": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
