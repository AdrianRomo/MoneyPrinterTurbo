"""Postiz public API integration for scheduled Article Mode publishing.

The adapter is intentionally narrow: it uploads a rendered video, verifies the
configured Instagram integration, computes a future schedule slot, and creates a
scheduled Postiz post. It never logs API keys or full upload URLs.
"""

from __future__ import annotations

import os
import random
import re
from datetime import datetime, time, timedelta, timezone
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

import requests
from loguru import logger

from app.config import config

_INSTAGRAM_PROVIDERS = {"instagram", "instagram-standalone"}
_MIN_SCHEDULE_LEAD = timedelta(minutes=30)
_POST_LOOKAHEAD_DAYS = 14


def _cfg_bool(key: str, default: bool) -> bool:
    value = config.app.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _cfg_int(key: str, default: int) -> int:
    try:
        return int(config.app.get(key, default))
    except (TypeError, ValueError):
        return default


def _utc_iso(value: datetime) -> str:
    value = value.astimezone(timezone.utc)
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_url_for_log(url: str) -> str:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return url
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return url
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def redact_postiz_text(value: Any, api_key: str = "") -> str:
    """Return text safe for logs/errors: no API key and no URL query strings."""
    text = str(value)
    secrets = [api_key, str(config.app.get("postiz_api_key", "") or "")]
    for secret in secrets:
        if secret and len(secret) >= 4:
            text = text.replace(secret, "[redacted]")
    text = re.sub(
        r"https?://[^\s'\"<>]+",
        lambda match: _safe_url_for_log(match.group(0)),
        text,
    )
    return text


class PostizService:
    """Small wrapper around Postiz's public API."""

    def __init__(self):
        self.enabled = _cfg_bool("postiz_enabled", False)
        self.base_url = str(config.app.get("postiz_base_url", "") or "").strip().rstrip("/")
        self.api_key = str(config.app.get("postiz_api_key", "") or "").strip()
        self.integration_id = str(config.app.get("postiz_integration_id", "") or "").strip()
        self.provider_type = str(
            config.app.get("postiz_provider_type", "instagram") or "instagram"
        ).strip()
        self.auto_schedule_enabled = _cfg_bool("postiz_auto_schedule_enabled", False)
        self.schedule_interval_hours = max(
            1, _cfg_int("postiz_schedule_interval_hours", 2)
        )
        self.schedule_jitter_minutes = max(
            0, _cfg_int("postiz_schedule_jitter_minutes", 30)
        )
        self.daily_post_cap = max(0, _cfg_int("postiz_daily_post_cap", 8))
        self.post_type = str(config.app.get("postiz_post_type", "post") or "post").strip()

    def is_api_configured(self) -> bool:
        return bool(self.enabled and self.base_url and self.api_key)

    def is_configured(self) -> bool:
        return bool(self.is_api_configured() and self.integration_id)

    def is_auto_schedule_configured(self) -> bool:
        return bool(self.is_configured() and self.auto_schedule_enabled)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": self.api_key}

    def _endpoint(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _failure(self, message: str) -> dict:
        safe = redact_postiz_text(message, self.api_key)
        logger.warning(f"Postiz scheduling skipped: {safe}")
        return {"success": False, "error": safe}

    def _request_failure(self, action: str, exc: Exception) -> dict:
        safe = redact_postiz_text(str(exc), self.api_key)
        logger.warning(f"Postiz {action} failed: {safe}")
        return {"success": False, "error": safe}

    def list_integrations(self) -> dict:
        if not self.is_api_configured():
            return self._failure("Postiz API is not configured")
        try:
            response = requests.get(
                self._endpoint("integrations"),
                headers=self._headers(),
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
        except (ValueError, requests.exceptions.RequestException) as exc:
            return self._request_failure("integration lookup", exc)

        if isinstance(payload, list):
            integrations = payload
        elif isinstance(payload, dict):
            integrations = payload.get("integrations")
        else:
            integrations = None
        if not isinstance(integrations, list):
            return self._failure("Postiz integrations response was not a list")
        return {"success": True, "integrations": integrations}

    def get_configured_integration(self) -> dict:
        if not self.integration_id:
            return self._failure("postiz_integration_id is empty")
        if self.provider_type not in _INSTAGRAM_PROVIDERS:
            return self._failure(
                f"unsupported postiz_provider_type: {self.provider_type}"
            )

        result = self.list_integrations()
        if not result.get("success"):
            return result

        for integration in result["integrations"]:
            if not isinstance(integration, dict):
                continue
            if str(integration.get("id") or "") != self.integration_id:
                continue
            identifier = str(
                integration.get("identifier")
                or integration.get("providerIdentifier")
                or ""
            )
            if identifier != self.provider_type:
                return self._failure(
                    f"Postiz integration {self.integration_id} is {identifier or 'unknown'}, "
                    f"not {self.provider_type}"
                )
            disabled = integration.get("disabled")
            if disabled is True or str(disabled).strip().lower() == "true":
                return self._failure(f"Postiz integration {self.integration_id} is disabled")
            return {"success": True, "integration": integration}

        return self._failure(f"Postiz integration {self.integration_id} was not found")

    def upload_media(self, video_path: str) -> dict:
        if not self.is_api_configured():
            return self._failure("Postiz API is not configured")
        if not video_path or not os.path.exists(video_path):
            return self._failure(f"video file not found: {video_path}")

        try:
            with open(video_path, "rb") as video_file:
                files = {
                    "file": (
                        os.path.basename(video_path),
                        video_file,
                        "video/mp4",
                    )
                }
                response = requests.post(
                    self._endpoint("upload"),
                    headers=self._headers(),
                    files=files,
                    timeout=300,
                )
            response.raise_for_status()
            payload = response.json()
        except (OSError, ValueError, requests.exceptions.RequestException) as exc:
            return self._request_failure("media upload", exc)

        if not isinstance(payload, dict):
            return self._failure("Postiz upload response was not an object")
        media_id = str(payload.get("id") or "").strip()
        media_path = str(payload.get("path") or "").strip()
        if not media_id or not media_path:
            return self._failure("Postiz upload response did not include id and path")
        logger.info(f"Postiz media uploaded: id={media_id}")
        return {
            "success": True,
            "media": {
                "id": media_id,
                "path": media_path,
                "name": payload.get("name") or os.path.basename(video_path),
            },
        }

    def find_available_slot(self) -> dict:
        if not self.is_configured():
            return self._failure("Postiz integration is not configured")
        try:
            response = requests.get(
                self._endpoint(f"find-slot/{self.integration_id}"),
                headers=self._headers(),
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
        except (ValueError, requests.exceptions.RequestException) as exc:
            return self._request_failure("slot lookup", exc)

        slot = _parse_datetime(payload.get("date") if isinstance(payload, dict) else None)
        if slot is None:
            return self._failure("Postiz slot response did not include a valid date")
        return {"success": True, "date": slot}

    def list_posts(self, start: datetime, end: datetime) -> dict:
        if not self.is_api_configured():
            return self._failure("Postiz API is not configured")
        try:
            response = requests.get(
                self._endpoint("posts"),
                headers=self._headers(),
                params={"startDate": _utc_iso(start), "endDate": _utc_iso(end)},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
        except (ValueError, requests.exceptions.RequestException) as exc:
            return self._request_failure("scheduled post lookup", exc)

        posts = payload.get("posts") if isinstance(payload, dict) else None
        if not isinstance(posts, list):
            return self._failure("Postiz posts response did not include posts")
        return {"success": True, "posts": posts}

    def _posts_for_integration(self, posts: list[dict]) -> list[datetime]:
        dates: list[datetime] = []
        for post in posts:
            if not isinstance(post, dict):
                continue
            integration = post.get("integration")
            if isinstance(integration, dict):
                post_integration_id = str(integration.get("id") or "")
            else:
                post_integration_id = str(integration or "")
            if post_integration_id != self.integration_id:
                continue
            published_at = _parse_datetime(
                post.get("publishDate")
                or post.get("date")
                or post.get("scheduledAt")
            )
            if published_at is not None:
                dates.append(published_at)
        return sorted(dates)

    def _with_jitter(self, candidate: datetime, now: datetime) -> datetime:
        if self.schedule_jitter_minutes > 0:
            offset = random.randint(
                -self.schedule_jitter_minutes,
                self.schedule_jitter_minutes,
            )
            candidate = candidate + timedelta(minutes=offset)
        return max(candidate, now + _MIN_SCHEDULE_LEAD)

    def select_publish_at(self, now: Optional[datetime] = None) -> dict:
        """Compute the next future schedule time while enforcing local caps."""
        if self.daily_post_cap <= 0:
            return self._failure("postiz_daily_post_cap is zero")

        integration = self.get_configured_integration()
        if not integration.get("success"):
            return integration

        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        minimum_candidate = now + timedelta(hours=self.schedule_interval_hours)
        slot = self.find_available_slot()
        if not slot.get("success"):
            return slot
        candidate = max(minimum_candidate, slot["date"])

        window_end = now + timedelta(days=_POST_LOOKAHEAD_DAYS)
        scheduled = self.list_posts(now, window_end)
        if not scheduled.get("success"):
            return scheduled
        existing = self._posts_for_integration(scheduled["posts"])

        for _ in range(_POST_LOOKAHEAD_DAYS):
            same_day = [dt for dt in existing if dt.date() == candidate.date()]
            if len(same_day) >= self.daily_post_cap:
                candidate = datetime.combine(
                    candidate.date() + timedelta(days=1),
                    time.min,
                    tzinfo=timezone.utc,
                ) + timedelta(hours=self.schedule_interval_hours)
                continue
            latest_same_day = max(same_day, default=None)
            if latest_same_day and latest_same_day >= candidate:
                candidate = latest_same_day + timedelta(hours=self.schedule_interval_hours)
                continue
            return {
                "success": True,
                "publish_at": self._with_jitter(candidate, now),
            }

        return self._failure("no Postiz schedule slot available inside lookahead window")

    def schedule_post(
        self,
        media: dict,
        caption: str,
        publish_at: datetime,
        *,
        integration: Optional[dict] = None,
    ) -> dict:
        caption = (caption or "").strip()
        if not caption:
            return self._failure("caption is empty")
        media_id = str((media or {}).get("id") or "").strip()
        media_path = str((media or {}).get("path") or "").strip()
        if not media_id or not media_path:
            return self._failure("Postiz media id/path is missing")

        if integration is None:
            integration_result = self.get_configured_integration()
            if not integration_result.get("success"):
                return integration_result

        payload = {
            "type": "schedule",
            "date": _utc_iso(publish_at),
            "shortLink": False,
            "tags": [],
            "posts": [
                {
                    "integration": {"id": self.integration_id},
                    "value": [
                        {
                            "content": caption[:2200],
                            "image": [{"id": media_id, "path": media_path}],
                        }
                    ],
                    "settings": {
                        "__type": self.provider_type,
                        "post_type": self.post_type or "post",
                    },
                }
            ],
        }

        try:
            response = requests.post(
                self._endpoint("posts"),
                headers={**self._headers(), "Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
        except (ValueError, requests.exceptions.RequestException) as exc:
            return self._request_failure("post creation", exc)

        if not isinstance(result, dict):
            return self._failure("Postiz create post response was not an object")
        post_id = str(
            result.get("postId")
            or result.get("post_id")
            or result.get("id")
            or ""
        ).strip()
        if not post_id:
            return self._failure("Postiz create post response did not include postId")
        logger.info(
            "Postiz post scheduled: "
            f"post_id={post_id}, integration_id={self.integration_id}, "
            f"publish_at={_utc_iso(publish_at)}"
        )
        return {
            "success": True,
            "post_id": post_id,
            "integration_id": self.integration_id,
            "publish_at": _utc_iso(publish_at),
        }

    def schedule_video(
        self,
        video_path: str,
        caption: str,
        publish_at: Optional[datetime] = None,
        *,
        now: Optional[datetime] = None,
    ) -> dict:
        if not self.is_auto_schedule_configured():
            return self._failure("Postiz auto-scheduling is not configured")
        caption = (caption or "").strip()
        if not caption:
            return self._failure("caption is empty")

        integration = self.get_configured_integration()
        if not integration.get("success"):
            return integration

        if publish_at is None:
            selected = self.select_publish_at(now=now)
            if not selected.get("success"):
                return selected
            publish_at = selected["publish_at"]

        upload = self.upload_media(video_path)
        if not upload.get("success"):
            return upload
        scheduled = self.schedule_post(
            upload["media"],
            caption,
            publish_at,
            integration=integration["integration"],
        )
        if not scheduled.get("success"):
            return scheduled
        scheduled["media_id"] = upload["media"]["id"]
        return scheduled


postiz_service = PostizService()


def schedule_video(video_path: str, caption: str, publish_at: Optional[datetime] = None) -> dict:
    return postiz_service.schedule_video(video_path, caption, publish_at)
