"""Postiz public API integration for scheduled Article Mode publishing.

The adapter is intentionally narrow: it uploads a rendered video, verifies the
configured Instagram integration, computes a future schedule slot, and creates a
scheduled Postiz post. It never logs API keys or full upload URLs.
"""

from __future__ import annotations

import json
import mimetypes
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
        # Per-type daily quotas, so one content type cannot consume the whole
        # allowance. postiz_daily_post_cap remains the global ceiling on top.
        #
        # These are counted from a LOCAL ledger, not from Postiz: the public
        # posts API returns no settings/media, so a post's type cannot be
        # recovered from it. We are the only publisher, so our own record is
        # authoritative for our own posts; anything published by hand still
        # counts against the global cap, which is queried live.
        self.type_quotas = {
            "reel": max(0, _cfg_int("postiz_daily_quota_reel", 1)),
            "post": max(0, _cfg_int("postiz_daily_quota_post", 1)),
            "story": max(0, _cfg_int("postiz_daily_quota_story", 2)),
            "carousel": max(0, _cfg_int("postiz_daily_quota_carousel", 1)),
        }

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
        """Upload a media file. Works for video *and* images.

        The content type must match the file: Postiz derives the stored
        extension from the upload's mimetype, and its Instagram provider decides
        between a video and a photo container by looking for '.mp4' in the
        stored path. Sending a JPEG as video/mp4 therefore stores it as .mp4 and
        Instagram tries to transcode a still image, failing with
        "Invalid video duration: None".
        """
        if not self.is_api_configured():
            return self._failure("Postiz API is not configured")
        if not video_path or not os.path.exists(video_path):
            return self._failure(f"media file not found: {video_path}")

        try:
            content_type = mimetypes.guess_type(video_path)[0]
            if not content_type:
                content_type = "video/mp4" if video_path.lower().endswith(".mp4") else "image/jpeg"
            with open(video_path, "rb") as video_file:
                files = {
                    "file": (
                        os.path.basename(video_path),
                        video_file,
                        content_type,
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

    # --- local per-type publish ledger ------------------------------------

    @staticmethod
    def _publish_log_path() -> str:
        d = "/influencer-automation-2.0/storage/postiz"
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "publish_log.json")

    @classmethod
    def _load_publish_log(cls) -> list[dict]:
        try:
            with open(cls._publish_log_path(), encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, list) else []
        except (OSError, ValueError):
            return []

    @classmethod
    def _record_publish(cls, kind: str, publish_at: datetime,
                        set_id: Optional[str] = None,
                        post_id: Optional[str] = None) -> None:
        entries = cls._load_publish_log()
        entries.append({"kind": kind, "date": publish_at.astimezone(timezone.utc).date().isoformat(),
                        "at": _utc_iso(publish_at),
                        # set_id/post_id let collect-insights.sh attribute reach
                        # back to the hashtag set that was used.
                        "set_id": set_id, "post_id": post_id})
        # 120 days is plenty for a daily quota and keeps the file small.
        cutoff = (datetime.now(timezone.utc).date() - timedelta(days=120)).isoformat()
        entries = [e for e in entries if str(e.get("date", "")) >= cutoff]
        try:
            with open(cls._publish_log_path(), "w", encoding="utf-8") as fh:
                json.dump(entries, fh)
        except OSError as exc:
            logger.warning(f"could not persist publish log: {exc}")

    @classmethod
    def _count_kind_on(cls, kind: str, day) -> int:
        target = day.isoformat() if hasattr(day, "isoformat") else str(day)
        return sum(1 for e in cls._load_publish_log()
                   if e.get("kind") == kind and str(e.get("date")) == target)

    def quota_for(self, kind: Optional[str]) -> Optional[int]:
        if not kind:
            return None
        return self.type_quotas.get(kind)

    # --- publishing windows -------------------------------------------

    def _utc_offset(self) -> int:
        return _cfg_int("content_utc_offset_hours", -6)

    def _window_for(self, kind: Optional[str]) -> Optional[tuple[float, float]]:
        """Local-time window for a format, as 'HH:MM-HH:MM'.

        Windows exist so posts land when the audience is actually awake, and so
        the exact minute varies day to day. A fixed hour plus jitter still
        clusters on the same minute; a uniform draw inside a window does not.
        """
        if not kind:
            return None
        raw = str(config.app.get(f"postiz_window_{kind}", "") or "").strip()
        if not raw or "-" not in raw:
            return None
        try:
            start, end = raw.split("-", 1)
            sh, sm = (int(x) for x in start.strip().split(":"))
            eh, em = (int(x) for x in end.strip().split(":"))
        except (ValueError, TypeError):
            logger.warning(f"unparsable postiz_window_{kind}: {raw!r}")
            return None
        return (sh + sm / 60.0, eh + em / 60.0)

    def _window_slot(self, kind: str, on_date, now: datetime) -> Optional[datetime]:
        """A uniform-random UTC instant inside that local window on that date."""
        window = self._window_for(kind)
        if not window:
            return None
        start_h, end_h = window
        if end_h <= start_h:
            return None
        offset = self._utc_offset()
        pick = random.uniform(start_h, end_h)
        # local hour -> utc
        slot = datetime.combine(on_date, time.min, tzinfo=timezone.utc) + timedelta(
            hours=pick - offset)
        if slot < now + _MIN_SCHEDULE_LEAD:
            return None
        return slot

    def _with_jitter(self, candidate: datetime, now: datetime) -> datetime:
        if self.schedule_jitter_minutes > 0:
            offset = random.randint(
                -self.schedule_jitter_minutes,
                self.schedule_jitter_minutes,
            )
            candidate = candidate + timedelta(minutes=offset)
        return max(candidate, now + _MIN_SCHEDULE_LEAD)

    def select_publish_at(self, now: Optional[datetime] = None,
                          kind: Optional[str] = None) -> dict:
        """Compute the next future schedule time while enforcing local caps.

        `kind` ('reel' | 'post' | 'story') additionally enforces that type's
        daily quota, so verse cards cannot eat the Reel's slot and vice versa.
        Omitting it keeps the previous behaviour (global cap only).
        """
        if self.daily_post_cap <= 0:
            return self._failure("postiz_daily_post_cap is zero")
        type_quota = self.quota_for(kind)
        if type_quota is not None and type_quota <= 0:
            return self._failure(f"postiz_daily_quota_{kind} is zero")

        integration = self.get_configured_integration()
        if not integration.get("success"):
            return integration

        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)

        # Window mode: when a format has a configured window, that window IS the
        # spacing, so the interval-based minimum does not apply — it would push
        # a 06:30 card to 10:30 and out of its own window.
        if self._window_for(kind):
            for day_offset in range(_POST_LOOKAHEAD_DAYS):
                on_date = (now + timedelta(days=day_offset)).date()
                if type_quota is not None and self._count_kind_on(kind, on_date) >= type_quota:
                    continue
                slot = self._window_slot(kind, on_date, now)
                if slot is not None:
                    return {"success": True, "publish_at": slot}
            return self._failure(f"no {kind} window available inside lookahead window")

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
            # Per-type quota, counted from our own ledger (see __init__).
            if type_quota is not None and self._count_kind_on(kind, candidate.date()) >= type_quota:
                logger.info(
                    f"{kind} quota ({type_quota}/day) reached for {candidate.date()}; "
                    "rolling to the next day"
                )
                candidate = datetime.combine(
                    candidate.date() + timedelta(days=1),
                    time.min,
                    tzinfo=timezone.utc,
                ) + timedelta(hours=self.schedule_interval_hours)
                continue
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
        kind: Optional[str] = None,
        set_id: Optional[str] = None,
    ) -> dict:
        caption = (caption or "").strip()
        if not caption:
            return self._failure("caption is empty")
        # A list of media becomes a carousel: Postiz passes the children through
        # to Instagram as media_type=CAROUSEL. Instagram caps carousels at 10.
        items = media if isinstance(media, list) else [media]
        images = []
        for item in items:
            item_id = str((item or {}).get("id") or "").strip()
            item_path = str((item or {}).get("path") or "").strip()
            if not item_id or not item_path:
                return self._failure("Postiz media id/path is missing")
            images.append({"id": item_id, "path": item_path})
        if not images:
            return self._failure("no media supplied")
        if len(images) > 10:
            logger.warning(f"carousel has {len(images)} items; Instagram allows 10 — truncating")
            images = images[:10]

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
                            "image": images,
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

        # Postiz's public API returns a JSON *array* of the created posts (one per
        # integration in the request); other/older deployments return a bare object.
        # Accept both. Getting this wrong is worse than it looks: the post has already
        # been created at this point (the call was 2xx), so reporting failure here
        # invites a retry that double-posts.
        if isinstance(result, list):
            result = next((item for item in result if isinstance(item, dict)), None)
        if not isinstance(result, dict):
            return self._failure(
                "Postiz create post response was neither an object nor a "
                "non-empty array of objects"
            )
        post_id = str(
            result.get("postId")
            or result.get("post_id")
            or result.get("id")
            or ""
        ).strip()
        if not post_id:
            return self._failure("Postiz create post response did not include postId")
        # Record against the per-type quota only once the post really exists.
        if kind:
            self._record_publish(kind, publish_at, set_id=set_id, post_id=post_id)
        logger.info(
            "Postiz post scheduled: "
            f"post_id={post_id}, integration_id={self.integration_id}, "
            f"kind={kind or 'unspecified'}, publish_at={_utc_iso(publish_at)}"
        )
        return {
            "success": True,
            "post_id": post_id,
            "integration_id": self.integration_id,
            "kind": kind,
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
            # Video posts are Reels; count them against the reel quota so verse
            # cards and stories cannot consume the day's video slot.
            selected = self.select_publish_at(now=now, kind="reel")
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
            kind="reel",
        )
        if not scheduled.get("success"):
            return scheduled
        scheduled["media_id"] = upload["media"]["id"]
        return scheduled


postiz_service = PostizService()


def schedule_video(video_path: str, caption: str, publish_at: Optional[datetime] = None) -> dict:
    return postiz_service.schedule_video(video_path, caption, publish_at)
