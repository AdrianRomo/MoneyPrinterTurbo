"""Collect Instagram insights and attribute reach back to hashtag sets.

Closes the loop started in hashtags.py: each published post recorded which
hashtag set it used, and this reads the resulting reach/saves/shares so future
selection is driven by measured performance instead of guesswork.

Run indirectly via collect-insights.sh, which supplies the Instagram token and
the Postiz post -> permalink map from docker-devops. The token deliberately
never lands in config.toml: it is short-lived (~59 days, refreshed by Postiz)
and copying it would go stale silently.

    IG_TOKEN=... python3 -m app.services.insights   # post map on stdin
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import requests
from loguru import logger

from app.services import hashtags
from app.services.postiz import PostizService

GRAPH = "https://graph.instagram.com/v21.0"
# Give Instagram time to accumulate numbers; a post measured after an hour tells
# you about posting time, not about the hashtags.
MIN_AGE_HOURS = 48
METRICS = "reach,likes,comments,saved,shares"


def _norm(url: str) -> str:
    return (url or "").split("?")[0].rstrip("/").lower()


def fetch_media(token: str, limit: int = 50) -> dict[str, str]:
    """{normalised permalink: media_id} for recent media."""
    try:
        r = requests.get(f"{GRAPH}/me/media",
                         params={"fields": "id,permalink,timestamp", "limit": limit,
                                 "access_token": token}, timeout=30)
        r.raise_for_status()
        data = r.json().get("data", [])
    except (requests.exceptions.RequestException, ValueError) as exc:
        logger.error(f"could not list media: {exc}")
        return {}
    return {_norm(m.get("permalink", "")): m["id"] for m in data if m.get("permalink")}


def fetch_insights(token: str, media_id: str) -> dict:
    try:
        r = requests.get(f"{GRAPH}/{media_id}/insights",
                         params={"metric": METRICS, "access_token": token}, timeout=30)
        if r.status_code != 200:
            # Stories and some media types expose a different metric set; a
            # failure here is not fatal, it just means no sample for this post.
            logger.warning(f"insights unavailable for {media_id}: http {r.status_code}")
            return {}
        payload = r.json().get("data", [])
    except (requests.exceptions.RequestException, ValueError) as exc:
        logger.warning(f"insights request failed for {media_id}: {exc}")
        return {}
    out = {}
    for entry in payload:
        name = entry.get("name")
        values = entry.get("values") or []
        value = values[0].get("value") if values else (entry.get("total_value") or {}).get("value")
        if name is not None and value is not None:
            out[name] = value
    return out


def collect(token: str, post_urls: dict[str, str]) -> dict:
    """post_urls: {postiz_post_id: releaseURL}. Returns a summary dict."""
    log = PostizService._load_publish_log()
    permalinks = fetch_media(token)
    if not permalinks:
        return {"collected": 0, "error": "no media listed"}

    cutoff = datetime.now(timezone.utc) - timedelta(hours=MIN_AGE_HOURS)
    collected, skipped_young, unmatched = 0, 0, 0

    for entry in log:
        set_id, post_id = entry.get("set_id"), entry.get("post_id")
        if not set_id or not post_id:
            continue  # e.g. Reels published before set tracking existed
        try:
            when = datetime.fromisoformat(str(entry.get("at", "")).replace("Z", "+00:00"))
        except ValueError:
            continue
        if when > cutoff:
            skipped_young += 1
            continue

        media_id = permalinks.get(_norm(post_urls.get(post_id, "")))
        if not media_id:
            unmatched += 1
            continue

        metrics = fetch_insights(token, media_id)
        if not metrics:
            continue
        hashtags.record_sample(set_id, media_id, metrics)
        collected += 1
        logger.info(f"insights for set '{set_id}': {metrics}")

    return {"collected": collected, "too_young": skipped_young, "unmatched": unmatched,
            "scores": hashtags.set_scores()}


def main() -> int:
    token = os.environ.get("IG_TOKEN", "").strip()
    if not token:
        print("IG_TOKEN not set", file=sys.stderr)
        return 2
    try:
        post_urls = json.load(sys.stdin)
    except ValueError:
        print("expected a JSON {post_id: releaseURL} map on stdin", file=sys.stderr)
        return 2
    if not isinstance(post_urls, dict):
        print("stdin JSON must be an object", file=sys.stderr)
        return 2

    summary = collect(token, post_urls)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
