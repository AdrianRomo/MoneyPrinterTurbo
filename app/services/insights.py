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
from typing import Optional

import requests
from loguru import logger

from app.services import hashtags
from app.services.postiz import PostizService

GRAPH = "https://graph.instagram.com/v21.0"
GROWTH_STORE = "/influencer-automation-2.0/storage/growth/account.json"

# Account-level metrics, i.e. the ones that answer "is this account growing?".
#
# Nothing collected them until 2026-08-25. Every number in this module was
# per-MEDIA — reach and saves on individual posts — which measures whether a
# post did well, never whether the account did. So the pipeline could report
# healthy for eleven days while followers sat at 47, and there is no history to
# reconstruct: a follower count is an instantaneous reading and cannot be
# backfilled. Hence a daily snapshot, kept forever, from the first run onward.
ACCOUNT_FIELDS = "username,media_count,followers_count,follows_count"
ACCOUNT_METRICS = ("reach", "profile_views", "accounts_engaged", "total_interactions")
ACCOUNT_WINDOW_DAYS = 28
# Give Instagram time to accumulate numbers; a post measured after an hour tells
# you about posting time, not about the hashtags.
MIN_AGE_HOURS = 48

METRICS = "reach,likes,comments,saved,shares"

# Reels are ranked on watch time and completion, and none of that was being
# collected — so the selector has been tuning hashtag sets while blind to the
# thing that actually drives reach. These two are Reels-only: asking for them on
# a feed image returns an error for the WHOLE request, which is why the metric
# set is chosen per media type rather than shared.
REEL_METRICS = METRICS + ",ig_reels_avg_watch_time,ig_reels_video_view_total_time"

RETENTION_METRICS = ("ig_reels_avg_watch_time", "ig_reels_video_view_total_time")


def _norm(url: str) -> str:
    return (url or "").split("?")[0].rstrip("/").lower()


def metrics_for(product_type: str) -> str:
    return REEL_METRICS if str(product_type or "").upper() == "REELS" else METRICS


def fetch_media(token: str, limit: int = 50) -> dict[str, dict]:
    """{normalised permalink: {id, product_type}} for recent media."""
    try:
        r = requests.get(f"{GRAPH}/me/media",
                         params={"fields": "id,permalink,timestamp,media_product_type",
                                 "limit": limit, "access_token": token}, timeout=30)
        r.raise_for_status()
        data = r.json().get("data", [])
    except (requests.exceptions.RequestException, ValueError) as exc:
        logger.error(f"could not list media: {exc}")
        return {}
    return {
        _norm(m.get("permalink", "")): {
            "id": m["id"],
            "product_type": m.get("media_product_type", ""),
        }
        for m in data if m.get("permalink")
    }


def _request_metrics(token: str, media_id: str, metric: str) -> Optional[list]:
    try:
        r = requests.get(f"{GRAPH}/{media_id}/insights",
                         params={"metric": metric, "access_token": token}, timeout=30)
        if r.status_code != 200:
            # Stories and some media types expose a different metric set; a
            # failure here is not fatal, it just means no sample for this post.
            logger.warning(f"insights unavailable for {media_id}: http {r.status_code}")
            return None
        return r.json().get("data", [])
    except (requests.exceptions.RequestException, ValueError) as exc:
        logger.warning(f"insights request failed for {media_id}: {exc}")
        return None


def fetch_insights(token: str, media_id: str, product_type: str = "") -> dict:
    metric = metrics_for(product_type)
    payload = _request_metrics(token, media_id, metric)

    if payload is None and metric != METRICS:
        # Meta renames and retires Reels metrics periodically. Losing the reach
        # and saves we have collected all along because a watch-time metric went
        # away would be a bad trade, so fall back to the base set.
        logger.warning(
            f"retrying {media_id} without retention metrics (the reel set was rejected)"
        )
        payload = _request_metrics(token, media_id, METRICS)

    if payload is None:
        return {}

    out = {}
    for entry in payload:
        name = entry.get("name")
        values = entry.get("values") or []
        value = values[0].get("value") if values else (entry.get("total_value") or {}).get("value")
        if name is not None and value is not None:
            out[name] = value
    return out


def fetch_account(token: str) -> dict:
    """Follower/following/media counts. {} if the call fails."""
    try:
        r = requests.get(f"{GRAPH}/me",
                         params={"fields": ACCOUNT_FIELDS, "access_token": token},
                         timeout=30)
        r.raise_for_status()
        data = r.json()
    except (requests.exceptions.RequestException, ValueError) as exc:
        logger.error(f"could not read account fields: {exc}")
        return {}
    return {k: data.get(k) for k in ACCOUNT_FIELDS.split(",") if data.get(k) is not None}


def fetch_account_insights(token: str, days: int = ACCOUNT_WINDOW_DAYS) -> dict:
    """Windowed account totals. Each metric is requested on its own.

    One request per metric on purpose: the media endpoint already taught us that
    a single rejected metric fails the WHOLE request, and losing the follower
    trend because 'profile_views' was renamed would be the same bad trade
    fetch_insights() already guards against.
    """
    until = int(datetime.now(timezone.utc).timestamp())
    since = until - days * 86400
    out: dict[str, int] = {}
    for metric in ACCOUNT_METRICS:
        try:
            r = requests.get(f"{GRAPH}/me/insights",
                             params={"metric": metric, "period": "day",
                                     "metric_type": "total_value",
                                     "since": since, "until": until,
                                     "access_token": token}, timeout=30)
            if r.status_code != 200:
                logger.warning(f"account metric {metric!r} unavailable: http {r.status_code}")
                continue
            data = r.json().get("data") or []
            if data and isinstance(data[0].get("total_value"), dict):
                out[metric] = data[0]["total_value"].get("value")
        except (requests.exceptions.RequestException, ValueError) as exc:
            logger.warning(f"account metric {metric!r} failed: {exc}")
    return {k: v for k, v in out.items() if v is not None}


def _load_growth() -> list:
    try:
        with open(GROWTH_STORE, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def record_account_snapshot(token: str) -> dict:
    """Append today's account reading to the growth series.

    One row per calendar day, last write wins, so running the collector twice in
    a day corrects the row rather than double-counting it.
    """
    account = fetch_account(token)
    if not account:
        return {}
    snapshot = {"at": datetime.now(timezone.utc).isoformat(),
                "date": datetime.now(timezone.utc).date().isoformat(),
                **account,
                **fetch_account_insights(token)}

    series = [row for row in _load_growth() if row.get("date") != snapshot["date"]]
    series.append(snapshot)
    series.sort(key=lambda row: str(row.get("date")))
    try:
        os.makedirs(os.path.dirname(GROWTH_STORE), exist_ok=True)
        with open(GROWTH_STORE, "w", encoding="utf-8") as fh:
            json.dump(series, fh, indent=2)
    except OSError as exc:
        logger.warning(f"could not write the growth series: {exc}")
    return snapshot


def growth_report(days: int = 28) -> dict:
    """Change in the account-level series over the window, or why it cannot say."""
    series = _load_growth()
    if len(series) < 2:
        return {"days": len(series),
                "note": "need at least two daily snapshots before growth is a number"}
    first, last = series[max(0, len(series) - days)], series[-1]
    span = max(1, len(series[max(0, len(series) - days):]) - 1)
    delta = {}
    for key in ("followers_count", "media_count", "reach", "profile_views",
                "accounts_engaged", "total_interactions"):
        if isinstance(first.get(key), (int, float)) and isinstance(last.get(key), (int, float)):
            delta[key] = last[key] - first[key]
    followers = delta.get("followers_count")
    return {
        "from": first.get("date"), "to": last.get("date"), "days_observed": span,
        "latest": {k: last.get(k) for k in ("followers_count", "follows_count",
                                            "media_count", "reach", "profile_views")},
        "delta": delta,
        "followers_per_day": round(followers / span, 2) if followers is not None else None,
        # The ratio a visitor reads as "is this a real account or a bot", and the
        # one thing on this page that is fixed by hand rather than by the pipeline.
        "follow_ratio": (round(last["followers_count"] / last["follows_count"], 2)
                         if last.get("follows_count") else None),
    }


def _local_hour(when: datetime) -> Optional[int]:
    """The hour of the account's civil day a post went out.

    Posting time is one of the dimensions the windows were set from, and the
    runbook is explicit that those windows are priors rather than measurements.
    They cannot stop being priors until the hour is recorded next to the reach.
    Uses the same local-day convention as the quota ledger.
    """
    try:
        from app.services.postiz import _local_tz

        return when.astimezone(_local_tz()).hour
    except Exception:  # noqa: BLE001
        return None


def collect(token: str, post_urls: dict[str, str]) -> dict:
    """post_urls: {postiz_post_id: releaseURL}. Returns a summary dict."""
    log = PostizService._load_publish_log()
    permalinks = fetch_media(token)
    if not permalinks:
        return {"collected": 0, "error": "no media listed"}

    cutoff = datetime.now(timezone.utc) - timedelta(hours=MIN_AGE_HOURS)
    collected, skipped_young, unmatched, stories = 0, 0, 0, 0

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

        media = permalinks.get(_norm(post_urls.get(post_id, "")))
        if not media:
            # Stories are not a measurement failure — /me/media never returns
            # them, so a story can NEVER be matched to a permalink and will be
            # retried, and re-counted, on every run forever. Lumping them in
            # with genuine misses is what drove "unmatched" from 10 to 26 and
            # made the number unreadable: it looked like a degrading matcher
            # when it was a format that is structurally unmeasurable.
            if str(entry.get("kind") or "").lower() == "story":
                stories += 1
            else:
                unmatched += 1
            continue

        metrics = fetch_insights(token, media["id"], media.get("product_type", ""))
        if not metrics:
            continue
        # The variant recorded at publish time is what makes retention readable:
        # watch time on its own says nothing without the script length and the
        # treatment that produced it.
        #
        # kind and hour come from the ledger entry rather than the variant: they
        # are known for EVERY post ever published, including the ones that
        # predate variants entirely, so format and timing can be read back over
        # the whole history instead of only over posts made from here on.
        hashtags.record_sample(set_id, media["id"], metrics,
                               variant=entry.get("variant"),
                               kind=entry.get("kind"),
                               local_hour=_local_hour(when))
        collected += 1
        retention = {k: v for k, v in metrics.items() if k in RETENTION_METRICS}
        logger.info(
            f"insights for set '{set_id}': {metrics}"
            + (f" (retention {retention})" if retention else " (no retention data)")
        )

    return {"collected": collected, "too_young": skipped_young, "unmatched": unmatched,
            "stories_unmeasurable": stories,
            "account": record_account_snapshot(token), "growth": growth_report(),
            "scores": hashtags.set_scores(), "retention": hashtags.retention_report(),
            "dimensions": hashtags.dimension_report()}


def main() -> int:
    # --report reads what has already been collected. It needs no token and
    # makes no API calls, so it is safe to run any time — including on a laptop
    # that cannot reach docker-devops.
    if "--report" in sys.argv[1:]:
        print(json.dumps({"growth": growth_report(),
                          "dimensions": hashtags.dimension_report()},
                         indent=2, default=str))
        return 0

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
