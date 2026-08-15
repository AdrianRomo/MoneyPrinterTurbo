"""Decides what to publish, and when.

The per-type quotas in postiz.py are ceilings, not a schedule: they stop one
format from eating another's slot, but nothing ever *asks* for a verse card or
a carousel. Article Mode's worker only ever produces Reels. This module is what
makes the other three formats actually happen.

Run it a few times a day (see mpt-content-scheduler.timer). Each run asks, per
format: is today's quota unspent, and for carousels has enough time passed since
the last one? If so it generates and hands the result to Postiz, which picks the
actual moment via select_publish_at(kind=...) — a uniform-random time inside
that format's publishing window.

This module decides *whether*; postiz.py decides *when*. Quota is read before
generating, so a double run is a no-op rather than a duplicate post, and no GPU
time is spent on something that would only be deferred.

Quota days are UTC, matching the ledger; windows are local via a fixed offset,
so there is no tzdata dependency.

    python3 -m app.services.content_scheduler --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone

from loguru import logger

from app.config import config
from app.services.postiz import PostizService

# per_day is a target, not a licence to spend: the real ceiling is still
# postiz_daily_quota_<kind>. The time of day comes from the publishing windows
# in postiz.py (postiz_window_<kind>), not from here.
#
# Only one story a day, and deliberately so: stories are shown mainly to people
# who already follow you, so on a young account they are a retention surface,
# not a discovery one. Reels and carousels are what actually grow reach, so they
# get the slots.
PLAN = {
    "post":     {"per_day": 1},
    "story":    {"per_day": 1},
    # Every other day, and Sunday counts double: faith audiences are markedly
    # more active on Sundays, and carousels reward unhurried scrolling.
    "carousel": {"per_day": 1, "min_interval_days": 2, "sunday_interval_days": 1},
}

STORY_THEMES = ["peace and rest", "trust in the everyday", "gratitude",
                "hope for tomorrow", "stillness", "God's nearness"]
POST_THEMES = ["hope and trust", "encouragement in hard seasons", "gratitude",
               "faith in ordinary days", "peace", "new beginnings"]


def _cfg_int(key: str, default: int) -> int:
    try:
        return int(config.app.get(key, default))
    except (TypeError, ValueError):
        return default


def _days_since_last(kind: str) -> int:
    entries = [e for e in PostizService._load_publish_log() if e.get("kind") == kind]
    if not entries:
        return 10_000
    try:
        last = max(str(e.get("date", "")) for e in entries)
        last_date = datetime.strptime(last, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return 10_000
    return (datetime.now(timezone.utc).date() - last_date).days


def due(kind: str, now_utc: datetime, svc: PostizService) -> tuple[bool, str]:
    """Is this format due? Returns (due, reason).

    Only decides *whether*; postiz.select_publish_at decides *when*, by drawing
    a random time inside the format's window.
    """
    plan = PLAN[kind]
    today = now_utc.date()
    used = PostizService._count_kind_on(kind, today)
    if used >= min(plan["per_day"], svc.type_quotas.get(kind, 0)):
        return False, f"already published {used} today (target {plan['per_day']})"

    interval = plan.get("min_interval_days")
    if interval:
        # Sunday = 6 in Python's weekday(); local day, not UTC.
        local_now = now_utc + timedelta(hours=_cfg_int("content_utc_offset_hours", -6))
        if local_now.weekday() == 6 and plan.get("sunday_interval_days"):
            interval = plan["sunday_interval_days"]
        gap = _days_since_last(kind)
        if gap < interval:
            return False, f"last one was {gap}d ago, interval is {interval}d"
    return True, "due"


def produce(kind: str) -> dict:
    """Generate and publish one item of this format."""
    import random

    from app.services import carousel as ca
    from app.services import verse_card as vc

    if kind == "carousel":
        car = ca.build(slides=_cfg_int("carousel_slides", 8))
        if not car:
            return {"success": False, "error": "carousel build failed"}
        return ca.publish(car)

    theme = random.choice(STORY_THEMES if kind == "story" else POST_THEMES)
    card = vc.create_card(kind="story" if kind == "story" else "post", theme=theme)
    if not card:
        return {"success": False, "error": f"{kind} generation failed"}
    return vc.publish_card(card)


def run_once(dry_run: bool = False, only: str | None = None) -> dict:
    svc = PostizService()
    if not svc.is_configured():
        return {"error": "Postiz is not configured"}

    now = datetime.now(timezone.utc)
    results: dict[str, dict] = {}

    for kind in PLAN:
        if only and kind != only:
            continue
        is_due, reason = due(kind, now, svc)
        if not is_due:
            logger.info(f"{kind}: skip — {reason}")
            results[kind] = {"action": "skip", "reason": reason}
            continue
        if dry_run:
            logger.info(f"{kind}: WOULD PUBLISH ({reason})")
            results[kind] = {"action": "would-publish"}
            continue

        logger.info(f"{kind}: due, producing")
        outcome = produce(kind)
        results[kind] = {
            "action": "published" if outcome.get("success") else "failed",
            "post_id": outcome.get("post_id"),
            "publish_at": outcome.get("publish_at"),
            "error": outcome.get("error"),
        }
        if not outcome.get("success"):
            logger.error(f"{kind}: {outcome.get('error')}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish whatever is due")
    parser.add_argument("--dry-run", action="store_true", help="decide but do not publish")
    parser.add_argument("--only", choices=sorted(PLAN), help="restrict to one format")
    args = parser.parse_args()

    results = run_once(dry_run=args.dry_run, only=args.only)
    print(json.dumps(results, indent=2, default=str))
    # Non-zero only on a real publish failure, so the timer's OnFailure fires
    # for genuine problems and stays quiet on "nothing was due".
    return 1 if any(r.get("action") == "failed" for r in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
