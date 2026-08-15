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
# Three stories a day: the first is the story twin of the day's feed card
# (same verse, same background, pointing at the post), published just after it
# to give it early engagement velocity; the other two stand alone.
#
# One window per story, not one window shared by three — see _windows_for in
# postiz.py. The scheduler produces at most one item per kind per run, so three
# stories need three runs, and by the second run a single shared window is
# mostly in the past.
PLAN = {
    "post":     {"per_day": 1},
    "story":    {"per_day": 3},
    "carousel": {"per_day": 1},
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
    from app.services import series
    from app.services import verse_card as vc

    if kind == "carousel":
        car = ca.build(slides=_cfg_int("carousel_slides", 8))
        if not car:
            return {"success": False, "error": "carousel build failed"}
        return ca.publish(car)

    # One story a day is the twin of the feed card, so the two read as one post.
    # PLAN runs 'post' before 'story', so within a single run the card already
    # exists by the time we get here. The other stories stand alone.
    if kind == "story" and vc.twin_pending():
        twin = vc.create_story_from_todays_post()
        if twin:
            result = vc.publish_card(twin)
            if result.get("success"):
                vc.mark_twin_done()
            return result

    # Feed cards run as a SERIES, not a random theme. A one-off card earns a
    # save; a numbered run earns a follow, because tomorrow is part 4 of
    # something someone already started. Stories stay themed and standalone —
    # they expire in a day, so continuity there buys nothing.
    if kind == "post":
        part = series.current()
        if part:
            card = vc.create_card(kind="post", subject=part["subject"],
                                  reference=part["reference"],
                                  series_label=series.label(part))
            if not card:
                return {"success": False, "error": "series card generation failed"}
            result = vc.publish_card(card)
            # Advance only on a real publish, so a failed render retries the
            # same part rather than silently skipping it.
            if result.get("success"):
                series.advance(part["series_id"])
            return result
        logger.warning("no series defined; falling back to a random theme")

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
