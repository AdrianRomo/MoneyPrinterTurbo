"""Decides what to publish, and when.

The per-type quotas in postiz.py are ceilings, not a schedule: they stop one
format from eating another's slot, but nothing ever *asks* for a verse card or
a carousel. Article Mode's worker produces article Reels; this module also
keeps a quiet quote Reel lane alive when that format is due.

Run it a few times a day (see mpt-content-scheduler.timer). Each run asks, per
format: is today's quota unspent, and is the format enabled? If so it generates
and hands the result to Postiz, which picks the
actual moment via select_publish_at(kind=...) — a uniform-random time inside
that format's publishing window.

This module decides *whether*; postiz.py decides *when*. Quota is read before
generating, so a double run is a no-op rather than a duplicate post, and no GPU
time is spent on something that would only be deferred.

Quota days and windows are both the account's LOCAL civil day (content_timezone,
default America/Mexico_City). They used to disagree — quota in UTC, windows
local — and at -6 the UTC day rolls over at 18:00 local, so every evening slot
drew from a fresh, empty quota.

    python3 -m app.services.content_scheduler --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timedelta, timezone

from loguru import logger

from app.config import config
from app.services.postiz import PostizService, _local_date

# per_day is a target, not a licence to spend: the real ceiling is still
# postiz_daily_quota_<kind>. The time of day comes from the publishing windows
# in postiz.py (postiz_window_<kind>), not from here.
#
# THE MIX IS A DISTRIBUTION DECISION, NOT A PRODUCTION ONE.
#
# This was post 1 / carousel 1 / reel 1 / story 3, so half the daily output went
# to Stories — a surface that only reaches people who ALREADY follow the
# account. At the reach this account is actually getting (8-14 per post, zero
# saves, zero shares over its first four days), that half was being published to
# almost nobody, while the two surfaces that reach strangers got one slot each.
#
# Reels are the only real discovery surface Instagram has; carousels reach
# non-followers through Explore. So the budget moves to them. Stories keep
# exactly one slot, and it is the twin of the day's feed card — the one story
# with a job to do, giving the post early engagement velocity.
#
# Revisit this once W3.1 attribution has reach broken down BY FORMAT. This is a
# reasoned prior, and the whole point of measuring per-format reach is that it
# stops being one.
PLAN = {
    "post": {"per_day": 1},
    "carousel": {"per_day": 1},
    # Shares the real Reel quota and windows with Article Mode: if Article Mode
    # has already scheduled today's Reels, quiet quote Reels wait for tomorrow.
    #
    # Two Reels needs TWO reel windows and TWO scheduler runs. run_once produces
    # at most one item per kind per run, and with a single window the second run
    # of the day would find it mostly in the past and roll to tomorrow — costing
    # a post while the ledger still looks fine. See _windows_for in postiz.py.
    "reel": {"per_day": 2},
    "story": {"per_day": 1},
}

STORY_THEMES = ["peace and rest", "trust in the everyday", "gratitude",
                "hope for tomorrow", "stillness", "God's nearness"]
POST_THEMES = ["hope and trust", "encouragement in hard seasons", "gratitude",
               "faith in ordinary days", "peace", "new beginnings"]
# Subjects, not quotes — the language of the finished Reel comes from
# quote_reel_default_language, so keep these as neutral English prompts.
QUOTE_REEL_THEMES = [
    "God in the ordinary",
    "faith in ordinary days",
    "quiet beauty and grace",
    "peace in simple moments",
    "God's nearness in daily life",
    "gratitude for small things",
]
# Empty means "every subject carousel.py defines", which is what you want: this
# list used to be 17 hardcoded landscape subjects, so creatures and the heavens
# were never scheduled at all, and new subjects had to be added in two places.
# Override per-deployment with `content_scheduler_carousel_subjects`.
CAROUSEL_SUBJECTS: list = []


def _cfg_int(key: str, default: int) -> int:
    try:
        return int(config.app.get(key, default))
    except (TypeError, ValueError):
        return default


def _cfg_bool(key: str, default: bool) -> bool:
    value = config.app.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _cfg_list(key: str, default: list[str]) -> list[str]:
    value = config.app.get(key)
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    elif isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value]
    else:
        return list(default)
    return [item for item in items if item] or list(default)


def _parse_publish_at(value) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
    except ValueError:
        return None


def _format_enabled(kind: str) -> tuple[bool, str]:
    if kind == "reel" and not _cfg_bool("quote_reel_auto_schedule_enabled", False):
        return False, "quote reel auto-scheduling disabled"
    return True, "enabled"


def _days_since_last(kind: str) -> int:
    entries = [e for e in PostizService._load_publish_log() if e.get("kind") == kind]
    if not entries:
        return 10_000
    try:
        last = max(str(e.get("date", "")) for e in entries)
        last_date = datetime.strptime(last, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return 10_000
    return (_local_date(datetime.now(timezone.utc)) - last_date).days


def due(kind: str, now_utc: datetime, svc: PostizService) -> tuple[bool, str]:
    """Is this format due? Returns (due, reason).

    Only decides *whether*; postiz.select_publish_at decides *when*, by drawing
    a random time inside the format's window.
    """
    enabled, enabled_reason = _format_enabled(kind)
    if not enabled:
        return False, enabled_reason

    plan = PLAN[kind]
    target = min(plan["per_day"], svc.type_quotas.get(kind, 0))
    if target <= 0:
        return False, f"{kind} quota is zero"

    # The local civil day, not the UTC one: with a -6 offset the UTC day rolls
    # over at 18:00 local, which handed every evening slot a fresh empty quota.
    today = _local_date(now_utc)
    slot = svc.select_publish_at(now=now_utc, kind=kind)
    if not slot.get("success"):
        return False, slot.get("error") or f"no {kind} slot available"
    publish_at = _parse_publish_at(slot.get("publish_at"))
    if publish_at is None:
        return False, f"no valid {kind} publish_at returned"
    quota_day = _local_date(publish_at)
    # How deep the scheduler is willing to QUEUE, as opposed to how far Postiz
    # will search for a free slot (postiz_post_lookahead_days, a month). Raised
    # from 1 to a week so the queue can absorb a night the GPU was busy, and
    # deliberately stopped there rather than following the lookahead out to a
    # month: at the account's current reach the format is not proven, and a
    # month of queued posts is a month of commitment to a format that may need
    # to change next week. The month-deep resource is the footage pool, which
    # costs nothing to discard.
    horizon_days = max(0, _cfg_int("content_scheduler_schedule_days_ahead", 7))
    horizon = today + timedelta(days=horizon_days)
    if quota_day > horizon:
        return (
            False,
            f"next {kind} slot is {quota_day}, outside {horizon_days}d scheduler horizon",
        )
    used = PostizService._count_kind_on(kind, quota_day)
    if used >= target:
        return False, f"already scheduled {used} {kind} for {quota_day} (target {target})"

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
    from app.models.schema import VideoParams
    from app.services import carousel as ca
    from app.services import quote_reel
    from app.services import series
    from app.services import verse_card as vc

    if kind == "carousel":
        slides = _cfg_int("carousel_slides", 8)
        attempts = max(1, _cfg_int("content_scheduler_carousel_attempts", 6))
        subjects = [
            subject
            for subject in _cfg_list("content_scheduler_carousel_subjects", CAROUSEL_SUBJECTS)
            if subject in ca.SUBJECTS
        ]
        if not subjects:
            subjects = list(ca.SUBJECTS)
        # Least-recently-used, NOT shuffled. A shuffle has no memory, so the
        # same handful of subjects kept winning and the account published
        # "the creativity of God in mountains" three times over.
        subjects = ca.rank_subjects(subjects)
        tried = []
        for subject in subjects[:attempts]:
            tried.append(subject)
            car = ca.build(subject=subject, slides=slides)
            if car:
                return ca.publish(car)
            logger.warning(f"carousel subject {subject!r} had too few usable images")
        return {
            "success": False,
            "error": f"carousel build failed after trying: {', '.join(tried)}",
        }

    if kind == "reel":
        subject = random.choice(QUOTE_REEL_THEMES)
        task_id = f"scheduled-quote-reel-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        result = quote_reel.render_quote_reel(
            task_id,
            VideoParams(
                video_subject=subject,
                content_mode=quote_reel.CONTENT_MODE,
                video_aspect="9:16",
                n_threads=_cfg_int("quote_reel_n_threads", 2),
            ),
        )
        publish_result = dict(result.get("publish_result") or {})
        if publish_result.get("success"):
            publish_result.setdefault("success", True)
            publish_result["video"] = (result.get("videos") or [""])[0]
            publish_result["artifact"] = result.get("quote_reel_artifact")
            return publish_result
        return {
            "success": False,
            "error": (
                publish_result.get("error")
                or result.get("error")
                or "quote reel was generated but not published"
            ),
            "video": (result.get("videos") or [""])[0],
            "artifact": result.get("quote_reel_artifact"),
            "review_queue": result.get("quote_reel_review_queue"),
            "review_required": result.get("review_required"),
        }

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


def run_once(
    dry_run: bool = False,
    only: str | None = None,
    *,
    force: bool = False,
) -> dict:
    svc = PostizService()
    if not svc.is_configured():
        return {"error": "Postiz is not configured"}

    now = datetime.now(timezone.utc)
    results: dict[str, dict] = {}

    for kind in PLAN:
        if only and kind != only:
            continue
        is_due, reason = (True, "forced") if force else due(kind, now, svc)
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
        for key in ("video", "artifact", "review_queue", "review_required"):
            if outcome.get(key):
                results[kind][key] = outcome.get(key)
        if not outcome.get("success"):
            logger.error(f"{kind}: {outcome.get('error')}")

    return results


def run_loop(interval_seconds: int, dry_run: bool = False, only: str | None = None) -> None:
    interval_seconds = max(60, int(interval_seconds or 900))
    logger.info(f"content scheduler loop started, interval={interval_seconds}s")
    while True:
        try:
            run_once(dry_run=dry_run, only=only)
        except Exception:
            logger.exception("content scheduler pass failed; continuing")
        time.sleep(interval_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish whatever is due")
    parser.add_argument("--dry-run", action="store_true", help="decide but do not publish")
    parser.add_argument("--only", choices=sorted(PLAN), help="restrict to one format")
    parser.add_argument(
        "--force",
        action="store_true",
        help="produce the selected format now; Postiz still chooses the legal slot",
    )
    parser.add_argument("--loop", action="store_true", help="run continuously")
    parser.add_argument("--interval", type=int, default=900, help="loop interval seconds")
    args = parser.parse_args()

    if args.force and not args.only:
        parser.error("--force requires --only")
    if args.force and args.loop:
        parser.error("--force is for one-off runs, not --loop")

    if args.loop:
        run_loop(args.interval, dry_run=args.dry_run, only=args.only)
        return 0

    results = run_once(dry_run=args.dry_run, only=args.only, force=args.force)
    print(json.dumps(results, indent=2, default=str))
    # Non-zero only on a real publish failure, so the timer's OnFailure fires
    # for genuine problems and stays quiet on "nothing was due".
    return 1 if any(
        isinstance(r, dict) and r.get("action") == "failed"
        for r in results.values()
    ) else 0


if __name__ == "__main__":
    sys.exit(main())
