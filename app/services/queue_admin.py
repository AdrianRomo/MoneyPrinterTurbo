"""Inspect and flush the Postiz publishing queue.

Why this exists as a tool rather than a one-off: on 2026-08-25 the content
pipeline changed (shorter Reels, no stories, a carousel verse slide) while 31
posts built under the OLD rules were already queued eight days out. Changing the
generator does nothing to work that is already scheduled, so "the change is live"
and "the change is visible in the feed" are more than a week apart unless the
queue is flushed. That will be true of every future content change, and the
ad-hoc version of this went wrong twice in ways worth encoding:

  * A date-ranged ``GET /posts`` returns PUBLISHED posts alongside QUEUE ones.
    Deleting what that returns would delete the account's live history. Nothing
    here ever deletes a post that is not in a flushable state.
  * Postiz throttles the API. A straight loop got 29 of 30 deletions through and
    then hit HTTP 429, leaving the queue in a half-flushed state that neither
    reported as failure nor finished the job.

Everything is dry-run by default; ``--apply`` is required to delete, and a
backup is always written first.

    python3 -m app.services.queue_admin list
    python3 -m app.services.queue_admin flush                 # dry run
    python3 -m app.services.queue_admin flush --apply --reason "cadence change"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from loguru import logger

from app.services.postiz import PostizService, redact_postiz_text

BACKUP_DIR = "/influencer-automation-2.0/storage/postiz"

# Only these may ever be deleted. PUBLISHED is deliberately absent: a published
# post is the account's history, and the Instagram media it produced would be
# orphaned rather than removed. ERROR is absent too — those pin their R2 media
# and are worth keeping visible until someone has looked at them.
FLUSHABLE_STATES = frozenset({"QUEUE", "DRAFT"})

# Postiz's throttler is per-window, not per-request, so pacing matters more than
# retrying: 29 rapid deletes tripped it. This is slow on purpose — a flush is
# rare and interactive, and taking a minute is cheaper than a half-flushed queue.
DELETE_PAUSE_SECONDS = 1.5
MAX_RETRIES = 6
BACKOFF_START_SECONDS = 15


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_when(value: Optional[str], default: datetime) -> datetime:
    if not value:
        return default
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit(f"could not parse {value!r} as a date: {exc}") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def list_posts(svc: PostizService, start: datetime, end: datetime) -> list[dict]:
    """Every post Postiz knows about in the window, whatever its state.

    Soft-deleted posts are already excluded by the API, so this is the live
    picture rather than the table.
    """
    try:
        response = requests.get(svc._endpoint("posts"), headers=svc._headers(),
                                params={"startDate": _iso(start), "endDate": _iso(end)},
                                timeout=30)
        response.raise_for_status()
        payload = response.json()
    except (requests.exceptions.RequestException, ValueError) as exc:
        raise SystemExit(f"could not list posts: {redact_postiz_text(str(exc), svc.api_key)}")
    posts = payload if isinstance(payload, list) else payload.get("posts", payload)
    return posts if isinstance(posts, list) else []


def flushable(posts: list[dict]) -> list[dict]:
    return [p for p in posts if str(p.get("state", "")).upper() in FLUSHABLE_STATES]


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "flush").lower()).strip("-")[:40] or "flush"


def write_backup(posts: list[dict], reason: str) -> str:
    """Full JSON of everything about to be deleted, before anything is deleted.

    Captions, publish dates and media references are all here, so a mistaken
    flush costs the SCHEDULE rather than the work — the content can be read back
    and re-queued by hand.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = os.path.join(BACKUP_DIR, f"queue-backup-{stamp}-{_slug(reason)}.json")
    os.makedirs(BACKUP_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"captured_at": datetime.now(timezone.utc).isoformat(),
                   "reason": reason, "count": len(posts), "posts": posts},
                  fh, indent=2, ensure_ascii=False)
    return path


def delete_post(svc: PostizService, post_id: str) -> tuple[bool, str]:
    """Delete one post, waiting out the throttler rather than giving up on it."""
    delay = BACKOFF_START_SECONDS
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.delete(svc._endpoint(f"posts/{post_id}"),
                                       headers=svc._headers(), timeout=30)
        except requests.exceptions.RequestException as exc:
            return False, redact_postiz_text(str(exc), svc.api_key)
        if response.status_code == 200:
            return True, "deleted"
        if response.status_code != 429:
            return False, f"http {response.status_code}: {redact_postiz_text(response.text, svc.api_key)[:160]}"
        if attempt < MAX_RETRIES:
            logger.warning(f"throttled deleting {post_id}; waiting {delay}s "
                           f"(attempt {attempt}/{MAX_RETRIES})")
            time.sleep(delay)
            delay *= 2
    return False, f"still throttled after {MAX_RETRIES} attempts"


def flush(svc: PostizService, start: datetime, end: datetime, *,
          apply: bool = False, reason: str = "queue flush") -> dict:
    posts = list_posts(svc, start, end)
    targets = flushable(posts)
    skipped = len(posts) - len(targets)

    summary = {"window": [_iso(start), _iso(end)], "found": len(posts),
               "flushable": len(targets), "protected": skipped, "applied": apply}
    if not targets:
        summary["note"] = "nothing to flush"
        return summary

    # Backup happens even on a dry run: it costs nothing and it means the
    # operator can read exactly what WOULD go before deciding to pass --apply.
    summary["backup"] = write_backup(targets, reason)
    if not apply:
        summary["note"] = "dry run — pass --apply to delete"
        summary["would_delete"] = [{"id": p.get("id"), "publishDate": p.get("publishDate")}
                                   for p in targets]
        return summary

    deleted, failures = 0, []
    for index, post in enumerate(targets):
        ok, detail = delete_post(svc, str(post.get("id")))
        if ok:
            deleted += 1
        else:
            failures.append({"id": post.get("id"), "error": detail})
        if index < len(targets) - 1:
            time.sleep(DELETE_PAUSE_SECONDS)

    summary["deleted"] = deleted
    summary["failed"] = failures
    # Ask Postiz again rather than trusting the loop's own tally: a delete that
    # returned 200 but did not stick is exactly the failure this needs to catch.
    summary["remaining"] = len(flushable(list_posts(svc, start, end)))
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["list", "flush"])
    parser.add_argument("--from", dest="start", help="ISO date (default: 30 days ago)")
    parser.add_argument("--to", dest="end", help="ISO date (default: 60 days ahead)")
    parser.add_argument("--apply", action="store_true",
                        help="actually delete; without it this is a dry run")
    parser.add_argument("--reason", default="queue flush",
                        help="recorded in the backup filename and its contents")
    args = parser.parse_args(argv)

    svc = PostizService()
    if not svc.is_api_configured():
        print("Postiz API is not configured", file=sys.stderr)
        return 2

    now = datetime.now(timezone.utc)
    start = _parse_when(args.start, now - timedelta(days=30))
    end = _parse_when(args.end, now + timedelta(days=60))

    if args.command == "list":
        posts = list_posts(svc, start, end)
        for post in sorted(posts, key=lambda p: str(p.get("publishDate"))):
            state = str(post.get("state", "?")).upper()
            mark = "flushable" if state in FLUSHABLE_STATES else "protected"
            first_line = str(post.get("content", "")).strip().splitlines()
            print(f"{post.get('publishDate','?')[:16]}  {state:9s} {mark:9s} "
                  f"{post.get('id')}  {(first_line[0] if first_line else '')[:60]}")
        print(f"\n{len(posts)} posts — {len(flushable(posts))} flushable, "
              f"{len(posts) - len(flushable(posts))} protected")
        return 0

    result = flush(svc, start, end, apply=args.apply, reason=args.reason)
    print(json.dumps(result, indent=2))
    return 1 if result.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
