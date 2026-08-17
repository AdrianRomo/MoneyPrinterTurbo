"""Drain the quote-Reel review queue.

`quote_reel.enqueue_review` has been writing to `storage/quote_reel_review/
queue.json` since it was added, and nothing has ever read it. A Reel that fails
the quality gate renders, queues, and then sits there — the queue was write-only,
so "needs a human" in practice meant "is lost".

Three verbs, and publishing needs `--apply`:

    python3 -m app.services.quote_reel_review --list
    python3 -m app.services.quote_reel_review --show <task_id>
    python3 -m app.services.quote_reel_review --discard <task_id>
    python3 -m app.services.quote_reel_review --approve <task_id> [--apply]

Approving is the only thing in the pipeline that bypasses the `publishable`
gate. It does so because a human said to, which is what the queue exists for —
every other rail (Postiz configured, auto-scheduling on, quota, windows, the
daily cap) still applies, and the Reel series counter still advances only on a
real publish.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from app.services import quote_reel


def _summarise(item: dict) -> dict:
    qc = item.get("qc") or {}
    final = str(item.get("final_path") or "")
    return {
        "task_id": item.get("task_id"),
        "queued_at": item.get("queued_at", "unknown"),
        "quote": (item.get("quote") or "")[:70],
        "reasons": item.get("review_reasons") or [],
        "duration": qc.get("duration"),
        "resolution": qc.get("resolution"),
        "file_present": os.path.exists(final),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="what is waiting")
    action.add_argument("--show", metavar="TASK_ID", help="full record for one item")
    action.add_argument("--discard", metavar="TASK_ID", help="drop it, publish nothing")
    action.add_argument("--approve", metavar="TASK_ID", help="release it for publishing")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually publish on --approve; without it, --approve is a dry run",
    )
    args = parser.parse_args()

    if args.list:
        queue = quote_reel.list_review()
        print(json.dumps([_summarise(i) for i in queue], indent=2, default=str))
        return 0

    if args.show:
        item = next((i for i in quote_reel.list_review()
                     if i.get("task_id") == args.show), None)
        if item is None:
            print(f"{args.show} is not in the review queue", file=sys.stderr)
            return 1
        print(json.dumps(item, indent=2, default=str))
        return 0

    if args.discard:
        if not quote_reel.discard_review(args.discard):
            print(f"{args.discard} is not in the review queue", file=sys.stderr)
            return 1
        print(f"discarded {args.discard}")
        return 0

    result = quote_reel.approve_review(args.approve, apply=args.apply)
    print(json.dumps(result, indent=2, default=str))
    if result.get("dry_run"):
        print("\ndry run — pass --apply to publish this to the live account",
              file=sys.stderr)
        return 0
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
