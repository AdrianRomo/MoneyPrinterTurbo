#!/usr/bin/env python3
"""The squint test, automated.

A feed is judged as a GRID, not as single posts, and the failure this reports is
invisible to every per-image gate: 41 cards that each passed contrast, colour
and sharpness individually, and together looked like one photograph taken 41
times.

Two outputs:

  1. A contact sheet of the most recent N cards at thumbnail size. Look at it
     the way a visitor does — small, fast, all at once. If the rows blur into
     each other, the feed is repetitive no matter what the numbers say.
  2. The distribution the eye is reacting to: palette buckets, how concentrated
     they are, and how often neighbouring posts share one.

Read-only. Nothing here publishes, deletes or regenerates anything.

    python scripts/feed_variety_report.py
    python scripts/feed_variety_report.py --n 24 --out /tmp/sheet.jpg
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

from app.services import quality  # noqa: E402

DEFAULT_DIR = "/influencer-automation-2.0/storage/verse_cards"
# Published cards are named YYYYMMDD-HHMMSS-<kind>-<ref>.jpg. Working files
# (todays_post_bg.jpg, tuning experiments) are not, and must not be counted as
# things the audience saw.
PUBLISHED_PREFIX = "2026"

# A story twin DELIBERATELY reuses its feed card's verse and background — see
# verse_card.create_story_from_todays_post. Measuring posts and stories in one
# pool therefore reports the twin as a variety failure, which it is not: the two
# land on different surfaces, and a follower who sees both is meant to recognise
# the pairing. They are measured separately, and `--kind` picks one.
KINDS = ("post", "story")


def card_kind(name: str) -> str:
    parts = os.path.basename(name).split("-")
    return parts[2] if len(parts) > 2 else ""


def published_cards(directory: str, kind: str = "") -> list:
    try:
        names = sorted(f for f in os.listdir(directory)
                       if f.lower().endswith((".jpg", ".png"))
                       and f.startswith(PUBLISHED_PREFIX))
    except OSError as exc:
        print(f"cannot read {directory}: {exc}", file=sys.stderr)
        return []
    if kind:
        names = [n for n in names if card_kind(n) == kind]
    return [os.path.join(directory, n) for n in names]


def contact_sheet(paths: list, out: str, cols: int = 3, thumb: int = 220) -> str:
    """The grid as a visitor sees it: small, and all at once."""
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), (18, 18, 20))
    for i, p in enumerate(paths):
        try:
            im = Image.open(p).convert("RGB")
        except OSError:
            continue
        side = min(im.size)
        im = im.crop((((im.width - side) // 2), ((im.height - side) // 2),
                      ((im.width - side) // 2) + side,
                      ((im.height - side) // 2) + side))
        sheet.paste(im.resize((thumb, thumb), Image.LANCZOS),
                    ((i % cols) * thumb, (i // cols) * thumb))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    sheet.save(out, "JPEG", quality=90)
    return out


def report(paths: list, window: int) -> int:
    buckets = []
    for p in paths:
        try:
            buckets.append(quality.palette_bucket(Image.open(p)))
        except OSError:
            continue
    if not buckets:
        print("no published cards found")
        return 1

    recent = buckets[-window:]
    counts = collections.Counter(buckets)
    top = counts.most_common(1)[0]
    adjacent = sum(1 for a, b in zip(buckets, buckets[1:]) if a == b)

    print(f"cards measured           {len(buckets)}")
    print(f"distinct palette buckets {len(counts)} over all, "
          f"{len(set(recent))} in the last {len(recent)}")
    print(f"most common bucket       {top[0]}  ({top[1]}, "
          f"{100 * top[1] / len(buckets):.0f}% of the feed)")
    print(f"neighbours that match    {adjacent} of {max(len(buckets) - 1, 1)} "
          f"({100 * adjacent / max(len(buckets) - 1, 1):.0f}%)")
    print()
    print("distribution:")
    for bucket, n in counts.most_common():
        bar = "#" * n
        print(f"  {bucket:<26} {n:>3}  {bar}")

    # Thresholds are advisory. They describe the shape of a feed that reads as
    # varied, not a gate — the gate that can actually refuse a card is
    # quality.check_variety, which runs before the card is ever written.
    print()
    concentration = top[1] / len(buckets)
    if concentration > 0.30:
        print(f"NOTE: {top[0]} carries {100 * concentration:.0f}% of the feed "
              f"(a varied feed keeps its top bucket under ~30%)")
    if len(set(recent)) < min(len(recent), 5):
        print(f"NOTE: only {len(set(recent))} distinct looks in the last "
              f"{len(recent)} — the grid a visitor sees is repetitive")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR, help="where published cards live")
    ap.add_argument("--n", type=int, default=9,
                    help="how many recent cards on the contact sheet (default: a 3x3 grid)")
    ap.add_argument("--out", default="", help="contact sheet path (default: alongside the cards)")
    ap.add_argument("--no-sheet", action="store_true", help="print the numbers only")
    ap.add_argument("--kind", default="", choices=("", *KINDS),
                    help="measure one surface only (default: each in turn)")
    args = ap.parse_args()

    kinds = [args.kind] if args.kind else list(KINDS)
    code = 0
    for kind in kinds:
        paths = published_cards(args.dir, kind)
        if not paths:
            print(f"no published {kind or 'card'}s in {args.dir}")
            continue
        print(f"=== {kind or 'all'} " + "=" * (56 - len(kind)))
        code = report(paths, args.n) or code
        if not args.no_sheet:
            out = args.out or os.path.join(args.dir, f"variety-contact-sheet-{kind}.jpg")
            if args.out and len(kinds) > 1:
                stem, ext = os.path.splitext(args.out)
                out = f"{stem}-{kind}{ext}"
            try:
                print(f"\ncontact sheet -> {contact_sheet(paths[-args.n:], out)}")
            except OSError as exc:
                print(f"could not write contact sheet: {exc}", file=sys.stderr)
        print()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
