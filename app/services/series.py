"""Series — named, ordered runs of feed cards instead of one-off random themes.

A standalone card earns a save. A *series* earns a follow: it gives someone a
reason to come back tomorrow, because tomorrow is part 4 of something they
started. That is the whole point of this module.

References are **curated and explicit**, not LLM-proposed. This is stricter than
the random-theme path, not looser: the verse text is still fetched verbatim from
the bible API by reference (see verse_card.fetch_verse), so the LLM still never
writes scripture — it just no longer chooses which one either. A curated run also
reads as edited rather than generated, which is the difference the audience
actually notices.

State lives in storage/verse_cards/series_state.json and advances only after a
card really publishes, so a failed render does not skip part 3.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from loguru import logger

# Kept deliberately short. Seven parts is one week — long enough to build a
# habit, short enough that someone joining at part 5 does not feel too late.
# Every reference is public domain in KJV/WEB (verse_card enforces that anyway).
SERIES: list[dict] = [
    {
        "id": "psalms-for-anxious-nights",
        "title": "Psalms for Anxious Nights",
        "subject": "night sky",
        "references": [
            "Psalm 4:8", "Psalm 23:4", "Psalm 34:4", "Psalm 46:1",
            "Psalm 56:3", "Psalm 91:5", "Psalm 121:3",
        ],
    },
    {
        "id": "when-the-work-feels-small",
        "title": "When the Work Feels Small",
        "subject": "morning light through a window",
        "references": [
            "Colossians 3:23", "Galatians 6:9", "1 Corinthians 15:58",
            "Zechariah 4:10", "Ecclesiastes 9:10", "Matthew 25:21",
            "1 Thessalonians 4:11",
        ],
    },
    {
        "id": "gratitude-in-ordinary-days",
        "title": "Gratitude in Ordinary Days",
        "subject": "wildflowers in a field",
        "references": [
            "1 Thessalonians 5:18", "Psalm 118:24", "James 1:17",
            "Philippians 4:6", "Psalm 100:4", "Colossians 3:15",
            "Psalm 107:1",
        ],
    },
    {
        "id": "promises-for-hard-seasons",
        "title": "Promises for Hard Seasons",
        "subject": "storm clearing over hills",
        "references": [
            "Isaiah 41:10", "Romans 8:28", "2 Corinthians 4:17",
            "Psalm 34:18", "Isaiah 43:2", "Lamentations 3:22",
            "John 16:33",
        ],
    },
    {
        "id": "rest-for-the-weary",
        "title": "Rest for the Weary",
        "subject": "still lake at dawn",
        "references": [
            "Matthew 11:28", "Psalm 23:2", "Exodus 33:14", "Mark 6:31",
            "Psalm 62:1", "Hebrews 4:9", "Isaiah 40:31",
        ],
    },
]


def _state_path() -> str:
    d = "/influencer-automation-2.0/storage/verse_cards"
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "series_state.json")


def _load() -> dict:
    try:
        with open(_state_path(), encoding="utf-8") as fh:
            state = json.load(fh)
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def _save(state: dict) -> None:
    try:
        with open(_state_path(), "w", encoding="utf-8") as fh:
            json.dump(state, fh)
    except OSError as exc:
        logger.warning(f"could not persist series state: {exc}")


def all_series() -> list:
    """Series definitions, from the pack if it defines any."""
    from app.services import pack

    return pack.typed("series", SERIES)


def _series_by_id(series_id: str) -> Optional[dict]:
    return next((s for s in all_series() if s["id"] == series_id), None)


def current() -> Optional[dict]:
    """The next part to publish.

    Returns {series_id, title, subject, reference, part, total} or None if no
    series are defined. Starts the first series on a cold state, and rotates to
    the next one when the current run finishes.
    """
    if not SERIES:
        return None
    state = _load()
    series = _series_by_id(str(state.get("series_id", ""))) or SERIES[0]
    index = int(state.get("index", 0) or 0)
    if index >= len(series["references"]):
        # Finished: roll to the next series, wrapping at the end.
        nxt = (SERIES.index(series) + 1) % len(SERIES)
        series, index = SERIES[nxt], 0
    return {
        "series_id": series["id"],
        "title": series["title"],
        "subject": series.get("subject") or None,
        "reference": series["references"][index],
        "part": index + 1,
        "total": len(series["references"]),
    }


def advance(series_id: str) -> None:
    """Move to the next part. Call only after a card has actually published, so
    a failed render retries the same part rather than silently skipping it."""
    series = _series_by_id(series_id)
    if not series:
        return
    state = _load()
    index = int(state.get("index", 0) or 0) if state.get("series_id") == series_id else 0
    index += 1
    if index >= len(series["references"]):
        nxt = (SERIES.index(series) + 1) % len(SERIES)
        _save({"series_id": SERIES[nxt]["id"], "index": 0})
        logger.info(f"series '{series['title']}' complete; next up "
                    f"'{SERIES[nxt]['title']}'")
        return
    _save({"series_id": series_id, "index": index})


def label(part: dict) -> str:
    """Natural-case series line, e.g. 'Psalms for Anxious Nights · Part 3 of 7'.

    Kept in sentence case here and uppercased only where it is *drawn*: the same
    string leads the caption, and `.title()` on it would render "3 Of 7".
    """
    return f"{part['title']}  ·  Part {part['part']} of {part['total']}"


# --- reels -------------------------------------------------------------------
#
# The cards run as finite, curated series; Reels cannot, because their subject
# comes from whatever the article worker found that morning. So a Reel series is
# an open-ended *format* rather than a fixed run: same name, same treatment, a
# number that keeps counting. "Ordinary Grace, no. 4" tells a viewer this is a
# thing with a back catalogue, and a follow is a subscription to a thing — which
# is what converts a viewer into a follower, and a standalone reel never does.
#
# No total, deliberately: "no. 4 of 7" on an open format would be a lie, and a
# format that visibly ends gives no reason to follow.

# Its own file, not a key in the card state: the card series' advance() writes
# a whole fresh dict, so a reel counter living beside it would be silently
# dropped the next time a card published.
_REEL_STATE_KEY = "reel_number"


def _reel_state_path() -> str:
    d = "/influencer-automation-2.0/storage/verse_cards"
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "reel_series_state.json")


def _reel_load() -> dict:
    try:
        with open(_reel_state_path(), encoding="utf-8") as fh:
            state = json.load(fh)
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def _reel_save(state: dict) -> None:
    try:
        with open(_reel_state_path(), "w", encoding="utf-8") as fh:
            json.dump(state, fh)
    except OSError as exc:
        logger.warning(f"could not persist reel series state: {exc}")


def reel_title() -> str:
    """The Reel series name, or "" when the feature is off."""
    from app.config import config

    return str(config.app.get("reel_series_title", "") or "").strip()


def reel_current() -> Optional[dict]:
    """The next Reel's place in the series, or None when disabled."""
    title = reel_title()
    if not title:
        return None
    state = _reel_load()
    try:
        published = int(state.get(_REEL_STATE_KEY, 0) or 0)
    except (TypeError, ValueError):
        published = 0
    return {"title": title, "number": published + 1}


def reel_advance() -> None:
    """Count a published Reel.

    Called only after a real publish, for the same reason the card series does:
    a failed render must not burn a number and leave a gap in the run.
    """
    if not reel_title():
        return
    state = _reel_load()
    try:
        published = int(state.get(_REEL_STATE_KEY, 0) or 0)
    except (TypeError, ValueError):
        published = 0
    state[_REEL_STATE_KEY] = published + 1
    _reel_save(state)


def reel_label(part: Optional[dict]) -> str:
    """e.g. 'Ordinary Grace, no. 4'. Empty string when the series is off."""
    if not part:
        return ""
    return f"{part['title']}, no. {part['number']}"
