"""Hashtag sets, rotation, and reach-driven selection.

There is no trending-hashtag API. `ig_hashtag_search` is Facebook-Login-only
(this account is Instagram Login / MEDIA_CREATOR and gets error 100), and even
that endpoint returns media for a hashtag you already name — Meta publishes no
"what is trending" endpoint at all. Everything claiming otherwise scrapes.

So instead of guessing at global trends, we rotate curated niche sets and let
*our own* reach data pick the winners. Insights are collected out-of-band by
collect-insights.sh (host side, so the Instagram token stays on docker-devops)
and land in scores.json, which this module reads.

Selection is epsilon-greedy: mostly exploit the best-performing set, sometimes
explore a random one so a set that got unlucky early is not written off
forever. With no data it degrades to least-recently-used rotation.

Instagram's own guidance is 3–5 relevant hashtags, and hashtags are a weak
ranking signal since 2024 — the keyword-bearing first line of the caption is
searchable and matters more. Sets are kept deliberately small for that reason.
"""

from __future__ import annotations

import json
import os
import random
from typing import Optional

from loguru import logger

STORAGE = "/influencer-automation-2.0/storage/hashtags"

# Curated for this account's niche. Small on purpose (see module docstring).
SETS: dict[str, dict] = {
    "devotional": {
        "label": "Daily devotional",
        "tags": ["#dailydevotional", "#bibleverseoftheday", "#scripture", "#godsword", "#devotional"],
    },
    "encouragement": {
        "label": "Encouragement",
        "tags": ["#christianencouragement", "#faithoverfear", "#hopeinjesus", "#trustgod", "#encouragement"],
    },
    "gratitude": {
        "label": "Gratitude",
        "tags": ["#gratitude", "#gratefulheart", "#countyourblessings", "#thankful", "#blessed"],
    },
    "morning": {
        "label": "Morning quiet time",
        "tags": ["#morningdevotion", "#quiettime", "#morningprayer", "#timewithgod", "#startyourday"],
    },
    "scripture_art": {
        "label": "Scripture art",
        "tags": ["#bibleverse", "#scriptureart", "#christianart", "#wordofgod", "#versedesign"],
    },
    "peace": {
        "label": "Peace and rest",
        "tags": ["#peaceofgod", "#restinhim", "#findingpeace", "#stillness", "#christianmeditation"],
    },
    "everyday_faith": {
        "label": "Everyday faith",
        "tags": ["#faithintheordinary", "#everydayfaith", "#simplefaith", "#livingbyfaith", "#ordinarydays"],
    },
    "hope": {
        "label": "Hope",
        "tags": ["#hopeinchrist", "#newmercies", "#freshstart", "#godisgood", "#hopeful"],
    },
}

# Saves and shares are far stronger ranking signals than a like, so the
# composite weights them well above raw reach.
SCORE_WEIGHTS = {"reach": 1.0, "saved": 12.0, "shares": 20.0, "likes": 2.0, "comments": 6.0}

MIN_SAMPLES = 2      # a set needs this many measured posts before it can be exploited
EPSILON = 0.25       # fraction of posts that explore rather than exploit


def _path(name: str) -> str:
    os.makedirs(STORAGE, exist_ok=True)
    return os.path.join(STORAGE, name)


def _load(name: str, default):
    try:
        with open(_path(name), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def _save(name: str, data) -> None:
    try:
        with open(_path(name), "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except OSError as exc:
        logger.warning(f"could not write {name}: {exc}")


def score_of(metrics: dict) -> float:
    return sum(float(metrics.get(k, 0) or 0) * w for k, w in SCORE_WEIGHTS.items())


def set_scores() -> dict[str, dict]:
    """{set_id: {samples, mean_score}} built from collected insights."""
    samples = _load("samples.json", [])
    grouped: dict[str, list[float]] = {}
    for s in samples:
        sid = s.get("set_id")
        if sid in SETS and isinstance(s.get("metrics"), dict):
            grouped.setdefault(sid, []).append(score_of(s["metrics"]))
    return {sid: {"samples": len(v), "mean_score": sum(v) / len(v)} for sid, v in grouped.items() if v}


def choose_set(explicit: Optional[str] = None) -> str:
    if explicit and explicit in SETS:
        return explicit

    scores = set_scores()
    eligible = {sid: d for sid, d in scores.items() if d["samples"] >= MIN_SAMPLES}

    if eligible and random.random() > EPSILON:
        best = max(eligible, key=lambda s: eligible[s]["mean_score"])
        logger.info(
            f"hashtag set '{best}' chosen by reach "
            f"(mean {eligible[best]['mean_score']:.1f} over {eligible[best]['samples']} posts)"
        )
        return best

    # Explore — or no data yet. Prefer whatever has been used least recently, so
    # early rotation is even rather than random-clumped.
    recent = _load("recent.json", [])
    unused = [s for s in SETS if s not in recent]
    if unused:
        choice = random.choice(unused)
    else:
        # recent is most-recent-last; the front of the list is the stalest.
        choice = next((s for s in recent if s in SETS), random.choice(list(SETS)))
    logger.info(f"hashtag set '{choice}' chosen by rotation ({'no data yet' if not eligible else 'explore'})")
    return choice


def mark_used(set_id: str, keep: int = 6) -> None:
    recent = [s for s in _load("recent.json", []) if s != set_id]
    recent.append(set_id)
    _save("recent.json", recent[-keep:])


def tags_for(set_id: str) -> list[str]:
    return list(SETS.get(set_id, {}).get("tags", []))


def record_sample(set_id: str, media_id: str, metrics: dict) -> None:
    """Called by the insights collector once a post's numbers are in."""
    samples = _load("samples.json", [])
    if any(s.get("media_id") == media_id for s in samples):
        return
    samples.append({"set_id": set_id, "media_id": media_id, "metrics": metrics})
    _save("samples.json", samples[-500:])


# --- caption assembly --------------------------------------------------------

# Instagram indexes caption text for search; the first line is what surfaces.
# Leading with a keyword-bearing sentence is worth more than the hashtag block.
KEYWORD_LEADS = {
    "devotional": "Daily Bible verse and devotional encouragement",
    "encouragement": "Bible verse for encouragement when life feels heavy",
    "gratitude": "A Bible verse about gratitude and thankfulness",
    "morning": "Morning Bible verse for your quiet time",
    "scripture_art": "Bible verse of the day",
    "peace": "A Bible verse about peace and rest",
    "everyday_faith": "Finding faith in ordinary, everyday moments",
    "hope": "A Bible verse about hope for today",
}


def build_caption(verse_text: str, reference: str, translation: str,
                  set_id: Optional[str] = None) -> tuple[str, str]:
    """Return (caption, set_id). Keyword line first, verse, then a small tag set."""
    set_id = choose_set(set_id)
    lead = KEYWORD_LEADS.get(set_id, "Bible verse of the day")
    tags = " ".join(tags_for(set_id))
    caption = (
        f"{lead} — {reference}\n\n"
        f"“{verse_text}”\n"
        f"— {reference} ({translation})\n\n"
        f"{tags}"
    )
    return caption, set_id
