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
import math
import os
import random
import re
import statistics
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

# A set needs this many measured posts before it can be exploited. Two was far
# too few: at a reach of 2-9 with zero saves and zero shares on every post so
# far, the composite score is reach-only noise, and exploiting on two samples of
# noise locks the rotation onto whichever set happened to get shown twice.
#
# Five was still too few. Measured spread across the first 17 samples was reach
# 6-108 — a single Reel carried 30% of all reach ever collected. A five-sample
# mean drawn from that distribution is dominated by whether the outlier landed
# in the set, so a threshold on sample COUNT alone cannot tell signal from luck
# no matter where it is set. Count is necessary and not sufficient, hence
# _is_separable() below.
MIN_SAMPLES = 12
EPSILON = 0.25       # fraction of posts that explore rather than exploit

# How far the leader must stand clear of the runner-up, in combined standard
# errors, before its lead is treated as real. 1.0 is deliberately permissive —
# this is a hashtag rotation, not a drug trial, and the cost of exploring a
# slightly worse set is one post. The cost of the alternative is what was
# happening: the whole rotation collapsing onto one lucky sample forever.
SEPARATION_SIGMAS = 1.0


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
    """{set_id: {samples, mean_score, stderr}} built from collected insights."""
    samples = _load("samples.json", [])
    grouped: dict[str, list[float]] = {}
    for s in samples:
        sid = s.get("set_id")
        if sid in _sets() and isinstance(s.get("metrics"), dict):
            grouped.setdefault(sid, []).append(score_of(s["metrics"]))
    return {
        sid: {
            "samples": len(v),
            "mean_score": sum(v) / len(v),
            # Standard error of the mean. One sample has no spread to speak of,
            # so it reports infinite uncertainty rather than a confident zero —
            # otherwise a single lucky post looks like a perfectly measured one.
            "stderr": (statistics.stdev(v) / math.sqrt(len(v))) if len(v) > 1 else float("inf"),
        }
        for sid, v in grouped.items()
        if v
    }


def _is_separable(eligible: dict[str, dict]) -> bool:
    """True when the leader's lead is bigger than the noise around it.

    Guards against the failure this account actually hit: reach that varies by
    more than an order of magnitude between posts of the SAME set, so ranking
    sets by mean is ranking them by which one caught an outlier.
    """
    if len(eligible) < 2:
        return False
    ranked = sorted(eligible.values(), key=lambda d: d["mean_score"], reverse=True)
    best, runner_up = ranked[0], ranked[1]
    lead = best["mean_score"] - runner_up["mean_score"]
    spread = math.hypot(best["stderr"], runner_up["stderr"])
    if not math.isfinite(spread):
        # A single-sample set carries infinite uncertainty by construction, so
        # nothing it is involved in can be called.
        return False
    if spread == 0:
        # Both sets scored identically on every post they ran. There is no noise
        # left to hide a real difference, so any lead at all is a real one —
        # treating this as "not separable" would refuse to act on the cleanest
        # evidence the selector can ever get.
        return lead > 0
    return lead >= SEPARATION_SIGMAS * spread


def _sets() -> dict:
    """Hashtag sets, from the pack if it defines any."""
    from app.services import pack

    return pack.typed("hashtags.sets", SETS)


def choose_set(explicit: Optional[str] = None) -> str:
    sets = _sets()
    if explicit and explicit in sets:
        return explicit

    scores = set_scores()
    eligible = {sid: d for sid, d in scores.items() if d["samples"] >= MIN_SAMPLES}

    if eligible and _is_separable(eligible) and random.random() > EPSILON:
        best = max(eligible, key=lambda s: eligible[s]["mean_score"])
        logger.info(
            f"hashtag set '{best}' chosen by reach "
            f"(mean {eligible[best]['mean_score']:.1f} over {eligible[best]['samples']} posts)"
        )
        return best

    # Explore — or no data yet. Prefer whatever has been used least recently, so
    # early rotation is even rather than random-clumped.
    recent = _load("recent.json", [])
    unused = [s for s in sets if s not in recent]
    if unused:
        choice = random.choice(unused)
    else:
        # recent is most-recent-last; the front of the list is the stalest.
        choice = next((s for s in recent if s in sets), random.choice(list(sets)))
    if not eligible:
        why = "no data yet"
    elif not _is_separable(eligible):
        why = "no separable winner"
    else:
        why = "explore"
    logger.info(f"hashtag set '{choice}' chosen by rotation ({why})")
    return choice


def mark_used(set_id: str, keep: int = 6) -> None:
    recent = [s for s in _load("recent.json", []) if s != set_id]
    recent.append(set_id)
    _save("recent.json", recent[-keep:])


def tags_for(set_id: str) -> list[str]:
    return list(_sets().get(set_id, {}).get("tags", []))


def record_sample(set_id: str, media_id: str, metrics: dict,
                  variant: Optional[dict] = None,
                  kind: Optional[str] = None,
                  local_hour: Optional[int] = None) -> None:
    """Called by the insights collector once a post's numbers are in."""
    samples = _load("samples.json", [])
    if any(s.get("media_id") == media_id for s in samples):
        return
    sample = {"set_id": set_id, "media_id": media_id, "metrics": metrics}
    if variant:
        sample["variant"] = variant
    if kind:
        sample["kind"] = kind
    if local_hour is not None:
        sample["local_hour"] = int(local_hour)
    samples.append(sample)
    _save("samples.json", samples[-500:])


# --- reach by dimension -------------------------------------------------------
#
# Until now this module scored exactly ONE variable — which hashtag set was used
# — and its own docstring says hashtags have been a weak ranking signal since
# 2024. Meanwhile format, cover variant, subject, verse and posting hour, all of
# which plausibly matter more, were recorded at publish time and never read.
#
# These are read-only reports. Nothing selects on them yet, and that is
# deliberate: with a handful of samples any of these breakdowns is noise, and
# wiring a bandit to noise is how the hashtag rotation locked onto whichever set
# happened to get shown twice. They exist so the priors in the runbook — the
# posting windows especially, which it explicitly flags as unmeasured — can stop
# being priors once there is enough data to read.

_MIN_DIMENSION_SAMPLES = 3


def _dimension_of(sample: dict, dimension: str):
    """Pull one axis out of a sample, wherever it happens to live."""
    if dimension in ("kind", "local_hour"):
        return sample.get(dimension)
    return (sample.get("variant") or {}).get(dimension)


def reach_by(dimension: str, samples: Optional[list] = None) -> dict:
    """{value: {samples, mean_score, mean_reach, saves, shares}} for one axis.

    `dimension` is a ledger field ("kind", "local_hour") or a variant key
    ("subject", "cover_variant", "reference", "series", ...).
    """
    if samples is None:
        samples = _load("samples.json", [])
    grouped: dict = {}
    for s in samples:
        value = _dimension_of(s, dimension)
        if value is None or not isinstance(s.get("metrics"), dict):
            continue
        grouped.setdefault(str(value), []).append(s["metrics"])
    out = {}
    for value, rows in grouped.items():
        n = len(rows)
        out[value] = {
            "samples": n,
            "mean_score": round(sum(score_of(m) for m in rows) / n, 2),
            "mean_reach": round(sum(float(m.get("reach", 0) or 0) for m in rows) / n, 2),
            "saves": sum(int(m.get("saved", 0) or 0) for m in rows),
            "shares": sum(int(m.get("shares", 0) or 0) for m in rows),
            # Below this, a breakdown is a story about one or two posts.
            "readable": n >= _MIN_DIMENSION_SAMPLES,
        }
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["mean_score"]))


def dimension_report(dimensions: Optional[list] = None) -> dict:
    """Every axis worth looking at, plus how much of the data carries it."""
    samples = _load("samples.json", [])
    dimensions = dimensions or ["kind", "local_hour", "subject", "cover_variant",
                                "series", "translation", "hashtag_set"]
    report = {"samples": len(samples), "dimensions": {}}
    for dim in dimensions:
        rows = reach_by(dim, samples)
        if rows:
            report["dimensions"][dim] = {
                "coverage": sum(r["samples"] for r in rows.values()),
                "readable": any(r["readable"] for r in rows.values()),
                "values": rows,
            }
    return report


# --- retention ---------------------------------------------------------------
#
# Watch time is the number Reels are actually ranked on, and the only one that
# means anything at a reach of 2-9: reach and follows are too sparse to read,
# but "did this person watch to the end" is measurable on a single viewer.

WATCH_TIME_MS = "ig_reels_avg_watch_time"
TOTAL_WATCH_MS = "ig_reels_video_view_total_time"


def completion_of(sample: dict) -> Optional[float]:
    """Average watch time as a fraction of the reel's length, if both are known.

    This is the closest thing to a completion rate the API will give us — and it
    is why the variant records the video's duration at publish time. Without the
    denominator, "watched 8 seconds" is unreadable: superb for a 9-second reel,
    dismal for a 70-second one.
    """
    watch_ms = (sample.get("metrics") or {}).get(WATCH_TIME_MS)
    seconds = (sample.get("variant") or {}).get("video_seconds")
    try:
        watch_seconds = float(watch_ms) / 1000.0
        seconds = float(seconds)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    return watch_seconds / seconds


def retention_report() -> dict:
    """Watch time grouped by the treatment that produced it."""
    samples = [s for s in _load("samples.json", [])
               if (s.get("metrics") or {}).get(WATCH_TIME_MS) is not None]
    if not samples:
        return {"samples": 0, "note": "no retention data collected yet"}

    by_variant: dict[str, list[dict]] = {}
    for sample in samples:
        variant = sample.get("variant") or {}
        # Group by treatment, not by post: "brand script + brand captions" is
        # the question, and the individual reel is only a sample of it.
        key = "|".join(
            f"{k}={variant.get(k)}" for k in ("script_style", "subtitle_renderer", "subtitle_cadence")
        ) or "unrecorded"
        by_variant.setdefault(key, []).append(sample)

    out = {"samples": len(samples), "variants": {}}
    for key, group in by_variant.items():
        watch = [float(s["metrics"][WATCH_TIME_MS]) / 1000.0 for s in group]
        completions = [c for c in (completion_of(s) for s in group) if c is not None]
        entry = {
            "samples": len(group),
            "mean_watch_seconds": round(sum(watch) / len(watch), 2),
        }
        if completions:
            entry["mean_completion"] = round(sum(completions) / len(completions), 3)
        out["variants"][key] = entry
    return out


# --- caption assembly --------------------------------------------------------

# Instagram indexes caption text for search; the first line is what surfaces.
# Leading with a keyword-bearing sentence is worth more than the hashtag block.
#
# Each set carries SEVERAL leads rather than one. A fixed lead per set meant the
# same eight sentences cycled forever: the one field Instagram actually searches
# was carrying no new keywords, and two posts from the same set read as a
# template. They stay keyword-bearing — that reasoning was right — they just
# stop being identical.
KEYWORD_LEADS: dict[str, list[str]] = {
    "devotional": [
        "Daily Bible verse and devotional encouragement",
        "A short devotional for today",
        "Today's scripture reading, in one verse",
    ],
    "encouragement": [
        "Bible verse for encouragement when life feels heavy",
        "A verse for the days that ask too much of you",
        "Encouragement from scripture for a hard week",
    ],
    "gratitude": [
        "A Bible verse about gratitude and thankfulness",
        "Scripture for a thankful heart",
        "A verse worth reading slowly before you say thank you",
    ],
    "morning": [
        "Morning Bible verse for your quiet time",
        "A verse to start the morning on",
        "Scripture for the first quiet minutes of the day",
    ],
    "scripture_art": [
        "Bible verse of the day",
        "Scripture, set plainly",
        "A verse worth keeping where you can see it",
    ],
    "peace": [
        "A Bible verse about peace and rest",
        "Scripture for an anxious mind",
        "A verse for when you cannot settle",
    ],
    "everyday_faith": [
        "Finding faith in ordinary, everyday moments",
        "A verse for an ordinary Tuesday",
        "Scripture for the unremarkable parts of the week",
    ],
    "hope": [
        "A Bible verse about hope for today",
        "Scripture for holding on a little longer",
        "A verse about hope when it is in short supply",
    ],
}

# Comments are the strongest ranking signal Instagram has, and effort is what
# kills reply rates — every one of these is answerable in a word or two without
# the reader having to compose anything. Same principle as carousel.QUESTIONS.
QUESTIONS = [
    "Which line did you need today?",
    "Who came to mind while you read this?",
    "One word for where you are this week?",
    "Have you sat with this verse before?",
    "What would change if you believed this today?",
]

# Saves are weighted x12 and shares x20 in SCORE_WEIGHTS, and until now nothing
# in the content ever asked for either. An explicit ask is the cheapest lever on
# the two metrics the scoring already says matter most.
SAVE_ASKS = [
    "Save this for the day it stops being theoretical.",
    "Save it — you will want it on a worse morning than this one.",
    "Save this one, and pass it on if it is not just for you.",
    "Keep this where you will see it again.",
    "Save it for later — this one keeps.",
]

# A reflection is commentary, never scripture. These carry the format when the
# model is unavailable or its output fails the guards below — the caption must
# never fall back to being a bare re-print of the image.
REFLECTION_FALLBACKS = [
    "Read it once for the sense of it, then once more slowly.",
    "Nothing here asks you to feel better first. It just says what is true.",
    "This was written to people who were not coping either.",
    "It is worth noticing what this verse does not ask of you.",
    "Short enough to carry around all day, which is probably the point.",
]

# The model may comment on scripture; it may never write, quote, paraphrase or
# cite it. A reflection that quotes is indistinguishable from a misquote to a
# reader, and a misquote is the error that costs credibility on a faith account.
_REF_PATTERN = re.compile(r"\b\d?\s*[A-Z][a-z]+\.?\s+\d+[:.]\d+")
_MAX_REFLECTION_CHARS = 190


def reflection(verse_text: str, reference: str) -> str:
    """One line of commentary on the verse, or "" if it cannot be trusted.

    Guarded the same way carousel.science_note is: the prompt constrains it, and
    anything that comes back looking like scripture is dropped rather than
    repaired. Returning "" is safe — the caller substitutes a fallback line.
    """
    from app.services import llm

    prompt = (
        "Write ONE short sentence of plain reflection on the verse below, for "
        "an Instagram caption. Maximum 22 words — shorter is better. Speak to "
        "the reader as a person having an ordinary week. Do NOT quote the "
        "verse, do not paraphrase it, do not cite any chapter or verse number, "
        "and do not use quotation marks, hashtags or emoji. No preamble.\n"
        f"Verse ({reference}): {verse_text}"
    )
    try:
        raw = (llm._generate_response(prompt) or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"reflection generation failed: {exc}")
        return ""
    line = " ".join(raw.splitlines()[0].split()) if raw else ""
    line = line.strip().strip("*").strip()
    if not line:
        return ""
    if len(line) > _MAX_REFLECTION_CHARS:
        logger.warning("reflection too long; dropping it")
        return ""
    if any(ch in line for ch in '"“”'):
        logger.warning("reflection quoted something; dropping it rather than risk a misquote")
        return ""
    if _REF_PATTERN.search(line):
        logger.warning("reflection cited a reference; dropping it rather than risk a wrong one")
        return ""
    if "#" in line:
        logger.warning("reflection contained a hashtag; dropping it")
        return ""
    return line


def rotate(bucket: str, options: list[str], keep: int = 8) -> str:
    """Pick from `options`, avoiding whatever was used most recently.

    Same least-recently-used shape as choose_set's explore branch. Without it,
    random.choice over five lines repeats within a couple of posts, which is
    exactly the templated feel these banks exist to remove.
    """
    if not options:
        return ""
    recent = _load("recent_copy.json", {})
    used = [s for s in recent.get(bucket, []) if s in options]
    fresh = [s for s in options if s not in used]
    choice = random.choice(fresh) if fresh else random.choice(options)
    recent[bucket] = ([s for s in used if s != choice] + [choice])[-keep:]
    _save("recent_copy.json", recent)
    return choice


def build_caption(verse_text: str, reference: str, translation: str,
                  set_id: Optional[str] = None, reflect: bool = True) -> tuple[str, str]:
    """Return (caption, set_id).

    Shape: keyword lead, the verse, one line of reflection, a question, a save
    ask, then a small tag set. The previous version led with a keyword line and
    then reprinted the verse that is already rendered on the image — nothing in
    it gave a reader a reason to stop, comment, or save, which are the three
    things the ranking (and SCORE_WEIGHTS) actually reward.

    `reflect=False` skips the model call for callers that cannot afford it.
    """
    from app.services import pack

    set_id = choose_set(set_id)
    leads = (pack.typed("captions.keyword_leads", KEYWORD_LEADS).get(set_id)
             or ["Bible verse of the day"])
    lead = rotate(f"lead:{set_id}", leads)
    note = reflection(verse_text, reference) if reflect else ""
    if not note:
        note = rotate("reflection", pack.typed("captions.reflection_fallbacks", REFLECTION_FALLBACKS))
    blocks = [
        f"{lead} — {reference}",
        f"“{verse_text}”\n— {reference} ({translation.upper()})",
        note,
        rotate("question", pack.typed("captions.questions", QUESTIONS)),
        rotate("save_ask", pack.typed("captions.save_asks", SAVE_ASKS)),
        " ".join(tags_for(set_id)),
    ]
    return "\n\n".join(b for b in blocks if b), set_id
