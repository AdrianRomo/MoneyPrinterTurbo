"""Cut captions on speech, not on punctuation.

`subtitle_provider = "edge"` derives cue boundaries from TTS word boundaries and
groups them on punctuation. A sentence without a comma in it is therefore one
cue, however long it runs — which is how the 2026-08-16 reel produced a single
8.5-second cue of 159 characters that rendered as a five-line wall.

Punctuation is a writing convention. What a viewer reads in a Reel is closer to
speech: two to four words at a time, roughly a second each, changing on the
beat of the narration. That is what this module produces from word-level
timings.

Two rules do most of the work:

  * **A pause is a cut.** A gap between words is the speaker taking a breath,
    and it is a better boundary than any comma, so a cue always ends there.
  * **A cue never blinks.** Cues are held until the next one starts whenever the
    gap between them is short, so captions replace each other rather than
    flashing off and back on.

Off by default; ``subtitle_cadence = "words"`` selects it.
"""

from __future__ import annotations

from typing import Iterable

from loguru import logger

from app.config import config

MIN_WORDS = 2
MAX_WORDS = 4
MAX_CHARS = 30          # what fits two rendered lines at caption size
MIN_SECONDS = 0.8
MAX_SECONDS = 1.6
PAUSE_GAP = 0.35        # a gap this long is a breath, and a breath is a cut
HOLD_GAP = 0.40         # gaps under this are closed so captions never blink


def enabled() -> bool:
    return str(config.app.get("subtitle_cadence", "punctuation") or "").strip().lower() == "words"


def _cfg_float(key: str, default: float) -> float:
    raw = config.app.get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(f"{key}={raw!r} is not a number; using {default}")
        return default
    return value if value > 0 else default


def _cfg_int(key: str, default: int) -> int:
    return int(_cfg_float(key, float(default)))


def limits() -> tuple[int, int, float, float]:
    return (
        _cfg_int("subtitle_cue_max_words", MAX_WORDS),
        _cfg_int("subtitle_cue_max_chars", MAX_CHARS),
        _cfg_float("subtitle_cue_min_seconds", MIN_SECONDS),
        _cfg_float("subtitle_cue_max_seconds", MAX_SECONDS),
    )


def group(words: Iterable) -> list[dict]:
    """Group word-level timings into short cues.

    ``words`` is any iterable of objects with ``word``, ``start`` and ``end``
    (faster-whisper's Word), or 3-tuples of the same.
    """
    max_words, max_chars, min_seconds, max_seconds = limits()

    items = []
    for word in words:
        if isinstance(word, (tuple, list)):
            text, start, end = word[0], float(word[1]), float(word[2])
        else:
            text, start, end = word.word, float(word.start), float(word.end)
        text = str(text).strip()
        if text:
            items.append((text, start, end))
    if not items:
        return []

    cues: list[dict] = []
    current: list[tuple[str, float, float]] = []

    def flush():
        if not current:
            return
        cues.append({
            "msg": " ".join(w[0] for w in current),
            "start_time": current[0][1],
            "end_time": current[-1][2],
        })
        current.clear()

    for index, item in enumerate(items):
        text, start, end = item
        candidate = " ".join([w[0] for w in current] + [text])
        if current and (
            len(current) >= max_words
            or len(candidate) > max_chars
            or (end - current[0][1]) > max_seconds
        ):
            flush()
        current.append(item)

        # A pause after this word is a natural cut — take it, unless doing so
        # would strand a single word that could still sit with the next one.
        following = items[index + 1] if index + 1 < len(items) else None
        if following is not None and (following[1] - end) >= PAUSE_GAP:
            flush()

    flush()

    cues = _merge_orphans(cues, max_words, max_chars, max_seconds)
    _hold_until_next(cues, min_seconds, max_seconds)
    return cues


def _fits(text: str, span: float, max_chars: int, max_seconds: float) -> bool:
    # Word count is deliberately not a constraint here. A one-word cue reads as
    # a stutter, and "in the rustle of / leaves," is worse than a five-word cue
    # that still fits the box and the clock — which are the limits that actually
    # decide whether a caption is readable.
    return len(text) <= max_chars and span <= max_seconds


def _merge_orphans(cues: list[dict], max_words: int, max_chars: int,
                   max_seconds: float) -> list[dict]:
    """Fold a stranded single word into a neighbour, backwards or forwards.

    Backwards is preferred: it keeps the phrase's own rhythm. Forwards is the
    fallback, and the only option for a leading orphan.
    """
    merged: list[dict] = []
    for cue in cues:
        previous = merged[-1] if merged else None
        if (
            previous is not None
            and len(cue["msg"].split()) < MIN_WORDS
            and _fits(previous["msg"] + " " + cue["msg"],
                      cue["end_time"] - previous["start_time"],
                      max_chars, max_seconds)
        ):
            previous["msg"] += " " + cue["msg"]
            previous["end_time"] = cue["end_time"]
            continue
        merged.append(cue)

    # Second pass for orphans that had no room behind them: try the cue ahead.
    result: list[dict] = []
    index = 0
    while index < len(merged):
        cue = merged[index]
        following = merged[index + 1] if index + 1 < len(merged) else None
        if (
            following is not None
            and len(cue["msg"].split()) < MIN_WORDS
            and _fits(cue["msg"] + " " + following["msg"],
                      following["end_time"] - cue["start_time"],
                      max_chars, max_seconds)
        ):
            result.append({
                "msg": cue["msg"] + " " + following["msg"],
                "start_time": cue["start_time"],
                "end_time": following["end_time"],
            })
            index += 2
            continue
        result.append(cue)
        index += 1

    return result


def _hold_until_next(cues: list[dict], min_seconds: float, max_seconds: float) -> None:
    """Close short gaps and lift very short cues to a readable minimum.

    Holding is capped at ``max_seconds`` so that "no cue is longer than the
    ceiling" stays true of the finished file — otherwise a 1.5s cue plus a 0.3s
    hold silently lands at 1.8s and the invariant is only true of the grouping,
    not of what actually renders.
    """
    for index, cue in enumerate(cues):
        following = cues[index + 1] if index + 1 < len(cues) else None
        ceiling = following["start_time"] if following else None
        latest = cue["start_time"] + max_seconds
        if ceiling is not None:
            latest = min(latest, ceiling)

        if ceiling is not None and 0 < (ceiling - cue["end_time"]) <= HOLD_GAP:
            cue["end_time"] = min(ceiling, latest)
        if (cue["end_time"] - cue["start_time"]) < min_seconds:
            wanted = cue["start_time"] + min_seconds
            cue["end_time"] = min(wanted, ceiling) if ceiling is not None else wanted
        cue["end_time"] = max(cue["end_time"], cue["start_time"])


def report(cues: list[dict]) -> str:
    """What the cadence produced, so a bad grouping is visible in the run log."""
    if not cues:
        return "cadence: no cues"
    _, max_chars, _, max_seconds = limits()
    durations = [c["end_time"] - c["start_time"] for c in cues]
    longest = max(cues, key=lambda c: c["end_time"] - c["start_time"])
    over_time = sum(1 for d in durations if d > max_seconds + 0.05)
    over_chars = sum(1 for c in cues if len(c["msg"]) > max_chars)
    return (
        f"cadence: {len(cues)} cues, {sum(durations) / len(durations):.2f}s avg, "
        f"longest {max(durations):.2f}s {longest['msg']!r}, "
        f"{over_time} over {max_seconds}s, {over_chars} over {max_chars} chars"
    )
