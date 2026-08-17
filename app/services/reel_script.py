"""Reel scripts shaped for retention rather than for length.

Reels are ranked on watch time and completion rate. A 70-second script is not a
longer version of a 20-second script — it is a worse one, because completion
rate collapses long before the payoff arrives. Everything here exists to make
the script short, single-minded and specific, and to give it a first line worth
staying for.

The rules are prompt-side, but three of them are also *checkable*, and those are
enforced in code rather than hoped for:

  * the character budget, derived from a measured speaking rate;
  * the hook, which must be the literal first line, not a metadata title;
  * the scripture anchor, which is fetched and verified rather than generated.

That last one matters most. The feed cards already refuse to put a verse on
screen unless bible-api returned it — locations are never invented, translations
are public-domain only. The Reels inherited none of that discipline, which meant
the one differentiator the account actually has stopped at the image path. Here
the LLM proposes a *reference only* (:func:`verse_card.pick_reference`) and the
API supplies the words; if the fetch fails the anchor is dropped and the script
is written without one. Scripture is never paraphrased from the model's memory.

Off by default. ``script_style = "brand"`` in ``[app]`` turns it on, so the
previous behaviour is one config key away.
"""

from __future__ import annotations

import re
from typing import Optional

from loguru import logger

from app.config import config

# Measured, not guessed: the 2026-08-16 reel ran 1253 characters in 70.3 s of
# narration, i.e. 17.8 chars/s for this voice at this delivery. Rounding to 18
# keeps the budget honest for the voice actually in use — a faster read would
# need a different number, which is why it is configurable.
DEFAULT_CHARS_PER_SECOND = 18.0
DEFAULT_TARGET_SECONDS = 20.0

# Below this a "script" is a fragment, not a devotional. Guards against a
# mis-typed config key silently producing one-sentence reels.
MIN_CHAR_BUDGET = 120

# A hook only works if it lands before the viewer's thumb moves. Asked for a
# short opening line, models happily return the entire script as one 40-word
# sentence — which reads as no hook at all, and gives the caption path nothing
# to cut on either (Phase 3 groups cues on sentence boundaries).
MAX_HOOK_WORDS = 9
MAX_SENTENCE_WORDS = 16
MIN_SENTENCES = 3

# Scripture ends on a closing quote — `…strengtheneth me.” Coffee drips…` — so a
# plain "split after .!?" treats the verse and the line following it as one
# 20-word sentence and reports a rule break that isn't there. The third
# alternative catches verses whose own punctuation is a comma or semicolon
# (`…not unto men;” Your quiet labor…`), which is common in the KJV and is still
# a spoken boundary. All lookbehinds are fixed-width, which Python requires.
_SENTENCE_END_RE = re.compile(
    r"(?<=[.!?])\s+"
    r"|(?<=[.!?][\"”'’\)\]])\s+"
    r"|(?<=[\"”'’])\s+(?=[A-Z“\"])"
)


def _cfg_float(key: str, default: float) -> float:
    raw = config.app.get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(f"{key}={raw!r} is not a number; using {default}")
        return default
    return value if value > 0 else default


def enabled() -> bool:
    """True when the brand reel discipline should shape the script."""
    return str(config.app.get("script_style", "default") or "").strip().lower() == "brand"


def verse_anchor_enabled() -> bool:
    value = config.app.get("script_verse_anchor", True)
    if isinstance(value, str):
        return value.strip().lower() not in ("", "false", "no", "0", "off")
    return bool(value)


def target_seconds() -> float:
    return _cfg_float("script_target_seconds", DEFAULT_TARGET_SECONDS)


def char_budget() -> int:
    """Characters of narration that fit the target duration at this voice's rate."""
    seconds = target_seconds()
    rate = _cfg_float("script_chars_per_second", DEFAULT_CHARS_PER_SECOND)
    return max(MIN_CHAR_BUDGET, int(seconds * rate))


def fetch_anchor(subject: str) -> Optional[tuple[str, str]]:
    """Return ``(reference, text)`` from bible-api, or None.

    Never returns model-generated scripture: the LLM only proposes a reference,
    and a reference the API will not serve is treated as not existing.
    """
    from app.services import verse_card

    try:
        reference = verse_card.pick_reference(theme=subject)
    except Exception as exc:  # noqa: BLE001 - an anchor is a bonus, never a blocker
        logger.warning(f"verse anchor: reference selection failed: {exc}")
        return None
    if not reference:
        return None

    try:
        verse = verse_card.fetch_verse(reference)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"verse anchor: fetch failed for {reference!r}: {exc}")
        return None
    if not verse or not verse.text:
        logger.info(f"verse anchor: {reference!r} not served by the API; writing without one")
        return None

    logger.info(f"verse anchor verified: {verse.reference} ({verse.translation})")
    return verse.reference, verse.text


def scene_target() -> int:
    """How many shots a reel of this length should have.

    Three to five per twenty seconds. A slow push-in on one image reads as
    considered; fifteen cuts in seventy seconds reads as a screensaver, and the
    baseline reel had exactly that.
    """
    return max(3, min(5, round(target_seconds() / 5.0)))


def guidance_block(subject: str = "", include_anchor: bool = True) -> str:
    """The reel discipline, as a prompt section appended to the system prompt.

    ``include_anchor`` is off for Article Mode, whose own contract is that every
    claim traces back to a source article. Handing that prompt a verse and
    telling it to quote it would put words in the script that the sources do not
    support — the length, hook and one-idea rules carry over, the scripture
    anchor does not.
    """
    budget = char_budget()
    seconds = int(round(target_seconds()))

    anchor_rule = (
        "7. Do not quote scripture. Refer to it only in your own words."
    )
    if include_anchor and verse_anchor_enabled():
        anchor = fetch_anchor(subject)
        if anchor:
            reference, text = anchor
            anchor_rule = (
                "7. Anchor the script on this verse, which has been verified against a "
                "public-domain source. Quote it EXACTLY as written here, or do not quote "
                "it at all — never reword, expand or complete it from memory, and never "
                "cite a different reference.\n"
                f"   {reference} — “{text}”"
            )

    return f"""
# Reel Discipline

This is a {seconds}-second vertical video. It is ranked on how many people watch
it to the end, so length is a cost, not a feature.

1. HARD LIMIT: {budget} characters of spoken words, total. Shorter is better.
   Count them before answering. Going over is a failure, not a stylistic choice.
2. The FIRST SENTENCE is the hook. It must be a SHORT, COMPLETE sentence of at
   most {MAX_HOOK_WORDS} words, ending in a full stop, that stands alone as the
   opening words the viewer hears. Make it concrete and arresting. No preamble,
   no throat-clearing, no "in today's video", no naming the topic before saying
   something about it.
3. ONE idea. Not two related ideas — one. If a sentence introduces a second
   subject, cut it.
4. Be specific, not abstract. "God speaks in ordinary moments" is a platitude
   nobody stops scrolling for. "The dishes in your sink are not keeping you from
   prayer" is a picture, and a picture holds attention.
5. The LAST line closes the loop back to the first, so the video can replay
   seamlessly. A rewatch is watch time. Do not end on a call to action.
6. Speak plainly, as one ordinary person to another. No sermon voice, no
   elevated register, no rhetorical questions stacked on each other.
{anchor_rule}
8. Write in SHORT SENTENCES — at least {MIN_SENTENCES} of them, none longer than
   about {MAX_SENTENCE_WORDS} words. Never return the whole script as a single
   run-on sentence joined by "and" or commas. Short sentences are how spoken
   narration breathes, and they are what the on-screen captions are cut on.
""".rstrip()


def sentences(script: str) -> list[str]:
    text = (script or "").strip().replace("\n", " ")
    return [s.strip() for s in _SENTENCE_END_RE.split(text) if s.strip()]


def first_line(script: str) -> str:
    parts = sentences(script)
    return parts[0] if parts else (script or "").strip()


def problems(script: str, budget: Optional[int] = None) -> list[str]:
    """Everything checkable that is wrong with this script, in plain words.

    Used both to decide whether a rewrite is worth asking for and to tell the
    model what to fix — a stated overshoot corrects far more reliably than a
    restated rule.
    """
    budget = budget or char_budget()
    text = (script or "").strip()
    parts = sentences(text)
    issues: list[str] = []

    if len(text) > budget:
        issues.append(
            f"it is {len(text)} characters, {len(text) - budget} over the hard "
            f"limit of {budget}"
        )
    if len(parts) < MIN_SENTENCES:
        issues.append(
            f"it is only {len(parts)} sentence(s); write at least {MIN_SENTENCES} "
            "short ones instead of joining everything with commas and 'and'"
        )
    if parts:
        hook_words = len(parts[0].split())
        if hook_words > MAX_HOOK_WORDS:
            issues.append(
                f"the opening sentence is {hook_words} words ({MAX_HOOK_WORDS} is "
                "the limit) — the hook has to land before the viewer scrolls"
            )
        longest = max(parts, key=lambda s: len(s.split()))
        if len(longest.split()) > MAX_SENTENCE_WORDS:
            issues.append(
                f"one sentence runs to {len(longest.split())} words (limit "
                f"{MAX_SENTENCE_WORDS}): {longest[:80]!r}"
            )
    return issues


def correction_note(issues: list[str]) -> str:
    """A correction appended to the prompt for one retry."""
    listed = "\n".join(f"- {issue}" for issue in issues)
    return (
        "\n\n# Correction\nYour previous draft is not usable:\n"
        f"{listed}\n"
        "Rewrite it completely — do not simply trim words off the end. Keep one "
        "idea, keep the closing line that loops back to the opening, and if a "
        "verse was supplied above, still quote it exactly."
    )


def trim_to_budget(script: str, budget: Optional[int] = None) -> str:
    """Last-resort trim on a sentence boundary, never mid-sentence.

    Only reached when the model has already been asked twice. A script cut in the
    middle of a clause is worse than one that runs slightly long, so a single
    over-long opening sentence is kept and reported rather than mangled.
    """
    budget = budget or char_budget()
    text = (script or "").strip()
    if len(text) <= budget:
        return text

    parts = sentences(text)
    kept: list[str] = []
    for sentence in parts:
        candidate = " ".join(kept + [sentence])
        if kept and len(candidate) > budget:
            break
        kept.append(sentence)

    trimmed = " ".join(kept).strip()
    if len(trimmed) > budget:
        logger.warning(
            f"script: opening sentence alone is {len(trimmed)} chars (budget {budget}); "
            "leaving it intact rather than cutting mid-sentence"
        )
        return trimmed

    logger.warning(
        f"script: trimmed {len(text)} -> {len(trimmed)} chars to fit the "
        f"{budget}-char budget ({len(parts) - len(kept)} sentence(s) dropped)"
    )
    return trimmed


def report(script: str) -> str:
    """One-line summary for the log, so budget misses are visible in a run."""
    budget = char_budget()
    length = len(script or "")
    seconds = length / _cfg_float("script_chars_per_second", DEFAULT_CHARS_PER_SECOND)
    issues = problems(script, budget)
    verdict = "ok" if not issues else "; ".join(issues)
    return (
        f"script: {length}/{budget} chars, ~{seconds:.1f}s spoken, "
        f"{len(sentences(script))} sentences, hook={first_line(script)!r} — {verdict}"
    )
