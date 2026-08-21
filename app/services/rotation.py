"""Least-recently-used rotation, shared by everything that picks from a pool.

The account published `mountains` three times while 45 carousel subjects had
never run once, because the picker was `random.choice`. Random selection *with
replacement* does not spread a pool out: over 54 draws from 30 candidates the
birthday paradox makes collisions the expected outcome, not the unlucky one.
carousel.py fixed that for its own subjects in f3f1ba2. This module is that fix
extracted, because the same bug was still live on two other axes:

  - the verse-card BACKGROUND subject (`random.choice(background_subjects())`)
  - the background STYLE, which was not a choice at all — one fixed suffix on
    every image the account has ever produced

Measured on the 41 cards published 2026-08-14..27, before this landed:

    saturation   never above 33/100   -> 100% "muted", no exceptions
    lightness    never above 54/100   -> nothing bright, ever
    hue family   66% in just two      -> amber and blue/teal carried the feed

Those are the fingerprints of a fixed style string, not of taste.

THE ORDERING RULE. Never-used candidates come first, shuffled among themselves;
then the rest, oldest-use first. A caller that walks the ranked list in order
therefore exhausts the whole pool before anything repeats, and the shuffle stops
a fresh pool from always opening in alphabetical order.

State lives in one small JSON list per axis, oldest first. It is advisory: a
missing or corrupt file means "nothing has been used yet", which degrades to the
old behaviour rather than taking a live unattended run down with it.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Iterable, Optional

from loguru import logger


def load_history(path: str, limit: Optional[int] = None) -> list:
    """Items already used, oldest first. Never raises.

    The window must be at least as long as the pool it tracks, or rotation
    starts repeating while candidates that have never run are still waiting.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            history = list(json.load(fh))
    except (OSError, ValueError):
        return []
    return history[-limit:] if limit else history


def remember(path: str, item: Any, keep: int = 500) -> None:
    """Record `item` as the most recently used, de-duplicating older uses."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except OSError as exc:  # noqa: BLE001
        logger.warning(f"could not create state dir for {path}: {exc}")
        return
    recent = [x for x in load_history(path, keep) if x != item]
    recent.append(item)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(recent[-keep:], fh)
    except OSError as exc:
        logger.warning(f"could not persist rotation state to {path}: {exc}")


def rank(candidates: Iterable, history: Iterable) -> list:
    """Candidates least-recently-used first; never-used first of all.

    Pure and side-effect free, so callers can rank against a hypothetical
    history in tests without touching disk.
    """
    candidates = list(candidates)
    last_used = {item: i for i, item in enumerate(history)}
    unused = [c for c in candidates if c not in last_used]
    random.shuffle(unused)
    used = sorted((c for c in candidates if c in last_used),
                  key=lambda c: last_used[c])
    return unused + used


def choose(candidates: Iterable, path: str, remember_it: bool = True,
           keep: int = 500) -> Optional[Any]:
    """Pick the least-recently-used candidate and record the pick.

    Returns None for an empty pool rather than raising: every caller here has a
    module-level default pool, so an empty one means a pack was mis-edited, and
    the right answer is to let the caller fall back.
    """
    candidates = list(candidates)
    if not candidates:
        return None
    picked = rank(candidates, load_history(path))[0]
    if remember_it:
        remember(path, picked, keep=keep)
    return picked
