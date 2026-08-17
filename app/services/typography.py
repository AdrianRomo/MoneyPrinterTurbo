"""Shared type system for cards and carousels.

Three roles, strictly separated — which is what keeps a three-family system
coherent rather than noisy:

    DISPLAY  Playfair Display        cover titles
    SERIF    Cormorant Garamond      the voice: wordmark, verse text, tagline
    SANS     Inter                   functional micro-text: location, credit

Why these. Playfair is a high-contrast transitional serif that stays legible at
feed-thumbnail size, where Cormorant's hairlines would disappear. Cormorant is a
Garamond revival — scripture set in a Garamond reads as timeless rather than as
a template. The sans appears only where a serif would turn to mush (~13px photo
credits), which is standard editorial practice.

Both serifs are variable fonts, so weights come from named instances rather than
separate files.

The functions here add what Pillow lacks: **tracking**. Letter-spacing is not
decoration — all-caps and small text are unreadable and ugly without it, and its
absence is the single biggest giveaway that type was set by code.
"""

from __future__ import annotations

import os
from typing import Optional

from PIL import ImageFont

# Derived from this file rather than hardcoded to the container's mount point, so
# the type system is exercisable outside the container too. The fonts ship in the
# repo, and the repo *is* what is mounted at /influencer-automation-2.0, so this
# resolves to the same directory in the container.
FONT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "resource",
    "fonts",
)

DISPLAY = os.path.join(FONT_DIR, "PlayfairDisplay[wght].ttf")
SERIF = os.path.join(FONT_DIR, "CormorantGaramond[wght].ttf")
SANS = os.path.join(FONT_DIR, "Inter[opsz,wght].ttf")

# Optical defaults. Display type wants tight leading and open tracking; body
# type wants the opposite.
TRACK_TITLE = 0.045      # all-caps display
TRACK_WORDMARK = 0.22    # lowercase wordmark, widely spaced
TRACK_MICRO = 0.10       # small caps / credits
LEAD_TITLE = 1.14
LEAD_BODY = 1.42


def font(path: str, size: int, instance: Optional[str] = None) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(path, max(1, int(size)))
    if instance:
        try:
            f.set_variation_by_name(instance)
        except (OSError, AttributeError, ValueError):
            pass  # static font, or no such instance — the default is acceptable
    return f


def width(draw, text: str, fnt, tracking: float = 0.0) -> float:
    """Advance width including tracking (Pillow ignores letter-spacing)."""
    if not text:
        return 0.0
    base = draw.textlength(text, font=fnt)
    return base + tracking * fnt.size * max(0, len(text) - 1)


def wrap(draw, text: str, fnt, max_w: float, tracking: float = 0.0) -> list[str]:
    lines, current = [], ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if width(draw, trial, fnt, tracking) <= max_w or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def fit(draw, text: str, path: str, max_w: float, max_h: float, *,
        start: int, min_size: int, instance: Optional[str] = None,
        tracking: float = 0.0, leading: float = LEAD_BODY):
    """Largest size at which the tracked, wrapped text still fits the box."""
    size = int(start)
    while size > min_size:
        fnt = font(path, size, instance)
        lines = wrap(draw, text, fnt, max_w, tracking)
        if len(lines) * int(size * leading) <= max_h:
            return fnt, lines
        size -= 2
    fnt = font(path, min_size, instance)
    return fnt, wrap(draw, text, fnt, max_w, tracking)


def draw_tracked(draw, xy: tuple[float, float], text: str, fnt, fill,
                 tracking: float = 0.0) -> float:
    """Draw text with tracking; returns the advance width used."""
    x, y = xy
    if tracking == 0:
        draw.text((x, y), text, font=fnt, fill=fill)
        return draw.textlength(text, font=fnt)
    step = tracking * fnt.size
    for char in text:
        draw.text((x, y), char, font=fnt, fill=fill)
        x += draw.textlength(char, font=fnt) + step
    return x - xy[0] - step


def draw_centered(draw, canvas_w: int, y: float, text: str, fnt, fill,
                  tracking: float = 0.0) -> float:
    w = width(draw, text, fnt, tracking)
    # Tracking adds a trailing space after the last glyph; shift by half of it
    # so the optical centre matches the geometric one.
    x = (canvas_w - w) / 2 + (tracking * fnt.size) / 2
    draw_tracked(draw, (x, y), text, fnt, fill, tracking)
    return w
