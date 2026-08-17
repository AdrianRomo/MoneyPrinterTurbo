"""Quality gates, so an unattended pipeline cannot publish something bad.

Contrast is measured **under the glyphs**, not over the text band. Band-mean
luminance is a poor proxy: a card can measure bright overall and still be
perfectly legible because the type happens to sit over the dark part of the
photograph. Measuring only the pixels the letters actually cover is what
legibility depends on.

The ratio is WCAG-style relative contrast against white type. WCAG AA wants 3:1
for large text; TARGET_RATIO aims above that and MIN_RATIO is the hard floor
below which a card is rejected and regenerated.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger
from PIL import Image

# White text (relative luminance 1.0) against background luminance L:
#   ratio = (1.0 + 0.05) / (L + 0.05)
TARGET_RATIO = 4.0   # what the scrim aims for  -> L <= 0.2125
MIN_RATIO = 2.8      # below this the card is rejected -> L >= 0.325

MIN_CAROUSEL_SLIDES = 4

# A slide is never built by enlarging its photograph. Carousels went out soft
# for weeks because nothing measured this: the images were correctly licensed,
# correctly labelled and distinct, and the gate had no opinion about whether
# there were enough pixels behind them.
MAX_SLIDE_UPSCALE = 1.02


def luminance_under(img: Image.Image, mask: Image.Image) -> Optional[float]:
    """Mean relative luminance of `img` where `mask` is non-zero (0.0–1.0)."""
    grey = img.convert("L")
    if grey.size != mask.size:
        mask = mask.resize(grey.size)
    px, mk = grey.load(), mask.load()
    w, h = grey.size
    # Sampling every 2nd pixel is ample and keeps this well under a second.
    total = count = 0
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            if mk[x, y] > 96:
                total += px[x, y]
                count += 1
    if count < 50:
        return None
    return (total / count) / 255.0


def contrast_ratio(bg_luminance: float) -> float:
    return (1.0 + 0.05) / (max(0.0, bg_luminance) + 0.05)


def alpha_for_target(bg_luminance: float, target_ratio: float = TARGET_RATIO) -> int:
    """Black-overlay alpha needed to bring bg luminance down to the target.

    Darkening by alpha a scales luminance by (1 - a/255), so the required alpha
    is exact — no iteration needed.
    """
    target_l = (1.05 / target_ratio) - 0.05
    if bg_luminance <= target_l or bg_luminance <= 0:
        return 0
    return int(255 * (1 - target_l / bg_luminance))


def check_card(bg_luminance: Optional[float]) -> tuple[bool, str]:
    if bg_luminance is None:
        return True, "no glyph sample (not enough text pixels) — allowed"
    ratio = contrast_ratio(bg_luminance)
    if ratio < MIN_RATIO:
        return False, f"text contrast {ratio:.1f}:1 below floor {MIN_RATIO}:1"
    return True, f"contrast {ratio:.1f}:1"


def check_carousel(car: dict) -> tuple[bool, str]:
    paths = car.get("paths") or []
    if len(paths) < MIN_CAROUSEL_SLIDES:
        return False, f"only {len(paths)} slides (minimum {MIN_CAROUSEL_SLIDES})"
    photos = car.get("photos") or []
    urls = [getattr(p, "url", None) for p in photos]
    urls = [u for u in urls if u]
    if len(set(urls)) != len(urls):
        return False, "carousel contains the same photograph twice"
    scales = [s for s in (car.get("scales") or []) if s]
    worst = max(scales) if scales else 0.0
    if worst > MAX_SLIDE_UPSCALE:
        return False, f"slide source upscaled {worst:.2f}x (max {MAX_SLIDE_UPSCALE}x)"
    sharpness = f", worst source scale {worst:.2f}x" if scales else ""
    return True, f"{len(paths)} slides, all distinct{sharpness}"


def log_result(label: str, ok: bool, reason: str) -> None:
    (logger.info if ok else logger.warning)(f"quality[{label}]: {'pass' if ok else 'REJECT'} — {reason}")
