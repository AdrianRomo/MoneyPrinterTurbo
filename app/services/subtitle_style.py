"""Captions set with the same type system as the cards.

The grid had editorial discipline the Reels did not. Cards are set in a Garamond
revival with real tracking, optical centring and a scrim measured against the
pixels the letters actually cover; captions were MoviePy `TextClip` — one static
bold face at 58px with a 4px black stroke, no tracking, wrapping to five lines.
That is the "shout" style every AI faith account ships, and it is the giveaway
that the type was set by code.

Why this cannot be fixed inside MoviePy: `TextClip` renders a variable font at
its **default instance**, so asking it for Cormorant Medium silently gives you
Cormorant Regular — which is exactly why a static face was used in the first
place. Going through PIL directly sidesteps it, and brings tracking with it,
since Pillow has no letter-spacing of its own (:mod:`typography` supplies it).

So a cue is rendered here to an RGBA overlay and composited as an image. The
scrim replaces the stroke: :func:`quality.alpha_for_target` computes the exact
black alpha needed to bring the *measured* luminance under the glyphs down to a
4:1 contrast ratio, which is the same calculation the cards use. A stroke shouts
at every background equally; a scrim only darkens as much as this frame needs.

Off by default. ``subtitle_renderer = "brand"`` in ``[app]`` selects it, so the
old renderer stays one config key away and the two can be rendered side by side.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from loguru import logger
from PIL import Image, ImageDraw, ImageFilter

from app.config import config
from app.services import quality, typography

# Type. Small and quiet: at 40–44px a caption sits under the footage rather than
# on top of it, and two lines is the most a viewer reads while the audio moves.
DEFAULT_FONT_SIZE = 42
MIN_FONT_SIZE = 30
MAX_LINES = 2
TRACKING = 0.03
LEADING = 1.34
TEXT_WIDTH_RATIO = 0.78   # of frame width; the cards' margins, not MoviePy's 90%

# Scrim. Padded well past the glyphs and blurred, so it reads as a shadow under
# the type rather than as a visible box with edges.
SCRIM_PAD_X = 56
SCRIM_PAD_Y = 34
SCRIM_BLUR = 26
SCRIM_MAX_ALPHA = 225   # a fully opaque plate would look like a lower third

# Video asks more of a caption than a card does: the background moves, the
# viewer is not looking for the text, and compression eats thin strokes. The
# cards aim for 4:1; captions aim higher, and never ship below the floor.
TARGET_RATIO = 6.0
MIN_RATIO = 4.0

# Motion, in seconds/pixels. Enough to feel considered, not enough to notice.
FADE_SECONDS = 0.15
RISE_PIXELS = 8


def enabled() -> bool:
    return str(config.app.get("subtitle_renderer", "moviepy") or "").strip().lower() == "brand"


def _font_size() -> int:
    try:
        size = int(config.app.get("subtitle_font_size", DEFAULT_FONT_SIZE))
    except (TypeError, ValueError):
        size = DEFAULT_FONT_SIZE
    return max(MIN_FONT_SIZE, size)


def _face() -> tuple[str, str]:
    """(font path, variable instance). Serif is the account's voice."""
    face = str(config.app.get("subtitle_face", "serif") or "serif").strip().lower()
    if face == "sans":
        return typography.SANS, "Light"
    return typography.SERIF, "Medium"


def _fit_lines(draw, text: str, path: str, instance: str, max_w: float):
    """Largest size at which the text wraps to at most MAX_LINES."""
    size = _font_size()
    while size > MIN_FONT_SIZE:
        fnt = typography.font(path, size, instance)
        lines = typography.wrap(draw, text, fnt, max_w, TRACKING)
        if len(lines) <= MAX_LINES:
            return fnt, lines
        size -= 2
    fnt = typography.font(path, MIN_FONT_SIZE, instance)
    lines = typography.wrap(draw, text, fnt, max_w, TRACKING)
    if len(lines) > MAX_LINES:
        # Phase 3 (word-level Whisper cues) is what stops this happening at all;
        # until then a long cue is reported rather than silently clipped.
        logger.warning(
            f"subtitle cue needs {len(lines)} lines at minimum size: {text[:60]!r}"
        )
    return fnt, lines


def _glyph_mask(size: tuple[int, int], lines: list[str], fnt, baseline: int,
                line_h: int, width: int) -> Image.Image:
    mask = Image.new("L", size, 0)
    mdraw = ImageDraw.Draw(mask)
    y = baseline
    for line in lines:
        typography.draw_centered(mdraw, width, y, line, fnt, 255, TRACKING)
        y += line_h
    return mask


def measure_background(frame: Optional[np.ndarray], box: tuple[int, int, int, int],
                       mask: Image.Image) -> Optional[float]:
    """Luminance of the video under the glyphs, or None if unmeasurable.

    `frame` is the RGB frame the cue will sit on. Sampling the real frame is the
    whole point: a caption over a bright sky and the same caption over dark water
    need different scrims, and a fixed stroke gives both the same one.
    """
    if frame is None:
        return None
    try:
        x0, y0, x1, y1 = box
        crop = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB").crop((x0, y0, x1, y1))
        return quality.luminance_under(crop, mask)
    except Exception as exc:  # noqa: BLE001 - fall back to the default scrim
        logger.warning(f"subtitle background measurement failed: {exc}")
        return None


def _layout(text: str, video_width: int):
    """Type layout for a cue: (probe, font, lines, line height, overlay height)."""
    path, instance = _face()
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    fnt, lines = _fit_lines(probe, text, path, instance, video_width * TEXT_WIDTH_RATIO)
    line_h = int(fnt.size * LEADING)
    # The overlay carries a margin on every side for the blur to fall off in.
    # Without it the Gaussian is clipped at the overlay boundary and the scrim
    # gains exactly the hard edges it exists to avoid — a visible plate instead
    # of a shadow. Pillow's blur radius is a standard deviation, so the margin
    # has to be ~3σ for the falloff to actually reach zero; at 1σ the edge still
    # carries ~19% alpha, which is a soft step rather than no step.
    overlay_h = line_h * len(lines) + 2 * (SCRIM_PAD_Y + SCRIM_BLUR * 3)
    return probe, fnt, lines, line_h, overlay_h


def cue_height(text: str, video_width: int) -> int:
    """Overlay height for a cue without drawing it.

    The caller needs the height to decide *where* the overlay goes, and needs
    that position before the scrim can be measured against the right pixels.
    Exposing the layout separately keeps that from costing a throwaway render.
    """
    text = " ".join((text or "").split())
    if not text:
        return 0
    return _layout(text, video_width)[4]


def render_cue(text: str, video_width: int, video_height: int,
               background: Optional[np.ndarray] = None,
               background_offset: tuple[int, int] = (0, 0)) -> Optional[np.ndarray]:
    """Render one caption to an RGBA array, or None if there is nothing to draw.

    ``background`` is the full video frame; ``background_offset`` is where this
    overlay will be composited on it, which is what lets the scrim be measured
    against the pixels actually behind the glyphs.
    """
    text = " ".join((text or "").split())
    if not text:
        return None

    probe, fnt, lines, line_h, overlay_h = _layout(text, video_width)
    margin = SCRIM_BLUR * 3
    text_top = SCRIM_PAD_Y + margin
    overlay = Image.new("RGBA", (video_width, overlay_h), (0, 0, 0, 0))

    mask = _glyph_mask((video_width, overlay_h), lines, fnt, text_top,
                       line_h, video_width)

    off_x, off_y = background_offset
    luminance = measure_background(
        background,
        (off_x, off_y, off_x + video_width, off_y + overlay_h),
        mask,
    )
    # Shape first, strength second. A blurred scrim delivers less alpha under
    # the glyphs than it was drawn with, so computing the alpha and *then*
    # softening the edges quietly undoes the calculation — the first version of
    # this measured 1.85:1 as needing alpha 150, applied 150, and still shipped a
    # caption below the floor. Instead: build the soft shape, find how much of it
    # actually lands under the letters, and scale so that what lands hits target.
    widest = max(typography.width(probe, line, fnt, TRACKING) for line in lines)
    box_w = min(video_width - 2 * margin, widest + 2 * SCRIM_PAD_X)
    x0 = (video_width - box_w) / 2
    shape = Image.new("L", (video_width, overlay_h), 0)
    ImageDraw.Draw(shape).rounded_rectangle(
        [x0, margin, x0 + box_w, overlay_h - margin],
        radius=int(fnt.size * 0.7),
        fill=255,
    )
    shape = shape.filter(ImageFilter.GaussianBlur(SCRIM_BLUR))

    shape_a = np.asarray(shape, dtype=np.float32) / 255.0
    glyphs = np.asarray(mask, dtype=np.uint8) > 96
    coverage = float(shape_a[glyphs].mean()) if glyphs.any() else 1.0

    if luminance is None:
        # No sample (or no frame): assume a mid-bright background rather than
        # none, so an unmeasurable cue still gets a usable scrim.
        peak = int(SCRIM_MAX_ALPHA * 0.6)
    else:
        needed = quality.alpha_for_target(luminance, TARGET_RATIO)
        peak = min(SCRIM_MAX_ALPHA, int(round(needed / max(coverage, 0.05))))

    if peak > 0:
        alpha_map = Image.fromarray((shape_a * peak).astype(np.uint8))
        overlay.paste(Image.new("RGBA", overlay.size, (0, 0, 0, 255)), (0, 0), alpha_map)

    if luminance is not None:
        delivered = luminance * (1.0 - (peak * coverage) / 255.0)
        ratio = quality.contrast_ratio(delivered)
        if ratio < MIN_RATIO:
            logger.warning(
                f"caption contrast {ratio:.1f}:1 is below the {MIN_RATIO}:1 floor "
                f"even at maximum scrim (background luminance {luminance:.2f}): "
                f"{text[:50]!r}"
            )
        else:
            logger.debug(f"caption contrast {ratio:.1f}:1 (scrim alpha {peak})")

    draw = ImageDraw.Draw(overlay)
    y = text_top
    for line in lines:
        typography.draw_centered(draw, video_width, y, line, fnt, (255, 255, 255, 255), TRACKING)
        y += line_h

    return np.array(overlay)


def describe(text: str, video_width: int) -> str:
    """What the renderer decided, for logs and for A/B comparison."""
    path, instance = _face()
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    fnt, lines = _fit_lines(probe, " ".join((text or "").split()), path, instance,
                            video_width * TEXT_WIDTH_RATIO)
    face = path.rsplit("/", 1)[-1]
    return f"{face} {instance} {fnt.size}px, {len(lines)} line(s), tracking {TRACKING}"
