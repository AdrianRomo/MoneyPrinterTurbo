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

# --- aesthetics --------------------------------------------------------------
#
# Every other size rule here measures PIXELS. None of them measures LIGHT, and a
# technically perfect photograph of a flat overcast sky still gets scrolled past.
# The worst cover the account has published (mountains, 2026-08-17) cleared every
# existing gate: correctly licensed, correctly labelled, distinct, downscaled.
#
# Calibrated against the 57 slides published 2026-08-15..17, not guessed:
#
#   colourfulness   min 4.3   p25 21.9   median 34.6   p75 45.3   max 108.2
#   contrast        min 15.7  p25 37.8   median 42.3   p75 48.4   max  62.0
#   brightness      min 32.1  p25 73.9   median 103.5  p75 122.9  max 173.2
#
# Two findings from that data decided the shape of this gate:
#
# 1. Colourfulness is the discriminating axis. The bad mountains cover scores
#    4.3 — three times below the next lowest slide in the whole set. Contrast
#    (35.9) and brightness (109) both put it mid-pack, so neither would catch it.
#
# 2. A colourfulness floor ALONE would reject good work. The deep-space cover
#    scores 12.7 and the best aurora scores 26.0 at a contrast of 15.7 — both are
#    genuinely near-monochrome, both are excellent, and both are exactly the
#    muted register this brand is built on. "Muted" and "lifeless" are not the
#    same thing and a single threshold cannot tell them apart.
#
# What separates them is brightness. Colourless AND dark is a night sky, which
# reads as deliberate. Colourless AND bright is an overcast whiteout, which reads
# as a snapshot. So the gate fires only on the conjunction.
#
# There is deliberately NO contrast floor. The measured evidence says it would
# reject the aurora and catch nothing that colourfulness does not already catch —
# the same trap as delta-vs-drift in the motion gate, where the metric that looked
# obvious was measuring texture rather than the thing being asked about.
FLAT_COLOUR = 15.0        # below this the frame is essentially monochrome
WASHED_BRIGHTNESS = 95.0  # ...and above this it is washed out rather than moody


def aesthetics(img: Image.Image) -> dict:
    """Colourfulness, tonal range and mean brightness of a slide.

    Colourfulness is Hasler & Susstrunk (2003), the standard cheap metric.
    Measured on a thumbnail: these are whole-frame statistics, and sampling the
    full 1440x1800 buys nothing but time.
    """
    import numpy as np

    small = img.convert("RGB").copy()
    small.thumbnail((400, 400))
    arr = np.asarray(small).astype(np.float64)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    colour = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                   + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return {"colour": colour, "contrast": float(lum.std()), "brightness": float(lum.mean())}


def check_slide_aesthetics(img: Image.Image) -> tuple[bool, str]:
    """Is this photograph worth a slide? Judged on light, not on pixels."""
    m = aesthetics(img)
    if m["colour"] < FLAT_COLOUR and m["brightness"] > WASHED_BRIGHTNESS:
        return False, (f"washed out: colourfulness {m['colour']:.1f} "
                       f"(floor {FLAT_COLOUR}) at brightness {m['brightness']:.0f}")
    return True, f"colour {m['colour']:.1f}, brightness {m['brightness']:.0f}"


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


# --- ink polarity -------------------------------------------------------------
#
# Everything above assumes WHITE type over a darkened photograph, because that
# was the only card this pipeline could make. That single assumption is why the
# feed collapsed on the light axis: the scrim darkens adaptively to guarantee
# white type stays legible, so a high-key background becomes a dark card and
# every card the account has ever published measures "dark". Widening the prompt
# vocabulary cannot fix that — measured end to end, six backgrounds spanning
# light, mid and dark all composed to "dark".
#
# The fix is the other polarity: DARK type over a lightened photograph. It needs
# its own contrast maths, because the WCAG ratio is not symmetric.
DARK_INK = (28, 26, 24)
# Relative luminance of DARK_INK, sRGB. Low enough that the ratio is generous
# and the gate is dominated by how light the background gets, not by the ink.
DARK_INK_LUM = 0.0116

# Dark type on a light ground reaches a given ratio far more easily than white
# type on a dark one, so a 4:1 target would pass a card that is merely mid-grey
# and defeat the point. These are set so a "dark ink" card is genuinely LIGHT.
TARGET_RATIO_DARK = 9.0    # -> background luminance about 0.50
MIN_RATIO_DARK = 5.5       # -> about 0.29, the floor below which it reads muddy


def contrast_ratio(bg_luminance: float, ink: str = "light") -> float:
    """WCAG-style ratio between the type and what is behind it.

    `ink` is the colour of the TYPE: "light" is the white type this pipeline
    started with, "dark" is DARK_INK on a lightened ground.
    """
    bg = max(0.0, bg_luminance)
    if ink == "dark":
        return (bg + 0.05) / (DARK_INK_LUM + 0.05)
    return (1.0 + 0.05) / (bg + 0.05)


def target_ratio_for(ink: str = "light") -> float:
    return TARGET_RATIO_DARK if ink == "dark" else TARGET_RATIO


def min_ratio_for(ink: str = "light") -> float:
    return MIN_RATIO_DARK if ink == "dark" else MIN_RATIO


def alpha_for_target(bg_luminance: float, target_ratio: Optional[float] = None,
                     ink: str = "light") -> int:
    """Overlay alpha needed to bring the background to the target luminance.

    For light ink the overlay is BLACK and darkening by alpha a scales luminance
    by (1 - a/255). For dark ink the overlay is WHITE and lightening by a moves
    luminance to L + (1 - L) * a/255. Both are exact — no iteration needed.
    """
    if target_ratio is None:
        target_ratio = target_ratio_for(ink)
    if ink == "dark":
        target_l = target_ratio * (DARK_INK_LUM + 0.05) - 0.05
        if bg_luminance >= target_l:
            return 0
        if bg_luminance >= 1.0:
            return 0
        return int(255 * (target_l - bg_luminance) / (1.0 - bg_luminance))
    target_l = (1.05 / target_ratio) - 0.05
    if bg_luminance <= target_l or bg_luminance <= 0:
        return 0
    return int(255 * (1 - target_l / bg_luminance))


def check_card(bg_luminance: Optional[float], ink: str = "light") -> tuple[bool, str]:
    if bg_luminance is None:
        return True, "no glyph sample (not enough text pixels) — allowed"
    ratio = contrast_ratio(bg_luminance, ink)
    floor = min_ratio_for(ink)
    if ratio < floor:
        return False, f"{ink}-ink contrast {ratio:.1f}:1 below floor {floor}:1"
    return True, f"{ink}-ink contrast {ratio:.1f}:1"


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
    # Per-slide aesthetics are enforced at candidate time in carousel.build(),
    # where a rejected photograph can still be replaced. Re-measuring the whole
    # set here would only be able to fail it, so this reports rather than gates.
    looks = ""
    try:
        # The closing CTA slide is deliberately desaturated and darkened, so it
        # is always the flattest frame in the set and reporting it as such says
        # nothing about the photography.
        colours = [aesthetics(Image.open(p))["colour"]
                   for p in paths if "-zz-cta" not in p]
        if colours:
            looks = f", flattest photo {min(colours):.1f}"
    except (OSError, ValueError) as exc:
        logger.debug(f"could not measure carousel aesthetics: {exc}")
    return True, f"{len(paths)} slides, all distinct{sharpness}{looks}"


def log_result(label: str, ok: bool, reason: str) -> None:
    (logger.info if ok else logger.warning)(f"quality[{label}]: {'pass' if ok else 'REJECT'} — {reason}")


# --- variety ------------------------------------------------------------------
#
# THE GAP THIS CLOSES. Every other gate in this module measures ONE image and
# asks "is this good?". None of them ever asked "is this DIFFERENT from the last
# few?", and nothing else in the pipeline did either. So the account published
# 41 cards that each passed every gate individually and, taken together, looked
# like one photograph taken 41 times:
#
#     saturation   never above 33/100    (100% of cards read as "muted")
#     lightness    never above 54/100    (nothing bright, ever)
#     hue family   66% in two families   (amber and blue/teal carried the feed)
#
# A feed is consumed as a grid, not as single posts. Variety is therefore a
# property of the SEQUENCE, and it needs a gate that can see more than one card.
#
# THIS GATE IS A PREFERENCE, NOT A FLOOR — deliberately, and unlike every other
# gate here. The contrast gate protects legibility, so it hard-fails and the
# card is regenerated or abandoned. Repetition is a taste failure: a repetitive
# post is worth much more than a missed slot on an unattended account. So
# create_card prefers a candidate that passes this and publishes the best
# contrast-passing candidate anyway when the pool will not yield one. Never
# promote this to a hard floor without giving the caller somewhere to fall back.

VARIETY_WINDOW = 8      # a 3x3 profile grid is what a visitor actually judges
CLASS_WINDOW = 3        # ...and adjacent posts are what a follower scrolls
LOOK_WINDOW = 3


def palette_bucket(img: Image.Image) -> str:
    """A coarse "what does this look like" label: hue family / light / colour.

    Deliberately coarse. The question is not what colour the image is, it is
    whether a reader would call this the same kind of picture as the last one,
    and at thumbnail size that judgement has about this many levels.
    """
    import colorsys

    small = img.convert("RGB").resize((32, 32), Image.LANCZOS)
    px = list(small.getdata())
    n = len(px)
    r = sum(p[0] for p in px) / n
    g = sum(p[1] for p in px) / n
    b = sum(p[2] for p in px) / n
    h, light, sat = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    h *= 360
    light *= 100
    sat *= 100

    family = ("warm" if (h < 70 or h > 330) else
              "green" if h < 170 else
              "blue" if h < 260 else "violet")
    band = "dark" if light < 35 else "mid" if light < 62 else "light"
    colour = "muted" if sat < 30 else "moderate" if sat < 55 else "vivid"
    return f"{family}/{band}/{colour}"


def register_of(img: Image.Image, meta: Optional[dict] = None) -> dict:
    """The comparable fingerprint of a card: what it is OF, and what it looks like.

    Subject class and look come from the generator (verse_card knows what it
    asked for); the palette is measured from the pixels that came back, because
    a prompt asking for "high-key airy" and a model returning something dark is
    exactly the disagreement worth catching.
    """
    meta = meta or {}
    return {
        "subject_class": meta.get("subject_class", ""),
        "look": meta.get("look", ""),
        "palette": palette_bucket(img),
        # Carried so the accepted card's polarity travels with it — the story
        # twin has to compose in the same ink. check_variety ignores it: ink is
        # a property of the look, so comparing it would double-count.
        "ink": meta.get("ink", "light"),
    }


def check_variety(register: dict, recent: list) -> tuple[bool, str]:
    """Is this card different enough from the ones just published?

    `recent` is oldest-first, as rotation.load_history returns it.
    """
    if not register:
        return True, "no register to compare"
    window = list(recent)[-VARIETY_WINDOW:]
    if not window:
        return True, "nothing published yet"

    key = (register.get("subject_class"), register.get("look"),
           register.get("palette"))
    for prior in window:
        if not isinstance(prior, dict):
            continue
        if (prior.get("subject_class"), prior.get("look"),
                prior.get("palette")) == key:
            return False, (f"identical register to a card in the last "
                           f"{len(window)}: {key[0]}/{key[1]}/{key[2]}")

    recent_classes = [p.get("subject_class") for p in window[-CLASS_WINDOW:]
                      if isinstance(p, dict)]
    if register.get("subject_class") and register["subject_class"] in recent_classes:
        return False, (f"subject class {register['subject_class']!r} used within "
                       f"the last {CLASS_WINDOW} cards")

    recent_looks = [p.get("look") for p in window[-LOOK_WINDOW:]
                    if isinstance(p, dict)]
    if register.get("look") and register["look"] in recent_looks:
        return False, (f"look {register['look']!r} used within the last "
                       f"{LOOK_WINDOW} cards")

    return True, f"{key[0]}/{key[1]}/{key[2]}"
