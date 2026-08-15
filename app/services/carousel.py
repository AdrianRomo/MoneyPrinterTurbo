"""Photo carousels — "the creativity of God in <subject>".

A cover slide with a large title, then real photographs of real places, each
labelled with its location and photographer. Published to Instagram as a
carousel.

The imagery comes from Wikimedia Commons (see wikimedia.py), never from image
generation: this format's whole appeal is that the places are real, and its
audience checks. Locations that cannot be parsed confidently are left off the
slide rather than guessed.

Instagram's Content Publishing API accepts at most 10 items in a carousel, so
sets are capped there even though the app itself allows more.
"""

from __future__ import annotations

import os
import random
import re
import time
from typing import Optional

from loguru import logger
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from app.config import config
from app.services import wikimedia
from app.services.verse_card import FONT_REF, FONT_VERSE, _cover, _draw_tracked, _fit_font

WIDTH, HEIGHT = 1080, 1350          # 4:5, the tallest ratio Instagram allows in feed
MAX_SLIDES = 10                      # hard API limit for carousel children
FONT_TITLE = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
# The wordmark is deliberately the light cut, not the bold: it should sit
# quietly on every slide, not compete with the photograph.
FONT_MARK = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"

OUT_DIR = "/influencer-automation-2.0/storage/carousels"

# Subject -> (title noun, Commons search terms). Keeping the search terms
# explicit avoids drifting into categories with poor or irrelevant imagery.
SUBJECTS = {
    "mountains":  ("MOUNTAINS",      "mountain landscape peak"),
    "auroras":    ("THE AURORAS",    "aurora borealis"),
    "sunsets":    ("THE SUNSET",     "sunset clouds sky"),
    "oceans":     ("THE OCEAN",      "ocean coast sea waves"),
    "forests":    ("THE FORESTS",    "forest trees woodland"),
    "deserts":    ("THE DESERT",     "desert dunes landscape"),
    "rivers":     ("THE RIVERS",     "river valley waterfall"),
    "storms":     ("THE STORM",      "storm clouds lightning sky"),
    "glaciers":   ("THE ICE",        "glacier iceberg arctic"),
    "night_sky":  ("THE NIGHT SKY",  "milky way night sky stars"),
}


def _cfg(key: str, default: str) -> str:
    value = config.app.get(key, default)
    return default if value in (None, "") else str(value)


def wordmark() -> str:
    return _cfg("brand_wordmark", "holy ordinary")


def tagline() -> tuple[str, str]:
    raw = _cfg("brand_tagline", "creation × wonder")
    parts = [p.strip() for p in re.split(r"[×x]", raw, maxsplit=1)]
    return (parts[0], "× " + parts[1]) if len(parts) == 2 else (raw, "")


def _furniture_scrim(img: Image.Image) -> Image.Image:
    """Soft top/bottom gradients so the small furniture text stays readable
    on any photograph, without flattening the image the way a full scrim does."""
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    band = int(h * 0.22)
    for y in range(band):
        a = int(120 * (1 - y / band) ** 1.5)
        od.line([(0, y), (w, y)], fill=(0, 0, 0, a))
        od.line([(0, h - 1 - y), (w, h - 1 - y)], fill=(0, 0, 0, int(a * 1.05)))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def _draw_furniture(img: Image.Image, location: Optional[str], credit: Optional[str],
                    mark_at_top: bool) -> Image.Image:
    w, h = img.size
    draw = ImageDraw.Draw(img)

    mark_size = int(w * 0.021)
    small_size = int(w * 0.0155)
    f_mark = ImageFont.truetype(FONT_MARK, mark_size)
    f_small = ImageFont.truetype(FONT_VERSE, small_size)

    # Wordmark and tagline swap ends between slides, as the reference does —
    # it keeps a long carousel from feeling like the same frame repeated.
    mark_y = int(h * 0.055) if mark_at_top else int(h * 0.905)
    tag_y = int(h * 0.90) if mark_at_top else int(h * 0.055)

    mw = draw.textlength(wordmark(), font=f_mark)
    draw.text(((w - mw) / 2, mark_y), wordmark(), font=f_mark, fill=(255, 255, 255, 235))

    t1, t2 = tagline()
    for i, line in enumerate([t1, t2]):
        if not line:
            continue
        lw = draw.textlength(line, font=f_small)
        draw.text(((w - lw) / 2, tag_y + i * int(small_size * 1.35)), line,
                  font=f_small, fill=(255, 255, 255, 205))

    margin = int(w * 0.062)
    if location:
        for i, line in enumerate(location.split("\n")[:2]):
            draw.text((margin, int(h * 0.845) + i * int(small_size * 1.3)), line,
                      font=f_small, fill=(255, 255, 255, 225))
    if credit:
        # CC BY requires attribution; keep it discreet but present on-slide.
        f_credit = ImageFont.truetype(FONT_VERSE, int(w * 0.0125))
        cw = draw.textlength(credit, font=f_credit)
        draw.text((w - margin - cw, int(h * 0.862)), credit, font=f_credit,
                  fill=(255, 255, 255, 150))
    return img


def _cover_slide(photo_img: Image.Image, title: str) -> Image.Image:
    img = _cover(photo_img, WIDTH, HEIGHT)
    img = ImageEnhance.Color(img).enhance(0.9)
    img = img.filter(ImageFilter.GaussianBlur(radius=WIDTH * 0.0015))
    # The cover carries a large title, so it needs a real scrim, unlike the
    # interior slides which only carry small furniture.
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 90))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    img = _furniture_scrim(img)

    draw = ImageDraw.Draw(img)
    max_w = int(WIDTH * 0.90)
    font, lines = _fit_font(draw, title, FONT_TITLE, max_w, int(HEIGHT * 0.26),
                            start=int(WIDTH * 0.070), min_size=int(WIDTH * 0.038),
                            line_ratio=1.14)
    line_h = int(font.size * 1.16)
    y = int(HEIGHT * 0.50) - (len(lines) * line_h) // 2
    for line in lines:
        lw = draw.textlength(line, font=font)
        # A soft shadow keeps the title readable over bright sky.
        draw.text(((WIDTH - lw) / 2 + 2, y + 2), line, font=font, fill=(0, 0, 0, 120))
        draw.text(((WIDTH - lw) / 2, y), line, font=font, fill=(255, 255, 255))
        y += line_h
    return img


def build(subject: Optional[str] = None, slides: int = 8,
          out_dir: str = OUT_DIR) -> Optional[dict]:
    """Build a carousel. Returns {paths, subject, title, photos, credits}."""
    subject = subject if subject in SUBJECTS else random.choice(list(SUBJECTS))
    noun, query = SUBJECTS[subject]
    slides = max(3, min(slides, MAX_SLIDES))

    found = wikimedia.search(query, limit=slides * 4)
    if len(found) < slides:
        logger.error(f"only {len(found)} usable photos for {subject!r}, need {slides}")
        return None

    # Spread across photographers so a carousel is not one person's portfolio.
    picked: list[wikimedia.Photo] = []
    per_author: dict[str, int] = {}
    for photo in found:
        key = photo.author.lower()[:40]
        if per_author.get(key, 0) >= 2:
            continue
        picked.append(photo)
        per_author[key] = per_author.get(key, 0) + 1
        if len(picked) == slides:
            break
    if len(picked) < slides:
        picked = found[:slides]

    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    paths, credits = [], []

    for index, photo in enumerate(picked):
        source = wikimedia.download(photo)
        if source is None:
            continue
        if index == 0:
            slide = _cover_slide(source, f"THE CREATIVITY OF GOD IN {noun}")
        else:
            slide = _cover(source, WIDTH, HEIGHT)
            slide = _furniture_scrim(slide)
        slide = _draw_furniture(slide, photo.location, wikimedia.credit_line(photo),
                                mark_at_top=(index % 2 == 0))
        path = os.path.join(out_dir, f"{stamp}-{subject}-{index:02d}.jpg")
        slide.save(path, "JPEG", quality=91, optimize=True, progressive=True)
        paths.append(path)
        credits.append(f"{photo.location or 'unlisted'} — {wikimedia.credit_line(photo)}")

    if len(paths) < 3:
        logger.error("too few slides rendered")
        return None

    logger.info(f"carousel built: {len(paths)} slides for {subject}")
    return {"paths": paths, "subject": subject, "title": noun.title(),
            "photos": picked[:len(paths)], "credits": credits}


def build_caption(car: dict) -> tuple[str, str]:
    from app.services import hashtags

    set_id = hashtags.choose_set()
    lead = f"The creativity of God in {car['title'].lower()}"
    body = ("Creation keeps saying something we did not invent. "
            "Swipe through and let it slow you down for a minute.")
    # CC BY obliges us to name the photographers; Commons is named as the source.
    credit_block = "Photos via Wikimedia Commons — " + "; ".join(
        c.split(" — ", 1)[1] for c in car["credits"][:10])
    tags = " ".join(hashtags.tags_for(set_id))
    return f"{lead}\n\n{body}\n\n{credit_block}\n\n{tags}", set_id


def publish(car: dict, publish_at=None) -> dict:
    from app.services import hashtags
    from app.services.postiz import PostizService

    svc = PostizService()
    svc.post_type = "post"
    integration = svc.get_configured_integration()
    if not integration.get("success"):
        return integration
    if publish_at is None:
        selected = svc.select_publish_at(kind="carousel")
        if not selected.get("success"):
            return selected
        publish_at = selected.get("publish_at") or selected.get("date")

    media = []
    for path in car["paths"]:
        up = svc.upload_media(path)
        if not up.get("success"):
            return up
        media.append(up["media"])

    caption, set_id = build_caption(car)
    result = svc.schedule_post(media, caption, publish_at,
                               integration=integration["integration"],
                               kind="carousel", set_id=set_id)
    if result.get("success"):
        hashtags.mark_used(set_id)
    return result
