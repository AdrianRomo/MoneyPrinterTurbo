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
from app.services import quality, typography as ty
from app.services.verse_card import _cover

WIDTH, HEIGHT = 1080, 1350          # 4:5, the tallest ratio Instagram allows in feed
MAX_SLIDES = 10                      # hard API limit for carousel children
# Type roles come from typography.py; see that module for why these faces.

OUT_DIR = "/influencer-automation-2.0/storage/carousels"

# Subject -> (title noun, Commons search terms). Keeping the search terms
# explicit avoids drifting into categories with poor or irrelevant imagery.
SUBJECTS = {
    "mountains":    ("MOUNTAINS",       "mountain landscape peak"),
    "auroras":      ("THE AURORAS",     "aurora borealis"),
    "sunsets":      ("THE SUNSET",      "sunset clouds sky"),
    "oceans":       ("THE OCEAN",       "ocean coast sea waves"),
    "forests":      ("THE FORESTS",     "forest trees woodland"),
    "deserts":      ("THE DESERT",      "desert dunes landscape"),
    "rivers":       ("THE RIVERS",      "river valley"),
    "storms":       ("THE STORM",       "storm clouds lightning sky"),
    "glaciers":     ("THE ICE",         "glacier iceberg arctic"),
    "night_sky":    ("THE NIGHT SKY",   "milky way night sky stars"),
    "waterfalls":   ("WATERFALLS",      "waterfall cascade"),
    "canyons":      ("CANYONS",         "canyon gorge cliffs"),
    "volcanoes":    ("VOLCANOES",       "volcano crater lava"),
    "lakes":        ("LAKES",           "lake reflection alpine"),
    "islands":      ("ISLANDS",         "island coastline aerial"),
    "caves":        ("CAVES",           "cave cavern limestone"),
    "autumn":       ("AUTUMN",          "autumn foliage forest"),
    "winter":       ("WINTER",          "snow winter landscape"),
    "wildflowers":  ("WILDFLOWERS",     "wildflower meadow bloom"),
    "fjords":       ("THE FJORDS",      "fjord landscape"),
    "rainforest":   ("THE RAINFOREST",  "rainforest jungle canopy"),
    "clouds":       ("THE CLOUDS",      "clouds sky formation"),
    "mist":         ("THE MIST",        "fog mist landscape morning"),
    "reefs":        ("CORAL REEFS",     "coral reef underwater"),
    "savanna":      ("THE SAVANNA",     "savanna grassland plain"),
    "hot_springs":  ("HOT SPRINGS",     "hot spring geyser thermal"),
    "salt_flats":   ("SALT FLATS",      "salt flat reflection"),
    "the_moon":     ("THE MOON",        "moon lunar surface"),
    "rain":         ("THE RAIN",        "rain droplets water nature"),
    "tundra":       ("THE TUNDRA",      "tundra arctic landscape"),
    "birds":        ("BIRDS IN FLIGHT", "birds flight flock nature"),
    "ancient_trees":("ANCIENT TREES",   "ancient tree solitary oak"),
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

    mark_size = int(w * 0.026)      # serif needs more size than a sans to read
    small_size = int(w * 0.0135)
    f_mark = ty.font(ty.SERIF, mark_size, "Light")
    f_small = ty.font(ty.SANS, small_size, "Light")

    # Wordmark and tagline swap ends between slides, as the reference does —
    # it keeps a long carousel from feeling like the same frame repeated.
    mark_y = int(h * 0.055) if mark_at_top else int(h * 0.905)
    tag_y = int(h * 0.90) if mark_at_top else int(h * 0.055)

    ty.draw_centered(draw, w, mark_y, wordmark(), f_mark, (255, 255, 255, 240),
                     ty.TRACK_WORDMARK)

    t1, t2 = tagline()
    for i, line in enumerate([t1, t2]):
        if not line:
            continue
        ty.draw_centered(draw, w, tag_y + i * int(small_size * 1.6), line.upper(),
                         f_small, (255, 255, 255, 200), ty.TRACK_MICRO)

    margin = int(w * 0.062)
    if location:
        for i, line in enumerate(location.split("\n")[:2]):
            ty.draw_tracked(draw, (margin, int(h * 0.845) + i * int(small_size * 1.5)),
                            line.upper(), f_small, (255, 255, 255, 230), ty.TRACK_MICRO)
    if credit:
        # CC BY requires attribution; keep it discreet but present on-slide.
        f_credit = ty.font(ty.SANS, int(w * 0.0105), "Regular")
        cw = ty.width(draw, credit, f_credit, ty.TRACK_MICRO)
        ty.draw_tracked(draw, (w - margin - cw, int(h * 0.868)), credit, f_credit,
                        (255, 255, 255, 145), ty.TRACK_MICRO)
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
    fnt, lines = ty.fit(draw, title, ty.DISPLAY, max_w, int(HEIGHT * 0.27),
                        start=int(WIDTH * 0.068), min_size=int(WIDTH * 0.036),
                        instance="Bold", tracking=ty.TRACK_TITLE, leading=ty.LEAD_TITLE)
    line_h = int(fnt.size * ty.LEAD_TITLE)
    y = int(HEIGHT * 0.50) - (len(lines) * line_h) // 2
    for line in lines:
        # No drop shadow: the scrim already guarantees contrast, and a hard
        # offset shadow is the fastest way to make display type look cheap.
        ty.draw_centered(draw, WIDTH, y, line, fnt, (255, 255, 255), ty.TRACK_TITLE)
        y += line_h
    return img


def _recent_subjects(limit: int = 12) -> list:
    import json
    try:
        with open(os.path.join(OUT_DIR, "recent_subjects.json"), encoding="utf-8") as fh:
            return list(json.load(fh))[-limit:]
    except (OSError, ValueError):
        return []


def _remember_subject(subject: str, keep: int = 24) -> None:
    import json
    os.makedirs(OUT_DIR, exist_ok=True)
    recent = [s for s in _recent_subjects(keep) if s != subject]
    recent.append(subject)
    try:
        with open(os.path.join(OUT_DIR, "recent_subjects.json"), "w", encoding="utf-8") as fh:
            json.dump(recent[-keep:], fh)
    except OSError as exc:
        logger.warning(f"could not persist recent subjects: {exc}")


def choose_subject() -> str:
    """Least-recently-used, so the pool is worked through before repeating."""
    recent = _recent_subjects()
    unused = [s for s in SUBJECTS if s not in recent]
    if unused:
        return random.choice(unused)
    # Everything has been used recently: take the stalest.
    return next((s for s in recent if s in SUBJECTS), random.choice(list(SUBJECTS)))


def _used_photos_path() -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, "used_photos.json")


def _recent_photo_urls(limit: int = 300) -> set:
    try:
        import json
        with open(_used_photos_path(), encoding="utf-8") as fh:
            return set(list(json.load(fh))[-limit:])
    except (OSError, ValueError):
        return set()


def _remember_photos(urls: list, keep: int = 300) -> None:
    import json
    existing = [u for u in _recent_photo_urls(keep) if u]
    existing.extend(u for u in urls if u not in existing)
    try:
        with open(_used_photos_path(), "w", encoding="utf-8") as fh:
            json.dump(existing[-keep:], fh)
    except OSError as exc:
        logger.warning(f"could not persist used photos: {exc}")


def build(subject: Optional[str] = None, slides: int = 8,
          out_dir: str = OUT_DIR) -> Optional[dict]:
    """Build a carousel. Returns {paths, subject, title, photos, credits}."""
    subject = subject if subject in SUBJECTS else choose_subject()
    noun, query = SUBJECTS[subject]
    slides = max(3, min(slides, MAX_SLIDES))

    found = wikimedia.search(query, limit=slides * 4)
    if len(found) < slides:
        logger.error(f"only {len(found)} usable photos for {subject!r}, need {slides}")
        return None

    # Spread across photographers so a carousel is not one person's portfolio,
    # and skip anything used in a recent carousel — otherwise the same striking
    # photograph reappears every few weeks.
    seen_before = _recent_photo_urls()
    fresh = [p for p in found if p.url not in seen_before]
    if len(fresh) >= slides:
        found = fresh
    else:
        logger.warning(f"only {len(fresh)} unseen photos for {subject!r}; allowing repeats")

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

    car = {"paths": paths, "subject": subject, "title": noun.title(),
           "photos": picked[:len(paths)], "credits": credits}
    _remember_photos([p.url for p in picked[:len(paths)]])
    _remember_subject(subject)
    ok, reason = quality.check_carousel(car)
    quality.log_result("carousel", ok, reason)
    if not ok:
        return None
    logger.info(f"carousel built: {len(paths)} slides for {subject}")
    return car


def science_note(subject: str) -> str:
    """A short, plain explanation of the phenomenon.

    The wonder-plus-science pairing is what makes this format work: the image
    carries the awe, the caption gives people something to learn and pass on.

    Deliberately asks for qualitative description and no statistics — a wrong
    figure is the kind of error a comment section corrects in public, and this
    text is LLM-written and unverified.
    """
    from app.services import llm

    prompt = (
        f"Write 2-3 short sentences explaining, in plain everyday English, how {subject} "
        "form or occur in nature. Be accurate and general. Do NOT include any numbers, "
        "statistics, dates, measurements or place names. No preamble, no title, no "
        "hashtags — just the sentences."
    )
    try:
        text = (llm._generate_response(prompt) or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"science note failed: {exc}")
        return ""
    text = " ".join(text.split())
    # Guard against the model ignoring the no-numbers instruction.
    if re.search(r"\d", text):
        logger.warning("science note contained figures; dropping it rather than risk a wrong one")
        return ""
    return text[:600]


def build_caption(car: dict) -> tuple[str, str]:
    from app.services import hashtags

    set_id = hashtags.choose_set()
    lead = f"The creativity of God in {car['title'].lower()}"
    body = ("Creation keeps saying something we did not invent. "
            "Swipe through and let it slow you down for a minute.")
    note = science_note(car["subject"].replace("_", " "))
    tags = " ".join(hashtags.tags_for(set_id))

    # No credit block here by choice: CC BY attribution is satisfied on-slide,
    # where every photographer is named beside their own image. Dropping the
    # caption list keeps it readable without breaching a licence.
    parts = [lead, body]
    if note:
        parts.append(note)
    parts.append(tags)
    return "\n\n".join(parts), set_id


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
