"""Verse cards — AI-generated image posts and stories with scripture.

Pipeline: pick a reference -> fetch the authoritative text -> generate a
background with local ComfyUI (SDXL) -> compose a legible card -> hand to
Postiz as a feed post or a story.

Two rules this module exists to enforce:

1. **The LLM never writes scripture.** It only proposes a *reference*; the text
   is always fetched from a bible API and rendered verbatim. Language models
   misquote verses confidently, and on a faith account a misquote is the one
   error that costs credibility outright. A reference that cannot be fetched is
   discarded, not guessed at.

2. **Only public-domain translations.** KJV and WEB are public domain. NIV/ESV/
   NLT are copyrighted with strict quoting limits and are not safe to post on an
   account being grown commercially.
"""

from __future__ import annotations

import io
import json
import os
import random
import re
import time
import urllib.parse
from dataclasses import dataclass
from typing import Optional

import requests
from loguru import logger
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from app.config import config
from app.services import quality, typography as ty

# --- constants ---------------------------------------------------------------

# Instagram: 4:5 is the tallest permitted feed ratio; 9:16 for stories.
ASPECTS = {
    "post": (1080, 1350),
    "story": (1080, 1920),
}
# SDXL is trained on ~1 megapixel buckets; generating at a native bucket and
# resampling beats asking for an off-bucket size (which produces duplicated
# subjects and mangled composition).
SDXL_BUCKET = {
    "post": (896, 1152),
    "story": (768, 1344),
}

PUBLIC_DOMAIN_TRANSLATIONS = {"kjv", "web"}

# Backgrounds are deliberately unpeopled. AI depictions of Jesus or biblical
# figures land in the uncanny valley and reliably draw criticism on faith
# accounts; landscape and light imagery carries the same mood without the risk.
NEGATIVE_PROMPT = (
    "people, person, human, face, portrait, jesus, christ, saint, angel, figure, "
    "crowd, hands, text, watermark, signature, logo, letters, words, caption, "
    "cross, crucifix, church interior, statue, "
    "lowres, blurry, jpeg artifacts, oversaturated, hdr, deformed, cartoon, 3d render"
)

BACKGROUND_SUBJECTS = [
    "soft morning mist over a quiet meadow at sunrise",
    "golden hour light through tall forest pines",
    "calm lake at dawn with gentle reflections",
    "rolling hills under a wide pastel sky",
    "sunlight breaking through low clouds over open fields",
    "gentle ocean waves on an empty shore at first light",
    "dew on wild grass in early morning light",
    "quiet mountain range under soft blue haze",
    "wheat field swaying in warm evening light",
    "still river winding through autumn woodland",
    "desert dunes under a soft dawn sky",
    "snowfall over silent evergreen trees",
]
STYLE_SUFFIX = (
    "photographic, natural light, shallow depth of field, muted warm tones, "
    "cinematic, serene, minimal composition, negative space, 35mm, high detail"
)


@dataclass
class Verse:
    reference: str
    text: str
    translation: str


# --- config helpers ----------------------------------------------------------


def _cfg(key: str, default):
    value = config.app.get(key, default)
    return default if value in (None, "") else value


def _comfy_url() -> str:
    return str(_cfg("comfyui_base_url", "http://192.168.0.135:8188")).rstrip("/")


def _translation() -> str:
    t = str(_cfg("verse_translation", "kjv")).strip().lower()
    if t not in PUBLIC_DOMAIN_TRANSLATIONS:
        logger.warning(
            f"verse_translation '{t}' is not public domain; falling back to kjv. "
            "Copyrighted translations (NIV/ESV/NLT) are not safe to publish."
        )
        return "kjv"
    return t


def _state_path() -> str:
    d = "/influencer-automation-2.0/storage/verse_cards"
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, "used_references.json")


def _recent_references(limit: int = 60) -> list[str]:
    try:
        with open(_state_path(), encoding="utf-8") as fh:
            return list(json.load(fh))[-limit:]
    except (OSError, ValueError):
        return []


def _remember_reference(reference: str, keep: int = 400) -> None:
    used = _recent_references(limit=keep)
    used.append(reference)
    try:
        with open(_state_path(), "w", encoding="utf-8") as fh:
            json.dump(used[-keep:], fh)
    except OSError as exc:
        logger.warning(f"could not persist used reference: {exc}")


# --- 1. reference selection (LLM proposes, API verifies) ---------------------

_REF_RE = re.compile(r"^([1-3]\s*)?[A-Za-z][A-Za-z ]{1,20}\s+\d{1,3}:\d{1,3}(-\d{1,3})?$")


def pick_reference(theme: str = "", avoid: Optional[list[str]] = None) -> Optional[str]:
    """Ask the LLM for a *reference only*. Never for verse text."""
    from app.services import llm

    avoid = avoid or _recent_references()
    avoid_clause = ""
    if avoid:
        avoid_clause = "Do NOT choose any of these recently used references: " + "; ".join(avoid[-40:]) + ". "

    prompt = (
        "You are helping select a Bible verse for an encouraging social media post.\n"
        f"Theme: {theme or 'everyday encouragement, hope, gratitude, trust, peace'}.\n"
        f"{avoid_clause}"
        "Reply with ONE Bible reference and nothing else — no verse text, no commentary, "
        "no quotation marks. Format exactly like: Philippians 4:6 or Psalm 23:1-3. "
        "Choose a well-known, encouraging verse suitable for a general audience."
    )
    try:
        raw = llm._generate_response(prompt)
    except Exception as exc:  # noqa: BLE001 - any LLM failure is non-fatal here
        logger.warning(f"reference selection failed: {exc}")
        return None

    candidate = (raw or "").strip().strip('"').strip("'").splitlines()[0].strip()
    candidate = re.sub(r"^[\s\-\*\d.]+", "", candidate).strip()
    if not _REF_RE.match(candidate):
        logger.warning(f"LLM returned an unusable reference: {candidate!r}")
        return None
    return candidate


# --- 2. authoritative text ---------------------------------------------------


def fetch_verse(reference: str, translation: Optional[str] = None) -> Optional[Verse]:
    """Fetch verse text by reference. Returns None if the reference is not real."""
    translation = (translation or _translation()).lower()
    url = "https://bible-api.com/%s?translation=%s" % (
        urllib.parse.quote(reference),
        translation,
    )
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "influencer-automation/1.0"})
    except requests.exceptions.RequestException as exc:
        logger.warning(f"verse fetch failed for {reference}: {exc}")
        return None
    if r.status_code != 200:
        logger.warning(f"verse fetch: http {r.status_code} for {reference!r} (likely not a real reference)")
        return None
    try:
        data = r.json()
    except ValueError:
        return None

    text = " ".join((data.get("text") or "").split())
    if not text:
        return None
    return Verse(
        reference=str(data.get("reference") or reference).strip(),
        text=text,
        translation=str(data.get("translation_id") or translation).upper(),
    )


def select_verse(theme: str = "", attempts: int = 4) -> Optional[Verse]:
    """LLM proposes, the API decides. Unfetchable references are discarded."""
    avoid = _recent_references()
    for _ in range(attempts):
        ref = pick_reference(theme, avoid=avoid)
        if not ref:
            continue
        verse = fetch_verse(ref)
        if verse:
            return verse
        avoid = avoid + [ref]
    logger.error("could not obtain a verified verse after %d attempts" % attempts)
    return None


# --- 3. background generation (local ComfyUI / SDXL) -------------------------


def _workflow(prompt: str, width: int, height: int, seed: int, ckpt: str) -> dict:
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed, "steps": 30, "cfg": 6.0,
                "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0,
                "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE_PROMPT, "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "versecard"}},
    }


def generate_background(kind: str = "post", subject: Optional[str] = None,
                        seed: Optional[int] = None, timeout: int = 300) -> Optional[Image.Image]:
    base = _comfy_url()
    width, height = SDXL_BUCKET.get(kind, SDXL_BUCKET["post"])
    subject = subject or random.choice(BACKGROUND_SUBJECTS)
    prompt = f"{subject}, {STYLE_SUFFIX}"
    seed = seed if seed is not None else random.randint(1, 2**31 - 1)
    ckpt = str(_cfg("comfyui_checkpoint", "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"))

    logger.info(f"generating background ({kind} {width}x{height}): {subject}")
    try:
        resp = requests.post(f"{base}/prompt",
                             json={"prompt": _workflow(prompt, width, height, seed, ckpt)},
                             timeout=30)
        resp.raise_for_status()
        prompt_id = resp.json()["prompt_id"]
    except (requests.exceptions.RequestException, KeyError, ValueError) as exc:
        logger.error(f"ComfyUI submit failed: {exc}")
        return None

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            h = requests.get(f"{base}/history/{prompt_id}", timeout=15).json()
        except (requests.exceptions.RequestException, ValueError):
            time.sleep(2)
            continue
        entry = h.get(prompt_id)
        if entry and entry.get("outputs"):
            for out in entry["outputs"].values():
                for img in out.get("images", []):
                    q = urllib.parse.urlencode({
                        "filename": img["filename"],
                        "subfolder": img.get("subfolder", ""),
                        "type": img.get("type", "output"),
                    })
                    raw = requests.get(f"{base}/view?{q}", timeout=60).content
                    return Image.open(io.BytesIO(raw)).convert("RGB")
            logger.error("ComfyUI finished but produced no image")
            return None
        time.sleep(2)
    logger.error(f"ComfyUI timed out after {timeout}s")
    return None


# --- 4. composition ----------------------------------------------------------


def _cover(img: Image.Image, w: int, h: int) -> Image.Image:
    """Resize + centre-crop to exactly w x h without distorting."""
    scale = max(w / img.width, h / img.height)
    img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                     Image.LANCZOS)
    left, top = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((left, top, left + w, top + h))


def _wrap(draw, text: str, font, max_w: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for word in words:
        trial = (cur + " " + word).strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _fit_font(draw, text: str, path: str, max_w: int, max_h: int,
              start: int, min_size: int, line_ratio: float) -> tuple[ImageFont.FreeTypeFont, list[str]]:
    """Largest size at which the wrapped text still fits the box."""
    size = start
    while size > min_size:
        font = ImageFont.truetype(path, size)
        lines = _wrap(draw, text, font, max_w)
        if len(lines) * int(size * line_ratio) <= max_h:
            return font, lines
        size -= 2
    font = ImageFont.truetype(path, min_size)
    return font, _wrap(draw, text, font, max_w)


def _mean_luminance(img: Image.Image, box: tuple[int, int, int, int]) -> float:
    region = img.crop(box).convert("L").resize((32, 32))
    px = list(region.getdata())
    return (sum(px) / len(px)) / 255.0


def _draw_tracked(draw, xy, text: str, font, fill, tracking: float):
    """Letter-spaced text — Pillow has no tracking, so step per glyph."""
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=font, fill=fill)
        x += draw.textlength(ch, font=font) + tracking


def compose_card(bg: Image.Image, verse: Verse, kind: str = "post",
                 out_path: Optional[str] = None) -> str:
    w, h = ASPECTS.get(kind, ASPECTS["post"])
    img = _cover(bg, w, h)

    # Keep the photograph a photograph. Enough softening that fine detail stops
    # fighting the type, not so much that the image turns to grey wash.
    img = ImageEnhance.Color(img).enhance(0.88)
    img = img.filter(ImageFilter.GaussianBlur(radius=w * 0.0018))

    # Text block. Stories reserve the top ~14% and bottom ~20% for Instagram's
    # own UI (avatar/close at the top, reply bar at the bottom).
    side = int(w * 0.11)
    if kind == "story":
        band_top, band_bottom = int(h * 0.20), int(h * 0.78)
    else:
        band_top, band_bottom = int(h * 0.14), int(h * 0.86)
    band_h = band_bottom - band_top
    max_w = w - 2 * side

    draw = ImageDraw.Draw(img)
    verse_text = f"“{verse.text}”"
    # Cap the type a little smaller than the box allows: restraint and negative
    # space are what separate a designed card from a generated one.
    # Scripture in a Garamond reads as timeless rather than as a template.
    # Cormorant is delicate, so it is set larger than a sans would be.
    font_v, lines = ty.fit(draw, verse_text, ty.SERIF, max_w, int(band_h * 0.62),
                           start=int(w * 0.092), min_size=int(w * 0.040),
                           instance="Medium", tracking=0.0, leading=ty.LEAD_BODY)
    line_h = int(font_v.size * ty.LEAD_BODY)
    ref_size = max(int(font_v.size * 0.26), int(w * 0.0155))
    font_r = ty.font(ty.SANS, ref_size, "Medium")

    rule_gap = int(font_v.size * 0.95)
    block_h = len(lines) * line_h + rule_gap + ref_size
    block_top = band_top + (band_h - block_h) // 2

    # Adaptive scrim: measure what is actually behind the text and darken only
    # as much as needed for white type to stay comfortably readable. A fixed
    # opacity either muddies dark images or fails on bright ones.
    # Measure the background *under the glyphs*, not across the whole band: a
    # card can be bright overall and still perfectly legible because the type
    # sits over the dark part of the photograph. Render the text to a mask
    # first, then darken exactly as much as those pixels need.
    mask = Image.new("L", (w, h), 0)
    mdraw = ImageDraw.Draw(mask)
    my = block_top
    for line in lines:
        lw = ty.width(mdraw, line, font_v)
        mdraw.text(((w - lw) / 2, my), line, font=font_v, fill=255)
        my += line_h

    lum = quality.luminance_under(img, mask)
    alpha = quality.alpha_for_target(lum) if lum is not None else 110
    # Floor keeps type separated from texture even over an already-dark photo;
    # ceiling stops the photograph being drowned.
    alpha = max(70, min(205, alpha))

    # The scrim is anchored to the text block and falls off within roughly one
    # block-height either side, so the top and bottom of the frame keep their
    # colour. An earlier version feathered across h/2 and turned bright images
    # into grey wash.
    # Full strength across the whole text block, feathering only outside it.
    # A centre-anchored gradient under-darkens the first and last lines, so the
    # measured contrast never reaches what alpha_for_target computed.
    pad = int(font_v.size * 0.35)          # ascenders/descenders overshoot the block
    top, bottom = block_top - pad, block_top + block_h + pad
    # Half the darkening is applied to the whole frame and half is eased in over
    # the text, on a long smoothstep falloff. Applying it all as a band — even a
    # feathered one — leaves a visible grey bar across the photograph.
    base = alpha * 0.5
    boost = alpha - base
    feather = max(int(h * 0.28), int(block_h * 0.8))
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    for y in range(h):
        if top <= y <= bottom:
            ease = 1.0
        else:
            t = min(1.0, (top - y if y < top else y - bottom) / feather)
            ease = 1.0 - (3 * t * t - 2 * t * t * t)   # smoothstep
        a = int(base + boost * ease)
        if a > 0:
            od.line([(0, y), (w, y)], fill=(0, 0, 0, a))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    # Gentle vignette, applied separately as a multiply so it adds depth at the
    # corners without lifting the whole frame.
    vign = Image.new("L", (w, h), 0)
    ImageDraw.Draw(vign).ellipse((-w * 0.30, -h * 0.20, w * 1.30, h * 1.20), fill=255)
    vign = vign.filter(ImageFilter.GaussianBlur(radius=w * 0.10)).point(lambda v: 200 + v * 55 // 255)
    img = Image.composite(img, Image.new("RGB", (w, h), (0, 0, 0)), vign)

    # Measure now: once the white type is drawn, sampling the glyph mask would
    # be reading the letters themselves, not the background behind them.
    final_lum = quality.luminance_under(img, mask)

    draw = ImageDraw.Draw(img)

    # Verse — centred, generous leading.
    y = block_top
    for line in lines:
        ty.draw_centered(draw, w, y, line, font_v, (255, 255, 255))
        y += line_h

    # Hairline rule, then the reference in tracked small caps.
    rule_y = y + int(rule_gap * 0.42)
    rule_w = int(w * 0.10)
    draw.line([((w - rule_w) / 2, rule_y), ((w + rule_w) / 2, rule_y)],
              fill=(255, 255, 255, 120), width=max(1, int(h * 0.0012)))

    ref_text = f"{verse.reference}  ·  {verse.translation}".upper()
    ty.draw_centered(draw, w, rule_y + int(rule_gap * 0.42), ref_text, font_r,
                     (255, 255, 255, 235), ty.TRACK_MICRO)

    ok, reason = quality.check_card(final_lum)
    quality.log_result("card", ok, reason)
    if not ok:
        return ""   # caller regenerates with a different background

    out_dir = "/influencer-automation-2.0/storage/verse_cards"
    os.makedirs(out_dir, exist_ok=True)
    if not out_path:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        safe = re.sub(r"[^A-Za-z0-9]+", "-", verse.reference).strip("-").lower()
        out_path = os.path.join(out_dir, f"{stamp}-{kind}-{safe}.jpg")
    # JPEG deliberately: Instagram rejects PNG.
    img.save(out_path, "JPEG", quality=92, optimize=True, progressive=True)
    logger.info(f"verse card written: {out_path}")
    return out_path


# --- 5. caption + publish ----------------------------------------------------


def build_caption(verse: Verse, set_id: Optional[str] = None) -> tuple[str, str]:
    """Caption + the hashtag set used, so reach can be attributed to it later."""
    from app.services import hashtags

    return hashtags.build_caption(verse.text, verse.reference, verse.translation, set_id)


def create_card(kind: str = "post", theme: str = "", subject: Optional[str] = None,
                hashtag_set: Optional[str] = None) -> Optional[dict]:
    """Full generation, no publishing. Returns {path, verse, caption, set_id}."""
    if kind not in ASPECTS:
        logger.error(f"unknown card kind: {kind}")
        return None
    verse = select_verse(theme)
    if not verse:
        return None
    # A rejected card means the background was too bright under the type even
    # after darkening; a different background is cheaper than a bad post.
    path = ""
    for attempt in range(1, 4):
        bg = generate_background(kind=kind, subject=subject)
        if bg is None:
            return None
        path = compose_card(bg, verse, kind=kind)
        if path:
            break
        logger.warning(f"card rejected on contrast; regenerating background ({attempt}/3)")
    if not path:
        logger.error("could not produce a legible card after 3 backgrounds")
        return None
    _remember_reference(verse.reference)
    caption, set_id = build_caption(verse, hashtag_set)
    return {"path": path, "verse": verse, "caption": caption, "kind": kind, "set_id": set_id}


def publish_card(card: dict, publish_at=None) -> dict:
    """Hand a generated card to Postiz as a feed post or a story."""
    from app.services.postiz import PostizService

    svc = PostizService()
    kind = "story" if card.get("kind") == "story" else "post"
    # post_type drives Instagram's media_type: 'story' -> STORIES, else a feed post.
    svc.post_type = kind

    integration = svc.get_configured_integration()
    if not integration.get("success"):
        return integration
    if publish_at is None:
        # kind enforces this type's daily quota as well as the global cap.
        selected = svc.select_publish_at(kind=kind)
        if not selected.get("success"):
            return selected
        publish_at = selected.get("publish_at") or selected.get("date")

    upload = svc.upload_media(card["path"])
    if not upload.get("success"):
        return upload
    result = svc.schedule_post(upload["media"], card["caption"], publish_at,
                               integration=integration["integration"], kind=kind,
                               set_id=card.get("set_id"))
    if result.get("success") and card.get("set_id"):
        # Only mark the set as used once the post actually exists, so a failed
        # publish does not skew the rotation.
        from app.services import hashtags

        hashtags.mark_used(card["set_id"])
    return result
