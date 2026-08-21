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

2. **Only public-domain translations.** See PUBLIC_DOMAIN_TRANSLATIONS, every
   entry of which was checked against bible-api.com's own licence field. The
   default is WEBBE — modern English without WEB's "Yahweh is my shepherd".
   NIV/ESV/NLT are copyrighted with strict quoting limits and are not safe to
   post on an account being grown commercially.
"""

from __future__ import annotations

import io
import json
import os
import random
import re
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Optional

import requests
from loguru import logger
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from app.config import config
from app.services import quality, typography as ty

# --- constants ---------------------------------------------------------------

# Instagram: 4:5 is the tallest permitted feed ratio; 9:16 for stories.
# Feed cards render at 1440 wide for the same reason carousels do — Instagram
# serves feed images up to 1440px on high-DPI screens. Stories stay at 1080
# because 1080x1920 IS the story surface; there is nothing above it to serve.
ASPECTS = {
    "post": (1440, 1800),
    "story": (1080, 1920),
}
# SDXL is trained on ~1 megapixel buckets; generating at a native bucket and
# resampling beats asking for an off-bucket size (which produces duplicated
# subjects and mangled composition).
SDXL_BUCKET = {
    "post": (896, 1152),
    "story": (768, 1344),
}

# Public domain only. NIV/ESV/NLT are copyrighted with strict quoting limits and
# are not safe on an account being grown commercially. Every id here was checked
# against bible-api.com/data, which reports its own licence per translation.
#
# `webbe` is the default rather than `kjv`: it is the modern-English option that
# does NOT have web's "Yahweh is my shepherd" problem (it renders Psalm 23:1 as
# "The LORD is my shepherd"), which was the whole reason KJV was chosen. KJV's
# "that ye may abound" / "which strengtheneth me" is a real shareability tax —
# people do not forward a verse they have to parse.
PUBLIC_DOMAIN_TRANSLATIONS = {"kjv", "web", "webbe", "asv", "bbe", "oeb-us"}
DEFAULT_TRANSLATION = "webbe"

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
    # --- the original thirty -------------------------------------------------
    # Kept verbatim so the account's established look is still in rotation; the
    # additions below widen the range rather than replacing it.
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
    "rain on a window with soft daylight behind",
    "a narrow country path between hedgerows at dusk",
    "olive trees on a hillside in warm afternoon light",
    "low tide on a wide empty beach under pale sky",
    "morning fog rolling through a shallow valley",
    "wildflowers in a field under an overcast sky",
    "sunlight through cathedral of tall birch trees",
    "a calm harbour at first light, no boats moving",
    "frost on bare branches in blue morning light",
    "clouds building over a wide open plain",
    "a mountain stream over smooth stones",
    "tall grass bending in coastal wind",
    "an empty wooden bench beside still water",
    "warm lamplight through a rain-streaked window",
    "sun setting behind distant rolling hills",
    "a quiet forest floor covered in fallen leaves",
    "moonlight over a calm sea",
    "first snow settling on a quiet field",

    # --- texture and macro ---------------------------------------------------
    # The whole account was WIDE. Nothing published was ever closer than a
    # landscape, which is half of why every card read the same at thumbnail
    # size: a distant vista has no dominant shape at 120px. Close focus gives
    # the grid a second scale to alternate with.
    "close detail of weathered olive wood grain",
    "cracked dry earth in soft directional light",
    "a single wheat stalk against a plain background",
    "raindrops beaded on a green leaf",
    "frost crystals forming on a dark surface",
    "close detail of lichen on grey stone",
    "the surface of still water disturbed by one drop",
    "sand ripples in low raking light",
    "close detail of tree bark with deep fissures",
    "a feather resting on dark stone",
    "pale sea glass on wet sand",
    "the underside of a fern frond backlit",
    "smooth river pebbles in shallow clear water",
    "close detail of woven linen cloth",
    "salt crystals on a dark tidal rock",
    "moss growing in the crevice of a boulder",
    "a spider web strung with morning dew",
    "close detail of an unopened flower bud",
    "wind-carved snow crust in blue shade",
    "the ridged surface of a seashell",

    # --- sky and weather -----------------------------------------------------
    # Sky-dominant frames are the cleanest negative space the type ever gets,
    # and they read at thumbnail size when a landscape does not.
    "towering cumulus clouds in clear afternoon light",
    "a break of blue between heavy grey clouds",
    "high cirrus streaks across a pale sky",
    "the last band of colour after sunset",
    "a rainbow forming against dark storm cloud",
    "mackerel sky at first light",
    "rain falling from a distant cloud over open ground",
    "clear night sky thick with stars",
    "the milky way over an empty horizon",
    "a crescent moon in deep blue twilight",
    "sunbeams fanning through a gap in cloud",
    "fog bank rolling in low over water",
    "heat haze rising from a flat plain",
    "a single cloud in an otherwise empty sky",
    "the blue hour over a silent landscape",

    # --- architecture and interior light -------------------------------------
    # Human-made but unpeopled: hard geometry that the all-organic pool never
    # offered, and the strongest available source of directional light.
    "light falling through a tall arched window onto a stone floor",
    "a worn stone staircase in soft shade",
    "sunlight moving across a plain plaster wall",
    "an open wooden door onto bright daylight",
    "a narrow alley of pale stone at midday",
    "rough-hewn stone wall in raking evening light",
    "a quiet cloister walkway in deep shadow",
    "weathered shutters on a sunlit wall",
    "a simple wooden roof beam against white plaster",
    "a stone well head in an empty courtyard",
    "terracotta rooftops under a wide sky",
    "an old stone bridge over still water",

    # --- still life ----------------------------------------------------------
    # Deliberate arrangement, controlled light, and the only frames in the pool
    # with a guaranteed single focal object.
    "an open book on a plain wooden table in window light",
    "a clay cup beside a shuttered window",
    "a loaf of bread on a bare wooden board",
    "olive branches laid on rough linen",
    "a beeswax candle burning in a dim room",
    "a small brass bell on a stone ledge",
    "a bowl of water reflecting a window",
    "dried lavender tied with plain twine",
    "a fishing net folded on a harbour wall",
    "an oil lamp unlit on a stone sill",
    "a handful of grain spilling from a cloth sack",
    "a shepherd\'s staff leaning against a wall",

    # --- water ---------------------------------------------------------------
    "a waterfall through green moss-covered rock",
    "a slow river bend under overhanging trees",
    "clear shallows over pale sand",
    "surf breaking white on dark rocks",
    "a mountain tarn perfectly still at dawn",
    "reeds standing in shallow marsh water",
    "a frozen lake surface under low sun",
    "spray rising at the base of a fall",

    # --- forest and field ----------------------------------------------------
    "a single oak alone in an open field",
    "bluebells carpeting a woodland floor",
    "an avenue of trees in full summer leaf",
    "a hedgerow heavy with autumn berries",
    "terraced vineyards on a warm hillside",
    "a lavender field in rows to the horizon",
    "bare winter trees against a white sky",
    "a stubble field after harvest",
    "pine forest under deep snow",
    "a clearing lit by a single shaft of sun",

    # --- mountain, desert and ice --------------------------------------------
    "a jagged ridge line above the cloud layer",
    "red rock canyon walls in warm reflected light",
    "a glacier face in cold blue shadow",
    "scree slopes under thin high cloud",
    "salt flats mirroring an empty sky",
    "a lone mesa on a flat desert plain",
    "icicles hanging from a dark rock overhang",
    "windswept dunes with a sharp shadow edge",
]

# --- style: one brand core, many looks ---------------------------------------
#
# There used to be exactly ONE style string, appended to every background the
# account has ever generated. Measured across the 41 cards published
# 2026-08-14..27, that produced saturation never above 33/100, lightness never
# above 54/100, and 66% of the feed inside two hue families. Different pixels,
# one impression — which is what a scrolling reader actually perceives.
#
# The fix is to split the string by what it is FOR:
#
#   STYLE_CORE  is identity and legibility. "negative space" and "minimal
#               composition" are not taste; the type has to go somewhere, and
#               the contrast gate in quality.py rejects the card when it
#               doesn\'t. These never rotate.
#
#   STYLE_LOOKS is light and palette, which are pure taste and the only axis
#               the eye reads at thumbnail size. These rotate least-recently-
#               used, exactly like carousel subjects.
#
# Lens focal length rides with the look rather than the core because it moves
# with the light in practice: a long lens suits compressed low-key drama, a
# wide one suits an open saturated midday frame.
#
# `ink` is the OTHER half of the fix, and the more important one. Rotating the
# prompt vocabulary alone cannot produce a light card: the scrim in
# compose_card darkens adaptively to keep WHITE type legible, so a high-key
# background composes to a dark card anyway — measured end to end, six
# backgrounds spanning light, mid and dark all landed "dark". A look that asks
# for bright light therefore also asks for dark type on a lightened ground, and
# rotation alternates the two polarities roughly half and half.
#
# TO REVERT THE ACCOUNT TO ALL-WHITE TYPE: set every `ink` to "light" in
# packs/<name>/pack.yaml. Nothing else needs to change.
STYLE_CORE = (
    "photographic, natural light, serene, minimal composition, "
    "negative space, high detail"
)

STYLE_LOOKS = [
    # The incumbent look, kept first and kept verbatim: everything published so
    # far is this, and it stays in rotation rather than being retired.
    {"name": "muted-warm", "ink": "light",
     "suffix": "shallow depth of field, muted warm tones, cinematic, 35mm"},
    {"name": "high-key-airy", "ink": "dark",
     "suffix": "high-key, bright airy exposure, pale delicate palette, "
               "gentle haze, 50mm"},
    {"name": "low-key-dramatic", "ink": "light",
     "suffix": "low-key, deep shadow, single directional light source, "
               "rich contrast, 85mm"},
    {"name": "backlit-golden", "ink": "light",
     "suffix": "strong backlight, golden hour rim light, warm glow, "
               "atmospheric, 35mm"},
    {"name": "overcast-cool", "ink": "dark",
     "suffix": "flat overcast light, cool blue-grey palette, soft even tone, "
               "50mm"},
    {"name": "clear-vivid", "ink": "dark",
     "suffix": "clear bright daylight, saturated natural colour, crisp, "
               "deep blue sky, 24mm"},
    {"name": "monochrome-quiet", "ink": "dark",
     "suffix": "near-monochrome, tonal greyscale palette, fine film grain, "
               "50mm"},
    {"name": "blue-hour", "ink": "light",
     "suffix": "blue hour, deep twilight, cool indigo palette, soft ambient "
               "light, 35mm"},
]

# The pre-rotation string, unchanged. Retained as the fallback for a pack that
# defines no looks, so "no looks configured" degrades to exactly the behaviour
# that shipped rather than to no style at all.
STYLE_SUFFIX = (
    "photographic, natural light, shallow depth of field, muted warm tones, "
    "cinematic, serene, minimal composition, negative space, 35mm, high detail"
)


@dataclass
class Verse:
    reference: str
    text: str
    translation: str
    # Individual verses behind this reference, as returned by the API:
    # [{"book": str, "chapter": int, "verse": int, "text": str}]. Splitting a
    # long passage happens on THESE boundaries — a card may never cut a verse
    # part-way, and each card can state its own exact range.
    verses: list = field(default_factory=list)


# How much scripture one card carries before it is split across several.
#
# Measured, not guessed. At the text box each layout actually uses, and at
# 0.062w — the smallest size where the type still reads as set rather than
# crammed — a post fits ~191 characters and a story ~219. Beyond that `fit()`
# grinds down to its min_size floor and then silently overflows the band: it
# returns the wrapped lines regardless of whether they fit, which is how a
# 1,100-character Psalm ran through the wordmark and into Instagram's own UI.
#
# Splitting instead of shrinking also keeps the photograph visible, which is
# most of why these cards work at all.
MAX_CHARS = {"post": 190, "story": 215}


# --- config helpers ----------------------------------------------------------


def _cfg(key: str, default):
    value = config.app.get(key, default)
    return default if value in (None, "") else value


def _comfy_url() -> str:
    return str(_cfg("comfyui_base_url", "http://192.168.0.135:8188")).rstrip("/")


# Which visual register a subject belongs to, for the variety gate. Ordered:
# the first pattern that matches wins, so the scale cues ("close detail",
# "macro") are tested before the scene nouns they contain — "close detail of
# lichen on grey stone" is a texture frame, not an architecture one.
#
# This is a bucketing heuristic, not a taxonomy. It exists so quality.py can ask
# "is this the same KIND of picture as the last few?" and a rough answer is
# worth far more than no answer. A subject that matches nothing lands in
# "other", which the gate treats as its own bucket rather than as a wildcard.
#
# "sky" is tested LAST of the scene classes on purpose. Almost every landscape
# prompt in the pool ends "...under a wide sky", so an early sky rule swallowed
# 21 of 115 subjects and starved the terrain buckets. Only a frame with no
# terrain noun at all is really a picture of the sky.
SUBJECT_CLASSES = [
    ("texture", ("close detail", "macro", "surface of", "crystals", "grain",
                 "beaded", "web strung", "woven", "ripples in low")),
    ("still-life", ("book", "cup", "bread", "candle", "bell", "bowl", "lamp",
                    "lavender tied", "net folded", "grain spilling", "staff",
                    "olive branches laid", "feather")),
    ("architecture", ("window", "staircase", "wall", "door", "alley", "cloister",
                      "shutters", "roof", "well head", "rooftops", "bridge",
                      "beam", "floor")),
    ("night", ("stars", "milky way", "moon", "moonlight", "night sky")),
    ("water", ("lake", "river", "sea", "ocean", "waves", "harbour", "stream",
               "waterfall", "shallows", "surf", "tarn", "marsh", "reeds",
               "tide", "beach", "spray", "water")),
    ("snow", ("snow", "frost", "ice", "glacier", "icicles", "frozen")),
    ("desert", ("desert", "dunes", "canyon", "mesa", "salt flats", "dry earth",
                "scree")),
    ("mountain", ("mountain", "ridge", "hills", "hillside", "valley", "boulder")),
    ("forest", ("forest", "trees", "tree", "woodland", "pines", "birch", "fern",
                "moss", "bluebells", "hedgerow", "clearing", "leaves")),
    ("field", ("meadow", "field", "grass", "wheat", "wildflowers", "vineyards",
               "plain", "stubble", "harvest", "path")),
    ("sky", ("cloud", "sky", "rainbow", "sunbeams", "haze rising", "fog bank",
             "blue hour", "after sunset", "cirrus")),
]


def subject_class(subject: str) -> str:
    """The visual register of a background subject. See SUBJECT_CLASSES."""
    text = (subject or "").lower()
    for name, needles in SUBJECT_CLASSES:
        if any(n in text for n in needles):
            return name
    return "other"


def background_subjects() -> list:
    """Background vocabulary, from the pack if it defines one."""
    from app.services import pack

    return pack.typed("verse_card.background_subjects", BACKGROUND_SUBJECTS)


def style_suffix() -> str:
    from app.services import pack

    return pack.value("verse_card.style_suffix", STYLE_SUFFIX)


def style_core() -> str:
    from app.services import pack

    return pack.value("verse_card.style_core", STYLE_CORE)


def style_looks() -> list:
    from app.services import pack

    return pack.typed("verse_card.style_looks", STYLE_LOOKS)


def _rotation_path(name: str) -> str:
    """Rotation state lives beside the used-reference list it is modelled on."""
    return os.path.join(os.path.dirname(_state_path()), name)


def choose_background_subject(remember_it: bool = True) -> str:
    """Least-recently-used background subject.

    This was `random.choice` — the only selection axis in the module without an
    anti-repeat, while verse references had one and carousel subjects had one.
    Random *with replacement* over 30 near-synonymous prompts is why the feed
    read as a single photograph taken 54 times.
    """
    from app.services import rotation

    pool = background_subjects() or BACKGROUND_SUBJECTS
    picked = rotation.choose(pool, _rotation_path("used_backgrounds.json"),
                             remember_it=remember_it)
    return picked or random.choice(pool)


def choose_look(remember_it: bool = True) -> dict:
    """Least-recently-used style look. See STYLE_LOOKS for why this rotates."""
    from app.services import rotation

    looks = style_looks() or STYLE_LOOKS
    names = [str(l.get("name", "")) for l in looks if isinstance(l, dict)]
    by_name = {str(l.get("name", "")): l for l in looks if isinstance(l, dict)}
    if not names:
        return {}
    name = rotation.choose(names, _rotation_path("used_looks.json"),
                           remember_it=remember_it)
    return by_name.get(name or "", {})


def style_for(look: Optional[dict] = None) -> str:
    """The full style string for a look: its light and palette, then the core.

    A look that is missing or malformed falls back to the pre-rotation string,
    so the failure mode is "the account's established look" rather than "a
    background with no style direction at all".
    """
    suffix = (look or {}).get("suffix")
    if not suffix:
        return style_suffix()
    return f"{suffix}, {style_core()}"


def negative_prompt() -> str:
    """The unpeopled guarantee.

    Overridable because a non-faith account has different things to exclude —
    but note brand_motion leans on this being tuned and trusted, so a pack that
    weakens it weakens the Reel imagery guarantee too.
    """
    from app.services import pack

    return pack.value("verse_card.negative_prompt", NEGATIVE_PROMPT)


def _preferred_chars() -> int:
    """Preferred verse length at SELECTION time (see select_verse).

    Distinct from MAX_CHARS, which is the split budget. Reference accounts in
    this niche hold to roughly 6-14 words; 110 characters is about that, and
    leaves the type large enough to read in the feed grid.
    """
    try:
        value = int(_cfg("verse_max_chars", 110))
    except (TypeError, ValueError):
        return 110
    return value if value > 0 else 110


def _translation() -> str:
    t = str(_cfg("verse_translation", DEFAULT_TRANSLATION)).strip().lower()
    if t not in PUBLIC_DOMAIN_TRANSLATIONS:
        logger.warning(
            f"verse_translation '{t}' is not public domain; falling back to "
            f"{DEFAULT_TRANSLATION}. Copyrighted translations (NIV/ESV/NLT) are "
            "not safe to publish."
        )
        return DEFAULT_TRANSLATION
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


def _todays_post_path() -> str:
    return os.path.join(os.path.dirname(_state_path()), "todays_post.json")


def _remember_todays_post(verse: Verse, bg: Image.Image, set_id: Optional[str],
                          series_label: Optional[str] = None,
                          ink: str = "light") -> None:
    """Keep the day's feed card so its story twin can reuse verse and background.

    Instagram's Content Publishing API cannot re-share a feed post to a story —
    the native share, the one with the tap-through sticker, is app-only. So the
    story has to be published as its own media, and rebuilding it from the same
    verse over the same background is what makes a viewer read the two as one
    post rather than two unrelated cards.
    """
    from datetime import datetime, timezone

    bg_path = os.path.join(os.path.dirname(_state_path()), "todays_post_bg.jpg")
    try:
        bg.save(bg_path, "JPEG", quality=95, optimize=True)
        with open(_todays_post_path(), "w", encoding="utf-8") as fh:
            json.dump({
                "date": datetime.now(timezone.utc).date().isoformat(),
                "reference": verse.reference,
                "text": verse.text,
                "translation": verse.translation,
                "set_id": set_id,
                "series_label": series_label,
                "bg_path": bg_path,
                # The twin must compose with the SAME ink. This background was
                # generated for one polarity; veiling a high-key frame in black
                # to carry white type would both look wrong and very likely fail
                # the contrast gate, costing the twin for no reason.
                "ink": ink,
            }, fh)
    except (OSError, ValueError) as exc:
        logger.warning(f"could not persist today's feed card: {exc}")


def load_todays_post() -> Optional[dict]:
    """The feed card generated today (UTC), or None. Quota days are UTC."""
    from datetime import datetime, timezone

    try:
        with open(_todays_post_path(), encoding="utf-8") as fh:
            record = json.load(fh)
    except (OSError, ValueError):
        return None
    if record.get("date") != datetime.now(timezone.utc).date().isoformat():
        return None
    if not os.path.exists(str(record.get("bg_path", ""))):
        return None
    return record


def twin_pending() -> bool:
    """Is today's feed card still waiting for its story twin?

    Deliberately not 'is this the first story of the day': a story left over
    from a previous day's roll-forward can occupy the first slot, and the twin
    would then be skipped on the one day it matters. The twin is defined by
    pairing with the feed card, not by ordering.
    """
    record = load_todays_post()
    return bool(record) and not record.get("twin_done")


def mark_twin_done() -> None:
    """Record that today's twin exists, so later runs publish standalone stories."""
    record = load_todays_post()
    if not record:
        return
    record["twin_done"] = True
    try:
        with open(_todays_post_path(), "w", encoding="utf-8") as fh:
            json.dump(record, fh)
    except OSError as exc:
        logger.warning(f"could not mark today's twin as done: {exc}")


# --- 1. reference selection (LLM proposes, API verifies) ---------------------

_REF_RE = re.compile(r"^([1-3]\s*)?[A-Za-z][A-Za-z ]{1,20}\s+(\d{1,3}):(\d{1,3})(?:-(\d{1,3}))?$")

# Splitting handles a long passage gracefully, but the best card is still one
# thought. The regex used to accept any range, so "Psalm 139:7-18" — twelve
# verses, ~1,100 characters — passed as a perfectly legal reference and produced
# a wall of type. Cap it at the source and let split_verse() be the safety net.
_MAX_VERSES_PER_REFERENCE = 4


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
        "Choose a well-known, encouraging verse suitable for a general audience. "
        f"Use at most {_MAX_VERSES_PER_REFERENCE} consecutive verses — a single verse "
        "is usually best. Long passages do not fit on a card."
    )
    try:
        raw = llm._generate_response(prompt)
    except Exception as exc:  # noqa: BLE001 - any LLM failure is non-fatal here
        logger.warning(f"reference selection failed: {exc}")
        return None

    candidate = (raw or "").strip().strip('"').strip("'").splitlines()[0].strip()
    candidate = re.sub(r"^[\s\-\*\d.]+", "", candidate).strip()
    match = _REF_RE.match(candidate)
    if not match:
        logger.warning(f"LLM returned an unusable reference: {candidate!r}")
        return None
    start, end = match.group(3), match.group(4)
    if end and int(end) - int(start) + 1 > _MAX_VERSES_PER_REFERENCE:
        logger.warning(
            f"reference {candidate!r} spans "
            f"{int(end) - int(start) + 1} verses (max {_MAX_VERSES_PER_REFERENCE}); "
            "rejecting so a shorter one is chosen"
        )
        return None
    return candidate


# --- 2. authoritative text ---------------------------------------------------


_QUOTE_CHARS = "“”\"'‘’"


def _strip_dangling_quotes(text: str) -> str:
    """Drop a leading or trailing quote mark that has no partner.

    Translations that mark direct speech hand back fragments like
    ``“Come to me, all you who labour`` (open, never closed) or
    ``...wherever you go.”`` (closed, never opened), because the quotation spans
    verses the reference does not. The card and the caption both wrap the text in
    their own quotes, so an unbalanced one renders as ``““Come to me...`` — the
    kind of small wrongness that reads as carelessness on a scripture account.

    Only strips when the count is genuinely unbalanced, so a verse that opens and
    closes its own quotation keeps both marks.
    """
    if not text:
        return text
    opens = text.count("“")
    closes = text.count("”")
    if opens > closes and text[0] in _QUOTE_CHARS:
        text = text[1:].lstrip()
    elif closes > opens and text[-1] in _QUOTE_CHARS:
        text = text[:-1].rstrip()
    return text


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

    text = _strip_dangling_quotes(" ".join((data.get("text") or "").split()))
    if not text:
        return None
    verses = []
    for item in (data.get("verses") or []):
        vtext = _strip_dangling_quotes(" ".join(str(item.get("text") or "").split()))
        if not vtext:
            continue
        verses.append({
            "book": str(item.get("book_name") or "").strip(),
            "chapter": int(item.get("chapter") or 0),
            "verse": int(item.get("verse") or 0),
            "text": vtext,
        })
    return Verse(
        reference=str(data.get("reference") or reference).strip(),
        text=text,
        translation=str(data.get("translation_id") or translation).upper(),
        verses=verses,
    )


def _format_reference(chunk: list) -> str:
    """Exact reference for a run of verses, e.g. 'Psalms 139:7-9'."""
    if not chunk:
        return ""
    book = chunk[0]["book"]
    first, last = chunk[0], chunk[-1]
    if first["chapter"] == last["chapter"]:
        if first["verse"] == last["verse"]:
            return f"{book} {first['chapter']}:{first['verse']}"
        return f"{book} {first['chapter']}:{first['verse']}-{last['verse']}"
    return (f"{book} {first['chapter']}:{first['verse']}-"
            f"{last['chapter']}:{last['verse']}")


def split_verse(verse: Verse, kind: str = "post") -> list:
    """Split a passage across cards, on verse boundaries only.

    Scripture is never cut mid-verse: a verse is the smallest unit a card may
    carry. Verses are packed greedily up to the card's character budget, and a
    single verse longer than the budget gets a card to itself rather than being
    broken — the type shrinks for that one card, which is the lesser evil.

    Each part gets its OWN exact reference (Psalms 139:7-9, then 139:10-13), so
    a reader is never shown a range wider than the words in front of them.
    Returns [verse] unchanged when it already fits.
    """
    budget = MAX_CHARS.get(kind, MAX_CHARS["post"])
    if len(verse.text) <= budget:
        return [verse]
    if not verse.verses:
        # No per-verse structure to split on (shouldn't happen with bible-api,
        # but a card that overflows is worse than one that is merely long).
        logger.warning(f"{verse.reference}: {len(verse.text)} chars and no verse "
                       "structure to split on; leaving as one card")
        return [verse]

    chunks, current, current_len = [], [], 0
    for item in verse.verses:
        addition = len(item["text"]) + (1 if current else 0)
        if current and current_len + addition > budget:
            chunks.append(current)
            current, current_len = [item], len(item["text"])
        else:
            current.append(item)
            current_len += addition
    if current:
        chunks.append(current)

    parts = [
        Verse(
            reference=_format_reference(chunk),
            text=" ".join(c["text"] for c in chunk),
            translation=verse.translation,
            verses=chunk,
        )
        for chunk in chunks
    ]
    logger.info(f"{verse.reference} ({len(verse.text)} chars) split into "
                f"{len(parts)} {kind} cards: "
                + ", ".join(f"{p.reference} [{len(p.text)}]" for p in parts))
    return parts


def select_verse(theme: str = "", attempts: int = 6) -> Optional[Verse]:
    """LLM proposes, the API decides. Unfetchable references are discarded.

    Verses longer than the preferred budget are re-rolled rather than rejected.
    MAX_CHARS is the point at which a passage is SPLIT across cards; this is a
    much lower bar, about what a card can set at a size that survives the feed
    grid. Romans 15:13 is 130 characters, fits on one card, and sets as seven
    lines of type — legible full-screen and grey mush at thumbnail size, which
    is where the decision to stop scrolling is actually made.

    It is a preference, not a gate: the first verified verse is kept as a
    fallback and returned if nothing shorter turns up, so a long verse costs
    variety rather than the whole slot.
    """
    budget = _preferred_chars()
    avoid = _recent_references()
    fallback: Optional[Verse] = None
    for _ in range(attempts):
        ref = pick_reference(theme, avoid=avoid)
        if not ref:
            continue
        verse = fetch_verse(ref)
        if not verse:
            avoid = avoid + [ref]
            continue
        if len(verse.text) <= budget:
            return verse
        if fallback is None:
            fallback = verse
        logger.info(f"{verse.reference} is {len(verse.text)} chars (over the "
                    f"{budget}-char card budget); asking for a shorter one")
        avoid = avoid + [ref]
    if fallback is not None:
        logger.warning(f"no verse under {budget} chars after {attempts} attempts; "
                       f"using {fallback.reference} ({len(fallback.text)} chars)")
        return fallback
    logger.error("could not obtain a verified verse after %d attempts" % attempts)
    return None


# --- 3. background generation (local ComfyUI / SDXL) -------------------------


def hires_size(bucket: tuple[int, int], target: tuple[int, int]) -> tuple[int, int]:
    """Second-pass size: big enough to cover the canvas, at the BUCKET's ratio.

    Upscaling straight to the canvas ratio would stretch the image — the 4:5
    canvas and the 896x1152 bucket are not the same shape. Keeping the bucket's
    ratio and letting `_cover` crop the difference preserves the geometry SDXL
    composed. Latent dimensions must be multiples of 8.
    """
    bw, bh = bucket
    tw, th = target
    if bw <= 0 or bh <= 0:
        return target
    scale = max(tw / bw, th / bh)
    if scale <= 1.0:
        return bucket                      # already covers the canvas
    return (max(8, round(bw * scale / 8) * 8), max(8, round(bh * scale / 8) * 8))


def _workflow(prompt: str, width: int, height: int, seed: int, ckpt: str,
              hires: Optional[tuple[int, int]] = None) -> dict:
    """SDXL at a native bucket, then an optional detail pass at output size.

    The cards were being generated at 896x1152 and enlarged to fill the canvas —
    a 1.21x upscale for feed posts and 1.43x for stories, so every card and
    story ever published was softer than it needed to be. Generating larger
    outright is not the fix: off-bucket sizes are what produce duplicated
    subjects, which is why the buckets are there.

    So the composition is still decided at the bucket, and a second low-denoise
    pass re-samples it at output size. Denoise is 0.4 — high enough to resolve
    real detail rather than interpolate it, low enough that the composition,
    and with it the negative prompt's guarantees about what is in frame, does
    not change.

    The first pass keeps its previous settings exactly (30 steps, cfg 6.0, same
    seed), so the composition a given seed produces is unchanged — only its
    resolution is. `brand_motion._seed_workflow` runs the same two-pass shape
    for Reel seed frames and stays deliberately separate: it is tuned for
    photoreal stills (cfg 5.0, denoise 0.45) and its own comment explains why
    it would not impose those settings on the daily feed. This pass exists for
    the cards' own sake, not to serve the Reels.
    """
    flow = {
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
        "7": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative_prompt(), "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "versecard"}},
    }
    if hires and hires != (width, height):
        hw, hh = hires
        flow["10"] = {"class_type": "LatentUpscale",
                      "inputs": {"samples": ["3", 0], "upscale_method": "bislerp",
                                 "width": hw, "height": hh, "crop": "disabled"}}
        flow["11"] = {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed, "steps": 20, "cfg": 6.0,
                "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 0.4,
                "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0],
                "latent_image": ["10", 0],
            },
        }
        flow["8"]["inputs"]["samples"] = ["11", 0]
    return flow


def generate_background(kind: str = "post", subject: Optional[str] = None,
                        seed: Optional[int] = None, timeout: int = 300,
                        look: Optional[dict] = None) -> Optional[Image.Image]:
    """One background image. See generate_background_tagged for the metadata."""
    img, _meta = generate_background_tagged(kind=kind, subject=subject, seed=seed,
                                            timeout=timeout, look=look)
    return img


def generate_background_tagged(kind: str = "post", subject: Optional[str] = None,
                               seed: Optional[int] = None, timeout: int = 300,
                               look: Optional[dict] = None) -> tuple:
    """(image, meta) — meta carries what the variety gate needs to compare.

    Split from generate_background so every existing caller keeps its plain
    Image return, while create_card can see which subject and look it got and
    refuse a card that repeats the recent feed.
    """
    base = _comfy_url()
    width, height = SDXL_BUCKET.get(kind, SDXL_BUCKET["post"])
    hires = hires_size((width, height), ASPECTS.get(kind, ASPECTS["post"]))
    # Rotation is only consulted when the caller has not pinned a value. A
    # pinned subject (the story twin, a series run, brand_footage's cache key)
    # must stay pinned, and recording it as "used" would corrupt the LRU with
    # picks it never made.
    subject = subject or choose_background_subject()
    look = look if look is not None else choose_look()
    prompt = f"{subject}, {style_for(look)}"
    seed = seed if seed is not None else random.randint(1, 2**31 - 1)
    ckpt = str(_cfg("comfyui_checkpoint", "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"))
    meta = {"subject": subject, "look": (look or {}).get("name", ""),
            "subject_class": subject_class(subject), "seed": seed,
            "ink": (look or {}).get("ink", "light")}

    logger.info(f"generating background ({kind} {width}x{height} -> "
                f"{hires[0]}x{hires[1]}): {subject} [{meta['look'] or 'default'}]")
    try:
        resp = requests.post(f"{base}/prompt",
                             json={"prompt": _workflow(prompt, width, height, seed, ckpt, hires)},
                             timeout=30)
        resp.raise_for_status()
        prompt_id = resp.json()["prompt_id"]
    except (requests.exceptions.RequestException, KeyError, ValueError) as exc:
        logger.error(f"ComfyUI submit failed: {exc}")
        return None, meta

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
                    return Image.open(io.BytesIO(raw)).convert("RGB"), meta
            logger.error("ComfyUI finished but produced no image")
            return None, meta
        time.sleep(2)
    logger.error(f"ComfyUI timed out after {timeout}s")
    return None, meta


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
                 out_path: Optional[str] = None, point_at_post: bool = False,
                 series_label: Optional[str] = None,
                 part_of: Optional[tuple] = None, ink: str = "light") -> str:
    """One finished card. `ink` is the colour of the TYPE, not of the card.

    "light" is white type over a darkened photograph — the only card this
    pipeline could make until now, and the reason every card it ever published
    measures dark: the scrim darkens adaptively to protect white type, so a
    high-key background composes to a dark card regardless. "dark" is the other
    polarity, DARK_INK over a lightened photograph, which is what actually lets
    a light card exist. See quality.py for the maths; the two are not
    symmetric.
    """
    w, h = ASPECTS.get(kind, ASPECTS["post"])
    dark_ink = ink == "dark"
    ink_rgb = quality.DARK_INK if dark_ink else (255, 255, 255)
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
    # Clamp to the top of the band. `fit()` returns wrapped lines even when they
    # do not fit its box, so an over-long passage produced a block taller than
    # the band; centring it then pushed the first line ABOVE band_top, through
    # the story-safe zone, and the last line down through the wordmark.
    # split_verse() should prevent this, so an overflow here is a bug worth
    # seeing rather than absorbing silently.
    if block_h > band_h:
        logger.warning(
            f"{verse.reference}: text block {block_h}px exceeds the {band_h}px "
            f"{kind} band at {font_v.size}px — expected split_verse() to have "
            "divided this passage"
        )
    block_top = max(band_top, band_top + (band_h - block_h) // 2)

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
    alpha = quality.alpha_for_target(lum, ink=ink) if lum is not None else 110
    # Floor keeps type separated from texture even over an already-suitable
    # photo; ceiling stops the photograph being drowned. The light-ground
    # ceiling is higher: white veiling reads as haze and stays photographic
    # further up than black veiling, which goes to mud.
    alpha = max(70, min(225 if dark_ink else 205, alpha))

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
    veil = (255, 255, 255) if dark_ink else (0, 0, 0)
    overlay = Image.new("RGBA", (w, h), (*veil, 0))
    od = ImageDraw.Draw(overlay)
    for y in range(h):
        if top <= y <= bottom:
            ease = 1.0
        else:
            t = min(1.0, (top - y if y < top else y - bottom) / feather)
            ease = 1.0 - (3 * t * t - 2 * t * t * t)   # smoothstep
        a = int(base + boost * ease)
        if a > 0:
            od.line([(0, y), (w, y)], fill=(*veil, a))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    # Gentle vignette, applied separately as a multiply so it adds depth at the
    # corners without lifting the whole frame. On a light card it is inverted:
    # darkening the corners of an airy frame is exactly the move that would drag
    # it back toward the one look this whole change exists to escape.
    vign = Image.new("L", (w, h), 0)
    ImageDraw.Draw(vign).ellipse((-w * 0.30, -h * 0.20, w * 1.30, h * 1.20), fill=255)
    vign = vign.filter(ImageFilter.GaussianBlur(radius=w * 0.10)).point(
        lambda v: (228 + v * 27 // 255) if dark_ink else (200 + v * 55 // 255))
    img = Image.composite(img, Image.new("RGB", (w, h), (255, 255, 255) if dark_ink
                                         else (0, 0, 0)), vign)

    # Measure now: once the white type is drawn, sampling the glyph mask would
    # be reading the letters themselves, not the background behind them.
    final_lum = quality.luminance_under(img, mask)

    draw = ImageDraw.Draw(img)

    # Verse — centred, generous leading.
    y = block_top
    for line in lines:
        ty.draw_centered(draw, w, y, line, font_v, ink_rgb)
        y += line_h

    # Hairline rule, then the reference in tracked small caps.
    rule_y = y + int(rule_gap * 0.42)
    rule_w = int(w * 0.10)
    draw.line([((w - rule_w) / 2, rule_y), ((w + rule_w) / 2, rule_y)],
              fill=(*ink_rgb, 120), width=max(1, int(h * 0.0012)))

    # The reference always states the verses ON THIS CARD, and when a passage is
    # split the card says which part it is — so a reader can tell at a glance
    # that there is more, and never sees a range wider than the words shown.
    ref_text = f"{verse.reference}  ·  {verse.translation}"
    if part_of and part_of[1] > 1:
        ref_text += f"  ·  {part_of[0]} of {part_of[1]}"
    ty.draw_centered(draw, w, rule_y + int(rule_gap * 0.42), ref_text.upper(), font_r,
                     (*ink_rgb, 235), ty.TRACK_MICRO)

    ok, reason = quality.check_card(final_lum, ink)
    quality.log_result("card", ok, reason)
    if not ok:
        return ""   # caller regenerates with a different background

    # Wordmark — the same discreet mark the carousels carry, so cards and
    # carousels read as one account on the profile grid.
    from app.services import carousel as _carousel

    mark_size = int(w * 0.019)
    f_mark = ty.font(ty.SERIF, mark_size, "Light")
    mark_y = int(h * (0.915 if kind == "post" else 0.845))
    ty.draw_centered(draw, w, mark_y, _carousel.wordmark(), f_mark,
                     (*ink_rgb, 190), ty.TRACK_WORDMARK)

    if point_at_post and kind == "story":
        f_ptr = ty.font(ty.SANS, int(w * 0.016), "Medium")
        ty.draw_centered(draw, w, int(h * 0.885), "NEW POST TODAY  ·  TAP THROUGH",
                         f_ptr, (*ink_rgb, 225), ty.TRACK_MICRO)

    # Series line, above the verse block: it is a label, not part of the
    # scripture, and putting it at the top stops it reading as a citation.
    # Dimmer than the reference so the verse still leads the eye.
    if series_label:
        f_series = ty.font(ty.SANS, int(w * 0.0155), "Medium")
        series_y = int(h * (0.085 if kind == "post" else 0.145))
        ty.draw_centered(draw, w, series_y, series_label.upper(), f_series,
                         (*ink_rgb, 170), ty.TRACK_MICRO)

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


def _discard(paths: list) -> None:
    """Remove cards this run wrote and then decided not to use.

    compose_card writes its JPEG before returning, and the variety check can
    only run on the composed card — so a rejected candidate has already landed
    in the published-cards directory. Leaving it there would both waste disk and
    corrupt feed_variety_report.py, which measures that directory to decide what
    the account has been publishing.

    Only ever called with paths this function created moments earlier.
    """
    for p in paths:
        try:
            os.remove(p)
        except OSError as exc:  # noqa: BLE001 - a leftover file is not fatal
            logger.debug(f"could not remove rejected card {p}: {exc}")


def create_card(kind: str = "post", theme: str = "", subject: Optional[str] = None,
                hashtag_set: Optional[str] = None,
                reference: Optional[str] = None,
                series_label: Optional[str] = None) -> Optional[dict]:
    """Full generation, no publishing. Returns {path, verse, caption, set_id}.

    `reference` pins the verse instead of asking the LLM to propose one (used by
    the series runs). It is still fetched from the bible API and rendered
    verbatim — pinning removes the LLM from the choice, not the verification.
    """
    if kind not in ASPECTS:
        logger.error(f"unknown card kind: {kind}")
        return None
    if reference:
        verse = fetch_verse(reference)
        if not verse:
            logger.error(f"series reference could not be fetched: {reference!r}")
            return None
    else:
        verse = select_verse(theme)
    if not verse:
        return None
    # A long passage becomes several cards rather than one crammed one. Each
    # part shares the SAME background, so the set reads as one piece across a
    # carousel or a run of stories.
    parts = split_verse(verse, kind)
    total = len(parts)

    # A rejected card means the background was too bright under the type even
    # after darkening; a different background is cheaper than a bad post.
    #
    # Two gates run here and they are NOT equal. Contrast is a floor: a card
    # that fails it is illegible and is never published. Variety is a
    # preference: a card that repeats the recent feed is worth less than a
    # different one but far more than a missed slot, so a repetitive card is
    # kept as a fallback and published if the pool will not do better. See
    # quality.check_variety.
    from app.services import quality, rotation

    register_state = _rotation_path("used_registers.json")
    history = rotation.load_history(register_state)

    paths: list[str] = []
    bg = None
    register: dict = {}
    fallback: Optional[tuple] = None   # (paths, bg, register, reason)
    for attempt in range(1, 5):
        bg, meta = generate_background_tagged(kind=kind, subject=subject)
        if bg is None:
            return None
        # Only the twin points at the feed post — see create_story_from_todays_post.
        # This used to fire on any story generated on a day a post went out, but
        # the pointer was decided at GENERATION time while the story can be
        # scheduled into the next day, so it could promise "new post today" on a
        # morning with no post. The twin cannot drift that way: it is built from
        # the card it points at.
        paths = []
        ink = meta.get("ink", "light")
        for i, part in enumerate(parts, start=1):
            p = compose_card(bg, part, kind=kind, series_label=series_label,
                             part_of=(i, total), ink=ink)
            if not p:
                paths = []
                break
            paths.append(p)
        if paths:
            # Measured on the COMPOSED card, not on `bg`. The scrim in
            # compose_card darkens adaptively to hit the contrast target, so a
            # bright background becomes a dark card — measuring the raw
            # background would score variety the audience never sees.
            try:
                composed = Image.open(paths[0])
            except OSError:
                composed = bg
            register = quality.register_of(composed, meta)
            varied, why = quality.check_variety(register, history)
            quality.log_result(f"variety/{kind}", varied, why)
            if varied:
                if fallback is not None:
                    _discard(fallback[0])   # a better card won; drop the spare
                break
            if fallback is None:
                fallback = (paths, bg, register, why)
            else:
                _discard(paths)
            paths = []
            logger.info(f"card repeats the recent feed; trying another "
                        f"background ({attempt}/4)")
            continue
        logger.warning(f"card rejected on contrast; regenerating background ({attempt}/4)")
    if not paths and fallback is not None:
        # Every candidate repeated the feed. Publishing the first legible one is
        # the lesser failure — see the note on the loop above.
        paths, bg, register, why = fallback
        logger.warning(f"publishing a repetitive card, no varied background "
                       f"available after 4 tries: {why}")
    if not paths:
        logger.error("could not produce a legible card after 4 backgrounds")
        return None
    if register:
        rotation.remember(register_state, register, keep=200)
    _remember_reference(verse.reference)
    caption, set_id = build_caption(verse, hashtag_set)
    if series_label:
        # The first line of a caption is what Instagram indexes for search, so
        # the series name leads rather than trailing after the verse.
        caption = f"{series_label}\n\n{caption}"
    if kind == "post" and bg is not None:
        _remember_todays_post(verse, bg, set_id, series_label,
                              ink=register.get("ink", "light"))
    return {"path": paths[0], "paths": paths, "parts": parts,
            "verse": verse, "caption": caption, "kind": kind,
            "set_id": set_id, "series_label": series_label}


def create_story_from_todays_post() -> Optional[dict]:
    """The story twin of today's feed card: same verse, same background, at 9:16.

    Returns None when there is no feed card today, or when the story crop fails
    the contrast gate — the caller then falls back to an independent story, so a
    bad crop costs variety rather than the whole slot.
    """
    record = load_todays_post()
    if not record:
        return None
    try:
        bg = Image.open(record["bg_path"]).convert("RGB")
    except (OSError, ValueError, KeyError) as exc:
        logger.warning(f"could not reopen today's background: {exc}")
        return None
    verse = Verse(reference=record["reference"], text=record["text"],
                  translation=record["translation"])
    # The background cleared the gate at 4:5; the 9:16 crop samples different
    # pixels, so it is measured again rather than assumed.
    #
    # The twin carries the SAME series label as the card it is built from. It is
    # the one story that is genuinely part of the series, and without the label
    # the pair reads as two unrelated cards — which is the exact thing rebuilding
    # it from the same verse and background exists to prevent.
    series_label = record.get("series_label") or None
    path = compose_card(bg, verse, kind="story", point_at_post=True,
                        series_label=series_label,
                        ink=record.get("ink", "light"))
    if not path:
        logger.warning("story twin rejected on contrast; falling back to a fresh story")
        return None
    # Deliberately NOT reusing the post's hashtag set: insights attribute reach
    # per set, and scoring one set with both a feed post and a story on the same
    # day would bias the rotation with numbers the two surfaces do not share.
    caption, set_id = build_caption(verse, None)
    caption = f"{caption}\n\nNew post on the grid today."
    logger.info(f"story twin of today's feed card: {verse.reference}")
    return {"path": path, "verse": verse, "caption": caption, "kind": "story",
            "set_id": set_id, "twin_of_post": True}


def publish_card(card: dict, publish_at=None) -> dict:
    """Hand a generated card to Postiz as a feed post or a story.

    A split passage publishes as a CAROUSEL in the feed and as a RUN of stories,
    because Instagram allows several images in one post but only one per story.
    """
    from datetime import timedelta

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

    paths = card.get("paths") or [card["path"]]
    # Instagram's hard limit on carousel children. A passage needing more than
    # ten cards is not a post, it is a chapter — publish the first ten and say so.
    if kind == "post" and len(paths) > 10:
        logger.warning(f"{len(paths)} cards exceeds Instagram's 10-image carousel "
                       "limit; publishing the first 10")
        paths = paths[:10]

    media = []
    for path in paths:
        upload = svc.upload_media(path)
        if not upload.get("success"):
            return upload
        media.append(upload["media"])

    # What produced this card, recorded at publish time because it cannot be
    # reconstructed later: the verse rotates, the series advances, and by the
    # time the numbers arrive two days on, the card is just a JPEG. Reels have
    # carried a variant since retention tracking; cards and carousels carried
    # none, so reach could only ever be attributed to the hashtag set.
    verse = card.get("verse")
    variant = {
        "format": kind,
        "reference": getattr(verse, "reference", None),
        "translation": getattr(verse, "translation", None),
        "verse_chars": len(getattr(verse, "text", "") or "") or None,
        "series": card.get("series_label"),
        "twin_of_post": bool(card.get("twin_of_post")) or None,
        "parts": len(paths) if len(paths) > 1 else None,
    }
    variant = {k: v for k, v in variant.items() if v is not None}

    if kind == "story":
        # Instagram stories take exactly one image each ("if it's a story, it can
        # have only one picture" — Postiz's own provider). A split passage is
        # therefore a RUN of stories, spaced a few minutes apart so they appear
        # in reading order in the tray.
        results = []
        for i, item in enumerate(media):
            at = publish_at + timedelta(minutes=2 * i) if i else publish_at
            res = svc.schedule_post(item, card["caption"], at,
                                    integration=integration["integration"], kind=kind,
                                    set_id=card.get("set_id") if i == 0 else None,
                                    variant=variant if i == 0 else None)
            results.append(res)
            if not res.get("success"):
                logger.error(f"story part {i + 1}/{len(media)} failed: "
                             f"{res.get('error') or res.get('message')}")
                break
        result = results[0] if results else {"success": False, "error": "no story parts"}
        result = dict(result)
        result["parts"] = len(results)
    else:
        # One post; several media makes it a carousel (media_type=CAROUSEL).
        result = svc.schedule_post(media if len(media) > 1 else media[0],
                                   card["caption"], publish_at,
                                   integration=integration["integration"], kind=kind,
                                   set_id=card.get("set_id"), variant=variant)
        result = dict(result)
        result["parts"] = len(media)

    if result.get("success") and card.get("set_id"):
        # Only mark the set as used once the post actually exists, so a failed
        # publish does not skew the rotation.
        from app.services import hashtags

        hashtags.mark_used(card["set_id"])
    return result
