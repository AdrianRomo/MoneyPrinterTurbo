"""Real, correctly-licensed photography from Wikimedia Commons.

Carousels in the "creativity of God in <subject>" format label real places, and
their audiences fact-check — the account this format was modelled on carries a
public correction about mislabelling one slide. So the images must be real
photographs with real place names and proper attribution. AI imagery is used
elsewhere in this codebase (verse_card.py) but is never labelled as a location.

Two rules here:

1. **Licence allowlist.** Only public-domain and CC BY / CC BY-SA files are
   accepted, and the author is always captured so the caption can credit them —
   CC BY *requires* attribution.

2. **A location is never invented.** It is parsed out of the file's own title,
   and if nothing parses confidently the slide simply carries no location label.
   An unlabelled slide is correct; a wrong label is not.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Optional

import requests
from loguru import logger
from PIL import Image

API = "https://commons.wikimedia.org/w/api.php"
UA = "influencer-automation/1.0 (homelab; https://adrianromo.me)"

# Public domain plus the CC licences that permit commercial use with credit.
# Deliberately excludes NC/ND variants, which forbid commercial posting.
ALLOWED_LICENCES = re.compile(
    r"^(cc0|public domain|pd|cc by(?:-sa)? ?[0-9.]*|cc-by(?:-sa)?-[0-9.]+)$", re.I
)

# Words that parse like a place but are not one. Printing "unsplash" under a
# photograph as though it were a location is exactly the class of error this
# module exists to avoid.
NON_PLACES = {
    "unsplash", "pexels", "pixabay", "flickr", "wikimedia", "commons", "wikipedia",
    "panorama", "panoramio", "hdr", "raw", "jpeg", "canon", "nikon", "sony", "fujifilm",
    "camera", "lens", "drone", "dji", "gopro", "iphone", "author", "own work",
    "creative commons", "public domain", "photo", "photograph", "image", "picture",
    "featured", "quality image", "wallpaper", "stock",
}

# --- Sizing, all of it measured against the crop -------------------------
#
# Slides are a 4:5 portrait frame, so a landscape photo loses most of its width
# to the centre crop. Every size rule below is therefore expressed against the
# CROP and never against the file, because the two come apart badly: a
# 12500x2332 panorama is an enormous file and a 286px-wide slide.
TARGET_W, TARGET_H = 1440, 1800

# Margin over the crop's hard requirement, applied only when a bigger fetch was
# needed anyway. Resolving detail *down* with LANCZOS is what makes a slide look
# sharp; arriving at the exact size leaves nothing to resolve.
OVERSAMPLE = 1.15

# upload.wikimedia.org serves thumbnails up to 3840px wide and silently snaps
# larger requests down to it — while the API still reports the width you asked
# for. Trust the downloaded image, never `thumbwidth`.
COMMONS_MAX_THUMB = 3840

# What search() asks for. Kept modest: most candidates are never downloaded,
# and download() re-asks at the width the crop actually needs.
SEARCH_THUMB_WIDTH = 1600

# Past this ratio the crop cannot fill the frame even from a 3840px thumbnail
# (3840 / 1800 = 2.13), and a 4:5 crop of a panorama is an arbitrary sliver of
# a photograph that was composed to be wide. Both reasons point the same way,
# so the wide limit is a quality gate and a composition gate at once.
MAX_ASPECT = 2.05
MIN_ASPECT = 0.50

# Commons' search ANDs its terms, so a three-word query like "whale ocean
# breaching" demands all three appear and returns nothing. Subject queries are
# therefore OR groups of single words — which restores the pool but widens the
# tail, because an OR clause only has to match *one* term to rank. These two
# guards keep the tail honest, and both work the way the location parser does:
# checked against the file's own metadata, never inferred.

# A result must actually mention one of the terms it was searched for. Latin
# binomials are deliberately included in the subject queries so that they widen
# the search and this check together.
#
# Matching starts at a word boundary but does not end at one, so "moon" still
# accepts "moons" while no longer accepting the "m" of "airborne". Plain
# substring matching let an iceberg through on the "milky" inside a sentence.
def _term_pattern(query: str) -> Optional[re.Pattern]:
    terms = [t for t in re.findall(r"[A-Za-zÀ-ÿ]{3,}", query) if t.upper() != "OR"]
    if not terms:
        return None
    # A trailing "s" is trimmed so a plural query term still matches singular
    # prose ("mountains" against "a mountain peak"). Irregular plurals are not
    # inferred — spell both forms in the query instead, as "galaxy OR galaxies"
    # does; guessing at morphology is how a guard starts inventing things.
    stems = {t[:-1] if len(t) > 4 and t.lower().endswith("s") else t for t in terms}
    return re.compile(r"\b(?:%s)" % "|".join(re.escape(t) for t in sorted(stems)), re.I)


# This format promises real photographs. The astronomy pools in particular are
# full of "artist's impression" renderings, which are exactly what the module
# exists to exclude.
NOT_PHOTOGRAPHS = re.compile(
    r"artist'?s\s+(impression|concept)|illustration|diagram|artwork|painting"
    r"|drawing|\brender(ing)?\b|simulation|\bchart\b|\bmap\b|\blogo\b",
    re.I,
)

# Commons' own curated quality pools — the difference between this format
# looking premium and looking like a stock-photo dump.
QUALITY_POOLS = [
    "incategory:Featured_pictures_of_landscapes",
    "incategory:Quality_images",
]


@dataclass
class Photo:
    title: str
    url: str
    width: int
    height: int
    author: str
    licence: str
    location: Optional[str]
    descriptionurl: str


def _plain(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html or "")).strip()


def crop_dimensions(width: int, height: int,
                    target: tuple[int, int] = (TARGET_W, TARGET_H)) -> tuple[int, int]:
    """The pixels of a source that survive a centre crop to the target ratio."""
    tw, th = target
    if width <= 0 or height <= 0:
        return (0, 0)
    if width * th > height * tw:      # wider than the frame: height is binding
        return (int(height * tw / th), height)
    return (width, int(width * th / tw))


def fetch_width_for(width: int, height: int,
                    target: tuple[int, int] = (TARGET_W, TARGET_H)) -> int:
    """The source width at which the centre crop exactly fills the frame.

    This is the whole fix for soft carousels: the search thumbnail is sized by
    *width*, but a portrait crop is paid for in *height*, so a 16:9 thumbnail
    that looks generous at 1920px yields 864 usable pixels and gets upscaled.

    Returns the hard requirement, without OVERSAMPLE — callers add that margin
    when they fetch, so that wanting a nicer downscale never on its own costs a
    round-trip for a source that was already big enough.
    """
    tw, th = target
    if width <= 0 or height <= 0:
        return tw
    return int(th * (width / height)) if width * th > height * tw else tw


def _thumb_url(title: str, width: int) -> Optional[str]:
    """Re-ask the API for one file at a specific rendition width."""
    params = {
        "action": "query", "format": "json", "titles": title,
        "prop": "imageinfo", "iiprop": "url|size", "iiurlwidth": width,
    }
    try:
        r = requests.get(API, params=params, headers={"User-Agent": UA}, timeout=30)
        r.raise_for_status()
        pages = r.json().get("query", {}).get("pages", {})
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            if info.get("thumburl"):
                return info["thumburl"]
    except (requests.exceptions.RequestException, ValueError, KeyError) as exc:
        logger.warning(f"could not size up {title}: {exc}")
    return None


def _extract_location(title: str, object_name: str = "") -> Optional[str]:
    """Parse a place out of the file title, or return None.

    Grounded by construction: everything returned is a substring of the title.
    Nothing is inferred, so a slide can be unlabelled but never mislabelled.
    """
    # Commons' ObjectName is often a cleaner caption than the filename, so try
    # it first; both are the file's own metadata, so neither can invent a place.
    for source in (object_name, title):
        if not source:
            continue
        found = _parse_place(source)
        if found:
            return found
    return None


def _parse_place(title: str) -> Optional[str]:
    name = re.sub(r"\.(jpe?g|png|tiff?)$", "", title, flags=re.I)
    name = re.sub(r"^File:", "", name)
    # Commons titles often carry an ID prefix and a trailing credit.
    name = re.sub(r"^\d+[\s\-_]+", "", name)
    name = re.split(r"\bphoto(graph)? by\b", name, flags=re.I)[0]
    name = re.sub(r"[_]+", " ", name).strip(" -–,")

    # "... (Iceland)" / "..., New Zealand" / "... over Mývatn"
    patterns = [
        r"\(([^)]{3,40})\)\s*$",
        r",\s*([A-ZÀ-Þ][\w'’.\-]*(?:\s+[A-ZÀ-Þ][\w'’.\-]*){0,3})\s*$",
        r"\b(?:over|above|in|at|near|from)\s+([A-ZÀ-Þ][\w'’.\-]*(?:\s+[A-ZÀ-Þ][\w'’.\-]*){0,3})",
    ]
    for pat in patterns:
        m = re.search(pat, name)
        if not m:
            continue
        loc = m.group(1).strip(" .,-")
        # Reject junk: measurements, pure numbers, over-long fragments.
        if len(loc) < 3 or len(loc) > 42:
            continue
        if re.search(r"\d{3,}|\bmm\b|\bf/\d|\bISO\b", loc, re.I):
            continue
        low = loc.lower()
        if low in NON_PLACES or any(w in low.split() for w in NON_PLACES):
            continue
        # "by Pudelek" is a photographer credit, not a place.
        if re.match(r"^(by|von|par|de|copyright|\(c\))\b", low):
            continue
        return low
    return None


def search(subject: str, limit: int = 24, extra_pool: Optional[str] = None) -> list[Photo]:
    """Find high-quality, correctly-licensed photos for a subject."""
    out: list[Photo] = []
    seen: set[str] = set()
    term_re = _term_pattern(subject)
    pools = ([extra_pool] if extra_pool else []) + QUALITY_POOLS
    for pool in pools:
        if len(out) >= limit:
            break
        params = {
            "action": "query", "format": "json",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {pool} {subject}",
            "gsrlimit": limit, "gsrnamespace": 6,
            # Categories ride along in the same request; they are what makes the
            # relevance check work on files titled with a Latin binomial.
            "prop": "imageinfo|categories",
            "iiprop": "url|size|extmetadata",
            "iiurlwidth": SEARCH_THUMB_WIDTH,
            "cllimit": "max",
        }
        try:
            r = requests.get(API, params=params, headers={"User-Agent": UA}, timeout=30)
            r.raise_for_status()
            pages = r.json().get("query", {}).get("pages", {})
        except (requests.exceptions.RequestException, ValueError) as exc:
            logger.warning(f"commons search failed for {subject!r}: {exc}")
            continue

        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            meta = info.get("extmetadata") or {}

            def field(key: str) -> str:
                return _plain((meta.get(key) or {}).get("value", ""))

            licence = field("LicenseShortName")
            if not licence or not ALLOWED_LICENCES.match(licence.strip()):
                continue

            # Everything the file says about itself. This must span the same
            # text Commons searched — title AND description — or the guard drops
            # results the search legitimately matched on prose.
            #
            # Categories are included when present but are never depended on:
            # the API truncates them across a generator batch (it returns a
            # clcontinue token), so a category-only rule would drop files purely
            # for appearing late in the response. That failure is invisible.
            title = page.get("title", "")
            cats = " ".join(c.get("title", "") for c in (page.get("categories") or []))
            about = f"{title} {cats} {field('ImageDescription')} {field('ObjectName')}"
            if term_re and not term_re.search(about):
                continue
            # Only title and categories for this one: a description is free prose
            # and mentioning "a map of the area" must not disqualify a photograph.
            if NOT_PHOTOGRAPHS.search(f"{title} {cats}"):
                continue
            # Judge the original by what survives the crop, not by its raw size.
            src_w, src_h = info.get("width", 0), info.get("height", 0)
            crop_w, crop_h = crop_dimensions(src_w, src_h)
            if crop_w < TARGET_W or crop_h < TARGET_H:
                continue
            aspect = src_w / src_h if src_h else 0
            if not MIN_ASPECT <= aspect <= MAX_ASPECT:
                continue
            url = info.get("thumburl") or info.get("url")
            if not url or url in seen:
                continue
            seen.add(url)

            out.append(Photo(
                title=title,
                url=url,
                width=info.get("width", 0),
                height=info.get("height", 0),
                author=field("Artist") or "Unknown",
                licence=licence.strip(),
                location=_extract_location(title, field("ObjectName")),
                descriptionurl=info.get("descriptionurl", ""),
            ))
            if len(out) >= limit:
                break
    logger.info(f"commons: {len(out)} usable photos for {subject!r}")
    return out


def download(photo: Photo,
             target: tuple[int, int] = (TARGET_W, TARGET_H)) -> Optional[Image.Image]:
    """Fetch a photo at a rendition big enough to survive the crop.

    `photo.url` is the search thumbnail, which is sized for browsing, not for a
    portrait crop. Only go back to the API when the crop actually needs more —
    most portrait sources are already fine at the search width.
    """
    url = photo.url
    if fetch_width_for(photo.width, photo.height, target) > SEARCH_THUMB_WIDTH:
        want = int(fetch_width_for(photo.width, photo.height, target) * OVERSAMPLE)
        want = min(want, photo.width or COMMONS_MAX_THUMB, COMMONS_MAX_THUMB)
        bigger = _thumb_url(photo.title, want)
        if bigger:
            url = bigger
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=90)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except (requests.exceptions.RequestException, OSError) as exc:
        logger.warning(f"could not download {photo.title}: {exc}")
        return None


def credit_line(photo: Photo) -> str:
    """Short attribution, as CC BY requires."""
    author = photo.author if photo.author and photo.author != "Unknown" else "unknown"
    return f"{author} / {photo.licence}"
