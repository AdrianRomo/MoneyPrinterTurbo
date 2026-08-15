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

MIN_WIDTH = 1800
MIN_HEIGHT = 1200

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


def _extract_location(title: str) -> Optional[str]:
    """Parse a place out of the file title, or return None.

    Grounded by construction: everything returned is a substring of the title.
    Nothing is inferred, so a slide can be unlabelled but never mislabelled.
    """
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
        return loc.lower()
    return None


def search(subject: str, limit: int = 24) -> list[Photo]:
    """Find high-quality, correctly-licensed photos for a subject."""
    out: list[Photo] = []
    seen: set[str] = set()
    for pool in QUALITY_POOLS:
        if len(out) >= limit:
            break
        params = {
            "action": "query", "format": "json",
            "generator": "search",
            "gsrsearch": f"filetype:bitmap {pool} {subject}",
            "gsrlimit": limit, "gsrnamespace": 6,
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata",
            "iiurlwidth": 1600,
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
            if info.get("width", 0) < MIN_WIDTH or info.get("height", 0) < MIN_HEIGHT:
                continue
            url = info.get("thumburl") or info.get("url")
            if not url or url in seen:
                continue
            seen.add(url)

            title = page.get("title", "")
            out.append(Photo(
                title=title,
                url=url,
                width=info.get("width", 0),
                height=info.get("height", 0),
                author=field("Artist") or "Unknown",
                licence=licence.strip(),
                location=_extract_location(title),
                descriptionurl=info.get("descriptionurl", ""),
            ))
            if len(out) >= limit:
                break
    logger.info(f"commons: {len(out)} usable photos for {subject!r}")
    return out


def download(photo: Photo) -> Optional[Image.Image]:
    try:
        r = requests.get(photo.url, headers={"User-Agent": UA}, timeout=60)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except (requests.exceptions.RequestException, OSError) as exc:
        logger.warning(f"could not download {photo.title}: {exc}")
        return None


def credit_line(photo: Photo) -> str:
    """Short attribution, as CC BY requires."""
    author = photo.author if photo.author and photo.author != "Unknown" else "unknown"
    return f"{author} / {photo.licence}"
