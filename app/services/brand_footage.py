"""Reel footage generated locally, instead of searched from stock.

Stock search cannot be made safe for this account, and the failure is not a
tuning problem. On 2026-08-15 the script for an article about Revelation
produced the search term "Revelation dramatic angelic figure"; Pexels returned a
woman in a gold mesh dress wearing skull face paint and costume wings. An
earlier failure of the same kind put a woman in activewear on a bed into a
devotional Reel. Both were caught only by a human-review gate.

The root cause is that theological language has no honest stock photography.
"Locust swarm", "angelic figure", "the end times" are not things a stock library
contains, so the search falls through to whatever is superficially similar —
which, for anything involving a figure, means costume and glamour shoots. A term
blocklist cannot fix this: it filters the words we send, not the pictures that
come back, and the offending images matched terms that were themselves
perfectly reasonable.

So we do not search. Every frame is generated locally with the SAME ComfyUI
pipeline, subject vocabulary, style and negative prompt the verse cards use — a
prompt that explicitly excludes people, faces, figures and religious iconography.
That makes an off-brand frame structurally impossible rather than unlikely, and
it makes Reels look like the rest of the account instead of like stock.

The script still chooses the *mood*; it can never introduce a subject. Terms are
matched against a fixed vocabulary, so the worst case is a serene landscape that
is slightly less apt — never a costume shoot.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

from loguru import logger

from app.models.schema import MaterialInfo, VideoAspect  # noqa: F401  (typing parity)
from app.services import verse_card
from app.utils import utils

# Mood -> the subjects allowed to express it. Every entry is drawn from
# verse_card.BACKGROUND_SUBJECTS, so the imagery is identical in character to
# the cards and carousels. Keywords are matched against the script's own search
# term, which is how the narration still steers the look.
MOODS: dict[str, dict] = {
    "hope": {
        "keywords": ("hope", "promise", "dawn", "new", "begin", "future", "light",
                     "joy", "resurrect", "morning"),
        "subjects": ["soft morning mist over a quiet meadow at sunrise",
                     "sunlight breaking through low clouds over open fields",
                     "calm lake at dawn with gentle reflections",
                     "gentle ocean waves on an empty shore at first light"],
    },
    "struggle": {
        "keywords": ("fear", "doubt", "storm", "trial", "suffer", "grief", "hard",
                     "dark", "lost", "wilderness", "judgment", "end", "wrath"),
        "subjects": ["desert dunes under a soft dawn sky",
                     "quiet mountain range under soft blue haze",
                     "rain on a window with soft daylight behind",
                     "snowfall over silent evergreen trees"],
    },
    "rest": {
        "keywords": ("rest", "peace", "still", "quiet", "calm", "sleep", "night",
                     "comfort", "abide", "wait"),
        "subjects": ["calm lake at dawn with gentle reflections",
                     "still river winding through autumn woodland",
                     "dew on wild grass in early morning light",
                     "a narrow country path between hedgerows at dusk"],
    },
    "gratitude": {
        "keywords": ("thank", "grateful", "gratitude", "praise", "bless", "harvest",
                     "abundance", "provide", "ordinary", "daily"),
        "subjects": ["wheat field swaying in warm evening light",
                     "golden hour light through tall forest pines",
                     "rolling hills under a wide pastel sky",
                     "dew on wild grass in early morning light"],
    },
}

_FALLBACK = "rolling hills under a wide pastel sky"


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", (text or "").lower()))


def subject_for(term: str, index: int = 0) -> str:
    """Pick an on-brand subject for a scene.

    Deterministic in (term, index) so a retried render reuses the same look, and
    `index` walks the matched mood's list so one Reel does not repeat a frame.
    """
    words = _tokens(term)
    best, best_score = None, 0
    for name, mood in MOODS.items():
        score = sum(1 for kw in mood["keywords"] if any(w.startswith(kw) for w in words))
        if score > best_score:
            best, best_score = name, score
    subjects = MOODS[best]["subjects"] if best else verse_card.BACKGROUND_SUBJECTS
    if not subjects:
        return _FALLBACK
    # Offset by the term's own hash so different scenes in one mood diverge.
    offset = sum(ord(c) for c in (term or "")) + index
    return subjects[offset % len(subjects)]


def generate_frame(term: str, index: int = 0, save_dir: str = "") -> Optional[str]:
    """Generate one on-brand still and return its local path (or None).

    Cached by subject: a script expands each concept into several near-identical
    terms ("selfie capture", "Timothy selfie capture", "God selfie capture"),
    and those collapse onto the same allowlisted subject. Regenerating per term
    would cost a diffusion pass each for the same picture.
    """
    subject = subject_for(term, index)
    save_dir = save_dir or utils.storage_dir("cache_images", create=True)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"brand-{utils.md5(f'{subject}|{index}')}.jpg")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        logger.info(f"reusing brand footage for {term!r}: {subject}")
        return path
    # 9:16 bucket — the same native SDXL size the story cards use, so the
    # composition holds instead of duplicating subjects at an off-bucket size.
    img = verse_card.generate_background(kind="story", subject=subject)
    if img is None:
        return None
    try:
        img.convert("RGB").save(path, "JPEG", quality=92, optimize=True)
    except OSError as exc:
        logger.error(f"could not write generated frame: {exc}")
        return None
    logger.info(f"generated brand footage for {term!r}: {subject}")
    return path


def asset_for_scene(term: str, beat_index: int):
    """One generated frame for one scene, guaranteed distinct per scene.

    `beat_index` walks the mood's subject list and is part of the cache key, so
    two scenes in the same mood get different pictures. This matters because the
    generic selection path dedups by asset key: subject-cached frames shared
    between scenes would be discarded as duplicates and the scene would fall
    back to a blank background.
    """
    from app.models.article import MediaAsset

    path = generate_frame(term, index=beat_index)
    if not path:
        return None
    return MediaAsset(
        media_type="image",
        provider="comfyui",
        url=path,
        width=768,
        height=1344,
        asset_id=os.path.basename(path),
        creator="",
        metadata_text=subject_for(term, beat_index),
        license_name="Locally generated (SDXL)",
        license_url="",
        attribution_text="",
        source_page_url="",
        search_query=term,
        relevance_score=1.0,
    )


def search_images_comfyui(search_term: str, video_aspect=None,
                          per_page: int = 1) -> List["MediaAsset"]:
    """Provider-shaped entry point: generate rather than search.

    Returns ONE frame per term. A stock API is asked for 20 candidates because
    they are free to list and most get discarded; here each candidate is a
    diffusion pass, and the script already asks for many terms — honouring
    per_page=20 would mean minutes of GPU per Reel to throw nearly all of it
    away.
    """
    from app.models.article import MediaAsset

    assets: List[MediaAsset] = []
    for i in range(1):
        path = generate_frame(search_term, index=i)
        if not path:
            continue
        assets.append(
            MediaAsset(
                media_type="image",
                provider="comfyui",
                url=path,          # local path; save_image passes it through
                width=768,
                height=1344,
                asset_id=os.path.basename(path),
                creator="",
                metadata_text=subject_for(search_term, i),
                # Locally generated from a self-hosted SDXL checkpoint: no stock
                # licence applies and there is no photographer to credit.
                license_name="Locally generated (SDXL)",
                license_url="",
                attribution_text="",
                source_page_url="",
                search_query=search_term,
            )
        )
    return assets
