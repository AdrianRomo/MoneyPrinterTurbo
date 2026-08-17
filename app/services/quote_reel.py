"""Quiet quote Reel renderer.

This mode is intentionally separate from the legacy topic pipeline: no spoken
narration, no all-caps opening hook, and no word-by-word subtitles. It is meant
for curated cinematic footage with one centered, saveable quote.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from loguru import logger
from moviepy import AudioFileClip, CompositeVideoClip, ImageClip, VideoFileClip, afx
from PIL import Image, ImageDraw, ImageFilter, ImageStat

from app.config import config
from app.models import const
from app.models.schema import VideoAspect, VideoParams
from app.services import llm, material, state as sm, typography, video
from app.utils import utils

CONTENT_MODE = "quiet_quote_reel"
MIN_SECONDS = 10.0
MAX_SECONDS = 24.0
DEFAULT_SECONDS = 15.0
MAX_QUOTE_CHARS = 180
MAX_QUOTE_LINES = 3

# The reference Reels hold 4-8 words a line. Wrapping purely on pixel width
# produced three near-full-width lines that read like a paragraph, not a quote.
MAX_WORDS_PER_LINE = 7

_SUPPORTED_VIDEO = {f".{ext}" for ext in const.FILE_TYPE_VIDEOS}
_SUPPORTED_IMAGE = {f".{ext}" for ext in const.FILE_TYPE_IMAGES}

DEFAULT_LANGUAGE = "English"

# One key drives the whole lane. Setting quote_reel_default_language to Spanish
# has to move the caption call-to-action and the fallbacks with it, or you get
# what shipped on 2026-08-16: a Spanish quote under English hashtags.
_LANGUAGE_PACKS = {
    "english": {
        "hashtags": ["#faith", "#stillness", "#christianfaith", "#God"],
        "cta": "Save this one to come back to slowly.",
        "fallbacks": [
            "God also speaks in whatever makes you stop.",
            "Beauty opens a window onto the eternal.",
            "Sometimes grace arrives quietly.",
        ],
    },
    "spanish": {
        "hashtags": ["#fe", "#belleza", "#vidacristiana", "#Dios"],
        "cta": "Guardalo para volver a mirar con calma.",
        "fallbacks": [
            "Dios tambien habla en lo que te hace detenerte.",
            "La belleza abre una ventana hacia lo eterno.",
            "A veces la gracia llega en silencio.",
        ],
    },
}
_BAD_PHRASES = (
    "faucet sings",
    "sink speaks",
    "grifo canta",
    "llave canta",
    "fregadero habla",
    "lavabo habla",
)
_REFERENCE_NAME_PATTERNS = ("reference", "example", "postiz", "final-output")
_PEOPLE_TERMS = (
    "person",
    "people",
    "woman",
    "man",
    "child",
    "face",
    "portrait",
    "hands",
)
_PROPERTY_TERMS = (
    "building",
    "church",
    "cathedral",
    "home",
    "house",
    "interior",
    "logo",
    "museum",
    "store",
)


@dataclass(frozen=True)
class QuoteReelAsset:
    path: str
    kind: str
    provider: str
    label: str
    source_info: dict


def _cfg_bool(key: str, default: bool = False) -> bool:
    value = config.app.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _cfg_float(key: str, default: float) -> float:
    try:
        value = float(config.app.get(key, default))
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    return value


def _cfg_int(key: str, default: int, *, minimum: int = 0, maximum: int | None = None) -> int:
    try:
        value = int(config.app.get(key, default))
    except (TypeError, ValueError):
        value = default
    value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def target_seconds() -> float:
    configured = _cfg_float("quote_reel_target_seconds", DEFAULT_SECONDS)
    return max(MIN_SECONDS, min(MAX_SECONDS, configured))


def language_name() -> str:
    return str(
        config.app.get("quote_reel_default_language", DEFAULT_LANGUAGE)
        or DEFAULT_LANGUAGE
    ).strip()


def _language_pack() -> dict:
    return _LANGUAGE_PACKS.get(language_name().lower(), _LANGUAGE_PACKS["english"])


def clean_quote(text: str) -> str:
    value = (text or "").strip()
    value = re.sub(r"^```(?:\w+)?|```$", "", value, flags=re.MULTILINE).strip()
    value = re.sub(r"(?i)^(quote|frase|caption|texto)\s*:\s*", "", value).strip()
    value = value.replace(" / ", "\n")
    lines = [line.strip().strip('"').strip("'") for line in value.splitlines()]
    lines = [line for line in lines if line and not line.lstrip().startswith("#")]
    if not lines:
        return ""

    # Keep the first quote-like paragraph, but allow intentional line breaks.
    quote = " ".join(lines[:MAX_QUOTE_LINES])
    quote = re.sub(r"\s+", " ", quote).strip()
    quote = quote.strip(" -")
    if len(quote) > MAX_QUOTE_CHARS:
        quote = quote[: MAX_QUOTE_CHARS - 1].rsplit(" ", 1)[0].rstrip(".,;:") + "."
    return _open_with_a_capital(quote)


def _open_with_a_capital(quote: str) -> str:
    """Capitalise the opening letter, skipping any accent marker.

    Models return the quote lowercased often enough that it shipped that way —
    "the breath that settles after sunrise" reads as a typo on a finished Reel,
    and the accent marker means the first character is not always a letter.
    """
    for index, char in enumerate(quote):
        if char.isalpha():
            return quote[:index] + char.upper() + quote[index + 1:]
        if char not in "*\"'“‘(":
            break
    return quote


def _is_shouty(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    if len(letters) < 12:
        return False
    uppercase = sum(1 for char in letters if char.isupper())
    return uppercase / len(letters) > 0.72


def _is_bad_quote(text: str) -> bool:
    lowered = text.lower()
    return (
        not text
        or _is_shouty(text)
        or any(phrase in lowered for phrase in _BAD_PHRASES)
    )


def _fallback_quote(subject: str = "") -> str:
    subject = clean_quote(subject)
    if subject and not _is_bad_quote(subject) and len(subject) <= 120:
        return subject.rstrip(".") + "."
    return _language_pack()["fallbacks"][0]


def resolve_quote(params: VideoParams) -> str:
    provided = clean_quote(getattr(params, "video_script", "") or "")
    if provided and not _is_bad_quote(provided):
        return provided

    language = language_name()
    subject = (getattr(params, "video_subject", "") or "faith in ordinary days").strip()
    # The word budget is deliberately tighter than the old 12-26: at 26 words the
    # overlay wrapped to three full-width lines and stopped reading as a quote.
    prompt = f"""Write one short poetic faith quote for an Instagram Reel.

Language: {language}.
Topic: {subject}

Style rules:
- Quiet, contemplative, and human.
- One sentence or two short lines, 10-18 words total.
- Wrap the two to four words that carry the meaning in *single asterisks*.
  Exactly one such phrase, and never the whole sentence.
- No all-caps hook, no hashtags, no CTA, no sermon.
- Do not personify random household objects.
- Return only the quote text.
"""
    try:
        generated = clean_quote(llm._generate_response(prompt))
    except Exception as exc:
        logger.warning(f"quote reel LLM quote generation failed: {exc}")
        generated = ""
    if generated and not _is_bad_quote(generated):
        return generated
    return _fallback_quote(subject)


def _configured_hashtags() -> list[str]:
    raw = config.app.get("quote_reel_caption_hashtags", _language_pack()["hashtags"])
    if isinstance(raw, str):
        tags = raw.split()
    elif isinstance(raw, Iterable):
        tags = [str(item) for item in raw]
    else:
        tags = []
    cleaned = []
    for tag in tags:
        tag = tag.strip()
        if not tag:
            continue
        if not tag.startswith("#"):
            tag = f"#{tag}"
        cleaned.append(tag)
    return cleaned[:6] or list(_language_pack()["hashtags"])


def build_caption_variant(quote: str, subject: str = "") -> dict:
    from app.services import hashtags

    quote = typography.strip_accent(clean_quote(quote)) or _fallback_quote(subject)
    first_line = quote.rstrip(".")
    if len(first_line) > 86:
        first_line = first_line[:83].rsplit(" ", 1)[0].rstrip(".,;:") + "..."
    explicit_set = str(config.app.get("quote_reel_hashtag_set", "") or "").strip()
    try:
        set_id = hashtags.choose_set(explicit_set or None)
        tags = hashtags.tags_for(set_id)
    except Exception as exc:
        logger.warning(f"quote reel hashtag selection failed: {exc}")
        set_id = ""
        tags = []
    tags = tags or _configured_hashtags()
    caption_style = "saveable_contemplative"
    caption = (
        f"{first_line}.\n\n"
        f"{_language_pack()['cta']}\n\n"
        f"{' '.join(tags)}"
    )
    return {
        "caption": caption,
        "caption_style": caption_style,
        "hashtag_set": set_id,
        "hashtags": tags,
    }


def build_caption(quote: str, subject: str = "") -> str:
    return build_caption_variant(quote, subject)["caption"]


def _asset_kind(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in _SUPPORTED_IMAGE:
        return "image"
    return "video"


def _material_assets(params: VideoParams) -> list[QuoteReelAsset]:
    materials = list(getattr(params, "video_materials", None) or [])
    if not materials:
        return []
    valid = video.preprocess_video(materials, clip_duration=int(target_seconds()))
    assets = []
    for source_material in valid:
        path = str(getattr(source_material, "url", "") or "").strip()
        if not path:
            continue
        assets.append(
            QuoteReelAsset(
                path=path,
                kind=_asset_kind(path),
                provider=str(getattr(source_material, "provider", "") or "local"),
                label=os.path.basename(path),
                source_info={
                    "provider": str(
                        getattr(source_material, "provider", "") or "local"
                    ),
                    "label": os.path.basename(path),
                    "kind": _asset_kind(path),
                    "raw_text_free": True,
                },
            )
        )
    return assets


def _configured_media_dirs() -> list[str]:
    configured = str(config.app.get("quote_reel_media_dir", "") or "").strip()
    if not configured:
        return [utils.storage_dir("quote_reel_media", create=True)]
    dirs = []
    for item in re.split(r"[;:]", configured):
        item = item.strip()
        if not item:
            continue
        dirs.append(item if os.path.isabs(item) else utils.storage_dir(item, create=True))
    return dirs or [utils.storage_dir("quote_reel_media", create=True)]


def _read_media_sidecar(path: Path) -> dict:
    candidates = [
        path.with_suffix(".json"),
        path.with_name(f"{path.name}.json"),
        path.with_name(f"{path.stem}.quote.json"),
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            with open(candidate, encoding="utf-8") as fp:
                data = json.load(fp)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError) as exc:
            logger.warning(f"could not read quote reel media sidecar: {candidate}: {exc}")
    return {}


def _sidecar_bool(metadata: dict, *keys: str, default: bool | None = None) -> bool | None:
    for key in keys:
        if key not in metadata:
            continue
        value = metadata.get(key)
        if isinstance(value, bool):
            return value
        if value in (None, ""):
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    return default


def _is_reference_asset(path: Path, metadata: dict) -> bool:
    if _sidecar_bool(metadata, "reference_only", "is_reference", default=False):
        return True
    lowered = path.name.lower()
    return any(pattern in lowered for pattern in _REFERENCE_NAME_PATTERNS)


def _library_assets() -> list[QuoteReelAsset]:
    assets = []
    for media_dir in _configured_media_dirs():
        if not os.path.isdir(media_dir):
            continue
        for path in sorted(Path(media_dir).rglob("*"), key=lambda item: str(item).lower()):
            if not path.is_file() or path.name.startswith("."):
                continue
            suffix = path.suffix.lower()
            if suffix not in _SUPPORTED_VIDEO and suffix not in _SUPPORTED_IMAGE:
                continue
            kind = "image" if suffix in _SUPPORTED_IMAGE else "video"
            metadata = _read_media_sidecar(path)
            reference_only = _is_reference_asset(path, metadata)
            raw_text_free = _sidecar_bool(
                metadata,
                "raw_text_free",
                "text_free",
                default=(
                    not reference_only
                    and _cfg_bool("quote_reel_assume_curated_text_free", True)
                ),
            )
            assets.append(
                QuoteReelAsset(
                    path=str(path),
                    kind=kind,
                    provider="curated_library",
                    label=path.name,
                    source_info={
                        "provider": "curated_library",
                        "label": path.name,
                        "kind": kind,
                        "license": str(metadata.get("license") or "").strip(),
                        "source_page": str(metadata.get("source_page") or "").strip(),
                        "reference_only": reference_only,
                        "raw_text_free": raw_text_free,
                        "contains_people": _sidecar_bool(
                            metadata, "contains_people", "has_people", default=False
                        ),
                        "contains_property": _sidecar_bool(
                            metadata, "contains_property", "has_property", default=False
                        ),
                        "has_talent_released": _sidecar_bool(
                            metadata, "has_talent_released", "talent_released"
                        ),
                        "has_property_released": _sidecar_bool(
                            metadata, "has_property_released", "property_released"
                        ),
                    },
                )
            )
    return assets


def _quote_search_terms(params: VideoParams) -> list[str]:
    raw = config.app.get("quote_reel_search_terms", [])
    if isinstance(raw, str):
        terms = [term.strip() for term in re.split(r"[,;\n]", raw) if term.strip()]
    elif isinstance(raw, Iterable):
        terms = [str(term).strip() for term in raw if str(term).strip()]
    else:
        terms = []
    subject = (getattr(params, "video_subject", "") or "").strip()
    if subject:
        terms.insert(0, subject)
    terms.extend(["soft light", "quiet nature", "church window", "clouds"])
    deduped = []
    for term in terms:
        if term.lower() not in {item.lower() for item in deduped}:
            deduped.append(term)
    return deduped[:6]


def _storyblocks_assets(
    task_id: str,
    params: VideoParams,
    *,
    duration: float,
) -> list[QuoteReelAsset]:
    if not material.storyblocks_is_configured():
        return []
    aspect = VideoAspect(getattr(params, "video_aspect", None) or VideoAspect.portrait)
    save_dir = os.path.join(utils.task_dir(task_id), "storyblocks")
    os.makedirs(save_dir, exist_ok=True)
    assets: list[QuoteReelAsset] = []
    covered = 0.0
    max_clip = _cfg_int("quote_reel_storyblocks_clip_seconds", 6, minimum=2, maximum=10)
    seen_urls: set[str] = set()
    for term in _quote_search_terms(params):
        items = material.search_videos_storyblocks(
            term,
            minimum_duration=max_clip,
            video_aspect=aspect,
        )
        for item in items:
            if item.url in seen_urls:
                continue
            seen_urls.add(item.url)
            try:
                local_path = material.save_video(item.url, save_dir=save_dir)
            except Exception as exc:
                logger.warning(f"storyblocks quote reel download failed: {exc}")
                continue
            if not local_path:
                continue
            source_info = dict(item.source_info or {})
            source_info["provider"] = "storyblocks"
            source_info["kind"] = "video"
            source_info["raw_text_free"] = True
            source_info["label"] = os.path.basename(local_path)
            assets.append(
                QuoteReelAsset(
                    path=local_path,
                    kind="video",
                    provider="storyblocks",
                    label=os.path.basename(local_path),
                    source_info=source_info,
                )
            )
            covered += min(max_clip, float(item.duration or max_clip))
            if covered >= duration:
                return assets
    return assets


def _comfyui_assets(params: VideoParams, *, count: int = 3) -> list[QuoteReelAsset]:
    """Generate the backdrop locally instead of searching stock.

    Article Mode already moved to this (see the commit that added
    brand_footage.search_images_comfyui): theological search terms have no honest
    stock photography, and "angelic figure" reliably returns costume shoots. The
    quote lane wants the same thing for a second reason — a generated still under
    a slow push is closer to the reference Reels than any stock clip, and no other
    account is running the same footage.
    """
    from app.services import brand_footage

    assets: list[QuoteReelAsset] = []
    # generate_frame caches by subject, so two different search terms can resolve
    # to the same subject and hand back the identical file — which cuts from a
    # shot straight back to that same shot mid-reel.
    seen: set[str] = set()
    for term in _quote_search_terms(params):
        if len(assets) >= count:
            break
        try:
            generated = brand_footage.search_images_comfyui(term, per_page=1)
        except Exception as exc:
            logger.warning(f"comfyui quote reel generation failed for {term!r}: {exc}")
            continue
        for item in generated:
            path = str(getattr(item, "url", "") or "").strip()
            if not path or not os.path.exists(path) or path in seen:
                continue
            seen.add(path)
            assets.append(
                QuoteReelAsset(
                    path=path,
                    kind="image",
                    provider="comfyui",
                    label=os.path.basename(path),
                    source_info={
                        "provider": "comfyui",
                        "kind": "image",
                        "label": os.path.basename(path),
                        "search_term": term,
                        "metadata_text": getattr(item, "metadata_text", "") or "",
                        "license": getattr(item, "license_name", "") or "",
                        # Locally generated: no stock licence, nobody to credit,
                        # and no release question to answer.
                        "raw_text_free": True,
                        "contains_people": False,
                        "contains_property": False,
                    },
                )
            )
            break
    return assets


def _comfyui_motion_assets(params: VideoParams, *, count: int = 3) -> list[QuoteReelAsset]:
    """Draw real motion clips from the pre-generated pool.

    The still path above puts a slow push on a photograph, which reads as a
    slideshow next to the reference Reels. These are ~5s clips generated ahead
    of time by brand_motion's overnight timer, from the same allowlisted
    subjects and the same SDXL seed frames — so the imagery is identical in
    character to the stills, it simply moves.

    Reads only. A clip costs minutes of GPU, so a miss here must never turn into
    an inline generation that blocks the publish path; the caller falls back to
    stills instead, which is a slightly worse Reel rather than a late one.

    Dedupes on BOTH axes, because either alone leaves a visible repeat:

    - by subject, because a Reel's search terms are near-synonyms that collapse
      onto one subject ("quiet morning kitchen" and "hope at first light" both
      resolve to the misty meadow), and
    - by path, because two different subjects can still be served the same file
      when the pool is only partly filled.

    Without both, the Reel cuts from a shot straight back to that same shot.
    """
    # Imported here, as _comfyui_assets does: these pull in the diffusion
    # stack, and the stock paths must not pay for it.
    from app.services import brand_footage, brand_motion

    assets: list[QuoteReelAsset] = []
    seen_paths: set[str] = set()
    seen_subjects: set[str] = set()

    for index, term in enumerate(_quote_search_terms(params)):
        if len(assets) >= count:
            break
        # `avoid` walks on to the next free subject in the mood rather than
        # handing back one already used in this Reel.
        subject = brand_footage.subject_for(term, index, avoid=seen_subjects)
        if subject in seen_subjects:
            continue
        path = brand_motion.clip_for_subject(subject, avoid=seen_paths)
        if not path:
            # The mapped subject is not pooled yet — normal while the pool is
            # filling. Take any other pooled clip rather than abandoning motion
            # for the whole Reel; they all come from the same allowlist.
            path = brand_motion.substitute_clip(avoid=seen_paths)
        if not path or path in seen_paths:
            continue
        seen_paths.add(path)
        seen_subjects.add(subject)
        assets.append(
            QuoteReelAsset(
                path=path,
                kind="video",
                provider="comfyui_motion",
                label=os.path.basename(path),
                source_info={
                    "provider": "comfyui_motion",
                    "kind": "video",
                    "label": os.path.basename(path),
                    "search_term": term,
                    "metadata_text": subject,
                    "license": "Locally generated (SDXL still animated locally)",
                    # Locally generated from a vetted seed frame: no stock
                    # licence, nobody to credit, no release question to answer.
                    "raw_text_free": True,
                    "contains_people": False,
                    "contains_property": False,
                },
            )
        )

    if len(assets) < count:
        logger.info(
            f"motion pool served {len(assets)}/{count} clips; "
            f"{brand_motion.pool_status()} — falling back for the remainder"
        )
    return assets


def _stock_provider() -> str:
    provider = str(
        config.app.get("quote_reel_stock_provider")
        or config.app.get("video_source")
        or "pexels"
    ).strip().lower()
    if provider not in {"pexels", "pixabay", "coverr", "storyblocks"}:
        logger.warning(f"unsupported quote reel stock provider {provider!r}; using pexels")
        return "pexels"
    return provider


def _stock_searcher(provider: str):
    return {
        "pexels": material.search_videos_pexels,
        "pixabay": material.search_videos_pixabay,
        "coverr": material.search_videos_coverr,
        "storyblocks": material.search_videos_storyblocks,
    }.get(provider)


def _stock_assets(
    task_id: str,
    params: VideoParams,
    *,
    duration: float,
    provider: str | None = None,
) -> list[QuoteReelAsset]:
    provider = (provider or _stock_provider()).strip().lower()
    if provider == "storyblocks":
        return _storyblocks_assets(task_id, params, duration=duration)
    searcher = _stock_searcher(provider)
    if searcher is None:
        return []

    aspect = VideoAspect(getattr(params, "video_aspect", None) or VideoAspect.portrait)
    save_dir = os.path.join(utils.task_dir(task_id), provider)
    os.makedirs(save_dir, exist_ok=True)
    assets: list[QuoteReelAsset] = []
    covered = 0.0
    max_clip = _cfg_int("quote_reel_stock_clip_seconds", 6, minimum=2, maximum=10)
    assume_text_free = _cfg_bool("quote_reel_assume_stock_text_free", True)
    seen_urls: set[str] = set()

    for term in _quote_search_terms(params):
        items = searcher(term, minimum_duration=max_clip, video_aspect=aspect)
        for item in items:
            if item.url in seen_urls:
                continue
            seen_urls.add(item.url)
            try:
                local_path = material.save_video(item.url, save_dir=save_dir)
            except Exception as exc:
                logger.warning(f"{provider} quote reel download failed: {exc}")
                continue
            if not local_path:
                continue
            item_provider = str(item.provider or provider).strip().lower() or provider
            source_info = dict(item.source_info or {})
            source_info["provider"] = item_provider
            source_info["kind"] = "video"
            source_info["raw_text_free"] = assume_text_free
            source_info["label"] = os.path.basename(local_path)
            candidate = QuoteReelAsset(
                path=local_path,
                kind="video",
                provider=item_provider,
                label=os.path.basename(local_path),
                source_info=source_info,
            )
            if _cfg_bool("quote_reel_skip_stock_review_risk", True):
                reasons = _asset_review_reasons(candidate)
                if reasons:
                    logger.info(
                        f"skipping {provider} quote reel asset "
                        f"{source_info.get('asset_id') or candidate.label}: {reasons[0]}"
                    )
                    continue
            assets.append(candidate)
            covered += min(max_clip, float(item.duration or max_clip))
            if covered >= duration:
                return assets
    return assets


def select_media_assets(
    params: VideoParams,
    *,
    task_id: str = "",
    duration: float | None = None,
) -> list[QuoteReelAsset]:
    uploaded = _material_assets(params)
    if uploaded:
        return uploaded
    media_source = str(config.app.get("quote_reel_media_source", "curated") or "curated").strip().lower()
    duration = duration or target_seconds()
    if media_source == "comfyui_motion":
        # Motion first, then the still generator, then the curated library. The
        # pool is filled overnight and may legitimately be empty or partial —
        # on a fresh install, or after a night the GPU was busy — so this
        # degrades to the previous behaviour rather than failing the render.
        motion = _comfyui_motion_assets(params)
        if len(motion) >= 3:
            return motion
        # A partial pool would otherwise mix one clip with two stills, which
        # cuts between moving and frozen footage inside one Reel. Prefer three
        # of the same kind.
        return _comfyui_assets(params) or motion or _library_assets()
    if media_source == "comfyui":
        return _comfyui_assets(params) or _library_assets()
    if media_source == "storyblocks":
        return _storyblocks_assets(task_id or "quote-reel", params, duration=duration) or _library_assets()
    if media_source in {"pexels", "pixabay", "coverr", "stock"}:
        provider = None if media_source == "stock" else media_source
        return _stock_assets(
            task_id or "quote-reel",
            params,
            duration=duration,
            provider=provider,
        ) or _library_assets()
    if media_source == "auto":
        task = task_id or "quote-reel"
        return (
            _library_assets()
            or _comfyui_assets(params)
            or _storyblocks_assets(task, params, duration=duration)
            or _stock_assets(task, params, duration=duration)
        )
    return _library_assets()


def _select_timeline_assets(assets: list[QuoteReelAsset]) -> list[QuoteReelAsset]:
    if len(assets) <= 3:
        return assets
    return assets[:3]


def build_background_clip(
    task_id: str,
    assets: list[QuoteReelAsset],
    *,
    duration: float,
    video_aspect: VideoAspect = VideoAspect.portrait,
    threads: int = 2,
) -> tuple[str, list[str]]:
    selected = _select_timeline_assets(assets)
    if not selected:
        raise ValueError("no quote reel media found")

    task_dir = utils.task_dir(task_id)
    per_clip = duration / len(selected)
    normalized_paths = []
    for index, asset in enumerate(selected, start=1):
        output = os.path.join(task_dir, f"quote-bg-{index}.mp4")
        if asset.kind == "image":
            video.image_to_video_clip(
                asset.path,
                output,
                per_clip,
                video_aspect=video_aspect,
                threads=threads,
            )
        else:
            video.video_to_normalized_clip(
                asset.path,
                output,
                per_clip,
                video_aspect=video_aspect,
                threads=threads,
                preserve_audio=False,
            )
        normalized_paths.append(output)

    combined = os.path.join(task_dir, "quote-background.mp4")
    video.combine_article_clips(normalized_paths, combined, duration, threads=threads)
    return combined, normalized_paths


TRACKING = 0.015
_TEXT_FILL = (255, 255, 255, 245)


def _fonts(size: int) -> dict[bool, object]:
    return {
        False: typography.font(typography.SERIF, size, "Medium"),
        True: typography.font(typography.SERIF, size, "Bold"),
    }


def _layout_quote(
    draw: ImageDraw.ImageDraw,
    quote: str,
    width: int,
) -> tuple[dict, list[list[tuple[str, bool]]]]:
    max_width = width * 0.62
    tokens = typography.tokens(typography.parse_accent(quote))
    for size in range(58, 35, -2):
        fonts = _fonts(size)
        lines = typography.wrap_tokens(draw, tokens, fonts, max_width, TRACKING, MAX_WORDS_PER_LINE)
        if len(lines) <= MAX_QUOTE_LINES:
            return fonts, lines
    fonts = _fonts(36)
    return fonts, typography.wrap_tokens(draw, tokens, fonts, max_width, TRACKING, MAX_WORDS_PER_LINE)[:4]


def _text_band(height: int, start_y: int, total_h: int) -> tuple[int, int]:
    """Vertical slice the quote occupies, clamped to the frame."""
    top = max(0, start_y - int(height * 0.02))
    bottom = min(height, start_y + total_h + int(height * 0.02))
    return top, bottom


def _scrim(width: int, height: int, band: tuple[int, int], strength: float) -> Image.Image:
    """A soft dark wash behind the quote.

    The reference Reels get away with no scrim because their footage is already
    dark where the text sits. Stock clips are not, and white serif on a pale sky
    is what made the 2026-08-16 render unreadable. Strength is derived from the
    measured luminance of the band, so dark footage still gets nothing.
    """
    top, bottom = band
    layer = Image.new("L", (1, height), 0)
    pixels = layer.load()
    peak = int(max(0.0, min(1.0, strength)) * 255)
    feather = max(1, int((bottom - top) * 0.55))
    for y in range(height):
        if top <= y <= bottom:
            value = peak
        elif y < top:
            value = int(peak * max(0.0, 1 - (top - y) / feather))
        else:
            value = int(peak * max(0.0, 1 - (y - bottom) / feather))
        pixels[0, y] = value
    mask = layer.resize((width, height))
    wash = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    wash.putalpha(mask)
    return wash


def quote_overlay_image(
    quote: str,
    width: int,
    height: int,
    *,
    backdrop_luma: float | None = None,
) -> Image.Image:
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    shadow_draw = ImageDraw.Draw(shadow)
    fonts, lines = _layout_quote(draw, quote, width)
    line_h = int(fonts[False].size * 1.28)
    total_h = line_h * len(lines)
    start_y = int(height * 0.43 - total_h / 2)
    band = _text_band(height, start_y, total_h)

    # White text needs the backdrop below roughly 45% luminance to stay readable.
    # Above that, wash the band down by however much is missing.
    if backdrop_luma is not None:
        target = _cfg_float("quote_reel_scrim_target_luma", 110.0)
        deficit = (float(backdrop_luma) - target) / 255.0
        strength = max(0.0, min(_cfg_float("quote_reel_scrim_max", 0.55), deficit * 1.6))
        if strength > 0.02:
            canvas = Image.alpha_composite(_scrim(width, height, band, strength), canvas)
            draw = ImageDraw.Draw(canvas)

    for offset_y, alpha in ((3, 150), (7, 90)):
        y = start_y + offset_y
        for line in lines:
            typography.draw_line_centered(shadow_draw, width, y, line, fonts, (0, 0, 0, alpha), TRACKING)
            y += line_h
    shadow = shadow.filter(ImageFilter.GaussianBlur(2.2))
    canvas = Image.alpha_composite(canvas, shadow)
    draw = ImageDraw.Draw(canvas)
    y = start_y
    for line in lines:
        typography.draw_line_centered(draw, width, y, line, fonts, _TEXT_FILL, TRACKING)
        y += line_h
    return canvas


def _band_luma(path: str, height: int) -> float | None:
    """Mean luminance of the strip the quote will sit in, across the clip."""
    frames = _sample_frames(path)
    if not frames:
        return None
    top, bottom = int(height * 0.34), int(height * 0.56)
    values = []
    for frame in frames:
        crop = frame.convert("L").crop((0, top, frame.width, min(bottom, frame.height)))
        values.append(float(ImageStat.Stat(crop).mean[0]))
    return sum(values) / len(values)


def _music_bed(duration: float):
    """A quiet loop under the quote, or None.

    Source audio is stripped from the raw clips (traffic, wind, a stray voice),
    which left the reel completely silent — the reference Reels all carry a bed,
    and a silent Reel gets skipped past on Instagram.
    """
    if not _cfg_bool("quote_reel_music_enabled", True):
        return None
    bed = str(config.app.get("quote_reel_music_file", "") or "").strip()
    try:
        path = video.get_bgm_file(bgm_type="random" if not bed else "", bgm_file=bed)
    except Exception as exc:
        logger.warning(f"quote reel music selection failed: {exc}")
        return None
    if not path or not os.path.exists(path):
        logger.info("quote reel: no music bed available, rendering silent")
        return None
    volume = _cfg_float("quote_reel_music_volume", 0.22)
    try:
        clip = AudioFileClip(path)
        return clip.with_effects(
            [
                afx.MultiplyVolume(volume),
                afx.AudioLoop(duration=duration),
                afx.AudioFadeIn(1.5),
                afx.AudioFadeOut(2.5),
            ]
        )
    except Exception as exc:
        logger.warning(f"quote reel music bed failed ({path}): {exc}")
        return None


def add_quote_overlay(
    background_path: str,
    output_path: str,
    quote: str,
    *,
    threads: int = 2,
) -> str:
    source = None
    overlay = None
    composite = None
    music = None
    try:
        source = VideoFileClip(background_path, audio=True)
        width, height = source.size
        rendered = quote_overlay_image(
            quote,
            width,
            height,
            backdrop_luma=_band_luma(background_path, height),
        )
        overlay = ImageClip(np.array(rendered), transparent=True).with_duration(
            source.duration
        )
        composite = CompositeVideoClip(
            [source, overlay],
            size=(width, height),
        ).with_duration(source.duration)
        music = _music_bed(float(source.duration or 0))
        if music is not None:
            composite = composite.with_audio(music)
        kwargs = {
            "fps": video.fps,
            "logger": None,
            "threads": threads or 2,
        }
        if getattr(composite, "audio", None) is not None:
            kwargs["audio_codec"] = video.audio_codec
            kwargs["audio_bitrate"] = video.audio_bitrate
        video._write_videofile_with_codec_fallback(
            composite,
            output_path,
            codec=video._get_configured_video_codec(),
            **kwargs,
        )
    finally:
        video.close_clip(composite)
        video.close_clip(music)
        video.close_clip(overlay)
        video.close_clip(source)
    return output_path


def _inspect_video(path: str) -> dict:
    clip = None
    try:
        clip = VideoFileClip(path, audio=False)
        width, height = clip.size
        return {
            "duration": float(clip.duration or 0),
            "width": int(width),
            "height": int(height),
        }
    finally:
        video.close_clip(clip)


def _sample_frames(path: str, *, max_samples: int = 3) -> list[Image.Image]:
    clip = None
    frames: list[Image.Image] = []
    try:
        clip = VideoFileClip(path, audio=False)
        duration = float(clip.duration or 0)
        if duration <= 0:
            return []
        if max_samples <= 1:
            times = [duration / 2]
        else:
            times = [
                duration * position
                for position in (0.2, 0.5, 0.8)[:max_samples]
            ]
        for sample_time in times:
            frame = clip.get_frame(min(max(0.0, sample_time), max(0.0, duration - 0.05)))
            frames.append(Image.fromarray(frame).convert("RGB"))
    except Exception as exc:
        logger.warning(f"quote reel frame sampling failed: {path}: {exc}")
    finally:
        video.close_clip(clip)
    return frames


def _visual_metrics(path: str) -> dict:
    frames = _sample_frames(path)
    if not frames:
        return {"sampled": 0}
    contrasts = []
    brightness = []
    for frame in frames:
        gray = frame.convert("L")
        stat = ImageStat.Stat(gray)
        brightness.append(float(stat.mean[0]))
        contrasts.append(float(stat.stddev[0]))
    return {
        "sampled": len(frames),
        "mean_brightness": sum(brightness) / len(brightness),
        "mean_contrast": sum(contrasts) / len(contrasts),
    }


def _scrim_strength(raw_luma: float) -> float:
    """The wash quote_overlay_image would apply at this backdrop luminance."""
    target = _cfg_float("quote_reel_scrim_target_luma", 110.0)
    deficit = (float(raw_luma) - target) / 255.0
    return max(0.0, min(_cfg_float("quote_reel_scrim_max", 0.55), deficit * 1.6))


def legibility_check(background_path: str, height: int) -> dict:
    """Will white serif read against this backdrop, once the scrim is applied?

    The old gate measured stddev over the whole frame, which a busy-but-pale sky
    passes easily — the 2026-08-16 reel scored 29.96 and was still unreadable.
    What matters is the luminance of the strip the text actually sits in.
    """
    raw = _band_luma(background_path, height)
    if raw is None:
        return {"measured": False}
    strength = _scrim_strength(raw)
    effective = raw * (1.0 - strength)
    ceiling = _cfg_float("quote_reel_max_text_band_luma", 150.0)
    return {
        "measured": True,
        "raw_band_luma": round(raw, 1),
        "scrim_strength": round(strength, 3),
        "effective_band_luma": round(effective, 1),
        "ceiling": ceiling,
        "readable": effective <= ceiling,
    }


def _ocr_text_from_image(frame: Image.Image) -> str:
    try:
        import pytesseract  # type: ignore
    except ImportError:
        return ""
    try:
        gray = frame.convert("L")
        text = pytesseract.image_to_string(gray, config="--psm 6")
    except Exception as exc:
        logger.warning(f"quote reel OCR scan failed: {exc}")
        return ""
    return " ".join(text.split())


def _embedded_text_scan(path: str) -> dict:
    frames = _sample_frames(path)
    if not frames:
        return {"available": False, "detected": False, "sampled": 0, "text": ""}
    scanned = []
    for frame in frames:
        text = _ocr_text_from_image(frame)
        if text:
            scanned.append(text)
    combined = " ".join(scanned).strip()
    alnum = re.sub(r"[^A-Za-z0-9ÁÉÍÓÚÜÑáéíóúüñ]+", "", combined)
    return {
        "available": bool(scanned),
        "detected": len(alnum) >= 10,
        "sampled": len(frames),
        "text": combined[:120],
    }


def _source_text(source: dict) -> str:
    return _coerce_str(
        " ".join(
            str(source.get(key) or "")
            for key in ("metadata_text", "label", "source_page", "license")
        )
    ).lower()


def _coerce_str(value: object) -> str:
    return str(value or "").strip()


def _asset_review_reasons(asset: QuoteReelAsset) -> list[str]:
    source = asset.source_info if isinstance(asset.source_info, dict) else {}
    reasons: list[str] = []
    if source.get("reference_only"):
        reasons.append(f"{asset.label}: marked as reference-only media")
    if source.get("raw_text_free") is False:
        reasons.append(f"{asset.label}: not marked as raw text-free footage")
    if source.get("is_editorial"):
        reasons.append(f"{asset.label}: editorial-only source")

    # The release checks are opt-in. Stock metadata never carries an explicit
    # talent release, so demanding one flagged every clip containing a person —
    # which blocked the 2026-08-16 render even though the reference Reels are
    # built on exactly that footage (a cyclist, a skater, a figure walking away).
    # Turn these on only for a client whose contract actually requires releases.
    text = _source_text(source)
    if _cfg_bool("quote_reel_require_talent_release", False):
        contains_people = bool(source.get("contains_people")) or any(
            term in text for term in _PEOPLE_TERMS
        )
        if contains_people and source.get("has_talent_released") is not True:
            reasons.append(f"{asset.label}: possible people/talent release risk")
    if _cfg_bool("quote_reel_require_property_release", False):
        contains_property = bool(source.get("contains_property")) or any(
            term in text for term in _PROPERTY_TERMS
        )
        if contains_property and source.get("has_property_released") is not True:
            reasons.append(f"{asset.label}: possible property release risk")
    return reasons


def quality_check(
    final_path: str,
    quote: str,
    assets: list[QuoteReelAsset],
    *,
    background_path: str = "",
) -> dict:
    issues: list[str] = []
    review_reasons: list[str] = []
    meta: dict = {}
    if not assets:
        issues.append("no curated or uploaded media selected")
    if _is_bad_quote(quote):
        issues.append("quote failed style guard")
    for asset in assets:
        review_reasons.extend(_asset_review_reasons(asset))
    if (
        not final_path
        or not os.path.exists(final_path)
        or os.path.getsize(final_path) < 1024
    ):
        issues.append("rendered video file is missing or empty")
    else:
        try:
            meta = _inspect_video(final_path)
        except Exception as exc:
            issues.append(f"rendered video cannot be inspected: {exc}")
        duration = float(meta.get("duration") or 0)
        width = int(meta.get("width") or 0)
        height = int(meta.get("height") or 0)
        if duration < MIN_SECONDS or duration > MAX_SECONDS:
            issues.append(f"duration outside quote reel range: {duration:.2f}s")
        if width <= 0 or height <= 0 or abs((width / height) - (9 / 16)) > 0.035:
            issues.append(f"video is not 9:16 portrait: {width}x{height}")

    visual_path = background_path or final_path
    visual = _visual_metrics(visual_path) if visual_path else {"sampled": 0}
    if visual.get("sampled"):
        min_contrast = _cfg_float("quote_reel_min_visual_contrast", 12.0)
        if float(visual.get("mean_contrast") or 0) < min_contrast:
            review_reasons.append(
                f"background contrast is low: {float(visual.get('mean_contrast') or 0):.1f}"
            )

    legibility = (
        legibility_check(visual_path, int(meta.get("height") or 1920))
        if visual_path
        else {"measured": False}
    )
    if legibility.get("measured") and not legibility.get("readable"):
        review_reasons.append(
            "quote may not read against this backdrop: text band sits at "
            f"{legibility['effective_band_luma']} after scrim "
            f"(ceiling {legibility['ceiling']})"
        )

    text_scan = _embedded_text_scan(background_path) if background_path else {
        "available": False,
        "detected": False,
        "sampled": 0,
        "text": "",
    }
    if text_scan.get("detected"):
        review_reasons.append("possible embedded text detected in source footage")

    publishable = not issues and not review_reasons
    return {
        "passed": not issues,
        "publishable": publishable,
        "review_required": bool(review_reasons),
        "issues": issues,
        "review_reasons": review_reasons,
        "duration": meta.get("duration"),
        "resolution": f"{meta.get('width')}x{meta.get('height')}" if meta else "",
        "asset_count": len(assets),
        "visual": visual,
        "legibility": legibility,
        "embedded_text_scan": text_scan,
    }


def _review_queue_path() -> str:
    directory = utils.storage_dir("quote_reel_review", create=True)
    return os.path.join(directory, "queue.json")


def _load_review_queue() -> list[dict]:
    try:
        with open(_review_queue_path(), encoding="utf-8") as fp:
            data = json.load(fp)
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def enqueue_review(
    task_id: str,
    *,
    final_path: str,
    quote: str,
    caption: str,
    qc: dict,
    assets: list[QuoteReelAsset],
) -> str:
    queue = [item for item in _load_review_queue() if item.get("task_id") != task_id]
    queue.append(
        {
            "task_id": task_id,
            "content_mode": CONTENT_MODE,
            "final_path": final_path,
            "quote": quote,
            "caption": caption,
            "review_reasons": qc.get("review_reasons") or qc.get("issues") or [],
            "qc": qc,
            "assets": [asset.source_info for asset in assets],
        }
    )
    path = _review_queue_path()
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(queue[-200:], fp, ensure_ascii=False, indent=2)
    return path


def _write_artifact(task_id: str, payload: dict) -> str:
    artifact_path = os.path.join(utils.task_dir(task_id), "quote_reel.json")
    with open(artifact_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    return artifact_path


def _with_series_line(caption: str, part: int | None) -> str:
    if not part:
        return caption
    from app.services import series

    return f"{series.reel_label(part)}\n\n{caption}"


def _schedule_if_enabled(
    final_path: str,
    caption: str,
    quote: str,
    qc: dict,
    *,
    caption_meta: dict | None = None,
    assets: list[QuoteReelAsset] | None = None,
) -> dict:
    if not qc.get("passed"):
        return {
            "success": False,
            "skipped": True,
            "error": "quality check failed",
        }
    if not qc.get("publishable"):
        return {
            "success": False,
            "skipped": True,
            "error": "review required before publishing",
        }
    if not _cfg_bool("quote_reel_auto_schedule_enabled", False):
        return {
            "success": False,
            "skipped": True,
            "error": "quote reel auto-scheduling disabled",
        }

    # Auto-publish only the format that was actually approved.
    #
    # `select_media_assets` falls back to stills whenever the motion pool cannot
    # supply three clips — on a fresh install, after a night the GPU was busy, or
    # once the pool is drained by daily use. That fallback is right for
    # *rendering*: a slideshow Reel beats a failed one. It is wrong for
    # *publishing*: sign-off was given on motion Reels, and a stills Reel going
    # out unattended under that approval is a silent substitution.
    #
    # So a Reel that fell back still renders and still lands in the review queue
    # — it just needs a human to release it.
    if _cfg_bool("quote_reel_require_motion_for_autopublish", True):
        wanted = str(config.app.get("quote_reel_media_source", "") or "").strip().lower()
        if wanted == "comfyui_motion":
            still_backed = [a for a in (assets or []) if a.kind != "video"]
            if still_backed or not assets:
                return {
                    "success": False,
                    "skipped": True,
                    "error": (
                        f"motion pool could not dress this Reel "
                        f"({len(still_backed)} of {len(assets or [])} scenes fell back "
                        "to stills); queued for review instead of auto-publishing"
                    ),
                }

    from app.services import hashtags, postiz, series

    if not (postiz.postiz_service.enabled or postiz.postiz_service.auto_schedule_enabled):
        return {"success": False, "skipped": True, "error": "Postiz is disabled"}
    if not postiz.postiz_service.is_auto_schedule_configured():
        return {
            "success": False,
            "skipped": True,
            "error": "Postiz auto-scheduling is not configured",
        }

    part = series.reel_current()
    caption_meta = caption_meta or {}
    assets = assets or []
    set_id = str(caption_meta.get("hashtag_set") or "").strip() or None
    variant = {
        "kind": "quiet_quote_reel",
        "style": "centered_quote",
        "script_style": "quiet_quote_reel",
        "subtitle_renderer": "centered_serif_quote",
        "subtitle_cadence": "none",
        "quote": typography.strip_accent(quote)[:120],
        "quote_chars": len(typography.strip_accent(quote)),
        "language": language_name(),
        "caption_style": caption_meta.get("caption_style"),
        "hashtag_set": set_id,
        "video_seconds": qc.get("duration"),
        "duration": qc.get("duration"),
        "asset_providers": sorted({asset.provider for asset in assets}),
    }
    if part:
        variant["series"] = series.reel_label(part)
    result = postiz.schedule_video(
        final_path,
        _with_series_line(caption, part),
        variant=variant,
        set_id=set_id,
    )
    result["provider"] = "postiz"
    if result.get("success"):
        series.reel_advance()
        if set_id:
            hashtags.mark_used(set_id)
    return result


def _fail(task_id: str, stage: str, error: str, **extra) -> dict:
    result = {
        "task_id": task_id,
        "state": const.TASK_STATE_FAILED,
        "error": error,
        "failed_stage": stage,
        "content_mode": CONTENT_MODE,
        **extra,
    }
    sm.state.update_task(
        task_id,
        state=const.TASK_STATE_FAILED,
        progress=100,
        **result,
    )
    return result


def render_quote_reel(
    task_id: str,
    params: VideoParams,
    *,
    stop_at: str = "video",
) -> dict:
    sm.state.update_task(task_id, state=const.TASK_STATE_PROCESSING, progress=5)
    duration = target_seconds()
    # ``quote`` keeps the *accent* markers the overlay renders in a heavier
    # weight; ``plain`` is what the caption, the ledger and the API report show.
    quote = resolve_quote(params)
    plain = typography.strip_accent(quote)
    caption_meta = build_caption_variant(
        quote,
        getattr(params, "video_subject", "") or "",
    )
    caption = caption_meta["caption"]

    if stop_at == "script":
        result = {
            "script": plain,
            "caption": caption,
            "content_mode": CONTENT_MODE,
            "ready_to_publish": False,
        }
        sm.state.update_task(
            task_id,
            state=const.TASK_STATE_COMPLETE,
            progress=100,
            **result,
        )
        return result

    try:
        assets = select_media_assets(params, task_id=task_id, duration=duration)
        if not assets:
            return _fail(
                task_id,
                "media",
                "no quote reel media found; add raw text-free clips to quote_reel_media_dir or upload local video_materials",
                script=plain,
                caption=caption,
            )
        aspect = VideoAspect(
            getattr(params, "video_aspect", None) or VideoAspect.portrait
        )
        background_path, normalized_paths = build_background_clip(
            task_id,
            assets,
            duration=duration,
            video_aspect=aspect,
            threads=int(getattr(params, "n_threads", 2) or 2),
        )
        final_path = os.path.join(utils.task_dir(task_id), "final-quote-reel.mp4")
        add_quote_overlay(
            background_path,
            final_path,
            quote,
            threads=int(getattr(params, "n_threads", 2) or 2),
        )
    except Exception as exc:
        logger.exception(f"quote reel render failed, task_id: {task_id}, error: {exc}")
        return _fail(
            task_id,
            "render",
            f"{type(exc).__name__}: {exc}",
            script=plain,
            caption=caption,
        )

    qc = quality_check(final_path, plain, assets, background_path=background_path)
    review_queue_path = ""
    if qc.get("review_required") or qc.get("issues"):
        review_queue_path = enqueue_review(
            task_id,
            final_path=final_path,
            quote=plain,
            caption=caption,
            qc=qc,
            assets=assets,
        )
    publish_result = _schedule_if_enabled(
        final_path,
        caption,
        quote,
        qc,
        caption_meta=caption_meta,
        assets=assets,
    )
    ready_to_publish = (
        bool(qc.get("publishable"))
        and not bool(publish_result.get("success"))
    )
    artifact = {
        "quote": plain,
        "caption": caption,
        "caption_meta": caption_meta,
        "assets": [asset.source_info for asset in assets],
        "normalized_clips": [os.path.basename(path) for path in normalized_paths],
        "background": os.path.basename(background_path),
        "final": os.path.basename(final_path),
        "qc": qc,
        "publish_result": publish_result,
        "ready_to_publish": ready_to_publish,
        "review_queue_path": review_queue_path,
    }
    artifact_path = _write_artifact(task_id, artifact)
    result = {
        "videos": [final_path],
        "combined_videos": [background_path],
        "script": plain,
        "caption": caption,
        "content_mode": CONTENT_MODE,
        "quote_reel_qc": qc,
        "quote_reel_artifact": artifact_path,
        "quote_reel_review_queue": review_queue_path,
        "review_required": bool(qc.get("review_required")),
        "media_attribution": [asset.label for asset in assets],
        "publish_result": publish_result,
        "ready_to_publish": ready_to_publish,
    }
    sm.state.update_task(
        task_id,
        state=const.TASK_STATE_COMPLETE,
        progress=100,
        **result,
    )
    logger.success(
        f"quote reel finished, task_id: {task_id}, qc_passed={qc.get('passed')}"
    )
    return result
