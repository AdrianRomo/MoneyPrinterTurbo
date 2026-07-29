"""Draft a grounded video script from a single article URL.

This ties together the three existing pieces — SSRF-safe fetching
(:mod:`app.services.article_ingestion`), a structured production brief
(:func:`app.services.article_llm.analyze_article_brief`) and strict,
no-invention script generation (:func:`app.services.llm.generate_script`) — into
one UI-agnostic entry point so the WebUI, the HTTP API and the CLI can all reuse
the exact same drafting logic instead of duplicating it inside a Streamlit
button handler.

Nothing here imports Streamlit; callers handle their own presentation and
caching. Article text is untrusted and is only ever passed to the LLM through
the injection-guarded prompts in ``article_llm``.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

from loguru import logger

from app.config import config
from app.services import article_ingestion, article_llm, llm

# Extraction can technically succeed with a couple of stray words (nav text,
# cookie banners). Require a small floor before we treat it as a real article so
# we don't ground a script on a paywall stub.
MIN_ARTICLE_WORDS = 40

# Process-level cache of fetched articles keyed by canonical URL. Articles change
# slowly, so a short TTL lets the WebUI, API and CLI avoid re-fetching the same
# link (e.g. draft then generate, or repeated automated runs) without ever
# serving stale content for long. The WebUI additionally caches per session.
_DEFAULT_CACHE_TTL = 900  # seconds
_MAX_CACHE_ENTRIES = 128
_REFERENCE_CACHE: "dict[str, tuple[float, ArticleReference]]" = {}
_REFERENCE_CACHE_LOCK = threading.Lock()

# Multi-source drafting bounds.
MAX_SOURCES = 3
MAX_COMBINED_CHARS = 12000
_WIRE_DUPLICATE_TITLE_SIMILARITY = 0.9


class ArticleDraftError(Exception):
    """Base class for draft failures the caller can turn into UI/API messages."""


class ArticleEmptyError(ArticleDraftError):
    """Raised when a URL fetched fine but yielded no usable article body."""


# Stable error categories so the WebUI and the HTTP API describe the same
# failure the same way (each maps to its own localized/So-API message).
FETCH_ERROR_CATEGORIES = (
    "blocked",      # SSRF / disallowed address
    "not_found",    # HTTP 404
    "unavailable",  # other HTTP >= 400
    "timeout",      # request timed out
    "network",      # connection / DNS failure
    "empty",        # fetched OK but no readable article text
    "unknown",      # anything else
)

_FETCH_ERROR_MESSAGES = {
    "blocked": "The link was blocked for security reasons (it may point to a private or disallowed address).",
    "not_found": "The article could not be found (HTTP 404).",
    "unavailable": "The source responded with an error.",
    "timeout": "The source took too long to respond.",
    "network": "Could not reach the source. Check the link and your connection.",
    "empty": "No readable article text was found at that URL.",
    "unknown": "Could not fetch the reference article. Check the URL and try again.",
}


def classify_fetch_error(exc: BaseException) -> str:
    """Map a fetch/draft exception to a stable :data:`FETCH_ERROR_CATEGORIES` code."""
    if isinstance(exc, ArticleEmptyError):
        return "empty"
    if isinstance(exc, article_ingestion.SecurityError):
        text = str(exc).lower()
        if "http status" in text:
            return "not_found" if "404" in text else "unavailable"
        return "blocked"
    try:
        import requests

        if isinstance(exc, requests.exceptions.Timeout):
            return "timeout"
        if isinstance(exc, requests.exceptions.RequestException):
            return "network"
    except Exception:  # pragma: no cover - requests always present in practice
        pass
    return "unknown"


def fetch_error_message(category: str) -> str:
    """English message for an error category (used by the HTTP API)."""
    return _FETCH_ERROR_MESSAGES.get(category, _FETCH_ERROR_MESSAGES["unknown"])


@dataclass
class ArticleReference:
    """A fetched + extracted article, ready to ground a script on."""

    requested_url: str
    url: str  # final URL after redirects
    domain: str
    title: str
    text: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def is_usable(self) -> bool:
        return self.word_count >= MIN_ARTICLE_WORDS

    def snippet(self, limit: int = 300) -> str:
        text = " ".join(self.text.split())
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "…"


@dataclass
class ArticleDraft:
    """The reviewable output of drafting: everything the UI/API fills in."""

    subject: str
    script: str
    requirements: str
    brief: dict = field(default_factory=dict)
    reference: Optional[ArticleReference] = None

    @property
    def visual_themes(self) -> list[str]:
        return list(self.brief.get("visual_themes") or [])

    @property
    def recommended_paragraphs(self) -> int:
        return int(self.brief.get("recommended_paragraphs") or 0)

    @property
    def suggested_terms(self) -> list[str]:
        """Merge the brief's visual themes with real entities from the article so
        the stock-footage search prefers the actual named subjects."""
        terms: list[str] = []
        seen: set[str] = set()
        for term in list(self.visual_themes) + reference_entities(self.reference):
            value = (term or "").strip()
            if value and value.lower() not in seen:
                terms.append(value)
                seen.add(value.lower())
        return terms[:12]


def _domain_of(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:  # pragma: no cover - defensive
        return ""


def reference_entities(reference: Optional["ArticleReference"], limit: int = 12) -> list[str]:
    """Real named entities (people/places/orgs) extracted from the article.

    Used to bias stock-footage search toward the article's actual subjects.
    Best-effort: returns an empty list if extraction is unavailable.
    """
    if reference is None or not reference.text.strip():
        return []
    try:
        return article_ingestion.entities_of(
            f"{reference.title}. {reference.text}", limit=limit
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(f"entity extraction failed: {exc}")
        return []


def _cache_ttl() -> int:
    try:
        return int(config.app.get("article_reference_cache_ttl", _DEFAULT_CACHE_TTL))
    except Exception:  # pragma: no cover - defensive
        return _DEFAULT_CACHE_TTL


def _cache_key(url: str) -> str:
    try:
        return article_ingestion.canonicalize_url(url) or url
    except Exception:  # pragma: no cover - defensive
        return url


def _cache_get(key: str) -> Optional["ArticleReference"]:
    ttl = _cache_ttl()
    if ttl <= 0:
        return None
    now = time.monotonic()
    with _REFERENCE_CACHE_LOCK:
        entry = _REFERENCE_CACHE.get(key)
        if entry is None:
            return None
        stored_at, reference = entry
        if now - stored_at > ttl:
            _REFERENCE_CACHE.pop(key, None)
            return None
        return reference


def _cache_put(key: str, reference: "ArticleReference") -> None:
    if _cache_ttl() <= 0:
        return
    with _REFERENCE_CACHE_LOCK:
        _REFERENCE_CACHE[key] = (time.monotonic(), reference)
        if len(_REFERENCE_CACHE) > _MAX_CACHE_ENTRIES:
            # Drop the oldest entry to bound memory (simple FIFO by insertion age).
            oldest_key = min(_REFERENCE_CACHE, key=lambda k: _REFERENCE_CACHE[k][0])
            _REFERENCE_CACHE.pop(oldest_key, None)


def clear_reference_cache() -> None:
    """Drop all cached articles (used by tests and manual cache busting)."""
    with _REFERENCE_CACHE_LOCK:
        _REFERENCE_CACHE.clear()


def fetch_reference(url: str, *, use_cache: bool = True) -> ArticleReference:
    """Fetch and extract a single article URL.

    Reuses :func:`article_ingestion.fetch_url` for SSRF validation, redirect
    re-validation, size caps and content-type whitelisting. Results are cached
    per canonical URL for a short TTL (see ``article_reference_cache_ttl``) so
    repeated drafts/generations don't re-hit the network. Raises
    :class:`article_ingestion.SecurityError` or the underlying network error on
    failure; returns an :class:`ArticleReference` (possibly with empty text) on
    success so the caller can decide how to handle thin pages.
    """
    requested = (url or "").strip()
    key = _cache_key(requested)
    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            logger.debug(f"article reference cache hit: {key}")
            return cached

    final_url, body, _content_type = article_ingestion.fetch_url(
        requested,
        allowed_content_types=["text/html", "application/xhtml+xml", "text/plain"],
    )
    extracted = article_ingestion.extract_article(body, url=final_url)
    reference = ArticleReference(
        requested_url=requested,
        url=final_url,
        domain=_domain_of(final_url),
        title=extracted.get("title", "") or "",
        text=extracted.get("text", "") or "",
    )
    # Only cache pages that actually yielded text; don't memoize empty/paywall hits.
    if use_cache and reference.text.strip():
        _cache_put(key, reference)
    return reference


def draft_from_reference(
    reference: ArticleReference,
    *,
    language: str = "",
    paragraph_number: int = 1,
    custom_system_prompt: str = "",
    fallback_subject: str = "",
) -> ArticleDraft:
    """Turn an already-fetched article into a grounded, reviewable draft.

    Runs the production brief (best-effort — a failed brief degrades to an empty
    one rather than aborting) and then a strict, no-invention script grounded on
    the article body.
    """
    if not reference.text.strip():
        raise ArticleEmptyError("no readable article text to draft from")

    try:
        brief = article_llm.analyze_article_brief(
            reference.title, reference.text, language=language
        )
    except Exception as exc:  # noqa: BLE001 - brief is optional context
        logger.error(f"article brief failed, drafting without it: {exc}")
        brief = {}

    requirements = article_llm.brief_to_requirements(brief) if brief else ""
    subject = (
        (brief.get("subject") or "").strip()
        or reference.title
        or fallback_subject
    )
    paragraphs = brief.get("recommended_paragraphs") or paragraph_number

    script = llm.generate_script(
        video_subject=subject,
        language=language,
        paragraph_number=paragraphs,
        video_script_prompt=requirements,
        custom_system_prompt=custom_system_prompt,
        reference_content=reference.text,
        strict_source=True,
    )

    return ArticleDraft(
        subject=subject,
        script=script,
        requirements=requirements,
        brief=brief,
        reference=reference,
    )


def build_article_draft(
    url: str,
    *,
    language: str = "",
    paragraph_number: int = 1,
    custom_system_prompt: str = "",
    fallback_subject: str = "",
    reference: Optional[ArticleReference] = None,
) -> ArticleDraft:
    """Fetch ``url`` (unless a ``reference`` is supplied) and draft from it."""
    if reference is None:
        reference = fetch_reference(url)
    return draft_from_reference(
        reference,
        language=language,
        paragraph_number=paragraph_number,
        custom_system_prompt=custom_system_prompt,
        fallback_subject=fallback_subject,
    )


def parse_source_urls(text: str, *, limit: int = MAX_SOURCES) -> list[str]:
    """Split a field that may contain several article links (whitespace/comma
    separated) into a de-duplicated, bounded list of URLs."""
    if not text:
        return []
    urls: list[str] = []
    seen: set[str] = set()
    for token in re.split(r"[\s,]+", text.strip()):
        token = token.strip()
        if token and token not in seen:
            seen.add(token)
            urls.append(token)
        if len(urls) >= limit:
            break
    return urls


def fetch_references(
    urls: list[str], *, use_cache: bool = True
) -> "tuple[list[ArticleReference], list[str]]":
    """Fetch several article URLs, dropping empties and near-duplicate wire
    copies (by title similarity). Returns ``(usable_references, failed_urls)``."""
    refs: list[ArticleReference] = []
    failed: list[str] = []
    for url in urls:
        try:
            reference = fetch_reference(url, use_cache=use_cache)
        except Exception as exc:  # noqa: BLE001 - collect per-URL failures
            logger.error(f"failed to fetch reference article {url}: {exc}")
            failed.append(url)
            continue
        if not reference.text.strip():
            failed.append(url)
            continue
        if any(
            reference.title
            and existing.title
            and article_ingestion.title_similarity(reference.title, existing.title)
            > _WIRE_DUPLICATE_TITLE_SIMILARITY
            for existing in refs
        ):
            logger.debug(f"skipping near-duplicate source: {url}")
            continue
        refs.append(reference)
    return refs, failed


def combine_references(
    refs: list[ArticleReference], *, requested_url: str = ""
) -> ArticleReference:
    """Merge multiple articles into one grounding reference with clear per-source
    markers, so a single strict draft can synthesize across all of them."""
    if not refs:
        raise ArticleEmptyError("no usable sources to combine")
    if len(refs) == 1 and requested_url:
        first = refs[0]
        return ArticleReference(
            requested_url=requested_url,
            url=first.url,
            domain=first.domain,
            title=first.title,
            text=first.text,
        )
    if len(refs) == 1:
        return refs[0]

    per_source = max(500, MAX_COMBINED_CHARS // len(refs))
    parts: list[str] = []
    for index, reference in enumerate(refs, start=1):
        header = f"[SOURCE {index} — {reference.domain or 'source'}]"
        title = f" {reference.title}" if reference.title else ""
        parts.append(f"{header}{title}\n{reference.text[:per_source]}")
    domains = list(dict.fromkeys(r.domain for r in refs if r.domain))
    return ArticleReference(
        requested_url=requested_url or ", ".join(r.requested_url for r in refs),
        url=refs[0].url,
        domain=", ".join(domains),
        title=next((r.title for r in refs if r.title), ""),
        text="\n\n".join(parts),
    )


def build_combined_reference(
    urls: list[str], *, use_cache: bool = True, requested_url: str = ""
) -> "tuple[Optional[ArticleReference], list[ArticleReference], list[str]]":
    """Fetch + dedupe + combine several URLs. Returns
    ``(combined_reference_or_None, used_references, failed_urls)``."""
    refs, failed = fetch_references(urls, use_cache=use_cache)
    if not refs:
        return None, [], failed
    combined = combine_references(refs, requested_url=requested_url)
    return combined, refs, failed


def detected_language(reference: Optional[ArticleReference]) -> str:
    """Best-effort language of the article as a lowercase English name (or "").

    Purely a UI hint — the script language is never forced from this, so an
    article can still be translated into another language on request."""
    if reference is None or not reference.text.strip():
        return ""
    try:
        return article_ingestion.guess_language(reference.text)
    except Exception:  # pragma: no cover - defensive
        return ""


def check_faithfulness(script_text: str, reference: Optional[ArticleReference]) -> dict:
    """Fact-check a script against a reference article (see
    :func:`article_llm.check_faithfulness`). Returns
    ``{"supported": bool, "confidence": float, "issues": [str]}``."""
    reference_text = reference.text if reference is not None else ""
    return article_llm.check_faithfulness(script_text, reference_text)
