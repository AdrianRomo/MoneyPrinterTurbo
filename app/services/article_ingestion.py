"""Article ingestion: fetch, extract, normalize, deduplicate and cluster.

Sources are untrusted input, so this module keeps hard *technical* guarantees:

* Only ``http``/``https`` URLs are fetched.
* SSRF protection: DNS is resolved and every resulting address is rejected if it
  is loopback, private, link-local, reserved, multicast or unspecified. Each
  redirect hop is re-validated, so a public URL cannot redirect to an internal
  one.
* Redirect count, response size, content-type and extracted-text length are all
  bounded; requests time out.
* Extracted text is treated strictly as data — scripts/markup are stripped and
  the text is never interpreted as instructions.

Clustering and keyword/entity extraction here are deterministic *grouping*
helpers (they organize candidates); they are not editorial fact gates. Story
quality/credibility judgement happens later via the LLM assessment step.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from datetime import datetime, timezone
from typing import Callable, Iterable, List, Optional, Tuple
from urllib.parse import parse_qsl, urljoin, urlsplit, urlunsplit

import requests
from loguru import logger

from app.config import config
from app.models.article import (
    ArticleCluster,
    ArticleRecord,
    ArticleStatus,
    SourceType,
    TopicSubscription,
    domain_of,
    text_hash,
    utcnow,
)

# ---------------------------------------------------------------------------
# Tunables (overridable via [app] config keys)
# ---------------------------------------------------------------------------


def _cfg_int(key: str, default: int) -> int:
    try:
        return int(config.app.get(key, default))
    except (TypeError, ValueError):
        return default


DEFAULT_TIMEOUT = (10, 20)  # (connect, read)
MAX_REDIRECTS = 5
# Tracking/campaign params that never change which article a URL points to.
_TRACKING_PARAM_PREFIXES = ("utm_", "utm-")
_TRACKING_PARAMS = {
    "gclid", "fbclid", "mc_cid", "mc_eid", "igshid", "spm", "ref", "ref_src",
    "cmpid", "cid", "ncid", "smid", "s_cid", "at_medium", "at_campaign",
    "recirc", "taid", "__twitter_impression",
}
_USER_AGENT = (
    "Mozilla/5.0 (compatible; MoneyPrinterTurbo Article Mode/1.0; +automation)"
)
# Common syndication/wire markers used to tag likely wire copy.
_WIRE_MARKERS = ("(reuters)", "(ap)", "associated press", "agence france-presse", "(afp)")

_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
    "at", "by", "from", "as", "is", "are", "was", "were", "be", "been", "it",
    "its", "this", "that", "these", "those", "he", "she", "they", "we", "you",
    "his", "her", "their", "our", "your", "has", "have", "had", "will", "would",
    "can", "could", "may", "might", "not", "no", "than", "then", "so", "into",
    "about", "after", "over", "new", "said", "says", "more", "who", "what",
    "when", "where", "how", "why", "which", "also", "up", "out", "one", "two",
}


class SecurityError(Exception):
    """Raised when a URL or response violates a fetch security rule."""


# ---------------------------------------------------------------------------
# URL security / SSRF
# ---------------------------------------------------------------------------


def _address_is_blocked(ip_text: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_text)
    except ValueError:
        return True
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _resolve_addresses(host: str) -> List[str]:
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise SecurityError(f"could not resolve host: {exc}") from exc
    return sorted({info[4][0] for info in infos})


def validate_url(url: str) -> str:
    """Return the URL if it is safe to fetch, otherwise raise SecurityError.

    Rejects non-http(s) schemes, credentials in the URL, unresolvable hosts and
    any host that resolves to a non-public address (SSRF protection).
    """
    if not isinstance(url, str) or not url.strip():
        raise SecurityError("empty url")
    parsed = urlsplit(url.strip())
    if parsed.scheme not in {"http", "https"}:
        raise SecurityError(f"unsupported scheme: {parsed.scheme or '<none>'}")
    if parsed.username or parsed.password:
        raise SecurityError("credentials in url are not allowed")
    host = parsed.hostname
    if not host:
        raise SecurityError("url has no host")
    # A literal IP host is checked directly; a name is resolved and every
    # resulting address must be public.
    try:
        ipaddress.ip_address(host)
        addresses = [host]
    except ValueError:
        addresses = _resolve_addresses(host)
    if not addresses:
        raise SecurityError("host did not resolve to any address")
    for address in addresses:
        if _address_is_blocked(address):
            raise SecurityError(f"host resolves to blocked address: {address}")
    return url.strip()


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _tls_verify() -> bool:
    value = config.app.get("tls_verify", True)
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off")
    return bool(value)


def _request_timeout() -> Tuple[int, int]:
    read_timeout = max(1, _cfg_int("article_request_timeout", DEFAULT_TIMEOUT[1]))
    connect_timeout = min(DEFAULT_TIMEOUT[0], read_timeout)
    return connect_timeout, read_timeout


def _read_capped(response: requests.Response, max_bytes: int) -> bytes:
    chunks = bytearray()
    for chunk in response.iter_content(chunk_size=65536):
        if not chunk:
            continue
        chunks.extend(chunk)
        if len(chunks) > max_bytes:
            raise SecurityError(f"response exceeds {max_bytes} bytes")
    return bytes(chunks)


def fetch_url(
    url: str,
    *,
    max_bytes: Optional[int] = None,
    timeout: Optional[Tuple[int, int]] = None,
    max_redirects: Optional[int] = None,
    allowed_content_types: Optional[Iterable[str]] = None,
) -> Tuple[str, bytes, str]:
    """Fetch ``url`` safely, following redirects manually and re-validating each
    hop. Returns ``(final_url, body, content_type)``.

    Every redirect target passes :func:`validate_url` again, so an initially
    public URL cannot bounce the request onto an internal address.
    """
    max_bytes = max_bytes or _cfg_int("article_max_bytes", 5 * 1024 * 1024)
    max_redirects = MAX_REDIRECTS if max_redirects is None else max_redirects
    timeout = timeout or _request_timeout()
    current = validate_url(url)
    headers = {"User-Agent": _USER_AGENT, "Accept-Encoding": "identity"}
    session = requests.Session()
    try:
        for _ in range(max_redirects + 1):
            response = session.get(
                current,
                headers=headers,
                timeout=timeout,
                allow_redirects=False,
                stream=True,
                verify=_tls_verify(),
            )
            try:
                if response.is_redirect or response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("Location")
                    if not location:
                        raise SecurityError("redirect without a location header")
                    current = validate_url(urljoin(current, location))
                    continue
                if response.status_code >= 400:
                    raise SecurityError(f"http status {response.status_code}")
                content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                if allowed_content_types is not None:
                    if content_type and not any(
                        content_type == allowed or content_type.endswith(allowed)
                        for allowed in allowed_content_types
                    ):
                        raise SecurityError(f"unexpected content-type: {content_type}")
                declared = response.headers.get("Content-Length")
                if declared and declared.isdigit() and int(declared) > max_bytes:
                    raise SecurityError(f"content-length exceeds {max_bytes} bytes")
                body = _read_capped(response, max_bytes)
                return current, body, content_type
            finally:
                response.close()
        raise SecurityError("too many redirects")
    finally:
        session.close()


# A fetcher signature that tests can substitute for the network.
Fetcher = Callable[[str], Tuple[str, bytes, str]]


def _default_fetcher(url: str) -> Tuple[str, bytes, str]:
    return fetch_url(
        url,
        allowed_content_types=[
            "text/html", "application/xhtml+xml", "text/plain",
        ],
    )


def _feed_fetcher(url: str) -> Tuple[str, bytes, str]:
    return fetch_url(
        url,
        allowed_content_types=[
            "application/rss+xml", "application/atom+xml", "application/xml",
            "text/xml", "text/html", "application/rss", "rss+xml", "atom+xml",
            "xml", "text/plain",
        ],
    )


# ---------------------------------------------------------------------------
# URL canonicalization
# ---------------------------------------------------------------------------


def canonicalize_url(url: str) -> str:
    """Normalize a URL so the same article resolves to the same key.

    Lowercases the host, drops the fragment and tracking query params, removes a
    trailing slash and default ports. Path case is preserved (paths can be
    case-sensitive)."""
    if not url:
        return ""
    parsed = urlsplit(url.strip())
    scheme = parsed.scheme.lower() or "https"
    host = (parsed.hostname or "").lower()
    if not host:
        return url.strip()
    netloc = host
    if parsed.port and parsed.port not in (80, 443):
        netloc = f"{host}:{parsed.port}"
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=False)
        if key.lower() not in _TRACKING_PARAMS
        and not key.lower().startswith(_TRACKING_PARAM_PREFIXES)
    ]
    query = "&".join(f"{k}={v}" for k, v in query_pairs)
    path = parsed.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return urlunsplit((scheme, netloc, path, query, ""))


# ---------------------------------------------------------------------------
# Text cleaning and extraction
# ---------------------------------------------------------------------------

_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS_RE = re.compile(r"[ \t ]+")


def clean_text(text: str, max_length: Optional[int] = None) -> str:
    """Strip control characters, collapse whitespace and bound length. Text is
    treated purely as data; nothing here interprets it as markup or commands."""
    if not text:
        return ""
    text = _CONTROL_RE.sub("", text)
    text = _WS_RE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = text.strip()
    max_length = max_length or _cfg_int("article_max_text_length", 40000)
    if len(text) > max_length:
        text = text[:max_length].rsplit(" ", 1)[0]
    return text


def _strip_html(html: str) -> str:
    without_scripts = _SCRIPT_STYLE_RE.sub(" ", html)
    return _TAG_RE.sub(" ", without_scripts)


def _parse_datetime(value) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    # Try a handful of common formats; feedparser already normalizes most feeds.
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S %Z",
    ):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def extract_article(html_or_text: bytes | str, url: str = "") -> dict:
    """Extract the main article body/title/author/date.

    Primary path uses trafilatura; if it yields nothing (or is unavailable), we
    fall back to a conservative HTML tag strip. Either way the returned ``text``
    is cleaned and bounded.
    """
    if isinstance(html_or_text, bytes):
        html = html_or_text.decode("utf-8", errors="replace")
    else:
        html = html_or_text or ""

    title = author = ""
    published: Optional[datetime] = None
    body = ""

    try:
        import trafilatura  # imported lazily so tests/CLI work without it loaded

        extracted = trafilatura.extract(
            html,
            url=url or None,
            output_format="json",
            with_metadata=True,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
        if extracted:
            import json as _json

            payload = _json.loads(extracted)
            body = payload.get("text") or payload.get("raw_text") or ""
            title = payload.get("title") or ""
            author = payload.get("author") or ""
            published = _parse_datetime(payload.get("date"))
    except Exception as exc:  # pragma: no cover - defensive; falls back below
        logger.debug(f"trafilatura extraction failed, using fallback: {exc}")

    if not body.strip():
        body = _strip_html(html)
        if not title:
            match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
            if match:
                title = _strip_html(match.group(1))

    return {
        "title": clean_text(title, 500),
        "author": clean_text(author, 200),
        "published_at": published,
        "text": clean_text(body),
    }


# ---------------------------------------------------------------------------
# Feeds
# ---------------------------------------------------------------------------


def parse_feed(feed_bytes: bytes | str) -> List[dict]:
    """Parse an RSS/Atom feed into a list of entry dicts."""
    import feedparser

    parsed = feedparser.parse(feed_bytes)
    if parsed.get("bozo") and not parsed.get("entries"):
        error = parsed.get("bozo_exception")
        detail = f": {error}" if error else ""
        raise SecurityError(f"malformed feed{detail}")
    entries: List[dict] = []
    for entry in parsed.get("entries", []):
        link = entry.get("link") or ""
        if not link:
            continue
        published = None
        for key in ("published_parsed", "updated_parsed"):
            struct = entry.get(key)
            if struct:
                published = datetime(*struct[:6], tzinfo=timezone.utc)
                break
        entries.append(
            {
                "title": clean_text(entry.get("title", ""), 500),
                "link": link,
                "summary": clean_text(entry.get("summary", ""), 2000),
                "author": clean_text(entry.get("author", ""), 200),
                "published_at": published,
            }
        )
    return entries


# ---------------------------------------------------------------------------
# Keywords, entities, similarity, source typing
# ---------------------------------------------------------------------------


def keywords_of(text: str, limit: int = 12) -> List[str]:
    counts: dict[str, int] = {}
    for token in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", (text or "").lower()):
        if token in _STOPWORDS:
            continue
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [word for word, _ in ranked[:limit]]


def entities_of(text: str, limit: int = 12) -> List[str]:
    """Very small deterministic named-entity heuristic (capitalized runs).

    This is only *metadata* used to help cluster and to seed visual queries; it
    is not used to prove or disprove any claim."""
    counts: dict[str, int] = {}
    for match in re.findall(
        r"\b([A-Z][a-zA-Z0-9.&'-]+(?:\s+[A-Z][a-zA-Z0-9.&'-]+){0,3})", text or ""
    ):
        candidate = match.strip()
        first = candidate.split()[0].lower()
        if first in _STOPWORDS or len(candidate) < 3:
            continue
        counts[candidate] = counts.get(candidate, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [entity for entity, _ in ranked[:limit]]


# Small, dependency-free language guesser. Good enough to *hint* the article's
# language in the UI (e.g. so the user knows to leave "Auto Detect" on, or to
# pick another language to translate); never used to force anything.
_LANG_STOPWORDS = {
    "english": {"the", "and", "of", "to", "in", "is", "that", "for", "it", "with",
                "as", "was", "on", "are", "this", "be", "at", "by", "an"},
    "spanish": {"el", "la", "de", "que", "y", "en", "los", "del", "las", "un",
                "una", "por", "con", "para", "es", "se", "su", "lo", "como"},
    "french": {"le", "la", "les", "de", "des", "et", "un", "une", "que", "qui",
               "dans", "pour", "sur", "est", "au", "aux", "ce", "pas", "plus"},
    "german": {"der", "die", "das", "und", "den", "von", "zu", "mit", "auf",
               "ist", "ein", "eine", "dem", "des", "im", "für", "nicht", "auch"},
    "portuguese": {"o", "a", "os", "as", "de", "que", "e", "do", "da", "dos",
                   "das", "um", "uma", "por", "com", "para", "se", "no", "na"},
    "italian": {"il", "la", "di", "che", "e", "un", "una", "per", "con", "non",
                "sono", "del", "della", "le", "gli", "nel", "al", "come"},
}


def guess_language(text: str) -> str:
    """Best-effort language guess as a lowercase English name, or "" if unsure.

    Uses Unicode script ranges for non-Latin languages and small stop-word sets
    for common Latin-script ones. Intentionally conservative: returns "" rather
    than guessing when the signal is weak.
    """
    sample = (text or "").strip()[:2000]
    if not sample:
        return ""
    if re.search(r"[぀-ヿ]", sample):
        return "japanese"  # kana is unique to Japanese
    if re.search(r"[가-힣]", sample):
        return "korean"
    if re.search(r"[一-鿿]", sample):
        return "chinese"
    if re.search(r"[Ѐ-ӿ]", sample):
        return "russian"
    if re.search(r"[؀-ۿ]", sample):
        return "arabic"
    if re.search(r"[฀-๿]", sample):
        return "thai"
    if re.search(r"[Ͱ-Ͽ]", sample):
        return "greek"

    words = re.findall(r"[a-zàâäáéèêëíîïóôöúûüñçã]+", sample.lower())
    if not words:
        return ""
    scores = {
        lang: sum(1 for w in words if w in stop) for lang, stop in _LANG_STOPWORDS.items()
    }
    best = max(scores, key=scores.get)
    # Require a minimal density of stop-word hits so short/ambiguous text stays "".
    if scores[best] == 0 or scores[best] / max(len(words), 1) < 0.02:
        return ""
    return best


def normalize_title(title: str) -> str:
    lowered = re.sub(r"[^a-z0-9 ]+", " ", (title or "").lower())
    tokens = [t for t in lowered.split() if t and t not in _STOPWORDS]
    return " ".join(tokens)


def _token_set(text: str) -> set:
    return {t for t in normalize_title(text).split() if t}


def title_similarity(a: str, b: str) -> float:
    set_a, set_b = _token_set(a), _token_set(b)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def classify_source(domain: str, title: str, text: str, trusted: Iterable[str]) -> Tuple[SourceType, bool]:
    """Best-effort deterministic source typing. Returns (type, is_primary)."""
    haystack = f"{title} {text[:400]}".lower()
    if any(marker in haystack for marker in _WIRE_MARKERS):
        return SourceType.wire, False
    trusted_domains = {d.lower() for d in trusted}
    is_primary = domain in trusted_domains and (
        ".gov" in domain or "press" in haystack[:200]
    )
    if domain.endswith(".gov") or domain.endswith(".gov.uk") or domain.endswith(".mil"):
        return SourceType.primary, True
    if domain in trusted_domains:
        return SourceType.news, is_primary
    return SourceType.news, is_primary


# ---------------------------------------------------------------------------
# Build article records
# ---------------------------------------------------------------------------

_MIN_BODY_LENGTH = 200  # technical minimum to have anything to work with


def build_article(
    url: str,
    body: bytes | str,
    subscription: Optional[TopicSubscription] = None,
    fallback_title: str = "",
    fallback_published: Optional[datetime] = None,
) -> ArticleRecord:
    """Turn fetched bytes into an :class:`ArticleRecord`. Raises SecurityError
    only for hard technical problems (no usable body). Editorial suitability is
    decided later by the LLM assessment, not here."""
    extracted = extract_article(body, url=url)
    text = extracted["text"]
    if len(text) < _MIN_BODY_LENGTH:
        raise SecurityError("no meaningful article body could be extracted")
    canonical = canonicalize_url(url)
    domain = domain_of(canonical)
    title = extracted["title"] or fallback_title
    published = extracted["published_at"] or fallback_published
    trusted = subscription.trusted_domains if subscription else []
    source_type, is_primary = classify_source(domain, title, text, trusted)
    return ArticleRecord(
        subscription_id=subscription.id if subscription else "",
        url=url,
        canonical_url=canonical,
        domain=domain,
        publisher=domain,
        title=title,
        author=extracted["author"],
        published_at=published,
        text=text,
        text_hash=text_hash(text),
        summary=text[:600],
        keywords=keywords_of(text),
        entities=entities_of(f"{title}. {text}"),
        source_type=source_type,
        is_authoritative_primary=is_primary,
        status=ArticleStatus.extracted,
    )


# ---------------------------------------------------------------------------
# Deduplication and clustering
# ---------------------------------------------------------------------------


def is_wire_duplicate(a: ArticleRecord, b: ArticleRecord, threshold: float = 0.9) -> bool:
    """Two articles are wire duplicates if their bodies are substantially the
    same. Identical text hashes are an exact match; otherwise compare title +
    body token overlap. Wire copies must not be counted as independent
    perspectives."""
    if a.text_hash and a.text_hash == b.text_hash:
        return True
    body_a = _token_set(a.text[:1500])
    body_b = _token_set(b.text[:1500])
    if not body_a or not body_b:
        return False
    overlap = len(body_a & body_b) / len(body_a | body_b)
    return overlap >= threshold


def dedupe_articles(articles: List[ArticleRecord]) -> List[ArticleRecord]:
    """Drop exact duplicates (same canonical URL or same text hash)."""
    seen_canonical: set = set()
    seen_hash: set = set()
    unique: List[ArticleRecord] = []
    for article in articles:
        key_url = article.canonical_url
        key_hash = article.text_hash
        if (key_url and key_url in seen_canonical) or (key_hash and key_hash in seen_hash):
            continue
        if key_url:
            seen_canonical.add(key_url)
        if key_hash:
            seen_hash.add(key_hash)
        unique.append(article)
    return unique


def cluster_articles(
    articles: List[ArticleRecord],
    subscription_id: str = "",
    title_threshold: float = 0.5,
    time_window_hours: float = 72.0,
) -> List[ArticleCluster]:
    """Group articles covering the same story.

    Deterministic signal combination (not an LLM): same canonical URL, high
    normalized-title similarity, keyword/entity overlap and publication-time
    proximity. This only *organizes* candidates; credibility is judged later.
    """
    clusters: List[ArticleCluster] = []
    cluster_reps: List[ArticleRecord] = []  # representative article per cluster

    def same_story(a: ArticleRecord, b: ArticleRecord) -> bool:
        if a.canonical_url and a.canonical_url == b.canonical_url:
            return True
        title_sim = title_similarity(a.title, b.title)
        keyword_overlap = len(set(a.keywords) & set(b.keywords))
        entity_overlap = len(set(a.entities) & set(b.entities))
        time_ok = True
        if a.published_at and b.published_at:
            delta_hours = abs((a.published_at - b.published_at).total_seconds()) / 3600.0
            time_ok = delta_hours <= time_window_hours
        strong_text = is_wire_duplicate(a, b, threshold=0.6)
        return time_ok and (
            title_sim >= title_threshold
            or strong_text
            or (title_sim >= 0.3 and (keyword_overlap >= 4 or entity_overlap >= 2))
        )

    for article in articles:
        placed = False
        for index, rep in enumerate(cluster_reps):
            if same_story(article, rep):
                cluster = clusters[index]
                cluster.article_ids.append(article.id)
                if article.domain and article.domain not in cluster.domains:
                    cluster.domains.append(article.domain)
                article.cluster_id = cluster.id
                article.status = ArticleStatus.clustered
                placed = True
                break
        if not placed:
            cluster = ArticleCluster(
                subscription_id=subscription_id or article.subscription_id,
                normalized_title=normalize_title(article.title),
                article_ids=[article.id],
                domains=[article.domain] if article.domain else [],
            )
            article.cluster_id = cluster.id
            article.status = ArticleStatus.clustered
            clusters.append(cluster)
            cluster_reps.append(article)
    return clusters


def article_is_fresh(
    article: ArticleRecord, freshness_hours: float, now: Optional[datetime] = None
) -> Optional[bool]:
    """Return whether an article is within the freshness window.

    ``None`` means the publication date is unknown (the LLM decides how to treat
    developing/undated stories via ``allow_unverified_developing_stories``)."""
    if not article.published_at:
        return None
    now = now or utcnow()
    age_hours = (now - article.published_at).total_seconds() / 3600.0
    return age_hours <= freshness_hours


def independent_domain_count(articles: List[ArticleRecord]) -> int:
    """Count distinct publisher domains, collapsing wire duplicates.

    Several URLs reproducing the same wire article count once, because they are
    not independent corroboration."""
    kept: List[ArticleRecord] = []
    domains: set = set()
    for article in articles:
        if any(is_wire_duplicate(article, other) for other in kept):
            continue
        kept.append(article)
        if article.domain:
            domains.add(article.domain)
    return len(domains)


# ---------------------------------------------------------------------------
# High-level ingestion
# ---------------------------------------------------------------------------


def ingest_url(
    url: str,
    subscription: Optional[TopicSubscription] = None,
    fetcher: Optional[Fetcher] = None,
) -> ArticleRecord:
    """Fetch and extract a single article URL."""
    fetch = fetcher or _default_fetcher
    final_url, body, _content_type = fetch(url)
    return build_article(final_url, body, subscription=subscription)


def ingest_subscription(
    subscription: TopicSubscription,
    *,
    feed_fetcher: Optional[Fetcher] = None,
    article_fetcher: Optional[Fetcher] = None,
    existing_hashes: Optional[set] = None,
    existing_canonicals: Optional[set] = None,
    max_articles: int = 40,
) -> Tuple[List[ArticleRecord], List[ArticleCluster], List[str]]:
    """Poll every feed of a subscription and return new articles + clusters.

    A failure on one feed or one article never aborts the others; the error is
    collected and returned. ``existing_*`` sets let the caller skip articles
    already stored (dedupe against the repository)."""
    feed_fetch = feed_fetcher or _feed_fetcher
    article_fetch = article_fetcher or _default_fetcher
    existing_hashes = existing_hashes or set()
    existing_canonicals = existing_canonicals or set()
    errors: List[str] = []
    articles: List[ArticleRecord] = []

    for feed_url in subscription.rss_urls:
        try:
            _final, feed_body, _ct = feed_fetch(feed_url)
            entries = parse_feed(feed_body)
        except Exception as exc:
            errors.append(f"feed {domain_of(feed_url) or feed_url}: {exc}")
            logger.warning(f"failed to poll feed: {exc}")
            continue

        for entry in entries:
            if len(articles) >= max_articles:
                break
            link = entry.get("link", "")
            canonical = canonicalize_url(link)
            if canonical in existing_canonicals:
                continue
            blocked = {d.lower() for d in subscription.blocked_domains}
            if domain_of(canonical) in blocked:
                continue
            try:
                final_url, body, _ct = article_fetch(link)
                article = build_article(
                    final_url,
                    body,
                    subscription=subscription,
                    fallback_title=entry.get("title", ""),
                    fallback_published=entry.get("published_at"),
                )
            except SecurityError as exc:
                errors.append(f"article {domain_of(link) or link}: {exc}")
                continue
            except Exception as exc:
                errors.append(f"article {domain_of(link) or link}: {exc}")
                logger.warning(f"failed to ingest article: {exc}")
                continue

            if article.text_hash in existing_hashes:
                continue
            existing_hashes.add(article.text_hash)
            existing_canonicals.add(article.canonical_url)
            articles.append(article)

    articles = dedupe_articles(articles)
    clusters = cluster_articles(articles, subscription_id=subscription.id)
    subscription.last_polled_at = utcnow()
    return articles, clusters, errors
