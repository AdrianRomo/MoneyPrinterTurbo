"""Typed models for Article Mode (automation-first).

Article Mode turns user-configured news topics/feeds into short videos with as
little human involvement as the operator wants. The design is automation-first:
an LLM assesses each story, assigns confidence/quality/virality/visual scores,
generates the script, and an automated reviewer checks it. Human review is
optional and gated only by configurable thresholds — not by mandatory
deterministic claim-by-claim proof.

Deterministic logic in this codebase is reserved for software correctness and
security (Pydantic validation, URL/SSRF checks, file validation, DB integrity),
never as an editorial fact-verification gate.

Nothing here performs network I/O or imports heavy media libraries, so it is
safe to import from the CLI, API layer and the polling worker.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    """Timezone-aware current time. Freshness math must never rely on naive
    local timestamps, which silently misbehave across hosts."""
    return datetime.now(timezone.utc)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if isinstance(value, datetime) else None


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:16]}"


def text_hash(text: str) -> str:
    """Stable hash of the cleaned article body. Used to detect wire-copy
    duplicates and to record provenance without persisting the full article."""
    normalized = re.sub(r"\s+", " ", (text or "")).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def domain_of(url: str) -> str:
    """Registrable-ish host of a URL, lowercased, without a leading ``www.``."""
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def safe_public_page(value: Optional[str]) -> str:
    """Keep only public http(s) page URLs; drop query strings and credentials so
    signed download URLs and API keys never reach persisted manifests."""
    if not isinstance(value, str) or not value.strip():
        return ""
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return ""
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ""
    if parsed.username or parsed.password:
        return ""
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ContentMode(str, Enum):
    """Selects the pipeline. ``topic`` preserves the original behaviour exactly."""

    topic = "topic"
    article_url = "article_url"
    article_feed = "article_feed"


class MediaMode(str, Enum):
    videos_only = "videos_only"
    images_only = "images_only"
    mixed = "mixed"


class AutomationMode(str, Enum):
    """How far the pipeline runs on its own.

    * ``assisted``   – prepare candidates + scripts, stop before render.
    * ``automated``  – generate + render, stop before publish.
    * ``autonomous`` – generate + render + publish, pausing only on repeated
      technical failure or very high AI-assessed risk.
    """

    assisted = "assisted"
    automated = "automated"
    autonomous = "autonomous"


class SourceType(str, Enum):
    primary = "primary"  # official first-party (org announcing its own news)
    news = "news"
    wire = "wire"  # syndicated wire copy redistributed by many sites
    blog = "blog"
    unknown = "unknown"


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class RecommendedAction(str, Enum):
    generate = "generate"
    review = "review"
    skip = "skip"


class VisualType(str, Enum):
    image = "image"
    video = "video"
    image_or_video = "image_or_video"


class ArticleStatus(str, Enum):
    discovered = "discovered"
    extracted = "extracted"
    rejected = "rejected"  # technical/security rejection only (not editorial)
    clustered = "clustered"
    scored = "scored"
    generating = "generating"
    generated = "generated"
    rendering = "rendering"
    rendered = "rendered"
    publishing = "publishing"
    published = "published"
    skipped = "skipped"  # below automation thresholds
    stale = "stale"
    failed = "failed"


# ---------------------------------------------------------------------------
# Subscriptions and poll runs
# ---------------------------------------------------------------------------


class TopicSubscription(BaseModel):
    """A persistent topic the worker polls on a schedule.

    ``minimum_independent_sources`` is retained for operators who *want* a
    corroboration preference, but it is a soft signal that feeds confidence
    scoring, not a hard gate: a single credible primary source can still be
    produced (see ``allow_single_source_stories`` in automation config).
    """

    id: str = Field(default_factory=lambda: new_id("sub"))
    name: str
    query: str = ""
    language: str = ""
    rss_urls: List[str] = Field(default_factory=list)
    trusted_domains: List[str] = Field(default_factory=list)
    blocked_domains: List[str] = Field(default_factory=list)
    freshness_hours: int = 72
    minimum_independent_sources: int = 1
    poll_interval_minutes: int = 60
    # Editorial preferences forwarded to the LLM prompts.
    audience: str = ""
    tone: str = ""
    platform: str = "tiktok"
    brand_preset: str = ""
    sensitive: bool = False
    # Optional per-subscription automation override; empty means use global.
    automation_mode: Optional[str] = None
    enabled: bool = True
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    last_polled_at: Optional[datetime] = None

    def is_due(self, now: Optional[datetime] = None) -> bool:
        if not self.enabled:
            return False
        if self.last_polled_at is None:
            return True
        now = now or utcnow()
        elapsed_minutes = (now - self.last_polled_at).total_seconds() / 60.0
        return elapsed_minutes >= self.poll_interval_minutes


class PollRun(BaseModel):
    id: str = Field(default_factory=lambda: new_id("poll"))
    subscription_id: str
    started_at: datetime = Field(default_factory=utcnow)
    finished_at: Optional[datetime] = None
    feeds_polled: int = 0
    articles_found: int = 0
    articles_accepted: int = 0
    articles_rejected: int = 0
    errors: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Sources, articles and clusters
# ---------------------------------------------------------------------------


class ArticleSource(BaseModel):
    """A retrieved article used as a source. Every field is safe for artifacts:
    no API keys, authorization headers or signed URLs."""

    id: str = Field(default_factory=lambda: new_id("source"))
    domain: str = ""
    publisher: str = ""
    title: str = ""
    canonical_url: str = ""
    author: str = ""
    published_at: Optional[datetime] = None
    retrieved_at: datetime = Field(default_factory=utcnow)
    text_hash: str = ""
    source_type: SourceType = SourceType.unknown
    is_authoritative_primary: bool = False

    @classmethod
    def from_article(cls, article: "ArticleRecord") -> "ArticleSource":
        return cls(
            id=f"source-{article.id.split('-', 1)[-1]}",
            domain=article.domain,
            publisher=article.publisher or article.domain,
            title=article.title,
            canonical_url=article.canonical_url,
            author=article.author,
            published_at=article.published_at,
            retrieved_at=article.retrieved_at,
            text_hash=article.text_hash,
            source_type=article.source_type,
            is_authoritative_primary=article.is_authoritative_primary,
        )


class ArticleRecord(BaseModel):
    """A fully extracted article candidate. ``text`` holds the cleaned body only
    during processing; persistence stores a bounded excerpt + ``text_hash`` and
    task artifacts never receive the full body."""

    id: str = Field(default_factory=lambda: new_id("article"))
    subscription_id: str = ""
    cluster_id: str = ""
    url: str = ""
    canonical_url: str = ""
    domain: str = ""
    publisher: str = ""
    title: str = ""
    author: str = ""
    published_at: Optional[datetime] = None
    retrieved_at: datetime = Field(default_factory=utcnow)
    text: str = ""
    text_hash: str = ""
    summary: str = ""
    keywords: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    source_type: SourceType = SourceType.unknown
    is_authoritative_primary: bool = False
    status: ArticleStatus = ArticleStatus.discovered
    rejection_reasons: List[str] = Field(default_factory=list)
    generated_task_ids: List[str] = Field(default_factory=list)

    def public_dict(self) -> dict:
        """Sanitized representation for artifacts/API (no full body text)."""
        return {
            "id": self.id,
            "subscription_id": self.subscription_id,
            "cluster_id": self.cluster_id,
            "canonical_url": self.canonical_url,
            "domain": self.domain,
            "publisher": self.publisher,
            "title": self.title,
            "author": self.author,
            "published_at": _iso(self.published_at),
            "retrieved_at": _iso(self.retrieved_at),
            "text_hash": self.text_hash,
            "summary": self.summary[:600],
            "keywords": self.keywords,
            "entities": self.entities,
            "source_type": self.source_type.value,
            "is_authoritative_primary": self.is_authoritative_primary,
            "status": self.status.value,
            "rejection_reasons": self.rejection_reasons,
            "generated_task_ids": self.generated_task_ids,
        }


class ArticleCluster(BaseModel):
    """A group of articles covering the same underlying story."""

    id: str = Field(default_factory=lambda: new_id("cluster"))
    subscription_id: str = ""
    normalized_title: str = ""
    article_ids: List[str] = Field(default_factory=list)
    domains: List[str] = Field(default_factory=list)
    summary: str = ""
    created_at: datetime = Field(default_factory=utcnow)

    @property
    def independent_domain_count(self) -> int:
        return len({d for d in self.domains if d})


# ---------------------------------------------------------------------------
# AI scoring, generated script and review
# ---------------------------------------------------------------------------


class StoryAssessment(BaseModel):
    """AI-generated scores for a story/cluster. Scores are normalized to 0..1
    on ingest regardless of whether the model returned 0..1 or 0..100."""

    story_score: float = 0.0
    confidence: float = 0.0
    source_quality: float = 0.0
    relevance: float = 0.0
    viral_potential: float = 0.0
    visual_potential: float = 0.0
    freshness: float = 0.0
    audience_fit: float = 0.0
    duplicate_story: float = 0.0
    risk_level: RiskLevel = RiskLevel.medium
    sensitive_categories: List[str] = Field(default_factory=list)
    recommended_action: RecommendedAction = RecommendedAction.review
    reasoning_summary: str = ""
    uncertainties: List[str] = Field(default_factory=list)

    def public_dict(self) -> dict:
        return self.model_dump(mode="json")


class SocialMetadata(BaseModel):
    youtube_title: str = ""
    youtube_description: str = ""
    tiktok_caption: str = ""
    instagram_caption: str = ""
    hashtags: List[str] = Field(default_factory=list)


class Scene(BaseModel):
    narration: str = ""
    visual_queries: List[str] = Field(default_factory=list)
    visual_type: VisualType = VisualType.image_or_video
    duration_weight: float = 1.0
    is_contextual_visual: bool = True


class GeneratedScript(BaseModel):
    """Structured script produced by the LLM and validated with Pydantic."""

    id: str = Field(default_factory=lambda: new_id("script"))
    cluster_id: str = ""
    primary_article_id: str = ""
    title: str = ""
    hook: str = ""
    summary: str = ""
    confidence: float = 0.0
    narration: str = ""
    scenes: List[Scene] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    source_ids: List[str] = Field(default_factory=list)
    social_metadata: SocialMetadata = Field(default_factory=SocialMetadata)
    language: str = ""
    media_mode: MediaMode = MediaMode.images_only
    sources: List[ArticleSource] = Field(default_factory=list)
    assessment: Optional["StoryAssessment"] = None
    review: Optional["ScriptReview"] = None
    created_at: datetime = Field(default_factory=utcnow)

    def narration_text(self) -> str:
        joined = "\n\n".join(
            scene.narration.strip() for scene in self.scenes if scene.narration.strip()
        )
        return joined or (self.narration or "").strip()

    def source_names(self, limit: int = 4) -> List[str]:
        names: List[str] = []
        for source in self.sources:
            label = source.publisher or source.domain
            if label and label not in names:
                names.append(label)
            if len(names) >= limit:
                break
        return names


class ScriptReview(BaseModel):
    approved: bool = False
    confidence: float = 0.0
    issues: List[str] = Field(default_factory=list)
    revised_script: Optional[dict] = None


GeneratedScript.model_rebuild()


# ---------------------------------------------------------------------------
# Automation configuration (typed view over the [article] config section)
# ---------------------------------------------------------------------------


class AutomationSettings(BaseModel):
    """Configurable thresholds that decide how far automation proceeds. All
    scores are compared on a 0..1 scale."""

    mode: AutomationMode = AutomationMode.assisted
    minimum_story_score: float = 0.6
    minimum_confidence_score: float = 0.6
    minimum_visual_score: float = 0.4
    maximum_risk_for_auto_publish: RiskLevel = RiskLevel.low
    auto_generate_enabled: bool = True
    auto_render_enabled: bool = False
    auto_publish_enabled: bool = False
    auto_rewrite_attempts: int = 1
    allow_single_source_stories: bool = True
    allow_unverified_developing_stories: bool = True
    require_review_for_sensitive_topics: bool = True
    add_illustrative_label: bool = True
    max_generations_per_day: int = 20
    max_publications_per_day: int = 10

    def resolve_mode(self, override: Optional[str]) -> AutomationMode:
        if override:
            try:
                return AutomationMode(override)
            except ValueError:
                return self.mode
        return self.mode


# ---------------------------------------------------------------------------
# Media assets
# ---------------------------------------------------------------------------


class MediaAsset(BaseModel):
    """A backward-compatible richer material record for Article Mode. The topic
    pipeline keeps using ``MaterialInfo``; ``MediaAsset`` carries the extra
    licensing/provenance/scene-ordering fields image and mixed-media article
    videos require. ``local_path`` is the normalized MP4 (or image) path handed
    to the existing combine pipeline."""

    media_type: str = "image"  # "image" | "video"
    provider: str = ""
    url: str = ""
    local_path: str = ""
    width: int = 0
    height: int = 0
    duration: float = 0.0
    asset_id: str = ""
    creator: str = ""
    metadata_text: str = ""
    license_name: str = ""
    license_url: str = ""
    attribution_text: str = ""
    source_page_url: str = ""
    search_query: str = ""
    beat_index: int = 0
    relevance_score: float = 0.0
    selection_reason: str = ""
    illustrative: bool = False

    def manifest_entry(self) -> dict:
        """Sanitized entry for media_manifest.json (no signed download URL)."""
        return {
            "media_type": self.media_type,
            "provider": self.provider,
            "asset_id": self.asset_id,
            "width": self.width,
            "height": self.height,
            "duration": round(float(self.duration), 3),
            "creator": self.creator,
            "license_name": self.license_name,
            "license_url": safe_public_page(self.license_url),
            "attribution_text": self.attribution_text,
            "source_page_url": safe_public_page(self.source_page_url),
            "search_query": self.search_query,
            "beat_index": self.beat_index,
            "relevance_score": round(float(self.relevance_score), 4),
            "selection_reason": self.selection_reason,
            "illustrative": self.illustrative,
            "local_file": self.local_path.rsplit("/", 1)[-1] if self.local_path else "",
        }


def normalize_score(value: object, default: float = 0.0) -> float:
    """Coerce an LLM score to 0..1. Accepts 0..1 or 0..100; clamps the rest."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    if number > 1.0:
        number = number / 100.0
    return max(0.0, min(1.0, number))


def scores_from_payload(payload: Dict) -> Dict[str, float]:
    """Normalize the numeric score fields of an assessment payload."""
    keys = [
        "story_score",
        "confidence",
        "source_quality",
        "relevance",
        "viral_potential",
        "visual_potential",
        "freshness",
        "audience_fit",
        "duplicate_story",
    ]
    return {key: normalize_score(payload.get(key)) for key in keys}
