"""Sanitized provenance artifacts for Article Mode.

Extends the existing task-artifact convention (JSON files written atomically
into the task directory) with Article-Mode files:

    article.json, sources.json, script_plan.json, media_manifest.json,
    assessment.json, review.json, provenance.json

None of these ever contain API keys, authorization headers or temporary signed
URLs — only public page URLs, license metadata and model output. They exist so a
generated video can be inspected after the fact; they never block generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from loguru import logger

from app.models.article import GeneratedScript, MediaAsset, StoryAssessment, safe_public_page
from app.services import task_artifacts
from app.utils import utils


def _write(task_id: str, name: str, payload) -> None:
    target = Path(utils.task_dir(task_id)) / name
    try:
        task_artifacts._write_json_atomic(target, payload)
    except Exception as exc:  # provenance must never break rendering
        logger.warning(f"failed to write article artifact {name}: {exc}")


def article_payload(script: GeneratedScript) -> dict:
    return {
        "title": script.title,
        "summary": script.summary,
        "language": script.language,
        "confidence": script.confidence,
        "cluster_id": script.cluster_id,
        "primary_article_id": script.primary_article_id,
        "uncertainties": script.uncertainties,
        "label": "Generated from available sources.",
    }


def sources_payload(script: GeneratedScript) -> list:
    return [
        {
            "id": source.id,
            "publisher": source.publisher,
            "domain": source.domain,
            "title": source.title,
            "canonical_url": safe_public_page(source.canonical_url),
            "author": source.author,
            "published_at": source.published_at.isoformat() if source.published_at else None,
            "retrieved_at": source.retrieved_at.isoformat() if source.retrieved_at else None,
            "text_hash": source.text_hash,
            "source_type": source.source_type.value,
            "is_authoritative_primary": source.is_authoritative_primary,
        }
        for source in script.sources
    ]


def script_plan_payload(script: GeneratedScript) -> dict:
    payload = script.model_dump(mode="json")
    # Drop nothing sensitive here, but strip potential download URLs defensively.
    payload.pop("sources", None)  # sources.json holds the sanitized version
    return payload


def write_all(
    task_id: str,
    *,
    script: GeneratedScript,
    assessment: Optional[StoryAssessment] = None,
    assets: Optional[List[MediaAsset]] = None,
) -> None:
    """Write the full set of sanitized Article-Mode artifacts."""
    assets = assets or []
    _write(task_id, "article.json", article_payload(script))
    _write(task_id, "sources.json", sources_payload(script))
    _write(task_id, "script_plan.json", script_plan_payload(script))
    _write(task_id, "media_manifest.json", [asset.manifest_entry() for asset in assets])
    if assessment is not None:
        _write(task_id, "assessment.json", assessment.public_dict())
    if script.review is not None:
        _write(task_id, "review.json", script.review.model_dump(mode="json"))
    _write(
        task_id,
        "provenance.json",
        {
            "label": "Generated from the listed sources.",
            "confidence": script.confidence,
            "source_names": script.source_names(),
            "uncertainties": script.uncertainties,
            "media_attribution": [a.attribution_text for a in assets if a.attribution_text],
            "synthetic_media": True,
        },
    )
