"""Verse card API endpoints (v1).

Generate image posts and stories with scripture over an AI-generated
background, and optionally hand them to Postiz.

The verse text is always fetched from a bible API by reference — the LLM only
proposes which reference to use. See app/services/verse_card.py.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field

from app.controllers.v1.base import new_router
from app.models.exception import HttpException
from app.services import verse_card

router = new_router()


class VerseCardRequest(BaseModel):
    kind: str = Field("post", description="'post' (1080x1350 feed) or 'story' (1080x1920)")
    theme: str = Field("", description="Theme to guide verse selection, e.g. 'hope', 'peace'")
    reference: Optional[str] = Field(
        None,
        description="Explicit reference (e.g. 'Philippians 4:6'). Skips LLM selection. "
        "Still fetched from the bible API — never rendered from user text.",
    )
    subject: Optional[str] = Field(None, description="Background subject override")
    publish: bool = Field(False, description="Publish via Postiz after generating")


@router.post("/verse-cards", summary="Generate a scripture image post or story")
def create_verse_card(body: VerseCardRequest):
    if body.kind not in verse_card.ASPECTS:
        raise HttpException(task_id="", status_code=400,
                            message=f"kind must be one of {sorted(verse_card.ASPECTS)}")

    if body.reference:
        verse = verse_card.fetch_verse(body.reference)
        if not verse:
            # A reference that cannot be fetched is rejected outright rather
            # than rendered on a card — see the module docstring.
            raise HttpException(task_id="", status_code=400,
                                message=f"could not fetch reference {body.reference!r}")
        bg = verse_card.generate_background(kind=body.kind, subject=body.subject)
        if bg is None:
            raise HttpException(task_id="", status_code=502, message="background generation failed")
        path = verse_card.compose_card(bg, verse, kind=body.kind)
        verse_card._remember_reference(verse.reference)
        card = {"path": path, "verse": verse,
                "caption": verse_card.build_caption(verse), "kind": body.kind}
    else:
        card = verse_card.create_card(kind=body.kind, theme=body.theme, subject=body.subject)
        if not card:
            raise HttpException(task_id="", status_code=502,
                                message="could not generate a verse card (see logs)")

    result = {
        "path": card["path"],
        "kind": card["kind"],
        "reference": card["verse"].reference,
        "translation": card["verse"].translation,
        "caption": card["caption"],
        "published": False,
    }

    if body.publish:
        outcome = verse_card.publish_card(card)
        result["published"] = bool(outcome.get("success"))
        result["publish_result"] = {k: v for k, v in outcome.items() if k != "integration"}
        if not outcome.get("success"):
            logger.warning(f"verse card generated but not published: {outcome.get('error')}")

    return result
