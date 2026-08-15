"""Photo carousel API endpoints (v1).

Builds "the creativity of God in <subject>" carousels from real, correctly
licensed Wikimedia Commons photography and optionally publishes them.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field

from app.controllers.v1.base import new_router
from app.models.exception import HttpException
from app.services import carousel

router = new_router()


class CarouselRequest(BaseModel):
    subject: Optional[str] = Field(
        None, description=f"One of: {', '.join(sorted(carousel.SUBJECTS))}. Random if omitted.")
    slides: int = Field(8, ge=3, le=carousel.MAX_SLIDES,
                        description="Instagram allows at most 10 carousel items")
    publish: bool = Field(False, description="Publish via Postiz after building")


@router.post("/carousels", summary="Build a real-photography carousel")
def create_carousel(body: CarouselRequest):
    if body.subject and body.subject not in carousel.SUBJECTS:
        raise HttpException(task_id="", status_code=400,
                            message=f"subject must be one of {sorted(carousel.SUBJECTS)}")

    car = carousel.build(subject=body.subject, slides=body.slides)
    if not car:
        raise HttpException(task_id="", status_code=502,
                            message="could not build a carousel (see logs)")

    caption, set_id = carousel.build_caption(car)
    result = {
        "subject": car["subject"],
        "slides": len(car["paths"]),
        "paths": car["paths"],
        # Attribution is part of the output, not an afterthought: CC BY requires it.
        "credits": car["credits"],
        "hashtag_set": set_id,
        "caption": caption,
        "published": False,
    }

    if body.publish:
        outcome = carousel.publish(car)
        result["published"] = bool(outcome.get("success"))
        result["publish_result"] = {k: v for k, v in outcome.items() if k != "integration"}
        if not outcome.get("success"):
            logger.warning(f"carousel built but not published: {outcome.get('error')}")

    return result
