from fastapi import Request

from app.controllers.v1.base import new_router
from app.models.schema import (
    VideoScriptRequest,
    VideoScriptResponse,
    VideoSocialMetadataRequest,
    VideoSocialMetadataResponse,
    VideoTermsRequest,
    VideoTermsResponse,
)
from app.services import article_draft, llm
from app.utils import utils
from loguru import logger

# authentication dependency
# router = new_router(dependencies=[Depends(base.verify_token)])
router = new_router()


@router.post(
    "/scripts",
    response_model=VideoScriptResponse,
    summary="Create a script for the video",
)
async def generate_video_script(request: Request, body: VideoScriptRequest):
    # Resolve grounding: explicit reference_content wins; otherwise fetch the URL
    # server-side through the SSRF-protected ingestion path.
    reference_content = (body.reference_content or "").strip()
    reference_url = (body.reference_url or "").strip()
    if not reference_content and reference_url:
        try:
            reference = article_draft.fetch_reference(reference_url)
            reference_content = reference.text
        except Exception as exc:  # noqa: BLE001 - surface as a categorized 400
            category = article_draft.classify_fetch_error(exc)
            logger.error(f"failed to fetch reference article ({category}): {exc}")
            return utils.get_response(
                400,
                {"reference_url": reference_url, "error_category": category},
                article_draft.fetch_error_message(category),
            )
        if not reference_content.strip():
            return utils.get_response(
                400,
                {"reference_url": reference_url, "error_category": "empty"},
                article_draft.fetch_error_message("empty"),
            )

    video_script = llm.generate_script(
        video_subject=body.video_subject,
        language=body.video_language,
        paragraph_number=body.paragraph_number,
        video_script_prompt=body.video_script_prompt,
        custom_system_prompt=body.custom_system_prompt,
        reference_content=reference_content,
        strict_source=body.strict_source or bool(reference_content),
    )
    response = {"video_script": video_script}
    return utils.get_response(200, response)


@router.post(
    "/terms",
    response_model=VideoTermsResponse,
    summary="Generate video terms based on the video script",
)
async def generate_video_terms(request: Request, body: VideoTermsRequest):
    video_terms = llm.generate_terms(
        video_subject=body.video_subject,
        video_script=body.video_script,
        amount=body.amount,
        match_script_order=body.match_materials_to_script,
    )
    response = {"video_terms": video_terms}
    return utils.get_response(200, response)


@router.post(
    "/social-metadata",
    response_model=VideoSocialMetadataResponse,
    summary="Generate social publishing metadata",
)
async def generate_video_social_metadata(
    request: Request, body: VideoSocialMetadataRequest
):
    metadata = llm.generate_social_metadata(
        video_subject=body.video_subject,
        video_script=body.video_script,
        language=body.language,
        platform=body.platform,
    )
    return utils.get_response(200, metadata)
