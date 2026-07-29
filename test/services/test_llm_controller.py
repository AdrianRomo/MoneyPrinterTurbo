"""Tests for the /v1/scripts grounding behaviour (network + LLM mocked)."""

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.controllers.v1 import llm as llm_controller
from app.models.schema import VideoScriptRequest
from app.services import article_draft


def _call(body):
    return asyncio.run(llm_controller.generate_video_script(None, body))


class TestScriptEndpointGrounding(unittest.TestCase):
    def test_reference_url_is_fetched_and_grounds_strictly(self):
        body = VideoScriptRequest(
            video_subject="Outage",
            reference_url="https://news.example.com/x",
        )
        ref = SimpleNamespace(text="Real body about the outage.")
        with patch.object(
            article_draft, "fetch_reference", return_value=ref
        ), patch.object(
            llm_controller.llm, "generate_script", return_value="grounded"
        ) as generate:
            resp = _call(body)

        self.assertEqual(resp["data"]["video_script"], "grounded")
        kwargs = generate.call_args.kwargs
        self.assertEqual(kwargs["reference_content"], "Real body about the outage.")
        self.assertTrue(kwargs["strict_source"])

    def test_explicit_reference_content_skips_fetch(self):
        body = VideoScriptRequest(
            video_subject="Outage", reference_content="Pasted article text."
        )
        with patch.object(article_draft, "fetch_reference") as fetch, patch.object(
            llm_controller.llm, "generate_script", return_value="ok"
        ) as generate:
            _call(body)
        fetch.assert_not_called()
        self.assertEqual(
            generate.call_args.kwargs["reference_content"], "Pasted article text."
        )

    def test_fetch_failure_returns_400(self):
        body = VideoScriptRequest(
            video_subject="Outage", reference_url="http://169.254.169.254/"
        )
        with patch.object(
            article_draft,
            "fetch_reference",
            side_effect=article_draft.article_ingestion.SecurityError("blocked"),
        ), patch.object(llm_controller.llm, "generate_script") as generate:
            resp = _call(body)
        self.assertEqual(resp["status"], 400)
        # categorized error surfaced to the caller
        self.assertEqual(resp["data"]["error_category"], "blocked")
        generate.assert_not_called()

    def test_no_reference_generates_normally(self):
        body = VideoScriptRequest(video_subject="Coffee")
        with patch.object(
            llm_controller.llm, "generate_script", return_value="plain"
        ) as generate:
            resp = _call(body)
        self.assertEqual(resp["data"]["video_script"], "plain")
        self.assertEqual(generate.call_args.kwargs["reference_content"], "")
        self.assertFalse(generate.call_args.kwargs["strict_source"])


if __name__ == "__main__":
    unittest.main()
