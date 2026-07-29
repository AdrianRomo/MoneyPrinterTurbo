"""Article Mode API tests (ASGI client, repository + LLM mocked)."""

import json
import os
import sys
import tempfile
import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import httpx
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from httpx import ASGITransport

from app.controllers.v1 import article as article_api
from app.models.article import ArticleRecord, ArticleStatus
from app.models.exception import HttpException
from app.services import article_repository
from app.services.article_repository import SqliteArticleRepository
from app.utils import utils


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(article_api.router)

    @app.exception_handler(HttpException)
    async def _http(request, exc):
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=exc.status_code,
            content=utils.get_response(exc.status_code, exc.data, exc.message),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request, exc):
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=400, content=utils.get_response(400, exc.errors(), "field required"))

    return app


class TestArticleApi(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.repo = SqliteArticleRepository(os.path.join(self.dir, "a.db"))
        article_repository.set_repository(self.repo)
        self.app = _build_app()

    def tearDown(self):
        article_repository.set_repository(None)  # reset singleton

    async def _request_async(self, method: str, path: str, **kwargs):
        async with httpx.AsyncClient(
            transport=ASGITransport(app=self.app),
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    def request(self, method: str, path: str, **kwargs):
        return asyncio.run(self._request_async(method, path, **kwargs))

    def test_subscription_crud(self):
        with patch(
            "app.services.article_ingestion._resolve_addresses",
            return_value=["93.184.216.34"],
        ):
            resp = self.request(
                "POST",
                "/api/v1/article-subscriptions",
                json={"name": "AI", "rss_urls": ["https://example.com/rss"]},
            )
        self.assertEqual(resp.status_code, 200)
        sub_id = resp.json()["data"]["id"]

        resp = self.request("GET", "/api/v1/article-subscriptions")
        self.assertEqual(len(resp.json()["data"]["subscriptions"]), 1)

        resp = self.request(
            "PUT",
            f"/api/v1/article-subscriptions/{sub_id}",
            json={"name": "AI News", "rss_urls": []},
        )
        self.assertEqual(resp.json()["data"]["name"], "AI News")

        resp = self.request("DELETE", f"/api/v1/article-subscriptions/{sub_id}")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            self.request("GET", f"/api/v1/article-subscriptions/{sub_id}").status_code,
            404,
        )

    def test_create_rejects_ssrf_rss_url(self):
        resp = self.request(
            "POST",
            "/api/v1/article-subscriptions",
            json={"name": "bad", "rss_urls": ["http://169.254.169.254/latest"]},
        )
        self.assertEqual(resp.status_code, 400)

    def test_create_validation_error(self):
        resp = self.request("POST", "/api/v1/article-subscriptions", json={"rss_urls": []})
        self.assertEqual(resp.status_code, 400)  # missing required "name"

    def test_list_and_get_articles(self):
        art = ArticleRecord(
            subscription_id="s", domain="ex.com", canonical_url="https://ex.com/a",
            title="Story", text_hash="h", status=ArticleStatus.extracted,
        )
        self.repo.save_article(art)
        resp = self.request("GET", "/api/v1/articles")
        self.assertEqual(len(resp.json()["data"]["articles"]), 1)
        resp = self.request("GET", f"/api/v1/articles/{art.id}")
        self.assertEqual(resp.json()["data"]["title"], "Story")
        self.assertEqual(self.request("GET", "/api/v1/articles/nope").status_code, 404)

    def test_generate_assisted_returns_script(self):
        art = ArticleRecord(
            subscription_id="", domain="ex.com", canonical_url="https://ex.com/a",
            title="Story", text="The event occurred. " * 30, text_hash="h",
            status=ArticleStatus.extracted,
        )
        self.repo.save_article(art)

        def mock_llm(prompt):
            if "Story Scorer" in prompt:
                return json.dumps({
                    "story_score": 0.9,
                    "confidence": 0.9,
                    "visual_potential": 0.9,
                    "risk_level": "low",
                    "recommended_action": "generate",
                })
            if "Editorial Reviewer" in prompt:
                return json.dumps({"approved": True, "confidence": 0.9, "issues": []})
            return json.dumps({"title": "Gen", "narration": "n", "scenes": [{"narration": "a", "visual_queries": ["x"]}], "social_metadata": {}})

        with patch("app.services.article_llm.llm._generate_response", side_effect=mock_llm):
            resp = self.request(
                "POST",
                f"/api/v1/articles/{art.id}/generate",
                json={"media_mode": "images_only", "render": False},
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()["data"]
        self.assertEqual(data["decision"], "generate")
        self.assertFalse(data["rendered"])
        self.assertEqual(data["script"]["title"], "Gen")

    def test_generate_invalid_media_mode(self):
        art = ArticleRecord(domain="ex.com", canonical_url="https://ex.com/a", title="S", text_hash="h")
        self.repo.save_article(art)
        resp = self.request(
            "POST",
            f"/api/v1/articles/{art.id}/generate", json={"media_mode": "bogus"}
        )
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()
