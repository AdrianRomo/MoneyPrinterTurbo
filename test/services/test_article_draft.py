"""Tests for the UI-agnostic article -> grounded draft service (LLM/network mocked)."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import article_draft
from app.services import article_ingestion


def _ref(text="word " * 100, title="Outage report", url="https://news.example.com/x"):
    return article_draft.ArticleReference(
        requested_url=url,
        url=url,
        domain="news.example.com",
        title=title,
        text=text.strip(),
    )


class TestArticleReference(unittest.TestCase):
    def test_word_count_and_usable(self):
        ref = _ref(text="one two three")
        self.assertEqual(ref.word_count, 3)
        self.assertFalse(ref.is_usable)  # below MIN_ARTICLE_WORDS
        self.assertTrue(_ref().is_usable)

    def test_snippet_truncates_and_collapses_whitespace(self):
        ref = _ref(text="a\n\n  b   c " + "x" * 500)
        snip = ref.snippet(limit=10)
        self.assertLessEqual(len(snip), 11)  # +ellipsis
        self.assertTrue(snip.endswith("…"))
        self.assertNotIn("\n", snip)


class TestSuggestedTerms(unittest.TestCase):
    def test_reference_entities_uses_ingestion(self):
        with patch.object(
            article_ingestion, "entities_of", return_value=["NASA", "Mars"]
        ):
            ents = article_draft.reference_entities(_ref())
        self.assertEqual(ents, ["NASA", "Mars"])

    def test_reference_entities_empty_for_none(self):
        self.assertEqual(article_draft.reference_entities(None), [])

    def test_suggested_terms_merges_and_dedupes(self):
        draft = article_draft.ArticleDraft(
            subject="s",
            script="x",
            requirements="",
            brief={"visual_themes": ["rocket", "nasa"]},
            reference=_ref(),
        )
        with patch.object(
            article_draft, "reference_entities", return_value=["NASA", "Mars"]
        ):
            terms = draft.suggested_terms
        self.assertIn("rocket", terms)
        self.assertIn("Mars", terms)
        # "nasa" theme and "NASA" entity collapse to a single term
        self.assertEqual(len([t for t in terms if t.lower() == "nasa"]), 1)


class TestErrorTaxonomy(unittest.TestCase):
    def test_classify_ssrf_and_http_statuses(self):
        self.assertEqual(
            article_draft.classify_fetch_error(
                article_ingestion.SecurityError("blocked by policy")
            ),
            "blocked",
        )
        self.assertEqual(
            article_draft.classify_fetch_error(
                article_ingestion.SecurityError("http status 404")
            ),
            "not_found",
        )
        self.assertEqual(
            article_draft.classify_fetch_error(
                article_ingestion.SecurityError("http status 503")
            ),
            "unavailable",
        )

    def test_classify_network_and_empty_and_unknown(self):
        import requests

        self.assertEqual(
            article_draft.classify_fetch_error(requests.exceptions.Timeout()), "timeout"
        )
        self.assertEqual(
            article_draft.classify_fetch_error(requests.exceptions.ConnectionError()),
            "network",
        )
        self.assertEqual(
            article_draft.classify_fetch_error(article_draft.ArticleEmptyError()), "empty"
        )
        self.assertEqual(article_draft.classify_fetch_error(ValueError("x")), "unknown")

    def test_fetch_error_message_covers_all_categories(self):
        for category in article_draft.FETCH_ERROR_CATEGORIES:
            self.assertTrue(article_draft.fetch_error_message(category))


class TestMultiSource(unittest.TestCase):
    def setUp(self):
        article_draft.clear_reference_cache()

    def tearDown(self):
        article_draft.clear_reference_cache()

    def test_parse_source_urls_splits_dedupes_limits(self):
        text = "https://a.com/1, https://b.com/2  https://a.com/1\nhttps://c.com/3 https://d.com/4"
        urls = article_draft.parse_source_urls(text)
        self.assertEqual(
            urls, ["https://a.com/1", "https://b.com/2", "https://c.com/3"]
        )  # deduped + capped at MAX_SOURCES(3)

    def test_fetch_references_collects_failures_and_dedupes(self):
        refs_by_url = {
            "https://a.com/1": _ref(title="Outage hits city", url="https://a.com/1"),
            # near-identical title -> treated as a wire duplicate of the first
            "https://b.com/2": _ref(title="Outage hits city", url="https://b.com/2"),
            "https://c.com/3": _ref(title="Totally different topic", url="https://c.com/3"),
        }

        def fake_fetch(url, **kwargs):
            if url == "https://x.com/boom":
                raise article_ingestion.SecurityError("blocked")
            return refs_by_url[url]

        with patch.object(article_draft, "fetch_reference", side_effect=fake_fetch):
            refs, failed = article_draft.fetch_references(
                [
                    "https://a.com/1",
                    "https://b.com/2",
                    "https://c.com/3",
                    "https://x.com/boom",
                ]
            )
        self.assertEqual([r.url for r in refs], ["https://a.com/1", "https://c.com/3"])
        self.assertEqual(failed, ["https://x.com/boom"])

    def test_combine_references_marks_each_source(self):
        refs = [
            _ref(title="A", text="alpha body", url="https://a.com/1"),
            _ref(title="B", text="beta body", url="https://b.com/2"),
        ]
        refs[0].domain, refs[1].domain = "a.com", "b.com"
        combined = article_draft.combine_references(refs, requested_url="raw input")
        self.assertIn("[SOURCE 1", combined.text)
        self.assertIn("[SOURCE 2", combined.text)
        self.assertIn("alpha body", combined.text)
        self.assertIn("beta body", combined.text)
        self.assertEqual(combined.requested_url, "raw input")
        self.assertIn("a.com", combined.domain)
        self.assertIn("b.com", combined.domain)

    def test_build_combined_reference_all_failed_returns_none(self):
        with patch.object(
            article_draft,
            "fetch_reference",
            side_effect=article_ingestion.SecurityError("blocked"),
        ):
            combined, used, failed = article_draft.build_combined_reference(
                ["https://a.com/1", "https://b.com/2"]
            )
        self.assertIsNone(combined)
        self.assertEqual(used, [])
        self.assertEqual(len(failed), 2)


class TestFetchReference(unittest.TestCase):
    def setUp(self):
        article_draft.clear_reference_cache()

    def tearDown(self):
        article_draft.clear_reference_cache()

    def test_fetch_uses_ingestion_and_derives_domain(self):
        html = b"<html><head><title>T</title></head><body><p>hello world</p></body></html>"
        with patch.object(
            article_ingestion,
            "fetch_url",
            return_value=("https://site.example.org/a/b", html, "text/html"),
        ):
            ref = article_draft.fetch_reference("https://site.example.org/a/b")
        self.assertEqual(ref.domain, "site.example.org")
        self.assertIn("hello world", ref.text)

    def test_fetch_propagates_security_error(self):
        with patch.object(
            article_ingestion,
            "fetch_url",
            side_effect=article_ingestion.SecurityError("blocked"),
        ):
            with self.assertRaises(article_ingestion.SecurityError):
                article_draft.fetch_reference("http://169.254.169.254/")


class TestReferenceCache(unittest.TestCase):
    def setUp(self):
        article_draft.clear_reference_cache()

    def tearDown(self):
        article_draft.clear_reference_cache()

    def _html(self):
        return (
            b"<html><head><title>Cached</title></head><body><p>"
            + b"word " * 100
            + b"</p></body></html>"
        )

    def test_second_fetch_served_from_cache(self):
        calls = []

        def fake_fetch(url, **kwargs):
            calls.append(url)
            return ("https://ex.org/a", self._html(), "text/html")

        with patch.object(article_ingestion, "fetch_url", side_effect=fake_fetch):
            r1 = article_draft.fetch_reference("https://ex.org/a")
            r2 = article_draft.fetch_reference("https://ex.org/a")
        self.assertEqual(len(calls), 1)  # the network was only hit once
        self.assertEqual(r1.text, r2.text)

    def test_use_cache_false_bypasses(self):
        calls = []

        def fake_fetch(url, **kwargs):
            calls.append(url)
            return ("https://ex.org/a", self._html(), "text/html")

        with patch.object(article_ingestion, "fetch_url", side_effect=fake_fetch):
            article_draft.fetch_reference("https://ex.org/a")
            article_draft.fetch_reference("https://ex.org/a", use_cache=False)
        self.assertEqual(len(calls), 2)

    def test_ttl_zero_disables_cache(self):
        calls = []

        def fake_fetch(url, **kwargs):
            calls.append(url)
            return ("https://ex.org/a", self._html(), "text/html")

        with patch.object(article_draft, "_cache_ttl", return_value=0), patch.object(
            article_ingestion, "fetch_url", side_effect=fake_fetch
        ):
            article_draft.fetch_reference("https://ex.org/a")
            article_draft.fetch_reference("https://ex.org/a")
        self.assertEqual(len(calls), 2)


class TestDraftFromReference(unittest.TestCase):
    def _brief(self):
        return {
            "subject": "Six-hour outage hits 1200 users",
            "content_type": "news",
            "tone": "measured and factual",
            "audience": "general",
            "hook": "Open with the scale.",
            "key_points": ["1200 users affected", "lasted six hours"],
            "recommended_paragraphs": 3,
            "visual_themes": ["data center", "network"],
            "sensitivity": "",
        }

    def test_empty_reference_raises(self):
        with self.assertRaises(article_draft.ArticleEmptyError):
            article_draft.draft_from_reference(_ref(text="   "))

    def test_draft_grounds_strictly_and_uses_brief(self):
        captured = {}

        def fake_generate_script(**kwargs):
            captured.update(kwargs)
            return "Para one.\n\nPara two.\n\nPara three."

        with patch.object(
            article_draft.article_llm, "analyze_article_brief", return_value=self._brief()
        ), patch.object(article_draft.llm, "generate_script", side_effect=fake_generate_script):
            draft = article_draft.draft_from_reference(_ref(), language="en")

        # subject comes from the brief; script generation must be strict + grounded
        self.assertEqual(draft.subject, "Six-hour outage hits 1200 users")
        self.assertTrue(captured["strict_source"])
        self.assertEqual(captured["paragraph_number"], 3)
        self.assertIn("word", captured["reference_content"])  # grounded on article text
        self.assertIn("1200 users affected", draft.requirements)
        self.assertEqual(draft.visual_themes, ["data center", "network"])
        self.assertEqual(draft.recommended_paragraphs, 3)

    def test_draft_degrades_when_brief_fails(self):
        with patch.object(
            article_draft.article_llm,
            "analyze_article_brief",
            side_effect=RuntimeError("llm down"),
        ), patch.object(
            article_draft.llm, "generate_script", return_value="script body"
        ):
            draft = article_draft.draft_from_reference(
                _ref(title="Fallback title"), fallback_subject="fb"
            )
        # brief failed -> empty brief, no requirements, subject falls back to title
        self.assertEqual(draft.brief, {})
        self.assertEqual(draft.requirements, "")
        self.assertEqual(draft.subject, "Fallback title")

    def test_build_article_draft_fetches_then_drafts(self):
        with patch.object(
            article_draft, "fetch_reference", return_value=_ref()
        ), patch.object(
            article_draft.article_llm, "analyze_article_brief", return_value=self._brief()
        ), patch.object(
            article_draft.llm, "generate_script", return_value="body"
        ):
            draft = article_draft.build_article_draft("https://news.example.com/x")
        self.assertEqual(draft.script, "body")
        self.assertIsNotNone(draft.reference)


if __name__ == "__main__":
    unittest.main()
