"""CLI backward-compatibility and Article Mode flag tests."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cli


class TestCliBackwardCompatibility(unittest.TestCase):
    def test_topic_still_works(self):
        args = cli.parse_args(["--video-subject", "spring flowers"])
        params = cli.build_video_params(args)
        self.assertEqual(params.video_subject, "spring flowers")
        # Article defaults untouched.
        self.assertEqual(params.content_mode, "topic")
        self.assertEqual(params.media_mode, "videos_only")

    def test_video_subject_required_without_article(self):
        with self.assertRaises(SystemExit):
            cli.parse_args([])


class TestCliArticleFlags(unittest.TestCase):
    def test_article_url_sets_content_mode(self):
        args = cli.parse_args(["--article-url", "https://example.com/story", "--media-mode", "images_only"])
        params = cli.build_video_params(args)
        self.assertEqual(params.content_mode, "article_url")
        self.assertEqual(params.article_url, "https://example.com/story")
        self.assertEqual(params.media_mode, "images_only")

    def test_article_id_implies_feed_mode(self):
        args = cli.parse_args(["--article-id", "article-123", "--image-source", "pixabay"])
        params = cli.build_video_params(args)
        self.assertEqual(params.content_mode, "article_feed")
        self.assertEqual(params.article_id, "article-123")
        self.assertEqual(params.image_source, "pixabay")

    def test_explicit_content_mode(self):
        args = cli.parse_args(["--content-mode", "article_feed", "--video-subject", "x"])
        self.assertEqual(args.content_mode, "article_feed")

    def test_no_subject_required_with_article_action(self):
        # Should not raise SystemExit.
        args = cli.parse_args(["--list-articles"])
        self.assertTrue(args.list_articles)

    def test_invalid_media_mode_choice_rejected(self):
        with self.assertRaises(SystemExit):
            cli.parse_args(["--video-subject", "x", "--media-mode", "bogus"])


if __name__ == "__main__":
    unittest.main()
