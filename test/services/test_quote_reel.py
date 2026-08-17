import os
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.models.schema import MaterialInfo, VideoParams
from app.services import quote_reel, typography


class TestQuoteReel(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_resolve_quote_prefers_clean_user_script(self):
        params = VideoParams(
            video_subject="Belleza ordinaria",
            video_script="Dios no se impone.\nSe revela en la belleza.",
            content_mode=quote_reel.CONTENT_MODE,
        )

        with patch.object(quote_reel.llm, "_generate_response") as generate:
            self.assertEqual(
                quote_reel.resolve_quote(params),
                "Dios no se impone. Se revela en la belleza.",
            )

        generate.assert_not_called()

    def test_resolve_quote_rejects_shouty_object_personification(self):
        params = VideoParams(
            video_subject="FEEL CALM AS FAUCET SINGS",
            video_script="FEEL CALM AS FAUCET SINGS",
            content_mode=quote_reel.CONTENT_MODE,
        )

        with patch.object(
            quote_reel.llm,
            "_generate_response",
            return_value="FEEL CALM AS FAUCET SINGS",
        ):
            quote = quote_reel.resolve_quote(params)

        self.assertEqual(quote, "God also speaks in whatever makes you stop.")

    def test_fallback_quote_follows_the_configured_language(self):
        with patch.dict(
            quote_reel.config.app,
            {"quote_reel_default_language": "Spanish"},
            clear=False,
        ):
            self.assertEqual(
                quote_reel._fallback_quote(),
                "Dios tambien habla en lo que te hace detenerte.",
            )

    def test_caption_is_short_saveable_and_configured_hashtags(self):
        with (
            patch.dict(
                quote_reel.config.app,
                {"quote_reel_caption_hashtags": ["fe", "#belleza"]},
                clear=False,
            ),
            patch("app.services.hashtags.choose_set", return_value=""),
            patch("app.services.hashtags.tags_for", return_value=[]),
        ):
            caption = quote_reel.build_caption("La belleza nos pone al limite de lo eterno.")

        self.assertIn("Save this one to come back to slowly.", caption)
        self.assertIn("#fe #belleza", caption)
        self.assertNotIn("FOLLOW", caption.upper())

    def test_caption_call_to_action_follows_the_configured_language(self):
        with (
            patch.dict(
                quote_reel.config.app,
                {"quote_reel_default_language": "Spanish"},
                clear=False,
            ),
            patch("app.services.hashtags.choose_set", return_value=""),
            patch("app.services.hashtags.tags_for", return_value=[]),
        ):
            caption = quote_reel.build_caption("La belleza abre una ventana.")

        # The whole caption moves with the language key, hashtags included —
        # shipping a Spanish quote under #bibleverse is what this prevents.
        self.assertIn("Guardalo para volver a mirar con calma.", caption)
        self.assertIn("#fe", caption)

    def test_caption_drops_the_overlay_accent_markers(self):
        with (
            patch("app.services.hashtags.choose_set", return_value=""),
            patch("app.services.hashtags.tags_for", return_value=[]),
        ):
            caption = quote_reel.build_caption("Beauty opens a window *onto the eternal*.")

        self.assertIn("Beauty opens a window onto the eternal.", caption)
        self.assertNotIn("*", caption)

    def test_caption_variant_uses_hashtag_set_for_performance_linkage(self):
        with (
            patch("app.services.hashtags.choose_set", return_value="peace") as choose,
            patch("app.services.hashtags.tags_for", return_value=["#peaceofgod"]),
        ):
            variant = quote_reel.build_caption_variant(
                "La belleza nos pone al limite de lo eterno.",
            )

        choose.assert_called_once_with(None)
        self.assertEqual(variant["hashtag_set"], "peace")
        self.assertEqual(variant["hashtags"], ["#peaceofgod"])
        self.assertIn("#peaceofgod", variant["caption"])

    def test_select_media_assets_prefers_uploaded_materials(self):
        params = VideoParams(
            video_subject="Belleza",
            content_mode=quote_reel.CONTENT_MODE,
            video_materials=[MaterialInfo(provider="local", url="clip.mp4")],
        )
        material = MaterialInfo(provider="local", url="/tmp/raw-clip.mp4")

        with (
            patch.object(quote_reel.video, "preprocess_video", return_value=[material]) as preprocess,
            patch.object(quote_reel, "_library_assets") as library_assets,
        ):
            assets = quote_reel.select_media_assets(params)

        preprocess.assert_called_once()
        library_assets.assert_not_called()
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0].kind, "video")
        self.assertEqual(assets[0].label, "raw-clip.mp4")

    def test_library_assets_read_sidecar_metadata_and_reference_flags(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            clip = Path(temp_dir) / "postiz-reference.mp4"
            clip.write_bytes(b"not a real video")
            sidecar = Path(temp_dir) / "postiz-reference.json"
            sidecar.write_text(
                json.dumps(
                    {
                        "license": "owned",
                        "reference_only": True,
                        "raw_text_free": False,
                        "contains_people": True,
                        "has_talent_released": False,
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(
                quote_reel.config.app,
                {"quote_reel_media_dir": temp_dir},
                clear=False,
            ):
                assets = quote_reel._library_assets()

        self.assertEqual(len(assets), 1)
        self.assertTrue(assets[0].source_info["reference_only"])
        self.assertFalse(assets[0].source_info["raw_text_free"])
        self.assertTrue(assets[0].source_info["contains_people"])

    def test_storyblocks_source_downloads_assets_for_quote_reel(self):
        params = VideoParams(
            video_subject="Church light",
            content_mode=quote_reel.CONTENT_MODE,
        )
        item = MaterialInfo(
            provider="storyblocks",
            url="https://cdn.storyblocks.example/a.mp4",
            duration=6,
            source_info={
                "provider": "storyblocks",
                "asset_id": "sb-1",
                "has_talent_released": True,
                "has_property_released": True,
            },
        )

        with (
            patch.dict(
                quote_reel.config.app,
                {"quote_reel_media_source": "storyblocks"},
                clear=False,
            ),
            patch.object(quote_reel.material, "storyblocks_is_configured", return_value=True),
            patch.object(quote_reel.material, "search_videos_storyblocks", return_value=[item]) as search,
            patch.object(quote_reel.material, "save_video", return_value="/tmp/sb.mp4"),
        ):
            assets = quote_reel.select_media_assets(
                params,
                task_id="quote-storyblocks",
                duration=5,
            )

        search.assert_called_once()
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0].provider, "storyblocks")
        self.assertTrue(assets[0].source_info["raw_text_free"])

    def test_pexels_source_downloads_assets_for_quote_reel(self):
        params = VideoParams(
            video_subject="Quiet light",
            content_mode=quote_reel.CONTENT_MODE,
        )
        item = MaterialInfo(
            provider="pexels",
            url="https://videos.pexels.example/quiet.mp4",
            duration=6,
            source_info={
                "provider": "pexels",
                "asset_id": "px-1",
                "source_page": "https://www.pexels.com/video/quiet-1/",
            },
        )

        with (
            patch.dict(
                quote_reel.config.app,
                {
                    "quote_reel_media_source": "pexels",
                    "quote_reel_assume_stock_text_free": True,
                },
                clear=False,
            ),
            patch.object(
                quote_reel.material,
                "search_videos_pexels",
                return_value=[item],
            ) as search,
            patch.object(quote_reel.material, "save_video", return_value="/tmp/px.mp4"),
        ):
            assets = quote_reel.select_media_assets(
                params,
                task_id="quote-pexels",
                duration=5,
            )

        search.assert_called_once()
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0].provider, "pexels")
        self.assertTrue(assets[0].source_info["raw_text_free"])

    def test_stock_source_skips_release_risk_assets(self):
        params = VideoParams(
            video_subject="Quiet light",
            content_mode=quote_reel.CONTENT_MODE,
        )
        risky = MaterialInfo(
            provider="pexels",
            url="https://videos.pexels.example/person.mp4",
            duration=6,
            source_info={
                "provider": "pexels",
                "asset_id": "risk-1",
                "metadata_text": "woman reading book on couch",
            },
        )
        safe = MaterialInfo(
            provider="pexels",
            url="https://videos.pexels.example/clouds.mp4",
            duration=6,
            source_info={
                "provider": "pexels",
                "asset_id": "safe-1",
                "metadata_text": "clouds at sunset",
            },
        )

        with (
            patch.dict(
                quote_reel.config.app,
                {
                    "quote_reel_media_source": "pexels",
                    "quote_reel_skip_stock_review_risk": True,
                    # Opt-in: stock metadata never carries a release, so leaving
                    # this off (the default) keeps people footage usable.
                    "quote_reel_require_talent_release": True,
                },
                clear=False,
            ),
            patch.object(
                quote_reel.material,
                "search_videos_pexels",
                return_value=[risky, safe],
            ),
            patch.object(
                quote_reel.material,
                "save_video",
                side_effect=["/tmp/person.mp4", "/tmp/clouds.mp4"],
            ),
        ):
            assets = quote_reel.select_media_assets(
                params,
                task_id="quote-pexels-risk",
                duration=5,
            )

        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0].path, "/tmp/clouds.mp4")
        self.assertEqual(assets[0].source_info["asset_id"], "safe-1")

    def test_build_background_clip_strips_source_audio(self):
        asset = quote_reel.QuoteReelAsset(
            path="/tmp/raw.mp4",
            kind="video",
            provider="curated_library",
            label="raw.mp4",
            source_info={"provider": "curated_library"},
        )

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            patch.object(quote_reel.utils, "task_dir", return_value=temp_dir),
            patch.object(quote_reel.video, "video_to_normalized_clip") as normalize,
            patch.object(quote_reel.video, "combine_article_clips") as combine,
        ):
            background, normalized = quote_reel.build_background_clip(
                "task-1",
                [asset],
                duration=12.0,
                video_aspect=quote_reel.VideoAspect.portrait,
                threads=2,
            )

        normalize.assert_called_once()
        self.assertFalse(normalize.call_args.kwargs["preserve_audio"])
        combine.assert_called_once()
        self.assertEqual(background, os.path.join(temp_dir, "quote-background.mp4"))
        self.assertEqual(normalized, [os.path.join(temp_dir, "quote-bg-1.mp4")])

    def test_quality_check_passes_portrait_render(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as fp:
            fp.write(b"0" * 2048)
            final_path = fp.name
        self.addCleanup(lambda: os.path.exists(final_path) and os.remove(final_path))
        asset = quote_reel.QuoteReelAsset(
            path="/tmp/raw.mp4",
            kind="video",
            provider="local",
            label="raw.mp4",
            source_info={},
        )

        with patch.object(
            quote_reel,
            "_inspect_video",
            return_value={"duration": 15.0, "width": 1080, "height": 1920},
        ), patch.object(quote_reel, "_visual_metrics", return_value={"sampled": 0}):
            qc = quote_reel.quality_check(
                final_path,
                "La belleza abre una ventana hacia lo eterno.",
                [asset],
            )

        self.assertTrue(qc["passed"])
        self.assertTrue(qc["publishable"])
        self.assertEqual(qc["issues"], [])

    def test_quality_check_requires_review_for_reference_and_embedded_text(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as fp:
            fp.write(b"0" * 2048)
            final_path = fp.name
        self.addCleanup(lambda: os.path.exists(final_path) and os.remove(final_path))
        asset = quote_reel.QuoteReelAsset(
            path="/tmp/postiz-reference.mp4",
            kind="video",
            provider="curated_library",
            label="postiz-reference.mp4",
            source_info={
                "provider": "curated_library",
                "reference_only": True,
                "raw_text_free": False,
            },
        )

        with (
            patch.object(
                quote_reel,
                "_inspect_video",
                return_value={"duration": 15.0, "width": 1080, "height": 1920},
            ),
            patch.object(
                quote_reel,
                "_visual_metrics",
                return_value={"sampled": 1, "mean_contrast": 30.0},
            ),
            patch.object(
                quote_reel,
                "_embedded_text_scan",
                return_value={
                    "available": True,
                    "detected": True,
                    "sampled": 1,
                    "text": "old quote",
                },
            ),
        ):
            qc = quote_reel.quality_check(
                final_path,
                "La belleza abre una ventana hacia lo eterno.",
                [asset],
                background_path="/tmp/background.mp4",
            )

        self.assertTrue(qc["passed"])
        self.assertFalse(qc["publishable"])
        self.assertTrue(qc["review_required"])
        self.assertTrue(any("reference-only" in reason for reason in qc["review_reasons"]))
        self.assertTrue(any("embedded text" in reason for reason in qc["review_reasons"]))

    def test_enqueue_review_writes_deduplicated_queue_item(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            queue_path = os.path.join(temp_dir, "queue.json")
            asset = quote_reel.QuoteReelAsset(
                path="/tmp/raw.mp4",
                kind="video",
                provider="curated_library",
                label="raw.mp4",
                source_info={"provider": "curated_library"},
            )
            with patch.object(quote_reel, "_review_queue_path", return_value=queue_path):
                first = quote_reel.enqueue_review(
                    "task-1",
                    final_path="/tmp/final.mp4",
                    quote="quote",
                    caption="caption",
                    qc={"review_reasons": ["reason one"]},
                    assets=[asset],
                )
                quote_reel.enqueue_review(
                    "task-1",
                    final_path="/tmp/final-2.mp4",
                    quote="quote 2",
                    caption="caption 2",
                    qc={"review_reasons": ["reason two"]},
                    assets=[asset],
                )

            data = json.loads(Path(queue_path).read_text(encoding="utf-8"))

        self.assertEqual(first, queue_path)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["final_path"], "/tmp/final-2.mp4")

    def test_schedule_skips_when_quote_reel_auto_schedule_disabled(self):
        with patch.dict(
            quote_reel.config.app,
            {"quote_reel_auto_schedule_enabled": False},
            clear=False,
        ):
            result = quote_reel._schedule_if_enabled(
                "/tmp/final.mp4",
                "caption",
                "quote",
                {"passed": True, "publishable": True, "duration": 15.0},
            )

        self.assertTrue(result["skipped"])
        self.assertIn("disabled", result["error"])

    def test_schedule_skips_when_review_required(self):
        result = quote_reel._schedule_if_enabled(
            "/tmp/final.mp4",
            "caption",
            "quote",
            {"passed": True, "publishable": False, "duration": 15.0},
        )

        self.assertTrue(result["skipped"])
        self.assertIn("review required", result["error"])

    def test_schedule_records_postiz_set_id_and_variant(self):
        asset = quote_reel.QuoteReelAsset(
            path="/tmp/raw.mp4",
            kind="video",
            provider="curated_library",
            label="raw.mp4",
            source_info={"provider": "curated_library"},
        )
        fake_service = SimpleNamespace(
            enabled=True,
            auto_schedule_enabled=True,
            is_auto_schedule_configured=lambda: True,
        )

        with (
            patch.dict(
                quote_reel.config.app,
                {"quote_reel_auto_schedule_enabled": True},
                clear=False,
            ),
            patch("app.services.postiz.postiz_service", fake_service),
            patch(
                "app.services.postiz.schedule_video",
                return_value={"success": True, "post_id": "post-1"},
            ) as schedule,
            patch("app.services.series.reel_current", return_value=None),
            patch("app.services.series.reel_advance") as advance,
            patch("app.services.hashtags.mark_used") as mark_used,
        ):
            result = quote_reel._schedule_if_enabled(
                "/tmp/final.mp4",
                "caption",
                "quote",
                {"passed": True, "publishable": True, "duration": 12.0},
                caption_meta={
                    "hashtag_set": "peace",
                    "caption_style": "saveable_contemplative",
                },
                assets=[asset],
            )

        self.assertTrue(result["success"])
        self.assertEqual(schedule.call_args.kwargs["set_id"], "peace")
        self.assertEqual(schedule.call_args.kwargs["variant"]["video_seconds"], 12.0)
        self.assertEqual(
            schedule.call_args.kwargs["variant"]["subtitle_renderer"],
            "centered_serif_quote",
        )
        advance.assert_called_once()
        mark_used.assert_called_once_with("peace")


class TestQuoteReelTypography(unittest.TestCase):
    """The overlay: one accented clause, short lines, and readable on pale footage."""

    def setUp(self):
        self.original_app_config = dict(config.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_parse_accent_splits_the_marked_clause(self):
        self.assertEqual(
            typography.parse_accent("Beauty opens a window *onto the eternal*."),
            [("Beauty opens a window ", False), ("onto the eternal", True), (".", False)],
        )

    def test_unmarked_quote_is_one_plain_run(self):
        self.assertEqual(
            typography.parse_accent("no accent at all"),
            [("no accent at all", False)],
        )

    def test_tokens_keep_punctuation_attached_to_the_accent(self):
        # Splitting per run tore "*eternal*." into "eternal" and ".", which the
        # renderer then drew a whole word-space apart.
        tokens = typography.tokens(
            typography.parse_accent("a window *onto the eternal*.")
        )
        self.assertEqual(tokens[-1], ("eternal.", True))
        self.assertNotIn(".", [word for word, _ in tokens])

    def test_lines_stay_short_enough_to_read_as_a_quote(self):
        from PIL import Image, ImageDraw

        draw = ImageDraw.Draw(Image.new("RGBA", (1080, 1920)))
        _, lines = quote_reel._layout_quote(
            draw,
            "In the hush of dawn I feel *gentle grace* whispering through my heart",
            1080,
        )
        self.assertLessEqual(len(lines), quote_reel.MAX_QUOTE_LINES)
        for line in lines:
            self.assertLessEqual(len(line), quote_reel.MAX_WORDS_PER_LINE)

    def test_scrim_scales_with_how_pale_the_backdrop_is(self):
        # Dark footage is left alone; a bright sky gets washed down.
        self.assertEqual(quote_reel._scrim_strength(80.0), 0.0)
        self.assertGreater(quote_reel._scrim_strength(200.0), 0.2)

    def test_scrim_never_exceeds_the_configured_ceiling(self):
        with patch.dict(
            quote_reel.config.app, {"quote_reel_scrim_max": 0.3}, clear=False
        ):
            self.assertLessEqual(quote_reel._scrim_strength(255.0), 0.3)

    def test_legibility_flags_a_backdrop_the_scrim_cannot_rescue(self):
        with (
            patch.dict(
                quote_reel.config.app,
                {"quote_reel_scrim_max": 0.0, "quote_reel_max_text_band_luma": 150.0},
                clear=False,
            ),
            patch.object(quote_reel, "_band_luma", return_value=220.0),
        ):
            result = quote_reel.legibility_check("/tmp/bg.mp4", 1920)

        self.assertTrue(result["measured"])
        self.assertFalse(result["readable"])

    def test_comfyui_source_never_repeats_a_cached_frame(self):
        # generate_frame caches by subject, so distinct terms can resolve to one
        # file — which cut from a shot straight back to that same shot.
        from types import SimpleNamespace

        frame = SimpleNamespace(url="/tmp/brand-a.jpg", metadata_text="a path at dusk")
        other = SimpleNamespace(url="/tmp/brand-b.jpg", metadata_text="mist over a meadow")
        params = VideoParams(
            video_subject="quiet beauty", content_mode=quote_reel.CONTENT_MODE
        )

        with (
            patch("app.services.brand_footage.search_images_comfyui") as generate,
            patch.object(quote_reel.os.path, "exists", return_value=True),
        ):
            generate.side_effect = [[frame], [other], [frame], [frame], [frame], [frame]]
            assets = quote_reel._comfyui_assets(params, count=3)

        self.assertEqual([a.path for a in assets], ["/tmp/brand-a.jpg", "/tmp/brand-b.jpg"])
        self.assertTrue(all(a.provider == "comfyui" for a in assets))

    def test_music_bed_is_skipped_when_disabled(self):
        with patch.dict(
            quote_reel.config.app, {"quote_reel_music_enabled": False}, clear=False
        ):
            self.assertIsNone(quote_reel._music_bed(15.0))

    def test_music_bed_is_skipped_when_the_library_is_empty(self):
        with patch.object(quote_reel.video, "get_bgm_file", return_value=""):
            self.assertIsNone(quote_reel._music_bed(15.0))


if __name__ == "__main__":
    unittest.main()


class TestQuoteCasing(unittest.TestCase):
    """Models return lowercase openings often enough that one shipped."""

    def test_lowercase_opening_is_capitalised(self):
        self.assertEqual(
            quote_reel.clean_quote("the breath that settles."),
            "The breath that settles.",
        )

    def test_capitalisation_skips_the_accent_marker(self):
        self.assertEqual(
            quote_reel.clean_quote("*the breath* settles."),
            "*The breath* settles.",
        )

    def test_an_existing_capital_is_left_alone(self):
        self.assertEqual(quote_reel.clean_quote("Already capital."), "Already capital.")

    def test_a_quote_opening_on_a_digit_is_untouched(self):
        self.assertEqual(quote_reel.clean_quote("123 numbers first"), "123 numbers first")
