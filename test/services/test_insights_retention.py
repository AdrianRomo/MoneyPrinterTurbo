import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import hashtags, insights


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code != 200:
            raise AssertionError(f"http {self.status_code}")


class TestMetricSelection(unittest.TestCase):
    def test_reels_ask_for_watch_time(self):
        metric = insights.metrics_for("REELS")
        self.assertIn("ig_reels_avg_watch_time", metric)
        self.assertIn("ig_reels_video_view_total_time", metric)
        self.assertIn("reach", metric)

    def test_feed_posts_do_not(self):
        """Reels 指标用在图文帖上会让整个请求失败，必须按媒体类型分开请求。"""
        metric = insights.metrics_for("FEED")
        self.assertNotIn("ig_reels", metric)
        self.assertEqual(metric, insights.METRICS)

    def test_unknown_type_is_treated_as_feed(self):
        self.assertEqual(insights.metrics_for(""), insights.METRICS)
        self.assertEqual(insights.metrics_for(None), insights.METRICS)


class TestFetchInsights(unittest.TestCase):
    def test_reel_metrics_are_returned(self):
        payload = {"data": [
            {"name": "reach", "values": [{"value": 9}]},
            {"name": "ig_reels_avg_watch_time", "values": [{"value": 5200}]},
        ]}
        with patch("app.services.insights.requests.get",
                   return_value=FakeResponse(200, payload)) as get:
            out = insights.fetch_insights("tok", "media-1", "REELS")
        self.assertEqual(out["ig_reels_avg_watch_time"], 5200)
        self.assertIn("ig_reels_avg_watch_time", get.call_args[1]["params"]["metric"])

    def test_total_value_shape_is_understood(self):
        payload = {"data": [{"name": "shares", "total_value": {"value": 3}}]}
        with patch("app.services.insights.requests.get",
                   return_value=FakeResponse(200, payload)):
            self.assertEqual(insights.fetch_insights("tok", "m", "REELS")["shares"], 3)

    def test_a_rejected_reel_metric_falls_back_to_the_base_set(self):
        """Meta 会改名/下线 Reels 指标；不能因此把一直在收的数据也弄丢。"""
        base_payload = {"data": [{"name": "reach", "values": [{"value": 4}]}]}
        responses = [FakeResponse(400, {"error": "unsupported metric"}),
                     FakeResponse(200, base_payload)]
        with patch("app.services.insights.requests.get", side_effect=responses) as get:
            out = insights.fetch_insights("tok", "media-1", "REELS")
        self.assertEqual(out, {"reach": 4})
        self.assertEqual(get.call_count, 2)
        self.assertEqual(get.call_args_list[1][1]["params"]["metric"], insights.METRICS)

    def test_a_feed_failure_does_not_retry(self):
        with patch("app.services.insights.requests.get",
                   return_value=FakeResponse(400, {})) as get:
            self.assertEqual(insights.fetch_insights("tok", "m", "FEED"), {})
        self.assertEqual(get.call_count, 1)

    def test_media_listing_carries_the_product_type(self):
        payload = {"data": [{"id": "1", "permalink": "https://ig.com/p/AAA/",
                             "media_product_type": "REELS"}]}
        with patch("app.services.insights.requests.get",
                   return_value=FakeResponse(200, payload)) as get:
            media = insights.fetch_media("tok")
        self.assertEqual(media["https://ig.com/p/aaa"]["product_type"], "REELS")
        self.assertIn("media_product_type", get.call_args[1]["params"]["fields"])


class TestRetentionReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patcher = patch.object(hashtags, "STORAGE", self.tmp.name)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.tmp.cleanup()

    def sample(self, media_id, watch_ms, seconds, style="brand"):
        hashtags.record_sample(
            "devotional", media_id,
            {"reach": 5, "ig_reels_avg_watch_time": watch_ms},
            variant={"video_seconds": seconds, "script_style": style,
                     "subtitle_renderer": style, "subtitle_cadence": "words"},
        )

    def test_completion_needs_both_halves(self):
        self.assertAlmostEqual(
            hashtags.completion_of({"metrics": {"ig_reels_avg_watch_time": 5000},
                                    "variant": {"video_seconds": 10}}), 0.5)
        self.assertIsNone(hashtags.completion_of(
            {"metrics": {"ig_reels_avg_watch_time": 5000}, "variant": {}}))
        self.assertIsNone(hashtags.completion_of(
            {"metrics": {}, "variant": {"video_seconds": 10}}))

    def test_zero_length_video_does_not_divide_by_zero(self):
        self.assertIsNone(hashtags.completion_of(
            {"metrics": {"ig_reels_avg_watch_time": 100}, "variant": {"video_seconds": 0}}))

    def test_report_is_empty_before_any_data(self):
        self.assertEqual(hashtags.retention_report()["samples"], 0)

    def test_report_groups_by_treatment(self):
        self.sample("m1", 5000, 10.0, style="brand")
        self.sample("m2", 7000, 10.0, style="brand")
        self.sample("m3", 8000, 70.0, style="default")
        report = hashtags.retention_report()
        self.assertEqual(report["samples"], 3)
        self.assertEqual(len(report["variants"]), 2)

    def test_report_reads_completion_not_just_seconds(self):
        """8 秒对 9 秒的片子是极好，对 70 秒的片子是灾难——必须有分母。"""
        self.sample("short", 8000, 9.0, style="brand")
        self.sample("long", 8000, 70.0, style="default")
        variants = hashtags.retention_report()["variants"]
        brand = next(v for k, v in variants.items() if "brand" in k)
        old = next(v for k, v in variants.items() if "default" in k)
        self.assertEqual(brand["mean_watch_seconds"], old["mean_watch_seconds"])
        self.assertGreater(brand["mean_completion"], old["mean_completion"])

    def test_samples_without_retention_are_ignored(self):
        hashtags.record_sample("devotional", "m0", {"reach": 3})
        self.assertEqual(hashtags.retention_report()["samples"], 0)

    def test_variant_is_persisted_with_the_sample(self):
        self.sample("m1", 5000, 10.0)
        stored = hashtags._load("samples.json", [])
        self.assertEqual(stored[0]["variant"]["video_seconds"], 10.0)

    def test_a_sample_is_recorded_once(self):
        self.sample("m1", 5000, 10.0)
        self.sample("m1", 9999, 10.0)
        self.assertEqual(len(hashtags._load("samples.json", [])), 1)


class TestSelectorGuard(unittest.TestCase):
    def test_min_samples_is_above_noise(self):
        """reach 2-9、saves/shares 全为 0 时，2 个样本只是噪声。"""
        self.assertGreaterEqual(hashtags.MIN_SAMPLES, 5)


if __name__ == "__main__":
    unittest.main()
