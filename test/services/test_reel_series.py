import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.services import article_worker, series


class ReelSeriesTestCase(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        config.app["reel_series_title"] = "Ordinary Grace"
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name) / "reel_series_state.json"
        self.patcher = patch.object(series, "_reel_state_path", lambda: str(self.state))
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.tmp.cleanup()
        config.app.clear()
        config.app.update(self.original_app_config)


class TestReelSeries(ReelSeriesTestCase):
    def test_disabled_until_a_title_is_configured(self):
        config.app["reel_series_title"] = ""
        self.assertIsNone(series.reel_current())
        self.assertEqual(series.reel_label(None), "")

    def test_counting_starts_at_one(self):
        self.assertEqual(series.reel_current()["number"], 1)

    def test_the_number_advances_only_on_publish(self):
        """渲染失败不能吃掉一个编号，否则这一季就缺了一集。"""
        self.assertEqual(series.reel_current()["number"], 1)
        self.assertEqual(series.reel_current()["number"], 1)  # peeking does not advance
        series.reel_advance()
        self.assertEqual(series.reel_current()["number"], 2)

    def test_advance_is_a_no_op_when_disabled(self):
        config.app["reel_series_title"] = ""
        series.reel_advance()
        config.app["reel_series_title"] = "Ordinary Grace"
        self.assertEqual(series.reel_current()["number"], 1)

    def test_label_reads_as_a_back_catalogue(self):
        series.reel_advance()
        series.reel_advance()
        series.reel_advance()
        self.assertEqual(series.reel_label(series.reel_current()), "Ordinary Grace, no. 4")

    def test_corrupt_state_restarts_rather_than_raising(self):
        self.state.write_text("not json at all", encoding="utf-8")
        self.assertEqual(series.reel_current()["number"], 1)

    def test_reel_state_is_isolated_from_the_card_series(self):
        """卡片系列的 advance() 会整体覆写状态文件，共用一个键就会被清掉。"""
        series.reel_advance()
        with patch.object(series, "_save") as card_save:
            series.advance("psalms-for-anxious-nights")
        self.assertTrue(card_save.called)
        self.assertEqual(series.reel_current()["number"], 2)


class TestCaptionLine(ReelSeriesTestCase):
    def test_series_line_leads_the_caption(self):
        out = article_worker._with_series_line("A quiet thought.\n\n#faith",
                                               series.reel_current())
        self.assertTrue(out.startswith("Ordinary Grace, no. 1"))
        self.assertIn("#faith", out)

    def test_no_line_when_the_series_is_off(self):
        self.assertEqual(article_worker._with_series_line("A quiet thought.", None),
                         "A quiet thought.")

    def test_an_empty_caption_is_left_alone(self):
        self.assertEqual(article_worker._with_series_line("", series.reel_current()), "")

    def test_the_line_is_not_added_twice(self):
        once = article_worker._with_series_line("A quiet thought.", series.reel_current())
        twice = article_worker._with_series_line(once, series.reel_current())
        self.assertEqual(once, twice)

    def test_caption_stays_within_instagram_limit(self):
        out = article_worker._with_series_line("x" * 2200, series.reel_current())
        self.assertLessEqual(len(out), 2200)


if __name__ == "__main__":
    unittest.main()
