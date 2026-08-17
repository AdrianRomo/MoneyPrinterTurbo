import threading
import tomllib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from app.config import config
from app.models.llm_provider import LLM_PROVIDER_REGISTRY, get_llm_provider


class TestConfigPersistence:
    @staticmethod
    def _load_example_config():
        config_path = Path(__file__).resolve().parents[2] / "config.example.toml"
        return tomllib.loads(config_path.read_text(encoding="utf-8"))

    def test_example_config_documents_runtime_settings(self):
        """示例配置应展示用户需要手工维护的服务、素材和高级运行参数。"""
        example_config = self._load_example_config()
        app_config = example_config["app"]

        assert example_config["listen_host"] == "0.0.0.0"
        assert example_config["listen_port"] == 8080
        assert example_config["log_level"] == "DEBUG"
        assert app_config["video_source"] in {
            "pexels",
            "pixabay",
            "coverr",
            "storyblocks",
            "local",
        }
        assert "match_materials_to_script" in app_config
        assert example_config["whisper"]["device"] == "cpu"

    def test_example_config_covers_llm_provider_registry(self):
        """Registry 中可配置的 Provider 字段必须能在示例文件中被发现。"""
        app_config = self._load_example_config()["app"]

        for provider in LLM_PROVIDER_REGISTRY:
            if provider.show_api_key:
                assert provider.config_key("api_key") in app_config
            if provider.show_base_url:
                assert provider.config_key("base_url") in app_config
            if provider.requires_model_name:
                assert provider.config_key("model_name") in app_config
            for field in provider.extra_fields:
                assert provider.config_key(field.config_suffix) in app_config

    def test_kimi_uses_current_default_model(self):
        """Kimi 未配置模型覆盖值时，应使用当前发布版本的默认模型。"""
        provider = get_llm_provider("moonshot")

        assert provider is not None
        assert provider.resolve_model_name("") == "kimi-k3"

    def test_upload_post_settings_belong_to_app_section(self):
        """发布配置必须位于 app 节点，确保示例文件与运行时读取路径一致。"""
        example_config = self._load_example_config()
        upload_post_keys = {
            "upload_post_enabled",
            "upload_post_api_key",
            "upload_post_username",
            "upload_post_platforms",
            "upload_post_auto_upload",
            "upload_post_youtube_privacy_status",
            "upload_post_max_pending_tasks",
        }

        assert upload_post_keys <= example_config["app"].keys()
        assert upload_post_keys.isdisjoint(example_config.get("ui", {}).keys())

    def test_postiz_settings_belong_to_app_section(self):
        """Postiz 调度配置必须位于 app 节点，避免运行时读取不到。"""
        example_config = self._load_example_config()
        postiz_keys = {
            "postiz_enabled",
            "postiz_base_url",
            "postiz_api_key",
            "postiz_integration_id",
            "postiz_provider_type",
            "postiz_auto_schedule_enabled",
            "content_timezone",
            "content_utc_offset_hours",
            "postiz_daily_quota_post",
            "postiz_daily_quota_carousel",
            "postiz_daily_quota_reel",
            "postiz_daily_quota_story",
            "postiz_window_post",
            "postiz_window_carousel",
            "postiz_window_reel",
            "postiz_window_story",
            "content_scheduler_schedule_days_ahead",
            "content_scheduler_carousel_attempts",
            "content_scheduler_carousel_subjects",
            "postiz_schedule_interval_hours",
            "postiz_schedule_jitter_minutes",
            "postiz_daily_post_cap",
            "postiz_post_type",
            "quote_reel_auto_schedule_enabled",
            "quote_reel_media_dir",
            "quote_reel_media_source",
            "quote_reel_stock_provider",
            "quote_reel_target_seconds",
            "quote_reel_default_language",
            "quote_reel_caption_hashtags",
            "quote_reel_hashtag_set",
            "quote_reel_search_terms",
            "quote_reel_assume_curated_text_free",
            "quote_reel_assume_stock_text_free",
            "quote_reel_skip_stock_review_risk",
            "quote_reel_min_visual_contrast",
            "quote_reel_storyblocks_clip_seconds",
            "quote_reel_stock_clip_seconds",
        }

        assert postiz_keys <= example_config["app"].keys()
        assert postiz_keys.isdisjoint(example_config.get("ui", {}).keys())

    def test_storyblocks_settings_belong_to_app_section(self):
        """Storyblocks API settings must be configured under app, not UI."""
        example_config = self._load_example_config()
        storyblocks_keys = {
            "storyblocks_public_key",
            "storyblocks_private_key",
            "storyblocks_project_id",
            "storyblocks_user_id",
            "storyblocks_quality",
            "storyblocks_download_quality",
            "storyblocks_results_per_page",
            "storyblocks_max_duration",
            "storyblocks_required_keywords",
            "storyblocks_filtered_keywords",
            "storyblocks_require_talent_release",
            "storyblocks_require_property_release",
        }

        assert storyblocks_keys <= example_config["app"].keys()
        assert storyblocks_keys.isdisjoint(example_config.get("ui", {}).keys())

    def test_save_config_uses_parseable_atomic_output(self):
        """
        配置保存先写临时文件再原子替换。测试同时确认输出仍是合法 TOML，
        且成功保存后不会在配置目录遗留临时文件。
        """
        original_cfg = dict(config._cfg)
        original_app = dict(config.app)
        try:
            with TemporaryDirectory() as temp_dir:
                config_path = Path(temp_dir) / "config.toml"
                config.app["atomic_save_test"] = "ok"
                with (
                    patch.object(config, "root_dir", temp_dir),
                    patch.object(config, "config_file", str(config_path)),
                ):
                    config.save_config()

                saved_config = tomllib.loads(config_path.read_text(encoding="utf-8"))
                assert saved_config["app"]["atomic_save_test"] == "ok"
                assert list(Path(temp_dir).glob(".config-*.toml.tmp")) == []
        finally:
            config.app.clear()
            config.app.update(original_app)
            config._cfg.clear()
            config._cfg.update(original_cfg)

    def test_runtime_config_lock_blocks_concurrent_config_writes(self):
        """长任务持有运行锁时，其它会话不能在任务中途改写全局配置。"""
        write_started = threading.Event()
        write_finished = threading.Event()

        def update_config():
            write_started.set()
            config.app["runtime_lock_test"] = "updated"
            write_finished.set()

        config.app.pop("runtime_lock_test", None)
        with config.runtime_config_lock():
            worker = threading.Thread(target=update_config)
            worker.start()
            assert write_started.wait(timeout=1)
            assert not write_finished.wait(timeout=0.05)

        worker.join(timeout=1)
        assert write_finished.is_set()
        config.app.pop("runtime_lock_test", None)

    def test_runtime_config_lock_allows_idempotent_page_writes(self):
        """生成期间刷新页面时，相同控件值的回写不能阻塞整页渲染。"""
        key = "runtime_lock_idempotent_test"
        config.app[key] = "unchanged"
        write_finished = threading.Event()

        def write_same_value():
            config.app[key] = "unchanged"
            assert config.app.setdefault(key, "other") == "unchanged"
            config.app.update({key: "unchanged"})
            assert config.app.pop("runtime_lock_missing_key", None) is None
            write_finished.set()

        with config.runtime_config_lock():
            worker = threading.Thread(target=write_same_value)
            worker.start()
            assert write_finished.wait(timeout=0.2)

        worker.join(timeout=1)
        assert config.app[key] == "unchanged"
        config.app.pop(key, None)

    def test_try_runtime_config_lock_returns_immediately_when_busy(self):
        """试听锁不能等待长任务释放全局配置，忙碌时应立即让 UI 提示重试。"""
        attempted = threading.Event()
        result = []

        def try_lock():
            with config.try_runtime_config_lock() as acquired:
                result.append(acquired)
            attempted.set()

        with config.runtime_config_lock():
            worker = threading.Thread(target=try_lock)
            worker.start()
            assert attempted.wait(timeout=0.2)

        worker.join(timeout=1)
        assert result == [False]

        with config.try_runtime_config_lock() as acquired:
            assert acquired is True
