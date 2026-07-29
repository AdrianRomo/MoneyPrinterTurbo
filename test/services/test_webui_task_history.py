import ast
import os
import re
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.utils import file_security, utils


ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"
TASK_HISTORY_HELPERS = {
    "_as_task_video_list",
    "_find_final_task_video",
    "_resolve_task_video_file",
    "_task_video_download_name",
    "_task_video_mime_type",
    "_build_restore_upload_requirements",
    "_get_unmet_restore_upload_requirements",
}
TASK_HISTORY_CONSTANTS = {
    "_FINAL_VIDEO_PATTERN",
    "VOICE_MODE_TTS",
    "VOICE_MODE_UPLOAD",
    "VOICE_MODE_NONE",
}


def _load_task_history_helpers():
    """
    从 WebUI 入口中隔离加载不依赖 Streamlit 的任务历史纯函数。

    直接导入 Main.py 会执行整套页面渲染。测试只编译目标常量和函数，既验证
    合并后的真实实现，也避免为了单元测试重新拆出一个只有少量函数的生产模块。
    """
    tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
    selected_nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in TASK_HISTORY_CONSTANTS
            for target in node.targets
        ):
            selected_nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in TASK_HISTORY_HELPERS:
            selected_nodes.append(node)

    namespace = {
        "file_security": file_security,
        "logger": MagicMock(),
        "mimetypes": __import__("mimetypes"),
        "os": os,
        "Path": Path,
        "re": re,
        "Mapping": Mapping,
        "utils": utils,
    }
    module = ast.fix_missing_locations(ast.Module(body=selected_nodes, type_ignores=[]))
    exec(compile(module, str(WEBUI_MAIN), "exec"), namespace)
    return namespace


TASK_HISTORY_NAMESPACE = _load_task_history_helpers()
find_final_task_video = TASK_HISTORY_NAMESPACE["_find_final_task_video"]
build_restore_upload_requirements = TASK_HISTORY_NAMESPACE[
    "_build_restore_upload_requirements"
]
get_unmet_restore_upload_requirements = TASK_HISTORY_NAMESPACE[
    "_get_unmet_restore_upload_requirements"
]
as_task_video_list = TASK_HISTORY_NAMESPACE["_as_task_video_list"]
resolve_task_video_file = TASK_HISTORY_NAMESPACE["_resolve_task_video_file"]
task_video_download_name = TASK_HISTORY_NAMESPACE["_task_video_download_name"]
task_video_mime_type = TASK_HISTORY_NAMESPACE["_task_video_mime_type"]


def test_find_final_task_video_ignores_intermediate_files(tmp_path):
    """任务历史只能把 final 成片识别为完成，不能使用合成中间文件。"""
    for file_name in (
        "combined-1.mp4",
        "temp-clip-1.mp4",
        "final-1TEMP_MPY_wvf_snd.mp4",
    ):
        (tmp_path / file_name).touch()

    assert find_final_task_video(str(tmp_path)) == ""


def test_find_final_task_video_returns_first_numbered_output(tmp_path):
    """多成片任务与运行时结果保持一致，默认播放序号最小的最终视频。"""
    (tmp_path / "final-10.mp4").touch()
    (tmp_path / "final-2.mp4").touch()
    (tmp_path / "final-1.mp4").touch()

    assert find_final_task_video(str(tmp_path)) == str(tmp_path / "final-1.mp4")


def test_task_video_list_accepts_legacy_string_state():
    """历史任务兼容 Redis 或旧状态里偶发保存成字符串的视频字段。"""
    assert as_task_video_list("final.mp4") == ["final.mp4"]
    assert as_task_video_list(["final-1.mp4", 1, "", "final-2.mp4"]) == [
        "final-1.mp4",
        "final-2.mp4",
    ]
    assert as_task_video_list(None) == []


def test_resolve_task_video_file_rejects_paths_outside_task_dir(tmp_path):
    """下载按钮只能暴露 storage/tasks 下真实存在的任务视频。"""
    task_dir = tmp_path / "tasks"
    task_dir.mkdir()
    video_file = task_dir / "task-1" / "final-1.mp4"
    video_file.parent.mkdir()
    video_file.write_bytes(b"video")
    secret_file = tmp_path / "secret.mp4"
    secret_file.write_bytes(b"secret")

    with patch.object(utils, "task_dir", return_value=str(task_dir)):
        assert resolve_task_video_file(str(video_file)) == os.path.realpath(video_file)
        assert resolve_task_video_file("../secret.mp4") == ""


def test_task_video_download_name_and_mime_are_browser_friendly():
    """下载文件名保留任务来源，但去掉路径分隔符和空白等不稳定字符。"""
    assert task_video_download_name("task 1/../x", "/tmp/final 1.mp4") == (
        "task-1-..-x-final-1.mp4"
    )
    assert task_video_mime_type("/tmp/final-1.mp4") == "video/mp4"


def test_restore_requirements_block_missing_uploaded_files():
    params = {
        "video_source": "local",
        "custom_audio_file": "/old-task/custom-audio.wav",
        "voice_name": "zh-CN-XiaoxiaoNeural-Female",
    }
    requirements = build_restore_upload_requirements(params)

    assert get_unmet_restore_upload_requirements(
        requirements,
        video_source="local",
        voice_name=params["voice_name"],
        has_local_materials=False,
        has_custom_audio=False,
    ) == {"local_materials", "custom_audio"}


def test_restore_requirements_allow_explicit_replacements():
    requirements = build_restore_upload_requirements(
        {
            "video_source": "local",
            "custom_audio_file": "/old-task/custom-audio.wav",
            "voice_name": "zh-CN-XiaoxiaoNeural-Female",
        }
    )

    assert not get_unmet_restore_upload_requirements(
        requirements,
        video_source="pexels",
        voice_name="en-US-JennyNeural-Female",
        has_local_materials=False,
        has_custom_audio=False,
    )


def test_restore_requirements_require_file_in_upload_voice_mode():
    """恢复上传配音任务时，继续使用上传模式必须重新选择音频文件。"""
    requirements = build_restore_upload_requirements(
        {
            "video_source": "pexels",
            "custom_audio_file": "/old-task/custom-audio.wav",
            "voice_name": "zh-CN-XiaoxiaoNeural-Female",
        }
    )

    assert get_unmet_restore_upload_requirements(
        requirements,
        video_source="pexels",
        voice_name="zh-CN-XiaoxiaoNeural-Female",
        has_local_materials=False,
        has_custom_audio=False,
        voice_mode="upload",
    ) == {"custom_audio"}


def test_restore_requirements_allow_replacing_upload_with_other_voice_modes():
    """用户主动切换到自动配音或无配音时，不再强制恢复历史上传文件。"""
    requirements = build_restore_upload_requirements(
        {
            "video_source": "pexels",
            "custom_audio_file": "/old-task/custom-audio.wav",
            "voice_name": "zh-CN-XiaoxiaoNeural-Female",
        }
    )

    for voice_mode in ("tts", "none"):
        assert not get_unmet_restore_upload_requirements(
            requirements,
            video_source="pexels",
            voice_name="zh-CN-XiaoxiaoNeural-Female",
            has_local_materials=False,
            has_custom_audio=False,
            voice_mode=voice_mode,
        )
