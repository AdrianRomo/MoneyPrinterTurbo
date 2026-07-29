from pathlib import Path
from unittest.mock import patch

from app.utils import utils


def test_storage_dir_uses_environment_override(tmp_path):
    with patch.dict("os.environ", {"MPT_STORAGE_DIR": str(tmp_path)}):
        assert utils.storage_dir() == str(tmp_path)
        assert utils.storage_dir("article", create=True) == str(tmp_path / "article")
        assert (tmp_path / "article").is_dir()


def test_task_dir_uses_environment_storage_root(tmp_path):
    with patch.dict("os.environ", {"MPT_STORAGE_DIR": str(tmp_path)}):
        task_path = Path(utils.task_dir("task-1"))

    assert task_path == tmp_path / "tasks" / "task-1"
    assert task_path.is_dir()

