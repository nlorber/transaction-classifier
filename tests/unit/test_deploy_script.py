"""Tests for scripts/deploy_model.sh, the manual promotion path."""

import os
import subprocess
import sys
from pathlib import Path


def _deploy(store: Path, version: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "TXCLS_ARTIFACT_DIR": str(store), "TXCLS_PYTHON": sys.executable}
    return subprocess.run(
        ["bash", "scripts/deploy_model.sh", version],
        env=env,
        capture_output=True,
        text=True,
    )


def test_second_deploy_repoints_current(tmp_path):
    """An existing `current` must be replaced, not have the new link nested inside it."""
    for version in ("v-1", "v-2"):
        (tmp_path / version).mkdir()

    for version in ("v-1", "v-2"):
        result = _deploy(tmp_path, version)
        assert result.returncode == 0, result.stderr

    assert Path(os.readlink(tmp_path / "current")).name == "v-2"
    assert list((tmp_path / "v-1").iterdir()) == []


def test_unknown_version_exits_nonzero(tmp_path):
    result = _deploy(tmp_path, "v-missing")
    assert result.returncode == 1
    assert not (tmp_path / "current").exists()
