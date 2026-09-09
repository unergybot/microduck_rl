"""Exercise restart handling without importing the simulator's ML dependencies."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_BOOTSTRAP = Path(__file__).parents[1] / "docker/rom-simulator/pid1_bootstrap.py"
_RUNNER = r"""
import os
import runpy
import signal
import stat
import sys
import types
from pathlib import Path

marker = Path(sys.argv[2])
original_open = os.open
original_unlink = os.unlink
# Redirect only the fixed container path; use real filesystem operations.
container_marker = '/tmp/.microduck-pid1-sigterm-ready'
def redirect(path):
    return marker if os.fspath(path) == container_marker else path
os.open = lambda path, *args, **kwargs: original_open(redirect(path), *args, **kwargs)
os.unlink = lambda path, *args, **kwargs: original_unlink(redirect(path), *args, **kwargs)

# The application closure is expensive and irrelevant to bootstrap lifecycle.
module = types.ModuleType('mjlab_microduck.rom.main')
def main():
    assert marker.is_file() and not marker.is_symlink()
    assert marker.read_bytes() == b''
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600
    assert callable(signal.getsignal(signal.SIGTERM))
    print('application reached', flush=True)
    os.kill(os.getpid(), signal.SIGTERM)
    raise AssertionError('SIGTERM was not handled')
module.main = main
sys.modules[module.__name__] = module
runpy.run_path(sys.argv[1], run_name='__main__')
"""


def _start(marker: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _RUNNER, str(_BOOTSTRAP), str(marker)],
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "application reached"


def test_same_container_filesystem_can_start_repeatedly(tmp_path: Path) -> None:
    marker = tmp_path / "ready"
    for _ in range(3):
        _start(marker)


@pytest.mark.parametrize("kind", ["regular", "symlink", "dangling_symlink"])
def test_stale_marker_is_replaced_without_touching_target(
    tmp_path: Path, kind: str,
) -> None:
    marker = tmp_path / "ready"
    target = tmp_path / "unrelated"
    if kind == "regular":
        marker.write_bytes(b"stale marker")
        marker.chmod(0o644)
    else:
        if kind == "symlink":
            target.write_bytes(b"preserve me")
        marker.symlink_to(target)
    _start(marker)
    if kind == "symlink":
        assert target.read_bytes() == b"preserve me"
    elif kind == "dangling_symlink":
        assert not target.exists()
