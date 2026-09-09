"""Install PID-1 termination handling before importing the application closure."""
# ruff: noqa: I001

from __future__ import annotations

import signal


def _terminate_before_server(_signum: int, _frame: object) -> None:
    """Exit cleanly while no simulator child can exist yet."""
    raise SystemExit(0)


signal.signal(signal.SIGTERM, _terminate_before_server)

# Publish a kernel-visible test/operator barrier only after PID 1 owns SIGTERM,
# and before importing any application closure that can create a runtime child.
import os

_ready_path = "/tmp/.microduck-pid1-sigterm-ready"
# Docker restarts preserve a writable /tmp. Discard the previous process's
# barrier (including a dangling symlink), then publish this process's marker.
# Keep exclusive creation so a raced-in path is never followed or truncated.
try:
    os.unlink(_ready_path)
except FileNotFoundError:
    pass

_ready_fd = os.open(
    _ready_path,
    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
    0o600,
)
os.close(_ready_fd)

from mjlab_microduck.rom.main import main


if __name__ == "__main__":
    main()
