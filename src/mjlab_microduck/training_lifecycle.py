"""Explicit, atomic lifecycle artifacts for ROM-observed training runs."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Mapping[str, Any], *, replace: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(payload, output, sort_keys=True, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


class TrainingRunLifecycle:
    """Own the durable lifecycle files for one training run directory."""

    def __init__(
        self,
        run_dir: Path,
        metadata: Mapping[str, Any],
        *,
        now: Clock = _utc_now,
        pid: int | None = None,
        hostname: str | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.metadata = dict(metadata)
        self.now = now
        self.pid = os.getpid() if pid is None else pid
        self.hostname = socket.gethostname() if hostname is None else hostname
        self.started_at: str | None = None

    def start(self) -> None:
        started_at = _timestamp(self.now())
        metadata = {**self.metadata, "started_at": started_at}
        _atomic_json(self.run_dir / "run_metadata.json", metadata, replace=False)
        self.started_at = started_at
        self.heartbeat()

    def heartbeat(self) -> None:
        if self.started_at is None:
            raise RuntimeError("training lifecycle has not started")
        _atomic_json(
            self.run_dir / ".running",
            {
                "schema_version": 1,
                "pid": self.pid,
                "hostname": self.hostname,
                "started_at": self.started_at,
                "heartbeat_at": _timestamp(self.now()),
            },
            replace=True,
        )

    def complete(self, *, exit_code: int = 0) -> None:
        self._terminal("COMPLETED", "completed.json", exit_code=exit_code)

    def fail(self, *, exit_code: int, error_summary: str) -> None:
        self._terminal(
            "FAILED",
            "failed.json",
            exit_code=exit_code,
            error_summary=error_summary[:1000],
        )

    def _terminal(self, status: str, filename: str, **values: Any) -> None:
        if self.started_at is None:
            raise RuntimeError("training lifecycle has not started")
        payload = {
            "schema_version": 1,
            "status": status,
            "started_at": self.started_at,
            "ended_at": _timestamp(self.now()),
            **values,
        }
        _atomic_json(self.run_dir / filename, payload, replace=True)
        (self.run_dir / ".running").unlink(missing_ok=True)


def _git(repository: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
    ).stdout


def collect_git_provenance(repository: Path) -> dict[str, Any]:
    """Return the exact commit and a deterministic digest for local changes."""

    repository = Path(repository).resolve()
    commit = _git(repository, "rev-parse", "HEAD").decode("ascii").strip()
    status = _git(repository, "status", "--porcelain=v1", "-z")
    if not status:
        return {
            "source_commit": commit,
            "source_dirty": False,
            "source_diff_sha256": None,
        }

    digest = hashlib.sha256()
    digest.update(_git(repository, "diff", "--binary", "HEAD"))
    untracked = _git(repository, "ls-files", "--others", "--exclude-standard", "-z")
    for raw_name in sorted(filter(None, untracked.split(b"\0"))):
        digest.update(raw_name)
        digest.update(b"\0")
        path = repository / os.fsdecode(raw_name)
        if path.is_file():
            digest.update(path.read_bytes())
        digest.update(b"\0")
    return {
        "source_commit": commit,
        "source_dirty": True,
        "source_diff_sha256": digest.hexdigest(),
    }


def _digest_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_run_metadata(
    *,
    task: str,
    source_root: Path,
    train_cfg: Mapping[str, Any],
    env: Any,
    device: str,
) -> dict[str, Any]:
    """Build the immutable identity consumed by the ROM training monitor."""

    unwrapped = getattr(env, "unwrapped", env)
    env_cfg = getattr(unwrapped, "cfg", unwrapped)
    actions = getattr(env_cfg, "actions", {})
    action = actions.get("joint_pos") if isinstance(actions, Mapping) else None
    action_scale = float(getattr(action, "scale", 1.0))
    scene_cfg = getattr(env_cfg, "scene", None)
    num_envs = int(getattr(unwrapped, "num_envs", getattr(scene_cfg, "num_envs", 0)))

    model_name = (
        "robot_allcollisions.xml"
        if any(name in task for name in ("SitStand", "StandUp", "GroundPick", "BallKick"))
        else "robot_walk.xml"
    )
    model_path = source_root / "src/mjlab_microduck/robot/microduck" / model_name
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    sim_cfg = getattr(env_cfg, "sim", None)
    mujoco_cfg = getattr(sim_cfg, "mujoco", None)
    timestep = float(getattr(sim_cfg, "timestep", getattr(sim_cfg, "dt", 0.005)))
    physics_contract = {
        "model": model_name,
        "model_sha256": model_sha256,
        "physics_hz": round(1.0 / timestep),
        "control_hz": round(
            1.0 / (timestep * int(getattr(env_cfg, "decimation", 4)))
        ),
        "jacobian": str(getattr(mujoco_cfg, "jacobian", "unknown")),
        "solver": str(getattr(mujoco_cfg, "solver", "unknown")),
    }
    controller_contract = {
        "controller": "BAM_XL330_M6",
        "kp_fw": 200.0,
        "vin_range": [6.5, 8.2],
        "vin_drop_gain_range": [0.0, 0.2],
        "vin_min": 6.0,
        "delay_substeps": [3, 6],
    }
    resume = bool(train_cfg.get("resume", False))
    return {
        "schema_version": 1,
        "task": task,
        **collect_git_provenance(source_root),
        "fresh_run": not resume,
        "parent_checkpoint_sha256": None,
        "num_envs": num_envs,
        "device": device,
        "action_scale": action_scale,
        "max_iterations": int(train_cfg.get("max_iterations", 0)),
        "save_interval": int(train_cfg.get("save_interval", 0)),
        "seed": int(train_cfg.get("seed", 0)),
        "physics_model": model_name,
        "physics_contract": physics_contract,
        "physics_contract_sha256": _digest_json(physics_contract),
        "controller": "BAM_XL330_M6",
        "controller_contract": controller_contract,
        "controller_contract_sha256": _digest_json(controller_contract),
    }
