from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mjlab_microduck.training_lifecycle import (
    TrainingRunLifecycle,
    collect_git_provenance,
)


def _now() -> datetime:
    return datetime(2026, 9, 4, 16, 0, tzinfo=UTC)


def test_running_lifecycle_is_explicit_and_completes_atomically(tmp_path: Path) -> None:
    lifecycle = TrainingRunLifecycle(
        tmp_path,
        {
            "schema_version": 1,
            "task": "Mjlab-SitStand-Flat-MicroDuck",
            "source_commit": "a" * 40,
            "source_dirty": False,
            "fresh_run": True,
            "parent_checkpoint_sha256": None,
            "num_envs": 1024,
            "device": "cuda:0",
            "action_scale": 0.8,
        },
        now=_now,
        pid=321,
        hostname="duale5",
    )

    lifecycle.start()

    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    running = json.loads((tmp_path / ".running").read_text())
    assert metadata["task"] == "Mjlab-SitStand-Flat-MicroDuck"
    assert metadata["fresh_run"] is True
    assert metadata["started_at"] == "2026-09-04T16:00:00Z"
    assert running == {
        "schema_version": 1,
        "pid": 321,
        "hostname": "duale5",
        "started_at": "2026-09-04T16:00:00Z",
        "heartbeat_at": "2026-09-04T16:00:00Z",
    }
    assert not (tmp_path / "completed.json").exists()

    lifecycle.complete(exit_code=0)

    assert not (tmp_path / ".running").exists()
    completed = json.loads((tmp_path / "completed.json").read_text())
    assert completed["status"] == "COMPLETED"
    assert completed["exit_code"] == 0
    assert completed["ended_at"] == "2026-09-04T16:00:00Z"


def test_failed_lifecycle_removes_running_marker_and_bounds_error(tmp_path: Path) -> None:
    lifecycle = TrainingRunLifecycle(
        tmp_path,
        {"schema_version": 1, "task": "Mjlab-SitStand-Flat-MicroDuck"},
        now=_now,
        pid=9,
        hostname="duale5",
    )
    lifecycle.start()

    lifecycle.fail(exit_code=1, error_summary="x" * 5000)

    assert not (tmp_path / ".running").exists()
    failed = json.loads((tmp_path / "failed.json").read_text())
    assert failed["status"] == "FAILED"
    assert failed["exit_code"] == 1
    assert len(failed["error_summary"]) == 1000


def test_git_provenance_identifies_commit_and_dirty_diff(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"], check=True)
    (tmp_path / "tracked.txt").write_text("one\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-qm", "initial"], check=True)

    clean = collect_git_provenance(tmp_path)
    assert clean["source_dirty"] is False
    assert len(clean["source_commit"]) == 40
    assert clean["source_diff_sha256"] is None

    (tmp_path / "tracked.txt").write_text("two\n")
    dirty = collect_git_provenance(tmp_path)
    assert dirty["source_dirty"] is True
    assert len(dirty["source_diff_sha256"]) == 64


def test_lifecycle_refuses_to_replace_existing_metadata(tmp_path: Path) -> None:
    first = TrainingRunLifecycle(tmp_path, {"schema_version": 1, "task": "first"}, now=_now)
    first.start()

    second = TrainingRunLifecycle(tmp_path, {"schema_version": 1, "task": "second"}, now=_now)
    with pytest.raises(FileExistsError):
        second.start()

    assert json.loads((tmp_path / "run_metadata.json").read_text())["task"] == "first"


def test_training_entry_publishes_running_before_environment_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from mjlab_microduck.train_cli import run_train_observed

    class Action:
        scale = 0.8

    class EnvironmentConfig:
        def __init__(self) -> None:
            self.actions = {"joint_pos": Action()}
            self.scene = SimpleNamespace(num_envs=1024)
            self.sim = SimpleNamespace(
                timestep=0.005,
                mujoco=SimpleNamespace(jacobian="sparse", solver="cg"),
            )
            self.decimation = 4

    monkeypatch.setenv("MICRODUCK_SOURCE_ROOT", str(Path(__file__).parents[1]))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    cfg = SimpleNamespace(
        env=EnvironmentConfig(),
        agent=SimpleNamespace(
            resume=False,
            max_iterations=15000,
            save_interval=250,
            seed=7,
            load_run=".*",
            load_checkpoint="model_.*.pt",
        ),
    )

    def initialize_environment(task: str, passed_cfg: object, log_dir: Path) -> None:
        assert task == "Mjlab-SitStand-Flat-MicroDuck"
        assert passed_cfg is cfg
        assert json.loads((log_dir / ".running").read_text())["hostname"]

    run_train_observed("Mjlab-SitStand-Flat-MicroDuck", cfg, tmp_path, initialize_environment)

    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    assert metadata["task"] == "Mjlab-SitStand-Flat-MicroDuck"
    assert metadata["fresh_run"] is True
    assert metadata["parent_checkpoint_sha256"] is None
    assert metadata["num_envs"] == 1024
    assert metadata["device"] == "cuda:0"
    assert metadata["action_scale"] == 0.8
    assert metadata["physics_model"] == "robot_allcollisions.xml"
    assert len(metadata["physics_contract_sha256"]) == 64
    assert metadata["controller"] == "BAM_XL330_M6"
    assert len(metadata["controller_contract_sha256"]) == 64

    assert not (tmp_path / ".running").exists()
    assert json.loads((tmp_path / "completed.json").read_text())["status"] == "COMPLETED"


def test_microduck_runner_does_not_mutate_existing_run_outside_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

    from mjlab_microduck.tasks import MicroduckOnPolicyRunner

    monkeypatch.delenv("MICRODUCK_TRAINING_TASK_ID", raising=False)
    monkeypatch.setattr(VelocityOnPolicyRunner, "__init__", lambda self, *args, **kwargs: None)
    runner = MicroduckOnPolicyRunner(object(), {"algorithm": {}}, log_dir=tmp_path, device="cpu")

    assert runner is not None
    assert not (tmp_path / "run_metadata.json").exists()
    assert not (tmp_path / ".running").exists()
