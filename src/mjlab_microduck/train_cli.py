"""`train` entry point: mjlab's trainer, plus `--hf-jobs` remote submission.

This project's [project.scripts] `train` shadows mjlab's so the everyday
command grows one flag:

    uv run train Mjlab-Kick-Flat-MicroDuck --env.scene.num-envs 4096 \
        --agent.max_iterations 4000              # local, exactly as before
    uv run train Mjlab-Kick-Flat-MicroDuck --env.scene.num-envs 4096 \
        --agent.max_iterations 4000 --hf-jobs    # same run, on HF Jobs

Without --hf-jobs, argv is passed to mjlab.scripts.train untouched. With it,
the submission flags (--flavor, --namespace, --detach, ... see hf_jobs.py)
are consumed here and everything else is forwarded to `uv run train` inside
the job.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from mjlab_microduck.training_lifecycle import TrainingRunLifecycle, build_run_metadata


def _mapping(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    return dict(vars(value))


def run_train_observed(
    task_id: str,
    cfg: Any,
    log_dir: Path,
    delegate: Callable[[str, Any, Path], Any],
) -> Any:
    """Run the complete trainer inside one explicit ROM-visible lifecycle."""

    source_root = Path(os.environ["MICRODUCK_SOURCE_ROOT"]).resolve()
    device = "cpu" if not os.environ.get("CUDA_VISIBLE_DEVICES") else f"cuda:{os.environ.get('LOCAL_RANK', '0')}"
    lifecycle = TrainingRunLifecycle(
        Path(log_dir),
        build_run_metadata(
            task=task_id,
            source_root=source_root,
            train_cfg=_mapping(cfg.agent),
            env=cfg.env,
            device=device,
        ),
    )
    lifecycle.start()
    heartbeat_stop = threading.Event()

    def heartbeat() -> None:
        while not heartbeat_stop.wait(15.0):
            lifecycle.heartbeat()

    heartbeat_thread = threading.Thread(
        target=heartbeat,
        name="training-lifecycle-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        result = delegate(task_id, cfg, Path(log_dir))
    except BaseException as exception:
        lifecycle.fail(exit_code=1, error_summary=f"{type(exception).__name__}: {exception}")
        raise
    else:
        lifecycle.complete(exit_code=0)
        return result
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)


def main() -> int | None:
    argv = sys.argv[1:]
    if "--hf-jobs" in argv:
        from mjlab_microduck.hf_jobs import submit

        return submit([a for a in argv if a != "--hf-jobs"])

    os.environ["MICRODUCK_SOURCE_ROOT"] = str(Path(__file__).resolve().parents[2])

    from mjlab.scripts import train as mjlab_train

    original_run_train = mjlab_train.run_train

    def observed(task_id: str, cfg: Any, log_dir: Path) -> Any:
        return run_train_observed(task_id, cfg, log_dir, original_run_train)

    mjlab_train.run_train = observed

    return mjlab_train.main()


if __name__ == "__main__":
    sys.exit(main())
