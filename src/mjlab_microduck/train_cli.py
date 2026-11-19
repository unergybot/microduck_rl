"""Run mjlab training with ROM lifecycle tracking and shared train options."""

from __future__ import annotations

import hashlib
import os
import sys
import threading
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from functools import partial, update_wrapper
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

    from mjlab_microduck.checkpoints import INIT_CHECKPOINT_ENV, resolve

    source_root = Path(os.environ.get("MICRODUCK_SOURCE_ROOT", Path(__file__).resolve().parents[2])).resolve()
    parent_digest = None
    if spec := os.environ.get(INIT_CHECKPOINT_ENV):
        checkpoint = resolve(spec).resolve()
        with checkpoint.open("rb") as source:
            parent_digest = hashlib.file_digest(source, "sha256").hexdigest()
        os.environ[INIT_CHECKPOINT_ENV] = str(checkpoint)
    device = "cpu" if not os.environ.get("CUDA_VISIBLE_DEVICES") else f"cuda:{os.environ.get('LOCAL_RANK', '0')}"
    lifecycle = TrainingRunLifecycle(
        Path(log_dir),
        build_run_metadata(
            task=task_id,
            source_root=source_root,
            train_cfg=_mapping(cfg.agent),
            env=cfg.env,
            device=device,
            parent_checkpoint_sha256=parent_digest,
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


def _run_train_with_lifecycle(
    task_id: str, cfg: Any, log_dir: Path, *, training_env: dict[str, str]
) -> Any:
    from mjlab.scripts import train as mjlab_train
    from mjlab_microduck.checkpoints import INIT_CHECKPOINT_ENV, install_init_checkpoint_patch

    # torchrunx serializes this entry point into separate worker processes.
    os.environ.update(training_env)
    if INIT_CHECKPOINT_ENV in os.environ:
        install_init_checkpoint_patch()
    delegate = getattr(mjlab_train, "_microduck_original_run_train", mjlab_train.run_train)
    if os.environ.get("RANK", "0") != "0" or "MicroDuck" not in task_id:
        return delegate(task_id, cfg, log_dir)
    return run_train_observed(task_id, cfg, log_dir, delegate)


def install_training_lifecycle() -> None:
    """Instrument either installed train entry point once, including rank-zero workers."""
    from mjlab.scripts import train as mjlab_train
    from mjlab_microduck.checkpoints import INIT_CHECKPOINT_ENV

    if hasattr(mjlab_train, "_microduck_original_run_train"):
        return
    names = (
        INIT_CHECKPOINT_ENV,
        "MICRODUCK_WARM_START",
        "MICRODUCK_SOURCE_ROOT",
        "MICRODUCK_IN_HF_JOB",
        "MICRODUCK_SOURCE_PROVENANCE",
    )
    training_env = {name: os.environ[name] for name in names if name in os.environ}
    mjlab_train._microduck_original_run_train = mjlab_train.run_train
    worker = partial(_run_train_with_lifecycle, training_env=training_env)
    mjlab_train.run_train = update_wrapper(worker, _run_train_with_lifecycle)


def main() -> int | None:
    from mjlab_microduck.train_hook import maybe_init_checkpoint, maybe_submit_to_hf_jobs

    maybe_submit_to_hf_jobs()
    maybe_init_checkpoint()
    install_training_lifecycle()
    from mjlab.scripts.train import main as mjlab_main

    return mjlab_main()


if __name__ == "__main__":
    from mjlab_microduck.train_cli import main

    sys.exit(main())
