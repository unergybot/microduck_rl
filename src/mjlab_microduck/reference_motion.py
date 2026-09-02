"""Strict loading and phase sampling for Blender-authored reference motion."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path

import numpy as np

from mjlab_microduck.blender_motion import MotionValidationError, validate_motion


class ReferenceMotionError(ValueError):
    """A reference path is missing or does not satisfy the canonical archive contract."""


@dataclass(frozen=True)
class ReferenceMotion:
    joint_pos: np.ndarray
    root_height: np.ndarray
    sha256: str
    fps: int = 50

    @property
    def frames(self) -> int:
        return int(self.joint_pos.shape[0])

    def sample_phase(self, phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        phase = np.asarray(phase, dtype=np.float64)
        if not np.all(np.isfinite(phase)):
            raise ReferenceMotionError("reference phase must be finite")
        source = np.linspace(0.0, 1.0, self.frames)
        clipped = np.clip(phase, 0.0, 1.0)
        joints = np.column_stack(
            [np.interp(clipped, source, self.joint_pos[:, index]) for index in range(14)]
        )
        height = np.interp(clipped, source, self.root_height)
        return joints.astype(np.float32), height.astype(np.float32)


def load_reference_motion(path: str | Path | None) -> ReferenceMotion:
    if path is None:
        configured = os.environ.get("MICRODUCK_REFERENCE_MOTION", "").strip()
        if not configured:
            raise ReferenceMotionError(
                "MICRODUCK_REFERENCE_MOTION must name a validated Blender NPZ"
            )
        path = configured
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ReferenceMotionError(f"reference motion does not exist: {resolved}")
    try:
        validation = validate_motion(resolved)
        with np.load(resolved, allow_pickle=False) as archive:
            joints = np.asarray(archive["joint_pos"], dtype=np.float32)
            root_height = np.asarray(archive["body_pos_w"], dtype=np.float32)[:, 0, 2]
    except (MotionValidationError, OSError, ValueError, KeyError) as exc:
        raise ReferenceMotionError(f"invalid reference motion: {exc}") from exc
    if validation.frames < 3:
        raise ReferenceMotionError("reference motion must contain at least three frames")
    return ReferenceMotion(
        joint_pos=joints,
        root_height=root_height,
        sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(),
    )
