"""Bounded interface between durable ROM tasks and a simulator runtime."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from .contracts import ActionDefinition, RobotStatus, TaskCreateRequest, canonical_json

type RuntimeMetric = str | int | float | bool | None
_MAX_METRICS = 32
_MAX_METRIC_KEY_LENGTH = 64
_MAX_METRIC_STRING_LENGTH = 128
_MAX_METRICS_ENCODED_BYTES = 1_024
_TRACKING_MEAN_DECIMAL_PLACES = 6


def canonical_tracking_mean(
    tracking_error_sum: float, tracking_sample_count: int
) -> float:
    """Serialize one finite tracking sum/sample mean at runtime precision."""
    if (
        not isinstance(tracking_error_sum, int | float)
        or isinstance(tracking_error_sum, bool)
        or not math.isfinite(float(tracking_error_sum))
        or tracking_error_sum < 0.0
    ):
        raise ValueError("tracking error sum must be a finite nonnegative number")
    if (
        not isinstance(tracking_sample_count, int)
        or isinstance(tracking_sample_count, bool)
        or tracking_sample_count <= 0
    ):
        raise ValueError("tracking sample count must be a positive integer")
    return round(
        float(tracking_error_sum) / tracking_sample_count,
        _TRACKING_MEAN_DECIMAL_PLACES,
    )


def _bounded_metrics(metrics: Mapping[str, RuntimeMetric]) -> dict[str, RuntimeMetric]:
    """Copy the bounded scalar-only metrics safe to persist as task evidence."""
    if len(metrics) > _MAX_METRICS:
        raise ValueError(f"runtime metrics must contain at most {_MAX_METRICS} entries")
    bounded: dict[str, RuntimeMetric] = {}
    for key, value in metrics.items():
        if not isinstance(key, str) or not key or len(key) > _MAX_METRIC_KEY_LENGTH:
            raise ValueError(
                "runtime metric names must be non-empty strings of bounded length"
            )
        if not isinstance(value, str | int | float | bool | type(None)):
            raise TypeError("runtime metrics must be scalar values")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("runtime metrics must be finite")
        if isinstance(value, str) and len(value) > _MAX_METRIC_STRING_LENGTH:
            raise ValueError("runtime metric string values must have bounded length")
        bounded[key] = value
    if len(canonical_json(bounded)) > _MAX_METRICS_ENCODED_BYTES:
        raise ValueError(
            "runtime metrics encoded size exceeds the bounded evidence limit"
        )
    return bounded


# These values are recoverable exactly from the verified bundle and task seed.
_PROVENANCE_KEYS = (
    "bundleDigest",
    "onnxDigest",
    "mjcfDigest",
    "sourceCommit",
    "checkpoint",
    "runIdentity",
    "terrainIdentity",
    "rngSeed",
    "scenarioProfile",
    "resetProfile",
)


def _provenance_digest(metrics: Mapping[str, RuntimeMetric]) -> str:
    provenance = {key: metrics[key] for key in _PROVENANCE_KEYS}
    payload = {"schema": "MICRODUCK_EVIDENCE_PROVENANCE_V1", **provenance}
    return "sha256:" + hashlib.sha256(canonical_json(payload)).hexdigest()


def compact_runtime_evidence(
    metrics: Mapping[str, RuntimeMetric],
) -> dict[str, RuntimeMetric]:
    """Keep legacy evidence when it fits; bind large provenance without losing metrics."""
    try:
        return _bounded_metrics(metrics)
    except ValueError:
        compact = {
            key: value for key, value in metrics.items() if key not in _PROVENANCE_KEYS
        }
        compact["provenanceDigest"] = _provenance_digest(metrics)
        # The same byte, scalar and item bounds still apply. Never trim measurements.
        return _bounded_metrics(compact)


def runtime_evidence_identity_matches(metrics, expected) -> bool:
    if "provenanceDigest" in metrics:
        if any(key in metrics for key in _PROVENANCE_KEYS):
            return False
        expected = {
            key: value for key, value in expected.items() if key not in _PROVENANCE_KEYS
        } | {"provenanceDigest": _provenance_digest(expected)}
    return all(
        key in metrics and type(metrics[key]) is type(value) and metrics[key] == value
        for key, value in expected.items()
    )


@dataclass(frozen=True)
class RuntimeHandle:
    """Opaque runtime ownership token for one discrete task."""

    taskId: str


@dataclass(frozen=True)
class RuntimeSample:
    """A bounded state sample; high-rate trajectories are deliberately not represented."""

    running: bool
    terminalState: Literal["SUCCEEDED", "FAILED"] | None = None
    metrics: Mapping[str, RuntimeMetric] = field(default_factory=dict)
    stopReason: str | None = None

    def __post_init__(self) -> None:
        if self.running == (self.terminalState is not None):
            raise ValueError(
                "a runtime sample must be running or have one terminal state"
            )
        object.__setattr__(self, "metrics", _bounded_metrics(self.metrics))


@dataclass(frozen=True)
class RuntimeEvidence:
    """Bounded final runtime evidence returned after a safe stop."""

    metrics: Mapping[str, RuntimeMetric] = field(default_factory=dict)
    stopReason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", _bounded_metrics(self.metrics))


class SimulationRuntime(Protocol):
    """Runtime operations required by the durable discrete-task service."""

    def validate(
        self, action: ActionDefinition, request: TaskCreateRequest
    ) -> None: ...

    def start(
        self, action: ActionDefinition, request: TaskCreateRequest
    ) -> RuntimeHandle: ...

    def command(
        self, handle: RuntimeHandle, parameters: Mapping[str, object]
    ) -> None: ...

    def sample(self, handle: RuntimeHandle) -> RuntimeSample: ...

    def safe_stop(
        self, handle: RuntimeHandle | None, reason: str
    ) -> RuntimeEvidence: ...

    def emergency_stop(self, reason: str) -> None:
        """Fail motion without acquiring or waiting for the primary runtime lock."""
        ...

    def status(self) -> RobotStatus: ...
