"""Additive, strict navigation protocol. V1 locomotion contracts stay unchanged."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Identifier = Annotated[
    str, StringConstraints(strict=True, pattern=r"^[A-Za-z0-9._:-]{1,128}$")
]
Digest = Annotated[
    str, StringConstraints(strict=True, pattern=r"^sha256:[0-9a-f]{64}$")
]
TaskId = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{32}$")]
Number = Annotated[float, Field(strict=True, allow_inf_nan=False, ge=-10000, le=10000)]


def canonical(value) -> str:
    """Sorted UTF-8 JSON; plain decimals with trailing zeros removed; negative zero is zero."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", by_alias=True)
    if value is None or isinstance(value, (str, bool)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (float, int, Decimal)):
        number = Decimal(str(value))
        if not number.is_finite():
            raise ValueError("non-finite canonical number")
        if not number:
            return "0"
        return (
            format(number, "f").rstrip("0").rstrip(".")
            if "." in format(number, "f")
            else format(number, "f")
        )
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canonical(item) for item in value) + "]"
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return (
            "{"
            + ",".join(canonical(k) + ":" + canonical(value[k]) for k in sorted(value))
            + "}"
        )
    raise ValueError("invalid canonical value")


def digest(value) -> str:
    return "sha256:" + hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


class Model(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, populate_by_name=True, frozen=True
    )


class Pose(Model):
    x: Number
    y: Number
    yaw: Annotated[
        float,
        Field(
            strict=True,
            allow_inf_nan=False,
            ge=-3.141592653589793,
            le=3.141592653589793,
        ),
    ]


class Area(Model):
    minX: Number
    minY: Number
    maxX: Number
    maxY: Number

    @model_validator(mode="after")
    def ordered(self):
        if self.minX >= self.maxX or self.minY >= self.maxY:
            raise ValueError("invalid area")
        return self

    def contains(self, x: float, y: float, margin: float = 0) -> bool:
        return (
            self.minX + margin <= x <= self.maxX - margin
            and self.minY + margin <= y <= self.maxY - margin
        )


class NavigationProfile(Model):
    robotRadiusM: Annotated[float, Field(strict=True, gt=0, le=1, allow_inf_nan=False)]
    clearanceM: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
    gridResolutionM: Annotated[
        float, Field(strict=True, gt=0, le=1, allow_inf_nan=False)
    ]
    maxSpeedMps: Annotated[float, Field(strict=True, gt=0, le=0.4, allow_inf_nan=False)]
    maxYawRateRadps: Annotated[
        float, Field(strict=True, gt=0, le=1.0, allow_inf_nan=False)
    ]
    poseFreshnessMs: Annotated[int, Field(gt=0, le=1000)]
    arrivalToleranceM: Annotated[
        float, Field(strict=True, gt=0, le=0.2, allow_inf_nan=False)
    ]
    headingToleranceRad: Annotated[
        float, Field(strict=True, gt=0, le=0.5, allow_inf_nan=False)
    ]
    settleMs: Annotated[int, Field(ge=200, le=5000)]
    stableSpeedMps: Annotated[
        float, Field(strict=True, gt=0, le=0.05, allow_inf_nan=False)
    ]
    stableYawRateRadps: Annotated[
        float, Field(strict=True, gt=0, le=0.1, allow_inf_nan=False)
    ]
    deadlineMs: Annotated[int, Field(gt=0, le=600000)]
    leaseMs: Annotated[int, Field(ge=500, le=2000)]

    @model_validator(mode="after")
    def valid_deadline(self):
        if self.deadlineMs <= max(self.leaseMs, self.settleMs):
            raise ValueError("deadline must exceed lease and settling")
        return self


class Scene(Model):
    revision: Identifier
    area: Area
    obstacles: Annotated[list[Area], Field(max_length=128)]
    landmarks: Annotated[dict[Identifier, Pose], Field(min_length=1, max_length=100)]

    @model_validator(mode="after")
    def bounded(self):
        if any(not self.area.contains(p.x, p.y) for p in self.landmarks.values()):
            raise ValueError("landmark outside area")
        return self


class Binding(Model):
    robotId: Identifier
    targetId: Identifier
    runtimeSession: Identifier


class Environment(Model):
    schema_: Literal["rom.microduck.environment.v1"] = Field(alias="schema")
    binding: Binding
    provenance: Literal["SIM_GROUND_TRUTH"]
    capturedAt: Identifier
    sequence: Annotated[int, Field(ge=0)]
    mapFrame: Literal["map"]
    bodyFrame: Literal["body"]
    transformRevision: Literal["xy-forward-left-z-up-v1"]
    pose: Pose
    valid: bool
    scene: Scene
    mapDigest: Digest
    bundleDigest: Digest
    navigationProfileDigest: Digest

    @model_validator(mode="after")
    def coherent(self):
        if self.mapDigest != digest(self.scene):
            raise ValueError("map digest mismatch")
        return self


class PlanningReferences(Model):
    source: Literal["MANUAL", "TAIROS"]
    requestId: Identifier
    inputSnapshotDigest: Digest
    providerResultRef: Identifier | None = None

    @model_validator(mode="after")
    def source_reference(self):
        if (self.source == "TAIROS") != (self.providerResultRef is not None):
            raise ValueError("planning source reference mismatch")
        return self


class NavigationProposal(Model):
    schema_: Literal["ROM_MICRODUCK_NAVIGATION_PROPOSAL_V2"] = Field(alias="schema")
    actionCode: Literal["NAVIGATE_TO_LANDMARK"]
    binding: Binding
    landmarkId: Identifier
    destination: Pose
    scene: Scene
    mapDigest: Digest
    bundleVersion: Identifier
    bundleDigest: Digest
    navigationProfileDigest: Digest
    profile: NavigationProfile
    safeStopBehavior: Literal["ZERO_THEN_HOLD"]
    planning: PlanningReferences

    @model_validator(mode="after")
    def bound(self):
        if self.mapDigest != digest(
            self.scene
        ) or self.navigationProfileDigest != digest(self.profile):
            raise ValueError("proposal digest mismatch")
        if self.scene.landmarks.get(self.landmarkId) != self.destination:
            raise ValueError("destination not bound to landmark")
        return self


class NavigationTaskRequest(Model):
    schema_: Literal["MICRODUCK_NAVIGATION_TASK_V2"] = Field(alias="schema")
    taskId: TaskId
    proposalDigest: Digest
    proposal: NavigationProposal
    requestedBy: Identifier

    @model_validator(mode="after")
    def approved_digest(self):
        if self.proposalDigest != digest(self.proposal):
            raise ValueError("proposal digest mismatch")
        return self


class NavigationLeaseRequest(Model):
    taskId: TaskId
    proposalDigest: Digest
    sequence: Annotated[int, Field(gt=0, le=9223372036854775807)]
