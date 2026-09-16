"""Small, dependency-light sensor, movement and scoring contracts."""

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

TARGET_RNG_DOMAIN = 0x54415247
PHYSICS_PROTOCOL = "runtime_owned_physics_render_copy_v1"
RANDOMIZATION_PROTOCOL = "body_seed_target_pcg64_seedsequence_54415247_v1"


class Action(IntEnum):
    STOP = 0
    ADVANCE = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3


COMMANDS = ((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0))
FAMILIES = ("stationary", "lateral", "receding", "disappearance", "perturbation")


@dataclass(frozen=True)
class SensorFrame:
    """Only inputs available to the controller; ticks are 20 ms simulation ticks."""

    tick: int
    captured_tick: int
    rgb: np.ndarray
    body: np.ndarray


class MovementGate:
    def __init__(self):
        self.action = Action.STOP
        self.changed_tick = -4
        self.last_tick = -1
        self.last_capture = -1
        self.fault = None

    def apply(self, action, observation):
        if self.fault:
            return COMMANDS[0]
        o = observation
        valid = (
            o.rgb.shape == (240, 320, 3)
            and o.rgb.dtype == np.uint8
            and o.body.shape == (34,)
            and np.isfinite(o.body).all()
            and isinstance(action, (int, np.integer))
            and int(action) in range(4)
            and o.tick >= self.last_tick
            and self.last_capture <= o.captured_tick <= o.tick
        )
        if not valid:
            self.fault = "INVALID_SENSOR"
        elif (o.tick - o.captured_tick) * 0.02 > 0.25:
            self.fault = "STALE_CAMERA"
        self.last_tick, self.last_capture = o.tick, o.captured_tick
        if self.fault:
            self.action = Action.STOP
        elif action != self.action and (
            action == Action.STOP or o.tick - self.changed_tick >= 4
        ):
            self.action, self.changed_tick = Action(action), o.tick
        return COMMANDS[self.action]


def teacher_action(bearing, distance, visible):
    if not math.isfinite(bearing) or not math.isfinite(distance) or distance < 0:
        raise ValueError("invalid teacher geometry")
    if not visible:
        return Action.STOP
    if abs(bearing) > math.radians(15):
        return Action.TURN_LEFT if bearing > 0 else Action.TURN_RIGHT
    return Action.ADVANCE if distance > 0.7 else Action.STOP


class SpatialPool:
    """Cell-type ordered 3x3 means retaining coarse retinotopy."""

    def __init__(self, types, x, y):
        self.types = sorted(set(np.asarray(types).astype(str)))
        x, y = np.asarray(x, float), np.asarray(y, float)
        bins = []
        for coord in (x, y):
            bins.append(
                np.minimum(
                    2,
                    ((coord - coord.min()) / max(np.ptp(coord), 1e-9) * 3).astype(int),
                )
            )
        type_ids = np.searchsorted(self.types, np.asarray(types).astype(str))
        self.indices = type_ids * 9 + bins[1] * 3 + bins[0]
        self.size = len(self.types) * 9
        self.count = np.maximum(np.bincount(self.indices, minlength=self.size), 1)

    def __call__(self, activity):
        activity = np.asarray(activity).reshape(-1)
        if activity.shape != self.indices.shape or not np.isfinite(activity).all():
            raise ValueError("invalid neural activity")
        return (
            np.bincount(self.indices, weights=activity, minlength=self.size)
            / self.count
        ).astype(np.float32)


@dataclass(frozen=True)
class EpisodeSpec:
    seed: int
    family: str
    partition: str


def episode_specs(partition, count, offset=0):
    base = {"train": 0, "validation": 100_000, "test": 200_000}[partition]
    if count < 1 or offset < 0 or offset + count >= 100_000:
        raise ValueError("invalid episode count/offset")
    return [
        EpisodeSpec(base + i, FAMILIES[i % 5], partition)
        for i in range(offset, offset + count)
    ]


def score_episode(rows, fault=None):
    eligible = [r for r in rows if r["time"] >= 2 and r["visible"]]
    tracked = [
        r
        for r in eligible
        if abs(r["bearing"]) <= math.radians(15) and 0.4 <= r["distance"] <= 0.8
    ]
    fraction = len(tracked) / len(eligible) if eligible else None
    fallen = any(r["fallen"] for r in rows)
    collision = any(r["collision"] for r in rows)
    return {
        "trackingFraction": fraction,
        "meanAbsBearingRad": float(np.mean([abs(r["bearing"]) for r in eligible]))
        if eligible
        else None,
        "meanDistanceM": float(np.mean([r["distance"] for r in eligible]))
        if eligible
        else None,
        "visibleSamples": len(eligible),
        "samples": len(rows),
        "fallen": fallen,
        "collision": collision,
        "fault": fault,
        "passed": bool(
            fraction is not None
            and fraction >= 0.8
            and not fallen
            and not collision
            and not fault
        ),
    }
