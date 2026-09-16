"""Sensor/command boundaries: prevent hidden-state leakage and unsafe stale motion."""

import math

import numpy as np
import pytest

from mjlab_microduck.rom.flyvis_tracking.core import (
    Action,
    MovementGate,
    SensorFrame,
    SpatialPool,
    episode_specs,
    score_episode,
    teacher_action,
)


def frame(tick=0, captured_tick=None):
    return SensorFrame(
        tick,
        tick if captured_tick is None else captured_tick,
        np.zeros((240, 320, 3), np.uint8),
        np.zeros(34, np.float32),
    )


def test_stop_bypasses_dwell_and_stale_camera_latches_fault():
    gate = MovementGate()
    assert gate.apply(Action.ADVANCE, frame()) == (0.2, 0.0, 0.0)
    assert gate.apply(Action.TURN_LEFT, frame(2)) == (0.2, 0.0, 0.0)
    assert gate.apply(Action.TURN_LEFT, frame(4)) == (0.0, 0.0, 1.0)
    assert gate.apply(Action.STOP, frame(5)) == (0.0, 0.0, 0.0)
    gate.apply(Action.ADVANCE, frame(10))
    assert gate.apply(Action.ADVANCE, frame(24, 10)) == (0.0, 0.0, 0.0)
    assert gate.fault == "STALE_CAMERA"
    assert gate.apply(Action.ADVANCE, frame(26)) == (0.0, 0.0, 0.0)


def test_invalid_body_and_out_of_order_capture_cannot_drive():
    gate = MovementGate()
    gate.apply(Action.ADVANCE, frame(10))
    bad = frame(12)
    bad.body[0] = np.nan
    assert gate.apply(Action.ADVANCE, bad) == (0.0, 0.0, 0.0)
    assert gate.fault == "INVALID_SENSOR"
    gate = MovementGate()
    gate.apply(Action.ADVANCE, frame(10))
    assert gate.apply(Action.ADVANCE, frame(12, 8)) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "bearing,distance,visible,want",
    [
        (0.0, 0.9, True, Action.ADVANCE),
        (0.0, 0.6, True, Action.STOP),
        (0.4, 0.9, True, Action.TURN_LEFT),
        (-0.4, 0.9, True, Action.TURN_RIGHT),
        (0.4, 0.9, False, Action.STOP),
    ],
)
def test_teacher_uses_world_bearing_with_correct_turn_sign(
    bearing, distance, visible, want
):
    assert teacher_action(bearing, distance, visible) == want


def test_spatial_pool_preserves_left_right_and_cell_type():
    pool = SpatialPool(
        np.array(["A", "A", "B", "B"]),
        np.array([-1, 1, -1, 1]),
        np.array([-1, 1, -1, 1]),
    )
    features = pool(np.array([2.0, 4.0, 6.0, 8.0]))
    assert features.shape == (18,)
    assert features[[0, 8, 9, 17]].tolist() == [2.0, 4.0, 6.0, 8.0]
    assert np.count_nonzero(features) == 4


def test_partitions_are_disjoint_and_repeatable():
    partitions = [episode_specs(p, 80) for p in ("train", "validation", "test")]
    seeds = [{e.seed for e in p} for p in partitions]
    assert not (seeds[0] & seeds[1] or seeds[1] & seeds[2] or seeds[0] & seeds[2])
    assert episode_specs("train", 80) == partitions[0]
    assert len({e.family for e in episode_specs("test", 50)}) == 5


def test_falls_and_empty_visible_windows_are_not_success():
    rows = [
        {
            "time": 3.0,
            "visible": True,
            "bearing": 0.0,
            "distance": 0.6,
            "fallen": False,
            "collision": False,
            "speed": 0.01,
            "yaw_rate": 0.0,
            "requested": 0,
            "applied": 0,
        }
    ]
    assert score_episode(rows)["passed"] is True
    rows[0]["fallen"] = True
    assert score_episode(rows)["passed"] is False
    rows[0]["visible"] = False
    assert score_episode(rows)["trackingFraction"] is None
    assert score_episode([])["passed"] is False


def test_sensor_contract_has_no_target_or_world_pose():
    assert set(frame().__dataclass_fields__) == {"tick", "captured_tick", "rgb", "body"}
    with pytest.raises(ValueError):
        teacher_action(math.nan, 0.6, True)
