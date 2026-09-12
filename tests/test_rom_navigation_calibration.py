import math

import pytest

from mjlab_microduck.rom.navigation.calibration import run_case, summarize
from tests.test_rom_mujoco_runtime import _write_verified_bundle


def test_vertical_settling_is_not_reported_as_planar_progress():
    samples = [
        {
            "time": 0.0,
            "x": 0.0,
            "y": 0.0,
            "z": 0.12,
            "yaw": 3.13,
            "vx": 0.0,
            "vy": 0.0,
            "yawRate": 0.0,
            "tilt": 0.0,
        },
        {
            "time": 1.0,
            "x": 0.03,
            "y": 0.04,
            "z": 0.10,
            "yaw": -3.13,
            "vx": 0.03,
            "vy": 0.04,
            "yawRate": 0.02,
            "tilt": 0.1,
        },
    ]
    result = summarize(samples, (0.03, 0.0, 0.02))
    assert result["planarDisplacementM"] == pytest.approx(0.05)
    assert result["verticalDisplacementM"] == pytest.approx(-0.02)
    assert result["yawRotationRad"] == pytest.approx(2 * math.pi - 6.26)
    assert result["trackingRmse"]["vy"] == pytest.approx(0.04)


def test_real_runtime_calibration_records_move_and_zero_command_stop(tmp_path):
    bundle = _write_verified_bundle(tmp_path)
    result = run_case(
        tmp_path, bundle, (0.05, 0.0, 0.0), 7, move_seconds=0.04, stop_seconds=0.04
    )
    assert result["seed"] == 7
    assert result["move"]["durationS"] == pytest.approx(0.04)
    assert result["stop"]["durationS"] == pytest.approx(0.04)
    assert result["trace"][-1]["command"] == [0.0, 0.0, 0.0]
    assert result["stoppedCommandConfirmed"] is True
    assert result["bundleDigest"] == bundle.bundleDigest
