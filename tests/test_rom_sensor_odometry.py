"""A controller estimate must come from sampled sensor values, not world pose."""

import json
import os
from pathlib import Path

import mujoco
import pytest

from mjlab_microduck.rom.navigation.sensor_odometry import SensorFailure, SensorOdometry
from mjlab_microduck.rom.navigation_contracts import Pose


def model():
    return mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody><body name="trunk_base"><freejoint/>
        <geom type="sphere" size=".02" mass=".1"/>
        <site name="imu"/></body></worldbody>
        <sensor><gyro name="imu_ang_vel" site="imu"/>
        <velocimeter name="imu_lin_vel" site="imu"/></sensor></mujoco>"""
    )


def test_integrates_sensor_velocity_and_yaw_without_reading_world_pose():
    fixture = model()
    data = mujoco.MjData(fixture)
    estimator = SensorOdometry(fixture, Pose(x=0.0, y=0.0, yaw=0.0))
    data.qpos[:2] = [50.0, -50.0]
    data.sensordata[estimator.velocity_address] = 0.2
    data.sensordata[estimator.gyro_address + 2] = 1.0
    estimator.update(data)
    data.time = 0.02
    pose, speed, yaw_rate = estimator.update(data)
    assert pose.x == pytest.approx(0.004, abs=1e-5)
    assert pose.y == pytest.approx(0.00004, abs=1e-5)
    assert pose.yaw == pytest.approx(0.02)
    assert speed == pytest.approx(0.2)
    assert yaw_rate == pytest.approx(1.0)


def test_rejects_missing_or_stale_sensor_samples():
    fixture = model()
    data = mujoco.MjData(fixture)
    estimator = SensorOdometry(fixture, Pose(x=0.0, y=0.0, yaw=0.0))
    estimator.update(data)
    data.time = 0.11
    with pytest.raises(SensorFailure, match="stale"):
        estimator.update(data)
    data.time = 0.02
    data.sensordata[estimator.gyro_address] = float("nan")
    with pytest.raises(SensorFailure, match="non-finite"):
        estimator.update(data)
    no_velocity = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><site name='imu'/></worldbody>"
        "<sensor><gyro name='imu_ang_vel' site='imu'/></sensor></mujoco>"
    )
    with pytest.raises(SensorFailure, match="imu_lin_vel"):
        SensorOdometry(no_velocity, Pose(x=0.0, y=0.0, yaw=0.0))


def test_rotating_offset_sensor_does_not_create_base_translation():
    xml = """<mujoco><worldbody><body name="trunk_base"><freejoint/>
        <geom type="sphere" size=".02" mass=".1"/>
        <site name="imu" pos="0 .1 0"/></body></worldbody>
        <sensor><gyro name="imu_ang_vel" site="imu"/>
        <velocimeter name="imu_lin_vel" site="imu"/></sensor></mujoco>"""
    fixture = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(fixture)
    estimator = SensorOdometry(fixture, Pose(x=0.0, y=0.0, yaw=0.0))
    data.sensordata[estimator.gyro_address] = 1.0
    data.sensordata[estimator.velocity_address + 2] = 0.1
    estimator.update(data)
    data.time = 0.02
    pose, _, _ = estimator.update(data)
    assert pose.x == pytest.approx(0.0, abs=1e-9)
    assert pose.y == pytest.approx(0.0, abs=1e-9)


@pytest.mark.skipif(
    not os.environ.get("MICRODUCK_TEST_BUNDLE"), reason="requires real bundle"
)
def test_sensor_loss_fails_and_safely_stops_navigation(monkeypatch):
    from scripts.qualify_rom_navigation import run

    from mjlab_microduck.rom.main import load_verified_bundle
    from mjlab_microduck.rom.navigation_contracts import NavigationProfile, Scene

    def lost_sample(_self, _data):
        raise SensorFailure("simulated dropout")

    monkeypatch.setattr(SensorOdometry, "update", lost_sample)
    root = Path(os.environ["MICRODUCK_TEST_BUNDLE"])
    fixture = json.loads(
        Path("tests/fixtures/navigation/calibrated-scenarios.json").read_text()
    )
    result = run(
        root,
        load_verified_bundle(root),
        Scene.model_validate(fixture["scene"]),
        NavigationProfile.model_validate(fixture["profile"]),
        fixture["scenarios"][0],
        7,
        pose_source="SIM_SENSOR_ODOMETRY",
    )
    assert result["reason"] == "LOCALIZATION_FAILED"
    assert result["passed"] is False
    assert result["evidence"]["stoppedCommandConfirmed"] is True
