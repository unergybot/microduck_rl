"""ToF returns ranges for mapped geometry and unknown for no return."""

import os
from pathlib import Path

import mujoco
import pytest

from mjlab_microduck.rom.navigation.tof import ToFSimulator


def scene(with_obstacle):
    obstacle = (
        '<geom name="desk" type="box" pos="1 0 .2" size=".1 .4 .3"/>'
        if with_obstacle
        else ""
    )
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="head" pos="0 0 .2">'
        '<site name="tof"/>'
        "</body>" + obstacle + "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_toF_returns_desk_range_and_preserves_no_return_unknown():
    model, data = scene(True)
    scan = ToFSimulator(model).capture(data)
    hits = [
        distance for row in scan.ranges_m for distance in row if distance is not None
    ]
    assert len(scan.ranges_m) == 8
    assert all(len(row) == 8 for row in scan.ranges_m)
    assert hits
    assert min(hits) < 1
    empty_model, empty_data = scene(False)
    empty = ToFSimulator(empty_model).capture(empty_data)
    assert all(distance is None for row in empty.ranges_m for distance in row)


@pytest.mark.skipif(
    not os.environ.get("MICRODUCK_TEST_BUNDLE"), reason="requires real bundle"
)
def test_installed_head_tof_faces_calibrated_desk():
    from mjlab_microduck.rom.main import load_verified_bundle
    from mjlab_microduck.rom.mujoco_runtime import MicroduckMujocoRuntime

    root = Path(os.environ["MICRODUCK_TEST_BUNDLE"])
    runtime = MicroduckMujocoRuntime(root, load_verified_bundle(root), realtime=False)
    try:
        scan = ToFSimulator(runtime._model).capture(runtime._data)
        ranges = [value for row in scan.ranges_m for value in row if value is not None]
        assert len(ranges) >= 1
        assert 0.25 < min(ranges) < 0.6
    finally:
        runtime._snapshot.cleanup()
