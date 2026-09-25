"""Installed landmarks and mapped desk geometry share the navigation scene."""

import json
from math import pi
from pathlib import Path
from types import SimpleNamespace

import mujoco
import pytest

from mjlab_microduck.rom.navigation.environment import add_geometry
from mjlab_microduck.rom.navigation.grid import Grid
from mjlab_microduck.rom.navigation_contracts import NavigationProfile, Scene
from mjlab_microduck.rom.viewer import export_geometry


def test_navigation_landmarks_are_visible_and_do_not_collide(tmp_path):
    path = tmp_path / "scene.xml"
    path.write_text(
        "<mujoco><worldbody><geom name='floor' type='plane' size='2 2 .1'/></worldbody></mujoco>"
    )
    pose = lambda x, y, yaw=0: SimpleNamespace(x=x, y=y, yaw=yaw)
    scene = SimpleNamespace(
        obstacles=[SimpleNamespace(minX=0.4, maxX=0.6, minY=-0.1, maxY=0.3)],
        landmarks={"home": pose(0, 0), "desk": pose(1, 0), "door": pose(0, 1, pi / 2)},
    )
    add_geometry(path, scene)
    model = mujoco.MjModel.from_xml_path(str(path))
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        for i in range(model.ngeom)
    ]
    assert {
        "rom_navigation_obstacle_0",
        "rom_door_header_1",
        "rom_desk_top_0",
        "rom_landmark_pad_2",
    } <= set(names)
    for index, name in enumerate(names):
        if name.startswith("rom_"):
            assert model.geom_contype[index] == model.geom_conaffinity[index] == 0
            assert model.geom_rgba[index, 3] > 0
    assert len(export_geometry(model)) == model.ngeom
    door = model.body(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rom_landmark_1")
    )
    assert list(door.pos[:2]) == pytest.approx([0, 1])
    assert list(door.quat[[0, 3]]) == pytest.approx([2**-0.5, 2**-0.5])


def test_navigation_geometry_rejects_missing_world(tmp_path):
    path = tmp_path / "scene.xml"
    path.write_text("<mujoco/>")
    with pytest.raises(ValueError, match="no world"):
        add_geometry(path, SimpleNamespace(obstacles=[], landmarks={}))


def test_calibrated_desk_uses_mapped_obstacle_footprint(tmp_path):
    path = tmp_path / "scene.xml"
    path.write_text("<mujoco><worldbody/></mujoco>")
    fixture = json.loads(
        Path("tests/fixtures/navigation/calibrated-scenarios.json").read_text()
    )
    scene = Scene.model_validate(fixture["scene"])
    add_geometry(path, scene)
    model = mujoco.MjModel.from_xml_path(str(path))
    top = model.geom("rom_navigation_obstacle_0")
    assert list(top.pos[:2]) == pytest.approx([0.5, 0.1])
    assert list(top.size[:2]) == pytest.approx([0.1, 0.2])
    for x_side in ("left", "right"):
        for y_side in ("front", "back"):
            model.geom(f"rom_navigation_obstacle_0_leg_{x_side}_{y_side}")
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rom_desk_top_0") == -1
    grid = Grid(scene, NavigationProfile.model_validate(fixture["profile"]))
    assert not grid.free(0.5, 0.1)
    assert grid.free(scene.landmarks["desk"].x, scene.landmarks["desk"].y)
