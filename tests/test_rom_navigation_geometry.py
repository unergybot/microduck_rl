"""Installed landmarks are visible in MuJoCo and remain noncontact display geometry."""

from math import pi
from types import SimpleNamespace
from xml.etree import ElementTree as ET

import mujoco
import pytest

from mjlab_microduck.rom.navigation.environment import add_geometry
from mjlab_microduck.rom.viewer import export_geometry


def test_navigation_landmarks_are_visible_and_do_not_collide(tmp_path):
    path = tmp_path / "scene.xml"
    path.write_text("<mujoco><worldbody><geom name='floor' type='plane' size='2 2 .1'/></worldbody></mujoco>")
    pose = lambda x, y, yaw=0: SimpleNamespace(x=x, y=y, yaw=yaw)
    scene = SimpleNamespace(
        obstacles=[SimpleNamespace(minX=.4, maxX=.6, minY=-.1, maxY=.3)],
        landmarks={"home": pose(0, 0), "desk": pose(1, 0), "door": pose(0, 1, pi / 2)},
    )
    add_geometry(path, scene)
    model = mujoco.MjModel.from_xml_path(str(path))
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)]
    assert {"rom_navigation_obstacle_0", "rom_door_header_1", "rom_desk_top_0", "rom_landmark_pad_2"} <= set(names)
    for index, name in enumerate(names):
        if name.startswith("rom_"):
            assert model.geom_contype[index] == model.geom_conaffinity[index] == 0
            assert model.geom_rgba[index, 3] > 0
    assert len(export_geometry(model)) == model.ngeom
    door = model.body(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rom_landmark_1"))
    assert list(door.pos[:2]) == pytest.approx([0, 1])
    assert list(door.quat[[0, 3]]) == pytest.approx([2**-.5, 2**-.5])


def test_navigation_geometry_rejects_missing_world(tmp_path):
    path = tmp_path / "scene.xml"
    path.write_text("<mujoco/>")
    with pytest.raises(ValueError, match="no world"):
        add_geometry(path, SimpleNamespace(obstacles=[], landmarks={}))
