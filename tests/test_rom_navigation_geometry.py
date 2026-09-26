"""Installed landmarks and mapped desk geometry share the navigation scene."""

import json
from math import pi
from pathlib import Path
from types import SimpleNamespace

import mujoco
import pytest

from mjlab_microduck.rom.navigation.environment import add_geometry
from mjlab_microduck.rom.navigation.calibrated_scene import canonical_v2_scene
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


def test_v2_mujoco_colliders_match_desk_and_door_map_footprints(tmp_path):
    from mjlab_microduck.rom.navigation.environment import v2_collision_geom_names

    path = tmp_path / "scene.xml"
    path.write_text("<mujoco><worldbody/></mujoco>")
    scene = canonical_v2_scene()
    add_geometry(path, scene)
    model = mujoco.MjModel.from_xml_path(str(path))

    for name, x, y, half_x, half_y in (
        ("rom_navigation_obstacle_0", 0.5, 0.1, 0.1, 0.2),
        ("rom_navigation_obstacle_1", -0.4, 1.0, 0.025, 0.025),
        ("rom_navigation_obstacle_2", 0.4, 1.0, 0.025, 0.025),
    ):
        geom = model.geom(name)
        assert list(geom.pos[:2]) == pytest.approx([x, y])
        assert list(geom.size[:2]) == pytest.approx([half_x, half_y])
        assert geom.contype == geom.conaffinity == 1

    header = model.geom("rom_navigation_obstacle_door_header")
    assert list(header.pos) == pytest.approx([0.0, 1.0, 0.39])
    assert list(header.size) == pytest.approx([0.425, 0.025, 0.02])
    assert header.pos[2] - header.size[2] == pytest.approx(0.37)

    names = v2_collision_geom_names(scene)
    assert names == frozenset(
        {"rom_navigation_obstacle_0", "rom_navigation_obstacle_1", "rom_navigation_obstacle_2", "rom_navigation_obstacle_door_header"}
        | {
            f"rom_navigation_obstacle_0_leg_{x}_{y}"
            for x in ("left", "right")
            for y in ("front", "back")
        }
    )
    for name in names:
        assert model.geom(name).conaffinity == 1
    assert model.geom("rom_landmark_pad_2").conaffinity == 0


def test_v2_door_posts_block_map_without_closing_doorway():
    scene = canonical_v2_scene()
    fixture = json.loads(
        Path("src/mjlab_microduck/rom/navigation/calibrated_v2.json").read_text()
    )
    grid = Grid(scene, NavigationProfile.model_validate(fixture["profile"]))
    assert not grid.free(-0.4, 1.0)
    assert not grid.free(0.4, 1.0)
    assert all(grid.free(x, 1.0) for x in (-0.1, -0.05, 0.0, 0.05, 0.1))
    assert grid.free(scene.landmarks["desk"].x, scene.landmarks["desk"].y)
    unreachable = fixture["scenarios"][3]["start"]
    assert not grid.free(*unreachable[:2])
    assert unreachable[0] - scene.obstacles[0].maxX > fixture["profile"]["robotRadiusM"]
