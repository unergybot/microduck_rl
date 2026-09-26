import json
from pathlib import Path

import pytest

from mjlab_microduck.rom.mujoco_runtime import MicroduckMujocoRuntime
from mjlab_microduck.rom.navigation_contracts import NavigationTaskRequest, digest
from mjlab_microduck.rom.navigation_service import NavigationRuntimeRequest
from tests.test_rom_mujoco_runtime import _write_verified_bundle


def request_for(bundle):
    data = json.loads(Path("tests/fixtures/navigation/wire.json").read_text())["task"]
    data["proposal"].update(
        bundleDigest=bundle.bundleDigest, bundleVersion=bundle.bundleVersion
    )
    data["proposalDigest"] = digest(data["proposal"])
    nav = NavigationTaskRequest.model_validate(data)
    return NavigationRuntimeRequest(
        schema="MICRODUCK_SIM_TASK_V1",
        taskId=nav.taskId,
        actionCode="WALK_VELOCITY",
        bundleVersion=bundle.bundleVersion,
        bundleDigest=bundle.bundleDigest,
        parameters={"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
        scenario={"terrain": "flat", "seed": 7},
        leaseMs=1000,
        requestedBy="test",
        navigation=nav,
    )


def test_navigation_cannot_use_unqualified_runtime(tmp_path):
    bundle = _write_verified_bundle(tmp_path / "bundle")
    runtime = MicroduckMujocoRuntime(tmp_path / "bundle", bundle, realtime=False)
    with pytest.raises(ValueError, match="navigation"):
        runtime.validate(bundle.actions[0], request_for(bundle))


def test_staged_navigation_geometry_cannot_change_v1_collisions(tmp_path, monkeypatch):
    import mujoco
    from mjlab_microduck.rom.navigation.installation import Installation
    from mjlab_microduck.rom.navigation_contracts import Scene, NavigationProfile

    data = json.loads(Path("tests/fixtures/navigation/candidate.json").read_text())
    bundle = _write_verified_bundle(tmp_path / "bundle")
    installed = Installation(
        Scene.model_validate(data["scene"]),
        NavigationProfile.model_validate(data["profile"]),
        "microduck-sim",
        "1",
        bundle.bundleDigest,
        "sha256:" + "c" * 64,
    )
    monkeypatch.setattr(
        "mjlab_microduck.rom.navigation.installation.load", lambda *_: installed
    )
    runtime = MicroduckMujocoRuntime(tmp_path / "bundle", bundle, realtime=False)
    geom = mujoco.mj_name2id(
        runtime._model, mujoco.mjtObj.mjOBJ_GEOM, "rom_navigation_obstacle_0"
    )
    assert (
        runtime._model.geom_contype[geom] == 0
        and runtime._model.geom_conaffinity[geom] == 0
    )


def test_malformed_optional_navigation_files_do_not_break_v1(tmp_path):
    bundle = _write_verified_bundle(tmp_path / "bundle")
    (tmp_path / "bundle" / "navigation.json").write_text("null")
    (tmp_path / "bundle" / "navigation-qualification.json").write_text("[]")
    runtime = MicroduckMujocoRuntime(tmp_path / "bundle", bundle, realtime=False)
    assert runtime._navigation_installation is None


def test_calibrated_desk_collides_only_during_navigation(tmp_path, monkeypatch):
    import mujoco
    from mjlab_microduck.rom.navigation.installation import Installation
    from mjlab_microduck.rom.navigation_contracts import NavigationProfile, Scene

    data = json.loads(
        Path("tests/fixtures/navigation/calibrated-scenarios.json").read_text()
    )
    bundle = _write_verified_bundle(tmp_path / "bundle")
    scene = Scene.model_validate(data["scene"])
    profile = NavigationProfile.model_validate(data["profile"])
    installed = Installation(
        scene, profile, "microduck-sim", "1", bundle.bundleDigest, "sha256:" + "c" * 64
    )
    monkeypatch.setattr(
        "mjlab_microduck.rom.navigation.installation.load", lambda *_: installed
    )
    runtime = MicroduckMujocoRuntime(tmp_path / "bundle", bundle, realtime=False)
    names = ["rom_navigation_obstacle_0", "rom_navigation_obstacle_0_leg_left_front"]
    ids = [
        mujoco.mj_name2id(runtime._model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in names
    ]
    assert all(runtime._model.geom_contype[i] == 0 for i in ids)

    raw = json.loads(Path("tests/fixtures/navigation/wire.json").read_text())["task"]
    raw["proposal"].update(
        scene=data["scene"],
        mapDigest=digest(scene),
        profile=data["profile"],
        navigationProfileDigest=digest(profile),
        bundleDigest=bundle.bundleDigest,
        bundleVersion=bundle.bundleVersion,
    )
    raw["proposalDigest"] = digest(raw["proposal"])
    nav = NavigationTaskRequest.model_validate(raw)
    request = NavigationRuntimeRequest(
        schema="MICRODUCK_SIM_TASK_V1",
        taskId=nav.taskId,
        actionCode="WALK_VELOCITY",
        bundleVersion=bundle.bundleVersion,
        bundleDigest=bundle.bundleDigest,
        parameters={"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
        scenario={"terrain": "flat", "seed": 7},
        leaseMs=1000,
        requestedBy="test",
        navigation=nav,
    )
    handle = runtime.start(bundle.actions[0], request)
    assert all(
        runtime._model.geom_contype[i] == 1 and runtime._model.geom_conaffinity[i] == 1
        for i in ids
    )
    runtime.safe_stop(handle, "TEST_END")


def _v2_runtime_and_request(tmp_path, monkeypatch):
    from mjlab_microduck.rom.navigation.installation import Installation
    from mjlab_microduck.rom.navigation_contracts import NavigationProfile, Scene

    fixture = json.loads(
        Path("src/mjlab_microduck/rom/navigation/calibrated_v2.json").read_text()
    )
    bundle = _write_verified_bundle(tmp_path / "bundle")
    scene = Scene.model_validate(fixture["scene"])
    profile = NavigationProfile.model_validate(fixture["profile"])
    installed = Installation(
        scene, profile, "microduck-sim", "1", bundle.bundleDigest, "sha256:" + "c" * 64
    )
    monkeypatch.setattr(
        "mjlab_microduck.rom.navigation.installation.load", lambda *_: installed
    )
    runtime = MicroduckMujocoRuntime(tmp_path / "bundle", bundle, realtime=False)
    raw = json.loads(Path("tests/fixtures/navigation/wire.json").read_text())["task"]
    raw["proposal"].update(
        scene=fixture["scene"],
        mapDigest=digest(scene),
        profile=fixture["profile"],
        navigationProfileDigest=digest(profile),
        bundleDigest=bundle.bundleDigest,
        bundleVersion=bundle.bundleVersion,
    )
    raw["proposalDigest"] = digest(raw["proposal"])
    nav = NavigationTaskRequest.model_validate(raw)
    request = NavigationRuntimeRequest(
        schema="MICRODUCK_SIM_TASK_V1",
        taskId=nav.taskId,
        actionCode="WALK_VELOCITY",
        bundleVersion=bundle.bundleVersion,
        bundleDigest=bundle.bundleDigest,
        parameters={"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
        scenario={"terrain": "flat", "seed": 7},
        leaseMs=1000,
        requestedBy="test",
        navigation=nav,
    )
    return runtime, bundle, request


def test_v2_desk_and_door_are_physical_only_during_navigation(tmp_path, monkeypatch):
    import mujoco
    from mjlab_microduck.rom.navigation.environment import v2_collision_geom_names

    runtime, bundle, request = _v2_runtime_and_request(tmp_path, monkeypatch)
    names = v2_collision_geom_names(runtime._navigation_installation.scene)
    ids = [runtime._model.geom(name).id for name in names]
    assert all(runtime._model.geom_contype[i] == 0 for i in ids)
    assert all(runtime._model.geom_rgba[i, 3] > 0 for i in ids)

    handle = runtime.start(bundle.actions[0], request)
    assert all(
        runtime._model.geom_contype[i] == runtime._model.geom_conaffinity[i] == 1
        for i in ids
    )
    with runtime._lock:
        address = runtime._free_qpos_address
        runtime._data.qpos[address : address + 3] = [0.3, 1.0, 0.12]
        mujoco.mj_forward(runtime._model, runtime._data)
        contacts = {
            mujoco.mj_id2name(runtime._model, mujoco.mjtObj.mjOBJ_GEOM, geom)
            for contact in runtime._data.contact
            for geom in (contact.geom1, contact.geom2)
        }
    assert "rom_navigation_obstacle_2" in contacts
    runtime.safe_stop(handle, "TEST_END")

    ordinary = request.model_copy(
        update={"taskId": "a" * 32, "navigation": None}
    )
    ordinary_handle = runtime.start(bundle.actions[0], ordinary)
    assert all(runtime._model.geom_contype[i] == 0 for i in ids)
    assert all(runtime._model.geom_rgba[i, 3] > 0 for i in ids)
    runtime.safe_stop(ordinary_handle, "TEST_END")


def test_v2_unknown_static_collider_invalidates_navigation(tmp_path, monkeypatch):
    from xml.etree import ElementTree as ET
    from mjlab_microduck.rom.navigation import environment

    original = environment.add_geometry

    def add_unknown_collider(model_path, scene):
        original(model_path, scene)
        tree = ET.parse(model_path)
        ET.SubElement(
            tree.getroot().find("worldbody"), "geom",
            name="unmapped_wall", type="box", pos="1.5 1.5 0.1",
            size="0.05 0.05 0.1", contype="1", conaffinity="1",
        )
        tree.write(model_path)

    monkeypatch.setattr(environment, "add_geometry", add_unknown_collider)
    runtime, _, _ = _v2_runtime_and_request(tmp_path, monkeypatch)
    assert runtime._navigation_installation is None
