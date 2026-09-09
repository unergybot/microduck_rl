import json
from pathlib import Path
import pytest
from mjlab_microduck.rom.navigation_contracts import NavigationTaskRequest, digest
from mjlab_microduck.rom.navigation_service import NavigationRuntimeRequest
from mjlab_microduck.rom.mujoco_runtime import MicroduckMujocoRuntime
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
