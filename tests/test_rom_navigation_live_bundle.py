"""Opt-in isolated child-process acceptance against a real installed bundle.

MUJOCO_GL=egl MICRODUCK_TEST_BUNDLE=/absolute/bundle PYTHONPATH=src python -m pytest
    tests/test_rom_navigation_live_bundle.py -q
"""

import json
import os
import shutil
import time
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mjlab_microduck.rom.main import create_configured_app
from mjlab_microduck.rom.navigation_contracts import digest

pytestmark = pytest.mark.skipif(
    not os.environ.get("MICRODUCK_TEST_BUNDLE"),
    reason="requires an explicitly selected real MuJoCo/ONNX bundle",
)


@pytest.fixture
def isolated_settings(tmp_path):
    root = tmp_path / "bundle"
    shutil.copytree(os.environ["MICRODUCK_TEST_BUNDLE"], root)
    root.chmod(0o700)
    (root / "navigation.json").chmod(0o600)
    config = json.loads((root / "navigation.json").read_text())
    calibrated = json.loads(
        Path("tests/fixtures/navigation/calibrated-scenarios.json").read_text()
    )
    config.update(scene=calibrated["scene"], profile=calibrated["profile"])
    (root / "navigation.json").write_text(json.dumps(config))
    (root / "navigation-qualification.json").unlink(missing_ok=True)
    token = uuid.uuid4().hex
    settings = {
        "MICRODUCK_ROM_BUNDLE_DIR": str(root),
        "MICRODUCK_ROM_STATE_DB": str(tmp_path / "tasks.sqlite"),
        "MICRODUCK_ROM_BEARER_TOKEN": token,
        "ROM_MICRODUCK_NAVIGATION_ENABLED": "true",
    }
    return settings


@pytest.fixture
def live(isolated_settings):
    with TestClient(create_configured_app(isolated_settings)) as client:
        assert client.get("/v2/navigation/capabilities").status_code == 401
        client.headers["Authorization"] = (
            "Bearer " + isolated_settings["MICRODUCK_ROM_BEARER_TOKEN"]
        )
        yield client


@pytest.fixture
def isolated_v2_settings(isolated_settings):
    root = Path(isolated_settings["MICRODUCK_ROM_BUNDLE_DIR"])
    config = json.loads((root / "navigation.json").read_text())
    canonical = json.loads(
        Path("src/mjlab_microduck/rom/navigation/calibrated_v2.json").read_text()
    )
    config.update(scene=canonical["scene"], profile=canonical["profile"])
    (root / "navigation.json").write_text(json.dumps(config))
    return isolated_settings


@pytest.fixture
def live_v2(isolated_v2_settings):
    with TestClient(create_configured_app(isolated_v2_settings)) as client:
        client.headers["Authorization"] = (
            "Bearer " + isolated_v2_settings["MICRODUCK_ROM_BEARER_TOKEN"]
        )
        yield client


@pytest.fixture
def live_v2_visual(isolated_v2_settings):
    isolated_v2_settings["ROM_MICRODUCK_NAVIGATION_POSE_SOURCE"] = (
        "SIM_VISUAL_ODOMETRY"
    )
    with TestClient(create_configured_app(isolated_v2_settings)) as client:
        client.headers["Authorization"] = (
            "Bearer " + isolated_v2_settings["MICRODUCK_ROM_BEARER_TOKEN"]
        )
        yield client


def submit(client, landmark="door"):
    caps = client.get("/v2/navigation/capabilities").json()
    assert caps["ready"], caps
    env = caps["environment"]
    proposal = json.loads(Path("tests/fixtures/navigation/wire.json").read_text())[
        "task"
    ]["proposal"]
    proposal.update(
        binding=env["binding"],
        scene=env["scene"],
        mapDigest=env["mapDigest"],
        profile=caps["profile"],
        navigationProfileDigest=env["navigationProfileDigest"],
        bundleDigest=env["bundleDigest"],
        bundleVersion=caps["bundleVersion"],
        landmarkId=landmark,
        destination=env["scene"]["landmarks"][landmark],
    )
    proposal["planning"].update(source="MANUAL", inputSnapshotDigest=digest(env))
    request = {
        "schema": "MICRODUCK_NAVIGATION_TASK_V2",
        "taskId": uuid.uuid4().hex,
        "proposalDigest": digest(proposal),
        "proposal": proposal,
        "requestedBy": "isolated-acceptance",
    }
    response = client.post("/v2/navigation/tasks", json=request)
    assert response.status_code == 202, response.json()
    return request


def terminal(client, request, renew=False, timeout_s=20):
    task_id = request["taskId"]
    deadline = time.monotonic() + timeout_s
    sequence = 1
    while time.monotonic() < deadline:
        state = client.get(f"/v2/navigation/tasks/{task_id}").json()
        if state["state"] not in {"RUNNING", "PENDING", "CANCELLING"}:
            return state, sequence - 1
        if renew:
            response = client.put(
                f"/v2/navigation/tasks/{task_id}/lease",
                json={
                    "taskId": task_id,
                    "proposalDigest": request["proposalDigest"],
                    "sequence": sequence,
                },
            )
            if response.status_code != 200:
                # The task may complete between the status read and renewal.
                settled = client.get(f"/v2/navigation/tasks/{task_id}").json()
                if settled["state"] not in {"RUNNING", "PENDING", "CANCELLING"}:
                    return settled, sequence - 1
                pytest.fail(
                    f"renewal {sequence} rejected while task was running: "
                    f"{response.json()}"
                )
            sequence += 1
        time.sleep(0.3)
    pytest.fail("isolated runtime did not terminate within the test deadline")


@pytest.mark.parametrize("landmark", ["door", "desk"])
def test_v2_real_child_reaches_mapped_landmark_and_stops(live_v2, landmark):
    environment = live_v2.get("/v2/navigation/capabilities").json()["environment"]
    assert environment["scene"]["revision"] == "microduck-navigation-calibration-v2"
    assert environment["mapDigest"] == digest(environment["scene"])
    request = submit(live_v2, landmark)
    result, renewals = terminal(live_v2, request, renew=True, timeout_s=90)
    assert renewals > 0
    assert result["state"] == "SUCCEEDED", result
    assert result["evidence"]["metrics"]["arrived"] is True
    assert result["evidence"]["metrics"]["stoppedCommandConfirmed"] is True


def test_v2_visual_pose_source_survives_child_start(live_v2_visual):
    caps = live_v2_visual.get("/v2/navigation/capabilities").json()
    assert caps["controllerPoseSource"] == "SIM_VISUAL_ODOMETRY"
    request = submit(live_v2_visual, "door")
    response = live_v2_visual.post(
        f"/v2/navigation/tasks/{request['taskId']}/cancel"
    )
    assert response.status_code == 200, response.json()


def test_v2_visual_real_child_reaches_door_and_stops(live_v2_visual):
    request = submit(live_v2_visual, "door")
    result, renewals = terminal(live_v2_visual, request, renew=True, timeout_s=90)
    assert renewals > 0
    assert result["state"] == "SUCCEEDED", (
        result["state"], result.get("stopReason"), result.get("evidence"),
        renewals, live_v2_visual.app.state.task_service._supervisor.trace[-15:]
    )
    assert result["evidence"]["metrics"]["stoppedCommandConfirmed"] is True


def test_v2_visual_real_child_reaches_desk_after_door(live_v2_visual):
    for landmark in ("door", "desk"):
        request = submit(live_v2_visual, landmark)
        result, renewals = terminal(live_v2_visual, request, renew=True, timeout_s=120)
        assert renewals > 0
        assert result["state"] == "SUCCEEDED", (
            landmark, result["state"], result.get("stopReason"),
            result.get("evidence"),
        )
        assert result["evidence"]["metrics"]["stoppedCommandConfirmed"] is True


def test_v2_visual_camera_stream_during_door_and_desk(live_v2_visual):
    if os.environ.get("ROM_MICRODUCK_HEAD_CAMERA_ENABLED") != "true":
        pytest.skip("requires explicitly enabled EGL camera observer")
    for landmark in ("door", "desk"):
        request = submit(live_v2_visual, landmark)
        deadline = time.monotonic() + 120
        sequence = 1
        seen = set()
        while time.monotonic() < deadline:
            state = live_v2_visual.get(f"/v2/navigation/tasks/{request['taskId']}").json()
            if state["state"] not in {"RUNNING", "PENDING", "CANCELLING"}:
                break
            camera = live_v2_visual.get("/v1/viewer/camera")
            if camera.status_code == 200:
                assert camera.headers["x-active-task-id"] == request["taskId"]
                assert camera.headers["content-type"] == "image/jpeg"
                seen.add(camera.headers["x-frame-sequence"])
            renewal = live_v2_visual.put(
                f"/v2/navigation/tasks/{request['taskId']}/lease",
                json={"taskId": request["taskId"], "proposalDigest": request["proposalDigest"], "sequence": sequence},
            )
            if renewal.status_code == 200:
                sequence += 1
            time.sleep(.3)
        else:
            pytest.fail(f"{landmark} did not terminate")
        assert state["state"] == "SUCCEEDED", state
        assert state["evidence"]["metrics"]["stoppedCommandConfirmed"] is True
        assert len(seen) >= 2, (landmark, seen)


def test_v2_ground_truth_camera_shows_apriltag_while_idle(live_v2):
    if os.environ.get("ROM_MICRODUCK_HEAD_CAMERA_ENABLED") != "true":
        pytest.skip("requires explicitly enabled EGL camera observer")
    import io

    import numpy as np
    from PIL import Image

    from mjlab_microduck.rom.navigation.apriltag import detect_tag_corners

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        camera = live_v2.get("/v1/viewer/camera")
        if camera.status_code == 200:
            with Image.open(io.BytesIO(camera.content)) as image:
                detected = detect_tag_corners(np.asarray(image.convert("RGB")))
            assert detected, "ground-truth camera did not include calibrated tags"
            assert camera.headers["x-active-task-id"] == ""
            return
        time.sleep(.2)
    pytest.fail("ground-truth camera did not produce an idle frame")


def test_camera_tag_setup_failure_does_not_prevent_control_runtime(
    isolated_v2_settings, monkeypatch
):
    from mjlab_microduck.rom.main import load_verified_bundle
    from mjlab_microduck.rom.mujoco_runtime import MicroduckMujocoRuntime
    from mjlab_microduck.rom.navigation import environment

    def fail_camera_tags(*_args):
        raise ValueError("camera-only tag setup failed")

    monkeypatch.setenv("ROM_MICRODUCK_HEAD_CAMERA_ENABLED", "true")
    monkeypatch.setattr(environment, "add_apriltag_probe", fail_camera_tags)
    root = Path(isolated_v2_settings["MICRODUCK_ROM_BUNDLE_DIR"])
    runtime = MicroduckMujocoRuntime(root, load_verified_bundle(root), realtime=False)
    assert runtime._model is not None
    assert runtime.viewer_model() is not None
    assert runtime._head_camera is None


@pytest.mark.parametrize("operation", ["cancel", "expire", "arrive", "moving_arrive"])
def test_real_child_navigation_stop_and_arrival(live, tmp_path, operation):
    request = submit(live, "home" if operation == "arrive" else "door")
    if operation == "cancel":
        response = live.post(f"/v2/navigation/tasks/{request['taskId']}/cancel")
        assert response.status_code == 200, response.json()
    result, renewals = terminal(
        live,
        request,
        renew=operation in {"arrive", "moving_arrive"},
        timeout_s=60 if operation == "moving_arrive" else 20,
    )
    (tmp_path / "acceptance.json").write_text(
        json.dumps(
            {"operation": operation, "result": result, "renewals": renewals},
            indent=2,
        )
    )
    metrics = result["evidence"]["metrics"]
    assert metrics["stoppedCommandConfirmed"] is True
    if operation in {"arrive", "moving_arrive"}:
        assert result["state"] == "SUCCEEDED"
        assert metrics["arrived"] is True
        if operation == "moving_arrive":
            assert renewals >= 3
    elif operation == "cancel":
        assert result["state"] == "CANCELLED"
    else:
        assert result["stopReason"] == "LEASE_EXPIRED"
    events = live.get(f"/v2/navigation/tasks/{request['taskId']}/events").json()[
        "events"
    ]
    sequences = [event["sequence"] for event in events]
    assert sequences == sorted(set(sequences)) and events


def test_restart_reads_original_task_without_restoring_motion_authority(
    isolated_settings, tmp_path
):
    headers = {
        "Authorization": "Bearer " + isolated_settings["MICRODUCK_ROM_BEARER_TOKEN"]
    }
    with TestClient(
        create_configured_app(isolated_settings), headers=headers
    ) as client:
        request = submit(client)
    with TestClient(
        create_configured_app(isolated_settings), headers=headers
    ) as restarted:
        result = restarted.get(f"/v2/navigation/tasks/{request['taskId']}").json()
        assert result["taskId"] == request["taskId"]
        assert result["state"] not in {"PENDING", "RUNNING"}
        capabilities = restarted.get("/v2/navigation/capabilities").json()
        assert (
            capabilities["environment"]["binding"]["runtimeSession"]
            != request["proposal"]["binding"]["runtimeSession"]
        )
        replay = restarted.post("/v2/navigation/tasks", json=request)
        assert replay.status_code == 202
        assert replay.json()["state"] == result["state"]
        (tmp_path / "acceptance.json").write_text(
            json.dumps({"operation": "restart", "result": result}, indent=2)
        )
