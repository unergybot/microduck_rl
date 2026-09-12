"""Opt-in isolated child-process acceptance against a real installed bundle.

MICRODUCK_TEST_BUNDLE=/absolute/bundle PYTHONPATH=src python -m pytest
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


def terminal(client, request, renew=False):
    task_id = request["taskId"]
    deadline = time.monotonic() + 20
    sequence = 1
    while time.monotonic() < deadline:
        state = client.get(f"/v2/navigation/tasks/{task_id}").json()
        if state["state"] not in {"RUNNING", "PENDING", "CANCELLING"}:
            return state
        if renew:
            response = client.put(
                f"/v2/navigation/tasks/{task_id}/lease",
                json={
                    "taskId": task_id,
                    "proposalDigest": request["proposalDigest"],
                    "sequence": sequence,
                },
            )
            assert response.status_code == 200, response.json()
            sequence += 1
        time.sleep(0.1)
    pytest.fail("isolated runtime did not terminate within twenty seconds")


@pytest.mark.parametrize("operation", ["cancel", "expire", "arrive"])
def test_real_child_navigation_stop_and_arrival(live, tmp_path, operation):
    request = submit(live, "home" if operation == "arrive" else "door")
    if operation == "cancel":
        response = live.post(f"/v2/navigation/tasks/{request['taskId']}/cancel")
        assert response.status_code == 200, response.json()
    result = terminal(live, request, renew=operation == "arrive")
    (tmp_path / "acceptance.json").write_text(
        json.dumps({"operation": operation, "result": result}, indent=2)
    )
    metrics = result["evidence"]["metrics"]
    assert metrics["stoppedCommandConfirmed"] is True
    if operation == "arrive":
        assert result["state"] == "SUCCEEDED"
        assert metrics["arrived"] is True
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
