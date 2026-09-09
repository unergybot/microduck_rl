import json
from pathlib import Path
import pytest
from mjlab_microduck.rom.navigation_contracts import (
    NavigationTaskRequest,
    NavigationLeaseRequest,
    digest,
    Scene,
    NavigationProfile,
)
from mjlab_microduck.rom.navigation.installation import Installation
from mjlab_microduck.rom.navigation_service import NavigationTaskService
from mjlab_microduck.rom.service import (
    InvalidParameters,
    RobotBusy,
    StaleCommand,
    TaskConflict,
)
from tests.test_rom_continuous_tasks import (
    bundle,
    store,
    runtime,
    clock,
    service,
    db_path,
    walk_request,
)


def nav_and_request(service):
    data = json.loads(Path("tests/fixtures/navigation/candidate.json").read_text())
    installation = Installation(
        Scene.model_validate(data["scene"]),
        NavigationProfile.model_validate(data["profile"]),
        "microduck-sim",
        "1",
        service._bundle.bundleDigest,
        "sha256:" + "c" * 64,
    )
    nav = NavigationTaskService(service, installation, enabled=True)
    raw = json.loads(Path("tests/fixtures/navigation/wire.json").read_text())["task"]
    raw["proposal"].update(
        binding=nav.capabilities()["environment"]["binding"],
        bundleDigest=service._bundle.bundleDigest,
        bundleVersion=service._bundle.bundleVersion,
    )
    raw["proposalDigest"] = digest(raw["proposal"])
    return nav, NavigationTaskRequest.model_validate(raw)


def test_navigation_shares_v1_owner_and_forbids_v1_renewal(service, walk_request):
    nav, request = nav_and_request(service)
    nav.create_task(request)
    with pytest.raises(RobotBusy):
        service.create_task(walk_request)
    from tests.test_rom_continuous_tasks import command

    with pytest.raises(InvalidParameters):
        service.command(request.taskId, command(sequence=1))
    assert nav.create_task(request)["taskId"] == request.taskId
    lease = NavigationLeaseRequest(
        taskId=request.taskId, proposalDigest=request.proposalDigest, sequence=1
    )
    assert nav.renew_lease(request.taskId, lease)["state"] == "RUNNING"
    with pytest.raises(StaleCommand):
        nav.renew_lease(request.taskId, lease)
    nav.cancel_task(request.taskId)


def test_uncertain_submit_is_recovered_without_new_authority(service):
    nav, request = nav_and_request(service)
    nav.create_task(request)
    nav.enabled = False
    assert nav.create_task(request)["taskId"] == request.taskId
    raw = request.model_dump(mode="json", by_alias=True)
    raw["requestedBy"] = "other"
    with pytest.raises(TaskConflict):
        nav.create_task(NavigationTaskRequest.model_validate(raw))


def test_changed_session_rejects_dispatch(service):
    nav, request = nav_and_request(service)
    nav.session = "different"
    with pytest.raises(InvalidParameters):
        nav.create_task(request)
    assert service._store.get(request.taskId) is None


def test_idle_environment_reads_actual_runtime_pose(service, runtime):
    runtime._navigation_test_pose = True
    original = runtime.status
    runtime.status = lambda: original().model_copy(
        update={"basePositionM": (0.23, 0.12, 0.2)}
    )
    nav, _ = nav_and_request(service)
    assert nav.capabilities()["environment"]["pose"]["x"] == 0.23
