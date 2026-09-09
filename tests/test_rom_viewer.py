"""Actual compiled geometry and same-owner read-only viewer regression tests."""

import json
import time

import mujoco
import numpy as np
import pytest
from fastapi.testclient import TestClient

from mjlab_microduck.rom.api import create_app
from mjlab_microduck.rom.mujoco_runtime import MicroduckMujocoRuntime
from tests.test_rom_mujoco_runtime import _write_verified_bundle, _request


def test_actual_geometry_and_idle_running_stopped_frames(tmp_path):
    bundle = _write_verified_bundle(tmp_path / "bundle")
    runtime = MicroduckMujocoRuntime(tmp_path / "bundle", bundle, realtime=False)
    assert callable(getattr(runtime, "viewer_model", None)), (
        "runtime must export its compiled model"
    )
    model = runtime.viewer_model()
    frame = runtime.viewer_frame()
    assert model["schema"] == "MICRODUCK_VIEWER_MODEL_V1"
    assert frame["schema"] == "MICRODUCK_VIEWER_FRAME_V1"
    for key in ("runtimeSession", "modelDigest", "bundleDigest"):
        assert model[key] == frame[key]
    assert model["modelDigest"] == bundle.model.digest
    assert frame["activeTaskId"] is None
    assert len(model["geometry"]) == runtime._model.ngeom
    for geom in model["geometry"]:
        np.testing.assert_allclose(geom["size"], runtime._model.geom_size[geom["id"]])
    np.testing.assert_allclose(frame["positions"], runtime._data.geom_xpos.ravel())
    before = runtime._data.qpos.copy()
    assert runtime.viewer_frame() == frame
    np.testing.assert_array_equal(runtime._data.qpos, before)
    request = _request().model_copy(update={"bundleDigest": bundle.bundleDigest})
    handle = runtime.start(bundle.actions[0], request)
    runtime.sample(handle)
    time.sleep(0.21)
    running = runtime.viewer_frame()
    assert running["activeTaskId"] == request.taskId
    assert running["positions"] != frame["positions"]
    assert running["sequence"] > frame["sequence"]
    np.testing.assert_allclose(running["matrices"], runtime._data.geom_xmat.ravel())
    runtime.safe_stop(handle, "USER_CANCELLED")
    time.sleep(0.21)
    assert runtime.viewer_frame()["activeTaskId"] is None
    other = MicroduckMujocoRuntime(tmp_path / "bundle", bundle, realtime=False)
    assert other.viewer_model()["runtimeSession"] != model["runtimeSession"]


def test_geometry_mesh_fidelity_and_bounds():
    from mjlab_microduck.rom.viewer import export_geometry

    model = mujoco.MjModel.from_xml_string(
        """<mujoco><asset><mesh name="m" vertex="0 0 0 1 0 0 0 1 0 0 0 1" face="0 1 2 0 1 3 0 2 3 1 2 3"/></asset><worldbody><geom type="mesh" mesh="m"/></worldbody></mujoco>"""
    )
    result = export_geometry(model)
    np.testing.assert_array_equal(result[0]["vertices"], model.mesh_vert.ravel())
    np.testing.assert_array_equal(result[0]["indices"], model.mesh_face.ravel())
    model.geom_rgba[0, 0] = np.nan
    with pytest.raises(ValueError):
        export_geometry(model)
    huge = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>" + '<geom size=".1"/>' * 513 + "</worldbody></mujoco>"
    )
    with pytest.raises(ValueError):
        export_geometry(huge)


def test_viewer_endpoints_require_bearer_and_fail_closed_without_feed():
    app = create_app(None, bearer_token="test-viewer-token")
    with TestClient(app) as client:
        for resource in ("model", "frame"):
            assert client.get("/v1/viewer/" + resource).status_code == 401
            response = client.get(
                "/v1/viewer/" + resource,
                headers={"Authorization": "Bearer test-viewer-token"},
            )
            assert response.status_code == 503


def test_process_feed_authenticated_same_child_and_no_viewer_fault_quarantine(tmp_path):
    from mjlab_microduck.rom.process_supervisor import RuntimeProcessSupervisor
    from types import SimpleNamespace

    bundle = _write_verified_bundle(tmp_path / "bundle")
    supervisor = RuntimeProcessSupervisor(
        bundle_root=tmp_path / "bundle",
        bundle_digest=bundle.bundleDigest,
        operation_timeout_s=5,
    )
    try:
        supervisor.ensure_ready()
        assert callable(getattr(supervisor, "viewer_model", None)), (
            "supervisor must expose bounded viewer feed"
        )
        model = supervisor.viewer_model()
        pid = supervisor.snapshot().pid
        service = SimpleNamespace(
            viewer_model=supervisor.viewer_model,
            viewer_frame=supervisor.viewer_frame,
            tick=lambda: None,
            close=lambda: None,
        )
        app = create_app(service, bearer_token="viewer-test")
        with TestClient(app) as client:
            for resource in ("model", "frame"):
                assert client.get("/v1/viewer/" + resource).status_code == 401
                response = client.get(
                    "/v1/viewer/" + resource,
                    headers={"Authorization": "Bearer viewer-test"},
                )
                assert response.status_code == 200
                assert response.headers["cache-control"] == "no-store"
                assert response.json()["runtimeSession"] == model["runtimeSession"]
        request = _request().model_copy(
            update={
                "bundleDigest": bundle.bundleDigest,
                "parameters": {"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
                "leaseMs": 2000,
            }
        )
        supervisor.start(request)
        time.sleep(0.21)
        frame = supervisor.viewer_frame()
        assert frame["activeTaskId"] == request.taskId
        assert frame["runtimeSession"] == model["runtimeSession"]
        assert supervisor.snapshot().pid == pid
        supervisor.stop(request.taskId, "USER_CANCELLED")
        time.sleep(0.21)
        assert supervisor.viewer_frame()["activeTaskId"] is None
        assert supervisor.snapshot().child_healthy
    finally:
        supervisor.close()


def test_viewer_protocol_fragments_remain_below_control_limit():
    from mjlab_microduck.rom import process_protocol as protocol

    assert hasattr(protocol, "ViewerReplyPayload"), (
        "viewer requires bounded IPC fragment contract"
    )
    payload = protocol.ViewerReplyPayload(
        offset=0, total=8 * 1024 * 1024, token="a" * 32, text="x" * 24000
    )
    message = protocol.RuntimeMessage(
        kind="VIEWER_REPLY", generation=1, operationSequence=1, payload=payload
    )
    assert len(protocol.encode_packet(message)) < protocol.PACKET_MAX_BYTES == 65536
    assert protocol.decode_packet(protocol.encode_packet(message)) == message
    with pytest.raises(ValueError):
        protocol.ViewerReplyPayload(
            offset=0, total=8 * 1024 * 1024 + 1, token="a" * 32, text="x"
        )


def test_child_viewer_failure_does_not_touch_active_task_and_chunks_are_stable(
    tmp_path,
):
    import socket
    from types import SimpleNamespace
    from mjlab_microduck.rom.runtime_child import RuntimeChildHost
    from mjlab_microduck.rom.process_protocol import (
        RuntimeMessage,
        ViewerRequestPayload,
        decode_packet,
    )

    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    host = RuntimeChildHost(child)
    try:
        host._handle = object()
        host._task_id = "1" * 32
        host._runtime = SimpleNamespace(
            viewer_model=lambda: {"geometry": ["x" * 23999] * 4},
            viewer_model_text=lambda: json.dumps({"geometry": ["x" * 23999] * 4}),
        )

        def read(offset=0, token=None, sequence=1):
            request = RuntimeMessage(
                kind="VIEWER",
                generation=1,
                operationSequence=sequence,
                payload=ViewerRequestPayload(
                    resource="model", offset=offset, token=token
                ),
            )
            assert host._handle_message(request)
            return decode_packet(parent.recv(65537)).payload

        first = read()
        assert first.total > 65536
        pieces = [first.text]
        offset = len(first.text)
        while offset < first.total:
            next_piece = read(offset, first.token, len(pieces) + 1)
            assert next_piece.token == first.token
            pieces.append(next_piece.text)
            offset += len(next_piece.text)
        assert json.loads("".join(pieces)) == host._runtime.viewer_model()
        bad = read(offset=1, token="f" * 32, sequence=20)
        assert bad.unavailable
        assert host._handle is not None and host._task_id == "1" * 32
        assert not host._safety_requested.is_set()
    finally:
        parent.close()
        child.close()


def test_runtime_busy_viewer_is_nonblocking(tmp_path):
    import threading

    bundle = _write_verified_bundle(tmp_path / "bundle")
    runtime = MicroduckMujocoRuntime(tmp_path / "bundle", bundle, realtime=False)
    locked, release = threading.Event(), threading.Event()

    def hold():
        with runtime._lock:
            locked.set()
            release.wait(2)

    thread = threading.Thread(target=hold)
    thread.start()
    assert locked.wait(1)
    try:
        start = time.monotonic()
        with pytest.raises(ValueError):
            runtime.viewer_frame()
        assert time.monotonic() - start < 0.1
    finally:
        release.set()
        thread.join()


def test_collision_proxy_geometry_is_omitted_but_preserves_ids():
    from mjlab_microduck.rom.viewer import export_geometry

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><geom size=".1" group="3"/><geom size=".2"/></worldbody></mujoco>'
    )
    assert [geom["id"] for geom in export_geometry(model)] == [1]


def test_compiled_material_color_is_used():
    from mjlab_microduck.rom.viewer import export_geometry

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><asset><material name="red" rgba="1 0 0 1"/></asset><worldbody><geom size=".1" material="red"/></worldbody></mujoco>'
    )
    assert export_geometry(model)[0]["rgba"] == [1.0, 0.0, 0.0, 1.0]


def test_viewer_request_behind_blocked_control_has_bounded_wait():
    import threading
    from tests.test_rom_process_supervisor import _supervisor, _request as fake_request

    supervisor, _ = _supervisor("block-start", operation_timeout_s=2.0)
    supervisor.ensure_ready()
    dispatched, viewer_done = threading.Event(), threading.Event()

    def start():
        try:
            supervisor.start(fake_request(), register_dispatch=dispatched.set)
        except Exception:
            pass

    def view():
        try:
            supervisor.viewer_model()
        except Exception:
            pass
        finally:
            viewer_done.set()

    control = threading.Thread(target=start)
    viewing = threading.Thread(target=view)
    control.start()
    assert dispatched.wait(1)
    try:
        viewing.start()
        assert viewer_done.wait(0.3), (
            "viewer must not wait for a blocked control operation"
        )
    finally:
        supervisor.close()
        viewing.join(2)
        control.join(2)


def test_late_viewer_reply_cannot_poison_task_stop(tmp_path):
    import sys
    from mjlab_microduck.rom.process_supervisor import (
        ChildLaunch,
        RuntimeProcessSupervisor,
        SupervisorUnavailable,
    )

    bundle = _write_verified_bundle(tmp_path / "bundle")
    script = """
import socket,sys,time
from pathlib import Path
from mjlab_microduck.rom.runtime_child import RuntimeChildHost
from mjlab_microduck.rom.mujoco_runtime import MicroduckMujocoRuntime
class DelayedViewer(MicroduckMujocoRuntime):
    def viewer_frame(self):
        time.sleep(.18)
        return super().viewer_frame()
host=RuntimeChildHost(socket.socket(fileno=int(sys.argv[1])),bundle_root=Path(sys.argv[2]),runtime_factory=DelayedViewer)
raise SystemExit(host.run())
"""
    supervisor = RuntimeProcessSupervisor(
        bundle_root=tmp_path / "bundle",
        bundle_digest=bundle.bundleDigest,
        operation_timeout_s=5,
        launch_factory=lambda fd: ChildLaunch(
            argv=(sys.executable, "-c", script, str(fd), str(tmp_path / "bundle"))
        ),
    )
    try:
        supervisor.ensure_ready()
        request = _request().model_copy(
            update={
                "bundleDigest": bundle.bundleDigest,
                "parameters": {"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
                "leaseMs": 2000,
            }
        )
        supervisor.start(request)
        with pytest.raises(SupervisorUnavailable):
            supervisor.viewer_frame()
        assert supervisor.snapshot().child_healthy
        terminal = supervisor.stop(request.taskId, "USER_CANCELLED")
        assert terminal.outcome == "CANCELLED"
        assert supervisor.snapshot().child_healthy
        import os
        import signal

        old_session = supervisor.viewer_model()["runtimeSession"]
        old_generation = supervisor.snapshot().generation
        os.kill(supervisor.snapshot().pid, signal.SIGKILL)
        deadline = time.monotonic() + 2
        while supervisor.snapshot().child_healthy and time.monotonic() < deadline:
            time.sleep(0.01)
        supervisor.start(request)
        assert supervisor.snapshot().generation > old_generation
        assert supervisor.viewer_model()["runtimeSession"] != old_session
        supervisor.stop(request.taskId, "USER_CANCELLED")
    finally:
        supervisor.close()


def test_first_child_model_read_uses_preencoded_geometry(tmp_path, monkeypatch):
    import socket
    from mjlab_microduck.rom import viewer
    from mjlab_microduck.rom.runtime_child import RuntimeChildHost
    from mjlab_microduck.rom.process_protocol import (
        RuntimeMessage,
        ViewerRequestPayload,
        decode_packet,
    )

    bundle = _write_verified_bundle(tmp_path / "bundle")
    runtime = MicroduckMujocoRuntime(tmp_path / "bundle", bundle, realtime=False)
    encoded = runtime._viewer.model_text

    def prohibit_encoding(*args, **kwargs):
        raise AssertionError("static geometry must be serialized during initialization")

    monkeypatch.setattr(viewer, "encode_display", prohibit_encoding)
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    host = RuntimeChildHost(child)
    host._runtime = runtime
    try:
        message = RuntimeMessage(
            kind="VIEWER",
            generation=1,
            operationSequence=1,
            payload=ViewerRequestPayload(resource="model"),
        )
        assert host._handle_message(message)
        response = decode_packet(parent.recv(65537)).payload
        assert not response.unavailable
        assert response.text == encoded[: viewer.CHUNK_CHARS]
        assert response.total == len(encoded)
    finally:
        parent.close()
        child.close()
