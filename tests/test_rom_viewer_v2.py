"""Lossless shared meshes and explicit robot ownership from compiled MuJoCo."""

import base64
import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from mjlab_microduck.rom.viewer import RuntimeViewer, MODEL_MAX_BYTES


def shared_model():
    return mujoco.MjModel.from_xml_string("""<mujoco><asset><mesh name="m" vertex="0 0 0 1 0 0 0 1 0 0 0 1" face="0 1 2 0 1 3 0 2 3 1 2 3"/></asset><worldbody>
    <geom type="plane" size="2 2 .1"/>
    <body name="decoy"><geom type="mesh" mesh="m"/></body>
    <body name="trunk_base"><freejoint/><geom type="mesh" mesh="m"/>
      <body name="leg"><joint/><geom type="mesh" mesh="m" group="2"/><geom size=".1" group="3"/></body>
    </body></worldbody></mujoco>""")


def viewer(model):
    bundle = SimpleNamespace(
        bundleDigest="sha256:" + "a" * 64,
        model=SimpleNamespace(digest="sha256:" + "b" * 64),
    )
    return RuntimeViewer(model, bundle, trunk_body_id=model.body("trunk_base").id)


def test_meshes_are_shared_bitexact_and_robot_ids_use_ancestry():
    model = shared_model()
    result = viewer(model).model
    assert result["schema"] == "MICRODUCK_VIEWER_MODEL_V2"
    assert set(result) == {
        "schema",
        "runtimeSession",
        "bundleDigest",
        "modelDigest",
        "geometry",
        "meshes",
        "robotGeometryIds",
    }
    assert result["robotGeometryIds"] == [2, 3]
    assert len(result["meshes"]) == 1
    entry = result["meshes"][0]
    assert set(entry) == {"id", "verticesBase64", "indicesBase64", "indexType"}
    assert entry["id"] == 0 and entry["indexType"] == "uint16"
    vertices = base64.b64decode(entry["verticesBase64"], validate=True)
    faces = base64.b64decode(entry["indicesBase64"], validate=True)
    assert vertices == model.mesh_vert.astype("<f4").tobytes()
    np.testing.assert_array_equal(
        np.frombuffer(faces, dtype="<u2"), model.mesh_face.ravel()
    )
    for geom in result["geometry"]:
        expected = {"id", "type", "size", "rgba"} | (
            {"meshId"} if geom["type"] == 7 else set()
        )
        assert set(geom) == expected
    assert [g["meshId"] for g in result["geometry"] if g["type"] == 7] == [0, 0, 0]
    assert len(viewer(model).model_text.encode()) < MODEL_MAX_BYTES


@pytest.mark.parametrize(
    "bad", ["nan_vertex", "negative_face", "out_of_range_face", "bad_mesh_id"]
)
def test_compiled_mesh_data_is_validated(bad):
    model = shared_model()
    if bad == "nan_vertex":
        model.mesh_vert[0, 0] = np.nan
    elif bad == "negative_face":
        model.mesh_face[0, 0] = -1
    elif bad == "out_of_range_face":
        model.mesh_face[0, 0] = model.mesh_vertnum[0]
    else:
        model.geom_dataid[2] = model.nmesh
    with pytest.raises(ValueError):
        viewer(model)


def test_invalid_robot_root_is_rejected():
    model = shared_model()
    bundle = SimpleNamespace(
        bundleDigest="sha256:" + "a" * 64,
        model=SimpleNamespace(digest="sha256:" + "b" * 64),
    )
    with pytest.raises(ValueError):
        RuntimeViewer(model, bundle, trunk_body_id=0)


def test_uint32_indices_and_preallocation_bound():
    from mjlab_microduck.rom.viewer import export_meshes

    model = SimpleNamespace(
        mesh_vertadr=[0],
        mesh_vertnum=[65537],
        mesh_faceadr=[0],
        mesh_facenum=[1],
        mesh_vert=np.zeros((65537, 3), dtype=np.float32),
        mesh_face=np.array([[0, 65535, 65536]], dtype=np.int32),
    )
    mesh = export_meshes(model, [{"type": 7, "meshId": 0}])[0]
    assert mesh["indexType"] == "uint32"
    assert (
        base64.b64decode(mesh["indicesBase64"])
        == model.mesh_face.astype("<u4").tobytes()
    )
    model.mesh_vertnum = [500001]
    with pytest.raises(ValueError, match="bound"):
        export_meshes(model, [{"type": 7, "meshId": 0}])


def test_real_bundle_http_transfer_within_existing_timeout():
    import os
    import time
    from pathlib import Path
    from fastapi.testclient import TestClient
    from mjlab_microduck.rom.api import create_app
    from mjlab_microduck.rom.process_supervisor import RuntimeProcessSupervisor

    bundle_path = os.environ.get("MICRODUCK_VIEWER_REAL_BUNDLE")
    if not bundle_path:
        pytest.skip(
            "Set MICRODUCK_VIEWER_REAL_BUNDLE to verify external real-model assets"
        )
    root = Path(bundle_path)
    manifest = json.loads((root / "microduck-policy-bundle.json").read_text())
    supervisor = RuntimeProcessSupervisor(
        bundle_root=root, bundle_digest=manifest["bundleDigest"], operation_timeout_s=10
    )
    try:
        supervisor.ensure_ready()
        pid = supervisor.snapshot().pid
        service = SimpleNamespace(
            viewer_model=supervisor.viewer_model,
            viewer_frame=supervisor.viewer_frame,
            tick=lambda: None,
            close=lambda: None,
        )
        with TestClient(
            create_app(service, bearer_token="isolated-viewer-test")
        ) as client:
            headers = {"Authorization": "Bearer isolated-viewer-test"}
            started = time.monotonic()
            response = client.get("/v1/viewer/model", headers=headers)
            elapsed = time.monotonic() - started
            assert response.status_code == 200, response.text[:300]
            assert elapsed < 5
            assert len(response.content) < MODEL_MAX_BYTES
            model = response.json()
            assert model["schema"] == "MICRODUCK_VIEWER_MODEL_V2"
            navigation = root / "navigation.json"
            obstacle_count = (
                len(json.loads(navigation.read_text())["scene"]["obstacles"])
                if navigation.is_file()
                else 0
            )
            assert (
                len(model["geometry"]) == 71 + obstacle_count
                and len(model["meshes"]) == 38
            )
            assert len(model["robotGeometryIds"]) == 70
            compiled = mujoco.MjModel.from_xml_path(
                str(root / manifest["model"]["path"])
            )
            for mesh in model["meshes"]:
                mesh_id = mesh["id"]
                va, vn = compiled.mesh_vertadr[mesh_id], compiled.mesh_vertnum[mesh_id]
                fa, fn = compiled.mesh_faceadr[mesh_id], compiled.mesh_facenum[mesh_id]
                assert (
                    base64.b64decode(mesh["verticesBase64"], validate=True)
                    == compiled.mesh_vert[va : va + vn].astype("<f4").tobytes()
                )
                assert (
                    base64.b64decode(mesh["indicesBase64"], validate=True)
                    == compiled.mesh_face[fa : fa + fn].astype("<u2").tobytes()
                )
            frame = client.get("/v1/viewer/frame", headers=headers).json()
            assert len(frame["positions"]) == (76 + obstacle_count) * 3
            if obstacle_count:
                environment = [
                    g
                    for g in model["geometry"]
                    if g["id"] not in model["robotGeometryIds"]
                ]
                assert len(environment) == 1 + obstacle_count
                assert sum(g["type"] == 6 for g in environment) == obstacle_count
            assert frame["runtimeSession"] == model["runtimeSession"]
            assert (
                supervisor.snapshot().pid == pid and supervisor.snapshot().child_healthy
            )
            # Public display data only, for cross-layer validator replay.
            Path("/tmp/microduck-real-viewer-v2-model.json").write_bytes(
                response.content
            )
            Path("/tmp/microduck-real-viewer-v2-frame.json").write_text(
                json.dumps(frame)
            )
            print(f"real model HTTP: {len(response.content)} bytes in {elapsed:.3f}s")
            from tests.test_rom_mujoco_runtime import _request

            request = _request().model_copy(
                update={
                    "bundleDigest": manifest["bundleDigest"],
                    "bundleVersion": manifest["bundleVersion"],
                    "parameters": {"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
                    "leaseMs": 5000,
                }
            )
            supervisor.start(request)
            time.sleep(0.21)
            running = client.get("/v1/viewer/frame", headers=headers)
            assert running.status_code == 200
            assert running.json()["activeTaskId"] == request.taskId
            assert client.get("/v1/viewer/model", headers=headers).json() == model
            supervisor.stop(request.taskId, "USER_CANCELLED")
            time.sleep(0.21)
            assert (
                client.get("/v1/viewer/frame", headers=headers).json()["activeTaskId"]
                is None
            )
            assert (
                supervisor.snapshot().pid == pid and supervisor.snapshot().child_healthy
            )
    finally:
        supervisor.close()


def test_combined_mesh_payload_bound():
    from mjlab_microduck.rom.viewer import export_meshes

    model = SimpleNamespace(
        mesh_vertadr=[0, 0],
        mesh_vertnum=[300000, 300000],
        mesh_faceadr=[0, 0],
        mesh_facenum=[1, 1],
        mesh_vert=np.zeros((300000, 3), dtype=np.float32),
        mesh_face=np.array([[0, 1, 2]], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="meshes exceed bound"):
        export_meshes(model, [{"type": 7, "meshId": 0}, {"type": 7, "meshId": 1}])


def test_mesh_requires_at_least_three_vertices():
    from mjlab_microduck.rom.viewer import export_meshes

    model = SimpleNamespace(
        mesh_vertadr=[0],
        mesh_vertnum=[2],
        mesh_faceadr=[0],
        mesh_facenum=[1],
        mesh_vert=np.zeros((2, 3), dtype=np.float32),
        mesh_face=np.array([[0, 1, 0]], dtype=np.int32),
    )
    with pytest.raises(ValueError, match="component bound"):
        export_meshes(model, [{"type": 7, "meshId": 0}])
