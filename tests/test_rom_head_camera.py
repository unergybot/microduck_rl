"""The ROM observer returns actual MuJoCo pixels without owning control state."""

import time

import mujoco
import numpy as np
import pytest

from mjlab_microduck.rom.head_camera import CameraPoseSnapshot, HeadCameraObserver
from mjlab_microduck.rom.mujoco_runtime import MicroduckMujocoRuntime
from tests.test_rom_mujoco_runtime import _write_verified_bundle

MODEL = """<mujoco><visual><global offwidth="640" offheight="480"/></visual>
<worldbody><light pos="0 0 3"/><camera name="head_camera" pos="0 -2 1" xyaxes="1 0 0 0 0 1"/>
<body name="target" pos="0 0 1"><freejoint/><geom type="sphere" size=".2" rgba="1 0 0 1"/></body>
</worldbody></mujoco>"""


def test_camera_wire_payload_rejects_oversize_or_wrong_dimensions():
    import io

    from PIL import Image

    from mjlab_microduck.rom.head_camera import (
        CameraFrame,
        decode_camera,
        encode_camera,
    )

    metadata = {
        "schema": "MICRODUCK_HEAD_CAMERA_FRAME_V1",
        "runtimeSession": "session",
        "modelDigest": "sha256:" + "1" * 64,
        "bundleDigest": "sha256:" + "2" * 64,
        "capturedAt": "2026-09-26T00:00:00Z",
        "sequence": 1,
        "activeTaskId": None,
        "width": 640,
        "height": 480,
        "mimeType": "image/jpeg",
    }
    output = io.BytesIO()
    Image.new("RGB", (640, 480), "red").save(output, format="JPEG")
    frame = CameraFrame(metadata, output.getvalue())
    assert decode_camera(encode_camera(frame)).image == frame.image
    with pytest.raises(ValueError):
        encode_camera(CameraFrame(metadata, b"x" * 262145))
    wrong = io.BytesIO()
    Image.new("RGB", (320, 240), "red").save(wrong, format="JPEG")
    with pytest.raises(ValueError):
        encode_camera(CameraFrame(metadata, wrong.getvalue()))
    gradient = Image.new("RGB", (640, 480))
    gradient.putdata([(i % 256, (i // 256) % 256, (i * 7) % 256) for i in range(640 * 480)])
    damaged = io.BytesIO()
    gradient.save(damaged, format="JPEG", quality=75)
    corrupt = bytearray(damaged.getvalue())
    corrupt[43445:43447] = b"\xff\xd8"
    with pytest.raises(ValueError):
        encode_camera(CameraFrame(metadata, bytes(corrupt)))


def test_observer_renders_current_pose_from_own_model(tmp_path):
    path = tmp_path / "scene.xml"
    path.write_text(MODEL)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    observer = HeadCameraObserver(
        path,
        {
            "runtimeSession": "test",
            "modelDigest": "sha256:" + "1" * 64,
            "bundleDigest": "sha256:" + "2" * 64,
        },
    )
    try:
        observer.activate()
        observer.submit(
            CameraPoseSnapshot(
                data.qpos.copy(), data.mocap_pos.copy(), data.mocap_quat.copy(), None
            )
        )
        deadline = time.monotonic() + 3
        while observer.latest() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        first = observer.latest()
        assert first is not None
        assert first.metadata["width"] == 640 and first.metadata["height"] == 480
        assert first.image.startswith(b"\xff\xd8") and first.image.endswith(b"\xff\xd9")
        assert len(first.image) <= 262144
        data.qpos[0] = 0.5
        control_qpos = data.qpos.copy()
        observer.submit(
            CameraPoseSnapshot(
                data.qpos.copy(),
                data.mocap_pos.copy(),
                data.mocap_quat.copy(),
                "a" * 32,
            )
        )
        deadline = time.monotonic() + 3
        while (
            observer.latest() is first
            or observer.latest().metadata["sequence"] == first.metadata["sequence"]
        ) and time.monotonic() < deadline:
            time.sleep(0.02)
        second = observer.latest()
        assert second is not None and second.image != first.image
        assert second.metadata["activeTaskId"] == "a" * 32
        np.testing.assert_array_equal(data.qpos, control_qpos)
    finally:
        observer.close()


def test_runtime_camera_read_only_queues_pose_and_returns_cached_frame(
    tmp_path, monkeypatch
):
    from mjlab_microduck.rom import head_camera

    class FakeObserver:
        def __init__(self, path, identity):
            self.identity = identity
            self.pose = None
            self.activated = False

        def activate(self):
            self.activated = True

        def submit(self, snapshot):
            self.pose = snapshot.qpos.copy()

        def latest(self):
            return head_camera.CameraFrame(
                {"schema": "MICRODUCK_HEAD_CAMERA_FRAME_V1"}, b"\xff\xd8\xff\xd9"
            )

    monkeypatch.setenv("ROM_MICRODUCK_HEAD_CAMERA_ENABLED", "true")
    monkeypatch.setattr(head_camera, "HeadCameraObserver", FakeObserver)
    bundle = _write_verified_bundle(tmp_path / "bundle")
    runtime = MicroduckMujocoRuntime(tmp_path / "bundle", bundle, realtime=False)
    before = runtime._data.qpos.copy()
    frame = runtime.viewer_camera()
    assert frame.image == b"\xff\xd8\xff\xd9"
    assert runtime._head_camera.activated
    np.testing.assert_array_equal(runtime._head_camera.pose, before)
    np.testing.assert_array_equal(runtime._data.qpos, before)
    runtime._head_camera.latest = lambda: head_camera.CameraFrame(
        {"activeTaskId": "a" * 32}, b"\xff\xd8\xff\xd9"
    )
    with pytest.raises(ValueError, match="camera frame unavailable"):
        runtime.viewer_camera()
