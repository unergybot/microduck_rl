"""Best-effort MuJoCo camera observer; no control state is ever shared."""

from __future__ import annotations

import base64
import io
import re
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

WIDTH = 640
HEIGHT = 480
MAX_JPEG_BYTES = 262144
FRAME_PERIOD_S = 0.5
DEMAND_TTL_S = 10.0


@dataclass(frozen=True, slots=True)
class CameraPoseSnapshot:
    qpos: np.ndarray
    mocap_pos: np.ndarray
    mocap_quat: np.ndarray
    active_task_id: str | None


@dataclass(frozen=True, slots=True)
class CameraFrame:
    metadata: dict[str, object]
    image: bytes


def _valid_image(image: bytes) -> None:
    if (
        not 0 < len(image) <= MAX_JPEG_BYTES
        or not image.startswith(b"\xff\xd8")
        or not image.endswith(b"\xff\xd9")
    ):
        raise ValueError("invalid camera image")
    try:
        with Image.open(io.BytesIO(image)) as decoded:
            if (
                decoded.format != "JPEG"
                or decoded.size != (WIDTH, HEIGHT)
                or decoded.mode != "RGB"
            ):
                raise ValueError("invalid camera dimensions")
            decoded.load()
    except (OSError, SyntaxError) as exc:
        raise ValueError("invalid camera image") from exc


def encode_camera(frame: CameraFrame) -> dict[str, object]:
    _valid_image(frame.image)
    return dict(
        frame.metadata, imageBase64=base64.b64encode(frame.image).decode("ascii")
    )


def decode_camera(payload: dict[str, object]) -> CameraFrame:
    expected = {
        "schema",
        "runtimeSession",
        "modelDigest",
        "bundleDigest",
        "capturedAt",
        "sequence",
        "activeTaskId",
        "width",
        "height",
        "mimeType",
        "imageBase64",
    }
    if (
        set(payload) != expected
        or payload["schema"] != "MICRODUCK_HEAD_CAMERA_FRAME_V1"
    ):
        raise ValueError("invalid camera contract")
    for field in ("modelDigest", "bundleDigest"):
        if not isinstance(payload[field], str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", payload[field]
        ):
            raise ValueError("invalid camera identity")
    if not isinstance(payload["runtimeSession"], str) or not re.fullmatch(
        r"[A-Za-z0-9:_-]{1,128}", payload["runtimeSession"]
    ):
        raise ValueError("invalid camera session")
    if (
        payload["width"] != WIDTH
        or payload["height"] != HEIGHT
        or payload["mimeType"] != "image/jpeg"
    ):
        raise ValueError("invalid camera format")
    if (
        type(payload["sequence"]) is not int
        or not 0 <= payload["sequence"] <= 9007199254740991
    ):
        raise ValueError("invalid camera sequence")
    task = payload["activeTaskId"]
    if task is not None and (
        not isinstance(task, str) or not re.fullmatch(r"[0-9a-f]{32}", task)
    ):
        raise ValueError("invalid camera task")
    if not isinstance(payload["capturedAt"], str):
        raise TypeError("invalid camera time")
    if not payload["capturedAt"].endswith("Z"):
        raise ValueError("invalid camera time")
    datetime.fromisoformat(payload["capturedAt"])
    encoded = payload["imageBase64"]
    if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_JPEG_BYTES + 2) // 3):
        raise ValueError("invalid camera encoding")
    try:
        image = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("invalid camera encoding") from exc
    _valid_image(image)
    return CameraFrame(
        {key: value for key, value in payload.items() if key != "imageBase64"}, image
    )


class HeadCameraObserver:
    def __init__(self, model_path: Path, identity: dict[str, str]):
        self._model_path = Path(model_path)
        self._identity = dict(identity)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closed = False
        self._requested_until = 0.0
        self._pending: CameraPoseSnapshot | None = None
        self._frame: CameraFrame | None = None
        self._sequence = 0
        self._thread: threading.Thread | None = None

    def activate(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._requested_until = time.monotonic() + DEMAND_TTL_S
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run, name="microduck-head-camera", daemon=True
                )
                self._thread.start()

    def submit(self, snapshot: CameraPoseSnapshot) -> None:
        with self._lock:
            if self._closed or time.monotonic() > self._requested_until:
                return
            self._pending = CameraPoseSnapshot(
                snapshot.qpos.copy(),
                snapshot.mocap_pos.copy(),
                snapshot.mocap_quat.copy(),
                snapshot.active_task_id,
            )
            self._wake.set()

    def latest(self) -> CameraFrame | None:
        with self._lock:
            return self._frame

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._wake.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)

    def _run(self) -> None:
        renderer = None
        try:
            model = mujoco.MjModel.from_xml_path(str(self._model_path))
            data = mujoco.MjData(model)
            camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
            if camera < 0:
                return
            renderer = mujoco.Renderer(model, width=WIDTH, height=HEIGHT)
            last_render = float("-inf")
            while True:
                self._wake.wait(timeout=0.5)
                self._wake.clear()
                with self._lock:
                    if self._closed:
                        break
                    pending = self._pending
                    self._pending = None
                    demanded = time.monotonic() <= self._requested_until
                if pending is None or not demanded:
                    continue
                delay = FRAME_PERIOD_S - (time.monotonic() - last_render)
                if delay > 0:
                    time.sleep(delay)
                with self._lock:
                    if self._closed:
                        break
                    if self._pending is not None:
                        pending, self._pending = self._pending, None
                data.qpos[:] = pending.qpos
                data.mocap_pos[:] = pending.mocap_pos
                data.mocap_quat[:] = pending.mocap_quat
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                rgb = renderer.render().copy()
                output = io.BytesIO()
                Image.fromarray(rgb, "RGB").save(output, format="JPEG", quality=75)
                image = output.getvalue()
                if not 0 < len(image) <= MAX_JPEG_BYTES:
                    continue
                last_render = time.monotonic()
                with self._lock:
                    self._frame = CameraFrame(
                        dict(
                            schema="MICRODUCK_HEAD_CAMERA_FRAME_V1",
                            **self._identity,
                            capturedAt=datetime.now(UTC)
                            .isoformat()
                            .replace("+00:00", "Z"),
                            sequence=self._sequence,
                            activeTaskId=pending.active_task_id,
                            width=WIDTH,
                            height=HEIGHT,
                            mimeType="image/jpeg",
                        ),
                        image,
                    )
                    self._sequence += 1
        except Exception:  # noqa: BLE001 - camera failure must not affect control.
            # EGL and display faults are deliberately nonfatal to navigation.
            return
        finally:
            if renderer is not None:
                renderer.close()
