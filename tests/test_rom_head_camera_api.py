import io
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

from mjlab_microduck.rom.api import create_app
from mjlab_microduck.rom.head_camera import CameraFrame


def test_camera_route_is_authenticated_and_returns_provenance():
    output = io.BytesIO()
    Image.new("RGB", (640, 480), "red").save(output, format="JPEG")
    frame = CameraFrame(
        {
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
        },
        output.getvalue(),
    )
    service = SimpleNamespace(
        viewer_camera=lambda: frame, tick=lambda: None, close=lambda: None
    )
    app = create_app(service, bearer_token="test-viewer-token")
    with TestClient(app) as client:
        denied = client.get("/v1/viewer/camera")
        assert denied.status_code == 401
        assert denied.headers["cache-control"] == "no-store"
        response = client.get(
            "/v1/viewer/camera", headers={"Authorization": "Bearer test-viewer-token"}
        )
        assert response.status_code == 200
        assert response.content == frame.image
        assert response.headers["content-type"] == "image/jpeg"
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-runtime-session"] == "session"
        assert response.headers["x-frame-sequence"] == "1"
