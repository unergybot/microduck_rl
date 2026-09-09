from fastapi.testclient import TestClient
from mjlab_microduck.rom.api import create_app


def test_v2_is_private_and_missing_navigation_does_not_enable_it():
    with TestClient(create_app(None, "private-test-token")) as client:
        assert client.get("/v2/navigation/capabilities").status_code == 401
        response = client.get(
            "/v2/navigation/capabilities",
            headers={"Authorization": "Bearer private-test-token"},
        )
        assert response.status_code == 200 and response.json()["ready"] is False
        assert (
            client.post(
                "/v2/navigation/tasks",
                headers={"Authorization": "Bearer private-test-token"},
                content=b"x" * 65537,
            ).status_code
            == 413
        )
        assert client.get("/v1/health").status_code == 200
