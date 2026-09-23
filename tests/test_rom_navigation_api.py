from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from mjlab_microduck.rom.api import create_app
from mjlab_microduck.rom.service import (
    NavigationAuthorizationChanged,
    NavigationIdentityMismatch,
    NavigationTaskNotRunning,
)


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


@pytest.mark.parametrize(
    ("failure", "code"),
    [
        (NavigationIdentityMismatch("secret detail"), "NAVIGATION_IDENTITY_MISMATCH"),
        (
            NavigationAuthorizationChanged("secret detail"),
            "NAVIGATION_AUTHORIZATION_CHANGED",
        ),
        (NavigationTaskNotRunning("secret detail"), "NAVIGATION_TASK_NOT_RUNNING"),
    ],
)
def test_lease_rejection_has_safe_specific_code(failure, code):
    def reject(*_args):
        raise failure

    service = SimpleNamespace(
        navigation=SimpleNamespace(renew_lease=reject),
        tick=lambda: None,
        close=lambda: None,
    )
    with TestClient(create_app(service, "private-test-token")) as client:
        response = client.put(
            "/v2/navigation/tasks/" + "a" * 32 + "/lease",
            headers={"Authorization": "Bearer private-test-token"},
            json={
                "taskId": "a" * 32,
                "proposalDigest": "sha256:" + "b" * 64,
                "sequence": 1,
            },
        )
    assert response.status_code == 400
    assert response.json()["code"] == code
    assert response.json()["details"] == {}
    assert "secret detail" not in response.text


def test_malformed_lease_is_distinguished_from_runtime_rejection():
    with TestClient(create_app(None, "private-test-token")) as client:
        response = client.put(
            "/v2/navigation/tasks/" + "a" * 32 + "/lease",
            headers={"Authorization": "Bearer private-test-token"},
            json={"taskId": "a" * 32, "proposalDigest": "bad", "sequence": 0},
        )
    assert response.status_code == 400
    assert response.json()["code"] == "NAVIGATION_LEASE_MALFORMED"
