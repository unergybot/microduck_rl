import math

import pytest
from pydantic import ValidationError

from mjlab_microduck.rom.navigation_contracts import (
    Area,
    NavigationLeaseRequest,
    NavigationProfile,
    Pose,
    digest,
)


def test_renewal_rejects_motion_intent_and_nonincreasing_type():
    valid = {"taskId": "a" * 32, "proposalDigest": "sha256:" + "b" * 64, "sequence": 1}
    assert NavigationLeaseRequest(**valid).sequence == 1
    for extra in (
        {"landmarkId": "door"},
        {"velocity": 0.1},
        {"leaseMs": 10000},
        {"sequence": True},
        {"sequence": 0},
    ):
        with pytest.raises(ValidationError):
            NavigationLeaseRequest(**(valid | extra))


@pytest.mark.parametrize("value", [math.nan, math.inf, True, "0.2"])
def test_pose_rejects_unusable_measurements(value):
    with pytest.raises(ValidationError):
        Pose(x=value, y=0.0, yaw=0.0)


def test_area_rejects_inverted_boundary():
    with pytest.raises(ValidationError):
        Area(minX=2.0, maxX=1.0, minY=0.0, maxY=1.0)


def test_canonical_digest_normalizes_numbers_and_key_order():
    assert digest({"x": 1, "label": "门"}) == digest({"label": "门", "x": 1.0})
    assert digest({"x": -0.0}) == digest({"x": 0})
    assert digest({"x": 0.001}) != digest({"x": 0.01})
    with pytest.raises(ValueError):
        digest({"x": math.nan})


def test_profile_never_accepts_sideways_or_unbounded_motion():
    fields = {
        "robotRadiusM": 0.15,
        "clearanceM": 0.05,
        "gridResolutionM": 0.05,
        "maxSpeedMps": 0.05,
        "maxYawRateRadps": 0.2,
        "poseFreshnessMs": 200,
        "arrivalToleranceM": 0.08,
        "headingToleranceRad": 0.15,
        "settleMs": 500,
        "stableSpeedMps": 0.02,
        "stableYawRateRadps": 0.05,
        "deadlineMs": 120000,
        "leaseMs": 1000,
    }
    assert NavigationProfile(**fields).maxSpeedMps == 0.05
    for patch in (
        {"maxSpeedMps": 0},
        {"leaseMs": 0},
        {"sideways": True},
        {"deadlineMs": 500},
    ):
        with pytest.raises(ValidationError):
            NavigationProfile(**(fields | patch))


def test_proposal_tamper_is_rejected():
    import json
    from pathlib import Path

    from mjlab_microduck.rom.navigation_contracts import NavigationTaskRequest

    value = json.loads(Path("tests/fixtures/navigation/wire.json").read_text())["task"]
    assert NavigationTaskRequest.model_validate(value).proposal.landmarkId == "desk"
    value["proposal"]["destination"]["x"] = 1.5
    with pytest.raises(ValidationError):
        NavigationTaskRequest.model_validate(value)


def test_calibrated_profile_accepts_walking_limits_and_rejects_excess():
    import json
    from pathlib import Path

    original = json.loads(Path("tests/fixtures/navigation/candidate.json").read_text())[
        "profile"
    ]
    calibrated = original | {"maxSpeedMps": 0.4, "maxYawRateRadps": 1.0}
    assert NavigationProfile.model_validate(calibrated).maxSpeedMps == 0.4
    assert digest(calibrated) != digest(original)
    for invalid in ({"maxSpeedMps": 0.4001}, {"maxYawRateRadps": 1.0001}):
        with pytest.raises(ValidationError):
            NavigationProfile.model_validate(calibrated | invalid)
