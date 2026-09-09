"""Stop evidence must remain truthful and bounded for long manifest provenance."""

import pytest

from mjlab_microduck.rom.contracts import canonical_json
from mjlab_microduck.rom.mujoco_runtime import MicroduckMujocoRuntime
from tests.test_rom_mujoco_runtime import _request, _write_verified_bundle


def test_long_provenance_does_not_prevent_stop_or_drop_measurements(tmp_path):
    bundle = _write_verified_bundle(tmp_path)
    runtime = MicroduckMujocoRuntime(tmp_path, bundle, realtime=False)
    request = _request().model_copy(update={"bundleDigest": bundle.bundleDigest})
    action = next(a for a in bundle.actions if a.actionCode == "WALK_VELOCITY")
    handle = runtime.start(action, request)
    runtime.sample(handle)
    # Manifest paths can exceed the scalar metric limit, independently of motion.
    runtime._active_policy = runtime._active_policy.model_copy(
        update={"experimentRef": "local/" + "r" * 180}
    )
    measured = runtime._action_metrics_locked()
    evidence = runtime.safe_stop(handle, "CANCELLED")
    assert len(canonical_json(evidence.metrics)) <= 1024
    assert all(evidence.metrics[k] == v for k, v in measured.items())
    assert "runIdentity" not in evidence.metrics
    assert evidence.metrics["provenanceDigest"].startswith("sha256:")
    assert runtime.status().activeTaskId is None
    assert runtime.safe_stop(handle, "CANCELLED") == evidence


def test_compact_identity_verification_rejects_tampering_and_mixed_formats():
    from mjlab_microduck.rom.runtime import (
        compact_runtime_evidence,
        runtime_evidence_identity_matches,
    )

    original = {
        "actionCode": "WALK_VELOCITY",
        "bundleDigest": "sha256:" + "a" * 64,
        "onnxDigest": "sha256:" + "b" * 64,
        "mjcfDigest": "sha256:" + "c" * 64,
        "sourceCommit": "d" * 40,
        "checkpoint": "model_3399.pt",
        "runIdentity": "local/" + "r" * 180,
        "terrainIdentity": "flat",
        "rngSeed": 7,
        "scenarioProfile": "SEEDED_SERVO_RESET_V1",
        "resetProfile": "DEFAULT_STANDING",
    }
    compact = compact_runtime_evidence(original)
    assert runtime_evidence_identity_matches(compact, original)
    for key, value in original.items():
        changed = original | {key: value + 1 if isinstance(value, int) else value + "x"}
        assert not runtime_evidence_identity_matches(compact, changed), key
    assert not runtime_evidence_identity_matches(compact | {"rngSeed": 7}, original)
    assert not runtime_evidence_identity_matches(
        compact | {"provenanceDigest": "sha256:" + "0" * 64}, original
    )
    with pytest.raises(ValueError):
        compact_runtime_evidence(original | {"measurement": "x" * 129})


def test_small_evidence_retains_legacy_representation():
    from mjlab_microduck.rom.runtime import compact_runtime_evidence

    small = {"steps": 1, "fallen": False}
    assert compact_runtime_evidence(small) == small


def test_byte_overflow_compacts_identity_without_relaxing_scalar_limits():
    from mjlab_microduck.rom.runtime import RuntimeEvidence, compact_runtime_evidence

    original = {
        "actionCode": "WALK_VELOCITY",
        "bundleDigest": "sha256:" + "a" * 64,
        "onnxDigest": "sha256:" + "b" * 64,
        "mjcfDigest": "sha256:" + "c" * 64,
        "sourceCommit": "d" * 40,
        "checkpoint": "model_3399.pt",
        "runIdentity": "r" * 115,
        "terrainIdentity": "flat",
        "rngSeed": 7,
        "scenarioProfile": "SEEDED_SERVO_RESET_V1",
        "resetProfile": "DEFAULT_STANDING",
        **{f"measured{i}": "x" * 120 for i in range(3)},
    }
    assert len(canonical_json(original)) > 1024
    assert all(not isinstance(v, str) or len(v) <= 128 for v in original.values())
    with pytest.raises(ValueError, match="encoded size"):
        RuntimeEvidence(metrics=original)
    compact = compact_runtime_evidence(original)
    assert len(canonical_json(compact)) <= 1024
    assert all(compact[f"measured{i}"] == original[f"measured{i}"] for i in range(3))
    assert RuntimeEvidence(metrics=compact).metrics == compact
