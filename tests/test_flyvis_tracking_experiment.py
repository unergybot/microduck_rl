import json

import numpy as np
import pytest

from mjlab_microduck.rom.flyvis_tracking.core import (
    PHYSICS_PROTOCOL,
    RANDOMIZATION_PROTOCOL,
)


def test_paused_or_unversioned_readout_is_rejected_before_simulation():
    from types import SimpleNamespace

    from mjlab_microduck.rom.flyvis_tracking.experiment import run_episode

    for metadata in ({}, {"startupProtocol": "standing_pause"}):
        with pytest.raises(ValueError, match="startup protocol"):
            run_episode(None, None, None, readout=SimpleNamespace(metadata=metadata))


def test_shared_randomization_readout_is_rejected_before_simulation():
    from types import SimpleNamespace

    from mjlab_microduck.rom.flyvis_tracking.experiment import run_episode

    with pytest.raises(ValueError, match="randomization protocol"):
        run_episode(
            None,
            None,
            None,
            readout=SimpleNamespace(
                metadata={"startupProtocol": "immediate_sensor_control"}
            ),
        )


def test_legacy_episode_cannot_be_mixed_into_current_training_data(tmp_path):
    from mjlab_microduck.rom.flyvis_tracking.experiment import load_dataset

    path = tmp_path / "legacy.npz"
    np.savez(
        path,
        neural=np.ones((1, 9)),
        retina=np.ones((1, 721)),
        body=np.ones((1, 34)),
        labels=np.array([1]),
        metadata=json.dumps(
            {"partition": "train", "startupProtocol": "immediate_sensor_control"}
        ),
    )
    with pytest.raises(ValueError, match="randomization protocol"):
        load_dataset([path], "flyvis", "train")


def test_dataset_loader_rejects_heldout_data_and_uses_only_sensor_features(tmp_path):
    from mjlab_microduck.rom.flyvis_tracking.experiment import load_dataset

    path = tmp_path / "episode.npz"
    np.savez(
        path,
        neural=np.ones((2, 9)),
        retina=np.ones((2, 721)),
        body=np.ones((2, 34)),
        labels=np.array([1, 2]),
        metadata=json.dumps({"partition": "test"}),
    )
    with pytest.raises(ValueError, match="partition"):
        load_dataset([path], "flyvis", "train")
    np.savez(
        path,
        neural=np.ones((2, 9)),
        retina=np.ones((2, 721)),
        body=np.ones((2, 34)),
        labels=np.array([1, 2]),
        target_pose=np.full((2, 3), 123),
        metadata=json.dumps(
            {
                "partition": "train",
                "startupProtocol": "immediate_sensor_control",
                "randomizationProtocol": RANDOMIZATION_PROTOCOL,
                "physicsProtocol": PHYSICS_PROTOCOL,
            }
        ),
    )
    x, y = load_dataset([path], "flyvis", "train")
    assert x.shape == (2, 43)
    assert np.all(x == 1)
    assert y.tolist() == [1, 2]


def test_metrics_measure_loss_stop_and_reacquisition():
    from mjlab_microduck.rom.flyvis_tracking.experiment import temporal_metrics

    rows = [
        {
            "time": t,
            "visible": v,
            "requested": a,
            "speed": s,
            "yaw_rate": 0.0,
            "bearing": 0.0,
            "distance": 0.6,
        }
        for t, v, a, s in [
            (3.0, True, 1, 0.03),
            (3.04, False, 1, 0.03),
            (3.08, False, 0, 0.03),
            (3.6, False, 0, 0.0),
            (4.12, False, 0, 0.0),
            (5.0, True, 2, 0.0),
        ]
    ]
    metrics = temporal_metrics(rows)
    assert metrics["lossStopCommandLatencyS"] == pytest.approx([0.04])
    assert metrics["lossSettlementLatencyS"] == pytest.approx([1.08])
    assert metrics["reacquisitionLatencyS"] == [0.0]


@pytest.mark.parametrize("physics", [None, "live_forward_v0"])
def test_live_forward_physics_artifacts_are_rejected(physics):
    from mjlab_microduck.rom.flyvis_tracking.core import RANDOMIZATION_PROTOCOL
    from mjlab_microduck.rom.flyvis_tracking.experiment import validate_protocol

    legacy = {
        "startupProtocol": "immediate_sensor_control",
        "randomizationProtocol": RANDOMIZATION_PROTOCOL,
    }
    if physics is not None:
        legacy["physicsProtocol"] = physics
    with pytest.raises(ValueError, match="physics protocol"):
        validate_protocol(legacy)


@pytest.mark.parametrize("kind", ["flyvis", "retina"])
@pytest.mark.parametrize("changed", ["bundleDigest", "vision", "missing"])
def test_dataset_rejects_episode_identity_mismatch(tmp_path, kind, changed):
    from mjlab_microduck.rom.flyvis_tracking.experiment import load_dataset

    vision = {
        "extent": 15,
        "kernelSize": 13,
        "grayscaleWeights": [0.299, 0.587, 0.114],
        "retinaSites": 721,
        "checkpoint": "expected",
    }
    identity = {"bundleDigest": "expected-bundle", "vision": vision}
    metadata = {
        "partition": "train",
        "startupProtocol": "immediate_sensor_control",
        "randomizationProtocol": RANDOMIZATION_PROTOCOL,
        "physicsProtocol": PHYSICS_PROTOCOL,
        **identity,
    }
    if changed == "bundleDigest":
        metadata["bundleDigest"] = "another-bundle"
    elif changed == "missing":
        metadata.pop("vision")
    else:
        metadata["vision"] = dict(vision, kernelSize=7)
    path = tmp_path / "episode.npz"
    np.savez(
        path,
        neural=np.ones((1, 9)),
        retina=np.ones((1, 721)),
        body=np.ones((1, 34)),
        labels=np.array([1]),
        metadata=json.dumps(metadata),
    )
    with pytest.raises(ValueError, match="identity"):
        load_dataset([path], kind, "train", identity=identity)


def test_retinal_dataset_ignores_neural_checkpoint_but_flyvis_rejects_it(tmp_path):
    from mjlab_microduck.rom.flyvis_tracking.experiment import load_dataset

    vision = {
        "extent": 15,
        "kernelSize": 13,
        "grayscaleWeights": [0.299, 0.587, 0.114],
        "retinaSites": 721,
        "checkpoint": "expected",
    }
    identity = {"bundleDigest": "same-bundle", "vision": vision}
    metadata = {
        "partition": "validation",
        "startupProtocol": "immediate_sensor_control",
        "randomizationProtocol": RANDOMIZATION_PROTOCOL,
        "physicsProtocol": PHYSICS_PROTOCOL,
        **identity,
        "vision": dict(vision, checkpoint="other"),
    }
    path = tmp_path / "episode.npz"
    np.savez(
        path,
        neural=np.ones((1, 9)),
        retina=np.ones((1, 721)),
        body=np.ones((1, 34)),
        labels=np.array([1]),
        metadata=json.dumps(metadata),
    )
    x, y = load_dataset([path], "retina", "validation", identity=identity)
    assert x.shape == (1, 755) and y.tolist() == [1]
    with pytest.raises(ValueError, match="identity"):
        load_dataset([path], "flyvis", "validation", identity=identity)
