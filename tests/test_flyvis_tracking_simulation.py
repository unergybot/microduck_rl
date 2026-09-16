"""Actual MuJoCo integration, enabled with MICRODUCK_TRACKING_BUNDLE."""

import os
from pathlib import Path

import numpy as np
import pytest

from mjlab_microduck.rom.flyvis_tracking.core import EpisodeSpec


@pytest.fixture
def source_bundle():
    value = os.environ.get("MICRODUCK_TRACKING_BUNDLE")
    if not value:
        pytest.skip("set MICRODUCK_TRACKING_BUNDLE to a verified walking bundle")
    return Path(value)


def test_real_camera_body_feedback_and_bundle_isolation(source_bundle, tmp_path):
    from mjlab_microduck.rom.flyvis_tracking.simulation import (
        TrackingWorld,
        prepare_bundle,
    )

    original = (source_bundle / "microduck-policy-bundle.json").read_bytes()
    bundle = prepare_bundle(source_bundle, tmp_path / "bundle")
    with TrackingWorld(bundle) as world:
        world.reset(EpisodeSpec(7, "stationary", "test"))
        initial = world.observe()
        assert initial.rgb.shape == (240, 320, 3)
        assert np.std(initial.rgb) > 10
        assert initial.body.shape == (34,)
        assert np.isfinite(initial.body).all()
        assert world.truth()["visible"]
        for _ in range(50):
            world.step((0.0, 0.0, 1.0))
        assert world.observe().tick == 50
        assert world.truth()["yaw"] > 0.01
        assert not world.runtime._navigator
        assert world.runtime._previous_action.shape == (14,)
    assert (source_bundle / "microduck-policy-bundle.json").read_bytes() == original
    with pytest.raises((ValueError, FileExistsError)):
        prepare_bundle(source_bundle, source_bundle)


def test_target_visibility_is_measured_from_rendered_scene(source_bundle, tmp_path):
    from mjlab_microduck.rom.flyvis_tracking.simulation import (
        TrackingWorld,
        prepare_bundle,
    )

    bundle = prepare_bundle(source_bundle, tmp_path / "bundle")
    with TrackingWorld(bundle) as world:
        world.reset(EpisodeSpec(7, "disappearance", "test"))
        world.observe()
        assert world.truth()["visible"]
        world.tick = 450
        world.update_target()
        world.observe()
        assert not world.truth()["visible"]


def test_fall_between_camera_frames_is_retained_and_next_episode_runs(
    source_bundle, tmp_path
):
    from mjlab_microduck.rom.flyvis_tracking.simulation import (
        TrackingWorld,
        prepare_bundle,
    )

    bundle = prepare_bundle(source_bundle, tmp_path / "bundle")
    with TrackingWorld(bundle) as world:
        world.reset(EpisodeSpec(7, "stationary", "test"))
        # Invert the body between camera captures: actual runtime fall detection.
        address = world.runtime._free_qpos_address
        world.data.qpos[address + 3 : address + 7] = [0.0, 1.0, 0.0, 0.0]
        assert world.step((0.0, 0.0, 0.0)) is False
        assert world.safety_outcome()["fallen"]
        world.reset(EpisodeSpec(8, "stationary", "test"))
        assert world.step((0.0, 0.0, 0.0)) is True
        assert world.safety_outcome()["fallen"] is False


def test_contact_at_control_tick_is_retained_after_target_moves(
    source_bundle, tmp_path
):
    from mjlab_microduck.rom.flyvis_tracking.simulation import (
        TrackingWorld,
        prepare_bundle,
    )

    bundle = prepare_bundle(source_bundle, tmp_path / "bundle")
    with TrackingWorld(bundle) as world:
        world.reset(EpisodeSpec(7, "stationary", "test"))
        world.data.mocap_pos[world.mocap_id] = world.runtime._base_position()
        import mujoco

        mujoco.mj_forward(world.model, world.data)
        world.step((0.0, 0.0, 0.0))
        assert world.safety_outcome()["collision"]
        world.update_target()
        assert world.safety_outcome()["collision"]


def test_parallel_collection_keeps_episode_files_and_partitions_separate(
    source_bundle, tmp_path
):
    checkpoint = os.environ.get("FLYVIS_TRACKING_MODEL")
    if not checkpoint:
        pytest.skip("requires official flyvis checkpoint")
    from mjlab_microduck.rom.flyvis_tracking.experiment import collect, load_dataset
    from mjlab_microduck.rom.flyvis_tracking.simulation import prepare_bundle

    bundle = prepare_bundle(source_bundle, tmp_path / "bundle")
    collect(
        bundle,
        checkpoint,
        tmp_path / "data",
        count=2,
        validation_count=2,
        seconds=1.2,
        jobs=2,
    )
    train = list((tmp_path / "data").glob("train-*.npz"))
    validation = list((tmp_path / "data").glob("validation-*.npz"))
    assert len(train) == len(validation) == 2
    assert load_dataset(train, "flyvis", "train")[0].shape[0] == 60
    with pytest.raises(ValueError, match="partition"):
        load_dataset(train + validation, "flyvis", "train")


def test_sensor_selected_movement_starts_on_first_frame(source_bundle, tmp_path):
    import json

    from mjlab_microduck.rom.flyvis_tracking.experiment import run_episode
    from mjlab_microduck.rom.flyvis_tracking.simulation import (
        TrackingWorld,
        prepare_bundle,
    )
    from mjlab_microduck.rom.flyvis_tracking.vision import RetinaVision

    bundle = prepare_bundle(source_bundle, tmp_path / "bundle")
    with TrackingWorld(bundle) as world:
        run_episode(
            world,
            RetinaVision(),
            EpisodeSpec(7, "stationary", "train"),
            seconds=0.08,
            record=tmp_path / "episode.npz",
        )
    trace = json.loads((tmp_path / "episode.json").read_text())["trace"]
    assert trace[0]["requested"] == 1
    assert trace[0]["command"] == [0.2, 0.0, 0.0]


def test_target_placement_does_not_encode_initial_joint_randomization(
    source_bundle, tmp_path
):
    from mjlab_microduck.rom.flyvis_tracking.simulation import (
        TrackingWorld,
        prepare_bundle,
    )

    bundle = prepare_bundle(source_bundle, tmp_path / "bundle")
    joints, geometry = [], []
    with TrackingWorld(bundle) as world:
        for seed in range(80):
            world.reset(EpisodeSpec(seed, "stationary", "train"))
            joints.append(world.observe().body[6:20])
            truth = world.truth()
            geometry.append([truth["bearing"], truth["distance"]])
        world.reset(EpisodeSpec(0, "stationary", "train"))
        np.testing.assert_array_equal(world.observe().body[6:20], joints[0])
        truth = world.truth()
        np.testing.assert_allclose(
            [truth["bearing"], truth["distance"]], geometry[0], atol=1e-12
        )
    # Reusing the body's RNG made these two correlations exactly one, leaking
    # target geometry into joint sensors before the camera could contribute.
    for joint, target in ((0, 0), (1, 1)):
        correlation = np.corrcoef(
            np.asarray(joints)[:, joint], np.asarray(geometry)[:, target]
        )[0, 1]
        assert abs(correlation) < 0.4


@pytest.mark.parametrize("seed", [400000, 400001, 400002])
def test_camera_and_target_updates_preserve_direct_runtime_trajectory(
    source_bundle, tmp_path, seed
):
    """Rendering must not change the robot's next ONNX input or dynamics state."""
    from mjlab_microduck.rom.flyvis_tracking.simulation import (
        TrackingWorld,
        prepare_bundle,
    )

    bundle = prepare_bundle(source_bundle, tmp_path / "bundle")
    with TrackingWorld(bundle) as world, TrackingWorld(bundle) as reference:
        for instance in (world, reference):
            instance.reset(EpisodeSpec(seed, "stationary", "test"))
            instance.target_start[:] = [10, 10, 1]
            instance.update_target()
        command = (0.2, 0.0, 0.0)
        for tick in range(1000):
            reference.runtime.command(
                reference.handle,
                dict(zip(("vxMps", "vyMps", "yawRateRadps"), command, strict=True)),
            )
            assert reference.runtime.sample(reference.handle).running
            assert world.step(command)
            if tick % 20 == 0:
                world.observe()
            np.testing.assert_allclose(
                world.data.qpos, reference.data.qpos, atol=1e-12, rtol=0
            )
            np.testing.assert_allclose(
                world.data.qvel, reference.data.qvel, atol=1e-12, rtol=0
            )
            np.testing.assert_allclose(
                world.runtime._previous_action,
                reference.runtime._previous_action,
                atol=1e-12,
                rtol=0,
            )
