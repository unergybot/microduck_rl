import os
from pathlib import Path

import numpy as np
import pytest

from mjlab_microduck.rom.flyvis_tracking.core import PHYSICS_PROTOCOL


def test_motion_score_rejects_standing_and_any_collision():
    from mjlab_microduck.rom.flyvis_scenarios import score_motion

    start = {
        "x": 0.0,
        "y": 0.0,
        "distance": 1.2,
        "bearing": 0.0,
        "visible": True,
        "collision": False,
        "fallen": False,
        "time": 0.0,
    }
    assert not score_motion([start, dict(start, time=20.0)], None)["passed"]
    finish = dict(start, x=0.5, distance=0.7, time=20.0)
    assert score_motion([start, finish], None)["passed"]
    assert not score_motion([start, dict(finish, collision=True)], None)["passed"]
    assert not score_motion([start, finish], "STALE_CAMERA")["passed"]


def test_obstacle_teacher_turns_toward_clear_side():
    from mjlab_microduck.rom.flyvis_scenarios import scenario_teacher

    row = {
        "x": 0.0,
        "y": 0.0,
        "yaw": 0.0,
        "distance": 1.2,
        "bearing": 0.0,
        "visible": True,
        "waypoint": [0.4, -0.32],
    }
    assert scenario_teacher(row) == 3
    assert scenario_teacher(dict(row, waypoint=[0.4, 0.32])) == 2
    assert scenario_teacher(dict(row, waypoint=None)) == 1


def test_scenario_checkpoints_cannot_silently_reuse_baseline_metadata():
    from mjlab_microduck.rom.flyvis_scenarios import validate_scenario

    with pytest.raises(ValueError, match="scenario protocol"):
        validate_scenario({"startupProtocol": "immediate_sensor_control"})


@pytest.fixture
def source_bundle():
    value = os.environ.get("MICRODUCK_TRACKING_BUNDLE")
    if not value:
        pytest.skip("requires calibrated bundle")
    return Path(value)


def test_external_camera_preserves_sensors_and_obstacles_have_robot_contacts(
    source_bundle, tmp_path
):
    import mujoco

    from mjlab_microduck.rom.flyvis_scenarios import ScenarioWorld, prepare_scenarios
    from mjlab_microduck.rom.flyvis_tracking.core import EpisodeSpec

    bundle = prepare_scenarios(source_bundle, tmp_path / "bundle")
    with ScenarioWorld(bundle) as world:
        world.reset(EpisodeSpec(7, "obstacle_left", "test"))
        before = world.observe()
        qpos = world.data.qpos.copy()
        external = world.external_view()
        after = world.observe()
        assert external.shape == (480, 640, 3)
        assert np.std(external) > 10
        assert np.array_equal(before.rgb, after.rgb)
        assert np.array_equal(before.body, after.body)
        assert np.array_equal(qpos, world.data.qpos)
        assert before.tick == after.tick
        assert world.truth()["distance"] > 1.0
        assert not world.truth()["collision"]
        world.data.mocap_pos[world.obstacle_mocap] = world.runtime._base_position()
        mujoco.mj_forward(world.model, world.data)
        assert world.truth()["collision"]
        world.step((0.0, 0.0, 0.0))
        world.data.mocap_pos[world.obstacle_mocap] = [5.0, 5.0, 1.0]
        mujoco.mj_forward(world.model, world.data)
        assert world.safety_outcome()["collision"]


def test_scenario_dataset_rejects_baseline_episode_inside_new_manifest(tmp_path):
    import json

    from mjlab_microduck.rom.flyvis_scenario_training import load_scenarios
    from mjlab_microduck.rom.flyvis_tracking.core import RANDOMIZATION_PROTOCOL

    p = tmp_path / "old.npz"
    np.savez(
        p,
        neural=np.ones((1, 585)),
        retina=np.ones((1, 721)),
        body=np.ones((1, 34)),
        labels=[1],
        metadata=json.dumps(
            {
                "partition": "train",
                "startupProtocol": "immediate_sensor_control",
                "randomizationProtocol": RANDOMIZATION_PROTOCOL,
                "physicsProtocol": PHYSICS_PROTOCOL,
            }
        ),
    )
    with pytest.raises(ValueError, match="scenario protocol"):
        load_scenarios([p], "flyvis", "train")


def test_scenario_seed_ranges_are_disjoint_and_balanced():
    from mjlab_microduck.rom.flyvis_scenario_training import specs

    sets = [specs(p, 10) for p in ("train", "validation", "test", "probe")]
    assert len({s.seed for group in sets for s in group}) == 40
    for group in sets:
        assert len({s.family for s in group}) == 5
    assert not {s.seed for s in specs("train", 10, 10)} & {s.seed for s in sets[0]}


def test_synchronized_recording_replays_both_views_without_sensor_leak(
    source_bundle, tmp_path
):
    import json

    import imageio.v2 as imageio

    from mjlab_microduck.rom.flyvis_scenarios import (
        ScenarioWorld,
        prepare_scenarios,
        replay_scenario,
        scenario_episode,
    )
    from mjlab_microduck.rom.flyvis_tracking.core import EpisodeSpec
    from mjlab_microduck.rom.flyvis_tracking.vision import RetinaVision

    bundle = prepare_scenarios(source_bundle, tmp_path / "bundle")
    path = tmp_path / "episode.npz"
    with ScenarioWorld(bundle) as world:
        result = scenario_episode(
            world,
            RetinaVision(),
            EpisodeSpec(7, "approach", "train"),
            path,
            seconds=0.16,
            video=True,
        )
    assert result["controllerRole"] == "privileged_geometry_teacher"
    with np.load(path, allow_pickle=False) as data:
        assert data["body"].shape == (4, 34)
        assert data["neural"].shape == (4, 721)
        assert data["rgb"].shape == (4, 240, 320, 3)
        assert data["observer_rgb"].shape == (4, 480, 640, 3)
        assert json.loads(str(data["metadata"]))["scenarioProtocol"]
    replay_scenario(path, tmp_path / "replay.mp4")
    with (
        imageio.get_reader(path.with_suffix(".mp4")) as first,
        imageio.get_reader(tmp_path / "replay.mp4") as second,
    ):
        assert first.get_data(2).shape == (480, 1280, 3)
        assert np.array_equal(first.get_data(2), second.get_data(2))


def test_motion_success_uses_fresh_settled_pose():
    from mjlab_microduck.rom.flyvis_scenarios import score_motion

    start = {
        "x": 0.0,
        "y": 0.0,
        "distance": 1.2,
        "bearing": 0.0,
        "visible": True,
        "collision": False,
        "fallen": False,
        "time": 0.0,
    }
    last_frame = dict(start, x=0.5, distance=0.7, time=19.96)
    final = dict(last_frame, x=0.2, distance=1.0, time=21.0)
    assert not score_motion([start, last_frame], None, final=final)["passed"]


def test_training_identity_checks_bundle_and_visual_preprocessing(tmp_path):
    import json

    from mjlab_microduck.rom.flyvis_scenario_training import load_scenarios
    from mjlab_microduck.rom.flyvis_scenarios import SCENARIO_PROTOCOL
    from mjlab_microduck.rom.flyvis_tracking.core import RANDOMIZATION_PROTOCOL

    p = tmp_path / "episode.npz"
    eye = {
        "extent": 15,
        "kernelSize": 13,
        "retinaSites": 721,
        "grayscaleWeights": [0.299, 0.587, 0.114],
    }
    meta = {
        "partition": "train",
        "scenarioProtocol": SCENARIO_PROTOCOL,
        "startupProtocol": "immediate_sensor_control",
        "randomizationProtocol": RANDOMIZATION_PROTOCOL,
        "physicsProtocol": PHYSICS_PROTOCOL,
        "bundleDigest": "bundle-a",
        "vision": dict(eye, checkpoint="network-a"),
    }

    def save(value):
        np.savez(
            p,
            neural=np.ones((1, 585)),
            retina=np.ones((1, 721)),
            body=np.ones((1, 34)),
            labels=[1],
            metadata=json.dumps(value),
        )

    save(meta)
    with pytest.raises(ValueError, match="identit"):
        load_scenarios([p], "flyvis", "train", dict(meta, bundleDigest="bundle-b"))
    with pytest.raises(ValueError, match="identit"):
        load_scenarios(
            [p], "flyvis", "train", dict(meta, vision=dict(eye, checkpoint="network-b"))
        )
    # Retinal features remain compatible across flyvis teacher / retinal DAgger traces.
    x, _ = load_scenarios([p], "retina", "train", dict(meta, vision=eye))
    assert x.shape == (1, 755)
    with pytest.raises(ValueError, match="identit"):
        load_scenarios(
            [p], "retina", "train", dict(meta, vision=dict(eye, kernelSize=7))
        )


def test_detour_requires_precise_heading_and_teacher_can_reacquire_target():
    from mjlab_microduck.rom.flyvis_scenarios import scenario_teacher

    row = {
        "x": 0.0,
        "y": 0.0,
        "yaw": 0.0,
        "distance": 1.2,
        "bearing": 0.4,
        "visible": True,
        "waypoint": [0.4, 0.07],
    }
    assert scenario_teacher(row) == 2
    assert scenario_teacher(dict(row, visible=False, waypoint=None)) == 2


def test_scenario_manifest_rejects_legacy_physics_before_training(tmp_path):
    import json

    from mjlab_microduck.rom.flyvis_scenario_training import train
    from mjlab_microduck.rom.flyvis_scenarios import SCENARIO_PROTOCOL
    from mjlab_microduck.rom.flyvis_tracking.core import RANDOMIZATION_PROTOCOL

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "dataset.json").write_text(
        json.dumps(
            {
                "scenarioProtocol": SCENARIO_PROTOCOL,
                "startupProtocol": "immediate_sensor_control",
                "randomizationProtocol": RANDOMIZATION_PROTOCOL,
            }
        )
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="physics protocol"):
        train(None, None, dataset, output)
    assert not output.exists()
