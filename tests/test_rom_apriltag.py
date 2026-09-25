"""AprilTag pose correction uses RGB and calibrated joint kinematics."""

import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mjlab_microduck.rom.navigation.apriltag import (
    PROBE_TAGS,
    AprilTagPose,
    observe_planar_pose,
)


def test_probe_world_corners_have_known_square_size():
    for probe in PROBE_TAGS:
        corners = probe.world_corners()
        assert corners.shape == (4, 3)
        for index in range(4):
            assert np.linalg.norm(corners[(index + 1) % 4] - corners[index]) == (
                pytest.approx(0.14)
            )


@pytest.mark.skipif(
    not os.environ.get("MICRODUCK_TEST_BUNDLE"), reason="requires real bundle"
)
def test_real_head_camera_tag_pose_matches_hidden_truth(monkeypatch):
    pytest.importorskip("cv2")
    from mjlab_microduck.rom.main import load_verified_bundle
    from mjlab_microduck.rom.mujoco_runtime import MicroduckMujocoRuntime
    from mjlab_microduck.rom.navigation.installation import Installation
    from mjlab_microduck.rom.navigation_contracts import NavigationProfile, Pose, Scene

    root = Path(os.environ["MICRODUCK_TEST_BUNDLE"])
    fixture = json.loads(
        Path("tests/fixtures/navigation/calibrated-scenarios.json").read_text()
    )
    bundle = load_verified_bundle(root)
    installed = Installation(
        Scene.model_validate(fixture["scene"]),
        NavigationProfile.model_validate(fixture["profile"]),
        "qualification",
        "qualification",
        bundle.bundleDigest,
        "sha256:" + "0" * 64,
    )
    runtime = MicroduckMujocoRuntime(
        root,
        bundle,
        realtime=False,
        _navigation_candidate=installed,
        _apriltag_experiment=True,
    )
    try:
        model, data = runtime._model, runtime._data
        camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
        renderer = mujoco.Renderer(model, height=480, width=640)
        cases = [
            (-0.3, 0.0, 0.0, 0),
            (0.0, 0.0, 0.0, 0),
            (0.0, -0.1, 0.0, 0),
            (0.0, 0.0, 0.2, 0),
            (0.0, 0.0, math.pi / 2, 1),
            (1.0, 0.0, 0.0, 2),
            (0.0, -0.5, 0.0, 3),
            (0.0, 1.0, math.pi / 2, 4),
        ]
        try:
            for x, y, yaw, tag_id in cases:
                address = runtime._free_qpos_address
                data.qpos[address : address + 2] = [x, y]
                data.qpos[address + 3 : address + 7] = [
                    math.cos(yaw / 2),
                    0.0,
                    0.0,
                    math.sin(yaw / 2),
                ]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                rgb = renderer.render().copy()
                probe = next(item for item in PROBE_TAGS if item.tag_id == tag_id)
                observed = observe_planar_pose(
                    rgb,
                    model=model,
                    marker_world_corners=probe.world_corners(),
                    marker_id=tag_id,
                    joint_qpos_indices=runtime._joint_qpos_indices,
                    joint_positions=runtime._encoder_positions(),
                )
                if tag_id == 1:
                    # Door-header tag at this range is below the pixel-size
                    # gate; only a well-resolved observation may correct pose.
                    assert observed is None
                    continue
                assert observed is not None
                assert math.hypot(observed.pose.x - x, observed.pose.y - y) < 0.02
                assert abs(observed.pose.yaw - yaw) < 0.03
                assert observed.reprojection_error_px < 1.0
                assert observed.tag_pixels > 30
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            assert (
                observe_planar_pose(
                    blank,
                    model=model,
                    marker_world_corners=PROBE_TAGS[0].world_corners(),
                    marker_id=0,
                    joint_qpos_indices=runtime._joint_qpos_indices,
                    joint_positions=runtime._encoder_positions(),
                )
                is None
            )
        finally:
            renderer.close()
        runtime.set_navigation_estimator_for_qualification(
            Pose(x=0.0, y=0.0, yaw=0.0), visual=True
        )
        estimator = runtime._navigation_estimator
        try:
            origin = Pose(x=0.0, y=0.0, yaw=0.0)
            assert not estimator.can_confirm_arrival(1.0, origin)
            estimator.visual_updates = 2
            estimator.last_visual_time = 1.0
            estimator.last_visual_pose = origin
            assert estimator.can_confirm_arrival(1.7, origin)
            assert not estimator.can_confirm_arrival(2.01, origin)
            assert not estimator.can_confirm_arrival(1.7, Pose(x=0.04, y=0.0, yaw=0.0))
            assert not estimator.can_confirm_arrival(1.7, Pose(x=0.0, y=0.0, yaw=0.1))
            monkeypatch.setattr(
                "mjlab_microduck.rom.navigation.visual_odometry.detect_tag_corners",
                lambda _rgb: {0: [np.zeros((4, 2))]},
            )
            monkeypatch.setattr(
                "mjlab_microduck.rom.navigation.visual_odometry.observe_planar_pose",
                lambda *_args, **_kwargs: AprilTagPose(
                    Pose(x=0.2, y=0.0, yaw=0.0), 0.12, 0.0, 200.0
                ),
            )
            estimator.update(data)
            assert estimator.visual_rejections == 1
            assert estimator.pose.x == pytest.approx(0.0)
        finally:
            estimator.close()
    finally:
        runtime._snapshot.cleanup()


@pytest.mark.skipif(
    not os.environ.get("MICRODUCK_TEST_BUNDLE"), reason="requires real bundle"
)
def test_missing_visual_fix_cannot_confirm_arrival(monkeypatch):
    pytest.importorskip("cv2")
    from scripts.qualify_rom_navigation import run

    from mjlab_microduck.rom.main import load_verified_bundle
    from mjlab_microduck.rom.navigation_contracts import NavigationProfile, Scene

    monkeypatch.setattr(
        "mjlab_microduck.rom.navigation.visual_odometry.observe_planar_pose",
        lambda *args, **kwargs: None,
    )
    root = Path(os.environ["MICRODUCK_TEST_BUNDLE"])
    fixture = json.loads(
        Path("tests/fixtures/navigation/calibrated-scenarios.json").read_text()
    )
    turn = next(item for item in fixture["scenarios"] if item["name"] == "turn")
    result = run(
        root,
        load_verified_bundle(root),
        Scene.model_validate(fixture["scene"]),
        NavigationProfile.model_validate(fixture["profile"]),
        turn,
        7,
        pose_source="SIM_VISUAL_ODOMETRY",
    )
    assert result["passed"] is False
    assert result["reason"] == "LOCALIZATION_FAILED"
    assert result["visualUpdates"] == 0
    assert result["evidence"]["stoppedCommandConfirmed"] is True
