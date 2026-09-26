"""RGB landmark extraction stays separate from evaluator-only MuJoCo truth."""

import math
import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mjlab_microduck.rom.navigation.vision import detect_red_sphere


def test_visual_arrival_waits_briefly_for_a_fresh_tag_fix():
    from mjlab_microduck.rom.mujoco_runtime import gate_visual_arrival
    from mjlab_microduck.rom.navigation.follower import Command
    from mjlab_microduck.rom.navigation_contracts import Pose

    class Estimator:
        confirmed = False

        def can_confirm_arrival(self, now, pose):
            return self.confirmed

    estimator = Estimator()
    pose = Pose(x=1.0, y=0.0, yaw=0.0)
    arrived = Command(arrived=True)
    pending, started = gate_visual_arrival(arrived, estimator, 10.0, pose, None)
    assert pending == Command() and started == 10.0
    pending, started = gate_visual_arrival(arrived, estimator, 12.9, pose, started)
    assert pending == Command() and started == 10.0
    failed, started = gate_visual_arrival(arrived, estimator, 13.0, pose, started)
    assert failed.reason == "LOCALIZATION_FAILED" and started == 10.0
    estimator.confirmed = True
    accepted, started = gate_visual_arrival(arrived, estimator, 13.1, pose, started)
    assert accepted == arrived and started is None


def test_red_sphere_requires_one_uncropped_round_blob():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    rows, columns = np.ogrid[:240, :320]
    frame[(rows - 120) ** 2 + (columns - 160) ** 2 <= 20**2] = [220, 10, 10]
    seen = detect_red_sphere(frame, sphere_radius_m=0.08, vertical_fov_deg=45)
    assert seen is not None
    assert seen.center_x_px == pytest.approx(160, abs=0.1)
    assert seen.center_y_px == pytest.approx(120, abs=0.1)
    assert seen.range_m == pytest.approx(
        0.08 * math.sqrt(1 + (240 / (2 * math.tan(math.pi / 8)) / 20) ** 2),
        rel=0.03,
    )
    frame[:60] = [220, 10, 10]
    assert detect_red_sphere(frame, sphere_radius_m=0.08, vertical_fov_deg=45) is None


@pytest.mark.skipif(
    not os.environ.get("MICRODUCK_TEST_BUNDLE"), reason="requires real bundle"
)
def test_corrected_head_camera_observes_known_marker_from_rgb(tmp_path):
    from mjlab_microduck.rom.flyvis_tracking.simulation import prepare_bundle
    from mjlab_microduck.rom.main import load_verified_bundle
    from mjlab_microduck.rom.mujoco_runtime import MicroduckMujocoRuntime

    candidate = prepare_bundle(
        Path(os.environ["MICRODUCK_TEST_BUNDLE"]), tmp_path / "vision-bundle"
    )
    runtime = MicroduckMujocoRuntime(
        candidate, load_verified_bundle(candidate), realtime=False
    )
    try:
        model, data = runtime._model, runtime._data
        target = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "tracking_target_geom"
        )
        camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
        model.geom_rgba[target] = [1, 0, 0, 1]
        renderer = mujoco.Renderer(model, height=240, width=320)
        try:
            renderer.update_scene(data, camera=camera)
            rgb = renderer.render().copy()
        finally:
            renderer.close()
        seen = detect_red_sphere(
            rgb, sphere_radius_m=0.08, vertical_fov_deg=float(model.cam_fovy[camera])
        )
        assert seen is not None
        # World geometry is evaluator-only evidence for the RGB range estimate.
        true_range = np.linalg.norm(data.geom_xpos[target] - data.cam_xpos[camera])
        assert seen.range_m == pytest.approx(true_range, abs=0.09)
        assert seen.pixel_count > 500
    finally:
        runtime._snapshot.cleanup()
