from pathlib import Path

import mujoco
import numpy as np
import pytest

from mjlab_microduck.reference_motion import (
    ReferenceMotionError,
    load_reference_motion,
)
from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_XML


def valid_reference(path: Path) -> Path:
    model = mujoco.MjModel.from_xml_path(str(MICRODUCK_WALK_XML))
    data = mujoco.MjData(model)
    joint_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(1, model.njnt)
    )
    body_names = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
        for index in range(1, model.nbody)
    )
    joints = np.zeros((3, 14), dtype=np.float32)
    joints[1, 3] = 1.0
    joints[1, 12] = -1.0
    body_pos = []
    body_quat = []
    for frame, z in enumerate((0.12, 0.07, 0.12)):
        data.qpos[:] = 0.0
        data.qpos[2] = z
        data.qpos[3] = 1.0
        data.qpos[model.jnt_qposadr[1:]] = joints[frame]
        mujoco.mj_forward(model, data)
        body_pos.append(data.xpos[1:].copy())
        body_quat.append(data.xquat[1:].copy())
    zeros_body = np.zeros((3, len(body_names), 3), dtype=np.float32)
    np.savez_compressed(
        path,
        joint_pos=joints,
        joint_vel=np.gradient(joints, 0.02, axis=0).astype(np.float32),
        body_pos_w=np.asarray(body_pos, dtype=np.float32),
        body_quat_w=np.asarray(body_quat, dtype=np.float32),
        body_lin_vel_w=zeros_body,
        body_ang_vel_w=zeros_body,
        fps=np.array([50], dtype=np.int32),
        schema_version=np.array([1], dtype=np.int32),
        joint_names=np.asarray(joint_names),
        body_names=np.asarray(body_names),
        source_hashes_json=np.asarray(['{"fixture":"reference"}']),
    )
    return path


def test_loads_validated_reference_and_interpolates_cycle_phase(tmp_path):
    reference = load_reference_motion(valid_reference(tmp_path / "squat.npz"))

    sampled_joints, sampled_height = reference.sample_phase(np.array([0.0, 0.5, 1.0]))

    np.testing.assert_allclose(sampled_joints[:, 3], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(sampled_height, [0.12, 0.07, 0.12], atol=1e-6)
    assert reference.frames == 3
    assert reference.sha256


def test_missing_reference_path_fails_closed(monkeypatch):
    monkeypatch.delenv("MICRODUCK_REFERENCE_MOTION", raising=False)

    with pytest.raises(ReferenceMotionError, match="MICRODUCK_REFERENCE_MOTION"):
        load_reference_motion(None)
