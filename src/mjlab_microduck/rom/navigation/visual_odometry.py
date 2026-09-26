"""Offline IMU propagation corrected by rendered AprilTag observations."""

import math

import mujoco
import numpy as np

from ..navigation_contracts import Pose
from .apriltag import PROBE_TAGS, detect_tag_corners, observe_planar_pose
from .sensor_odometry import SensorFailure, SensorOdometry


class VisualOdometry:
    def __init__(self, model, initial_pose, joint_qpos_indices):
        self.model = model
        self.odometry = SensorOdometry(model, initial_pose)
        self.joint_qpos_indices = np.asarray(joint_qpos_indices, dtype=np.int32)
        self.camera_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera"
        )
        if self.camera_id < 0:
            raise SensorFailure("missing calibrated head camera")
        self.render_data = mujoco.MjData(model)
        self.renderer = mujoco.Renderer(model, height=480, width=640)
        self.last_image_time = -math.inf
        self.last_visual_time = -math.inf
        self.last_visual_pose = None
        self.visual_updates = 0
        self.visual_rejections = 0
        self.last_tag_id = None

    @property
    def pose(self):
        return self.odometry.pose

    def close(self):
        self.renderer.close()

    def can_confirm_arrival(self, now, pose):
        """Require a recent landmark fix with little motion since its capture."""
        if self.last_visual_pose is None:
            return False
        yaw_since_fix = math.atan2(
            math.sin(pose.yaw - self.last_visual_pose.yaw),
            math.cos(pose.yaw - self.last_visual_pose.yaw),
        )
        return (
            self.visual_updates >= 2
            and math.isfinite(self.last_visual_time)
            # The door turn briefly points the head away from the last tag.
            # Keep the fix valid for that measured interval only while the
            # independently integrated pose has barely moved since capture.
            and 0 <= now - self.last_visual_time <= 1.5
            and math.hypot(
                pose.x - self.last_visual_pose.x,
                pose.y - self.last_visual_pose.y,
            )
            <= 0.03
            and abs(yaw_since_fix) <= 0.08
        )

    def update(self, data):
        pose, speed, yaw_rate = self.odometry.update(data)
        now = float(data.time)
        if now - self.last_image_time < 0.2:
            return pose, speed, yaw_rate
        self.last_image_time = now
        # The renderer reads only a copy. Forward dynamics on live data would
        # perturb the next policy observation and contact solver state.
        mujoco.mj_copyData(self.render_data, self.model, data)
        mujoco.mj_forward(self.model, self.render_data)
        self.renderer.update_scene(self.render_data, camera=self.camera_id)
        rgb = self.renderer.render().copy()
        detections = detect_tag_corners(rgb)
        candidates = []
        joints = np.asarray(data.qpos[self.joint_qpos_indices], dtype=np.float64)
        for probe in PROBE_TAGS:
            if probe.tag_id not in detections:
                continue
            observed = observe_planar_pose(
                rgb,
                model=self.model,
                marker_world_corners=probe.world_corners(),
                marker_id=probe.tag_id,
                joint_qpos_indices=self.joint_qpos_indices,
                joint_positions=joints,
                detections=detections,
            )
            if observed is not None:
                candidates.append((probe.tag_id, observed))
        if not candidates:
            return pose, speed, yaw_rate
        tag_id, observed = min(
            candidates, key=lambda item: item[1].reprojection_error_px
        )
        delta_x = observed.pose.x - pose.x
        delta_y = observed.pose.y - pose.y
        delta_yaw = math.atan2(
            math.sin(observed.pose.yaw - pose.yaw),
            math.cos(observed.pose.yaw - pose.yaw),
        )
        # A single planar PnP solution can flip under partial occlusion while
        # retaining a small reprojection residual. Reject any correction that
        # exceeds the qualifier's position/yaw error envelope.
        if math.hypot(delta_x, delta_y) > 0.08 or abs(delta_yaw) > 0.1:
            self.visual_rejections += 1
            return pose, speed, yaw_rate
        # A full per-frame snap can repeatedly cross the tight 8 cm arrival
        # boundary and reset settlement. Blend valid corrections while still
        # using each raw PnP solution for the innovation safety check above.
        gain = 0.5
        half = gain * delta_yaw / 2
        correction = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])
        w, x, y, z = correction
        a, b, c, d = self.odometry.orientation
        self.odometry.orientation = np.array(
            [
                w * a - x * b - y * c - z * d,
                w * b + x * a + y * d - z * c,
                w * c - x * d + y * a + z * b,
                w * d + x * c - y * b + z * a,
            ]
        )
        fused_pose = Pose(
            x=pose.x + gain * delta_x,
            y=pose.y + gain * delta_y,
            yaw=math.atan2(
                math.sin(pose.yaw + gain * delta_yaw),
                math.cos(pose.yaw + gain * delta_yaw),
            ),
        )
        self.odometry.pose = fused_pose
        self.last_visual_time = now
        self.last_visual_pose = fused_pose
        self.last_tag_id = tag_id
        self.visual_updates += 1
        return fused_pose, speed, yaw_rate
