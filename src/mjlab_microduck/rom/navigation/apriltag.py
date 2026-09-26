"""Offline AprilTag pose observation from RGB and measured head joints.

The marker's four world corners are declared map calibration, not MuJoCo
body/geom positions. Ground truth is reserved for the qualification evaluator.
"""

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from ..navigation_contracts import Pose


@dataclass(frozen=True)
class TagProbe:
    tag_id: int
    center_xyz: tuple[float, float, float]
    quat_wxyz: tuple[float, float, float, float]

    def world_corners(self) -> np.ndarray:
        rotation = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, np.asarray(self.quat_wxyz))
        local = np.array(
            [[-0.07, 0.07, 0], [0.07, 0.07, 0], [0.07, -0.07, 0], [-0.07, -0.07, 0]],
            dtype=np.float64,
        )
        return local @ rotation.reshape(3, 3).T + self.center_xyz


PROBE_TAGS = (
    TagProbe(0, (0.38, 0.0, 0.2), (2**-0.5, 0.0, -(2**-0.5), 0.0)),
    TagProbe(1, (0.0, 1.0, 0.52), (2**-0.5, 2**-0.5, 0.0, 0.0)),
    TagProbe(2, (1.5, 0.0, 0.25), (2**-0.5, 0.0, -(2**-0.5), 0.0)),
    TagProbe(3, (0.5, -0.45, 0.25), (2**-0.5, 0.0, -(2**-0.5), 0.0)),
    TagProbe(4, (0.0, 1.5, 0.25), (2**-0.5, 2**-0.5, 0.0, 0.0)),
)


def tag_texture(tag_id: int) -> np.ndarray:
    """Deterministic tag36h11 image with a white detection margin."""
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dictionary, tag_id, 256)
    image = np.full((320, 320), 255, dtype=np.uint8)
    image[32:288, 32:288] = marker
    return image


def detect_tag_corners(rgb: np.ndarray) -> dict[int, list[np.ndarray]]:
    """Decode all visible tag36h11 IDs from one RGB camera frame."""
    import cv2

    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("expected uint8 RGB camera frame")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    corners, identifiers, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(gray)
    found: dict[int, list[np.ndarray]] = {}
    if identifiers is not None:
        for pixels, identifier in zip(corners, identifiers.reshape(-1), strict=True):
            found.setdefault(int(identifier), []).append(pixels.reshape(4, 2))
    return found


@dataclass(frozen=True)
class AprilTagPose:
    pose: Pose
    base_height_m: float
    reprojection_error_px: float
    tag_pixels: float


def observe_planar_pose(
    rgb: np.ndarray,
    *,
    model: mujoco.MjModel,
    marker_world_corners: np.ndarray,
    marker_id: int,
    joint_qpos_indices: np.ndarray,
    joint_positions: np.ndarray,
    camera_name: str = "head_camera",
    detections: dict[int, list[np.ndarray]] | None = None,
) -> AprilTagPose | None:
    """Solve robot x/y/yaw from one calibrated rendered AprilTag.

    The camera extrinsic is recomputed on a separate scratch state at origin,
    using encoder samples. The live
    simulator base pose and camera transform are never read here.
    """
    import cv2

    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("expected uint8 RGB camera frame")
    height, width = rgb.shape[:2]
    if min(height, width) < 64:
        raise ValueError("invalid camera frame")
    points = np.asarray(marker_world_corners, dtype=np.float64)
    joint_indices = np.asarray(joint_qpos_indices, dtype=np.int32)
    joints = np.asarray(joint_positions, dtype=np.float64)
    if (
        points.shape != (4, 3)
        or joint_indices.shape != (14,)
        or joints.shape != (14,)
        or not np.isfinite(points).all()
        or not np.isfinite(joints).all()
        or len(set(joint_indices.tolist())) != 14
        or joint_indices.min() < 0
        or joint_indices.max() >= model.nq
    ):
        raise ValueError("invalid marker or joint calibration")
    camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    if camera < 0 or trunk < 0:
        raise ValueError("missing calibrated head camera or trunk")
    freejoint = int(model.body_jntadr[trunk])
    if freejoint < 0 or model.jnt_type[freejoint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("trunk has no calibrated free joint")
    address = int(model.jnt_qposadr[freejoint])
    fovy = float(model.cam_fovy[camera])
    if not 0 < fovy < 120:
        raise ValueError("invalid camera field of view")
    matches = (detections if detections is not None else detect_tag_corners(rgb)).get(
        marker_id, []
    )
    if len(matches) != 1:
        return None
    pixels = np.asarray(matches[0], dtype=np.float64)
    if (
        np.min(pixels[:, 0]) <= 1
        or np.min(pixels[:, 1]) <= 1
        or np.max(pixels[:, 0]) >= width - 2
        or np.max(pixels[:, 1]) >= height - 2
    ):
        return None
    edge_px = min(np.linalg.norm(pixels[(i + 1) % 4] - pixels[i]) for i in range(4))
    # A small planar tag has ambiguous pose under one-pixel corner jitter.
    # Only use close, well-resolved observations for navigation correction.
    if edge_px < 90:
        return None
    focal = height / (2 * math.tan(math.radians(fovy) / 2))
    intrinsic = np.array(
        [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    solved, rotation_vector, translation = cv2.solvePnP(
        points, pixels, intrinsic, None, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not solved:
        return None
    reprojected = cv2.projectPoints(
        points, rotation_vector, translation, intrinsic, None
    )[0].reshape(4, 2)
    residual = float(np.max(np.linalg.norm(reprojected - pixels, axis=1)))
    if not math.isfinite(residual) or residual > 2:
        return None
    world_to_cv = cv2.Rodrigues(rotation_vector)[0]
    camera_world = (-world_to_cv.T @ translation).reshape(3)
    cv_to_world = world_to_cv.T
    # OpenCV: x right, y down, z forward. MuJoCo camera: x right, y up,
    # z backward. The diagonal transform relates their optical axes.
    camera_world_matrix = cv_to_world @ np.diag([1.0, -1.0, -1.0])
    scratch = mujoco.MjData(model)
    mujoco.mj_resetData(model, scratch)
    scratch.qpos[address : address + 3] = [0.0, 0.0, 0.0]
    scratch.qpos[address + 3 : address + 7] = [1.0, 0.0, 0.0, 0.0]
    scratch.qpos[joint_indices] = joints
    mujoco.mj_forward(model, scratch)
    relative_camera_matrix = scratch.cam_xmat[camera].reshape(3, 3)
    relative_camera_position = scratch.cam_xpos[camera].copy()
    base_rotation = camera_world_matrix @ relative_camera_matrix.T
    tilt = math.acos(float(np.clip(base_rotation[2, 2], -1, 1)))
    if tilt > 0.25:
        return None
    yaw = math.atan2(base_rotation[1, 0], base_rotation[0, 0])
    base_position = camera_world - base_rotation @ relative_camera_position
    if not np.isfinite(base_position).all() or not 0.05 <= base_position[2] <= 0.3:
        return None
    return AprilTagPose(
        pose=Pose(
            x=float(base_position[0]),
            y=float(base_position[1]),
            yaw=yaw,
        ),
        base_height_m=float(base_position[2]),
        reprojection_error_px=residual,
        tag_pixels=edge_px,
    )
