"""Experimental RGB observation of a known red spherical landmark.

Only pixels enter the detector. MuJoCo geom IDs, depth, and world pose are
reserved for the qualification evaluator and are never detection inputs.
"""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SphereObservation:
    center_x_px: float
    center_y_px: float
    range_m: float
    pixel_count: int


def detect_red_sphere(
    rgb: np.ndarray, *, sphere_radius_m: float, vertical_fov_deg: float
) -> SphereObservation | None:
    """Return a range/bearing landmark sample only for one intact red blob.

    The range derives from known physical marker radius and apparent angular
    radius. This is a simulator feasibility probe, not a calibrated camera or
    an AprilTag detector.
    """
    if (
        rgb.ndim != 3
        or rgb.shape[2] != 3
        or rgb.dtype != np.uint8
        or not math.isfinite(sphere_radius_m)
        or sphere_radius_m <= 0
        or not math.isfinite(vertical_fov_deg)
        or not 0 < vertical_fov_deg < 120
    ):
        raise ValueError("invalid camera sample or marker calibration")
    height, width = rgb.shape[:2]
    if min(height, width) < 32:
        raise ValueError("camera frame is too small")
    red = rgb[:, :, 0].astype(np.int16)
    green = rgb[:, :, 1].astype(np.int16)
    blue = rgb[:, :, 2].astype(np.int16)
    mask = (red >= 100) & (red > 2 * green) & (red > 2 * blue)
    rows, columns = np.where(mask)
    count = len(rows)
    if count < 20:
        return None
    top, bottom = int(rows.min()), int(rows.max())
    left, right = int(columns.min()), int(columns.max())
    if top == 0 or left == 0 or bottom == height - 1 or right == width - 1:
        return None
    box_height = bottom - top + 1
    box_width = right - left + 1
    if not 0.7 <= box_width / box_height <= 1.3:
        return None
    fill = count / (box_width * box_height)
    if not 0.55 <= fill <= 0.9:
        return None
    radius_px = math.sqrt(count / math.pi)
    focal_px = height / (2 * math.tan(math.radians(vertical_fov_deg) / 2))
    range_m = sphere_radius_m * math.sqrt(1 + (focal_px / radius_px) ** 2)
    if not math.isfinite(range_m):
        return None
    return SphereObservation(
        center_x_px=float(columns.mean()),
        center_y_px=float(rows.mean()),
        range_m=range_m,
        pixel_count=count,
    )
