"""Offline 8×8 head ToF ray simulation; no-return cells remain unknown.

Beam axes and zone centres match the upstream MicroDuck ToF reprojector:
sensor +x forward, +y left, +z up, with 45 degrees per image axis.
"""

import math
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class ToFScan:
    captured_at_s: float
    ranges_m: tuple[tuple[float | None, ...], ...]


class ToFSimulator:
    def __init__(self, model: mujoco.MjModel, *, axis_fov_deg=45.0, max_range_m=4.0):
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tof")
        if site < 0:
            raise ValueError("model has no head ToF site")
        if not 0 < axis_fov_deg < 180 or not 0 < max_range_m <= 4:
            raise ValueError("invalid ToF geometry")
        self.model = model
        self.site = site
        self.max_range_m = max_range_m
        half = math.radians(axis_fov_deg / 2 - axis_fov_deg / 16)
        centers = [half - i * 2 * half / 7 for i in range(8)]
        self.directions = [
            np.array(
                [
                    math.cos(elevation) * math.cos(azimuth),
                    math.cos(elevation) * math.sin(azimuth),
                    math.sin(elevation),
                ],
                dtype=np.float64,
            )
            for elevation in centers
            for azimuth in centers
        ]
        # The qualified robot's geoms use group 2 and the installed scene uses
        # group 0. Excluding robot visuals prevents self-range artifacts.
        self.geom_groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
        if any(
            model.geom_group[geom] == 0
            and model.body_weldid[model.geom_bodyid[geom]] != 0
            for geom in range(model.ngeom)
        ):
            raise ValueError("ToF scene mask would include moving robot geometry")

    def capture(self, data: mujoco.MjData) -> ToFScan:
        origin = np.asarray(data.site_xpos[self.site], dtype=np.float64)
        rotation = np.asarray(data.site_xmat[self.site], dtype=np.float64).reshape(3, 3)
        if not np.isfinite(origin).all() or not np.isfinite(rotation).all():
            raise ValueError("invalid ToF transform")
        ranges = []
        hit = np.empty(1, dtype=np.int32)
        for local in self.directions:
            distance = mujoco.mj_ray(
                self.model,
                data,
                origin,
                rotation @ local,
                self.geom_groups,
                True,
                -1,
                hit,
            )
            ranges.append(
                float(distance) if 0 <= distance <= self.max_range_m else None
            )
        return ToFScan(
            captured_at_s=float(data.time),
            ranges_m=tuple(tuple(ranges[row * 8 : (row + 1) * 8]) for row in range(8)),
        )
