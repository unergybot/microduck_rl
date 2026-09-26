"""Offline dead reckoning from MuJoCo IMU-site sensor samples.

The navigation controller receives only this estimate. Ground-truth position
and orientation remain available to the qualification evaluator separately.
The MuJoCo velocimeter is a simulator sensor, not a MicroDuck hardware claim.
"""

import math

import mujoco
import numpy as np

from ..navigation_contracts import Pose


class SensorFailure(ValueError):
    """A required sample is absent, stale, or non-finite."""


class SensorOdometry:
    def __init__(self, model: mujoco.MjModel, initial_pose: Pose):
        self.pose = initial_pose
        self.last_time: float | None = None
        self.gyro_address, gyro_site = self._sensor(
            model, "imu_ang_vel", mujoco.mjtSensor.mjSENS_GYRO
        )
        self.velocity_address, velocity_site = self._sensor(
            model, "imu_lin_vel", mujoco.mjtSensor.mjSENS_VELOCIMETER
        )
        if gyro_site != velocity_site:
            raise SensorFailure("IMU sensors must share one calibrated site")
        trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        if (
            trunk < 0
            or model.site_bodyid[gyro_site] != trunk
            or not np.allclose(
                model.site_quat[gyro_site], [1.0, 0.0, 0.0, 0.0], atol=1e-7
            )
        ):
            raise SensorFailure("IMU site must be identity-aligned on trunk_base")
        self.site_offset = np.asarray(model.site_pos[gyro_site], dtype=np.float64)
        self.orientation = np.array(
            [math.cos(initial_pose.yaw / 2), 0.0, 0.0, math.sin(initial_pose.yaw / 2)],
            dtype=np.float64,
        )

    @staticmethod
    def _sensor(model, name, kind):
        sensor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if (
            sensor < 0
            or model.sensor_type[sensor] != kind
            or model.sensor_dim[sensor] != 3
            or model.sensor_objtype[sensor] != mujoco.mjtObj.mjOBJ_SITE
        ):
            raise SensorFailure(f"missing calibrated {name} sensor")
        return int(model.sensor_adr[sensor]), int(model.sensor_objid[sensor])

    def update(self, data: mujoco.MjData) -> tuple[Pose, float, float]:
        now = float(data.time)
        gyro = np.asarray(data.sensordata[self.gyro_address : self.gyro_address + 3])
        velocity = np.asarray(
            data.sensordata[self.velocity_address : self.velocity_address + 3]
        )
        if (
            not math.isfinite(now)
            or not np.isfinite(gyro).all()
            or not np.isfinite(velocity).all()
        ):
            raise SensorFailure("non-finite IMU sample")
        if self.last_time is None:
            self.last_time = now
            return self.pose, float(np.linalg.norm(velocity[:2])), float(gyro[2])
        dt = now - self.last_time
        if not 0 < dt <= 0.1:
            raise SensorFailure("stale IMU sample")
        self.last_time = now
        angular = np.asarray(gyro, dtype=np.float64)
        mid_orientation = _integrate_orientation(self.orientation, angular, dt / 2)
        self.orientation = _integrate_orientation(self.orientation, angular, dt)
        base_velocity = velocity - np.cross(angular, self.site_offset)
        world_velocity = _rotate(mid_orientation, base_velocity)
        w, x, y, z = self.orientation
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        self.pose = Pose(
            x=self.pose.x + float(world_velocity[0]) * dt,
            y=self.pose.y + float(world_velocity[1]) * dt,
            yaw=yaw,
        )
        return self.pose, float(np.linalg.norm(base_velocity[:2])), float(angular[2])


def _integrate_orientation(quaternion, angular_velocity, dt):
    angle = float(np.linalg.norm(angular_velocity)) * dt
    if angle == 0:
        return quaternion.copy()
    half = angle / 2
    delta = np.concatenate(
        ([math.cos(half)], angular_velocity * (math.sin(half) * dt / angle))
    )
    w, x, y, z = quaternion
    a, b, c, d = delta
    result = np.array(
        [
            w * a - x * b - y * c - z * d,
            w * b + x * a + y * d - z * c,
            w * c - x * d + y * a + z * b,
            w * d + x * c - y * b + z * a,
        ]
    )
    return result / np.linalg.norm(result)


def _rotate(quaternion, vector):
    xyz = quaternion[1:]
    intermediate = 2 * np.cross(xyz, vector)
    return vector + quaternion[0] * intermediate + np.cross(xyz, intermediate)
