"""Heading-first local follower with measured settlement and monotonic deadlines."""

import math
from dataclasses import dataclass

from .grid import Grid
from .planner import plan_cells


@dataclass(frozen=True)
class Command:
    vx: float = 0.0
    vy: float = 0.0
    yaw: float = 0.0
    arrived: bool = False
    reason: str | None = None


def angle(value):
    return math.atan2(math.sin(value), math.cos(value))


class Navigator:
    def __init__(self, scene, profile, landmark_id, started):
        self.scene = scene
        self.profile = profile
        self.goal = scene.landmarks[landmark_id]
        self.grid = Grid(scene, profile)
        self.started = started
        self.settled_since = None
        self.reason = None
        self.path = []
        self.index = 0

    def update(self, pose, *, now, captured, speed, yaw_rate, _replanned=False):
        p = self.profile
        if self.reason:
            return Command(reason=self.reason)
        if (
            not all(
                math.isfinite(v)
                for v in (now, captured, speed, yaw_rate, pose.x, pose.y, pose.yaw)
            )
            or captured > now
            or now - captured > p.poseFreshnessMs / 1000
        ):
            self.reason = "LOCALIZATION_FAILED"
        elif now - self.started >= p.deadlineMs / 1000:
            self.reason = "DEADLINE_EXCEEDED"
        elif not self.grid.free(pose.x, pose.y) or not self.grid.free(
            self.goal.x, self.goal.y
        ):
            self.reason = "PATH_BLOCKED"
        if self.reason:
            return Command(reason=self.reason)
        distance = math.hypot(self.goal.x - pose.x, self.goal.y - pose.y)
        if distance <= p.arrivalToleranceM:
            heading = angle(self.goal.yaw - pose.yaw)
            if abs(heading) > p.headingToleranceRad:
                self.settled_since = None
                return Command(yaw=math.copysign(p.maxYawRateRadps, heading))
            if abs(speed) <= p.stableSpeedMps and abs(yaw_rate) <= p.stableYawRateRadps:
                if self.settled_since is None:
                    self.settled_since = now
                return Command(arrived=now - self.settled_since >= p.settleMs / 1000)
            self.settled_since = None
            return Command()
        self.settled_since = None
        if not self.path or self.index >= len(self.path):
            cells = plan_cells(
                self.grid.blocked,
                self.grid.cell(pose.x, pose.y),
                self.grid.cell(self.goal.x, self.goal.y),
                self.grid.bounds,
                prefer_clearance=True,
            )
            if cells is None:
                self.reason = "PATH_BLOCKED"
                return Command(reason=self.reason)
            # Collapse collinear segments only; never cut obstacle corners.
            points = [self.grid.point(c) for c in cells]
            self.path = []
            for i in range(1, len(points) - 1):
                if (cells[i][0] - cells[i - 1][0], cells[i][1] - cells[i - 1][1]) != (
                    cells[i + 1][0] - cells[i][0],
                    cells[i + 1][1] - cells[i][1],
                ):
                    self.path.append(points[i])
            self.path.append((self.goal.x, self.goal.y))
            self.index = 0
        waypoint = self.path[self.index]
        if (
            math.dist((pose.x, pose.y), waypoint) < self.grid.resolution / 2
            and self.index < len(self.path) - 1
        ):
            self.index += 1
            waypoint = self.path[self.index]
        if not self.grid.segment_free((pose.x, pose.y), waypoint):
            self.path = []
            if _replanned:
                self.reason = "PATH_BLOCKED"
                return Command(reason=self.reason)
            return self.update(
                pose,
                now=now,
                captured=captured,
                speed=speed,
                yaw_rate=yaw_rate,
                _replanned=True,
            )
        heading = angle(
            math.atan2(waypoint[1] - pose.y, waypoint[0] - pose.x) - pose.yaw
        )
        # The walking policy has a command deadband. Shrinking a command with
        # remaining distance/heading can leave a valid task permanently stalled.
        # Use the approved effective command, alternating turning and advancing;
        # arrival/settlement above always commands zero. No startup overdrive.
        if abs(heading) > p.headingToleranceRad:
            return Command(yaw=math.copysign(p.maxYawRateRadps, heading))
        return Command(vx=p.maxSpeedMps)
