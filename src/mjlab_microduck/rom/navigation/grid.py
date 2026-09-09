"""Occupancy derived from the exact fixture used for MuJoCo collision geometry."""

import math


class Grid:
    def __init__(self, scene, profile):
        self.scene = scene
        self.resolution = profile.gridResolutionM
        a = scene.area
        r = self.resolution
        self.bounds = (
            math.ceil((a.maxX - a.minX) / r),
            math.ceil((a.maxY - a.minY) / r),
        )
        if self.bounds[0] * self.bounds[1] > 250000:
            raise ValueError("map exceeds planner budget")
        # Half diagonal makes every point in a free cell satisfy the footprint margin.
        self.margin = profile.robotRadiusM + profile.clearanceM
        inflated = self.margin + r / math.sqrt(2)
        self.blocked = set()
        for x in range(self.bounds[0]):
            for y in range(self.bounds[1]):
                px, py = self.point((x, y))
                if not a.contains(px, py, inflated) or any(
                    o.minX - inflated <= px <= o.maxX + inflated
                    and o.minY - inflated <= py <= o.maxY + inflated
                    for o in scene.obstacles
                ):
                    self.blocked.add((x, y))

    def cell(self, x, y):
        return (
            math.floor((x - self.scene.area.minX) / self.resolution),
            math.floor((y - self.scene.area.minY) / self.resolution),
        )

    def point(self, cell):
        return (
            self.scene.area.minX + (cell[0] + 0.5) * self.resolution,
            self.scene.area.minY + (cell[1] + 0.5) * self.resolution,
        )

    def free(self, x, y):
        c = self.cell(x, y)
        return (
            0 <= c[0] < self.bounds[0]
            and 0 <= c[1] < self.bounds[1]
            and c not in self.blocked
        )

    def segment_free(self, start, end):
        distance = math.dist(start, end)
        steps = max(1, math.ceil(distance / (self.resolution / 4)))
        return all(
            self.free(
                start[0] + (end[0] - start[0]) * i / steps,
                start[1] + (end[1] - start[1]) * i / steps,
            )
            for i in range(steps + 1)
        )
