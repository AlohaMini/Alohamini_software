"""Deterministic planar kinematics and segment-ray scans; no tire/contact dynamics."""

import math


def rectangle(rect):
    if (
        len(rect) != 4
        or not all(math.isfinite(x) for x in rect)
        or rect[0] >= rect[2]
        or rect[1] >= rect[3]
    ):
        raise ValueError("Invalid world rectangle")
    x0, y0, x1, y1 = rect
    points = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return list(zip(points, points[1:] + points[:1], strict=True))


class World:
    def __init__(self, config):
        self.bounds = config["bounds"]
        self.obstacles = config.get("obstacles", [])
        self.segments = rectangle(self.bounds)
        for rect in self.obstacles:
            self.segments += rectangle(rect)
        self.x, self.y, self.yaw = config["initial_pose"]
        self.radius = float(config.get("robot_radius", 0.32))
        if (
            not all(math.isfinite(x) for x in (self.x, self.y, self.yaw, self.radius))
            or self.radius <= 0
        ):
            raise ValueError("Invalid initial pose or robot radius")
        if not self.free(self.x, self.y):
            raise ValueError("Initial footprint intersects world geometry")

    def free(self, x, y):
        x0, y0, x1, y1 = self.bounds
        r = self.radius
        if not x0 + r < x < x1 - r or not y0 + r < y < y1 - r:
            return False
        return all(
            math.hypot(x - min(max(x, a), c), y - min(max(y, b), d)) > r
            for a, b, c, d in self.obstacles
        )

    def step(self, vx, vy, wz, dt):
        if not all(math.isfinite(x) for x in (vx, vy, wz, dt)) or not 0 < dt <= 0.1:
            raise ValueError("Invalid simulation step")
        yaw = self.yaw + wz * dt / 2
        x = self.x + (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
        y = self.y + (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
        if self.free(x, y):
            self.x, self.y = x, y
        else:
            vx = vy = 0.0
        self.yaw = math.atan2(math.sin(self.yaw + wz * dt), math.cos(self.yaw + wz * dt))
        return vx, vy, wz

    def scan(self, xyz, rpy, count=360, maximum=8.0):
        x = self.x + xyz[0] * math.cos(self.yaw) - xyz[1] * math.sin(self.yaw)
        y = self.y + xyz[0] * math.sin(self.yaw) + xyz[1] * math.cos(self.yaw)
        result = []
        for i in range(count):
            angle = self.yaw + rpy[2] - math.pi + i * 2 * math.pi / (count - 1)
            dx, dy = math.cos(angle), math.sin(angle)
            distance = math.inf
            for (ax, ay), (bx, by) in self.segments:
                sx, sy = bx - ax, by - ay
                denominator = dx * sy - dy * sx
                if abs(denominator) < 1e-9:
                    continue
                t = ((ax - x) * sy - (ay - y) * sx) / denominator
                u = ((ax - x) * dy - (ay - y) * dx) / denominator
                if 0 <= t <= maximum and 0 <= u <= 1:
                    distance = min(distance, t)
            result.append(distance)
        return result
