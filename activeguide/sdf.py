"""Signed-distance-field primitives.

Convention used throughout ActiveGuide:
    distance(p) > 0   -> p is OUTSIDE the forbidden region (safe)
    distance(p) == 0  -> p is exactly on the forbidden surface
    distance(p) < 0   -> p is INSIDE the forbidden region (penetrated)

    gradient(p)       -> points in the direction of INCREASING distance,
                         i.e. the outward / "escape" direction.

This single representation feeds both the controller (penetration + push-out
normal) and, later, the motion planner (clearance). Keep it the source of truth.
"""

from __future__ import annotations

import numpy as np


class SDF:
    """Base class. Subclasses must implement ``distance``.

    A finite-difference ``gradient`` is provided so any new primitive (or a
    mesh-based SDF later) works out of the box; override it where an analytic
    form is cheaper or more accurate.
    """

    eps: float = 1e-6

    def distance(self, p: np.ndarray) -> float:
        raise NotImplementedError

    def gradient(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        g = np.zeros(3)
        for i in range(3):
            dp = np.zeros(3)
            dp[i] = self.eps
            g[i] = (self.distance(p + dp) - self.distance(p - dp)) / (2.0 * self.eps)
        return g

    def normal(self, p: np.ndarray) -> np.ndarray:
        """Unit outward normal at ``p`` (falls back to +x if undefined)."""
        g = self.gradient(p)
        n = np.linalg.norm(g)
        if n < 1e-9:
            return np.array([1.0, 0.0, 0.0])
        return g / n


class Sphere(SDF):
    """Forbidden region = the inside of a sphere (e.g. a protected organ)."""

    def __init__(self, center, radius: float):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)

    def distance(self, p) -> float:
        return float(np.linalg.norm(np.asarray(p, dtype=float) - self.center) - self.radius)

    def gradient(self, p) -> np.ndarray:
        d = np.asarray(p, dtype=float) - self.center
        n = np.linalg.norm(d)
        if n < 1e-9:
            return np.array([1.0, 0.0, 0.0])
        return d / n


class Plane(SDF):
    """Forbidden region = the half-space behind a plane.

    ``normal`` points INTO the safe half-space (away from the forbidden side).
    """

    def __init__(self, point, normal):
        self.point = np.asarray(point, dtype=float)
        n = np.asarray(normal, dtype=float)
        self.n = n / np.linalg.norm(n)

    def distance(self, p) -> float:
        return float(np.dot(np.asarray(p, dtype=float) - self.point, self.n))

    def gradient(self, p) -> np.ndarray:
        return self.n.copy()


class Box(SDF):
    """Forbidden region = the inside of an axis-aligned box (uses numeric grad)."""

    def __init__(self, center, half_extents):
        self.center = np.asarray(center, dtype=float)
        self.half = np.asarray(half_extents, dtype=float)

    def distance(self, p) -> float:
        q = np.abs(np.asarray(p, dtype=float) - self.center) - self.half
        outside = np.linalg.norm(np.maximum(q, 0.0))
        inside = min(float(np.max(q)), 0.0)
        return outside + inside


class Union(SDF):
    """Union of forbidden regions: a point is unsafe if inside ANY child.

    ``distance`` is the min over children (positive only when outside all).
    """

    def __init__(self, sdfs):
        self.sdfs = list(sdfs)

    def distance(self, p) -> float:
        return min(s.distance(p) for s in self.sdfs)
