"""Guidance paths.

A guidance virtual fixture needs a reference curve and, at any query point, the
nearest point on it plus the local tangent (the "preferred direction"). A
polyline is enough for Phase 1; swap in a spline later without touching the
fixture, since `GuidanceFixture` only relies on `closest_point`.
"""

from __future__ import annotations

import numpy as np


class Polyline:
    def __init__(self, points):
        self.pts = np.asarray(points, dtype=float)
        if self.pts.ndim != 2 or self.pts.shape[1] != 3 or len(self.pts) < 2:
            raise ValueError("Polyline needs >=2 points of shape (N,3)")
        self.seg = np.diff(self.pts, axis=0)               # (N-1, 3)
        self.seg_len2 = np.einsum("ij,ij->i", self.seg, self.seg)

    def closest_point(self, p):
        """Return (closest_point, unit_tangent) for the nearest segment."""
        p = np.asarray(p, dtype=float)
        best_d2 = np.inf
        best_c = self.pts[0]
        best_t = np.array([1.0, 0.0, 0.0])
        for i in range(len(self.seg)):
            a = self.pts[i]
            ab = self.seg[i]
            L2 = self.seg_len2[i]
            if L2 < 1e-12:
                c, tang = a, np.zeros(3)
            else:
                s = np.clip(np.dot(p - a, ab) / L2, 0.0, 1.0)
                c = a + s * ab
                tang = ab / np.sqrt(L2)
            d = p - c
            d2 = float(np.dot(d, d))
            if d2 < best_d2:
                best_d2, best_c, best_t = d2, c, tang
        return best_c, best_t

    def distance(self, p) -> float:
        """Perpendicular distance from p to the path (>= 0)."""
        c, _ = self.closest_point(p)
        return float(np.linalg.norm(np.asarray(p, dtype=float) - c))
