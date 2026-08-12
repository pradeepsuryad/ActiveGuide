"""Active constraints (virtual fixtures).

Phase 0 ships the Forbidden-Region Virtual Fixture (FRVF): a virtual wall that
keeps the tool out of protected anatomy. Two responsibilities, deliberately
separated so they can be retargeted independently later:

  1. ENFORCEMENT  (filter_velocity) -- runs on the robot/slave side. Projects
     the commanded motion so the tool cannot cross the wall. This is the part
     that is *always* active, regardless of what master device is used.

  2. FEEDBACK     (feedback_force / warning_level) -- the operator-facing cue.
     `feedback_force` is what you would render on a force-capable master (dVRK
     MTM); `warning_level` is the 0..1 signal you map to Quest controller
     vibration when real force isn't available (sensory substitution).

Distances/normals come from an `sdf.SDF` (see sdf.py for the sign convention).
"""

from __future__ import annotations

import numpy as np

from .sdf import SDF


class ForbiddenRegionFixture:
    def __init__(
        self,
        sdf: SDF,
        margin: float = 0.010,
        influence: float = 0.040,
        stiffness: float = 300.0,
        damping: float = 5.0,
        restore_rate: float = 5.0,
    ):
        """
        margin        SDF distance the tool tip is held at (wall standoff, m).
                      Set to tool_radius + clearance so the tool body never
                      visually intersects the protected surface.
        influence     distance from the wall at which the warning cue begins (m).
        stiffness     virtual-wall spring constant for the rendered force (N/m).
        damping       normal-direction damping for the rendered force (N*s/m).
        restore_rate  outward recovery speed gain if the tool overshoots inside
                      the margin (1/s).
        """
        self.sdf = sdf
        self.margin = float(margin)
        self.influence = float(influence)
        self.K = float(stiffness)
        self.D = float(damping)
        self.restore_rate = float(restore_rate)

    # --- 1. ENFORCEMENT (robot side) --------------------------------------
    def filter_velocity(self, p, v_des, dt):
        """Project commanded velocity so the tool cannot cross the wall.

        Returns (v_allowed, distance, outward_normal).

        The tool approaches freely until the *next* Euler step would carry it
        past the margin; then the inward normal speed is capped so that step
        lands exactly on the margin (no overshoot, dt-aware). The tangential
        component is always preserved, so the tool slides ALONG the wall, and
        outward motion is never opposed. A restoring term recovers from any
        residual penetration (e.g. if the wall itself moved).
        """
        p = np.asarray(p, dtype=float)
        v = np.array(v_des, dtype=float)
        d = self.sdf.distance(p)
        n = self.sdf.normal(p)

        vn = float(np.dot(v, n))               # +ve = moving outward (safe)
        # inward distance we may travel this step before reaching the margin
        max_inward_speed = max(0.0, d - self.margin) / dt
        if vn < -max_inward_speed:             # this step would breach the wall
            v = v - (vn + max_inward_speed) * n  # cap inward speed to the margin
        if d < self.margin and self.restore_rate > 0.0:
            v = v + self.restore_rate * (self.margin - d) * n  # recover outward
        return v, d, n

    # --- 2a. FEEDBACK as force (force-capable master, e.g. dVRK MTM) -------
    def feedback_force(self, p, v):
        """Spring-damper wall force to render on the master. Zero until the
        tool enters the margin; then it pushes outward along the normal."""
        p = np.asarray(p, dtype=float)
        d = self.sdf.distance(p)
        n = self.sdf.normal(p)
        pen = self.margin - d
        if pen <= 0.0:
            return np.zeros(3), 0.0
        vn = float(np.dot(np.asarray(v, dtype=float), n))
        force = (self.K * pen) * n - (self.D * vn) * n
        return force, float(np.linalg.norm(force))

    # --- 2b. FEEDBACK as warning (vibrotactile substitution, Quest) -------
    def warning_level(self, p) -> float:
        """0..1 proximity signal for controller vibration amplitude.

        0 outside the influence zone, ramps to 1 at the wall (margin)."""
        d = self.sdf.distance(p)
        if d >= self.influence:
            return 0.0
        if d <= self.margin:
            return 1.0
        return float((self.influence - d) / (self.influence - self.margin))


class GuidanceFixture:
    """Guidance Virtual Fixture (GVF): assist motion ALONG a reference path.

    The complement of the forbidden region. Motion along the path's tangent is
    left free (the surgeon sets the pace); perpendicular motion is blended with
    an attraction back onto the path. ``strength`` in [0,1] is the assistance
    level: 0 = no guidance (free), 1 = locked to the path (free only to slide
    along it). This is the "software-assisted guideline".
    """

    def __init__(self, path, strength: float = 0.85,
                 attract_gain: float = 40.0, max_attract: float = 0.15):
        self.path = path
        self.strength = float(strength)
        self.attract_gain = float(attract_gain)
        self.max_attract = float(max_attract)

    def filter_velocity(self, p, v_des, dt):
        """Return (v_assisted, perpendicular_offset, tangent)."""
        p = np.asarray(p, dtype=float)
        v = np.array(v_des, dtype=float)
        c, t = self.path.closest_point(p)
        offset = p - c                              # displacement off the path

        v_t = np.dot(v, t) * t                      # along-path (free)
        v_perp = v - v_t                            # off-path component

        v_attract = -self.attract_gain * offset     # pull back toward the path
        v_attract = v_attract - np.dot(v_attract, t) * t   # keep it perpendicular
        s = np.linalg.norm(v_attract)
        if s > self.max_attract:
            v_attract *= self.max_attract / s

        k = self.strength
        v_out = v_t + (1.0 - k) * v_perp + k * v_attract
        return v_out, offset, t


class FixtureStack:
    """Compose fixtures. Guidance assists first; the forbidden-region clamp is
    applied LAST so safety always overrides assistance."""

    def __init__(self, guidance: "GuidanceFixture | None" = None,
                 forbidden: "ForbiddenRegionFixture | None" = None):
        self.guidance = guidance
        self.forbidden = forbidden

    def filter_velocity(self, p, v_des, dt):
        v = np.array(v_des, dtype=float)
        info: dict = {}
        if self.guidance is not None:
            v, offset, _ = self.guidance.filter_velocity(p, v, dt)
            info["path_offset"] = float(np.linalg.norm(offset))
        if self.forbidden is not None:
            v, d, n = self.forbidden.filter_velocity(p, v, dt)
            info["distance"] = d
            info["warning"] = self.forbidden.warning_level(p)
        return v, info
