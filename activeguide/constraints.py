"""Vector Field Inequality (VFI) constraints on joint velocity.

Every constraint here reduces to the same linear form, which is what makes the
QP controller possible:

    G qdot <= h

The VFI idea (Marinho et al., "Dynamic Active Constraints for Surgical Robots
Using Vector Field Inequalities"): for a scalar distance ``d(q)`` that must stay
non-negative, forbid it from shrinking faster than proportionally to how much
room is left,

    d_dot >= -eta * d        <=>    -J_d qdot <= eta * d

so the tool decelerates smoothly as it approaches a boundary instead of being
clipped at it. ``eta`` sets how hard the approach is braked (1/s).

The chain rule is what lets the Phase-0 SDF classes be reused verbatim:

    J_d = grad_p d(p)^T @ J_p(q)
          \\_ sdf.normal(p)   \\_ RobotKinematics.site_jacobian

So ``sdf.py`` never needed to change; it was already the constraint-geometry
backend for a real controller.

Note on guidance: a guidance fixture is *not* a hard constraint -- it reshapes
the operator's intent. It therefore stays in the QP objective (it pre-filters
``v_des``), not here. ``GuidanceFixture`` is reused unchanged.
"""

from __future__ import annotations

import numpy as np

from .kinematics import RobotKinematics
from .sdf import SDF


class Constraint:
    """Base class. Subclasses return VFI rows for the current configuration."""

    name: str = "constraint"

    def rows(self, kin: RobotKinematics) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(G, h)`` with shapes ``(k, nv)`` and ``(k,)``."""
        raise NotImplementedError

    def violation(self, kin: RobotKinematics) -> float:
        """How far this constraint is currently breached, in metres.

        ``<= 0`` means satisfied. Used for benchmark metrics, never by the QP.
        """
        raise NotImplementedError


class SdfConstraint(Constraint):
    """Keep a site outside an SDF region by at least ``margin``.

    This is the joint-space counterpart of
    ``fixtures.ForbiddenRegionFixture.filter_velocity``: identical geometry, but
    enforced on joint velocity through the Jacobian, so it composes with joint
    limits and the RCM instead of overriding them.
    """

    def __init__(self, sdf: SDF, site: str, margin: float = 0.010,
                 eta: float = 10.0, name: str | None = None):
        self.sdf = sdf
        self.site = site
        self.margin = float(margin)
        self.eta = float(eta)
        self.name = name or f"sdf[{site}]"

    def _distance(self, kin: RobotKinematics) -> tuple[float, np.ndarray]:
        p = kin.site_position(self.site)
        return self.sdf.distance(p) - self.margin, p

    def rows(self, kin):
        d, p = self._distance(kin)
        n = self.sdf.normal(p)                  # outward: increasing distance
        Jp, _ = kin.site_jacobian(self.site)
        G = -(n @ Jp).reshape(1, -1)            # -J_d
        h = np.array([self.eta * d])            # eta * d
        return G, h

    def violation(self, kin):
        d, _ = self._distance(kin)
        return float(-d)                        # >0 when inside the margin


class ShaftClearanceConstraint(Constraint):
    """Keep *every* sampled point on the instrument shaft clear of a region.

    A tip-only fixture cannot catch the shaft sweeping through anatomy while the
    tip itself stays legal -- a real failure mode in minimally invasive surgery,
    and one the Phase-0/1 point formulation is structurally blind to. One VFI row
    per sample point.
    """

    def __init__(self, sdf: SDF, sites, margin: float = 0.006,
                 eta: float = 10.0, name: str | None = None):
        self.sdf = sdf
        self.sites = list(sites)
        self.margin = float(margin)
        self.eta = float(eta)
        self.name = name or f"shaft[{len(self.sites)} pts]"

    def rows(self, kin):
        G = np.zeros((len(self.sites), kin.nv))
        h = np.zeros(len(self.sites))
        for i, site in enumerate(self.sites):
            p = kin.site_position(site)
            d = self.sdf.distance(p) - self.margin
            n = self.sdf.normal(p)
            Jp, _ = kin.site_jacobian(site)
            G[i] = -(n @ Jp)
            h[i] = self.eta * d
        return G, h

    def violation(self, kin):
        worst = -np.inf
        for site in self.sites:
            d = self.sdf.distance(kin.site_position(site)) - self.margin
            worst = max(worst, -d)
        return float(worst)


class JointLimitConstraint(Constraint):
    """Brake each joint before it reaches its mechanical stop.

    Same VFI form applied to the two distances ``q - q_min`` and ``q_max - q``.
    Cartesian point projection cannot express this at all, which is precisely
    why it silently drives joints into their limits.
    """

    name = "joint_limits"

    def __init__(self, eta: float = 10.0, margin: float = 0.0):
        self.eta = float(eta)
        self.margin = float(margin)

    def rows(self, kin):
        lo, hi = kin.joint_limits()
        q = kin.q
        nv = kin.nv
        I = np.eye(nv)

        d_lo = q - (lo + self.margin)           # room before the lower stop
        d_hi = (hi - self.margin) - q           # room before the upper stop

        # d_lo_dot = +qdot  ->  -qdot <= eta * d_lo
        # d_hi_dot = -qdot  ->  +qdot <= eta * d_hi
        G = np.vstack([-I, I])
        h = np.concatenate([self.eta * d_lo, self.eta * d_hi])

        # Unlimited joints produce inf room; a huge finite bound keeps the QP
        # numerically sane without constraining anything in practice.
        h = np.where(np.isfinite(h), h, 1e6)
        return G, h

    def violation(self, kin):
        lo, hi = kin.joint_limits()
        q = kin.q
        over = np.concatenate([lo - q, q - hi])
        over = over[np.isfinite(over)]
        return float(over.max()) if over.size else 0.0


class RcmConstraint(Constraint):
    """Hold the instrument shaft pivoting through a fixed trocar point.

    Formulation
    -----------
    The shaft is the rigid segment between sites ``proximal`` and ``distal``.
    For a rigid body the velocity of an interior point is the affine blend of
    the endpoint velocities, so the material point currently nearest the trocar
    has Jacobian

        J_c = (1 - lam) * J_proximal + lam * J_distal,    lam = s / L

    Let ``e = c - trocar`` be the lateral error and ``{b1, b2}`` an orthonormal
    basis of the plane perpendicular to the shaft axis. Rather than a single
    row on ``||e||`` -- which is undefined at ``e = 0``, exactly the state we
    start in and most want to hold -- constrain each signed component
    two-sidedly:

         b^T J_c qdot <= eta * (tol - e_b)
        -b^T J_c qdot <= eta * (tol + e_b)

    Four rows, always well defined, and a genuine box on lateral error in the
    shaft-perpendicular plane. This is the QP-native counterpart of the
    weighted RCM rows in ``panda_ik_cpp/src/rcm_ik_solver.cpp``: there the RCM
    was a soft, weighted block inside a damped-least-squares stack and could be
    traded away against pose error; here it is a hard inequality that pose
    tracking cannot violate.

    Why the tolerance is divided by sqrt(2)
    ---------------------------------------
    Bounding two perpendicular *components* by ``t`` bounds the radial error by
    the box corner, ``sqrt(2) * t``, not by ``t``. Measured directly: with a
    per-axis bound of 500 um the radial deviation settled at exactly 707.1 um,
    invariant to ``eta`` -- the constraint was active and tight, just enforcing a
    circle-inscribing square rather than the circle. Since a caller reads ``tol``
    as a radial tolerance, the per-axis bound is set to ``tol / sqrt(2)`` so the
    guarantee matches the name. The cost is conservatism along the diagonals,
    which is the right way to be wrong for a safety constraint.
    """

    _SQRT2 = float(np.sqrt(2.0))

    def __init__(self, trocar, proximal: str = "shaft_00", distal: str = "tool_tip",
                 tol: float = 0.0005, eta: float = 10.0):
        self.trocar = np.asarray(trocar, dtype=float)
        self.proximal = proximal
        self.distal = distal
        self.tol = float(tol)
        self.eta = float(eta)
        self.name = "rcm"

    @property
    def axis_tol(self) -> float:
        """Per-axis bound giving a radial guarantee of ``tol``."""
        return self.tol / self._SQRT2

    def _geometry(self, kin: RobotKinematics):
        pa = kin.site_position(self.proximal)
        pb = kin.site_position(self.distal)
        seg = pb - pa
        L = float(np.linalg.norm(seg))
        if L < 1e-9:
            raise ValueError(
                f"shaft sites {self.proximal!r} and {self.distal!r} coincide")
        u = seg / L
        s = float(np.dot(self.trocar - pa, u))
        lam = s / L                              # unclamped: report honestly if
                                                 # the trocar leaves the shaft
        c = pa + s * u                           # closest point on the axis
        e = c - self.trocar                      # lateral error (perp to u)
        return pa, pb, u, lam, c, e

    @staticmethod
    def _perp_basis(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Two unit vectors spanning the plane perpendicular to ``u``."""
        seed = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(seed, u)) > 0.9:           # nearly parallel: pick another
            seed = np.array([0.0, 1.0, 0.0])
        b1 = seed - np.dot(seed, u) * u
        b1 /= np.linalg.norm(b1)
        b2 = np.cross(u, b1)
        return b1, b2

    def rows(self, kin):
        _, _, u, lam, _, e = self._geometry(kin)
        Ja, _ = kin.site_jacobian(self.proximal)
        Jb, _ = kin.site_jacobian(self.distal)
        Jc = (1.0 - lam) * Ja + lam * Jb

        b1, b2 = self._perp_basis(u)
        t = self.axis_tol
        G = np.zeros((4, kin.nv))
        h = np.zeros(4)
        for i, b in enumerate((b1, b2)):
            row = b @ Jc
            e_b = float(np.dot(b, e))
            G[2 * i] = row
            h[2 * i] = self.eta * (t - e_b)
            G[2 * i + 1] = -row
            h[2 * i + 1] = self.eta * (t + e_b)
        return G, h

    def deviation(self, kin: RobotKinematics) -> float:
        """Perpendicular trocar-to-shaft-axis distance, in metres.

        The headline metric: comparable to the sub-millimetre RCM figures
        reported for closed-form RCM controllers.
        """
        *_, e = self._geometry(kin)
        return float(np.linalg.norm(e))

    def violation(self, kin):
        return float(self.deviation(kin) - self.tol)


def stack(constraints, kin: RobotKinematics) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate every constraint's rows into one ``(G, h)`` pair."""
    if not constraints:
        return np.zeros((0, kin.nv)), np.zeros(0)
    Gs, hs = zip(*(c.rows(kin) for c in constraints))
    return np.vstack(Gs), np.concatenate(hs)
