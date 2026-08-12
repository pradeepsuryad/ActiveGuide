"""The Phase 0/1 controller, extended to a robot the obvious way.

This is the honest baseline the joint-space QP is measured against. Given a
manipulator, the natural thing to do with a Cartesian fixture is:

    1. filter the commanded tool velocity in Cartesian space  (FixtureStack)
    2. map the result to joint velocity with a damped pseudo-inverse

That is a reasonable design, and for the *tool tip* it works: the tip really
does stay out of the forbidden region. What it cannot do is see anything that
does not live in tool-tip Cartesian space --

    * joint limits          the pseudo-inverse has no notion of them
    * the trocar / RCM      not expressible as a constraint on the tip
    * the instrument shaft  the fixture constrains one point, not a body

so it can satisfy the wall perfectly while driving a joint into its stop or
levering the shaft out of the patient's body wall. Those failures are the point
of the comparison, and they are invisible until you put a real robot underneath.

Deliberately charitable: same damping, same velocity clamp, same fixtures, same
integration as ``qp.QpController``. The only difference is *where* enforcement
happens -- Cartesian output space here, joint velocity space there.
"""

from __future__ import annotations

import time

import numpy as np

from .kinematics import RobotKinematics


class CartesianBaselineController:
    """Cartesian fixture filtering + damped least-squares inverse kinematics.

    Parameters mirror ``qp.QpController`` so the two can be swapped in a
    benchmark without changing anything else.
    """

    def __init__(self, kin: RobotKinematics, site: str, stack,
                 damping: float = 1e-3, qd_limit: float = 1.5, equalities=()):
        self.kin = kin
        self.site = site
        self.stack = stack                 # fixtures.FixtureStack
        self.damping = float(damping)
        self.qd_limit = float(qd_limit)
        # Joint couplings are properties of the *mechanism*, not choices a
        # controller gets to make -- on the PSM the parallelogram is welded
        # steel. Withholding them from the baseline would let it command motion
        # the hardware cannot execute and then blame it for the resulting RCM
        # error, which would be a strawman. The baseline therefore gets the same
        # couplings the QP does, imposed by projecting onto their null space.
        self.equalities = list(equalities)

        self.n_solves = 0
        self.n_infeasible = 0              # never infeasible: it cannot refuse
        self.solve_times: list[float] = []

    def _nullspace(self):
        """Orthonormal basis ``N`` of ``{qdot : C qdot = 0}``, or None.

        Solving in reduced coordinates -- rather than projecting an
        unconstrained solution afterwards -- is the difference between a
        baseline that handles the mechanism correctly and one that is merely
        sloppy. Post-hoc projection of a damped-least-squares solution is *not*
        the least-squares solution subject to the constraint, and using it would
        manufacture a win that says nothing about enforcement space.
        """
        if not self.equalities:
            return None
        from .constraints import stack_eq
        C, _ = stack_eq(self.equalities, self.kin)
        if C.size == 0:
            return None
        _, s, Vt = np.linalg.svd(C)
        rank = int((s > 1e-9).sum())
        return Vt[rank:].T                    # (nv, nv - rank)

    def solve(self, v_des, dt: float):
        t0 = time.perf_counter()
        p = self.kin.site_position(self.site)

        # 1. Cartesian enforcement, exactly as Phases 0-1 did it.
        v_safe, info_fx = self.stack.filter_velocity(p, v_des, dt)

        # 2. Damped least squares, solved in the mechanism's reduced coordinates
        #    so joint couplings hold exactly:  qdot = N (J N)^+_damped v
        J, _ = self.kin.site_jacobian(self.site)
        v_safe = np.asarray(v_safe, dtype=float)
        N = self._nullspace()
        Jr = J if N is None else J @ N
        A = Jr @ Jr.T + self.damping * np.eye(3)
        qr = Jr.T @ np.linalg.solve(A, v_safe)
        qdot = qr if N is None else N @ qr

        # Clamp to the same velocity box the QP is given. Note this is a *post
        # hoc* clip: it can bend the achieved direction, whereas the QP treats
        # the box as a constraint and re-optimises within it.
        big = np.abs(qdot).max()
        if big > self.qd_limit:
            qdot *= self.qd_limit / big

        dt_solve = time.perf_counter() - t0
        self.n_solves += 1
        self.solve_times.append(dt_solve)

        v_actual = J @ qdot
        info = dict(
            status="ok",
            slack_used=False,
            solve_time=dt_solve,
            v_actual=v_actual,
            tracking_error=float(np.linalg.norm(v_actual - np.asarray(v_des, float))),
            manipulability=self.kin.manipulability(J),
            **info_fx,
        )
        return qdot, info

    def step(self, v_des, dt: float):
        qdot, info = self.solve(v_des, dt)
        self.kin.set_q(self.kin.q + qdot * dt)
        return qdot, info

    def timing(self) -> dict:
        if not self.solve_times:
            return {}
        t = np.asarray(self.solve_times) * 1e6
        return dict(
            mean_us=float(t.mean()),
            p50_us=float(np.percentile(t, 50)),
            p99_us=float(np.percentile(t, 99)),
            max_us=float(t.max()),
            n=int(t.size),
            infeasible_rate=0.0,
        )
