"""Joint-space virtual-fixture controller: one QP per control step.

Phases 0-1 enforced fixtures by *sequentially* projecting a Cartesian velocity --
guidance first, then a safety clamp last (``fixtures.FixtureStack``). That
ordering is a workaround for not being able to satisfy several constraints at
once: whichever fixture runs last wins, and nothing in the chain knows about
joint limits or the trocar at all.

Here every fixture, joint limit, and the RCM enter one quadratic program, so they
are satisfied *simultaneously* or the problem is reported infeasible:

    min_qdot   || J_p qdot - v_des ||^2  +  lambda || qdot ||^2
    s.t.       G qdot <= h                 (all VFI rows: walls, shaft, limits, RCM)
               qdot_min <= qdot <= qdot_max

``lambda`` is Tikhonov damping: it resolves the arm's redundancy by preferring
small joint motion, and keeps ``P`` positive definite near singularities. It
plays the same role as the DLS damping in ``panda_ik_cpp``, but here it damps a
*constrained* solve rather than a pseudo-inverse.
"""

from __future__ import annotations

import time

import numpy as np
import qpsolvers

from .constraints import stack
from .kinematics import RobotKinematics


class QpController:
    """Map a desired tool-tip velocity to a safe joint velocity.

    Parameters
    ----------
    kin           robot kinematics wrapper
    site          tool site whose velocity tracks ``v_des`` (e.g. "tool_tip")
    constraints   list of ``constraints.Constraint``
    damping       Tikhonov weight (lambda) on ``||qdot||^2``
    qd_limit      symmetric joint-velocity bound (rad/s)
    solver        qpsolvers backend name. Default ``daqp``: this QP is small and
                  dense (7 variables, ~15 rows), which is exactly the regime a
                  dense active-set solver wins. Measured on the Panda reach task,
                  p99 solve time was 1355 us for daqp and 1519 us for quadprog
                  against 5371 us for OSQP -- only the first two fit inside a
                  500 Hz control period. All three produce the same trajectory,
                  so the choice is purely about latency.
    slack_penalty weight on constraint relaxation, used only after a hard solve
                  fails (see ``solve``)
    """

    def __init__(self, kin: RobotKinematics, site: str, constraints,
                 damping: float = 1e-3, qd_limit: float = 1.5,
                 solver: str = "daqp", slack_penalty: float = 1e6):
        self.kin = kin
        self.site = site
        self.constraints = list(constraints)
        self.damping = float(damping)
        self.qd_limit = float(qd_limit)
        self.solver = solver
        self.slack_penalty = float(slack_penalty)

        # Running diagnostics, reported by the benchmark.
        self.n_solves = 0
        self.n_infeasible = 0
        self.solve_times: list[float] = []

    # --- the QP ------------------------------------------------------------
    def _objective(self, v_des):
        """Return (P, q) for ``||J qdot - v_des||^2 + lambda ||qdot||^2``."""
        Jp, _ = self.kin.site_jacobian(self.site)
        nv = self.kin.nv
        P = Jp.T @ Jp + self.damping * np.eye(nv)
        q = -(Jp.T @ np.asarray(v_des, dtype=float))
        return P, q, Jp

    def solve(self, v_des) -> tuple[np.ndarray, dict]:
        """Solve for joint velocity. Returns ``(qdot, info)``.

        Tries the hard problem first, so every constraint is inviolable whenever
        that is achievable. Only if it is genuinely infeasible do we re-solve
        with penalized slack -- that keeps a run going instead of crashing, while
        ``info["slack_used"]`` and ``n_infeasible`` record honestly that a
        constraint had to be relaxed. Silently slackening safety constraints on
        every step would make the benchmark meaningless.
        """
        t0 = time.perf_counter()
        P, q, Jp = self._objective(v_des)
        G, h = stack(self.constraints, self.kin)
        nv = self.kin.nv
        lb = np.full(nv, -self.qd_limit)
        ub = np.full(nv, self.qd_limit)

        qdot = qpsolvers.solve_qp(
            P, q,
            G=G if G.size else None,
            h=h if h.size else None,
            lb=lb, ub=ub, solver=self.solver,
        )

        slack_used = False
        if qdot is None:
            slack_used = True
            self.n_infeasible += 1
            qdot = self._solve_with_slack(P, q, G, h, lb, ub)

        dt_solve = time.perf_counter() - t0
        self.n_solves += 1
        self.solve_times.append(dt_solve)

        if qdot is None:                        # even the relaxed QP failed
            qdot = np.zeros(nv)                 # safest possible action: stop
            info = dict(status="failed", slack_used=True, solve_time=dt_solve,
                        tracking_error=float(np.linalg.norm(v_des)))
            return qdot, info

        v_actual = Jp @ qdot
        info = dict(
            status="slack" if slack_used else "ok",
            slack_used=slack_used,
            solve_time=dt_solve,
            v_actual=v_actual,
            tracking_error=float(np.linalg.norm(v_actual - np.asarray(v_des, float))),
            manipulability=self.kin.manipulability(Jp),
            n_rows=int(G.shape[0]),
        )
        return qdot, info

    def _solve_with_slack(self, P, q, G, h, lb, ub):
        """Re-solve with per-row slack ``s >= 0`` penalized at ``slack_penalty``.

        Variables become ``[qdot, s]``. Constraints relax to ``G qdot - s <= h``.
        """
        nv = self.kin.nv
        m = int(G.shape[0])
        if m == 0:
            return None

        Pa = np.zeros((nv + m, nv + m))
        Pa[:nv, :nv] = P
        Pa[nv:, nv:] = self.slack_penalty * np.eye(m)
        qa = np.concatenate([q, np.zeros(m)])

        Ga = np.hstack([G, -np.eye(m)])
        lba = np.concatenate([lb, np.zeros(m)])
        uba = np.concatenate([ub, np.full(m, np.inf)])

        sol = qpsolvers.solve_qp(Pa, qa, G=Ga, h=h, lb=lba, ub=uba,
                                 solver=self.solver)
        return None if sol is None else sol[:nv]

    # --- integration -------------------------------------------------------
    def step(self, v_des, dt: float) -> tuple[np.ndarray, dict]:
        """Solve, integrate the configuration, and refresh kinematics."""
        qdot, info = self.solve(v_des)
        self.kin.set_q(self.kin.q + qdot * dt)
        return qdot, info

    # --- diagnostics -------------------------------------------------------
    def timing(self) -> dict:
        """Solve-time summary in microseconds, for the real-time check."""
        if not self.solve_times:
            return {}
        t = np.asarray(self.solve_times) * 1e6
        return dict(
            mean_us=float(t.mean()),
            p50_us=float(np.percentile(t, 50)),
            p99_us=float(np.percentile(t, 99)),
            max_us=float(t.max()),
            n=int(t.size),
            infeasible_rate=self.n_infeasible / max(self.n_solves, 1),
        )
