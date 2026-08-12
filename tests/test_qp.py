import os
import unittest

import numpy as np
import mujoco

from activeguide.constraints import (
    JointLimitConstraint,
    RcmConstraint,
    SdfConstraint,
)
from activeguide.fixtures import ForbiddenRegionFixture
from activeguide.kinematics import RobotKinematics
from activeguide.qp import QpController
from activeguide.sdf import Sphere

SCENE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "model", "scene_panda.xml")


class QpTestBase(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(SCENE)
        self.kin = RobotKinematics(self.model)
        self.reset()
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "forbidden")
        self.sdf = Sphere(self.model.geom_pos[gid], float(self.model.geom_size[gid][0]))
        self.anat = np.asarray(self.model.geom_pos[gid], dtype=float)
        self.trocar = self.kin.site_position("trocar")
        self.goal = self.kin.site_position("goal")

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.kin.data, 0)
        self.kin.forward()


class TestBaselineEquivalence(QpTestBase):
    """The QP must reproduce the trusted Phase-0 law in the regime where they
    are meant to agree.

    The two enforcement laws are not the same in general: ``fixtures``
    hard-clamps the inward normal speed so the next Euler step lands exactly on
    the margin, while the VFI brakes smoothly via ``d_dot >= -eta * d``. They
    coincide *exactly* when ``eta = 1/dt``, since the VFI bound then reads
    ``vn >= -(d - margin)/dt`` -- literally the baseline's dt-aware cap.

    That makes this a real cross-validation of the new stack: the Jacobian, the
    SDF chain rule, and the QP assembly must all be right for it to hold.
    """

    MARGIN = 0.004
    ATOL = 1e-8

    def _compare(self, dt, speed):
        p = self.kin.site_position("tool_tip")
        n = self.sdf.normal(p)
        v_des = -n * speed                      # straight at the forbidden region

        base = ForbiddenRegionFixture(self.sdf, margin=self.MARGIN, restore_rate=0.0)
        v_base, _, _ = base.filter_velocity(p, v_des, dt)

        con = SdfConstraint(self.sdf, "tool_tip", margin=self.MARGIN, eta=1.0 / dt)
        # Near-zero damping and a slack velocity box so nothing but the wall
        # constraint can shape the answer.
        ctl = QpController(self.kin, "tool_tip", [con], damping=1e-9, qd_limit=1e4)
        _, info = ctl.solve(v_des)
        return info["v_actual"], v_base

    def test_matches_baseline_unclamped(self):
        """Free space: both laws are the identity."""
        v_qp, v_base = self._compare(dt=0.002, speed=0.02)
        np.testing.assert_allclose(v_qp, v_base, atol=self.ATOL)

    def test_matches_baseline_when_clamp_engages(self):
        """The regime that matters: the wall is actively limiting the motion.

        dt=0.1 with a 2 m/s command leaves only d/dt of inward speed, so the cap
        is genuinely active rather than trivially satisfied.
        """
        dt, speed = 0.1, 2.0
        p = self.kin.site_position("tool_tip")
        d = self.sdf.distance(p) - self.MARGIN
        self.assertGreater(speed, d / dt)        # assert the clamp really binds
        v_qp, v_base = self._compare(dt=dt, speed=speed)
        np.testing.assert_allclose(v_qp, v_base, atol=self.ATOL)
        # and the capped inward speed is exactly d/dt
        n = self.sdf.normal(p)
        self.assertAlmostEqual(float(np.dot(v_qp, n)), -d / dt, places=6)

    def test_matches_baseline_across_regimes(self):
        for dt in (0.002, 0.02, 0.1):
            for speed in (0.02, 0.5, 2.0):
                with self.subTest(dt=dt, speed=speed):
                    v_qp, v_base = self._compare(dt, speed)
                    np.testing.assert_allclose(v_qp, v_base, atol=self.ATOL)


class TestTracking(QpTestBase):
    def test_tracks_desired_velocity_when_unconstrained(self):
        ctl = QpController(self.kin, "tool_tip", [], damping=1e-9)
        v_des = np.array([0.0, -0.01, 0.0])
        _, info = ctl.solve(v_des)
        np.testing.assert_allclose(info["v_actual"], v_des, atol=1e-6)
        self.assertLess(info["tracking_error"], 1e-6)

    def test_no_constraints_is_well_formed(self):
        ctl = QpController(self.kin, "tool_tip", [])
        qdot, info = ctl.solve(np.zeros(3))
        self.assertEqual(qdot.shape, (7,))
        self.assertEqual(info["status"], "ok")

    def test_velocity_box_is_respected(self):
        ctl = QpController(self.kin, "tool_tip", [], damping=1e-9, qd_limit=0.05)
        qdot, _ = ctl.solve(np.array([0.0, -5.0, 0.0]))   # absurdly fast command
        self.assertTrue(np.all(np.abs(qdot) <= 0.05 + 1e-6))


class TestClosedLoopSafety(QpTestBase):
    """Drive the tip at the goal for a full run and assert nothing is breached."""

    DT = 0.002
    MARGIN = 0.004

    def _run(self, steps=2500, speed=0.03):
        wall = SdfConstraint(self.sdf, "tool_tip", margin=self.MARGIN)
        limits = JointLimitConstraint()
        rcm = RcmConstraint(self.trocar, tol=0.0005)
        ctl = QpController(self.kin, "tool_tip", [wall, limits, rcm])

        worst = dict(wall=-np.inf, limits=-np.inf, rcm=0.0)
        for _ in range(steps):
            p = self.kin.site_position("tool_tip")
            v = self.goal - p
            s = float(np.linalg.norm(v))
            if s > speed:
                v *= speed / s
            ctl.step(v, self.DT)
            worst["wall"] = max(worst["wall"], wall.violation(self.kin))
            worst["limits"] = max(worst["limits"], limits.violation(self.kin))
            worst["rcm"] = max(worst["rcm"], rcm.deviation(self.kin))
            if float(np.linalg.norm(self.goal - self.kin.site_position("tool_tip"))) < 1e-3:
                break
        return ctl, worst

    def test_reaches_goal(self):
        ctl, _ = self._run()
        err = float(np.linalg.norm(self.goal - self.kin.site_position("tool_tip")))
        self.assertLess(err, 1.5e-3)

    def test_never_enters_forbidden_region(self):
        _, worst = self._run()
        self.assertLessEqual(worst["wall"], 0.0)

    def test_never_violates_joint_limits(self):
        _, worst = self._run()
        self.assertLessEqual(worst["limits"], 0.0)

    def test_rcm_held_within_radial_tolerance(self):
        """The sqrt(2) correction is what makes this bound the *radial* error."""
        _, worst = self._run()
        self.assertLessEqual(worst["rcm"], 0.0005 + 1e-9)

    def test_no_infeasible_solves_on_a_reasonable_task(self):
        ctl, _ = self._run()
        self.assertEqual(ctl.n_infeasible, 0)

    def test_timing_is_recorded(self):
        ctl, _ = self._run()
        t = ctl.timing()
        self.assertGreater(t["n"], 100)
        self.assertGreater(t["p99_us"], 0.0)


class TestInfeasibility(QpTestBase):
    """An unsatisfiable problem must degrade, not crash."""

    def test_slack_path_returns_finite_velocity_and_flags_itself(self):
        # A margin larger than the sphere's surroundings puts the tip deep inside
        # the violated zone, so the VFI *demands* a large outward velocity; a
        # tiny velocity box then makes that impossible.
        con = SdfConstraint(self.sdf, "tool_tip", margin=0.5, eta=50.0)
        ctl = QpController(self.kin, "tool_tip", [con], qd_limit=1e-4)
        qdot, info = ctl.solve(np.array([0.0, 0.0, 0.0]))
        self.assertEqual(qdot.shape, (7,))
        self.assertTrue(np.all(np.isfinite(qdot)))
        self.assertTrue(info["slack_used"])
        self.assertIn(info["status"], ("slack", "failed"))
        self.assertEqual(ctl.n_infeasible, 1)
        self.assertGreater(ctl.timing()["infeasible_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
