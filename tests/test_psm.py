import os
import unittest

import numpy as np
import mujoco

from activeguide.constraints import (
    PSM_COUPLING,
    JointLimitConstraint,
    RcmConstraint,
    SdfConstraint,
    stack_eq,
)
from activeguide.kinematics import RobotKinematics
from activeguide.qp import QpController
from activeguide.sdf import Sphere

MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "model", "psm.xml")

# From the official description: remote_center_joint places the RCM at this
# offset from base_link.
RCM_IN_BASE = np.array([0.0, 0.4864, 0.0])

# model/psm.xml is generated, not committed -- it derives from upstream dVRK
# assets that are fetched rather than redistributed (see model/psm/NOTICE.md).
# Skip rather than fail on a fresh clone, and say how to fix it.
requires_psm = unittest.skipUnless(
    os.path.exists(MODEL),
    "model/psm.xml not built - run: python tools/build_psm.py --fetch",
)


@requires_psm
class PsmTestBase(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(MODEL)
        self.kin = RobotKinematics(self.model)
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.kin.data, 0)
        self.kin.forward()

    def jid(self, name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)


class TestPsmModel(PsmTestBase):
    def test_compiles_with_all_meshes(self):
        self.assertEqual(self.model.nmesh, 13)

    def test_mimic_couplings_present_as_equality_constraints(self):
        """The builder must restore what MuJoCo's URDF importer drops."""
        self.assertEqual(self.model.neq, 5)

    def test_six_actuated_instrument_joints(self):
        self.assertEqual(self.model.nu, 6)

    def test_required_sites_exist(self):
        for s in ("remote_center", "shaft_00", "shaft_50", "shaft_end", "tool_tip"):
            self.assertIsInstance(self.kin.site_position(s), np.ndarray)

    def test_remote_centre_matches_official_offset(self):
        base = self.kin.data.xpos[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")]
        np.testing.assert_allclose(
            self.kin.site_position("remote_center"), base + RCM_IN_BASE, atol=1e-9)

    def test_tool_tip_is_inserted_past_the_port(self):
        """At the home pose the instrument should be inside the patient."""
        rcm = self.kin.site_position("remote_center")
        a, b = self.kin.site_position("shaft_00"), self.kin.site_position("shaft_end")
        u = (b - a) / np.linalg.norm(b - a)
        depth = float(np.dot(self.kin.site_position("tool_tip") - rcm, u))
        self.assertGreater(depth, 0.05)          # >5 cm past the remote centre


class TestParallelogramIsLoadBearing(PsmTestBase):
    """Regression guard on the most dangerous failure mode in this conversion.

    The PSM's remote centre is mechanical, produced by a parallelogram the URDF
    declares via <mimic>. MuJoCo's URDF importer ignores <mimic> silently, so a
    naive import yields a model that loads cleanly, reports the wrong DOF count,
    and whose remote centre wanders by ~90 mm. These tests pin both halves of
    that: with the couplings the pivot holds to microns, and without them it
    does not -- so if someone ever removes the equality restoration, the second
    test fails and says why.
    """

    MIMIC = {
        "pitch_bottom_joint": ("pitch_back_joint", -1.0),
        "pitch_end_joint": ("pitch_back_joint", +1.0),
        "pitch_top_joint": ("pitch_back_joint", -1.0),
        "pitch_front_joint": ("pitch_back_joint", +1.0),
    }

    def _set_pose(self, yaw, pitch, insertion, coupled):
        q = np.zeros(self.kin.nv)
        q[self.jid("yaw_joint")] = yaw
        q[self.jid("pitch_back_joint")] = pitch
        q[self.jid("main_insertion_joint")] = insertion
        if coupled:
            for child, (driver, mult) in self.MIMIC.items():
                q[self.jid(child)] = mult * q[self.jid(driver)]
        self.kin.set_q(q)

    def _rcm_error(self):
        rcm = self.kin.site_position("remote_center")
        a, b = self.kin.site_position("shaft_00"), self.kin.site_position("shaft_end")
        u = (b - a) / np.linalg.norm(b - a)
        w = rcm - a
        return float(np.linalg.norm(w - np.dot(w, u) * u))

    POSES = [(y, p, i)
             for y in (-0.8, -0.4, 0.0, 0.4, 0.8)
             for p in (-0.6, -0.3, 0.0, 0.3, 0.6)
             for i in (0.0, 0.12, 0.24)]

    def test_coupled_parallelogram_holds_the_remote_centre(self):
        worst = 0.0
        for yaw, pitch, ins in self.POSES:
            self._set_pose(yaw, pitch, ins, coupled=True)
            worst = max(worst, self._rcm_error())
        self.assertLess(worst, 50e-6, f"worst RCM error {worst*1e6:.1f} um")

    def test_dropping_the_coupling_breaks_the_remote_centre(self):
        worst = 0.0
        for yaw, pitch, ins in self.POSES:
            self._set_pose(yaw, pitch, ins, coupled=False)
            worst = max(worst, self._rcm_error())
        self.assertGreater(worst, 10e-3, "uncoupled model should be badly wrong")


class TestPsmCoupling(PsmTestBase):
    def test_home_pose_is_coupling_consistent(self):
        self.assertLess(PSM_COUPLING.configuration_residual(self.kin), 1e-12)

    def test_equality_row_count(self):
        A, b = stack_eq([PSM_COUPLING], self.kin)
        self.assertEqual(A.shape, (7, self.kin.nv))   # 5 couplings + 2 locked
        np.testing.assert_allclose(b, 0.0)

    def test_locked_joints_are_pinned(self):
        ctl = QpController(self.kin, "tool_tip", [], equalities=[PSM_COUPLING])
        qdot, _ = ctl.solve(np.array([0.0, 0.01, 0.0]))
        for j in ("rev_joint", "tool_gripper2_joint"):
            self.assertLess(abs(float(qdot[self.kin.dof_index(j)])), 1e-9)

    def test_qp_respects_the_coupling_exactly(self):
        ctl = QpController(self.kin, "tool_tip", [], equalities=[PSM_COUPLING])
        qdot, _ = ctl.solve(np.array([0.0, 0.01, 0.0]))
        self.assertLess(PSM_COUPLING.residual(self.kin, qdot), 1e-9)


class TestPsmClosedLoop(PsmTestBase):
    """Drive the real PSM with the fixture stack and check nothing degrades."""

    DT = 0.002

    def _run(self, steps=4000, use_wall=True):
        rcm_pt = self.kin.site_position("remote_center")
        tip0 = self.kin.site_position("tool_tip")
        goal = tip0 + np.array([0.0, 0.026, 0.0])
        sdf = Sphere(tip0 + np.array([0.0, 0.012, 0.010]), 0.010)

        rcm = RcmConstraint(rcm_pt, proximal="shaft_00", distal="shaft_end",
                            tol=0.0005)
        limits = JointLimitConstraint()
        cons = [limits, rcm]
        wall = SdfConstraint(sdf, "tool_tip", margin=0.003)
        if use_wall:
            cons.insert(0, wall)
        ctl = QpController(self.kin, "tool_tip", cons,
                           equalities=[PSM_COUPLING], qd_limit=0.8)

        worst = dict(rcm=0.0, wall=-np.inf, limits=-np.inf, coupling=0.0)
        for _ in range(steps):
            p = self.kin.site_position("tool_tip")
            v = goal - p
            s = float(np.linalg.norm(v))
            if s > 0.01:
                v *= 0.01 / s
            qdot, _ = ctl.step(v, self.DT)
            worst["coupling"] = max(worst["coupling"],
                                    PSM_COUPLING.residual(self.kin, qdot))
            worst["rcm"] = max(worst["rcm"], rcm.deviation(self.kin))
            worst["wall"] = max(worst["wall"], wall.violation(self.kin))
            worst["limits"] = max(worst["limits"], limits.violation(self.kin))
            if float(np.linalg.norm(goal - self.kin.site_position("tool_tip"))) < 3e-4:
                break
        err = float(np.linalg.norm(goal - self.kin.site_position("tool_tip")))
        return ctl, worst, err

    def test_reaches_goal_through_the_port(self):
        _, _, err = self._run()
        self.assertLess(err, 1e-3)

    def test_mechanical_rcm_is_preserved_under_control(self):
        """The PSM's RCM is structural, so it should hold far tighter than the
        500 um the constraint merely permits."""
        _, worst, _ = self._run()
        self.assertLess(worst["rcm"], 50e-6)

    def test_no_violations(self):
        _, worst, _ = self._run()
        self.assertLessEqual(worst["wall"], 0.0)
        # Allow floating-point rounding: DAQP 0.10 lands ~1e-17 past the bound.
        self.assertLessEqual(worst["limits"], 1e-12)

    def test_coupling_never_drifts(self):
        _, worst, _ = self._run()
        self.assertLess(worst["coupling"], 1e-9)
        self.assertLess(PSM_COUPLING.configuration_residual(self.kin), 1e-9)

    def test_no_infeasible_solves(self):
        ctl, _, _ = self._run()
        self.assertEqual(ctl.n_infeasible, 0)


if __name__ == "__main__":
    unittest.main()
