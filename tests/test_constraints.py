import os
import unittest

import numpy as np
import mujoco

from activeguide.constraints import (
    JointLimitConstraint,
    RcmConstraint,
    SdfConstraint,
    ShaftClearanceConstraint,
    stack,
)
from activeguide.kinematics import RobotKinematics
from activeguide.sdf import Sphere

SCENE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "model", "scene_panda.xml")


class ConstraintTestBase(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(SCENE)
        self.kin = RobotKinematics(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.kin.data, 0)
        self.kin.forward()
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "forbidden")
        self.sdf = Sphere(self.model.geom_pos[gid], float(self.model.geom_size[gid][0]))
        self.trocar = self.kin.site_position("trocar")


class TestSdfConstraint(ConstraintTestBase):
    def test_row_shape(self):
        c = SdfConstraint(self.sdf, "tool_tip", margin=0.004)
        G, h = c.rows(self.kin)
        self.assertEqual(G.shape, (1, 7))
        self.assertEqual(h.shape, (1,))

    def test_slack_is_positive_when_far(self):
        """Far from the region the bound is generous, so the row is inactive."""
        c = SdfConstraint(self.sdf, "tool_tip", margin=0.004, eta=10.0)
        _, h = c.rows(self.kin)
        self.assertGreater(h[0], 0.0)

    def test_vfi_row_equals_negative_distance_jacobian(self):
        """G must be -grad(d)^T J, i.e. the chain rule through the SDF."""
        c = SdfConstraint(self.sdf, "tool_tip", margin=0.004)
        G, _ = c.rows(self.kin)
        p = self.kin.site_position("tool_tip")
        n = self.sdf.normal(p)
        J, _ = self.kin.site_jacobian("tool_tip")
        np.testing.assert_allclose(G[0], -(n @ J), atol=1e-12)

    def test_moving_toward_region_is_the_constrained_direction(self):
        """A qdot that reduces distance must consume the constraint's slack."""
        c = SdfConstraint(self.sdf, "tool_tip", margin=0.004)
        G, h = c.rows(self.kin)
        p = self.kin.site_position("tool_tip")
        n = self.sdf.normal(p)
        J, _ = self.kin.site_jacobian("tool_tip")
        # Least-squares qdot that drives the tip inward (along -n).
        qdot = np.linalg.lstsq(J, -n * 0.01, rcond=None)[0]
        self.assertGreater(float(G[0] @ qdot), 0.0)   # positive => eats into h

    def test_violation_sign(self):
        c = SdfConstraint(self.sdf, "tool_tip", margin=0.004)
        self.assertLess(c.violation(self.kin), 0.0)   # safe at the home pose


class TestShaftClearanceConstraint(ConstraintTestBase):
    def test_one_row_per_sample_point(self):
        sites = ["shaft_25", "shaft_50", "shaft_75"]
        c = ShaftClearanceConstraint(self.sdf, sites, margin=0.004)
        G, h = c.rows(self.kin)
        self.assertEqual(G.shape, (3, 7))
        self.assertEqual(h.shape, (3,))

    def test_violation_is_worst_over_points(self):
        c = ShaftClearanceConstraint(self.sdf, ["shaft_50", "shaft_75"], margin=0.004)
        worst = max(
            -(self.sdf.distance(self.kin.site_position(s)) - 0.004)
            for s in ("shaft_50", "shaft_75")
        )
        self.assertAlmostEqual(c.violation(self.kin), worst, places=12)


class TestJointLimitConstraint(ConstraintTestBase):
    def test_two_rows_per_joint(self):
        G, h = JointLimitConstraint().rows(self.kin)
        self.assertEqual(G.shape, (14, 7))
        self.assertEqual(h.shape, (14,))

    def test_home_pose_has_room_both_ways(self):
        _, h = JointLimitConstraint().rows(self.kin)
        self.assertTrue(np.all(h > 0.0))

    def test_bound_shrinks_as_joint_approaches_stop(self):
        """The whole point of the VFI: available velocity goes to zero at the stop."""
        lo, hi = self.kin.joint_limits()
        q = self.kin.q.copy()
        q[0] = hi[0] - 1e-4                       # nearly at the upper stop
        self.kin.set_q(q)
        _, h = JointLimitConstraint(eta=10.0).rows(self.kin)
        self.assertLess(h[7], 1e-2)               # upper-limit row for joint 0

    def test_violation_negative_when_inside(self):
        self.assertLess(JointLimitConstraint().violation(self.kin), 0.0)


class TestRcmConstraint(ConstraintTestBase):
    def test_four_rows(self):
        G, h = RcmConstraint(self.trocar).rows(self.kin)
        self.assertEqual(G.shape, (4, 7))
        self.assertEqual(h.shape, (4,))

    def test_home_pose_is_on_axis(self):
        """The scene is built so the home shaft axis passes through the trocar."""
        self.assertLess(RcmConstraint(self.trocar).deviation(self.kin), 5e-5)

    def test_axis_tol_is_radial_tol_over_sqrt2(self):
        """Guards the sqrt(2) box-corner correction from silently regressing."""
        c = RcmConstraint(self.trocar, tol=0.001)
        self.assertAlmostEqual(c.axis_tol, 0.001 / np.sqrt(2.0), places=12)

    def test_deviation_grows_when_shaft_tilts_off_the_port(self):
        q = self.kin.q.copy()
        q[0] += 0.05                              # yaw the base: shaft leaves the port
        self.kin.set_q(q)
        self.assertGreater(RcmConstraint(self.trocar).deviation(self.kin), 1e-3)

    def test_coincident_shaft_sites_raise(self):
        c = RcmConstraint(self.trocar, proximal="tool_tip", distal="tool_tip")
        with self.assertRaises(ValueError):
            c.rows(self.kin)

    def test_perp_basis_is_orthonormal_and_perpendicular(self):
        for u in (np.array([0.0, 0.0, 1.0]),
                  np.array([1.0, 0.0, 0.0]),
                  np.array([0.3, -0.5, 0.81])):
            u = u / np.linalg.norm(u)
            b1, b2 = RcmConstraint._perp_basis(u)
            self.assertAlmostEqual(float(np.linalg.norm(b1)), 1.0, places=12)
            self.assertAlmostEqual(float(np.linalg.norm(b2)), 1.0, places=12)
            self.assertAlmostEqual(float(np.dot(b1, u)), 0.0, places=12)
            self.assertAlmostEqual(float(np.dot(b2, u)), 0.0, places=12)
            self.assertAlmostEqual(float(np.dot(b1, b2)), 0.0, places=12)


class TestStack(ConstraintTestBase):
    def test_concatenates_all_rows(self):
        cons = [
            SdfConstraint(self.sdf, "tool_tip", margin=0.004),
            ShaftClearanceConstraint(self.sdf, ["shaft_50", "shaft_75"], margin=0.004),
            JointLimitConstraint(),
            RcmConstraint(self.trocar),
        ]
        G, h = stack(cons, self.kin)
        self.assertEqual(G.shape, (1 + 2 + 14 + 4, 7))
        self.assertEqual(h.shape, (21,))

    def test_empty_stack_is_well_formed(self):
        G, h = stack([], self.kin)
        self.assertEqual(G.shape, (0, 7))
        self.assertEqual(h.shape, (0,))


if __name__ == "__main__":
    unittest.main()
