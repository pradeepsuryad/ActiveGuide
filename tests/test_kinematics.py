import os
import unittest

import numpy as np
import mujoco

from activeguide.kinematics import RobotKinematics

SCENE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "model", "scene_panda.xml")


class TestRobotKinematics(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_path(SCENE)
        self.kin = RobotKinematics(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.kin.data, 0)
        self.kin.forward()

    def test_dof_count(self):
        self.assertEqual(self.kin.nv, 7)          # 7-DOF arm, no gripper DOFs

    def test_unknown_site_raises(self):
        with self.assertRaises(KeyError):
            self.kin.site_position("no_such_site")

    def test_jacobian_matches_finite_difference(self):
        """J_pos must be the true derivative of site position w.r.t. q.

        This is the load-bearing check for the whole QP: every VFI row is built
        from this Jacobian, so if it is wrong the constraints are wrong in a way
        no amount of solver tuning would reveal.
        """
        J, _ = self.kin.site_jacobian("tool_tip")
        q0 = self.kin.q
        eps = 1e-6
        J_fd = np.zeros_like(J)
        for i in range(self.kin.nv):
            dq = np.zeros(self.kin.nv)
            dq[i] = eps
            self.kin.set_q(q0 + dq)
            p_plus = self.kin.site_position("tool_tip")
            self.kin.set_q(q0 - dq)
            p_minus = self.kin.site_position("tool_tip")
            J_fd[:, i] = (p_plus - p_minus) / (2 * eps)
        self.kin.set_q(q0)
        np.testing.assert_allclose(J, J_fd, atol=1e-6)

    def test_jacobian_shape_and_buffer_independence(self):
        Ja, _ = self.kin.site_jacobian("tool_tip")
        self.assertEqual(Ja.shape, (3, 7))
        # Fetching another site must not mutate a previously returned array.
        before = Ja.copy()
        self.kin.site_jacobian("shaft_00")
        np.testing.assert_array_equal(Ja, before)

    def test_joint_limits_are_finite_and_ordered(self):
        lo, hi = self.kin.joint_limits()
        self.assertEqual(lo.shape, (7,))
        self.assertTrue(np.all(np.isfinite(lo)))   # every Panda joint is limited
        self.assertTrue(np.all(np.isfinite(hi)))
        self.assertTrue(np.all(lo < hi))

    def test_home_pose_within_limits(self):
        lo, hi = self.kin.joint_limits()
        q = self.kin.q
        self.assertTrue(np.all(q >= lo) and np.all(q <= hi))

    def test_manipulability_positive_away_from_singularity(self):
        J, _ = self.kin.site_jacobian("tool_tip")
        self.assertGreater(self.kin.manipulability(J), 0.05)

    def test_set_q_moves_the_tip(self):
        p0 = self.kin.site_position("tool_tip")
        q = self.kin.q.copy()
        q[0] += 0.2
        self.kin.set_q(q)
        self.assertGreater(np.linalg.norm(self.kin.site_position("tool_tip") - p0), 1e-3)


if __name__ == "__main__":
    unittest.main()
