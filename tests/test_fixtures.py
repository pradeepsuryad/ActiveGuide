import unittest

import numpy as np

from activeguide.sdf import Sphere
from activeguide.fixtures import ForbiddenRegionFixture


class TestForbiddenRegionFixture(unittest.TestCase):
    def setUp(self):
        # sphere radius 1 at origin; hold tool 0.1 outside the surface
        self.sdf = Sphere([0, 0, 0], 1.0)
        self.fx = ForbiddenRegionFixture(self.sdf, margin=0.1, influence=0.5,
                                         restore_rate=0.0)

    DT = 0.002

    def test_far_motion_unchanged(self):
        p = np.array([3.0, 0, 0])              # well outside influence
        v = np.array([-1.0, 0.2, 0])
        out, d, n = self.fx.filter_velocity(p, v, self.DT)
        np.testing.assert_allclose(out, v)

    def test_inward_normal_removed_at_wall(self):
        # at the margin, pushing straight in (-x): inward speed capped to ~0
        p = np.array([1.1, 0, 0])              # distance == margin
        v = np.array([-1.0, 0, 0])
        out, d, n = self.fx.filter_velocity(p, v, self.DT)
        self.assertLess(abs(float(np.dot(out, n))), 1e-6)  # no inward normal vel
        np.testing.assert_allclose(out, [0, 0, 0], atol=1e-6)

    def test_tangential_preserved_at_wall(self):
        p = np.array([1.1, 0, 0])
        v = np.array([-1.0, 0.5, 0])           # inward + tangential
        out, d, n = self.fx.filter_velocity(p, v, self.DT)
        np.testing.assert_allclose(out, [0, 0.5, 0], atol=1e-6)  # slides along

    def test_outward_motion_allowed(self):
        p = np.array([1.05, 0, 0])             # inside the margin
        v = np.array([1.0, 0, 0])              # moving out
        out, d, n = self.fx.filter_velocity(p, v, self.DT)
        np.testing.assert_allclose(out, v)     # never opposed when leaving

    def test_force_zero_outside_margin(self):
        f, mag = self.fx.feedback_force([2.0, 0, 0], [0, 0, 0])
        self.assertEqual(mag, 0.0)

    def test_force_pushes_outward_when_penetrated(self):
        p = np.array([1.05, 0, 0])             # 0.05 inside the margin
        f, mag = self.fx.feedback_force(p, [0, 0, 0])
        self.assertGreater(mag, 0.0)
        self.assertGreater(f[0], 0.0)          # pushes +x, away from center

    def test_warning_level_monotonic(self):
        # closer to the wall -> larger warning, clamped to [0,1]
        self.assertEqual(self.fx.warning_level([2.0, 0, 0]), 0.0)
        mid = self.fx.warning_level([1.3, 0, 0])
        self.assertTrue(0.0 < mid < 1.0)
        self.assertEqual(self.fx.warning_level([1.08, 0, 0]), 1.0)  # inside margin

    def test_integration_never_penetrates(self):
        # drive straight at the center for 4 s; tool must never cross the margin
        p = np.array([2.0, 1e-3, 0.0])         # tiny offset to break symmetry
        dt = 0.002
        min_d = np.inf
        for _ in range(2000):
            v_des = np.array([-3.0, 0, 0])     # aggressive inward push
            v, d, n = self.fx.filter_velocity(p, v_des, dt)
            p = p + v * dt
            min_d = min(min_d, self.sdf.distance(p))
        self.assertGreaterEqual(min_d, self.fx.margin - 1e-4)


if __name__ == "__main__":
    unittest.main()
