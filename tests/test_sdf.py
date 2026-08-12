import unittest

import numpy as np

from activeguide.sdf import Sphere, Plane, Box, Union


class TestSphere(unittest.TestCase):
    def setUp(self):
        self.s = Sphere([0, 0, 0], 1.0)

    def test_distance_sign(self):
        self.assertAlmostEqual(self.s.distance([2, 0, 0]), 1.0)   # outside
        self.assertAlmostEqual(self.s.distance([1, 0, 0]), 0.0)   # surface
        self.assertAlmostEqual(self.s.distance([0.25, 0, 0]), -0.75)  # inside

    def test_gradient_points_outward(self):
        n = self.s.normal([2, 0, 0])
        np.testing.assert_allclose(n, [1, 0, 0], atol=1e-9)

    def test_analytic_matches_numeric(self):
        p = np.array([0.3, -0.7, 0.5])
        analytic = self.s.gradient(p)
        numeric = super(Sphere, self.s).gradient(p)  # base finite-difference
        np.testing.assert_allclose(analytic, numeric, atol=1e-4)


class TestPlane(unittest.TestCase):
    def test_halfspace(self):
        # safe side is +z; forbidden below z=0
        p = Plane([0, 0, 0], [0, 0, 1])
        self.assertAlmostEqual(p.distance([0, 0, 0.5]), 0.5)
        self.assertAlmostEqual(p.distance([5, -3, -0.2]), -0.2)
        np.testing.assert_allclose(p.normal([1, 1, 1]), [0, 0, 1], atol=1e-9)


class TestBox(unittest.TestCase):
    def setUp(self):
        self.b = Box([0, 0, 0], [1, 1, 1])

    def test_outside_and_inside(self):
        self.assertAlmostEqual(self.b.distance([2, 0, 0]), 1.0)
        self.assertGreater(self.b.distance([2, 2, 0]), 1.0)   # corner is farther
        self.assertAlmostEqual(self.b.distance([0, 0, 0]), -1.0)  # center

    def test_numeric_gradient_outward(self):
        n = self.b.normal([2, 0, 0])
        np.testing.assert_allclose(n, [1, 0, 0], atol=1e-4)


class TestUnion(unittest.TestCase):
    def test_min_distance(self):
        u = Union([Sphere([-2, 0, 0], 1.0), Sphere([2, 0, 0], 1.0)])
        # origin is outside both; nearest surface is 1.0 away
        self.assertAlmostEqual(u.distance([0, 0, 0]), 1.0)
        # inside the right sphere -> negative
        self.assertLess(u.distance([2, 0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
