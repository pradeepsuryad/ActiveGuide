import unittest

import numpy as np

from activeguide.paths import Polyline


class TestPolyline(unittest.TestCase):
    def setUp(self):
        self.line = Polyline([[0, 0, 0], [1, 0, 0]])

    def test_closest_on_segment(self):
        c, t = self.line.closest_point([0.5, 1.0, 0.0])
        np.testing.assert_allclose(c, [0.5, 0, 0], atol=1e-9)
        np.testing.assert_allclose(t, [1, 0, 0], atol=1e-9)
        self.assertAlmostEqual(self.line.distance([0.5, 1.0, 0.0]), 1.0)

    def test_clamps_to_endpoints(self):
        c, _ = self.line.closest_point([-2.0, 0.3, 0.0])
        np.testing.assert_allclose(c, [0, 0, 0], atol=1e-9)
        c, _ = self.line.closest_point([5.0, 0.0, 0.0])
        np.testing.assert_allclose(c, [1, 0, 0], atol=1e-9)

    def test_corner_picks_nearest_segment(self):
        L = Polyline([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        c, t = L.closest_point([1.2, 0.5, 0.0])      # nearest the vertical leg
        np.testing.assert_allclose(c, [1, 0.5, 0], atol=1e-9)
        np.testing.assert_allclose(t, [0, 1, 0], atol=1e-9)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            Polyline([[0, 0, 0]])


if __name__ == "__main__":
    unittest.main()
