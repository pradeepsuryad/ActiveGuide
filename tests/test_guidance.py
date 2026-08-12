import unittest

import numpy as np

from activeguide.sdf import Sphere
from activeguide.paths import Polyline
from activeguide.fixtures import (
    ForbiddenRegionFixture, GuidanceFixture, FixtureStack,
)

DT = 0.002


class TestGuidanceFixture(unittest.TestCase):
    def setUp(self):
        self.path = Polyline([[0, 0, 0], [1, 0, 0]])     # straight line along +x

    def test_along_path_motion_is_free(self):
        gf = GuidanceFixture(self.path, strength=1.0)
        out, off, t = gf.filter_velocity([0.5, 0, 0], [0.3, 0, 0], DT)
        np.testing.assert_allclose(out, [0.3, 0, 0], atol=1e-9)

    def test_attracts_back_to_path_when_locked(self):
        gf = GuidanceFixture(self.path, strength=1.0)
        out, off, t = gf.filter_velocity([0.5, 0.1, 0], [0, 0, 0], DT)
        self.assertLess(out[1], 0.0)                     # pulled toward path (-y)
        self.assertAlmostEqual(out[0], 0.0, places=9)    # no spurious along-path

    def test_zero_strength_passes_through(self):
        gf = GuidanceFixture(self.path, strength=0.0)
        v = np.array([0.2, 0.3, 0.0])
        out, off, t = gf.filter_velocity([0.5, 0.1, 0], v, DT)
        np.testing.assert_allclose(out, v, atol=1e-9)    # no guidance at all

    def test_offset_reported(self):
        gf = GuidanceFixture(self.path, strength=0.5)
        out, off, t = gf.filter_velocity([0.5, 0.1, 0], [0, 0, 0], DT)
        np.testing.assert_allclose(off, [0, 0.1, 0], atol=1e-9)


class TestFixtureStackSafetyWins(unittest.TestCase):
    def test_forbidden_overrides_guidance(self):
        # path runs straight through the forbidden sphere; at the wall, even
        # though guidance wants to continue inward, the clamp removes it.
        sdf = Sphere([0, 0, 0], 1.0)
        forbidden = ForbiddenRegionFixture(sdf, margin=1.1, restore_rate=0.0)
        path = Polyline([[2, 0, 0], [-2, 0, 0]])
        guidance = GuidanceFixture(path, strength=1.0)
        stack = FixtureStack(guidance=guidance, forbidden=forbidden)

        p = np.array([1.1, 0, 0])            # exactly at the margin
        v_des = np.array([-1.0, 0, 0])       # operator drives inward along path
        out, info = stack.filter_velocity(p, v_des, DT)
        n = sdf.normal(p)
        self.assertLess(abs(float(np.dot(out, n))), 1e-6)   # no inward motion
        self.assertIn("distance", info)
        self.assertIn("path_offset", info)


if __name__ == "__main__":
    unittest.main()
