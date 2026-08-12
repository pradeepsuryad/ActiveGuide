import unittest

import numpy as np

from activeguide.sdf import Sphere, Plane, Box, Union, VoxelSdf


def sample_to_grid(sdf, lower, upper, spacing):
    """Sample an analytic SDF onto a regular grid -> (grid, origin, spacing)."""
    lower = np.asarray(lower, float)
    sp = np.full(3, float(spacing))
    n = np.floor((np.asarray(upper, float) - lower) / sp).astype(int) + 1
    axes = [lower[i] + np.arange(n[i]) * sp[i] for i in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    flat = pts.reshape(-1, 3)
    grid = np.array([sdf.distance(p) for p in flat]).reshape(tuple(n))
    return grid, lower, sp


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


class TestVoxelSdf(unittest.TestCase):
    """Sampling an analytic SDF onto a grid gives a known right answer to
    compare against, so accuracy claims are checked rather than asserted."""

    @classmethod
    def setUpClass(cls):
        cls.ref = Sphere([0.0, 0.0, 0.0], 0.030)
        grid, origin, sp = sample_to_grid(cls.ref, [-0.06] * 3, [0.06] * 3, 0.002)
        cls.field = VoxelSdf(grid, origin, sp)
        cls.rng = np.random.default_rng(0)

    def in_box(self, n=400, shrink=0.999):
        lo, hi = self.field.lower * shrink, self.field.upper * shrink
        return self.rng.uniform(lo, hi, size=(n, 3))

    # --- accuracy ---------------------------------------------------------
    def test_interpolates_the_analytic_field(self):
        P = self.in_box()
        err = np.array([self.field.distance(p) - self.ref.distance(p) for p in P])
        # Trilinear error on a smooth field scales with the square of the cell.
        self.assertLess(np.abs(err).max(), 2e-4)

    def test_exact_on_grid_nodes(self):
        """No interpolation happens at a node, so it must reproduce exactly."""
        for idx in [(0, 0, 0), (7, 13, 21), (30, 30, 30)]:
            p = self.field.origin + np.array(idx) * self.field.spacing
            self.assertAlmostEqual(self.field.distance(p),
                                   self.field.grid[idx], places=12)

    def test_scalar_and_vectorised_paths_agree(self):
        P = self.in_box()
        scalar = np.array([self.field.distance(p) for p in P])
        np.testing.assert_allclose(scalar, self.field.distance_many(P), atol=1e-12)

    # --- gradient ---------------------------------------------------------
    def test_gradient_matches_finite_difference_of_itself(self):
        """The analytic gradient must differentiate the interpolant actually
        used, not the underlying field it approximates."""
        eps = 1e-7
        for p in self.in_box(60, shrink=0.9):
            fd = np.array([
                (self.field.distance(p + e) - self.field.distance(p - e)) / (2 * eps)
                for e in np.eye(3) * eps])
            np.testing.assert_allclose(self.field.gradient(p), fd, atol=1e-4)

    def test_normal_is_unit_and_outward(self):
        """Length is exact; direction carries a known, bounded error.

        The interpolant's gradient is a forward difference across the cell, so
        it reports the true normal from about half a cell away. The angular
        error is therefore O(spacing / radius) -- roughly 1.8 degrees at 45 mm
        on a 2 mm grid, which is what this pins. It is not a defect to be
        tuned away: a centred stencil would need a wider one and would still
        be discontinuous somewhere.
        """
        n = self.field.normal([0.045, 0.0, 0.0])
        self.assertAlmostEqual(np.linalg.norm(n), 1.0, places=9)
        self.assertGreater(np.dot(n, [1.0, 0.0, 0.0]), 0.999)     # < 2.6 deg

    # --- out-of-box behaviour --------------------------------------------
    def test_outside_the_box_never_overestimates(self):
        """Claiming clearance that is not there is the one unsafe failure, so
        the out-of-range bound is checked for sign, not just magnitude."""
        P = self.rng.uniform(-0.5, 0.5, size=(3000, 3))
        P = P[np.any((P < self.field.lower) | (P > self.field.upper), axis=1)]
        self.assertGreater(len(P), 100)
        err = np.array([self.field.distance(p) for p in P]) - \
            np.array([self.ref.distance(p) for p in P])
        self.assertLessEqual(err.max(), 1e-3)      # allows interpolation error only

    def test_outside_scalar_matches_vectorised(self):
        P = self.rng.uniform(-0.4, 0.4, size=(500, 3))
        P = P[np.any((P < self.field.lower) | (P > self.field.upper), axis=1)]
        scalar = np.array([self.field.distance(p) for p in P])
        np.testing.assert_allclose(scalar, self.field.distance_many(P), atol=1e-12)

    def test_outside_gradient_is_outward_and_lipschitz(self):
        """Outside the box the field is a conservative *bound*, not a distance,
        and its gradient reflects that: every valid lower bound on distance is
        1-Lipschitz, so the norm must not exceed 1. It sits furthest below 1
        off an edge or corner, where the clipped point cannot move in two or
        three axes and the bound is loosest."""
        for p in ([0.3, 0.0, 0.0], [-0.25, 0.1, 0.0], [0.2, 0.2, 0.2]):
            p = np.asarray(p, dtype=float)
            g = self.field.gradient(p)
            n = float(np.linalg.norm(g))
            self.assertLessEqual(n, 1.0 + 1e-6)
            self.assertGreater(n, 0.9)
            self.assertGreater(np.dot(g, p), 0.0)      # points away from the box

    def test_outside_gradient_is_tight_off_a_face(self):
        """Clipped in one axis only, the bound is tight and the gradient is
        unit to within the interpolation of the field it reads."""
        g = self.field.gradient(np.array([0.3, 0.0, 0.0]))
        self.assertAlmostEqual(float(np.linalg.norm(g)), 1.0, delta=0.01)
        self.assertGreater(g[0], 0.99)

    def test_distance_grows_moving_away(self):
        base = np.array([0.061, 0.0, 0.0])       # just outside the box
        ds = [self.field.distance(base + np.array([t, 0.0, 0.0]))
              for t in (0.0, 0.05, 0.2, 1.0)]
        self.assertTrue(all(b > a for a, b in zip(ds, ds[1:])), ds)

    def test_boundary_query_terminates(self):
        """The out-of-box path re-enters the interior path at the clipped
        point; rounding there must not bounce it straight back out."""
        for corner in (self.field.lower, self.field.upper):
            self.assertTrue(np.isfinite(self.field.distance(corner)))
            self.assertTrue(np.all(np.isfinite(self.field.gradient(corner))))

    # --- offset and placement --------------------------------------------
    def test_offset_dilates_exactly(self):
        dilated = VoxelSdf(self.field.grid, self.field.origin,
                           self.field.spacing, offset=0.005)
        P = self.in_box(200)
        np.testing.assert_allclose(dilated.distance_many(P),
                                   self.field.distance_many(P) - 0.005, atol=1e-12)

    def test_translated_shifts_without_resampling(self):
        t = np.array([0.1, -0.2, 0.05])
        moved = self.field.translated(t)
        P = self.in_box(200)
        np.testing.assert_allclose(moved.distance_many(P + t),
                                   self.field.distance_many(P), atol=1e-12)
        self.assertIs(moved.grid, self.field.grid)     # shared, not copied

    def test_contains_point(self):
        self.assertTrue(self.field.contains_point([0.0, 0.0, 0.0]))
        self.assertFalse(self.field.contains_point([0.5, 0.0, 0.0]))

    # --- construction -----------------------------------------------------
    def test_rejects_malformed_grids(self):
        g = np.zeros((4, 4, 4))
        with self.assertRaises(ValueError):
            VoxelSdf(np.zeros((4, 4)), [0, 0, 0], 0.01)
        with self.assertRaises(ValueError):
            VoxelSdf(np.zeros((1, 4, 4)), [0, 0, 0], 0.01)
        with self.assertRaises(ValueError):
            VoxelSdf(g, [0, 0, 0], 0.0)
        with self.assertRaises(ValueError):
            VoxelSdf(g, [0, 0, 0], -0.01)

    def test_anisotropic_spacing(self):
        """CT voxels are routinely fine in-plane and coarse through-plane."""
        ref = Sphere([0, 0, 0], 0.02)
        sp = np.array([0.002, 0.002, 0.005])
        n = np.array([41, 41, 17])
        axes = [-0.04 + np.arange(n[i]) * sp[i] for i in range(3)]
        pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        grid = np.array([ref.distance(p) for p in pts]).reshape(tuple(n))
        f = VoxelSdf(grid, [-0.04, -0.04, -0.04], sp)
        np.testing.assert_allclose(f.spacing, sp)
        self.assertAlmostEqual(f.distance([0.0, 0.0, 0.0]), -0.02, places=9)
        # The coarse z axis dominates the direction error, as it should:
        # O(spacing_z / radius) = 2.5 mm / 30 mm here.
        g = f.gradient([0.03, 0.0, 0.0])
        self.assertGreater(np.dot(g / np.linalg.norm(g), [1.0, 0.0, 0.0]), 0.99)


if __name__ == "__main__":
    unittest.main()
