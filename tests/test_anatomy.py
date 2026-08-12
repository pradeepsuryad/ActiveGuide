"""Tests for the segmented-anatomy -> SDF pipeline.

The pipeline's job is to answer "how far is the instrument from the organ, and
which side of it is it on". Both halves are checked against geometry whose
answer is known independently of this code:

  * a **box mesh is exactly a box**, so `sdf.Box` is a closed-form oracle for
    both distance and sign -- no tolerance for tessellation, no circularity;
  * a voxelised **ball** has a closed-form distance too, which pins the label
    volume path;
  * the candidate-pruned search is checked against brute force over every
    triangle, so "exact" means measured rather than argued.
"""

import os
import tempfile
import unittest

import numpy as np
import trimesh

from activeguide.anatomy import (
    MeshSdf,
    bake_labels,
    bake_mesh,
    closest_point_on_triangles,
    load_mesh,
    load_sdf,
    make_phantom,
    save_sdf,
)
from activeguide.sdf import Box, Sphere, VoxelSdf

BOX_HALF = np.array([0.030, 0.020, 0.015])


def box_mesh():
    return trimesh.creation.box(extents=2 * BOX_HALF)


def brute_force_distance(P, triangles):
    """Unsigned distance using every triangle -- the reference for pruning."""
    out = np.empty(len(P))
    for s in range(0, len(P), 128):
        b = P[s:s + 128]
        T = np.broadcast_to(triangles, (len(b),) + triangles.shape)
        out[s:s + 128] = np.linalg.norm(
            closest_point_on_triangles(b, T) - b[:, None, :], axis=2).min(axis=1)
    return out


class TestClosestPointOnTriangles(unittest.TestCase):
    """One triangle, one query per barycentric region."""

    def setUp(self):
        self.tri = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])

    def check(self, p, expected):
        got = closest_point_on_triangles(np.array([p], dtype=float),
                                         self.tri[None, ...])[0, 0]
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_face_interior(self):
        self.check([0.25, 0.25, 1.0], [0.25, 0.25, 0.0])

    def test_vertex_regions(self):
        self.check([-1.0, -1.0, 0.0], [0.0, 0.0, 0.0])
        self.check([2.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        self.check([0.0, 3.0, 0.5], [0.0, 1.0, 0.0])

    def test_edge_regions(self):
        self.check([0.5, -1.0, 0.0], [0.5, 0.0, 0.0])      # edge AB
        self.check([-1.0, 0.5, 0.0], [0.0, 0.5, 0.0])      # edge AC
        self.check([1.0, 1.0, 0.0], [0.5, 0.5, 0.0])       # edge BC

    def test_degenerate_triangle_resolves(self):
        """A zero-area triangle satisfies several region tests at once; it must
        still produce one finite point rather than a blend of them."""
        tri = np.zeros((1, 1, 3, 3))
        got = closest_point_on_triangles(np.array([[1.0, 1.0, 1.0]]), tri)
        self.assertTrue(np.all(np.isfinite(got)))
        np.testing.assert_allclose(got[0, 0], [0, 0, 0], atol=1e-12)


class TestMeshSdfAgainstClosedForm(unittest.TestCase):
    """A box mesh IS a box, so `sdf.Box` grades the mesh code exactly."""

    @classmethod
    def setUpClass(cls):
        cls.mesh = box_mesh()
        cls.exact = MeshSdf(cls.mesh)
        cls.ref = Box([0, 0, 0], BOX_HALF)
        cls.P = np.random.default_rng(0).uniform(-0.06, 0.06, size=(500, 3))

    def test_signed_distance_is_exact(self):
        got = self.exact.distance_many(self.P)
        want = np.array([self.ref.distance(p) for p in self.P])
        np.testing.assert_allclose(got, want, atol=1e-12)

    def test_sign_convention_matches_the_rest_of_the_package(self):
        """Positive outside, negative inside -- inverting this would silently
        turn a safety constraint into an attractor."""
        self.assertGreater(self.exact.distance([0.5, 0.0, 0.0]), 0.0)
        self.assertLess(self.exact.distance([0.0, 0.0, 0.0]), 0.0)
        want_in = np.array([self.ref.distance(p) for p in self.P]) < 0
        np.testing.assert_array_equal(self.exact.inside(self.P), want_in)

    def test_winding_number_is_one_inside_zero_outside(self):
        self.assertAlmostEqual(abs(self.exact.winding_number(
            np.zeros((1, 3)))[0]), 1.0, places=6)
        self.assertAlmostEqual(self.exact.winding_number(
            np.array([[1.0, 1.0, 1.0]]))[0], 0.0, places=6)

    def test_candidate_pruning_equals_brute_force(self):
        """The k-d tree only proposes candidates; this pins that the proposal
        never loses the true closest triangle."""
        mesh = make_phantom(radius=0.03, seed=1)
        exact = MeshSdf(mesh)
        P = np.random.default_rng(2).uniform(-0.055, 0.055, size=(220, 3))
        np.testing.assert_allclose(exact.unsigned_many(P),
                                   brute_force_distance(P, exact.triangles),
                                   atol=1e-12)

    def test_certify_within_does_not_change_the_answer(self):
        P = np.random.default_rng(3).uniform(-0.06, 0.06, size=(200, 3))
        np.testing.assert_allclose(self.exact.unsigned_many(P),
                                   self.exact.unsigned_many(P, certify_within=0.004),
                                   atol=1e-12)

    def test_rejects_empty_mesh(self):
        empty = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), int))
        with self.assertRaises(ValueError):
            MeshSdf(empty)


class TestBakeMesh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh = box_mesh()
        cls.ref = Box([0, 0, 0], BOX_HALF)
        cls.field, cls.info = bake_mesh(cls.mesh, spacing=0.002, pad=0.02,
                                        verify=400, report=True)

    def test_produces_a_usable_sdf(self):
        self.assertIsInstance(self.field, VoxelSdf)
        self.assertEqual(self.field.grid.ndim, 3)

    def test_sign_algorithms_agree(self):
        """Flood fill and the winding number are unrelated methods; a genuine
        disagreement means one of them is wrong."""
        self.assertEqual(self.info["sign_mismatch"], 0)

    def test_reported_error_actually_bounds_the_error(self):
        """`bake_error` is what a caller folds into their margin, so it has to
        hold on probes the bake never saw -- including the edges and corners
        that area-weighted surface sampling barely reaches."""
        rng = np.random.default_rng(7)
        P = rng.uniform(self.field.lower, self.field.upper, size=(3000, 3))
        err = self.field.distance_many(P) - np.array([self.ref.distance(p) for p in P])
        self.assertLessEqual(err.max(), self.field.bake_error + 1e-9)

    def test_bake_error_is_an_overestimate_only(self):
        self.assertGreaterEqual(self.field.bake_error, 0.0)
        self.assertLess(self.field.bake_error, 0.002)

    def test_offset_is_baked_in(self):
        dilated = bake_mesh(self.mesh, spacing=0.003, pad=0.02, offset=0.004,
                            verify=0)
        plain = bake_mesh(self.mesh, spacing=0.003, pad=0.02, verify=0)
        P = np.random.default_rng(1).uniform(plain.lower, plain.upper, size=(200, 3))
        np.testing.assert_allclose(dilated.distance_many(P),
                                   plain.distance_many(P) - 0.004, atol=1e-12)

    def test_finer_spacing_reduces_error(self):
        coarse = bake_mesh(self.mesh, spacing=0.004, pad=0.02, verify=300)
        fine = bake_mesh(self.mesh, spacing=0.001, pad=0.02, verify=300)
        self.assertLess(fine.bake_error, coarse.bake_error)

    def test_box_is_padded_clear_of_the_anatomy(self):
        """The out-of-box bound and the flood fill both assume the boundary is
        outside the forbidden region."""
        f = self.field
        for corner in (f.lower, f.upper):
            self.assertGreater(f.distance(corner), 0.0)

    def test_refuses_a_grid_too_coarse_to_resolve_the_anatomy(self):
        """Silently returning an all-outside field would disable the fixture."""
        with self.assertRaises(ValueError):
            bake_mesh(self.mesh, spacing=0.05, pad=0.05, verify=0)


class TestBakeLabels(unittest.TestCase):
    """The CT-native path: a segmentation label volume, no meshing step."""

    @classmethod
    def setUpClass(cls):
        n, cls.sp = 48, 0.002
        cls.radius_vox = 15
        c = (n - 1) / 2
        gz, gy, gx = np.mgrid[0:n, 0:n, 0:n]
        cls.vol = ((gz - c) ** 2 + (gy - c) ** 2 + (gx - c) ** 2) < cls.radius_vox ** 2
        cls.field = bake_labels(cls.vol, spacing=cls.sp, origin=(-c * cls.sp,) * 3)

    def test_matches_the_analytic_ball_to_half_a_voxel(self):
        ref = Sphere([0, 0, 0], self.radius_vox * self.sp)
        P = np.random.default_rng(0).uniform(-0.035, 0.035, size=(600, 3))
        P = P[np.all((P > self.field.lower) & (P < self.field.upper), axis=1)]
        err = self.field.distance_many(P) - np.array([ref.distance(p) for p in P])
        self.assertLess(np.abs(err).max(), np.linalg.norm([self.sp] * 3))

    def test_sign_convention(self):
        self.assertLess(self.field.distance([0, 0, 0]), 0.0)
        self.assertGreater(self.field.distance([0.04, 0, 0]), 0.0)

    def test_anisotropic_spacing_is_honoured(self):
        f = bake_labels(self.vol, spacing=(0.002, 0.002, 0.006))
        np.testing.assert_allclose(f.spacing, [0.002, 0.002, 0.006])

    def test_rejects_degenerate_volumes(self):
        with self.assertRaises(ValueError):
            bake_labels(np.zeros((8, 8, 8), bool), spacing=0.002)
        with self.assertRaises(ValueError):
            bake_labels(np.ones((8, 8, 8), bool), spacing=0.002)
        with self.assertRaises(ValueError):
            bake_labels(np.ones((8, 8), bool), spacing=0.002)


class TestPhantom(unittest.TestCase):
    def test_watertight_and_closed(self):
        """Sign is only defined by a closed surface."""
        m = make_phantom()
        self.assertTrue(m.is_watertight)
        self.assertTrue(m.is_volume)
        self.assertGreater(m.volume, 0.0)

    def test_deterministic(self):
        np.testing.assert_allclose(make_phantom(seed=3).vertices,
                                   make_phantom(seed=3).vertices)
        self.assertFalse(np.allclose(make_phantom(seed=3).vertices,
                                     make_phantom(seed=4).vertices))

    def test_is_not_a_sphere(self):
        """The whole point of Phase 3 is anatomy a sphere cannot represent, so
        the phantom has to actually be non-spherical."""
        m = make_phantom(radius=0.030, seed=0)
        r = np.linalg.norm(m.vertices, axis=1)
        exact = MeshSdf(m)
        r_in = float(exact.unsigned_many(np.zeros((1, 3)))[0])
        self.assertGreater(r.max() - r_in, 0.008)

    def test_lumpiness_does_not_break_the_surface(self):
        for lump in (0.0, 0.3, 0.6):
            self.assertTrue(make_phantom(lumpiness=lump).is_watertight)


class TestRoundTrip(unittest.TestCase):
    def test_save_and_load_preserve_the_field(self):
        field = bake_mesh(make_phantom(radius=0.02, seed=5), spacing=0.004,
                          pad=0.015, offset=0.001, verify=120)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.npz")
            save_sdf(field, path)
            back = load_sdf(path)
        np.testing.assert_allclose(back.grid, field.grid)
        np.testing.assert_allclose(back.origin, field.origin)
        np.testing.assert_allclose(back.spacing, field.spacing)
        self.assertAlmostEqual(back.offset, field.offset)
        self.assertAlmostEqual(back.bake_error, field.bake_error)

    def test_load_can_override_the_offset(self):
        field = bake_mesh(make_phantom(radius=0.02, seed=5), spacing=0.005,
                          pad=0.015, offset=0.001, verify=0)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.npz")
            save_sdf(field, path)
            self.assertAlmostEqual(load_sdf(path, offset=0.007).offset, 0.007)

    def test_load_mesh_scales_to_metres(self):
        """Segmentation exports are usually millimetres; getting this wrong is
        silent and catastrophic."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.stl")
            trimesh.creation.box(extents=[20.0, 20.0, 20.0]).export(path)
            m = load_mesh(path, scale=0.001)
            np.testing.assert_allclose(m.extents, [0.02, 0.02, 0.02], atol=1e-9)


class TestControllerIntegration(unittest.TestCase):
    """Phase 3 claims the controller needed no changes. This checks that."""

    def test_baked_field_drops_into_an_sdf_constraint(self):
        import mujoco

        from activeguide.constraints import SdfConstraint, ShaftClearanceConstraint
        from activeguide.kinematics import RobotKinematics

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model = mujoco.MjModel.from_xml_path(
            os.path.join(root, "model", "scene_panda.xml"))
        kin = RobotKinematics(model)
        mujoco.mj_resetDataKeyframe(model, kin.data, 0)
        kin.forward()

        field = bake_mesh(make_phantom(radius=0.025, seed=0), spacing=0.004,
                          pad=0.02, verify=0)
        placed = field.translated(kin.site_position("tool_tip") + [0, 0.03, -0.03])

        for c in (SdfConstraint(placed, "tool_tip", margin=0.004),
                  ShaftClearanceConstraint(placed, ["shaft_50", "shaft_75"],
                                           margin=0.004)):
            G, h = c.rows(kin)
            self.assertEqual(G.shape[1], kin.nv)
            self.assertTrue(np.all(np.isfinite(G)))
            self.assertTrue(np.all(np.isfinite(h)))
            self.assertTrue(np.isfinite(c.violation(kin)))


if __name__ == "__main__":
    unittest.main()
