"""Segmented anatomy -> signed distance field.

Phases 0-2 protected a *sphere*, because a closed-form SDF was all the
controller needed to be interesting. Real protected anatomy arrives from a CT or
MR segmentation, in one of two forms:

    label volume  (voxels tagged "liver")  --> bake_labels
    surface mesh  (marching cubes, STL)    --> bake_mesh

Both land on the same object, ``sdf.VoxelSdf``, and therefore need *no* new
controller code: ``SdfConstraint`` and ``ShaftClearanceConstraint`` already take
an ``SDF``. That is the payoff of having kept the SDF interface as the single
representation since Phase 0.

Why bake at all
---------------
Exact mesh queries are not real-time. A control step evaluates the field at the
tool tip plus each shaft sample, and the finite-difference gradient multiplies
that again, so a 2 ms step needs the field roughly twenty times. Exact
closest-triangle search costs on the order of a millisecond per point, which is
one to two orders of magnitude over budget. Baking once moves that cost offline
and makes the online cost independent of mesh complexity -- a 200k-triangle
liver queries exactly as fast as a sphere.

The exact query is still built here, as ``MeshSdf``. It is not used by the
controller; it is the *oracle*. The demo drives the robot with the interpolated
grid and then audits the resulting trajectory against ``MeshSdf``, so the safety
claim is checked against true mesh geometry rather than against the same
approximation that produced it.

Two independent sign algorithms
-------------------------------
Deciding inside from outside is where mesh SDFs usually go wrong, so it is done
twice, by unrelated methods:

  * ``MeshSdf`` uses the **generalised winding number** (solid angle summed over
    triangles) -- a continuous, per-point test that degrades gracefully on
    meshes with small holes.
  * ``bake_mesh`` uses **grid flood fill** -- label the cells the surface cannot
    reach, then flood inward from the box boundary, which is outside by
    construction.

The bake then verifies a random sample of one against the other. A single sign
convention error is the kind of bug that silently inverts a safety constraint,
so it is worth catching two ways.

No native extension is required: ``trimesh``'s own proximity queries need
``rtree``, so closest-triangle search is done here on ``scipy.spatial.cKDTree``,
which is already a dependency.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .sdf import SDF, VoxelSdf

# Points processed per chunk. Bounds peak memory: the point-triangle kernel
# materialises (chunk x candidates x 3 x 3) floats.
_CHUNK = 4096


# --------------------------------------------------------------------------
# Exact geometry
# --------------------------------------------------------------------------
def closest_point_on_triangles(P: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Closest point to ``P`` on each triangle of ``T``.

    ``P`` is ``(N, 3)``; ``T`` is ``(N, K, 3, 3)`` (K triangles per point, each
    three vertices). Returns ``(N, K, 3)``.

    Ericson's barycentric region test (*Real-Time Collision Detection*, §5.1.5),
    vectorised. The seven cases -- three vertices, three edges, the face
    interior -- are evaluated for every pair and then selected between, which is
    branchless and so vectorises cleanly. Cases are applied in priority order
    through a "still unassigned" mask so that a degenerate (zero-area) triangle,
    which can satisfy several tests at once, still resolves to one answer
    instead of a blend of two.
    """
    A, B, C = T[..., 0, :], T[..., 1, :], T[..., 2, :]
    p = P[:, None, :]

    ab, ac = B - A, C - A
    ap = p - A
    d1 = np.einsum("nkj,nkj->nk", ab, ap)
    d2 = np.einsum("nkj,nkj->nk", ac, ap)
    bp = p - B
    d3 = np.einsum("nkj,nkj->nk", ab, bp)
    d4 = np.einsum("nkj,nkj->nk", ac, bp)
    cp = p - C
    d5 = np.einsum("nkj,nkj->nk", ab, cp)
    d6 = np.einsum("nkj,nkj->nk", ac, cp)

    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    def _safe(num, den):
        """num/den, with the ratio defined as 0 where den vanishes."""
        return np.divide(num, den, out=np.zeros_like(num), where=den != 0.0)

    # General case: interior of the face.
    denom = va + vb + vc
    v = _safe(vb, denom)
    w = _safe(vc, denom)
    out = A + ab * v[..., None] + ac * w[..., None]
    todo = np.ones(va.shape, dtype=bool)

    def _assign(mask, value):
        nonlocal out, todo
        m = mask & todo
        if m.any():
            out = np.where(m[..., None], value, out)
            todo &= ~m

    # Vertex regions.
    _assign((d1 <= 0) & (d2 <= 0), A)
    _assign((d3 >= 0) & (d4 <= d3), B)
    _assign((d6 >= 0) & (d5 <= d6), C)
    # Edge regions.
    _assign((vc <= 0) & (d1 >= 0) & (d3 <= 0),
            A + ab * _safe(d1, d1 - d3)[..., None])
    _assign((vb <= 0) & (d2 >= 0) & (d6 <= 0),
            A + ac * _safe(d2, d2 - d6)[..., None])
    _assign((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0),
            B + (C - B) * _safe(d4 - d3, (d4 - d3) + (d5 - d6))[..., None])
    return out


class MeshSdf(SDF):
    """Exact signed distance to a triangle mesh. The oracle, not the controller.

    Distance is the true closest-triangle distance; the sign comes from the
    generalised winding number. Correct but slow -- use ``bake_mesh`` for
    anything in a control loop, and use this to check it.

    Search strategy
    ---------------
    A k-d tree over triangle *centroids* proposes candidates, then exact
    point-triangle distance is computed for those. Centroid proximity does not
    by itself imply triangle proximity, so the result is **verified rather than
    assumed**: with ``R`` the largest centroid-to-vertex radius in the mesh, no
    triangle whose centroid lies farther than ``d + R`` can beat a candidate
    already at distance ``d``. Where the k-th candidate fails that test the
    point is re-queried by radius, so the answer is exact and not merely likely.
    ``n_refined`` counts how often that second pass was needed.

    That certificate is cheap near the surface and structurally unobtainable
    far from it: at distance ``d`` from a mesh of extent much smaller than
    ``d``, *every* centroid sits at roughly ``d``, so no finite ``k`` short of
    the whole mesh can satisfy ``d_k >= d + R``. Hence ``certify_within``, which
    restricts the second pass to the band where a boundary constraint can
    actually be active. Beyond it the k-candidate estimate stands; measured
    against brute force over all triangles it was exact at every one of 3000
    probes even at ``k = 8``, so the band is a bound on proof, not on accuracy.
    """

    def __init__(self, mesh, k: int = 24):
        self.mesh = mesh
        self.triangles = np.ascontiguousarray(
            mesh.triangles, dtype=float)                 # (F, 3, 3)
        if len(self.triangles) == 0:
            raise ValueError("mesh has no faces")
        self.centroids = self.triangles.mean(axis=1)
        self.radius = float(np.linalg.norm(
            self.triangles - self.centroids[:, None, :], axis=2).max())
        self.tree = cKDTree(self.centroids)
        self.k = int(min(k, len(self.triangles)))
        self.n_refined = 0

    # --- unsigned distance ------------------------------------------------
    def unsigned_many(self, P: np.ndarray, certify_within: float | None = None) -> np.ndarray:
        """Exact distance to the surface for an ``(N, 3)`` array of points.

        ``certify_within`` limits the certification pass to points estimated to
        lie within that distance of the surface; ``None`` certifies everywhere.
        """
        P = np.atleast_2d(np.asarray(P, dtype=float))
        out = np.empty(len(P))
        for s in range(0, len(P), _CHUNK):
            block = P[s:s + _CHUNK]
            dk, idx = self.tree.query(block, k=self.k)
            # query() drops the candidate axis when k == 1; keep it 2-D.
            dk = dk.reshape(len(block), -1)
            idx = idx.reshape(len(block), -1)
            d = np.linalg.norm(
                closest_point_on_triangles(block, self.triangles[idx])
                - block[:, None, :], axis=2).min(axis=1)

            # Candidates are provably sufficient where the farthest one searched
            # already exceeds d + R; elsewhere widen to a radius query.
            need = dk[:, -1] < d + self.radius
            if certify_within is not None:
                need &= d <= certify_within
            for i in np.nonzero(need)[0]:
                cand = self.tree.query_ball_point(block[i], d[i] + self.radius)
                if cand:
                    tri = self.triangles[np.asarray(cand)][None, ...]
                    d[i] = np.linalg.norm(
                        closest_point_on_triangles(block[i][None, :], tri)
                        - block[i][None, None, :], axis=2).min()
            self.n_refined += int(need.sum())
            out[s:s + _CHUNK] = d
        return out

    # --- sign -------------------------------------------------------------
    def winding_number(self, P: np.ndarray) -> np.ndarray:
        """Generalised winding number: ~+-1 inside the surface, ~0 outside.

        Van Oosterom & Strackee's signed solid angle per triangle, summed and
        scaled by 4*pi. Unlike ray casting there is no special case for rays
        that grze an edge, and unlike a parity test it stays meaningful on a
        mesh with small holes.
        """
        P = np.atleast_2d(np.asarray(P, dtype=float))
        w = np.empty(len(P))
        tri = self.triangles
        for s in range(0, len(P), _CHUNK):
            p = P[s:s + _CHUNK][:, None, :]
            a = tri[None, :, 0, :] - p
            b = tri[None, :, 1, :] - p
            c = tri[None, :, 2, :] - p
            na = np.linalg.norm(a, axis=2)
            nb = np.linalg.norm(b, axis=2)
            nc = np.linalg.norm(c, axis=2)
            num = np.einsum("nkj,nkj->nk", a, np.cross(b, c))
            den = (na * nb * nc
                   + np.einsum("nkj,nkj->nk", a, b) * nc
                   + np.einsum("nkj,nkj->nk", b, c) * na
                   + np.einsum("nkj,nkj->nk", c, a) * nb)
            w[s:s + _CHUNK] = 2.0 * np.arctan2(num, den).sum(axis=1)
        return w / (4.0 * np.pi)

    def inside(self, P) -> np.ndarray:
        return np.abs(self.winding_number(P)) > 0.5

    # --- SDF interface ----------------------------------------------------
    def distance_many(self, P) -> np.ndarray:
        P = np.atleast_2d(np.asarray(P, dtype=float))
        d = self.unsigned_many(P)
        return np.where(self.inside(P), -d, d)

    def distance(self, p) -> float:
        return float(self.distance_many(np.asarray(p, dtype=float).reshape(1, 3))[0])


# --------------------------------------------------------------------------
# Baking
# --------------------------------------------------------------------------
def _grid_axes(lower, upper, spacing):
    """Cell-aligned axes covering ``[lower, upper]``, and the realised spacing."""
    n = np.maximum(np.ceil((upper - lower) / spacing).astype(int) + 1, 2)
    axes = [lower[i] + np.arange(n[i]) * spacing[i] for i in range(3)]
    return axes, n


def _sign_by_flood_fill(unsigned: np.ndarray, diagonal: float) -> np.ndarray:
    """Classify grid cells inside/outside from the unsigned distance field.

    Cells farther than one voxel diagonal from the surface cannot contain it, so
    the surface cannot separate two such cells that are face-connected. Label
    those "free" cells: any component touching the box boundary is outside (the
    bake pads the box so the boundary is guaranteed clear of the anatomy), and
    any component that does not is an interior cavity. The remaining thin shell
    of near-surface cells inherits the sign of its nearest free cell.

    Returns +1.0 outside, -1.0 inside.
    """
    free = unsigned > diagonal
    if not free.any():
        raise ValueError(
            "no cell lies more than one voxel diagonal from the surface: "
            "the grid is far too coarse for this mesh")

    labels, n = ndimage.label(free, structure=ndimage.generate_binary_structure(3, 1))
    edge = np.concatenate([
        labels[0].ravel(), labels[-1].ravel(),
        labels[:, 0].ravel(), labels[:, -1].ravel(),
        labels[:, :, 0].ravel(), labels[:, :, -1].ravel(),
    ])
    outer = np.unique(edge[edge > 0])
    if outer.size == 0:
        raise ValueError("no free cell touches the box boundary; increase pad")
    if n == outer.size:
        raise ValueError(
            "no interior region found: every free cell connects to the outside. "
            "The anatomy is thinner than one voxel diagonal -- reduce spacing.")

    sign = np.where(np.isin(labels, outer), 1.0, -1.0)
    # Near-surface cells were excluded from the fill; give each the sign of the
    # nearest cell that did take part.
    _, idx = ndimage.distance_transform_edt(~free, return_indices=True)
    return np.where(free, sign, sign[idx[0], idx[1], idx[2]])


def bake_mesh(mesh, spacing: float = 0.002, pad: float = 0.025,
              offset: float = 0.0, verify: int = 512, seed: int = 0,
              certify_band: float = 0.010, report: bool = False):
    """Bake a triangle mesh into a ``VoxelSdf``.

    Parameters
    ----------
    spacing  grid cell size (m). Halving it costs 8x memory and bake time and
             roughly quarters the interpolation error, which falls with the
             square of cell size for a smooth field.
    pad      empty margin added around the mesh bounds (m). It must be positive
             so the box boundary is outside the anatomy, which is what makes
             both the flood fill and ``VoxelSdf``'s out-of-box bound valid.
    offset   safety dilation baked into the returned field (m); see ``VoxelSdf``.
    verify   number of random near-surface probes used to (a) cross-check the
             flood-fill sign against the winding number and (b) measure the
             interpolation error. Set 0 to skip.
    certify_band  distance within which closest-triangle search is *proved*
             exhaustive rather than merely exact; see ``MeshSdf``.

    Returns ``VoxelSdf``, or ``(VoxelSdf, dict)`` when ``report``.
    """
    t0 = time.perf_counter()
    exact = MeshSdf(mesh)

    sp = np.full(3, float(spacing))
    lower = np.asarray(mesh.bounds[0], dtype=float) - pad
    upper = np.asarray(mesh.bounds[1], dtype=float) + pad
    axes, n = _grid_axes(lower, upper, sp)

    pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    unsigned = exact.unsigned_many(pts, certify_within=certify_band).reshape(n)
    sign = _sign_by_flood_fill(unsigned, float(np.linalg.norm(sp)))

    field = VoxelSdf(sign * unsigned, origin=lower, spacing=sp, offset=offset)
    bake_s = time.perf_counter() - t0

    info = dict(shape=tuple(int(v) for v in n), cells=int(np.prod(n)),
                spacing=float(spacing), pad=float(pad), offset=float(offset),
                faces=int(len(mesh.faces)), bake_s=bake_s,
                certified=int(exact.n_refined))

    if verify:
        rng = np.random.default_rng(seed)
        # Two probe populations, because they fail differently. Jittered
        # surface samples concentrate where the constraint actually engages.
        # Uniform samples are needed as well: surface sampling is area-weighted
        # and so barely lands on edges and corners, yet those are exactly where
        # trilinear interpolation is worst -- it smooths a crease in the true
        # field, and smoothing a convex crease over-estimates distance. Probing
        # only the surface under-reports the error on a faceted mesh by around
        # a factor of two.
        n = int(verify)
        surf = np.asarray(mesh.sample(n), dtype=float)
        probe = np.vstack([
            surf + rng.normal(scale=2.0 * spacing, size=surf.shape),
            rng.uniform(field.lower, field.upper, size=(n, 3)),
        ])
        probe = np.clip(probe, field.lower, field.upper)

        d_exact = exact.distance_many(probe)
        d_grid = field.distance_many(probe) + offset      # compare undilated

        err = d_grid - d_exact
        # Only over-estimates are unsafe: they promise clearance that is absent.
        field.bake_error = float(max(err.max(), 0.0))
        # A disagreement about *sign* only means something where the point is
        # farther from the surface than the interpolation error: closer in, the
        # two fields bracket a zero level that sits within their common
        # uncertainty, and either answer is as good as the other. Counting
        # those would report the grid resolution, not a fault in the sign.
        genuine = np.abs(d_exact) > np.abs(err).max()
        info.update(
            max_abs_error=float(np.abs(err).max()),
            max_overestimate=field.bake_error,
            rms_error=float(np.sqrt((err ** 2).mean())),
            sign_mismatch=int(np.count_nonzero(
                ((d_grid < 0) != (d_exact < 0)) & genuine)),
            sign_mismatch_in_noise=int(np.count_nonzero(
                ((d_grid < 0) != (d_exact < 0)) & ~genuine)),
        )

    return (field, info) if report else field


def bake_labels(volume, spacing, origin=(0.0, 0.0, 0.0), offset: float = 0.0):
    """Bake a binary segmentation label volume into a ``VoxelSdf``.

    This is the shortest path from a CT/MR segmentation to an active constraint:
    no meshing step at all. ``volume[i, j, k]`` is True where the voxel belongs
    to the protected structure.

    The field is ``EDT(outside) - EDT(inside)`` in metres, the standard
    two-sided Euclidean distance transform. Its zero level sits on the voxel
    boundary layer rather than on a sub-voxel surface, so it is accurate to
    about half a voxel -- fine when the segmentation itself is voxel-resolution,
    which is the case being modelled. Go through ``bake_mesh`` instead when a
    marching-cubes surface has already been extracted, since that recovers
    sub-voxel detail.

    ``spacing`` may be scalar or per-axis, matching CT's usual anisotropic
    voxels (fine in-plane, coarse through-plane).
    """
    vol = np.ascontiguousarray(volume).astype(bool)
    if vol.ndim != 3:
        raise ValueError(f"volume must be 3-D, got shape {vol.shape}")
    if not vol.any():
        raise ValueError("volume is empty: nothing is labelled")
    if vol.all():
        raise ValueError("volume is entirely labelled: no outside region")

    sp = np.asarray(spacing, dtype=float)
    sp = np.full(3, float(sp)) if sp.ndim == 0 else sp.reshape(3)

    d_out = ndimage.distance_transform_edt(~vol, sampling=sp)
    d_in = ndimage.distance_transform_edt(vol, sampling=sp)
    return VoxelSdf(d_out - d_in, origin=origin, spacing=sp, offset=offset,
                    bake_error=float(np.linalg.norm(sp)) / 2.0)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def load_mesh(path: str, scale: float = 1.0, repair: bool = True):
    """Load a surface mesh and make it fit to bake.

    ``scale`` converts to metres -- surgical meshes exported from segmentation
    tools are almost always in millimetres, so ``scale=0.001`` is the common
    case and getting it wrong is silent and catastrophic.

    Watertightness matters because the sign of the field is only defined by a
    closed surface. Segmentation output frequently is not closed, so small holes
    are filled and winding made consistent when ``repair``. A mesh that is still
    open afterwards is returned with a flag rather than rejected: the winding
    number degrades gracefully, but the caller deserves to know.
    """
    import trimesh

    mesh = trimesh.load(path, force="mesh", process=True)
    if scale != 1.0:
        mesh.apply_scale(float(scale))
    if repair and not mesh.is_watertight:
        mesh.remove_unreferenced_vertices()
        mesh.fill_holes()
        mesh.fix_normals()
    return mesh


def make_phantom(radius: float = 0.030, lumpiness: float = 0.22,
                 aspect=(1.0, 0.72, 0.58), subdivisions: int = 3,
                 center=(0.0, 0.0, 0.0), seed: int = 0):
    """A deterministic lobed organ phantom, watertight by construction.

    The repository ships no patient data, and the upstream assets it does use
    are fetched rather than redistributed (see ``model/psm/NOTICE.md``). A
    procedural phantom keeps the demo and the tests runnable offline with no
    licence question, while still exercising everything a real segmentation
    would: a non-convex surface with concave saddles between lobes, where the
    naive "sign from the nearest face normal" shortcut fails and a sphere would
    not have caught it.

    Vertices are displaced radially by a sum of smooth directional bumps, so the
    surface stays a star domain and cannot self-intersect for ``lumpiness < 1``;
    the icosphere's topology, and hence watertightness, is untouched. Pass a
    real mesh to ``bake_mesh`` instead whenever one is available.
    """
    import trimesh

    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)
    v = np.asarray(sphere.vertices, dtype=float)

    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(5, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    amps = rng.uniform(-1.0, 1.0, size=5)

    bumps = (v @ dirs.T) ** 2 @ amps               # smooth, low frequency
    r = 1.0 + lumpiness * bumps / np.abs(amps).sum()

    sphere.vertices = (v * r[:, None]) * (radius * np.asarray(aspect, dtype=float))
    sphere.vertices += np.asarray(center, dtype=float)
    return sphere


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def save_sdf(field: VoxelSdf, path: str) -> None:
    """Cache a baked field so a demo need not re-bake on every run."""
    np.savez_compressed(
        path, grid=field.grid, origin=field.origin, spacing=field.spacing,
        offset=field.offset, bake_error=field.bake_error)


def load_sdf(path: str, offset: float | None = None) -> VoxelSdf:
    """Load a cached field, optionally re-setting the safety dilation."""
    z = np.load(path)
    return VoxelSdf(z["grid"], origin=z["origin"], spacing=z["spacing"],
                    offset=float(z["offset"]) if offset is None else offset,
                    bake_error=float(z["bake_error"]))
