"""Signed-distance-field primitives.

Convention used throughout ActiveGuide:
    distance(p) > 0   -> p is OUTSIDE the forbidden region (safe)
    distance(p) == 0  -> p is exactly on the forbidden surface
    distance(p) < 0   -> p is INSIDE the forbidden region (penetrated)

    gradient(p)       -> points in the direction of INCREASING distance,
                         i.e. the outward / "escape" direction.

This single representation feeds both the controller (penetration + push-out
normal) and, later, the motion planner (clearance). Keep it the source of truth.
"""

from __future__ import annotations

import numpy as np


class SDF:
    """Base class. Subclasses must implement ``distance``.

    A finite-difference ``gradient`` is provided so any new primitive (or a
    mesh-based SDF later) works out of the box; override it where an analytic
    form is cheaper or more accurate.
    """

    eps: float = 1e-6

    def distance(self, p: np.ndarray) -> float:
        raise NotImplementedError

    def gradient(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        g = np.zeros(3)
        for i in range(3):
            dp = np.zeros(3)
            dp[i] = self.eps
            g[i] = (self.distance(p + dp) - self.distance(p - dp)) / (2.0 * self.eps)
        return g

    def normal(self, p: np.ndarray) -> np.ndarray:
        """Unit outward normal at ``p`` (falls back to +x if undefined)."""
        g = self.gradient(p)
        n = np.linalg.norm(g)
        if n < 1e-9:
            return np.array([1.0, 0.0, 0.0])
        return g / n


class Sphere(SDF):
    """Forbidden region = the inside of a sphere (e.g. a protected organ)."""

    def __init__(self, center, radius: float):
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)

    def distance(self, p) -> float:
        return float(np.linalg.norm(np.asarray(p, dtype=float) - self.center) - self.radius)

    def gradient(self, p) -> np.ndarray:
        d = np.asarray(p, dtype=float) - self.center
        n = np.linalg.norm(d)
        if n < 1e-9:
            return np.array([1.0, 0.0, 0.0])
        return d / n


class Plane(SDF):
    """Forbidden region = the half-space behind a plane.

    ``normal`` points INTO the safe half-space (away from the forbidden side).
    """

    def __init__(self, point, normal):
        self.point = np.asarray(point, dtype=float)
        n = np.asarray(normal, dtype=float)
        self.n = n / np.linalg.norm(n)

    def distance(self, p) -> float:
        return float(np.dot(np.asarray(p, dtype=float) - self.point, self.n))

    def gradient(self, p) -> np.ndarray:
        return self.n.copy()


class Box(SDF):
    """Forbidden region = the inside of an axis-aligned box (uses numeric grad)."""

    def __init__(self, center, half_extents):
        self.center = np.asarray(center, dtype=float)
        self.half = np.asarray(half_extents, dtype=float)

    def distance(self, p) -> float:
        q = np.abs(np.asarray(p, dtype=float) - self.center) - self.half
        outside = np.linalg.norm(np.maximum(q, 0.0))
        inside = min(float(np.max(q)), 0.0)
        return outside + inside


class Union(SDF):
    """Union of forbidden regions: a point is unsafe if inside ANY child.

    ``distance`` is the min over children (positive only when outside all).
    """

    def __init__(self, sdfs):
        self.sdfs = list(sdfs)

    def distance(self, p) -> float:
        return min(s.distance(p) for s in self.sdfs)


class VoxelSdf(SDF):
    """Signed distance sampled on a regular grid, read by trilinear interpolation.

    This is the primitive real anatomy arrives as. A segmented organ is a mesh or
    a label volume, not a closed-form surface, and querying a mesh exactly costs
    milliseconds per point -- far past a 2 ms control step (see ``anatomy.py``).
    Baking the distance field once and interpolating it turns an arbitrarily
    complicated surface into a constant-time lookup, so the *controller* cost no
    longer depends on how complex the anatomy is. Nothing downstream changes:
    this is an ``SDF`` like any other, so ``SdfConstraint`` and
    ``ForbiddenRegionFixture`` accept it unmodified.

    Parameters
    ----------
    grid      (nx, ny, nz) signed distances in metres, following this module's
              convention (positive outside the forbidden region).
    origin    world coordinates of ``grid[0, 0, 0]``.
    spacing   cell size, scalar or per-axis.
    offset    safety dilation in metres, subtracted from every query. This grows
              the forbidden region to cover *segmentation* uncertainty, and is
              deliberately distinct from a fixture's ``margin``, which covers
              *tool* radius. They add.
    bake_error  reported upper bound on how much the interpolated field may
              exceed the true distance (see ``anatomy.bake_mesh``). It is
              metadata for the caller's margin budget, not applied here.

    Accuracy and the sign of the error
    ----------------------------------
    Trilinear interpolation of a distance field is not itself a distance field.
    Where the true field is smooth the error goes as the square of the cell
    size; where it creases -- at a convex edge of the surface, where the medial
    axis reaches it -- interpolation rounds the crease off and *over-estimates*
    distance, which is the unsafe direction because it promises clearance that
    is not there. ``bake_error`` is the measured bound on that over-estimate,
    and a caller's ``margin`` must cover it for the constraint to mean what it
    says. Every guarantee below is stated on the underlying field and inherits
    this same term.

    Queries outside the grid
    ------------------------
    The controller must not be handed nonsense if the tool leaves the baked
    volume, so out-of-range queries return a *lower bound* on the true distance
    rather than an extrapolation. Let ``p' = clip(p)`` be the nearest point of
    the grid box. Provided the forbidden region lies entirely inside the box
    (the bake pads to guarantee this), every surface point ``s`` is in the box,
    so in each clipped axis ``p`` and ``s`` sit on opposite sides of ``p'`` and
    the displacements add with the same sign:

        ||p - s||^2 >= ||p - p'||^2 + ||p' - s||^2

    Minimising over ``s`` gives ``d(p) >= hypot(d(p'), ||p - p'||)``, which is
    what is returned: never optimistic about the true field, and still growing
    with distance so the gradient keeps pointing sensibly away. (Were ``p'``
    somehow inside the region, that argument lapses and the plain 1-Lipschitz
    bound ``d(p') - ||p - p'||`` is used instead.) ``d(p')`` is itself read by
    interpolation, so in practice the bound holds to within ``bake_error`` like
    everything else -- it is not a separate escape from discretisation.
    """

    def __init__(self, grid, origin, spacing, offset: float = 0.0,
                 bake_error: float = 0.0):
        self.grid = np.ascontiguousarray(grid, dtype=float)
        if self.grid.ndim != 3:
            raise ValueError(f"grid must be 3-D, got shape {self.grid.shape}")
        if min(self.grid.shape) < 2:
            raise ValueError(
                f"grid needs >= 2 samples per axis to interpolate, got {self.grid.shape}")

        self.origin = np.asarray(origin, dtype=float).reshape(3)
        sp = np.asarray(spacing, dtype=float)
        self.spacing = np.full(3, float(sp)) if sp.ndim == 0 else sp.reshape(3).copy()
        if np.any(self.spacing <= 0.0):
            raise ValueError(f"spacing must be positive, got {self.spacing}")

        self.offset = float(offset)
        self.bake_error = float(bake_error)
        self._shape = np.array(self.grid.shape, dtype=int)

        # Scalar-path constants as Python floats/ints. The control loop calls
        # distance() and gradient() point-by-point, where NumPy's per-call
        # overhead dominates the eight multiply-adds that do the actual work:
        # going through the vectorised path costs 63 us a query, and this one
        # costs 3 us for bit-identical output.
        self._o = tuple(float(v) for v in self.origin)
        self._hi = tuple(float(v) for v in self.upper)
        self._inv = tuple(1.0 / float(v) for v in self.spacing)
        self._cap = tuple(int(v) - 2 for v in self.grid.shape)

    # --- geometry ---------------------------------------------------------
    @property
    def lower(self) -> np.ndarray:
        """World coordinates of the low corner of the baked box."""
        return self.origin.copy()

    @property
    def upper(self) -> np.ndarray:
        """World coordinates of the high corner of the baked box."""
        return self.origin + (self._shape - 1) * self.spacing

    def contains_point(self, p) -> bool:
        """Is ``p`` inside the baked box (i.e. answered by interpolation)?"""
        p = np.asarray(p, dtype=float)
        return bool(np.all(p >= self.lower) and np.all(p <= self.upper))

    def translated(self, t, offset: float | None = None) -> "VoxelSdf":
        """The same field moved by ``t``, sharing the underlying grid.

        Baking is expensive and depends only on the anatomy, so a field is baked
        once in its own frame and *placed* into a scene here. Moving the origin
        is exact -- no resampling, no interpolation error, and the grid array is
        shared rather than copied.
        """
        out = VoxelSdf(self.grid, self.origin + np.asarray(t, dtype=float),
                       self.spacing,
                       offset=self.offset if offset is None else offset,
                       bake_error=self.bake_error)
        return out

    # --- interpolation ----------------------------------------------------
    def _corners(self, U):
        """Base indices, fractional offsets, and the 8 corner values for ``U``."""
        i0 = np.floor(U).astype(int)
        np.clip(i0, 0, self._shape - 2, out=i0)
        f = U - i0
        x, y, z = i0[:, 0], i0[:, 1], i0[:, 2]
        g = self.grid
        c = np.empty((U.shape[0], 2, 2, 2))
        c[:, 0, 0, 0] = g[x, y, z]
        c[:, 1, 0, 0] = g[x + 1, y, z]
        c[:, 0, 1, 0] = g[x, y + 1, z]
        c[:, 1, 1, 0] = g[x + 1, y + 1, z]
        c[:, 0, 0, 1] = g[x, y, z + 1]
        c[:, 1, 0, 1] = g[x + 1, y, z + 1]
        c[:, 0, 1, 1] = g[x, y + 1, z + 1]
        c[:, 1, 1, 1] = g[x + 1, y + 1, z + 1]
        return f, c

    @staticmethod
    def _blend(f, c):
        """Trilinear value from fractional offsets ``f`` and corners ``c``."""
        fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
        c00 = c[:, 0, 0, 0] * (1 - fx) + c[:, 1, 0, 0] * fx
        c10 = c[:, 0, 1, 0] * (1 - fx) + c[:, 1, 1, 0] * fx
        c01 = c[:, 0, 0, 1] * (1 - fx) + c[:, 1, 0, 1] * fx
        c11 = c[:, 0, 1, 1] * (1 - fx) + c[:, 1, 1, 1] * fx
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        return c0 * (1 - fz) + c1 * fz

    def distance_many(self, P) -> np.ndarray:
        """Vectorised ``distance`` for an ``(N, 3)`` array of points."""
        P = np.atleast_2d(np.asarray(P, dtype=float))
        Pc = np.clip(P, self.lower, self.upper)
        delta = np.linalg.norm(P - Pc, axis=1)

        U = (Pc - self.origin) / self.spacing
        f, c = self._corners(U)
        d = self._blend(f, c)

        out = delta > 0.0                       # queried outside the baked box
        if np.any(out):
            dc, dl = d[out], delta[out]
            d[out] = np.where(dc >= 0.0, np.hypot(dc, dl), dc - dl)
        return d - self.offset

    def _cell(self, p):
        """Locate ``p``: base cell corner values and fractional offsets.

        Returns ``None`` when ``p`` lies outside the baked box, so the caller
        can fall back to the out-of-box treatment. Pulls the 2x2x2 block out as
        a contiguous slice and converts once to Python floats, which is what
        makes the scalar path cheap: one NumPy call instead of eight gathers.

        The bounds test carries a tolerance of a billionth of a cell. That is
        far below any physical scale here, and it guarantees termination: the
        out-of-box path re-enters this method at the clipped point, which lands
        on the boundary and must be *accepted* there. Rounding in
        ``(upper - origin) / spacing`` can otherwise place it a bit past the
        last index and bounce the call straight back out again.
        """
        o, inv, cap = self._o, self._inv, self._cap
        u0 = (float(p[0]) - o[0]) * inv[0]
        u1 = (float(p[1]) - o[1]) * inv[1]
        u2 = (float(p[2]) - o[2]) * inv[2]
        tol = 1e-9
        if (u0 < -tol or u1 < -tol or u2 < -tol
                or u0 > cap[0] + 1 + tol or u1 > cap[1] + 1 + tol
                or u2 > cap[2] + 1 + tol):
            return None
        u0 = 0.0 if u0 < 0.0 else u0
        u1 = 0.0 if u1 < 0.0 else u1
        u2 = 0.0 if u2 < 0.0 else u2

        i = cap[0] if u0 >= cap[0] else int(u0)
        j = cap[1] if u1 >= cap[1] else int(u1)
        k = cap[2] if u2 >= cap[2] else int(u2)
        block = self.grid[i:i + 2, j:j + 2, k:k + 2].tolist()
        return block, u0 - i, u1 - j, u2 - k

    def _outside(self, p):
        """Distance and gradient for a point beyond the baked box.

        Not a rare path in practice: the shaft-clearance fixture samples points
        along the whole instrument, and most of those sit well outside a box
        baked around one organ, on every control step. Evaluating them through
        the vectorised path and then finite-differencing it costs about 400 us
        a step -- a fifth of the budget, spent entirely on points that are
        nowhere near the anatomy. Both quantities have closed forms, so this
        takes the same few microseconds as an interior query.

        With ``p'`` the clipped point, ``u = p - p'`` (non-zero only on clipped
        axes) and ``d'``, ``g'`` the field and its gradient at ``p'``:

            d = hypot(d', |u|)     and    grad = (d' * g'_free + u) / d

        where ``g'_free`` masks off the clipped axes, along which ``p'`` cannot
        move. The two terms are orthogonal by construction, so
        ``|grad| <= 1``, with equality exactly when ``g'_free`` is a unit
        vector. It dips below 1 off an edge or corner, where two or three axes
        are masked and the bound is at its loosest -- which is correct rather
        than a defect: this is a lower bound on distance, not a distance, and
        every valid lower bound is 1-Lipschitz. The constraint is slack by
        centimetres out there in any case.
        """
        o, hi = self._o, self._hi
        px, py, pz = float(p[0]), float(p[1]), float(p[2])
        cx = o[0] if px < o[0] else (hi[0] if px > hi[0] else px)
        cy = o[1] if py < o[1] else (hi[1] if py > hi[1] else py)
        cz = o[2] if pz < o[2] else (hi[2] if pz > hi[2] else pz)
        ux, uy, uz = px - cx, py - cy, pz - cz
        delta = (ux * ux + uy * uy + uz * uz) ** 0.5

        d_edge = self.distance((cx, cy, cz)) + self.offset      # undilated
        g = self.gradient((cx, cy, cz))
        # p' cannot move along a clipped axis, so the field's gradient there
        # says nothing about how d varies with p.
        if ux:
            g[0] = 0.0
        if uy:
            g[1] = 0.0
        if uz:
            g[2] = 0.0

        u = np.array([ux, uy, uz])
        if d_edge >= 0.0:
            d = (d_edge * d_edge + delta * delta) ** 0.5
            grad = (d_edge * g + u) / d if d > 0.0 else g
        else:
            # p' inside the region: the containment argument lapses, fall back
            # to the plain 1-Lipschitz bound.
            d = d_edge - delta
            grad = g - u / delta if delta > 0.0 else g
        return d - self.offset, grad

    def distance(self, p) -> float:
        cell = self._cell(p)
        if cell is None:                        # outside: conservative bound
            return self._outside(p)[0]

        ((c000, c001), (c010, c011)), ((c100, c101), (c110, c111)) = cell[0]
        fx, fy, fz = cell[1], cell[2], cell[3]
        mx, my, mz = 1.0 - fx, 1.0 - fy, 1.0 - fz

        c0 = ((c000 * mx + c100 * fx) * my + (c010 * mx + c110 * fx) * fy)
        c1 = ((c001 * mx + c101 * fx) * my + (c011 * mx + c111 * fx) * fy)
        return c0 * mz + c1 * fz - self.offset

    def gradient(self, p) -> np.ndarray:
        """Analytic gradient of the trilinear interpolant inside the box.

        Exact for the interpolated field, and so consistent with ``distance``:
        the base class's finite differences would instead measure the same
        interpolant through a second approximation. It is piecewise constant
        across cell faces -- a real discontinuity of the interpolant, not an
        artefact -- which the VFI tolerates because the constraint row is
        rebuilt from scratch every control step. Outside the box a different
        closed form applies; see ``_outside``.
        """
        cell = self._cell(p)
        if cell is None:
            return self._outside(p)[1]

        ((c000, c001), (c010, c011)), ((c100, c101), (c110, c111)) = cell[0]
        fx, fy, fz = cell[1], cell[2], cell[3]
        mx, my, mz = 1.0 - fx, 1.0 - fy, 1.0 - fz
        ix, iy, iz = self._inv

        gx = ((c100 - c000) * my * mz + (c110 - c010) * fy * mz
              + (c101 - c001) * my * fz + (c111 - c011) * fy * fz) * ix
        gy = ((c010 - c000) * mx * mz + (c110 - c100) * fx * mz
              + (c011 - c001) * mx * fz + (c111 - c101) * fx * fz) * iy
        gz = ((c001 - c000) * mx * my + (c101 - c100) * fx * my
              + (c011 - c010) * mx * fy + (c111 - c110) * fx * fy) * iz
        return np.array([gx, gy, gz])
