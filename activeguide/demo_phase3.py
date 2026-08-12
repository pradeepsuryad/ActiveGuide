"""Phase 3 demo: what a sphere costs you once the anatomy is real.

Phases 0-2 protected a sphere, because a closed-form SDF was all the controller
needed. Real anatomy is neither spherical nor convex, and approximating it puts
the surgeon in a bind that has nothing to do with the controller:

    fit the sphere INSIDE the organ   -> the fixture is unsafe. Everywhere the
                                         organ bulges past the sphere, the wall
                                         says "clear" and the tool cuts in.
    fit the sphere AROUND the organ   -> the fixture is safe but takes with it
                                         every cubic centimetre between organ
                                         and sphere, and the target may end up
                                         inside the forbidden region entirely.

Neither knob setting is right, because the error is in the *shape*, not the
size. This demo runs one task three times -- inscribed sphere, circumscribed
sphere, baked mesh field -- with identical fixtures, margin, damping, velocity
box and integrator, and scores all three against the true mesh.

    python -m activeguide.demo_phase3
    python -m activeguide.demo_phase3 --no-video     # plots only (faster)
    python -m activeguide.demo_phase3 --rebuild      # re-bake the anatomy

Scoring is deliberately not self-referential. Each controller is judged by
``anatomy.MeshSdf``, exact closest-triangle distance to the real surface, never
by the approximation it was driving on -- so the mesh run cannot be flattered by
grading it against its own grid.

Writes to logs/:
    phase3_dashboard.png    metric panels, all three runs overlaid
    phase3_field.png        the baked field, sliced, with both spheres on it
    phase3.mp4              side-by-side video
    phase3_results.md       the results table
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import mujoco

from .anatomy import MeshSdf, bake_mesh, load_sdf, make_phantom, save_sdf
from .constraints import (
    JointLimitConstraint,
    RcmConstraint,
    SdfConstraint,
    ShaftClearanceConstraint,
)
from .kinematics import RobotKinematics
from .qp import QpController
from .sdf import Sphere

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR = os.path.join(HERE, "logs")
SCENE = os.path.join(HERE, "model", "scene_panda_anatomy.xml")
SDF_NPZ = os.path.join(HERE, "model", "anatomy_sdf.npz")
MESH_STL = os.path.join(HERE, "model", "anatomy.stl")

DT = 0.002
MAX_STEPS = 6000
SPEED = 0.02                     # commanded tool speed (m/s)
GOAL_TOL = 5e-4
MARGIN = 0.004                   # tool radius + clearance
SPACING = 0.0015                 # SDF cell size

TIP = "tool_tip"
SHAFT_SITES = ["shaft_50", "shaft_75"]
AUDIT_SITES = [TIP] + SHAFT_SITES

# Placement relative to the tool tip at the "ready" keyframe. The organ sits
# between the start pose and the goal, so the tip cannot reach the target
# without negotiating the surface -- which is the only way the *shape* of the
# forbidden region can show up in the result at all.
ANATOMY_OFFSET = (0.0, -0.032, -0.006)
GOAL_OFFSET = (0.0, -0.070, -0.004)

# The instrument works in the y-z plane at x ~ 0.307, so the camera looks back
# along +x to see that plane face-on; from any other angle the tip's excursion
# around the organ is foreshortened into nothing.
CAMERA = dict(azimuth=0, elevation=-12, distance=0.30,
              lookat=(0.307, -0.035, 0.168))


# --------------------------------------------------------------------------
# Anatomy asset
# --------------------------------------------------------------------------
def anatomy_asset(rebuild: bool = False, spacing: float = SPACING):
    """Return ``(mesh, field, info)`` for the protected organ, baking if needed.

    Cached because baking is seconds and querying is microseconds, and because
    the cost belongs to the anatomy, not to the control loop -- leaving it
    inside the loop would quietly make the reported step times a lie.
    """
    import trimesh

    if not rebuild and os.path.exists(SDF_NPZ) and os.path.exists(MESH_STL):
        field = load_sdf(SDF_NPZ)
        mesh = trimesh.load(MESH_STL, force="mesh")
        return mesh, field, {"cached": True, "spacing": float(field.spacing[0])}

    mesh = make_phantom(radius=0.030, seed=0)
    field, info = bake_mesh(mesh, spacing=spacing, pad=0.025, report=True)
    info["cached"] = False
    os.makedirs(os.path.dirname(SDF_NPZ), exist_ok=True)
    save_sdf(field, SDF_NPZ)
    mesh.export(MESH_STL)
    return mesh, field, info


def sphere_fits(mesh):
    """Inscribed and circumscribed radii of ``mesh`` about its own origin.

    The anatomy is baked and stored in its own frame and *placed* into the scene
    by translation, so both radii are taken about that frame's origin -- the
    same point the field is placed at, which is what keeps the two spheres and
    the mesh concentric in the scene.

    The circumscribed radius is the farthest vertex. The inscribed radius is the
    exact distance to the nearest point of the *surface*, not to the nearest
    vertex: on a coarse mesh the closest surface point usually lies inside a
    triangle, and using vertices would overstate how much of the organ a sphere
    can hold.
    """
    r_out = float(np.linalg.norm(np.asarray(mesh.vertices, dtype=float), axis=1).max())
    r_in = float(MeshSdf(mesh).unsigned_many(np.zeros((1, 3)))[0])
    return r_in, r_out


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
def run(wall_kind, mesh, field, record: bool):
    """Drive the tip to the goal with one model of the anatomy."""
    model = mujoco.MjModel.from_xml_path(SCENE)
    kin = RobotKinematics(model)
    mujoco.mj_resetDataKeyframe(model, kin.data, 0)
    kin.forward()

    tip0 = kin.site_position(TIP)
    center = tip0 + np.asarray(ANATOMY_OFFSET, dtype=float)
    goal = tip0 + np.asarray(GOAL_OFFSET, dtype=float)
    trocar = kin.site_position("trocar")

    r_in, r_out = sphere_fits(mesh)
    if wall_kind == "sphere_in":
        sdf, shown_radius = Sphere(center, r_in), r_in
    elif wall_kind == "sphere_out":
        sdf, shown_radius = Sphere(center, r_out), r_out
    else:
        # Dilate the field by its own measured interpolation error, so the
        # discretisation can only ever cost clearance, never borrow it.
        sdf = field.translated(center, offset=field.bake_error)
        shown_radius = 1e-6

    ctl = QpController(
        kin, TIP,
        [SdfConstraint(sdf, TIP, margin=MARGIN),
         ShaftClearanceConstraint(sdf, SHAFT_SITES, margin=MARGIN),
         JointLimitConstraint(),
         RcmConstraint(trocar, proximal="shaft_00", distal=TIP, tol=5e-4)],
        qd_limit=1.5)

    # Keep the picture and the constraint tied to the same numbers.
    def _set(kind, name, field_, value):
        i = mujoco.mj_name2id(model, kind, name)
        field_[i] = value

    _set(mujoco.mjtObj.mjOBJ_GEOM, "forbidden", model.geom_pos, center)
    _set(mujoco.mjtObj.mjOBJ_GEOM, "model_sphere", model.geom_pos, center)
    _set(mujoco.mjtObj.mjOBJ_GEOM, "model_sphere", model.geom_size,
         [shown_radius, 0, 0])
    _set(mujoco.mjtObj.mjOBJ_SITE, "goal", model.site_pos, goal)

    renderer = frames = cam = None
    if record:
        renderer = mujoco.Renderer(model, height=480, width=640)
        frames = []
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        for k, v in CAMERA.items():
            setattr(cam, k, np.asarray(v) if k == "lookat" else v)

    log = {k: [] for k in ("t", "tip", "sites", "modelled", "rcm", "jlim",
                           "solve_us", "goal_d")}
    lo, hi = kin.joint_limits()
    rcm = ctl.constraints[3]

    for step in range(MAX_STEPS):
        p = kin.site_position(TIP)
        v = goal - p
        s = float(np.linalg.norm(v))
        if s > SPEED:
            v *= SPEED / s

        _, info = ctl.step(v, DT)

        tip = kin.site_position(TIP)
        log["t"].append(step * DT)
        log["tip"].append(tip)
        log["sites"].append([kin.site_position(s_) for s_ in AUDIT_SITES])
        # What this controller *believed* its clearance was.
        log["modelled"].append(sdf.distance(tip) - MARGIN)
        log["rcm"].append(rcm.deviation(kin))
        q = kin.q
        log["jlim"].append(float(np.max(np.concatenate([lo - q, q - hi]))))
        log["solve_us"].append(info["solve_time"] * 1e6)
        log["goal_d"].append(float(np.linalg.norm(goal - tip)))

        if record and step % 8 == 0:
            mujoco.mj_forward(model, kin.data)
            renderer.update_scene(kin.data, camera=cam)
            frames.append(renderer.render())

        if log["goal_d"][-1] < GOAL_TOL:
            break

    for k in log:
        log[k] = np.asarray(log[k])
    log.update(frames=frames, wall=wall_kind, steps=len(log["t"]),
               infeasible=ctl.n_infeasible, timing=ctl.timing(),
               center=center, goal=goal, r_in=r_in, r_out=r_out, sdf=sdf)
    if renderer is not None:
        renderer.close()
    return log


def audit(log, oracle: MeshSdf, center) -> dict:
    """Score a run against the true mesh, in one batched pass.

    Every logged sample point -- tip and shaft -- is measured, because a fixture
    on the tip alone cannot see the shaft sweeping through the organ, and that
    is a real failure mode rather than a hypothetical one.
    """
    pts = log["sites"].reshape(-1, 3) - center
    true_d = oracle.distance_many(pts).reshape(log["sites"].shape[:2])

    tip_true = true_d[:, 0]
    worst = float(true_d.min())
    return dict(
        wall=log["wall"],
        steps=int(log["steps"]),
        reached=bool(log["goal_d"][-1] < GOAL_TOL),
        goal_mm=float(log["goal_d"][-1]) * 1e3,
        true_min_mm=worst * 1e3,
        organ_viol_mm=float(max(0.0, -worst)) * 1e3,
        tip_true_min_mm=float(tip_true.min()) * 1e3,
        modelled_min_mm=float(log["modelled"].min()) * 1e3,
        rcm_max_mm=float(log["rcm"].max()) * 1e3,
        jlim_viol_rad=float(max(0.0, log["jlim"].max())),
        p99_us=float(np.percentile(log["solve_us"], 99)),
        infeasible=int(log["infeasible"]),
        true_d=true_d,
    )


def coverage(field, r_in, r_out) -> dict:
    """How much each sphere gets wrong, independently of any trajectory.

    The run below reports what happened on one path, which necessarily depends
    on where that path went. This is the path-free companion: integrate the
    disagreement between each sphere and the organ over the baked grid, which
    already holds the signed distance at every cell.

        unprotected  organ volume the inscribed sphere leaves outside itself --
                     real tissue the fixture would let an instrument enter.
        denied       free volume the circumscribed sphere swallows -- workspace
                     forbidden although nothing is there.

    Both are properties of the shape alone, so neither can be improved by
    re-tuning the radius: shrinking the sphere trades one for the other. That
    is the trade the mesh field removes rather than balances.
    """
    axes = [field.origin[i] + np.arange(field.grid.shape[i]) * field.spacing[i]
            for i in range(3)]
    R = np.sqrt(sum(a.reshape([-1 if j == i else 1 for j in range(3)]) ** 2
                    for i, a in enumerate(axes)))
    inside = field.grid < 0.0
    cell = float(np.prod(field.spacing)) * 1e6            # cm^3
    return dict(
        organ_cm3=float(inside.sum()) * cell,
        unprotected_cm3=float((inside & (R > r_in)).sum()) * cell,
        denied_cm3=float((~inside & (R < r_out)).sum()) * cell,
    )


def query_cost(sdf, pts, n=2000):
    """Time distance+gradient the way the control loop actually calls them."""
    probe = pts[np.linspace(0, len(pts) - 1, min(n, len(pts))).astype(int)]
    t0 = time.perf_counter()
    for p in probe:
        sdf.distance(p)
        sdf.normal(p)
    return (time.perf_counter() - t0) / len(probe) * 1e6


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
STYLE = {
    "sphere_in": dict(color="tab:red", label="inscribed sphere"),
    "sphere_out": dict(color="tab:orange", label="circumscribed sphere"),
    "mesh": dict(color="tab:blue", label="baked mesh SDF (ours)"),
}


def make_dashboard(runs, scores):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    fig.suptitle("ActiveGuide Phase 3 - one organ, three models of it "
                 "(all scored against the true mesh)", fontsize=13)

    ref = runs[0]
    center, goal = ref["center"], ref["goal"]

    # Organ cross-section in the plane of travel, straight from the field.
    ax = axes[0, 0]
    field = next(r["sdf"] for r in runs if r["wall"] == "mesh")
    ys = np.linspace(center[1] - 0.05, center[1] + 0.05, 220)
    zs = np.linspace(center[2] - 0.05, center[2] + 0.05, 220)
    Y, Z = np.meshgrid(ys, zs)
    P = np.stack([np.full(Y.size, center[0]), Y.ravel(), Z.ravel()], axis=1)
    D = field.distance_many(P).reshape(Y.shape)
    ax.contourf(Y * 1e3, Z * 1e3, D <= 0, levels=[0.5, 1.5], colors=["#d9534f"],
                alpha=0.35)
    ax.contour(Y * 1e3, Z * 1e3, D, levels=[0.0], colors="#b52b26", linewidths=1.2)
    for r, label in ((ref["r_in"], "inscribed"), (ref["r_out"], "circumscribed")):
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot((center[1] + r * np.cos(th)) * 1e3,
                (center[2] + r * np.sin(th)) * 1e3, ls="--", lw=1,
                color=STYLE["sphere_in" if label == "inscribed" else "sphere_out"]["color"])

    for log in runs:
        tip = log["tip"]
        ax.plot(tip[:, 1] * 1e3, tip[:, 2] * 1e3, lw=1.6, **STYLE[log["wall"]])
    ax.plot(ref["tip"][0, 1] * 1e3, ref["tip"][0, 2] * 1e3, "ko", ms=5)
    ax.plot(goal[1] * 1e3, goal[2] * 1e3, "*", color="tab:green", ms=13)
    ax.set(xlabel="y (mm)", ylabel="z (mm)",
           title="tool-tip path through the organ cross-section")
    ax.set_aspect("equal", adjustable="datalim")

    for log, sc in zip(runs, scores):
        st = STYLE[log["wall"]]
        t = log["t"]
        axes[0, 1].plot(t, sc["true_d"].min(axis=1) * 1e3, **st)
        axes[0, 2].plot(t, log["modelled"] * 1e3, **st)
        axes[1, 0].plot(t, log["goal_d"] * 1e3, **st)
        axes[1, 1].plot(t, log["rcm"] * 1e3, **st)
        axes[1, 2].plot(t, log["solve_us"], lw=0.8, **st)

    axes[0, 1].axhline(0, color="k", ls="--", lw=1, label="organ surface")
    axes[0, 1].set(xlabel="time (s)", ylabel="true clearance (mm)",
                   title="TRUE clearance to the mesh (below 0 = inside the organ)")
    axes[0, 2].axhline(0, color="k", ls="--", lw=1, label="believed margin")
    axes[0, 2].set(xlabel="time (s)", ylabel="modelled clearance (mm)",
                   title="what each controller believed (all stay >= 0)")
    axes[1, 0].set(xlabel="time (s)", ylabel="distance to goal (mm)",
                   title="task progress")
    axes[1, 1].axhline(0.5, color="r", ls="--", lw=1, label="0.5 mm tolerance")
    axes[1, 1].set(xlabel="time (s)", ylabel="RCM deviation (mm)",
                   title="trocar / remote centre")
    axes[1, 2].set(xlabel="time (s)", ylabel="solve time (us)",
                   title="QP solve time")

    for ax in axes.ravel():
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(LOGDIR, "phase3_dashboard.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def make_field_figure(runs):
    """The baked field itself: why a sphere cannot be the right answer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = runs[0]
    center = ref["center"]
    field = next(r["sdf"] for r in runs if r["wall"] == "mesh")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, (ai, bi, an, bn) in zip(axes, [(1, 2, "y", "z"), (0, 1, "x", "y")]):
        a = np.linspace(center[ai] - 0.05, center[ai] + 0.05, 260)
        b = np.linspace(center[bi] - 0.05, center[bi] + 0.05, 260)
        A, B = np.meshgrid(a, b)
        P = np.tile(center, (A.size, 1))
        P[:, ai] = A.ravel()
        P[:, bi] = B.ravel()
        D = field.distance_many(P).reshape(A.shape) * 1e3

        m = np.abs(D).max()
        im = ax.pcolormesh(A * 1e3, B * 1e3, D, cmap="RdBu", vmin=-m, vmax=m,
                           shading="auto")
        ax.contour(A * 1e3, B * 1e3, D, levels=np.arange(-30, 31, 5),
                   colors="k", linewidths=0.3, alpha=0.4)
        ax.contour(A * 1e3, B * 1e3, D, levels=[0.0], colors="k", linewidths=1.6)
        th = np.linspace(0, 2 * np.pi, 200)
        for r, key in ((ref["r_in"], "sphere_in"), (ref["r_out"], "sphere_out")):
            ax.plot((center[ai] + r * np.cos(th)) * 1e3,
                    (center[bi] + r * np.sin(th)) * 1e3, ls="--", lw=1.4,
                    **STYLE[key])
        ax.set(xlabel=f"{an} (mm)", ylabel=f"{bn} (mm)",
               title=f"baked signed distance, {an}-{bn} slice")
        ax.set_aspect("equal")
        ax.legend(fontsize=7, loc="upper right")
        fig.colorbar(im, ax=ax, label="signed distance (mm)")

    fig.suptitle("The forbidden region the controller actually sees "
                 "(black = organ surface)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(LOGDIR, "phase3_field.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def save_media(runs):
    written = []
    try:
        from PIL import Image
    except ImportError:
        return written

    have = [r for r in runs if r.get("frames")]
    for log in have:
        still = os.path.join(LOGDIR, f"phase3_{log['wall']}.png")
        Image.fromarray(log["frames"][-1]).save(still)
        written.append(still)

    if len(have) >= 2:
        try:
            import imageio.v2 as imageio
        except ImportError:
            return written
        from .viz import stack_labeled

        combo = stack_labeled([r["frames"] for r in have],
                              [STYLE[r["wall"]]["label"] for r in have])
        vid = os.path.join(LOGDIR, "phase3.mp4")
        imageio.mimsave(vid, combo, fps=25, quality=8, macro_block_size=1)
        written.append(vid)
    return written


def results_table(scores) -> str:
    hdr = (f"| {'anatomy model':22} | {'goal':>10} | {'TRUE clear':>10} | "
           f"{'organ viol':>10} | {'believed':>9} | {'RCM max':>8} | {'p99':>7} |")
    sep = "|" + "|".join("-" * (len(c) + 2) for c in hdr.split("|")[1:-1]) + "|"
    lines = [hdr, sep]
    for s in scores:
        name = STYLE[s["wall"]]["label"]
        goal = f"{s['goal_mm']:.2f} mm" if s["reached"] else f"{s['goal_mm']:.1f} STALL"
        lines.append(
            f"| {name:22} | {goal:>10} | {s['true_min_mm']:>7.2f} mm | "
            f"{s['organ_viol_mm']:>7.2f} mm | {s['modelled_min_mm']:>6.2f} mm | "
            f"{s['rcm_max_mm']:>5.2f} mm | {s['p99_us']:>4.0f} us |")
    return "\n".join(lines)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--rebuild", action="store_true",
                    help="re-bake the anatomy SDF even if cached")
    ap.add_argument("--spacing", type=float, default=SPACING,
                    help="SDF cell size in metres when baking")
    args = ap.parse_args()

    os.makedirs(LOGDIR, exist_ok=True)

    mesh, field, info = anatomy_asset(rebuild=args.rebuild, spacing=args.spacing)
    if info.get("cached"):
        print(f"anatomy    cached field, {args.spacing * 0 + info['spacing'] * 1e3:.2f} mm cells, "
              f"error offset {field.bake_error * 1e3:.3f} mm")
    else:
        print(f"anatomy    baked {info['cells']:,} cells @ "
              f"{info['spacing'] * 1e3:.2f} mm in {info['bake_s']:.1f} s | "
              f"max over-estimate {info['max_overestimate'] * 1e3:.3f} mm | "
              f"{info['sign_mismatch']} sign mismatches")
    print(f"           {len(mesh.faces)} faces, watertight={mesh.is_watertight}, "
          f"volume {mesh.volume * 1e6:.1f} cm^3")

    oracle = MeshSdf(mesh)
    runs, scores = [], []
    for wall in ("sphere_out", "sphere_in", "mesh"):
        log = run(wall, mesh, field, record=not args.no_video)
        sc = audit(log, oracle, log["center"])
        runs.append(log)
        scores.append(sc)
        verdict = "reached" if sc["reached"] else "STALLED"
        print(f"  {STYLE[wall]['label']:22} {verdict:8} in {sc['steps']:4d} steps | "
              f"true clearance {sc['true_min_mm']:7.2f} mm | "
              f"organ violation {sc['organ_viol_mm']:5.2f} mm | "
              f"p99 {sc['p99_us']:5.0f} us")

    ref = runs[0]
    cov = coverage(field, ref["r_in"], ref["r_out"])
    print(f"\nsphere fits about the organ centre: inscribed "
          f"{ref['r_in'] * 1e3:.1f} mm, circumscribed {ref['r_out'] * 1e3:.1f} mm "
          f"(shape error {(ref['r_out'] - ref['r_in']) * 1e3:.1f} mm)")
    print(f"organ {cov['organ_cm3']:.1f} cm^3 | inscribed sphere leaves "
          f"{cov['unprotected_cm3']:.1f} cm^3 "
          f"({cov['unprotected_cm3'] / cov['organ_cm3'] * 100:.0f}% of it) "
          f"unprotected | circumscribed sphere denies "
          f"{cov['denied_cm3']:.1f} cm^3 of free space")

    mesh_sdf = next(r["sdf"] for r in runs if r["wall"] == "mesh")
    tips = next(r["tip"] for r in runs if r["wall"] == "mesh")
    print(f"field query (distance+normal): {query_cost(mesh_sdf, tips):.1f} us, "
          f"vs {query_cost(oracle, tips, n=40):.0f} us for the exact mesh")
    for log, sc in zip(runs, scores):
        t = log["timing"]
        print(f"  {STYLE[log['wall']]['label']:22} solve p50 {t['p50_us']:5.0f} us  "
              f"p99 {t['p99_us']:5.0f} us  infeasible {sc['infeasible']}")

    artifacts = [make_dashboard(runs, scores), make_field_figure(runs)]
    if not args.no_video:
        artifacts += save_media(runs)

    table = results_table(scores)
    print("\n" + table)

    md = os.path.join(LOGDIR, "phase3_results.md")
    with open(md, "w") as f:
        f.write(
            "# ActiveGuide Phase 3 results\n\n"
            "One organ, three models of it. Identical fixtures, margin, damping,\n"
            "velocity box and integrator; only the controller's model of the\n"
            "forbidden region changes.\n\n"
            + table + "\n\n"
            "`TRUE clear` is the smallest exact distance from any sampled point on\n"
            "the instrument (tip and shaft) to the real mesh surface, measured with\n"
            f"`anatomy.MeshSdf`. Negative means the instrument was inside the organ.\n\n"
            f"`believed` is clearance past the {MARGIN * 1e3:.0f} mm margin, as the "
            "controller's own model\nreported it. All three sit at essentially zero: "
            "each rode its safety margin\nand thought it was doing exactly that. "
            "(Slightly negative is normal -- a VFI\nbrakes proportionally to the room "
            "left rather than stopping dead at the\nboundary, so a small transient "
            "into the margin is expected.) None of them\ncould tell that its own model "
            "of the anatomy was wrong; that is the point.\n\n"
            "## The sphere trade-off, independently of this trajectory\n\n"
            "A single run only shows what happened along one path. Integrating the\n"
            "disagreement between each sphere and the organ over the whole baked\n"
            "field gives the path-free version:\n\n"
            f"| organ volume | inscribed leaves unprotected | circumscribed denies |\n"
            f"|---|---|---|\n"
            f"| {cov['organ_cm3']:.1f} cm^3 | {cov['unprotected_cm3']:.1f} cm^3 "
            f"({cov['unprotected_cm3'] / cov['organ_cm3'] * 100:.0f}%) | "
            f"{cov['denied_cm3']:.1f} cm^3 |\n\n"
            f"Inscribed radius {ref['r_in'] * 1e3:.1f} mm, circumscribed "
            f"{ref['r_out'] * 1e3:.1f} mm: no sphere about this centre is within\n"
            f"{(ref['r_out'] - ref['r_in']) * 1e3:.1f} mm of the surface everywhere, so "
            "shrinking the radius only trades\ntissue left unprotected against "
            "workspace needlessly denied. The measured\npenetration above is whatever "
            "that error happened to cost on this path.\n")
    artifacts.append(md)

    print("\nwrote:")
    for a in artifacts:
        print("  " + os.path.relpath(a, HERE))


if __name__ == "__main__":
    main()
