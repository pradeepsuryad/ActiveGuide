"""Phase 2 demo: Cartesian projection vs joint-space QP, on two robots.

Runs the identical task four ways -- {Cartesian baseline, joint-space QP} x
{Franka Panda, dVRK PSM} -- and reports what each does to the robot, not just to
the tool tip. Both controllers get the same fixtures, damping, velocity box and
integrator; the only difference is where enforcement happens.

    python -m activeguide.demo_phase2                # run everything
    python -m activeguide.demo_phase2 --robot panda  # one robot
    python -m activeguide.demo_phase2 --no-video     # plots only (faster)

Writes to logs/:
    phase2_<robot>_dashboard.png    metric panels, both controllers overlaid
    phase2_<robot>.mp4              side-by-side video
    phase2_<robot>_<method>.png     final still
    phase2_results.md               the results table

The PSM run needs model/psm.xml:  python tools/build_psm.py --fetch
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import mujoco

from .baseline import CartesianBaselineController
from .constraints import (
    PSM_COUPLING,
    JointLimitConstraint,
    RcmConstraint,
    SdfConstraint,
    ShaftClearanceConstraint,
)
from .fixtures import ForbiddenRegionFixture, FixtureStack
from .kinematics import RobotKinematics
from .qp import QpController
from .sdf import Sphere

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGDIR = os.path.join(HERE, "logs")
DT = 0.002
MAX_STEPS = 4000
SPEED = 0.02                    # commanded tool speed (m/s)
GOAL_TOL = 5e-4


# --------------------------------------------------------------------------
# Scenes
# --------------------------------------------------------------------------
class Scene:
    """One robot plus a task: reach a goal past protected anatomy, through a port."""

    def __init__(self, name, model_path, tip, shaft_sites, rcm_site,
                 anatomy_offset, anatomy_radius, goal_offset, margin,
                 qd_limit, rcm_distal, equalities=(), keyframe=0, camera=None):
        self.name = name
        self.model_path = os.path.join(HERE, model_path)
        self.tip = tip
        self.shaft_sites = shaft_sites
        self.rcm_site = rcm_site
        # Distal end of the *shaft* for the RCM measurement. On the Panda the
        # shaft is rigid all the way to the tip, so the tip is on the axis. On
        # the PSM the tip is past an articulating wrist, so using it would
        # measure wrist bend as if it were RCM error.
        self.rcm_distal = rcm_distal
        self.anatomy_offset = np.asarray(anatomy_offset, float)
        self.anatomy_radius = anatomy_radius
        self.goal_offset = np.asarray(goal_offset, float)
        self.margin = margin
        self.qd_limit = qd_limit
        self.equalities = list(equalities)
        self.keyframe = keyframe
        self.camera = camera or {}

    def available(self) -> bool:
        return os.path.exists(self.model_path)

    def build(self):
        model = mujoco.MjModel.from_xml_path(self.model_path)
        kin = RobotKinematics(model)
        mujoco.mj_resetDataKeyframe(model, kin.data, self.keyframe)
        kin.forward()

        tip0 = kin.site_position(self.tip)
        sdf = Sphere(tip0 + self.anatomy_offset, self.anatomy_radius)
        goal = tip0 + self.goal_offset
        rcm_pt = kin.site_position(self.rcm_site)
        return model, kin, sdf, goal, rcm_pt, tip0


PANDA = Scene(
    name="panda",
    model_path="model/scene_panda.xml",
    tip="tool_tip",
    shaft_sites=["shaft_50", "shaft_75"],
    rcm_site="trocar",
    anatomy_offset=(0.0, 0.030, -0.030),
    anatomy_radius=0.025,
    goal_offset=(0.0, -0.035, -0.020),
    margin=0.004,
    qd_limit=1.5,
    rcm_distal="tool_tip",          # rigid shaft: tip lies on the shaft axis
    camera=dict(azimuth=135, elevation=-20, distance=0.9,
                lookat=(0.30, 0.0, 0.25)),
)

PSM = Scene(
    name="psm",
    model_path="model/psm.xml",
    tip="tool_tip",
    shaft_sites=["shaft_50", "shaft_75"],
    rcm_site="remote_center",
    anatomy_offset=(0.0, 0.014, 0.012),
    anatomy_radius=0.010,
    goal_offset=(0.0, 0.026, 0.0),
    margin=0.003,
    qd_limit=0.8,
    rcm_distal="shaft_end",         # tip is past the wrist; measure the shaft
    equalities=[PSM_COUPLING],
    camera=dict(azimuth=140, elevation=-18, distance=0.95,
                lookat=(0.0, 0.35, 0.05)),
)


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
def run(scene: Scene, method: str, record: bool):
    """Drive the tip at the goal. Returns a log dict."""
    model, kin, sdf, goal, rcm_pt, tip0 = scene.build()

    wall = SdfConstraint(sdf, scene.tip, margin=scene.margin)
    shaft = ShaftClearanceConstraint(sdf, scene.shaft_sites, margin=scene.margin)
    limits = JointLimitConstraint()
    rcm = RcmConstraint(rcm_pt, proximal=scene.shaft_sites[0],
                        distal=scene.rcm_distal, tol=5e-4)

    if method == "qp":
        ctl = QpController(kin, scene.tip, [wall, shaft, limits, rcm],
                           qd_limit=scene.qd_limit, equalities=scene.equalities)
    else:
        # Same wall geometry, but enforced in Cartesian space then pushed
        # through a pseudo-inverse -- no joint limits, no RCM, no shaft.
        stack = FixtureStack(
            forbidden=ForbiddenRegionFixture(sdf, margin=scene.margin,
                                             influence=scene.margin * 8))
        # Same couplings as the QP: they describe the mechanism, so withholding
        # them would let the baseline command impossible motion (see baseline.py).
        ctl = CartesianBaselineController(kin, scene.tip, stack,
                                          qd_limit=scene.qd_limit,
                                          equalities=scene.equalities)

    renderer = frames = None
    if record:
        renderer = mujoco.Renderer(model, height=480, width=640)
        frames = []
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        for k, v in scene.camera.items():
            setattr(cam, k, np.asarray(v) if k == "lookat" else v)

    log = {k: [] for k in ("t", "tip", "wall_d", "rcm", "jlim", "manip",
                           "solve_us", "goal_d")}
    lo, hi = kin.joint_limits()

    for step in range(MAX_STEPS):
        p = kin.site_position(scene.tip)
        v = goal - p
        s = float(np.linalg.norm(v))
        if s > SPEED:
            v *= SPEED / s

        if method == "qp":
            qdot, info = ctl.step(v, DT)
        else:
            qdot, info = ctl.step(v, DT)

        log["t"].append(step * DT)
        log["tip"].append(kin.site_position(scene.tip))
        # distance past the safety margin: <0 means inside the forbidden margin
        log["wall_d"].append(sdf.distance(kin.site_position(scene.tip)) - scene.margin)
        log["rcm"].append(rcm.deviation(kin))
        q = kin.q
        log["jlim"].append(float(np.max(np.concatenate([lo - q, q - hi]))))
        log["manip"].append(info.get("manipulability", 0.0))
        log["solve_us"].append(info["solve_time"] * 1e6)
        log["goal_d"].append(float(np.linalg.norm(goal - kin.site_position(scene.tip))))

        if record and step % 8 == 0:
            mujoco.mj_forward(model, kin.data)
            renderer.update_scene(kin.data, camera=cam)
            frames.append(renderer.render())

        if log["goal_d"][-1] < GOAL_TOL:
            break

    for k in log:
        log[k] = np.asarray(log[k])
    log["frames"] = frames
    log["method"] = method
    log["robot"] = scene.name
    log["steps"] = len(log["t"])
    log["infeasible"] = ctl.n_infeasible
    log["timing"] = ctl.timing()
    if renderer is not None:
        renderer.close()
    return log


def summarize(log) -> dict:
    """Collapse a run into the numbers that matter."""
    return dict(
        robot=log["robot"],
        method=log["method"],
        steps=int(log["steps"]),
        goal_mm=float(log["goal_d"][-1]) * 1e3,
        reached=bool(log["goal_d"][-1] < GOAL_TOL),
        wall_viol_mm=float(max(0.0, -log["wall_d"].min())) * 1e3,
        jlim_viol_rad=float(max(0.0, log["jlim"].max())),
        rcm_max_mm=float(log["rcm"].max()) * 1e3,
        manip_min=float(log["manip"].min()),
        p50_us=float(np.percentile(log["solve_us"], 50)),
        p99_us=float(np.percentile(log["solve_us"], 99)),
        infeasible=int(log["infeasible"]),
    )


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
def make_dashboard(scene, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {"qp": dict(color="tab:blue", label="joint-space QP (ours)"),
             "cartesian": dict(color="tab:orange", label="Cartesian projection")}

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    fig.suptitle(f"ActiveGuide Phase 2 - {scene.name.upper()}: "
                 "Cartesian projection vs joint-space QP", fontsize=13)

    for log in runs:
        st = style[log["method"]]
        t = log["t"]

        ax = axes[0, 0]
        tip = log["tip"]
        ax.plot(tip[:, 1] * 1e3, tip[:, 2] * 1e3, **st)
        ax.plot(tip[0, 1] * 1e3, tip[0, 2] * 1e3, "ko", ms=5)
        ax.set(xlabel="y (mm)", ylabel="z (mm)", title="tool-tip path")
        ax.set_aspect("equal", adjustable="datalim")

        axes[0, 1].plot(t, log["wall_d"] * 1e3, **st)
        axes[0, 2].plot(t, log["rcm"] * 1e3, **st)
        axes[1, 0].plot(t, log["jlim"], **st)
        axes[1, 1].plot(t, log["manip"], **st)
        axes[1, 2].plot(t, log["goal_d"] * 1e3, **st)

    axes[0, 1].axhline(0, color="r", ls="--", lw=1, label="margin (violation below)")
    axes[0, 1].set(xlabel="time (s)", ylabel="distance past margin (mm)",
                   title="forbidden region")
    axes[0, 2].axhline(0.5, color="r", ls="--", lw=1, label="0.5 mm tolerance")
    axes[0, 2].set(xlabel="time (s)", ylabel="RCM deviation (mm)",
                   title="trocar / remote centre")
    axes[1, 0].axhline(0, color="r", ls="--", lw=1, label="limit (violation above)")
    axes[1, 0].set(xlabel="time (s)", ylabel="worst limit excess (rad)",
                   title="joint limits")
    axes[1, 1].set(xlabel="time (s)", ylabel=r"$\sigma_{min}(J)$",
                   title="manipulability (higher = further from singularity)")
    axes[1, 2].set(xlabel="time (s)", ylabel="distance to goal (mm)",
                   title="task progress")

    for ax in axes.ravel():
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(LOGDIR, f"phase2_{scene.name}_dashboard.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def save_media(scene, runs):
    written = []
    from PIL import Image

    for log in runs:
        if not log.get("frames"):
            continue
        still = os.path.join(LOGDIR, f"phase2_{scene.name}_{log['method']}.png")
        Image.fromarray(log["frames"][-1]).save(still)
        written.append(still)

    have = [r for r in runs if r.get("frames")]
    if len(have) == 2:
        try:
            import imageio.v2 as imageio
        except ImportError:
            return written
        from .viz import stack_labeled

        label = lambda r: ("joint-space QP (ours)" if r["method"] == "qp"
                           else "Cartesian projection")
        combo = stack_labeled([r["frames"] for r in have],
                              [label(r) for r in have])
        vid = os.path.join(LOGDIR, f"phase2_{scene.name}.mp4")
        imageio.mimsave(vid, combo, fps=25, quality=8, macro_block_size=1)
        written.append(vid)
    return written


def results_table(rows) -> str:
    hdr = (f"| {'robot':6} | {'controller':20} | {'goal':>9} | {'wall':>9} | "
           f"{'joint lim':>10} | {'RCM max':>9} | {'min sig':>8} | {'p50':>8} | "
           f"{'p99':>8} |")
    sep = "|" + "|".join("-" * (len(c) + 2) for c in hdr.split("|")[1:-1]) + "|"
    lines = [hdr, sep]
    for r in rows:
        name = "joint-space QP" if r["method"] == "qp" else "Cartesian proj."
        goal = f"{r['goal_mm']:.2f} mm" if r["reached"] else f"{r['goal_mm']:.1f} STALL"
        lines.append(
            f"| {r['robot']:6} | {name:20} | {goal:>9} | "
            f"{r['wall_viol_mm']:>6.3f} mm | {r['jlim_viol_rad']:>7.4f} rad | "
            f"{r['rcm_max_mm']:>6.2f} mm | {r['manip_min']:>8.4f} | "
            f"{r['p50_us']:>5.0f} us | {r['p99_us']:>5.0f} us |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=("panda", "psm", "all"), default="all")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    os.makedirs(LOGDIR, exist_ok=True)
    scenes = {"panda": PANDA, "psm": PSM}
    chosen = list(scenes.values()) if args.robot == "all" else [scenes[args.robot]]

    all_rows, artifacts = [], []
    for scene in chosen:
        if not scene.available():
            print(f"[skip] {scene.name}: {os.path.relpath(scene.model_path, HERE)} "
                  f"not found - run: python tools/build_psm.py --fetch\n")
            continue

        print(f"=== {scene.name.upper()} " + "=" * (58 - len(scene.name)))
        runs = []
        for method in ("cartesian", "qp"):
            log = run(scene, method, record=not args.no_video)
            row = summarize(log)
            runs.append(log)
            all_rows.append(row)
            verdict = "reached" if row["reached"] else "STALLED"
            print(f"  {method:10} {verdict:8} in {row['steps']:4d} steps | "
                  f"wall {row['wall_viol_mm']:6.3f} mm | "
                  f"jlim {row['jlim_viol_rad']:7.4f} rad | "
                  f"RCM {row['rcm_max_mm']:6.2f} mm | "
                  f"p50 {row['p50_us']:5.0f} us | p99 {row['p99_us']:5.0f} us")

        artifacts.append(make_dashboard(scene, runs))
        if not args.no_video:
            artifacts += save_media(scene, runs)
        print()

    if not all_rows:
        sys.exit("nothing ran")

    table = results_table(all_rows)
    print(table)
    print()

    md = os.path.join(LOGDIR, "phase2_results.md")
    with open(md, "w") as f:
        f.write("# ActiveGuide Phase 2 results\n\n"
                "Identical task, fixtures, damping, velocity box and integrator.\n"
                "The only difference is where enforcement happens: Cartesian\n"
                "output space vs joint-velocity space.\n\n"
                + table + "\n\n"
                "`wall` and `joint lim` are worst *violations* (0 is perfect).\n"
                "`min sig` is the smallest singular value of J reached -- lower\n"
                "means the controller drove the arm nearer a singularity.\n")
    artifacts.append(md)

    print("wrote:")
    for a in artifacts:
        print("  " + os.path.relpath(a, HERE))


if __name__ == "__main__":
    main()
