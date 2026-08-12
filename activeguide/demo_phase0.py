"""Phase 0 demo: a Forbidden-Region Virtual Fixture deflecting a tool.

A P-controller pulls the tool tip straight toward a goal on the far side of a
protected sphere. The *ghost* (orange) follows that raw command and plows
through the anatomy. The *tool* (blue) has the fixture applied: it slides
around the wall and still reaches the goal, never penetrating.

Run headless (default): verifies the invariant and writes plots to logs/.
    python -m activeguide.demo_phase0
Watch it live in the MuJoCo viewer:
    python -m activeguide.demo_phase0 --view
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import mujoco

from .sdf import Sphere
from .fixtures import ForbiddenRegionFixture

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(HERE, "model", "scene.xml")
LOGDIR = os.path.join(HERE, "logs")

TOOL_RADIUS = 0.006
KP = 2.0              # P gain pulling toward goal
MAX_SPEED = 0.05      # m/s command cap
GOAL_TOL = 0.004      # stop when this close to goal (m)


def _name2id(model, objtype, name):
    return mujoco.mj_name2id(model, objtype, name)


def build_scene():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)

    gid = _name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "forbidden")
    center = model.geom_pos[gid].copy()
    radius = float(model.geom_size[gid][0])

    goal = model.site_pos[_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "goal")].copy()

    tool_mocap = model.body_mocapid[_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tool")]
    ghost_mocap = model.body_mocapid[_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ghost")]

    sdf = Sphere(center, radius)
    # Hold the tool's CENTER margin out so its 6 mm body never touches anatomy.
    fixture = ForbiddenRegionFixture(sdf, margin=TOOL_RADIUS + 0.004, influence=0.045)

    return dict(model=model, data=data, goal=goal, sdf=sdf, fixture=fixture,
                tool_mocap=tool_mocap, ghost_mocap=ghost_mocap)


def simulate(scene, max_steps=6000, on_step=None):
    model, data = scene["model"], scene["data"]
    goal, sdf, fixture = scene["goal"], scene["sdf"], scene["fixture"]
    tool_mocap, ghost_mocap = scene["tool_mocap"], scene["ghost_mocap"]
    dt = model.opt.timestep

    tool = data.mocap_pos[tool_mocap].copy()
    ghost = data.mocap_pos[ghost_mocap].copy()

    log = {k: [] for k in
           ("t", "tool", "ghost", "d_tool", "d_ghost", "force", "warn")}

    for step in range(max_steps):
        # Raw command: P-controller toward the goal, speed-capped.
        err = goal - tool
        v_des = KP * err
        s = np.linalg.norm(v_des)
        if s > MAX_SPEED:
            v_des *= MAX_SPEED / s

        v_allowed, d_tool, _ = fixture.filter_velocity(tool, v_des, dt)
        force, fmag = fixture.feedback_force(tool, v_allowed)
        warn = fixture.warning_level(tool)

        tool = tool + v_allowed * dt
        # Ghost obeys the same raw command, unconstrained.
        gerr = goal - ghost
        gv = KP * gerr
        gs = np.linalg.norm(gv)
        if gs > MAX_SPEED:
            gv *= MAX_SPEED / gs
        ghost = ghost + gv * dt

        data.mocap_pos[tool_mocap] = tool
        data.mocap_pos[ghost_mocap] = ghost
        mujoco.mj_forward(model, data)

        log["t"].append(step * dt)
        log["tool"].append(tool.copy())
        log["ghost"].append(ghost.copy())
        log["d_tool"].append(d_tool)
        log["d_ghost"].append(sdf.distance(ghost))
        log["force"].append(fmag)
        log["warn"].append(warn)

        if on_step is not None:
            on_step(step)

        if np.linalg.norm(goal - tool) < GOAL_TOL:
            break

    for k in log:
        log[k] = np.array(log[k])
    return log


def verify(log, fixture):
    """The core invariant: the tool tip never enters the protected surface
    (its center stays >= tool radius from it), while the ghost does."""
    min_d_tool = float(log["d_tool"].min())
    min_d_ghost = float(log["d_ghost"].min())
    reached = bool(np.linalg.norm(log["tool"][-1] - log["ghost"][-1]) < 0.05
                   or log["d_tool"][-1] > 0)  # tool ended in free space
    tool_safe = min_d_tool >= TOOL_RADIUS - 1e-4
    ghost_breached = min_d_ghost < TOOL_RADIUS
    print(f"  tool  min SDF distance : {min_d_tool*1000:7.2f} mm  "
          f"(must stay >= {TOOL_RADIUS*1000:.1f} mm tip radius)")
    print(f"  ghost min SDF distance : {min_d_ghost*1000:7.2f} mm")
    print(f"  peak warning level     : {float(log['warn'].max()):7.2f}  "
          f"(vibrotactile cue, 0..1 -- fires as tool nears the wall)")
    print(f"  peak rendered force    : {float(log['force'].max()):7.2f} N  "
          f"(0 is expected: robot-side clamp stops AT the margin, no penetration)")
    print(f"  tool stays out of anatomy : {'PASS' if tool_safe else 'FAIL'}")
    print(f"  ghost penetrates anatomy  : {'PASS' if ghost_breached else 'FAIL'}")
    return tool_safe and ghost_breached


def make_plots(log, scene):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(LOGDIR, exist_ok=True)
    margin = scene["fixture"].margin
    sdf = scene["sdf"]

    # 1) distance-to-anatomy vs time
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0.0, color="k", lw=1)
    ax.axhline(margin * 1000, color="r", ls="--", lw=1, label="fixture margin")
    ax.plot(log["t"], log["d_tool"] * 1000, color="tab:blue", label="tool (constrained)")
    ax.plot(log["t"], log["d_ghost"] * 1000, color="tab:orange", label="ghost (raw)")
    ax.fill_between(log["t"], -100, 0, color="red", alpha=0.08)
    ax.set_ylim(min(-5, log["d_ghost"].min() * 1000 - 5), log["d_tool"].max() * 1000 + 5)
    ax.set_xlabel("time (s)"); ax.set_ylabel("signed distance to anatomy (mm)")
    ax.set_title("Phase 0: tool stays outside the forbidden region")
    ax.legend(); fig.tight_layout()
    p1 = os.path.join(LOGDIR, "phase0_distance.png")
    fig.savefig(p1, dpi=120); plt.close(fig)

    # 2) top-down (x-y) trajectory at the working plane
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(sdf.center[0] + sdf.radius * np.cos(th),
            sdf.center[1] + sdf.radius * np.sin(th), "r-", label="anatomy")
    ax.fill(sdf.center[0] + sdf.radius * np.cos(th),
            sdf.center[1] + sdf.radius * np.sin(th), color="red", alpha=0.1)
    ax.plot(log["ghost"][:, 0], log["ghost"][:, 1], color="tab:orange", label="ghost (raw)")
    ax.plot(log["tool"][:, 0], log["tool"][:, 1], color="tab:blue", label="tool (constrained)")
    ax.plot(log["tool"][0, 0], log["tool"][0, 1], "ko", label="start")
    ax.plot(scene["goal"][0], scene["goal"][1], "g*", ms=14, label="goal")
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Phase 0: fixture routes the tool around anatomy")
    ax.legend(loc="upper right", fontsize=8); fig.tight_layout()
    p2 = os.path.join(LOGDIR, "phase0_trajectory.png")
    fig.savefig(p2, dpi=120); plt.close(fig)
    return p1, p2


def run_viewer(scene):
    import mujoco.viewer
    with mujoco.viewer.launch_passive(scene["model"], scene["data"]) as viewer:
        def step_cb(_):
            viewer.sync()
        simulate(scene, on_step=step_cb)
        # hold the final frame briefly
        import time
        end = time.time() + 3.0
        while viewer.is_running() and time.time() < end:
            viewer.sync()
            time.sleep(0.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", action="store_true", help="show the MuJoCo viewer")
    ap.add_argument("--no-plot", action="store_true", help="skip writing plots")
    args = ap.parse_args()

    scene = build_scene()
    if args.view:
        run_viewer(scene)
        return

    log = simulate(scene)
    print("Phase 0 -- Forbidden-Region Virtual Fixture")
    ok = verify(log, scene["fixture"])
    if not args.no_plot:
        p1, p2 = make_plots(log, scene)
        print(f"  wrote {p1}")
        print(f"  wrote {p2}")
    print("RESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
