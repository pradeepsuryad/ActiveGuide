"""Phase 1 demo: guidance fixture + forbidden region, composed.

A guidance path arcs over the protected anatomy from start to goal. A
(simulated) operator naively pushes straight toward the goal -- which would cut
through the anatomy -- with an added out-of-plane tremor. The guidance fixture
pulls the tool onto the arc and the forbidden-region fixture guarantees it never
enters the anatomy even if guidance were turned down. The ghost shows the raw,
unassisted command.

Headless (default): verifies guidance + safety, writes plots to logs/.
    python -m activeguide.demo_phase1
Interactive: drive the tool yourself and feel the guideline + wall.
    python -m activeguide.demo_phase1 --view
        WASD = x/y,  Q/E = down/up,  mouse = camera,  ESC = quit
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import mujoco

from .sdf import Sphere
from .paths import Polyline
from .fixtures import ForbiddenRegionFixture, GuidanceFixture, FixtureStack

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(HERE, "model", "scene.xml")
LOGDIR = os.path.join(HERE, "logs")

TOOL_RADIUS = 0.006
KP = 2.0
MAX_SPEED = 0.05
GOAL_TOL = 0.004
TREMOR_AMP = 0.06          # out-of-plane operator tremor (m/s)
TREMOR_HZ = 0.6


def _id(model, t, name):
    return mujoco.mj_name2id(model, t, name)


def build_scene():
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)

    gid = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "forbidden")
    center = model.geom_pos[gid].copy()
    radius = float(model.geom_size[gid][0])
    goal = model.site_pos[_id(model, mujoco.mjtObj.mjOBJ_SITE, "goal")].copy()
    start = model.body_pos[_id(model, mujoco.mjtObj.mjOBJ_BODY, "tool")].copy()

    tool_mocap = model.body_mocapid[_id(model, mujoco.mjtObj.mjOBJ_BODY, "tool")]
    ghost_mocap = model.body_mocapid[_id(model, mujoco.mjtObj.mjOBJ_BODY, "ghost")]
    tool_geom = _id(model, mujoco.mjtObj.mjOBJ_GEOM, "tool_tip")

    # Guidance path: arc over the top of the anatomy (stays well clear of it).
    path = Polyline([
        start,
        [0.07, 0.075, 0.10],
        [0.0, 0.090, 0.10],
        [-0.07, 0.075, 0.10],
        goal,
    ])

    sdf = Sphere(center, radius)
    forbidden = ForbiddenRegionFixture(sdf, margin=TOOL_RADIUS + 0.004, influence=0.045)
    guidance = GuidanceFixture(path, strength=0.85)
    stack = FixtureStack(guidance=guidance, forbidden=forbidden)

    return dict(model=model, data=data, goal=goal, start=start, sdf=sdf, path=path,
                forbidden=forbidden, guidance=guidance, stack=stack,
                tool_mocap=tool_mocap, ghost_mocap=ghost_mocap, tool_geom=tool_geom)


def simulate(scene, max_steps=8000):
    model, data = scene["model"], scene["data"]
    goal, stack, path, sdf = scene["goal"], scene["stack"], scene["path"], scene["sdf"]
    tool_mocap, ghost_mocap = scene["tool_mocap"], scene["ghost_mocap"]
    dt = model.opt.timestep

    tool = data.mocap_pos[tool_mocap].copy()
    ghost = data.mocap_pos[ghost_mocap].copy()
    log = {k: [] for k in ("t", "tool", "ghost", "err_tool", "err_ghost", "d_tool")}

    for step in range(max_steps):
        t = step * dt
        # Operator command: head straight to goal + out-of-plane tremor.
        err = goal - tool
        v_des = KP * err
        sp = np.linalg.norm(v_des)
        if sp > MAX_SPEED:
            v_des *= MAX_SPEED / sp
        v_des = v_des + np.array([0.0, 0.0, TREMOR_AMP * np.sin(2 * np.pi * TREMOR_HZ * t)])

        v_out, info = stack.filter_velocity(tool, v_des, dt)
        tool = tool + v_out * dt

        gerr = goal - ghost
        gv = KP * gerr
        gsp = np.linalg.norm(gv)
        if gsp > MAX_SPEED:
            gv *= MAX_SPEED / gsp
        gv = gv + np.array([0.0, 0.0, TREMOR_AMP * np.sin(2 * np.pi * TREMOR_HZ * t)])
        ghost = ghost + gv * dt

        data.mocap_pos[tool_mocap] = tool
        data.mocap_pos[ghost_mocap] = ghost
        mujoco.mj_forward(model, data)

        log["t"].append(t)
        log["tool"].append(tool.copy())
        log["ghost"].append(ghost.copy())
        log["err_tool"].append(path.distance(tool))
        log["err_ghost"].append(path.distance(ghost))
        log["d_tool"].append(sdf.distance(tool))

        if np.linalg.norm(goal - tool) < GOAL_TOL:
            break

    for k in log:
        log[k] = np.array(log[k])
    return log


def verify(log, scene):
    mean_err_tool = float(log["err_tool"].mean())
    mean_err_ghost = float(log["err_ghost"].mean())
    min_d_tool = float(log["d_tool"].min())
    margin = scene["forbidden"].margin

    guided = mean_err_tool < 0.01 and mean_err_tool < 0.5 * mean_err_ghost
    safe = min_d_tool >= margin - 1e-4
    print(f"  mean path error  tool : {mean_err_tool*1000:7.2f} mm")
    print(f"  mean path error  ghost: {mean_err_ghost*1000:7.2f} mm")
    print(f"  tool min SDF distance : {min_d_tool*1000:7.2f} mm  (margin {margin*1000:.1f} mm)")
    print(f"  guidance keeps tool on path : {'PASS' if guided else 'FAIL'}")
    print(f"  safety keeps tool out anatomy: {'PASS' if safe else 'FAIL'}")
    return guided and safe


def make_plots(log, scene):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(LOGDIR, exist_ok=True)
    sdf, path = scene["sdf"], scene["path"]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    th = np.linspace(0, 2 * np.pi, 200)
    ax.fill(sdf.center[0] + sdf.radius * np.cos(th),
            sdf.center[1] + sdf.radius * np.sin(th), color="red", alpha=0.12)
    ax.plot(sdf.center[0] + sdf.radius * np.cos(th),
            sdf.center[1] + sdf.radius * np.sin(th), "r-", label="anatomy")
    ax.plot(path.pts[:, 0], path.pts[:, 1], "g--", lw=1.5, label="guidance path")
    ax.plot(log["ghost"][:, 0], log["ghost"][:, 1], color="tab:orange", label="ghost (raw)")
    ax.plot(log["tool"][:, 0], log["tool"][:, 1], color="tab:blue", label="tool (assisted)")
    ax.plot(scene["start"][0], scene["start"][1], "ko", label="start")
    ax.plot(scene["goal"][0], scene["goal"][1], "g*", ms=14, label="goal")
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Phase 1: guidance follows path, safety avoids anatomy")
    ax.legend(loc="upper right", fontsize=8); fig.tight_layout()
    p1 = os.path.join(LOGDIR, "phase1_trajectory.png")
    fig.savefig(p1, dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(log["t"], log["err_tool"] * 1000, color="tab:blue", label="tool (assisted)")
    ax.plot(log["t"], log["err_ghost"] * 1000, color="tab:orange", label="ghost (raw)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("distance from guidance path (mm)")
    ax.set_title("Phase 1: guidance fixture suppresses path deviation / tremor")
    ax.legend(); fig.tight_layout()
    p2 = os.path.join(LOGDIR, "phase1_path_error.png")
    fig.savefig(p2, dpi=120); plt.close(fig)
    return p1, p2


def run_viewer(scene):
    import glfw
    from .interactive import GlfwTeleop

    model, data = scene["model"], scene["data"]
    stack, forbidden = scene["stack"], scene["forbidden"]
    tool_mocap, ghost_mocap, tool_geom = scene["tool_mocap"], scene["ghost_mocap"], scene["tool_geom"]
    dt = 1.0 / 60.0
    speed = 0.05
    state = dict(tool=data.mocap_pos[tool_mocap].copy(),
                 ghost=data.mocap_pos[ghost_mocap].copy(), warn=0.0, off=0.0)

    def step_fn(app):
        v = np.zeros(3)
        if app.key(glfw.KEY_D): v[0] += speed
        if app.key(glfw.KEY_A): v[0] -= speed
        if app.key(glfw.KEY_W): v[1] += speed
        if app.key(glfw.KEY_S): v[1] -= speed
        if app.key(glfw.KEY_E): v[2] += speed
        if app.key(glfw.KEY_Q): v[2] -= speed
        v_out, info = stack.filter_velocity(state["tool"], v, dt)
        state["tool"] = state["tool"] + v_out * dt
        state["ghost"] = state["ghost"] + v * dt
        state["warn"] = info.get("warning", 0.0)
        state["off"] = info.get("path_offset", 0.0)
        data.mocap_pos[tool_mocap] = state["tool"]
        data.mocap_pos[ghost_mocap] = state["ghost"]
        # tool color encodes the warning cue (blue -> red) = vibration proxy
        w = state["warn"]
        model.geom_rgba[tool_geom] = [0.2 + 0.8 * w, 0.6 * (1 - w), 1.0 * (1 - w), 1.0]

    def status_fn():
        return (f"ActiveGuide  |  wall warning {state['warn']*100:3.0f}%  "
                f"|  off-path {state['off']*1000:4.1f} mm  |  WASD/QE move, ESC quit")

    GlfwTeleop(model, data, lookat=scene["sdf"].center).run(step_fn, status_fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    scene = build_scene()
    if args.view:
        run_viewer(scene)
        return

    log = simulate(scene)
    print("Phase 1 -- Guidance Fixture + Forbidden Region (composed)")
    ok = verify(log, scene)
    if not args.no_plot:
        p1, p2 = make_plots(log, scene)
        print(f"  wrote {p1}")
        print(f"  wrote {p2}")
    print("RESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
