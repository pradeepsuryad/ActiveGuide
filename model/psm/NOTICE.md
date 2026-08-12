# dVRK PSM model — provenance and licensing

The PSM model used by ActiveGuide is **derived, not vendored**. This directory is
empty in a fresh clone; run the builder to populate it:

```bash
python tools/build_psm.py --fetch
```

That downloads the upstream description and produces `model/psm.xml`.

## Upstream source

| | |
|---|---|
| **Repository** | [WPI-AIM/dvrk_env](https://github.com/WPI-AIM/dvrk_env) |
| **Files used** | `dvrk_description/psm/psm.urdf.xacro` + 13 `.STL` meshes |
| **Authors** | Ankur Agrawal, Radian Gondokaryono (Worcester Polytechnic Institute) |
| **Declared license** | `BSD`, per `dvrk_description/package.xml` |

## Why nothing here is committed

The upstream repository has **no `LICENSE` file** — the GitHub API reports
`license: null` — even though the ROS package manifest declares BSD. Rather than
redistribute assets whose license terms are only implied, the builder fetches
them from upstream at setup time. Anyone running it obtains the files under
whatever terms the authors intend, directly from the authors.

If you later confirm the BSD grant (or get written permission), committing the
meshes is a one-line change: drop the `model/psm/` entries from `.gitignore`.

## There is no official dVRK MJCF

Checked, as of 2026-08-12:

- **mujoco_menagerie** — 80+ models, none surgical, no dVRK/PSM.
- **jhu-dvrk/dvrk_model** (official JHU) — xacro + STL/DAE only, zero MJCF, zero URDF.
- **WPI-AIM/dvrk_env** — `psm.urdf.xacro`, no MJCF.
- Published dVRK simulators target CoppeliaSim/V-REP, Unity/PhysX (CRESSim),
  Gazebo/RViz, Isaac Sim (ORBIT-Surgical, USD), and PyBullet (SurRoL, URDF).

Hence `tools/build_psm.py`.

## What the builder changes, and why it must

Three things break on a plain MuJoCo URDF import:

1. **xacro** — the source is `.urdf.xacro`. It uses exactly one feature (a single
   `<xacro:macro>` with four `${}` parameters), so the builder expands it
   directly; no ROS toolchain is required.

2. **Zero inertias** — every `<inertial>` carries an all-zero inertia tensor,
   which MuJoCo rejects (*"mass and inertia of moving bodies must be larger than
   mjMINVAL"*). `inertiafromgeom="true"` recomputes them from mesh volume.
   **Masses therefore do not match the real PSM**; this model is for kinematic
   work, not dynamics.

3. **Dropped `<mimic>` joints — the dangerous one.** The PSM's remote centre of
   motion is *mechanical*: a parallelogram whose members the URDF declares as
   `<mimic>` joints slaved to `pitch_back_joint`. **MuJoCo's URDF importer
   ignores `<mimic>` silently.** The model loads without warning, reports 13
   independent DOFs instead of 6 + jaw, and the parallelogram comes apart.

   Measured over 75 poses (yaw × pitch × insertion), perpendicular distance from
   the official remote centre to the instrument shaft axis:

   | | worst error |
   |---|---|
   | couplings applied | **0.013 mm** |
   | couplings dropped | **90.1 mm** |

   A naive import gives a plausible-looking robot that is 90 mm wrong. The
   builder restores the couplings as `<equality><joint>` constraints, and
   `tests/test_psm.py::TestParallelogramIsLoadBearing` pins *both* directions so
   the restoration cannot be silently removed.

## Additions beyond the raw import

- `remote_center` site at `base_link + (0, 0.4864, 0)` — the RCM the official
  description defines via `remote_center_joint`, which MuJoCo fuses away because
  it is a fixed joint on a massless body.
- `shaft_00 … shaft_end` sites along the instrument shaft, for shaft-clearance
  and RCM constraints.
- `tool_tip` site at the closed-jaw tip.
- Velocity actuators on the six instrument joints. `rev_joint` (the setup-arm
  mount) and the jaws are left unactuated — this is a fixed-base PSM.
