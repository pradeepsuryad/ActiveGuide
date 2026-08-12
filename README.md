# ActiveGuide

Software/VR-assisted **virtual fixtures (active constraints)** for surgical
robot control. The system identifies protected anatomy, builds **virtual walls**
(forbidden-region fixtures) and **guidance paths** (guidance fixtures) from it,
enforces them on the robot, and surfaces them to the surgeon through a VR UI plus
force or vibrotactile feedback.

North star (Levels of Autonomy 1–2): give a *solo* surgeon enough active
assistance to do what normally needs a second pair of hands.

## Status — Phases 0–1 (done)
**Phase 0 — Forbidden-Region Virtual Fixture (virtual wall):**
- `activeguide/sdf.py` — signed-distance primitives (sphere/plane/box/union).
- `activeguide/fixtures.py` — `ForbiddenRegionFixture`: dt-aware predictive
  velocity projection (robot-side enforcement) + spring-damper feedback force
  (MTM) + 0..1 warning level (Quest vibration / sensory substitution).
- `model/scene.xml` — tool, protected sphere, goal.
- `activeguide/demo_phase0.py` — drives the tool at the anatomy; the fixture
  routes it around without ever penetrating.

**Phase 1 — Guidance fixture + interactive teleop:**
- `activeguide/paths.py` — `Polyline` (closest point + tangent).
- `activeguide/fixtures.py` — `GuidanceFixture` (assist along a path,
  anisotropic strength) and `FixtureStack` (guidance first, safety clamp last).
- `activeguide/interactive.py` — GLFW teleop (hold-to-move keys, mouse camera);
  the seam where Quest master input plugs in later.
- `activeguide/demo_phase1.py` — guidance routes the tool along an arc over the
  anatomy; suppresses operator tremor ~135× (0.22 mm vs 29.8 mm path error).

**Phase 2 — Joint-space enforcement on a real manipulator (in progress):**
Phases 0–1 constrained a free-floating *point* in Cartesian space (the tool was
a MuJoCo mocap body). Phase 2 puts an actual 7-DOF arm in the loop and enforces
every fixture as a **Vector Field Inequality** inside one QP per control step:

```
min_q̇   ‖J_p q̇ − v_des‖² + λ‖q̇‖²
s.t.    −J_dᵢ q̇ ≤ ηᵢ dᵢ     walls, shaft clearance, joint limits, RCM
        q̇_min ≤ q̇ ≤ q̇_max
```

so constraints are satisfied *simultaneously* rather than by last-writer-wins
ordering. `sdf.py` needed **no changes** — `sdf.normal()` is already `∇d`, and
the chain rule `J_d = ∇dᵀ J_p` turns any existing SDF primitive into a linear
constraint on joint velocity.

- `activeguide/kinematics.py` — `mj_jacSite` wrapper, joint limits, `σ_min(J)`.
- `activeguide/constraints.py` — `SdfConstraint`, `ShaftClearanceConstraint`
  (samples the whole shaft, not just the tip), `JointLimitConstraint`,
  `RcmConstraint` (trocar pivot, ported from `panda_ik_cpp`).
- `activeguide/qp.py` — `QpController`; hard solve first, penalized slack only
  on genuine infeasibility, which is counted and reported.
- `model/scene_panda.xml` — 7-DOF Panda + 0.42 m laparoscopic shaft, trocar,
  anatomy, goal.

Measured on the Panda reach task (2 ms control step): **zero** wall and
joint-limit violations, RCM deviation **≤ 500 µm** (the requested radial
tolerance, hit exactly), 0% infeasible solves, p50/p99 solve time **355/1317 µs**.

The new layer is cross-validated against Phase 0: with `η = 1/dt` the VFI bound
reduces algebraically to the old dt-aware clamp, and the two agree to **1e-8**
in both the free and actively-clamped regimes (`tests/test_qp.py`).

## Roadmap
- **Phase 1** Guidance fixtures + interactive teleop. *(done; Quest viz pending)*
- **Phase 2** Joint-space QP enforcement + RCM. *(Panda done; dVRK PSM next)*
- **Phase 2b** Swappable feedback: dVRK MTM force vs. Quest vibration.
- **Phase 3** Build walls from segmented anatomy (CT/mesh → offset → SDF).
- **Phase 4** AI task recognition selects which fixtures are active.
- **Phase 5** "Second-person" features (camera / retraction / next-step).
- **Phase 6** User study: solo+system vs. solo vs. two-person.

## Stack & requirements

| | |
|---|---|
| **Language** | Python 3.11+ (developed on 3.14) |
| **Physics** | [MuJoCo](https://mujoco.org) ≥ 3.9 |
| **Viewer / input** | `glfw` (interactive teleop window, mouse camera) |
| **Numerics** | `numpy`, `scipy` |
| **QP controller** | `qpsolvers` ≥ 4.13 with `daqp` (default) / `quadprog` |
| **Geometry** | `trimesh` (Phase 3 — anatomy meshes → SDF) |
| **Plots** | `matplotlib` (writes to `logs/`) |
| **Tests** | `unittest` (stdlib) |
| **Target hardware** | dVRK (MTM force feedback) + Meta Quest (vibrotactile) — Phase 2+ |

Runs headless for verification and plotting; `--view` opens the GLFW window and
needs a GPU/display. `logs/` output is regenerable and not tracked.

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run
```powershell
# Phase 0 — virtual wall (headless verify + plots, then live)
.\.venv\Scripts\python.exe -m activeguide.demo_phase0
.\.venv\Scripts\python.exe -m activeguide.demo_phase0 --view
# Phase 1 — guidance + wall (headless, then interactive teleop)
.\.venv\Scripts\python.exe -m activeguide.demo_phase1
.\.venv\Scripts\python.exe -m activeguide.demo_phase1 --view   # WASD/QE move, mouse camera
# tests (64: 24 from Phases 0-1, 40 for the Phase 2 QP layer)
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Key references
- Bowyer, Davies & Rodriguez y Baena, *Active Constraints / Virtual Fixtures:
  A Survey*, IEEE T-RO 2014.
- Yang et al., *Medical robotics—Regulatory, ethical, and legal considerations
  for increasing levels of autonomy*, Science Robotics 2017.
- AMBF (JHU/WPI), dVRK/CRTK; datasets: JIGSAWS, Cholec80.
