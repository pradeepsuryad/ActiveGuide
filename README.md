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

## Roadmap
- **Phase 1** Guidance fixtures + interactive teleop. *(done; Quest viz pending)*
- **Phase 2** Swappable feedback: dVRK MTM force vs. Quest vibration.
- **Phase 3** Build walls from segmented anatomy (CT/mesh → offset → SDF).
- **Phase 4** AI task recognition selects which fixtures are active.
- **Phase 5** "Second-person" features (camera / retraction / next-step).
- **Phase 6** User study: solo+system vs. solo vs. two-person.

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
# tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Key references
- Bowyer, Davies & Rodriguez y Baena, *Active Constraints / Virtual Fixtures:
  A Survey*, IEEE T-RO 2014.
- Yang et al., *Medical robotics—Regulatory, ethical, and legal considerations
  for increasing levels of autonomy*, Science Robotics 2017.
- AMBF (JHU/WPI), dVRK/CRTK; datasets: JIGSAWS, Cholec80.
