"""Bake protected anatomy into a signed distance field for the fixture stack.

Run:  python tools/build_anatomy.py                      # procedural phantom
      python tools/build_anatomy.py --mesh liver.stl --scale 0.001
      python tools/build_anatomy.py --spacing 0.001 --offset 0.003

Writes model/anatomy_sdf.npz (the field) and model/anatomy.stl (the surface the
field was baked from, so the demo can render it and audit against it).

Why a build step
----------------
Baking is the expensive part -- seconds, against microseconds for a query -- and
it depends only on the anatomy, not on the robot, the task, or the run. Doing it
once and caching it keeps the demo honest about control-loop cost: the numbers
it reports are query cost, and nothing in the loop is secretly re-deriving
geometry. It is the same split as tools/build_psm.py: derive the model
reproducibly offline, keep the online path thin.

Choosing the parameters
-----------------------
``--spacing`` is the accuracy/size trade. The bake reports the measured
over-estimate of distance, which is the error that matters, since claiming
clearance that is not there is what lets a tool touch the organ. The fixture
margin must exceed it -- the script checks this and says so.

``--offset`` dilates the forbidden region to cover *segmentation* uncertainty
(how well the contour matches the organ). It is deliberately separate from the
fixture margin, which covers *tool* radius. The two add.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from activeguide.anatomy import bake_mesh, load_mesh, make_phantom, save_sdf

ROOT = Path(__file__).resolve().parent.parent
OUT_SDF = ROOT / "model" / "anatomy_sdf.npz"
OUT_MESH = ROOT / "model" / "anatomy.stl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", help="surface mesh to bake (STL/OBJ/PLY). "
                                   "Default: the procedural phantom.")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="factor converting the mesh to metres "
                         "(segmentation exports are usually mm: 0.001)")
    ap.add_argument("--spacing", type=float, default=0.0015, help="cell size (m)")
    ap.add_argument("--pad", type=float, default=0.025,
                    help="empty margin around the mesh bounds (m)")
    ap.add_argument("--offset", type=float, default=0.0,
                    help="safety dilation of the forbidden region (m)")
    ap.add_argument("--margin", type=float, default=0.004,
                    help="fixture margin the field will be used with; checked "
                         "against the measured bake error (m)")
    ap.add_argument("--radius", type=float, default=0.030,
                    help="phantom radius when --mesh is not given (m)")
    ap.add_argument("--seed", type=int, default=0, help="phantom seed")
    args = ap.parse_args()

    if args.mesh:
        mesh = load_mesh(args.mesh, scale=args.scale)
        source = f"{args.mesh} (scale {args.scale})"
    else:
        mesh = make_phantom(radius=args.radius, seed=args.seed)
        source = f"procedural phantom (radius {args.radius} m, seed {args.seed})"

    print(f"source     {source}")
    print(f"           {len(mesh.faces)} faces, watertight={mesh.is_watertight}, "
          f"volume={mesh.volume * 1e6:.1f} cm^3")
    if not mesh.is_watertight:
        print("  ! not watertight -- inside/outside is only as good as the "
              "winding number can infer; check the segmentation")

    field, info = bake_mesh(mesh, spacing=args.spacing, pad=args.pad,
                            offset=args.offset, report=True)

    print(f"grid       {info['shape']} = {info['cells']:,} cells "
          f"@ {args.spacing * 1e3:.2f} mm, baked in {info['bake_s']:.1f} s")
    print(f"accuracy   max over-estimate {info['max_overestimate'] * 1e3:.3f} mm, "
          f"rms {info['rms_error'] * 1e3:.3f} mm")
    print(f"sign       {info['sign_mismatch']} genuine mismatches vs the winding "
          f"number ({info['sign_mismatch_in_noise']} within interpolation noise)")

    if info["sign_mismatch"]:
        sys.exit("inside/outside disagrees between flood fill and winding "
                 "number: refusing to write a field whose sign is unresolved")

    budget = args.margin - args.offset
    if info["max_overestimate"] >= budget:
        print(f"\n  ! over-estimate {info['max_overestimate'] * 1e3:.3f} mm does not "
              f"fit the {budget * 1e3:.3f} mm left by margin-offset.\n"
              f"    The wall could be crossed by that much. Reduce --spacing.")
    else:
        print(f"margin     {args.margin * 1e3:.1f} mm covers dilation "
              f"{args.offset * 1e3:.1f} + error "
              f"{info['max_overestimate'] * 1e3:.3f} mm -- OK")

    OUT_SDF.parent.mkdir(parents=True, exist_ok=True)
    save_sdf(field, str(OUT_SDF))
    mesh.export(str(OUT_MESH))
    size = OUT_SDF.stat().st_size / 1e6
    print(f"\nwrote {OUT_SDF.relative_to(ROOT)} ({size:.2f} MB) "
          f"and {OUT_MESH.relative_to(ROOT)}")
    print(f"     bounds {np.round(field.lower, 4).tolist()} .. "
          f"{np.round(field.upper, 4).tolist()}")


if __name__ == "__main__":
    main()
