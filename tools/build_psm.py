"""Build model/psm.xml (MJCF) from the official dVRK PSM description.

Run:  python tools/build_psm.py            # uses the vendored xacro + meshes
      python tools/build_psm.py --fetch    # re-download from WPI-AIM/dvrk_env

Why this script exists
----------------------
There is no MJCF model of the dVRK PSM -- not in mujoco_menagerie, not in
jhu-dvrk/dvrk_model (xacro + STL/DAE only), not in WPI-AIM/dvrk_env
(psm.urdf.xacro). Published dVRK simulators target CoppeliaSim/V-REP, Unity, or
Gazebo. So the model has to be derived, and derived *reproducibly* rather than
hand-edited, which is what this script is for.

Three problems have to be fixed on the way in, none of which MuJoCo does for you:

1. xacro. The source is a .urdf.xacro. It happens to use exactly one feature --
   a single <xacro:macro> with four plain ${} parameters -- so it is expanded
   here directly and no ROS toolchain is needed.

2. Zero inertias. Every <inertial> in the dVRK URDF carries an all-zero inertia
   tensor, which MuJoCo rejects outright ("mass and inertia of moving bodies
   must be larger than mjMINVAL"). An embedded <mujoco><compiler
   inertiafromgeom="true"> recomputes mass and inertia from mesh volume. Masses
   therefore do NOT match the real PSM; this model is for kinematic work.

3. *** Dropped <mimic> joints -- the dangerous one. *** The PSM's remote centre
   of motion is produced mechanically, by a parallelogram whose members are
   declared in URDF as <mimic> joints slaved to pitch_back_joint. MuJoCo's URDF
   importer ignores <mimic> SILENTLY: the model loads, reports 13 independent
   DOFs instead of 6+jaw, and the parallelogram falls apart. Measured over 75
   poses, the perpendicular distance from the official remote centre to the
   instrument shaft axis is 0.013 mm with the couplings applied and 90.1 mm
   without. A naive import gives you a plausible-looking, 90 mm-wrong robot.
   This script restores the couplings as MJCF <equality><joint> constraints.

Everything added beyond the raw import is listed in ADDITIONS below.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parent.parent
PSM_DIR = ROOT / "model" / "psm"
XACRO = PSM_DIR / "psm.urdf.xacro"
URDF = PSM_DIR / "psm.urdf"
OUT = ROOT / "model" / "psm.xml"

UPSTREAM = ("https://raw.githubusercontent.com/WPI-AIM/dvrk_env/master/"
            "dvrk_description/psm")
MESHES = [
    "base_link", "yaw_link", "pitch_back_link", "pitch_bottom_link",
    "pitch_end_link", "pitch_front_link", "pitch_top_link",
    "main_insertion_link", "tool_roll_link", "tool_pitch_link",
    "tool_yaw_link", "tool_gripper1_link", "tool_gripper2_link",
]

# --- facts taken from the official description ---------------------------------

# remote_center_joint: fixed, base_link -> remote_center_link, xyz="0 0.4864 0".
# Intuitive's own RCM location. MuJoCo fuses away fixed joints on massless
# bodies, so it is re-added here as a site.
RCM_IN_BASE = (0.0, 0.4864, 0.0)

# URDF <mimic multiplier="..."> table: child = multiplier * parent.
# pitch_bottom and pitch_end are on the chain to the tool and so are load
# bearing; pitch_top and pitch_front are leaf bodies and purely visual; the
# gripper pair mirrors the jaws.
MIMIC = {
    "pitch_bottom_joint": ("pitch_back_joint", -1.0),
    "pitch_end_joint": ("pitch_back_joint", +1.0),
    "pitch_top_joint": ("pitch_back_joint", -1.0),
    "pitch_front_joint": ("pitch_back_joint", +1.0),
    "tool_gripper1_joint": ("tool_gripper2_joint", -1.0),
}

# The six joints that actually position the instrument, in chain order.
ACTUATED = [
    "yaw_joint", "pitch_back_joint", "main_insertion_joint",
    "tool_roll_joint", "tool_pitch_joint", "tool_yaw_joint",
]

# Sites added for the fixture controller: (body, name, local xyz).
# Shaft samples run along main_insertion_link's local +z, which is the shaft
# axis (tool_roll_link attaches at z=0.4162). The distal span is what passes
# through the port, so that is where clearance matters.
ADDITIONS = [
    ("base_link", "remote_center", RCM_IN_BASE),
    ("main_insertion_link", "shaft_00", (0.0, 0.0, 0.26)),
    ("main_insertion_link", "shaft_25", (0.0, 0.0, 0.3039)),
    ("main_insertion_link", "shaft_50", (0.0, 0.0, 0.3478)),
    ("main_insertion_link", "shaft_75", (0.0, 0.0, 0.3917)),
    ("main_insertion_link", "shaft_end", (0.0, 0.0, 0.4162)),
    # Jaw tip with the gripper closed: tool_gripper1_link shares tool_yaw_link's
    # frame and its mesh reaches y=+0.0106.
    ("tool_yaw_link", "tool_tip", (0.0, 0.0106, 0.0)),
]


def fetch() -> None:
    PSM_DIR.mkdir(parents=True, exist_ok=True)
    (PSM_DIR / "meshes").mkdir(exist_ok=True)
    urllib.request.urlretrieve(f"{UPSTREAM}/psm.urdf.xacro", XACRO)
    for name in MESHES:
        urllib.request.urlretrieve(f"{UPSTREAM}/meshes/{name}.STL",
                                   PSM_DIR / "meshes" / f"{name}.STL")
    print(f"fetched xacro + {len(MESHES)} meshes from {UPSTREAM}")


def expand_xacro() -> None:
    """Expand the single <xacro:macro> into a plain URDF MuJoCo can compile."""
    src = XACRO.read_text()
    m = re.search(r"<xacro:macro[^>]*>(.*)</xacro:macro>", src, re.S)
    if not m:
        sys.exit("no <xacro:macro> found -- upstream format changed")
    body = m.group(1)
    for k, v in {
        "${prefix}": "",
        "${parent_link}": "world",
        "${xyz}": "0 0 0",
        "${rpy}": "0 0 0",
    }.items():
        body = body.replace(k, v)
    leftover = set(re.findall(r"\$\{[^}]*\}", body))
    if leftover:
        sys.exit(f"unsubstituted xacro expressions: {leftover}")

    # package:// -> bare filename; <compiler meshdir> supplies the directory.
    body = body.replace("package://dvrk_description/psm/meshes/", "")

    URDF.write_text(
        '<?xml version="1.0"?>\n'
        '<robot name="psm">\n'
        "  <!-- Generated by tools/build_psm.py from psm.urdf.xacro. Do not edit. -->\n"
        "  <mujoco>\n"
        '    <compiler meshdir="meshes" strippath="false" discardvisual="false"\n'
        '              inertiafromgeom="true" balanceinertia="true"/>\n'
        "  </mujoco>\n"
        '  <link name="world"/>\n'
        f"{body}\n"
        "</robot>\n"
    )
    print(f"expanded -> {URDF.relative_to(ROOT)}")


def _find_body(root: ET.Element, name: str) -> ET.Element:
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    sys.exit(f"body {name!r} not found in the compiled model")


def build() -> None:
    model = mujoco.MjModel.from_xml_path(str(URDF))
    raw = PSM_DIR / "psm_raw.xml"
    mujoco.mj_saveLastXML(str(raw), model)
    tree = ET.parse(raw)
    root = tree.getroot()
    raw.unlink()

    root.set("model", "dvrk_psm")

    # Meshes live under model/psm/meshes, but psm.xml sits in model/.
    comp = root.find("compiler")
    if comp is None:
        comp = ET.SubElement(root, "compiler")
    comp.set("meshdir", "psm/meshes/")
    comp.set("angle", "radian")

    # Kinematic study: velocity-commanded, so gravity would only fight the
    # integrator (same choice as scene.xml and scene_panda.xml).
    opt = root.find("option")
    if opt is None:
        opt = ET.Element("option")
        root.insert(0, opt)
    opt.set("timestep", "0.002")
    opt.set("gravity", "0 0 0")

    # 1. Restore the parallelogram and jaw couplings dropped by the URDF import.
    eq = ET.SubElement(root, "equality")
    for child, (parent, mult) in MIMIC.items():
        ET.SubElement(eq, "joint", {
            "name": f"mimic_{child}",
            "joint1": child,
            "joint2": parent,
            # child = polycoef[0] + polycoef[1]*parent + ...
            "polycoef": f"0 {mult:g} 0 0 0",
            "solimp": "0.9999 0.9999 0.001 0.5 2",
            "solref": "0.001 1",
        })

    # 2. Sites for the fixture controller.
    for body_name, site_name, xyz in ADDITIONS:
        ET.SubElement(_find_body(root, body_name), "site", {
            "name": site_name,
            "pos": " ".join(f"{v:g}" for v in xyz),
            "size": "0.0015",
            "rgba": ("1 0.85 0.2 0.9" if site_name == "remote_center"
                     else "0.2 0.6 1 0.9"),
        })

    # 3. Velocity actuators on the six instrument DOFs. rev_joint (the setup-arm
    #    mount) and the jaws are deliberately left unactuated: this model is a
    #    fixed-base PSM and the fixture work is about tool pose.
    act = ET.SubElement(root, "actuator")
    for j in ACTUATED:
        ET.SubElement(act, "velocity", {"name": f"v_{j}", "joint": j, "kv": "50"})

    # 4. Home pose: partially inserted, so the tool is through the port and the
    #    wrist has room to work.
    key = ET.SubElement(root, "keyframe")
    nq = model.nq
    qpos = ["0"] * nq
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
             for j in range(model.njnt)]
    qpos[names.index("main_insertion_joint")] = "0.12"
    ET.SubElement(key, "key", {"name": "home", "qpos": " ".join(qpos)})

    ET.indent(tree, space="  ")
    header = (
        "<!-- GENERATED by tools/build_psm.py from the official dVRK PSM\n"
        "     description (WPI-AIM/dvrk_env psm.urdf.xacro). DO NOT EDIT BY HAND;\n"
        "     re-run the builder instead.\n\n"
        "     Beyond the raw URDF import this file adds:\n"
        "       * <equality> joint couplings restoring the parallelogram that the\n"
        "         URDF declares via <mimic> and MuJoCo's importer drops silently.\n"
        "         Without them the remote centre wanders by up to 90 mm.\n"
        "       * a remote_center site at base_link + (0, 0.4864, 0), the RCM the\n"
        "         official description defines via remote_center_joint.\n"
        "       * shaft_* and tool_tip sites for the fixture constraints.\n"
        "       * velocity actuators on the six instrument joints.\n\n"
        "     Mass/inertia are recomputed from mesh volume (the source URDF has\n"
        "     all-zero inertia tensors), so they are NOT the real PSM's. This\n"
        "     model is intended for kinematic work. -->\n"
    )
    OUT.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        + header
        + ET.tostring(root, encoding="unicode")
        + "\n"
    )
    print(f"wrote {OUT.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true",
                    help="re-download the xacro and meshes from upstream")
    args = ap.parse_args()
    if args.fetch:
        fetch()
    expand_xacro()
    build()

    # Compile the result and report what came out.
    m = mujoco.MjModel.from_xml_path(str(OUT))
    print(f"\ncompiled: nq={m.nq} nv={m.nv} neq={m.neq} nu={m.nu} "
          f"nsite={m.nsite} nmesh={m.nmesh}")
    if m.neq != len(MIMIC):
        sys.exit(f"expected {len(MIMIC)} equality constraints, got {m.neq}")
    print("equality couplings restored OK")


if __name__ == "__main__":
    main()
