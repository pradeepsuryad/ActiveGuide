"""Print the PSM's worst remote-centre error with and without the parallelogram couplings.

    python tools/report_psm_rcm.py

Same 75 poses (yaw x pitch x insertion) and the same error as
tests/test_psm.py::TestParallelogramIsLoadBearing: the perpendicular distance
from the official remote centre to the instrument shaft axis. The tests only
bound these two numbers; this prints them. Needs model/psm.xml
(python tools/build_psm.py --fetch).
"""
import os

import mujoco
import numpy as np

from activeguide.kinematics import RobotKinematics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "model", "psm.xml")

# URDF <mimic> couplings: child = multiplier * pitch_back_joint.
MIMIC = {
    "pitch_bottom_joint": ("pitch_back_joint", -1.0),
    "pitch_end_joint": ("pitch_back_joint", +1.0),
    "pitch_top_joint": ("pitch_back_joint", -1.0),
    "pitch_front_joint": ("pitch_back_joint", +1.0),
}
POSES = [(y, p, i)
         for y in (-0.8, -0.4, 0.0, 0.4, 0.8)
         for p in (-0.6, -0.3, 0.0, 0.3, 0.6)
         for i in (0.0, 0.12, 0.24)]


def worst_rcm_error(kin, model, coupled):
    def jid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

    worst = 0.0
    for yaw, pitch, insertion in POSES:
        q = np.zeros(kin.nv)
        q[jid("yaw_joint")] = yaw
        q[jid("pitch_back_joint")] = pitch
        q[jid("main_insertion_joint")] = insertion
        if coupled:
            for child, (driver, mult) in MIMIC.items():
                q[jid(child)] = mult * q[jid(driver)]
        kin.set_q(q)
        rcm = kin.site_position("remote_center")
        a, b = kin.site_position("shaft_00"), kin.site_position("shaft_end")
        u = (b - a) / np.linalg.norm(b - a)
        w = rcm - a
        worst = max(worst, float(np.linalg.norm(w - np.dot(w, u) * u)))
    return worst


def main():
    model = mujoco.MjModel.from_xml_path(MODEL)
    kin = RobotKinematics(model)
    mujoco.mj_resetDataKeyframe(model, kin.data, 0)
    kin.forward()
    print(f"PSM remote-centre error, worst over {len(POSES)} poses (yaw x pitch x insertion):")
    print(f"  couplings applied:            {worst_rcm_error(kin, model, True) * 1e3:.3f} mm")
    print(f"  couplings dropped (naive import): {worst_rcm_error(kin, model, False) * 1e3:.1f} mm")


if __name__ == "__main__":
    main()
