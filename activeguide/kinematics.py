"""Robot kinematics: the bridge between MuJoCo and the fixture QP.

Phases 0-1 constrained a free-floating point in Cartesian space. To enforce a
fixture on an actual manipulator we need three things this module supplies:

  1. Where a site is now                  -> site_position
  2. How joint velocity moves it          -> site_jacobian  (the chain-rule term
                                             that turns an SDF into a linear
                                             constraint on q-dot)
  3. What the arm itself forbids          -> joint limits, and a distance-to-
                                             singularity measure

Deliberately embodiment-agnostic: nothing here knows about the Panda. Swapping
in the dVRK PSM means passing a different MJCF, not editing this file.
"""

from __future__ import annotations

import numpy as np
import mujoco


class RobotKinematics:
    """Thin, allocation-conscious wrapper over one MuJoCo model/data pair.

    Jacobian buffers are allocated once and reused, because the QP controller
    calls into here every control step (target: 500 Hz-1 kHz).
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData | None = None):
        self.model = model
        self.data = mujoco.MjData(model) if data is None else data

        # This module assumes every DOF is a 1-DOF joint (hinge/slide), so
        # qpos and qvel index alike. That holds for serial manipulators; it
        # would not for free or ball joints, hence the explicit check.
        if model.nq != model.nv:
            raise ValueError(
                f"RobotKinematics needs nq == nv (got nq={model.nq}, nv={model.nv}); "
                "free/ball joints are not supported."
            )

        self.nv = int(model.nv)
        self._Jp = np.zeros((3, self.nv))
        self._Jr = np.zeros((3, self.nv))
        self._site_ids: dict[str, int] = {}

    # --- lookup -----------------------------------------------------------
    def site_id(self, name: str) -> int:
        """Cached name -> id. Raises on a typo'd site rather than returning -1."""
        sid = self._site_ids.get(name)
        if sid is None:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid < 0:
                raise KeyError(f"no site named {name!r} in this model")
            self._site_ids[name] = sid
        return sid

    # --- state ------------------------------------------------------------
    @property
    def q(self) -> np.ndarray:
        return self.data.qpos[: self.nv].copy()

    def set_q(self, q) -> None:
        """Set the configuration and refresh derived quantities."""
        self.data.qpos[: self.nv] = np.asarray(q, dtype=float)
        self.forward()

    def forward(self) -> None:
        """Refresh site positions and Jacobians for the current qpos.

        ``mj_kinematics`` + ``mj_comPos`` is all ``mj_jacSite`` needs, and is
        markedly cheaper than a full ``mj_forward`` in the control loop.
        """
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def site_position(self, name: str) -> np.ndarray:
        return self.data.site_xpos[self.site_id(name)].copy()

    def site_jacobian(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (J_pos, J_rot), each (3, nv), for ``name``.

        Copies out of the reusable buffers so callers can hold onto the result
        without it changing under them on the next call.
        """
        mujoco.mj_jacSite(self.model, self.data, self._Jp, self._Jr, self.site_id(name))
        return self._Jp.copy(), self._Jr.copy()

    # --- limits -----------------------------------------------------------
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """(q_min, q_max) for the DOFs, +/-inf where a joint is unlimited."""
        lo = np.full(self.nv, -np.inf)
        hi = np.full(self.nv, np.inf)
        for j in range(self.model.njnt):
            dof = self.model.jnt_dofadr[j]
            if dof < 0 or dof >= self.nv:
                continue
            if self.model.jnt_limited[j]:
                lo[dof], hi[dof] = self.model.jnt_range[j]
        return lo, hi

    def velocity_limits(self, default: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
        """Symmetric joint-velocity box. MJCF has no standard velocity-limit
        field, so this is a controller parameter, not a model property."""
        v = np.full(self.nv, float(default))
        return -v, v

    # --- conditioning ------------------------------------------------------
    @staticmethod
    def manipulability(J: np.ndarray) -> float:
        """Smallest singular value of ``J``.

        Reported rather than yoked to a constraint: it is the honest measure of
        how close enforcement has driven the arm to a singularity, and one of
        the metrics a Cartesian point-projection controller cannot see at all.
        """
        if J.size == 0:
            return 0.0
        return float(np.linalg.svd(J, compute_uv=False).min())
