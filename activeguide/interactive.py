"""Minimal GLFW teleoperation viewer.

Polls keyboard state every frame (true hold-to-move, unlike the passive viewer's
press-only callback), renders the MuJoCo scene, and gives mouse camera control.
The display window is driven by the user; the control logic lives in the
``step_fn`` you pass to ``run``. This is the seam the Quest master input will
plug into later -- swap ``key`` polling for controller poses.
"""

from __future__ import annotations

import glfw
import mujoco
import numpy as np


class GlfwTeleop:
    def __init__(self, model, data, lookat=(0.0, 0.0, 0.10), title="ActiveGuide"):
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")
        self.window = glfw.create_window(1200, 900, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create GLFW window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.model = model
        self.data = data
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.opt)
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.azimuth = 90.0
        self.cam.elevation = -25.0
        self.cam.distance = 0.55
        self.cam.lookat[:] = lookat

        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

        self._lx = self._ly = 0.0
        self._left = self._right = False
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_mouse_button_callback(self.window, self._on_button)
        glfw.set_scroll_callback(self.window, self._on_scroll)

    # --- input ------------------------------------------------------------
    def key(self, glfw_key) -> bool:
        return glfw.get_key(self.window, glfw_key) == glfw.PRESS

    def _on_button(self, w, button, action, mods):
        self._left = glfw.get_mouse_button(w, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        self._right = glfw.get_mouse_button(w, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        self._lx, self._ly = glfw.get_cursor_pos(w)

    def _on_cursor(self, w, x, y):
        dx, dy = x - self._lx, y - self._ly
        self._lx, self._ly = x, y
        if not (self._left or self._right):
            return
        _, h = glfw.get_window_size(w)
        action = (mujoco.mjtMouse.mjMOUSE_ROTATE_V if self._left
                  else mujoco.mjtMouse.mjMOUSE_MOVE_V)
        mujoco.mjv_moveCamera(self.model, action, dx / h, dy / h, self.scene, self.cam)

    def _on_scroll(self, w, xoff, yoff):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                              0.0, -0.05 * yoff, self.scene, self.cam)

    # --- loop -------------------------------------------------------------
    def run(self, step_fn, status_fn=None):
        while not glfw.window_should_close(self.window):
            if self.key(glfw.KEY_ESCAPE):
                break
            step_fn(self)
            mujoco.mj_forward(self.model, self.data)

            w, h = glfw.get_framebuffer_size(self.window)
            viewport = mujoco.MjrRect(0, 0, w, h)
            mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                                   mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)
            mujoco.mjr_render(viewport, self.scene, self.ctx)
            if status_fn is not None:
                glfw.set_window_title(self.window, status_fn())
            glfw.swap_buffers(self.window)
            glfw.poll_events()
        glfw.terminate()
