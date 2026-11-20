"""Pick a UR5e joint at its displayed hinge and drag its angle in the MuJoCo viewer."""

import math

import glfw
import mujoco
import numpy as np

from .teleop_types import JointDrag


_RADII = np.array([0.090, 0.088, 0.076, 0.062, 0.057, 0.056])
_PHASE = np.linspace(0., 2. * np.pi, 28, endpoint=False)


def _ring_points(anchors, axes):
    # Complete each world-space hinge axis with two unit vectors in its ring plane.
    reference = np.tile([1., 0., 0.], (len(axes), 1))
    reference[np.abs(axes[:, 0]) > 0.9] = (0., 1., 0.)
    u = np.cross(reference, axes)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(axes, u)
    circle = np.cos(_PHASE)[:, None] * u[:, None] + np.sin(_PHASE)[:, None] * v[:, None]
    return anchors[:, None] + _RADII[:, None, None] * circle


def _nearest_ring(pixels, mouse, depth, tolerance, visible=None):
    """Find the closest projected *segment*, using the front ring to break ties."""
    edges = np.roll(pixels, -1, axis=1) - pixels
    offset = mouse - pixels
    fraction = np.sum(offset * edges, axis=2) / np.maximum(np.sum(edges * edges, axis=2), 1e-12)
    distance = np.linalg.norm(offset - np.clip(fraction, 0., 1.)[:, :, None] * edges, axis=2)
    distance = np.where(np.isfinite(distance), distance, np.inf)
    # Multiple segments can share one pixel when a ring is nearly edge-on.
    joints, segments = np.where(distance <= tolerance)
    for candidate in np.lexsort((depth[joints], distance[joints, segments])):
        joint, segment = int(joints[candidate]), int(segments[candidate])
        t = float(np.clip(fraction[joint, segment], 0., 1.))
        if visible is None or visible(joint, segment, t):
            return joint
    return None


class JointRings:
    """Call update with the displayed MjData before each draw; all output is a target."""

    def __init__(self, sim):
        # load_simulation has selected the same vendored camera math as the SG2 viewer.
        from ri_motion_v5_package.mujoco_sim import InteractiveMarker

        self.sim = sim
        self.window = sim.viewer.window
        self.camera = InteractiveMarker(env=sim, control_mode="R", mouse_view_offset=(0., 0.))
        self.ids = sim.model.actuator_trnid[:, 0].copy()
        self.q_indices = sim.model.jnt_qposadr[self.ids]
        self.limits = sim.model.actuator_ctrlrange
        self.epoch = 0
        self.serial = 0
        self._ready = False
        self._active = self._hover = None
        self._points = self._pixels = None
        self._scale = np.ones(2)
        self._previous_button = glfw.set_mouse_button_callback(self.window, self._button)
        self._previous_focus = glfw.set_window_focus_callback(self.window, self._focus)
        self._previous_enter = glfw.set_cursor_enter_callback(self.window, self._enter)

    def _cancel(self):
        self._active = self._hover = None
        self._ready = False

    def _focus(self, window, focused):
        if not focused:
            self._cancel()
        if self._previous_focus:
            self._previous_focus(window, focused)

    def _enter(self, window, entered):
        if not entered:
            self._cancel()
        if self._previous_enter:
            self._previous_enter(window, entered)

    def _cursor_available(self):
        x, y = glfw.get_cursor_pos(self.window)
        width, height = glfw.get_window_size(self.window)
        return (glfw.get_window_attrib(self.window, glfw.FOCUSED)
                and glfw.get_window_attrib(self.window, glfw.HOVERED)
                and 0 <= x < width and 0 <= y < height)

    def _pick(self, mouse):
        if self._pixels is None:
            return None
        return _nearest_ring(self._pixels, mouse, self._depth, 7. * self._scale.mean(), self._visible)

    def _visible(self, joint, segment, fraction):
        """Leave a click on solid robot geometry to the viewer if the ring is behind it."""
        ring = self._points[joint]
        point = ring[segment] + fraction * (ring[(segment + 1) % len(ring)] - ring[segment])
        direction = point - self._eye
        distance = np.linalg.norm(direction)
        hit = mujoco.mj_ray(self.sim.model, self._data, self._eye, direction / distance,
                            self.sim.viewer.vopt.geomgroup, 1, -1, None)
        return hit < 0 or hit >= distance - 0.003

    def _button(self, window, button, action, mods):
        # Claim the press before the viewer's camera callback. A camera drag that
        # subsequently crosses a ring must stay a camera drag for its entire life.
        if self.epoch and button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                eligible = self._ready and not mods and self._cursor_available()
                self._ready = False
                mouse = np.asarray(glfw.get_cursor_pos(window)) * self._scale
                joint = self._pick(mouse) if eligible else None
                if joint is not None:
                    self._active = joint
                    self.serial += 1
                    self._angle = self._q[joint]
                    self._center = self._centers[joint].copy()
                    self._sign = self._signs[joint]
                    self._polar = self._cursor_angle(mouse)
                    return
            elif action == glfw.RELEASE:
                owned = self._active is not None
                self._active = None
                self._ready = bool(self._cursor_available())
                if owned:
                    return
        if self._previous_button:
            self._previous_button(window, button, action, mods)

    def _cursor_angle(self, mouse):
        dx, dy = mouse - self._center
        # Passing exactly through the pivot does not define a turn.
        return math.atan2(dy, dx) if math.hypot(dx, dy) > 2. * self._scale.mean() else None

    def _project(self, data):
        width, height = glfw.get_window_size(self.window)
        if min(width, height, *glfw.get_framebuffer_size(self.window)) <= 0:
            return False
        camera = self.camera._begin_frame()
        if camera is None:
            return False
        # In mono the renderer averages the two GL eyes. The generic SG2 marker
        # projects through the left eye and applies a fixed mouse offset instead.
        left, right = self.sim.viewer.scn.camera
        camera["pos"] = (np.asarray(left.pos) + right.pos) * 0.5
        camera["center"] = (left.frustum_center + right.frustum_center) * 0.5
        self._eye, self._data = camera["pos"], data
        self._scale = np.array([camera["W"] / width, camera["H"] / height])
        centers, axes = data.xanchor[self.ids], data.xaxis[self.ids]
        # J6's mathematical pivot is inside the wrist housing. A small offset
        # along that same rotation axis exposes the ring at the wrist seam.
        centers[5] -= 0.025 * axes[5]
        self._points = _ring_points(centers, axes)
        projected = self.camera._project_many(self._points.reshape(-1, 3), camera)
        pixels = [p if p is not None else (np.nan, np.nan) for p in projected]
        self._pixels = np.asarray(pixels).reshape(len(self.ids), -1, 2)
        centers_px = self.camera._project_many(centers, camera)
        self._centers = np.asarray([p if p is not None else (np.nan, np.nan) for p in centers_px])
        self._pixels[~np.isfinite(self._centers).all(axis=1)] = np.nan
        self._depth = (centers - camera["pos"]) @ camera["fwd"]
        self._signs = np.where(axes @ camera["fwd"] >= 0., 1., -1.)
        self._q = data.qpos[self.q_indices].copy()
        return True

    def update(self, data, epoch):
        """Return a held angle, a confirmed release, or None when the mouse is unavailable."""
        if self.epoch != epoch:
            self.epoch = epoch
            self._cancel()
        if not epoch or not self._project(data) or not self._cursor_available():
            self._cancel()
            return None
        mouse = np.asarray(glfw.get_cursor_pos(self.window)) * self._scale
        if glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) != glfw.PRESS:
            self._active = None
            self._ready = True
            self._hover = self._pick(mouse)
            return JointDrag(self.serial, None)
        joint = self._active
        if joint is None:
            self._hover = None
            return None
        polar = self._cursor_angle(mouse)
        if polar is not None and self._polar is not None:
            delta = math.remainder(polar - self._polar, math.tau) * self._sign
            self._angle = float(np.clip(self._angle + delta, *self.limits[joint]))
        self._polar = polar
        return JointDrag(self.serial, joint, self._angle)

    def draw(self):
        """Queue all visible rings and identify the joint under the mouse."""
        if not self.epoch or self._points is None:
            return
        selected = self._active if self._active is not None else self._hover
        for joint, ring in enumerate(self._points):
            color = (1., 0.78, 0.08, 1.) if joint == selected else (0.10, 0.70, 0.90, 0.95)
            for start, end in zip(ring, np.roll(ring, -1, axis=0)):
                self.sim.plot_cylinder_fr2to(start, end, r=0.003, rgba=color)
        if selected is not None:
            name = self.sim.model.joint(int(self.ids[selected])).name
            angle = self._angle if self._active is not None else self._q[selected]
            self.sim.viewer_text_overlay(f"J{selected + 1} · {name}", f"{math.degrees(angle):+.1f} deg",
                                         loc="top left")
