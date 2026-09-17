"""
interactive_marker.py
"""
import glfw
import mujoco
import numpy as np

from ri_motion_v5_package.kinematics import np_uv, pr2t, rodrigues, t2p, t2r
from ri_motion_v5_package.utility import normalize_angle


class InteractiveMarker:
    """
    Interactive marker for MuJoCo viewer that supports translation and rotation
    editing of a 4x4 pose matrix T.

    Usage:
        obj = InteractiveMarker(...)
    """

    AXIS_LABELS = ("X", "Y", "Z")
    AXIS_COLORS = ((1.0, 0.18, 0.18, 1.0), (0.15, 0.85, 0.20, 1.0), (0.20, 0.45, 1.0, 1.0))
    PLANE_COLORS = ((1.0, 0.18, 0.18, 0.45), (0.15, 0.85, 0.20, 0.45), (0.20, 0.45, 1.0, 0.45))
    RING_COLORS = ((1.0, 0.18, 0.18, 0.95), (0.15, 0.85, 0.20, 0.95), (0.20, 0.45, 1.0, 0.95))
    HI_COLOR = (1.0, 0.92, 0.15, 1.0)

    def __init__(
            self,
            env,
            T                  = None,
            axis_len           = 0.20,
            ring_r             = 0.20,
            ring_segs          = 50,
            translation_frame  = "global",
            hover_scale         = 1.2,
            mouse_view_offset   = (0.03, 0.0),
            control_mode        = "pR",
            show_frame_toggle   = True,
            show_rotation_reset = True,
            plane               = None,
            hover_px           = (28.0, 22.0, 18.0),
            top_handle_px      = 18.0,
            top_handle_r       = 0.014,
            top_handle_offset  = 0.07,
            view_q_min         = 0.10,
            ring_thick         = 0.004,
            arrow_r            = 0.008,
            click_sphere_r     = 0.006,
            mouse_sphere_r     = 0.003,
            label_offset       = 0.01,
            hover_score_bias   = 0.02,
            top_score_bias     = 0.05,
            rotate_edge_min    = 0.03,
            axis_colors        = AXIS_COLORS,
            plane_colors       = PLANE_COLORS,
            ring_colors        = RING_COLORS,
            hi_color           = HI_COLOR,
            top_handle_color   = (0.15, 0.15, 0.15, 1.0),
            click_color        = (1.0, 0.0, 1.0, 1.0),
            mouse_color        = (0.0, 1.0, 1.0, 1.0),
        ):
        """
        Initialize InteractiveMarker.

        Parameters:
            env (Any): Input env.
            T (Any | None): Input T. Defaults to `None`.
            axis_len (float): Input axis len. Defaults to `0.2`.
            ring_r (float): Input ring r. Defaults to `0.2`.
            ring_segs (int): Input ring segs. Defaults to `50`.
            translation_frame (str): Input translation frame. Defaults to `'global'`.
            hover_scale (float): Input hover scale. Defaults to `1.2`.
            mouse_view_offset (tuple): Input mouse view offset. Defaults to `(0.03,
                                       0.0)`.
            control_mode (str): Input control mode. Defaults to `'pR'`.
            show_frame_toggle (bool): Input show frame toggle. Defaults to `True`.
            show_rotation_reset (bool): Input show rotation reset. Defaults to `True`.
            plane (Any | None): Input plane. Defaults to `None`.
            hover_px (tuple): Input hover px. Defaults to `(28.0, 22.0, 18.0)`.
            top_handle_px (float): Input top handle px. Defaults to `18.0`.
            top_handle_r (float): Input top handle r. Defaults to `0.014`.
            top_handle_offset (float): Input top handle offset. Defaults to `0.07`.
            view_q_min (float): Input view q min. Defaults to `0.1`.
            ring_thick (float): Input ring thick. Defaults to `0.004`.
            arrow_r (float): Input arrow r. Defaults to `0.008`.
            click_sphere_r (float): Input click sphere r. Defaults to `0.006`.
            mouse_sphere_r (float): Input mouse sphere r. Defaults to `0.003`.
            label_offset (float): Input label offset. Defaults to `0.01`.
            hover_score_bias (float): Input hover score bias. Defaults to `0.02`.
            top_score_bias (float): Input top score bias. Defaults to `0.05`.
            rotate_edge_min (float): Input rotate edge min. Defaults to `0.03`.
            axis_colors (Any): Input axis colors. Defaults to `AXIS_COLORS`.
            plane_colors (Any): Input plane colors. Defaults to `PLANE_COLORS`.
            ring_colors (Any): Input ring colors. Defaults to `RING_COLORS`.
            hi_color (Any): Input hi color. Defaults to `HI_COLOR`.
            top_handle_color (tuple): Input top handle color. Defaults to `(0.15,
                                      0.15, 0.15, 1.0)`.
            click_color (tuple): Input click color. Defaults to `(1.0, 0.0, 1.0,
                                 1.0)`.
            mouse_color (tuple): Input mouse color. Defaults to `(0.0, 1.0, 1.0,
                                 1.0)`.

        Returns:
            None: Initializes the instance in place.
        """
        self.env, self.viewer = env, env.viewer
        self.T = np.eye(4) if T is None else np.asarray(T, dtype=float).copy()
        self.T0 = self.T.copy()
        # T0 is the reference pose used by reset() and the Shift-mode reset-rotation handle.
        self.axis_len, self.ring_r, self.ring_segs = float(axis_len), float(ring_r), int(ring_segs)
        self.hover_scale = float(hover_scale)
        self.mouse_view_offset = np.asarray(mouse_view_offset, dtype=float).reshape(2)
        control_mode = str(control_mode)
        if control_mode in ("p","position","translate"):
            control_mode = "p"
        elif control_mode in ("R","r","rotation","rotate"):
            control_mode = "R"
        elif control_mode in ("pR","pr","PR","pose","both"):
            control_mode = "pR"
        else:
            raise ValueError("control_mode must be 'p', 'R', or 'pR'")
        self.control_mode = control_mode
        self.show_frame_toggle = bool(show_frame_toggle)
        self.show_rotation_reset = bool(show_rotation_reset)
        # Translation handles can be drawn/manipulated in marker-local or world-global axes.
        if translation_frame not in ("local", "global"):
            raise ValueError("translation_frame must be 'local' or 'global'")
        self.translation_frame = translation_frame
        if plane is None:
            plane = self._default_plane_from_axis_len(self.axis_len)
        self.plane_off, self.plane_size, self.plane_thick = map(float, plane)
        self.hover_px_axis, self.hover_px_plane, self.hover_px_ring = map(float, hover_px)
        # Pixel/world sizes for the top mode handles. hover_scale thickens hovered handles only.
        self.top_handle_px = float(top_handle_px)
        self.top_handle_r = float(top_handle_r)
        self.top_handle_offset = float(top_handle_offset)
        self.view_q_min = float(view_q_min)
        self.ring_thick = float(ring_thick)
        self.arrow_r = float(arrow_r)
        self.click_sphere_r = float(click_sphere_r)
        self.mouse_sphere_r = float(mouse_sphere_r)
        self.label_offset = float(label_offset)
        self.hover_score_bias = float(hover_score_bias)
        self.top_score_bias = float(top_score_bias)
        self.rotate_edge_min = float(rotate_edge_min)
        self.axis_colors = tuple(axis_colors)
        self.plane_colors = tuple(plane_colors)
        self.ring_colors = tuple(ring_colors)
        self.hi_color = tuple(hi_color)
        self.top_handle_color = tuple(top_handle_color)
        self.click_color = tuple(click_color)
        self.mouse_color = tuple(mouse_color)
        self.shift_held = False
        self.hover_mode = self.hover_axis = None
        self.drag_mode = self.drag_axis = None
        self.rot_cum_angle, self.rot_sign = 0.0, 1.0
        self.last_click_p_local = self.last_click_mode = self.last_click_axis = None
        self._prev_lmb_pressed = self._owns_lmb = self._suppressing_viewer_lmb = False
        # Cached screen/world handle data. axis is for translation; rot_axis is always local rotation.
        self._proj = dict(
            axis=[None] * 3,
            rot_axis=[None] * 3,
            p0=None,
            p1=[None] * 3,
            pix_dir=[None] * 3,
            pix_len=[0.0] * 3,
            ring_pix=[None] * 3,
            reset_p=None,
            reset_pix=None,
            frame_p=None,
            frame_pix=None,
        )
        self._drag = {}
        self._cb = None
        self._raw_mouse_xy = None
        self._last_sync_key = None
        self._last_error = None

    @staticmethod
    def _default_plane_from_axis_len(axis_len):
        """
        Return plane handle geometry scaled from the translation axis length.

        Parameters:
            axis_len (Any): Input axis len.

        Returns:
            Any: Result computed from the current inputs.
        """
        axis_len = float(axis_len)
        return 0.32 * axis_len, 0.28 * axis_len, 0.01 * axis_len

    def set_pose(self, p=None, R=None):
        """
        Set the marker pose by specifying a new position p and/or rotation R. This
        does not affect the reference pose T0 used by reset() and the Shift-mode
        reset-rotation handle.

        Parameters:
            p (Any | None): Input p. Defaults to `None`.
            R (Any | None): Input R. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if p is not None:
            self.T = pr2t(p=np.asarray(p, dtype=float), R=t2r(self.T))
        if R is not None:
            self.T = pr2t(p=t2p(self.T), R=self._orthonormalize(np.asarray(R, dtype=float)))
        return self

    def set_T(self, T, update_T0=False, clear_state=True):
        """
        Set the marker transform directly.

        Parameters:
            T (Any): Input T.
            update_T0 (bool): Input update T0. Defaults to `False`.
            clear_state (bool): Input clear state. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.T = np.asarray(T, dtype=float).copy()
        if update_T0:
            self.T0 = self.T.copy()
        if clear_state:
            self._clear_drag()
            self._clear_hover()
        return self

    def set_T0(self, T=None):
        """
        Set the marker reset/reference transform. If T is None, the current marker
        transform is used.

        Parameters:
            T (Any | None): Input T. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.T0 = self.T.copy() if T is None else np.asarray(T, dtype=float).copy()
        return self

    def reset(self):
        """
        Reset the marker pose to its initial pose T0, which is also the reference for
        the Shift-mode reset-rotation handle.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        self.T = self.T0.copy()
        self._clear_drag()
        self._clear_hover()
        return self

    def reset_rotation(self):
        """
        Reset only the marker rotation to its initial rotation.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        # Keep the current position, but restore the initial rotation R0.
        self.set_pose(R=t2r(self.T0))
        self._clear_drag()
        return self

    def toggle_translation_frame(self):
        """
        Toggle translation handles between marker-local and world-global axes.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        self.translation_frame = "global" if self.translation_frame == "local" else "local"
        self._clear_hover()
        return self

    def update(self, draw=True):
        """
        Poll input, refresh hover/drag state, optionally draw the marker, and return
        the transform.

        Parameters:
            draw (bool): Input draw. Defaults to `True`.

        Returns:
            Any: Result computed from the current inputs.
        """
        cb = self._begin_frame(sync_camera=True)
        if cb is None:
            return self.T
        self.shift_held = self._key(glfw.KEY_LEFT_SHIFT) or self._key(glfw.KEY_RIGHT_SHIFT)
        active_set = self._active_set()
        p_now, R_now = t2p(self.T), t2r(self.T)
        self._refresh_projection(p_now, R_now, cb)
        if self.drag_mode is None:
            # Keep hover available on the press edge so a marker drag can start.
            # Once LMB is already held by the viewer, suppress marker hover/highlight.
            self._update_hover(active_set) if (self._owns_lmb or not self._plain_lmb() or not self._prev_lmb_pressed) else self._clear_hover()
        self._poll_mouse()
        if draw:
            self._draw()
        return self.T

    def draw(self, alpha=None):
        """
        Refresh projection for the current camera and draw the marker once.

        Parameters:
            alpha (Any | None): Input alpha. Defaults to `None`.

        Returns:
            Any: Result computed from the current inputs.
        """
        cb = self._begin_frame(sync_camera=True)
        if cb is None:
            return self
        self._refresh_projection(t2p(self.T), t2r(self.T), cb)
        self._draw(alpha=alpha)
        return self

    @staticmethod
    def _rgba_alpha(rgba, alpha):
        """
        Return rgba with alpha overridden when alpha is not None.

        Parameters:
            rgba (tuple | list): Input rgba.
            alpha (float): Input alpha.

        Returns:
            Any: Result computed from the current inputs.
        """
        if alpha is None:
            return rgba
        out = list(rgba)
        out[3] = float(alpha)
        return tuple(out)

    def get_T(self, update=True, draw=False):
        """
        Return the current marker transform, optionally polling input and updating
        hover/drag state first.

        Parameters:
            update (bool): Input update. Defaults to `True`.
            draw (bool): Input draw. Defaults to `False`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if update:
            self.update(draw=draw)
        return self.T.copy()

    def get_status(self):
        """
        Return compact text describing the current hover/drag state and marker pose,
        useful for overlay display.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        return self.status_text

    @property
    def status_text(self):
        """
        Return compact text describing the current hover/drag state and marker pose,
        useful for overlay display.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        op = "ROTATE" if self._active_set() == "rotate" else "TRANSLATE"
        mode = self.drag_mode if self.drag_mode is not None else self.hover_mode
        axis = self.drag_axis if self.drag_axis is not None else self.hover_axis
        axis_text = self.AXIS_LABELS[axis] if axis is not None else "-"
        extra = " angle=%+.1f deg" % np.degrees(self.rot_sign * self.rot_cum_angle) if self.drag_mode == "rotate" else ""
        p = t2p(self.T)
        return "[%s] [%s %s] mode=%s frame=%s p=[%+.3f, %+.3f, %+.3f]%s" % (
            op, mode or "-", axis_text, self.control_mode, self.translation_frame, p[0], p[1], p[2], extra
        )

    def _active_set(self):
        """
        Return the currently editable handle set: 'translate' or 'rotate'.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        if self.control_mode == "R":
            return "rotate"
        if self.control_mode == "p":
            return "translate"
        return "rotate" if (self.shift_held or self.drag_mode == "rotate") else "translate"

    def _key(self, key):
        """
        Return True when a GLFW key is pressed.

        Parameters:
            key (Any): Input key.

        Returns:
            Any: Result computed from the current inputs.
        """
        try:
            pressed = glfw.get_key(self.viewer.window, key) == glfw.PRESS
            self._last_error = None
            return pressed
        except Exception as exc:
            self._last_error = repr(exc)
            return False

    def _plain_lmb(self):
        """
        Return True when left mouse is pressed without Ctrl modifiers.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        lmb = glfw.get_mouse_button(self.viewer.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        return bool(lmb and not (self._key(glfw.KEY_LEFT_CONTROL) or self._key(glfw.KEY_RIGHT_CONTROL)))

    def _mouse_xy(self, x=None, y=None):
        """
        Return mouse pixels using the same scale convention as the viewer callbacks.

        Parameters:
            x (Any | None): Input x. Defaults to `None`.
            y (Any | None): Input y. Defaults to `None`.

        Returns:
            Any: Result computed from the current inputs.
        """
        if x is None or y is None:
            x, y = glfw.get_cursor_pos(self.viewer.window)
        scale = getattr(self.viewer, "_scale", None)
        if scale is None:
            fb_w, _ = glfw.get_framebuffer_size(self.viewer.window)
            win_w, _ = glfw.get_window_size(self.viewer.window)
            scale = fb_w / max(1.0, win_w)
        return float(int(scale * x)), float(int(scale * y))

    def _begin_frame(self, sync_camera=True):
        """
        Cache camera and mouse values once for the current marker update/draw call.

        Parameters:
            sync_camera (bool): Input sync camera. Defaults to `True`.

        Returns:
            Any: Result computed from the current inputs.
        """
        if sync_camera:
            self._sync_scene_camera()
        self._cb = self._read_cam()
        self._raw_mouse_xy = self._mouse_xy()
        return self._cb

    def _camera_sync_key(self):
        """
        Return a compact key describing the viewer camera and framebuffer state.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        fb_w, fb_h = glfw.get_framebuffer_size(self.viewer.window)
        cam = self.viewer.cam
        lookat = tuple(float(v) for v in cam.lookat)
        return (
            int(fb_w),
            int(fb_h),
            int(cam.type),
            int(cam.fixedcamid),
            int(cam.trackbodyid),
            round(float(cam.azimuth), 9),
            round(float(cam.elevation), 9),
            round(float(cam.distance), 9),
            tuple(round(v, 9) for v in lookat),
        )

    def _sync_scene_camera(self):
        """
        Update MuJoCo's scene camera before using scn.camera[0] for ray/project math.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        try:
            fb_w, fb_h = glfw.get_framebuffer_size(self.viewer.window)
            self.viewer.viewport.width, self.viewer.viewport.height = int(fb_w), int(fb_h)
            sync_key = self._camera_sync_key()
            is_free_camera = int(self.viewer.cam.type) == mujoco.mjtCamera.mjCAMERA_FREE.value
            if is_free_camera and self._last_sync_key == sync_key and self.viewer.scn.camera[0].frustum_near > 0:
                return self
            lock = getattr(self.viewer, "_gui_lock", None)
            if lock is None:
                mujoco.mjv_updateScene(
                    self.viewer.model,
                    self.viewer.data,
                    self.viewer.vopt,
                    self.viewer.pert,
                    self.viewer.cam,
                    mujoco.mjtCatBit.mjCAT_ALL.value,
                    self.viewer.scn,
                )
            else:
                with lock:
                    mujoco.mjv_updateScene(
                        self.viewer.model,
                        self.viewer.data,
                        self.viewer.vopt,
                        self.viewer.pert,
                        self.viewer.cam,
                        mujoco.mjtCatBit.mjCAT_ALL.value,
                        self.viewer.scn,
                    )
            self._last_sync_key = sync_key
            self._last_error = None
        except Exception as exc:
            self._last_error = repr(exc)
        return self

    def _cam(self):
        """
        Return the cached camera basis, computing it only as a fallback.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        if self._cb is None:
            self._cb = self._read_cam()
        return self._cb

    def _read_cam(self):
        """
        Read camera basis and viewport values used for project/unproject.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        cam = self.viewer.scn.camera[0]
        if cam.frustum_near <= 0:
            return None
        pos = np.array(cam.pos, dtype=float)
        fwd = np.array(cam.forward, dtype=float)
        up0 = np.array(cam.up, dtype=float)
        right = np.cross(fwd, up0)
        nr = float(np.linalg.norm(right))
        if nr < 1e-9:
            return None
        try:
            fb_w, fb_h = glfw.get_framebuffer_size(self.viewer.window)
            self.viewer.viewport.width, self.viewer.viewport.height = int(fb_w), int(fb_h)
            self._last_error = None
        except Exception as exc:
            self._last_error = repr(exc)
        W, H = int(self.viewer.viewport.width), int(self.viewer.viewport.height)
        top = float(cam.frustum_top)
        bottom = float(getattr(cam, "frustum_bottom", -cam.frustum_top))
        # MuJoCo cameras may use an off-center frustum. Keep center/top/bottom here so projected handle pixels
        # and mouse rays stay aligned with what the viewer actually renders.
        half_w = (W / H) * (top - bottom) * 0.5
        if W <= 0 or H <= 0 or top <= bottom or half_w <= 0:
            return None
        right = right / nr
        return dict(pos=pos, fwd=fwd, right=right, up=np.cross(right, fwd), W=W, H=H,
                    top=top, bottom=bottom, center=float(getattr(cam, "frustum_center", 0.0)),
                    half_w=half_w, near=float(cam.frustum_near))

    def _project(self, p_world, cb=None):
        """
        Project a world point to framebuffer pixel coordinates.

        Parameters:
            p_world (np.ndarray): Input p world.
            cb (Any | None): Input cb. Defaults to `None`.

        Returns:
            Any: Result computed from the current inputs.
        """
        cb = self._cam() if cb is None else cb
        if cb is None:
            return None
        rel = np.asarray(p_world, dtype=float) - cb["pos"]
        x_c, y_c, z_c = float(rel @ cb["right"]), float(rel @ cb["up"]), float(rel @ cb["fwd"])
        if z_c <= 1e-6:
            return None
        # Project through the near plane, mirroring _ray() so hover tests and mouse-depth debug points use
        # one consistent camera model.
        x_near = cb["near"] * x_c / z_c
        y_near = cb["near"] * y_c / z_c
        x_ndc = (x_near - cb["center"]) / cb["half_w"]
        y_ndc = 2.0 * (y_near - cb["bottom"]) / (cb["top"] - cb["bottom"]) - 1.0
        return tuple(np.array([0.5 * (x_ndc + 1.0) * cb["W"], 0.5 * (1.0 - y_ndc) * cb["H"]]))

    def _project_many(self, points, cb=None):
        """
        Project a sequence of world points using one cached camera basis.

        Parameters:
            points (Any): Input points.
            cb (Any | None): Input cb. Defaults to `None`.

        Returns:
            Any: Result computed from the current inputs.
        """
        cb = self._cam() if cb is None else cb
        if cb is None:
            return [None] * len(points)
        pts = np.asarray(points, dtype=float)
        rel = pts - cb["pos"][None, :]
        x_c = rel @ cb["right"]
        y_c = rel @ cb["up"]
        z_c = rel @ cb["fwd"]
        valid = z_c > 1e-6
        out = [None] * len(pts)
        if not np.any(valid):
            return out
        x_near = cb["near"] * x_c[valid] / z_c[valid]
        y_near = cb["near"] * y_c[valid] / z_c[valid]
        x_ndc = (x_near - cb["center"]) / cb["half_w"]
        y_ndc = 2.0 * (y_near - cb["bottom"]) / (cb["top"] - cb["bottom"]) - 1.0
        pix = np.column_stack([0.5 * (x_ndc + 1.0) * cb["W"], 0.5 * (1.0 - y_ndc) * cb["H"]])
        for idx, q in zip(np.flatnonzero(valid), pix):
            out[int(idx)] = tuple(q)
        return out

    def _ray(self, mx, my, cb=None):
        """
        Unproject a framebuffer mouse point to a world-space ray.

        Parameters:
            mx (Any): Input mx.
            my (Any): Input my.
            cb (Any | None): Input cb. Defaults to `None`.

        Returns:
            Any: Result computed from the current inputs.
        """
        cb = self._cam() if cb is None else cb
        if cb is None:
            return None
        mx, my = np.asarray([mx, my], dtype=float)
        # GLFW cursor y is top-down, so convert to OpenGL-style NDC before building the near-plane point.
        x_ndc, y_ndc = (mx / cb["W"]) * 2.0 - 1.0, 1.0 - (my / cb["H"]) * 2.0
        x_near = cb["center"] + x_ndc * cb["half_w"]
        y_near = cb["bottom"] + 0.5 * (y_ndc + 1.0) * (cb["top"] - cb["bottom"])
        d = x_near * cb["right"] + y_near * cb["up"] + cb["near"] * cb["fwd"]
        return cb["pos"], d / float(np.linalg.norm(d))

    def _ray_plane(self, ray, plane_p, plane_n):
        """
        Intersect a ray with a plane.

        Parameters:
            ray (Any): Input ray.
            plane_p (Any): Input plane p.
            plane_n (Any): Input plane n.

        Returns:
            Any: Result computed from the current inputs.
        """
        if ray is None:
            return None
        ray_o, ray_d = ray
        denom = float(ray_d @ plane_n)
        if abs(denom) < 1e-9:
            return None
        t = float((plane_p - ray_o) @ plane_n) / denom
        return None if t < 0.0 else ray_o + t * ray_d

    def _mouse_offset(self, cb):
        """
        Return the combined mouse offset in world coordinates for the current camera
        basis.

        Parameters:
            cb (Any): Input cb.

        Returns:
            Any: Result computed from the current inputs.
        """
        return self.mouse_view_offset[0] * cb["right"] + self.mouse_view_offset[1] * cb["up"]

    def _mouse_xy_hit(self, p_ref=None, cb=None):
        """
        Return the virtual mouse pixel used by marker hit tests.

        Parameters:
            p_ref (Any | None): Input p ref. Defaults to `None`.
            cb (Any | None): Input cb. Defaults to `None`.

        Returns:
            Any: Result computed from the current inputs.
        """
        mx, my = self._raw_mouse_xy if self._raw_mouse_xy is not None else self._mouse_xy()
        if p_ref is None:
            p_ref = t2p(self.T)
        cb = self._cam() if cb is None else cb
        if cb is None:
            return mx, my
        base = self._project(p_ref, cb)
        shifted = self._project(np.asarray(p_ref, dtype=float) + self._mouse_offset(cb), cb)
        if base is None or shifted is None:
            return mx, my
        return mx + shifted[0] - base[0], my + shifted[1] - base[1]

    @staticmethod
    def _seg_dist(p, a, b):
        """
        Return the pixel distance from point p to segment ab.

        Parameters:
            p (np.ndarray): Input p.
            a (Any): Input a.
            b (Any): Input b.

        Returns:
            Any: Result computed from the current inputs.
        """
        px, py = p
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 < 1e-9:
            return float(np.hypot(px - ax, py - ay))
        t = float(np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0.0, 1.0))
        return float(np.hypot(px - (ax + t * dx), py - (ay + t * dy)))

    @staticmethod
    def _orthonormalize(R):
        """
        Project a 3x3 matrix onto the nearest valid rotation matrix.

        Parameters:
            R (Any): Input R.

        Returns:
            Any: Result computed from the current inputs.
        """
        U, _, Vt = np.linalg.svd(R)
        Rn = U @ Vt
        if np.linalg.det(Rn) < 0:
            U[:, -1] *= -1.0
            Rn = U @ Vt
        return Rn

    def _ring_pts(self, p, R, axis):
        """
        Sample world-space points on one rotation ring.

        Parameters:
            p (np.ndarray): Input p.
            R (Any): Input R.
            axis (Any): Input axis.

        Returns:
            Any: Result computed from the current inputs.
        """
        u, v = R[:, (axis + 1) % 3], R[:, (axis + 2) % 3]
        th = np.linspace(0.0, 2.0 * np.pi, self.ring_segs, endpoint=False)
        return p[None, :] + self.ring_r * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v)

    def _translation_R(self, R):
        """
        Return the frame used by translation handles.

        Parameters:
            R (Any): Input R.

        Returns:
            Any: Result computed from the current inputs.
        """
        # Only translation/plane handles use this frame; rotation handles stay marker-local.
        return np.eye(3) if self.translation_frame == "global" else R

    def _refresh_projection(self, p, R, cb=None):
        """
        Cache screen-space handles used by hover and drag tests.

        Parameters:
            p (np.ndarray): Input p.
            R (Any): Input R.
            cb (Any | None): Input cb. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        cb = self._cam() if cb is None else cb
        Rt = self._translation_R(R)
        # The rotate reset handle stays on local +Z; the translation-frame toggle follows whichever axes
        # translation currently uses, so local/global mode changes are visible immediately.
        self._proj["reset_p"] = p + np_uv(R[:, 2]) * (self.ring_r + self.top_handle_offset)
        self._proj["reset_pix"] = self._project(self._proj["reset_p"], cb)
        self._proj["frame_p"] = p + np_uv(Rt[:, 2]) * (self.ring_r + self.top_handle_offset)
        self._proj["frame_pix"] = self._project(self._proj["frame_p"], cb)
        self._proj["p0"] = self._project(p, cb)
        for k in range(3):
            # Translation axes may be local or global depending on translation_frame.
            ax = np_uv(Rt[:, k])
            # Rotation axes are always the marker-local axes, independent of translation_frame.
            self._proj["rot_axis"][k] = np_uv(R[:, k])
            p1 = self._project(p + ax * self.axis_len, cb)
            self._proj["axis"][k], self._proj["p1"][k] = ax, p1
            if self._proj["p0"] is not None and p1 is not None:
                d = np.asarray(p1) - np.asarray(self._proj["p0"])
                L = float(np.linalg.norm(d))
                self._proj["pix_len"][k] = L
                self._proj["pix_dir"][k] = tuple(d / L) if L > 1e-3 else None
            else:
                self._proj["pix_len"][k], self._proj["pix_dir"][k] = 0.0, None
            self._proj["ring_pix"][k] = self._project_many(self._ring_pts(p, R, k), cb)

    def _poly_dist(self, p, pts):
        """
        Return the closest pixel distance from p to a closed projected polyline.

        Parameters:
            p (np.ndarray): Input p.
            pts (Any): Input pts.

        Returns:
            Any: Result computed from the current inputs.
        """
        valid = [(i, q) for i, q in enumerate(pts or []) if q is not None]
        if len(valid) < 2:
            return 1e9
        best = 1e9
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            if a is not None and b is not None:
                best = min(best, self._seg_dist(p, a, b))
        return best

    def _update_hover(self, active_set):
        """
        Choose the hovered translate axis, plane handle, or rotation ring.

        Parameters:
            active_set (Any): Input active set.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.hover_mode = self.hover_axis = None
        p, R, cb = t2p(self.T), self._translation_R(t2r(self.T)), self._cam()
        if cb is None:
            return
        mx, my = self._mouse_xy_hit(p, cb)
        ray = self._ray(mx, my, cb)
        if ray is None:
            return
        candidates = []
        # Shift-mode reset handle hover test. If it wins, clicking restores R0 immediately.
        if active_set == "rotate" and self.show_rotation_reset and self._proj["reset_pix"] is not None:
            d_reset = float(np.hypot(mx - self._proj["reset_pix"][0], my - self._proj["reset_pix"][1]))
            if d_reset < self.top_handle_px:
                candidates.append((d_reset / self.top_handle_px - self.top_score_bias, "reset_rotation", None))
        # Translation-mode frame handle hover test. If it wins, clicking toggles local/global translation axes.
        if active_set == "translate" and self.show_frame_toggle and self._proj["frame_pix"] is not None:
            d_frame = float(np.hypot(mx - self._proj["frame_pix"][0], my - self._proj["frame_pix"][1]))
            if d_frame < self.top_handle_px:
                candidates.append((d_frame / self.top_handle_px - self.top_score_bias, "toggle_translation_frame", None))
        for k in range(3):
            # Choose local rotation axes in rotate mode, translation-frame axes otherwise.
            aw = self._proj["rot_axis"][k] if active_set == "rotate" else self._proj["axis"][k]
            af = float(aw @ cb["fwd"])
            if active_set == "rotate":
                if abs(af) >= self.rotate_edge_min:
                    score = self._poly_dist((mx, my), self._proj["ring_pix"][k]) / self.hover_px_ring - self.hover_score_bias * abs(af)
                    if score < 1.0:
                        candidates.append((score, "rotate", k))
                continue

            ps, pe = self._proj["p0"], self._proj["p1"][k]
            vq_axis, vq_plane = float(np.sqrt(max(0.0, 1.0 - af * af))), abs(af)
            if vq_axis >= self.view_q_min and ps is not None and pe is not None:
                score = self._seg_dist((mx, my), ps, pe) / self.hover_px_axis - self.hover_score_bias * vq_axis
                if score < 1.0:
                    candidates.append((score, "translate", k))
            if vq_plane >= self.view_q_min:
                q = self._ray_plane(ray, p, aw)
                if q is None:
                    continue
                u, v = R[:, (k + 1) % 3], R[:, (k + 2) % 3]
                uc, vc = float((q - p) @ u), float((q - p) @ v)
                du = max(self.plane_off - uc, uc - (self.plane_off + self.plane_size), 0.0)
                dv = max(self.plane_off - vc, vc - (self.plane_off + self.plane_size), 0.0)
                depth = float((q - cb["pos"]) @ cb["fwd"])
                px_per_m = cb["H"] * cb["near"] / max(1e-9, (cb["top"] - cb["bottom"]) * depth)
                score = np.hypot(du, dv) * px_per_m / self.hover_px_plane - self.hover_score_bias * vq_plane
                if score < 1.0:
                    candidates.append((score, "plane", k))
        if candidates:
            _, self.hover_mode, self.hover_axis = min(candidates, key=lambda c: c[0])

    def _clear_hover(self):
        """
        Clear hover state without touching an active drag.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.hover_mode = self.hover_axis = None

    def _handle_point(self, mode, axis, mx, my, cb=None):
        """
        Return the world-space handle point represented by the current click.

        Parameters:
            mode (Any): Input mode.
            axis (Any): Input axis.
            mx (Any): Input mx.
            my (Any): Input my.
            cb (Any | None): Input cb. Defaults to `None`.

        Returns:
            Any: Result computed from the current inputs.
        """
        cb = self._cam() if cb is None else cb
        p, R_pose = t2p(self.T), t2r(self.T)
        R = self._translation_R(R_pose)
        if mode == "translate":
            ps, pe, aw = self._proj["p0"], self._proj["p1"][axis], self._proj["axis"][axis]
            if ps is None or pe is None:
                return None
            d = np.asarray(pe) - np.asarray(ps)
            L2 = float(d @ d)
            t = 0.0 if L2 < 1e-9 else float(np.clip((np.asarray([mx, my]) - np.asarray(ps)) @ d / L2, 0.0, 1.0))
            return p + aw * (t * self.axis_len)
        if mode == "plane":
            return self._ray_plane(self._ray(mx, my, cb), p, self._proj["axis"][axis])
        if mode == "rotate":
            pts = self._proj["ring_pix"][axis]
            distances = [(float(np.hypot(mx - q[0], my - q[1])), i) for i, q in enumerate(pts) if q is not None]
            return None if not distances else self._ring_pts(p, R_pose, axis)[min(distances)[1]]
        return None

    def _start_drag(self, mode, axis, mx, my, cb=None):
        """
        Begin a translate, plane, or rotate drag operation.

        Parameters:
            mode (Any): Input mode.
            axis (Any): Input axis.
            mx (Any): Input mx.
            my (Any): Input my.
            cb (Any | None): Input cb. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        cb = self._cam() if cb is None else cb
        self.drag_mode, self.drag_axis = mode, axis
        p0, R0 = np.array(t2p(self.T), copy=True), np.array(t2r(self.T), copy=True)
        axis_world = self._proj["rot_axis"][axis] if mode == "rotate" else self._proj["axis"][axis]
        self._drag = dict(mouse=(mx, my), p=p0, R=R0, axis=axis_world.copy(),
                          pix_dir=self._proj["pix_dir"][axis], pix_len=self._proj["pix_len"][axis])
        click_p = self._handle_point(mode, axis, mx, my, cb)
        self.last_click_p_local = None if click_p is None else R0.T @ (click_p - p0)
        self.last_click_mode, self.last_click_axis = mode, axis
        if mode == "plane":
            self._drag["plane_origin"] = p0
            self._drag["plane_start"] = self._ray_plane(self._ray(mx, my, cb), self._drag["plane_origin"], self._drag["axis"])
            if self._drag["plane_start"] is None:
                self._clear_drag()
        elif mode == "rotate":
            center = self._project(p0, cb)
            if center is None or cb is None:
                self._clear_drag()
                return
            self.rot_sign = 1.0 if float(self._drag["axis"] @ cb["fwd"]) >= 0.0 else -1.0
            self.rot_cum_angle = 0.0
            self._drag["center"] = center
            self._drag["prev_angle"] = float(np.arctan2(my - center[1], mx - center[0]))

    def _drag_to(self, mx, my, cb=None):
        """
        Update marker pose for the current drag mode and mouse position.

        Parameters:
            mx (Any): Input mx.
            my (Any): Input my.
            cb (Any | None): Input cb. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        cb = self._cam() if cb is None else cb
        k, d = self.drag_axis, self._drag
        if self.drag_mode == "translate":
            dx, dy = mx - d["mouse"][0], my - d["mouse"][1]
            pix_dir, pix_len = d["pix_dir"], d["pix_len"]
            if pix_dir is not None and pix_len > 1e-3:
                offset = (dx * pix_dir[0] + dy * pix_dir[1]) * self.axis_len / pix_len
                self.set_pose(p=d["p"] + offset * d["axis"])
        elif self.drag_mode == "plane":
            q = self._ray_plane(self._ray(mx, my, cb), d["plane_origin"], d["axis"])
            if q is not None:
                self.set_pose(p=d["p"] + (q - d["plane_start"]))
        elif self.drag_mode == "rotate":
            angle = float(np.arctan2(my - d["center"][1], mx - d["center"][0]))
            da = float(normalize_angle(angle - d["prev_angle"]))
            self.rot_cum_angle += da
            d["prev_angle"] = angle
            self.set_pose(R=rodrigues(a=d["axis"], q_rad=self.rot_sign * self.rot_cum_angle) @ d["R"])

    def _clear_drag(self):
        """
        Clear active drag state and transient click visualization.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.drag_mode = self.drag_axis = None
        self.last_click_p_local = self.last_click_mode = self.last_click_axis = None
        self._drag = {}

    def _suppress_viewer_drag(self):
        """
        Stop the MuJoCo viewer callback only while this marker owns the current LMB
        press.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        active = self._owns_lmb
        if active or self._suppressing_viewer_lmb:
            try:
                self.viewer._button_left_pressed = False
                self.viewer.pert.active = 0
                self._last_error = None
            except Exception as exc:
                self._last_error = repr(exc)
        self._suppressing_viewer_lmb = bool(active)

    def _poll_mouse(self):
        """
        Advance marker drag state from current GLFW mouse state.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        cb = self._cam()
        if cb is None:
            return
        lmb_pressed = self._plain_lmb()
        p_ref = self._drag.get("p", t2p(self.T)) if self.drag_mode is not None else t2p(self.T)
        mx, my = self._mouse_xy_hit(p_ref, cb)
        # Press edge: either trigger the Shift reset handle or start a normal drag.
        if lmb_pressed and not self._prev_lmb_pressed and self.drag_mode is None and self.hover_mode is not None:
            # The marker only owns a drag when the press started on a marker handle.
            # If a viewer camera drag merely passes over the marker later, leave it to the viewer.
            self._owns_lmb = True
            # Reset and frame-toggle handles are click-only actions; they consume the press but do not start a drag.
            if self.hover_mode == "reset_rotation":
                self.reset_rotation()
            elif self.hover_mode == "toggle_translation_frame":
                self.toggle_translation_frame()
            else:
                self._start_drag(self.hover_mode, self.hover_axis, mx, my, cb)
        if self.drag_mode is not None and lmb_pressed:
            self._drag_to(mx, my, cb)
        if (not lmb_pressed) and self._prev_lmb_pressed and self.drag_mode is not None:
            self._clear_drag()
        if not lmb_pressed:
            self._owns_lmb = False
        self._prev_lmb_pressed = lmb_pressed
        self._suppress_viewer_drag()

    def _draw(self, alpha=None):
        """
        Draw translate or rotate handles plus the mouse/click reference spheres.

        Parameters:
            alpha (Any | None): Input alpha. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        p, R = t2p(self.T), t2r(self.T)
        Rt = self._translation_R(R)
        mode = self.drag_mode if self.drag_mode is not None else self.hover_mode
        axis = self.drag_axis if self.drag_axis is not None else self.hover_axis
        # alpha is a render-only override. Hit testing uses the cached projection geometry from _refresh_projection().
        # Draw rotate handles while requested by the marker control mode.
        rotate_view = self._active_set() == "rotate"
        if rotate_view:
            for k in range(3):
                is_hot = mode == "rotate" and axis == k
                color = self._rgba_alpha(self.hi_color if is_hot else self.ring_colors[k], alpha)
                ring = self._ring_pts(p, R, k)
                ring_r = self.ring_thick * (self.hover_scale if is_hot else 1.0)
                for i in range(len(ring)):
                    self.env.plot_cylinder_fr2to(p_fr=ring[i], p_to=ring[(i + 1) % len(ring)], r=ring_r, rgba=color)
            # Draw the clickable reset-rotation handle above the marker in rotate mode.
            if self.show_rotation_reset:
                reset_hot = mode == "reset_rotation"
                reset_color = self._rgba_alpha(self.hi_color if reset_hot else self.top_handle_color, alpha)
                self.env.plot_sphere(p=self._proj["reset_p"], r=self.top_handle_r * (self.hover_scale if reset_hot else 1.0), rgba=reset_color)
                self.env.plot_text(p=np.asarray(self._proj["reset_p"]) + np.array([0.0, 0.0, self.label_offset]), text="Reset")
        else:
            # Draw the clickable translation-frame toggle handle in normal translation mode.
            if self.show_frame_toggle:
                frame_hot = mode == "toggle_translation_frame"
                frame_color = self._rgba_alpha(self.hi_color if frame_hot else self.top_handle_color, alpha)
                self.env.plot_sphere(p=self._proj["frame_p"], r=self.top_handle_r * (self.hover_scale if frame_hot else 1.0), rgba=frame_color)
                self.env.plot_text(p=np.asarray(self._proj["frame_p"]) + np.array([0.0, 0.0, self.label_offset]), text=self.translation_frame.title())
            for k in range(3):
                is_hot = mode == "translate" and axis == k
                color = self._rgba_alpha(self.hi_color if is_hot else self.axis_colors[k], alpha)
                axis_len = self.axis_len
                arrow_r = self.arrow_r * (self.hover_scale if is_hot else 1.0)
                self.env.plot_arrow_fr2to(p_fr=p, p_to=p + Rt[:, k] * axis_len, r=arrow_r, rgba=color)
            for k in range(3):
                u, v, n = Rt[:, (k + 1) % 3], Rt[:, (k + 2) % 3], Rt[:, k]
                is_hot = mode == "plane" and axis == k
                color = self._rgba_alpha(self.hi_color if is_hot else self.plane_colors[k], alpha)
                plane_size = self.plane_size
                center = p + (self.plane_off + 0.5 * plane_size) * (u + v)
                self.env.plot_box(p=center, R=np.column_stack([u, v, n]), xlen=plane_size,
                                  ylen=plane_size, zlen=self.plane_thick * (self.hover_scale if is_hot else 1.0), rgba=color)
        if self.last_click_p_local is not None and self.drag_mode is not None:
            self.env.plot_sphere(p=p + R @ self.last_click_p_local, r=self.click_sphere_r, rgba=self._rgba_alpha(self.click_color, alpha), label="")
        cb = self._cam()
        if cb is None:
            return
        p_ref = self._drag.get("p", p) if self.drag_mode is not None else p
        mx, my = self._mouse_xy_hit(p_ref, cb)
        ray = self._ray(mx, my, cb)
        if ray is None:
            return
        denom = float(ray[1] @ cb["fwd"])
        depth = float((p_ref - cb["pos"]) @ cb["fwd"])
        if abs(denom) >= 1e-9 and depth / denom >= 0.0:
            mouse_p = ray[0] + (depth / denom) * ray[1]
            self.env.plot_sphere(p=mouse_p, r=self.mouse_sphere_r, rgba=self._rgba_alpha(self.mouse_color, alpha), label="")
