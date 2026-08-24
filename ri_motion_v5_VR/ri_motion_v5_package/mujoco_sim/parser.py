"""
MuJoCo parser and lightweight viewer utilities used across the workspace.
"""
from __future__ import annotations

import os
import time
import mujoco
import glfw
import pathlib
import cv2
import warnings
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from termcolor import colored

from typing import Optional
from threading import Lock

# MuJoCo version
MUJOCO_VERSION = tuple(map(int,mujoco.__version__.split('.')))

from ri_motion_v5_package.utility.utils import ( # type: ignore
    get_idxs,
    get_monitor_size,
    xy2xyz,
    TicToc,
    farthest_point_sampling,
)
from ri_motion_v5_package.utility.viz import ( # type: ignore
    get_colors,
    meters2xyz,
    print_red,
    print_yellow,
    print_green,
)
from ri_motion_v5_package.kinematics.transforms import ( # type: ignore
    np_uv,
    t2p,
    t2r,
    pr2t,
    rpy2r,
    get_R_from_twopoints,
    r2quat,
)
from ri_motion_v5_package.mujoco_sim.mujoco_utils import (
    normalize_mujoco_joint_types,
)

def _print_with_text_color(text,text_color=None):
    """
    Print text with an optional short color code.

    Parameters:
        text (Any): Input text.
        text_color (Any | None): Input text color. Defaults to `None`.

    Returns:
        None: This function mutates state, renders output, or performs side
              effects in place.
    """
    color_map = {
        "r":"red",
        "g":"green",
        "b":"blue",
        "c":"cyan",
    }
    if text_color is None:
        print(text)
        return

    color_key = str(text_color).lower()
    if color_key in color_map:
        print(colored(text,color_map[color_key]))
    else:
        print(text)

# MinimalCallbacks class
class MinimalCallbacks:
    """
    Store viewer callback hooks for MuJoCoMinimalViewer.

    Usage:
        obj = MinimalCallbacks(...)
    """
    def __init__(self):
        """
        Initialize MinimalCallbacks.

        Parameters:
            None.

        Returns:
            None: Initializes the instance in place.
        """
        # Mouse
        self._gui_lock                   = Lock()
        self._button_left_pressed        = False
        self._button_right_pressed       = False
        self._left_double_click_pressed  = False
        self._right_double_click_pressed = False
        self._last_left_click_time       = None
        self._last_right_click_time      = None
        self._last_mouse_x               = 0
        self._last_mouse_y               = 0
        self._paused                     = False
        self._render_every_frame         = True
        self._time_per_render            = 1/60.0
        self._run_speed                  = 1.0
        self._loop_count                 = 0
        self._advance_by_one_step        = False

        # Keyboard
        self._key_pressed                = None
        self._is_key_pressed             = False

        # Keyboard buffer
        self._key_pressed_set            = set()
        self._key_repeated_set           = set()

    def _key_callback(self, window, key, scancode, action, mods):
        """
        GLFW key callback to handle key press and release events.

        Parameters:
            window (Any): Input window.
            key (Any): Input key.
            scancode (Any): Input scancode.
            action (Any): Input action.
            mods (Any): Input mods.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        # Flags for key pressed
        is_key_pressed  = (action==glfw.PRESS)
        is_key_released = (action==glfw.RELEASE)
        is_key_repeated = (action==glfw.REPEAT)

        # Add and discard keys
        if is_key_pressed:
            self._key_pressed_set.add(key)
        if is_key_repeated:
            self._key_repeated_set.add(key)
        if is_key_released:
            # Remove from pressed and repeated lists (if present)
            self._key_pressed_set.discard(key)
            self._key_repeated_set.discard(key)

        # Quit (escape)
        if (key==glfw.KEY_ESCAPE):
            glfw.set_window_should_close(window, True)

        # Return
        return

    def _cursor_pos_callback(self, window, xpos, ypos):
        """
        GLFW cursor position callback to handle mouse movement events.

        Parameters:
            window (Any): Input window.
            xpos (Any): Input xpos.
            ypos (Any): Input ypos.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        # Only proceed if a mouse button is pressed
        if not (self._button_left_pressed or self._button_right_pressed):
            return

        # Determine action based on mouse buttons and modifier keys
        mod_shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
            glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
        if self._button_right_pressed:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self._button_left_pressed:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        # Calculate mouse movement deltas
        dx = int(self._scale * xpos) - self._last_mouse_x
        dy = int(self._scale * ypos) - self._last_mouse_y
        width, height = glfw.get_framebuffer_size(window)

        # Update last mouse positions
        with self._gui_lock:
            if self.pert.active:
                mujoco.mjv_movePerturb(
                    self.model,
                    self.data,
                    action,
                    dx / height,
                    dy / height,
                    self.scn,
                    self.pert)
            else:
                mujoco.mjv_moveCamera(
                    self.model,
                    action,
                    dx / height,
                    dy / height,
                    self.scn,
                    self.cam)

        # Update last mouse positions
        self._last_mouse_x = int(self._scale * xpos)
        self._last_mouse_y = int(self._scale * ypos)

    def _mouse_button_callback(self, window, button, act, mods):
        """
        GLFW callback for mouse button actions.

        Parameters:
            window (Any): Input window.
            button (Any): Input button.
            act (Any): Input act.
            mods (Any): Input mods.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self._button_left_pressed = button == glfw.MOUSE_BUTTON_LEFT and act == glfw.PRESS
        self._button_right_pressed = button == glfw.MOUSE_BUTTON_RIGHT and act == glfw.PRESS

        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = int(self._scale * x)
        self._last_mouse_y = int(self._scale * y)

        # detect a left- or right- doubleclick
        self._left_double_click_pressed = False
        self._right_double_click_pressed = False
        time_now = glfw.get_time()

        if self._button_left_pressed:
            if self._last_left_click_time is None:
                self._last_left_click_time = glfw.get_time()

            time_diff = (time_now - self._last_left_click_time)
            if time_diff > 0.01 and time_diff < 0.3:
                self._left_double_click_pressed = True
            self._last_left_click_time = time_now

        if self._button_right_pressed:
            if self._last_right_click_time is None:
                self._last_right_click_time = glfw.get_time()

            time_diff = (time_now - self._last_right_click_time)
            if time_diff > 0.01 and time_diff < 0.3:
                self._right_double_click_pressed = True
            self._last_right_click_time = time_now

        # set perturbation
        key = mods == glfw.MOD_CONTROL
        newperturb = 0
        if key and self.pert.select > 0:
            # right: translate, left: rotate
            if self._button_right_pressed:
                newperturb = mujoco.mjtPertBit.mjPERT_TRANSLATE
            if self._button_left_pressed:
                newperturb = mujoco.mjtPertBit.mjPERT_ROTATE

            # perturbation onste: reset reference
            if newperturb and not self.pert.active:
                mujoco.mjv_initPerturb(
                    self.model, self.data, self.scn, self.pert)
        self.pert.active = newperturb
        # 3D release
        if act == glfw.RELEASE:
            self.pert.active = 0

    def _scroll_callback(self, window, x_offset, y_offset):
        """
        GLFW callback for mouse wheel scrolling.

        Parameters:
            window (Any): Input window.
            x_offset (Any): Input x offset.
            y_offset (Any): Input y offset.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        with self._gui_lock:
            mujoco.mjv_moveCamera(
                self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * y_offset, self.scn, self.cam)

# MuJoCoMinimalViewer class
class MuJoCoMinimalViewer(MinimalCallbacks):
    """
    Manage the GLFW window and render state for MuJoCo.

    Usage:
        obj = MuJoCoMinimalViewer(...)
    """
    def __init__(
            self,
            model,
            data,
            mode              = 'window',
            title             = "MuJoCo Minimal Viewer",
            width             = None,
            height            = None,
            x_offset          = None,
            y_offset          = None,
            maxgeom           = 50000,
            perturbation      = True,
        ):
        """
        Initialize MuJoCoMinimalViewer.

        Parameters:
            model (Any): Input model.
            data (Any): Input data.
            mode (str): Input mode. Defaults to `'window'`.
            title (str): Input title. Defaults to `'MuJoCo Minimal Viewer'`.
            width (Any | None): Input width. Defaults to `None`.
            height (Any | None): Input height. Defaults to `None`.
            x_offset (Any | None): Input x offset. Defaults to `None`.
            y_offset (Any | None): Input y offset. Defaults to `None`.
            maxgeom (int): Input maxgeom. Defaults to `50000`.
            perturbation (bool): Input perturbation. Defaults to `True`.

        Returns:
            None: Initializes the instance in place.
        """
        super().__init__()

        # Store parameters
        self.model       = model
        self.data        = data
        self.render_mode = mode
        self.is_alive    = True # viewer is alive
        self.CONFIG_PATH = pathlib.Path.joinpath(pathlib.Path.home(), ".config/mujoco_viewer/config.yaml")

        # glfw init
        glfw.init()

        # Get monitor size
        monitor_w,monitor_h = glfw.get_video_mode(glfw.get_primary_monitor()).size
        if not width:
            width = monitor_w
        if not height:
            height = monitor_h
        if self.render_mode == 'offscreen':
            glfw.window_hint(glfw.VISIBLE, 0)

        # Create window
        self.maxgeom = maxgeom
        self.window = glfw.create_window(width, height, title, None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # Get framebuffer size
        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(self.window)

        # install callbacks only for 'window' mode
        if self.render_mode == 'window':
            window_width, _ = glfw.get_window_size(self.window)
            # Mouse callbacks use framebuffer pixels, so keep the conversion
            # from logical window coordinates for HiDPI displays.
            self._scale = framebuffer_width * 1.0 / window_width
            # set callbacks
            glfw.set_cursor_pos_callback(
                self.window, self._cursor_pos_callback)
            glfw.set_mouse_button_callback(
                self.window, self._mouse_button_callback)
            glfw.set_scroll_callback(self.window, self._scroll_callback)
            glfw.set_key_callback(self.window, self._key_callback)

        # create options, camera, scene, context
        self.vopt = mujoco.MjvOption()
        self.cam  = mujoco.MjvCamera()
        self.scn  = mujoco.MjvScene(self.model, maxgeom=self.maxgeom)
        self.pert = mujoco.MjvPerturb()

        # Context is initialized outside MuJoCoMinimalViewer class
        # self.ctx  = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

        # set up rgb overlay
        width, height = glfw.get_framebuffer_size(self.window)

        # # Set window position
        # if x_offset is not None or y_offset is not None:
        #     # window size
        #     win_w,win_h = glfw.get_window_size(self.window)

        #     # default = center
        #     x_px = int((monitor_w-win_w)*0.5)
        #     y_px = int((monitor_h-win_h)*0.5)

        #     # x offset (ratio: monitor-based, else pixel)
        #     if x_offset is not None:
        #         if 0.0 <= x_offset <= 1.0:
        #             x_px = int(monitor_w*float(x_offset))
        #         else:
        #             x_px = int(x_offset)

        #     # y offset (ratio: monitor-based, else pixel)
        #     if y_offset is not None:
        #         if 0.0 <= y_offset <= 1.0:
        #             y_px = int(monitor_h*float(y_offset))
        #         else:
        #             y_px = int(y_offset)

        #     # clamp so window stays fully on screen
        #     x_px = max(0,min(x_px,monitor_w-win_w))
        #     y_px = max(0,min(y_px,monitor_h-win_h))

        #     glfw.set_window_pos(self.window,x_px,y_px)

        # Set window position
        win_w,win_h = glfw.get_window_size(self.window)

        # default = center
        x_px = int((monitor_w-win_w)*0.5)
        y_px = int((monitor_h-win_h)*0.5)

        # Offsets accept either monitor-relative ratios (0~1) or raw pixels.
        if x_offset is not None:
            if 0.0 <= x_offset <= 1.0:
                x_px = int(monitor_w*float(x_offset))
            else:
                x_px = int(x_offset)

        # Same convention for vertical placement.
        if y_offset is not None:
            if 0.0 <= y_offset <= 1.0:
                y_px = int(monitor_h*float(y_offset))
            else:
                y_px = int(y_offset)

        # clamp so window stays fully on screen
        x_px = max(0,min(x_px,monitor_w-win_w))
        y_px = max(0,min(y_px,monitor_h-win_h))

        glfw.set_window_pos(self.window,x_px,y_px)

        # get viewport
        self.viewport = mujoco.MjrRect(0, 0, framebuffer_width, framebuffer_height)

        # overlay, markers
        self._overlay = {}
        self._markers = []

        # rgb images to overlay
        self.rgb_overlay_top_right    = None
        self.rgb_overlay_top_left     = None
        self.rgb_overlay_bottom_right = None
        self.rgb_overlay_bottom_left  = None

        # Perturbation
        self.perturbation = perturbation

    def add_marker(self, **marker_params):
        """
        Add a marker to be rendered in the viewer.

        Parameters:
            **marker_params (Any): Additional keyword arguments.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self._markers.append(marker_params)

    def _add_marker_to_scene(self, marker):
        """
        Add a marker's properties to the MuJoCo scene.

        Parameters:
            marker (Any): Input marker.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if self.scn.ngeom >= self.scn.maxgeom:
            raise RuntimeError(
                'Ran out of geoms. maxgeom: %d' %
                self.scn.maxgeom)

        g = self.scn.geoms[self.scn.ngeom]

        # default values.
        g.dataid      = -1
        g.objtype     = mujoco.mjtObj.mjOBJ_UNKNOWN
        g.objid       = -1
        g.category    = mujoco.mjtCatBit.mjCAT_DECOR
        g.matid       = -1 # newly added (by Jihwan, 2025-02-27)
        g.emission    = 0
        g.specular    = 0.5
        g.shininess   = 0.5
        g.reflectance = 0
        g.type        = mujoco.mjtGeom.mjGEOM_BOX
        g.size[:]     = np.ones(3) * 0.1
        g.mat[:]      = np.eye(3)
        g.rgba[:]     = np.ones(4)

        for key, value in marker.items():
            # setattr(g, key, value)
            if isinstance(value, (int, float, mujoco._enums.mjtGeom)):
                setattr(g, key, value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                attr = getattr(g, key)
                attr[:] = np.asarray(value).reshape(attr.shape)
            elif isinstance(value, str):
                # assert key == "label", "Only label is a string in mjtGeom."
                if value is None:
                    g.label[0] = 0
                else:
                    g.label = value
            elif hasattr(g, key):
                raise ValueError(
                    "mjtGeom has attr {} but type {} is invalid".format(
                        key, type(value)))
            else:
                raise ValueError("mjtGeom doesn't have field %s" % key)

        # Increment number of geoms
        self.scn.ngeom += 1
        return

    def apply_perturbations(self):
        """
        Apply any active perturbations to the simulation data by resetting external
        forces.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.data.xfrc_applied = np.zeros_like(self.data.xfrc_applied)
        mujoco.mjv_applyPerturbPose(self.model, self.data, self.pert, 0)
        mujoco.mjv_applyPerturbForce(self.model, self.data, self.pert)

    def add_overlay(
            self,
            loc     = 'bottom left',
            gridpos = mujoco.mjtGridPos.mjGRID_TOPLEFT,
            text1   = '',
            text2   = '',
        ):
        """
        Add a text overlay to the viewer at a specified location.

        Parameters:
            loc (str): Input loc. Defaults to `'bottom left'`.
            gridpos (Any): Input gridpos. Defaults to
                           `mujoco.mjtGridPos.mjGRID_TOPLEFT`.
            text1 (str): Input text1. Defaults to `''`.
            text2 (str): Input text2. Defaults to `''`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if loc is not None:
            if loc == 'top': gridpos = mujoco.mjtGridPos.mjGRID_TOP
            elif loc == 'top right': gridpos = mujoco.mjtGridPos.mjGRID_TOPRIGHT
            elif loc == 'top left': gridpos = mujoco.mjtGridPos.mjGRID_TOPLEFT
            elif loc == 'bottom': gridpos = mujoco.mjtGridPos.mjGRID_BOTTOM
            elif loc == 'bottom right': gridpos = mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT
            elif loc == 'bottom left': gridpos = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT

        if gridpos not in self._overlay:
            self._overlay[gridpos] = ["", ""]
            self._overlay[gridpos][0] += text1
            self._overlay[gridpos][1] += text2
        else:
            self._overlay[gridpos][0] += "\n" + text1
            self._overlay[gridpos][1] += "\n" + text2

    def plot_rgb_overlay(self,rgb=None,loc='top right'):
        """
        Plot an RGB overlay at a specified location in the viewer window.

        Parameters:
            rgb (Any | None): Input rgb. Defaults to `None`.
            loc (str): Input loc. Defaults to `'top right'`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        w_window,h_window = glfw.get_framebuffer_size(self.window)
        h_overlay,w_overlay = h_window//4,w_window//4
        # Fix aspect ratio
        h_raw,w_raw = rgb.shape[:2]
        # Calculate scale to preserve aspect ratio
        scale = min(w_overlay/w_raw,h_overlay/h_raw)
        w_new = int(w_raw*scale)
        h_new = int(h_raw*scale)
        # Resize
        rgb_resized = cv2.resize(rgb,(w_new,h_new),interpolation=cv2.INTER_NEAREST)
        # Create a black canvas with the target size
        rgb_padded = np.zeros((h_overlay,w_overlay,3),dtype=np.uint8)
        # Calculate the top-left corner for centering the resized image
        x_offset = (w_overlay-w_new) // 2
        y_offset = (h_overlay-h_new) // 2
        # Place the resized image onto the canvas
        rgb_padded[y_offset:y_offset+h_new,x_offset:x_offset+w_new] = rgb_resized
        # Store the RGB overlay
        if loc=='top right':
            self.rgb_overlay_top_right = rgb_padded
        elif loc=='top left':
            self.rgb_overlay_top_left = rgb_padded
        elif loc=='bottom right':
            self.rgb_overlay_bottom_right = rgb_padded
        elif loc=='bottom left':
            self.rgb_overlay_bottom_left = rgb_padded
        else:
            print ("Invalid location for RGB overlay. Use 'top right', 'top left', 'bottom right', or 'bottom left'.")

    def reset_rgb_overlay(self,loc=None):
        """
        Reset the RGB overlay(s).

        Parameters:
            loc (Any | None): Input loc. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if loc is None:
            self.rgb_overlay_top_right    = None
            self.rgb_overlay_top_left     = None
            self.rgb_overlay_bottom_right = None
            self.rgb_overlay_bottom_left  = None
        else:
            if loc=='top right':
                self.rgb_overlay_top_right = None
            if loc=='top left':
                self.rgb_overlay_top_left = None
            if loc=='bottom right':
                self.rgb_overlay_bottom_right = None
            if loc=='bottom left':
                self.rgb_overlay_bottom_left = None

    def update_render_scene(self):
        """
        Render one viewer frame and process overlays.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        render_start = time.time()
        width,height = glfw.get_framebuffer_size(self.window)
        self.viewport.width,self.viewport.height = width,height

        with self._gui_lock:
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                self.vopt,
                self.pert,
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL.value,
                self.scn,
            )

            for marker in self._markers:
                self._add_marker_to_scene(marker)

            mujoco.mjr_render(self.viewport,self.scn,self.ctx)

            for gridpos,[t1,t2] in self._overlay.items():
                mujoco.mjr_overlay(
                    mujoco.mjtFontScale.mjFONTSCALE_100,
                    gridpos,
                    self.viewport,
                    t1,
                    t2,
                    self.ctx,
                )

            if self.rgb_overlay_top_right is not None:
                h_overlay,w_overlay = self.rgb_overlay_top_right.shape[:2]
                viewport_rgb_top_right = mujoco.MjrRect(
                    left   = 3*w_overlay,
                    bottom = 3*h_overlay,
                    width  = w_overlay,
                    height = h_overlay,
                )
                mujoco.mjr_drawPixels(
                    rgb      = np.flipud(self.rgb_overlay_top_right).flatten(),
                    depth    = None,
                    viewport = viewport_rgb_top_right,
                    con      = self.ctx,
                )
            if self.rgb_overlay_top_left is not None:
                h_overlay,w_overlay = self.rgb_overlay_top_left.shape[:2]
                viewport_rgb_top_left = mujoco.MjrRect(
                    left   = 0*w_overlay,
                    bottom = 3*h_overlay,
                    width  = w_overlay,
                    height = h_overlay,
                )
                mujoco.mjr_drawPixels(
                    rgb      = np.flipud(self.rgb_overlay_top_left).flatten(),
                    depth    = None,
                    viewport = viewport_rgb_top_left,
                    con      = self.ctx,
                )
            if self.rgb_overlay_bottom_right is not None:
                h_overlay,w_overlay = self.rgb_overlay_bottom_right.shape[:2]
                viewport_rgb_bottom_right = mujoco.MjrRect(
                    left   = 3*w_overlay,
                    bottom = 0*h_overlay,
                    width  = w_overlay,
                    height = h_overlay,
                )
                mujoco.mjr_drawPixels(
                    rgb      = np.flipud(self.rgb_overlay_bottom_right).flatten(),
                    depth    = None,
                    viewport = viewport_rgb_bottom_right,
                    con      = self.ctx,
                )
            if self.rgb_overlay_bottom_left is not None:
                h_overlay,w_overlay = self.rgb_overlay_bottom_left.shape[:2]
                viewport_rgb_bottom_left = mujoco.MjrRect(
                    left   = 0*w_overlay,
                    bottom = 0*h_overlay,
                    width  = w_overlay,
                    height = h_overlay,
                )
                mujoco.mjr_drawPixels(
                    rgb      = np.flipud(self.rgb_overlay_bottom_left).flatten(),
                    depth    = None,
                    viewport = viewport_rgb_bottom_left,
                    con      = self.ctx,
                )

            glfw.swap_buffers(self.window)

        glfw.poll_events()
        self._time_per_render = (
            0.9*self._time_per_render + 0.1*(time.time()-render_start)
        )

    def render(self,reset_rgb_overlay=True):
        """
        Render the current scene in the viewer window.

        Parameters:
            reset_rgb_overlay (bool): Input reset rgb overlay. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if not self.is_alive:
            raise Exception(
                "GLFW window does not exist but you tried to render.")
        if glfw.window_should_close(self.window):
            self.close()
            return

        if self._paused: # if paused
            while self._paused:
                self.update_render_scene()
                if glfw.window_should_close(self.window):
                    self.close()
                    break
                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
                # Resume using 'glfw.KEY_SPACE'
        else:
            self._loop_count += self.model.opt.timestep / \
                (self._time_per_render * self._run_speed)
            if self._render_every_frame:
                self._loop_count = 1
            while self._loop_count > 0:
                self.update_render_scene()
                self._loop_count -= 1

        # clear markers
        self._markers[:] = []

        # clear overlay
        self._overlay.clear()
        if reset_rgb_overlay:
            # clear rgb overlay
            self.reset_rgb_overlay()

        # apply perturbation (should this come before mj_step?)
        if self.perturbation:
            self.apply_perturbations()

    def close(self):
        """
        Close the viewer, terminate the GLFW window, and free the MuJoCo context
        resources.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        # if not self.is_alive:
        #     return
        # self.is_alive=False
        # try:
        #     if hasattr(self,"ctx") and self.ctx is not None:
        #         self.ctx.free()
        # except Exception:
        #     pass
        # try:
        #     if self.window is not None:
        #         glfw.destroy_window(self.window)
        # except Exception:
        #     pass
        # try:
        #     glfw.terminate()
        # except Exception:
        #     pass
        if not self.is_alive:
            return

        self.is_alive = False

        try:
            if self.window is not None:
                glfw.set_window_should_close(self.window,True)
                glfw.hide_window(self.window)
                glfw.make_context_current(self.window)
        except Exception:
            pass

        try:
            if hasattr(self,"ctx") and self.ctx is not None:
                self.ctx.free()
                # self.ctx = None # Don't set to None to avoid potential double free if close() is called multiple times
        except Exception:
            pass

        try:
            glfw.make_context_current(None)
        except Exception:
            pass

        try:
            if self.window is not None:
                glfw.destroy_window(self.window)
                self.window = None
        except Exception:
            pass

        try:
            glfw.poll_events()
        except Exception:
            pass


    # Viewer settings
    def set_transparency(self,transparent=True):
        """
        Set the transparency (alpha value) for rendering.

        Parameters:
            transparent (bool): Input transparent. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = transparent

    def toggle_transparency(self):
        """
        Toggle the transparency setting for rendering.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = not self.vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT]

    def get_cam_info(self):
        """
        Get the current camera information.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        azimuth   = float(self.cam.azimuth)
        distance  = float(self.cam.distance)
        elevation = float(self.cam.elevation)
        lookat    = self.cam.lookat.copy()
        return azimuth,distance,elevation,lookat

    def set_cam_info(self,azimuth=None,distance=None,elevation=None,lookat=None):
        """
        Set the camera information.

        Parameters:
            azimuth (Any | None): Input azimuth. Defaults to `None`.
            distance (Any | None): Input distance. Defaults to `None`.
            elevation (Any | None): Input elevation. Defaults to `None`.
            lookat (Any | None): Input lookat. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.cam.azimuth   = azimuth if azimuth is not None else self.cam.azimuth
        self.cam.distance  = distance if distance is not None else self.cam.distance
        self.cam.elevation = elevation if elevation is not None else self.cam.elevation
        if lookat is not None:
            self.cam.lookat[:] = np.array(lookat).reshape(self.cam.lookat.shape)

    def set_geomgroup(
            self,
            group_0 = True,
            group_1 = True,
            group_2 = True,
            group_3 = False,
            group_4 = None,
            group_5 = None,
        ):
        """
        Set the geometry group visibility flags.

        Parameters:
            group_0 (bool): Input group 0. Defaults to `True`.
            group_1 (bool): Input group 1. Defaults to `True`.
            group_2 (bool): Input group 2. Defaults to `True`.
            group_3 (bool): Input group 3. Defaults to `False`.
            group_4 (Any | None): Input group 4. Defaults to `None`.
            group_5 (Any | None): Input group 5. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if group_0 is not None: self.vopt.geomgroup[0] = group_0
        if group_1 is not None: self.vopt.geomgroup[1] = group_1
        if group_2 is not None: self.vopt.geomgroup[2] = group_2
        if group_3 is not None: self.vopt.geomgroup[3] = group_3
        if group_4 is not None: self.vopt.geomgroup[4] = group_4
        if group_5 is not None: self.vopt.geomgroup[5] = group_5

    def set_sitegroup(
            self,
            group_0 = None,
            group_1 = None,
            group_2 = None,
            group_3 = None,
            group_4 = None,
            group_5 = None,
        ):
        """
        Set the site group visibility flags.

        Parameters:
            group_0 (Any | None): Input group 0. Defaults to `None`.
            group_1 (Any | None): Input group 1. Defaults to `None`.
            group_2 (Any | None): Input group 2. Defaults to `None`.
            group_3 (Any | None): Input group 3. Defaults to `None`.
            group_4 (Any | None): Input group 4. Defaults to `None`.
            group_5 (Any | None): Input group 5. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if group_0 is not None: self.vopt.sitegroup[0] = group_0
        if group_1 is not None: self.vopt.sitegroup[1] = group_1
        if group_2 is not None: self.vopt.sitegroup[2] = group_2
        if group_3 is not None: self.vopt.sitegroup[3] = group_3
        if group_4 is not None: self.vopt.sitegroup[4] = group_4
        if group_5 is not None: self.vopt.sitegroup[5] = group_5

    def set_convexhull(self,convexhull=False):
        """
        Set the visibility of convex hulls in the viewer.

        Parameters:
            convexhull (bool): Input convexhull. Defaults to `False`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.vopt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = convexhull

# MuJoCoParser class
class MuJoCoParser(object):
    """
    MuJoCo model parser class to parse MuJoCo XML files and extract model
    information.

    Usage:
        obj = MuJoCoParser(...)
    """
    def __init__(
            self,
            name         = None,
            rel_xml_path = None,
            verbose      = False,
        ):
        """
        Initialize MuJoCoParser.

        Parameters:
            name (Any | None): Input name. Defaults to `None`.
            rel_xml_path (Any | None): Input rel xml path. Defaults to `None`.
            verbose (bool): Input verbose. Defaults to `False`.

        Returns:
            None: Initializes the instance in place.
        """
        # Store parameters
        self.name         = name
        self.rel_xml_path = rel_xml_path
        self.verbose      = verbose

        # Parse XML
        self._parse_xml(self.rel_xml_path)

        # Set name if not provided
        if self.name is None: self.name = self.model_name

        # Get monitor size
        self.monitor_width,self.monitor_height = get_monitor_size()

        # Flags
        self.use_mujoco_viewer = False

        # Timing variables used in self.is_viewer_alive() method
        self._last_alive_t: Optional[float] = None
        self.loop_dt: Optional[float]       = 0.0
        self.loop_hz: Optional[float]       = 0.0
        self.loop_hz_ema: Optional[float]   = 0.0
        self._loop_hz_ema_alpha: float      = 0.05

        # Initial joint positions
        self.q0       = self.get_qpos()
        self.qrev0    = self.get_qpos(joint_names=self.rev_joint_names,flatten=True)
        self.qactive0 = self.get_qpos(joint_names=self.active_joint_names,flatten=True)

        # Final RGB image
        self.final_rgb_img = None

        # Object point cloud cache (body-local canonical pcd)
        self.obj_pcd_local_cache = {}

        # TicToc instance
        self.tt = TicToc()

        # Reset
        self.reset()

        # Print information
        if self.verbose:
            self.print_info()
        else:
            print ("MuJoCo model loaded from [%s]."%(self.rel_xml_path))

    def _parse_xml(self,rel_xml_path):
        """
        Parse the MuJoCo XML file.

        Parameters:
            rel_xml_path (str | Path): Input rel xml path.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.full_xml_path = os.path.abspath(os.path.join(os.getcwd(),rel_xml_path))
        self.model         = mujoco.MjModel.from_xml_path(self.full_xml_path)

        # Parse xml model name
        parsed_strings = [s for s in self.model.names.split(b'\x00') if s]
        parsed_strings = [s.decode('utf-8') for s in parsed_strings]
        self.model_name = parsed_strings[0]

        # Create MjData instance
        self.data = mujoco.MjData(self.model)
        self.dt   = self.model.opt.timestep
        self.HZ   = int(1/self.dt)

        # Parse integrator type
        """
        MuJoCo integrator details can be found here:
        https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjtintegrator
        """
        self.integrator = self.model.opt.integrator
        if self.integrator == mujoco.mjtIntegrator.mjINT_EULER:
            self.integrator_name = 'EULER'
        elif self.integrator == mujoco.mjtIntegrator.mjINT_RK4:
            self.integrator_name = 'RK4'
        elif self.integrator == mujoco.mjtIntegrator.mjINT_IMPLICIT:
            self.integrator_name = 'IMPLICIT'
        elif self.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST:
            self.integrator_name = 'IMPLICITFAST'
        else:
            self.integrator_name = 'UNKNOWN'

        # Parse model dimensions (I prefer to use snake_case for variable names)
        self.n_q = self.model.nq  # number of generalized coordinates
        self.n_v = self.model.nv  # number of generalized velocities
        self.n_u = self.model.nu  # number of actuators

        # Geometry information
        self.n_geom           = self.model.ngeom # number of geometries
        self.geom_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,geom_idx)
                                 for geom_idx in range(self.n_geom)]
        self.geom_name_to_id  = {name:idx for idx,name in enumerate(self.geom_names) if name is not None}

        # Mesh information
        self.n_mesh           = self.model.nmesh # number of meshes
        self.mesh_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_MESH,mesh_idx)
                              for mesh_idx in range(self.n_mesh)]
        self.mesh_name_to_id  = {name:idx for idx,name in enumerate(self.mesh_names) if name is not None}

        # Body information
        self.n_body           = self.model.nbody # number of bodies
        self.body_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,body_idx)
                                 for body_idx in range(self.n_body)]
        self.body_name_to_id  = {name:idx for idx,name in enumerate(self.body_names) if name is not None}
        self.body_masses      = self.model.body_mass
        self.body_total_mass  = self.body_masses.sum()
        # Keep name-level parent lookup because many helpers operate on body
        # names rather than integer body ids.
        self.parent_body_names = [] # parent body names
        for b_idx in range(self.n_body):
            parent_id = self.model.body_parentid[b_idx]
            parent_body_name = self.body_names[parent_id]
            self.parent_body_names.append(parent_body_name)

        # Joint information
        self.n_joint          = self.model.njnt # number of joints
        self.joint_names      = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_JOINT,joint_idx)
                                 for joint_idx in range(self.n_joint)]
        self.joint_name_to_id = {name:idx for idx,name in enumerate(self.joint_names) if name is not None}
        self.joint_types      = self.model.jnt_type # joint types
        self.joint_ranges     = self.model.jnt_range # joint ranges
        self.joint_mins       = self.joint_ranges[:,0]
        self.joint_maxs       = self.joint_ranges[:,1]

        # Free joint
        self.free_joint_idxs  = np.where(self.joint_types==mujoco.mjtJoint.mjJNT_FREE)[0].astype(np.int32)
        self.free_joint_names = [self.joint_names[joint_idx] for joint_idx in self.free_joint_idxs]
        self.n_free_joint     = len(self.free_joint_idxs)
        self.free_dof_idxs    = []
        for jidx in self.free_joint_idxs:
            dofadr = int(self.model.jnt_dofadr[jidx])
            self.free_dof_idxs.extend(range(dofadr,dofadr+6))
        self.free_dof_idxs = np.asarray(self.free_dof_idxs,dtype=np.int32)

        # Revolute Joint
        self.rev_joint_idxs   = np.where(self.joint_types==mujoco.mjtJoint.mjJNT_HINGE)[0].astype(np.int32)
        self.rev_joint_names  = [self.joint_names[joint_idx] for joint_idx in self.rev_joint_idxs]
        self.n_rev_joint      = len(self.rev_joint_idxs)
        self.rev_joint_mins   = self.joint_ranges[self.rev_joint_idxs,0]
        self.rev_joint_maxs   = self.joint_ranges[self.rev_joint_idxs,1]
        self.rev_joint_ranges = self.rev_joint_maxs - self.rev_joint_mins
        self.rev_qpos_idxs    = self.model.jnt_qposadr[self.rev_joint_idxs].astype(np.int32)
        self.rev_qpos_set     = set(int(idx) for idx in self.rev_qpos_idxs)

        # Prismatic Joint
        self.pri_joint_idxs   = np.where(self.joint_types==mujoco.mjtJoint.mjJNT_SLIDE)[0].astype(np.int32)
        self.pri_joint_names  = [self.joint_names[joint_idx] for joint_idx in self.pri_joint_idxs]
        self.n_pri_joint      = len(self.pri_joint_idxs)
        self.pri_joint_mins   = self.joint_ranges[self.pri_joint_idxs,0]
        self.pri_joint_maxs   = self.joint_ranges[self.pri_joint_idxs,1]
        self.pri_joint_ranges = self.pri_joint_maxs - self.pri_joint_mins
        self.pri_qpos_idxs    = self.model.jnt_qposadr[self.pri_joint_idxs].astype(np.int32)
        self.pri_qpos_set     = set(int(idx) for idx in self.pri_qpos_idxs)

        # Revolute + prismatic Joint Information
        self.n_rev_pri_joint      = self.n_rev_joint + self.n_pri_joint
        self.rev_pri_joint_idxs   = np.concatenate([self.rev_joint_idxs,self.pri_joint_idxs])
        self.rev_pri_joint_names  = self.rev_joint_names + self.pri_joint_names
        self.rev_pri_joint_mins   = np.concatenate([self.rev_joint_mins,self.pri_joint_mins])
        self.rev_pri_joint_maxs   = np.concatenate([self.rev_joint_maxs,self.pri_joint_maxs])
        self.rev_pri_joint_ranges = self.rev_pri_joint_maxs - self.rev_pri_joint_mins
        self.rev_pri_qpos_idxs    = self.model.jnt_qposadr[self.rev_pri_joint_idxs].astype(np.int32)

        # Equality-constrained revolute joints are treated as passive so
        # downstream helpers can target only independently driven joints.
        self.n_eq = self.model.eq_data.shape[0] # number of equality constraints
        self.active_joint_names = self.rev_pri_joint_names.copy()
        self.passive_joint_names = []
        for eq_idx in range(self.n_eq): # for each joint equality constraints
            if self.model.eq_type[eq_idx] == mujoco.mjtEq.mjEQ_JOINT: # joint equality constraint
                joint1 = self.model.joint(self.model.eq_obj1id[eq_idx]).name # passive joint name
                # If the passive joint is in the active joint list, move it to the passive joint list
                if joint1 in self.active_joint_names:
                    self.active_joint_names.remove(joint1)
                    self.passive_joint_names.append(joint1)
        self.active_rev_joint_names = [j for j in self.active_joint_names if j in self.rev_joint_names]
        self.active_rev_pri_joint_names = self.active_joint_names.copy()
        self.n_active_joint  = len(self.active_joint_names)
        self.active_joint_mins = self.joint_mins[get_idxs(self.joint_names,self.active_joint_names)]
        self.active_joint_maxs = self.joint_maxs[get_idxs(self.joint_names,self.active_joint_names)]
        self.active_joint_ranges = self.active_joint_maxs - self.active_joint_mins
        self.n_passive_joint = len(self.passive_joint_names)
        self.passive_joint_mins = self.joint_mins[get_idxs(self.joint_names,self.passive_joint_names)]
        self.passive_joint_maxs = self.joint_maxs[get_idxs(self.joint_names,self.passive_joint_names)]
        self.passive_joint_ranges = self.passive_joint_maxs - self.passive_joint_mins

        # Actuator (or control) information
        self.n_ctrl           = self.model.nu # number of actuators (or controls)
        self.ctrl_names       = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,ctrl_idx)
                                 for ctrl_idx in range(self.n_ctrl)]
        self.ctrl_name_to_id  = {name:idx for idx,name in enumerate(self.ctrl_names) if name is not None}
        self.ctrl_ranges      = self.model.actuator_ctrlrange # control range
        self.ctrl_mins        = self.ctrl_ranges[:,0]
        self.ctrl_maxs        = self.ctrl_ranges[:,1]
        self.ctrl_gears       = self.model.actuator_gear[:,0] # gears
        self.ctrl_trntypes    = self.model.actuator_trntype.copy()
        self.ctrl_gaintypes   = self.model.actuator_gaintype.copy()
        self.ctrl_biastypes   = self.model.actuator_biastype.copy()
        self.actuator_gainprm_0    = self.model.actuator_gainprm.copy()
        self.actuator_biasprm_0    = self.model.actuator_biasprm.copy()
        self.actuator_ctrlrange_0  = self.model.actuator_ctrlrange.copy()
        self.actuator_forcerange_0 = self.model.actuator_forcerange.copy()

        # Camera information
        self.n_cam            = self.model.ncam
        self.cam_names        = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_CAMERA,cam_idx)
                                 for cam_idx in range(self.n_cam)]
        self.cam_name_to_id   = {name:idx for idx,name in enumerate(self.cam_names) if name is not None}
        self.cams             = []
        self.cam_fovs         = []
        self.cam_viewports    = []
        for cam_idx in range(self.n_cam):
            cam_name = self.cam_names[cam_idx]
            cam      = mujoco.MjvCamera()
            cam.fixedcamid = self.model.cam(cam_name).id
            cam.type       = mujoco.mjtCamera.mjCAMERA_FIXED
            cam_fov        = self.model.cam_fovy[cam_idx]
            viewport       = mujoco.MjrRect(0,0,800,600) # SVGA?
            # Append
            self.cams.append(cam)
            self.cam_fovs.append(cam_fov)
            self.cam_viewports.append(viewport)

        # Build actuator-centric joint metadata once so later helpers do not
        # repeatedly traverse MuJoCo transmission fields.
        self.ctrl_qpos_idxs  = [] # qpos index attached to control, -1 if not attached to any joint
        self.ctrl_qpos_names = [] # qpos name attached to control, empty string if not attached to any joint
        self.ctrl_qpos_mins  = [] # minimum qpos value attached to control
        self.ctrl_qpos_maxs  = [] # maximum qpos value attached to control
        self.ctrl_qvel_idxs  = [] # qvel index attached to control, -1 if not attached to any joint
        self.ctrl_types      = [] # type of control (JOINT, TENDON, UNKNOWN)

        for ctrl_idx in range(self.n_ctrl):
            trntype = self.model.actuator_trntype[ctrl_idx]

            if trntype == mujoco.mjtTrn.mjTRN_JOINT:
                joint_idx = self.model.actuator(self.ctrl_names[ctrl_idx]).trnid[0]

                self.ctrl_qpos_idxs.append(self.model.jnt_qposadr[joint_idx])
                self.ctrl_qpos_names.append(self.joint_names[joint_idx])
                self.ctrl_qpos_mins.append(self.joint_ranges[joint_idx,0])
                self.ctrl_qpos_maxs.append(self.joint_ranges[joint_idx,1])
                self.ctrl_qvel_idxs.append(self.model.jnt_dofadr[joint_idx])
                self.ctrl_types.append('JOINT')

            elif trntype == mujoco.mjtTrn.mjTRN_TENDON:
                self.ctrl_qpos_idxs.append(-1)
                self.ctrl_qpos_names.append('')
                self.ctrl_qpos_mins.append(np.nan)
                self.ctrl_qpos_maxs.append(np.nan)
                self.ctrl_qvel_idxs.append(-1)
                self.ctrl_types.append('TENDON')

            else:
                self.ctrl_qpos_idxs.append(-1)
                self.ctrl_qpos_names.append('')
                self.ctrl_qpos_mins.append(np.nan)
                self.ctrl_qpos_maxs.append(np.nan)
                self.ctrl_qvel_idxs.append(-1)
                self.ctrl_types.append('UNKNOWN (trntype=%d)'%(trntype))

        # Semantic aliases for control to joint mapping
        self.ctrl_joint_names = self.ctrl_qpos_names.copy()
        self.ctrl_joint_mins  = self.ctrl_qpos_mins.copy()
        self.ctrl_joint_maxs  = self.ctrl_qpos_maxs.copy()
        self.ctrl_qpos_name_to_id = {
            name:idx for idx,name in enumerate(self.ctrl_qpos_names) if name != ''
        }

        # Mirror the mapping in the opposite direction because notebook code
        # often starts from a joint selection and asks which actuators drive it.
        self.joint_ctrl_idxs   = [] # control indices attached to each joint
        self.joint_ctrl_names  = [] # control names attached to each joint
        self.joint_ctrl_qpos_idxs = [] # qpos indices attached to each joint through controls
        self.joint_ctrl_qvel_idxs = [] # qvel indices attached to each joint through controls
        self.joint_ctrl_types  = [] # control types attached to each joint

        for joint_idx in range(self.n_joint):
            ctrl_idxs_for_joint      = []
            ctrl_names_for_joint     = []
            ctrl_qpos_idxs_for_joint = []
            ctrl_qvel_idxs_for_joint = []
            ctrl_types_for_joint     = []

            for ctrl_idx in range(self.n_ctrl):
                trntype = self.model.actuator_trntype[ctrl_idx]

                # Only joint transmission is directly mapped to a joint
                if trntype != mujoco.mjtTrn.mjTRN_JOINT:
                    continue

                attached_joint_idx = self.model.actuator(self.ctrl_names[ctrl_idx]).trnid[0]

                if attached_joint_idx == joint_idx:
                    ctrl_idxs_for_joint.append(ctrl_idx)
                    ctrl_names_for_joint.append(self.ctrl_names[ctrl_idx])
                    ctrl_qpos_idxs_for_joint.append(self.model.jnt_qposadr[joint_idx])
                    ctrl_qvel_idxs_for_joint.append(self.model.jnt_dofadr[joint_idx])
                    ctrl_types_for_joint.append('JOINT')

            self.joint_ctrl_idxs.append(ctrl_idxs_for_joint)
            self.joint_ctrl_names.append(ctrl_names_for_joint)
            self.joint_ctrl_qpos_idxs.append(ctrl_qpos_idxs_for_joint)
            self.joint_ctrl_qvel_idxs.append(ctrl_qvel_idxs_for_joint)
            self.joint_ctrl_types.append(ctrl_types_for_joint)

        # Position-style actuator subset
        # Practical rule:
        # - joint transmission only
        # - non-zero gain term
        # - non-zero q-position bias term
        # This excludes common velocity actuators that usually use only qvel bias.
        # Tuning helpers use this subset to avoid touching tendon actuators or
        # non-position-style controllers by default.
        self.position_ctrl_idxs        = []
        self.position_ctrl_names       = []
        self.position_ctrl_joint_names = []
        for ctrl_idx in range(self.n_ctrl):
            if self.ctrl_types[ctrl_idx] != 'JOINT':
                continue

            gainprm = np.asarray(self.model.actuator_gainprm[ctrl_idx],dtype=np.float64)
            biasprm = np.asarray(self.model.actuator_biasprm[ctrl_idx],dtype=np.float64)

            is_position_style = (
                np.isfinite(gainprm[0]) and
                (np.abs(gainprm[0]) > 0.0) and
                (np.abs(biasprm[1]) > 0.0)
            )
            if not is_position_style:
                continue

            self.position_ctrl_idxs.append(ctrl_idx)
            self.position_ctrl_names.append(self.ctrl_names[ctrl_idx])
            self.position_ctrl_joint_names.append(self.ctrl_joint_names[ctrl_idx])

        # Sensor information
        self.n_sensor      = self.model.nsensor
        self.sensor_names  = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_SENSOR,sensor_idx)
                              for sensor_idx in range(self.n_sensor)]
        self.sensor_name_to_id = {name:idx for idx,name in enumerate(self.sensor_names) if name is not None}

        # Site information
        self.n_site        = self.model.nsite
        self.site_names    = [mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_SITE,site_idx)
                              for site_idx in range(self.n_site)]
        self.site_name_to_id = {name:idx for idx,name in enumerate(self.site_names) if name is not None}

        # Geometry original parameters
        self.geom_friction_0 = self.model.geom_friction.copy()
        self.geom_rgba_0     = self.model.geom_rgba.copy()

        # Cache the body tree once so subtree queries stay cheap and name-based.
        self.body_children_list = [[] for _ in range(self.model.nbody)]
        for cid in range(1,self.model.nbody):
            pid = int(self.model.body_parentid[cid])
            self.body_children_list[pid].append(cid)

    def print_info(self):
        """
        Print out a comprehensive summary of the model's configuration.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        print("")
        print("-----------------------------------------------------------------------------")
        print("name:[%s] dt:[%.3f] HZ:[%d]" % (self.model_name, self.dt, self.HZ))
        print(" n_q:[%d] n_v:[%d] n_u:[%d]" % (self.n_q, self.n_v, self.n_u))
        print(" integrator:[%s]" % (self.integrator_name))

        # ------------------------------------------------------------------
        # Bodies
        # ------------------------------------------------------------------
        print("")
        print("Body information. n_body:[%d]" % (self.n_body))
        for body_idx, body_name in enumerate(self.body_names):
            body_mass = self.body_masses[body_idx]
            parent_name = self.parent_body_names[body_idx]
            print(" [%d/%d] [%s] parent:[%s] mass:[%.2f]kg" %
                (body_idx, self.n_body, body_name, parent_name, body_mass))
        print("body_total_mass:[%.2f]kg" % (self.body_total_mass))

        # ------------------------------------------------------------------
        # Geometry
        # ------------------------------------------------------------------
        print("")
        print("Geometry information. n_geom:[%d]" % (self.n_geom))
        for geom_idx, geom_name in enumerate(self.geom_names):
            body_id = self.model.geom_bodyid[geom_idx]
            body_name = self.body_names[body_id]
            print(" [%d/%d] [%s] attached_body:[%s]" %
                (geom_idx, self.n_geom, geom_name, body_name))

        # ------------------------------------------------------------------
        # Mesh
        # ------------------------------------------------------------------
        print("")
        print("Mesh information. n_mesh:[%d]" % (self.n_mesh))
        for mesh_idx, mesh_name in enumerate(self.mesh_names):
            print(" [%d/%d] [%s]" %
                (mesh_idx, self.n_mesh, mesh_name))

        # ------------------------------------------------------------------
        # Joints (all)
        # ------------------------------------------------------------------
        print("")
        print("Joint information. n_joint:[%d]" % (self.n_joint))
        for joint_idx, joint_name in enumerate(self.joint_names):
            axis = self.model.joint(joint_idx).axis
            print(" [%d/%d] [%s] axis:%s" %
                (joint_idx, self.n_joint, joint_name, axis))

        # ------------------------------------------------------------------
        # Free joints
        # ------------------------------------------------------------------
        print("\nFree joint information. n_free_joint:[%d]" % (self.n_free_joint))
        for idx, joint_name in enumerate(self.free_joint_names):
            body_id = self.model.joint(self.free_joint_idxs[idx]).bodyid[0]
            body_name = self.body_names[body_id]
            print(" [%d/%d] [%s] body_attached:[%s]" %
                (idx, self.n_free_joint, joint_name, body_name))

        # ------------------------------------------------------------------
        # Revolute joints
        # ------------------------------------------------------------------
        print("\nRevolute joint information. n_rev_joint:[%d]" % (self.n_rev_joint))
        for idx, joint_name in enumerate(self.rev_joint_names):
            print(" [%d/%d] [%s] range:[%.3f]~[%.3f]" %
                (idx, self.n_rev_joint, joint_name,
                self.rev_joint_mins[idx], self.rev_joint_maxs[idx]))

        # ------------------------------------------------------------------
        # Prismatic joints
        # ------------------------------------------------------------------
        print("\nPrismatic joint information. n_pri_joint:[%d]" % (self.n_pri_joint))
        for idx, joint_name in enumerate(self.pri_joint_names):
            print(" [%d/%d] [%s] range:[%.3f]~[%.3f]" %
                (idx, self.n_pri_joint, joint_name,
                self.pri_joint_mins[idx], self.pri_joint_maxs[idx]))

        # ------------------------------------------------------------------
        # Revolute + Prismatic joints
        # ------------------------------------------------------------------
        print("\nRevolute+Prismatic joint information. n_rev_pri_joint:[%d]" % (self.n_rev_pri_joint))
        for idx, joint_name in enumerate(self.rev_pri_joint_names):
            print(" [%d/%d] [%s] range:[%.3f]~[%.3f]" %
                (idx, self.n_rev_pri_joint, joint_name,
                self.rev_pri_joint_mins[idx], self.rev_pri_joint_maxs[idx]))

        # ------------------------------------------------------------------
        # Active / Passive joints
        # ------------------------------------------------------------------
        print("\nActive joint information. n_active_joint:[%d]" % (self.n_active_joint))
        for idx, joint_name in enumerate(self.active_joint_names):
            print(" [%d/%d] [%s] range:[%.3f]~[%.3f]" %
                (idx, self.n_active_joint, joint_name,
                self.active_joint_mins[idx], self.active_joint_maxs[idx]))

        print("\nPassive joint information. n_passive_joint:[%d]" % (self.n_passive_joint))
        for idx, joint_name in enumerate(self.passive_joint_names):
            print(" [%d/%d] [%s] range:[%.3f]~[%.3f]" %
                (idx, self.n_passive_joint, joint_name,
                self.passive_joint_mins[idx], self.passive_joint_maxs[idx]))

        # ------------------------------------------------------------------
        # Actuators / Controls
        # ------------------------------------------------------------------
        print("\nControl information. n_ctrl:[%d]" % (self.n_ctrl))
        for idx, ctrl_name in enumerate(self.ctrl_names):
            print(" [%d/%d] [%s] range:[%.3f]~[%.3f] gear:[%.2f] type:[%s]" %
                (idx, self.n_ctrl, ctrl_name,
                self.ctrl_mins[idx], self.ctrl_maxs[idx],
                self.ctrl_gears[idx], self.ctrl_types[idx]))

        # ------------------------------------------------------------------
        # Control-to-joint mapping
        # ------------------------------------------------------------------
        print("\nControl-to-joint mapping.")
        for idx in range(self.n_ctrl):
            print(" [%d/%d] ctrl:[%s] qpos_idx:[%d] qpos_name:[%s] qpos_range:[%.3f]~[%.3f] qvel_idx:[%d] type:[%s]" %
                (idx, self.n_ctrl,
                self.ctrl_names[idx],
                self.ctrl_qpos_idxs[idx],
                self.ctrl_qpos_names[idx],
                self.ctrl_qpos_mins[idx],
                self.ctrl_qpos_maxs[idx],
                self.ctrl_qvel_idxs[idx],
                self.ctrl_types[idx]))

        # ------------------------------------------------------------------
        # Joint-to-control mapping
        # ------------------------------------------------------------------
        print("\nJoint-to-control mapping.")
        for idx in range(self.n_joint):
            print(" [%d/%d] joint:[%s] ctrl_idxs:%s ctrl_names:%s" %
                (idx, self.n_joint,
                self.joint_names[idx],
                self.joint_ctrl_idxs[idx],
                self.joint_ctrl_names[idx]))

        # ------------------------------------------------------------------
        # Cameras
        # ------------------------------------------------------------------
        print("\nCamera information. n_cam:[%d]" % (self.n_cam))
        for idx, cam_name in enumerate(self.cam_names):
            print(" [%d/%d] [%s] fov:[%.1f] viewport:%s" %
                (idx, self.n_cam, cam_name, self.cam_fovs[idx], self.cam_viewports[idx]))

        # ------------------------------------------------------------------
        # Sensors
        # ------------------------------------------------------------------
        print("")
        print("Sensor information. n_sensor:[%d]" % (self.n_sensor))
        for idx, sensor_name in enumerate(self.sensor_names):
            print(" [%d/%d] [%s]" % (idx, self.n_sensor, sensor_name))

        # ------------------------------------------------------------------
        # Sites
        # ------------------------------------------------------------------
        print("")
        print("Site information. n_site:[%d]" % (self.n_site))
        for idx, site_name in enumerate(self.site_names):
            print(" [%d/%d] [%s]" % (idx, self.n_site, site_name))

        print("-----------------------------------------------------------------------------")

    def plot_body_graph(
            self,
            root_body_name  = "world",
            align           = "horizontal",
            figsize         = (6,4),
            node_size       = 700,
            font_size_node  = 6,
            node_colors     = None,
            font_size_title = 8,
            root_on_top     = True,
        ):
        """
        Plot the MuJoCo body tree graph using matplotlib.

        Parameters:
            root_body_name (str): Input root body name. Defaults to `'world'`.
            align (str): Input align. Defaults to `'horizontal'`.
            figsize (tuple): Input figsize. Defaults to `(6, 4)`.
            node_size (int): Input node size. Defaults to `700`.
            font_size_node (int): Input font size node. Defaults to `6`.
            node_colors (Any | None): Input node colors. Defaults to `None`.
            font_size_title (int): Input font size title. Defaults to `8`.
            root_on_top (bool): Input root on top. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        body_names = self.get_subtree_body_names(
            root_body_name = root_body_name,
            include_root   = True,
        )
        body_name_set = set(body_names)

        tree = nx.DiGraph(name="%s body graph"%(self.name))
        tree.add_nodes_from(body_names)

        for body_name in body_names:
            body_idx = self.body_name_to_id[body_name]
            parent_body_name = self.parent_body_names[body_idx]
            if parent_body_name in body_name_set and parent_body_name != body_name:
                tree.add_edge(parent_body_name,body_name)

        children_dict = {}
        for body_name in body_names:
            body_idx = self.body_name_to_id[body_name]
            children_dict[body_name] = [
                self.body_names[child_idx]
                for child_idx in self.body_children_list[body_idx]
                if self.body_names[child_idx] in body_name_set
            ]

        depth_dict = {}
        visit_order = []
        stack = [(root_body_name,0)]
        while stack:
            body_name,depth = stack.pop()
            depth_dict[body_name] = depth
            visit_order.append(body_name)
            for child_body_name in reversed(children_dict[body_name]):
                stack.append((child_body_name,depth+1))

        post_order = []
        stack = [(root_body_name,False)]
        while stack:
            body_name,visited = stack.pop()
            if visited:
                post_order.append(body_name)
                continue
            stack.append((body_name,True))
            for child_body_name in reversed(children_dict[body_name]):
                stack.append((child_body_name,False))

        x_dict = {}
        x_leaf = 0.0
        for body_name in post_order:
            child_body_names = children_dict[body_name]
            if len(child_body_names) == 0:
                x_dict[body_name] = x_leaf
                x_leaf = x_leaf + 1.0
            else:
                x_dict[body_name] = float(np.mean([
                    x_dict[child_body_name] for child_body_name in child_body_names
                ]))

        x_center = 0.5*max(x_leaf-1.0,0.0)
        if align == "horizontal":
            pos = {
                body_name:(x_dict[body_name]-x_center,-depth_dict[body_name])
                for body_name in body_names
            }
            if not root_on_top:
                pos = {node:(x,-y) for node,(x,y) in pos.items()}
        elif align == "vertical":
            pos = {
                body_name:(depth_dict[body_name],-(x_dict[body_name]-x_center))
                for body_name in body_names
            }
            if not root_on_top:
                pos = {node:(-x,y) for node,(x,y) in pos.items()}
        else:
            raise ValueError("Unsupported align:[%s]. Use 'horizontal' or 'vertical'."%(align))

        fig,ax = plt.subplots(figsize=figsize)

        if node_colors is None:
            node_colors = [(1,1,1,0.5) for _ in tree.nodes]

        nx.draw_networkx(
            tree,
            pos         = pos,
            ax          = ax,
            with_labels = True,
            node_size   = node_size,
            font_size   = font_size_node,
            node_color  = node_colors,
            linewidths  = 1,
            edgecolors  = "k",
        )

        ax.set_title("%s"%(tree.name),fontsize=font_size_title)
        ax.set_axis_off()
        ax.margins(0.15)
        fig.tight_layout()
        plt.show()

    def reset(self,forward=True,step=False,clear_cache=True,sleep_sec=0.0):
        """
        Reset the MuJoCo simulation data to the initial state.

        Parameters:
            forward (bool): Input forward. Defaults to `True`.
            step (bool): Input step. Defaults to `False`.
            clear_cache (bool): Input clear cache. Defaults to `True`.
            sleep_sec (float): Input sleep sec. Defaults to `0.0`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if sleep_sec > 0.0:
            time.sleep(float(sleep_sec))
        mujoco.mj_resetData(self.model,self.data) # reset data
        if forward:
            # self.data.qpos = self.model.qpos0.copy()
            self.data.qpos[:] = self.q0
            mujoco.mj_forward(self.model,self.data)
        if step:
            mujoco.mj_step(self.model,self.data) # step once to update

        # Reset wall time
        self.init_sim_time    = self.data.time
        self.init_wall_time   = time.time()
        self.accum_wall_time  = 0.0  # accumulated wall time
        self.last_wall_update = time.time()

        # Reset double click info
        self.xyz_left_double_click  = None
        self.xyz_right_double_click = None

        # Clear object pointcloud cache
        if clear_cache:
            self.clear_obj_pcd_cache()

        # Reset final RGB image
        self.final_rgb_img = None

    def open_legacy_viewer(self):
        """
        Launch an legacy viewer for the simulation.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        from mujoco import viewer
        viewer.launch(self.model)

    def init_viewer(
            self,
            title            = None,
            width            = 0.6,
            height           = 1.0,
            x_offset         = None,
            y_offset         = None,
            maxgeom          = 50000,
            perturbation     = True,
            fontscale        = 200,
        ):
        """
        Initialize the MuJoCo viewer.

        Parameters:
            title (Any | None): Input title. Defaults to `None`.
            width (float): Input width. Defaults to `0.6`.
            height (float): Input height. Defaults to `1.0`.
            x_offset (Any | None): Input x offset. Defaults to `None`.
            y_offset (Any | None): Input y offset. Defaults to `None`.
            maxgeom (int): Input maxgeom. Defaults to `50000`.
            perturbation (bool): Input perturbation. Defaults to `True`.
            fontscale (int): Input fontscale. Defaults to `200`.

        Returns:
            Any: Result computed from the current inputs.
        """

        # Create viewer instance
        self.use_mujoco_viewer = True

        # Set title
        if title is None: title = self.name

        # Notebook code usually passes monitor-relative ratios, while scripts may
        # pass absolute pixels. Support both without changing the API.
        if width <= 1.0 and height <= 1.0:
            w_monitor,h_monitor = get_monitor_size()
            width  = int(width*w_monitor)
            height = int(height*h_monitor)

        # Initialize viewer
        self.viewer = MuJoCoMinimalViewer(
            self.model,
            self.data,
            mode              = 'window',
            title             = title,
            width             = width,
            height            = height,
            x_offset          = x_offset,
            y_offset          = y_offset,
            maxgeom           = maxgeom,
            perturbation      = perturbation,
        )
        self.viewer.ctx = mujoco.MjrContext(self.model,fontscale)

        # Set some default viewer attributes
        self.viewer.set_transparency(transparent=False)
        self.viewer.set_cam_info(azimuth=173,distance=3.56,elevation=-22.56,lookat=[0,0,0.59])

        """
        Usual convention for geom groups:
            - group_0: floor or ground
            - group_1: visual and collision (e.g., box or cylinder)
            - group_2: visual only (e.g., accurate mesh) => contype="0" conaffinity="0"
            - group_3: collision only (e.g., simplified collision mesh) <= viewer off by default
        """
        self.viewer.set_geomgroup(group_0=True,group_1=True,group_2=True,group_3=False)

        # Hide site group 0 and 1 by default
        self.viewer.set_sitegroup(group_0=False,group_1=False)

    def is_viewer_alive(self):
        """
        Check if the viewer window is active.

        Parameters:
            None.

        Returns:
            bool: Boolean result for the requested condition.
        """
        if self.viewer is None:
            return False

        # Honor GLFW close requests immediately even when render() is throttled.
        # Without this, pressing ESC can leave the outer loop spinning until the
        # next scheduled render call notices window_should_close().
        try:
            if (
                getattr(self.viewer,"is_alive",False)
                and getattr(self.viewer,"window",None) is not None
                and glfw.window_should_close(self.viewer.window)
            ):
                self.close_viewer()
                return False
        except Exception:
            pass

        # Update loop timing
        """
        Update loop timing information for performance monitoring.
            - loop_dt: Time difference between current and last alive check.
            - loop_hz: Raw frequency (Hz) calculated from loop_dt.
            - loop_hz_ema: Exponentially moving average of loop_hz for smoother tracking.
        """
        now = time.perf_counter()
        if self._last_alive_t is not None:
            self.loop_dt = now - self._last_alive_t
            if self.loop_dt > 0.0:
                # raw Hz
                self.loop_hz = 1.0 / self.loop_dt
                # EMA Hz
                if self.loop_hz_ema == 0.0:
                    self.loop_hz_ema = self.loop_hz
                else:
                    a = self._loop_hz_ema_alpha
                    self.loop_hz_ema = (1.0 - a) * self.loop_hz_ema + a * self.loop_hz
        self._last_alive_t = now

        # Return alive status
        return self.viewer.is_alive

    def close_viewer(self):
        """
        Close the MuJoCo viewer and release resources.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        # self.use_mujoco_viewer = False
        # # Capture final RGB image
        # if self.final_rgb_img is None:
        #     self.final_rgb_img,_ = self.grab_current_rgbd_imgs()
        # # If viewer attribute does not exist
        # if not hasattr(self,'viewer'):
        #     return
        # # If viewer is not initialized
        # if self.viewer is None:
        #     return
        # # Close viewer
        # self.viewer.close()
        self.use_mujoco_viewer = False
        if not hasattr(self,"viewer"): # if viewer attribute does not exist,
            return
        if self.viewer is None: # if viewer is not initialized,
            return
        if not getattr(self.viewer,"is_alive",False): # if viewer is not alive,
            # self.viewer = None
            return
        if self.final_rgb_img is None: # if final RGB image is not captured yet,
            self.final_rgb_img,_ = self.capture_clean_viewer_rgbd_imgs()
        self.viewer.close()
        # self.viewer = None

    def capture_clean_viewer_rgbd_imgs(self):
        """
        Render one clean viewer frame without text/RGB overlays and capture it.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        if (
            not hasattr(self,"viewer")
            or self.viewer is None
            or not getattr(self.viewer,"is_alive",False)
        ):
            raise RuntimeError("[capture_clean_viewer_rgbd_imgs] Viewer is not alive.")

        window = self.viewer.window
        should_close_prev = False
        try:
            should_close_prev = glfw.window_should_close(window)
        except Exception:
            pass

        # Clear transient overlays first; otherwise old framebuffer content can
        # leak into the final screenshot.
        if hasattr(self.viewer,"_markers") and self.viewer._markers is not None:
            self.viewer._markers[:] = []
        if hasattr(self.viewer,"_overlay") and self.viewer._overlay is not None:
            self.viewer._overlay.clear()
        if hasattr(self.viewer,"reset_rgb_overlay"):
            self.viewer.reset_rgb_overlay()

        try:
            glfw.set_window_should_close(window,False)
        except Exception:
            pass

        self.viewer.render()
        rgb_img,depth_img = self.grab_current_rgbd_imgs()

        try:
            glfw.set_window_should_close(window,should_close_prev)
        except Exception:
            pass

        return rgb_img,depth_img

    def viewer_text_overlay(
            self,
            text1 = '',
            text2 = '',
            loc   = 'bottom left',
        ):
        """
        Add a text overlay on the viewer window.

        Parameters:
            text1 (str): Input text1. Defaults to `''`.
            text2 (str): Input text2. Defaults to `''`.
            loc (str): Input loc. Defaults to `'bottom left'`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.viewer.add_overlay(loc=loc,text1=text1,text2=text2)

    def viewer_rgb_overlay(
            self,
            rgb = None,
            loc = 'top right',
        ):
        """
        Display an RGB image overlay in the viewer.

        Parameters:
            rgb (Any | None): Input rgb. Defaults to `None`.
            loc (str): Input loc. Defaults to `'top right'`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if rgb is None:
            return
        rgb = np.asarray(rgb)
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[:, :, None], 3, axis=2)
        self.viewer.plot_rgb_overlay(rgb=rgb,loc=loc)

    def render(self, reset_rgb_overlay=True):
        """
        Render the current simulation state to the viewer.

        Parameters:
            reset_rgb_overlay (bool): Input reset rgb overlay. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        # Preserve the last clean frame before shutdown so notebooks can inspect
        # the terminal state after the viewer loop exits.
        if glfw.KEY_ESCAPE in self.get_key_pressed_list():
            print ("Final RGB image captured.")
            if self.final_rgb_img is None:
                self.final_rgb_img,_ = self.capture_clean_viewer_rgbd_imgs()

        # Render
        if self.use_mujoco_viewer:
            # Assert
            assert (
                hasattr(self,"viewer")
                and self.viewer is not None
                and getattr(self.viewer,"is_alive",False)
            ), "[%s] Mujoco viewer is not alive."%(self.name)

            # Render
            self.viewer.render(reset_rgb_overlay=reset_rgb_overlay)
        else:
            # Viewer not initialized
            print ("[%s] Viewer NOT initialized."%(self.name))

    def update_render_settings(self):
        """
        Update the rendering settings in the viewer.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        mujoco.mj_forward(self.model,self.data)
        mujoco.mjv_updateScene(
            self.model,self.data,self.viewer.vopt,self.viewer.pert,self.viewer.cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,self.viewer.scn)
        mujoco.mjr_render(self.viewer.viewport,self.viewer.scn,self.viewer.ctx)

    def step(
            self,
            ctrl          = None,
            ctrl_idxs     = None,
            ctrl_names    = None,
            joint_names   = None,
            nstep         = 1,
            step_flag     = True,
        ):
        """
        Advance the simulation by a certain number of steps, optionally applying
        control inputs.

        Parameters:
            ctrl (Any | None): Input ctrl. Defaults to `None`.
            ctrl_idxs (Any | None): Input ctrl idxs. Defaults to `None`.
            ctrl_names (Any | None): Input ctrl names. Defaults to `None`.
            joint_names (Any | None): Input joint names. Defaults to `None`.
            nstep (int): Input nstep. Defaults to `1`.
            step_flag (bool): Input step flag. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        # Apply control inputs and step simulation
        if step_flag:
            if ctrl is not None:
                # determine control indices
                if ctrl_names is not None: ctrl_idxs = get_idxs(self.ctrl_names,ctrl_names)
                elif joint_names is not None: ctrl_idxs = self.get_idxs_step(joint_names=joint_names)
                # set control inputs
                if ctrl_idxs is None:  self.data.ctrl[:] = ctrl
                else: self.data.ctrl[ctrl_idxs] = ctrl
            # Forward simulation
            mujoco.mj_step(self.model,self.data,nstep=nstep)

        # Update wall time
        self.increase_wall_time(step_flag=step_flag)

    def forward(
            self,
            q             = None,
            joint_names   = None,
            joint_idxs    = None,
            clip_position = True,
            rev_unit      = "rad",
            pri_unit      = "m",
        ):
        """
        Perform forward kinematics, optionally updating the state qpos.

        Parameters:
            q (Any | None): Input q. Defaults to `None`.
            joint_names (Any | None): Input joint names. Defaults to `None`.
            joint_idxs (Any | None): Input joint idxs. Defaults to `None`.
            clip_position (bool): Input clip position. Defaults to `True`.
            rev_unit (str): Input rev unit. Defaults to `'rad'`.
            pri_unit (str): Input pri unit. Defaults to `'m'`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        rev_unit = str(rev_unit).lower()
        pri_unit = str(pri_unit).lower()
        if rev_unit not in ("rad","deg"):
            raise ValueError("rev_unit must be either 'rad' or 'deg'.")
        if pri_unit not in ("m","cm","mm"):
            raise ValueError("pri_unit must be one of 'm', 'cm', or 'mm'.")
        pri_scale = {"m":1.0,"cm":0.01,"mm":0.001}[pri_unit]

        # Update qpos if provided
        if q is not None:
            if joint_names is not None: # if 'joint_names' is not None, it override 'joint_idxs'
                joint_idxs = self.get_idxs_fwd(joint_names=joint_names)
            if joint_idxs is not None:
                joint_idxs = list(joint_idxs)
                q_set = np.asarray(q,dtype=np.float64).copy()
                if q_set.ndim == 0:
                    q_set = np.full(len(joint_idxs),float(q_set),dtype=np.float64)
                elif q_set.size == 1 and len(joint_idxs) != 1:
                    q_set = np.full(len(joint_idxs),float(q_set.reshape(-1)[0]),dtype=np.float64)
                if rev_unit == "deg":
                    rev_mask = np.array([int(idx) in self.rev_qpos_set for idx in joint_idxs],dtype=bool)
                    q_set[rev_mask] = np.deg2rad(q_set[rev_mask])
                if pri_unit != "m":
                    pri_mask = np.array([int(idx) in self.pri_qpos_set for idx in joint_idxs],dtype=bool)
                    q_set[pri_mask] = pri_scale*q_set[pri_mask]
                self.data.qpos[joint_idxs] = q_set
            else:
                q_set = np.asarray(q,dtype=np.float64).copy()
                if rev_unit == "deg":
                    q_set[self.rev_qpos_idxs] = np.deg2rad(q_set[self.rev_qpos_idxs])
                if pri_unit != "m":
                    q_set[self.pri_qpos_idxs] = pri_scale*q_set[self.pri_qpos_idxs]
                self.data.qpos[:] = q_set

        # Clip qpos to joint limits
        if clip_position:
            self.data.qpos[self.rev_pri_qpos_idxs] = np.clip(
                self.data.qpos[self.rev_pri_qpos_idxs],
                self.rev_pri_joint_mins,
                self.rev_pri_joint_maxs,
            )

        # Perform forward kinematics
        mujoco.mj_forward(self.model,self.data)

        # Update wall time
        self.increase_wall_time()

    def sync_from(
            self,
            src_env,
            forward      = True,
            model_params = True,
        ):
        """
        Synchronize this environment's state from another MuJoCoParser.

        Parameters:
            src_env (Any): Input src env.
            forward (bool): Input forward. Defaults to `True`.
            model_params (bool): Input model params. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if self.model.nq != src_env.model.nq:
            raise ValueError("Cannot sync envs with different nq.")
        if self.model.nv != src_env.model.nv:
            raise ValueError("Cannot sync envs with different nv.")
        if self.model.nu != src_env.model.nu:
            raise ValueError("Cannot sync envs with different nu.")
        if self.model.nbody != src_env.model.nbody:
            raise ValueError("Cannot sync envs with different nbody.")
        if self.model.nmocap != src_env.model.nmocap:
            raise ValueError("Cannot sync envs with different nmocap.")

        self.data.time = float(src_env.data.time)
        self.data.qpos[:] = src_env.data.qpos
        self.data.qvel[:] = src_env.data.qvel
        self.data.qacc[:] = src_env.data.qacc
        self.data.ctrl[:] = src_env.data.ctrl
        self.data.qfrc_applied[:] = src_env.data.qfrc_applied
        self.data.xfrc_applied[:] = src_env.data.xfrc_applied
        self.data.mocap_pos[:] = src_env.data.mocap_pos
        self.data.mocap_quat[:] = src_env.data.mocap_quat
        if hasattr(self.data,"act") and self.data.act.shape == src_env.data.act.shape:
            self.data.act[:] = src_env.data.act
        if hasattr(self.data,"qacc_warmstart") and self.data.qacc_warmstart.shape == src_env.data.qacc_warmstart.shape:
            self.data.qacc_warmstart[:] = src_env.data.qacc_warmstart
        if hasattr(self.data,"userdata") and self.data.userdata.shape == src_env.data.userdata.shape:
            self.data.userdata[:] = src_env.data.userdata
        if hasattr(self.data,"eq_active") and self.data.eq_active.shape == src_env.data.eq_active.shape:
            self.data.eq_active[:] = src_env.data.eq_active

        if model_params:
            self.model.geom_friction[:] = src_env.model.geom_friction
            self.model.actuator_gainprm[:] = src_env.model.actuator_gainprm
            self.model.actuator_biasprm[:] = src_env.model.actuator_biasprm
            self.model.actuator_forcerange[:] = src_env.model.actuator_forcerange
            self.model.actuator_ctrlrange[:] = src_env.model.actuator_ctrlrange

        if forward:
            mujoco.mj_forward(self.model,self.data)

    def set_zero_qvel(self):
        """
        Set all joint velocities to zero and perform forward kinematics.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model,self.data)

    def compensate_gravity(self,root_body_names,on=True,exclude_free_joint=True):
        """
        Apply gravity compensation to the given root bodies.

        Parameters:
            root_body_names (str | Path): Input root body names.
            on (bool): Input on. Defaults to `True`.
            exclude_free_joint (bool): Input exclude free joint. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """

        # Always clear first so the result is not accumulated from previous calls
        self.data.qfrc_applied[:] = 0.0

        # Early return for OFF mode
        if not on:
            return
        jac = np.zeros((3,self.model.nv),dtype=np.float64) # (3, n_v) Jacobian for center of mass

        for root_body_name in root_body_names:
            subtree_id = self.model.body(root_body_name).id
            total_mass = float(self.model.body_subtreemass[subtree_id])

            mujoco.mj_jacSubtreeCom(
                self.model,
                self.data,
                jac,
                subtree_id,
            )

            f_grav = total_mass*self.model.opt.gravity
            qfrc_sub = -(jac.T@f_grav)

            if exclude_free_joint and (self.n_free_joint>0):
                qfrc_sub[self.free_dof_idxs] = 0.0

            self.data.qfrc_applied[:] += qfrc_sub

    def get_idxs_fwd(self,joint_names):
        """
        Get the indices of joints used for forward kinematics based on joint names.

        Parameters:
            joint_names (list[str]): Input joint names.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        return [int(self.model.jnt_qposadr[self.joint_name_to_id[jname]]) for jname in joint_names]

    def get_idxs_step(self,joint_names):
        """
        Get the indices used for applying control during simulation steps based on
        joint names.

        Parameters:
            joint_names (list[str]): Input joint names.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        return [self.ctrl_qpos_name_to_id[jname] for jname in joint_names]

    def get_idxs_jac(self,joint_names):
        """
        Get DOF indices (Jacobian column indices) for given joint names.

        Parameters:
            joint_names (list[str]): Input joint names.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        idxs=[]
        for jname in joint_names:
            jidx = self.joint_name_to_id[jname]
            jtype=int(self.model.jnt_type[jidx])
            dofadr=int(self.model.jnt_dofadr[jidx])

            if jtype==mujoco.mjtJoint.mjJNT_FREE:
                dof_count=6
            elif jtype==mujoco.mjtJoint.mjJNT_BALL:
                dof_count=3
            else:
                # HINGE / SLIDE
                dof_count=1

            idxs.extend(range(dofadr,dofadr+dof_count))
        return idxs

    def get_sim_time(self,init_flag=False):
        """
        Get the simulation time elapsed since initialization.

        Parameters:
            init_flag (bool): Input init flag. Defaults to `False`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if init_flag:
            self.init_sim_time = self.data.time
        elapsed_time = self.data.time - self.init_sim_time
        return elapsed_time

    def reset_sim_time(self):
        """
        Reset the simulation time reference to the current data.time.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.init_sim_time = self.data.time

    def reset_wall_time(self):
        """
        Reset the wall-clock time reference.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        current_wall_time = time.time()
        self.init_wall_time = current_wall_time
        self.accum_wall_time = 0.0
        self.last_wall_update = current_wall_time

    def reset_sim_wall_time(self):
        """
        Reset both simulation time and wall-clock time references.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.reset_sim_time()
        self.reset_wall_time()

    def increase_wall_time(self,step_flag=True):
        """
        Update the accumulated wall-clock time.

        Parameters:
            step_flag (bool): Input step flag. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        current_wall_time = time.time()
        if step_flag:
            # Increment wall time only when 'step_flag' is True
            self.accum_wall_time += current_wall_time - self.last_wall_update
        self.last_wall_update = current_wall_time

    def get_wall_time(self,init_flag=False):
        """
        Get the wall-clock time elapsed since the last reset.

        Parameters:
            init_flag (bool): Input init flag. Defaults to `False`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if init_flag:
            self.accum_wall_time = 0.0
            self.last_wall_update = time.time()
        return self.accum_wall_time

    def get_key_pressed_list(self):
        """
        Get the list of keys that have been pressed.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if self.viewer is None: # if viewer is not initialized
            return []

        # Return
        return list(self.viewer._key_pressed_set)

    def get_key_repeated_list(self):
        """
        Get the list of keys that have been repeatedly pressed.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        return list(self.viewer._key_repeated_set)

    def pop_key_pressed_list(self,key=None):
        """
        Pop the last key from the pressed keys list.

        Parameters:
            key (Any | None): Input key. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if key is not None:
            self.viewer._key_pressed_set.discard(key)

    def is_key_pressed_once(self,key=None,key_list=None):
        """
        Check if a specific key has been pressed once.

        Parameters:
            key (Any | None): Input key. Defaults to `None`.
            key_list (Any | None): Input key list. Defaults to `None`.

        Returns:
            bool: Boolean result for the requested condition.
        """
        if key is not None:
            if key in self.get_key_pressed_list():
                self.pop_key_pressed_list(key=key)
                return True
            else:
                return False
        elif key_list is not None:
            for key in key_list:
                if key in self.get_key_pressed_list():
                    self.pop_key_pressed_list(key=key)
                    return True
            return False
        else:
            return False

    def is_space_clicked_once(self,button='left'):
        """
        Check if the space key has been clicked once.

        Parameters:
            button (str): Input button. Defaults to `'left'`.

        Returns:
            bool: Boolean result for the requested condition.
        """
        return self.is_key_pressed_once(glfw.KEY_SPACE)

    def is_key_pressed_repeat(self,key=None,key_list=None):
        """
        Check if specific key(s) have been pressed continuously.

        Parameters:
            key (Any | None): Input key. Defaults to `None`.
            key_list (Any | None): Input key list. Defaults to `None`.

        Returns:
            bool: Boolean result for the requested condition.
        """
        if key is not None:
            return key in self.get_key_pressed_list()+self.get_key_repeated_list()
        elif key_list is not None:
            for key in key_list:
                if key in self.get_key_pressed_list()+self.get_key_repeated_list():
                    return True
            return False
        else:
            return False

    def grab_current_rgbd_imgs(self):
        """
        Capture the current RGB and depth images from the viewer.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        rgb_img   = np.zeros((self.viewer.viewport.height,self.viewer.viewport.width,3),dtype=np.uint8)
        depth_img = np.zeros((self.viewer.viewport.height,self.viewer.viewport.width,1), dtype=np.float32)
        mujoco.mjr_readPixels(rgb_img,depth_img,self.viewer.viewport,self.viewer.ctx)
        rgb_img,depth_img = np.flipud(rgb_img),np.flipud(depth_img) # flip up-down

        # Rescale depth image
        extent = self.model.stat.extent
        near   = self.model.vis.map.znear * extent
        far    = self.model.vis.map.zfar * extent
        scaled_depth_img = near / (1 - depth_img * (1 - near / far))
        depth_img = scaled_depth_img.squeeze()
        return rgb_img,depth_img

    def get_qpos(self,joint_names=None,ctrl_names=None,flatten=True):
        """
        Get joint position(s) (qpos) from the simulation.

        Parameters:
            joint_names (Any | None): Input joint names. Defaults to `None`.
            ctrl_names (Any | None): Input ctrl names. Defaults to `None`.
            flatten (bool): Input flatten. Defaults to `True`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if joint_names is not None and ctrl_names is not None:
            raise ValueError("Provide only one of joint_names or ctrl_names.")

        # full qpos
        if joint_names is None and ctrl_names is None:
            return self.data.qpos.copy()

        # resolve ctrl_names -> joint_names
        if ctrl_names is not None:
            joint_names = []
            for cname in ctrl_names:
                act = self.model.actuator(cname)
                joint_id = int(act.trnid[0])
                if joint_id < 0:
                    raise ValueError(f"Actuator '{cname}' is not attached to a joint.")
                joint_names.append(self.joint_names[joint_id])

        # collect qpos slices for each joint
        q_list = []
        for jname in joint_names:
            j = self.model.joint(jname)
            addr = int(j.qposadr[0])
            L = int(len(j.qpos0))  # FREE=7, BALL=4, HINGE/SLIDE=1
            q_list.append(self.data.qpos[addr:addr+L].copy())

        # return
        if flatten:
            return np.concatenate(q_list) if q_list else np.array([])
        else:
            return np.array(q_list,dtype=object)

    def get_ctrl(self,ctrl_names=None):
        """
        Get control input values from the simulation.

        Parameters:
            ctrl_names (Any | None): Input ctrl names. Defaults to `None`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if ctrl_names is None:
            return self.data.ctrl.copy()
        ctrl_idxs = [self.ctrl_name_to_id[name] for name in ctrl_names]
        return self.data.ctrl[ctrl_idxs].copy()

    def set_ctrl(self,ctrl,ctrl_names=None,ctrl_idxs=None):
        """
        Set control input values without stepping the simulation.

        Parameters:
            ctrl (Any): Input ctrl.
            ctrl_names (Any | None): Input ctrl names. Defaults to `None`.
            ctrl_idxs (Any | None): Input ctrl idxs. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if ctrl_idxs is None and ctrl_names is not None:
            ctrl_idxs = [self.ctrl_name_to_id[name] for name in ctrl_names]
        if ctrl_idxs is None:
            self.data.ctrl[:] = ctrl
        else:
            self.data.ctrl[ctrl_idxs] = ctrl

    def get_ctrl_qpos(self,ctrl_names=None):
        """
        Get joint positions attached to actuators.

        Parameters:
            ctrl_names (Any | None): Input ctrl names. Defaults to `None`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if ctrl_names is None:
            ctrl_names = self.ctrl_names

        ctrl_idxs = get_idxs(self.ctrl_names,ctrl_names)

        # Reject non-joint transmissions such as tendon actuators
        unsupported_ctrls = []
        for ctrl_idx in ctrl_idxs:
            if self.ctrl_types[ctrl_idx] != 'JOINT':
                unsupported_ctrls.append(self.ctrl_names[ctrl_idx])

        if len(unsupported_ctrls) > 0:
            raise ValueError(
                "[get_ctrl_qpos] Unsupported control type for ctrl_names:%s. "
                "Only JOINT transmission is supported."%(unsupported_ctrls)
            )

        qpos_idxs = np.array(self.ctrl_qpos_idxs,dtype=np.int32)[ctrl_idxs]
        return self.data.qpos[qpos_idxs].copy()

    def write_ctrl_qpos(
            self,
            ctrl,
            joint_names = None,
            qpos        = None,
            ctrl_idxs   = None,
            clip_position = False,
        ):
        """
        Write joint-position targets into a control buffer.

        Parameters:
            ctrl (Any): Input ctrl.
            joint_names (Any | None): Input joint names. Defaults to `None`.
            qpos (Any | None): Input qpos. Defaults to `None`.
            ctrl_idxs (Any | None): Input ctrl idxs. Defaults to `None`.
            clip_position (bool): Input clip position. Defaults to `False`.

        Returns:
            Any: Result computed from the current inputs.
        """
        if ctrl_idxs is None:
            if joint_names is None:
                raise ValueError("Provide either joint_names or ctrl_idxs.")
            ctrl_idxs = self.get_ctrl_idxs_attached_joints(joint_names=joint_names)
        ctrl_idxs = list(ctrl_idxs)

        q = np.asarray(qpos,dtype=np.float64).copy()
        if q.ndim == 0:
            q = np.full(len(ctrl_idxs),float(q),dtype=np.float64)
        else:
            q = q.reshape(-1)
            if q.size == 1 and len(ctrl_idxs) != 1:
                q = np.full(len(ctrl_idxs),float(q[0]),dtype=np.float64)

        if q.shape[0] != len(ctrl_idxs):
            raise ValueError(
                "[write_ctrl_qpos] qpos length [%d] does not match ctrl index length [%d]."%
                (q.shape[0],len(ctrl_idxs))
            )

        unsupported_ctrls = []
        for ctrl_idx in ctrl_idxs:
            if self.ctrl_types[ctrl_idx] != 'JOINT':
                unsupported_ctrls.append(self.ctrl_names[ctrl_idx])
        if len(unsupported_ctrls) > 0:
            raise ValueError(
                "[write_ctrl_qpos] Unsupported control type for ctrl_names:%s. "
                "Only JOINT transmission is supported."%(unsupported_ctrls)
            )

        if clip_position:
            q_mins = np.asarray(self.ctrl_qpos_mins,dtype=np.float64)[ctrl_idxs]
            q_maxs = np.asarray(self.ctrl_qpos_maxs,dtype=np.float64)[ctrl_idxs]
            q = np.clip(q,q_mins,q_maxs)

        ctrl[ctrl_idxs] = q
        return ctrl

    def get_qpos_from_ctrl(self,ctrl,ctrl_names=None,clip_position=True):
        """
        Convert joint-actuator control targets to qpos targets and joint names.

        Parameters:
            ctrl (Any): Input ctrl.
            ctrl_names (Any | None): Input ctrl names. Defaults to `None`.
            clip_position (bool): Input clip position. Defaults to `True`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if ctrl_names is None:
            ctrl_names = self.ctrl_names

        ctrl_idxs = get_idxs(self.ctrl_names,ctrl_names)

        unsupported_ctrls = []
        for ctrl_idx in ctrl_idxs:
            if self.ctrl_types[ctrl_idx] != 'JOINT':
                unsupported_ctrls.append(self.ctrl_names[ctrl_idx])

        if len(unsupported_ctrls) > 0:
            raise ValueError(
                "[get_qpos_from_ctrl] Unsupported control type for ctrl_names:%s. "
                "Only JOINT transmission is supported."%(unsupported_ctrls)
            )

        q = np.asarray(ctrl,dtype=np.float64).reshape(-1).copy()
        if q.shape[0] != len(ctrl_idxs):
            raise ValueError(
                "[get_qpos_from_ctrl] ctrl length [%d] does not match ctrl_names length [%d]."%
                (q.shape[0],len(ctrl_idxs))
            )

        joint_names = [self.ctrl_qpos_names[idx] for idx in ctrl_idxs]

        if clip_position:
            q_mins = np.asarray(self.ctrl_qpos_mins,dtype=np.float64)[ctrl_idxs]
            q_maxs = np.asarray(self.ctrl_qpos_maxs,dtype=np.float64)[ctrl_idxs]
            q = np.clip(q,q_mins,q_maxs)

        return q,joint_names

    def get_ctrl_names_attached_joints(self,joint_names=None):
        """
        Get the first control name attached to each specified joint.

        Parameters:
            joint_names (Any | None): Input joint names. Defaults to `None`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        # If joint_names is None, use all joints
        if joint_names is None:
            joint_names = self.joint_names
        joint_idxs = get_idxs(self.joint_names,joint_names)

        # Collect control names attached to the specified joints
        ctrl_names = []
        for idx in joint_idxs:
            attached_ctrl_names = self.joint_ctrl_names[idx]

            assert len(attached_ctrl_names) > 0, \
                "[get_ctrl_names_attached_joints] No control attached to joint [%s]." \
                % (self.joint_names[idx])

            if len(attached_ctrl_names) > 1:
                warnings.warn(
                    "[get_ctrl_names_attached_joints] Multiple controls [%s] are attached to joint [%s]. "
                    "Returning the first one [%s]."
                    % (attached_ctrl_names,self.joint_names[idx],attached_ctrl_names[0])
                )

            ctrl_names.append(attached_ctrl_names[0])

        return ctrl_names

    def get_ctrl_idxs(self,ctrl_names):
        """
        Get the indices of the specified control names.

        Parameters:
            ctrl_names (list[str]): Input ctrl names.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        idxs = get_idxs(self.ctrl_names,ctrl_names)
        return idxs

    def get_ctrl_idxs_attached_joints(self,joint_names):
        """
        Get the indices of the controls attached to each specified joint.

        Parameters:
            joint_names (list[str]): Input joint names.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        idxs = get_idxs(self.ctrl_names,self.get_ctrl_names_attached_joints(joint_names=joint_names))
        return idxs

    def _resolve_position_ctrl_idxs(self,ctrl_names=None,joint_names=None):
        """
        Resolve position actuator indices by control names or joint names.

        Parameters:
            ctrl_names (Any | None): Input ctrl names. Defaults to `None`.
            joint_names (Any | None): Input joint names. Defaults to `None`.

        Returns:
            Any: Result computed from the current inputs.
        """
        if ctrl_names is not None and joint_names is not None:
            raise ValueError("Provide only one of ctrl_names or joint_names.")

        if ctrl_names is None and joint_names is None:
            return list(self.position_ctrl_idxs)

        if ctrl_names is not None:
            ctrl_idxs = get_idxs(self.ctrl_names,ctrl_names)
        else:
            joint_idxs = get_idxs(self.joint_names,joint_names)
            ctrl_idxs = []
            for joint_idx in joint_idxs:
                ctrl_idxs_joint = [
                    ctrl_idx for ctrl_idx in self.joint_ctrl_idxs[joint_idx]
                    if ctrl_idx in self.position_ctrl_idxs
                ]
                if len(ctrl_idxs_joint) == 0:
                    raise ValueError(
                        "[_resolve_position_ctrl_idxs] No position actuator attached to joint [%s]."%
                        (self.joint_names[joint_idx])
                    )
                ctrl_idxs.extend(ctrl_idxs_joint)

            # keep order while removing duplicates
            seen = set()
            ctrl_idxs = [idx for idx in ctrl_idxs if not (idx in seen or seen.add(idx))]

        bad_ctrls = []
        for ctrl_idx in ctrl_idxs:
            if ctrl_idx not in self.position_ctrl_idxs:
                bad_ctrls.append(self.ctrl_names[ctrl_idx])
        if len(bad_ctrls) > 0:
            raise ValueError(
                "[_resolve_position_ctrl_idxs] Non-position actuators found:%s"%(bad_ctrls)
            )
        return list(ctrl_idxs)

    def _resolve_ctrl_idxs(self,ctrl_names=None):
        """
        Resolve control indices by control names.

        Parameters:
            ctrl_names (Any | None): Input ctrl names. Defaults to `None`.

        Returns:
            Any: Result computed from the current inputs.
        """
        if ctrl_names is None:
            return list(range(self.n_ctrl))
        return list(get_idxs(self.ctrl_names,ctrl_names))

    def _get_trntype_name(self,trntype):
        """
        Convert actuator transmission type to a readable string.

        Parameters:
            trntype (Any): Input trntype.

        Returns:
            Any: Result computed from the current inputs.
        """
        trntype_map = {
            int(mujoco.mjtTrn.mjTRN_JOINT):         "JOINT",
            int(mujoco.mjtTrn.mjTRN_JOINTINPARENT): "JOINTINPARENT",
            int(mujoco.mjtTrn.mjTRN_SLIDERCRANK):   "SLIDERCRANK",
            int(mujoco.mjtTrn.mjTRN_TENDON):        "TENDON",
            int(mujoco.mjtTrn.mjTRN_SITE):          "SITE",
            int(mujoco.mjtTrn.mjTRN_BODY):          "BODY",
            int(mujoco.mjtTrn.mjTRN_UNDEFINED):     "UNDEFINED",
        }
        return trntype_map.get(int(trntype),"UNKNOWN (%d)"%(int(trntype)))

    def _get_gaintype_name(self,gaintype):
        """
        Convert actuator gain type to a readable string.

        Parameters:
            gaintype (Any): Input gaintype.

        Returns:
            Any: Result computed from the current inputs.
        """
        gaintype_map = {
            int(mujoco.mjtGain.mjGAIN_FIXED):  "FIXED",
            int(mujoco.mjtGain.mjGAIN_AFFINE): "AFFINE",
            int(mujoco.mjtGain.mjGAIN_MUSCLE): "MUSCLE",
            int(mujoco.mjtGain.mjGAIN_USER):   "USER",
        }
        return gaintype_map.get(int(gaintype),"UNKNOWN (%d)"%(int(gaintype)))

    def _get_biastype_name(self,biastype):
        """
        Convert actuator bias type to a readable string.

        Parameters:
            biastype (Any): Input biastype.

        Returns:
            Any: Result computed from the current inputs.
        """
        biastype_map = {
            int(mujoco.mjtBias.mjBIAS_NONE):   "NONE",
            int(mujoco.mjtBias.mjBIAS_AFFINE): "AFFINE",
            int(mujoco.mjtBias.mjBIAS_MUSCLE): "MUSCLE",
            int(mujoco.mjtBias.mjBIAS_USER):   "USER",
        }
        return biastype_map.get(int(biastype),"UNKNOWN (%d)"%(int(biastype)))

    def _get_ctrl_gain_analysis(self,ctrl_idx,d_gain_ratio_warn=0.01):
        """
        Get minimal gain analysis for the control.

        Parameters:
            ctrl_idx (int): Input ctrl idx.
            d_gain_ratio_warn (float): Input d gain ratio warn. Defaults to `0.01`.

        Returns:
            Any: Result computed from the current inputs.
        """
        p_gain    = float(self.model.actuator_gainprm[ctrl_idx,0])
        d_gain    = float(-self.model.actuator_biasprm[ctrl_idx,2])
        d_over_p = None
        if np.abs(p_gain) > 1e-12:
            d_over_p = float(np.abs(d_gain)/np.abs(p_gain))

        if np.abs(d_gain) <= 1e-12:
            status  = "missing_d_gain"
            message = "D gain is too small."
            color   = "red"
        elif d_over_p is None:
            status  = "n/a"
            message = "Unable to evaluate D/P ratio."
            color   = "yellow"
        elif d_over_p > float(d_gain_ratio_warn):
            status  = "high_d_gain_ratio"
            message = "D gain is too high."
            color   = "yellow"
        else:
            status  = "ok"
            message = "P/D gains look okay."
            color   = "green"

        return {
            "p_gain"      : p_gain,
            "d_gain"      : d_gain,
            "d_over_p"    : d_over_p,
            "status"      : status,
            "message"     : message,
            "color"       : color,
            "warn_ratio"  : float(d_gain_ratio_warn),
        }

    def _print_ctrl_info_row(
            self,
            ctrl_idx,
            idx_print=0,
            n_print=1,
            raw=False,
            analysis=False,
            d_gain_ratio_warn=0.01,
        ):
        """
        Print one control information row.

        Parameters:
            ctrl_idx (int): Input ctrl idx.
            idx_print (int): Input idx print. Defaults to `0`.
            n_print (int): Input n print. Defaults to `1`.
            raw (bool): Input raw. Defaults to `False`.
            analysis (bool): Input analysis. Defaults to `False`.
            d_gain_ratio_warn (float): Input d gain ratio warn. Defaults to `0.01`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        gainprm    = np.array(self.model.actuator_gainprm[ctrl_idx],dtype=np.float64)
        biasprm    = np.array(self.model.actuator_biasprm[ctrl_idx],dtype=np.float64)
        ctrlrange  = np.array(self.model.actuator_ctrlrange[ctrl_idx],dtype=np.float64)
        forcerange = np.array(self.model.actuator_forcerange[ctrl_idx],dtype=np.float64)

        p_gain = float(gainprm[0])
        d_gain = float(-biasprm[2])

        print(
            "[%d/%d] ctrl_idx:[%02d] ctrl_name:[%s] "%
            (
                idx_print,n_print,ctrl_idx,self.ctrl_names[ctrl_idx]
            )
        )
        print(
            " joint:[%s] force:[%.2f~%.2f] ctrl_range:[%.2f~%.2f]"%
            (
                self.ctrl_joint_names[ctrl_idx],
                forcerange[0],forcerange[1],
                ctrlrange[0],ctrlrange[1],
            )
        )
        print(
            " type:[%s|%s] gear:[%.4f] gain:[%s] bias:[%s] P gain:[%.2f] D gain:[%.2f]"%
            (
                self.ctrl_types[ctrl_idx],
                self._get_trntype_name(self.ctrl_trntypes[ctrl_idx]),
                self.ctrl_gears[ctrl_idx],
                self._get_gaintype_name(self.ctrl_gaintypes[ctrl_idx]),
                self._get_biastype_name(self.ctrl_biastypes[ctrl_idx]),
                p_gain,d_gain,
            )
        )
        if raw:
            print("raw gainprm:%s"%(gainprm))
            print("raw biasprm:%s"%(biasprm))

        if analysis:
            info = self._get_ctrl_gain_analysis(
                ctrl_idx          = ctrl_idx,
                d_gain_ratio_warn = d_gain_ratio_warn,
            )
            msg = " analysis: %s"%(info["message"])
            if info["d_over_p"] is not None:
                msg += " D/P=[%.5f] warn>[%.5f]"%(info["d_over_p"],info["warn_ratio"])
            if info["color"] == "red":
                print_red(msg)
            elif info["color"] == "yellow":
                print_yellow(msg)
            else:
                print_green(msg)

    def print_ctrl_info(self,ctrl_names=None,raw=False,analysis=False,d_gain_ratio_warn=0.1):
        """
        Print control information for all or selected controls.

        Parameters:
            ctrl_names (Any | None): Input ctrl names. Defaults to `None`.
            raw (bool): Input raw. Defaults to `False`.
            analysis (bool): Input analysis. Defaults to `False`.
            d_gain_ratio_warn (float): Input d gain ratio warn. Defaults to `0.1`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        ctrl_idxs = self._resolve_ctrl_idxs(ctrl_names=ctrl_names)
        for i, ctrl_idx in enumerate(ctrl_idxs):
            if analysis: print ("")
            self._print_ctrl_info_row(
                ctrl_idx          = ctrl_idx,
                idx_print         = i,
                n_print           = len(ctrl_idxs),
                raw               = raw,
                analysis          = analysis,
                d_gain_ratio_warn = d_gain_ratio_warn,
            )

    def _resolve_geom_idxs(
            self,
            geom_names        = None,
            body_names        = None,
            include_keywords  = None,
            exclude_keywords  = None,
        ):
        """
        Resolve geometry indices by names, body names, and keyword filters.

        Parameters:
            geom_names (Any | None): Input geom names. Defaults to `None`.
            body_names (Any | None): Input body names. Defaults to `None`.
            include_keywords (Any | None): Input include keywords. Defaults to `None`.
            exclude_keywords (Any | None): Input exclude keywords. Defaults to `None`.

        Returns:
            Any: Result computed from the current inputs.
        """
        if include_keywords is None:
            include_keywords = []
        if exclude_keywords is None:
            exclude_keywords = []
        include_keywords = [str(k).lower() for k in include_keywords]
        exclude_keywords = [str(k).lower() for k in exclude_keywords]

        if geom_names is not None:
            geom_idxs = get_idxs(self.geom_names,geom_names)
        else:
            geom_idxs = list(range(self.n_geom))

        if body_names is not None:
            body_name_set = set(body_names)
            geom_idxs = [
                geom_idx for geom_idx in geom_idxs
                if self.body_names[int(self.model.geom_bodyid[geom_idx])] in body_name_set
            ]

        if (len(include_keywords) > 0) or (len(exclude_keywords) > 0):
            geom_idxs_filt = []
            for geom_idx in geom_idxs:
                geom_name = self.geom_names[geom_idx]
                body_name = self.body_names[int(self.model.geom_bodyid[geom_idx])]
                text = "%s %s"%(str(geom_name),str(body_name))
                text = text.lower()

                if len(include_keywords) > 0:
                    if not any([k in text for k in include_keywords]):
                        continue
                if len(exclude_keywords) > 0:
                    if any([k in text for k in exclude_keywords]):
                        continue
                geom_idxs_filt.append(geom_idx)
            geom_idxs = geom_idxs_filt

        return list(geom_idxs)

    def restore_position_actuators(self,ctrl_names=None,joint_names=None):
        """
        Restore position-style actuator parameters from originals.

        Parameters:
            ctrl_names (Any | None): Input ctrl names. Defaults to `None`.
            joint_names (Any | None): Input joint names. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        ctrl_idxs = self._resolve_position_ctrl_idxs(
            ctrl_names  = ctrl_names,
            joint_names = joint_names,
        )
        self.model.actuator_gainprm[ctrl_idxs,:] = self.actuator_gainprm_0[ctrl_idxs,:]
        self.model.actuator_biasprm[ctrl_idxs,:] = self.actuator_biasprm_0[ctrl_idxs,:]

    @staticmethod
    def _normalize_ctrl_range_input(
            value,
            name="range",
        ):
        """
        Normalize control/force range input to a [min,max] array.

        Parameters:
            value (Any): Input value.
            name (str): Input name. Defaults to `'range'`.

        Returns:
            Any: Result computed from the current inputs.
        """
        if np.isscalar(value):
            vmax = float(np.abs(value))
            if not np.isfinite(vmax):
                raise ValueError("[%s] scalar value should be finite."%(name))
            return np.array([-vmax,+vmax],dtype=np.float64)

        arr = np.asarray(value,dtype=np.float64).reshape(-1)
        if arr.shape[0] != 2:
            raise ValueError("[%s] should be scalar or length-2."%(name))
        if not np.isfinite(arr).all():
            raise ValueError("[%s] should contain finite values."%(name))
        if arr[0] > arr[1]:
            raise ValueError("[%s] min should be <= max."%(name))
        return arr

    def restore_ctrl_info(
            self,
            ctrl_names  = None,
            joint_names = None,
            verbose     = False,
        ):
        """
        Restore original actuator gain, force range, and control range.

        Parameters:
            ctrl_names (Any | None): Input ctrl names. Defaults to `None`.
            joint_names (Any | None): Input joint names. Defaults to `None`.
            verbose (bool): Input verbose. Defaults to `False`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        ctrl_idxs = self._resolve_position_ctrl_idxs(
            ctrl_names  = ctrl_names,
            joint_names = joint_names,
        )

        self.restore_position_actuators(
            ctrl_names  = ctrl_names,
            joint_names = joint_names,
        )
        self.model.actuator_ctrlrange[ctrl_idxs,:]  = self.actuator_ctrlrange_0[ctrl_idxs,:]
        self.model.actuator_forcerange[ctrl_idxs,:] = self.actuator_forcerange_0[ctrl_idxs,:]

        if verbose:
            for ctrl_idx in ctrl_idxs:
                print(
                    "[restore_ctrl_info] ctrl [%s]: P [%.2f], D [%.2f], force [%.2f~%.2f], ctrl_range [%.2f~%.2f]."%
                    (
                        self.ctrl_names[ctrl_idx],
                        self.model.actuator_gainprm[ctrl_idx,0],
                        -self.model.actuator_biasprm[ctrl_idx,2],
                        self.model.actuator_forcerange[ctrl_idx,0],
                        self.model.actuator_forcerange[ctrl_idx,1],
                        self.model.actuator_ctrlrange[ctrl_idx,0],
                        self.model.actuator_ctrlrange[ctrl_idx,1],
                    )
                )

    def restore_ctrl_gain(
            self,
            ctrl_names  = None,
            joint_names = None,
            verbose     = True,
        ):
        """
        Restore original P/D gains for position-style controls.

        Parameters:
            ctrl_names (Any | None): Input ctrl names. Defaults to `None`.
            joint_names (Any | None): Input joint names. Defaults to `None`.
            verbose (bool): Input verbose. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.restore_ctrl_info(
            ctrl_names  = ctrl_names,
            joint_names = joint_names,
            verbose     = False,
        )
        if verbose:
            ctrl_idxs = self._resolve_position_ctrl_idxs(
                ctrl_names  = ctrl_names,
                joint_names = joint_names,
            )
            for ctrl_idx in ctrl_idxs:
                print(
                    "[restore_ctrl_gain] ctrl [%s]: P gain restored to [%.2f], D gain restored to [%.2f]."%
                    (
                        self.ctrl_names[ctrl_idx],
                        self.model.actuator_gainprm[ctrl_idx,0],
                        -self.model.actuator_biasprm[ctrl_idx,2],
                    )
                )

    def set_ctrl_info(
            self,
            ctrl_names     = None,
            joint_names    = None,
            p_gain         = None,
            d_gain         = None,
            force          = None,
            ctrl_range     = None,
            restore_first  = True,
            verbose        = False,
            text_color     = None,
        ):
        """
        Set actuator P/D gain, force range, and control range.

        Parameters:
            ctrl_names (Any | None): Input ctrl names. Defaults to `None`.
            joint_names (Any | None): Input joint names. Defaults to `None`.
            p_gain (Any | None): Input p gain. Defaults to `None`.
            d_gain (Any | None): Input d gain. Defaults to `None`.
            force (Any | None): Input force. Defaults to `None`.
            ctrl_range (Any | None): Input ctrl range. Defaults to `None`.
            restore_first (bool): Input restore first. Defaults to `True`.
            verbose (bool): Input verbose. Defaults to `False`.
            text_color (Any | None): Input text color. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if (
            p_gain is None and
            d_gain is None and
            force is None and
            ctrl_range is None
        ):
            raise ValueError(
                "[set_ctrl_info] At least one of p_gain, d_gain, force, or ctrl_range should be provided."
            )

        ctrl_idxs = self._resolve_position_ctrl_idxs(
            ctrl_names  = ctrl_names,
            joint_names = joint_names,
        )

        if restore_first:
            self.restore_ctrl_info(
                ctrl_names  = ctrl_names,
                joint_names = joint_names,
                verbose     = False,
            )

        force_range_val = None
        ctrl_range_val  = None
        if force is not None:
            force_range_val = self._normalize_ctrl_range_input(force,name="force")
        if ctrl_range is not None:
            ctrl_range_val = self._normalize_ctrl_range_input(ctrl_range,name="ctrl_range")

        for ctrl_idx in ctrl_idxs:
            gainprm_0    = np.array(self.actuator_gainprm_0[ctrl_idx],dtype=np.float64)
            biasprm_0    = np.array(self.actuator_biasprm_0[ctrl_idx],dtype=np.float64)
            force_0      = np.array(self.model.actuator_forcerange[ctrl_idx],dtype=np.float64)
            ctrlrange_0  = np.array(self.model.actuator_ctrlrange[ctrl_idx],dtype=np.float64)

            if p_gain is not None:
                p_gain_val = float(p_gain)
                if not np.isfinite(p_gain_val):
                    raise ValueError("[set_ctrl_info] p_gain should be finite.")

                self.model.actuator_gainprm[ctrl_idx,0] = p_gain_val

                if np.abs(gainprm_0[0]) > 0.0:
                    p_ratio = p_gain_val / gainprm_0[0]
                    self.model.actuator_biasprm[ctrl_idx,1] = biasprm_0[1] * p_ratio
                else:
                    self.model.actuator_biasprm[ctrl_idx,1] = -p_gain_val

            if d_gain is not None:
                d_gain_val = float(d_gain)
                if not np.isfinite(d_gain_val):
                    raise ValueError("[set_ctrl_info] d_gain should be finite.")

                if np.abs(biasprm_0[2]) > 0.0:
                    d_ratio = d_gain_val / (-biasprm_0[2])
                    self.model.actuator_biasprm[ctrl_idx,2] = biasprm_0[2] * d_ratio
                else:
                    self.model.actuator_biasprm[ctrl_idx,2] = -d_gain_val

            if force_range_val is not None:
                self.model.actuator_forcerange[ctrl_idx,:] = force_range_val

            if ctrl_range_val is not None:
                self.model.actuator_ctrlrange[ctrl_idx,:] = ctrl_range_val

            if verbose:
                msg_list = []
                if p_gain is not None:
                    msg_list.append(
                        "P [%.2f]=>[%.2f]"%
                        (
                            gainprm_0[0],
                            self.model.actuator_gainprm[ctrl_idx,0],
                        )
                    )
                if d_gain is not None:
                    msg_list.append(
                        "D [%.2f]=>[%.2f]"%
                        (
                            -biasprm_0[2],
                            -self.model.actuator_biasprm[ctrl_idx,2],
                        )
                    )
                if force_range_val is not None:
                    msg_list.append(
                        "force [%.2f~%.2f]=>[%.2f~%.2f]"%
                        (
                            force_0[0],
                            force_0[1],
                            self.model.actuator_forcerange[ctrl_idx,0],
                            self.model.actuator_forcerange[ctrl_idx,1],
                        )
                    )
                if ctrl_range_val is not None:
                    msg_list.append(
                        "ctrl_range [%.2f~%.2f]=>[%.2f~%.2f]"%
                        (
                            ctrlrange_0[0],
                            ctrlrange_0[1],
                            self.model.actuator_ctrlrange[ctrl_idx,0],
                            self.model.actuator_ctrlrange[ctrl_idx,1],
                        )
                    )
                _print_with_text_color(
                    "ctrl:[%s] %s"%
                    (
                        self.ctrl_names[ctrl_idx],
                        ", ".join(msg_list),
                    ),
                    text_color = text_color,
                )

    def set_ctrl_gain(
            self,
            ctrl_names     = None,
            joint_names    = None,
            p_gain         = None,
            d_gain         = None,
            restore_first  = True,
            verbose        = True,
        ):
        """
        Set absolute P/D gains for position-style controls.

        Parameters:
            ctrl_names (Any | None): Input ctrl names. Defaults to `None`.
            joint_names (Any | None): Input joint names. Defaults to `None`.
            p_gain (Any | None): Input p gain. Defaults to `None`.
            d_gain (Any | None): Input d gain. Defaults to `None`.
            restore_first (bool): Input restore first. Defaults to `True`.
            verbose (bool): Input verbose. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.set_ctrl_info(
            ctrl_names    = ctrl_names,
            joint_names   = joint_names,
            p_gain        = p_gain,
            d_gain        = d_gain,
            force         = None,
            ctrl_range    = None,
            restore_first = restore_first,
            verbose       = False,
        )

        if verbose:
            ctrl_idxs = self._resolve_position_ctrl_idxs(
                ctrl_names  = ctrl_names,
                joint_names = joint_names,
            )
            for ctrl_idx in ctrl_idxs:
                print(
                    "[set_ctrl_gain] ctrl [%s]: P gain set to [%.2f], D gain set to [%.2f]."%
                    (
                        self.ctrl_names[ctrl_idx],
                        self.model.actuator_gainprm[ctrl_idx,0],
                        -self.model.actuator_biasprm[ctrl_idx,2],
                    )
                )

    def tune_position_actuators(
            self,
            ctrl_names     = None,
            joint_names    = None,
            p_rate         = 1.0,
            d_rate         = 1.0,
            restore_first  = True,
        ):
        """
        Tune position-style actuator gain/bias parameters from original values.

        Parameters:
            ctrl_names (Any | None): Input ctrl names. Defaults to `None`.
            joint_names (Any | None): Input joint names. Defaults to `None`.
            p_rate (float): Input p rate. Defaults to `1.0`.
            d_rate (float): Input d rate. Defaults to `1.0`.
            restore_first (bool): Input restore first. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        ctrl_idxs = self._resolve_position_ctrl_idxs(
            ctrl_names  = ctrl_names,
            joint_names = joint_names,
        )

        if restore_first:
            self.restore_position_actuators(
                ctrl_names  = ctrl_names,
                joint_names = joint_names,
            )

        self.model.actuator_gainprm[ctrl_idxs,0] = self.actuator_gainprm_0[ctrl_idxs,0]*p_rate
        self.model.actuator_biasprm[ctrl_idxs,1] = self.actuator_biasprm_0[ctrl_idxs,1]*p_rate
        self.model.actuator_biasprm[ctrl_idxs,2] = self.actuator_biasprm_0[ctrl_idxs,2]*d_rate

    def print_position_actuators(self,ctrl_names=None,joint_names=None,raw=False):
        """
        Print position actuator information.

        Parameters:
            ctrl_names (Any | None): Input ctrl names. Defaults to `None`.
            joint_names (Any | None): Input joint names. Defaults to `None`.
            raw (bool): Input raw. Defaults to `False`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        # Resolve position actuator indices
        ctrl_idxs = self._resolve_position_ctrl_idxs(
            ctrl_names  = ctrl_names,
            joint_names = joint_names,
        )

        # Print information for each actuator
        for i, ctrl_idx in enumerate(ctrl_idxs): # print info for each actuator
            self._print_ctrl_info_row(
                ctrl_idx   = ctrl_idx,
                idx_print  = i,
                n_print    = len(ctrl_idxs),
                raw        = raw,
            )

    def _get_friction_info_analysis(
            self,
            geom_idx,
            friction_change_tol = 1e-12,
            sliding_warn        = 2.0,
            torsional_warn      = 0.05,
            rolling_warn        = 0.01,
        ):
        """
        Get minimal geometry friction analysis summary.

        Parameters:
            geom_idx (int): Input geom idx.
            friction_change_tol (float): Input friction change tol. Defaults to
                                         `1e-12`.
            sliding_warn (float): Input sliding warn. Defaults to `2.0`.
            torsional_warn (float): Input torsional warn. Defaults to `0.05`.
            rolling_warn (float): Input rolling warn. Defaults to `0.01`.

        Returns:
            Any: Result computed from the current inputs.
        """
        friction    = np.array(self.model.geom_friction[geom_idx],dtype=np.float64)
        friction_0  = np.array(self.geom_friction_0[geom_idx],dtype=np.float64)
        delta       = friction-friction_0
        is_changed  = np.abs(delta) > float(friction_change_tol)
        is_high     = np.array(
            [
                friction[0] > float(sliding_warn),
                friction[1] > float(torsional_warn),
                friction[2] > float(rolling_warn),
            ],
            dtype=bool,
        )

        if np.any(is_high):
            status  = "high_friction"
            color   = "yellow"
            parts   = []
            if is_high[0]: parts.append("sliding")
            if is_high[1]: parts.append("torsional")
            if is_high[2]: parts.append("rolling")
            message = "High friction component: %s."%(",".join(parts))
        elif np.any(is_changed):
            status  = "changed"
            color   = "green"
            message = "Friction changed from original values."
        else:
            status  = "restored"
            color   = "green"
            message = "Friction matches original values."

        return {
            "friction"             : friction,
            "friction_0"           : friction_0,
            "delta"                : delta,
            "is_changed"           : is_changed,
            "is_high"              : is_high,
            "status"               : status,
            "message"              : message,
            "color"                : color,
            "friction_change_tol"  : float(friction_change_tol),
            "sliding_warn"         : float(sliding_warn),
            "torsional_warn"       : float(torsional_warn),
            "rolling_warn"         : float(rolling_warn),
        }

    def _print_friction_info_row(
            self,
            geom_idx,
            idx_print            = 0,
            n_print              = 1,
            raw                  = False,
            analysis             = False,
            inspect              = False,
            friction_change_tol  = 1e-12,
            sliding_warn         = 2.0,
            torsional_warn       = 0.05,
            rolling_warn         = 0.01,
        ):
        """
        Print one geometry friction information row.

        Parameters:
            geom_idx (int): Input geom idx.
            idx_print (int): Input idx print. Defaults to `0`.
            n_print (int): Input n print. Defaults to `1`.
            raw (bool): Input raw. Defaults to `False`.
            analysis (bool): Input analysis. Defaults to `False`.
            inspect (bool): Input inspect. Defaults to `False`.
            friction_change_tol (float): Input friction change tol. Defaults to
                                         `1e-12`.
            sliding_warn (float): Input sliding warn. Defaults to `2.0`.
            torsional_warn (float): Input torsional warn. Defaults to `0.05`.
            rolling_warn (float): Input rolling warn. Defaults to `0.01`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        body_name   = self.body_names[int(self.model.geom_bodyid[geom_idx])]
        friction    = np.array(self.model.geom_friction[geom_idx],dtype=np.float64)
        friction_0  = np.array(self.geom_friction_0[geom_idx],dtype=np.float64)
        priority    = int(self.model.geom_priority[geom_idx])

        print(
            "[%d/%d] geom_idx:[%03d] geom_name:[%s] body:[%s] priority:[%d]"%
            (
                idx_print,n_print,geom_idx,self.geom_names[geom_idx],body_name,priority,
            )
        )
        print(
            " friction (sliding, torsional, rolling): [%.4f %.4f %.4f] => [%.4f %.4f %.4f]"%
            (
                friction_0[0],friction_0[1],friction_0[2],
                friction[0],friction[1],friction[2],
            )
        )
        if raw:
            print("raw friction:%s"%(friction))
            print("raw friction_0:%s"%(friction_0))

        if analysis or inspect:
            info = self._get_friction_info_analysis(
                geom_idx             = geom_idx,
                friction_change_tol  = friction_change_tol,
                sliding_warn         = sliding_warn,
                torsional_warn       = torsional_warn,
                rolling_warn         = rolling_warn,
            )
            comp_names = ["sliding","torsional","rolling"]
            comp_texts = []
            for comp_idx, comp_name in enumerate(comp_names):
                value = info["friction"][comp_idx]
                text = "%s=[%.4f]"%(comp_name,value)
                if info["is_high"][comp_idx]:
                    text = colored(text,"yellow")
                elif info["is_changed"][comp_idx]:
                    text = colored(text,"green")
                comp_texts.append(text)
            print(
                " analysis: %s %s"%(
                    info["message"],
                    " ".join(comp_texts),
                )
            )

    def print_friction_info(
            self,
            geom_names        = None,
            body_names        = None,
            include_keywords  = None,
            exclude_keywords  = None,
            raw               = False,
            analysis          = False,
            inspect           = False,
            friction_change_tol = 1e-12,
            sliding_warn      = 2.0,
            torsional_warn    = 0.05,
            rolling_warn      = 0.01,
            title             = None,
        ):
        """
        Print geometry friction information for all or selected geometries.

        Parameters:
            geom_names (Any | None): Input geom names. Defaults to `None`.
            body_names (Any | None): Input body names. Defaults to `None`.
            include_keywords (Any | None): Input include keywords. Defaults to `None`.
            exclude_keywords (Any | None): Input exclude keywords. Defaults to `None`.
            raw (bool): Input raw. Defaults to `False`.
            analysis (bool): Input analysis. Defaults to `False`.
            inspect (bool): Input inspect. Defaults to `False`.
            friction_change_tol (float): Input friction change tol. Defaults to
                                         `1e-12`.
            sliding_warn (float): Input sliding warn. Defaults to `2.0`.
            torsional_warn (float): Input torsional warn. Defaults to `0.05`.
            rolling_warn (float): Input rolling warn. Defaults to `0.01`.
            title (Any | None): Input title. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        geom_idxs = self._resolve_geom_idxs(
            geom_names       = geom_names,
            body_names       = body_names,
            include_keywords = include_keywords,
            exclude_keywords = exclude_keywords,
        )

        if title is not None:
            print("\n"+"="*60)
            print("[%s]"%(title))
            print("="*60)

        for i, geom_idx in enumerate(geom_idxs):
            if analysis or inspect:
                print("")
            self._print_friction_info_row(
                geom_idx            = geom_idx,
                idx_print           = i,
                n_print             = len(geom_idxs),
                raw                 = raw,
                analysis            = analysis,
                inspect             = inspect,
                friction_change_tol = friction_change_tol,
                sliding_warn        = sliding_warn,
                torsional_warn      = torsional_warn,
                rolling_warn        = rolling_warn,
            )

    def restore_friction_info(
            self,
            geom_names        = None,
            body_names        = None,
            include_keywords  = None,
            exclude_keywords  = None,
            verbose           = False,
        ):
        """
        Restore geometry friction from original values.

        Parameters:
            geom_names (Any | None): Input geom names. Defaults to `None`.
            body_names (Any | None): Input body names. Defaults to `None`.
            include_keywords (Any | None): Input include keywords. Defaults to `None`.
            exclude_keywords (Any | None): Input exclude keywords. Defaults to `None`.
            verbose (bool): Input verbose. Defaults to `False`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        geom_idxs = self._resolve_geom_idxs(
            geom_names       = geom_names,
            body_names       = body_names,
            include_keywords = include_keywords,
            exclude_keywords = exclude_keywords,
        )
        self.model.geom_friction[geom_idxs,:] = self.geom_friction_0[geom_idxs,:]

        if verbose:
            for geom_idx in geom_idxs:
                body_name  = self.body_names[int(self.model.geom_bodyid[geom_idx])]
                friction_0 = self.geom_friction_0[geom_idx]
                print(
                    "[restore_friction_info] geom [%s] body [%s]: friction [%.4f %.4f %.4f]."%
                    (
                        self.geom_names[geom_idx],
                        body_name,
                        friction_0[0],
                        friction_0[1],
                        friction_0[2],
                    )
                )

    def restore_geom_friction(
            self,
            geom_names        = None,
            body_names        = None,
            include_keywords  = None,
            exclude_keywords  = None,
        ):
        """
        Restore geom friction from original values.

        Parameters:
            geom_names (Any | None): Input geom names. Defaults to `None`.
            body_names (Any | None): Input body names. Defaults to `None`.
            include_keywords (Any | None): Input include keywords. Defaults to `None`.
            exclude_keywords (Any | None): Input exclude keywords. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.restore_friction_info(
            geom_names       = geom_names,
            body_names       = body_names,
            include_keywords = include_keywords,
            exclude_keywords = exclude_keywords,
            verbose          = False,
        )

    def set_friction_info(
            self,
            geom_names        = None,
            body_names        = None,
            include_keywords  = None,
            exclude_keywords  = None,
            sliding           = None, # sliding friction
            torsional         = None, # torsional friction
            rolling           = None, # rolling friction
            material          = None, # material friction preset
            restore_first     = True,
            verbose           = False,
            text_color        = None,
        ):
        """
        Set absolute geometry friction values.

        Parameters:
            geom_names (Any | None): Input geom names. Defaults to `None`.
            body_names (Any | None): Input body names. Defaults to `None`.
            include_keywords (Any | None): Input include keywords. Defaults to `None`.
            exclude_keywords (Any | None): Input exclude keywords. Defaults to `None`.
            sliding (Any | None): Input sliding. Defaults to `None`.
            torsional (Any | None): Input torsional. Defaults to `None`.
            rolling (Any | None): Input rolling. Defaults to `None`.
            material (Any | None): Input material. Defaults to `None`.
            restore_first (bool): Input restore first. Defaults to `True`.
            verbose (bool): Input verbose. Defaults to `False`.
            text_color (Any | None): Input text color. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        material_presets = {
            "rubber":  (3.0,  0.10,  0.02),
            "wood":    (0.7,  0.01,  0.002),
            "glass":   (0.25, 0.003, 0.0005),
            "metal":   (0.35, 0.004, 0.0008),
            "plastic": (0.45, 0.006, 0.001),
            "table":   (0.8,  0.01,  0.001),
        }

        if material is not None:
            if sliding is not None or torsional is not None or rolling is not None:
                raise ValueError(
                    "[set_friction_info] sliding, torsional, and rolling should be None "
                    "when material is provided."
                )
            material_key = str(material).lower()
            if material_key not in material_presets:
                raise ValueError(
                    "[set_friction_info] material should be one of %s."%
                    sorted(material_presets.keys())
                )
            sliding, torsional, rolling = material_presets[material_key]

        if sliding is None and torsional is None and rolling is None:
            raise ValueError(
                "[set_friction_info] At least one of sliding, torsional, rolling, "
                "or material should be provided."
            )

        geom_idxs = self._resolve_geom_idxs(
            geom_names       = geom_names,
            body_names       = body_names,
            include_keywords = include_keywords,
            exclude_keywords = exclude_keywords,
        )

        if restore_first:
            self.restore_friction_info(
                geom_names       = geom_names,
                body_names       = body_names,
                include_keywords = include_keywords,
                exclude_keywords = exclude_keywords,
                verbose          = False,
            )

        for geom_idx in geom_idxs:
            friction_0 = np.array(self.geom_friction_0[geom_idx],dtype=np.float64)
            friction   = np.array(self.model.geom_friction[geom_idx],dtype=np.float64)

            if sliding is not None:
                sliding_val = float(sliding)
                if not np.isfinite(sliding_val):
                    raise ValueError("[set_friction_info] sliding should be finite.")
                friction[0] = sliding_val

            if torsional is not None:
                torsional_val = float(torsional)
                if not np.isfinite(torsional_val):
                    raise ValueError("[set_friction_info] torsional should be finite.")
                friction[1] = torsional_val

            if rolling is not None:
                rolling_val = float(rolling)
                if not np.isfinite(rolling_val):
                    raise ValueError("[set_friction_info] rolling should be finite.")
                friction[2] = rolling_val

            self.model.geom_friction[geom_idx,:] = friction

            if verbose:
                body_name = self.body_names[int(self.model.geom_bodyid[geom_idx])]
                msg_list = []
                if sliding is not None:
                    msg_list.append(
                        "sliding [%.4f]=>[%.4f]"%
                        (
                            friction_0[0],
                            friction[0],
                        )
                    )
                if torsional is not None:
                    msg_list.append(
                        "torsional [%.4f]=>[%.4f]"%
                        (
                            friction_0[1],
                            friction[1],
                        )
                    )
                if rolling is not None:
                    msg_list.append(
                        "rolling [%.4f]=>[%.4f]"%
                        (
                            friction_0[2],
                            friction[2],
                        )
                    )
                _print_with_text_color(
                    "geom:[%s] body:[%s] %s"%
                    (
                        self.geom_names[geom_idx],
                        body_name,
                        ", ".join(msg_list),
                    ),
                    text_color = text_color,
                )

    def tune_geom_friction(
            self,
            geom_names        = None,
            body_names        = None,
            include_keywords  = None,
            exclude_keywords  = None,
            sliding           = None,
            torsional         = None,
            rolling           = None,
            slide_rate        = None,
            torsion_rate      = None,
            roll_rate         = None,
            restore_first     = True,
        ):
        """
        Tune geometry friction from original values.

        Parameters:
            geom_names (Any | None): Input geom names. Defaults to `None`.
            body_names (Any | None): Input body names. Defaults to `None`.
            include_keywords (Any | None): Input include keywords. Defaults to `None`.
            exclude_keywords (Any | None): Input exclude keywords. Defaults to `None`.
            sliding (Any | None): Input sliding. Defaults to `None`.
            torsional (Any | None): Input torsional. Defaults to `None`.
            rolling (Any | None): Input rolling. Defaults to `None`.
            slide_rate (Any | None): Input slide rate. Defaults to `None`.
            torsion_rate (Any | None): Input torsion rate. Defaults to `None`.
            roll_rate (Any | None): Input roll rate. Defaults to `None`.
            restore_first (bool): Input restore first. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if restore_first:
            self.restore_friction_info(
                geom_names       = geom_names,
                body_names       = body_names,
                include_keywords = include_keywords,
                exclude_keywords = exclude_keywords,
                verbose          = False,
            )

        geom_idxs = self._resolve_geom_idxs(
            geom_names       = geom_names,
            body_names       = body_names,
            include_keywords = include_keywords,
            exclude_keywords = exclude_keywords,
        )

        friction_0 = self.geom_friction_0[geom_idxs,:].copy()
        friction_new = friction_0.copy()

        if slide_rate is not None and sliding is None:
            friction_new[:,0] = friction_0[:,0]*float(slide_rate)
        elif sliding is not None:
            friction_new[:,0] = float(sliding)

        if torsion_rate is not None and torsional is None:
            friction_new[:,1] = friction_0[:,1]*float(torsion_rate)
        elif torsional is not None:
            friction_new[:,1] = float(torsional)

        if roll_rate is not None and rolling is None:
            friction_new[:,2] = friction_0[:,2]*float(roll_rate)
        elif rolling is not None:
            friction_new[:,2] = float(rolling)

        self.model.geom_friction[geom_idxs,:] = friction_new

    def print_geom_friction(
            self,
            geom_names        = None,
            body_names        = None,
            include_keywords  = None,
            exclude_keywords  = None,
            title             = None,
        ):
        """
        Print geometry friction information.

        Parameters:
            geom_names (Any | None): Input geom names. Defaults to `None`.
            body_names (Any | None): Input body names. Defaults to `None`.
            include_keywords (Any | None): Input include keywords. Defaults to `None`.
            exclude_keywords (Any | None): Input exclude keywords. Defaults to `None`.
            title (Any | None): Input title. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.print_friction_info(
            geom_names       = geom_names,
            body_names       = body_names,
            include_keywords = include_keywords,
            exclude_keywords = exclude_keywords,
            raw              = False,
            title            = title,
        )

    def get_p(self,name,type):
        """
        Get the position of a specified object in the simulation.

        Parameters:
            name (str): Input name.
            type (Any): Input type.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        # Validate type
        assert type in ['body','geom','site','camera','joint','sensor'], "Unknown type: %s"%(type)

        # Get position based on type
        if type == 'body':
            obj_id = self.body_name_to_id[name]
            p = self.data.xpos[obj_id]
        elif type == 'geom':
            obj_id = self.geom_name_to_id[name]
            p = self.data.geom_xpos[obj_id]
        elif type == 'site':
            obj_id = self.site_name_to_id[name]
            p = self.data.site_xpos[obj_id]
        elif type == 'camera':
            obj_id = self.cam_name_to_id[name]
            p = self.data.cam_xpos[obj_id]
        elif type == 'joint':
            obj_id = self.joint_name_to_id[name]
            body_id = self.model.jnt_bodyid[obj_id]
            p = self.data.xpos[body_id]
        elif type == 'sensor':
            sensor_id = self.sensor_name_to_id[name] # get sensor ID
            sensor_objtype = self.model.sensor_objtype[sensor_id] # get attached object type (i.e., site)
            sensor_objid = self.model.sensor_objid[sensor_id] # get attached object ID
            obj_name = mujoco.mj_id2name(self.model,sensor_objtype,sensor_objid) # get the site name
            if sensor_objtype == mujoco.mjtObj.mjOBJ_JOINT:
                body_id = self.model.jnt_bodyid[self.joint_name_to_id[obj_name]]
                p = self.data.xpos[body_id] # get the position of the body attached to the joint
            elif sensor_objtype == mujoco.mjtObj.mjOBJ_SITE:
                p = self.data.site_xpos[self.site_name_to_id[obj_name]] # get the position of the site
            else:
                raise ValueError("Unknown sensor object type:[%d]"%(sensor_objtype))
        else:
            raise ValueError("Unknown type: %s"%(type))

        # Return
        return p.copy()

    def get_R(self,name,type):
        """
        Get the rotation matrix of a specified object in the simulation.

        Parameters:
            name (str): Input name.
            type (Any): Input type.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        # Validate type
        assert type in ['body','geom','site','camera','joint','sensor'], "Unknown type: %s"%(type)

        # Get rotation based on type
        if type == 'body':
            obj_id = self.body_name_to_id[name]
            R = self.data.xmat[obj_id].reshape(3,3)
        elif type == 'geom':
            obj_id = self.geom_name_to_id[name]
            R = self.data.geom_xmat[obj_id].reshape(3,3)
        elif type == 'site':
            obj_id = self.site_name_to_id[name]
            R = self.data.site_xmat[obj_id].reshape(3,3)
        elif type == 'camera':
            obj_id = self.cam_name_to_id[name]
            R = self.data.cam_xmat[obj_id].reshape(3,3)
        elif type == 'joint':
            obj_id = self.joint_name_to_id[name]
            body_id = self.model.jnt_bodyid[obj_id]
            R = self.data.xmat[body_id].reshape(3,3)
        elif type == 'sensor':
            sensor_id = self.sensor_name_to_id[name] # get sensor ID
            sensor_objtype = self.model.sensor_objtype[sensor_id] # get attached object type (i.e., site)
            sensor_objid = self.model.sensor_objid[sensor_id] # get attached object ID
            obj_name = mujoco.mj_id2name(self.model,sensor_objtype,sensor_objid) # get the site name
            if sensor_objtype == mujoco.mjtObj.mjOBJ_JOINT:
                body_id = self.model.jnt_bodyid[self.joint_name_to_id[obj_name]]
                R = self.data.xmat[body_id].reshape(3,3) # get the rotation of the body attached to the joint
            elif sensor_objtype == mujoco.mjtObj.mjOBJ_SITE:
                R = self.data.site_xmat[self.site_name_to_id[obj_name]].reshape(3,3) # get the rotation of the site
            else:
                raise ValueError("Unknown sensor object type:[%d]"%(sensor_objtype))
        else:
            raise ValueError("Unknown type: %s"%(type))

        # Return
        return R.copy()

    def get_T(self,name,type):
        """
        Get the transformation matrix of a specified object in the simulation.

        Parameters:
            name (str): Input name.
            type (Any): Input type.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        p = self.get_p(name,type)
        R = self.get_R(name,type)
        T = pr2t(p,R)
        return T

    def get_p_com(self,body_name):
        """
        Get the center of mass position of the specified body.

        Parameters:
            body_name (str): Input body name.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        body_idx = self.body_names.index(body_name)
        p_com = self.data.subtree_com[body_idx,:] # (3,)
        return p_com.copy()

    def get_J_body(self,body_name):
        """
        Compute the Jacobian matrices (position and rotation) for the specified body.

        Parameters:
            body_name (str): Input body name.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        J_p = np.zeros((3,self.n_v)) # nv: nDoF
        J_R = np.zeros((3,self.n_v))
        mujoco.mj_jacBody(self.model,self.data,J_p,J_R,self.data.body(body_name).id)
        J_full = np.array(np.vstack([J_p,J_R]))
        return J_p,J_R,J_full

    def get_J_geom(self,geom_name):
        """
        Compute the Jacobian matrices for the specified geometry.

        Parameters:
            geom_name (str): Input geom name.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        J_p = np.zeros((3,self.n_v)) # nv: nDoF
        J_R = np.zeros((3,self.n_v))
        mujoco.mj_jacGeom(self.model,self.data,J_p,J_R,self.data.geom(geom_name).id)
        J_full = np.array(np.vstack([J_p,J_R]))
        return J_p,J_R,J_full

    def get_J_site(self,site_name,return_type='all'):
        """
        Compute the Jacobian matrices for the specified site.

        Parameters:
            site_name (str): Input site name.
            return_type (str): Input return type. Defaults to `'all'`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        J_p = np.zeros((3,self.n_v)) # nv: nDoF
        J_R = np.zeros((3,self.n_v))
        mujoco.mj_jacSite(self.model,self.data,J_p,J_R,self.data.site(site_name).id)
        J_full = np.array(np.vstack([J_p,J_R]))

        if return_type=='all':
            return J_p,J_R,J_full
        elif return_type=='pos':
            return J_p
        elif return_type=='rot':
            return J_R
        elif return_type=='full':
            return J_full
        else:
            print ("[get_J_site] return_type:[%s] is not supported!"%(return_type))
            return None

    def get_body_idxs_from_root_body_name(
            self,
            root_body_name,
            more_body_names = None,
            include_root    = True,
        ):
        """
        Get all body indices in the subtree rooted at 'root_body_name',.

        Parameters:
            root_body_name (str | Path): Input root body name.
            more_body_names (Any | None): Input more body names. Defaults to `None`.
            include_root (bool): Input include root. Defaults to `True`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        # subtree body names
        body_names = self.get_subtree_body_names(root_body_name,include_root=include_root)

        # add extras
        if more_body_names is not None:
            body_names = body_names + list(more_body_names)

        # name -> idx (skip None / unknown)
        body_idxs = []
        for bname in body_names:
            if bname is None:
                continue
            try:
                body_idxs.append(self.body_names.index(bname))
            except ValueError:
                # unknown body name, skip
                continue

        # unique
        body_idxs = list(set(body_idxs))

        # stable order (ascending)
        body_idxs.sort()
        return body_idxs

    def get_geom_idxs_from_body_name(self,body_name):
        """
        Get all geometry indices attached to a given body.

        Parameters:
            body_name (str): Input body name.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        body_idx = self.body_names.index(body_name)
        geom_idxs = [idx for idx,val in enumerate(self.model.geom_bodyid) if val==body_idx]
        return geom_idxs

    def get_geom_idxs_from_root_body_name(
            self,
            root_body_name,
            more_body_names = None,
            include_root    = True,
        ):
        """
        Get all geometry indices in the subtree rooted at 'root_body_name',.

        Parameters:
            root_body_name (str | Path): Input root body name.
            more_body_names (Any | None): Input more body names. Defaults to `None`.
            include_root (bool): Input include root. Defaults to `True`.

        Returns:
            Any: Requested value computed from the current inputs.
        """

        # Get all body names in the subtree rooted at 'root_body_name'
        body_names = self.get_subtree_body_names(root_body_name,include_root)
        if more_body_names is not None:
            body_names += more_body_names

        # Aggregate geom indices
        geom_idxs = []
        for body_name in body_names:
            geom_idxs = geom_idxs + self.get_geom_idxs_from_body_name(body_name)

        # Remove duplications
        geom_idxs = list(set(geom_idxs))

        # Return
        return geom_idxs

    def _set_camera_local_pose(
            self,
            cam_name,
            p_world = None,
            R_world = None,
        ):
        """
        Set a camera pose using world-frame position/rotation.

        Parameters:
            cam_name (str): Input cam name.
            p_world (Any | None): Input p world. Defaults to `None`.
            R_world (Any | None): Input R world. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        cam_id     = self.model.cam(cam_name).id
        parent_bid = int(self.model.cam_bodyid[cam_id])

        if parent_bid == 0:
            p_parent = np.zeros(3,dtype=np.float64)
            R_parent = np.eye(3,dtype=np.float64)
        else:
            p_parent = self.data.xpos[parent_bid].copy()
            R_parent = self.data.xmat[parent_bid].reshape(3,3).copy()

        if p_world is not None:
            p_world = np.asarray(p_world,dtype=np.float64).reshape(3,)
            p_local = R_parent.T @ (p_world-p_parent)
            self.model.cam_pos[cam_id] = p_local

        if R_world is not None:
            R_world = np.asarray(R_world,dtype=np.float64).reshape(3,3)
            R_local = R_parent.T @ R_world
            self.model.cam_quat[cam_id] = r2quat(R_local)

    def get_T_cam_lookat(
            self,
            p_cam,
            p_target,
            up_vector = np.array([0,0,1]),
        ):
        """
        Build a camera pose whose optical axis looks at a target point.

        Parameters:
            p_cam (np.ndarray): Input p cam.
            p_target (np.ndarray): Input p target.
            up_vector (Any): Input up vector. Defaults to `np.array([0, 0, 1])`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        p_cam    = np.asarray(p_cam,dtype=np.float64).reshape(3,)
        p_target = np.asarray(p_target,dtype=np.float64).reshape(3,)
        up_vector = np_uv(up_vector)

        forward = p_target-p_cam
        if np.linalg.norm(forward) < 1e-8:
            raise ValueError("[get_T_cam_lookat] p_cam and p_target are too close.")

        z_axis = -np_uv(forward)                 # camera local -Z looks forward
        x_axis = np.cross(up_vector,z_axis)
        if np.linalg.norm(x_axis) < 1e-8:
            ref = np.array([1.0,0.0,0.0],dtype=np.float64)
            if np.abs(np.dot(ref,z_axis)) > 0.95:
                ref = np.array([0.0,1.0,0.0],dtype=np.float64)
            x_axis = np.cross(ref,z_axis)
        x_axis = np_uv(x_axis)
        y_axis = np_uv(np.cross(z_axis,x_axis))
        R_cam  = np.column_stack([x_axis,y_axis,z_axis])
        T_cam  = pr2t(p_cam,R_cam)
        return T_cam

    def set_cam_lookat(
            self,
            cam_name,
            p_cam,
            p_target,
            up_vector = np.array([0,0,1]),
            forward   = True,
        ):
        """
        Set a named camera pose from a world-frame look-at specification.

        Parameters:
            cam_name (str): Input cam name.
            p_cam (np.ndarray): Input p cam.
            p_target (np.ndarray): Input p target.
            up_vector (Any): Input up vector. Defaults to `np.array([0, 0, 1])`.
            forward (bool): Input forward. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        T_cam = self.get_T_cam_lookat(
            p_cam     = p_cam,
            p_target  = p_target,
            up_vector = up_vector,
        )
        self.set_T(
            name    = cam_name,
            type    = 'camera',
            T       = T_cam,
            forward = forward,
        )

    """
    Set position, rotation, and transformation matrix of an object
    ('base', 'base_body', 'body', 'mocap', 'camera')
        - set_p: set position
        - set_R: set rotation
        - set_T: set transformation
    """
    def set_p(self,name,type,p,forward=True):
        """
        Set the position of a specified object in the simulation.

        Parameters:
            name (str): Input name.
            type (Any): Input type.
            p (np.ndarray): Input p.
            forward (bool): Input forward. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        p = np.asarray(p).flatten()
        if type == 'base' or type == 'base_body': # set position of the base body
            jntadr  = self.model.body(name).jntadr[0]
            qposadr = self.model.jnt_qposadr[jntadr]
            self.data.qpos[qposadr:qposadr+3] = p
        elif type == 'body': # set position of a body
            self.model.body(name).pos = p
        elif type == 'mocap': # set position of a mocap body
            mocap_idx = self.model.body_mocapid[self.body_names.index(name)]
            self.data.mocap_pos[mocap_idx] = p
        elif type == 'camera': # set world position of a camera
            self._set_camera_local_pose(cam_name=name,p_world=p,R_world=None)
        else:
            raise ValueError("Cannot set position for type:[%s]"%(type))

        # Forward kinematics
        if forward:
            mujoco.mj_forward(self.model,self.data)

    def set_R(self,name,type,R,forward=True):
        """
        Set the rotation of a specified object in the simulation.

        Parameters:
            name (str): Input name.
            type (Any): Input type.
            R (Any): Input R.
            forward (bool): Input forward. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        R = np.asarray(R).reshape(3,3)
        if type == 'base' or type == 'base_body': # set rotation of the base body
            jntadr  = self.model.body(name).jntadr[0]
            qposadr = self.model.jnt_qposadr[jntadr]
            self.data.qpos[qposadr+3:qposadr+7] = r2quat(R)
        elif type == 'body': # set rotation of a body
            self.model.body(name).quat = r2quat(R)
        elif type == 'mocap': # set rotation of a mocap body
            mocap_idx = self.model.body_mocapid[self.body_names.index(name)]
            self.data.mocap_quat[mocap_idx] = r2quat(R)
        elif type == 'camera': # set world rotation of a camera
            self._set_camera_local_pose(cam_name=name,p_world=None,R_world=R)
        else:
            raise ValueError("Cannot set rotation for type:[%s]"%(type))

        # Forward kinematics
        if forward:
            mujoco.mj_forward(self.model,self.data)

    def set_pR(self,name,type,p,R,forward=True):
        """
        Set both position and rotation of a specified object in the simulation.

        Parameters:
            name (str): Input name.
            type (Any): Input type.
            p (np.ndarray): Input p.
            R (Any): Input R.
            forward (bool): Input forward. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.set_p(name,type,p,forward=False)
        self.set_R(name,type,R,forward=False)

        # Forward kinematics
        if forward:
            mujoco.mj_forward(self.model,self.data)

    def set_T(self,name,type,T,forward=True):
        """
        Set the transformation matrix of a specified object in the simulation.

        Parameters:
            name (str): Input name.
            type (Any): Input type.
            T (Any): Input T.
            forward (bool): Input forward. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        p = t2p(T)
        R = t2r(T)
        self.set_p(name,type,p,forward=False)
        self.set_R(name,type,R,forward=False)

        # Forward kinematics
        if forward:
            mujoco.mj_forward(self.model,self.data)

    def set_body_color(self,body_name,rgba=(1,0,0,1)): # modified from self to env, parser
        """
        Set the color of a body.

        Parameters:
            body_name (str): Input body name.
            rgba (tuple): Input rgba. Defaults to `(1, 0, 0, 1)`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        body_idx  = self.body_names.index(body_name)
        geom_idxs = [idx for idx,val in enumerate(self.model.geom_bodyid) if val==body_idx]
        for geom_idx in geom_idxs: # for each geom in the body
            self.model.geom(geom_idx).rgba = rgba

    def set_body_colors(self,body_names,rgba=(1,0,0,1)):
        """
        Set the color of multiple bodies.

        Parameters:
            body_names (list[str]): Input body names.
            rgba (tuple): Input rgba. Defaults to `(1, 0, 0, 1)`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if isinstance(body_names,str):
            body_names = [body_names]
        for body_name in body_names:
            self.set_body_color(body_name=body_name,rgba=rgba)

    def revert_body_color(self,body_names=None):
        """
        Revert body geom colors to the original model colors.

        Parameters:
            body_names (Any | None): Input body names. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if body_names is None:
            self.model.geom_rgba[:] = self.geom_rgba_0
            return

        if isinstance(body_names,str):
            body_names = [body_names]

        for body_name in body_names:
            body_idx = self.body_names.index(body_name)
            geom_idxs = [idx for idx,val in enumerate(self.model.geom_bodyid) if val==body_idx]
            for geom_idx in geom_idxs:
                self.model.geom_rgba[geom_idx,:] = self.geom_rgba_0[geom_idx,:]

    def set_geom_alpha(self,alpha=1.0,body_exclude=[]):
        """
        Adjust alpha (transparency) for all geometries, optionally excluding some
        bodies.

        Parameters:
            alpha (float): Input alpha. Defaults to `1.0`.
            body_exclude (list): Input body exclude. Defaults to `[]`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        for g_idx in range(self.n_geom): # for each geom
            geom = self.model.geom(g_idx)
            body_name = self.body_names[geom.bodyid[0]]
            if body_name in body_exclude: continue # exclude certain bodies
            # Change geom alpha
            self.model.geom(g_idx).rgba[3] = alpha

    """
    Grab RGBD images from a fixed camera by name
    """
    def grab_cam_rgbd_imgs(self,cam_name):
        """
        Capture an RGB image from a fixed camera by name.

        Parameters:
            cam_name (str): Input cam name.

        Returns:
            Any: Result computed from the current inputs.
        """
        # Parse camera information
        cam_idx  = self.cam_names.index(cam_name)
        cam      = self.cams[cam_idx]
        cam_fov  = self.cam_fovs[cam_idx]
        viewport = self.cam_viewports[cam_idx]
        # Update
        mujoco.mjv_updateScene(
            self.model,self.data,self.viewer.vopt,self.viewer.pert,
            cam,mujoco.mjtCatBit.mjCAT_ALL,self.viewer.scn)
        mujoco.mjr_render(viewport,self.viewer.scn,self.viewer.ctx)
        # Grab RGBD
        rgb = np.zeros((viewport.height,viewport.width,3),dtype=np.uint8)
        depth_img = np.zeros((viewport.height,viewport.width),dtype=np.float32)
        mujoco.mjr_readPixels(rgb,depth_img,viewport,self.viewer.ctx)
        rgb_img,depth_img = np.flipud(rgb),np.flipud(depth_img)

        # Return
        return rgb_img,depth_img

    def plot_T(
            self,
            T           = None,
            p           = np.array([0,0,0]),
            R           = np.eye(3),
            plot_axis   = True,
            axis_len    = 1.0,
            axis_width  = 0.005,
            axis_rgba   = None,
            axis_alpha  = None,
            plot_sphere = False,
            sphere_r    = 0.05,
            sphere_rgba = [1,0,0,0.5],
            label       = None,
            print_xyz   = False,
        ):
        """
        Plot coordinate axes (and optionally a sphere and label) at the given pose.

        Parameters:
            T (Any | None): Input T. Defaults to `None`.
            p (np.ndarray): Input p. Defaults to `np.array([0, 0, 0])`.
            R (Any): Input R. Defaults to `np.eye(3)`.
            plot_axis (bool): Input plot axis. Defaults to `True`.
            axis_len (float): Input axis len. Defaults to `1.0`.
            axis_width (float): Input axis width. Defaults to `0.005`.
            axis_rgba (Any | None): Input axis rgba. Defaults to `None`.
            axis_alpha (Any | None): Input axis alpha. Defaults to `None`.
            plot_sphere (bool): Input plot sphere. Defaults to `False`.
            sphere_r (float): Input sphere r. Defaults to `0.05`.
            sphere_rgba (list): Input sphere rgba. Defaults to `[1, 0, 0, 0.5]`.
            label (Any | None): Input label. Defaults to `None`.
            print_xyz (bool): Input print xyz. Defaults to `False`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if T is not None: # if T is not None, it overrides p and R
            p = t2p(T)
            R = t2r(T)
        # Add axis markers
        if plot_axis:
            if axis_alpha is None: axis_alpha = 0.9
            if axis_rgba is None:
                rgba_x = [1.0,0.0,0.0,axis_alpha]
                rgba_y = [0.0,1.0,0.0,axis_alpha]
                rgba_z = [0.0,0.0,1.0,axis_alpha]
            else:
                rgba_x = axis_rgba
                rgba_y = axis_rgba
                rgba_z = axis_rgba
            R_x = R@rpy2r(np.deg2rad([0,0,90]))@rpy2r(np.pi/2*np.array([1,0,0]))
            p_x = p+R_x[:,2]*axis_len/2
            if print_xyz: axis_label = 'X-axis'
            else: axis_label = ''
            self.viewer.add_marker(
                pos   = p_x,
                type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
                size  = [axis_width,axis_width,axis_len/2],
                mat   = R_x,
                rgba  = rgba_x,
                label = axis_label,
            )
            R_y = R@rpy2r(np.deg2rad([0,0,90]))@rpy2r(np.pi/2*np.array([0,1,0]))
            p_y = p + R_y[:,2]*axis_len/2
            if print_xyz: axis_label = 'Y-axis'
            else: axis_label = ''
            self.viewer.add_marker(
                pos   = p_y,
                type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
                size  = [axis_width,axis_width,axis_len/2],
                mat   = R_y,
                rgba  = rgba_y,
                label = axis_label,
            )
            R_z = R@rpy2r(np.deg2rad([0,0,90]))@rpy2r(np.pi/2*np.array([0,0,1]))
            p_z = p + R_z[:,2]*axis_len/2
            if print_xyz: axis_label = 'Z-axis'
            else: axis_label = ''
            self.viewer.add_marker(
                pos   = p_z,
                type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
                size  = [axis_width,axis_width,axis_len/2],
                mat   = R_z,
                rgba  = rgba_z,
                label = axis_label,
            )
        # Add sphere marker
        if plot_sphere:
            self.viewer.add_marker(
                pos   = p,
                size  = [sphere_r,sphere_r,sphere_r],
                rgba  = sphere_rgba,
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                label = '')
        # Add label
        if label is not None:
            self.viewer.add_marker(
                pos   = p,
                size  = [0.0001,0.0001,0.0001],
                rgba  = [1,1,1,0.01],
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                label = label,
            )

    def plot_global_coordinate_axes(
            self,
            p_offset  = np.array([0,0,0]),
            axis_len  = 1.0,
            r         = 0.001,
            alpha     = 0.5,
            print_xyz = False,
        ):
        """
        Plot the global coordinate axes in the viewer.

        Parameters:
            p_offset (np.ndarray): Input p offset. Defaults to `np.array([0, 0, 0])`.
            axis_len (float): Input axis len. Defaults to `1.0`.
            r (float): Input r. Defaults to `0.001`.
            alpha (float): Input alpha. Defaults to `0.5`.
            print_xyz (bool): Input print xyz. Defaults to `False`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.plot_T(
            p          = np.array(p_offset),
            axis_len   = axis_len,
            axis_width = r,
            axis_rgba  = None,
            axis_alpha = alpha,
            print_xyz  = print_xyz,
        )

    def plot_sphere(self,p,r=0.1,rgba=[1,1,1,1],label='',show_xyz=False):
        """
        Plot a sphere marker at the specified position.

        Parameters:
            p (np.ndarray): Input p.
            r (float): Input r. Defaults to `0.1`.
            rgba (list): Input rgba. Defaults to `[1, 1, 1, 1]`.
            label (str): Input label. Defaults to `''`.
            show_xyz (bool): Input show xyz. Defaults to `False`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if p is None: return
        p = np.asarray(p)
        if show_xyz:
            label = '(%.2f, %.2f, %.2f)'%(p[0],p[1],p[2])
        if len(p) == 2: # only x and y are given (pad z=0)
            self.viewer.add_marker(
                pos   = np.append(p,[0]),
                size  = [r,r,r],
                rgba  = rgba,
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                label = label,
            )
        elif len(p) == 3:
            self.viewer.add_marker(
                pos   = p,
                size  = [r,r,r],
                rgba  = rgba,
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                label = label,
            )

    def plot_text(self,p,text):
        """
        Plot a text marker at the specified position.

        Parameters:
            p (np.ndarray): Input p.
            text (Any): Input text.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if p is None: return
        p = np.asarray(p)
        self.viewer.add_marker(
            pos   = p,
            size  = [0.0001,0.0001,0.0001],
            rgba  = [1,1,1,0.01],
            type  = mujoco.mjtGeom.mjGEOM_SPHERE,
            label = text,
        )

    def plot_spheres(self,p_list,r=0.1,rgba=[1,1,1,1],label=''):
        """
        Plot multiple spheres at the positions given in p_list.

        Parameters:
            p_list (np.ndarray): Input p list.
            r (float): Input r. Defaults to `0.1`.
            rgba (list): Input rgba. Defaults to `[1, 1, 1, 1]`.
            label (str): Input label. Defaults to `''`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        for p in p_list:
            self.plot_sphere(p=p,r=r,rgba=rgba,label=label)

    def plot_box(
            self,
            T          = None,
            p          = np.array([0,0,0]),
            R          = np.eye(3),
            xlen       = 0.1,
            ylen       = 0.1,
            zlen       = 0.1,
            xyz_len    = None,
            rgba       = [0.5,0.5,0.5,0.5],
            label      = '',
            plot_axes  = False,
            axis_len   = 0.05,
            axis_width = 0.002,
            axis_alpha = 0.9,
        ):
        """
        Plot a box marker at the specified pose.

        Parameters:
            T (Any | None): Input T. Defaults to `None`.
            p (np.ndarray): Input p. Defaults to `np.array([0, 0, 0])`.
            R (Any): Input R. Defaults to `np.eye(3)`.
            xlen (float): Input xlen. Defaults to `0.1`.
            ylen (float): Input ylen. Defaults to `0.1`.
            zlen (float): Input zlen. Defaults to `0.1`.
            xyz_len (Any | None): Input xyz len. Defaults to `None`.
            rgba (list): Input rgba. Defaults to `[0.5, 0.5, 0.5, 0.5]`.
            label (str): Input label. Defaults to `''`.
            plot_axes (bool): Input plot axes. Defaults to `False`.
            axis_len (float): Input axis len. Defaults to `0.05`.
            axis_width (float): Input axis width. Defaults to `0.002`.
            axis_alpha (float): Input axis alpha. Defaults to `0.9`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if T is not None: # if T is given, it overrides p and R
            p = t2p(T)
            R = t2r(T)
        pos = np.asarray(p)
        mat = np.asarray(R, dtype=np.float64).reshape(9,)

        if xyz_len is not None: # if xyz_len is given, it overrides xlen, ylen, zlen
            xyz_len = np.asarray(xyz_len)
            xlen = xyz_len[0]
            ylen = xyz_len[1]
            zlen = xyz_len[2]

        self.viewer.add_marker(
            pos   = pos,
            mat   = mat,
            type  = mujoco.mjtGeom.mjGEOM_BOX,
            size  = [xlen/2,ylen/2,zlen/2],
            rgba  = rgba,
            label = label,
        )

        if plot_axes:
            self.plot_T(
                p          = p,
                R          = R,
                axis_len   = axis_len,
                axis_width = axis_width,
                axis_alpha = axis_alpha,
            )

    def plot_box_from_ranges(
            self,
            x_range    = (0,1),
            y_range    = (0,1),
            z_range    = (0,1),
            rgba       = [0.5,0.5,0.5,0.5],
            label      = '',
            plot_axes  = False,
            axis_len   = 0.05,
            axis_width = 0.002,
            axis_alpha = 0.9,
        ):
        """
        Plot an axis-aligned box marker from x, y, and z ranges.

        Parameters:
            x_range (tuple): Input x range. Defaults to `(0, 1)`.
            y_range (tuple): Input y range. Defaults to `(0, 1)`.
            z_range (tuple): Input z range. Defaults to `(0, 1)`.
            rgba (list): Input rgba. Defaults to `[0.5, 0.5, 0.5, 0.5]`.
            label (str): Input label. Defaults to `''`.
            plot_axes (bool): Input plot axes. Defaults to `False`.
            axis_len (float): Input axis len. Defaults to `0.05`.
            axis_width (float): Input axis width. Defaults to `0.002`.
            axis_alpha (float): Input axis alpha. Defaults to `0.9`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        x_range = np.asarray(x_range,dtype=np.float64)
        y_range = np.asarray(y_range,dtype=np.float64)
        z_range = np.asarray(z_range,dtype=np.float64)
        p = np.array([
            0.5*(x_range[0]+x_range[1]),
            0.5*(y_range[0]+y_range[1]),
            0.5*(z_range[0]+z_range[1]),
        ])
        xyz_len = np.array([
            abs(x_range[1]-x_range[0]),
            abs(y_range[1]-y_range[0]),
            abs(z_range[1]-z_range[0]),
        ])
        self.plot_box(
            p          = p,
            R          = np.eye(3),
            xyz_len    = xyz_len,
            rgba       = rgba,
            label      = label,
            plot_axes  = plot_axes,
            axis_len   = axis_len,
            axis_width = axis_width,
            axis_alpha = axis_alpha,
        )

    def plot_capsule(
            self,
            T          = None,
            p          = np.array([0,0,0]),
            R          = np.eye(3),
            r          = 0.1,
            h          = 0.1,
            rgba       = [0.5,0.5,0.5,0.5],
            label      = '',
            plot_axes  = False,
            axis_len   = 0.05,
            axis_width = 0.002,
            axis_alpha = 0.9,
        ):
        """
        Plot a capsule marker.

        Parameters:
            T (Any | None): Input T. Defaults to `None`.
            p (np.ndarray): Input p. Defaults to `np.array([0, 0, 0])`.
            R (Any): Input R. Defaults to `np.eye(3)`.
            r (float): Input r. Defaults to `0.1`.
            h (float): Input h. Defaults to `0.1`.
            rgba (list): Input rgba. Defaults to `[0.5, 0.5, 0.5, 0.5]`.
            label (str): Input label. Defaults to `''`.
            plot_axes (bool): Input plot axes. Defaults to `False`.
            axis_len (float): Input axis len. Defaults to `0.05`.
            axis_width (float): Input axis width. Defaults to `0.002`.
            axis_alpha (float): Input axis alpha. Defaults to `0.9`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if T is not None: # if T is not None, it overrides p and R
            p = t2p(T)
            R = t2r(T)
        else:
            p = np.asarray(p)
            R = np.asarray(R)
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_CAPSULE,
            size  = [r,r,h],
            rgba  = rgba,
            label = label,
        )

        if plot_axes:
            self.plot_T(
                p          = p,
                R          = R,
                axis_len   = axis_len,
                axis_width = axis_width,
                axis_alpha = axis_alpha,
            )

    def plot_cylinder(
            self,
            T          = None,
            p          = np.array([0,0,0]),
            R          = np.eye(3),
            r          = 0.1,
            h          = 0.1,
            rgba       = [0.5,0.5,0.5,0.5],
            label      = '',
            plot_axes  = False,
            axis_len   = 0.05,
            axis_width = 0.002,
            axis_alpha = 0.9,
        ):
        """
        Plot a cylinder marker.

        Parameters:
            T (Any | None): Input T. Defaults to `None`.
            p (np.ndarray): Input p. Defaults to `np.array([0, 0, 0])`.
            R (Any): Input R. Defaults to `np.eye(3)`.
            r (float): Input r. Defaults to `0.1`.
            h (float): Input h. Defaults to `0.1`.
            rgba (list): Input rgba. Defaults to `[0.5, 0.5, 0.5, 0.5]`.
            label (str): Input label. Defaults to `''`.
            plot_axes (bool): Input plot axes. Defaults to `False`.
            axis_len (float): Input axis len. Defaults to `0.05`.
            axis_width (float): Input axis width. Defaults to `0.002`.
            axis_alpha (float): Input axis alpha. Defaults to `0.9`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if T is not None: # if T is not None, it overrides p and R
            p = t2p(T)
            R = t2r(T)
        else:
            p = np.asarray(p)
            R = np.asarray(R)
        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
            size  = [r,r,h],
            rgba  = rgba,
            label = label,
        )

        if plot_axes:
            self.plot_T(
                p          = p,
                R          = R,
                axis_len   = axis_len,
                axis_width = axis_width,
                axis_alpha = axis_alpha,
            )

    def plot_ellipsoid(
            self,
            T          = None,
            p          = np.array([0,0,0]),
            R          = np.eye(3),
            rx         = 0.1,
            ry         = 0.1,
            rz         = 0.1,
            rgba       = [0.5,0.5,0.5,0.5],
            label      = '',
            plot_axes  = False,
            axis_len   = 0.05,
            axis_width = 0.002,
            axis_alpha = 0.9
        ):
        """
        Plot an ellipsoid marker.

        Parameters:
            T (Any | None): Input T. Defaults to `None`.
            p (np.ndarray): Input p. Defaults to `np.array([0, 0, 0])`.
            R (Any): Input R. Defaults to `np.eye(3)`.
            rx (float): Input rx. Defaults to `0.1`.
            ry (float): Input ry. Defaults to `0.1`.
            rz (float): Input rz. Defaults to `0.1`.
            rgba (list): Input rgba. Defaults to `[0.5, 0.5, 0.5, 0.5]`.
            label (str): Input label. Defaults to `''`.
            plot_axes (bool): Input plot axes. Defaults to `False`.
            axis_len (float): Input axis len. Defaults to `0.05`.
            axis_width (float): Input axis width. Defaults to `0.002`.
            axis_alpha (float): Input axis alpha. Defaults to `0.9`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if T is not None: # if T is not None, it overrides p and R
            p = t2p(T)
            R = t2r(T)
        else:
            p = np.asarray(p)
            R = np.asarray(R)

        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_ELLIPSOID,
            size  = [rx,ry,rz],
            rgba  = rgba,
            label = label,
        )

        if plot_axes:
            self.plot_T(
                p          = p,
                R          = R,
                axis_len   = axis_len,
                axis_width = axis_width,
                axis_alpha = axis_alpha,
            )

    def plot_arrow(
            self,
            T          = None,
            p          = np.array([0,0,0]),
            R          = np.eye(3),
            r          = 0.01,
            h          = 0.1,
            rgba       = [1,0,0,1],
            label      = '',
            plot_axes  = False,
            axis_len   = 0.05,
            axis_width = 0.002,
            axis_alpha = 0.9
        ):
        """
        Plot an arrow marker at the given pose.

        Parameters:
            T (Any | None): Input T. Defaults to `None`.
            p (np.ndarray): Input p. Defaults to `np.array([0, 0, 0])`.
            R (Any): Input R. Defaults to `np.eye(3)`.
            r (float): Input r. Defaults to `0.01`.
            h (float): Input h. Defaults to `0.1`.
            rgba (list): Input rgba. Defaults to `[1, 0, 0, 1]`.
            label (str): Input label. Defaults to `''`.
            plot_axes (bool): Input plot axes. Defaults to `False`.
            axis_len (float): Input axis len. Defaults to `0.05`.
            axis_width (float): Input axis width. Defaults to `0.002`.
            axis_alpha (float): Input axis alpha. Defaults to `0.9`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if T is not None: # if T is not None, it overrides p and R
            p = t2p(T)
            R = t2r(T)
        else:
            p = np.asarray(p)
            R = np.asarray(R)

        self.viewer.add_marker(
            pos   = p,
            mat   = R,
            type  = mujoco.mjtGeom.mjGEOM_ARROW,
            size  = [r,r,h*2],
            rgba  = rgba,
            label = label,
        )

        if plot_axes:
            self.plot_T(
                p          = p,
                R          = R,
                axis_len   = axis_len,
                axis_width = axis_width,
                axis_alpha = axis_alpha,
            )

    def plot_arrow_fr2to(
            self,
            p_fr,
            p_to,
            r          = 1.0,
            rgba       = [0.5,0.5,0.5,0.5],
            label      = '',
            label_to   = None,
            plot_axes  = False,
            axis_len   = 0.05,
            axis_width = 0.002,
            axis_alpha = 0.9
        ):
        """
        Plot an arrow from point p_fr to point p_to.

        Parameters:
            p_fr (np.ndarray): Input p fr.
            p_to (np.ndarray): Input p to.
            r (float): Input r. Defaults to `1.0`.
            rgba (list): Input rgba. Defaults to `[0.5, 0.5, 0.5, 0.5]`.
            label (str): Input label. Defaults to `''`.
            label_to (Any | None): Input label to. Defaults to `None`.
            plot_axes (bool): Input plot axes. Defaults to `False`.
            axis_len (float): Input axis len. Defaults to `0.05`.
            axis_width (float): Input axis width. Defaults to `0.002`.
            axis_alpha (float): Input axis alpha. Defaults to `0.9`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        # Ensure p_fr and p_to are numpy arrays
        p_fr = np.asarray(p_fr)
        p_to = np.asarray(p_to)
        R_fr2to = get_R_from_twopoints(p_fr=p_fr,p_to=p_to)
        self.viewer.add_marker(
            pos   = p_fr,
            mat   = R_fr2to,
            type  = mujoco.mjtGeom.mjGEOM_ARROW,
            size  = [r,r,np.linalg.norm(p_to-p_fr)*2],
            rgba  = rgba,
            label = label,
        )
        if label_to is not None:
            self.viewer.add_marker(
                pos   = p_to,
                size  = [0.0001,0.0001,0.0001],
                rgba  = [1,1,1,0.0],
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                label = label_to,
            )

        if plot_axes:
            self.plot_T(
                p          = p_fr,
                R          = R_fr2to,
                axis_len   = axis_len,
                axis_width = axis_width,
                axis_alpha = axis_alpha,
            )

    def plot_line_fr2to(
            self,
            p_fr,
            p_to,
            rgba  = [0.5,0.5,0.5,0.5],
            label = '',
        ):
        """
        Plot a line connecting two points.

        Parameters:
            p_fr (np.ndarray): Input p fr.
            p_to (np.ndarray): Input p to.
            rgba (list): Input rgba. Defaults to `[0.5, 0.5, 0.5, 0.5]`.
            label (str): Input label. Defaults to `''`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        # Ensure p_fr and p_to are numpy arrays
        p_fr = np.asarray(p_fr)
        p_to = np.asarray(p_to)
        if np.linalg.norm(p_to-p_fr) < 1e-6: # if the two points are too close, do not plot
            return
        R_fr2to = get_R_from_twopoints(p_fr=p_fr,p_to=p_to)
        self.viewer.add_marker(
            pos   = p_fr,
            mat   = R_fr2to,
            type  = mujoco.mjtGeom.mjGEOM_LINE,
            size  = np.linalg.norm(p_to-p_fr),
            rgba  = rgba,
            label = label,
        )

    def plot_cylinder_fr2to(
            self,
            p_fr,
            p_to,
            r        = 0.01,
            rgba     = [0.5,0.5,0.5,0.5],
            label    = '',
            label_to = None
        ):
        """
        Plot a cylinder marker between two points.

        Parameters:
            p_fr (np.ndarray): Input p fr.
            p_to (np.ndarray): Input p to.
            r (float): Input r. Defaults to `0.01`.
            rgba (list): Input rgba. Defaults to `[0.5, 0.5, 0.5, 0.5]`.
            label (str): Input label. Defaults to `''`.
            label_to (Any | None): Input label to. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        # Ensure p_fr and p_to are numpy arrays
        p_fr = np.asarray(p_fr)
        p_to = np.asarray(p_to)
        if np.linalg.norm(p_to-p_fr) < 1e-6: # if the two points are too close, do not plot
            return
        R_fr2to = get_R_from_twopoints(p_fr=p_fr,p_to=p_to)
        self.viewer.add_marker(
            pos   = (p_fr+p_to)/2,
            mat   = R_fr2to,
            type  = mujoco.mjtGeom.mjGEOM_CYLINDER,
            size  = [r,r,np.linalg.norm(p_to-p_fr)/2],
            rgba  = rgba,
            label = label,
        )
        if label_to is not None:
            self.viewer.add_marker(
                pos   = p_to,
                size  = [0.0001,0.0001,0.0001],
                rgba  = [1,1,1,0.0],
                type  = mujoco.mjtGeom.mjGEOM_SPHERE,
                label = label_to,
            )

    def plot_mesh(
            self,
            mesh_id    = None,
            mesh_name  = None,
            p          = np.array([0,0,0]),
            R          = np.eye(3),
            scale      = (1,1,1),
            rgba       = (0.2,0.2,0.5,1),
            label      = '',
            plot_axes  = False,
            axis_len   = 0.05,
            axis_width = 0.002,
            axis_alpha = 0.9,
        ):
        """
        TODO: Check the sanity of this function..

        Parameters:
            mesh_id (Any | None): Input mesh id. Defaults to `None`.
            mesh_name (Any | None): Input mesh name. Defaults to `None`.
            p (np.ndarray): Input p. Defaults to `np.array([0, 0, 0])`.
            R (Any): Input R. Defaults to `np.eye(3)`.
            scale (tuple): Input scale. Defaults to `(1, 1, 1)`.
            rgba (tuple): Input rgba. Defaults to `(0.2, 0.2, 0.5, 1)`.
            label (str): Input label. Defaults to `''`.
            plot_axes (bool): Input plot axes. Defaults to `False`.
            axis_len (float): Input axis len. Defaults to `0.05`.
            axis_width (float): Input axis width. Defaults to `0.002`.
            axis_alpha (float): Input axis alpha. Defaults to `0.9`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """

        # Convert inputs to numpy arrays
        p     = np.asarray(p)
        R     = np.asarray(R)
        scale = np.asarray(scale)

        # Get mesh ID if not provided
        if mesh_id is None:
            mesh_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_MESH,
                mesh_name,
            )

        # Add marker
        self.viewer.add_marker(
            pos    = p,
            mat    = R,
            type   = mujoco.mjtGeom.mjGEOM_MESH,
            size   = scale,
            dataid = mesh_id,
            rgba   = rgba,
            label  = label,
        )

        # Plot axes if needed
        if plot_axes:
            self.plot_T(
                p          = p,
                R          = R,
                axis_len   = axis_len,
                axis_width = axis_width,
                axis_alpha = axis_alpha,
            )

    def plot_sites(self,site_names=None,show_names=False):
        """
        Plot spheres at the positions of specified sites.

        Parameters:
            site_names (Any | None): Input site names. Defaults to `None`.
            show_names (bool): Input show names. Defaults to `False`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if site_names is None: # if site_names is not provided, plot all sites
            site_names = self.site_names
        for site_name in site_names: # iterate through all site names
            if site_name: # if site_name is valid
                p    = self.get_p(site_name,'site')
                R    = self.get_R(site_name,'site')
                size = self.model.site(site_name).size
                type = self.model.site(site_name).type
                rgba = self.model.site(site_name).rgba
                if show_names: label = site_name
                else: label = ''

                if type == mujoco.mjtGeom.mjGEOM_SPHERE:
                    r = size[0]
                    self.plot_sphere(p,r,rgba,label)
                elif type == mujoco.mjtGeom.mjGEOM_BOX:
                    xlen,ylen,zlen = size[0]*2,size[1]*2,size[2]*2
                    self.plot_box(
                        p        = p,
                        R        = R,
                        xlen     = xlen,
                        ylen     = ylen,
                        zlen     = zlen,
                        rgba     = rgba,
                        label    = label,
                    )
                else:
                    print("Site type [%d] not supported for plotting."%(type))

    def get_contact_info(self,forward=True,compute_force=True):
        """
        Get detailed contact information from the simulation.

        Parameters:
            forward (bool): Input forward. Defaults to `True`.
            compute_force (bool): Input compute force. Defaults to `True`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        # Perform a forward simulation step to ensure contact data is up-to-date
        if forward:
            self.forward()

        # Initialize lists to store contact information
        n_contact,contact_idx_list,contact_list,p_contact_list = 0,[],[],[]
        geom1_idx_list,geom2_idx_list = [],[]
        body1_idx_list,body2_idx_list = [],[]
        R_frame_list,f_world_list,norm_dir_list = [],[],[]
        min_contact_dist = np.inf

        # Iterate through all contacts
        for c_idx in range(self.data.ncon): # iterate through all contacts
            contact = self.data.contact[c_idx]

            # Contact position and frame orientation
            p_contact = np.array(contact.pos,dtype=np.float64) # (3,)
            R_frame = np.array(contact.frame,dtype=np.float64).reshape(3,3) # (3 x 3)

            # Contact force in world coordinates
            if compute_force:
                f_local = np.zeros(6,dtype=np.float64) # (6,)
                mujoco.mj_contactForce(self.model,self.data,c_idx,f_local) # contact force
                f_world = R_frame.T@f_local[:3] # (3,)
            else:
                f_world = np.zeros(3,dtype=np.float64) # (3,)
            norm_dir = R_frame[0,:] # contact normal direction (3,)

            # Contacting geoms
            geom1_idx,geom2_idx = contact.geom1,contact.geom2
            if (geom1_idx<0) or (geom2_idx<0): continue # skip invalid contacts

            # Append contact related information
            n_contact = n_contact + 1
            body1_idx,body2_idx = self.model.geom_bodyid[geom1_idx],self.model.geom_bodyid[geom2_idx]
            contact_idx_list.append(c_idx)
            contact_list.append(contact)
            p_contact_list.append(p_contact)
            geom1_idx_list.append(geom1_idx)
            geom2_idx_list.append(geom2_idx)
            body1_idx_list.append(body1_idx)
            body2_idx_list.append(body2_idx)
            R_frame_list.append(R_frame)
            f_world_list.append(f_world)
            norm_dir_list.append(norm_dir)

            # Min contact distance
            if contact.dist < min_contact_dist:
                min_contact_dist = contact.dist

        # Return
        contact_info = {
            'n_contact': n_contact,
            'contact_idx_list': contact_idx_list,
            'contact_list': contact_list,
            'p_contact_list': p_contact_list,
            'geom1_idx_list': geom1_idx_list,
            'geom2_idx_list': geom2_idx_list,
            'body1_idx_list': body1_idx_list,
            'body2_idx_list': body2_idx_list,
            'R_frame_list': R_frame_list,
            'f_world_list': f_world_list,
            'norm_dir_list': norm_dir_list,
            'min_contact_dist': min_contact_dist,
        }
        return contact_info

    def get_closest_contact_info(
            self,
            contact_info  = None,
            forward       = True,
            compute_force = False,
            topk          = None,
        ):
        """
        Return the closest currently generated MuJoCo contact(s).

        This uses ``mjData.contact`` after ``mj_forward``. Therefore it only
        considers contacts that MuJoCo's collision pipeline generated under the
        current margin/gap and collision-filter settings.

        Parameters:
            contact_info (dict | None): Optional result from get_contact_info().
            forward (bool): If True and contact_info is None, refresh contacts.
            compute_force (bool): If True and contact_info is None, compute force.
            topk (int | None): If None, return one closest contact dictionary.
                If an integer is given, return a list with up to topk closest
                contact dictionaries sorted by signed contact distance.

        Returns:
            dict | list[dict] | None: Closest contact summary, top-k contact
                summaries, or None when there is no generated contact and topk
                is None.
        """
        if contact_info is None:
            contact_info = self.get_contact_info(
                forward       = forward,
                compute_force = compute_force,
            )

        if topk is not None:
            topk = max(0,int(topk))

        if contact_info["n_contact"] <= 0:
            if topk is None:
                return None
            return []

        dists = np.array(
            [contact.dist for contact in contact_info["contact_list"]],
            dtype = np.float64,
        )
        if topk is None:
            c_info_idxs = [int(np.argmin(dists))]
        else:
            c_info_idxs = [int(idx) for idx in np.argsort(dists)[:topk]]

        closest_contact_infos = []
        for c_info_idx in c_info_idxs:
            contact = contact_info["contact_list"][c_info_idx]
            geom1_idx = int(contact_info["geom1_idx_list"][c_info_idx])
            geom2_idx = int(contact_info["geom2_idx_list"][c_info_idx])
            body1_idx = int(contact_info["body1_idx_list"][c_info_idx])
            body2_idx = int(contact_info["body2_idx_list"][c_info_idx])
            point = contact_info["p_contact_list"][c_info_idx].copy()
            normal = contact_info["norm_dir_list"][c_info_idx].copy()
            p1 = point - 0.5*float(contact.dist)*normal
            p2 = point + 0.5*float(contact.dist)*normal

            closest_contact_infos.append({
                "contact_info_idx": c_info_idx,
                "contact_idx": int(contact_info["contact_idx_list"][c_info_idx]),
                "contact": contact,
                "dist": float(contact.dist),
                "point": point,
                "p_contact": point.copy(),
                "p1": p1,
                "p2": p2,
                "fromto": np.concatenate((p1,p2)),
                "R_frame": contact_info["R_frame_list"][c_info_idx].copy(),
                "normal": normal,
                "f_world": contact_info["f_world_list"][c_info_idx].copy(),
                "geom1_idx": geom1_idx,
                "geom2_idx": geom2_idx,
                "geom1_name": self.geom_names[geom1_idx],
                "geom2_name": self.geom_names[geom2_idx],
                "body1_idx": body1_idx,
                "body2_idx": body2_idx,
                "body1_name": self.body_names[body1_idx],
                "body2_name": self.body_names[body2_idx],
                "includemargin": float(contact.includemargin),
                "efc_address": int(contact.efc_address),
            })

        if topk is None:
            return closest_contact_infos[0]
        return closest_contact_infos

    def get_geom_distance_info(
            self,
            geom1_idx  = None,
            geom2_idx  = None,
            geom1_name = None,
            geom2_name = None,
            distmax    = np.inf,
            forward    = False,
        ):
        """
        Return signed distance and nearest surface points for one geom pair.

        Parameters:
            geom1_idx (Any | None): Input geom1 idx. Defaults to `None`.
            geom2_idx (Any | None): Input geom2 idx. Defaults to `None`.
            geom1_name (Any | None): Input geom1 name. Defaults to `None`.
            geom2_name (Any | None): Input geom2 name. Defaults to `None`.
            distmax (Any): Input distmax. Defaults to `np.inf`.
            forward (bool): Input forward. Defaults to `False`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if forward:
            self.forward()

        if geom1_name is not None:
            geom1_idx = self.geom_name_to_id[geom1_name]
        if geom2_name is not None:
            geom2_idx = self.geom_name_to_id[geom2_name]
        if geom1_idx is None or geom2_idx is None:
            raise ValueError("geom1/geom2 must be provided by index or name.")

        geom1_idx = int(geom1_idx)
        geom2_idx = int(geom2_idx)
        fromto = np.zeros(6,dtype=np.float64)
        dist = float(mujoco.mj_geomDistance(
            self.model,self.data,geom1_idx,geom2_idx,float(distmax),fromto))
        p1 = fromto[:3].copy()
        p2 = fromto[3:].copy()
        vec12 = p2 - p1
        vec12_norm = float(np.linalg.norm(vec12))
        if vec12_norm > 0.0:
            normal = vec12 / vec12_norm
        else:
            normal = np.zeros(3,dtype=np.float64)
        body1_idx = int(self.model.geom_bodyid[geom1_idx])
        body2_idx = int(self.model.geom_bodyid[geom2_idx])

        return {
            "dist": dist,
            "fromto": fromto.copy(),
            "p1": p1,
            "p2": p2,
            "point": 0.5*(p1+p2),
            "normal": normal,
            "geom1_idx": geom1_idx,
            "geom2_idx": geom2_idx,
            "geom1_name": self.geom_names[geom1_idx],
            "geom2_name": self.geom_names[geom2_idx],
            "body1_idx": body1_idx,
            "body2_idx": body2_idx,
            "body1_name": self.body_names[body1_idx],
            "body2_name": self.body_names[body2_idx],
        }

    def get_root_geom_pair_candidates(
            self,
            root_body_name,
            init_exclude_dist               = 0.01,
            exclude_subtree_root_body_names = None,
            forward                         = True,
            exclude_current_contacts        = True,
            exclude_initial_close_pairs     = True,
            skip_visual_geoms               = True,
            skip_same_body                  = True,
            skip_parent_child               = True,
            respect_body_excludes           = True,
            respect_contype_affinity        = True,
        ):
        """
        Build geom-pair candidates where at least one geom belongs to a root body
        subtree.

        Parameters:
            root_body_name (str | Path): Input root body name.
            init_exclude_dist (float): Input init exclude dist. Defaults to `0.01`.
            exclude_subtree_root_body_names (Any | None): Input exclude subtree root
                                                          body names. Defaults to
                                                          `None`.
            forward (bool): Input forward. Defaults to `True`.
            exclude_current_contacts (bool): Input exclude current contacts. Defaults
                                             to `True`.
            exclude_initial_close_pairs (bool): Input exclude initial close pairs.
                                                Defaults to `True`.
            skip_visual_geoms (bool): Input skip visual geoms. Defaults to `True`.
            skip_same_body (bool): Input skip same body. Defaults to `True`.
            skip_parent_child (bool): Input skip parent child. Defaults to `True`.
            respect_body_excludes (bool): Input respect body excludes. Defaults to
                                          `True`.
            respect_contype_affinity (bool): Input respect contype affinity. Defaults
                                             to `True`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if forward:
            self.forward()

        if exclude_subtree_root_body_names is None:
            exclude_subtree_root_body_names = []
        elif isinstance(exclude_subtree_root_body_names,str):
            exclude_subtree_root_body_names = [exclude_subtree_root_body_names]

        if skip_visual_geoms:
            collision_geom_idxs = [
                g_idx for g_idx in range(self.model.ngeom)
                if not (
                    int(self.model.geom_contype[g_idx]) == 0 and
                    int(self.model.geom_conaffinity[g_idx]) == 0
                )
            ]
        else:
            collision_geom_idxs = list(range(self.model.ngeom))
        collision_geom_idx_set = set(collision_geom_idxs)

        root_geom_idxs = [
            g_idx for g_idx in self.get_geom_idxs_from_root_body_name(root_body_name)
            if g_idx in collision_geom_idx_set
        ]
        exclude_geom_idx_sets = [
            set(self.get_geom_idxs_from_root_body_name(body_name))
            for body_name in exclude_subtree_root_body_names
        ]

        excluded_geom_pair_keys = set()
        if exclude_current_contacts:
            contact_info = self.get_contact_info(forward=False,compute_force=False)
            for g1,g2 in zip(contact_info["geom1_idx_list"],contact_info["geom2_idx_list"]):
                excluded_geom_pair_keys.add((min(int(g1),int(g2)),max(int(g1),int(g2))))

        body_exclude_signatures = set()
        if respect_body_excludes and hasattr(self.model,"exclude_signature"):
            body_exclude_signatures = set(int(sig) for sig in self.model.exclude_signature)

        candidate_geom_pair_keys = set()
        fromto = np.zeros(6,dtype=np.float64)
        for g1 in root_geom_idxs:
            b1 = int(self.model.geom_bodyid[g1])
            for g2 in collision_geom_idxs:
                if g1 == g2:
                    continue
                key = (min(g1,g2),max(g1,g2))
                if key in candidate_geom_pair_keys or key in excluded_geom_pair_keys:
                    continue
                if any(g1 in geom_idx_set and g2 in geom_idx_set for geom_idx_set in exclude_geom_idx_sets):
                    continue

                b2 = int(self.model.geom_bodyid[g2])
                if skip_same_body and b1 == b2:
                    continue

                if respect_body_excludes:
                    sig12 = ((b1+1) << 16) + (b2+1)
                    sig21 = ((b2+1) << 16) + (b1+1)
                    if sig12 in body_exclude_signatures or sig21 in body_exclude_signatures:
                        continue

                if skip_parent_child:
                    parent1 = int(self.model.body_parentid[b1])
                    parent2 = int(self.model.body_parentid[b2])
                    if (parent1 == b2 and b2 != 0) or (parent2 == b1 and b1 != 0):
                        continue

                if respect_contype_affinity:
                    can_collide = (
                        int(self.model.geom_contype[g1]) & int(self.model.geom_conaffinity[g2])
                    ) or (
                        int(self.model.geom_contype[g2]) & int(self.model.geom_conaffinity[g1])
                    )
                    if not can_collide:
                        continue

                lower_bound = (
                    np.linalg.norm(self.data.geom_xpos[g2]-self.data.geom_xpos[g1])
                    - float(self.model.geom_rbound[g1])
                    - float(self.model.geom_rbound[g2])
                )
                if exclude_initial_close_pairs and lower_bound <= init_exclude_dist:
                    fromto[:] = 0.0
                    dist = float(mujoco.mj_geomDistance(
                        self.model,self.data,g1,g2,float(init_exclude_dist),fromto))
                    if dist <= init_exclude_dist:
                        excluded_geom_pair_keys.add(key)
                        continue

                candidate_geom_pair_keys.add(key)

        candidate_geom_pairs = np.array(sorted(candidate_geom_pair_keys),dtype=np.int32)
        return {
            "candidate_geom_pairs": candidate_geom_pairs,
            "excluded_geom_pair_keys": excluded_geom_pair_keys,
            "root_geom_idxs": root_geom_idxs,
            "collision_geom_idxs": collision_geom_idxs,
            "exclude_subtree_geom_idx_sets": exclude_geom_idx_sets,
            "stats": {
                "n_root_geom": len(root_geom_idxs),
                "n_collision_geom": len(collision_geom_idxs),
                "n_candidate_geom_pair": len(candidate_geom_pairs),
                "n_excluded_geom_pair": len(excluded_geom_pair_keys),
            },
        }

    def get_root_body_involved_geom_pair_candidates(self,*args,**kwargs):
        """
        Backward-compatible alias for get_root_geom_pair_candidates().

        Parameters:
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        return self.get_root_geom_pair_candidates(*args,**kwargs)

    def get_geom_pair_distance_infos(
            self,
            geom_pairs,
            distmax            = np.inf,
            topk               = None,
            forward            = False,
            sort               = True,
            sanity_check       = True,
            lower_bound_tol    = 1e-6,
            fromto_dist_tol    = 1e-4,
        ):
        """
        Return signed distance information for geom pairs at the current pose.

        Parameters:
            geom_pairs (Any): Input geom pairs.
            distmax (Any): Input distmax. Defaults to `np.inf`.
            topk (Any | None): Input topk. Defaults to `None`.
            forward (bool): Input forward. Defaults to `False`.
            sort (bool): Input sort. Defaults to `True`.
            sanity_check (bool): Input sanity check. Defaults to `True`.
            lower_bound_tol (float): Input lower bound tol. Defaults to `1e-06`.
            fromto_dist_tol (float): Input fromto dist tol. Defaults to `0.0001`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if forward:
            self.forward()

        geom_pairs = np.asarray(geom_pairs,dtype=np.int32)
        if geom_pairs.size == 0:
            return []
        geom_pairs = geom_pairs.reshape(-1,2)

        if topk is not None:
            topk = max(0,int(topk))
            if topk == 0:
                return []

        geom1_idxs = geom_pairs[:,0]
        geom2_idxs = geom_pairs[:,1]
        lower_bounds = (
            np.linalg.norm(self.data.geom_xpos[geom2_idxs]-self.data.geom_xpos[geom1_idxs],axis=1)
            - self.model.geom_rbound[geom1_idxs]
            - self.model.geom_rbound[geom2_idxs]
        )
        pair_order = np.argsort(lower_bounds) if topk is not None else range(len(geom_pairs))

        geom_distance_infos = []
        kth_best_dist = float(distmax)
        fromto = np.zeros(6,dtype=np.float64)
        for pair_idx in pair_order: # iterate through all candidate pairs in ascending order of lower bound
            if topk is not None and len(geom_distance_infos) >= topk and lower_bounds[pair_idx] > kth_best_dist:
                break

            g1 = int(geom_pairs[pair_idx,0])
            g2 = int(geom_pairs[pair_idx,1])
            lower_bound = float(lower_bounds[pair_idx])

            fromto[:] = 0.0
            query_distmax = kth_best_dist if topk is not None else float(distmax)
            dist = float(mujoco.mj_geomDistance(self.model,self.data,g1,g2,query_distmax,fromto)) # compute distance for this pair

            if sanity_check:
                if dist < lower_bound - lower_bound_tol:
                    continue
                if not np.all(np.isfinite(fromto)):
                    continue
                fromto_dist = np.linalg.norm(fromto[3:]-fromto[:3])
                if dist <= 1e-12 and fromto_dist <= 1e-12:
                    continue
                if abs(fromto_dist - dist) > fromto_dist_tol:
                    continue

            b1 = int(self.model.geom_bodyid[g1])
            b2 = int(self.model.geom_bodyid[g2])
            geom_distance_infos.append({
                "dist": dist,
                "fromto": fromto.copy(),
                "p1": fromto[:3].copy(),
                "p2": fromto[3:].copy(),
                "point": 0.5*(fromto[:3]+fromto[3:]),
                "geom1_idx": g1,
                "geom2_idx": g2,
                "geom1_name": self.geom_names[g1],
                "geom2_name": self.geom_names[g2],
                "body1_idx": b1,
                "body2_idx": b2,
                "body1_name": self.body_names[b1],
                "body2_name": self.body_names[b2],
            })

            if topk is not None:
                geom_distance_infos = sorted(geom_distance_infos,key=lambda info: info["dist"])[:topk]
                if len(geom_distance_infos) >= topk:
                    kth_best_dist = min(float(distmax),geom_distance_infos[-1]["dist"])

        if topk is None and sort:
            geom_distance_infos = sorted(geom_distance_infos,key=lambda info: info["dist"])
        elif topk is not None:
            geom_distance_infos = sorted(geom_distance_infos,key=lambda info: info["dist"])

        return geom_distance_infos

    def plot_geom_pair_distance_infos(
            self,
            geom_distance_infos,
            topk              = None,
            rgba              = (0,0,1,1),
            threshold_rgba    = (1,0,0,1),
            distance_threshold = 0.0,
            sphere_r          = 0.005,
            plot_sphere       = True,
            plot_line         = True,
            plot_text_overlay = True,
            overlay_loc       = "bottom left",
            overlay_title     = "geom distance",
            distance_unit     = "mm",
            dist_format       = "%.3f",
        ):
        """
        Plot closest-point spheres, connecting lines, and optional text overlay.

        Parameters:
            geom_distance_infos (Any): Input geom distance infos.
            topk (Any | None): Input topk. Defaults to `None`.
            rgba (tuple): Color for pairs outside the threshold. Defaults to
                `(0, 0, 1, 1)`.
            threshold_rgba (tuple): Color for pairs inside the threshold.
                Defaults to `(1, 0, 0, 1)`.
            distance_threshold (float | None): Distance threshold in meters.
                Pairs with `dist <= distance_threshold` use `threshold_rgba`.
                If `None`, all pairs use `rgba`. Defaults to `0.0`.
            sphere_r (float): Input sphere r. Defaults to `0.005`.
            plot_sphere (bool): Input plot sphere. Defaults to `True`.
            plot_line (bool): Input plot line. Defaults to `True`.
            plot_text_overlay (bool): Input plot text overlay. Defaults to `True`.
            overlay_loc (str): Input overlay loc. Defaults to `'bottom left'`.
            overlay_title (str): Input overlay title. Defaults to `'geom distance'`.
            distance_unit (str): Input distance unit. Defaults to `'mm'`.
            dist_format (str): Input dist format. Defaults to `'%.3f'`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        distance_unit_scale_dict = {"m":1.0,"cm":100.0,"mm":1000.0}
        if distance_unit not in distance_unit_scale_dict:
            raise ValueError("Unsupported distance_unit:[%s]. Use 'mm', 'cm', or 'm'."%(distance_unit))
        distance_scale = distance_unit_scale_dict[distance_unit]

        infos = geom_distance_infos if topk is None else geom_distance_infos[:max(0,int(topk))]
        for info in infos:
            info_rgba = rgba
            if distance_threshold is not None and float(info["dist"]) <= float(distance_threshold):
                info_rgba = threshold_rgba
            if plot_sphere:
                self.plot_sphere(info["p1"],r=sphere_r,rgba=info_rgba)
                self.plot_sphere(info["p2"],r=sphere_r,rgba=info_rgba)
            if plot_line:
                self.plot_line_fr2to(info["p1"],info["p2"],rgba=info_rgba)

        if plot_text_overlay:
            for idx,info in enumerate(infos):
                self.viewer_text_overlay(
                    "%s [%d] dist:[%s%s]"%(
                        overlay_title,idx,dist_format%(distance_scale*info["dist"]),distance_unit
                    ),
                    "(%s-%s)"%(info["body1_name"],info["body2_name"]),
                    loc = overlay_loc,
                )

    def get_geom_pair_signed_distance_infos(
            self,
            geom_pairs,
            distmax            = np.inf,
            topk               = None,
            forward            = False,
            sort               = True,
            contact_info       = None,
            compute_force      = False,
            sanity_check       = True,
            lower_bound_tol    = 1e-6,
            fromto_dist_tol    = 1e-4,
        ):
        """
        Return signed geom-pair distances using active contacts when available.

        Parameters:
            geom_pairs (Any): Input geom pairs.
            distmax (Any): Input distmax. Defaults to `np.inf`.
            topk (Any | None): Input topk. Defaults to `None`.
            forward (bool): Input forward. Defaults to `False`.
            sort (bool): Input sort. Defaults to `True`.
            contact_info (Any | None): Input contact info. Defaults to `None`.
            compute_force (bool): Input compute force. Defaults to `False`.
            sanity_check (bool): Input sanity check. Defaults to `True`.
            lower_bound_tol (float): Input lower bound tol. Defaults to `1e-06`.
            fromto_dist_tol (float): Input fromto dist tol. Defaults to `0.0001`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if forward:
            self.forward()

        geom_pairs = np.asarray(geom_pairs,dtype=np.int32)
        if geom_pairs.size == 0:
            return []
        geom_pairs = geom_pairs.reshape(-1,2)

        if topk is not None:
            topk = max(0,int(topk))
            if topk == 0:
                return []

        geom_pair_key_set = set()
        for g1,g2 in geom_pairs:
            geom_pair_key_set.add((min(int(g1),int(g2)),max(int(g1),int(g2))))

        signed_distance_infos = []
        contact_geom_pair_keys = set()

        if contact_info is None:
            contact_info = self.get_contact_info(
                forward       = False,
                compute_force = compute_force,
            )

        contact_distance_infos = self.get_closest_contact_info(
            contact_info  = contact_info,
            forward       = False,
            compute_force = compute_force,
            topk          = contact_info["n_contact"],
        )
        for contact_distance_info in contact_distance_infos:
            g1 = int(contact_distance_info["geom1_idx"])
            g2 = int(contact_distance_info["geom2_idx"])
            key = (min(g1,g2),max(g1,g2))
            if key not in geom_pair_key_set:
                continue
            if key in contact_geom_pair_keys:
                continue
            signed_distance_info = dict(contact_distance_info)
            signed_distance_info["signed_dist"] = float(signed_distance_info["dist"])
            signed_distance_info["source"] = "contact"
            signed_distance_info["is_contact"] = True
            signed_distance_infos.append(signed_distance_info)
            contact_geom_pair_keys.add(key)

        geom1_idxs = geom_pairs[:,0]
        geom2_idxs = geom_pairs[:,1]
        lower_bounds = (
            np.linalg.norm(self.data.geom_xpos[geom2_idxs]-self.data.geom_xpos[geom1_idxs],axis=1)
            - self.model.geom_rbound[geom1_idxs]
            - self.model.geom_rbound[geom2_idxs]
        )
        pair_order = np.argsort(lower_bounds) if topk is not None else range(len(geom_pairs))

        kth_best_dist = float(distmax)
        if topk is not None and len(signed_distance_infos) >= topk:
            signed_distance_infos = sorted(signed_distance_infos,key=lambda info: info["dist"])[:topk]
            kth_best_dist = min(float(distmax),signed_distance_infos[-1]["dist"])

        fromto = np.zeros(6,dtype=np.float64)
        for pair_idx in pair_order: # iterate through all candidate pairs in ascending order of lower bound
            if topk is not None and len(signed_distance_infos) >= topk and lower_bounds[pair_idx] > kth_best_dist:
                break

            g1 = int(geom_pairs[pair_idx,0])
            g2 = int(geom_pairs[pair_idx,1])
            key = (min(g1,g2),max(g1,g2))
            if key in contact_geom_pair_keys:
                continue
            lower_bound = float(lower_bounds[pair_idx])

            fromto[:] = 0.0
            query_distmax = kth_best_dist if topk is not None else float(distmax)
            dist = float(mujoco.mj_geomDistance(self.model,self.data,g1,g2,query_distmax,fromto))

            if sanity_check:
                if dist < lower_bound - lower_bound_tol:
                    continue
                if not np.all(np.isfinite(fromto)):
                    continue
                if abs(np.linalg.norm(fromto[3:]-fromto[:3]) - dist) > fromto_dist_tol:
                    continue

            b1 = int(self.model.geom_bodyid[g1])
            b2 = int(self.model.geom_bodyid[g2])
            signed_distance_infos.append({
                "dist": dist,
                "signed_dist": dist,
                "source": "geom_distance",
                "is_contact": False,
                "fromto": fromto.copy(),
                "p1": fromto[:3].copy(),
                "p2": fromto[3:].copy(),
                "point": 0.5*(fromto[:3]+fromto[3:]),
                "geom1_idx": g1,
                "geom2_idx": g2,
                "geom1_name": self.geom_names[g1],
                "geom2_name": self.geom_names[g2],
                "body1_idx": b1,
                "body2_idx": b2,
                "body1_name": self.body_names[b1],
                "body2_name": self.body_names[b2],
            })

            if topk is not None:
                signed_distance_infos = sorted(signed_distance_infos,key=lambda info: info["dist"])[:topk]
                if len(signed_distance_infos) >= topk:
                    kth_best_dist = min(float(distmax),signed_distance_infos[-1]["dist"])

        if topk is None and sort:
            signed_distance_infos = sorted(signed_distance_infos,key=lambda info: info["dist"])
        elif topk is not None:
            signed_distance_infos = sorted(signed_distance_infos,key=lambda info: info["dist"])

        return signed_distance_infos

    def get_geom_pair_signed_distance_infos_bind(
            self,
            geom_pairs,
            distmax            = np.inf,
            topk               = None,
            forward            = False,
            sort               = True,
            sanity_check       = True,
            lower_bound_tol    = 1e-6,
            fromto_dist_tol    = 1e-4,
            return_min_signed_dist = False,
            *,
            be,
        ):
        """
        Return signed geom-pair distances using a compiled backend.

        Parameters:
            geom_pairs (Any): Input geom pairs.
            distmax (Any): Input distmax. Defaults to `np.inf`.
            topk (Any | None): Input topk. Defaults to `None`.
            forward (bool): Input forward. Defaults to `False`.
            sort (bool): Input sort. Defaults to `True`.
            sanity_check (bool): Input sanity check. Defaults to `True`.
            lower_bound_tol (float): Input lower bound tol. Defaults to `1e-06`.
            fromto_dist_tol (float): Input fromto dist tol. Defaults to `0.0001`.
            return_min_signed_dist (bool): If True, also return the minimum
                signed distance from the returned records. Defaults to `False`.
            be (module): Compiled backend from get_native_backend(...).

        Returns:
            list[dict] | tuple[list[dict], float]: Signed-distance records, and
                optionally the minimum signed distance.
        """
        if be is None:
            warnings.warn(
                (
                    "signed-distance bind backend is unavailable; falling back "
                    "to Python get_geom_pair_signed_distance_infos(...)."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            signed_distance_infos = self.get_geom_pair_signed_distance_infos(
                geom_pairs,
                distmax         = distmax,
                topk            = topk,
                forward         = forward,
                sort            = sort,
                sanity_check    = sanity_check,
                lower_bound_tol = lower_bound_tol,
                fromto_dist_tol = fromto_dist_tol,
            )
            if return_min_signed_dist:
                min_signed_dist = min(
                    [info["signed_dist"] for info in signed_distance_infos],
                    default = np.nan,
                )
                return signed_distance_infos,min_signed_dist
            return signed_distance_infos

        from ri_motion_v5_package.mujoco_sim._native import _signed_distance_infos_from_backend
        signed_distance_infos = _signed_distance_infos_from_backend(
            be,
            self,
            geom_pairs,
            distmax         = distmax,
            topk            = topk,
            forward         = forward,
            sort            = sort,
            sanity_check    = sanity_check,
            lower_bound_tol = lower_bound_tol,
            fromto_dist_tol = fromto_dist_tol,
        )
        if return_min_signed_dist:
            min_signed_dist = min(
                [info["signed_dist"] for info in signed_distance_infos],
                default = np.nan,
            )
            return signed_distance_infos,min_signed_dist
        return signed_distance_infos

    def get_dep_trgts_from_signed_dists(
            self,
            geom_signed_distance_infos,
            root_body_name = None,
            body_idx_set   = None,
            safe_margin    = 0.0,
            dep_step       = 0.01,
        ):
        """
        Convert signed-distance violations into body-point depenetration targets.

        This reads signed-distance summaries from
        get_geom_pair_signed_distance_infos(...) and creates body-position
        targets only for pairs that violate safe_margin and involve the selected
        body set.

        Parameters:
            geom_signed_distance_infos (list[dict]): Signed-distance summaries.
            root_body_name (str | None): Root body whose subtree is movable.
                Used only when body_idx_set is None.
            body_idx_set (set[int] | None): Explicit movable body indices.
            safe_margin (float): Minimum allowed signed distance.
            dep_step (float): Fixed target displacement along the push direction.

        Returns:
            tuple[list[dict], dict]: Depenetration targets and compact summary
                information with n_violation and n_dep_trgt.
        """
        if body_idx_set is None:
            if root_body_name is None:
                raise ValueError("Either root_body_name or body_idx_set must be provided.")
            body_idx_set = set(self.get_body_idxs_from_root_body_name(root_body_name))
        else:
            body_idx_set = set(int(body_idx) for body_idx in body_idx_set)

        dep_trgts = []
        n_violation = 0
        for info in geom_signed_distance_infos:
            signed_dist = float(info["signed_dist"])
            if signed_dist >= safe_margin:
                continue

            body1_idx = int(info["body1_idx"])
            body2_idx = int(info["body2_idx"])
            body1_is_target = body1_idx in body_idx_set
            body2_is_target = body2_idx in body_idx_set
            if (not body1_is_target) and (not body2_is_target):
                continue

            if info.get("is_contact",False) and ("normal" in info):
                push_dir = np.array(info["normal"],dtype=np.float64)
                p_contact = info["p_contact"] if "p_contact" in info else info["point"]
                p1_curr = np.array(p_contact,dtype=np.float64)
                p2_curr = p1_curr.copy()
            else:
                p1_curr = np.array(info["p1"],dtype=np.float64)
                p2_curr = np.array(info["p2"],dtype=np.float64)
                push_dir = p2_curr - p1_curr

            push_norm = np.linalg.norm(push_dir)
            if push_norm <= 1e-9:
                continue
            push_dir = push_dir / push_norm

            n_violation = n_violation + 1
            violation = safe_margin - signed_dist
            if body1_is_target:
                dep_trgts.append({
                    "body_idx": body1_idx,
                    "p_curr": p1_curr,
                    "p_trgt": p1_curr - dep_step*push_dir,
                    "signed_dist": signed_dist,
                    "violation": violation,
                    "source": info.get("source",None),
                })
            if body2_is_target:
                dep_trgts.append({
                    "body_idx": body2_idx,
                    "p_curr": p2_curr,
                    "p_trgt": p2_curr + dep_step*push_dir,
                    "signed_dist": signed_dist,
                    "violation": violation,
                    "source": info.get("source",None),
                })

        dep_info = {
            "n_violation": n_violation,
            "n_dep_trgt": len(dep_trgts),
        }
        return dep_trgts,dep_info

    def plot_geom_pair_signed_distance_infos(
            self,
            geom_signed_distance_infos,
            topk              = None,
            rgba              = (0,0,1,1),
            contact_rgba      = (1,0,0,1),
            distance_threshold = 0.0,
            sphere_r          = 0.005,
            plot_sphere       = True,
            plot_line         = True,
            plot_text_overlay = True,
            overlay_loc       = "bottom left",
            overlay_title     = "signed distance",
            distance_unit     = "mm",
            dist_format       = "%.3f",
        ):
        """
        Plot signed-distance infos with threshold-based red/blue coloring.

        Parameters:
            geom_signed_distance_infos (Any): Input geom signed distance infos.
            topk (Any | None): Input topk. Defaults to `None`.
            rgba (tuple): Color for pairs outside the threshold. Defaults to
                `(0, 0, 1, 1)`.
            contact_rgba (tuple): Color for pairs inside the threshold.
                Defaults to `(1, 0, 0, 1)`.
            distance_threshold (float | None): Signed-distance threshold in
                meters. Pairs with `signed_dist <= distance_threshold` use
                `contact_rgba`. If `None`, active contacts use `contact_rgba`
                and non-contact pairs use `rgba`. Defaults to `0.0`.
            sphere_r (float): Input sphere r. Defaults to `0.005`.
            plot_sphere (bool): Input plot sphere. Defaults to `True`.
            plot_line (bool): Input plot line. Defaults to `True`.
            plot_text_overlay (bool): Input plot text overlay. Defaults to `True`.
            overlay_loc (str): Input overlay loc. Defaults to `'bottom left'`.
            overlay_title (str): Input overlay title. Defaults to `'signed distance'`.
            distance_unit (str): Input distance unit. Defaults to `'mm'`.
            dist_format (str): Input dist format. Defaults to `'%.3f'`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        distance_unit_scale_dict = {"m":1.0,"cm":100.0,"mm":1000.0}
        if distance_unit not in distance_unit_scale_dict:
            raise ValueError("Unsupported distance_unit:[%s]. Use 'mm', 'cm', or 'm'."%(distance_unit))
        distance_scale = distance_unit_scale_dict[distance_unit]

        infos = geom_signed_distance_infos if topk is None else geom_signed_distance_infos[:max(0,int(topk))]
        for info in infos:
            if distance_threshold is None:
                info_rgba = contact_rgba if info.get("is_contact",False) else rgba
            else:
                info_rgba = contact_rgba if float(info["signed_dist"]) <= float(distance_threshold) else rgba
            if plot_sphere:
                self.plot_sphere(info["p1"],r=sphere_r,rgba=info_rgba)
                self.plot_sphere(info["p2"],r=sphere_r,rgba=info_rgba)
            if plot_line:
                self.plot_line_fr2to(info["p1"],info["p2"],rgba=info_rgba)

        if plot_text_overlay:
            for idx,info in enumerate(infos):
                source = "cnt" if info.get("is_contact",False) else "dist"
                self.viewer_text_overlay(
                    "%s [%d] %s:[%s%s]"%(
                        overlay_title,idx,source,
                        dist_format%(distance_scale*info["signed_dist"]),distance_unit
                    ),
                    "(%s-%s)"%(info["body1_name"],info["body2_name"]),
                    loc = overlay_loc,
                )

    def get_closest_geom_pair_distance_info(
            self,
            geom_idxs                 = None,
            geom_names                = None,
            body_names                = None,
            distmax                   = np.inf,
            forward                   = True,
            respect_contype_affinity  = True,
            skip_visual_geoms         = True,
            skip_same_body            = True,
            skip_parent_child         = True,
            respect_body_excludes     = True,
        ):
        """
        Brute-force closest signed distance over candidate geom pairs.

        Parameters:
            geom_idxs (Any | None): Input geom idxs. Defaults to `None`.
            geom_names (Any | None): Input geom names. Defaults to `None`.
            body_names (Any | None): Input body names. Defaults to `None`.
            distmax (Any): Input distmax. Defaults to `np.inf`.
            forward (bool): Input forward. Defaults to `True`.
            respect_contype_affinity (bool): Input respect contype affinity. Defaults
                                             to `True`.
            skip_visual_geoms (bool): Input skip visual geoms. Defaults to `True`.
            skip_same_body (bool): Input skip same body. Defaults to `True`.
            skip_parent_child (bool): Input skip parent child. Defaults to `True`.
            respect_body_excludes (bool): Input respect body excludes. Defaults to
                                          `True`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if forward:
            self.forward()

        if geom_idxs is None:
            if geom_names is not None:
                geom_idxs = [self.geom_name_to_id[name] for name in geom_names]
            elif body_names is not None:
                if isinstance(body_names,str):
                    body_names = [body_names]
                body_name_set = set(body_names)
                geom_idxs = [
                    geom_idx for geom_idx in range(self.model.ngeom)
                    if self.body_names[int(self.model.geom_bodyid[geom_idx])] in body_name_set
                ]
            else:
                geom_idxs = list(range(self.model.ngeom))
        else:
            geom_idxs = list(geom_idxs)

        if len(geom_idxs) < 2:
            return None

        exclude_signatures = set()
        if respect_body_excludes and hasattr(self.model,"exclude_signature"):
            exclude_signatures = set(int(sig) for sig in self.model.exclude_signature)

        best = None
        best_dist = float(distmax)
        fromto = np.zeros(6,dtype=np.float64)
        for idx1 in range(len(geom_idxs)-1):
            geom1_idx = int(geom_idxs[idx1])
            body1_idx = int(self.model.geom_bodyid[geom1_idx])
            for idx2 in range(idx1+1,len(geom_idxs)):
                geom2_idx = int(geom_idxs[idx2])
                body2_idx = int(self.model.geom_bodyid[geom2_idx])

                if skip_same_body and body1_idx == body2_idx:
                    continue
                if skip_parent_child:
                    parent1 = int(self.model.body_parentid[body1_idx])
                    parent2 = int(self.model.body_parentid[body2_idx])
                    if (parent1 == body2_idx and body2_idx != 0) or (parent2 == body1_idx and body1_idx != 0):
                        continue
                if skip_visual_geoms:
                    visual1 = (
                        int(self.model.geom_contype[geom1_idx]) == 0 and
                        int(self.model.geom_conaffinity[geom1_idx]) == 0
                    )
                    visual2 = (
                        int(self.model.geom_contype[geom2_idx]) == 0 and
                        int(self.model.geom_conaffinity[geom2_idx]) == 0
                    )
                    if visual1 or visual2:
                        continue
                if respect_contype_affinity:
                    can_collide = (
                        (int(self.model.geom_contype[geom1_idx]) & int(self.model.geom_conaffinity[geom2_idx])) or
                        (int(self.model.geom_contype[geom2_idx]) & int(self.model.geom_conaffinity[geom1_idx]))
                    )
                    if not can_collide:
                        continue
                if respect_body_excludes:
                    sig12 = ((body1_idx+1) << 16) + (body2_idx+1)
                    sig21 = ((body2_idx+1) << 16) + (body1_idx+1)
                    if sig12 in exclude_signatures or sig21 in exclude_signatures:
                        continue

                lower_bound = (
                    np.linalg.norm(self.data.geom_xpos[geom2_idx]-self.data.geom_xpos[geom1_idx])
                    - float(self.model.geom_rbound[geom1_idx])
                    - float(self.model.geom_rbound[geom2_idx])
                )
                if lower_bound > best_dist:
                    continue

                fromto[:] = 0.0
                dist = float(mujoco.mj_geomDistance(
                    self.model,self.data,geom1_idx,geom2_idx,best_dist,fromto))
                if dist < best_dist:
                    best_dist = dist
                    best = self.get_geom_distance_info(
                        geom1_idx = geom1_idx,
                        geom2_idx = geom2_idx,
                        distmax   = distmax,
                        forward   = False,
                    )

        return best

    def print_contact_info(self,contact_info=None,simple=False):
        """
        Print detailed contact information to the console.

        Parameters:
            contact_info (Any | None): Input contact info. Defaults to `None`.
            simple (bool): Input simple. Defaults to `False`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """

        # Get contact information
        if contact_info is None:
            contact_info = self.get_contact_info()

        n = contact_info['n_contact']
        if not simple: print("----------------------------------------------")
        print("n_contact = [%d]" % n)
        print("min_contact_dist = [%.6f]" % contact_info['min_contact_dist'])
        if not simple: print("----------------------------------------------")

        for i in range(n):
            contact = contact_info['contact_list'][i]
            contact_dist = contact.dist

            g1 = contact_info['geom1_idx_list'][i]
            g2 = contact_info['geom2_idx_list'][i]
            b1 = contact_info['body1_idx_list'][i]
            b2 = contact_info['body2_idx_list'][i]

            # body names
            b1_name = self.body_names[b1]
            b2_name = self.body_names[b2]

            p   = contact_info['p_contact_list'][i]
            f   = contact_info['f_world_list'][i]
            nrm = contact_info['norm_dir_list'][i]
            R   = contact_info['R_frame_list'][i]

            # Contact header with body names
            print(" Contact [%d]: geom([%d],[%d]), body([%d]=[%s],[%d]=[%s]), dist=[%.6f]" %
                (i, g1, g2, b1, b1_name, b2, b2_name, contact_dist))

            if not simple:
                print("   p_contact = [%.5f %.5f %.5f]" %
                    (p[0], p[1], p[2]))

                print("   f_world   = [%.5f %.5f %.5f]" %
                    (f[0], f[1], f[2]))

                print("   normal    = [%.5f %.5f %.5f]" %
                    (nrm[0], nrm[1], nrm[2]))

                print("   R_frame:")
                print("      [%.5f %.5f %.5f]" % (R[0,0], R[0,1], R[0,2]))
                print("      [%.5f %.5f %.5f]" % (R[1,0], R[1,1], R[1,2]))
                print("      [%.5f %.5f %.5f]" % (R[2,0], R[2,1], R[2,2]))

                print("----------------------------------------------")

    def plot_contact_info(
            self,
            contact_info = None,
            plot_arrow   = True,
            cmap_arrow   = None, # "rainbow"
            arrow_len    = 0.05,
            arrow_r      = 0.005,
            arrow_rgba1  = (0.9,0.3,0,1),
            arrow_rgba2  = (0.9,0,0.3,1),
            arrow_alpha  = None,
            plot_contact = True,
            contact_rgba = (1,0,0,0.5),
            contact_r    = 0.02,
            contact_h    = 0.01,
            rate         = 1.0,
            max_contact_plot = None,
            forward      = False,
        ):
        """
        Visualize contact information in the viewer.

        Parameters:
            contact_info (Any | None): Input contact info. Defaults to `None`.
            plot_arrow (bool): Input plot arrow. Defaults to `True`.
            cmap_arrow (Any | None): Input cmap arrow. Defaults to `None`.
            arrow_len (float): Input arrow len. Defaults to `0.05`.
            arrow_r (float): Input arrow r. Defaults to `0.005`.
            arrow_rgba1 (tuple): Input arrow rgba1. Defaults to `(0.9, 0.3, 0, 1)`.
            arrow_rgba2 (tuple): Input arrow rgba2. Defaults to `(0.9, 0, 0.3, 1)`.
            arrow_alpha (Any | None): Input arrow alpha. Defaults to `None`.
            plot_contact (bool): Input plot contact. Defaults to `True`.
            contact_rgba (tuple): Input contact rgba. Defaults to `(1, 0, 0, 0.5)`.
            contact_r (float): Input contact r. Defaults to `0.02`.
            contact_h (float): Input contact h. Defaults to `0.01`.
            rate (float): Input rate. Defaults to `1.0`.
            max_contact_plot (Any | None): Input max contact plot. Defaults to `None`.
            forward (bool): Input forward. Defaults to `False`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        # Default arrow alpha
        if arrow_alpha is None:
            arrow_alpha = 0.8

        # Get contact information
        if contact_info is None:
            contact_info = self.get_contact_info(forward=forward)

        n_contact      = contact_info["n_contact"]
        n_contact_plot = n_contact
        if max_contact_plot is not None:
            n_contact_plot = min(n_contact,max(0,int(max_contact_plot)))
        p_contact_list = contact_info["p_contact_list"]
        geom1_idx_list = contact_info["geom1_idx_list"]
        geom2_idx_list = contact_info["geom2_idx_list"]
        body1_idx_list = contact_info["body1_idx_list"]
        body2_idx_list = contact_info["body2_idx_list"]
        norm_dir_list  = contact_info["norm_dir_list"]
        R_frame_list   = contact_info["R_frame_list"]

        if cmap_arrow is not None:
            colors_arrow = get_colors(
                n_color   = n_contact_plot*2,
                cmap_name = cmap_arrow,
                alpha     = arrow_alpha,
            )

        for c_idx in range(n_contact_plot):
            p_contact = p_contact_list[c_idx]
            norm_dir  = norm_dir_list[c_idx] # normal direction (3,)
            R_frame   = R_frame_list[c_idx]

            # Plot contact point
            if plot_contact:
                frame_norm = R_frame[0,:]
                norm_sq    = frame_norm[0]*frame_norm[0] + frame_norm[1]*frame_norm[1] + frame_norm[2]*frame_norm[2]
                if norm_sq > 0.0:
                    inv_norm   = 1.0 / np.sqrt(norm_sq)
                    frame_norm = frame_norm * inv_norm
                    if (frame_norm[0]*norm_dir[0] + frame_norm[1]*norm_dir[1] + frame_norm[2]*norm_dir[2]) < 0.0:
                        frame_norm = -frame_norm
                    norm_dir = frame_norm

                self.plot_cylinder_fr2to(
                    p_fr = p_contact - 0.5*contact_h*norm_dir*rate,
                    p_to = p_contact + 0.5*contact_h*norm_dir*rate,
                    r    = contact_r*rate,
                    rgba = contact_rgba,
                )

            # Plot contacting arrows
            geom1_idx,geom2_idx = geom1_idx_list[c_idx],geom2_idx_list[c_idx]
            body1_idx,body2_idx = body1_idx_list[c_idx],body2_idx_list[c_idx]
            body1_name,body2_name = self.body_names[body1_idx],self.body_names[body2_idx]

            if cmap_arrow is not None:
                rgba_arrow1 = colors_arrow[2*c_idx]
                rgba_arrow2 = colors_arrow[2*c_idx+1]
                rgba_arrow1 = (rgba_arrow1[0],rgba_arrow1[1],rgba_arrow1[2],arrow_alpha)
                rgba_arrow2 = (rgba_arrow2[0],rgba_arrow2[1],rgba_arrow2[2],arrow_alpha)
            else:
                rgba_arrow1 = (arrow_rgba1[0],arrow_rgba1[1],arrow_rgba1[2],arrow_alpha)
                rgba_arrow2 = (arrow_rgba2[0],arrow_rgba2[1],arrow_rgba2[2],arrow_alpha)

            p1_to = p_contact - arrow_len*norm_dir*rate
            p2_to = p_contact + arrow_len*norm_dir*rate

            if plot_arrow:
                self.plot_arrow_fr2to(p_contact,p1_to,arrow_r*rate,rgba_arrow1)
                self.plot_arrow_fr2to(p_contact,p2_to,arrow_r*rate,rgba_arrow2)

    def plot_links_between_bodies(
            self,
            root_body_name = 'world',
            pose_env       = None,
            r              = 0.002,
            rgba           = (0.0,0.0,0.0,0.5),
            plot_sphere    = False,
            r_sphere       = 0.01,
            rgba_sphere    = (0.0,0.0,0.0,0.5),
            eps            = 1e-9,
            exclude_list   = None,
        ):
        """
        Plot cylinders connecting parent-child bodies within the subtree rooted at
        root_body_name.

        Parameters:
            root_body_name (str): Input root body name. Defaults to `'world'`.
            pose_env (Any | None): Input pose env. Defaults to `None`.
            r (float): Input r. Defaults to `0.002`.
            rgba (tuple): Input rgba. Defaults to `(0.0, 0.0, 0.0, 0.5)`.
            plot_sphere (bool): Input plot sphere. Defaults to `False`.
            r_sphere (float): Input r sphere. Defaults to `0.01`.
            rgba_sphere (tuple): Input rgba sphere. Defaults to `(0.0, 0.0, 0.0,
                                 0.5)`.
            eps (float): Input eps. Defaults to `1e-09`.
            exclude_list (Any | None): Input exclude list. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        pose_env = self if pose_env is None else pose_env

        # Get subtree body names (exclude root by default)
        subtree_body_names=self.get_subtree_body_names(root_body_name=root_body_name,include_root=False)

        # Draw links: for each body in subtree, connect it to its parent (if parent exists)
        for body_name in subtree_body_names:
            if body_name is None: continue
            body_idx         = self.body_name_to_id[body_name]
            parent_body_name = self.parent_body_names[body_idx]

            # Skip world parent
            if parent_body_name is None:
                continue

            # Skip excluded bodies
            if exclude_list is not None and body_name in exclude_list:
                continue

            p_parent = pose_env.get_p(parent_body_name,'body')
            p_child  = pose_env.get_p(body_name,'body')

            if np.linalg.norm(p_child-p_parent)<eps:
                continue

            self.plot_cylinder_fr2to(p_fr=p_parent,p_to=p_child,r=r,rgba=rgba)

            if plot_sphere:
                self.plot_sphere(p=p_parent,r=r_sphere,rgba=rgba_sphere)

    def plot_rev_joint_axis(
            self,
            rev_joint_names = None,
            axis_len        = 0.1,
            axis_r          = 0.01,
            alpha           = 0.2,
            rate            = 1.0,
            print_name      = False,
            axis_colors     = None,
        ):
        """
        Plot the axis of revolute (hinge) joints as arrows in world coordinates.

        Parameters:
            rev_joint_names (Any | None): Input rev joint names. Defaults to `None`.
            axis_len (float): Input axis len. Defaults to `0.1`.
            axis_r (float): Input axis r. Defaults to `0.01`.
            alpha (float): Input alpha. Defaults to `0.2`.
            rate (float): Input rate. Defaults to `1.0`.
            print_name (bool): Input print name. Defaults to `False`.
            axis_colors (Any | None): Input axis colors. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if rev_joint_names is None:
            rev_joint_names=self.rev_joint_names

        idxs=get_idxs(self.rev_joint_names,rev_joint_names)
        rev_joint_idxs=self.rev_joint_idxs[idxs]

        for i,(jidx,jname) in enumerate(zip(rev_joint_idxs,rev_joint_names)):
            axis_joint=self.model.jnt_axis[jidx]

            body_id=int(self.model.joint(jname).bodyid[0])
            body_name=self.body_names[body_id]

            p_joint=self.get_p(body_name,'body')
            R_joint=self.get_R(body_name,'body')
            axis_world=R_joint@axis_joint

            if axis_colors is None:
                axis_rgba=np.append(np.eye(3)[:,int(np.argmax(np.abs(axis_joint)))],alpha)
            else:
                axis_rgba=axis_colors[i] if i<len(axis_colors) else axis_colors[-1]

            self.plot_arrow_fr2to(
                p_fr=p_joint,
                p_to=p_joint+rate*axis_len*axis_world,
                r=rate*axis_r,
                rgba=axis_rgba,
            )

            if print_name:
                self.viewer.add_marker(
                    pos=p_joint,
                    size=[0.0001,0.0001,0.0001],
                    rgba=[1,1,1,0.01],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    label=jname,
                )

    def plot_time(
            self,
            loc = 'bottom left',
            plot_sim_time  = True,
            plot_wall_time = True,
            plot_loop_freq = False,
            plot_time_diff = None,
        ):
        """
        Overlay the current simulation tick, simulation time, and wall-clock time on
        the viewer.

        Parameters:
            loc (str): Input loc. Defaults to `'bottom left'`.
            plot_sim_time (bool): Input plot sim time. Defaults to `True`.
            plot_wall_time (bool): Input plot wall time. Defaults to `True`.
            plot_loop_freq (bool): Input plot loop freq. Defaults to `False`.
            plot_time_diff (Any | None): Input plot time diff. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if plot_time_diff is None:
            plot_time_diff = plot_sim_time

        if plot_sim_time:
            self.viewer_text_overlay(text1='sim time',text2='[%.2f]sec'%(self.get_sim_time()),loc=loc)
        if plot_wall_time:
            self.viewer_text_overlay(text1='wall time',text2='[%.2f]sec'%(self.get_wall_time()),loc=loc)
        if plot_loop_freq:
            self.viewer_text_overlay(text1="Loop freq",text2="[%.1f]Hz"%(self.loop_hz),loc=loc)
            self.viewer_text_overlay(text1="Loop freq (ema)",text2="[%.1f]Hz"%(self.loop_hz_ema),loc=loc)
        if plot_time_diff:
            time_diff = self.get_wall_time() - self.get_sim_time()
            self.viewer_text_overlay(text1="Time diff",text2="[%.2f]sec"%(time_diff),loc=loc)

    def plot_tmr_lag(
            self,
            tmr,
            loc = 'bottom left',
        ):
        """
        Overlay how far a SimpleTimer is behind its ideal fixed-rate schedule.

        Parameters:
            tmr (Any): Input tmr.
            loc (str): Input loc. Defaults to `'bottom left'`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.plot_tmr_info(tmr=tmr,loc=loc,plot_period=False)

    def plot_tmr_info(
            self,
            tmr,
            loc = 'bottom left',
            plot_period = True,
        ):
        """
        Overlay SimpleTimer schedule lag, recent run time, and optionally target
        period.

        Parameters:
            tmr (Any): Input tmr.
            loc (str): Input loc. Defaults to `'bottom left'`.
            plot_period (bool): Input plot period. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        # Keep timer formatting in the parser so notebooks only pass the timer they already own.
        lag_info = tmr.get_lag_info()
        run_ms = tmr.get_ems() if hasattr(tmr,"get_ems") else 0.0
        text2 = "lag:[%d]tick [%.1f]ms run:[%.2f]ms"%(lag_info["tick"],lag_info["ms"],run_ms)
        if plot_period:
            if hasattr(tmr,"sec_period"):
                period_ms = 1000.0*float(tmr.sec_period)
            elif hasattr(tmr,"dt"):
                period_ms = 1000.0*float(tmr.dt)
            elif hasattr(tmr,"Hz"):
                period_ms = 1000.0/float(tmr.Hz)
            else:
                period_ms = 0.0
            text2 = "%s period:[%.1f]ms"%(text2,period_ms)
        self.viewer_text_overlay(
            text1 = "%s timer"%(tmr.name),
            text2 = text2,
            loc   = loc,
        )

    def plot_tmrs_info(
            self,
            tmrs,
            loc = 'bottom left',
        ):
        """
        Overlay summary information for multiple SimpleTimer instances.

        Parameters:
            tmrs (Any): Input tmrs.
            loc (str): Input loc. Defaults to `'bottom left'`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        tmrs = list(tmrs)
        if len(tmrs) == 0:
            return

        run_ms_list = []
        period_ms_list = []
        for tmr in tmrs:
            run_ms = tmr.get_ems() if hasattr(tmr,"get_ems") else 0.0
            if hasattr(tmr,"sec_period"):
                period_ms = 1000.0*float(tmr.sec_period)
            elif hasattr(tmr,"dt"):
                period_ms = 1000.0*float(tmr.dt)
            elif hasattr(tmr,"Hz"):
                period_ms = 1000.0/float(tmr.Hz)
            else:
                period_ms = np.nan
            run_ms_list.append(float(run_ms))
            period_ms_list.append(float(period_ms))

        run_sum_ms = float(np.nansum(run_ms_list))
        finite_periods = [p for p in period_ms_list if np.isfinite(p) and p > 0.0]
        min_period_ms = min(finite_periods) if len(finite_periods) > 0 else np.nan
        load = float(np.nansum([
            run_ms/period_ms
            for run_ms,period_ms in zip(run_ms_list,period_ms_list)
            if np.isfinite(run_ms) and np.isfinite(period_ms) and period_ms > 0.0
        ]))
        burst_load = run_sum_ms/min_period_ms if np.isfinite(min_period_ms) else np.nan
        self.viewer_text_overlay(
            text1 = "Timers",
            text2 = "run_sum:[%.2f]ms load:[%.2f]x burst:[%.2f]x"%(
                run_sum_ms,load,burst_load),
            loc   = loc,
        )

        for tmr,run_ms,period_ms in zip(tmrs,run_ms_list,period_ms_list):
            lag_info = tmr.get_lag_info()
            self.viewer_text_overlay(
                text1 = "%s timer"%(tmr.name),
                text2 = "lag:[%d]tick [%.1f]ms run:[%.2f]ms period:[%.1f]ms"%(
                    lag_info["tick"],lag_info["ms"],run_ms,period_ms),
                loc   = loc,
            )

    @staticmethod
    def eval_quartic(x,coef=np.zeros(5),x0=0.0,y0=0.0):
        """
        Evaluate a quartic polynomial function at a given point x.

        Parameters:
            x (Any): Input x.
            coef (Any): Input coef. Defaults to `np.zeros(5)`.
            x0 (float): Input x0. Defaults to `0.0`.
            y0 (float): Input y0. Defaults to `0.0`.

        Returns:
            Any: Result computed from the current inputs.
        """
        x_bar = float(x)-x0
        return y0+coef[0]+coef[1]*x_bar+coef[2]*x_bar**2+coef[3]*x_bar**3+coef[4]*x_bar**4

    @staticmethod
    def eval_quartic_derivative(x,coef=np.zeros(5),x0=0.0,y0=0.0):
        """
        Evaluate the derivative of a quartic polynomial function at a given point x.

        Parameters:
            x (Any): Input x.
            coef (Any): Input coef. Defaults to `np.zeros(5)`.
            x0 (float): Input x0. Defaults to `0.0`.
            y0 (float): Input y0. Defaults to `0.0`.

        Returns:
            Any: Result computed from the current inputs.
        """
        x_bar = x-x0
        return coef[1]+2*coef[2]*x_bar+3*coef[3]*x_bar**2+4*coef[4]*x_bar**3

    def get_qrev_from_qactive(self,qactive):
        """
        Given active joint positions, compute the full set of joint positions.

        Parameters:
            qactive (np.ndarray): Input qactive.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        # Initialize rev joint positions
        qrev = self.get_qpos(joint_names=self.rev_joint_names)

        # Set active revolute joint positions to rev joint positions
        active_rev_joint_names = self.active_rev_joint_names
        qactive = np.asarray(qactive,dtype=np.float64).reshape(-1,)
        if len(qactive) == len(self.active_joint_names):
            qactive_map = {j:qactive[idx] for idx,j in enumerate(self.active_joint_names)}
            qactive_rev = np.array([qactive_map[j] for j in active_rev_joint_names],dtype=np.float64)
        else:
            qactive_rev = qactive
        qrev[get_idxs(self.rev_joint_names,active_rev_joint_names)] = qactive_rev

        # Update passive joint values based on equality constraints
        for eq_idx in range(self.n_eq): # for each joint equality constraints
            if self.model.eq_type[eq_idx] != mujoco.mjtEq.mjEQ_JOINT:
                continue
            passive_joint = self.model.joint(self.model.eq_obj1id[eq_idx]).name
            active_joint  = self.model.joint(self.model.eq_obj2id[eq_idx]).name
            coef = self.model.eq_data[eq_idx,:5]
            x0   = self.model.joint(active_joint).qpos0[0]
            y0   = self.model.joint(passive_joint).qpos0[0]
            x    = qactive_rev[active_rev_joint_names.index(active_joint)]
            # Evaluate quartic polynomial
            y    = self.eval_quartic(x=x,coef=coef,x0=x0,y0=y0)
            # Set passive joint position
            qrev[self.rev_joint_names.index(passive_joint)] = y

        # Return
        return qrev

    def get_qrevpri_from_qactive(self,qactive):
        """
        Given active revolute/prismatic joint positions, compute the full.

        Parameters:
            qactive (np.ndarray): Input qactive.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        return self.get_qfull_from_qactive(
            full_joints    = self.rev_pri_joint_names,
            active_joints  = self.active_joint_names,
            qactive        = qactive,
        )

    def get_qfull_from_qactive(self,full_joints,active_joints,qactive):
        """
        Given a list of full joint names, a list of active joint names, and.

        Parameters:
            full_joints (Any): Input full joints.
            active_joints (Any): Input active joints.
            qactive (np.ndarray): Input qactive.

        Returns:
            Any: Requested value computed from the current inputs.
        """

        # Sanity check and convert inputs
        full_joints   = list(full_joints)
        active_joints = list(active_joints)
        qactive       = np.asarray(qactive,dtype=np.float64).reshape(-1,)

        # Check lengths
        if len(active_joints)!=len(qactive):
            raise ValueError("len(active_joints)=[%d] but len(qactive)=[%d]"%(len(active_joints),len(qactive)))

        # Start from current q (keeps unrelated joints stable if user passes mixed lists)
        qfull = self.get_qpos(joint_names=full_joints,flatten=True).copy()

        # Create name to index mappings
        j2i = {j:i for i,j in enumerate(full_joints)}
        a2i = {j:i for i,j in enumerate(active_joints)}

        # Set actives
        for j in active_joints:
            if j in j2i:
                qfull[j2i[j]] = float(qactive[a2i[j]])

        # Fill passives via equality constraints
        for eq_idx in range(self.n_eq): # for each joint equality constraints
            if self.model.eq_type[eq_idx]!=mujoco.mjtEq.mjEQ_JOINT:
                continue
            pj = self.model.joint(self.model.eq_obj1id[eq_idx]).name # passive joint
            aj = self.model.joint(self.model.eq_obj2id[eq_idx]).name # active joint
            if (pj not in j2i) or (aj not in a2i):
                continue
            coef = self.model.eq_data[eq_idx,:5]
            x0   = self.model.joint(aj).qpos0[0]
            y0   = self.model.joint(pj).qpos0[0]
            x    = qactive[a2i[aj]]
            qfull[j2i[pj]] = float(self.eval_quartic(x=x,coef=coef,x0=x0,y0=y0))

        # Return
        return qfull

    def get_active_among_joints(self,joints,return_passive=False):
        """
        Given a list of joint names, return the subset that are active joints.

        Parameters:
            joints (Any): Input joints.
            return_passive (bool): Input return passive. Defaults to `False`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        active_set     = set(self.active_joint_names)
        active_joints  = [j for j in joints if j in active_set]
        passive_joints = [j for j in joints if j not in active_set]
        if return_passive:
            return active_joints,passive_joints
        else:
            return active_joints

    def get_joint_mins(self,joint_names=None):
        """
        Get the minimum position limits for specified joints.

        Parameters:
            joint_names (Any | None): Input joint names. Defaults to `None`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if joint_names is None:
            joint_names=self.joint_names
        joint_idxs = get_idxs(self.joint_names,joint_names)
        joint_mins = self.joint_mins[joint_idxs]
        return joint_mins

    def get_joint_maxs(self,joint_names=None):
        """
        Get the maximum position limits for specified joints.

        Parameters:
            joint_names (Any | None): Input joint names. Defaults to `None`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if joint_names is None:
            joint_names=self.joint_names
        joint_idxs = get_idxs(self.joint_names,joint_names)
        joint_maxs = self.joint_maxs[joint_idxs]
        return joint_maxs

    def get_subtree_body_names(
            self,
            root_body_name,
            include_root = False,
        ):
        """
        Get the names of all bodies in the subtree rooted at the specified body or
        bodies.

        Parameters:
            root_body_name (str | Path): Input root body name.
            include_root (bool): Input include root. Defaults to `False`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if isinstance(root_body_name,str):
            root_body_names = [root_body_name]
        else:
            root_body_names = list(root_body_name)

        body_names = []
        for root_body_name in root_body_names:
            root_id=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,root_body_name)
            if root_id==-1:
                raise ValueError(f"Body '{root_body_name}' not found.")

            if include_root:
                body_names.append(root_body_name)

            stack = list(reversed(self.body_children_list[root_id]))
            while stack:
                body_id = stack.pop()
                body_name = self.body_names[body_id]
                if body_name is None:
                    continue
                body_names.append(body_name)
                stack.extend(reversed(self.body_children_list[body_id]))

        body_names = list(dict.fromkeys(body_names))
        return body_names

    def get_joint_names_from_root_body_name(
            self,
            root_body_name,
            include_root       = True,
            joint_types        = None,
            return_idxs        = False,
            use_body_idx_order = True,
            stop_body_name     = None,
        ):
        """
        Get joint names in the subtree rooted at root_body_name.

        Parameters:
            root_body_name (str | Path): Input root body name.
            include_root (bool): Input include root. Defaults to `True`.
            joint_types (Any | None): Input joint types. Defaults to `None`.
            return_idxs (bool): Input return idxs. Defaults to `False`.
            use_body_idx_order (bool): Input use body idx order. Defaults to `True`.
            stop_body_name (Any | None): Input stop body name. Defaults to `None`.

        Returns:
            Any: Requested value computed from the current inputs.
        """

        # Normalize joint type filter
        joint_type_set = normalize_mujoco_joint_types(joint_types)

        # Get subtree body names
        if use_body_idx_order:
            body_idxs = self.get_body_idxs_from_root_body_name(
                root_body_name = root_body_name,
                include_root   = include_root,
            )
            body_names = [self.body_names[bidx] for bidx in body_idxs]
        else:
            body_names = self.get_subtree_body_names(
                root_body_name = root_body_name,
                include_root   = include_root,
            )

        # Collect joints attached to each body
        joint_idxs  = []
        joint_names = []

        for body_name in body_names:
            # Stop from this body
            if (stop_body_name is not None) and (body_name == stop_body_name):
                break

            body = self.model.body(body_name)

            jntadr = int(body.jntadr[0])
            jntnum = int(body.jntnum[0])

            if jntnum <= 0:
                continue

            for jidx in range(jntadr,jntadr+jntnum):
                jtype = int(self.model.jnt_type[jidx])

                if joint_type_set is not None:
                    if jtype not in joint_type_set:
                        continue

                joint_idxs.append(jidx)
                joint_names.append(self.joint_names[jidx])

        if return_idxs:
            return joint_idxs,joint_names
        else:
            return joint_names

    # -------------------------------------------------------------------------
    # Object point cloud cache
    # -------------------------------------------------------------------------
    @staticmethod
    def _transform_pcd(pcd,p,R):
        """
        Transform point cloud with p and R.

        Parameters:
            pcd (Any): Input pcd.
            p (np.ndarray): Input p.
            R (Any): Input R.

        Returns:
            Any: Result computed from the current inputs.
        """
        pcd = np.asarray(pcd,dtype=np.float64).reshape(-1,3)
        p   = np.asarray(p,dtype=np.float64).reshape(3,)
        R   = np.asarray(R,dtype=np.float64).reshape(3,3)
        return (R@pcd.T).T+p

    def clear_obj_pcd_cache(self,obj_body_names=None):
        """
        Clear cached object-local point clouds.

        Parameters:
            obj_body_names (Any | None): Input obj body names. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        cache_attr_list = [attr for attr in dir(self) if attr.startswith("obj_pcd_local_cache")]
        if len(cache_attr_list) == 0:
            self.obj_pcd_local_cache = {}
            return

        for cache_attr in cache_attr_list:
            cache = getattr(self,cache_attr)
            if obj_body_names is None:
                setattr(self,cache_attr,{})
            else:
                for obj_body_name in obj_body_names:
                    cache.pop(obj_body_name,None)

    def _get_mesh_points_from_geom_idx(
            self,
            geom_idx,
            pcd_mode = "face",
        ):
        """
        Extract simple compiled-mesh points from one geom in geom-local coordinates.

        Parameters:
            geom_idx (int): Input geom idx.
            pcd_mode (str): Input pcd mode. Defaults to `'face'`.

        Returns:
            Any: Result computed from the current inputs.
        """
        if pcd_mode not in ["face","vertex"]:
            raise ValueError(
                "[_get_mesh_points_from_geom_idx] unsupported pcd_mode:[%s]."%(pcd_mode)
            )

        geom_type = int(self.model.geom_type[geom_idx])
        if geom_type!=mujoco.mjtGeom.mjGEOM_MESH:
            return None

        mesh_idx = int(self.model.geom_dataid[geom_idx])
        if mesh_idx<0:
            return None

        start_idx = int(self.model.mesh_vertadr[mesh_idx])
        n_vert    = int(self.model.mesh_vertnum[mesh_idx])
        if n_vert<=0:
            return None

        # MuJoCo stores mesh vertices in the compiled mesh-local frame.
        vertices = self.model.mesh_vert[start_idx:start_idx+n_vert].reshape(-1,3).copy()

        if pcd_mode == "vertex":
            return vertices

        face_idx = int(self.model.mesh_faceadr[mesh_idx])
        n_face   = int(self.model.mesh_facenum[mesh_idx])
        if n_face<=0:
            return None

        # For the face mode, collapse each triangle to one representative point.
        faces = self.model.mesh_face[face_idx:face_idx+n_face].reshape(-1,3)
        v0 = vertices[faces[:,0]]
        v1 = vertices[faces[:,1]]
        v2 = vertices[faces[:,2]]
        centroids = (v0+v1+v2)/3.0
        return centroids

    def _build_body_local_pcd(
            self,
            obj_body_name,
            n_sample         = 256,
            pcd_mode         = "mesh",
            seed             = 0,
        ):
        """
        Build a body-local canonical point cloud for one object body.

        Parameters:
            obj_body_name (str): Input obj body name.
            n_sample (int): Input n sample. Defaults to `256`.
            pcd_mode (str): Input pcd mode. Defaults to `'mesh'`.
            seed (int): Input seed. Defaults to `0`.

        Returns:
            Any: Result computed from the current inputs.
        """
        if pcd_mode not in ["face","vertex","mesh"]:
            raise ValueError(
                "[_build_body_local_pcd] unsupported pcd_mode:[%s]."%(pcd_mode)
            )
        DENSE_SAMPLE_FACTOR  = 5
        MESH_FIT_CHECK_N     = 512
        MESH_FIT_CHECK_RATIO = 0.02
        rng = np.random.default_rng(seed)

        geom_idxs = self.get_geom_idxs_from_root_body_name(
            root_body_name = obj_body_name,
            include_root   = True,
        )

        p_body = self.get_p(obj_body_name,'body')
        R_body = self.get_R(obj_body_name,'body')

        def _to_body_local(geom_idx,points_geom_local):
            """
            Convert geom-local points into the root-body-local frame.

            Parameters:
                geom_idx (int): Input geom idx.
                points_geom_local (Any): Input points geom local.

            Returns:
                Any: Result computed from the current inputs.
            """
            p_geom = self.data.geom_xpos[geom_idx].copy()
            R_geom = self.data.geom_xmat[geom_idx].reshape(3,3).copy()
            R_offset = R_body.T@R_geom
            p_offset = R_body.T@(p_geom-p_body)
            return self._transform_pcd(points_geom_local,p_offset,R_offset)

        def _get_mesh_face_data(mesh_idx):
            """
            Return compiled mesh triangle data in mesh-local coordinates.

            Parameters:
                mesh_idx (int): Input mesh idx.

            Returns:
                Any: Result computed from the current inputs.
            """
            vert_adr = int(self.model.mesh_vertadr[mesh_idx])
            n_vert   = int(self.model.mesh_vertnum[mesh_idx])
            if n_vert <= 0:
                return None
            vertices = self.model.mesh_vert[vert_adr:vert_adr+n_vert].reshape(-1,3)

            face_adr = int(self.model.mesh_faceadr[mesh_idx])
            n_face   = int(self.model.mesh_facenum[mesh_idx])
            if n_face <= 0:
                return None
            faces = self.model.mesh_face[face_adr:face_adr+n_face].reshape(-1,3)

            v0 = vertices[faces[:,0]]
            v1 = vertices[faces[:,1]]
            v2 = vertices[faces[:,2]]

            cross = np.cross(v1-v0,v2-v0)
            area  = 0.5*np.linalg.norm(cross,axis=1)
            valid_mask = area > 1e-12
            if not np.any(valid_mask):
                return None

            v0   = v0[valid_mask]
            v1   = v1[valid_mask]
            v2   = v2[valid_mask]
            area = area[valid_mask]
            return v0,v1,v2,area

        def _sample_triangle_surface(v0,v1,v2,area,n_sample_geom):
            """
            Sample triangle interiors with area-weighted barycentric sampling.

            Parameters:
                v0 (Any): Input v0.
                v1 (Any): Input v1.
                v2 (Any): Input v2.
                area (Any): Input area.
                n_sample_geom (int): Input n sample geom.

            Returns:
                Any: Result computed from the current inputs.
            """
            if len(area) == 0:
                return None

            prob      = area/np.sum(area)
            face_idxs = rng.choice(len(area),size=n_sample_geom,p=prob)
            sv0 = v0[face_idxs]
            sv1 = v1[face_idxs]
            sv2 = v2[face_idxs]

            u      = rng.random((n_sample_geom,1))
            v      = rng.random((n_sample_geom,1))
            sqrt_u = np.sqrt(u)
            w0 = 1.0 - sqrt_u
            w1 = sqrt_u*(1.0-v)
            w2 = sqrt_u*v
            return w0*sv0 + w1*sv1 + w2*sv2

        def _sample_compiled_mesh_surface(mesh_idx,n_sample_geom):
            """
            Sample the compiled MuJoCo mesh surface in mesh-local coordinates.

            Parameters:
                mesh_idx (int): Input mesh idx.
                n_sample_geom (int): Input n sample geom.

            Returns:
                Any: Result computed from the current inputs.
            """
            face_data = _get_mesh_face_data(mesh_idx)
            if face_data is None:
                return None
            return _sample_triangle_surface(*face_data,n_sample_geom)

        def _get_mesh_geom_idxs(geom_idxs,visual_only=False):
            """
            Collect mesh geoms, optionally preferring visual geoms.

            Parameters:
                geom_idxs (Any): Input geom idxs.
                visual_only (bool): Input visual only. Defaults to `False`.

            Returns:
                Any: Result computed from the current inputs.
            """
            mesh_geom_idxs = []
            visual_geom_idxs = []
            for geom_idx in geom_idxs:
                if int(self.model.geom_type[geom_idx]) != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                mesh_geom_idxs.append(geom_idx)
                is_visual = (
                    int(self.model.geom_group[geom_idx]) == 2 or
                    (
                        int(self.model.geom_contype[geom_idx]) == 0 and
                        int(self.model.geom_conaffinity[geom_idx]) == 0
                    )
                )
                if is_visual:
                    visual_geom_idxs.append(geom_idx)
            if visual_only:
                return visual_geom_idxs if len(visual_geom_idxs) > 0 else mesh_geom_idxs
            return mesh_geom_idxs

        vertices_tf_list = []
        if pcd_mode == "mesh":
            # The mesh mode uses the original asset files. We keep the parsing
            # and alignment data cached on the parser instance because the same
            # mesh assets are often reused across multiple calls.
            import trimesh
            import xml.etree.ElementTree as ET

            if not hasattr(self,"mesh_asset_info_dict"):
                self.mesh_asset_info_dict = {}

                def _parse_mesh_asset_info_dict(xml_path,visited_xml_paths=None):
                    """
                    Recursively parse mesh asset definitions from the main XML and
                    all included XML files.

                    We only collect the information needed to reconstruct raw mesh
                    samples in the same nominal asset frame that MuJoCo expects:
                    the source file, the asset scale, and the asset refquat.

                    Parameters:
                        xml_path (str | Path): XML file to inspect.
                        visited_xml_paths (set | None): XML files already parsed
                            during this recursive pass.

                    Returns:
                        None: Updates mesh_asset_info_dict in place.
                    """
                    if visited_xml_paths is None:
                        visited_xml_paths = set()
                    xml_path = os.path.abspath(xml_path)
                    if xml_path in visited_xml_paths:
                        return
                    visited_xml_paths.add(xml_path)
                    if not os.path.exists(xml_path):
                        return
                    xml_dir = os.path.dirname(xml_path)
                    root = ET.parse(xml_path).getroot()
                    for mesh_elem in root.findall("./asset/mesh"):
                        mesh_name = mesh_elem.get("name",None)
                        mesh_file = mesh_elem.get("file",None)
                        if mesh_name is None or mesh_file is None:
                            continue
                        scale = np.ones(3,dtype=np.float64)
                        scale_attr = mesh_elem.get("scale",None)
                        if scale_attr is not None:
                            scale_vals = np.fromstring(scale_attr,sep=" ",dtype=np.float64)
                            if len(scale_vals) == 1:
                                scale = np.repeat(scale_vals,3)
                            elif len(scale_vals) == 3:
                                scale = scale_vals
                        refquat = np.array([1.0,0.0,0.0,0.0],dtype=np.float64)
                        refquat_attr = mesh_elem.get("refquat",None)
                        if refquat_attr is not None:
                            refquat_vals = np.fromstring(refquat_attr,sep=" ",dtype=np.float64)
                            if len(refquat_vals) == 4:
                                refquat = refquat_vals
                        self.mesh_asset_info_dict[mesh_name] = dict(
                            mesh_file = os.path.abspath(os.path.join(xml_dir,mesh_file)),
                            scale     = scale,
                            refquat   = refquat,
                        )
                    for include_elem in root.findall("./include"):
                        include_file = include_elem.get("file",None)
                        if include_file is None:
                            continue
                        include_path = os.path.abspath(os.path.join(xml_dir,include_file))
                        _parse_mesh_asset_info_dict(
                            xml_path           = include_path,
                            visited_xml_paths  = visited_xml_paths,
                        )

                # Parse once from the top-level scene XML.
                _parse_mesh_asset_info_dict(self.full_xml_path)

            if not hasattr(self,"mesh_raw_to_compiled_tf_dict"):
                self.mesh_raw_to_compiled_tf_dict = {}

            visual_geom_idxs = _get_mesh_geom_idxs(geom_idxs,visual_only=True)
            n_sample_geom = max(n_sample*2,n_sample)
            for geom_idx in visual_geom_idxs:
                mesh_idx = int(self.model.geom_dataid[geom_idx])
                if mesh_idx < 0:
                    continue
                mesh_name = self.mesh_names[mesh_idx]
                mesh_info = self.mesh_asset_info_dict.get(mesh_name,None)
                if mesh_info is None:
                    continue
                mesh_file = mesh_info["mesh_file"]
                if not os.path.exists(mesh_file):
                    continue

                # Sample directly from the original mesh surface with the same
                # deterministic triangle sampler used by the face mode.
                mesh = trimesh.load(mesh_file,force="mesh",process=False)
                raw_vertices = np.asarray(mesh.vertices,dtype=np.float64).reshape(-1,3)
                raw_faces = np.asarray(mesh.faces,dtype=np.int32).reshape(-1,3)
                raw_v0 = raw_vertices[raw_faces[:,0]]
                raw_v1 = raw_vertices[raw_faces[:,1]]
                raw_v2 = raw_vertices[raw_faces[:,2]]
                raw_cross = np.cross(raw_v1-raw_v0,raw_v2-raw_v0)
                raw_area = 0.5*np.linalg.norm(raw_cross,axis=1)
                raw_valid_mask = raw_area > 1e-12
                if not np.any(raw_valid_mask):
                    continue
                mesh_pcd_raw = _sample_triangle_surface(
                    raw_v0[raw_valid_mask],
                    raw_v1[raw_valid_mask],
                    raw_v2[raw_valid_mask],
                    raw_area[raw_valid_mask],
                    n_sample_geom,
                )
                if mesh_pcd_raw is None:
                    continue
                mesh_pcd_raw = mesh_pcd_raw * mesh_info["scale"].reshape(1,3)

                # Apply the mesh asset refquat so the raw asset sample is expressed
                # in the same nominal mesh frame expected by MuJoCo.
                R_ref_flat = np.zeros(9,dtype=np.float64)
                mujoco.mju_quat2Mat(
                    R_ref_flat,
                    np.asarray(mesh_info["refquat"],dtype=np.float64).reshape(4,),
                )
                mesh_pcd_raw = (R_ref_flat.reshape(3,3) @ mesh_pcd_raw.T).T

                if mesh_name not in self.mesh_raw_to_compiled_tf_dict:
                    # Estimate a rigid transform from the raw asset mesh vertices
                    # to the compiled MuJoCo mesh vertices once, then validate the
                    # fit. If the alignment looks unreliable, fall back to the
                    # compiled MuJoCo mesh surface for safety.
                    raw_mesh_vertices = raw_vertices.copy()
                    raw_mesh_vertices = raw_mesh_vertices * mesh_info["scale"].reshape(1,3)
                    raw_mesh_vertices = (R_ref_flat.reshape(3,3) @ raw_mesh_vertices.T).T

                    vert_adr = int(self.model.mesh_vertadr[mesh_idx])
                    n_vert = int(self.model.mesh_vertnum[mesh_idx])
                    compiled_mesh_vertices = None if n_vert <= 0 else \
                        np.asarray(self.model.mesh_vert[vert_adr:vert_adr+n_vert],dtype=np.float64).reshape(-1,3)

                    if compiled_mesh_vertices is None or raw_mesh_vertices.shape != compiled_mesh_vertices.shape:
                        self.mesh_raw_to_compiled_tf_dict[mesh_name] = dict(mode="fallback")
                    else:
                        c_raw = raw_mesh_vertices.mean(axis=0)
                        c_compiled = compiled_mesh_vertices.mean(axis=0)
                        H = (raw_mesh_vertices-c_raw).T @ (compiled_mesh_vertices-c_compiled)
                        U,_,Vt = np.linalg.svd(H)
                        R_fit = Vt.T @ U.T
                        if np.linalg.det(R_fit) < 0:
                            Vt[-1,:] *= -1.0
                            R_fit = Vt.T @ U.T
                        p_fit = c_compiled - R_fit @ c_raw
                        raw_vertices_tf = (R_fit @ raw_mesh_vertices.T).T + p_fit

                        # Validate the fit with a lightweight symmetric NN check.
                        # This catches the occasional rotation mismatch that can
                        # happen when raw and compiled vertex orderings differ.
                        idx_raw = np.linspace(
                            0,len(raw_vertices_tf)-1,
                            num=min(MESH_FIT_CHECK_N,len(raw_vertices_tf)),
                            dtype=int,
                        )
                        idx_compiled = np.linspace(
                            0,len(compiled_mesh_vertices)-1,
                            num=min(MESH_FIT_CHECK_N,len(compiled_mesh_vertices)),
                            dtype=int,
                        )
                        A = raw_vertices_tf[idx_raw]
                        B = compiled_mesh_vertices[idx_compiled]
                        d_ab = np.linalg.norm(A[:,None,:]-B[None,:,:],axis=2)
                        nn_err = 0.5*(
                            np.mean(np.min(d_ab,axis=1)) +
                            np.mean(np.min(d_ab,axis=0))
                        )
                        bbox_diag = np.linalg.norm(
                            np.max(compiled_mesh_vertices,axis=0) -
                            np.min(compiled_mesh_vertices,axis=0)
                        )
                        if bbox_diag <= 1e-12:
                            bbox_diag = 1.0
                        fit_ratio = nn_err / bbox_diag

                        if fit_ratio <= MESH_FIT_CHECK_RATIO:
                            self.mesh_raw_to_compiled_tf_dict[mesh_name] = dict(
                                mode  = "fit",
                                p_fit = p_fit,
                                R_fit = R_fit,
                            )
                        else:
                            self.mesh_raw_to_compiled_tf_dict[mesh_name] = dict(mode="fallback")

                tf_info = self.mesh_raw_to_compiled_tf_dict.get(mesh_name,None)
                if tf_info is None:
                    continue
                if tf_info["mode"] == "fit":
                    p_fit = tf_info["p_fit"]
                    R_fit = tf_info["R_fit"]

                    # Convert sampled raw-mesh points into the compiled mesh frame,
                    # then into the root body-local frame.
                    mesh_pcd_geom = (R_fit @ mesh_pcd_raw.T).T + p_fit
                    vertices_tf_list.append(_to_body_local(geom_idx,mesh_pcd_geom))
                else:
                    compiled_mesh_pcd = _sample_compiled_mesh_surface(mesh_idx,n_sample_geom)
                    if compiled_mesh_pcd is None:
                        continue
                    vertices_tf_list.append(_to_body_local(geom_idx,compiled_mesh_pcd))

        elif pcd_mode == "face":
            # The face mode samples the compiled MuJoCo mesh surface more densely
            # than the requested output size, then uses FPS to keep good spatial
            # coverage in the final fixed-size point cloud.
            mesh_geom_idxs = _get_mesh_geom_idxs(geom_idxs)
            face_data_dict = {}
            geom_area_list = []
            for geom_idx in mesh_geom_idxs:
                mesh_idx = int(self.model.geom_dataid[geom_idx])
                if mesh_idx < 0:
                    continue
                face_data = _get_mesh_face_data(mesh_idx)
                if face_data is None:
                    continue
                face_data_dict[geom_idx] = face_data
                geom_area_list.append((geom_idx,np.sum(face_data[3])))

            total_area = sum(area_sum for _,area_sum in geom_area_list)
            for geom_idx,geom_area in geom_area_list:
                mesh_idx = int(self.model.geom_dataid[geom_idx])
                face_data = face_data_dict[geom_idx]
                if total_area > 1e-12:
                    n_sample_geom = max(
                        1,
                        int(np.ceil(n_sample*DENSE_SAMPLE_FACTOR*geom_area/total_area)),
                    )
                else:
                    n_sample_geom = max(
                        1,
                        int(np.ceil(n_sample*DENSE_SAMPLE_FACTOR/max(1,len(face_data_dict)))),
                    )
                vertices_sampled = _sample_triangle_surface(*face_data,n_sample_geom)
                vertices_tf_list.append(_to_body_local(geom_idx,vertices_sampled))
        else:
            # The vertex mode operates purely on MuJoCo's compiled mesh
            # representation without any extra surface resampling.
            for geom_idx in geom_idxs:
                vertices = self._get_mesh_points_from_geom_idx(
                    geom_idx = geom_idx,
                    pcd_mode = pcd_mode,
                )
                if vertices is None:
                    continue
                vertices_tf_list.append(_to_body_local(geom_idx,vertices))

        if len(vertices_tf_list)==0:
            return None

        pcd_concat = np.concatenate(vertices_tf_list,axis=0) # [N_total x 3]

        # Downsample to the target number of points while preserving spatial coverage.
        if len(pcd_concat)>=n_sample:
            # FPS keeps broad geometric coverage with deterministic spacing.
            idxs       = farthest_point_sampling(pcd_concat,n_sample,rng=rng)
            pcd_sample = pcd_concat[idxs]
        else:
            # Keep the output size fixed even for tiny meshes.
            idxs       = rng.choice(len(pcd_concat),n_sample,replace=True)
            pcd_sample = pcd_concat[idxs]

        return pcd_sample

    def _get_obj_pcd_cache_attr(
            self,
            pcd_mode         = "mesh",
            seed             = 0,
        ):
        """
        Return the cache attribute name for the requested PCD configuration.

        Parameters:
            pcd_mode (str): Input pcd mode. Defaults to `'mesh'`.
            seed (int): Input seed. Defaults to `0`.

        Returns:
            Any: Result computed from the current inputs.
        """
        if pcd_mode not in ["face","vertex","mesh"]:
            raise ValueError(
                "[_get_obj_pcd_cache_attr] unsupported pcd_mode:[%s]."%(pcd_mode)
            )
        if pcd_mode == "face":
            return "obj_pcd_local_cache" if seed == 0 else \
                "obj_pcd_local_cache_face_seed_%d"%(seed)
        if pcd_mode == "vertex":
            return "obj_pcd_local_cache_vertex"
        return "obj_pcd_local_cache_mesh" if seed == 0 else \
            "obj_pcd_local_cache_mesh_seed_%d"%(seed)

    def build_obj_pcd_cache(
            self,
            obj_body_names,
            n_sample         = 256,
            pcd_mode         = "mesh",
            seed             = 0,
            verbose          = False,
            overwrite        = True,
        ):
        """
        Build object-local point cloud cache for the specified bodies.

        Parameters:
            obj_body_names (list[str]): Input obj body names.
            n_sample (int): Input n sample. Defaults to `256`.
            pcd_mode (str): Input pcd mode. Defaults to `'mesh'`.
            seed (int): Input seed. Defaults to `0`.
            verbose (bool): Input verbose. Defaults to `False`.
            overwrite (bool): Input overwrite. Defaults to `True`.

        Returns:
            Any: Result computed from the current inputs.
        """
        cache_attr = self._get_obj_pcd_cache_attr(
            pcd_mode         = pcd_mode,
            seed             = seed,
        )
        if not hasattr(self,cache_attr):
            setattr(self,cache_attr,{})
        obj_pcd_local_cache = getattr(self,cache_attr)

        for obj_idx,obj_body_name in enumerate(obj_body_names):
            if (not overwrite) and (obj_body_name in obj_pcd_local_cache):
                if verbose:
                    print("[%d/%d] skip cached obj body:[%s]"%
                        (obj_idx+1,len(obj_body_names),obj_body_name))
                continue

            pcd_local = self._build_body_local_pcd(
                obj_body_name    = obj_body_name,
                n_sample         = n_sample,
                pcd_mode         = pcd_mode,
                seed             = seed,
            )

            if pcd_local is None:
                if verbose:
                    print(" No valid mesh geom found.")
                continue

            obj_pcd_local_cache[obj_body_name] = pcd_local

            if verbose:
                print("[build_obj_pcd_cache] [%d/%d] build obj body:[%s] mode:[%s] cached pcd:%s"%
                    (obj_idx+1,len(obj_body_names),obj_body_name,pcd_mode,pcd_local.shape))

        return obj_pcd_local_cache

    def cache_obj_pcd(
            self,
            obj_body_names    = None,
            n_sample         = 256,
            pcd_mode         = "mesh",
            seed             = 0,
            verbose          = True,
            overwrite        = False,
        ):
        """
        Eagerly build object point-cloud cache for one or more object bodies.

        Parameters:
            obj_body_names (Any | None): Input obj body names. Defaults to `None`.
            n_sample (int): Input n sample. Defaults to `256`.
            pcd_mode (str): Input pcd mode. Defaults to `'mesh'`.
            seed (int): Input seed. Defaults to `0`.
            verbose (bool): Input verbose. Defaults to `True`.
            overwrite (bool): Input overwrite. Defaults to `False`.

        Returns:
            Any: Result computed from the current inputs.
        """
        if obj_body_names is None:
            obj_body_names = [name for name in self.body_names if str(name).startswith("body_obj")]

        if isinstance(obj_body_names,str):
            obj_body_names = [obj_body_names]

        obj_body_names = list(obj_body_names)

        return self.build_obj_pcd_cache(
            obj_body_names   = obj_body_names,
            n_sample         = n_sample,
            pcd_mode         = pcd_mode,
            seed             = seed,
            verbose          = verbose,
            overwrite        = overwrite,
        )

    def ensure_obj_pcd_cache(
            self,
            obj_body_names,
            n_sample         = 256,
            pcd_mode         = "mesh",
            seed             = 0,
            verbose          = False,
        ):
        """
        Ensure object-local point cloud cache exists for the specified bodies.

        Parameters:
            obj_body_names (list[str]): Input obj body names.
            n_sample (int): Input n sample. Defaults to `256`.
            pcd_mode (str): Input pcd mode. Defaults to `'mesh'`.
            seed (int): Input seed. Defaults to `0`.
            verbose (bool): Input verbose. Defaults to `False`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        cache_attr = self._get_obj_pcd_cache_attr(
            pcd_mode         = pcd_mode,
            seed             = seed,
        )
        if not hasattr(self,cache_attr):
            setattr(self,cache_attr,{})
        obj_pcd_local_cache = getattr(self,cache_attr)

        missing_names = [name for name in obj_body_names if name not in obj_pcd_local_cache]
        if len(missing_names)>0:
            self.build_obj_pcd_cache(
                obj_body_names    = missing_names,
                n_sample         = n_sample,
                pcd_mode         = pcd_mode,
                seed             = seed,
                verbose          = verbose,
                overwrite        = False,
            )

    def get_obj_pcd(
            self,
            obj_body_name,
            n_sample         = 256,
            pcd_mode         = "mesh",
            seed             = 0,
            auto_build       = True,
            verbose          = True,
        ):
        """
        Get current world-frame object point cloud.

        Parameters:
            obj_body_name (str): Input obj body name.
            n_sample (int): Input n sample. Defaults to `256`.
            pcd_mode (str): Input pcd mode. Defaults to `'mesh'`.
            seed (int): Input seed. Defaults to `0`.
            auto_build (bool): Input auto build. Defaults to `True`.
            verbose (bool): Input verbose. Defaults to `True`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        cache_attr = self._get_obj_pcd_cache_attr(
            pcd_mode         = pcd_mode,
            seed             = seed,
        )
        if not hasattr(self,cache_attr):
            setattr(self,cache_attr,{})
        obj_pcd_local_cache = getattr(self,cache_attr)

        if obj_body_name not in obj_pcd_local_cache:
            if not auto_build: # don't build cache, just raise error
                raise KeyError(
                    "[get_obj_pcd] obj_body_name:[%s] mode:[%s] not found in cache."%
                    (obj_body_name,pcd_mode)
                )

            # Build cache for this body (and skip if already built by another thread in the meantime)
            print_red ("[get_obj_pcd] cache miss for obj_body_name:[%s] mode:[%s] n_sample:[%d]."%(
                obj_body_name,pcd_mode,n_sample))
            self.build_obj_pcd_cache(
                obj_body_names    = [obj_body_name],
                n_sample         = n_sample,
                pcd_mode         = pcd_mode,
                seed             = seed,
                verbose          = verbose,
                overwrite        = False,
            )
            if verbose:
                print ("[get_obj_pcd] built cache for obj_body_name:[%s] mode:[%s] n_sample:[%d]."%(
                    obj_body_name,pcd_mode,n_sample))

        if obj_body_name not in obj_pcd_local_cache:
            raise ValueError(
                "[get_obj_pcd] failed to build object pcd for [%s] mode:[%s] n_sample:[%d]."%(
                    obj_body_name,pcd_mode,n_sample))

        pcd_local = obj_pcd_local_cache[obj_body_name]
        p_body    = self.get_p(obj_body_name,'body')
        R_body    = self.get_R(obj_body_name,'body')
        pcd_world = self._transform_pcd(pcd_local,p_body,R_body)
        return pcd_world

    def set_obj_bottom_T(
            self,
            obj_name,
            T,
            pcd_mode = "mesh",
            n_sample = 128,
        ):
        """
        Set an object's pose so the bottom of its point cloud lands at `T`'s z
        position.

        Parameters:
            obj_name (str): Input obj name.
            T (Any): Input T.
            pcd_mode (str): Input pcd mode. Defaults to `'mesh'`.
            n_sample (int): Input n sample. Defaults to `128`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        T = np.asarray(T,dtype=np.float64).copy()
        p_target = t2p(T)

        self.set_T(obj_name,"base_body",T=T)
        obj_pcd = self.get_obj_pcd(
            obj_body_name = obj_name,
            n_sample      = n_sample,
            pcd_mode      = pcd_mode,
            verbose       = False,
        )
        bottom_z = float(obj_pcd[:,2].min())
        T[2,3] += float(p_target[2]-bottom_z)
        self.set_T(obj_name,"base_body",T=T)

    def get_obj_pcd_center(self,obj_body_name):
        """
        Get the center of the object point cloud in world coordinates.

        Parameters:
            obj_body_name (str): Input obj body name.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        pcd = self.get_obj_pcd(obj_body_name=obj_body_name)
        center = 0.5*(pcd.min(axis=0)+pcd.max(axis=0))
        return center

    def get_obj_pcds(
            self,
            obj_body_names,
            n_sample         = 256,
            pcd_mode         = "mesh",
            seed             = 0,
            auto_build       = True,
        ):
        """
        Get current world-frame object point clouds for multiple bodies.

        Parameters:
            obj_body_names (list[str]): Input obj body names.
            n_sample (int): Input n sample. Defaults to `256`.
            pcd_mode (str): Input pcd mode. Defaults to `'mesh'`.
            seed (int): Input seed. Defaults to `0`.
            auto_build (bool): Input auto build. Defaults to `True`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if auto_build:
            self.ensure_obj_pcd_cache(
                obj_body_names    = obj_body_names,
                n_sample         = n_sample,
                pcd_mode         = pcd_mode,
                seed             = seed,
                verbose          = False,
            )

        obj_pcd_dict = {}
        for obj_body_name in obj_body_names:
            obj_pcd_dict[obj_body_name] = self.get_obj_pcd(
                obj_body_name    = obj_body_name,
                n_sample         = n_sample,
                pcd_mode         = pcd_mode,
                seed             = seed,
                auto_build       = auto_build,
            )
        return obj_pcd_dict

    def plot_obj_pcd(
            self,
            obj_body_name,
            n_sample         = 256,
            pcd_mode         = "mesh",
            seed             = 0,
            auto_build       = True,
            stride           = 1,
            r                = 0.002,
            rgba             = [0,0,1,0.5],
            label            = '',
        ):
        """
        Plot current object point cloud in the viewer.

        Parameters:
            obj_body_name (str): Input obj body name.
            n_sample (int): Input n sample. Defaults to `256`.
            pcd_mode (str): Input pcd mode. Defaults to `'mesh'`.
            seed (int): Input seed. Defaults to `0`.
            auto_build (bool): Input auto build. Defaults to `True`.
            stride (int): Input stride. Defaults to `1`.
            r (float): Input r. Defaults to `0.002`.
            rgba (list): Input rgba. Defaults to `[0, 0, 1, 0.5]`.
            label (str): Input label. Defaults to `''`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """

        # Get object point cloud
        obj_pcd = self.get_obj_pcd(
            obj_body_name    = obj_body_name,
            n_sample         = n_sample,
            pcd_mode         = pcd_mode,
            seed             = seed,
            auto_build       = auto_build,
        )

        # Plot spheres at point cloud locations
        self.plot_spheres(
            p_list = obj_pcd[::max(1,int(stride))],
            r      = r,
            rgba   = rgba,
            label  = label,
        )

    def plot_screw(
            self,
            q,
            w,
            R          = None,
            r_axis     = 0.0025,
            l          = 10,
            r_sphere   = 0.01,
            axis_len   = 0.1,
            axis_width = 0.01,
            axis_alpha = 0.5,
            rgba       = (1,0,0,0.5),
            label      = '',
            plot_axis  = True,
            plot_q     = True,
            plot_T     = True,
        ):
        """
        Plot a screw defined by a position, axis of rotation, and radius.

        Parameters:
            q (np.ndarray): Input q.
            w (Any): Input w.
            R (Any | None): Input R. Defaults to `None`.
            r_axis (float): Input r axis. Defaults to `0.0025`.
            l (int): Input l. Defaults to `10`.
            r_sphere (float): Input r sphere. Defaults to `0.01`.
            axis_len (float): Input axis len. Defaults to `0.1`.
            axis_width (float): Input axis width. Defaults to `0.01`.
            axis_alpha (float): Input axis alpha. Defaults to `0.5`.
            rgba (tuple): Input rgba. Defaults to `(1, 0, 0, 0.5)`.
            label (str): Input label. Defaults to `''`.
            plot_axis (bool): Input plot axis. Defaults to `True`.
            plot_q (bool): Input plot q. Defaults to `True`.
            plot_T (bool): Input plot T. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if plot_axis:
            p_fr = q - l*w
            p_to = q + l*w
            self.plot_cylinder_fr2to(
                p_fr  = p_fr,
                p_to  = p_to,
                r     = r_axis,
                rgba  = rgba,
                label = '',
            )
        if plot_q:
            self.plot_sphere(
                p     = q,
                r     = r_sphere,
                rgba  = rgba,
                label = label,
            )
        if plot_T:
            T = pr2t(p=q,R=R)
            self.plot_T(
                T          = T,
                plot_axis  = True,
                axis_len   = axis_len,
                axis_width = axis_width,
                axis_alpha = axis_alpha,
            )

    """
    Click handler
    """
    def get_viewer_mouse_xy(self):
        """
        Get the current mouse (x, y) coordinates from the viewer.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        viewer_mouse_xy = np.array([self.viewer._last_mouse_x,self.viewer._last_mouse_y])
        return viewer_mouse_xy

    @staticmethod
    def compute_view_params(
            camera_pos,
            target_pos,
            up_vector = np.array([0,0,1]), # Z-up convention
        ):
        """
        Compute view parameters (azimuth, distance, elevation, and lookat point) for a
        camera in 3D space.

        Parameters:
            camera_pos (Any): Input camera pos.
            target_pos (Any): Input target pos.
            up_vector (Any): Input up vector. Defaults to `np.array([0, 0, 1])`.

        Returns:
            Any: Result computed from the current inputs.
        """
        # Compute camera-to-target vector and distance
        cam_to_target = target_pos - camera_pos
        distance = np.linalg.norm(cam_to_target)

        # Compute azimuth and elevation
        azimuth   = np.arctan2(cam_to_target[1], cam_to_target[0])
        azimuth   = np.rad2deg(azimuth) # [deg]
        elevation = np.arcsin(cam_to_target[2] / distance)
        elevation = np.rad2deg(elevation) # [deg]

        # Compute lookat point
        lookat = target_pos

        # Compute camera orientation matrix
        zaxis = cam_to_target / distance
        xaxis = np.cross(up_vector, zaxis)
        yaxis = np.cross(zaxis, xaxis)
        cam_orient = np.array([xaxis, yaxis, zaxis]) # 3 x 3

        # Return computed values
        return azimuth, distance, elevation, lookat

    def get_T_viewer(self):
        """
        Compute the viewer camera's 4x4 transform.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        cam_lookat    = self.viewer.cam.lookat
        cam_elevation = self.viewer.cam.elevation
        cam_azimuth   = self.viewer.cam.azimuth
        cam_distance  = self.viewer.cam.distance

        p_lookat = cam_lookat
        R_lookat = rpy2r(np.deg2rad([0,-cam_elevation,cam_azimuth])) # Z-up convention
        T_lookat = pr2t(p_lookat,R_lookat)
        T_viewer = T_lookat @ pr2t(np.array([-cam_distance,0,0]),np.eye(3)) # camera looks along -X axis
        return T_viewer

    def _get_depth_world_pcd(
            self,
            depth_img,
            fovy       = 45,
            grid_shape = None,
        ):
        """
        Convert selected depth pixels directly into a world-frame point cloud.

        Parameters:
            depth_img (Any): Input depth img.
            fovy (int): Input fovy. Defaults to `45`.
            grid_shape (Any | None): Input grid shape. Defaults to `None`.

        Returns:
            Any: Result computed from the current inputs.
        """
        depth_img = np.asarray(depth_img)
        img_height,img_width = depth_img.shape[:2]
        focal_scaling = 0.5*img_height/np.tan(fovy*np.pi/360)

        if grid_shape is None:
            row_idxs,col_idxs = np.indices((img_height,img_width))
            out_shape = (img_height,img_width)
        else:
            if isinstance(grid_shape,int):
                n_row,n_col = int(grid_shape),int(grid_shape)
            else:
                n_row,n_col = [int(v) for v in grid_shape]
            if n_row <= 0 or n_col <= 0:
                raise ValueError("grid_shape entries must be positive.")
            grid_rows = np.linspace(0,img_height-1,n_row).astype(int)
            grid_cols = np.linspace(0,img_width-1,n_col).astype(int)
            row_idxs,col_idxs = np.meshgrid(grid_rows,grid_cols,indexing="ij")
            out_shape = (n_row,n_col)

        depth = depth_img[row_idxs,col_idxs].reshape(-1)
        rows  = row_idxs.reshape(-1)
        cols  = col_idxs.reshape(-1)
        pcd_cam = np.column_stack(
            [
                depth,
                -(cols-img_width/2)*depth/focal_scaling,
                -(rows-img_height/2)*depth/focal_scaling,
            ]
        )
        T_viewer = self.get_T_viewer()
        pcd_world = pcd_cam@T_viewer[:3,:3].T + T_viewer[:3,3]
        return pcd_cam,pcd_world,out_shape

    def _get_xyz_from_depth_img(
            self,
            depth_img,
            fovy               = 45,
            grid_shape         = None,
        ):
        """
        Convert a depth image or sampled depth grid into camera/world xyz arrays.

        Parameters:
            depth_img (Any): Input depth img.
            fovy (int): Input fovy. Defaults to `45`.
            grid_shape (Any | None): Input grid shape. Defaults to `None`.

        Returns:
            Any: Result computed from the current inputs.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=RuntimeWarning)
            xyz_cam,xyz_world,out_shape = self._get_depth_world_pcd(
                depth_img  = depth_img,
                fovy       = fovy,
                grid_shape = grid_shape,
            )
            xyz_img       = xyz_cam.reshape(out_shape[0],out_shape[1],3)
            xyz_img_world = xyz_world.reshape(out_shape[0],out_shape[1],3)

        return xyz_img,xyz_world,xyz_img_world

    def get_pcd_from_depth_img(
            self,
            depth_img,
            fovy               = 45,
            grid_shape         = None,
            return_finite_only = False,
        ):
        """
        Convert a depth image into a flattened world-frame point cloud.

        Parameters:
            depth_img (Any): Input depth img.
            fovy (int): Input fovy. Defaults to `45`.
            grid_shape (Any | None): Input grid shape. Defaults to `None`.
            return_finite_only (bool): Input return finite only. Defaults to `False`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        _,pcd,_ = self._get_depth_world_pcd(
            depth_img  = depth_img,
            fovy       = fovy,
            grid_shape = grid_shape,
        )
        if return_finite_only:
            pcd = pcd[np.all(np.isfinite(pcd),axis=1)]
        return pcd

    def get_xyz_from_depth_pixel(
            self,
            depth_img,
            row,
            col,
            fovy = 45,
        ):
        """
        Reconstruct one world-frame xyz point from one depth pixel.

        Parameters:
            depth_img (Any): Input depth img.
            row (Any): Input row.
            col (Any): Input col.
            fovy (int): Input fovy. Defaults to `45`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        row = int(np.clip(int(row),0,depth_img.shape[0]-1))
        col = int(np.clip(int(col),0,depth_img.shape[1]-1))
        depth = float(depth_img[row,col])
        img_height,img_width = depth_img.shape[:2]
        focal_scaling = 0.5*img_height/np.tan(fovy*np.pi/360)
        p_cam = np.array(
            [
                depth,
                -(col-img_width/2)*depth/focal_scaling,
                -(row-img_height/2)*depth/focal_scaling,
            ],
            dtype = np.float64,
        )
        T_viewer = self.get_T_viewer()
        return T_viewer[:3,:3]@p_cam + T_viewer[:3,3]

    def grab_rgbd_img(
            self,
            update_render_settings = True,
        ):
        """
        Capture the current RGB and depth images from the viewer.

        Parameters:
            update_render_settings (bool): Input update render settings. Defaults to
                                           `True`.

        Returns:
            Any: Result computed from the current inputs.
        """
        if update_render_settings:
            self.update_render_settings()
        rgb_img   = np.zeros((self.viewer.viewport.height,self.viewer.viewport.width,3),dtype=np.uint8)
        depth_img = np.zeros((self.viewer.viewport.height,self.viewer.viewport.width,1), dtype=np.float32)
        mujoco.mjr_readPixels(rgb_img,depth_img,self.viewer.viewport,self.viewer.ctx)
        rgb_img,depth_img = np.flipud(rgb_img),np.flipud(depth_img) # flip up-down

        # Rescale depth image
        extent = self.model.stat.extent
        near   = self.model.vis.map.znear * extent
        far    = self.model.vis.map.zfar * extent
        scaled_depth_img = near / (1.0 - depth_img * (1.0 - near / far)) # in meters
        depth_img = scaled_depth_img.squeeze()
        return rgb_img,depth_img

    def get_egocentric_rgbd_pcd(
            self,
            p_ego            = None,
            p_trgt           = None,
            rsz_rate_for_pcd = None,
            rsz_rate_for_img = None,
            fovy             = None,
            restore_view     = True,
            grid_shape       = None,
            return_finite_only = False,
        ):
        """
        Capture egocentric RGB/Depth images and generate a point cloud.

        Parameters:
            p_ego (Any | None): Input p ego. Defaults to `None`.
            p_trgt (Any | None): Input p trgt. Defaults to `None`.
            rsz_rate_for_pcd (Any | None): Input rsz rate for pcd. Defaults to `None`.
            rsz_rate_for_img (Any | None): Input rsz rate for img. Defaults to `None`.
            fovy (Any | None): Input fovy. Defaults to `None`.
            restore_view (bool): Input restore view. Defaults to `True`.
            grid_shape (Any | None): Input grid shape. Defaults to `None`.
            return_finite_only (bool): Input return finite only. Defaults to `False`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if restore_view:
            # Backup camera information
            viewer_azimuth,viewer_distance,viewer_elevation,viewer_lookat = self.viewer.get_cam_info()

        if (p_ego is not None) and (p_trgt is not None):
            cam_azimuth,cam_distance,cam_elevation,cam_lookat = self.compute_view_params(
                camera_pos = p_ego,
                target_pos = p_trgt,
                up_vector  = np.array([0,0,1]),
            )
            self.viewer.set_cam_info(
                azimuth   = cam_azimuth,
                distance  = cam_distance,
                elevation = cam_elevation,
                lookat    = cam_lookat,
            )

        # Update viewer and render to ensure the latest view is captured
        mujoco.mj_forward(self.model,self.data)
        mujoco.mjv_updateScene(
            self.model,self.data,self.viewer.vopt,self.viewer.pert,self.viewer.cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,self.viewer.scn)
        mujoco.mjr_render(self.viewer.viewport,self.viewer.scn,self.viewer.ctx)

        # Grab RGB and depth image
        rgb_img,depth_img = self.grab_rgbd_img(update_render_settings=False) # get rgb and depth images

        # Resize depth image for reducing point clouds
        if rsz_rate_for_pcd is not None:
            h_rsz         = int(depth_img.shape[0]*rsz_rate_for_pcd)
            w_rsz         = int(depth_img.shape[1]*rsz_rate_for_pcd)
            depth_img_rsz = cv2.resize(depth_img,(w_rsz,h_rsz),interpolation=cv2.INTER_NEAREST)
        else:
            depth_img_rsz = depth_img

        # Get PCD
        if fovy is None:
            if len(self.model.cam_fovy)==0: fovy = 45.0 # if cam is not defined, use 45 deg (default value)
            else: fovy = self.model.cam_fovy[0] # otherwise use the fovy of the first camera
        xyz_img,pcd,xyz_img_world = self._get_xyz_from_depth_img(
            depth_img   = depth_img_rsz,
            fovy        = fovy,
            grid_shape  = grid_shape,
        ) # [N x 3]
        if return_finite_only:
            pcd = pcd[np.all(np.isfinite(pcd),axis=1)]

        # Resize rgb_image and depth_img (optional)
        if rsz_rate_for_img is not None:
            h = int(rgb_img.shape[0]*rsz_rate_for_img)
            w = int(rgb_img.shape[1]*rsz_rate_for_img)
            rgb_img   = cv2.resize(rgb_img,(w,h),interpolation=cv2.INTER_NEAREST)
            depth_img = cv2.resize(depth_img,(w,h),interpolation=cv2.INTER_NEAREST)

        # Restore view
        if restore_view:
            # Restore camera information
            self.viewer.set_cam_info(
                azimuth   = viewer_azimuth,
                distance  = viewer_distance,
                elevation = viewer_elevation,
                lookat    = viewer_lookat,
            )
            mujoco.mj_forward(self.model,self.data)
            mujoco.mjv_updateScene(
                self.model,self.data,self.viewer.vopt,self.viewer.pert,self.viewer.cam,
                mujoco.mjtCatBit.mjCAT_ALL.value,self.viewer.scn)
            mujoco.mjr_render(self.viewer.viewport,self.viewer.scn,self.viewer.ctx)

        return rgb_img,depth_img,pcd,xyz_img,xyz_img_world

    def get_xyz_left_double_click(
            self,
            verbose     = False,
            fovy        = 45,
            pick_opaque = True,
        ):
        """
        Get xyz location of left double click.

        Parameters:
            verbose (bool): Input verbose. Defaults to `False`.
            fovy (int): Input fovy. Defaults to `45`.
            pick_opaque (bool): Input pick opaque. Defaults to `True`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        flag_click = False
        if self.viewer._left_double_click_pressed: # left double click
            # Get clicked pixel
            viewer_mouse_xy = self.get_viewer_mouse_xy()
            # Get point cloud. Transparent rendering is good for visualization,
            # but opaque depth is more reliable for depth-buffer picking.
            transparent_prev = bool(
                self.viewer.vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT]
            )
            try:
                if pick_opaque and transparent_prev:
                    self.viewer.set_transparency(transparent=False)
                _,depth_img_pick = self.grab_rgbd_img(update_render_settings=True)
            finally:
                if pick_opaque and transparent_prev:
                    self.viewer.set_transparency(transparent=transparent_prev)
            # Get clicked xyz
            self.xyz_left_double_click = self.get_xyz_from_depth_pixel(
                depth_img = depth_img_pick,
                row       = viewer_mouse_xy[1],
                col       = viewer_mouse_xy[0],
                fovy      = fovy,
            )
            # Reset flag in viewer
            self.viewer._left_double_click_pressed = False
            flag_click = True # set flag
            if verbose:
                print ("left double click:(%.3f,%.3f,%.3f)"%
                       (self.xyz_left_double_click[0],
                        self.xyz_left_double_click[1],
                        self.xyz_left_double_click[2])
                        )
        # Return
        return self.xyz_left_double_click,flag_click

    def plot_left_double_click(
            self,
            r     = 0.01,
            rgba  = (1,0,0,1),
            label = '',
        ):
        """
        Plot a sphere at the last left double-clicked position.

        Parameters:
            r (float): Input r. Defaults to `0.01`.
            rgba (tuple): Input rgba. Defaults to `(1, 0, 0, 1)`.
            label (str): Input label. Defaults to `''`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if self.xyz_left_double_click is not None:
            p = self.xyz_left_double_click
            self.plot_sphere(p=p,r=r,rgba=rgba,label=label)

    def get_obj_body_name_left_double_click(
            self,
            obj_body_names  = None,
            obj_body_prefix = "body_obj",
            max_dist        = 0.08,
            n_sample        = 128,
            pcd_mode        = "mesh",
            seed            = 0,
            verbose         = False,
            pick_opaque     = True,
        ):
        """
        Select the nearest object body to the last left double-click point.

        Parameters:
            obj_body_names (Any | None): Input obj body names. Defaults to `None`.
            obj_body_prefix (str): Input obj body prefix. Defaults to `'body_obj'`.
            max_dist (float): Input max dist. Defaults to `0.08`.
            n_sample (int): Input n sample. Defaults to `128`.
            pcd_mode (str): Input pcd mode. Defaults to `'mesh'`.
            seed (int): Input seed. Defaults to `0`.
            verbose (bool): Input verbose. Defaults to `False`.
            pick_opaque (bool): Input pick opaque. Defaults to `True`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        xyz_click, flag_click = self.get_xyz_left_double_click(
            verbose     = verbose,
            pick_opaque = pick_opaque,
        )
        if not flag_click:
            return None, xyz_click
        if xyz_click is None or not np.all(np.isfinite(xyz_click)):
            return None, xyz_click

        if obj_body_names is None:
            obj_body_names = [name for name in self.body_names if name.startswith(obj_body_prefix)]
        obj_body_names = list(obj_body_names)
        if len(obj_body_names) == 0:
            return None, xyz_click

        self.ensure_obj_pcd_cache(
            obj_body_names = obj_body_names,
            n_sample       = n_sample,
            pcd_mode       = pcd_mode,
            seed           = seed,
            verbose        = True, # <= verbose here to show which caches are being built for the click operation
        )

        best_name, best_dist = None, np.inf
        xyz_click = np.asarray(xyz_click,dtype=float).reshape(3)
        for obj_body_name in obj_body_names:
            pcd = self.get_obj_pcd(
                obj_body_name = obj_body_name,
                n_sample      = n_sample,
                pcd_mode      = pcd_mode,
                seed          = seed,
                auto_build    = False,
                verbose       = False,
            )
            if pcd.size == 0:
                continue
            dist = float(np.linalg.norm(pcd - xyz_click[None,:],axis=1).min())
            if dist < best_dist:
                best_name, best_dist = obj_body_name, dist

        if best_dist > float(max_dist):
            best_name = None
        return best_name, xyz_click

    def sync_sim_wall_time(self,rate=1.0):
        """
        Synchronize the simulation time with the wall time.

        Parameters:
            rate (float): Input rate. Defaults to `1.0`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        time_diff = self.get_sim_time() - self.get_wall_time()
        if time_diff > 0: time.sleep(rate*time_diff)
