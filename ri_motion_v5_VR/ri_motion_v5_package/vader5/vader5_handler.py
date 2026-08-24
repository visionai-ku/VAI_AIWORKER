"""
vader5_handler.py
"""
import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
    PYGAME_IMPORT_ERROR = None
except Exception as e:
    pygame = None
    PYGAME_AVAILABLE = False
    PYGAME_IMPORT_ERROR = e

from ri_motion_v5_package.utility.utils import (
    deadzone,
)
from ri_motion_v5_package.kinematics.transforms import (
    pr2t,
    t2p,
    t2r,
    rpy2r,
    t2x,
    t2y,
    t2z,
)

class Vader5Handler(object):
    """
    Vader5Handler handles Vader5 gamepad input parsing,rumble control,.

    Usage:
        obj = Vader5Handler(...)
    """
    def __init__(
            self,
            joystick_idx     = 0,
            axis_map         = None,
            button_map       = None,
            rot_dir_init     = "yaw",
            auto_init_pygame = True,
            required         = False,
            verbose          = True,
        ):
        """
        Initialize Vader5Handler.

        Parameters:
            joystick_idx (int): Input joystick idx. Defaults to `0`.
            axis_map (Any | None): Input axis map. Defaults to `None`.
            button_map (Any | None): Input button map. Defaults to `None`.
            rot_dir_init (str): Input rot dir init. Defaults to `'yaw'`.
            auto_init_pygame (bool): Input auto init pygame. Defaults to `True`.
            required (bool): Input required. Defaults to `False`.
            verbose (bool): Input verbose. Defaults to `True`.

        Returns:
            None: Initializes the instance in place.
        """
        self.joystick_idx = joystick_idx
        self.verbose      = verbose
        self.rot_dir      = rot_dir_init
        self.available    = False
        self.status_msg   = ""
        self.js           = None
        self.name         = "VADER5 unavailable"
        self.n_axis       = 0
        self.n_button     = 0
        self.axes_raw     = []
        self.buttons_raw  = []

        # Default logical mapping.
        # These indices may vary depending on OS,driver,and controller mode.
        self.axis_map = {
            "stick_left_lr"      : 0,
            "stick_left_ud"      : 1,
            "stick_right_lr"     : 2,
            "stick_right_ud"     : 3,
            "trigger_top_left"   : 4,
            "trigger_top_right"  : 5,
        }
        self.button_map = {
            "button_face_down"   : 0,
            "button_face_right"  : 1,
            "button_face_left"   : 2,
            "button_face_up"     : 3,
            "button_top_left"    : 9,
            "button_top_right"   : 10,
            "button_dpad_down"   : 12,
            "button_dpad_right"  : 14,
            "button_dpad_left"   : 13,
            "button_dpad_up"     : 11,
        }

        if axis_map is not None:
            self.axis_map.update(axis_map)
        if button_map is not None:
            self.button_map.update(button_map)

        if not PYGAME_AVAILABLE:
            self.mark_unavailable(
                "[Vader5] pygame import failed: [%s]. "
                "Running without VADER5 hardware."%(PYGAME_IMPORT_ERROR),
                required = required,
            )
            return

        try:
            if auto_init_pygame:
                if not pygame.get_init():
                    pygame.init()
                if not pygame.joystick.get_init():
                    pygame.joystick.init()

            n_joystick = pygame.joystick.get_count()
        except Exception as e:
            self.mark_unavailable(
                "[Vader5] pygame joystick initialization failed: [%s]. "
                "Running without VADER5 hardware."%(e),
                required = required,
            )
            return

        if n_joystick <= 0:
            self.mark_unavailable(
                "[Vader5] No joystick detected. Running without VADER5 hardware.",
                required = required,
            )
            return
        if (self.joystick_idx < 0) or (self.joystick_idx >= n_joystick):
            self.mark_unavailable(
                "Invalid joystick_idx:[%d]. Available joystick count:[%d]."
                " Running without VADER5 hardware."
                % (self.joystick_idx,n_joystick),
                required = required,
            )
            return

        self.js = pygame.joystick.Joystick(self.joystick_idx)
        self.js.init()
        self.available = True

        self.name     = self.js.get_name()
        self.n_axis   = self.js.get_numaxes()
        self.n_button = self.js.get_numbuttons()

        self.axes_raw    = [0.0 for _ in range(self.n_axis)]
        self.buttons_raw = [0 for _ in range(self.n_button)]

        if self.verbose:
            print("name:[%s]"%(self.name))
            print("axes:[%d]"%(self.n_axis))
            print("buttons:[%d]"%(self.n_button))

    def mark_unavailable(self,status_msg,required=False):
        """
        Store a non-fatal unavailable hardware state.

        Parameters:
            status_msg (Any): Input status msg.
            required (bool): Input required. Defaults to `False`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.available   = False
        self.status_msg  = status_msg
        self.js          = None
        self.name        = "VADER5 unavailable"
        self.n_axis      = 0
        self.n_button    = 0
        self.axes_raw    = []
        self.buttons_raw = []

        if required:
            raise RuntimeError(status_msg)
        if self.verbose:
            print(status_msg)

    def is_available(self):
        """
        Return whether a joystick is currently connected and opened.

        Parameters:
            None.

        Returns:
            bool: Boolean result for the requested condition.
        """
        return bool(self.available)

    def get_status_msg(self):
        """
        Return the current hardware status message.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if self.available:
            return "[Vader5] joystick available: [%s]"%(self.name)
        return self.status_msg

    def reset_motion_memory(self,rot_dir="yaw"):
        """
        Reset internal motion-related memory.

        Parameters:
            rot_dir (str): Input rot dir. Defaults to `'yaw'`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.rot_dir = rot_dir

    def update(self):
        """
        Update the cached raw gamepad state.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        if not self.available:
            return False
        pygame.event.pump()
        self.axes_raw    = [float(self.js.get_axis(i)) for i in range(self.n_axis)]
        self.buttons_raw = [int(self.js.get_button(i)) for i in range(self.n_button)]
        return True

    def get_state(self,update_first=True):
        """
        Get the parsed logical state using the configured maps.

        Parameters:
            update_first (bool): Input update first. Defaults to `True`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if update_first:
            self.update()

        state = {}

        if not self.available:
            for key in self.axis_map:
                if "trigger" in key:
                    state[key] = 1.0
                else:
                    state[key] = 0.0
            for key in self.button_map:
                state[key] = 0
            return state

        for key,idx in self.axis_map.items():
            if (idx is None) or (idx < 0) or (idx >= self.n_axis):
                state[key] = None
            else:
                state[key] = float(self.axes_raw[idx])

        for key,idx in self.button_map.items():
            if (idx is None) or (idx < 0) or (idx >= self.n_button):
                state[key] = None
            else:
                state[key] = int(self.buttons_raw[idx])

        return state

    def get_raw_state(self):
        """
        Get the cached raw axis/button state.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        return {
            "axes_raw"   : list(self.axes_raw),
            "buttons_raw": list(self.buttons_raw),
        }

    def rumble(self,low_frequency=0.0,high_frequency=1.0,duration=200):
        """
        Trigger controller rumble if supported.

        Parameters:
            low_frequency (float): Input low frequency. Defaults to `0.0`.
            high_frequency (float): Input high frequency. Defaults to `1.0`.
            duration (int): Input duration. Defaults to `200`.

        Returns:
            Any: Result computed from the current inputs.
        """
        low_frequency  = max(0.0,min(1.0,float(low_frequency)))
        high_frequency = max(0.0,min(1.0,float(high_frequency)))
        duration       = max(0,int(duration))

        if (not self.available) or (not hasattr(self.js,"rumble")):
            return False

        try:
            return bool(self.js.rumble(low_frequency,high_frequency,duration))
        except Exception:
            return False

    def get_motion_command(
            self,
            update_first   = True,
            th             = 0.1,
            xyz_scale      = 0.02,
            rpy_scale      = 1.0,
            trigger_center = 1.0,
        ):
        """
        Convert current gamepad input to translation and rotation commands.

        Parameters:
            update_first (bool): Input update first. Defaults to `True`.
            th (float): Input th. Defaults to `0.1`.
            xyz_scale (float): Input xyz scale. Defaults to `0.02`.
            rpy_scale (float): Input rpy scale. Defaults to `1.0`.
            trigger_center (float): Input trigger center. Defaults to `1.0`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        state = self.get_state(update_first=update_first)

        dx   = -xyz_scale*deadzone(state["stick_left_ud"],th=th,center=0.0,rescale=True)
        dy   = -xyz_scale*deadzone(state["stick_left_lr"],th=th,center=0.0,rescale=True)
        dz   = -xyz_scale*deadzone(state["stick_right_ud"],th=th,center=0.0,rescale=True)
        p_tf = np.array([dx,dy,dz],dtype=np.float64)

        if state.get("button_face_left",0):
            self.rot_dir = "roll"
        elif state.get("button_face_down",0):
            self.rot_dir = "pitch"
        elif state.get("button_face_right",0):
            self.rot_dir = "yaw"

        ddeg = rpy_scale*(
            - deadzone(state["trigger_top_left"],th=th,center=trigger_center,rescale=True)
            + deadzone(state["trigger_top_right"],th=th,center=trigger_center,rescale=True)
        )

        if self.rot_dir == "roll":
            drpy_deg = np.array([ddeg,0.0,0.0],dtype=np.float64)
        elif self.rot_dir == "pitch":
            drpy_deg = np.array([0.0,ddeg,0.0],dtype=np.float64)
        elif self.rot_dir == "yaw":
            drpy_deg = np.array([0.0,0.0,ddeg],dtype=np.float64)
        else:
            raise ValueError("Invalid rot_dir:[%s]"%(self.rot_dir))

        R_tf = rpy2r(drpy_deg,unit="deg")

        cmd = {
            "state"       : state,
            "p_tf"        : p_tf,
            "ddeg"        : float(ddeg),
            "drpy_deg"    : drpy_deg,
            "R_tf"        : R_tf,
            "rot_dir"     : self.rot_dir,
            "th"          : float(th),
            "xyz_scale"   : float(xyz_scale),
            "rpy_scale"   : float(rpy_scale),
            "trigger_center": float(trigger_center),
        }
        return cmd

    def update_T_trgt(
            self,
            T_curr,
            dt                = 1.0/50.0,
            update_first      = True,
            th                = 0.1,
            xyz_scale         = 1.0,
            rpy_scale         = 50.0,
            trigger_center    = 1.0,
            translation_frame = "world",
            rotation_frame    = "local",
            return_cmd        = False,
        ):
        """
        Update a target transform using the current gamepad state.

        Parameters:
            T_curr (Any): Input T curr.
            dt (float): Input dt. Defaults to `1.0 / 50.0`.
            update_first (bool): Input update first. Defaults to `True`.
            th (float): Input th. Defaults to `0.1`.
            xyz_scale (float): Input xyz scale. Defaults to `1.0`.
            rpy_scale (float): Input rpy scale. Defaults to `50.0`.
            trigger_center (float): Input trigger center. Defaults to `1.0`.
            translation_frame (str): Input translation frame. Defaults to `'world'`.
            rotation_frame (str): Input rotation frame. Defaults to `'local'`.
            return_cmd (bool): Input return cmd. Defaults to `False`.

        Returns:
            Any: Result computed from the current inputs.
        """
        cmd = self.get_motion_command(
            update_first   = update_first,
            th             = th,
            xyz_scale      = xyz_scale*dt,
            rpy_scale      = rpy_scale*dt,
            trigger_center = trigger_center,
        )

        p_curr = t2p(T_curr).copy()
        R_curr = t2r(T_curr).copy()

        if translation_frame == "world":
            p_next = p_curr + cmd["p_tf"]
        elif translation_frame == "local":
            p_next = p_curr + R_curr@cmd["p_tf"]
        else:
            raise ValueError("Invalid translation_frame:[%s]"%(translation_frame))

        if rotation_frame == "local":
            R_next = R_curr@cmd["R_tf"]
        elif rotation_frame == "world":
            R_next = cmd["R_tf"]@R_curr
        else:
            raise ValueError("Invalid rotation_frame:[%s]"%(rotation_frame))

        T_next = pr2t(p_next,R_next)

        if return_cmd:
            cmd["dt"]     = float(dt)
            cmd["p_curr"] = p_curr.copy()
            cmd["R_curr"] = R_curr.copy()
            cmd["p_next"] = p_next.copy()
            cmd["R_next"] = R_next.copy()
            cmd["T_next"] = T_next.copy()
            return T_next,cmd
        else:
            return T_next

    def print_info(
            self,
            env,
            print_info = True,
            print_raw  = True,
            decimals   = 3,
        ):
        """
        Print processed and/or raw controller state to the viewer overlay.

        Parameters:
            env (Any): Input env.
            print_info (bool): Input print info. Defaults to `True`.
            print_raw (bool): Input print raw. Defaults to `True`.
            decimals (int): Input decimals. Defaults to `3`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        fmt_float = "[%%.%df]"%(decimals)

        if not self.available:
            env.viewer_text_overlay("vader5",self.get_status_msg())
            return

        if print_info:
            state = self.get_state(update_first=True)
            for key,value in state.items():
                if value is None:
                    value_str = "[None]"
                elif isinstance(value,float):
                    value_str = fmt_float%(value)
                else:
                    value_str = "[%d]"%(value)
                env.viewer_text_overlay(key,value_str)
            env.viewer_text_overlay("rot_dir","[%s]"%(self.rot_dir))

        if print_raw:
            for a_idx,value in enumerate(self.axes_raw):
                env.viewer_text_overlay("axis:[%d]"%(a_idx),fmt_float%(value))
            for b_idx,value in enumerate(self.buttons_raw):
                env.viewer_text_overlay("button:[%d]"%(b_idx),"[%d]"%(value))

    def close(self):
        """
        Close the joystick handle.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if self.js is None:
            return
        try:
            self.js.quit()
        except Exception:
            pass

    def plot_T_trgt(
            self,
            env,
            T_trgt,
            axis_len        = 0.2,
            axis_width      = 0.01,
            show_pos_label  = True,
            label           = None,
            arrow_len       = 0.2,
            arrow_r         = 0.03,
            arrow_alpha     = 0.2,
        ):
        """
        Plot the target transform and the currently selected rotation axis.

        Parameters:
            env (Any): Input env.
            T_trgt (Any): Input T trgt.
            axis_len (float): Input axis len. Defaults to `0.2`.
            axis_width (float): Input axis width. Defaults to `0.01`.
            show_pos_label (bool): Input show pos label. Defaults to `True`.
            label (Any | None): Input label. Defaults to `None`.
            arrow_len (float): Input arrow len. Defaults to `0.2`.
            arrow_r (float): Input arrow r. Defaults to `0.03`.
            arrow_alpha (float): Input arrow alpha. Defaults to `0.2`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        p = t2p(T_trgt)

        if label is None and show_pos_label:
            label = "(%.2f,%.2f,%.2f)"%(p[0],p[1],p[2])
        elif label is None:
            label = ""

        env.plot_T(
            T          = T_trgt,
            axis_len   = axis_len,
            axis_width = axis_width,
            label      = label,
        )

        if self.rot_dir == "roll":
            p_to = t2x(T_trgt,aw=arrow_len)
            rgba = (1,0,0,arrow_alpha)
        elif self.rot_dir == "pitch":
            p_to = t2y(T_trgt,aw=arrow_len)
            rgba = (0,1,0,arrow_alpha)
        elif self.rot_dir == "yaw":
            p_to = t2z(T_trgt,aw=arrow_len)
            rgba = (0,0,1,arrow_alpha)
        else:
            return

        env.plot_arrow_fr2to(
            p_fr = p,
            p_to = p_to,
            r    = arrow_r,
            rgba = rgba,
        )

    @staticmethod
    def wrap_deg(deg):
        """
        Wrap angle in degrees to [-180,180).

        Parameters:
            deg (Any): Input deg.

        Returns:
            Any: Result computed from the current inputs.
        """
        return ((float(deg)+180.0)%360.0)-180.0

    def update_viewer_cam(
            self,
            env,
            dt,
            update_first     = True,
            azim_speed       = 30.0,
            elev_speed       = 30.0,
            elev_min         = -89.0,
            elev_max         = 89.0,
        ):
        """
        Update viewer camera using D-pad buttons.

        Parameters:
            env (Any): Input env.
            dt (float): Input dt.
            update_first (bool): Input update first. Defaults to `True`.
            azim_speed (float): Input azim speed. Defaults to `30.0`.
            elev_speed (float): Input elev speed. Defaults to `30.0`.
            elev_min (Any): Input elev min. Defaults to `-89.0`.
            elev_max (float): Input elev max. Defaults to `89.0`.

        Returns:
            Any: Result computed from the current inputs.
        """
        state = self.get_state(update_first=update_first)

        azimuth,distance,elevation,lookat = env.viewer.get_cam_info()

        # D-pad left/right -> azimuth
        dazim = azim_speed*dt*(
            - state["button_dpad_left"]
            + state["button_dpad_right"]
        )

        # D-pad up/down -> elevation
        delev = elev_speed*dt*(
            - state["button_dpad_up"]
            + state["button_dpad_down"]
        )

        azimuth   = self.wrap_deg(azimuth+dazim)
        elevation = np.clip(elevation+delev,elev_min,elev_max)

        env.viewer.set_cam_info(
            azimuth   = azimuth,
            elevation = elevation,
        )
        return azimuth,elevation
