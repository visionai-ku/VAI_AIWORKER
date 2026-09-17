"""
env_widgets.py
"""

from __future__ import annotations
import numpy as np

from .qt_widgets import MultiSliderQtWidget
from ri_motion_v5_package.kinematics.transforms import (
    prpy2t,
    get_T_global_indep,
    t2prpy,
)
from ri_motion_v5_package.utility.utils import (
    get_idxs,
)
from ri_motion_v5_package.mujoco_sim.mujoco_utils import (
    get_label_colors_from_joint_axis,
)

class EnvPoseSlider(object):
    """
    A widget for controlling the pose of an object in the environment.

    Usage:
        obj = EnvPoseSlider(...)
    """
    def __init__(
            self,
            env,
            target_name,
            x_offset      = 0.0,
            y_offset      = 0.05,
            window_width  = 0.2,
            window_height = 0.15,
            T_0           = None,
            title         = None,
            get_type      = "body",
            set_type      = "base_body",
            slider_range  = [0.5,0.5,0.5,np.pi,np.pi,np.pi],
        ):
        """
        Initialize EnvPoseSlider.

        Parameters:
            env (Any): Input env.
            target_name (str): Input target name.
            x_offset (float): Input x offset. Defaults to `0.0`.
            y_offset (float): Input y offset. Defaults to `0.05`.
            window_width (float): Input window width. Defaults to `0.2`.
            window_height (float): Input window height. Defaults to `0.15`.
            T_0 (Any | None): Input T 0. Defaults to `None`.
            title (Any | None): Input title. Defaults to `None`.
            get_type (str): Input get type. Defaults to `'body'`.
            set_type (str): Input set type. Defaults to `'base_body'`.
            slider_range (list): Input slider range. Defaults to `[0.5, 0.5, 0.5,
                                 np.pi, np.pi, np.pi]`.

        Returns:
            None: Initializes the instance in place.
        """
        self.env          = env
        self.target_name  = target_name
        self.get_type     = get_type
        self.set_type     = set_type
        self.slider_range = np.array(slider_range)

        if title is None:
            title = "%s pose"%(target_name)

        if T_0 is None:
            T_0 = env.get_T(target_name,get_type)
        self.T_0 = T_0.copy()

        self.sliders = MultiSliderQtWidget(
            title         = title,
            x_offset      = x_offset,
            y_offset      = y_offset,
            window_width  = window_width,
            window_height = window_height,
            label_texts   = ["dX","dY","dZ","dRoll","dPitch","dYaw"],
            slider_mins   = -self.slider_range,
            slider_maxs   = +self.slider_range,
            slider_vals   = np.zeros(6),
        )

    def update(self,update_env=True,return_T=False):
        """
        Update the pose of the target object in the environment based on the current
        slider values.

        Parameters:
            update_env (bool): Input update env. Defaults to `True`.
            return_T (bool): Input return T. Defaults to `False`.

        Returns:
            Any: Result computed from the current inputs.
        """
        dT = prpy2t(self.sliders.get_values())
        self.T  = get_T_global_indep(self.T_0,dT)
        if update_env:
            self.env.set_T(name=self.target_name,type=self.set_type,T=self.T)
        if return_T:
            return self.T

    def get_T(self):
        """
        Get the current pose of the target object in the environment.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        return self.env.get_T(self.target_name,self.get_type)

    def plot_T(self,axis_len=0.1,axis_alpha=0.5,show_label=False):
        """
        Plot the current pose of the target object in the environment.

        Parameters:
            axis_len (float): Input axis len. Defaults to `0.1`.
            axis_alpha (float): Input axis alpha. Defaults to `0.5`.
            show_label (bool): Input show label. Defaults to `False`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        T = self.get_T()
        label = ""
        if show_label:
            label = self.target_name
        else:
            label = ""
        self.env.plot_T(T=T,axis_len=axis_len,axis_alpha=axis_alpha,label=label)

    def close(self):
        """
        Close the widget and clean up any resources.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.sliders.close()

class EnvJointSlider(object):
    """
    A widget for controlling the joint angles of a robot in the environment.

    Usage:
        obj = EnvJointSlider(...)
    """
    def __init__(
            self,
            env,
            joint_names,
            x_offset      = 0.8,
            y_offset      = 0.05,
            window_width  = 0.2,
            window_height = 0.3,
            title         = None,
            label_colors  = None,
            slider_colors = None,
            color_by_axis = True,
            bright_axis_colors = True,
        ):
        """
        Initialize EnvJointSlider.

        Parameters:
            env (Any): Input env.
            joint_names (list[str]): Input joint names.
            x_offset (float): Input x offset. Defaults to `0.8`.
            y_offset (float): Input y offset. Defaults to `0.05`.
            window_width (float): Input window width. Defaults to `0.2`.
            window_height (float): Input window height. Defaults to `0.3`.
            title (Any | None): Input title. Defaults to `None`.
            label_colors (Any | None): Input label colors. Defaults to `None`.
            slider_colors (Any | None): Input slider colors. Defaults to `None`.
            color_by_axis (bool): Input color by axis. Defaults to `True`.
            bright_axis_colors (bool): Input bright axis colors. Defaults to `True`.

        Returns:
            None: Initializes the instance in place.
        """
        self.env = env

        # The list of joint names to control with the slider.
        # The widget will automatically filter out any joints that are not active in the environment.
        self.joint_names = list(joint_names)
        self.active_joint_names = env.get_active_among_joints(self.joint_names)

        if len(self.active_joint_names) == 0:
            raise ValueError("No active joints found in joint_names:[%s]"%(self.joint_names))

        if title is None:
            title = "Joint Slider"

        if color_by_axis:
            axis_colors = get_label_colors_from_joint_axis(
                env,
                self.active_joint_names,
                bright_colors=bright_axis_colors,
            )
            if label_colors is None:
                label_colors = axis_colors
            if slider_colors is None:
                slider_colors = axis_colors

        self.sliders = MultiSliderQtWidget(
            title         = title,
            x_offset      = x_offset,
            y_offset      = y_offset,
            window_width  = window_width,
            window_height = window_height,
            label_texts   = self.active_joint_names,
            label_colors  = label_colors,
            slider_colors = slider_colors,
            slider_mins   = env.get_joint_mins(self.active_joint_names),
            slider_maxs   = env.get_joint_maxs(self.active_joint_names),
            slider_vals   = env.get_qpos(self.active_joint_names),
        )

    def get_qactive(self):
        """
        Get the current joint configuration of the active joints based on the slider
        values.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        return np.array(self.sliders.get_values(),dtype=float)

    def get_qfull(self):
        """
        Get the full joint configuration of the robot in the environment based on the
        current slider values for the active joints.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        qactive = self.get_qactive()
        qfull = self.env.get_qfull_from_qactive(self.joint_names,self.active_joint_names,qactive)
        return qfull

    def update(self,update_env=True,return_qfull=False):
        """
        Update the joint angles of the robot in the environment based on the current
        slider values.

        Parameters:
            update_env (bool): Input update env. Defaults to `True`.
            return_qfull (bool): Input return qfull. Defaults to `False`.

        Returns:
            Any: Result computed from the current inputs.
        """
        qfull = self.get_qfull()
        if update_env:
            self.env.forward(q=qfull,joint_names=self.joint_names)
        if return_qfull:
            return qfull

    def close(self):
        """
        Close the widget and release Qt resources.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.sliders.close()

class EnvDeltaPoseSlider(object):
    """
    A widget for controlling the delta pose of an object in the environment.

    Usage:
        obj = EnvDeltaPoseSlider(...)
    """
    def __init__(
            self,
            env,
            x_offset       = 0.8,
            y_offset       = 0.05,
            window_width   = 0.2,
            window_height  = 0.15,
            ref_name       = None,
            ref_type       = "body",
            T_0            = None,
            title          = None,
            slider_range   = [0.5,0.5,0.5,np.pi,np.pi,np.pi],
        ):
        """
        Initialize EnvDeltaPoseSlider.

        Parameters:
            env (Any): Input env.
            x_offset (float): Input x offset. Defaults to `0.8`.
            y_offset (float): Input y offset. Defaults to `0.05`.
            window_width (float): Input window width. Defaults to `0.2`.
            window_height (float): Input window height. Defaults to `0.15`.
            ref_name (Any | None): Input ref name. Defaults to `None`.
            ref_type (str): Input ref type. Defaults to `'body'`.
            T_0 (Any | None): Input T 0. Defaults to `None`.
            title (Any | None): Input title. Defaults to `None`.
            slider_range (list): Input slider range. Defaults to `[0.5, 0.5, 0.5,
                                 np.pi, np.pi, np.pi]`.

        Returns:
            None: Initializes the instance in place.
        """
        self.env          = env
        self.ref_name     = ref_name
        self.ref_type     = ref_type
        self.slider_range = np.asarray(slider_range,dtype=float)

        # Set reference pose
        if T_0 is None:
            if ref_name is None:
                raise ValueError("Either T_0 or ref_name must be provided.")
            T_0 = self.env.get_T(ref_name,ref_type)

        self.T_0 = np.asarray(T_0,dtype=float).copy()
        self.dT  = np.eye(4)
        self.T   = self.T_0.copy()

        # Title
        if title is None:
            if ref_name is None:
                title = "Delta Pose"
            else:
                title = "%s delta pose"%(ref_name)

        # Slider widget
        self.sliders = MultiSliderQtWidget(
            title         = title,
            x_offset      = x_offset,
            y_offset      = y_offset,
            window_width  = window_width,
            window_height = window_height,
            label_texts   = ["dX","dY","dZ","dRoll","dPitch","dYaw"],
            slider_mins   = -self.slider_range,
            slider_maxs   = +self.slider_range,
            slider_vals   = np.zeros(6,dtype=float),
        )

    def get_dpose(self):
        """
        Get raw slider values: [dX,dY,dZ,dRoll,dPitch,dYaw].

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        return np.asarray(self.sliders.get_values(),dtype=float)

    def get_dT(self):
        """
        Get delta transform from current slider values.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        self.dT = prpy2t(self.get_dpose())
        return self.dT

    def get_T(self):
        """
        Get current target transform:.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        self.T = get_T_global_indep(self.T_0,self.get_dT())
        return self.T

    def reset_T_0(self,T_0=None,ref_name=None,ref_type=None):
        """
        Reset reference pose T_0.

        Parameters:
            T_0 (Any | None): Input T 0. Defaults to `None`.
            ref_name (Any | None): Input ref name. Defaults to `None`.
            ref_type (Any | None): Input ref type. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if T_0 is None:
            if ref_name is None:
                ref_name = self.ref_name
            if ref_type is None:
                ref_type = self.ref_type
            if ref_name is None:
                raise ValueError("ref_name is required when T_0 is not given.")
            T_0 = self.env.get_T(ref_name,ref_type)

        self.T_0 = np.asarray(T_0,dtype=float).copy()
        self.T   = self.T_0.copy()

    def close(self):
        """
        Close widget.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.sliders.close()

class EnvDeltaPositionSlider(object):
    """
    A widget for controlling the delta position of a target point.

    Usage:
        obj = EnvDeltaPositionSlider(...)
    """
    def __init__(
            self,
            env,
            x_offset      = 0.8,
            y_offset      = 0.05,
            window_width  = 0.2,
            window_height = 0.10,
            p_0           = None,
            title         = None,
            slider_range  = [0.5,0.5,0.5],
        ):
        """
        Initialize EnvDeltaPositionSlider.

        Parameters:
            env (Any): Input env.
            x_offset (float): Input x offset. Defaults to `0.8`.
            y_offset (float): Input y offset. Defaults to `0.05`.
            window_width (float): Input window width. Defaults to `0.2`.
            window_height (float): Input window height. Defaults to `0.1`.
            p_0 (Any | None): Input p 0. Defaults to `None`.
            title (Any | None): Input title. Defaults to `None`.
            slider_range (list): Input slider range. Defaults to `[0.5, 0.5, 0.5]`.

        Returns:
            None: Initializes the instance in place.
        """
        self.env          = env
        self.slider_range = np.asarray(slider_range,dtype=float).reshape(3,)

        if p_0 is None:
            raise ValueError("p_0 is required.")
        self.p_0 = np.asarray(p_0,dtype=float).reshape(3,)
        self.p   = self.p_0.copy()

        if title is None:
            title = "Delta Position"

        self.sliders = MultiSliderQtWidget(
            title         = title,
            x_offset      = x_offset,
            y_offset      = y_offset,
            window_width  = window_width,
            window_height = window_height,
            label_texts   = ["dX","dY","dZ"],
            slider_mins   = -self.slider_range,
            slider_maxs   = +self.slider_range,
            slider_vals   = np.zeros(3,dtype=float),
        )

    def get_dpos(self):
        """
        Get raw slider values: [dX,dY,dZ].

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        return np.asarray(self.sliders.get_values(),dtype=float)

    def get_p(self):
        """
        Get current target position:.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        self.p = self.p_0 + self.get_dpos()
        return self.p

    def reset_p_0(self,p_0=None):
        """
        Reset reference position p_0.

        Parameters:
            p_0 (Any | None): Input p 0. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if p_0 is None:
            raise ValueError("p_0 is required.")
        self.p_0 = np.asarray(p_0,dtype=float).reshape(3,)
        self.p   = self.p_0.copy()

    def close(self):
        """
        Close widget.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.sliders.close()

class JointPositionMonitorQtWidget(object):
    """
    A read-mostly widget for monitoring joint positions.

    Usage:
        obj = JointPositionMonitorQtWidget(...)
    """
    def __init__(
            self,
            env,
            x_offset        = 0.8,
            y_offset        = 0.05,
            window_width    = 0.2,
            window_height   = 0.30,
            joint_names     = None,
            title           = None,
            label_colors    = None,
            slider_mins     = None,
            slider_maxs     = None,
            include_free_root_pose = False,
            free_root_body_names   = None,
            root_pose_xyz_range    = 5.0,
            root_pose_rpy_range    = np.pi,
        ):
        """
        Initialize JointPositionMonitorQtWidget.

        Parameters:
            env (Any): Input env.
            x_offset (float): Input x offset. Defaults to `0.8`.
            y_offset (float): Input y offset. Defaults to `0.05`.
            window_width (float): Input window width. Defaults to `0.2`.
            window_height (float): Input window height. Defaults to `0.3`.
            joint_names (Any | None): Input joint names. Defaults to `None`.
            title (Any | None): Input title. Defaults to `None`.
            label_colors (Any | None): Input label colors. Defaults to `None`.
            slider_mins (Any | None): Input slider mins. Defaults to `None`.
            slider_maxs (Any | None): Input slider maxs. Defaults to `None`.
            include_free_root_pose (bool): Input include free root pose. Defaults to
                                           `False`.
            free_root_body_names (Any | None): Input free root body names. Defaults to
                                               `None`.
            root_pose_xyz_range (float): Input root pose xyz range. Defaults to `5.0`.
            root_pose_rpy_range (str | Path): Input root pose rpy range. Defaults to
                                              `np.pi`.

        Returns:
            None: Initializes the instance in place.
        """
        self.env = env

        if joint_names is None:
            joint_names = env.rev_pri_joint_names
        self.joint_names = list(joint_names)
        self.include_free_root_pose = bool(include_free_root_pose or (free_root_body_names is not None))

        if title is None:
            title = "Joint position monitor"

        if not self.include_free_root_pose:
            self.free_root_body_names = []
            self.root_pose_label_texts = []
            self.label_texts = self.joint_names

            if label_colors is None:
                label_colors = get_label_colors_from_joint_axis(env,self.joint_names)
            if slider_mins is None:
                slider_mins = env.get_joint_mins(self.joint_names)
            if slider_maxs is None:
                slider_maxs = env.get_joint_maxs(self.joint_names)

            slider_vals = env.get_qpos(joint_names=self.joint_names)
        else:
            if free_root_body_names is None:
                free_root_body_names = self._get_default_free_root_body_names()
            self.free_root_body_names = list(free_root_body_names)
            self.root_pose_label_texts = self._get_root_pose_label_texts(self.free_root_body_names)
            self.label_texts = self.joint_names + self.root_pose_label_texts

            if label_colors is None:
                label_colors = (
                    get_label_colors_from_joint_axis(env,self.joint_names)
                    + self._get_root_pose_label_colors(self.free_root_body_names)
                )
            if slider_mins is None:
                slider_mins = np.concatenate([
                    env.get_joint_mins(self.joint_names),
                    self._get_root_pose_slider_mins(self.free_root_body_names,root_pose_xyz_range,root_pose_rpy_range),
                ])
            if slider_maxs is None:
                slider_maxs = np.concatenate([
                    env.get_joint_maxs(self.joint_names),
                    self._get_root_pose_slider_maxs(self.free_root_body_names,root_pose_xyz_range,root_pose_rpy_range),
                ])

            slider_vals = self.get_values_from_env(env)

        self.sliders = MultiSliderQtWidget(
            title         = title,
            x_offset      = x_offset,
            y_offset      = y_offset,
            window_width  = window_width,
            window_height = window_height,
            label_texts   = self.label_texts,
            label_colors  = label_colors,
            slider_vals   = slider_vals,
            slider_mins   = slider_mins,
            slider_maxs   = slider_maxs,
        )

    def _get_default_free_root_body_names(self):
        """
        Get body names attached to free joints.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        body_names = []
        for free_joint_name in getattr(self.env,"free_joint_names",[]):
            body_id = int(self.env.model.joint(free_joint_name).bodyid[0])
            body_names.append(self.env.body_names[body_id])
        return body_names

    @staticmethod
    def _get_root_pose_label_texts(body_names):
        """
        Build flat labels for root pose channels.

        Parameters:
            body_names (list[str]): Input body names.

        Returns:
            Any: Result computed from the current inputs.
        """
        label_texts = []
        for body_name in body_names:
            label_texts += [
                f"{body_name}_x",
                f"{body_name}_y",
                f"{body_name}_z",
                f"{body_name}_roll",
                f"{body_name}_pitch",
                f"{body_name}_yaw",
            ]
        return label_texts

    @staticmethod
    def _get_root_pose_label_colors(body_names):
        """
        Use XYZ/RPY color convention for free-body root pose channels.

        Parameters:
            body_names (list[str]): Input body names.

        Returns:
            Any: Result computed from the current inputs.
        """
        label_colors = []
        for _ in body_names:
            label_colors += ["#FF6666","#66FF66","#66CCFF","#FF6666","#66FF66","#66CCFF"]
        return label_colors

    @staticmethod
    def _get_root_pose_slider_mins(body_names,xyz_range,rpy_range):
        """
        Build monitor ranges for root pose channels.

        Parameters:
            body_names (list[str]): Input body names.
            xyz_range (Any): Input xyz range.
            rpy_range (Any): Input rpy range.

        Returns:
            Any: Result computed from the current inputs.
        """
        if len(body_names) == 0:
            return np.array([],dtype=float)
        xyz_range = np.broadcast_to(np.asarray(xyz_range,dtype=float),3)
        rpy_range = np.broadcast_to(np.asarray(rpy_range,dtype=float),3)
        one_root = np.concatenate([-xyz_range,-rpy_range])
        return np.tile(one_root,len(body_names))

    @staticmethod
    def _get_root_pose_slider_maxs(body_names,xyz_range,rpy_range):
        """
        Build monitor ranges for root pose channels.

        Parameters:
            body_names (list[str]): Input body names.
            xyz_range (Any): Input xyz range.
            rpy_range (Any): Input rpy range.

        Returns:
            Any: Result computed from the current inputs.
        """
        if len(body_names) == 0:
            return np.array([],dtype=float)
        xyz_range = np.broadcast_to(np.asarray(xyz_range,dtype=float),3)
        rpy_range = np.broadcast_to(np.asarray(rpy_range,dtype=float),3)
        one_root = np.concatenate([xyz_range,rpy_range])
        return np.tile(one_root,len(body_names))

    def get_values_from_env(self,env=None):
        """
        Get displayed values from env in the widget's label order.

        Parameters:
            env (Any | None): Input env. Defaults to `None`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if env is None:
            env = self.env

        values = [env.get_qpos(joint_names=self.joint_names)]
        for body_name in self.free_root_body_names:
            values.append(t2prpy(env.get_T(body_name,"body")))
        return np.concatenate(values) if values else np.array([],dtype=float)

    def set_values(self,values):
        """
        Update displayed joint positions.

        Parameters:
            values (Any): Input values.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        values = np.asarray(values,dtype=float).reshape(-1,)

        if not self.include_free_root_pose:
            self.sliders.set_values(values)
            return

        if len(values) == len(self.label_texts):
            self.sliders.set_values(values)
            return

        if len(values) == len(self.joint_names):
            root_pose_values = []
            for body_name in self.free_root_body_names:
                root_pose_values.append(t2prpy(self.env.get_T(body_name,"body")))
            if root_pose_values:
                values = np.concatenate([values] + root_pose_values)
            self.sliders.set_values(values)
            return

        raise ValueError(
            "values length (%d) must match joint count (%d) or total monitor count (%d)." % (
                len(values),len(self.joint_names),len(self.label_texts)
            )
        )

    def set_values_from_env(self,env):
        """
        Update displayed joint positions parsed from env.

        Parameters:
            env (Any): Input env.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.set_values(env.get_qpos(joint_names=self.joint_names))

    def close(self):
        """
        Close widget.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.sliders.close()

class EnvCtrlSlider(object):
    """
    A widget for controlling actuator values in the environment.

    Usage:
        obj = EnvCtrlSlider(...)
    """

    def __init__(
            self,
            env,
            ctrl_names,
            x_offset      = 0.8,
            y_offset      = 0.05,
            window_width  = 0.2,
            window_height = 0.3,
            title         = None,
        ):
        """
        Initialize EnvCtrlSlider.

        Parameters:
            env (Any): Input env.
            ctrl_names (list[str]): Input ctrl names.
            x_offset (float): Input x offset. Defaults to `0.8`.
            y_offset (float): Input y offset. Defaults to `0.05`.
            window_width (float): Input window width. Defaults to `0.2`.
            window_height (float): Input window height. Defaults to `0.3`.
            title (Any | None): Input title. Defaults to `None`.

        Returns:
            None: Initializes the instance in place.
        """
        self.env        = env
        self.ctrl_names = list(ctrl_names)
        self.ctrl_idxs  = get_idxs(env.ctrl_names,self.ctrl_names)

        if len(self.ctrl_names) == 0:
            raise ValueError("No ctrl names found.")

        if title is None:
            title = "Ctrl Slider"

        self.sliders = MultiSliderQtWidget(
            title         = title,
            x_offset      = x_offset,
            y_offset      = y_offset,
            window_width  = window_width,
            window_height = window_height,
            label_texts   = self.ctrl_names,
            slider_vals   = env.get_ctrl(self.ctrl_names),
            slider_mins   = env.ctrl_mins[self.ctrl_idxs],
            slider_maxs   = env.ctrl_maxs[self.ctrl_idxs],
        )

    def get_ctrl(self):
        """
        Get the current control values from the slider.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        return np.array(self.sliders.get_values(),dtype=np.float64)

    def close(self):
        """
        Close the widget and clean up any resources.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.sliders.close()

# -----------------------------------------------------------------------------
