"""In-viewer joint controller for FFW SG2.

`joint_control.py` drives the joints from Qt slider windows sitting next to the
viewer. This module puts the control back inside the 3D scene: every commandable
joint carries its own InteractiveMarker anchored at the joint itself -- a ring
around each hinge axis, an arrow along the lift slide -- so a joint is grabbed
where it actually lives, the same way `sim_interactive_marker_wrist_ik_test.ipynb`
grabs a wrist.

Still no IK anywhere: dragging a handle writes that one joint's qpos and nothing
else. The wrist marker asks "put the hand here, solve for the arm"; these markers
ask "turn this joint by this much".

Each marker is locked to a single local axis (Z, aligned with the joint axis), so
the two off-axis rings of a normal rotation marker are neither drawn nor
clickable. Between drags the markers are re-anchored to their joints every tick,
which is what keeps a child joint's ring glued to its link while a parent moves.
"""

from __future__ import annotations

import numpy as np

from ri_motion_v5_package.kinematics import np_uv, pr2t, t2p, t2r
from ri_motion_v5_package.mujoco_sim import InteractiveMarker
from ri_motion_v5_package.qt import MultiRadioQtWidget

from utils import (
    SG2_GRIPPER_DRIVER_JOINT_NAMES,
    SG2_HAND_TYPES,
    set_sg2_gripper_qpos,
)
from VAI_AIWORKER.ri_motion_v5_VR.project.ffw_sg2_vr_teleoperation.package.joint_control import (
    SG2_JOINT_COMMAND_ROWS,
    SG2_JOINT_PRESETS,
    SG2_JOINT_QPOS_PATH,
    SG2_SLIDER_JOINT_GROUPS,
    handle_sg2_joint_command_widget,
)

SG2_JOINT_MARKER_GROUPS = dict(SG2_SLIDER_JOINT_GROUPS)
SG2_JOINT_MARKER_GROUP_COLORS = {
    "arm_l": (0.20, 0.55, 1.00, 0.95),
    "arm_r": (1.00, 0.45, 0.15, 0.95),
    "body": (0.35, 0.85, 0.35, 0.95),
}

SG2_JOINT_MARKER_COMMAND_ROWS = dict(SG2_JOINT_COMMAND_ROWS)
SG2_JOINT_MARKER_COMMAND_ROWS["Show"] = ["all"] + list(SG2_JOINT_MARKER_GROUPS)
SG2_JOINT_MARKER_COMMAND_LAYOUT = (0.00, 0.00, 0.22, 0.30)


def get_R_from_z_axis(axis):
    """Orthonormal frame whose local Z is `axis`; X and Y are an arbitrary but
    deterministic completion, which is all the ring needs to be drawn."""
    z_axis = np_uv(np.asarray(axis, dtype=np.float64).reshape(3))
    reference = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np_uv(np.cross(reference, z_axis))
    return np.column_stack([x_axis, np.cross(z_axis, x_axis), z_axis])


class SG2JointAxisMarker(InteractiveMarker):
    """InteractiveMarker restricted to one local axis: a ring for a hinge joint, an
    arrow for a slide joint.

    The base class always projects and hit-tests all three axes; the two overrides
    below drop the off-axis handles from hover and draw only the one that matters,
    so 19 of these can share a scene without burying the robot in gizmos.
    """

    def __init__(
            self,
            env,
            joint_kind="hinge",
            handle_rgba=(1.0, 0.6, 0.0, 0.95),
            ring_segs=24,
            **kwargs,
        ):
        if joint_kind not in ("hinge", "slide"):
            raise ValueError("joint_kind must be 'hinge' or 'slide'.")
        super().__init__(
            env=env,
            control_mode="R" if joint_kind == "hinge" else "p",
            translation_frame="local",
            show_rotation_reset=False,
            show_frame_toggle=False,
            ring_segs=ring_segs,
            **kwargs,
        )
        self.joint_kind = joint_kind
        self.handle_rgba = tuple(handle_rgba)

    @property
    def is_active(self):
        return (self.drag_mode or self.hover_mode) is not None

    @property
    def drag_delta(self):
        """Signed motion the live drag has accumulated along the marker axis:
        radians for a hinge, meters for a slide. 0.0 when nothing is being dragged."""
        if self.drag_mode is None:
            return 0.0
        if self.joint_kind == "hinge":
            return float(self.rot_sign * self.rot_cum_angle)
        return float((t2p(self.T) - self._drag["p"]) @ self._drag["axis"])

    def _update_hover(self, active_set):
        super()._update_hover(active_set)
        # Keep only the local-Z handle: no plane drags, no off-axis rings or arrows.
        if self.hover_mode in ("translate", "rotate") and self.hover_axis != 2:
            self.hover_mode = self.hover_axis = None
        elif self.hover_mode == "plane":
            self.hover_mode = self.hover_axis = None

    def _draw(self, alpha=None):
        p, R = t2p(self.T), t2r(self.T)
        rgba = self._rgba_alpha(self.hi_color if self.is_active else self.handle_rgba, alpha)
        hover_scale = self.hover_scale if self.is_active else 1.0
        if self.joint_kind == "hinge":
            ring = self._ring_pts(p, R, 2)
            for idx in range(len(ring)):
                self.env.plot_cylinder_fr2to(
                    p_fr=ring[idx],
                    p_to=ring[(idx + 1) % len(ring)],
                    r=self.ring_thick * hover_scale,
                    rgba=rgba,
                )
        else:
            self.env.plot_arrow_fr2to(
                p_fr=p - R[:, 2] * self.axis_len,
                p_to=p + R[:, 2] * self.axis_len,
                r=self.arrow_r * hover_scale,
                rgba=rgba,
            )


class SG2JointMarkerController(object):
    """One axis marker per commandable joint, driving the MuJoCo model directly.

    Exposes the same `joint_names` / `get_qpos` / `set_qpos` / `sync_from_env` /
    `apply_preset` surface as `SG2JointSliderPanel`, so the shared command widget
    handler in `joint_control.py` works against either controller.
    """

    def __init__(
            self,
            env,
            joint_groups=None,
            group_colors=None,
            ring_r=0.05,
            axis_len=0.12,
            ring_segs=24,
        ):
        if joint_groups is None:
            joint_groups = SG2_JOINT_MARKER_GROUPS
        if group_colors is None:
            group_colors = SG2_JOINT_MARKER_GROUP_COLORS

        self.env = env
        self.group_joint_names = {}
        self.joint_names = []
        self.group_of_joint = {}
        for group_key, joint_names in joint_groups.items():
            active_joint_names = env.get_active_among_joints(joint_names)
            if len(active_joint_names) == 0:
                continue
            self.group_joint_names[group_key] = list(active_joint_names)
            self.joint_names += list(active_joint_names)
            for joint_name in active_joint_names:
                self.group_of_joint[joint_name] = group_key

        if len(self.joint_names) == 0:
            raise ValueError("No active joints found for the SG2 joint marker controller.")

        self.joint_ids = [int(env.model.joint(joint_name).id) for joint_name in self.joint_names]
        self.joint_kinds = [
            "slide" if joint_name in set(env.pri_joint_names) else "hinge"
            for joint_name in self.joint_names
        ]
        self.q_min = env.get_joint_mins(self.joint_names)
        self.q_max = env.get_joint_maxs(self.joint_names)

        self.markers = {
            joint_name: SG2JointAxisMarker(
                env=env,
                T=self.get_joint_anchor_T(idx),
                joint_kind=self.joint_kinds[idx],
                handle_rgba=group_colors[self.group_of_joint[joint_name]],
                ring_r=ring_r,
                axis_len=axis_len,
                ring_segs=ring_segs,
            )
            for idx, joint_name in enumerate(self.joint_names)
        }
        self.drag_q0 = {joint_name: None for joint_name in self.joint_names}
        self.visible_group = "all"
        self.active_joint_name = None

    def get_joint_anchor_T(self, idx):
        """Joint frame as MuJoCo currently reports it: anchor point, local Z on the
        joint axis. Both move with the parent chain, hence the per-tick re-anchor."""
        joint_id = self.joint_ids[idx]
        return pr2t(
            p=np.array(self.env.data.xanchor[joint_id], dtype=np.float64),
            R=get_R_from_z_axis(self.env.data.xaxis[joint_id]),
        )

    def is_joint_shown(self, joint_name):
        return self.visible_group in ("all", self.group_of_joint[joint_name])

    def shown_joint_names(self):
        return [joint_name for joint_name in self.joint_names if self.is_joint_shown(joint_name)]

    def get_qpos(self):
        return self.env.get_qpos(joint_names=self.joint_names)

    def set_joint_qpos(self, idx, qpos):
        joint_name = self.joint_names[idx]
        qpos = float(np.clip(qpos, self.q_min[idx], self.q_max[idx]))
        self.env.forward(q=[qpos], joint_names=[joint_name])
        for hand_type in SG2_HAND_TYPES:
            if joint_name == SG2_GRIPPER_DRIVER_JOINT_NAMES[hand_type]:
                set_sg2_gripper_qpos(self.env, hand_type, qpos)
        return qpos

    def set_qpos(self, qpos, joint_names=None):
        if joint_names is None:
            joint_names = self.joint_names
        for joint_name, value in zip(joint_names, np.asarray(qpos, dtype=np.float64).ravel()):
            if joint_name not in self.joint_names:
                continue
            self.set_joint_qpos(self.joint_names.index(joint_name), float(value))
        self.reanchor_markers()
        return self.get_qpos()

    def sync_from_env(self):
        self.reanchor_markers()
        return self.get_qpos()

    def apply_preset(self, preset_name, presets=None):
        if presets is None:
            presets = SG2_JOINT_PRESETS
        if preset_name not in presets:
            raise ValueError(
                "Unknown preset:[%s]. Available:%s" % (preset_name, list(presets))
            )
        preset = presets[preset_name]
        return self.set_qpos(
            [float(preset.get(joint_name, 0.0)) for joint_name in self.joint_names]
        )

    def reanchor_markers(self):
        """Snap idle markers back onto their joints. A marker mid-drag is left alone:
        its pose is the drag's own state, and re-writing it would fight the drag."""
        for idx, joint_name in enumerate(self.joint_names):
            marker = self.markers[joint_name]
            if marker.drag_mode is None:
                marker.set_T(self.get_joint_anchor_T(idx), clear_state=False)

    def update(self):
        """Poll the visible markers and write whichever joint is being dragged.

        Once a drag is live only that marker is polled, so a second ring passing
        under the cursor cannot steal the press mid-drag.
        """
        dragging_joint_name = next(
            (
                joint_name
                for joint_name in self.shown_joint_names()
                if self.markers[joint_name].drag_mode is not None
            ),
            None,
        )
        self.active_joint_name = None

        for idx, joint_name in enumerate(self.joint_names):
            marker = self.markers[joint_name]
            if not self.is_joint_shown(joint_name):
                self.drag_q0[joint_name] = None
                continue
            if dragging_joint_name is not None and joint_name != dragging_joint_name:
                continue

            marker.update(draw=False)
            if marker.drag_mode is None:
                self.drag_q0[joint_name] = None
                continue
            if self.drag_q0[joint_name] is None:
                self.drag_q0[joint_name] = float(self.env.get_qpos(joint_names=[joint_name])[0])
            self.active_joint_name = joint_name
            self.set_joint_qpos(idx, self.drag_q0[joint_name] + marker.drag_delta)

        self.reanchor_markers()
        return self.get_qpos()

    def draw(self):
        for joint_name in self.shown_joint_names():
            self.markers[joint_name].draw()


def create_sg2_joint_marker_command_widget(layout=SG2_JOINT_MARKER_COMMAND_LAYOUT):
    x_offset, y_offset, window_width, window_height = layout
    return MultiRadioQtWidget(
        title="SG2 Joint Marker Command",
        x_offset=x_offset,
        y_offset=y_offset,
        window_width=window_width,
        window_height=window_height,
        label_texts=list(SG2_JOINT_MARKER_COMMAND_ROWS),
        option_texts=list(SG2_JOINT_MARKER_COMMAND_ROWS.values()),
        initial_texts=["idle", "idle", "on", "on", "all"],
    )


def handle_sg2_joint_marker_command_widget(widget, controller, qpos_path=SG2_JOINT_QPOS_PATH):
    """Same one-shot Preset/File handling as the slider notebook, plus the Show row
    that decides which joint group keeps its markers. Call once per tick."""
    command_info = handle_sg2_joint_command_widget(
        widget=widget,
        panel=controller,
        qpos_path=qpos_path,
    )
    controller.visible_group = widget.get_button_option("Show") or "all"
    command_info["show_group"] = controller.visible_group
    return command_info


def make_sg2_joint_marker_overlay_text(controller):
    """One line per joint group, hinges in degrees and the slide in meters."""
    lines = []
    for group_key, joint_names in controller.group_joint_names.items():
        entries = []
        for joint_name in joint_names:
            idx = controller.joint_names.index(joint_name)
            qpos = float(controller.env.get_qpos(joint_names=[joint_name])[0])
            if controller.joint_kinds[idx] == "hinge":
                entries.append("%.1f" % np.rad2deg(qpos))
            else:
                entries.append("%.3fm" % qpos)
        lines.append("%-6s: %s" % (group_key, " ".join(entries)))
    return "\n".join(lines)


def make_sg2_joint_marker_status_text(controller):
    if controller.active_joint_name is None:
        return "show:%s | drag a ring to turn that joint" % controller.visible_group

    joint_name = controller.active_joint_name
    idx = controller.joint_names.index(joint_name)
    qpos = float(controller.env.get_qpos(joint_names=[joint_name])[0])
    marker = controller.markers[joint_name]
    if controller.joint_kinds[idx] == "hinge":
        return "show:%s | %s %+.1f deg (drag %+.1f deg, limit %.1f~%.1f)" % (
            controller.visible_group,
            joint_name,
            np.rad2deg(qpos),
            np.rad2deg(marker.drag_delta),
            np.rad2deg(controller.q_min[idx]),
            np.rad2deg(controller.q_max[idx]),
        )
    return "show:%s | %s %+.3f m (drag %+.3f m, limit %.3f~%.3f)" % (
        controller.visible_group,
        joint_name,
        qpos,
        marker.drag_delta,
        controller.q_min[idx],
        controller.q_max[idx],
    )


__all__ = [
    "SG2JointAxisMarker",
    "SG2JointMarkerController",
    "SG2_JOINT_MARKER_COMMAND_LAYOUT",
    "SG2_JOINT_MARKER_COMMAND_ROWS",
    "SG2_JOINT_MARKER_GROUPS",
    "SG2_JOINT_MARKER_GROUP_COLORS",
    "create_sg2_joint_marker_command_widget",
    "get_R_from_z_axis",
    "handle_sg2_joint_marker_command_widget",
    "make_sg2_joint_marker_overlay_text",
    "make_sg2_joint_marker_status_text",
]
