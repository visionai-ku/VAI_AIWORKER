"""Joint-space slider control for FFW SG2.

The interactive-marker notebook is an end-effector controller: drag a wrist
gizmo and let IK find the joints. This module is its joint-space counterpart —
one slider per commandable joint, no IK in the loop — in the spirit of opening a
mujoco_menagerie scene with `python -m mujoco.viewer` and moving the model from
the control panel.

Sliders are split into three panels so 19 joints stay readable:
    arm_l : arm_l_joint1~7
    arm_r : arm_r_joint1~7
    body  : lift_joint, head_joint1~2, gripper_l_joint1, gripper_r_joint1
The gripper rows drive `gripper_?_joint1` only, exactly like ros2_control on the
real robot; `set_sg2_gripper_qpos` propagates the three follower joints for
visualization.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from ri_motion_v5_package.kinematics import t2prpy
from ri_motion_v5_package.qt import EnvJointSlider, MultiRadioQtWidget

from utils import (
    PROJECT_DIR,
    SG2_ARM_JOINT_NAMES,
    SG2_GRIPPER_DRIVER_JOINT_NAMES,
    SG2_HAND_LABELS,
    SG2_HAND_TYPES,
    get_sg2_wrist_control_T,
    set_sg2_gripper_qpos,
)

SG2_SLIDER_JOINT_GROUPS = {
    "arm_l": list(SG2_ARM_JOINT_NAMES["left"]),
    "arm_r": list(SG2_ARM_JOINT_NAMES["right"]),
    "body": [
        "lift_joint",
        "head_joint1",
        "head_joint2",
        SG2_GRIPPER_DRIVER_JOINT_NAMES["left"],
        SG2_GRIPPER_DRIVER_JOINT_NAMES["right"],
    ],
}
SG2_SLIDER_GROUP_TITLES = {
    "arm_l": "SG2 arm_l",
    "arm_r": "SG2 arm_r",
    "body": "SG2 lift / head / gripper",
}

# (x_offset, y_offset, window_width, window_height) as monitor ratios.
SG2_SLIDER_GROUP_LAYOUTS = {
    "arm_l": (0.00, 0.03, 0.20, 0.44),
    "arm_r": (0.00, 0.50, 0.20, 0.44),
    "body": (0.20, 0.03, 0.20, 0.34),
}
SG2_JOINT_COMMAND_LAYOUT = (0.20, 0.40, 0.20, 0.24)

SG2_JOINT_COMMAND_ROWS = {
    "Preset": ["idle", "zero", "init", "sync"],
    "File": ["idle", "save", "load"],
    "Collision": ["off", "on"],
    "EEF": ["off", "on"],
}

# "init" mirrors the last step of ffw_sg2_follower_initial_positions.yaml
# ([0,0,0,-1.57,0,0,0] per arm), which is also the elbow-bent pose the other
# SG2 notebooks start from. Joints absent from a preset go to 0.
SG2_JOINT_PRESETS = {
    "zero": {},
    "init": {"arm_l_joint4": -0.5 * np.pi, "arm_r_joint4": -0.5 * np.pi},
}

SG2_JOINT_QPOS_PATH = PROJECT_DIR / "qpos" / "sg2_joint_qpos.json"


class SG2JointSliderPanel(object):
    """Grouped joint sliders that drive the MuJoCo model directly."""

    def __init__(
            self,
            env,
            joint_groups=None,
            group_titles=None,
            group_layouts=None,
        ):
        if joint_groups is None:
            joint_groups = SG2_SLIDER_JOINT_GROUPS
        if group_titles is None:
            group_titles = SG2_SLIDER_GROUP_TITLES
        if group_layouts is None:
            group_layouts = SG2_SLIDER_GROUP_LAYOUTS

        self.env = env
        self.group_sliders = {}
        for group_key, joint_names in joint_groups.items():
            if len(env.get_active_among_joints(joint_names)) == 0:
                continue
            x_offset, y_offset, window_width, window_height = group_layouts[group_key]
            self.group_sliders[group_key] = EnvJointSlider(
                env=env,
                joint_names=joint_names,
                title=group_titles.get(group_key, group_key),
                x_offset=x_offset,
                y_offset=y_offset,
                window_width=window_width,
                window_height=window_height,
            )

        if len(self.group_sliders) == 0:
            raise ValueError("No active joints found for the SG2 joint slider panel.")

        self.joint_names = [
            joint_name
            for slider in self.group_sliders.values()
            for joint_name in slider.active_joint_names
        ]
        self.q_min = env.get_joint_mins(self.joint_names)
        self.q_max = env.get_joint_maxs(self.joint_names)
        self.revolute_flags = [
            joint_name in set(env.rev_joint_names) for joint_name in self.joint_names
        ]

    def get_qpos(self):
        return np.concatenate(
            [slider.get_qactive() for slider in self.group_sliders.values()]
        )

    def update(self):
        for slider in self.group_sliders.values():
            slider.update(update_env=True)

        qpos = self.get_qpos()
        for hand_type in SG2_HAND_TYPES:
            driver_joint_name = SG2_GRIPPER_DRIVER_JOINT_NAMES[hand_type]
            if driver_joint_name not in self.joint_names:
                continue
            set_sg2_gripper_qpos(
                self.env,
                hand_type,
                float(qpos[self.joint_names.index(driver_joint_name)]),
            )
        return qpos

    def set_qpos(self, qpos, joint_names=None):
        if joint_names is None:
            joint_names = self.joint_names
        qpos_by_joint_name = {
            joint_name: float(value)
            for joint_name, value in zip(joint_names, np.asarray(qpos, dtype=np.float64).ravel())
        }

        for slider in self.group_sliders.values():
            idxs, values = [], []
            for slider_idx, joint_name in enumerate(slider.active_joint_names):
                if joint_name not in qpos_by_joint_name:
                    continue
                panel_idx = self.joint_names.index(joint_name)
                idxs.append(slider_idx)
                values.append(
                    float(
                        np.clip(
                            qpos_by_joint_name[joint_name],
                            self.q_min[panel_idx],
                            self.q_max[panel_idx],
                        )
                    )
                )
            if len(idxs) > 0:
                slider.sliders.set_values(values, idxs=idxs)
        return self.update()

    def sync_from_env(self):
        return self.set_qpos(self.env.get_qpos(joint_names=self.joint_names))

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

    def is_visible(self):
        return any(
            slider.sliders.isVisible() for slider in self.group_sliders.values()
        )

    def close(self):
        for slider in self.group_sliders.values():
            slider.close()


def save_sg2_joint_qpos(joint_names, qpos, path=SG2_JOINT_QPOS_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "joint_names": list(joint_names),
                "qpos": [float(value) for value in np.asarray(qpos, dtype=np.float64).ravel()],
            },
            indent=2,
        )
    )
    return path


def load_sg2_joint_qpos(path=SG2_JOINT_QPOS_PATH):
    data = json.loads(Path(path).read_text())
    return list(data["joint_names"]), np.array(data["qpos"], dtype=np.float64)


def create_sg2_joint_command_widget(layout=SG2_JOINT_COMMAND_LAYOUT):
    x_offset, y_offset, window_width, window_height = layout
    return MultiRadioQtWidget(
        title="SG2 Joint Command",
        x_offset=x_offset,
        y_offset=y_offset,
        window_width=window_width,
        window_height=window_height,
        label_texts=list(SG2_JOINT_COMMAND_ROWS),
        option_texts=list(SG2_JOINT_COMMAND_ROWS.values()),
        initial_texts=["idle", "idle", "on", "on"],
    )


def handle_sg2_joint_command_widget(widget, panel, qpos_path=SG2_JOINT_QPOS_PATH):
    """Run the one-shot Preset/File actions and read the Collision/EEF toggles.

    Call this exactly once per tick: `MultiRadioQtWidget.is_toggled` consumes the
    change event, so a second call in the same tick would miss it. Action rows
    snap back to "idle" once handled.
    """
    message = ""

    preset_name, preset_toggled = widget.get_button_option("Preset", return_toggled=True)
    if preset_toggled and preset_name not in (None, "idle"):
        if preset_name == "sync":
            panel.sync_from_env()
            message = "sync: sliders <- MuJoCo qpos"
        else:
            panel.apply_preset(preset_name)
            message = "preset: %s" % preset_name
        widget.set_button_option("Preset", "idle")

    file_action, file_toggled = widget.get_button_option("File", return_toggled=True)
    if file_toggled and file_action not in (None, "idle"):
        if file_action == "save":
            message = "saved: %s" % save_sg2_joint_qpos(
                panel.joint_names, panel.get_qpos(), path=qpos_path
            )
        else:
            try:
                saved_joint_names, saved_qpos = load_sg2_joint_qpos(path=qpos_path)
                panel.set_qpos(saved_qpos, joint_names=saved_joint_names)
                message = "loaded: %s" % qpos_path
            except FileNotFoundError:
                message = "load failed: no file at %s" % qpos_path
        widget.set_button_option("File", "idle")

    return {
        "message": message,
        "check_collision": widget.get_button_option("Collision") == "on",
        "show_eef": widget.get_button_option("EEF") == "on",
    }


def get_sg2_self_collision_status(
        env,
        collision_geom_pairs,
        native_backend=None,
        safe_margin=0.05,
        signed_distance_topk=5,
    ):
    """Signed distances for the candidate geom pairs at the current qpos.

    No IK is run here — the sliders own the joints, this only reports whether the
    pose the user dialled in is safe.
    """
    if native_backend is None:
        signed_distance_infos = env.get_geom_pair_signed_distance_infos(
            collision_geom_pairs,
            topk=signed_distance_topk,
        )
        min_signed_dist = min(
            [info["signed_dist"] for info in signed_distance_infos],
            default=np.inf,
        )
    else:
        signed_distance_infos, min_signed_dist = env.get_geom_pair_signed_distance_infos_bind(
            collision_geom_pairs,
            topk=signed_distance_topk,
            be=native_backend,
            return_min_signed_dist=True,
        )

    min_signed_dist = float(min_signed_dist)
    safe_margin = float(safe_margin)
    if not np.isfinite(min_signed_dist):
        level = "safe"
    elif min_signed_dist < 0.0:
        level = "collision"
    elif min_signed_dist <= safe_margin:
        level = "warning"
    else:
        level = "safe"

    return {
        "signed_distance_infos": signed_distance_infos,
        "min_signed_dist": min_signed_dist,
        "safe_margin": safe_margin,
        "level": level,
    }


def get_sg2_self_collision_rgba(signed_dist, safe_margin):
    if signed_dist < 0.0:
        return (1.0, 0.0, 0.0, 0.90)
    if signed_dist <= safe_margin:
        return (1.0, 0.6, 0.0, 0.85)
    return (0.0, 0.0, 1.0, 0.55)


def draw_sg2_self_collision(env, collision_status, sphere_r=0.004):
    if collision_status is None:
        return
    for info in collision_status["signed_distance_infos"]:
        rgba = get_sg2_self_collision_rgba(
            info["signed_dist"], collision_status["safe_margin"]
        )
        env.plot_line_fr2to(p_fr=info["p1"], p_to=info["p2"], rgba=rgba)
        env.plot_sphere(p=info["p1"], r=sphere_r, rgba=rgba)
        env.plot_sphere(p=info["p2"], r=sphere_r, rgba=rgba)


def draw_sg2_wrist_frames(env, hand_types=SG2_HAND_TYPES, hand_labels=SG2_HAND_LABELS, axis_len=0.12):
    for hand_type in hand_types:
        env.plot_T(
            T=get_sg2_wrist_control_T(env, hand_type),
            axis_len=axis_len,
            axis_width=0.005,
            label=hand_labels[hand_type],
        )


def make_sg2_joint_slider_overlay_text(panel):
    """One line per panel group, revolute joints in degrees, prismatic in meters."""
    lines = []
    qpos = panel.get_qpos()
    joint_idx = 0
    for group_key, slider in panel.group_sliders.items():
        entries = []
        for joint_name in slider.active_joint_names:
            if panel.revolute_flags[joint_idx]:
                entries.append("%.1f" % np.rad2deg(qpos[joint_idx]))
            else:
                entries.append("%.3fm" % qpos[joint_idx])
            joint_idx += 1
        lines.append("%-6s: %s" % (group_key, " ".join(entries)))
    return "\n".join(lines)


def make_sg2_eef_overlay_text(env, hand_types=SG2_HAND_TYPES, hand_labels=SG2_HAND_LABELS):
    lines = []
    for hand_type in hand_types:
        prpy = t2prpy(get_sg2_wrist_control_T(env, hand_type), unit="deg")
        lines.append(
            "%s p:[%+.3f %+.3f %+.3f] rpy:[%+6.1f %+6.1f %+6.1f]"
            % (hand_labels[hand_type], prpy[0], prpy[1], prpy[2], prpy[3], prpy[4], prpy[5])
        )
    return "\n".join(lines)


def make_sg2_self_collision_overlay_text(collision_status):
    if collision_status is None:
        return "off"
    if not np.isfinite(collision_status["min_signed_dist"]):
        return "%s | no candidate pair in range" % collision_status["level"].upper()
    return "%s | min signed dist:%+.4f m (margin %.3f m)" % (
        collision_status["level"].upper(),
        collision_status["min_signed_dist"],
        collision_status["safe_margin"],
    )


__all__ = [
    "SG2JointSliderPanel",
    "SG2_JOINT_COMMAND_LAYOUT",
    "SG2_JOINT_COMMAND_ROWS",
    "SG2_JOINT_PRESETS",
    "SG2_JOINT_QPOS_PATH",
    "SG2_SLIDER_GROUP_LAYOUTS",
    "SG2_SLIDER_GROUP_TITLES",
    "SG2_SLIDER_JOINT_GROUPS",
    "create_sg2_joint_command_widget",
    "draw_sg2_self_collision",
    "draw_sg2_wrist_frames",
    "get_sg2_self_collision_rgba",
    "get_sg2_self_collision_status",
    "handle_sg2_joint_command_widget",
    "load_sg2_joint_qpos",
    "make_sg2_eef_overlay_text",
    "make_sg2_joint_slider_overlay_text",
    "make_sg2_self_collision_overlay_text",
    "save_sg2_joint_qpos",
]
