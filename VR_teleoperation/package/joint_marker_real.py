"""Real-robot additions for the in-viewer joint marker controller.

`joint_marker_control.py` is the whole controller; nothing here replaces it. What a
real SG2 needs on top of the sim version is three things:

1. A joint speed limit. A marker drag is a human hand and never moves fast, but a
   `Preset` or a `File` load is an instant whole-arm jump. On the robot that jump
   is a swing at whatever is standing nearby, so the published pose is ramped and
   the ramped pose -- not the marker pose -- is what MuJoCo shows.
2. A `Robot` OFF/ON gate, off at startup, the same one the MetaQuest and wrist
   marker notebooks use.
3. A render payload for `real_render_MetaQuest3_sg2.ipynb`, which expects the key
   set the Quest notebooks send.

The `Preset` row drops `zero` here: reaching an all-zero pose from a folded arm is
exactly the motion `ffw_sg2_follower_initial_positions.yaml` splits into two steps
to avoid. `init` (elbows at -90 deg) is that yaml's own end state.
"""

from __future__ import annotations

import numpy as np

from utils import (
    SG2_GRIPPER_DRIVER_JOINT_NAMES,
    SG2_HAND_TYPES,
    copy_hand_T_dict,
    get_sg2_wrist_control_T,
    set_sg2_gripper_qpos,
)
from .joint_control import SG2_JOINT_QPOS_PATH
from .joint_marker_control import (
    SG2_JOINT_MARKER_GROUPS,
    handle_sg2_joint_marker_command_widget,
)
from ri_motion_v5_package.qt import MultiRadioQtWidget

# Conservative first-run ceilings. 60 deg/s turns a 90 deg preset into a 1.5 s move.
SG2_MAX_REV_JOINT_SPEED = np.deg2rad(60.0)
SG2_MAX_PRI_JOINT_SPEED = 0.05

SG2_JOINT_MARKER_REAL_COMMAND_ROWS = {
    "Robot": ["OFF", "ON"],
    "Preset": ["idle", "sync", "init"],
    "File": ["idle", "save", "load"],
    "Show": ["all"] + list(SG2_JOINT_MARKER_GROUPS),
    "Collision": ["off", "on"],
    "EEF": ["off", "on"],
}
SG2_JOINT_MARKER_REAL_COMMAND_LAYOUT = (0.00, 0.00, 0.24, 0.34)


def create_sg2_joint_marker_real_command_widget(layout=SG2_JOINT_MARKER_REAL_COMMAND_LAYOUT):
    x_offset, y_offset, window_width, window_height = layout
    return MultiRadioQtWidget(
        title="SG2 Joint Marker Command (real)",
        x_offset=x_offset,
        y_offset=y_offset,
        window_width=window_width,
        window_height=window_height,
        label_texts=list(SG2_JOINT_MARKER_REAL_COMMAND_ROWS),
        option_texts=list(SG2_JOINT_MARKER_REAL_COMMAND_ROWS.values()),
        initial_texts=["OFF", "idle", "idle", "all", "on", "on"],
    )


def handle_sg2_joint_marker_real_command_widget(widget, controller, qpos_path=SG2_JOINT_QPOS_PATH):
    """Sim handler plus the `Robot` gate. Call once per loop pass, outside the
    timers, so the gate reacts on the same tick it is switched."""
    command_info = handle_sg2_joint_marker_command_widget(
        widget=widget,
        controller=controller,
        qpos_path=qpos_path,
    )
    command_info["robot_on"] = widget.get_button_option("Robot") == "ON"
    return command_info


def get_sg2_joint_speed_limits(
        env,
        joint_names,
        max_rev_speed=SG2_MAX_REV_JOINT_SPEED,
        max_pri_speed=SG2_MAX_PRI_JOINT_SPEED,
    ):
    """Per-joint ceiling in rad/s for hinges and m/s for the lift slide."""
    pri_joint_names = set(env.pri_joint_names)
    return np.array(
        [
            max_pri_speed if joint_name in pri_joint_names else max_rev_speed
            for joint_name in joint_names
        ],
        dtype=np.float64,
    )


def apply_sg2_joint_rate_limit(env, joint_names, q_prev, dt, speed_limits,
                               coordinated_groups=None):
    """Clamp this tick's joint motion, write the clamped pose back into MuJoCo, and
    report whether anything actually hit the ceiling.

    Writing it back matters: the viewer, the markers and the published command then
    all show the one pose the robot was told to reach, instead of the sim racing
    ahead of a robot that is still ramping. Optional disjoint joint-name groups
    share a reduction factor, preserving the IK joint-displacement direction.
    Ungrouped joints retain the existing independent clipping behavior.
    """
    joint_names = list(joint_names)
    q_prev = np.asarray(q_prev, dtype=np.float64).ravel()
    dq_max = max(float(dt), 0.0) * np.asarray(speed_limits, dtype=np.float64)
    dq = env.get_qpos(joint_names=joint_names) - q_prev
    limited_dq = np.clip(dq, -dq_max, dq_max)
    group_scales = []
    used = set()
    for group in (() if coordinated_groups is None else coordinated_groups):
        indices = [joint_names.index(name) for name in group]
        if not indices or len(set(indices)) != len(indices) or used.intersection(indices):
            raise ValueError("Coordinated rate-limit groups must be nonempty and disjoint")
        used.update(indices)
        step = np.abs(dq[indices])
        moving = step > 0.
        factor = min(1., float(np.min(dq_max[indices][moving] / step[moving]))) if np.any(moving) else 1.
        limited_dq[indices] = factor * dq[indices]
        group_scales.append(factor)
    q_use = q_prev + limited_dq
    env.forward(q=q_use, joint_names=joint_names)

    for hand_type in SG2_HAND_TYPES:
        driver_joint_name = SG2_GRIPPER_DRIVER_JOINT_NAMES[hand_type]
        if driver_joint_name in joint_names:
            set_sg2_gripper_qpos(env, hand_type, float(q_use[joint_names.index(driver_joint_name)]))

    return {
        "qpos": q_use,
        "group_scales": group_scales,
        "limited": bool(np.any(np.abs(dq) > dq_max + 1e-12)),
        "max_dq_ratio": float(np.max(np.abs(dq) / np.maximum(dq_max, 1e-12))) if len(dq) else 0.0,
    }


def make_sg2_robot_command_overlay_text(robot_on, robot_pub_hz_actual, rate_limit_info=None):
    text = "robot %s | pub %.1fHz" % ("ON" if robot_on else "OFF", robot_pub_hz_actual)
    if rate_limit_info is not None and rate_limit_info["limited"]:
        text += " | RATE LIMITED (x%.1f over)" % rate_limit_info["max_dq_ratio"]
    return text


def make_sg2_joint_marker_render_extra(
        env,
        controller,
        robot_on,
        collision_status=None,
        gripper_q_by_hand=None,
        hand_types=SG2_HAND_TYPES,
    ):
    """Render payload shaped like the Quest notebooks send, so
    `real_render_MetaQuest3_sg2.ipynb` can display this notebook unchanged."""
    if gripper_q_by_hand is None:
        gripper_q_by_hand = {
            hand_type: float(
                env.get_qpos(joint_names=[SG2_GRIPPER_DRIVER_JOINT_NAMES[hand_type]])[0]
            )
            for hand_type in hand_types
        }
    return {
        "raw_robot_command_on": robot_on,
        "attached": {hand_type: True for hand_type in hand_types},
        "attach_phase": {hand_type: "joint_marker" for hand_type in hand_types},
        "attach_progress": {hand_type: 1.0 for hand_type in hand_types},
        "quest_ok": {hand_type: False for hand_type in hand_types},
        "quest_all_ok": False,
        "collision_ik_mode": "joint_marker",
        "min_signed_dist": float(
            np.nan if collision_status is None else collision_status["min_signed_dist"]
        ),
        "safe_margin": float(
            0.05 if collision_status is None else collision_status["safe_margin"]
        ),
        "T_sg2_wrist": copy_hand_T_dict(
            {hand_type: get_sg2_wrist_control_T(env, hand_type) for hand_type in hand_types}
        ),
        "active_joint": controller.active_joint_name,
        "visible_group": controller.visible_group,
        "gripper_q": dict(gripper_q_by_hand),
    }


__all__ = [
    "SG2_JOINT_MARKER_REAL_COMMAND_LAYOUT",
    "SG2_JOINT_MARKER_REAL_COMMAND_ROWS",
    "SG2_MAX_PRI_JOINT_SPEED",
    "SG2_MAX_REV_JOINT_SPEED",
    "apply_sg2_joint_rate_limit",
    "create_sg2_joint_marker_real_command_widget",
    "get_sg2_joint_speed_limits",
    "handle_sg2_joint_marker_real_command_widget",
    "make_sg2_joint_marker_render_extra",
    "make_sg2_robot_command_overlay_text",
]
