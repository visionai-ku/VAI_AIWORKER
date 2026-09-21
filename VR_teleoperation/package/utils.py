from __future__ import annotations

from enum import IntEnum
from pathlib import Path
import time

import mujoco
import numpy as np

try:
    import zmq
except ModuleNotFoundError:
    zmq = None

from ri_motion_v5_package.kinematics.transforms import (
    pr2t,
    quat_conjugate,
    quat_multiply,
    r2quat,
    r2w,
    t2p,
    t2r,
    w2r,
)
from ri_motion_v5_package.utility import ZMQPubSub, get_idxs, group_by_prefix


THIS_FILE = Path(__file__).resolve()
PROJECT_DIR = THIS_FILE.parents[1]
REPO_ROOT = PROJECT_DIR


def quat2r_mj(q):
    """Convert MuJoCo quaternion [w, x, y, z] to a rotation matrix."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / q_norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


SG2_PROJECT_XML_DIR = PROJECT_DIR / "xml"
SG2_XML_PATH = SG2_PROJECT_XML_DIR / "ffw_sg2.xml"
FLOOR_XML_PATH = SG2_PROJECT_XML_DIR / "floor_simple_white.xml"
SG2_SCENE_XML_PATH = SG2_PROJECT_XML_DIR / "scene_ffw_sg2_metaquest3.xml"
SG2_BODY_ADDED_VIS_XML_PATH = SG2_PROJECT_XML_DIR / "ffw_sg2_body_added_vis.xml"
SG2_SCENE_CACHE_DIR = PROJECT_DIR / ".cache"
SG2_RENDER_SCENE_XML_PATH = SG2_SCENE_CACHE_DIR / "scene_ffw_sg2_metaquest3_render.xml"

SG2_HAND_TYPES = ("left", "right")
SG2_HAND_LABELS = {"left": "L", "right": "R"}
SG2_ARM_JOINT_PREFIXES = {"left": "arm_l_joint", "right": "arm_r_joint"}
SG2_WRIST_BODY_NAMES = {"left": "arm_l_link7", "right": "arm_r_link7"}
SG2_WRIST_P_LOCAL = {
    "left": np.array([0.0, 0.0, -0.08], dtype=np.float64),
    "right": np.array([0.0, 0.0, -0.08], dtype=np.float64),
}
SG2_WRIST_R_LOCAL = {
    "left": quat2r_mj(np.array([0.5, -0.5, 0.5, 0.5], dtype=np.float64)),
    "right": quat2r_mj(np.array([0.5, 0.5, 0.5, -0.5], dtype=np.float64)),
}

SG2_ARM_JOINT_NAMES = {
    "left": ["arm_l_joint%d" % idx for idx in range(1, 8)],
    "right": ["arm_r_joint%d" % idx for idx in range(1, 8)],
}
SG2_GRIPPER_DRIVER_JOINT_NAMES = {
    "left": "gripper_l_joint1",
    "right": "gripper_r_joint1",
}
SG2_GRIPPER_JOINT_NAMES = {
    "left": ["gripper_l_joint%d" % idx for idx in range(1, 5)],
    "right": ["gripper_r_joint%d" % idx for idx in range(1, 5)],
}
SG2_GRIPPER_JOINT_SIGNS = np.array([1.0, -1.0, -1.0, 1.0], dtype=np.float64)
SG2_REAL_COMMAND_JOINT_NAMES = (
    ["lift_joint", "head_joint1", "head_joint2"]
    + SG2_ARM_JOINT_NAMES["left"]
    + SG2_ARM_JOINT_NAMES["right"]
)
SG2_COLLISION_IK_JOINT_NAMES = (
    SG2_ARM_JOINT_NAMES["left"]
    + SG2_ARM_JOINT_NAMES["right"]
)
SG2_UPPER_ARM_YAW_JOINT_NAMES = ("arm_r_joint3", "arm_l_joint3")
SG2_NULL_REFERENCE_MODES = ("previous", "home", "self_motion")

SG2_COLLISION_AWARE_IK_MODULE_NAME = "_mujoco_collision_aware_ik_bind"
SG2_COLLISION_AWARE_IK_CPP_SOURCE_NAMES = (
    "bindings.cpp",
    "signed_distance_bindings.cpp",
    "signed_distance.cpp",
    "ik_bindings.cpp",
    "ik.cpp",
)
SG2_COLLISION_AWARE_IK_HPP_HEADER_NAMES = (
    "signed_distance_bindings.hpp",
    "signed_distance.hpp",
    "ik_bindings.hpp",
    "ik.hpp",
    "mujoco_handles.hpp",
)
SG2_COLLISION_AWARE_IK_DEFINE_NAMES = (
    "RI_NATIVE_ENABLE_SIGNED_DISTANCE",
    "RI_NATIVE_ENABLE_IK",
)

QUEST_PACKET_HEADER_FLOATS = 3
QUEST_POSE_FLOATS = 7
QUEST_HAND_JOINT_OFFSETS = {
    "thumb": {"cmc": 1, "mcp": 2, "ip": 3, "tip": 4},
    "index": {"mcp": 6, "pip": 7, "dip": 8, "tip": 9},
    "middle": {"mcp": 11, "pip": 12, "dip": 13, "tip": 14},
    "ring": {"mcp": 16, "pip": 17, "dip": 18, "tip": 19},
    "little": {"mcp": 21, "pip": 22, "dip": 23, "tip": 24},
}


class QuestJointIdx(IntEnum):
    LEFT_PALM = 2
    RIGHT_PALM = 28


class QuestPalmRetargeting:
    def __init__(self, start_T):
        self.start_T = np.asarray(start_T, dtype=np.float64).reshape(4, 4)
        self.quat_meta2mj = self._meta2mj_quat()

    def parse_packet(self, data):
        data = np.asarray(data, dtype=np.float64).reshape(-1)
        poses = {
            "left_palm": self._pose_from_packet(data, QuestJointIdx.LEFT_PALM),
            "right_palm": self._pose_from_packet(data, QuestJointIdx.RIGHT_PALM),
        }
        return {key: None if T is None else self.start_T @ T for key, T in poses.items()}

    def parse_all_pose_dict(self, data):
        data = np.asarray(data, dtype=np.float64).reshape(-1)
        n_pose = max(0, (len(data) - QUEST_PACKET_HEADER_FLOATS) // QUEST_POSE_FLOATS)
        T_all = {}
        for joint_idx in range(n_pose):
            T = self._pose_from_packet(data, joint_idx)
            if T is not None:
                T_all[int(joint_idx)] = self.start_T @ T
        return T_all

    def _pose_from_packet(self, data, joint_idx):
        base = QUEST_PACKET_HEADER_FLOATS + QUEST_POSE_FLOATS * int(joint_idx)
        if base + QUEST_POSE_FLOATS > len(data):
            return None

        pose = data[base : base + QUEST_POSE_FLOATS]
        if not np.isfinite(pose).all():
            return None

        q_meta = np.array([pose[6], pose[3], pose[4], pose[5]], dtype=np.float64)
        if np.linalg.norm(q_meta) < 1e-12:
            return None

        p_mj = np.array([-pose[2], -pose[0], pose[1]], dtype=np.float64)
        q_mj = quat_multiply(
            quat_multiply(self.quat_meta2mj, q_meta),
            quat_conjugate(self.quat_meta2mj),
        )
        return pr2t(p=p_mj, R=quat2r_mj(q_mj))

    @staticmethod
    def _meta2mj_quat():
        R_meta2mj = np.array(
            [
                [0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        return r2quat(R_meta2mj)


def _require_zmq():
    if zmq is None:
        raise ModuleNotFoundError("pyzmq is required for MetaQuest3 VR teleoperation.")
    return zmq


def build_sg2_collision_aware_ik_backend(clean=False, verbose=True, required=True):
    from ri_motion_v5_package.mujoco_sim import build_native, clean_native, get_native_backend

    if clean:
        clean_native(
            module_name=SG2_COLLISION_AWARE_IK_MODULE_NAME,
            verbose=verbose,
        )

    build_native(
        cpp_source_names=SG2_COLLISION_AWARE_IK_CPP_SOURCE_NAMES,
        hpp_header_names=SG2_COLLISION_AWARE_IK_HPP_HEADER_NAMES,
        define_names=SG2_COLLISION_AWARE_IK_DEFINE_NAMES,
        module_name=SG2_COLLISION_AWARE_IK_MODULE_NAME,
        verbose=verbose,
    )
    return get_native_backend(
        required=required,
        module_name=SG2_COLLISION_AWARE_IK_MODULE_NAME,
    )


def create_zmq_sub_socket(ip_address, port, wait_seconds_after_connect=1.0):
    zmq_module = _require_zmq()
    ctx = zmq_module.Context.instance()
    sock = ctx.socket(zmq_module.SUB)
    sock.setsockopt(zmq_module.CONFLATE, 1)
    sock.set_hwm(1)
    sock.setsockopt(zmq_module.SUBSCRIBE, b"")
    sock.connect("tcp://%s:%d" % (ip_address, int(port)))
    time.sleep(wait_seconds_after_connect)
    return sock


def close_zmq_socket(sock):
    if sock is not None:
        sock.close(0)


def receive_metaquest3(socket_sub):
    zmq_module = _require_zmq()
    try:
        data = socket_sub.recv(flags=zmq_module.NOBLOCK)
    except zmq_module.Again:
        return None
    if data is None or len(data) == 0:
        return None
    return np.frombuffer(data, dtype=np.float32)


def update_metaquest3_wrist_data(
        socket_sub,
        quest_retargeting,
        hand_types,
        T_quest_wrist,
        T_quest_pose_by_idx,
        quest_missing_ticks,
        quest_all_missing_ticks,
        quest_attach_state=None,
        check_distance=0.10,
    ):
    quest_packet = receive_metaquest3(socket_sub)
    quest_wrist_poses = None if quest_packet is None else quest_retargeting.parse_packet(quest_packet)
    next_pose_by_idx = None if quest_packet is None else quest_retargeting.parse_all_pose_dict(quest_packet)

    if next_pose_by_idx is None:
        quest_all_missing_ticks += 1
    else:
        T_quest_pose_by_idx = next_pose_by_idx
        quest_all_missing_ticks = 0

    quest_updated = {hand_type: False for hand_type in hand_types}
    wrist_jump_rejected = {hand_type: False for hand_type in hand_types}

    for hand_type in hand_types:
        T_received_wrist = None if quest_wrist_poses is None else quest_wrist_poses.get("%s_palm" % hand_type)
        if T_received_wrist is None:
            quest_missing_ticks[hand_type] += 1
            continue

        wrist_jump_rejected[hand_type] = is_quest_wrist_jump_rejected(
            T_wrist_prev=T_quest_wrist[hand_type],
            T_wrist_new=T_received_wrist,
            hand_type=hand_type,
            check_distance=check_distance,
        )
        T_quest_wrist[hand_type] = T_received_wrist
        quest_missing_ticks[hand_type] = 0
        quest_updated[hand_type] = True

        if wrist_jump_rejected[hand_type] and quest_attach_state is not None:
            quest_attach_state[hand_type]["is_attached"] = False
            quest_attach_state[hand_type]["phase"] = "idle"

    return (
        T_quest_wrist,
        T_quest_pose_by_idx,
        quest_missing_ticks,
        quest_all_missing_ticks,
        quest_updated,
        wrist_jump_rejected,
    )


def normalize_hand_type(hand_type):
    hand_type = str(hand_type).lower()
    if hand_type not in SG2_HAND_TYPES:
        raise ValueError("hand_type must be one of %s." % (SG2_HAND_TYPES,))
    return hand_type


def get_sg2_wrist_body_name(hand_type):
    return SG2_WRIST_BODY_NAMES[normalize_hand_type(hand_type)]


def get_sg2_wrist_local_T(hand_type):
    hand_type = normalize_hand_type(hand_type)
    return pr2t(
        p=SG2_WRIST_P_LOCAL[hand_type],
        R=SG2_WRIST_R_LOCAL[hand_type],
    )


def get_sg2_wrist_control_T(env, hand_type):
    hand_type = normalize_hand_type(hand_type)
    return env.get_T(SG2_WRIST_BODY_NAMES[hand_type], "body") @ get_sg2_wrist_local_T(hand_type)


def compute_quest_target_T_from_wrist_transform(T_wrist, hand_type):
    normalize_hand_type(hand_type)
    if T_wrist is None:
        return None
    return np.asarray(T_wrist, dtype=np.float64).reshape(4, 4)


def transform_T_between_frames(T_source, T_target, T):
    if T_source is None or T_target is None or T is None:
        return None
    return (
        np.asarray(T_target, dtype=np.float64).reshape(4, 4)
        @ np.linalg.inv(np.asarray(T_source, dtype=np.float64).reshape(4, 4))
        @ np.asarray(T, dtype=np.float64).reshape(4, 4)
    )


def get_quest_wrist_joint_idx(hand_type):
    hand_type = normalize_hand_type(hand_type)
    return int(QuestJointIdx.LEFT_PALM if hand_type == "left" else QuestJointIdx.RIGHT_PALM)


def get_quest_hand_joint_idx(hand_type, finger_name, joint_name):
    hand_type = normalize_hand_type(hand_type)
    finger_name = str(finger_name).lower()
    joint_name = str(joint_name).lower()
    if finger_name not in QUEST_HAND_JOINT_OFFSETS:
        raise ValueError("finger_name must be one of %s." % (tuple(QUEST_HAND_JOINT_OFFSETS),))
    if joint_name not in QUEST_HAND_JOINT_OFFSETS[finger_name]:
        raise ValueError("joint_name must be one of %s." % (tuple(QUEST_HAND_JOINT_OFFSETS[finger_name]),))
    return get_quest_wrist_joint_idx(hand_type) + int(QUEST_HAND_JOINT_OFFSETS[finger_name][joint_name])


def get_quest_thumb_index_tip_positions(T_pose_by_idx, hand_type):
    if not isinstance(T_pose_by_idx, dict):
        return None, None

    thumb_tip_idx = get_quest_hand_joint_idx(hand_type, "thumb", "tip")
    index_tip_idx = get_quest_hand_joint_idx(hand_type, "index", "tip")
    if thumb_tip_idx not in T_pose_by_idx or index_tip_idx not in T_pose_by_idx:
        return None, None

    p_thumb_tip = t2p(T_pose_by_idx[thumb_tip_idx])
    p_index_tip = t2p(T_pose_by_idx[index_tip_idx])
    if not np.isfinite(p_thumb_tip).all() or not np.isfinite(p_index_tip).all():
        return None, None
    return p_thumb_tip, p_index_tip


def get_quest_thumb_index_tip_distance(T_pose_by_idx, hand_type):
    p_thumb_tip, p_index_tip = get_quest_thumb_index_tip_positions(T_pose_by_idx, hand_type)
    if p_thumb_tip is None or p_index_tip is None:
        return np.nan
    return float(np.linalg.norm(p_thumb_tip - p_index_tip))


def get_quest_finger_bend_ratio(T_pose_by_idx, hand_type, finger_name):
    if not isinstance(T_pose_by_idx, dict):
        return np.nan

    hand_type = normalize_hand_type(hand_type)
    finger_name = str(finger_name).lower()
    joint_names = ("mcp", "pip", "dip", "tip")
    if finger_name == "thumb":
        joint_names = ("cmc", "mcp", "ip", "tip")

    p_list = []
    for joint_name in joint_names:
        joint_idx = get_quest_hand_joint_idx(hand_type, finger_name, joint_name)
        if joint_idx not in T_pose_by_idx:
            return np.nan
        p_joint = t2p(T_pose_by_idx[joint_idx])
        if not np.isfinite(p_joint).all():
            return np.nan
        p_list.append(p_joint)

    chain_len = sum(
        float(np.linalg.norm(p_list[idx + 1] - p_list[idx]))
        for idx in range(len(p_list) - 1)
    )
    tip_span = float(np.linalg.norm(p_list[-1] - p_list[0]))
    if chain_len < 1e-9:
        return np.nan
    return float(np.clip(1.0 - tip_span / chain_len, 0.0, 1.0))


def get_quest_hand_bend_ratios(
        T_pose_by_idx,
        hand_type,
        finger_names=("middle", "ring", "little"),
    ):
    return {
        finger_name: get_quest_finger_bend_ratio(T_pose_by_idx, hand_type, finger_name)
        for finger_name in finger_names
    }


def create_sg2_detach_gesture_state(
        hand_types=SG2_HAND_TYPES,
        finger_names=("middle", "ring", "little"),
    ):
    finger_names = tuple(str(finger_name).lower() for finger_name in finger_names)
    hand_types = tuple(normalize_hand_type(hand_type) for hand_type in hand_types)
    return {
        "finger_names": finger_names,
        "bend_ratio_by_hand": {
            hand_type: {finger_name: np.nan for finger_name in finger_names}
            for hand_type in hand_types
        },
        "mean_bend_ratio_by_hand": {hand_type: np.nan for hand_type in hand_types},
        "closed_finger_count_by_hand": {hand_type: 0 for hand_type in hand_types},
        "is_detach_gesture_by_hand": {hand_type: False for hand_type in hand_types},
        "gesture_active": False,
        "hold_seconds": 0.0,
        "progress": 0.0,
        "should_detach": False,
    }


def update_sg2_detach_gesture_state(
        T_pose_by_idx,
        hand_types=SG2_HAND_TYPES,
        quest_missing_ticks=None,
        max_missing_quest_ticks=None,
        dt=0.0,
        detach_gesture_state=None,
        finger_names=("middle", "ring", "little"),
        enter_bend_ratio=0.55,
        exit_bend_ratio=0.35,
        mean_enter_bend_ratio=None,
        mean_exit_bend_ratio=None,
        min_closed_finger_count=2,
        hold_seconds_to_detach=1.2,
    ):
    finger_names = tuple(str(finger_name).lower() for finger_name in finger_names)
    hand_types = tuple(normalize_hand_type(hand_type) for hand_type in hand_types)
    if detach_gesture_state is None:
        detach_gesture_state = create_sg2_detach_gesture_state(
            hand_types=hand_types,
            finger_names=finger_names,
        )

    mean_enter_bend_ratio = (
        float(enter_bend_ratio)
        if mean_enter_bend_ratio is None else float(mean_enter_bend_ratio)
    )
    mean_exit_bend_ratio = (
        float(exit_bend_ratio)
        if mean_exit_bend_ratio is None else float(mean_exit_bend_ratio)
    )
    min_closed_finger_count = int(max(min_closed_finger_count, 1))
    hold_seconds_to_detach = float(max(hold_seconds_to_detach, 1e-6))

    bend_ratio_by_hand = {}
    mean_bend_ratio_by_hand = {}
    closed_finger_count_by_hand = {}
    is_detach_gesture_by_hand = {}

    prev_gesture_by_hand = detach_gesture_state.get("is_detach_gesture_by_hand", {})
    for hand_type in hand_types:
        quest_ok = True
        if quest_missing_ticks is not None and max_missing_quest_ticks is not None:
            quest_ok = quest_missing_ticks.get(hand_type, max_missing_quest_ticks + 1) <= max_missing_quest_ticks

        bend_ratios = get_quest_hand_bend_ratios(T_pose_by_idx, hand_type, finger_names=finger_names)
        valid_ratios = np.asarray(
            [
                bend_ratio
                for bend_ratio in bend_ratios.values()
                if np.isfinite(bend_ratio)
            ],
            dtype=np.float64,
        )
        if not quest_ok or len(valid_ratios) == 0:
            mean_bend_ratio = np.nan
            closed_finger_count = 0
            is_detach_gesture = False
        else:
            was_detach_gesture = bool(prev_gesture_by_hand.get(hand_type, False))
            finger_threshold = float(exit_bend_ratio if was_detach_gesture else enter_bend_ratio)
            mean_threshold = float(mean_exit_bend_ratio if was_detach_gesture else mean_enter_bend_ratio)
            mean_bend_ratio = float(np.mean(valid_ratios))
            closed_finger_count = int(np.sum(valid_ratios >= finger_threshold))
            is_detach_gesture = (
                closed_finger_count >= min_closed_finger_count
                and mean_bend_ratio >= mean_threshold
            )

        bend_ratio_by_hand[hand_type] = bend_ratios
        mean_bend_ratio_by_hand[hand_type] = float(mean_bend_ratio)
        closed_finger_count_by_hand[hand_type] = int(closed_finger_count)
        is_detach_gesture_by_hand[hand_type] = bool(is_detach_gesture)

    gesture_active = all(is_detach_gesture_by_hand[hand_type] for hand_type in hand_types)
    hold_seconds = (
        float(detach_gesture_state.get("hold_seconds", 0.0)) + max(float(dt), 0.0)
        if gesture_active else 0.0
    )
    progress = float(np.clip(hold_seconds / hold_seconds_to_detach, 0.0, 1.0))

    detach_gesture_state.update(
        finger_names=finger_names,
        bend_ratio_by_hand=bend_ratio_by_hand,
        mean_bend_ratio_by_hand=mean_bend_ratio_by_hand,
        closed_finger_count_by_hand=closed_finger_count_by_hand,
        is_detach_gesture_by_hand=is_detach_gesture_by_hand,
        gesture_active=bool(gesture_active),
        hold_seconds=hold_seconds,
        progress=progress,
        should_detach=bool(progress >= 1.0),
    )
    return detach_gesture_state


def apply_sg2_detach_gesture_to_attach_state(
        quest_attach_state,
        hand_types=SG2_HAND_TYPES,
    ):
    for hand_type in hand_types:
        hand_type = normalize_hand_type(hand_type)
        attach_state = quest_attach_state[hand_type]
        attach_state.update(
            phase="detached_by_gesture",
            is_attached=False,
            progress=0.0,
            T_target=None,
            position_error=np.inf,
            rotation_error_rad=np.inf,
            message="Detached by gesture.",
        )
    return quest_attach_state


def virtualize_quest_pose_dict_from_wrist(T_pose_by_idx, T_quest_wrist, T_quest_target, hand_type):
    if not isinstance(T_pose_by_idx, dict):
        return {}

    hand_type = normalize_hand_type(hand_type)
    wrist_idx = get_quest_wrist_joint_idx(hand_type)
    hand_start_idx = max(0, wrist_idx - 1)
    if hand_type == "left":
        idx_range = range(hand_start_idx, int(QuestJointIdx.RIGHT_PALM) - 1)
    else:
        idx_max = max(T_pose_by_idx.keys()) + 1 if len(T_pose_by_idx) > 0 else wrist_idx + 1
        idx_range = range(hand_start_idx, idx_max)

    T_source = T_quest_wrist.get(hand_type, None) if isinstance(T_quest_wrist, dict) else None
    T_target = T_quest_target.get(hand_type, None) if isinstance(T_quest_target, dict) else None
    if T_source is None or T_target is None:
        return {}

    return {
        joint_idx: transform_T_between_frames(T_source, T_target, T_pose_by_idx[joint_idx])
        for joint_idx in idx_range
        if joint_idx in T_pose_by_idx
    }


def virtualize_quest_pose_dicts_from_wrists(
        T_pose_by_idx,
        T_quest_wrist,
        T_quest_target,
        hand_types=SG2_HAND_TYPES,
    ):
    T_virtual_by_idx = {}
    for hand_type in hand_types:
        T_virtual_by_idx.update(
            virtualize_quest_pose_dict_from_wrist(
                T_pose_by_idx=T_pose_by_idx,
                T_quest_wrist=T_quest_wrist,
                T_quest_target=T_quest_target,
                hand_type=hand_type,
            )
        )
    return T_virtual_by_idx


def collect_sg2_arm_joint_names(env):
    joint_groups = group_by_prefix(
        items=env.rev_pri_joint_names,
        prefix_dict={
            "%s_arm_joint_names" % hand_type: [SG2_ARM_JOINT_PREFIXES[hand_type]]
            for hand_type in SG2_HAND_TYPES
        },
    )
    return {
        hand_type: joint_groups["%s_arm_joint_names" % hand_type]
        for hand_type in SG2_HAND_TYPES
    }


def filter_existing_joint_names(env, joint_names):
    joint_set = set(env.rev_pri_joint_names)
    return [joint_name for joint_name in joint_names if joint_name in joint_set]


def get_sg2_gripper_joint_names(hand_type, driver_only=False):
    hand_type = normalize_hand_type(hand_type)
    if driver_only:
        return [SG2_GRIPPER_DRIVER_JOINT_NAMES[hand_type]]
    return list(SG2_GRIPPER_JOINT_NAMES[hand_type])


def get_sg2_gripper_driver_qpos_from_pinch_distance(
        pinch_distance,
        pinch_closed_distance=0.025,
        pinch_open_distance=0.095,
        q_open=0.0,
        q_closed=1.1,
    ):
    if not np.isfinite(pinch_distance):
        return float(q_open), 0.0

    closed_distance = float(pinch_closed_distance)
    open_distance = float(pinch_open_distance)
    if open_distance <= closed_distance:
        raise ValueError("pinch_open_distance must be larger than pinch_closed_distance.")

    closed_ratio = float(np.clip((open_distance - float(pinch_distance)) / (open_distance - closed_distance), 0.0, 1.0))
    q_gripper = float(q_open) + closed_ratio * (float(q_closed) - float(q_open))
    return q_gripper, closed_ratio


def get_sg2_gripper_qpos_from_driver_qpos(driver_qpos):
    return SG2_GRIPPER_JOINT_SIGNS * float(driver_qpos)


def set_sg2_gripper_qpos(env, hand_type, driver_qpos):
    joint_names = get_sg2_gripper_joint_names(hand_type)
    qpos = get_sg2_gripper_qpos_from_driver_qpos(driver_qpos)

    names_use = []
    qpos_use = []
    joint_name_set = set(env.rev_pri_joint_names)
    for joint_name, q in zip(joint_names, qpos):
        if joint_name in joint_name_set:
            names_use.append(joint_name)
            qpos_use.append(float(q))

    if len(names_use) > 0:
        env.forward(q=qpos_use, joint_names=names_use)
    return names_use, np.asarray(qpos_use, dtype=np.float64)


def update_sg2_gripper_from_quest_pinch(
        env,
        T_pose_by_idx,
        hand_types=SG2_HAND_TYPES,
        quest_missing_ticks=None,
        max_missing_quest_ticks=None,
        prev_gripper_q_by_hand=None,
        pinch_closed_distance=0.025,
        pinch_open_distance=0.095,
        q_open=0.0,
        q_closed=1.1,
        smoothing_alpha=None,
    ):
    gripper_q_by_hand = {}
    gripper_info_by_hand = {}
    prev_gripper_q_by_hand = prev_gripper_q_by_hand or {}

    for hand_type in hand_types:
        hand_type = normalize_hand_type(hand_type)
        prev_q = float(prev_gripper_q_by_hand.get(hand_type, q_open))
        quest_ok = True
        if quest_missing_ticks is not None and max_missing_quest_ticks is not None:
            quest_ok = quest_missing_ticks[hand_type] <= max_missing_quest_ticks

        p_thumb_tip, p_index_tip = get_quest_thumb_index_tip_positions(T_pose_by_idx, hand_type)
        pinch_distance = np.nan
        q_gripper = prev_q
        closed_ratio = np.nan
        status = "missing"

        if quest_ok and p_thumb_tip is not None and p_index_tip is not None:
            pinch_distance = float(np.linalg.norm(p_thumb_tip - p_index_tip))
            q_target, closed_ratio = get_sg2_gripper_driver_qpos_from_pinch_distance(
                pinch_distance=pinch_distance,
                pinch_closed_distance=pinch_closed_distance,
                pinch_open_distance=pinch_open_distance,
                q_open=q_open,
                q_closed=q_closed,
            )
            if smoothing_alpha is None:
                q_gripper = q_target
            else:
                alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
                q_gripper = (1.0 - alpha) * prev_q + alpha * q_target
            status = "tracking"
        elif not quest_ok:
            status = "quest_missing"

        set_sg2_gripper_qpos(env, hand_type, q_gripper)
        gripper_q_by_hand[hand_type] = float(q_gripper)
        gripper_info_by_hand[hand_type] = {
            "status": status,
            "pinch_distance": float(pinch_distance),
            "closed_ratio": float(closed_ratio) if np.isfinite(closed_ratio) else np.nan,
            "q_driver": float(q_gripper),
            "p_thumb_tip": None if p_thumb_tip is None else np.asarray(p_thumb_tip, dtype=np.float64).copy(),
            "p_index_tip": None if p_index_tip is None else np.asarray(p_index_tip, dtype=np.float64).copy(),
        }

    return gripper_q_by_hand, gripper_info_by_hand


def build_q_msg_layout(env, q_msg_format_dict, use_joint_names=None, rest_key="_unused"):
    sim_joint_names = list(env.rev_pri_joint_names)
    joint_names_to_group = sim_joint_names if use_joint_names is None else list(use_joint_names)
    q_msg_keys = list(q_msg_format_dict)
    q_msg_prefix_dict = {
        q_msg_key: q_msg_format_dict[q_msg_key]["prefixes"]
        for q_msg_key in q_msg_keys
    }
    qmsgkeys_to_usejointnames_dict = group_by_prefix(
        items=joint_names_to_group,
        prefix_dict=q_msg_prefix_dict,
        rest_key=rest_key,
    )
    unused_joint_names = qmsgkeys_to_usejointnames_dict.pop(rest_key, [])

    if use_joint_names is None:
        use_joint_names = [
            joint_name
            for q_msg_key in q_msg_keys
            for joint_name in qmsgkeys_to_usejointnames_dict[q_msg_key]
        ]
    else:
        use_joint_names = joint_names_to_group

    qmsgkeys_to_usejointidxs_dict = {
        q_msg_key: get_idxs(
            query_list=use_joint_names,
            domain_list=qmsgkeys_to_usejointnames_dict[q_msg_key],
        )
        for q_msg_key in q_msg_keys
    }

    return {
        "q_msg_keys": q_msg_keys,
        "sim_joint_names": sim_joint_names,
        "use_joint_names": use_joint_names,
        "unused_joint_names": unused_joint_names,
        "qmsgkeys_to_usejointnames_dict": qmsgkeys_to_usejointnames_dict,
        "qmsgkeys_to_usejointidxs_dict": qmsgkeys_to_usejointidxs_dict,
    }


def get_qmsg_from_qpos(qpos, q_msg_layout):
    qpos = np.asarray(qpos, dtype=np.float64)
    return {
        q_msg_key: qpos[q_msg_layout["qmsgkeys_to_usejointidxs_dict"][q_msg_key]].tolist()
        for q_msg_key in q_msg_layout["q_msg_keys"]
    }


def get_qpos_from_robot_state(robot_state, q_msg_layout, env=None, q_default=None):
    if q_default is None:
        qpos = env.get_qpos(joint_names=q_msg_layout["use_joint_names"]).copy()
    else:
        qpos = np.asarray(q_default, dtype=np.float64).copy()

    if robot_state is None:
        raise ValueError("Robot State is not received.")

    joints = robot_state.get("joints", {})
    q_name_to_idx = {
        joint_name: idx
        for idx, joint_name in enumerate(q_msg_layout["use_joint_names"])
    }
    for q_msg_key in q_msg_layout["q_msg_keys"]:
        q_msg_state = joints.get(q_msg_key, None)
        if q_msg_state is None:
            continue
        name_to_position = dict(zip(q_msg_state["name"], q_msg_state["position"]))
        for joint_name in q_msg_layout["qmsgkeys_to_usejointnames_dict"][q_msg_key]:
            if joint_name in name_to_position:
                qpos[q_name_to_idx[joint_name]] = float(name_to_position[joint_name])
    return qpos


def get_real_q_from_robot_state(robot_state, q_msg_layout):
    qpos = []
    joint_names = []

    for q_msg_key in q_msg_layout["q_msg_keys"]:
        if q_msg_key not in robot_state["joints"]:
            continue

        q_msg_state = robot_state["joints"][q_msg_key]
        name_to_position = dict(zip(q_msg_state["name"], q_msg_state["position"]))

        for joint_name in q_msg_layout["qmsgkeys_to_usejointnames_dict"][q_msg_key]:
            if joint_name in name_to_position:
                joint_names.append(joint_name)
                qpos.append(float(name_to_position[joint_name]))

    return joint_names, np.asarray(qpos, dtype=np.float64)


def get_vis_q_from_render_msg(render_msg, vis_joint_set, vis_prefix="vis_"):
    joint_names = list(render_msg.get("joint_names", []))
    qpos = np.asarray(render_msg.get("qpos", []), dtype=np.float64)

    if len(joint_names) != len(qpos):
        raise ValueError(
            "render_msg joint_names/qpos length mismatch: %d != %d"
            % (len(joint_names), len(qpos))
        )

    vis_joint_names = []
    vis_qpos = []
    for joint_name, q in zip(joint_names, qpos):
        vis_joint_name = "%s%s" % (vis_prefix, joint_name)
        if vis_joint_name in vis_joint_set:
            vis_joint_names.append(vis_joint_name)
            vis_qpos.append(float(q))

    return vis_joint_names, np.asarray(vis_qpos, dtype=np.float64)


def get_sg2_vis_q_from_render_msg(render_msg, vis_joint_set, vis_prefix="vis_"):
    joint_names = list(render_msg.get("joint_names", []))
    qpos = np.asarray(render_msg.get("qpos", []), dtype=np.float64)

    if len(joint_names) != len(qpos):
        raise ValueError(
            "render_msg joint_names/qpos length mismatch: %d != %d"
            % (len(joint_names), len(qpos))
        )

    vis_q_by_name = {}
    for joint_name, q in zip(joint_names, qpos):
        vis_joint_name = "%s%s" % (vis_prefix, joint_name)
        if vis_joint_name in vis_joint_set:
            vis_q_by_name[vis_joint_name] = float(q)

        for hand_type in SG2_HAND_TYPES:
            if joint_name != SG2_GRIPPER_DRIVER_JOINT_NAMES[hand_type]:
                continue
            for gripper_joint_name, gripper_q in zip(
                SG2_GRIPPER_JOINT_NAMES[hand_type],
                get_sg2_gripper_qpos_from_driver_qpos(float(q)),
            ):
                vis_gripper_joint_name = "%s%s" % (vis_prefix, gripper_joint_name)
                if vis_gripper_joint_name in vis_joint_set:
                    vis_q_by_name[vis_gripper_joint_name] = float(gripper_q)

    vis_joint_names = list(vis_q_by_name)
    vis_qpos = [vis_q_by_name[joint_name] for joint_name in vis_joint_names]
    return vis_joint_names, np.asarray(vis_qpos, dtype=np.float64)


def get_latest_robot_state(sub, timeout_sec=1.0, sleep_sec=1e-4):
    time_start = time.time()
    robot_state_latest = None
    recv_count = 0
    while time.time() - time_start < timeout_sec:
        robot_state = sub.recv()
        if robot_state is not None:
            robot_state_latest = robot_state
            recv_count += 1
        time.sleep(sleep_sec)
    return robot_state_latest, recv_count


def get_latest_robot_qpos(
        env,
        q_msg_layout,
        use_joint_names,
        sub_ip,
        sub_port,
        timeout_sec=1.0,
        verbose=False,
    ):
    sub_robot_init = ZMQPubSub(
        role="sub",
        ip=sub_ip,
        port=sub_port,
        verbose=verbose,
    )
    try:
        robot_state, recv_count = get_latest_robot_state(
            sub=sub_robot_init,
            timeout_sec=timeout_sec,
        )
        if robot_state is None:
            raise ValueError(
                "[Robot init] failed to receive robot state from tcp://%s:%d"
                % (sub_ip, sub_port)
            )
        q_curr = get_qpos_from_robot_state(
            env=env,
            q_msg_layout=q_msg_layout,
            robot_state=robot_state,
            q_default=env.get_qpos(joint_names=use_joint_names),
        )
    finally:
        sub_robot_init.close()

    return q_curr, recv_count


def send_render_qpos(
        pub_render,
        qpos,
        joint_names,
        q_msg_layout,
        msg_type,
        on_flag,
        robot_hz,
        render_hz,
        robot_pub_hz_actual,
        extra=None,
    ):
    obj = {
        "msg_type": msg_type,
        "time": time.time(),
        "joint_names": joint_names,
        "qpos": np.asarray(qpos, dtype=np.float64).copy(),
        "q_names_dict": q_msg_layout["qmsgkeys_to_usejointnames_dict"],
        "pub_robot": on_flag,
        "robot_hz": robot_hz,
        "render_hz": render_hz,
        "robot_pub_hz_actual": robot_pub_hz_actual,
    }
    if extra is not None:
        obj.update(extra)
    pub_render.send(obj=obj)


def quest_missing_tick_limit(input_hz, missing_grace_seconds=1.0):
    return int(np.ceil(max(float(missing_grace_seconds), 0.0) * max(float(input_hz), 0.0)))


def _pose_error(T_target, T_curr):
    T_target = np.asarray(T_target, dtype=np.float64).reshape(4, 4)
    T_curr = np.asarray(T_curr, dtype=np.float64).reshape(4, 4)
    p_err = float(np.linalg.norm(t2p(T_target) - t2p(T_curr)))
    R_err = t2r(T_target) @ t2r(T_curr).T
    c = float(np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0))
    return p_err, float(np.arccos(c))


def _blend_T(T_from, T_to, alpha):
    T_from = np.asarray(T_from, dtype=np.float64).reshape(4, 4)
    T_to = np.asarray(T_to, dtype=np.float64).reshape(4, 4)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    p = (1.0 - alpha) * t2p(T_from) + alpha * t2p(T_to)
    R_from = t2r(T_from)
    R_to = t2r(T_to)
    R = R_from @ w2r(alpha * r2w(R_from.T @ R_to))
    return pr2t(p=p, R=R)


def _apply_world_offset(T, R_offset, p_offset):
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R_offset = np.asarray(R_offset, dtype=np.float64).reshape(3, 3)
    p_offset = np.asarray(p_offset, dtype=np.float64).reshape(3)
    return pr2t(
        p=R_offset @ t2p(T) + p_offset,
        R=R_offset @ t2r(T),
    )


def _world_offset_from_to(T_source, T_target):
    T_source = np.asarray(T_source, dtype=np.float64).reshape(4, 4)
    T_target = np.asarray(T_target, dtype=np.float64).reshape(4, 4)
    R_offset = t2r(T_target) @ t2r(T_source).T
    p_offset = t2p(T_target) - R_offset @ t2p(T_source)
    return R_offset, p_offset


def create_quest_wrist_attach_state(
        hand_type,
        attach_seconds=2.5,
        attach_start_distance=0.2,
    ):
    hand_type = normalize_hand_type(hand_type)
    return {
        "hand_type": hand_type,
        "phase": "idle",
        "is_attached": False,
        "progress": 0.0,
        "attach_seconds": float(max(attach_seconds, 1e-6)),
        "attach_start_distance": float(attach_start_distance),
        "R_offset": np.eye(3, dtype=np.float64),
        "p_offset": np.zeros(3, dtype=np.float64),
        "T_target": None,
        "position_error": np.inf,
        "rotation_error_rad": np.inf,
        "message": "Waiting for %s Quest wrist." % hand_type,
    }


def update_quest_wrist_attach_state(
        attach_state,
        T_quest_target_raw,
        T_robot_wrist,
        dt,
        quest_updated=True,
        hand_type="right",
        attach_seconds=2.5,
        attach_start_distance=0.2,
    ):
    hand_type = normalize_hand_type(hand_type)
    if attach_state is None:
        attach_state = create_quest_wrist_attach_state(
            hand_type=hand_type,
            attach_seconds=attach_seconds,
            attach_start_distance=attach_start_distance,
        )

    attach_state["hand_type"] = hand_type
    attach_state["attach_seconds"] = float(max(attach_seconds, 1e-6))
    attach_state["attach_start_distance"] = float(attach_start_distance)

    if T_quest_target_raw is None or T_robot_wrist is None:
        attach_state.update(
            phase="idle",
            is_attached=False,
            progress=0.0,
            T_target=None,
            position_error=np.inf,
            rotation_error_rad=np.inf,
            message="Waiting for %s Quest wrist." % hand_type,
        )
        return None, attach_state

    T_quest_target_raw = np.asarray(T_quest_target_raw, dtype=np.float64).reshape(4, 4)
    T_robot_wrist = np.asarray(T_robot_wrist, dtype=np.float64).reshape(4, 4)
    raw_position_distance = float(np.linalg.norm(t2p(T_robot_wrist) - t2p(T_quest_target_raw)))

    if attach_state.get("phase") == "attached" and attach_state.get("is_attached", False):
        T_target = _apply_world_offset(
            T=T_quest_target_raw,
            R_offset=attach_state["R_offset"],
            p_offset=attach_state["p_offset"],
        )
        attach_state["T_target"] = T_target
        attach_state["position_error"], attach_state["rotation_error_rad"] = _pose_error(
            T_target,
            T_robot_wrist,
        )
        attach_state["message"] = "Attached."
        return T_target, attach_state

    if raw_position_distance > attach_state["attach_start_distance"]:
        attach_state.update(
            phase="waiting_close",
            is_attached=False,
            progress=0.0,
            T_target=None,
            position_error=raw_position_distance,
            rotation_error_rad=np.inf,
            message="Move %s wrist within %.0fcm. Current %.1fcm."
            % (
                hand_type,
                100.0 * attach_state["attach_start_distance"],
                100.0 * raw_position_distance,
            ),
        )
        return None, attach_state

    if attach_state.get("phase") != "attaching":
        attach_state["phase"] = "attaching"
        attach_state["progress"] = 0.0

    attach_state["is_attached"] = False
    attach_state["progress"] = min(
        1.0,
        float(attach_state["progress"])
        + max(float(dt), 0.0) / attach_state["attach_seconds"],
    )
    T_target = _blend_T(T_from=T_quest_target_raw, T_to=T_robot_wrist, alpha=attach_state["progress"])
    position_error, rotation_error_rad = _pose_error(T_target, T_robot_wrist)
    attach_state["T_target"] = T_target
    attach_state["position_error"] = position_error
    attach_state["rotation_error_rad"] = rotation_error_rad

    if attach_state["progress"] >= 1.0:
        R_offset, p_offset = _world_offset_from_to(T_quest_target_raw, T_robot_wrist)
        attach_state["R_offset"] = R_offset
        attach_state["p_offset"] = p_offset
        attach_state["phase"] = "attached"
        attach_state["is_attached"] = True
        attach_state["progress"] = 1.0
        attach_state["T_target"] = _apply_world_offset(
            T=T_quest_target_raw,
            R_offset=R_offset,
            p_offset=p_offset,
        )
        attach_state["position_error"], attach_state["rotation_error_rad"] = _pose_error(
            attach_state["T_target"],
            T_robot_wrist,
        )
        attach_state["message"] = "Attached."
        return attach_state["T_target"], attach_state

    stale_text = "" if bool(quest_updated) else " stale"
    attach_state["message"] = "Attaching%s %.0f%% | pos %.3fm | rot %.1fdeg" % (
        stale_text,
        100.0 * attach_state["progress"],
        position_error,
        np.rad2deg(rotation_error_rad),
    )
    return T_target, attach_state


def is_quest_wrist_jump_rejected(T_wrist_prev, T_wrist_new, hand_type, check_distance=0.10):
    normalize_hand_type(hand_type)
    if check_distance is None or T_wrist_prev is None or T_wrist_new is None:
        return False
    T_target_prev = compute_quest_target_T_from_wrist_transform(T_wrist_prev, hand_type)
    T_target_new = compute_quest_target_T_from_wrist_transform(T_wrist_new, hand_type)
    distance = float(np.linalg.norm(t2p(T_target_new) - t2p(T_target_prev)))
    return distance > float(check_distance)


def quest_pose_delta(T_prev, T_now):
    if T_prev is None or T_now is None:
        return np.inf, np.inf
    T_prev = np.asarray(T_prev, dtype=np.float64).reshape(4, 4)
    T_now = np.asarray(T_now, dtype=np.float64).reshape(4, 4)
    p_delta = float(np.linalg.norm(t2p(T_now) - t2p(T_prev)))
    R_delta = t2r(T_prev).T @ t2r(T_now)
    c = float(np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0))
    return p_delta, float(np.arccos(c))


def update_sg2_wrist_attach_targets(
        env,
        hand_types,
        T_quest_wrist,
        quest_missing_ticks,
        quest_updated,
        quest_attach_state,
        last_solved_wrist_target,
        max_missing_quest_ticks,
        dt,
        attach_seconds=2.5,
        attach_start_distance=0.20,
    ):
    T_raw_wrist_target = {}
    T_attach_wrist_target = {}
    T_virtual_quest_wrist = {}
    attached_wrist_targets = {}
    wrist_target_motion = {}

    for hand_type in hand_types:
        wrist_target_motion[hand_type] = (np.inf, np.inf)

        quest_ok = quest_missing_ticks[hand_type] <= max_missing_quest_ticks
        T_robot_wrist = get_sg2_wrist_control_T(env, hand_type)
        T_raw_wrist_target[hand_type] = (
            compute_quest_target_T_from_wrist_transform(T_quest_wrist[hand_type], hand_type)
            if quest_ok else None
        )
        T_attach_wrist_target[hand_type], quest_attach_state[hand_type] = update_quest_wrist_attach_state(
            attach_state=quest_attach_state[hand_type],
            T_quest_target_raw=T_raw_wrist_target[hand_type],
            T_robot_wrist=T_robot_wrist,
            dt=dt,
            quest_updated=quest_updated[hand_type],
            hand_type=hand_type,
            attach_seconds=attach_seconds,
            attach_start_distance=attach_start_distance,
        )
        T_virtual_quest_wrist[hand_type] = transform_T_between_frames(
            T_source=T_quest_wrist[hand_type],
            T_target=T_attach_wrist_target[hand_type],
            T=T_quest_wrist[hand_type],
        )

        if not quest_attach_state[hand_type]["is_attached"] or T_virtual_quest_wrist[hand_type] is None:
            last_solved_wrist_target[hand_type] = None
            continue

        wrist_target_motion[hand_type] = quest_pose_delta(
            last_solved_wrist_target[hand_type],
            T_virtual_quest_wrist[hand_type],
        )
        attached_wrist_targets[hand_type] = T_virtual_quest_wrist[hand_type]

    return (
        T_raw_wrist_target,
        T_attach_wrist_target,
        T_virtual_quest_wrist,
        attached_wrist_targets,
        quest_attach_state,
        last_solved_wrist_target,
        wrist_target_motion,
    )


def update_last_solved_wrist_targets(last_solved_wrist_target, arm_ik_info_by_hand, attached_wrist_targets, ik_info):
    for hand_type, T_target in attached_wrist_targets.items():
        arm_ik_info_by_hand[hand_type] = ik_info
        last_solved_wrist_target[hand_type] = np.asarray(T_target, dtype=np.float64).copy()
    return last_solved_wrist_target, arm_ik_info_by_hand


def add_sg2_wrist_ik_coordinate_target(
        ik_solver,
        env,
        hand_type,
        T_target,
        weight=1.0,
        axis_width=0.10,
        kind=None,
    ):
    hand_type = normalize_hand_type(hand_type)
    T_curr = get_sg2_wrist_control_T(env, hand_type)
    ik_solver.add_ik_coordinate_target(
        body_name=SG2_WRIST_BODY_NAMES[hand_type],
        T_curr=T_curr,
        T_trgt=np.asarray(T_target, dtype=np.float64).reshape(4, 4),
        weight=weight,
        aw=axis_width,
        kind="%s_sg2_wrist" % hand_type if kind is None else kind,
    )


def create_sg2_arm_ik_solvers(env, mujoco_parser_cls, body_ik_solver_cls):
    arm_joint_names = collect_sg2_arm_joint_names(env)
    solvers = {}
    for hand_type in SG2_HAND_TYPES:
        joints_use = arm_joint_names[hand_type]
        dls_damping = 1e-6 * np.ones(len(joints_use), dtype=np.float64)
        if len(dls_damping) > 0:
            dls_damping[-1] = 1e-4
        q_home_rev_pri = env.get_qpos(joint_names=env.rev_pri_joint_names)
        solver = body_ik_solver_cls(
            name="%s SG2 arm IK" % hand_type,
            ik_env=mujoco_parser_cls(rel_xml_path=env.rel_xml_path, verbose=False),
            joints_use=joints_use,
            max_ik_tick=8,
            ik_stepsize_rev=np.deg2rad(8.0),
            ik_stepsize_pri=0.05,
            ik_update_th_rev=np.deg2rad(8.0),
            ik_update_th_pri=0.05,
            dls_damping=dls_damping,
            max_probe_rev=np.deg2rad(3.0),
            max_probe_pri=0.01,
            k_null=0.5,
            q_home_rev_pri=q_home_rev_pri,
        )
        # Keep the startup posture: the "self_motion" null reference anchors the
        # arm's unobservable coordinate to it, while every tick overwrites
        # q_home_rev_pri itself.  "previous" and "home" ignore this copy.
        solver.q_null_anchor_rev_pri = np.array(q_home_rev_pri, dtype=np.float64, copy=True)
        solvers[hand_type] = solver
    return solvers, arm_joint_names


def create_sg2_collision_aware_arm_ik_solvers(
        env,
        mujoco_parser_cls,
        body_ik_solver_cls,
        enable_depenetration=True,
    ):
    arm_joint_names = collect_sg2_arm_joint_names(env)
    joints_use = filter_existing_joint_names(env, SG2_COLLISION_IK_JOINT_NAMES)

    tracking_solver = body_ik_solver_cls(
        name="SG2 collision-aware tracking IK",
        ik_env=mujoco_parser_cls(rel_xml_path=env.rel_xml_path, verbose=False),
        joints_use=joints_use,
        max_ik_tick=8,
        ik_stepsize_rev=np.deg2rad(8.0),
        ik_stepsize_pri=0.05,
        ik_update_th_rev=np.deg2rad(8.0),
        ik_update_th_pri=0.05,
        max_probe_rev=np.deg2rad(3.0),
        max_probe_pri=0.01,
        k_null=0.5,
        q_home_rev_pri=env.get_qpos(joint_names=env.rev_pri_joint_names),
    )
    solvers = {"tracking": tracking_solver}
    if not enable_depenetration:
        return solvers, arm_joint_names

    depenetration_solver = body_ik_solver_cls(
        name="SG2 depenetration IK",
        ik_env=mujoco_parser_cls(rel_xml_path=env.rel_xml_path, verbose=False),
        joints_use=joints_use,
        max_ik_tick=1,
        ik_stepsize_rev=np.deg2rad(8.0),
        ik_stepsize_pri=0.05,
        ik_update_th_rev=np.deg2rad(8.0),
        ik_update_th_pri=0.05,
        max_probe_rev=np.deg2rad(3.0),
        max_probe_pri=0.01,
        k_null=0.0,
        q_home_rev_pri=env.get_qpos(joint_names=env.rev_pri_joint_names),
    )
    solvers["depenetration"] = depenetration_solver
    return solvers, arm_joint_names


def create_sg2_collision_candidate_geom_pairs(
        env,
        init_exclude_dist=0.05,
        exclude_subtree_root_body_names=None,
    ):
    q_backup = env.get_qpos(joint_names=env.rev_pri_joint_names)
    try:
        env.forward(
            q=[np.deg2rad(90.0), np.deg2rad(-90.0)],
            joint_names=["arm_l_joint2", "arm_r_joint2"],
        )
        candidate_info = env.get_root_geom_pair_candidates(
            root_body_name="base_link",
            init_exclude_dist=init_exclude_dist,
            exclude_subtree_root_body_names=exclude_subtree_root_body_names,
            forward=True,
        )
    finally:
        env.forward(q=q_backup, joint_names=env.rev_pri_joint_names)

    return candidate_info["candidate_geom_pairs"], candidate_info


def print_sg2_collision_candidate_summary(candidate_info):
    stats = candidate_info["stats"]
    print("Collision candidates")
    print("--------------------")
    print("robot collision geom:[%d]" % stats["n_root_geom"])
    print("collision geom:[%d]" % stats["n_collision_geom"])
    print("candidate geom pairs:[%d]" % stats["n_candidate_geom_pair"])
    print("excluded initial geom pairs:[%d]" % stats["n_excluded_geom_pair"])


def make_idle_collision_ik_result(active_solver, safe_margin=0.05):
    return {
        "mode": "idle",
        "solver": active_solver,
        "ik_info": None,
        "signed_distance_infos": [],
        "min_signed_dist": np.nan,
        "safe_margin": float(safe_margin),
        "depenetration_target_info": None,
        "track_weight": 0.0,
        "upper_arm_yaw_info": None,
        "null_reference": None,
        "relax_upper_arm_yaw": None,
    }


def get_sg2_arm_self_motion_direction(env, joint_names, hand_type):
    """Return the unit joint direction the wrist control pose cannot see.

    A seven-joint arm holding a six-dimensional pose has exactly one such
    direction, its self-motion.  On this robot it is dominated by "joint3 up /
    joint5 down" - the two rolls that sit on either side of the elbow - but the
    other components matter away from the straight-elbow poses, so it is solved
    per tick instead of assumed.

    Returns ``None`` when the arm is not redundant for this target or the model
    produces a non-finite Jacobian, so callers can fall back.
    """
    hand_type = normalize_hand_type(hand_type)
    model = env.model
    cache = getattr(env, "_sg2_self_motion_cache", None)
    if cache is None or cache[0] is not model:
        cache = env._sg2_self_motion_cache = (model, mujoco.MjData(model))
    data = cache[1]
    # Read the pose on a private MjData so the caller's own state is untouched.
    data.qpos[:] = env.data.qpos
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, SG2_WRIST_BODY_NAMES[hand_type])
    point = data.xpos[body] + data.xmat[body].reshape(3, 3) @ SG2_WRIST_P_LOCAL[hand_type]
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jac(model, data, jacp, jacr, point, body)
    columns = [int(model.joint(name).dofadr[0]) for name in joint_names]
    jacobian = np.vstack([jacp[:, columns], jacr[:, columns]])
    if jacobian.shape[0] >= jacobian.shape[1] or not np.isfinite(jacobian).all():
        return None
    direction = np.linalg.svd(jacobian)[2][-1]
    norm = float(np.linalg.norm(direction))
    return None if norm < 1e-12 else direction / norm


def _sg2_self_motion_reference(ik_solver, env, q_rev_pri):
    """Restore only the self-motion coordinate toward the solver's anchor pose."""
    anchor = getattr(ik_solver, "q_null_anchor_rev_pri", None)
    joint_names = list(getattr(ik_solver, "joints_use", ()))
    hand_type = next(
        (hand for hand in SG2_HAND_TYPES if list(SG2_ARM_JOINT_NAMES[hand]) == joint_names),
        None,
    )
    if anchor is None or hand_type is None:
        # Not a single-arm SG2 solver, or no anchor was captured: keep "previous".
        return q_rev_pri
    direction = get_sg2_arm_self_motion_direction(env, joint_names, hand_type)
    if direction is None:
        return q_rev_pri
    index = [env.rev_pri_joint_names.index(name) for name in joint_names]
    offset = np.asarray(anchor, dtype=np.float64)[index] - q_rev_pri[index]
    reference = np.array(q_rev_pri, dtype=np.float64, copy=True)
    reference[index] += float(offset @ direction) * direction
    return reference


def set_sg2_null_space_reference(ik_solver, env, null_reference="previous"):
    """Choose what the solver's null-space term pulls the arm toward.

    The solver reads ``q_home_rev_pri`` on every solve tick, on both the Python
    and the compiled backend path, so rebinding the attribute is enough to
    change the null-space objective without touching the solver package.

    "previous" points the objective at the pose the solver ended on last tick,
    which turns the null-space term into "move the joints as little as
    possible".  "home" leaves the startup snapshot captured when the solver was
    created, i.e. the original behavior.

    "self_motion" points it at the current pose corrected along the arm's
    self-motion direction only, so the one coordinate the end-effector cannot
    see is held near the startup posture while every task-relevant component is
    left to the IK.  "previous" leaves that coordinate with no restoring term at
    all, which let joint3 and joint5 wind up to nearly +-180 degrees in the
    2026-09-15 REAL logs while the tracking error stayed under a millimetre.
    """
    if null_reference not in SG2_NULL_REFERENCE_MODES:
        raise ValueError(
            "null_reference must be one of %s." % (SG2_NULL_REFERENCE_MODES,)
        )
    if null_reference == "home":
        return None

    q_prev_rev_pri = env.get_qpos(joint_names=env.rev_pri_joint_names)
    if null_reference == "self_motion":
        q_prev_rev_pri = _sg2_self_motion_reference(ik_solver, env, q_prev_rev_pri)
    ik_solver.q_home_rev_pri = q_prev_rev_pri
    return q_prev_rev_pri


def relax_sg2_upper_arm_yaw_joints(
        env,
        joint_names=SG2_UPPER_ARM_YAW_JOINT_NAMES,
        home_position=0.0,
        max_step_rad=np.deg2rad(1.0),
    ):
    joint_names = filter_existing_joint_names(env, joint_names)
    if len(joint_names) == 0:
        return {
            "joint_names": [],
            "q_prev": np.zeros(0),
            "q_next": np.zeros(0),
        }

    q_prev = env.get_qpos(joint_names=joint_names)
    q_home = np.full_like(q_prev, float(home_position))
    q_next = q_prev + np.clip(
        q_home - q_prev,
        -float(max_step_rad),
        float(max_step_rad),
    )
    env.forward(q=q_next, joint_names=joint_names)
    return {
        "joint_names": joint_names,
        "q_prev": q_prev,
        "q_next": q_next,
    }


def get_solver_target_infos(solver, env):
    if hasattr(solver, "get_target_infos"):
        return solver.get_target_infos(env=env)

    targets = []
    for target_idx in range(solver.n_target):
        body_id = int(solver.body_idx_list[target_idx])
        p_local = np.asarray(solver.p_local_list[target_idx], dtype=np.float64)
        targets.append(
            {
                "label": "target_%d" % target_idx,
                "body_name": env.body_names[body_id],
                "p_curr": solver._p_world_from_local(env.data, body_id, p_local),
                "p_trgt": np.asarray(solver.p_trgt_list[target_idx], dtype=np.float64).copy(),
                "weight": float(solver.weight_list[target_idx]),
                "kind": None,
            }
        )
    return targets


def _normalize_wrist_targets(wrist_targets_by_hand):
    if not isinstance(wrist_targets_by_hand, dict):
        return {}
    return {
        normalize_hand_type(hand_type): np.asarray(T_target, dtype=np.float64).reshape(4, 4)
        for hand_type, T_target in wrist_targets_by_hand.items()
        if T_target is not None
    }


def solve_sg2_collision_aware_wrist_ik(
        env,
        tracking_solver,
        depenetration_solver,
        wrist_targets_by_hand,
        collision_geom_pairs,
        native_backend,
        safe_margin=0.05,
        signed_distance_topk=5,
        dep_step=0.01,
        enable_depenetration=True,
        null_reference="previous",
        relax_upper_arm_yaw=True,
    ):
    wrist_targets_by_hand = _normalize_wrist_targets(wrist_targets_by_hand)
    signed_distance_infos, min_signed_dist = env.get_geom_pair_signed_distance_infos_bind(
        collision_geom_pairs,
        topk=signed_distance_topk,
        be=native_backend,
        return_min_signed_dist=True,
    )
    safe_margin = float(safe_margin)
    is_safe = (not np.isfinite(min_signed_dist)) or (float(min_signed_dist) > safe_margin)

    out = {
        "mode": "idle",
        "solver": tracking_solver,
        "ik_info": None,
        "signed_distance_infos": signed_distance_infos,
        "min_signed_dist": float(min_signed_dist),
        "safe_margin": safe_margin,
        "depenetration_target_info": None,
        "track_weight": 0.0,
        "upper_arm_yaw_info": None,
        "null_reference": null_reference,
        "relax_upper_arm_yaw": bool(relax_upper_arm_yaw),
    }
    if len(wrist_targets_by_hand) == 0:
        tracking_solver.reset_buffers(env=env)
        if depenetration_solver is not None:
            depenetration_solver.reset_buffers(env=env)
        return out

    if is_safe or (not enable_depenetration) or depenetration_solver is None:
        out["mode"] = "tracking" if is_safe else "tracking_unsafe"
        out["solver"] = tracking_solver
        set_sg2_null_space_reference(tracking_solver, env, null_reference)
        if relax_upper_arm_yaw:
            out["upper_arm_yaw_info"] = relax_sg2_upper_arm_yaw_joints(env)
        tracking_solver.reset_buffers(env=env)
        for hand_type, T_target in wrist_targets_by_hand.items():
            add_sg2_wrist_ik_coordinate_target(
                ik_solver=tracking_solver,
                env=env,
                hand_type=hand_type,
                T_target=T_target,
                weight=1.0,
                axis_width=0.10,
                kind="tracking",
            )
        _, ik_info = tracking_solver.solve_ik_bind(
            env=env,
            joint_limit_handle_flag=True,
            nullspace_control_flag=True,
            task_null_mask_flag=True,
            base_control_flag=False,
            be=native_backend,
        )
    else:
        out["mode"] = "depenetration"
        out["solver"] = depenetration_solver
        set_sg2_null_space_reference(depenetration_solver, env, null_reference)
        depenetration_solver.reset_buffers(env=env)
        dep_targets, dep_target_info = env.get_dep_trgts_from_signed_dists(
            signed_distance_infos,
            root_body_name="base_link",
            safe_margin=safe_margin,
            dep_step=dep_step,
        )
        depenetration_solver.add_body_trgts(dep_targets, kind="depenetration")
        track_weight = float(
            np.clip(
                (float(min_signed_dist) - 0.5 * safe_margin) / (0.5 * safe_margin),
                1e-3,
                1.0,
            )
        )
        for hand_type, T_target in wrist_targets_by_hand.items():
            add_sg2_wrist_ik_coordinate_target(
                ik_solver=depenetration_solver,
                env=env,
                hand_type=hand_type,
                T_target=T_target,
                weight=track_weight,
                axis_width=0.10,
                kind="tracking",
            )
        _, ik_info = depenetration_solver.solve_ik_bind(
            env=env,
            joint_limit_handle_flag=True,
            nullspace_control_flag=True,
            task_null_mask_flag=True,
            base_control_flag=False,
            be=native_backend,
        )
        out["depenetration_target_info"] = dep_target_info
        out["track_weight"] = track_weight

    env.forward(q=ik_info["qpos_used_best"], joint_names=out["solver"].joints_use)
    out["ik_info"] = ik_info
    return out


def make_signed_distance_draw_cmds(rp, signed_distance_infos):
    draw_cmds = []
    for info in signed_distance_infos:
        rgba = (1.0, 0.0, 0.0, 0.90) if info["signed_dist"] < 0.0 else (0.0, 0.0, 1.0, 0.75)
        draw_cmds.append(rp.draw("plot_line_fr2to", p_fr=info["p1"], p_to=info["p2"], rgba=rgba))
        draw_cmds.append(rp.draw("plot_sphere", p=info["p1"], r=0.002, rgba=rgba))
        draw_cmds.append(rp.draw("plot_sphere", p=info["p2"], r=0.002, rgba=rgba))
    return draw_cmds


def make_body_ik_target_draw_cmds(
        rp,
        solver,
        env,
        r=0.004,
        alpha=0.55,
        rgba=(0.0, 0.4, 1.0, 0.6),
        plot_arrows=True,
    ):
    draw_cmds = []
    n_target = int(getattr(solver, "n_target", len(getattr(solver, "p_trgt_list", []))))
    if n_target <= 0:
        return draw_cmds

    for target_idx in range(n_target):
        p_trgt = np.asarray(solver.p_trgt_list[target_idx], dtype=np.float64)
        body_id = int(solver.body_idx_list[target_idx])
        p_local = np.asarray(solver.p_local_list[target_idx], dtype=np.float64)
        p_curr = solver._p_world_from_local(env.data, body_id, p_local)
        rgba_i = list(rgba)
        rgba_i[3] = alpha
        draw_cmds.append(rp.draw("plot_sphere", p=p_curr, r=r, rgba=rgba_i))
        draw_cmds.append(rp.draw("plot_sphere", p=p_trgt, r=r, rgba=rgba_i))
        if plot_arrows:
            draw_cmds.append(rp.draw("plot_arrow_fr2to", p_fr=p_curr, p_to=p_trgt, r=0.5 * r, rgba=rgba_i))
        else:
            draw_cmds.append(rp.draw("plot_line_fr2to", p_fr=p_curr, p_to=p_trgt, rgba=rgba_i))
    return draw_cmds


def make_sg2_attach_overlay_text(hand_types, hand_labels, quest_attach_state):
    attach_parts = []
    for hand_type in hand_types:
        label = hand_labels[hand_type]
        attach_state = quest_attach_state[hand_type] or {"phase": "idle", "progress": 0.0}
        attach_parts.append(
            "%s %s %.0f%%" % (
                label,
                attach_state["phase"],
                100.0 * attach_state["progress"],
            )
        )
    return " | ".join(attach_parts)


def make_sg2_collision_overlay_text(collision_ik_result):
    return "%s %.1f/%.1fmm" % (
        collision_ik_result.get("mode", "idle"),
        1000.0 * collision_ik_result.get("min_signed_dist", np.nan),
        1000.0 * collision_ik_result.get("safe_margin", 0.05),
    )


def make_sg2_ik_overlay_text(collision_ik_result):
    ik_info = collision_ik_result.get("ik_info", None)
    ik_error_text = "-" if ik_info is None else "%.4f" % ik_info["ik_err_best"]
    return "%s err %s" % (collision_ik_result.get("mode", "idle"), ik_error_text)


def make_sg2_detach_gesture_overlay_text(hand_types, hand_labels, detach_gesture_state):
    if not isinstance(detach_gesture_state, dict):
        return "-"

    finger_count = len(detach_gesture_state.get("finger_names", ("middle", "ring", "little")))
    mean_bend_ratio_by_hand = detach_gesture_state.get("mean_bend_ratio_by_hand", {})
    closed_finger_count_by_hand = detach_gesture_state.get("closed_finger_count_by_hand", {})
    parts = []
    for hand_type in hand_types:
        label = hand_labels[hand_type]
        mean_bend_ratio = mean_bend_ratio_by_hand.get(hand_type, np.nan)
        closed_finger_count = closed_finger_count_by_hand.get(hand_type, 0)
        if np.isfinite(mean_bend_ratio):
            parts.append(
                "%s %.0f%% %d/%d" % (
                    label,
                    100.0 * mean_bend_ratio,
                    int(closed_finger_count),
                    finger_count,
                )
            )
        else:
            parts.append("%s -" % label)

    progress = float(detach_gesture_state.get("progress", 0.0))
    wait_text = "detached" if detach_gesture_state.get("should_detach", False) else "wait"
    parts.append("%s %.0f%%" % (wait_text, 100.0 * progress))
    return " | ".join(parts)


def make_sg2_gripper_overlay_text(hand_types, hand_labels, gripper_info_by_hand):
    parts = []
    for hand_type in hand_types:
        label = hand_labels[hand_type]
        info = gripper_info_by_hand.get(hand_type, {}) if isinstance(gripper_info_by_hand, dict) else {}
        closed_ratio = info.get("closed_ratio", np.nan)
        pinch_distance = info.get("pinch_distance", np.nan)
        if np.isfinite(closed_ratio) and np.isfinite(pinch_distance):
            parts.append("%s %.0f%% %.0fmm" % (label, 100.0 * closed_ratio, 1000.0 * pinch_distance))
        else:
            parts.append("%s -" % label)
    return " | ".join(parts)


def make_sg2_gripper_draw_cmds(rp, gripper_info_by_hand):
    draw_cmds = []
    if not isinstance(gripper_info_by_hand, dict):
        return draw_cmds

    for info in gripper_info_by_hand.values():
        p_thumb_tip = info.get("p_thumb_tip", None)
        p_index_tip = info.get("p_index_tip", None)
        closed_ratio = float(info.get("closed_ratio", 0.0))
        if p_thumb_tip is None or p_index_tip is None:
            continue

        rgba = (
            float(np.clip(closed_ratio, 0.0, 1.0)),
            float(np.clip(1.0 - closed_ratio, 0.0, 1.0)),
            0.0,
            0.85,
        )
        draw_cmds.append(rp.draw("plot_line_fr2to", p_fr=p_thumb_tip, p_to=p_index_tip, rgba=rgba))
        draw_cmds.append(rp.draw("plot_sphere", p=p_thumb_tip, r=0.004, rgba=rgba))
        draw_cmds.append(rp.draw("plot_sphere", p=p_index_tip, r=0.004, rgba=rgba))
    return draw_cmds


def copy_sg2_detach_gesture_state(detach_gesture_state):
    if not isinstance(detach_gesture_state, dict):
        return None
    return {
        "finger_names": tuple(detach_gesture_state.get("finger_names", ("middle", "ring", "little"))),
        "mean_bend_ratio_by_hand": {
            hand_type: float(v)
            for hand_type, v in detach_gesture_state.get("mean_bend_ratio_by_hand", {}).items()
        },
        "closed_finger_count_by_hand": {
            hand_type: int(v)
            for hand_type, v in detach_gesture_state.get("closed_finger_count_by_hand", {}).items()
        },
        "is_detach_gesture_by_hand": {
            hand_type: bool(v)
            for hand_type, v in detach_gesture_state.get("is_detach_gesture_by_hand", {}).items()
        },
        "gesture_active": bool(detach_gesture_state.get("gesture_active", False)),
        "hold_seconds": float(detach_gesture_state.get("hold_seconds", 0.0)),
        "progress": float(detach_gesture_state.get("progress", 0.0)),
        "should_detach": bool(detach_gesture_state.get("should_detach", False)),
    }


def make_sg2_metaquest_draw_cmds(
        rp,
        env,
        hand_types,
        hand_labels,
        quest_missing_ticks,
        T_quest_wrist,
        T_virtual_quest_wrist,
        collision_ik_result,
        active_ik_solver,
        plot_signed_distance=True,
        plot_ik_targets=True,
    ):
    draw_cmds = []
    if plot_signed_distance:
        draw_cmds.extend(
            make_signed_distance_draw_cmds(
                rp=rp,
                signed_distance_infos=collision_ik_result.get("signed_distance_infos", []),
            )
        )

    for hand_type in hand_types:
        label = hand_labels[hand_type]
        draw_cmds.append(
            rp.draw(
                "plot_T",
                T=get_sg2_wrist_control_T(env, hand_type),
                label="%s SG2 wrist" % label,
                axis_len=0.06,
                axis_width=0.0025,
                axis_alpha=0.65,
            )
        )
        if quest_missing_ticks[hand_type] == 0 and T_quest_wrist[hand_type] is not None:
            draw_cmds.append(
                rp.draw(
                    "plot_T",
                    T=T_quest_wrist[hand_type],
                    label="%s Quest wrist" % label,
                    axis_len=0.08,
                    axis_width=0.004,
                    axis_alpha=0.85,
                )
            )
        if T_virtual_quest_wrist[hand_type] is not None:
            draw_cmds.append(
                rp.draw(
                    "plot_T",
                    T=T_virtual_quest_wrist[hand_type],
                    label="%s virtual wrist" % label,
                    axis_len=0.06,
                    axis_width=0.0025,
                    axis_alpha=0.55,
                )
            )

    if plot_ik_targets:
        draw_cmds.extend(
            make_body_ik_target_draw_cmds(
                rp=rp,
                solver=active_ik_solver,
                env=env,
                r=0.004,
                alpha=0.50,
                rgba=(0.0, 0.4, 1.0, 0.6),
            )
        )
    return draw_cmds


def make_sg2_real_render_extra(
        hand_types,
        raw_robot_command_on,
        attached,
        quest_attach_state,
        quest_missing_ticks,
        quest_all_missing_ticks,
        max_missing_quest_ticks,
        collision_ik_result,
        T_quest_start,
        T_quest_pose_by_idx,
        T_virtual_quest_pose_by_idx,
        T_quest_wrist,
        T_virtual_quest_wrist,
        T_raw_wrist_target,
        T_attach_wrist_target,
        T_sg2_wrist,
        active_arm_ik_targets,
        arm_ik_info_by_hand,
        wrist_target_motion,
        detach_gesture_state=None,
        plot_quest_nodes=True,
        plot_arm_ik_targets=True,
    ):
    return {
        "raw_robot_command_on": raw_robot_command_on,
        "attached": attached,
        "attach_phase": {
            hand_type: quest_attach_state[hand_type]["phase"]
            for hand_type in hand_types
        },
        "attach_progress": {
            hand_type: float(quest_attach_state[hand_type]["progress"])
            for hand_type in hand_types
        },
        "attach_message": {
            hand_type: str(quest_attach_state[hand_type].get("message", ""))
            for hand_type in hand_types
        },
        "quest_ok": {
            hand_type: bool(quest_missing_ticks[hand_type] <= max_missing_quest_ticks)
            for hand_type in hand_types
        },
        "quest_all_ok": bool(quest_all_missing_ticks <= max_missing_quest_ticks),
        "collision_ik_mode": collision_ik_result.get("mode", "idle"),
        "min_signed_dist": float(collision_ik_result.get("min_signed_dist", np.nan)),
        "safe_margin": float(collision_ik_result.get("safe_margin", 0.05)),
        "T_quest_start": copy_T(T_quest_start),
        "T_quest_all_by_idx": copy_T_dict(T_quest_pose_by_idx) if plot_quest_nodes else {},
        "T_quest_virtual_by_idx": copy_T_dict(T_virtual_quest_pose_by_idx) if plot_quest_nodes else {},
        "T_quest_wrist": copy_hand_T_dict(T_quest_wrist),
        "T_quest_virtual_wrist": copy_hand_T_dict(T_virtual_quest_wrist),
        "T_quest_target_raw": copy_hand_T_dict(T_raw_wrist_target),
        "T_quest_target": copy_hand_T_dict(T_attach_wrist_target),
        "T_sg2_wrist": copy_hand_T_dict(T_sg2_wrist),
        "active_arm_ik_targets": active_arm_ik_targets if plot_arm_ik_targets else [],
        "arm_ik_targets": {
            hand_type: active_arm_ik_targets if plot_arm_ik_targets else []
            for hand_type in hand_types
        },
        "arm_ik_err_best": {
            hand_type: None
            if arm_ik_info_by_hand[hand_type] is None
            else float(arm_ik_info_by_hand[hand_type]["ik_err_best"])
            for hand_type in hand_types
        },
        "hand_target_count": {hand_type: 0 for hand_type in hand_types},
        "gripper_retargeting": "disabled",
        "arm_target_delta": {
            hand_type: {
                "pos": float(wrist_target_motion[hand_type][0]),
                "rot_deg": float(np.rad2deg(wrist_target_motion[hand_type][1])),
            }
            for hand_type in hand_types
        },
        "detach_gesture_state": copy_sg2_detach_gesture_state(detach_gesture_state),
    }


def copy_T(T):
    if T is None:
        return None
    return np.asarray(T, dtype=np.float64).reshape(4, 4).copy()


def copy_T_dict(T_by_key):
    if not isinstance(T_by_key, dict):
        return {}
    return {
        int(key) if isinstance(key, (int, np.integer)) else key: copy_T(T)
        for key, T in T_by_key.items()
        if T is not None
    }


def copy_hand_T_dict(T_by_hand):
    return {hand_type: copy_T(T_by_hand.get(hand_type, None)) for hand_type in SG2_HAND_TYPES}


__all__ = [
    "PROJECT_DIR",
    "REPO_ROOT",
    "SG2_XML_PATH",
    "FLOOR_XML_PATH",
    "SG2_PROJECT_XML_DIR",
    "SG2_SCENE_XML_PATH",
    "SG2_BODY_ADDED_VIS_XML_PATH",
    "SG2_SCENE_CACHE_DIR",
    "SG2_RENDER_SCENE_XML_PATH",
    "SG2_HAND_TYPES",
    "SG2_HAND_LABELS",
    "SG2_ARM_JOINT_PREFIXES",
    "SG2_WRIST_BODY_NAMES",
    "SG2_GRIPPER_DRIVER_JOINT_NAMES",
    "SG2_GRIPPER_JOINT_NAMES",
    "SG2_GRIPPER_JOINT_SIGNS",
    "SG2_REAL_COMMAND_JOINT_NAMES",
    "SG2_COLLISION_IK_JOINT_NAMES",
    "SG2_UPPER_ARM_YAW_JOINT_NAMES",
    "SG2_NULL_REFERENCE_MODES",
    "SG2_COLLISION_AWARE_IK_MODULE_NAME",
    "SG2_COLLISION_AWARE_IK_CPP_SOURCE_NAMES",
    "SG2_COLLISION_AWARE_IK_HPP_HEADER_NAMES",
    "SG2_COLLISION_AWARE_IK_DEFINE_NAMES",
    "QUEST_HAND_JOINT_OFFSETS",
    "QuestJointIdx",
    "QuestPalmRetargeting",
    "build_sg2_collision_aware_ik_backend",
    "create_zmq_sub_socket",
    "close_zmq_socket",
    "receive_metaquest3",
    "update_metaquest3_wrist_data",
    "normalize_hand_type",
    "get_sg2_wrist_body_name",
    "get_sg2_wrist_local_T",
    "get_sg2_wrist_control_T",
    "get_sg2_arm_self_motion_direction",
    "compute_quest_target_T_from_wrist_transform",
    "transform_T_between_frames",
    "get_quest_wrist_joint_idx",
    "get_quest_hand_joint_idx",
    "get_quest_thumb_index_tip_positions",
    "get_quest_thumb_index_tip_distance",
    "get_quest_finger_bend_ratio",
    "get_quest_hand_bend_ratios",
    "create_sg2_detach_gesture_state",
    "update_sg2_detach_gesture_state",
    "apply_sg2_detach_gesture_to_attach_state",
    "virtualize_quest_pose_dict_from_wrist",
    "virtualize_quest_pose_dicts_from_wrists",
    "collect_sg2_arm_joint_names",
    "filter_existing_joint_names",
    "get_sg2_gripper_joint_names",
    "get_sg2_gripper_driver_qpos_from_pinch_distance",
    "get_sg2_gripper_qpos_from_driver_qpos",
    "set_sg2_gripper_qpos",
    "update_sg2_gripper_from_quest_pinch",
    "build_q_msg_layout",
    "get_qmsg_from_qpos",
    "get_qpos_from_robot_state",
    "get_real_q_from_robot_state",
    "get_vis_q_from_render_msg",
    "get_sg2_vis_q_from_render_msg",
    "get_latest_robot_state",
    "get_latest_robot_qpos",
    "send_render_qpos",
    "quest_missing_tick_limit",
    "create_quest_wrist_attach_state",
    "update_quest_wrist_attach_state",
    "is_quest_wrist_jump_rejected",
    "quest_pose_delta",
    "update_sg2_wrist_attach_targets",
    "update_last_solved_wrist_targets",
    "add_sg2_wrist_ik_coordinate_target",
    "create_sg2_arm_ik_solvers",
    "create_sg2_collision_aware_arm_ik_solvers",
    "create_sg2_collision_candidate_geom_pairs",
    "print_sg2_collision_candidate_summary",
    "make_idle_collision_ik_result",
    "set_sg2_null_space_reference",
    "relax_sg2_upper_arm_yaw_joints",
    "get_solver_target_infos",
    "solve_sg2_collision_aware_wrist_ik",
    "make_signed_distance_draw_cmds",
    "make_body_ik_target_draw_cmds",
    "make_sg2_attach_overlay_text",
    "make_sg2_collision_overlay_text",
    "make_sg2_ik_overlay_text",
    "make_sg2_detach_gesture_overlay_text",
    "make_sg2_gripper_overlay_text",
    "make_sg2_gripper_draw_cmds",
    "make_sg2_metaquest_draw_cmds",
    "copy_sg2_detach_gesture_state",
    "make_sg2_real_render_extra",
    "copy_T",
    "copy_T_dict",
    "copy_hand_T_dict",
]
