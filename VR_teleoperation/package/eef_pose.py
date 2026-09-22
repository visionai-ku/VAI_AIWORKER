"""SG2 wrist (EEF) pose from joint positions, for dataset recording.

Uses the same wrist control frame the teleop tracks (get_sg2_wrist_control_T: arm_*_link7
body pose times SG2_WRIST_*_LOCAL), expressed in the base_link frame. The teleop scene lifts
base_link 1 cm above the floor, so its world-frame z is 0.01 m higher; relative to base_link
the two agree. Keeps its own MuJoCoParser so it can run in the recorder thread without
touching the teleop's models.
"""
import numpy as np

from ri_motion_v5_package.mujoco_sim import MuJoCoParser
from utils import SG2_XML_PATH, get_sg2_wrist_control_T
from package.worker.fixed_quest_protocol import JOINT_NAMES

EEF_HANDS = ("left", "right")
EEF_POSE_NAMES = ["eef_%s_%s" % (hand[0], axis) for hand in EEF_HANDS
                  for axis in ("x", "y", "z", "qw", "qx", "qy", "qz")]


def _quat_wxyz(R):
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
    y = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
    z = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


class SG2EEFPose:
    """Callable q (JOINT_NAMES order) -> float32[14]: per hand (left, right) xyz [m] + quat wxyz.

    Quaternion sign is kept continuous with the previous call (q and -q are the same
    rotation); call reset() at each episode start. Non-finite q gives NaNs.
    """

    def __init__(self, xml_path=SG2_XML_PATH):
        self.env = MuJoCoParser(rel_xml_path=str(xml_path), verbose=False)
        # Gripper joints do not move the wrist frame and are driven differently in the model.
        self.joint_names = [name for name in JOINT_NAMES if not name.startswith("gripper")]
        self.idxs = [list(JOINT_NAMES).index(name) for name in self.joint_names]
        self.reset()

    def reset(self):
        self._prev_quat = {hand: None for hand in EEF_HANDS}

    def __call__(self, q):
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (len(JOINT_NAMES),) or not np.all(np.isfinite(q)):
            return np.full(len(EEF_POSE_NAMES), np.nan, dtype=np.float32)
        self.env.forward(q=q[self.idxs], joint_names=self.joint_names, clip_position=False)
        T_base_inv = np.linalg.inv(self.env.get_T("base_link", "body"))
        out = []
        for hand in EEF_HANDS:
            T = T_base_inv @ get_sg2_wrist_control_T(self.env, hand)
            quat = _quat_wxyz(T[:3, :3])
            prev = self._prev_quat[hand]
            if prev is not None and np.dot(prev, quat) < 0:
                quat = -quat
            self._prev_quat[hand] = quat
            out.extend(T[:3, 3])
            out.extend(quat)
        return np.asarray(out, dtype=np.float32)
