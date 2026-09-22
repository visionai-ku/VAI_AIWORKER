"""Cartesian safety box for SG2 wrist EEF positions (robot base frame, meters).

Bounds were measured by hand via free-drive + scripts/live_fk_monitor.py,
which computes the wrist TCP point the same way get_sg2_wrist_control_T /
SG2_WRIST_P_LOCAL (utils.py) does -- the same point wrist_safety_box_violations()
below checks against, e.g. from policy_deploy.py before an autonomous policy
action reaches the real robot. Measurement and enforcement share one frame by
construction, so there is no tcp-offset/gripper-length mismatch between them
(see github.com/plaxxma/DiffusionPoicy_UR5e_real's workspace_safety_box.py
for a case where they didn't: the pendant TCP used to measure its box (0.13)
didn't match the real gripper's length (0.23m), so the measured box wasn't
where the physical gripper tip actually was).

That said, SG2_WRIST_P_LOCAL itself assumes a fixed flange-to-TCP offset
(currently -0.08m in local z) for whichever gripper is mounted. If the
physical gripper is a different length, both the numbers below AND the
runtime clip inherit that same error -- re-measure with live_fk_monitor.py
whenever the gripper changes.

Only the bounds actually requested are enforced; an unset side is +-inf.
"""
from __future__ import annotations

import numpy as np

SAFETY_BOX = {
    "left": {
        "x": (-0.30, np.inf),
        "y": (-np.inf, 0.59),
        "z": (0.95, np.inf),
    },
    "right": {
        "x": (-0.30, np.inf),
        "y": (-0.59, np.inf),
        "z": (0.95, np.inf),
    },
}


def clip_wrist_position(hand, position):
    """Clip a wrist TCP [x, y, z] into SAFETY_BOX[hand]. Returns (clipped, was_clipped)."""
    bounds = SAFETY_BOX[hand]
    position = np.asarray(position, dtype=np.float64).reshape(3)
    clipped = position.copy()
    for axis, index in (("x", 0), ("y", 1), ("z", 2)):
        low, high = bounds[axis]
        clipped[index] = min(max(position[index], low), high)
    was_clipped = not np.array_equal(clipped, position)
    return clipped, was_clipped


def wrist_safety_box_violations(env, q, joint_names):
    """FK `q` (in `joint_names` order) on `env` -- a dedicated scratch
    MuJoCoParser; this mutates its qpos -- and report which wrists land
    outside SAFETY_BOX.

    For a Cartesian IK target, clip_wrist_position() can just move the target;
    an autonomous policy instead predicts joint angles directly (see
    policy_inference.py), so there is no target position here to move -- only
    a resulting one to check. This is a guard for the caller to reject/hold
    on (see policy_deploy.py's step()), not a joint-space clip.

    -> {hand: (requested_xyz, box_xyz)}, empty when both wrists are inside.
    """
    from utils import SG2_HAND_TYPES, get_sg2_wrist_control_T

    env.forward(q=np.asarray(q, dtype=np.float64), joint_names=list(joint_names))
    violations = {}
    for hand in SG2_HAND_TYPES:
        position = get_sg2_wrist_control_T(env, hand)[:3, 3]
        clipped, was_clipped = clip_wrist_position(hand, position)
        if was_clipped:
            violations[hand] = (position, clipped)
    return violations
