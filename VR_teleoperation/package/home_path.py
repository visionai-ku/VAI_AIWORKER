"""Whole-path check for the Initial pose return, before any HOME motion.

The previous return clipped every joint independently toward the captured
start pose and checked only one control tick at a time. Nobody checked the full
swept path, so a return could travel for several seconds and then stop half
way with the arms interleaved (2026-09-15 REAL poses: left link5 vs right
gripper). A retry followed the same path and was rejected again.

A return path here is a joint-space polyline. Each stage moves one joint group
to its start value while every other joint holds, and the last stage always
reaches the full start pose. Every segment is swept by the existing
SG2CommandGuard, with the same joint limits and 5 mm clearance, before any
command. Candidates are tried in a fixed order and the first fully accepted
path is used. The controller then follows the waypoints with one coordinated
rate limit, so commands stay on the checked segments.
"""
from __future__ import annotations

import time

import numpy as np

HOME_AUX_JOINTS = ("lift_joint", "head_joint1", "head_joint2", "gripper_l_joint1", "gripper_r_joint1")
# Joint 3/5 (upper-arm and forearm roll) are where teleoperation winds the arms
# up; joint 6/7 roll the wrist. Joint 1/2/4 place the elbow and hand.
HOME_TWIST_JOINTS = tuple("arm_%s_joint%d" % (side, k) for side in "lr" for k in (3, 5, 6, 7))
HOME_J3J5_JOINTS = tuple("arm_%s_joint%d" % (side, k) for side in "lr" for k in (3, 5))
HOME_ARM_JOINTS = {side: tuple("arm_%s_joint%d" % (side, k) for k in range(1, 8)) for side in "lr"}

# Fixed order: the direct line first, then unwinding before repositioning,
# then one arm at a time. The final implicit stage moves all joints to home.
HOME_PATH_STRATEGIES = (
    ("direct", ()),
    ("unwind_j3j5_first", (HOME_AUX_JOINTS, HOME_J3J5_JOINTS)),
    ("unwind_forearm_wrist_first", (HOME_AUX_JOINTS, HOME_TWIST_JOINTS)),
    ("right_arm_first", (HOME_AUX_JOINTS, HOME_ARM_JOINTS["r"])),
    ("left_arm_first", (HOME_AUX_JOINTS, HOME_ARM_JOINTS["l"])),
)


def home_path_waypoints(joint_names, q_start, q_home, stages):
    """Start, each distinct stage pose, and exactly q_home last."""
    joint_names = list(joint_names)
    q_start = np.asarray(q_start, dtype=float)
    q_home = np.asarray(q_home, dtype=float)
    waypoints = [q_start.copy()]
    q = q_start.copy()
    for stage in stages:
        indices = [joint_names.index(name) for name in stage if name in joint_names]
        q = q.copy()
        q[indices] = q_home[indices]
        if not np.array_equal(q, waypoints[-1]):
            waypoints.append(q)
    if not np.array_equal(q_home, waypoints[-1]):
        waypoints.append(q_home.copy())
    return waypoints


def home_path_duration(waypoints, speed_limits):
    """Nominal time with each segment's joints scaled to arrive together."""
    speed_limits = np.asarray(speed_limits, dtype=float)
    return float(sum(np.max(np.abs(b - a) / speed_limits) for a, b in zip(waypoints[:-1], waypoints[1:])))


def _failure_text(attempt):
    pair = attempt.get("pair") or {}
    bodies = pair.get("body_names")
    text = attempt["reason"]
    if bodies:
        text += " %s-%s" % tuple(bodies)
    return "%s: %s (segment %d/%d)" % (attempt["strategy"], text, attempt["segment"], attempt["segments"])


def plan_home_path(guard, joint_names, q_start, q_home, speed_limits, strategies=HOME_PATH_STRATEGIES):
    """Return the first guard-accepted path, or a blocked result with no path.

    The caller synchronizes ``guard.env_check`` for uncommanded bodies, as for
    any other guard check. Nothing here moves or commands the robot.
    """
    started = time.perf_counter()
    q_start = np.asarray(q_start, dtype=float).copy()
    q_home = np.asarray(q_home, dtype=float).copy()
    if q_start.shape != (len(joint_names),) or q_home.shape != q_start.shape:
        raise ValueError("HOME path poses must match joint_names")
    attempts = []
    chosen = None
    for strategy, stages in strategies:
        waypoints = home_path_waypoints(joint_names, q_start, q_home, stages)
        segments = max(1, len(waypoints) - 1)
        # Reported guard distances can be lower bounds, so candidates are not
        # ranked by clearance: acceptance of every segment is the criterion.
        attempt = {"strategy": strategy, "accepted": True, "reason": "accepted", "segment": 0,
                   "segments": segments, "samples_checked": 0, "pair": None}
        pairs = list(zip(waypoints[:-1], waypoints[1:])) or [(q_start, q_start)]
        for index, (a, b) in enumerate(pairs, 1):
            check = guard.check(a, b)
            attempt["samples_checked"] += int(check.get("samples_checked", 0))
            if not check["accepted"]:
                attempt.update(accepted=False, reason=check["reason"], segment=index, pair=check.get("pair"),
                               sweep_fraction=check.get("sweep_fraction"), guard=check)
                break
        attempts.append(attempt)
        if attempt["accepted"]:
            chosen = (strategy, waypoints)
            break
    plan = {
        "ok": chosen is not None, "q_start": q_start, "q_home": q_home, "attempts": attempts,
        "compute_s": time.perf_counter() - started,
    }
    if chosen is None:
        at_start = all(attempt["segment"] == 1 and attempt["guard"].get("samples_checked", 0) <= 1
                       for attempt in attempts)
        if at_start:
            # No path can leave a pose the guard already rejects; say so rather
            # than listing identical failures for every candidate.
            reason = ("initial pose blocked: current pose already fails the command guard (%s)"
                      % _failure_text(attempts[0]).split(": ", 1)[1])
        else:
            reason = ("initial pose path blocked; move the arms apart, then retry: "
                      + "; ".join(_failure_text(attempt) for attempt in attempts))
        plan.update(strategy=None, waypoints=[q_start.copy()], duration_s=0., reason=reason)
    else:
        plan.update(strategy=chosen[0], waypoints=chosen[1],
                    duration_s=home_path_duration(chosen[1], speed_limits),
                    reason="initial pose path verified: %s" % chosen[0])
    return plan


def home_plan_summary(plan):
    """JSON-sized record of a plan for logs and the control window."""
    if plan is None:
        return None
    return {key: plan[key] for key in ("ok", "strategy", "duration_s", "compute_s", "reason", "attempts", "waypoints")}


__all__ = ["HOME_PATH_STRATEGIES", "home_path_waypoints", "home_path_duration", "plan_home_path", "home_plan_summary"]
