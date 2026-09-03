"""Motion executor for FFW SG2: goal qpos in, one position command per control cycle out.

The marker notebooks write a pose into MuJoCo the instant the user asks for it. A
preset, a file load or a bring-up sequence is therefore a whole-arm jump, which
`joint_marker_real.py` can only blunt after the fact with a per-tick speed clamp.

This module moves that decision earlier: given a goal (or a list of via poses) it
plans a **minimum-jerk trajectory over time** first, then hands out one sample per
tick. The rate limit stays in the output stage as a safety floor, but on a
well-planned trajectory it should never engage.

    plan_min_jerk(q_start, q_goal, v_max, a_max)   pure function, no env
    SG2MotionExecutor(env, joint_names).start([...])   then .tick() every cycle

`SG2MotionExecutor.tick` keeps the output order the real notebook uses
(`real_joint_controller_sg2_teleop.ipynb`, cell 14):

    write the target into env -> apply_sg2_joint_rate_limit -> np.clip -> env.forward

so moving to hardware only adds the `pub_robot.send(...)` line after the tick.

The bring-up waypoints below are the poses the real robot walks through when it is
powered on, transcribed from `ffw_sg2_follower_initial_positions.yaml` plus the
observed start pose. They are stored as **{joint name: value} dicts, never as bare
19-vectors**: `command_joint_names` starts with `lift_joint, head_joint1,
head_joint2` while the vendor yaml starts at `arm_l_joint1`, and a positional copy
silently lands 1.57 in the lift slide (limit -0.5~0.0 m).
"""

from __future__ import annotations

import numpy as np

from ri_motion_v5_package.kinematics import t2p
from ri_motion_v5_package.qt import MultiRadioQtWidget

from utils import (
    SG2_GRIPPER_DRIVER_JOINT_NAMES,
    SG2_HAND_TYPES,
    add_sg2_wrist_ik_coordinate_target,
    get_sg2_wrist_control_T,
    set_sg2_gripper_qpos,
)
from package.joint_control import (
    SG2_JOINT_PRESETS,
    SG2_JOINT_QPOS_PATH,
    load_sg2_joint_qpos,
    save_sg2_joint_qpos,
)
from package.joint_marker_real import (
    SG2_MAX_PRI_JOINT_SPEED,
    SG2_MAX_REV_JOINT_SPEED,
    apply_sg2_joint_rate_limit,
    get_sg2_joint_speed_limits,
)

# ---------------------------------------------------------------------------
# Acceleration ceilings
# ---------------------------------------------------------------------------
# Speed ceilings already exist in joint_marker_real.py. Acceleration ceilings do
# not: nothing in the code or the MJCF states one. Rather than inventing a bare
# number, derive them from "how long may a joint take to reach v_max from rest",
# which keeps a single tunable knob and one stated assumption.
# PROVISIONAL -- check against the DY-80 / DY-70 / DP-42 motor specs before
# running this on the real robot.
SG2_JOINT_ACC_RAMP_SEC = 0.5
SG2_MAX_REV_JOINT_ACC = SG2_MAX_REV_JOINT_SPEED / SG2_JOINT_ACC_RAMP_SEC  # ~2.09 rad/s^2
SG2_MAX_PRI_JOINT_ACC = SG2_MAX_PRI_JOINT_SPEED / SG2_JOINT_ACC_RAMP_SEC  # 0.10 m/s^2

# Peak values of the normalized min-jerk profile (verifiable with min_jerk_scalar)
MIN_JERK_PEAK_VEL = 1.875                 # max ds/dtau, at tau=0.5
MIN_JERK_PEAK_ACC = 10.0 / np.sqrt(3.0)   # max |d2s/dtau2| = ~5.7735


# ---------------------------------------------------------------------------
# Trajectory generation -- pure functions, no env / viewer / widget
# ---------------------------------------------------------------------------
def min_jerk_scalar(tau):
    """Progress s and its first two derivatives for normalized time tau in [0,1].

        s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5

    Boundary conditions: s(0)=0, s(1)=1, s'(0)=s'(1)=0, s''(0)=s''(1)=0
    -> velocity and acceleration are exactly zero at both ends, so there is no
       start/stop jolt.

    Returns: (s, ds_dtau, dds_dtau2), each an ndarray shaped like the input.
    """
    tau = np.clip(np.asarray(tau, dtype=np.float64), 0.0, 1.0)
    s   =  10.0 * tau**3 -  15.0 * tau**4 +   6.0 * tau**5
    ds  =  30.0 * tau**2 -  60.0 * tau**3 +  30.0 * tau**4
    dds =  60.0 * tau    - 180.0 * tau**2 + 120.0 * tau**3
    return s, ds, dds


def estimate_duration(q_start, q_goal, v_max, a_max, min_duration=0.5):
    """Shortest duration T [s] for which every joint stays within v_max and a_max.

    Moving a displacement D over time T with a min-jerk profile gives
        peak |velocity|     = MIN_JERK_PEAK_VEL * |D| / T
        peak |acceleration| = MIN_JERK_PEAK_ACC * |D| / T^2
    so joint j requires
        T >= MIN_JERK_PEAK_VEL * |D_j| / v_max_j
        T >= sqrt(MIN_JERK_PEAK_ACC * |D_j| / a_max_j)
    Take the largest requirement across all joints (and min_duration).
    """
    q_start = np.asarray(q_start, dtype=np.float64).ravel()
    q_goal  = np.asarray(q_goal,  dtype=np.float64).ravel()
    if q_start.shape != q_goal.shape:
        raise ValueError(f"q_start/q_goal shape mismatch: {q_start.shape} vs {q_goal.shape}")

    D     = np.abs(q_goal - q_start)
    v_max = np.broadcast_to(np.asarray(v_max, dtype=np.float64).ravel(), D.shape)
    a_max = np.broadcast_to(np.asarray(a_max, dtype=np.float64).ravel(), D.shape)
    if np.any(v_max <= 0.0) or np.any(a_max <= 0.0):
        raise ValueError("v_max and a_max must all be positive.")

    T_vel = float(np.max(MIN_JERK_PEAK_VEL * D / v_max))
    T_acc = float(np.max(np.sqrt(MIN_JERK_PEAK_ACC * D / a_max)))
    return float(max(T_vel, T_acc, float(min_duration)))


def plan_min_jerk(q_start, q_goal, v_max, a_max, hz=100.0, min_duration=0.5, duration=None):
    """Min-jerk trajectory from q_start to q_goal, sampled on an hz grid.

    Returns a dict:
        "times"    : (L,)    sample times [s], times[0]=0, times[-1]=T
        "qpos"     : (L, n)  position commands -- the only thing the loop consumes
        "qvel"     : (L, n)  analytic velocity (not a finite difference)
        "qacc"     : (L, n)  analytic acceleration
        "duration" : T [s]
        "hz"       : sampling rate
    """
    q_start = np.asarray(q_start, dtype=np.float64).ravel()
    q_goal  = np.asarray(q_goal,  dtype=np.float64).ravel()
    if q_start.shape != q_goal.shape:
        raise ValueError(f"q_start/q_goal shape mismatch: {q_start.shape} vs {q_goal.shape}")
    if hz <= 0.0:
        raise ValueError("hz must be positive.")

    T = estimate_duration(q_start, q_goal, v_max, a_max, min_duration=min_duration) \
        if duration is None else float(duration)
    if T <= 0.0:
        raise ValueError("duration must be positive.")

    dt     = 1.0 / float(hz)
    n_step = max(int(np.ceil(T / dt)), 1)
    T      = n_step * dt                      # snap to the hz grid
    times  = np.arange(n_step + 1, dtype=np.float64) * dt

    s, ds, dds = min_jerk_scalar(times / T)
    delta = (q_goal - q_start)[None, :]       # (1, n)

    return {
        "times"    : times,
        "qpos"     : q_start[None, :] + s[:, None] * delta,
        "qvel"     : (ds[:, None]  * delta) / T,
        "qacc"     : (dds[:, None] * delta) / (T * T),
        "duration" : T,
        "hz"       : float(hz),
    }


# ---------------------------------------------------------------------------
# Bring-up sequence
# ---------------------------------------------------------------------------
# The poses the real SG2 walks through on power-on. `step1` / `step2` are step0 /
# step1 of ffw_sg2_follower_initial_positions.yaml; `start` is the pose the robot
# is in before the sequence begins. Stored per joint name -- see the module
# docstring for why a bare 19-vector is a trap here.
SG2_BRINGUP_WAYPOINTS = {
    "start": {
        "arm_l_joint1": 0.2, "arm_l_joint4": -0.3, "arm_l_joint6": -0.2,
        "arm_r_joint1": 0.2, "arm_r_joint4": -0.3, "arm_r_joint6": -0.2,
    },
    "step1": {
        "arm_l_joint1": 1.57, "arm_l_joint4": -2.8,
        "arm_r_joint1": 1.57, "arm_r_joint4": -2.8,
    },
    "step2": {
        "arm_l_joint4": -0.5 * np.pi,
        "arm_r_joint4": -0.5 * np.pi,
    },
}

# "bring_up" is the ROS term for powering a robot into its ready pose; "shut_down"
# is its counterpart -- the same waypoints walked backwards, which is what folds
# the arms back in before the robot is powered off.
#
# These are the *joint-space* legs only. `shut_down` additionally gets two
# pre-steps computed from the pose the robot is actually in when the command is
# given (open the grippers, then lift the wrists clear) -- see
# `build_sg2_shut_down_goals`.
SG2_MOTION_SEQUENCES = {
    "bring_up" : ["start", "step1", "step2"],
    "shut_down": ["step2", "step1", "start"],
}

# rh_p12_rn driver joint, range 0 ~ 1.1: 0 is fully open, 1.1 fully closed. Same
# convention `update_sg2_gripper_from_quest_pinch` uses (utils.py, q_open/q_closed).
SG2_GRIPPER_OPEN_QPOS  = 0.0
SG2_GRIPPER_CLOSE_QPOS = 1.1

# The gripper-only Motion commands. Both go through the executor like any other
# move, so a grip is ramped rather than snapped; the key is also the segment label
# printed in the status overlay.
SG2_GRIP_COMMAND_QPOS = {
    "grip_close": SG2_GRIPPER_CLOSE_QPOS,
    "grip_open" : SG2_GRIPPER_OPEN_QPOS,
}


def make_sg2_gripper_qpos_goal(env, joint_names, driver_qpos, base_qpos=None):
    """The current pose with only the two gripper driver joints changed.

    Every other joint is held where it is, so this is a gripper-only move: the
    executor still ramps it, but nothing else on the robot shifts.
    """
    joint_names = list(joint_names)
    q = (env.get_qpos(joint_names=joint_names).copy() if base_qpos is None
         else expand_sg2_qpos(joint_names, base_qpos).copy())
    for hand_type in SG2_HAND_TYPES:
        driver = SG2_GRIPPER_DRIVER_JOINT_NAMES[hand_type]
        if driver in joint_names:
            q[joint_names.index(driver)] = float(driver_qpos)
    return q


def get_sg2_wrist_p_at_qpos(env, joint_names, qpos_spec, hand_types=SG2_HAND_TYPES):
    """Forward-kinematics probe: where the wrists sit at some other pose.

    Writes that pose into MuJoCo, reads the wrist frames, then puts the original
    pose back -- `env.forward` is kinematics only, so this costs nothing and
    leaves no trace.
    """
    joint_names = list(joint_names)
    q_save = env.get_qpos(joint_names=joint_names).copy()
    try:
        env.forward(q=expand_sg2_qpos(joint_names, qpos_spec), joint_names=joint_names)
        return {hand_type: t2p(get_sg2_wrist_control_T(env, hand_type)).copy()
                for hand_type in hand_types}
    finally:
        env.forward(q=q_save, joint_names=joint_names)


def solve_sg2_wrist_lift_goal(
        env,
        joint_names,
        ik_solvers,
        z_by_hand,
        base_qpos=None,
        min_lift=0.01,
        hand_types=SG2_HAND_TYPES,
    ):
    """Raise each wrist straight up to world z = `z_by_hand[hand]`, x/y and
    orientation unchanged, and return the joint pose that achieves it.

    This is the "lift clear before folding" step: whatever the hand was reaching
    into, it comes straight up out of it before the arm starts folding toward the
    stow pose.

    Returns (qpos, info). qpos is None when no wrist needs lifting (both are
    already at or above the target). `env` is restored either way.
    """
    joint_names = list(joint_names)
    q_save = env.get_qpos(joint_names=joint_names).copy()
    base = q_save.copy() if base_qpos is None else expand_sg2_qpos(joint_names, base_qpos)
    try:
        env.forward(q=base, joint_names=joint_names)

        targets = {}
        for hand_type in hand_types:
            T_now = get_sg2_wrist_control_T(env, hand_type)
            dz = float(z_by_hand[hand_type]) - float(t2p(T_now)[2])
            if dz > min_lift:
                T_target = np.array(T_now, dtype=np.float64)
                T_target[2, 3] += dz
                targets[hand_type] = T_target
        if len(targets) == 0:
            return None, {"lifted": {}, "skipped": "both wrists already at or above the lift height"}

        for hand_type, T_target in targets.items():
            solver = ik_solvers[hand_type]
            solver.reset_buffers(env=env)
            add_sg2_wrist_ik_coordinate_target(
                ik_solver  = solver,
                env        = env,
                hand_type  = hand_type,
                T_target   = T_target,
                weight     = 1.0,
                axis_width = 0.10,)
            _, ik_info = solver.solve_ik(
                env                     = env,
                joint_limit_handle_flag = True,
                nullspace_control_flag  = True,)
            env.forward(q=ik_info["qpos_used_best"], joint_names=solver.joints_use)

        q_goal = env.get_qpos(joint_names=joint_names).copy()
        # IK owns the arm joints only; carry the gripper values over from `base`
        # so this step cannot undo the gripper move that precedes it.
        for hand_type in SG2_HAND_TYPES:
            driver = SG2_GRIPPER_DRIVER_JOINT_NAMES[hand_type]
            if driver in joint_names:
                q_goal[joint_names.index(driver)] = base[joint_names.index(driver)]

        lifted = {hand_type: float(t2p(get_sg2_wrist_control_T(env, hand_type))[2])
                  for hand_type in targets}
        return q_goal, {"lifted": lifted, "skipped": ""}
    finally:
        env.forward(q=q_save, joint_names=joint_names)


def build_sg2_shut_down_goals(env, joint_names, ik_solvers=None, lift_reference="step2"):
    """The full shut-down goal list, computed from the pose the robot is in now.

        1. open both grippers        (gripper joints only)
        2. lift both wrists          (to the wrist height of the `lift_reference`
                                      waypoint -- IK, skipped when ik_solvers is None
                                      or the wrists are already high enough)
        3. step2 -> step1 -> start   (SG2_MOTION_SEQUENCES["shut_down"])

    Steps 1 and 2 cannot be constants: they depend on where the arms happen to be
    when the command is given, which is the whole reason they exist.

    Returns (goals, notes) where goals is the [(label, qpos), ...] list the
    executor takes and notes is a short human-readable trail.
    """
    joint_names = list(joint_names)
    notes = []

    q_open = make_sg2_gripper_qpos_goal(env, joint_names, SG2_GRIPPER_OPEN_QPOS)
    goals = [("gripper_open", q_open)]

    if ik_solvers is None:
        notes.append("wrist lift skipped: no ik_solvers passed")
    else:
        z_by_hand = {hand_type: float(p[2]) for hand_type, p in
                     get_sg2_wrist_p_at_qpos(env, joint_names,
                                             SG2_BRINGUP_WAYPOINTS[lift_reference]).items()}
        q_lift, lift_info = solve_sg2_wrist_lift_goal(
            env         = env,
            joint_names = joint_names,
            ik_solvers  = ik_solvers,
            z_by_hand   = z_by_hand,
            base_qpos   = q_open,)
        if q_lift is None:
            notes.append(f"wrist lift skipped: {lift_info['skipped']}")
        else:
            goals.append(("wrist_lift", q_lift))
            notes.append("wrist lift to z " + ", ".join(
                f"{hand_type}={z:.3f}m" for hand_type, z in lift_info["lifted"].items()))

    goals += [(step, expand_sg2_qpos(joint_names, SG2_BRINGUP_WAYPOINTS[step]))
              for step in SG2_MOTION_SEQUENCES["shut_down"]]
    return goals, notes


def expand_sg2_qpos(joint_names, spec):
    """{joint name: value} dict (or an already-ordered array) -> (n,) vector in
    `joint_names` order. Joints the dict omits go to 0.0, the same rule
    `joint_control.apply_preset` uses. An unknown joint name is an error, not a
    silent no-op -- that is how a mistyped name used to vanish into `dict.get`.
    """
    joint_names = list(joint_names)
    if not isinstance(spec, dict):
        qpos = np.asarray(spec, dtype=np.float64).ravel()
        if qpos.shape != (len(joint_names),):
            raise ValueError(f"qpos has {qpos.shape[0]} values, expected {len(joint_names)}")
        return qpos
    unknown = set(spec) - set(joint_names)
    if unknown:
        raise ValueError(f"unknown joint name(s): {sorted(unknown)}")
    return np.array([float(spec.get(name, 0.0)) for name in joint_names], dtype=np.float64)


def get_sg2_joint_acc_limits(
        env,
        joint_names,
        max_rev_acc=SG2_MAX_REV_JOINT_ACC,
        max_pri_acc=SG2_MAX_PRI_JOINT_ACC,
    ):
    """Per-joint ceiling in rad/s^2 for hinges and m/s^2 for the lift slide."""
    pri_joint_names = set(env.pri_joint_names)
    return np.array(
        [max_pri_acc if name in pri_joint_names else max_rev_acc for name in joint_names],
        dtype=np.float64,
    )


def check_sg2_joint_limits(env, joint_names, qpos):
    """Which of `qpos` sit outside the model's joint range.

    The whole point of a motion executor is that a goal is checked *before* the
    robot starts moving toward it. Without this, `np.clip` in the output stage
    truncates an impossible goal silently and the executor reports a large
    "gap" it can never close.

    Returns {"ok": bool, "violations": [{"joint","value","min","max","over"}, ...]}
    """
    joint_names = list(joint_names)
    qpos = np.asarray(qpos, dtype=np.float64).ravel()
    q_min = env.get_joint_mins(joint_names)
    q_max = env.get_joint_maxs(joint_names)
    violations = [
        {
            "joint": name,
            "value": float(value),
            "min"  : float(lo),
            "max"  : float(hi),
            "over" : float(max(lo - value, value - hi)),
        }
        for name, value, lo, hi in zip(joint_names, qpos, q_min, q_max)
        if value < lo - 1e-9 or value > hi + 1e-9
    ]
    return {"ok": len(violations) == 0, "violations": violations}


def format_sg2_joint_limit_violations(limit_info):
    return "\n".join(
        f"  {v['joint']:<18s} {v['value']:+9.4f}  outside [{v['min']:+.4f}, {v['max']:+.4f}]"
        f"  (over by {v['over']:+.4f})"
        for v in limit_info["violations"]
    )


# ---------------------------------------------------------------------------
# The executor
# ---------------------------------------------------------------------------
class SG2MotionExecutor(object):
    """Plans a min-jerk path to each goal and emits one position command per tick.

    Non-blocking on purpose: `tick()` is called from whatever render/control loop
    already exists, so the viewer stays responsive and a marker drag can still be
    read on the ticks the executor is idle.

    A multi-waypoint request is planned **one segment at a time**, each segment
    starting from the pose actually reached. A segment that was rate-limited or
    clipped therefore cannot leave a discontinuity at the via pose.
    """

    def __init__(
            self,
            env,
            joint_names,
            hz=100.0,
            min_duration=0.5,
            speed_limits=None,
            acc_limits=None,
            dt_max_factor=10.0,
        ):
        self.env = env
        self.joint_names = list(joint_names)
        self.hz = float(hz)
        self.dt_nominal = 1.0 / self.hz
        self.dt_max = float(dt_max_factor) * self.dt_nominal
        self.min_duration = float(min_duration)

        self.speed_limits = (
            get_sg2_joint_speed_limits(env=env, joint_names=self.joint_names)
            if speed_limits is None else np.asarray(speed_limits, dtype=np.float64)
        )
        self.acc_limits = (
            get_sg2_joint_acc_limits(env=env, joint_names=self.joint_names)
            if acc_limits is None else np.asarray(acc_limits, dtype=np.float64)
        )
        self.q_min = env.get_joint_mins(self.joint_names)
        self.q_max = env.get_joint_maxs(self.joint_names)

        self.reset()

    # -- state ------------------------------------------------------------
    def reset(self):
        self.goal_queue = []        # remaining goals, each (label, (n,) vector)
        self.traj = None            # current segment
        self.step_idx = 0
        self.q_use = None           # last committed command (the robot's q_use)
        self.label = ""
        self.segment_label = ""
        self.segment_idx = 0
        self.n_segment = 0
        self.last_rate_limit_info = None
        self.last_dt = self.dt_nominal
        self.limited_tick_count = 0
        self.clipped_tick_count = 0

    @property
    def is_active(self):
        return self.traj is not None or len(self.goal_queue) > 0

    def get_qpos(self):
        return self.env.get_qpos(joint_names=self.joint_names)

    # -- planning ---------------------------------------------------------
    def start(self, goals, label="", q_start=None, check_limits=True):
        """Queue one goal or a list of them and begin.

        `goals` items may be {joint name: value} dicts or ordered (n,) arrays.
        With check_limits=True an out-of-range goal raises here, before anything
        moves, instead of being silently truncated by the output stage.
        """
        if isinstance(goals, dict) or not isinstance(goals, (list, tuple)):
            goals = [goals]

        queue = []
        for idx, goal in enumerate(goals):
            if isinstance(goal, (list, tuple)) and len(goal) == 2 and isinstance(goal[0], str):
                goal_label, goal_spec = goal
            else:
                goal_label, goal_spec = f"goal{idx + 1}", goal
            q_goal = expand_sg2_qpos(self.joint_names, goal_spec)
            if check_limits:
                limit_info = check_sg2_joint_limits(self.env, self.joint_names, q_goal)
                if not limit_info["ok"]:
                    raise ValueError(
                        f"goal '{goal_label}' is outside the joint limits:\n"
                        + format_sg2_joint_limit_violations(limit_info)
                    )
            queue.append((goal_label, q_goal))

        self.reset()
        self.goal_queue = queue
        self.label = label or "motion"
        self.n_segment = len(queue)
        self.q_use = (
            self.get_qpos() if q_start is None
            else expand_sg2_qpos(self.joint_names, q_start)
        )
        self._plan_next_segment()
        return self

    def _plan_next_segment(self):
        """Pop the next goal and plan from the pose actually reached."""
        if len(self.goal_queue) == 0:
            self.traj = None
            self.segment_label = ""
            return False
        self.segment_label, q_goal = self.goal_queue.pop(0)
        self.segment_idx += 1
        self.traj = plan_min_jerk(
            q_start      = self.q_use,
            q_goal       = q_goal,
            v_max        = self.speed_limits,
            a_max        = self.acc_limits,
            hz           = self.hz,
            min_duration = self.min_duration,)
        self.step_idx = 0
        return True

    def cancel(self):
        """Stop where we are. The pose already committed by the last tick stays."""
        was_active = self.is_active
        self.goal_queue = []
        self.traj = None
        self.segment_label = ""
        return was_active

    # -- execution --------------------------------------------------------
    def tick(self, dt=None):
        """Emit one control cycle. Returns None when there is nothing to run.

        Output order matches the real notebook:
            env.forward(q_cmd) -> apply_sg2_joint_rate_limit -> np.clip -> env.forward

        `dt` should be the **measured** cycle time. SimpleTimer.dt is the constant
        1.0/Hz, so on a loop that has fallen behind it under-reports and the rate
        limit clamps a motion that was actually within spec. It is clipped into
        [dt_nominal, dt_max] so a one-off stall cannot authorize a huge jump.
        """
        if self.traj is None and not self._plan_next_segment():
            return None

        dt_used = float(np.clip(
            self.dt_nominal if dt is None else float(dt),
            self.dt_nominal, self.dt_max))
        self.last_dt = dt_used

        # 1) write this tick's trajectory target into env first: the rate limit
        #    takes no target argument, it reads env.get_qpos() - q_prev.
        q_cmd = self.traj["qpos"][self.step_idx]
        self.env.forward(q=q_cmd, joint_names=self.joint_names)

        # 2) speed limit
        rate_limit_info = apply_sg2_joint_rate_limit(
            env          = self.env,
            joint_names  = self.joint_names,
            q_prev       = self.q_use,
            dt           = dt_used,
            speed_limits = self.speed_limits,)

        # 3) joint limit clip, kept separate from (2) so the two are never
        #    confused when the gap between q_cmd and q_use is explained.
        q_rate = rate_limit_info["qpos"]
        q_use = np.clip(q_rate, self.q_min, self.q_max)
        clipped = bool(np.any(np.abs(q_use - q_rate) > 1e-12))

        # 4) commit
        self.env.forward(q=q_use, joint_names=self.joint_names)
        for hand_type in SG2_HAND_TYPES:
            driver = SG2_GRIPPER_DRIVER_JOINT_NAMES[hand_type]
            if driver in self.joint_names:
                set_sg2_gripper_qpos(self.env, hand_type,
                                     float(q_use[self.joint_names.index(driver)]))

        self.q_use = q_use
        self.last_rate_limit_info = rate_limit_info
        self.limited_tick_count += int(rate_limit_info["limited"])
        self.clipped_tick_count += int(clipped)

        info = {
            "q_cmd"        : q_cmd,
            "q_use"        : q_use,
            "dt"           : dt_used,
            "limited"      : bool(rate_limit_info["limited"]),
            "clipped"      : clipped,
            "max_dq_ratio" : float(rate_limit_info["max_dq_ratio"]),
            "segment"      : self.segment_label,
            "segment_idx"  : self.segment_idx,
            "n_segment"    : self.n_segment,
            "progress"     : (self.step_idx + 1) / len(self.traj["times"]),
            "done"         : False,
        }

        self.step_idx += 1
        if self.step_idx >= len(self.traj["times"]):
            self.traj = None
            if not self._plan_next_segment():
                info["done"] = True
        return info

    # -- reporting --------------------------------------------------------
    def status_text(self):
        if not self.is_active:
            return "executor idle"
        if self.traj is None:
            return f"{self.label}: planning"
        progress = (self.step_idx + 1) / len(self.traj["times"])
        flags = ""
        if self.last_rate_limit_info is not None and self.last_rate_limit_info["limited"]:
            flags = " | RATE LIMITED"
        return (
            f"{self.label} [{self.segment_idx}/{self.n_segment}] {self.segment_label} "
            f"{100.0 * progress:5.1f}%  T={self.traj['duration']:.2f}s{flags}"
        )


# ---------------------------------------------------------------------------
# Command widget
# ---------------------------------------------------------------------------
# Same shape as SG2_JOINT_MARKER_COMMAND_ROWS, with two differences:
#   - a `Motion` row for the bring-up sequences;
#   - `Show` selects which *wrist* gizmo is live, since this notebook drives the
#     arms through EEF markers and IK, not one marker per joint.
SG2_MOTION_COMMAND_ROWS = {
    "Motion"   : ["idle", "bring_up", "grip_close", "grip_open", "shut_down", "stop"],
    "Preset"   : ["idle", "zero", "init", "sync"],
    "File"     : ["idle", "save", "load"],
    "Collision": ["off", "on"],
    "EEF"      : ["off", "on"],
    "Show"     : ["all"] + list(SG2_HAND_TYPES),
}
SG2_MOTION_COMMAND_LAYOUT = (0.00, 0.00, 0.22, 0.36)


def create_sg2_motion_command_widget(layout=SG2_MOTION_COMMAND_LAYOUT):
    x_offset, y_offset, window_width, window_height = layout
    return MultiRadioQtWidget(
        title="SG2 Joint Marker Command",
        x_offset=x_offset,
        y_offset=y_offset,
        window_width=window_width,
        window_height=window_height,
        label_texts=list(SG2_MOTION_COMMAND_ROWS),
        option_texts=list(SG2_MOTION_COMMAND_ROWS.values()),
        initial_texts=["idle", "idle", "idle", "on", "on", "all"],
    )


def handle_sg2_motion_command_widget(
        widget,
        executor,
        qpos_path=SG2_JOINT_QPOS_PATH,
        presets=None,
        ik_solvers=None,
    ):
    """Turn the one-shot rows into executor goals and read the plain toggles.

    The difference from `handle_sg2_joint_command_widget`: a Preset or a File load
    is **not** written into MuJoCo directly. It becomes a goal for the executor,
    so every whole-arm move in this notebook is planned and ramped, not jumped.

    `ik_solvers` (the per-arm dict from `create_sg2_arm_ik_solvers`) is only needed
    for the wrist-lift pre-step of `shut_down`; without it that step is skipped and
    the rest of the sequence still runs.

    Call exactly once per loop pass: `MultiRadioQtWidget.is_toggled` consumes the
    change event, so a second call in the same pass would miss it.
    """
    if presets is None:
        presets = SG2_JOINT_PRESETS
    message = ""

    def start(goals, label):
        try:
            executor.start(goals, label=label)
            return f"{label}: started ({executor.n_segment} segment(s))"
        except ValueError as err:
            return f"{label}: REJECTED -- {err}"

    motion_name, motion_toggled = widget.get_button_option("Motion", return_toggled=True)
    if motion_toggled and motion_name not in (None, "idle"):
        if motion_name == "stop":
            message = "stop: cancelled" if executor.cancel() else "stop: nothing was running"
        elif motion_name in SG2_GRIP_COMMAND_QPOS:
            # Gripper-only move: every other joint is held where it is.
            message = start(
                [(motion_name, make_sg2_gripper_qpos_goal(
                    executor.env, executor.joint_names, SG2_GRIP_COMMAND_QPOS[motion_name]))],
                label=motion_name,)
        elif motion_name == "shut_down":
            # Open the grippers and lift the wrists clear before folding the arms.
            # Both pre-steps depend on the pose the robot is in right now, so they
            # are built here rather than stored as constants.
            goals, notes = build_sg2_shut_down_goals(
                env         = executor.env,
                joint_names = executor.joint_names,
                ik_solvers  = ik_solvers,)
            message = start(goals, label="shut_down")
            if notes:
                message += " | " + "; ".join(notes)
        else:
            message = start(
                [(step, SG2_BRINGUP_WAYPOINTS[step]) for step in SG2_MOTION_SEQUENCES[motion_name]],
                label=motion_name,)
        widget.set_button_option("Motion", "idle")

    preset_name, preset_toggled = widget.get_button_option("Preset", return_toggled=True)
    if preset_toggled and preset_name not in (None, "idle"):
        if preset_name == "sync":
            executor.cancel()
            message = "sync: markers <- MuJoCo qpos"
        elif preset_name in presets:
            message = start([(preset_name, presets[preset_name])], label=f"preset {preset_name}")
        else:
            message = f"preset: unknown [{preset_name}]"
        widget.set_button_option("Preset", "idle")

    file_action, file_toggled = widget.get_button_option("File", return_toggled=True)
    if file_toggled and file_action not in (None, "idle"):
        if file_action == "save":
            message = f"saved: {save_sg2_joint_qpos(executor.joint_names, executor.get_qpos(), path=qpos_path)}"
        else:
            try:
                saved_joint_names, saved_qpos = load_sg2_joint_qpos(path=qpos_path)
                message = start(
                    [("file", dict(zip(saved_joint_names, [float(v) for v in saved_qpos])))],
                    label="file load",)
            except FileNotFoundError:
                message = f"load failed: no file at {qpos_path}"
        widget.set_button_option("File", "idle")

    return {
        "message"        : message,
        "check_collision": widget.get_button_option("Collision") == "on",
        "show_eef"       : widget.get_button_option("EEF") == "on",
        "show_hand"      : widget.get_button_option("Show") or "all",
        "sync_requested" : bool(preset_toggled) and preset_name == "sync",
    }


__all__ = [
    "MIN_JERK_PEAK_ACC",
    "MIN_JERK_PEAK_VEL",
    "SG2MotionExecutor",
    "SG2_BRINGUP_WAYPOINTS",
    "SG2_GRIPPER_CLOSE_QPOS",
    "SG2_GRIPPER_OPEN_QPOS",
    "SG2_GRIP_COMMAND_QPOS",
    "SG2_JOINT_ACC_RAMP_SEC",
    "SG2_MAX_PRI_JOINT_ACC",
    "SG2_MAX_REV_JOINT_ACC",
    "SG2_MOTION_COMMAND_LAYOUT",
    "SG2_MOTION_COMMAND_ROWS",
    "SG2_MOTION_SEQUENCES",
    "build_sg2_shut_down_goals",
    "check_sg2_joint_limits",
    "create_sg2_motion_command_widget",
    "estimate_duration",
    "expand_sg2_qpos",
    "format_sg2_joint_limit_violations",
    "get_sg2_joint_acc_limits",
    "get_sg2_wrist_p_at_qpos",
    "handle_sg2_motion_command_widget",
    "make_sg2_gripper_qpos_goal",
    "min_jerk_scalar",
    "plan_min_jerk",
    "solve_sg2_wrist_lift_goal",
]
