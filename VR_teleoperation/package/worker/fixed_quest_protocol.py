"""Strict SG2 message validation, independent of ROS, NumPy and MuJoCo.

Times in this protocol are the robot's monotonic seconds. The PC echoes the
feedback timestamp; its own clock is never used to authorize a robot command.
"""
import json
import math
import uuid


JOINT_GROUPS = {
    "lift": ["lift_joint"],
    "head": ["head_joint1", "head_joint2"],
    "arm_l": ["arm_l_joint%d" % i for i in range(1, 8)],
    "arm_r": ["arm_r_joint%d" % i for i in range(1, 8)],
    "gripper_l": ["gripper_l_joint1"],
    "gripper_r": ["gripper_r_joint1"],
}
JOINT_NAMES = tuple(name for names in JOINT_GROUPS.values() for name in names)
# ROBOTIS cyclo_control 8d982a0: SG2 URDF arm velocity bounds, VR timer
# and activation ramp. This is a command ceiling, not measured motor speed.
OFFICIAL_VR_ARM_SPEED = 4.8
OFFICIAL_VR_CONTROL_HZ = 100.0
OFFICIAL_VR_RAMP_S = 8.0
# Our bounded command lead (not an official ROBOTIS parameter): 100 ms of
# full-speed arm travel. Home/head/grippers retain the existing 5-degree guard.
VR_ARM_TRACKING_LAG = OFFICIAL_VR_ARM_SPEED * .1
FIXED_QUEST_MIN_ARM_SPEED = math.radians(10)
FIXED_QUEST_MAX_ARM_SPEED = math.radians(120)
FIXED_QUEST_DEFAULT_ARM_SPEED = math.radians(60)
FIXED_QUEST_MAX_GRIPPER_SPEED = math.radians(120)
# Feedback-only encoder boundary allowance (0.02865 degrees), limited to the
# shoulder joints whose home position lies exactly on a mechanical bound.
# Commands retain JOINT_LIMITS without any allowance.
FEEDBACK_BOUNDARY_TOLERANCES = {"arm_l_joint2": .0005, "arm_r_joint2": .0005}
# Controller-button reset keeps initial arm alignment at 30 deg/s, then permits
# the remaining fold/bringup steps at 60 deg/s. Normal HOME still requests 10 deg/s.
FIXED_QUEST_VR_HOME_SPEED = math.radians(60)
FIXED_QUEST_VR_GATE_OPTIONS = {
    "max_home_joint_speed": FIXED_QUEST_VR_HOME_SPEED,
    "max_arm_joint_speed": FIXED_QUEST_MAX_ARM_SPEED,
    "max_arm_tracking_lag": VR_ARM_TRACKING_LAG,
    "max_gripper_joint_speed": FIXED_QUEST_MAX_GRIPPER_SPEED,
}
FIXED_QUEST_VR_BRIDGE_LIMITS = {
    "teleop_arm_joint_speed_rad_s": FIXED_QUEST_MAX_ARM_SPEED,
    "max_arm_tracking_lag_rad": VR_ARM_TRACKING_LAG,
    "command_arm_speed_limit": True,
}
# Radian/metre ranges from xml/ffw_sg2.xml. Wrist7 uses the conservative
# intersection with official SG2 rev1 URDF (the asymmetric ranges differ).
JOINT_LIMITS = {
    "lift_joint": (-.5, 0.), "head_joint1": (-.2317, .6951), "head_joint2": (-.35, .35),
    "arm_l_joint1": (-3.14, 3.14), "arm_l_joint2": (0., 3.14),
    "arm_l_joint3": (-3.14, 3.14), "arm_l_joint4": (-2.9361, 1.0786),
    "arm_l_joint5": (-3.14, 3.14), "arm_l_joint6": (-1.57, 1.57), "arm_l_joint7": (-1.5804, 1.5804),
    "arm_r_joint1": (-3.14, 3.14), "arm_r_joint2": (-3.14, 0.),
    "arm_r_joint3": (-3.14, 3.14), "arm_r_joint4": (-2.9361, 1.0786),
    "arm_r_joint5": (-3.14, 3.14), "arm_r_joint6": (-1.57, 1.57), "arm_r_joint7": (-1.5804, 1.5804),
    "gripper_l_joint1": (0., 1.1), "gripper_r_joint1": (0., 1.1),
}


def _number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(label)
    return float(value)


def _joint_limit_message(name, value):
    low, high = JOINT_LIMITS[name]
    unit = "m" if name == "lift_joint" else "rad"
    return ("joint_limit:%s: position=%.12g %s, allowed=[%.12g, %.12g] %s, "
            "excess=%.12g %s" % (name, value, unit, low, high, unit,
                                  max(low - value, value - high), unit))


def _identifier(value, label):
    if not isinstance(value, str) or not 1 <= len(value) <= 128:
        raise ValueError(label)
    return value


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(label)
    return value


def _constant_rejected(value):
    raise ValueError("non_finite_json")


def load_json_message(payload):
    """Decode only bounded JSON objects, with no pickle or non-finite constants."""
    if not isinstance(payload, (str, bytes)) or len(payload) > 65536:
        raise ValueError("invalid_message_size")
    data = json.loads(payload, parse_constant=_constant_rejected)
    if not isinstance(data, dict):
        raise ValueError("invalid_message")
    return data


def parse_joint_groups(joints, named=False):
    """Read all 19 joints in JOINT_NAMES order; reject missing or extra groups."""
    if not isinstance(joints, dict) or set(joints) != set(JOINT_GROUPS):
        raise ValueError("incomplete_joint_groups")
    positions = {}
    for group, names in JOINT_GROUPS.items():
        values = joints[group]
        if named:
            if not isinstance(values, dict) or set(values) != {"name", "position"}:
                raise ValueError("invalid_joint_group:" + group)
            actual_names, values = values["name"], values["position"]
            if not isinstance(actual_names, list) or len(actual_names) != len(names) or set(actual_names) != set(names):
                raise ValueError("invalid_joint_names:" + group)
        else:
            actual_names = names
        if not isinstance(values, list) or len(values) != len(names):
            raise ValueError("invalid_joint_positions:" + group)
        for name, value in zip(actual_names, values):
            value = _number(value, "non_finite_joint:" + name)
            low, high = JOINT_LIMITS[name]
            if not low <= value <= high:
                raise ValueError(_joint_limit_message(name, value))
            positions[name] = value
    return [positions[name] for name in JOINT_NAMES]


def make_joint_groups(q, named=False):
    """Create either command lists or feedback name/position groups."""
    if len(q) != len(JOINT_NAMES):
        raise ValueError("invalid_joint_count")
    by_name = dict(zip(JOINT_NAMES, q))
    groups = {group: [float(by_name[name]) for name in names] for group, names in JOINT_GROUPS.items()}
    parse_joint_groups(groups)
    if named:
        return {group: {"name": list(JOINT_GROUPS[group]), "position": values} for group, values in groups.items()}
    return groups


class JointStateTracker:
    """Track real callback freshness per required joint, never zero-fill gaps."""

    def __init__(self, timeout_s=.2, stream_id=None, boundary_tolerances=None):
        self.timeout_s = _number(timeout_s, "invalid_timeout")
        if self.timeout_s <= 0:
            raise ValueError("invalid_timeout")
        self.stream_id = stream_id or str(uuid.uuid4())
        self.seq = 0
        self.positions, self.updated_at, self.stamps = {}, {}, {}
        self.last_callback_at = None
        self.last_callback_result = "no_callback"
        self.last_rejection = None
        self.boundary_tolerances = dict(FEEDBACK_BOUNDARY_TOLERANCES if boundary_tolerances is None else boundary_tolerances)
        for name, value in self.boundary_tolerances.items():
            if name not in JOINT_LIMITS or not 0 <= _number(value, "invalid_boundary_tolerance") <= .0005:
                raise ValueError("invalid_boundary_tolerance")
        self.boundary_adjustments = {}
        self.boundary_adjustment_count = 0

    def diagnostics(self, now):
        """Observe rejected ROS feedback without refreshing its validity/age."""
        return {
            "callback_age_s": None if self.last_callback_at is None else max(0., now - self.last_callback_at),
            "last_callback_result": self.last_callback_result,
            "last_rejection": None if self.last_rejection is None else dict(self.last_rejection),
            "boundary_adjustments": {name: dict(value) for name, value in self.boundary_adjustments.items()},
            "boundary_adjustment_count": self.boundary_adjustment_count,
            "stale_joints": [name for name in JOINT_NAMES if name in self.updated_at
                             and now - self.updated_at[name] > self.timeout_s],
            "missing_joints": [name for name in JOINT_NAMES if name not in self.positions],
        }

    def update(self, names, positions, source_stamp_ns, now):
        now = _number(now, "invalid_now")
        self.last_callback_at = now
        try:
            changed = self._update_validated(names, positions, source_stamp_ns, now)
        except (ValueError, TypeError, OverflowError) as exc:
            self.last_callback_result = "rejected"
            self.last_rejection = {"at_s": now, "reason": str(exc)}
            raise
        self.last_callback_result = "updated" if changed else "no_source_progress"
        return changed

    def _update_validated(self, names, positions, source_stamp_ns, now):
        now = _number(now, "invalid_now")
        stamp = _integer(source_stamp_ns, "invalid_source_stamp", minimum=1)
        if len(names) != len(positions) or len(set(names)) != len(names):
            raise ValueError("invalid_joint_state")
        updates, adjustments = {}, {}
        for name, value in zip(names, positions):
            if name not in JOINT_LIMITS:
                continue
            value = _number(value, "non_finite_joint:" + name)
            low, high = JOINT_LIMITS[name]
            if not low <= value <= high:
                tolerance = self.boundary_tolerances.get(name, 0.)
                if max(low - value, value - high) > tolerance:
                    raise ValueError(_joint_limit_message(name, value))
                normalized = min(high, max(low, value))
                adjustments[name] = {"raw_position": value, "position": normalized,
                                     "correction_rad": normalized - value, "at_s": now}
                value = normalized
            if stamp > self.stamps.get(name, -1):
                if now < self.updated_at.get(name, -math.inf):
                    raise ValueError("non_monotonic_clock")
                updates[name] = value
        for name, value in updates.items():
            self.positions[name], self.updated_at[name], self.stamps[name] = value, now, stamp
        if updates:
            self.seq += 1
            self.boundary_adjustments = {name: value for name, value in adjustments.items() if name in updates}
            self.boundary_adjustment_count += len(self.boundary_adjustments)
        return bool(updates)

    def snapshot(self, now):
        now = _number(now, "invalid_now")
        missing = [name for name in JOINT_NAMES if name not in self.positions]
        source_time = min(self.updated_at.values()) if self.updated_at else None
        age = None if source_time is None else now - source_time
        ready = not missing and 0 <= age <= self.timeout_s
        return {
            "type": "sg2_state", "version": 1, "stream_id": self.stream_id, "seq": self.seq,
            "source_time_s": source_time, "sent_at_s": now,
            "source_stamp_ns": min(self.stamps.values()) if self.stamps else None,
            "ready": ready, "state": "incomplete_joint_state" if missing else ("ready" if ready else "stale_joint_state"),
            "missing_joints": missing,
            "joint_state_diagnostics": self.diagnostics(now),
            "joints": {} if missing else make_joint_groups([self.positions[name] for name in JOINT_NAMES], named=True),
        }

    def current(self, now):
        snapshot = self.snapshot(now)
        if not snapshot["ready"]:
            raise ValueError(snapshot["state"])
        return parse_joint_groups(snapshot["joints"], named=True)


class StrictCommandGate:
    """One owner, explicit arming, finite command lease and bounded joint steps.

    Each command selects teleop or home (10 degrees/s). Generic rotary teleop
    defaults to 60 degrees/s; the fixed-Quest worker explicitly selects the
    120 degree/s arm ceiling. A teleop command may request a lower arm cap.
    Lower per-joint speeds also cap home.
    Lift keeps its metre/s limit.
    A timeout emits at most one hold at a fresh measured pose. This is a software
    command gate; it cannot establish the hardware controller's stop behavior.
    """

    def __init__(self, tracker, bridge_id=None, timeout_s=.2, max_step_dt=.05,
                 max_joint_speed=math.radians(60), max_lift_speed=.01,
                 arm_tolerance=math.radians(1), arm_lift_tolerance=.002,
                 max_tracking_lag=math.radians(5), max_lift_lag=.02,
                 max_home_joint_speed=math.radians(10),
                 max_arm_joint_speed=None, max_arm_tracking_lag=None, max_gripper_joint_speed=None):
        self.tracker, self.bridge_id = tracker, bridge_id or str(uuid.uuid4())
        max_arm_joint_speed = max_joint_speed if max_arm_joint_speed is None else max_arm_joint_speed
        max_arm_tracking_lag = max_tracking_lag if max_arm_tracking_lag is None else max_arm_tracking_lag
        max_gripper_joint_speed = max_joint_speed if max_gripper_joint_speed is None else max_gripper_joint_speed
        parameters = (timeout_s, max_step_dt, max_joint_speed, max_lift_speed,
                      arm_tolerance, arm_lift_tolerance, max_tracking_lag, max_lift_lag,
                      max_home_joint_speed, max_arm_joint_speed, max_arm_tracking_lag, max_gripper_joint_speed)
        if any(_number(value, "invalid_gate_parameter") <= 0 for value in parameters) or timeout_s > .2:
            raise ValueError("invalid_gate_parameter")
        self.timeout_s, self.max_step_dt = timeout_s, max_step_dt
        self.speed = [max_lift_speed if name == "lift_joint" else
                      max_gripper_joint_speed if name.startswith("gripper_") else
                      max_arm_joint_speed if name.startswith("arm_") else max_joint_speed for name in JOINT_NAMES]
        self.home_speed = [speed if name == "lift_joint" else min(speed, max_home_joint_speed)
                           for name, speed in zip(JOINT_NAMES, self.speed)]
        self.arm_tolerance = [arm_lift_tolerance if name == "lift_joint" else arm_tolerance for name in JOINT_NAMES]
        self.lag_limit = [max_lift_lag if name == "lift_joint" else max_tracking_lag for name in JOINT_NAMES]
        self.home_lag_limit = self.lag_limit.copy()
        self.lag_limit = [max_arm_tracking_lag if name.startswith("arm_") else limit
                          for name, limit in zip(JOINT_NAMES, self.lag_limit)]
        self._limits = {
            "teleop_joint_speed_rad_s": float(max_joint_speed),
            "teleop_arm_joint_speed_rad_s": float(max_arm_joint_speed),
            "max_arm_tracking_lag_rad": float(max_arm_tracking_lag),
            "command_arm_speed_limit": True,
            "teleop_gripper_joint_speed_rad_s": float(max_gripper_joint_speed),
            "command_gripper_speed_limit": True,
            "home_joint_speed_rad_s": float(min(max_joint_speed, max_home_joint_speed)),
            "lift_speed_m_s": float(max_lift_speed), "max_step_dt_s": float(max_step_dt),
            "max_tracking_lag_rad": float(max_tracking_lag), "max_lift_lag_m": float(max_lift_lag),
        }
        self.owner_session_id, self.last_command_seq = None, -1
        self.state, self.reason = "DISARMED", "awaiting_arm"
        self.last_q, self.last_command_at, self.deadline = None, None, None
        self.session_sequences = {}
        self.expired_sessions = set()
        self.command_started = False
        self.rejection_detail = None

    def status(self, now):
        return {
            "type": "sg2_bridge", "version": 1, "bridge_id": self.bridge_id, "strict": True,
            "ready": self.tracker.snapshot(now)["ready"], "status_time_s": float(now),
            "state": self.state, "reason": self.reason, "owner_session_id": self.owner_session_id,
            "last_command_seq": self.last_command_seq,
            "limits": dict(self._limits),
            "joint_state_diagnostics": self.tracker.diagnostics(now),
            "rejection_detail": None if self.rejection_detail is None else dict(self.rejection_detail),
        }

    def _hold(self, now, state, reason):
        try:
            joints = make_joint_groups(self.tracker.current(now))
        except ValueError:
            joints = None
        if state == "EXPIRED" and self.owner_session_id is not None:
            self.expired_sessions.add(self.owner_session_id)
        self.owner_session_id, self.deadline = None, None
        self.command_started = False
        self.state, self.reason = state, reason
        self.rejection_detail = None
        return {"accepted": False, "reason": reason, "publish_joints": joints}

    def expire(self, now):
        now = _number(now, "invalid_now")
        if self.owner_session_id is not None:
            if self.deadline is None or now > self.deadline:
                return self._hold(now, "EXPIRED", "command_expired")
            if not self.tracker.snapshot(now)["ready"]:
                return self._hold(now, "EXPIRED", "feedback_expired")
        return {"accepted": False, "reason": "no_expiry", "publish_joints": None}

    def receive(self, message, now):
        """Validate without publishing; the ROS wrapper owns the actual send."""
        now = _number(now, "invalid_now")
        expired = self.expire(now)
        try:
            if not isinstance(message, dict) or message.get("type") != "sg2_command" or type(message.get("version")) is not int or message["version"] != 1:
                raise ValueError("invalid_command_version")
            session = _identifier(message.get("session_id"), "invalid_session")
            seq = _integer(message.get("seq"), "invalid_sequence")
            if message.get("bridge_id") != self.bridge_id:
                raise ValueError("wrong_bridge")
            mode = message.get("mode")
            if mode not in ("arm", "command", "stop"):
                raise ValueError("invalid_mode")
            # Older PCs omit motion and keep the teleoperation rate limit.
            # Select the limit from this packet, including the first home step.
            motion = message.get("motion", "teleop")
            if motion not in ("teleop", "home"):
                raise ValueError("invalid_motion")
            if self.owner_session_id not in (None, session):
                raise ValueError("owner_conflict")
            if seq <= self.session_sequences.get(session, -1):
                raise ValueError("repeated_sequence")
            q = parse_joint_groups(message.get("joints"))
            ttl = _number(message.get("valid_for_s"), "invalid_ttl")
            feedback_time = _number(message.get("feedback_time_s"), "invalid_feedback_time")
            if not 0 < ttl <= self.timeout_s:
                raise ValueError("invalid_ttl")
            if mode == "stop":
                result = self._hold(now, "STOPPED", "stopped")
                self.session_sequences[session], self.last_command_seq = seq, seq
                result["accepted"] = True
                return result
            if not 0 <= now - feedback_time <= ttl:
                raise ValueError("feedback_echo_expired")
            measured = self.tracker.current(now)
            if mode == "arm":
                if session in self.expired_sessions:
                    raise ValueError("new_session_required")
                if self.owner_session_id is not None and self.command_started:
                    raise ValueError("already_armed")
                if any(abs(a - b) > tolerance + 1e-12 for a, b, tolerance in zip(q, measured, self.arm_tolerance)):
                    raise ValueError("arm_pose_mismatch")
                q = measured
            else:
                if self.owner_session_id != session or self.state != "ACTIVE":
                    raise ValueError("explicit_arm_required")
                elapsed = now - self.last_command_at
                dt = min(max(elapsed, 0.), self.max_step_dt)
                tolerances = [.0001 if name == "lift_joint" else .001 for name in JOINT_NAMES]
                speeds = self.home_speed if motion == "home" else self.speed
                if motion == "teleop":
                    # Legacy senders without the new field keep the old ceiling.
                    requested_gripper = _number(message.get("gripper_speed_rad_s", min(max(self.speed[-2:]),
                        self._limits["teleop_joint_speed_rad_s"])), "invalid_gripper_speed")
                    if not 0 < requested_gripper <= self._limits["teleop_gripper_joint_speed_rad_s"]:
                        raise ValueError("gripper_speed_exceeds_limit")
                    speeds = [min(speed, requested_gripper) if name.startswith("gripper_") else speed
                              for name, speed in zip(JOINT_NAMES, speeds)]
                if motion == "teleop" and "arm_speed_rad_s" in message:
                    requested_speed = _number(message["arm_speed_rad_s"], "invalid_arm_speed")
                    arm_ceiling = self._limits["teleop_arm_joint_speed_rad_s"]
                    if not 0 < requested_speed <= arm_ceiling:
                        raise ValueError("arm_speed_exceeds_limit")
                    speeds = [min(speed, requested_speed) if name.startswith("arm_") else speed
                              for name, speed in zip(JOINT_NAMES, speeds)]
                for name, a, b, speed, tolerance in zip(JOINT_NAMES, q, self.last_q, speeds, tolerances):
                    delta, allowed = abs(a - b), speed * dt + tolerance
                    if delta > allowed:
                        # Keep one bounded diagnostic for the first violation;
                        # this uses exactly the same comparison as enforcement.
                        self.rejection_detail = {
                            "reason": "joint_rate_limit", "joint_name": name,
                            "delta": delta, "unit": "m" if name == "lift_joint" else "rad",
                            "elapsed_dt_s": elapsed, "capped_dt_s": dt,
                            "allowed_step": allowed, "speed_limit": float(speed),
                            "tolerance": tolerance, "motion": motion,
                            "command_seq": seq, "last_accepted_seq": self.last_command_seq,
                        }
                        raise ValueError("joint_rate_limit")
                lag_limits = self.home_lag_limit if motion == "home" else self.lag_limit
                if any(abs(a - b) > limit + 1e-12 for a, b, limit in zip(q, measured, lag_limits)):
                    raise ValueError("tracking_lag")
                self.command_started = True
            self.owner_session_id, self.last_command_seq = session, seq
            self.session_sequences[session] = seq
            self.last_q, self.last_command_at = q, now
            self.deadline = min(now + self.timeout_s, feedback_time + ttl)
            self.state, self.reason = "ACTIVE", "accepted"
            self.rejection_detail = None
            return {"accepted": True, "reason": "accepted", "publish_joints": make_joint_groups(q)}
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            self.reason = str(exc)
            return {"accepted": False, "reason": str(exc), "publish_joints": expired["publish_joints"]}
