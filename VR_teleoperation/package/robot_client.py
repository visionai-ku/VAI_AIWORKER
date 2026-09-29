"""Shared strict-JSON robot access for joint/EEF/base notebook controllers.

Input widgets and IK keep producing their existing joint-group targets. This
module owns feedback, arming, command leases and bounded transmission, using the
same transport and robot-side gate as VR. It has no GUI, IK or ROS dependency.
"""
from __future__ import annotations

import copy
import time
import uuid

import numpy as np

from .fixed_quest_transport import SG2RobotCommandPublisher, SG2RobotStateReceiver
from .worker.fixed_quest_protocol import JOINT_NAMES, make_joint_groups, parse_joint_groups


class SG2RobotStateSubscriber:
    """Fresh named state for existing viewers/recorders, with no command socket.

    ``recv`` returns each source measurement at most once. Camera and rendering
    transports are independent and retain their existing message formats.
    """

    def __init__(self, robot_ip="192.168.6.2", state_port=5560, *, receiver=None):
        self.receiver = receiver if receiver is not None else SG2RobotStateReceiver(robot_ip, state_port).start()
        self.last_feedback = {"ready": False, "reason": "no_feedback"}
        self._last_measurement = None
        self._closed = False

    @property
    def reason(self):
        return self.last_feedback["reason"]

    def recv(self):
        if self._closed:
            return None
        self.last_feedback = self.receiver.latest()
        if not self.last_feedback.get("ready"):
            return None
        state = self.last_feedback["state"]
        measurement = (state["stream_id"], state["seq"])
        if measurement == self._last_measurement:
            return None
        self._last_measurement = measurement
        result = copy.deepcopy(state)
        result.setdefault("sensors", {})
        return result

    def close(self):
        self._closed = True
        self.last_feedback = {"ready": False, "reason": "receiver_closed"}
        self.receiver.close()


def get_latest_strict_robot_qpos(env, q_msg_layout, use_joint_names, sub_ip,
                                 sub_port=5560, timeout_sec=1., verbose=False):
    """Read a complete fresh state and bridge status, without a command socket."""
    receiver = SG2RobotStateReceiver(sub_ip, sub_port).start()
    deadline = time.monotonic() + float(timeout_sec)
    feedback = {"reason": "no_feedback"}
    try:
        while time.monotonic() < deadline:
            feedback = receiver.latest()
            if feedback["ready"]:
                indices = [JOINT_NAMES.index(name) for name in use_joint_names]
                return feedback["q"][indices].copy(), 1
            time.sleep(.005)
        raise ValueError("[Robot init] strict feedback unavailable: " + feedback["reason"])
    finally:
        receiver.close()


class SG2JointCommandClient:
    """One explicitly enabled controller; OFF also releases the command port.

    Call ``step`` every control tick, including while OFF. An ON edge discards
    the old preview once and returns ``synchronize=True`` so widgets can follow
    the current measured pose. A fault requires OFF before another ON attempt.
    Uncontrolled joints are held at this synchronized pose, never zero-filled.
    ``sent`` means queued locally; robot acceptance is reported separately.
    """

    def __init__(self, robot_ip="192.168.6.2", state_port=5560, command_port=5561,
                 controlled_joint_names=None, joint_speed=np.deg2rad(60.),
                 lift_speed=.05, receiver=None, publisher_factory=None):
        names = tuple(JOINT_NAMES if controlled_joint_names is None else controlled_joint_names)
        if len(set(names)) != len(names) or not set(names).issubset(JOINT_NAMES):
            raise ValueError("Unknown or repeated controlled joint")
        if not all(np.isfinite(v) and v > 0 for v in (joint_speed, lift_speed)):
            raise ValueError("Joint speeds must be finite and positive")
        self.receiver = receiver if receiver is not None else SG2RobotStateReceiver(robot_ip, state_port).start()
        self.publisher_factory = publisher_factory or (
            lambda: SG2RobotCommandPublisher(enabled=True, port=command_port))
        self.controlled = np.asarray([name in names for name in JOINT_NAMES])
        self.requested_speed = np.full(len(JOINT_NAMES), float(joint_speed))
        self.requested_speed[0] = float(lift_speed)
        self.publisher = None
        self.session_id = None
        self.identity = None
        self.sequence = 0
        self.armed = False
        self.enabled_before = False
        self.fault = None
        self.last_feedback = None
        self._last_valid_feedback = None
        self.q_command = None
        self.held_q = None
        self.last_tick = None
        self.last_send_at = None
        self.arm_started = None
        self.records = {}
        self.ack_seq = -1
        self.last_ack_at = None
        self.rate_recovery_seq = None
        self.travel = np.zeros(len(JOINT_NAMES))
        self.ack_travel = None
        self._closed = False
        self._pending_publishers = []

    def _result(self, state, reason, *, sent=False, synchronize=False):
        return dict(state=state, reason=reason, sent=sent, synchronize=synchronize,
                    q_command=None if self.q_command is None else self.q_command.copy(),
                    feedback=self.last_feedback, acknowledged_seq=self.ack_seq)

    def _send(self, mode, q, feedback, now):
        packet = dict(type="sg2_command", version=1, session_id=self.session_id,
                      seq=self.sequence, bridge_id=self.identity[1],
                      feedback_time_s=feedback["state"]["source_time_s"],
                      valid_for_s=.2, mode=mode, motion="teleop",
                      joints=make_joint_groups(q))
        self.sequence += 1
        sent = self.publisher.send(packet)
        if sent:
            if mode == "command" and self.q_command is not None:
                self.travel += np.abs(q - self.q_command)
            self.q_command = np.asarray(q, dtype=float).copy()
            self.last_send_at = now
            if mode != "stop":
                self.records[packet["seq"]] = (mode, self.travel.copy())
                if len(self.records) > 256:
                    raise ValueError("command acknowledgment stalled")
        return bool(sent)

    def _release(self):
        errors = []
        try:
            if self.publisher is not None and self.session_id is not None and self._last_valid_feedback is not None:
                self._send("stop", self.q_command, self._last_valid_feedback, time.monotonic())
        except Exception as exc:
            errors.append(str(exc))
        finally:
            pending = self._pending_publishers + ([self.publisher] if self.publisher is not None else [])
            self._pending_publishers = []
            for publisher in pending:
                try:
                    publisher.close()
                except Exception as exc:
                    self._pending_publishers.append(publisher)
                    errors.append(str(exc))
            self.publisher = None
            self.session_id = None
            self.armed = False
            self.arm_started = None
            self.last_send_at = None
            self.records.clear()
            self.ack_seq = -1
            self.last_ack_at = None
            self.rate_recovery_seq = None
            self.travel[:] = 0
            self.ack_travel = None
        if errors:
            raise RuntimeError("; ".join(errors))

    def _fail(self, reason):
        self.fault = str(reason)
        try:
            self._release()
        except Exception as exc:
            self.fault += "; cleanup: " + str(exc)
        return self._result("FAULT", self.fault)

    def _limits(self, feedback):
        limits = feedback["bridge"]["limits"]
        caps = []
        for name in JOINT_NAMES:
            key = ("lift_speed_m_s" if name == "lift_joint" else
                   "teleop_arm_joint_speed_rad_s" if name.startswith("arm_") else
                   "teleop_gripper_joint_speed_rad_s" if name.startswith("gripper_") else
                   "teleop_joint_speed_rad_s")
            caps.append(float(limits.get(key, limits["teleop_joint_speed_rad_s"])))
        speed = np.minimum(self.requested_speed, caps)
        # Without an explicit per-command gripper field the bridge uses its
        # generic rotary ceiling, including on older compatible bridges.
        speed[-2:] = np.minimum(speed[-2:], float(limits["teleop_joint_speed_rad_s"]))
        lag = np.asarray([float(limits["max_lift_lag_m"]) if name == "lift_joint" else
                          float(limits.get("max_arm_tracking_lag_rad", limits["max_tracking_lag_rad"]))
                          if name.startswith("arm_") else float(limits["max_tracking_lag_rad"])
                          for name in JOINT_NAMES])
        if not np.all(np.isfinite(speed)) or np.any(speed <= 0) or not np.all(np.isfinite(lag)) or np.any(lag <= 0):
            raise ValueError("invalid bridge limits")
        step_dt = float(limits["max_step_dt_s"])
        if not np.isfinite(step_dt) or step_dt <= 0:
            raise ValueError("invalid bridge step duration")
        return speed, lag, step_dt

    def step(self, joints, enabled=False, *, now=None, feedback=None):
        if self._closed:
            return self._result("FAULT", "client is closed")
        try:
            observed = self.receiver.latest() if feedback is None else feedback
            # Polling can block: choose the control clock after reading state,
            # then age the validated cache again before authorizing a command.
            now = time.monotonic() if now is None else float(now)
            if feedback is None and hasattr(self.receiver, "cache"):
                observed = self.receiver.cache.latest(now)
            if not isinstance(observed, dict):
                raise ValueError("invalid robot feedback")
        except Exception as exc:
            return self._fail("feedback read failed: " + str(exc))
        previous_tick, self.last_tick = self.last_tick, now
        rising = bool(enabled) and not self.enabled_before
        self.enabled_before = bool(enabled)
        if not enabled:
            try:
                self._release()
            except Exception as exc:
                return self._fail("release failed: " + str(exc))
            self.fault = None
            self.last_feedback = observed
            if observed.get("ready"):
                try:
                    measured = np.asarray(observed["q"], dtype=float)
                    make_joint_groups(measured)
                    self.identity = (observed["generation"], observed["bridge_id"])
                    self.q_command = measured.copy()
                    self._last_valid_feedback = observed
                except Exception as exc:
                    return self._fail("invalid robot feedback: " + str(exc))
            return self._result("OFF", "control disabled; command port released")
        if self.fault:
            return self._result("FAULT", self.fault + "; switch OFF then ON")
        if not np.isfinite(now) or (previous_tick is not None and
                                   (now < previous_tick or (self.session_id and now - previous_tick > .1))):
            return self._fail("control loop stalled or clock moved backwards")
        if not observed.get("ready"):
            self.last_feedback = observed
            return self._fail(observed.get("reason", "robot feedback unavailable"))
        try:
            identity = (observed["generation"], observed["bridge_id"])
            measured = np.asarray(observed["q"], dtype=float)
            make_joint_groups(measured)
        except Exception as exc:
            return self._fail("invalid robot feedback: " + str(exc))
        if self.identity is not None and identity != self.identity:
            return self._fail("robot bridge restarted; switch OFF then ON to synchronize")
        self.last_feedback = observed
        self._last_valid_feedback = observed
        self.identity = identity
        try:
            target = np.asarray(parse_joint_groups(joints), dtype=float)
            speed, lag, max_step_dt = self._limits(observed)
            owner = observed.get("owner_session_id")
            if owner not in (None, self.session_id):
                return self._fail("another controller owns the robot; release it before enabling")
            if rising:
                self.publisher = self.publisher_factory()
                self.session_id = str(uuid.uuid4())
                self.sequence = 0
                self.arm_started = now
                self.held_q = measured.copy()
                self.q_command = measured.copy()
                sent = self._send("arm", measured, observed, now)
                return self._result("ARMING", "synchronized to measured pose; awaiting bridge", sent=sent, synchronize=True)
            if self.session_id is None:
                return self._fail("switch OFF then ON to start a new session")
            if not self.armed:
                if owner == self.session_id and observed.get("bridge_state") == "ACTIVE":
                    self.armed = True
                elif now - self.arm_started > .5:
                    return self._fail("robot did not acknowledge arming")
                else:
                    sent = self._send("arm", measured, observed, now)
                    self.held_q = measured.copy()
                    return self._result("ARMING", "awaiting bridge acknowledgment", sent=sent, synchronize=True)
            if owner != self.session_id or observed.get("bridge_state") != "ACTIVE":
                return self._fail("robot command lease lost")
            rejection = observed["bridge"].get("reason", "accepted")
            if rejection in ("tracking_lag", "feedback_echo_expired",
                             "repeated_sequence", "invalid_ttl", "explicit_arm_required"):
                return self._fail("robot rejected command: " + rejection)
            ack = observed["last_command_seq"]
            if type(ack) is not int or ack < self.ack_seq:
                return self._fail("robot acknowledgment regressed")
            if ack != self.ack_seq:
                if ack not in self.records:
                    return self._fail("unknown robot acknowledgment")
                mode, travel = self.records[ack]
                if mode == "command":
                    self.ack_travel = travel.copy()
                self.ack_seq = ack
                self.last_ack_at = now
                self.records = {seq: record for seq, record in self.records.items() if seq >= ack}
            if self.last_ack_at is None or now - self.last_ack_at > .2:
                return self._fail("robot command acknowledgment stalled")
            target[~self.controlled] = self.held_q[~self.controlled]
            gap = self.q_command - measured
            if np.any(np.abs(gap) > lag):
                return self._fail("robot is not following the commanded pose")
            # Arrival intervals at the robot can be shorter than PC send
            # intervals. A rate rejection does not accept a new pose or renew
            # the lease. Freeze the already bounded in-flight target and retry
            # it, without spending more travel, until that pose is acknowledged.
            # Feedback, ownership, ACK and loop deadlines above still apply.
            if rejection == "joint_rate_limit" and self.rate_recovery_seq is None:
                self.rate_recovery_seq = self.sequence - 1
            if self.rate_recovery_seq is not None:
                if rejection == "accepted" and ack >= self.rate_recovery_seq:
                    self.rate_recovery_seq = None
                else:
                    sent = self._send("command", self.q_command, observed, now)
                    if not sent:
                        return self._fail("robot command could not be queued")
                    return self._result("ACTIVE", "holding queued pose; waiting for joint_rate_limit recovery",
                                        sent=sent)
            # Bound each step and all unacknowledged travel. Coordinated scaling
            # preserves the segment supplied by EEF IK instead of mixing axes.
            dt = min(.02, max_step_dt, max(0., now - self.last_send_at))
            delta = target - self.q_command
            allowance = speed * dt * .9
            if self.ack_travel is None:
                allowance[:] = 0.  # Baseline command itself needs an ACK.
            else:
                window = min(.04, max_step_dt * .8)
                allowance = np.minimum(allowance, np.maximum(0., speed * window - (self.travel - self.ack_travel)))
            moving = np.abs(delta) > 1e-15
            scale = min(1., float(np.min(allowance[moving] / np.abs(delta[moving])))) if np.any(moving) else 1.
            for change, distance, limit in zip(delta, gap, lag * .8):
                if change > 0:
                    scale = min(scale, max(0., (max(limit, abs(distance)) - distance) / change))
                elif change < 0:
                    scale = min(scale, max(0., (max(limit, abs(distance)) + distance) / -change))
            candidate = self.q_command + scale * delta
            sent = self._send("command", candidate, observed, now)
            if not sent:
                return self._fail("robot command could not be queued")
            return self._result("ACTIVE", "commanding; robot acknowledgments monitored", sent=sent)
        except Exception as exc:
            return self._fail(str(exc))

    def close(self):
        self._closed = True
        try:
            self._release()
        finally:
            self.receiver.close()


class SG2BaseCommandClient:
    """Leased base-only commands, sharing exclusive ownership with joint control.

    Enabling always arms at zero and waits for an acknowledged zero command.
    Faults latch until OFF; OFF releases the PUB socket. The robot watchdog is
    still responsible for stopping if the PC disappears before sending stop.
    """

    def __init__(self, robot_ip="192.168.6.2", state_port=5560, command_port=5561,
                 *, receiver=None, publisher_factory=None):
        self.receiver = receiver if receiver is not None else SG2RobotStateReceiver(robot_ip, state_port).start()
        self.publisher_factory = publisher_factory or (
            lambda: SG2RobotCommandPublisher(enabled=True, port=command_port))
        self.publisher = None
        self._pending_publishers = []
        self.session_id = None
        self.identity = None
        self.sequence = 0
        self.enabled_before = False
        self.armed = False
        self.arm_started = None
        self.baseline_seq = None
        self.last_tick = None
        self.last_ack_at = None
        self.ack_seq = -1
        self.records = set()
        self.fault = None
        self.last_feedback = None
        self._last_valid_feedback = None
        self.base_command = np.zeros(3)
        self._closed = False

    def _result(self, state, reason, *, sent=False):
        return dict(state=state, reason=reason, sent=sent,
                    base_command=self.base_command.copy(), feedback=self.last_feedback,
                    acknowledged_seq=self.ack_seq)

    def _send(self, mode, base, feedback):
        seq = self.sequence
        packet = dict(type="sg2_base_command", version=1, session_id=self.session_id,
                      seq=seq, bridge_id=self.identity[1],
                      feedback_time_s=feedback["state"]["source_time_s"], valid_for_s=.2,
                      mode=mode, base=[float(v) for v in base])
        self.sequence += 1
        if not self.publisher.send(packet):
            raise RuntimeError("base command could not be queued")
        self.base_command = np.asarray(base, dtype=float).copy()
        if mode != "stop":
            self.records.add(seq)
            if len(self.records) > 256:
                raise RuntimeError("base acknowledgment stalled")
        return seq

    def _release(self):
        errors = []
        try:
            if self.publisher is not None and self.session_id is not None and self._last_valid_feedback is not None:
                self._send("stop", np.zeros(3), self._last_valid_feedback)
        except Exception as exc:
            errors.append(str(exc))
        finally:
            pending = self._pending_publishers + ([self.publisher] if self.publisher is not None else [])
            self._pending_publishers = []
            for publisher in pending:
                try:
                    publisher.close()
                except Exception as exc:
                    self._pending_publishers.append(publisher)
                    errors.append(str(exc))
            self.publisher = None
            self.session_id = None
            self.armed = False
            self.arm_started = self.baseline_seq = None
            self.last_ack_at = None
            self.ack_seq = -1
            self.records.clear()
            self.base_command[:] = 0.
        if errors:
            raise RuntimeError("; ".join(errors))

    def _fail(self, reason):
        self.fault = str(reason)
        try:
            self._release()
        except Exception as exc:
            self.fault += "; cleanup: " + str(exc)
        return self._result("FAULT", self.fault)

    def step(self, base, enabled=False, *, now=None, feedback=None):
        if self._closed:
            return self._result("FAULT", "client is closed")
        rising = bool(enabled) and not self.enabled_before
        self.enabled_before = bool(enabled)
        if not enabled:
            try:
                self._release()
            except Exception as exc:
                return self._fail("release failed: " + str(exc))
            self.fault = None
            self.identity = None
            self.last_tick = None
            return self._result("OFF", "control disabled; command port released")
        if self.fault:
            return self._result("FAULT", self.fault + "; switch OFF then ON")
        try:
            observed = self.receiver.latest() if feedback is None else feedback
            now = time.monotonic() if now is None else float(now)
            if feedback is None and hasattr(self.receiver, "cache"):
                observed = self.receiver.cache.latest(now)
            previous_tick, self.last_tick = self.last_tick, now
            if not np.isfinite(now) or (previous_tick is not None and
                    (now < previous_tick or now - previous_tick > .1)):
                raise ValueError("base control loop stalled or clock moved backwards")
            if not isinstance(observed, dict) or not observed.get("ready"):
                raise ValueError(observed.get("reason", "robot feedback unavailable")
                                 if isinstance(observed, dict) else "invalid robot feedback")
            identity = (observed["generation"], observed["bridge_id"])
            if self.identity is not None and identity != self.identity:
                raise ValueError("robot bridge restarted; switch OFF then ON")
            bridge = observed["bridge"]
            if bridge.get("capabilities", {}).get("base_json") is not True:
                raise ValueError("robot worker does not support Base JSON; update worker first")
            linear = float(bridge["limits"]["base_linear_speed_m_s"])
            angular = float(bridge["limits"]["base_angular_speed_rad_s"])
            if not all(np.isfinite(v) and v > 0 for v in (linear, angular)):
                raise ValueError("invalid base limits")
            target = np.asarray(base, dtype=float)
            if target.shape != (3,) or not np.all(np.isfinite(target)):
                raise ValueError("base target must be finite [vx, vy, wz]")
            target = target.copy()
            norm = np.linalg.norm(target[:2])
            target[:2] *= min(1., min(.3, linear) / norm) if norm else 1.
            target[2] = np.clip(target[2], -min(.6, angular), min(.6, angular))
            owner = observed.get("owner_session_id")
            if owner not in (None, self.session_id):
                raise ValueError("another controller owns the robot")
            self.identity = identity
            self.last_feedback = self._last_valid_feedback = observed
            if rising:
                self.publisher = self.publisher_factory()
                self.session_id = str(uuid.uuid4())
                self.sequence = 0
                self.arm_started = now
                self._send("arm", np.zeros(3), observed)
                return self._result("ARMING", "arming at zero; awaiting bridge", sent=True)
            if self.session_id is None:
                raise ValueError("switch OFF then ON to start a new session")
            if not self.armed:
                if owner == self.session_id and observed.get("bridge_state") == "ACTIVE":
                    if bridge.get("command_kind") != "base":
                        raise ValueError("robot command kind mismatch")
                    self.armed = True
                elif now - self.arm_started > .5:
                    raise ValueError("robot did not acknowledge base arming")
                else:
                    self._send("arm", np.zeros(3), observed)
                    return self._result("ARMING", "awaiting bridge acknowledgment", sent=True)
            if owner != self.session_id or observed.get("bridge_state") != "ACTIVE":
                raise ValueError("robot base command lease lost")
            if bridge.get("command_kind") != "base":
                raise ValueError("robot command kind mismatch")
            if bridge.get("reason") != "accepted":
                raise ValueError("robot rejected base command: " + str(bridge.get("reason")))
            ack = observed["last_command_seq"]
            if type(ack) is not int or ack < self.ack_seq:
                raise ValueError("robot acknowledgment regressed")
            if ack != self.ack_seq:
                if ack not in self.records:
                    raise ValueError("unknown robot acknowledgment")
                self.ack_seq, self.last_ack_at = ack, now
                self.records = {seq for seq in self.records if seq >= ack}
            if self.last_ack_at is None or now - self.last_ack_at > .2:
                raise ValueError("robot base acknowledgment stalled")
            if self.baseline_seq is None:
                self.baseline_seq = self._send("command", np.zeros(3), observed)
                return self._result("ARMING", "awaiting zero-command acknowledgment", sent=True)
            if ack < self.baseline_seq:
                self._send("command", np.zeros(3), observed)
                return self._result("ARMING", "awaiting zero-command acknowledgment", sent=True)
            self._send("command", target, observed)
            return self._result("ACTIVE", "base command queued; acknowledgments monitored", sent=True)
        except Exception as exc:
            return self._fail(str(exc))

    def close(self):
        self._closed = True
        try:
            self._release()
        finally:
            self.receiver.close()
