"""Real output gate around the tested controller mapper and existing SG2 IK.

The injected planner owns command-preview MuJoCo state; env_measured owns robot
feedback. This module never steps physics or invents missing robot positions.
"""
from __future__ import annotations

import copy
import time
import uuid

import numpy as np

from utils import get_qmsg_from_qpos, get_sg2_wrist_control_T, set_sg2_gripper_qpos
from .controller_mapping import ControllerMapper
from .fixed_quest_sim import wrist_error
from .home_path import home_plan_summary, plan_home_path
from .vr_buttons import BRINGUP_POSE
from .worker.fixed_quest_protocol import (
    JOINT_NAMES, JOINT_LIMITS, FIXED_QUEST_MIN_ARM_SPEED, FIXED_QUEST_MAX_ARM_SPEED,
    FIXED_QUEST_MAX_GRIPPER_SPEED,
)

# Requested gripper speed is negotiated separately from head/arm/HOME limits.
FIXED_QUEST_MIN_GRIPPER_SPEED = np.deg2rad(10.)
# Hold commands keep refreshing the bridge lease while paused, so a pause may
# outlast the 200 ms command TTL.
FIXED_QUEST_MAX_TRACKING_PAUSE_S = 1.


def _tracking_pause(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value) or not 0 <= value <= FIXED_QUEST_MAX_TRACKING_PAUSE_S):
        raise ValueError("tracking_pause_s must be between 0 and %g seconds" % FIXED_QUEST_MAX_TRACKING_PAUSE_S)
    return float(value)


_STARTUP_INPUT_REASONS = frozenset({
    "no_frame", "stale_pose", "session_not_visible",
    "tracking_lost_left", "tracking_lost_right",
    "emulated_pose_left", "emulated_pose_right",
    "head_calibration_unavailable", "head_yaw_unavailable",
})


class _DisabledPublisher:
    """Never opens or sends; retain failed cleanup for a later close retry."""

    enabled = False

    def __init__(self):
        self._pending_publishers = []

    def send(self, packet):
        return False

    def close(self):
        pending, self._pending_publishers = self._pending_publishers, []
        failure = None
        for publisher in pending:
            try:
                publisher.close()
            except Exception as exc:
                self._pending_publishers.append(publisher)
                if failure is None:
                    failure = exc
        if failure is not None:
            raise RuntimeError("previous publisher cleanup failed: " + str(failure)) from failure


class SG2FixedQuestReal:
    def __init__(self, planner, env_measured, q_msg_layout, publisher, dry_run=True,
                 max_joint_lag=np.deg2rad(5), max_lift_lag=.01, command_ttl_s=.2,
                 max_home_joint_speed=np.deg2rad(10), max_arm_joint_lag=None,
                 teleop_ramp_s=0., required_bridge_limits=None, command_arm_speed_limit=False,
                 tracking_pause_s=0.):
        if planner.mode != "kinematic" or env_measured in (planner.env, planner.env_ik, planner.guard.env_check):
            raise ValueError("Real adapter needs separate measured and command-preview environments")
        if list(planner.joint_names) != list(JOINT_NAMES):
            raise ValueError("Real joint order must match the robot bridge exactly")
        if bool(publisher.enabled) == bool(dry_run):
            raise ValueError("Shadow mode must not allocate a robot publisher; live mode requires one")
        if not (all(np.isfinite(value) for value in (command_ttl_s, max_joint_lag, max_lift_lag,
                                                    max_home_joint_speed))
                and 0 < command_ttl_s <= .2 and max_joint_lag > 0 and max_lift_lag > 0
                and max_home_joint_speed > 0):
            raise ValueError("Invalid real command limits")
        self.planner, self.env_measured, self.publisher = planner, env_measured, publisher
        self.q_msg_layout, self.dry_run = q_msg_layout, bool(dry_run)
        self.command_ttl_s = command_ttl_s
        self.teleop_speed_limits = np.asarray(planner.speed_limits, dtype=float).copy()
        self.home_speed_limits = self.teleop_speed_limits.copy()
        rotary = np.asarray([name != "lift_joint" for name in JOINT_NAMES])
        self.home_speed_limits[rotary] = np.minimum(self.home_speed_limits[rotary], max_home_joint_speed)
        self.arm_mask = np.asarray([name.startswith("arm_") for name in JOINT_NAMES])
        self.gripper_mask = np.asarray([name.startswith("gripper_") for name in JOINT_NAMES])
        if not np.isfinite(teleop_ramp_s) or teleop_ramp_s < 0:
            raise ValueError("Invalid teleop ramp duration")
        if max_arm_joint_lag is not None and (not np.isfinite(max_arm_joint_lag) or max_arm_joint_lag <= 0):
            raise ValueError("Invalid arm tracking limit")
        self.teleop_ramp_s = float(teleop_ramp_s)
        self.teleop_started_at = None
        self.teleop_ramp_scale = 1.
        self.required_bridge_limits = dict(required_bridge_limits or {})
        if type(command_arm_speed_limit) is not bool:
            raise ValueError("command_arm_speed_limit must be a bool")
        self.command_arm_speed_limit = command_arm_speed_limit
        self._pending_teleop_speed = None
        self._pending_gripper_speed = None
        self.bounds = np.asarray([JOINT_LIMITS[name] for name in JOINT_NAMES], dtype=float)
        self.lag_limits = np.full(len(JOINT_NAMES), max_joint_lag)
        self.lag_limits[list(JOINT_NAMES).index("lift_joint")] = max_lift_lag
        self.home_lag_limits = self.lag_limits.copy()
        self.teleop_lag_limits = self.lag_limits.copy()
        if max_arm_joint_lag is not None:
            self.teleop_lag_limits[self.arm_mask] = max_arm_joint_lag
        self.lag_limits = self.teleop_lag_limits.copy()
        self.initialized = False
        self.generation = None
        self.bridge_id = None
        self.session_id = None
        self.sequence = 0
        self.armed = False
        self.arm_started = None
        self.enabled_before = False
        self.control_has_started = False
        self.latched_reason = None
        self.last_feedback = None
        self.last_sent_q = None
        self.home_wait_started = None
        self.home_pending = False
        self._home_arm_deferred = False
        self._joint_preview = None
        self._pose_request = None
        self._manual_motion = False
        self._grip_reader = ControllerMapper(timeout_s=planner.mapper.timeout_s,
                                             grip_threshold=planner.mapper.grip_threshold,
                                             R_BQ=planner.mapper.R_BQ,
                                             accept_emulated=planner.mapper.accept_emulated)
        self._grip_engaged = False
        self._grip_start_requested = False
        self._grip_home_active = False
        self._grip_needs_release = False
        self._grip_has_started = False
        self._grip_rearm_reason = None
        self._grip_rearm_after = None
        self._grip_last_now = None
        self._grip_frame = None
        self._grip_intent = None
        self.tracking_pause_s = _tracking_pause(tracking_pause_s)
        self._tracking_pause_started = None
        self._tracking_pause_seq = -1
        self._grip_mode_barrier_pending = False
        self.home_tolerance = np.full(len(JOINT_NAMES), np.deg2rad(.5))
        self.home_tolerance[list(JOINT_NAMES).index("lift_joint")] = .001
        # The head settles ~0.7 deg off its target, so the 0.5 deg arm tolerance would keep a
        # return waiting until the settling timeout. It is a low-force pointing axis whose
        # exact angle does not gate anything, so it gets its own tolerance.
        for _head in ("head_joint1", "head_joint2"):
            self.home_tolerance[list(JOINT_NAMES).index(_head)] = np.deg2rad(2.)
        self.ack_window_s = .04
        self.tracking_soft_fraction = .8
        self._reset_command_window()

    def _set_q(self, env, q):
        env.forward(q=q, joint_names=self.planner.joint_names)
        for hand, index in self.planner.gripper_indices.items():
            set_sg2_gripper_qpos(env, hand, q[index])

    def preview_joint_adjustment(self, joint_name, degrees, feedback):
        """Preview one absolute arm angle with existing FK/guard; never send.

        Execution rechecks the path from fresh feedback and uses HOME's 10 deg/s
        coordinated runner. The captured session home is never overwritten.
        """
        self._joint_preview = None
        if (not self.initialized or not feedback.get("ready") or self.armed
                or self._grip_engaged or self.home_pending or self.planner.homing):
            raise ValueError("수동 조정은 조작·복귀를 멈추고 정상 로봇 피드백 상태에서 미리보기 하세요.")
        if joint_name not in self.planner.joint_names or not joint_name.startswith("arm_"):
            raise ValueError("팔 관절을 선택하세요.")
        if isinstance(degrees, (bool, np.bool_)) or not isinstance(degrees, (int, float, np.integer, np.floating)) or not np.isfinite(degrees):
            raise ValueError("관절 목표 각도는 유한한 숫자여야 합니다.")
        start = np.asarray(feedback["q"], dtype=float).copy()
        if start.shape != self.planner.q_home.shape or not np.isfinite(start).all():
            raise ValueError("invalid manual feedback")
        index = self.planner.joint_names.index(joint_name)
        target = start.copy()
        target[index] = np.deg2rad(float(degrees))
        low, high = JOINT_LIMITS[joint_name]
        if not low <= target[index] <= high:
            raise ValueError("%s 허용 범위: %.2f~%.2f°" % (joint_name, np.rad2deg(low), np.rad2deg(high)))
        self._set_q(self.planner.guard.env_check, start)
        plan = plan_home_path(self.planner.guard, self.planner.joint_names, start, target,
                              self.home_speed_limits, strategies=(("manual_joint", ()),))
        if not plan["ok"]:
            raise ValueError(plan["reason"])
        self._joint_preview = {"start": start, "target": target, "joint": joint_name,
                               "degrees": float(degrees), "bridge_id": feedback.get("bridge_id"),
                               "generation": feedback.get("generation"), "dry_run": self.dry_run}
        self._set_q(self.planner.env_ik, target)
        return {"joint": joint_name, "degrees": float(degrees), "duration_s": plan["duration_s"],
                "q_target": target.copy(), "plan": home_plan_summary(plan)}

    def request_pose_target(self, target):
        """Queue a full-pose target for the next "home" action (controller buttons).

        Uses the same path as manual adjustment: the straight joint path is
        guard-checked before anything is sent, then HOME's 10 deg/s runner moves
        there. A blocked path sends nothing. q_home (Initial pose) is untouched.
        """
        target = np.asarray(target, dtype=float).copy()
        if target.shape != (len(JOINT_NAMES),) or not np.isfinite(target).all():
            raise ValueError("invalid pose target")
        if np.any(target < self.bounds[:, 0]) or np.any(target > self.bounds[:, 1]):
            raise ValueError("pose target outside verified joint limits")
        self._pose_request = target

    def _sync_command(self, q):
        self._set_q(self.planner.env, q)
        self.planner.reset()

    @property
    def teleop_speed_deg_s(self):
        return float(np.rad2deg(np.max(self.teleop_speed_limits[self.arm_mask])))

    def set_position_gains(self, lateral, vertical, now=None, *, forward=None):
        """Apply XYZ gains after an idle tick with both grips released.

        Omitted forward preserves X for existing callers. Z may be signed/zero.
        Preserve orientation, speed limits and the saved robot home pose.
        The next grip anchors the user's comfortable pose to the current EEF.
        """
        values = (lateral, vertical) if forward is None else (forward, lateral, vertical)
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating))
               for v in values):
            raise ValueError("위치 배율은 숫자로 입력하세요. X/Y: 0.05~3.0, Z: -3.0~3.0")
        if (any(not np.isfinite(v) for v in values) or not .05 <= lateral <= 3.
                or not -3. <= vertical <= 3. or (forward is not None and not .05 <= forward <= 3.)):
            raise ValueError("위치 배율 범위: X/Y 0.05~3.0, Z -3.0~3.0")
        self._require_released_idle(now, "배율을")
        scales = self.planner.mapper.position_scales
        for gains in scales.values():
            if forward is not None:
                gains[0] = float(forward)
            gains[1:] = [float(lateral), float(vertical)]
        self.planner.mapper.set_position_scales(scales)
        return {hand: gains.tolist() for hand, gains in scales.items()}

    def _require_released_idle(self, now, subject):
        """Allow input-policy changes only between clutches, after a recent both-released sample."""
        now = time.monotonic() if now is None else float(now)
        if self._grip_engaged or self.armed or self._grip_home_active or self.home_pending or self.planner.homing:
            raise ValueError("조작·복귀가 끝난 뒤 양손 grip을 놓고 %s 적용하세요." % subject)
        intent = self._grip_intent
        if (not np.isfinite(now) or not intent or not intent["valid"] or not intent["both_released"]
                or self._grip_last_now is None or now < self._grip_last_now
                or not 0 <= now - intent["effective_sample_at"] <= self.planner.mapper.timeout_s):
            raise ValueError("양손 컨트롤러 추적이 유효하고 두 grip을 놓은 상태에서 적용하세요.")

    @property
    def tracking_loss_mode(self):
        return {"tracking_pause_s": self.tracking_pause_s,
                "accept_emulated": self.planner.mapper.accept_emulated}

    def set_tracking_loss_mode(self, tracking_pause_s, accept_emulated, now=None):
        """Test switch for tracking loss: pause length and emulated-pose use.

        Applies to SIM and REAL/DRY RUN alike, only between clutches. Both the
        planner mapper and the grip reader switch together so a start/stop
        gesture and its targets read the same poses.
        """
        pause = _tracking_pause(tracking_pause_s)
        if type(accept_emulated) is not bool:
            raise ValueError("accept_emulated must be a bool")
        self._require_released_idle(now, "추적 손실 모드를")
        self.tracking_pause_s = pause
        self.planner.mapper.accept_emulated = accept_emulated
        self._grip_reader.accept_emulated = accept_emulated
        return self.tracking_loss_mode

    @property
    def requested_teleop_speed_deg_s(self):
        return (self.teleop_speed_deg_s if self._pending_teleop_speed is None
                else float(np.rad2deg(self._pending_teleop_speed)))

    def set_teleop_speed(self, degrees_per_s):
        """Change only the arm cap. Decreases drain in-flight travel first.

        Called by the control loop, never from a socket/Qt background thread.
        A held/retried pose keeps its old cap until acknowledgment, so lowering
        the cap cannot strand an unacknowledged pose beyond the new rate budget.
        """
        if isinstance(degrees_per_s, (bool, np.bool_)) or not isinstance(degrees_per_s, (int, float, np.integer, np.floating)):
            raise ValueError("팔 속도는 10~120°/s의 숫자로 입력하세요.")
        try:
            speed = float(np.deg2rad(float(degrees_per_s)))
        except (ValueError, OverflowError) as exc:
            raise ValueError("팔 속도는 10~120°/s 범위여야 합니다.") from exc
        if not np.isfinite(speed) or not FIXED_QUEST_MIN_ARM_SPEED <= speed <= FIXED_QUEST_MAX_ARM_SPEED:
            raise ValueError("팔 속도는 10~120°/s 범위여야 합니다.")
        if np.isclose(degrees_per_s, self.requested_teleop_speed_deg_s, rtol=0, atol=1e-9):
            return False
        self._pending_teleop_speed = speed
        if (self.dry_run or not self.armed or self.home_pending or self.planner.homing
                or speed >= np.max(self.teleop_speed_limits[self.arm_mask])):
            self._apply_pending_teleop_speed(force=True)
        else:
            self._apply_pending_teleop_speed()
        return True

    def _apply_pending_teleop_speed(self, force=False):
        if self._pending_teleop_speed is None:
            return
        if force or (self._ack_travel is not None
                     and np.allclose(self._sent_travel, self._ack_travel, rtol=0, atol=1e-12)):
            self.teleop_speed_limits[self.arm_mask] = self._pending_teleop_speed
            self._pending_teleop_speed = None

    @property
    def gripper_speed_deg_s(self):
        return float(np.rad2deg(np.max(self.teleop_speed_limits[self.gripper_mask])))

    @property
    def requested_gripper_speed_deg_s(self):
        return (self.gripper_speed_deg_s if self._pending_gripper_speed is None
                else float(np.rad2deg(self._pending_gripper_speed)))

    def gripper_bridge_cap(self, feedback=None):
        if self.dry_run:
            return FIXED_QUEST_MAX_GRIPPER_SPEED
        limits = ((feedback or self.last_feedback or {}).get("bridge") or {}).get("limits", {})
        key = ("teleop_gripper_joint_speed_rad_s" if limits.get("command_gripper_speed_limit") is True
               else "teleop_joint_speed_rad_s")
        cap = limits.get(key, np.deg2rad(60.))
        if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not np.isfinite(cap) or cap <= 0:
            return 0.
        return min(float(cap), FIXED_QUEST_MAX_GRIPPER_SPEED,
                   FIXED_QUEST_MAX_GRIPPER_SPEED if key.startswith("teleop_gripper") else np.deg2rad(60.))

    def set_gripper_speed(self, degrees_per_s):
        """Change only the gripper cap. Decreases drain in-flight travel first.

        Arms, head, lift and the HOME return keep their own limits: HOME still
        moves the grippers at max_home_joint_speed.  A pending decrease holds the
        command exactly like an arm decrease, so the slower cap cannot strand an
        unacknowledged gripper pose.  It is tracked apart from the arm request so
        applying one never forces the other past its acknowledgment wait.
        """
        if isinstance(degrees_per_s, (bool, np.bool_)) or not isinstance(degrees_per_s, (int, float, np.integer, np.floating)):
            raise ValueError("그리퍼 속도는 10~120°/s의 숫자로 입력하세요.")
        try:
            speed = float(np.deg2rad(float(degrees_per_s)))
        except (ValueError, OverflowError) as exc:
            raise ValueError("그리퍼 속도는 10~120°/s 범위여야 합니다.") from exc
        if not np.isfinite(speed) or not FIXED_QUEST_MIN_GRIPPER_SPEED <= speed <= FIXED_QUEST_MAX_GRIPPER_SPEED:
            raise ValueError("그리퍼 속도는 10~120°/s 범위여야 합니다.")
        if speed > self.gripper_bridge_cap() + 1e-12:
            raise ValueError("현재 bridge 그리퍼 상한은 %.0f°/s입니다. 로봇 PC protocol/inbound 업데이트가 필요합니다." %
                             np.rad2deg(self.gripper_bridge_cap()))
        if np.isclose(degrees_per_s, self.requested_gripper_speed_deg_s, rtol=0, atol=1e-9):
            return False
        self._pending_gripper_speed = speed
        if (self.dry_run or not self.armed or self.home_pending or self.planner.homing
                or speed >= np.max(self.teleop_speed_limits[self.gripper_mask])):
            self._apply_pending_gripper_speed(force=True)
        else:
            self._apply_pending_gripper_speed()
        return True

    def _apply_pending_gripper_speed(self, force=False):
        if self._pending_gripper_speed is None:
            return
        if force or (self._ack_travel is not None
                     and np.allclose(self._sent_travel, self._ack_travel, rtol=0, atol=1e-12)):
            self.teleop_speed_limits[self.gripper_mask] = self._pending_gripper_speed
            self._pending_gripper_speed = None

    def _set_motion_speed(self, home, now=None):
        self.planner.speed_limits = (self.home_speed_limits if home else self.teleop_speed_limits).copy()
        self.lag_limits = (self.home_lag_limits if home else self.teleop_lag_limits).copy()
        self.teleop_ramp_scale = 1.
        if home:
            self.teleop_started_at = None
        elif now is not None and self.teleop_ramp_s > 0:
            if self.teleop_started_at is None:
                self.teleop_started_at = now
            self.teleop_ramp_scale = float(np.clip((now - self.teleop_started_at) / self.teleop_ramp_s, 0., 1.))
            self.planner.speed_limits[self.arm_mask] *= self.teleop_ramp_scale

    def _reset_command_window(self):
        self._sent_records = {}
        self._sent_travel = np.zeros(len(JOINT_NAMES))
        self._last_command_q = None
        self._last_ack_seq = -1
        self._ack_travel = None

    def _observe_command_ack(self, feedback):
        ack = feedback.get("last_command_seq", -1)
        if type(ack) is not int or ack < -1 or ack < self._last_ack_seq:
            raise ValueError("robot command acknowledgment regressed")
        if ack == self._last_ack_seq:
            return
        if ack not in self._sent_records:
            raise ValueError("unknown robot command acknowledgment")
        record = self._sent_records[ack]
        if record["mode"] == "command":
            self._ack_travel = record["travel"].copy()
        self._last_ack_seq = ack
        self._sent_records = {seq: item for seq, item in self._sent_records.items() if seq >= ack}

    def _pace_candidate(self, previous, candidate, measured, home=False):
        if self._pending_teleop_speed is not None or self._pending_gripper_speed is not None:
            return previous.copy(), {"tracking_paced": False, "ack_paced": True,
                                     "command_scale": 0., "speed_change_waiting_ack": True}
        delta = candidate - previous
        gap = previous - measured
        # If measured motion has already put a held command outside the soft
        # envelope, permit inward progress but never increase that discrepancy.
        soft = np.maximum(self.lag_limits * self.tracking_soft_fraction, np.abs(gap))
        tracking = np.ones(len(delta))
        for i, (change, distance, limit) in enumerate(zip(delta, gap, soft)):
            if change > 0:
                tracking[i] = (limit - distance) / change
            elif change < 0:
                tracking[i] = (limit + distance) / -change
        tracking = np.clip(tracking, 0., 1.)
        ack = np.zeros(len(delta))
        if self._ack_travel is not None:
            remaining = np.maximum(0., self.planner.speed_limits * self.ack_window_s
                                   - (self._sent_travel - self._ack_travel))
            moving = np.abs(delta) > 1e-15
            ack[:] = 1.
            ack[moving] = np.minimum(1., remaining[moving] / np.abs(delta[moving]))
        # HOME must stay on its checked segment. During teleop, a slow finger
        # does not scale both arms; every final mixed candidate is still guarded.
        groups = ([np.arange(len(delta))] if home or self.home_pending or self.planner.homing else
                  [np.flatnonzero(~self.gripper_mask)] + [np.array([i]) for i in np.flatnonzero(self.gripper_mask)])
        scales = np.ones(len(delta))
        for indices in groups:
            scales[indices] = min(float(np.min(tracking[indices])), float(np.min(ack[indices])))
        return previous + scales * delta, {
            "tracking_paced": bool(np.any(tracking < 1. - 1e-9)),
            "ack_paced": bool(np.any(ack < 1. - 1e-9)),
            "command_scale": float(np.min(scales[~self.gripper_mask])),
            "gripper_command_scales": scales[self.gripper_mask].tolist(),
        }

    def _send(self, mode, q, feedback, motion="teleop"):
        if self.dry_run or self.session_id is None:
            return False
        packet = {
            "type": "sg2_command", "version": 1, "session_id": self.session_id,
            "seq": self.sequence, "bridge_id": self.bridge_id,
            "feedback_time_s": feedback["state"]["source_time_s"],
            "valid_for_s": self.command_ttl_s, "mode": mode, "motion": motion,
            "joints": get_qmsg_from_qpos(q, self.q_msg_layout),
        }
        if self.command_arm_speed_limit and mode == "command" and motion == "teleop":
            packet["arm_speed_rad_s"] = max(1e-12, float(np.max(self.planner.speed_limits[self.arm_mask])))
        limits = (feedback.get("bridge") or {}).get("limits", {})
        if mode == "command" and motion == "teleop" and limits.get("command_gripper_speed_limit") is True:
            packet["gripper_speed_rad_s"] = float(np.max(self.planner.speed_limits[self.gripper_mask]))
        self.sequence += 1
        sent = self.publisher.send(packet)
        if sent:
            self.last_sent_q = np.asarray(q).copy()
            if mode == "command":
                if self._last_command_q is not None:
                    self._sent_travel += np.abs(self.last_sent_q - self._last_command_q)
                self._last_command_q = self.last_sent_q.copy()
            if mode in ("arm", "command"):
                self._sent_records[packet["seq"]] = {"mode": mode, "travel": self._sent_travel.copy()}
                if len(self._sent_records) > 256:
                    del self._sent_records[next(iter(self._sent_records))]
        return bool(sent)

    def stop(self):
        """Request lease release; return whether local stop enqueue succeeded."""
        sent = True
        try:
            if self.session_id and self.last_feedback is not None:
                q = self.last_sent_q if self.last_sent_q is not None else self.planner.q_command
                sent = self._send("stop", q, self.last_feedback)
        finally:
            self._tracking_pause_started = None
            self._tracking_pause_seq = -1
            self.session_id, self.armed, self.arm_started = None, False, None
            self.planner.mapper.require_release()
            self.planner.homing = False
            self.home_wait_started = None
            self.home_pending = False
            self._home_arm_deferred = False
            self.teleop_started_at = None
            self.planner._clear_ik_recovery()
            self._apply_pending_teleop_speed(force=True)
            self._apply_pending_gripper_speed(force=True)
            self._set_motion_speed(home=False)
            self._grip_engaged = False
            self._grip_start_requested = False
            self._grip_home_active = False
            self._reset_command_window()
        return sent

    def set_dry_run(self, dry_run, publisher_factory):
        """Switch mode after requesting lease release and closing its publisher.

        ``publisher_factory(enabled=...)`` owns transport construction. Return
        False for the current mode, otherwise True. Failure disables output and
        raises; failed closes are retained for retry by the fallback's close().
        Every change requires a subsequent control-OFF tick;
        the captured Initial pose, feedback generation and speed limits survive.
        Callers must use ``self.publisher`` for subsequent cleanup, even on error.
        """
        if type(dry_run) is not bool:
            raise TypeError("dry_run must be a bool")
        if dry_run == self.dry_run:
            return False
        if not callable(publisher_factory):
            raise TypeError("publisher_factory must be callable")

        old_publisher = self.publisher
        fallback = _DisabledPublisher()
        failure = None
        try:
            if not self.stop():
                failure = RuntimeError("previous robot stop could not be queued")
        except Exception as exc:
            failure = exc
        finally:
            try:
                old_publisher.close()
            except Exception as exc:
                if isinstance(old_publisher, _DisabledPublisher):
                    fallback._pending_publishers.extend(old_publisher._pending_publishers)
                else:
                    fallback._pending_publishers.append(old_publisher)
                if failure is None:
                    failure = exc
            self.publisher, self.dry_run = fallback, True
            self.sequence, self.last_sent_q = 0, None
            self.enabled_before = False
            self.control_has_started = False
            self._grip_needs_release = True
            self._grip_rearm_reason = "run mode changed"
            self._grip_rearm_after = self._grip_last_now
            self._grip_mode_barrier_pending = True
            self.latched_reason = "run mode changed; switch control OFF before resuming"
            self.planner.reset()

        if failure is not None:
            self.latched_reason = "run mode change failed; control OFF: " + str(failure)
            raise RuntimeError(self.latched_reason) from failure

        candidate = None
        try:
            candidate = publisher_factory(enabled=not dry_run)
            if bool(candidate.enabled) != (not dry_run):
                raise ValueError("publisher enabled state does not match requested run mode")
        except Exception as exc:
            if candidate is not None:
                try:
                    candidate.close()
                except Exception as cleanup_error:
                    fallback._pending_publishers.append(candidate)
                    self.latched_reason = "run mode change failed; control OFF: %s; cleanup: %s" % (exc, cleanup_error)
                    raise RuntimeError(self.latched_reason) from cleanup_error
            self.latched_reason = "run mode change failed; control OFF: " + str(exc)
            raise RuntimeError(self.latched_reason) from exc
        self.publisher, self.dry_run = candidate, dry_run
        return True

    def _result(self, result, feedback, q, transmitted=False):
        if q is not None and (q.shape != (len(JOINT_NAMES),) or not np.all(np.isfinite(q))):
            q = None
        result = {"tracking_paced": False, "ack_paced": False, "command_scale": 1., **result}
        return dict(result, dry_run=self.dry_run, robot_ready=bool(feedback.get("ready")) and q is not None,
                    robot_q=None if q is None else q.copy(), feedback_age_s=feedback.get("age_s"),
                    home_error=result["home_error"] if self.dry_run or q is None else q - self.planner.q_home,
                    bridge_id=self.bridge_id, session_id=self.session_id, armed=self.armed,
                    transmitted=bool(transmitted), needs_disable=self.latched_reason is not None,
                    control_has_started=self.control_has_started, home_pending=self.home_pending,
                    teleop_ramp_scale=self.teleop_ramp_scale,
                    teleop_speed_deg_s=self.teleop_speed_deg_s,
                    requested_teleop_speed_deg_s=self.requested_teleop_speed_deg_s,
                    speed_change_pending=self._pending_teleop_speed is not None,
                    gripper_speed_deg_s=self.gripper_speed_deg_s,
                    requested_gripper_speed_deg_s=self.requested_gripper_speed_deg_s,
                    gripper_speed_change_pending=self._pending_gripper_speed is not None,
                    tracking_loss_mode=self.tracking_loss_mode)

    def _hold(self, frame, feedback, now, reason, latch, state="HOLD"):
        reason = reason.replace("SIM control", "control")
        recovery = self.planner.ik_recovery_info(now)
        self.stop()
        if latch:
            self.latched_reason = reason
        result = self.planner.tick(frame, enabled=False, now=now)
        if recovery["status"] == "waiting":
            result["ik_recovery"] = dict(recovery, status="cancelled", reason=reason)
        result.update(state=state, reason=reason, active=[], targets={})
        q = feedback.get("q")
        return self._result(result, feedback, None if q is None else np.asarray(q))

    def tick(self, frame, feedback, enabled=False, now=None, action=None, pause_motion=False, q_target=None):
        """action="policy": drive q_target (JOINT_NAMES order, already safety-clipped
        by the caller -- see policy_inference.clip_to_joint_limits) through the same
        arming/lease/pacing/guard pipeline teleop already goes through, at the
        always-checked "home" collision profile (see guard_motion below) but at
        teleop speed/lag limits -- a trained policy is expected to move at
        operating speed, not HOME's slow guarded-return crawl. No VR mapper/IK
        runs; `frame` may be None. Pass action="policy" with a fresh q_target on
        every tick for the whole session (unlike "home", this is not a one-shot
        request that then self-sustains via self.planner.homing).
        """
        now = time.monotonic() if now is None else float(now)
        if not np.isfinite(now):
            raise ValueError("now must be finite")
        if action not in (None, "home", "cancel_home", "joint_adjust", "policy"):
            raise ValueError("Unknown real control action")
        is_policy = action == "policy"
        if is_policy:
            q_target = None if q_target is None else np.asarray(q_target, dtype=float).copy()
            if (q_target is None or q_target.shape != (len(JOINT_NAMES),)
                    or not np.isfinite(q_target).all()):
                raise ValueError("policy action requires a finite q_target of shape (%d,)" % len(JOINT_NAMES))
        elif q_target is not None:
            raise ValueError("q_target is only used with action='policy'")
        recovery_before = self.planner.ik_recovery_info(now)
        feedback = feedback or {"ready": False, "reason": "waiting for robot state"}
        rising = bool(enabled) and not self.enabled_before
        self.enabled_before = bool(enabled)
        if not enabled:
            self.latched_reason = None
            self.control_has_started = False
        if not feedback.get("ready"):
            return self._hold(frame, feedback, now, feedback.get("reason", "robot feedback unavailable"),
                              latch=bool(enabled), state="WAIT_ROBOT")
        q = np.asarray(feedback.get("q"), dtype=float)
        if (q.shape != (len(JOINT_NAMES),) or not np.all(np.isfinite(q))
                or np.any(q < self.bounds[:, 0]) or np.any(q > self.bounds[:, 1])):
            return self._hold(frame, feedback, now, "measured joints outside verified real limits", bool(enabled), "FAULT")
        self._set_q(self.env_measured, q)
        changed = self.initialized and (feedback["generation"] != self.generation
                                       or feedback["bridge_id"] != self.bridge_id)
        self.last_feedback = feedback
        if not self.initialized or changed:
            self.stop()
            self._sync_command(q)
            if not self.initialized:
                # Initial pose is always the fixed worker_bringup pose (BRINGUP_POSE),
                # not whatever pose the robot happened to be in when this session
                # connected -- head_joint2 (yaw) is not part of BRINGUP_POSE (see
                # vr_buttons.py), so it keeps whatever this run's first measured
                # value was. A worker/bridge restart resynchronizes commands but
                # keeps this target, so a restart in a wound-up pose never becomes
                # the Initial pose.
                q_home = q.copy()
                for name, value in BRINGUP_POSE.items():
                    q_home[JOINT_NAMES.index(name)] = value
                self.planner.q_home = q_home
            self.initialized = True
            self.generation, self.bridge_id = feedback["generation"], feedback["bridge_id"]
            if enabled:
                return self._hold(frame, feedback, now, "robot state synchronized; switch control OFF then ON", True)
        if not enabled:
            self.stop()
            self._sync_command(q)
            result = self.planner.tick(frame, enabled=False, now=now)
            if recovery_before["status"] == "waiting":
                result["ik_recovery"] = dict(recovery_before, status="cancelled", reason="control disabled")
            if action == "home":
                result["reason"] = "enable control before returning to session start"
            return self._result(result, feedback, q)
        returning = action in ("home", "joint_adjust") or self.home_pending or self.planner.homing
        if not returning and self.gripper_speed_deg_s > np.rad2deg(self.gripper_bridge_cap(feedback)) + 1e-9:
            return self._hold(frame, feedback, now, "worker gripper speed profile mismatch; lower speed or update worker", True, "FAULT")
        if self.latched_reason:
            return self._hold(frame, feedback, now, self.latched_reason, True)
        if rising:
            self._sync_command(q)
        if action == "cancel_home":
            return self._hold(frame, feedback, now, "return cancelled; control OFF", True)
        if action in ("home", "joint_adjust") and (self.home_pending or self.planner.homing):
            action = None  # already requested or returning along a checked path
        if not self.dry_run and self.required_bridge_limits:
            limits = (feedback.get("bridge") or {}).get("limits", {})
            if any(not isinstance(limits.get(key), (int, float))
                   or not np.isclose(limits[key], expected, rtol=0, atol=1e-9)
                   for key, expected in self.required_bridge_limits.items()):
                return self._hold(frame, feedback, now,
                    "worker speed profile mismatch; update inbound.py and fixed_quest_protocol.py, then restart inbound",
                    True, "FAULT")
        home_plan = None
        if action in ("home", "joint_adjust"):
            target = None
            if action == "joint_adjust":
                preview = self._joint_preview
                self._joint_preview = None
                if (preview is None or np.any(np.abs(q - preview["start"]) > self.home_tolerance)
                        or preview["bridge_id"] != feedback.get("bridge_id")
                        or preview["generation"] != feedback.get("generation") or preview["dry_run"] != self.dry_run):
                    return self._hold(frame, feedback, now, "manual preview expired; preview current pose again", True)
                target = q.copy()
                selected = self.planner.joint_names.index(preview["joint"])
                target[selected] = preview["target"][selected]
            else:
                self._joint_preview = None
                if self._pose_request is not None:
                    target, self._pose_request = self._pose_request, None
            if not self.dry_run and self.armed:
                # Checking the path can outlast the 200 ms command lease, so
                # release it first; the return re-arms at the measured pose.
                self.stop()
                self._sync_command(q)
            # The whole return is checked before arming or any HOME command. A
            # blocked path sends nothing; the next click checks again from here.
            self._manual_motion = target is not None
            home_plan = self.planner.plan_home(speed_limits=self.home_speed_limits, target=target)
            action = "home"
            if not home_plan["ok"]:
                held = self._hold(frame, feedback, now, home_plan["reason"], True, "FAULT")
                held["home_plan"] = home_plan_summary(home_plan)
                held["guard"] = next((a["guard"] for a in home_plan["attempts"] if "guard" in a), held["guard"])
                return held
            if not self.dry_run:
                self.home_pending = True
                # Feedback sampled before the check may be too old to echo, so
                # a new lease is requested on the next tick. An arming already
                # in progress keeps its own session and acknowledgment deadline.
                self._home_arm_deferred = self.session_id is None
        self._set_motion_speed(home=self.home_pending or self.planner.homing or action == "home", now=now)
        if self.armed and (feedback.get("owner_session_id") != self.session_id
                           or feedback.get("bridge_state") not in ("ACTIVE", "active")):
            return self._hold(frame, feedback, now, "robot command lease lost; control OFF", True, "FAULT")

        # A live run first arms at the measured pose, and waits for robot-side
        # acknowledgment before permitting any IK-driven command.
        if not self.dry_run and self.session_id is not None and not self.armed:
            self._sync_command(q)
            # An explicit Initial pose request needs measured robot feedback,
            # not controller tracking. Ordinary teleoperation still needs both.
            # A policy session has no VR frame at all -- arm in place, at the
            # current measured pose, exactly like an Initial pose request does.
            if is_policy:
                arming_result = self.planner.tick(
                    frame, enabled=True, now=now, action="policy", q_target=q,
                    posture_reference_qpos=self.env_measured.data.qpos.copy())
            else:
                arming_result = self.planner.tick(
                    frame, enabled=not self.home_pending, now=now,
                    posture_reference_qpos=self.env_measured.data.qpos.copy())
            if arming_result["state"] in ("HOLD", "FAULT"):
                return self._hold(frame, feedback, now, arming_result["reason"], True,
                                  arming_result["state"])
            if self.home_pending or is_policy:
                self.planner.guard.env_check.data.qpos[:] = self.env_measured.data.qpos
                self.planner.guard.env_check.data.qvel[:] = 0
                self.planner.guard.env_check.forward()
                arming_check = self.planner.check_command(q, q, motion="home")
                if not arming_check["accepted"]:
                    held = self._hold(frame, feedback, now,
                                      ("policy arming rejected: " if is_policy else "initial pose arming rejected: ")
                                      + arming_check["reason"], True, "FAULT")
                    held["guard"] = arming_check
                    return held
            if feedback.get("owner_session_id") == self.session_id and feedback.get("bridge_state") in ("ACTIVE", "active"):
                self.armed = True
            elif now - self.arm_started > .5:
                return self._hold(frame, feedback, now, "robot bridge did not acknowledge arming", True, "FAULT")
            else:
                sent = self._send("arm", q, feedback)
                arming_result.update(enabled=True, state="ARMING", reason=(
                    "initial pose requested; waiting for robot bridge" if self.home_pending
                    else ("waiting for robot bridge; keep both grips held" if self._grip_engaged
                          else "waiting for robot bridge; release both grips")))
                return self._result(arming_result, feedback, q, sent)

        if not self.dry_run and self.armed and np.any(np.abs(self.planner.q_command - q) > self.lag_limits):
            return self._hold(frame, feedback, now, "command/feedback tracking difference exceeds limit", True, "FAULT")
        if not self.dry_run and self.armed:
            try:
                self._observe_command_ack(feedback)
                self._apply_pending_teleop_speed()
                self._apply_pending_gripper_speed()
                self._set_motion_speed(home=self.home_pending or self.planner.homing or action == "home", now=now)
            except ValueError as exc:
                return self._hold(frame, feedback, now, str(exc), True, "FAULT")
        eef = {hand: get_sg2_wrist_control_T(self.env_measured, hand) for hand in self.planner.hand_indices}
        if not self.control_has_started and self.session_id is None:
            # While acquiring input, hold the latest actual pose without losing
            # the mapper's freshness history or the captured session-start pose.
            self._set_q(self.planner.env, q)
            self.planner.q_command = q.copy()
            self.planner.q_goal = q.copy()
        # Queue return until the bridge owns a lease at the measured pose. No
        # return target reaches the planner before that acknowledgment.
        waiting_home_arm = self.home_pending and not self.armed
        waiting_grip_arm = self._grip_start_requested and not self.dry_run and not self.armed
        waiting_policy_arm = is_policy and not self.dry_run and not self.armed
        home_action = "home" if self.home_pending and self.armed else action
        previous = self.planner.q_command.copy()
        if waiting_home_arm or waiting_grip_arm or waiting_policy_arm:
            result = self.planner.tick(frame, enabled=False, now=now, eef_poses=eef)
            result["enabled"] = True
        else:
            if home_action == "home":
                self.home_pending = False
            if self._grip_start_requested:
                initialized = self.planner.mapper.activate_held(frame, now=now)
                if initialized["state"] == "HOLD":
                    return self._hold(frame, feedback, now, initialized["reason"], True)
                self._grip_start_requested = False
            result = self.planner.tick(frame, enabled=True, now=now, action=home_action, q_target=q_target,
                                       eef_poses=eef, pause_motion=pause_motion,
                                       posture_reference_qpos=self.env_measured.data.qpos.copy())
        if result["state"] in ("HOLD", "FAULT"):
            if (result["state"] == "HOLD" and result["reason"] in _STARTUP_INPUT_REASONS
                    and not self.control_has_started and self.session_id is None):
                self.planner.mapper.require_release()
                result.update(state="WAIT_INPUT", active=[], targets={})
                return self._result(result, feedback, q)
            held = self._hold(frame, feedback, now, result["reason"], True, result["state"])
            # Preserve the attempted input mapping for offline diagnosis. These
            # are rejected targets, not commands that the robot executed.
            held["attempted_mapping_stages"] = result.get("mapping_stages", {})
            held["attempted_ik_errors"] = result.get("ik_errors", {})
            held["attempted_posture_ik"] = result.get("posture_ik", {})
            held["ik_recovery"] = result.get("ik_recovery", held["ik_recovery"])
            return held
        if self.dry_run and result["state"] in ("ACTIVE", "HOMING", "HOME", "POLICY"):
            self.control_has_started = True
        candidate = result["q_command"]
        # Capture the automatic-motion policy through the final HOME tick,
        # even after the planner has cleared its homing flag on arrival. An
        # autonomous policy is not teleoperation either: always the checked
        # "home" collision profile, regardless of teleop_collision_check.
        guard_motion = "home" if returning or is_policy or result["state"] in ("HOMING", "HOME") else "teleop"
        if np.any(candidate < self.bounds[:, 0]) or np.any(candidate > self.bounds[:, 1]):
            return self._hold(frame, feedback, now, "candidate outside verified real joint limits", True, "FAULT")
        if not self.dry_run and self.armed:
            paced, pacing = self._pace_candidate(previous, candidate, q, home=result["state"] in ("HOMING", "HOME"))
            result.update(pacing)
            if not np.array_equal(paced, candidate):
                self.planner.q_command = paced.copy()
                self._set_q(self.planner.env, paced)
                result.update(q_command=paced.copy(), q_measured=paced.copy(), eef_errors={
                    hand: wrist_error(get_sg2_wrist_control_T(self.planner.env, hand), target)
                    for hand, target in result["targets"].items()})
                self.planner.guard.env_check.data.qpos[:] = self.planner.env.data.qpos
                self.planner.guard.env_check.data.qvel[:] = 0
                self.planner.guard.env_check.forward()
                paced_check = self.planner.check_command(previous, paced, motion=guard_motion)
                if not paced_check["accepted"]:
                    held = self._hold(frame, feedback, now, "paced path rejected: " + paced_check["reason"], True, "FAULT")
                    held["guard"] = paced_check
                    return held
                result["guard"] = paced_check
            candidate = paced
            if result["state"] == "HOME" and (
                    self._ack_travel is None or not np.allclose(candidate, self.planner.return_target, rtol=0, atol=1e-9)):
                self.planner.homing = True
                result.update(state="HOMING", reason="returning to initial pose", homing=True)
        if result["state"] in ("ACTIVE", "IK_WAIT", "POLICY"):
            posture_check = self.planner.check_posture_command(candidate)
            result["posture_command_guard"] = posture_check
            if not posture_check["accepted"]:
                held = self._hold(frame, feedback, now,
                                  "posture command rejected: " + posture_check["reason"], True, "FAULT")
                held["attempted_mapping_stages"] = result.get("mapping_stages", {})
                held["attempted_ik_errors"] = result.get("ik_errors", {})
                held["attempted_posture_ik"] = result.get("posture_ik", {})
                held["posture_command_guard"] = posture_check
                return held
        if not self.dry_run and self.armed and np.any(np.abs(candidate - q) > self.lag_limits):
            return self._hold(frame, feedback, now, "candidate/feedback tracking difference exceeds limit", True, "FAULT")
        # Validate physical-to-command lag as well as the planner's own sweep.
        self.planner.guard.env_check.data.qpos[:] = self.env_measured.data.qpos
        self.planner.guard.env_check.data.qvel[:] = 0
        self.planner.guard.env_check.forward()
        check = self.planner.check_command(q, candidate, motion=guard_motion)
        if not check["accepted"]:
            held = self._hold(frame, feedback, now, "measured path rejected: " + check["reason"], True, "FAULT")
            held["guard"] = check
            return held
        if not self.dry_run and result["state"] == "HOME" and np.any(np.abs(q - self.planner.return_target) > self.home_tolerance):
            if self.home_wait_started is None:
                self.home_wait_started = now
            if now - self.home_wait_started > 3.:
                return self._hold(frame, feedback, now, "robot did not reach session start within settling time", True, "FAULT")
            self.planner.homing = True
            result.update(state="HOMING", reason="waiting for measured session-start pose", homing=True)
        sent = False
        if not self.dry_run:
            if waiting_home_arm and self._home_arm_deferred:
                self._home_arm_deferred = False
                result.update(state="ARMING", reason="initial pose path verified; arming robot bridge")
            elif not self.armed and (result["state"] == "READY" or waiting_home_arm or waiting_grip_arm
                                     or waiting_policy_arm):
                self._reset_command_window()
                self.session_id, self.sequence, self.arm_started = str(uuid.uuid4()), 0, now
                self.control_has_started = True
                sent = self._send("arm", q, feedback)
                result.update(state="ARMING", reason=(
                    "policy control requested; waiting for robot bridge" if is_policy
                    else "initial pose requested; waiting for robot bridge" if self.home_pending
                    else ("waiting for robot bridge; keep both grips held" if self._grip_engaged
                          else "waiting for robot bridge; release both grips")))
            elif self.armed and result["state"] in ("READY", "ACTIVE", "HOMING", "HOME", "TRACKING_PAUSED",
                                                     "IK_WAIT", "POLICY"):
                motion = "home" if result["state"] in ("HOMING", "HOME") else "teleop"
                sent = self._send("command", candidate, feedback, motion=motion)
                if not sent:
                    return self._hold(frame, feedback, now, "robot command could not be queued", True, "FAULT")
        if result["state"] == "HOME":
            self.stop()
            self.latched_reason = ("session-start preview reached; control OFF" if self.dry_run
                                   else "session-start pose reached; control OFF")
            result["reason"] = self.latched_reason
        result["manual_adjustment"] = self._manual_motion and result["state"] in ("ARMING", "HOMING", "HOME")
        if result["manual_adjustment"]:
            result["reason"] = ("manual joint target reached; release both grips before restarting" if result["state"] == "HOME"
                                else "manual joint adjustment at HOME speed")
        if home_plan is not None:
            result["home_plan"] = home_plan_summary(home_plan)
        return self._result(result, feedback, q, sent)

    def _grip_failure(self, result):
        self._grip_engaged = False
        self._grip_start_requested = False
        self._grip_home_active = False
        self._grip_needs_release = True
        self._grip_rearm_reason = result["reason"]
        self._grip_rearm_after = self._grip_last_now
        return self._grip_result(result)

    def _grip_result(self, result, state=None, reason=None):
        result = dict(result, grip_control=True, grip_rearm_required=self._grip_needs_release,
                      rearm_reason=self._grip_rearm_reason)
        if state is not None:
            result.update(state=state, reason=reason, active=[], targets={})
        for field in ("reason", "rearm_reason"):
            if result.get(field):
                result[field] = (result[field].replace("switch control OFF then ON", "release both grips, then hold both to retry")
                                 .replace("switch control OFF before resuming", "release both grips before restarting")
                                 .replace("control OFF", "release both grips before restarting"))
        return result

    def _released_button_during_tracking_loss(self, frame, now):
        """Position validity is independent of a fresh controller grip release.

        Only stopping uses these buttons. They can never authorize motion.
        Duplicates, old sessions and stale packets cannot alter the gesture.
        """
        try:
            previous = self._grip_frame
            if (not previous or frame["version"] != 1 or frame["visibility"] != "visible"
                    or (frame["session_id"], frame["origin_id"]) != (previous["session_id"], previous["origin_id"])
                    or type(frame["seq"]) is not int
                    or frame["seq"] <= max(previous["seq"], self._tracking_pause_seq)
                    or frame["sample_time_ms"] <= previous["sample_time_ms"]
                    or not 0 <= now - frame["received_at"] <= self.planner.mapper.timeout_s):
                return False
            for hand in ("left", "right"):
                grip = frame["hands"][hand]["grip"]
                if (isinstance(grip, (int, float)) and not isinstance(grip, bool)
                        and np.isfinite(grip) and 0 <= grip < self.planner.mapper.grip_threshold):
                    return True
        except (KeyError, TypeError, ValueError):
            pass
        return False

    def tick_grip(self, frame, feedback, now=None, action=None):
        """Both grips start/hold both arms; either release stops their lease.

        Initial held input can start directly. Fault recovery requires a fresh
        both-released sample then a newer both-held sample. Explicit Initial
        pose requests have priority and never depend on controller tracking.
        """
        now = time.monotonic() if now is None else float(now)
        if not np.isfinite(now):
            raise ValueError("now must be finite")
        if action not in (None, "home", "cancel_home", "joint_adjust"):
            raise ValueError("Unknown real control action")
        self._grip_last_now = now
        if self._grip_mode_barrier_pending:
            # Use the same injected clock as input processing. The first tick
            # after a mode switch is a conservative barrier against queued input.
            self._grip_rearm_after = now
            self._grip_mode_barrier_pending = False
        feedback = feedback or {"ready": False, "reason": "waiting for robot state"}

        if action == "cancel_home":
            result = self.tick(frame, feedback, enabled=True, now=now, action=action)
            return self._grip_failure(result)
        if action == "joint_adjust" and (self._grip_engaged or self.armed or self._grip_home_active):
            return self._grip_failure(self._hold(frame, feedback, now, "stop teleoperation before manual adjustment", True))
        if action in ("home", "joint_adjust") and not self._grip_home_active:
            # A deliberate return click can recover from an earlier teleop
            # latch, but still must synchronize fresh feedback and get an ACK.
            self.tick(frame, feedback, enabled=False, now=now)
            self._grip_needs_release = True
            self._grip_rearm_reason = "initial pose requested"
            self._grip_home_active = True
            result = self.tick(frame, feedback, enabled=True, now=now, action=action)
            if result["needs_disable"] or result["state"] == "HOME":
                return self._grip_failure(result)
            return self._grip_result(result)
        if self._grip_home_active:
            result = self.tick(frame, feedback, enabled=True, now=now)
            if result["needs_disable"] or result["state"] == "HOME":
                return self._grip_failure(result)
            return self._grip_result(result)

        intent = self._grip_reader.read_intent(frame, now=now)
        self._grip_intent = intent
        if intent["valid"]:
            if intent["fresh"]:
                self._grip_frame = copy.deepcopy(frame)
            # The planner may reset during ACK handling. Keep the full accepted
            # sample so duplicate packets cannot alter poses, yaw or buttons.
            frame = self._grip_frame
        if self._grip_engaged:
            pausing = self._tracking_pause_started is not None
            if pausing and now - self._tracking_pause_started >= self.tracking_pause_s - 1e-9:
                result = self._hold(frame, feedback, now, "controller tracking pause exceeded %.0f ms"
                                    % (1000 * self.tracking_pause_s), True)
                return self._grip_failure(result)
            if not intent["valid"]:
                if self.tracking_pause_s > 0 and self._released_button_during_tracking_loss(frame, now):
                    return self._grip_failure(self._hold(frame, feedback, now, "grip released during tracking loss", True))
                transient = intent["reason"] in ("tracking_lost_left", "tracking_lost_right",
                                                 "emulated_pose_left", "emulated_pose_right")
                if transient and self.tracking_pause_s > 0 and (self.dry_run or self.armed):
                    if not pausing:
                        self._tracking_pause_started = now
                    if isinstance(frame, dict) and type(frame.get("seq")) is int:
                        self._tracking_pause_seq = max(self._tracking_pause_seq, frame["seq"])
                    elapsed = now - self._tracking_pause_started
                    result = self.tick(frame, feedback, enabled=True, now=now, pause_motion=True)
                    result["tracking_pause_elapsed_s"] = elapsed
                    if result["needs_disable"] or result["state"] != "TRACKING_PAUSED":
                        return self._grip_failure(result)
                    return self._grip_result(result)
                result = self._hold(frame, feedback, now, intent["reason"], True)
                return self._grip_failure(result)
            if pausing:
                if not intent["fresh"] or frame["seq"] <= self._tracking_pause_seq:
                    result = self._hold(frame, feedback, now, "tracking recovery requires a new valid sample", True)
                    return self._grip_failure(result)
                self._tracking_pause_started = None
                self._tracking_pause_seq = -1
            if intent["both_held"]:
                result = self.tick(frame, feedback, enabled=True, now=now)
                if result["needs_disable"] or result["state"] in ("HOLD", "FAULT", "WAIT_ROBOT"):
                    return self._grip_failure(result)
                return self._grip_result(result)
            # A valid release stops both arms. A newer both-held edge may start
            # again even if the other hand stayed held throughout this pause.

        changed = self.initialized and feedback.get("ready") and (
            feedback.get("generation") != self.generation or feedback.get("bridge_id") != self.bridge_id)
        idle = self.tick(frame, feedback, enabled=False, now=now)
        if changed:
            self._grip_needs_release = True
            self._grip_rearm_reason = "robot feedback identity changed"
            self._grip_rearm_after = now
        if idle["state"] in ("WAIT_ROBOT", "FAULT"):
            if self._grip_has_started or idle["state"] == "FAULT":
                return self._grip_failure(idle)
            return self._grip_result(idle)
        if not intent["valid"]:
            if self._grip_has_started or intent["reason"] == "reference_changed":
                self._grip_needs_release = True
                self._grip_rearm_reason = intent["reason"]
                self._grip_rearm_after = now
            return self._grip_result(idle, "WAIT_INPUT", intent["reason"])
        if self._grip_needs_release:
            if (intent["fresh"] and intent["both_released"]
                    and (self._grip_rearm_after is None
                         or intent["effective_sample_at"] > self._grip_rearm_after)):
                self._grip_needs_release = False
                self._grip_rearm_reason = None
                self._grip_rearm_after = None
                return self._grip_result(idle, "WAIT_GRIP", "hold_both_grips")
            return self._grip_result(idle, "WAIT_RELEASE", "release_both_grips")
        if not (intent["fresh"] and intent["both_held"]):
            return self._grip_result(idle, "WAIT_GRIP", "hold_both_grips")
        self._grip_engaged = True
        self._joint_preview = None
        self._grip_start_requested = True
        self._grip_has_started = True
        result = self.tick(frame, feedback, enabled=True, now=now)
        if result["needs_disable"] or result["state"] in ("HOLD", "FAULT", "WAIT_ROBOT"):
            return self._grip_failure(result)
        return self._grip_result(result)
