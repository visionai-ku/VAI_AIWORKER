"""
Quest controller shortcuts for run_intent_teleop(on_tick=...).

    left  X          press       toggle recording (REC <-> STOP)
    left  Y          press       discard the running episode, else mark the last saved one as
                                 rejected (reversible; review_dataset.ipynb drops the marks).
                                 Measured Quest taps are 0.03-0.26 s, so only B still asks for
                                 a hold -- it is the one that moves the whole robot.
    right A          press       Initial pose: guarded return to the fixed worker_bringup pose.
                                 A single press, exactly like clicking the "Initial pose" button:
                                 the whole path is guard-checked before anything is sent and the
                                 return runs at the 10 deg/s HOME speed, so there is nothing for a
                                 hold-to-confirm to protect. Quest taps are often shorter than
                                 100 ms, so a hold would simply never fire.
    right B          hold 1 s    reset pose: both wrists up 5 cm + 5 cm apart -> fold like worker_shutdown
                                 (4 steps) -> worker_bringup initial pose (5 steps)

Initial pose and A/B keep head_joint1 at the bringup target HEAD_TILT (0.69 rad),
including every B reset stage; head_joint2 (yaw) keeps its measured value.

Poses go through SG2FixedQuestReal.request_pose_target(): each straight joint path is
guard-checked before anything is sent. The wrist lift/spread and shutdown1 alignment use
speed_scale x the 10 deg/s home speed (default 3 -> 30 deg/s).
Shutdown2 onward uses fold_speed_scale (default 10 -> 100 deg/s). Arm reset speeds
are independent of the teleop speed setting; head/grippers keep their existing caps.
The live bridge's HOME ceiling caps every step, so older 30/60 deg/s workers run the
full sequence at their supported speed and report that an update is needed. The on-screen
Initial pose / manual adjustment keep 10 deg/s. A blocked path sends nothing and the
sequence stops. Teleop input is ignored while the B sequence runs. Requires the Quest page
to send "buttons" (quest_client/client.js); older pages never trigger anything.
"""
import mujoco
import numpy as np

from utils import SG2_ARM_JOINT_NAMES, SG2_HAND_TYPES, SG2_WRIST_BODY_NAMES, SG2_WRIST_P_LOCAL
from package.worker.fixed_quest_protocol import JOINT_LIMITS, FIXED_QUEST_MAX_ARM_SPEED

# Step interruptions that are worth retrying from the current pose (loop stall, robot
# settling late, command lease expiry). Guard/collision refusals are never retried.
TRANSIENT_REASONS = ("control loop delay", "settling time", "lease lost", "command_expired",
                     "stale", "waiting for robot", "feedback")
MAX_STEP_RETRIES = 2
RETRY_DELAY_S = 0.3

# ffw_bringup/config/ffw_sg2_rev1_follower/shutdown_position.yaml (step1..step4, arms + grippers)
# Only the right arm's joint3 has the opposite sign.
_FOLD_LEFT = [
    [0.3, 0.0, 0.0, -1.8, 0.0, 0.0, 0.0, 0.0],
    [0.3, 0.0, 1.5, -1.8, 0.0, 0.0, 0.0, 0.0],
    [0.3, 0.0, 1.5, -0.45, 0.2, 0.0, 0.0, 0.0],
    [0.2, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0],
]
FOLD_STEPS = []
for _i, _step in enumerate(_FOLD_LEFT):
    _right = list(_step)
    _right[2] = -_right[2]
    FOLD_STEPS.append(("shutdown%d" % (_i + 1), {
        **{"arm_l_joint%d" % (j + 1): v for j, v in enumerate(_step[:7])}, "gripper_l_joint1": _step[7],
        **{"arm_r_joint%d" % (j + 1): v for j, v in enumerate(_right[:7])}, "gripper_r_joint1": _right[7],
    }))

# ffw_bringup/config/ffw_sg2_rev1_follower/ffw_sg2_follower_initial_positions.yaml
# start -> step1 -> step2 -> step3 -> step4 (both arms identical); lift goes home [0] and the
# head keeps its bringup pitch throughout the reset sequence.
# head_joint1 is the pitch axis (world +y), where positive looks down: the bringup target
# is 0.69 rad (~39.5 deg), within its limit (-0.2317, 0.6951). head_joint2 (yaw) is still
# left alone because it settles ~0.7 deg off its target. SG2FixedQuestReal widens the HOME
# tolerance for both head joints so that settling error cannot stall a return.
HEAD_TILT = .69
_BRINGUP_ARM = [
    ("start", [0.2, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0]),
    ("step1", [1.0, 0.0, 0.0, -0.3, 0.0, -1.4, 0.0, 0.0]),
    ("step2", [1.32, 0.0, 0.0, -1.5, 0.0, -1.4, 0.0, 0.0]),
    ("step3", [1.57, 0.0, 0.0, -2.8, 0.0, 0.0, 0.0, 0.0]),
    ("step4", [0.0, 0.0, 0.0, -1.5708, 0.0, 0.0, 0.0, 0.0]),
]
BRINGUP_STEPS = []
for _i, (_name, _step) in enumerate(_BRINGUP_ARM):
    _pose = {}
    for _side in ("l", "r"):
        _pose.update({"arm_%s_joint%d" % (_side, j + 1): v for j, v in enumerate(_step[:7])})
        _pose["gripper_%s_joint1" % _side] = _step[7]
    if _i == 0:
        _pose["lift_joint"] = 0.0
        _pose["head_joint1"] = HEAD_TILT
    BRINGUP_STEPS.append(("bringup " + _name, _pose))
BRINGUP_POSE = dict(BRINGUP_STEPS[-1][1], lift_joint=0.0, head_joint1=HEAD_TILT)


def lifted_arm_pose(env, joint_names, q_now, dz=0.05, dy=0.0, iters=200, pos_tol=2e-3, rot_tol=np.deg2rad(1.)):
    """Joint pose with both wrist control points dz higher (world z), spread dy apart per hand
    (left +y, right -y), with the same orientation.

    Damped least squares on a private MjData copy of env's model; never touches env.
    Returns (q, ok, max_position_error_m).
    """
    model = env.model
    data = mujoco.MjData(model)
    data.qpos[:] = env.data.qpos
    adr = {name: int(model.jnt_qposadr[model.joint(name).id]) for name in joint_names}
    for name, value in zip(joint_names, q_now):
        data.qpos[adr[name]] = value

    def wrist(hand):
        body = model.body(SG2_WRIST_BODY_NAMES[hand]).id
        R = data.xmat[body].reshape(3, 3).copy()
        return body, data.xpos[body] + R @ SG2_WRIST_P_LOCAL[hand], R

    mujoco.mj_kinematics(model, data)
    targets = {}
    for hand in SG2_HAND_TYPES:
        _, p, R = wrist(hand)
        side = 1.0 if hand == "left" else -1.0
        targets[hand] = (p + np.array([0.0, side * dy, dz]), R)
    err, done = np.inf, False
    for _ in range(iters):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        err, done = 0.0, True
        for hand in SG2_HAND_TYPES:
            body, p, R = wrist(hand)
            p_t, R_t = targets[hand]
            e_p = p_t - p
            e_r = 0.5 * sum(np.cross(R[:, i], R_t[:, i]) for i in range(3))
            err = max(err, float(np.linalg.norm(e_p)))
            if np.linalg.norm(e_p) > pos_tol or np.linalg.norm(e_r) > rot_tol:
                done = False
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jac(model, data, jacp, jacr, p, body)
            names = SG2_ARM_JOINT_NAMES[hand]
            cols = [int(model.joint(name).dofadr[0]) for name in names]
            J = np.vstack([jacp[:, cols], jacr[:, cols]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.r_[e_p, e_r])
            dq = np.clip(dq, -0.1, 0.1)
            for name, d in zip(names, dq):
                low, high = JOINT_LIMITS[name]
                data.qpos[adr[name]] = np.clip(data.qpos[adr[name]] + d, low, high)
        if done:
            break
    q = np.array([data.qpos[adr[name]] for name in joint_names])
    return q, done, err


class ControllerButtonActions:
    def __init__(self, record_control=None, hold_s=1.0, speed_scale=3.0, lift_dz=0.05,
                 spread_dy=0.05, *, fold_speed_scale=10.0):
        self.record_control = record_control
        self.hold_s = float(hold_s)
        if not 0 < speed_scale <= 3.0:
            raise ValueError("speed_scale must be in (0, 3]; alignment uses up to 30 deg/s")
        if not 0 < fold_speed_scale <= 10.0:
            raise ValueError("fold_speed_scale must be in (0, 10]; fold/bringup uses up to 100 deg/s")
        self.speed_scale = float(speed_scale)
        self.fold_speed_scale = float(fold_speed_scale)
        self.lift_dz = float(lift_dz)
        # Hands held together make the fold path pass within millimetres; spreading them
        # first keeps every step clear of the 5 mm collision margin.
        self.spread_dy = float(spread_dy)
        self._base_home_speed = None
        self._speed_notice = ""
        self.lock_input = False
        self.status = "X 녹화 · Y 삭제 · A Initial pose · B(1초) reset pose (위로·바깥 5cm → 접기 → bringup)"
        self._prev = {}
        self._held_since = {}
        self._fired = set()
        self._sequence = []          # remaining (label, pose dict or full q vector, speed scale)
        self._seq_name = None
        self._step_active = False
        self._step_reached = False
        self._current = None          # (label, pose, speed scale) of the running step, for retries
        self._retries = 0
        self._retry_at = None

    # ---------- buttons ----------
    @staticmethod
    def _read(frame):
        b = (frame or {}).get("buttons") or {}
        left, right = b.get("left") or {}, b.get("right") or {}
        return {"X": bool(left.get("x")), "Y": bool(left.get("y")),
                "A": bool(right.get("a")), "B": bool(right.get("b"))}

    def _pressed_edge(self, name, state):
        return state[name] and not self._prev.get(name, False)

    def _held(self, name, state, now):
        """True once when the button has been held for hold_s; re-arms after release."""
        if not state[name]:
            self._held_since.pop(name, None)
            self._fired.discard(name)
            return False
        start = self._held_since.setdefault(name, now)
        if name not in self._fired and now - start >= self.hold_s:
            self._fired.add(name)
            return True
        return False

    def _hold_progress(self, now):
        parts = []
        for name in ("B",):
            if name in self._held_since and name not in self._fired:
                parts.append("%s %.0f%%" % (name, 100 * min(1.0, (now - self._held_since[name]) / self.hold_s)))
        return " · ".join(parts)

    # ---------- pose sequence ----------
    @staticmethod
    def _bridge_home_speed(real):
        if real.dry_run:
            return np.inf
        limits = ((real.last_feedback or {}).get("bridge") or {}).get("limits") or {}
        speed = limits.get("home_joint_speed_rad_s")
        if (isinstance(speed, bool) or not isinstance(speed, (int, float))
                or not np.isfinite(speed) or speed <= 0):
            raise ValueError("로봇 HOME 속도 제한 확인 불가: worker 피드백을 확인하세요")
        return float(speed)

    def _sequence_speed_limits(self, real, speed_scale):
        if self._base_home_speed is None:
            self._base_home_speed = real.home_speed_limits.copy()
        speeds = self._base_home_speed.copy()
        if speed_scale is not None:
            rotary = np.asarray([name != "lift_joint" for name in real.planner.joint_names])
            speeds[rotary] = np.minimum(real.teleop_speed_limits[rotary], speeds[rotary] * speed_scale)
            # B's automatic arm motion has its own speed profile. A normal
            # teleop setting of 60 deg/s must not silently cap the 100 deg/s reset.
            speeds[real.arm_mask] = np.minimum(FIXED_QUEST_MAX_ARM_SPEED,
                                               self._base_home_speed[real.arm_mask] * speed_scale)
        return speeds

    def _set_sequence_speed(self, real, speed_scale=None):
        """Scale rotary reset speeds; preserve lift and restore normal HOME afterwards."""
        speeds = self._sequence_speed_limits(real, speed_scale)
        if speed_scale is not None:
            rotary = np.asarray([name != "lift_joint" for name in real.planner.joint_names])
            speeds[rotary] = np.minimum(speeds[rotary], self._bridge_home_speed(real))
        real.home_speed_limits = speeds

    def _finish(self, real, message):
        self._sequence = []
        self._step_active = False
        self._set_sequence_speed(real)
        self.lock_input = False
        self.status = message
        print("[buttons]", message)

    def _start_home(self, real, ui_actions):
        """Request the same guarded return the "Initial pose" button/action performs
        (SG2FixedQuestReal.tick_grip(action="home")), targeting the fixed worker_bringup
        pose (planner.q_home; see SG2FixedQuestReal.tick's initialization)."""
        if (real.last_feedback or {}).get("q") is None:
            self.status = "Initial pose 불가: 로봇 피드백 없음"
            return
        ui_actions.append("home")
        self.status = "Initial pose로 복귀 요청 (A)"
        print("[buttons]", self.status)

    def _start_reset(self, real):
        q_now = (real.last_feedback or {}).get("q")
        if q_now is None:
            self.status = "reset pose 불가: 로봇 피드백 없음"
            return
        try:
            bridge_speed = self._bridge_home_speed(real)
        except ValueError as exc:
            self.status = "reset pose 불가: " + str(exc)
            print("[buttons]", self.status)
            return
        rotary = np.asarray([name != "lift_joint" for name in real.planner.joint_names])
        requested = float(np.max(self._sequence_speed_limits(
            real, max(self.speed_scale, self.fold_speed_scale))[rotary]))
        self._speed_notice = ""
        if bridge_speed + 1e-9 < requested:
            self._speed_notice = " · 로봇 상한 %.0f°/s (고속 reset: Orin protocol 업데이트·inbound 재시작 필요)" % np.rad2deg(bridge_speed)
            print("[buttons] 전체 접기·펴기 유지" + self._speed_notice)
        steps = []
        names = list(real.planner.joint_names)
        err = np.nan
        for dz, dy in ((self.lift_dz, self.spread_dy), (self.lift_dz, 0.0), (self.lift_dz / 2, 0.0)):
            q_up, ok, err = lifted_arm_pose(real.env_measured, names, q_now, dz=dz, dy=dy)
            if ok:
                steps.append(("위로 %.1fcm%s" % (100 * dz, " · 바깥 %.0fcm" % (100 * dy) if dy else ""), q_up, self.speed_scale))
                break
        else:
            print("[buttons] 위로 올리기 IK 실패 (오차 %.1f cm) - 올리기 없이 접기부터 진행" % (100 * err))
        # shutdown1 is the first fixed arm alignment pose. Only accelerate after
        # HOME confirms it was reached, even if the optional lift IK was skipped.
        steps.append((*FOLD_STEPS[0], self.speed_scale))
        steps += [(label, pose, self.fold_speed_scale) for label, pose in FOLD_STEPS[1:] + BRINGUP_STEPS]
        self._sequence = [("%d/%d %s" % (i + 1, len(steps), label), pose, scale)
                          for i, (label, pose, scale) in enumerate(steps)]
        self._seq_name = "reset pose (B)"
        self._step_active = False
        self._current, self._retries, self._retry_at = None, 0, None
        self.lock_input = True

    def _pose_vector(self, real, values, q_now):
        names = list(real.planner.joint_names)
        if isinstance(values, dict):
            q = np.asarray(q_now, dtype=float).copy()
            for joint, value in values.items():
                q[names.index(joint)] = value
        else:
            q = np.asarray(values, dtype=float).copy()
        # Lift IK and partial arm poses otherwise inherit the measured head angle.
        # Keep the bringup target through lift, fold, unfold and retried B stages.
        q[names.index("head_joint1")] = HEAD_TILT
        return q

    def _advance(self, real, result, ui_actions, now):
        if self._step_active:
            returning = real._grip_home_active or real.home_pending or real.planner.homing
            if result.get("state") == "HOME":
                self._step_reached = True
            if returning:
                return
            self._step_active = False
            if not self._step_reached:
                reason = result.get("reason") or "경로 확인 실패"
                if any(key in reason for key in TRANSIENT_REASONS) and self._retries < MAX_STEP_RETRIES:
                    self._retries += 1
                    self._sequence.insert(0, self._current)
                    self._retry_at = now + RETRY_DELAY_S
                    self.status = "%s: %s 재시도 %d/%d (%s)" % (
                        self._seq_name, self._current[0], self._retries, MAX_STEP_RETRIES, reason[:60])
                    print("[buttons]", self.status)
                    return
                if "blocked" in reason or "guard" in reason or "collision" in reason:
                    reason += " → 팔을 조금 벌린 뒤 다시 B"
                self._finish(real, "%s 중단: %s" % (self._seq_name, reason))
                return
            self._retries = 0
            if not self._sequence:
                self._finish(real, "%s 완료 · 양손 grip을 놓았다가 다시 잡으면 조작" % self._seq_name)
                return
        if not self._sequence:
            return
        if self._retry_at is not None:
            if now < self._retry_at:
                return
            self._retry_at = None
        label, values, speed_scale = self._sequence.pop(0)
        self._current = (label, values, speed_scale)
        q_now = (real.last_feedback or {}).get("q")
        if q_now is None:
            self.status = "%s 대기: 로봇 피드백 없음" % self._seq_name
            self._sequence.insert(0, self._current)
            return
        try:
            self._set_sequence_speed(real, speed_scale)
            real.request_pose_target(self._pose_vector(real, values, q_now))
        except ValueError as exc:
            self._finish(real, "%s 중단: %s" % (self._seq_name, exc))
            return
        ui_actions.append("home")
        self._step_active, self._step_reached = True, False
        self.status = "%s: %s 이동 중 (%.0f°/s)" % (
            self._seq_name, label, np.rad2deg(np.max(real.home_speed_limits[real.arm_mask]))) + self._speed_notice
        print("[buttons]", self.status)

    # ---------- per tick ----------
    def __call__(self, now, frame, result, real, ui_actions):
        state = self._read(frame)
        busy = bool(self._sequence) or self._step_active

        if self._pressed_edge("X", state) and self.record_control is not None:
            recording = self.record_control.widget.get_button_option(label_text="Record") == "REC"
            self.record_control.widget.set_button_option(label_text="Record", option_text="STOP" if recording else "REC")
            self.status = "녹화 %s (X)" % ("종료" if recording else "시작")
        if self._pressed_edge("Y", state) and self.record_control is not None:
            self.record_control.widget.set_button_option(label_text="Record", option_text="DEL")
            self.status = "삭제 (Y)"
        if self._pressed_edge("A", state) and not busy:
            self._start_home(real, ui_actions)
        if self._held("B", state, now) and not busy:
            self._start_reset(real)

        self._advance(real, result, ui_actions, now)

        progress = self._hold_progress(now)
        if progress:
            self.status = "누르는 중: " + progress
        self._prev = state
