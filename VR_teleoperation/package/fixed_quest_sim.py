"""Controller-input experiment around the existing SG2 EEF and joint helpers.

No IK, FK, renderer, or real-robot publisher is implemented here. The notebook
owns the visible/physics environment, candidate environment, solvers and guard.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .home_path import plan_home_path, home_plan_summary
from .joint_marker_real import apply_sg2_joint_rate_limit, get_sg2_joint_speed_limits
from utils import (
    SG2_GRIPPER_DRIVER_JOINT_NAMES, SG2_HAND_TYPES, SG2_NULL_REFERENCE_MODES,
    add_sg2_wrist_ik_coordinate_target, get_sg2_wrist_control_T,
    set_sg2_gripper_qpos, set_sg2_null_space_reference,
)


def reset_sg2_sim(env):
    """Same bent-elbow initial pose as sim_eef_controller, plus actuator seeding."""
    env.reset()
    env.forward(q=[-np.pi / 2, -np.pi / 2], joint_names=["arm_l_joint4", "arm_r_joint4"])
    for hand in SG2_HAND_TYPES:
        set_sg2_gripper_qpos(env, hand, 0.0)
    # All retained position actuators start at the current joint position;
    # wheel velocity actuators remain zero (the generated scene fixes the base).
    for index in range(env.model.nu):
        joint_id = int(env.model.actuator_trnid[index, 0])
        if joint_id >= 0 and env.model.actuator_biasprm[index, 1] != 0:
            env.data.ctrl[index] = env.data.qpos[env.model.jnt_qposadr[joint_id]]
    env.forward()


def wrist_error(T_actual, T_target):
    """Report existing FK pose error in metres and radians."""
    dR = T_target[:3, :3] @ T_actual[:3, :3].T
    return {
        "position_m": float(np.linalg.norm(T_actual[:3, 3] - T_target[:3, 3])),
        "rotation_rad": float(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))),
    }


class IKTargetUnreachable(ValueError):
    """A finite IK attempt that needs a new clutch target, not a SIM reset."""


class SG2FixedQuestSimulation:
    """Only simulation-specific orchestration; existing EEF IK computes every arm."""

    def __init__(self, env, env_ik, ik_solvers, arm_joint_names, joint_names,
                 mapper, guard, mode="kinematic", ik_hz=30.0,
                 max_joint_speed=np.deg2rad(20), max_loop_dt=.05,
                 max_ik_position_error=.03, max_ik_rotation_error=np.deg2rad(15),
                 max_command_dt=.02, ik_position_step=.02, ik_rotation_step=np.deg2rad(10),
                 coordinated_arm_rate_limit=False, retry_ik_near_target=False,
                 ik_recovery_timeout_s=0., null_reference="self_motion", null_anchor_on_grip=False,
                 posture_solvers=None, teleop_collision_check=True):
        if mode not in ("kinematic", "dynamic"):
            raise ValueError("mode must be kinematic or dynamic")
        if null_reference not in SG2_NULL_REFERENCE_MODES:
            raise ValueError("null_reference must be one of %s" % (SG2_NULL_REFERENCE_MODES,))
        # "self_motion" holds the arm's unobservable joint coordinate near the
        # startup posture; "previous" leaves it free to wind joint3/joint5 up.
        self.null_reference = null_reference
        if type(null_anchor_on_grip) is not bool:
            raise ValueError("null_anchor_on_grip must be a bool")
        # Re-anchor the self-motion reference at each clutch engagement, so the
        # twist is held where the operator started this grip.
        self.null_anchor_on_grip = null_anchor_on_grip
        self._null_anchor_engaged = False
        if type(coordinated_arm_rate_limit) is not bool:
            raise ValueError("coordinated_arm_rate_limit must be a bool")
        self.coordinated_arm_rate_limit = coordinated_arm_rate_limit
        if type(retry_ik_near_target) is not bool:
            raise ValueError("retry_ik_near_target must be a bool")
        self.retry_ik_near_target = retry_ik_near_target
        if type(teleop_collision_check) is not bool:
            raise ValueError("teleop_collision_check must be a bool")
        self.teleop_collision_check = teleop_collision_check
        if (isinstance(ik_recovery_timeout_s, (bool, np.bool_))
                or not isinstance(ik_recovery_timeout_s, (int, float, np.integer, np.floating))
                or not np.isfinite(ik_recovery_timeout_s) or not 0 <= ik_recovery_timeout_s <= 5.):
            raise ValueError("ik_recovery_timeout_s must be between 0 and 5 seconds")
        # Existing callers retain immediate HOLD; the intent runtime opts in.
        self.ik_recovery_timeout_s = float(ik_recovery_timeout_s)
        if env is env_ik or guard.env_check is env or guard.env_check is env_ik:
            raise ValueError("Physics/visible state, IK and guard must have separate owners")
        if not all(np.isfinite(value) and value > 0 for value in (
                ik_hz, max_joint_speed, max_loop_dt, max_command_dt,
                ik_position_step, ik_rotation_step, max_ik_position_error, max_ik_rotation_error)):
            raise ValueError("Control periods and speed must be positive")
        self.env, self.env_ik = env, env_ik
        self.ik_solvers, self.arm_joint_names = ik_solvers, arm_joint_names
        self.joint_names, self.mapper, self.guard = list(joint_names), mapper, guard
        self.mode, self.ik_period, self.max_loop_dt = mode, 1 / ik_hz, max_loop_dt
        self.max_ik_position_error, self.max_ik_rotation_error = max_ik_position_error, max_ik_rotation_error
        self.max_command_dt = max_command_dt
        self.ik_position_step, self.ik_rotation_step = ik_position_step, ik_rotation_step
        self.q_home = self.env.get_qpos(joint_names=self.joint_names).copy()
        self.return_target = self.q_home.copy()
        self.speed_limits = get_sg2_joint_speed_limits(env, self.joint_names, max_rev_speed=max_joint_speed)
        self.hand_indices = {
            hand: [self.joint_names.index(name) for name in arm_joint_names[hand]]
            for hand in SG2_HAND_TYPES
        }
        # Captured at the first active, feedback-anchored clutch. Ordinary
        # reset()/release/re-arm must never adopt an already twisted posture.
        self.posture_solvers = dict(posture_solvers or {})
        if self.posture_solvers and set(self.posture_solvers) != set(SG2_HAND_TYPES):
            raise ValueError("posture_solvers must contain both arms")
        self.gripper_indices = {
            hand: self.joint_names.index(SG2_GRIPPER_DRIVER_JOINT_NAMES[hand])
            for hand in SG2_HAND_TYPES
        }
        # A checked return path survives reset(): the real adapter plans at the
        # click, then resynchronizes while arming. It is used only if its start
        # still matches the command and its target is still q_home.
        self.home_plan = None
        self.home_plan_start_tolerance = np.where(
            [name in env.pri_joint_names for name in self.joint_names], .001, np.deg2rad(.5))
        self._home_planned = False
        self._home_planning_gap = False
        self.reset()

    def reset(self):
        """Call after resetting the visible environment when rerunning a cell."""
        self.mapper.reset()
        self.q_command = self.env.get_qpos(joint_names=self.joint_names).copy()
        self.q_goal = self.q_command.copy()
        self.last_time, self.last_ik_time = None, -np.inf
        self.last_active = ()
        self.fault_reason = None
        self.physics_remainder = 0.0
        self.ik_errors = {}
        self.ik_retries = {}
        self.ik_targets = {}
        self.posture_ik = {}
        self.ik_target_limited = False
        self.homing = False
        self.home_failed = False
        self.home_rearm_required = False
        self.home_waypoints = []
        self.home_path = None
        self.last_enabled = False
        self._clear_ik_recovery()

    def _reanchor_null_space(self):
        """Point every arm solver's self-motion anchor at the current command pose."""
        rev_pri = list(self.env_ik.rev_pri_joint_names)
        anchor = self.env_ik.get_qpos(joint_names=rev_pri).copy()
        for name, value in zip(self.joint_names, self.q_command):
            if name in rev_pri:
                anchor[rev_pri.index(name)] = value
        for solver in self.ik_solvers.values():
            solver.q_null_anchor_rev_pri = anchor.copy()

    def capture_posture_reference(self, qpos):
        """Capture the initial measured full configuration once per runtime."""
        qpos = np.asarray(qpos, dtype=float)
        pending = [solver for solver in self.posture_solvers.values() if not solver.has_reference]
        try:
            for solver in pending:
                solver.capture_reference(qpos)
        except (ValueError, FloatingPointError):
            for solver in pending:
                solver.reset_reference()
            raise

    def check_posture_command(self, q):
        """Check the actual limited command, not just the IK endpoint."""
        result = {"accepted": True, "reason": "posture disabled", "by_hand": {}}
        if not self.posture_solvers:
            return result
        q = np.asarray(q, dtype=float)
        if q.shape != (len(self.joint_names),) or not np.isfinite(q).all():
            return dict(result, accepted=False, reason="invalid posture command")
        full = self.env.data.qpos.copy()
        for name, value in zip(self.joint_names, q):
            full[int(self.env.model.joint(name).qposadr[0])] = value
        result["reason"] = "posture accepted"
        for hand, solver in self.posture_solvers.items():
            if not solver.has_reference:
                continue
            info = solver.configuration_info(full)
            result["by_hand"][hand] = info
            if not info["valid"]:
                result.update(accepted=False, reason=hand + " " + info["reason"])
                break
        return result

    def check_command(self, q_start, q_end, *, motion="teleop"):
        """Keep automatic moves guarded while teleoperation may skip collision checks."""
        if motion not in ("teleop", "home"):
            raise ValueError("motion must be teleop or home")
        if motion == "home" or self.teleop_collision_check:
            return self.guard.check(q_start, q_end)
        return self.guard.check(q_start, q_end, check_collision=False)

    def plan_home(self, q_start=None, speed_limits=None, target=None):
        """Check the whole return to q_home before any HOME command is made.

        Uses the command-preview context, like every other guard check here.
        Planning time is not a control-loop stall: the next tick may exceed
        max_loop_dt once, while its command step stays capped by max_command_dt.
        """
        q_start = self.q_command.copy() if q_start is None else np.asarray(q_start, dtype=float).copy()
        self._copy_context(self.guard.env_check)
        self.return_target = self.q_home.copy() if target is None else np.asarray(target, dtype=float).copy()
        options = {} if target is None else {"strategies": (("manual_joint", ()),)}
        self.home_plan = plan_home_path(self.guard, self.joint_names, q_start, self.return_target,
                                        self.speed_limits if speed_limits is None else speed_limits, **options)
        self._home_planned = True
        return self.home_plan

    def _home_plan_matches(self, plan):
        return (plan is not None and np.array_equal(plan["q_home"], self.return_target)
                and np.all(np.abs(plan["q_start"] - self.q_command) <= self.home_plan_start_tolerance))

    def _clear_ik_recovery(self):
        self._ik_recovery_started = None
        self._ik_recovery_attempts = 0
        self._ik_recovery_reason = None

    def ik_recovery_info(self, now, status=None):
        """Snapshot the bounded retry episode without changing its deadline."""
        return {
            "status": status or ("idle" if self._ik_recovery_started is None else "waiting"),
            "elapsed_s": (0. if self._ik_recovery_started is None
                          else max(0., now - self._ik_recovery_started)),
            "timeout_s": self.ik_recovery_timeout_s,
            "attempts": self._ik_recovery_attempts,
            "reason": self._ik_recovery_reason,
        }

    def _copy_context(self, target):
        target.data.qpos[:] = self.env.data.qpos
        target.data.qvel[:] = 0
        target.forward()

    def _write_candidate(self, q):
        self.env_ik.forward(q=q, joint_names=self.joint_names)
        for hand, index in self.gripper_indices.items():
            set_sg2_gripper_qpos(self.env_ik, hand, q[index])

    def _solve(self, targets):
        self._copy_context(self.env_ik)
        self._write_candidate(self.q_command)
        errors = {}
        self.ik_errors = errors
        self.ik_retries = {}
        self.ik_targets = {}
        self.ik_target_limited = False
        self.posture_ik = {}
        for hand, target in targets.items():
            # Give the existing IK a nearby target when input outruns the arm.
            # Keep the full operator target for tracking-error reporting.
            current = get_sg2_wrist_control_T(self.env_ik, hand)
            delta_p = target[:3, 3] - current[:3, 3]
            delta_rot = Rotation.from_matrix(target[:3, :3] @ current[:3, :3].T).as_rotvec()
            fraction = min(1., self.ik_position_step / max(np.linalg.norm(delta_p), 1e-12),
                           self.ik_rotation_step / max(np.linalg.norm(delta_rot), 1e-12))
            solver = self.ik_solvers[hand]
            start_q = self.env_ik.get_qpos(joint_names=solver.joints_use).copy()
            for attempt, retry_scale in enumerate((1., .5) if self.retry_ik_near_target else (1.,)):
                if attempt:
                    self.env_ik.forward(q=start_q, joint_names=solver.joints_use)
                solve_target = current.copy()
                solve_target[:3, 3] += fraction * retry_scale * delta_p
                solve_target[:3, :3] = Rotation.from_rotvec(fraction * retry_scale * delta_rot).as_matrix() @ current[:3, :3]
                self.ik_targets[hand] = solve_target
                self.ik_target_limited |= fraction * retry_scale < 1.
                self.ik_retries[hand] = attempt
                # Retry from the same pose with a nearer target, never a looser tolerance.
                posture_solver = self.posture_solvers.get(hand)
                if posture_solver is not None:
                    info = posture_solver.solve(self.env_ik, solve_target)
                    self.posture_ik[hand] = {key: value for key, value in info.items()
                                             if key != "qpos_used_best"}
                    position_tolerance = min(posture_solver.position_tolerance,
                                             self.ik_position_step * retry_scale / 2)
                    rotation_tolerance = min(posture_solver.rotation_tolerance,
                                             self.ik_rotation_step * retry_scale / 2)
                else:
                    set_sg2_null_space_reference(solver, self.env_ik, null_reference=self.null_reference)
                    solver.reset_buffers(self.env_ik)
                    add_sg2_wrist_ik_coordinate_target(solver, self.env_ik, hand, solve_target, axis_width=.10)
                    _, info = solver.solve_ik(env=self.env_ik, joint_limit_handle_flag=True, nullspace_control_flag=True)
                    position_tolerance = min(self.max_ik_position_error, self.ik_position_step * retry_scale / 2)
                    rotation_tolerance = min(self.max_ik_rotation_error, self.ik_rotation_step * retry_scale / 2)
                if not np.all(np.isfinite(info["qpos_used_best"])):
                    raise FloatingPointError("%s IK returned non-finite joints" % hand)
                self.env_ik.forward(q=info["qpos_used_best"], joint_names=solver.joints_use)
                errors[hand] = wrist_error(get_sg2_wrist_control_T(self.env_ik, hand), solve_target)
                if not all(np.isfinite(value) for value in errors[hand].values()):
                    raise FloatingPointError("%s IK returned non-finite error" % hand)
                if (info.get("success", True)
                        and errors[hand]["position_m"] <= position_tolerance
                        and errors[hand]["rotation_rad"] <= rotation_tolerance):
                    break
            else:
                raise IKTargetUnreachable("%s IK target unavailable; release both grips and retry" % hand)
        return self.env_ik.get_qpos(joint_names=self.joint_names).copy()

    def tick(self, frame, enabled=False, now=None, action=None, eef_poses=None, pause_motion=False,
             posture_reference_qpos=None, q_target=None):
        """Validate input, solve at ik_hz, limit/check one command, then apply to SIM.

        action="policy": drive q_target (JOINT_NAMES order; the caller must have
        already clipped it into JOINT_LIMITS, e.g. via policy_inference's
        clip_to_joint_limits) through the same rate-limit + collision/posture
        guard as teleop and HOME motion, at the "home" collision profile --
        always checked, regardless of teleop_collision_check -- since an
        autonomous policy is not teleoperation. No VR mapper/IK runs on this
        tick; `frame` is ignored. `enabled=False` holds position, like a
        released grip. Call with a fresh q_target every tick: nothing here
        repeats a stale one.
        """
        started = time.perf_counter()
        now = time.monotonic() if now is None else float(now)
        if not np.isfinite(now):
            raise ValueError("now must be a finite monotonic timestamp")
        if action not in (None, "home", "cancel_home", "policy"):
            raise ValueError("Unknown SIM action")
        if action == "policy":
            q_target = None if q_target is None else np.asarray(q_target, dtype=float).copy()
            if (q_target is None or q_target.shape != (len(self.joint_names),)
                    or not np.isfinite(q_target).all()):
                raise ValueError("policy action requires a finite q_target of shape (%d,)"
                                 % len(self.joint_names))
        elif q_target is not None:
            raise ValueError("q_target is only used with action='policy'")
        dt = 0.0 if self.last_time is None else now - self.last_time
        self.last_time = now
        planning_gap, self._home_planning_gap = self._home_planning_gap, False
        delayed = not 0 <= dt <= self.max_loop_dt and not (planning_gap and dt >= 0)
        if not enabled and self.last_enabled:
            self.home_failed = False
        if not enabled and not self.homing and not self.home_failed:
            self.fault_reason = None
            self.home_rearm_required = False
        home_plan = None
        home_blocked = None
        if action == "home":
            # Never start a return whose full path is unchecked. A blocked path
            # leaves the command in place; a later request plans again from it.
            if not self._home_plan_matches(self.home_plan):
                # A manual checked path may need rechecking after the arm ACK.
                manual_target = (self.return_target if self.home_plan is not None
                                 and self.home_plan.get("strategy") == "manual_joint" else None)
                self.plan_home(target=manual_target)
            home_plan, self.home_plan = self.home_plan, None
            self.home_rearm_required = True
            self.mapper.require_release()
            if home_plan["ok"]:
                self.homing, self.home_failed, self.fault_reason = True, False, None
                self.home_waypoints = [q.copy() for q in home_plan["waypoints"][1:]] or [self.return_target.copy()]
                self.home_path = {"strategy": home_plan["strategy"], "stages": len(self.home_waypoints),
                                  "duration_s": home_plan["duration_s"]}
                self.q_goal = self.home_waypoints[0].copy()
            else:
                self.homing, self.home_failed = False, True
                self.fault_reason = home_plan["reason"]
                self.home_waypoints, self.home_path = [], None
                self.q_goal = self.q_command.copy()
                home_blocked = next((attempt["guard"] for attempt in home_plan["attempts"] if "guard" in attempt), None)
        if action == "cancel_home" or (self.homing and self.last_enabled and not enabled and action != "home"):
            self.homing = False
            if self.home_failed:
                self.fault_reason = None
                self.home_failed = False
            self.home_rearm_required = True
            self.home_waypoints, self.home_path = [], None
            self.q_goal = self.q_command.copy()
            self.mapper.require_release()
        self.last_enabled = bool(enabled)
        if delayed:
            if enabled or self.homing:
                self.fault_reason = "control loop delay; switch SIM control OFF then ON"
                self.home_failed = self.homing
            self.homing = False
            dt = 0.0
        command_dt = min(dt, self.max_command_dt)
        measured = self.env.get_qpos(joint_names=self.joint_names)
        if not np.all(np.isfinite(measured)):
            raise RuntimeError("Non-finite MuJoCo state; reset simulation")
        # Real adapters may anchor each clutch on fresh measured wrist FK.
        eef = ({hand: get_sg2_wrist_control_T(self.env, hand) for hand in SG2_HAND_TYPES}
               if eef_poses is None else eef_poses)
        if action == "policy":
            # No VR grip/IK on a policy tick -- q_target (already validated above)
            # is the command, exactly like a HOME waypoint is.
            mapping = dict(state="POLICY" if enabled else "HOLD",
                           reason="policy target" if enabled else "SIM control OFF",
                           targets={}, active=(), triggers={})
        elif pause_motion and enabled and not self.homing:
            # Keep clutch anchors but never consume an emulated pose or advance IK.
            mapping = dict(state="TRACKING_PAUSED", reason="waiting for valid controller tracking",
                           targets={}, active=(), triggers={})
        else:
            mapping = self.mapper.update(frame, eef, enabled=enabled and self.fault_reason is None
                                         and not self.homing and not self.home_rearm_required, now=now)
        targets, active = mapping["targets"], tuple(mapping["active"])
        if active and self.posture_solvers:
            try:
                self.capture_posture_reference(self.env.data.qpos if posture_reference_qpos is None
                                               else posture_reference_qpos)
            except (ValueError, FloatingPointError) as exc:
                self.fault_reason = "posture reference unavailable: " + str(exc)
        if self.null_anchor_on_grip and not self.posture_solvers:
            if active and not self._null_anchor_engaged:
                self._reanchor_null_space()
            self._null_anchor_engaged = bool(active)
        guard_info = home_blocked or {"accepted": True, "reason": "hold", "min_distance_m": None}
        limited = False
        arm_rate_scales = {}
        if self.homing:
            mapping = dict(mapping, state="HOMING", reason="returning to initial pose")
        elif self.home_rearm_required and enabled:
            mapping = dict(mapping, reason="switch SIM control OFF before resuming")
        if self.fault_reason is not None:
            mapping = dict(mapping, state="FAULT", reason=self.fault_reason)
            targets, active = {}, ()

        recovery = self.ik_recovery_info(now)
        if self._ik_recovery_started is not None:
            # A short tracking pause preserves the original IK deadline. OFF,
            # HOME and invalid input must never leave a recovery to resume later.
            if (not enabled or self.homing or self.fault_reason is not None
                    or (not active and mapping["state"] != "TRACKING_PAUSED")):
                recovery = dict(self.ik_recovery_info(now, "cancelled"), reason=mapping["reason"])
                self._clear_ik_recovery()
            elif now - self._ik_recovery_started >= self.ik_recovery_timeout_s - 1e-9:
                recovery = self.ik_recovery_info(now, "timeout")
                reason = "IK recovery timeout after %.2f s: %s; release both grips and retry" % (
                    recovery["elapsed_s"], recovery["reason"])
                self._clear_ik_recovery()
                self.mapper.require_release()
                self.q_goal = self.q_command.copy()
                targets, active = {}, ()
                mapping = dict(mapping, state="HOLD", reason=reason)

        # A released hand must not continue converging toward its old EEF target.
        # Not for a policy tick: q_target below already covers every joint.
        if action != "policy":
            for hand in SG2_HAND_TYPES:
                if hand not in active and not self.homing:
                    indices = self.hand_indices[hand] + [self.gripper_indices[hand]]
                    self.q_goal[indices] = self.q_command[indices]
        if active and (now - self.last_ik_time >= self.ik_period - 1e-9 or active != self.last_active):
            # Failed attempts also consume the IK cadence. Keep last_active as
            # the mapped hands, even when output motion is held below.
            self.last_ik_time = now
            if self._ik_recovery_started is not None:
                self._ik_recovery_attempts += 1
            try:
                self.q_goal = self._solve(targets)
                if self._ik_recovery_started is not None:
                    recovery = self.ik_recovery_info(now, "recovered")
                    self._clear_ik_recovery()
            except IKTargetUnreachable as exc:
                self.q_goal = self.q_command.copy()
                if self.ik_recovery_timeout_s > 0:
                    if self._ik_recovery_started is None:
                        self._ik_recovery_started = now
                        self._ik_recovery_attempts = 1
                    self._ik_recovery_reason = str(exc).split("; release both grips")[0]
                    recovery = self.ik_recovery_info(now)
                else:
                    self.mapper.require_release()
                    targets, active = {}, ()
                    mapping = dict(mapping, state="HOLD", reason=str(exc))
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                if self._ik_recovery_started is not None:
                    recovery = dict(self.ik_recovery_info(now, "cancelled"), reason=str(exc))
                    self._clear_ik_recovery()
                self.fault_reason = str(exc)
                self.q_goal = self.q_command.copy()
                targets, active = {}, ()
                mapping = dict(mapping, state="FAULT", reason=self.fault_reason)
        if mapping["state"] != "TRACKING_PAUSED" or self._ik_recovery_started is None:
            self.last_active = active
        if self._ik_recovery_started is not None:
            # Freeze all 19 commands, including triggers and any unexecuted
            # previous q_goal. Fresh input still updates the mapping/anchors.
            self.q_goal = self.q_command.copy()
            active = ()
            if mapping["state"] != "TRACKING_PAUSED":
                mapping = dict(mapping, state="IK_WAIT", reason="IK retry pending: " + self._ik_recovery_reason)

        policy_driving = action == "policy" and enabled and self.fault_reason is None
        if policy_driving:
            self.q_goal = q_target.copy()

        if (active or self.homing or policy_driving) and command_dt > 0 and self.fault_reason is None:
            if self.homing:
                # Advance only from a waypoint the command actually reached. A
                # candidate held back by real pacing then cannot cut a corner.
                if not self.home_waypoints:
                    self.home_waypoints = [self.return_target.copy()]
                while (len(self.home_waypoints) > 1
                       and np.allclose(self.q_command, self.home_waypoints[0], rtol=0, atol=1e-9)):
                    self.home_waypoints.pop(0)
                self.q_goal = self.home_waypoints[0].copy()
            for hand in active:
                index = self.gripper_indices[hand]
                name = self.joint_names[index]
                low, high = self.env.get_joint_mins([name])[0], self.env.get_joint_maxs([name])[0]
                self.q_goal[index] = low + mapping["triggers"][hand] * (high - low)
            self._copy_context(self.env_ik)
            self._write_candidate(self.q_goal)
            # Reuse the real joint-controller rate limiter on the candidate env.
            # Preserve the seven-joint IK direction separately for each arm.
            # HOME scales all joints together so commands stay on the checked
            # straight segment; callers that did not opt in keep independent limits.
            coordinated_hands = list(active) if self.coordinated_arm_rate_limit and not self.homing else []
            coordinated_groups = ([self.joint_names] if self.homing
                                  else [self.arm_joint_names[hand] for hand in coordinated_hands])
            rate_info = apply_sg2_joint_rate_limit(
                self.env_ik, self.joint_names, self.q_command, command_dt, self.speed_limits,
                coordinated_groups=coordinated_groups)
            arm_rate_scales = dict(zip(coordinated_hands, rate_info["group_scales"]))
            candidate, limited = rate_info["qpos"], rate_info["limited"]
            self._copy_context(self.guard.env_check)
            # A policy is not teleoperation: keep it on the always-checked "home"
            # collision profile even when teleop_collision_check is False.
            motion = "home" if self.homing or policy_driving else "teleop"
            guard_info = self.check_command(self.q_command, candidate, motion=motion)
            if guard_info["accepted"] and not self.homing:
                posture_check = self.check_posture_command(candidate)
                if not posture_check["accepted"]:
                    guard_info = dict(guard_info, accepted=False,
                                      reason=posture_check["reason"], posture=posture_check)
            # Also check measured-to-command lag in a dynamic run.
            if guard_info["accepted"] and self.mode == "dynamic":
                guard_info = self.check_command(measured, candidate, motion=motion)
            if guard_info["accepted"]:
                self.q_command = candidate.copy()
            else:
                self.fault_reason = "command rejected: " + guard_info["reason"]
                self.home_failed = self.homing
                self.homing = False
                self.q_goal = self.q_command.copy()
                mapping = dict(mapping, state="FAULT", reason=self.fault_reason)
                targets, active = {}, ()
        elif self._ik_recovery_started is not None:
            # A held target uses the teleoperation policy, including measured
            # drift in a dynamic SIM. No solver candidate is used.
            self._copy_context(self.guard.env_check)
            guard_info = self.check_command(self.q_command, self.q_command, motion="teleop")
            if guard_info["accepted"] and self.mode == "dynamic":
                guard_info = self.check_command(measured, self.q_command, motion="teleop")
            if not guard_info["accepted"]:
                self.fault_reason = "held command rejected: " + guard_info["reason"]
                recovery = dict(self.ik_recovery_info(now, "cancelled"), reason=self.fault_reason)
                self._clear_ik_recovery()
                targets, active = {}, ()
                mapping = dict(mapping, state="FAULT", reason=self.fault_reason)

        home_path = None
        if self.homing and self.fault_reason is None:
            # Checked on every tick, not only motion ticks, but never on the
            # request tick itself, so a return always reports HOMING first.
            if (action != "home" and len(self.home_waypoints) <= 1
                    and np.allclose(self.q_command, self.return_target, rtol=0, atol=1e-9)):
                self.homing = False
                self.home_waypoints, self.home_path = [], None
                mapping = dict(mapping, state="HOME", reason="initial joint target reached; SIM control remains OFF")
            elif self.home_path is not None:
                home_path = dict(self.home_path, stage=self.home_path["stages"] - max(1, len(self.home_waypoints)) + 1)

        if self.mode == "kinematic":
            self.env.forward(q=self.q_command, joint_names=self.joint_names)
            for hand, index in self.gripper_indices.items():
                set_sg2_gripper_qpos(self.env, hand, self.q_command[index])
        else:
            self.physics_remainder += dt
            nstep = int((self.physics_remainder + 1e-12) / self.env.model.opt.timestep)
            self.physics_remainder -= nstep * self.env.model.opt.timestep
            if nstep:
                # Physics owns qpos; only the existing actuator interface is used.
                self.env.step(ctrl=self.q_command, joint_names=self.joint_names, nstep=nstep)
                self.env.forward()  # refresh FK only, without writing qpos
        measured = self.env.get_qpos(joint_names=self.joint_names).copy()
        errors = {hand: wrist_error(get_sg2_wrist_control_T(self.env, hand), target) for hand, target in targets.items()}
        if self._home_planned:
            # A path check before or during this tick lengthens only the gap to
            # the next tick. Excuse that one gap, not any later loop delay.
            self._home_planned, self._home_planning_gap = False, True
        return {
            "enabled": bool(enabled), "state": mapping["state"], "reason": mapping["reason"], "active": list(active),
            "action": action, "homing": self.homing, "home_error": measured - self.q_home,
            "home_path": home_path, "home_plan": home_plan_summary(home_plan),
            "pose_age_s": mapping.get("pose_age_s"), "targets": targets,
            "q_command": self.q_command.copy(), "q_measured": measured,
            "eef_errors": errors, "ik_errors": copy.deepcopy(self.ik_errors),
            "posture_ik": copy.deepcopy(self.posture_ik),
            "posture_reference_captured": bool(self.posture_solvers) and all(
                solver.has_reference for solver in self.posture_solvers.values()),
            "ik_recovery": recovery,
            "mapping_stages": copy.deepcopy(mapping.get("mapping_stages", {})),
            "intent_profile_id": self.mapper.intent_profile_id,
            "ik_retries": self.ik_retries.copy(),
            "coordinated_arm_rate_limit": self.coordinated_arm_rate_limit,
            "arm_rate_scales": arm_rate_scales,
            "ik_targets": copy.deepcopy(self.ik_targets), "ik_target_limited": self.ik_target_limited,
            "guard": guard_info, "rate_limited": limited, "dt": dt, "command_dt": command_dt,
            "sim_time_s": float(self.env.data.time), "compute_ms": 1000 * (time.perf_counter() - started),
        }


class SyntheticControllerSource:
    """Released -> left small translation -> release -> right -> release, at 30 Hz."""

    def __init__(self, start_time=0.0, hz=30.0):
        self.start_time, self.hz, self._last = start_time, hz, None

    def latest(self, now):
        t = max(0.0, now - self.start_time)
        seq = int(t * self.hz + 1e-9)
        if self._last is not None and self._last["seq"] == seq:
            return self._last
        hands = {}
        for hand, begin, x in (("left", 1.0, .20), ("right", 6.0, -.20)):
            active = begin <= t < begin + 4.0
            u = np.clip((t - begin) / 2.0, 0, 1)
            blend = u * u * (3 - 2 * u)
            hands[hand] = {
                "position_m": [x, 1.1 + .04 * blend, -.6],
                "quaternion_xyzw": [0., 0., 0., 1.],
                "valid": True, "emulated": False, "grip": float(active),
                "trigger": .3 * blend if active else 0.,
            }
        self._last = {
            "version": 1, "session_id": "synthetic", "origin_id": 0, "seq": seq,
            "sample_time_ms": 1000 * seq / self.hz, "received_at": now,
            "visibility": "visible", "head": {"position_m": [0, 1.6, 0], "quaternion_xyzw": [0, 0, 0, 1]},
            "hands": hands,
        }
        return self._last


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class ControllerRecorder:
    """Write raw input and SIM results together; no robot transport is opened."""

    def __init__(self, path, joint_names, mode, configuration=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.file = Path(path).open("w", encoding="utf-8")
        self.file.write(json.dumps({"type": "metadata", "version": 1, "joint_names": list(joint_names),
                                   "mode": mode, "configuration": configuration or {}}, default=_json_default) + "\n")

    def write(self, now, frame, result):
        self.file.write(json.dumps({"type": "tick", "recorded_at": now, "frame": frame, "result": result}, default=_json_default) + "\n")

    def close(self):
        self.file.close()


class ReplayControllerSource:
    """Replay original input arrival timing, duplicates and outages at original speed."""

    def __init__(self, path, start_time=0.0):
        with Path(path).open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
        self.metadata = next((row for row in records if row.get("type") == "metadata"), {})
        self.rows = [row for row in records if row.get("type") == "tick"]
        if not self.rows:
            raise ValueError("Recording has no ticks")
        times = np.array([row["recorded_at"] for row in self.rows], dtype=float)
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0):
            raise ValueError("Recording timestamps must be finite and monotonic")
        if any(type(row.get("result", {}).get("enabled")) is not bool for row in self.rows):
            raise ValueError("Recording must include the original SIM enabled state")
        self.start_time, self.original_start = start_time, times[0]
        self.duration_s = float(times[-1] - times[0])
        self.index, self._latest, self.enabled = 0, None, False
        self.action = None

    def latest(self, now):
        shift = self.start_time - self.original_start
        self.action = None
        while self.index < len(self.rows) and self.rows[self.index]["recorded_at"] + shift <= now + 1e-9:
            self._latest = copy.deepcopy(self.rows[self.index]["frame"])
            self.enabled = self.rows[self.index]["result"]["enabled"]
            action = self.rows[self.index]["result"].get("action")
            if action is not None:
                self.action = action
            if self._latest is not None:
                self._latest["received_at"] += shift
            self.index += 1
        return self._latest
