"""Clutched EEF and joint commands shared by the UR5e teleoperation backends."""

import numpy as np
from scipy.spatial.transform import Rotation


HZ = 50
MAX_TICK_INTERVAL = 0.12      # s before the operator must release the clutch again
GOAL_TOLERANCE = 1e-8         # m for position errors, rad for angle errors
MAX_LINEAR_SPEED = 0.10       # m/s
MAX_ANGULAR_SPEED = 0.60      # rad/s
MAX_POSITION_LEAD = 0.025     # m from feedback
MAX_ROTATION_LEAD = 0.12      # rad from feedback
MAX_JOINT_SPEED = 0.5         # rad/s per joint
MAX_JOINT_LEAD = 0.12         # rad per joint from feedback
JOINT_RAD_PER_METRE = 5.0     # translation channels drive joints 1–3 in joint-delta
COMMAND_MODES = ("eef-delta", "eef-absolute", "joint-delta", "joint-absolute")
CONTROL_LAWS = ("position", "impedance")


def bounded_vector(vector, limit):
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("Expected a finite 3D vector")
    norm = np.linalg.norm(vector)
    return vector * (limit / norm) if norm > limit else vector


class TeleopController:
    """Anchor clutched targets to feedback and dispatch the selected control law.

    Each tick advances either the TCP pose in `target` or the angles in
    `joint_target`. An optional numerical `goal` can take several ticks to reach.
    """

    def __init__(self, backend, *, dry_run=False, command_mode="eef-delta", control_law="position",
                 joint_limits=None):
        if command_mode not in COMMAND_MODES or control_law not in CONTROL_LAWS:
            raise ValueError("Unknown UR5e command mode or control law")
        self.command_mode = command_mode
        self.control_law = control_law
        self.measured_joints = None
        self.joint_limits = joint_limits
        self.backend = backend
        self.dry_run = dry_run
        self.measured = None
        self._reset_targets()

    @property
    def joint_space(self):
        return self.command_mode.startswith("joint-")

    @property
    def absolute(self):
        return self.command_mode.endswith("-absolute")

    @property
    def active(self):
        return self.target is not None or self.joint_target is not None

    def _reset_targets(self):
        self.ready_after_release = False
        self.last_tick = None
        self.target = self.joint_target = self.goal = None
        self.target_from_device = False

    def set_command_mode(self, command_mode, *, simulation, sample, device=None):
        if command_mode not in COMMAND_MODES:
            return "지원하지 않는 Command Mode입니다."
        return self._configure(command_mode, self.control_law, simulation, sample, device)

    def set_control_law(self, control_law, *, simulation, sample, device=None):
        if control_law not in CONTROL_LAWS:
            return "지원하지 않는 Control Law입니다."
        return self._configure(self.command_mode, control_law, simulation, sample, device)

    def _release_check(self, simulation, sample, device):
        if sample is not None:
            if sample.enabled or not sample.buttons_released:
                return "Clutch와 Gripper 버튼을 놓은 뒤 다시 선택하세요."
            return ""

        # An idle simulation or preview can be configured with no input device.
        if (self.active or not (simulation or self.dry_run) or
                getattr(device, "clutch_pressed", None) is True or
                getattr(device, "gripper_chord", None) is True or
                getattr(device, "waiting_for_release", False) is True):
            return "입력 장치 연결을 확인하고 Clutch를 놓은 뒤 다시 선택하세요."
        return ""

    def set_impedance_gains(self, space, gains, *, simulation, sample, device=None):
        if self.control_law != "impedance" or space != ("joint" if self.joint_space else "eef"):
            return "현재 Command Mode에 맞는 TCP 또는 Joint 설정인지, Impedance Control이 선택되어 있는지 확인하세요."
        reason = self._release_check(simulation, sample, device)
        if reason:
            return reason
        if self.active:
            return "Clutch를 놓고 팔 조작을 마친 뒤 적용하세요."
        self.backend.hold()
        try:
            self.backend.set_impedance_gains(space, gains)
        except ValueError as error:
            return f"Stiffness / Damping을 적용할 수 없습니다. {error}"
        self._reset_targets()
        self.measured = self.measured_joints = None
        return ""

    def _configure(self, command_mode, control_law, simulation, sample, device):
        if (command_mode, control_law) == (self.command_mode, self.control_law):
            return ""
        reason = self._release_check(simulation, sample, device)
        if reason:
            return reason

        using_joint_writer = self.joint_space and self.control_law == "impedance"
        needs_joint_writer = command_mode.startswith("joint-") and control_law == "impedance"
        change_writer = not simulation and needs_joint_writer != using_joint_writer
        if change_writer and not self.dry_run:
            return "Joint Impedance로 전환하거나 해제하려면 DRY RUN에서 설정을 바꾼 뒤 REAL RUN을 선택하세요."

        self.backend.hold()
        if change_writer:
            self.backend.set_joint_impedance_writer(needs_joint_writer)
        self._reset_targets()
        self.measured = self.measured_joints = None
        self.command_mode = command_mode
        self.control_law = control_law
        return ""

    @staticmethod
    def _real_run_input_error(sample, device):
        if sample is not None and not sample.enabled and sample.buttons_released:
            return ""
        if not hasattr(device, "clutch_pressed"):
            return "Quest tracking 상태를 확인하고 Clutch를 놓은 뒤 REAL RUN을 다시 선택하세요."
        if device.status:
            return "SpaceMouse를 USB에 연결한 뒤 다시 시도하세요."
        if device.clutch_pressed is None:
            return f"SpaceMouse {device.button}번 버튼을 한 번 눌렀다 놓은 뒤 REAL RUN을 다시 선택하세요."
        return "SpaceMouse의 모든 버튼을 놓은 뒤 REAL RUN을 다시 선택하세요."

    def set_mode(self, dry_run, *, simulation, sample, device=None):
        if dry_run == self.dry_run:
            return ""
        if not dry_run and not simulation:
            reason = self._real_run_input_error(sample, device)
            if reason:
                return reason
        self._reset_targets()
        if simulation:
            self.backend.hold()
        else:
            try:
                # Detaching the real writer also stops its servo and script.
                self.backend.set_motion_enabled(not dry_run)
            except (RuntimeError, OSError, ImportError) as error:
                if not dry_run:
                    return f"REAL RUN 전환 실패 · {error}"
                if self.backend.motion_enabled:
                    raise
                self.dry_run = True
                return f"실기 제어 연결 해제 · 정지 응답을 확인하지 못했습니다. Teach Pendant를 확인하세요. ({error})"
        self.dry_run = dry_run
        return ""

    def update(self, sample, now, *, scale=1.0, values=None):
        """Apply one device tick plus an optional one-shot six-axis numerical command (SI)."""
        notice = self.plan(sample, now, measured=self.backend.pose(),
                           measured_joints=self.backend.joints() if self.joint_space else None,
                           scale=scale, values=values)
        if not self.active:
            self.backend.hold()
        elif not self.dry_run:
            self._send_target()
        return notice

    def plan(self, sample, now, *, measured, measured_joints=None, scale=1.0, values=None):
        """Advance the input and target state using supplied feedback, without robot I/O."""
        dt = 0.0 if self.last_tick is None else now - self.last_tick
        self.last_tick = now
        self.measured = measured
        self.measured_joints = measured_joints

        if not self._update_clutch(sample, dt):
            self.target = self.joint_target = self.goal = None
            if values is not None:
                return "Clutch를 놓았다 다시 누른 상태에서 Apply Target을 누르세요."
            return

        # A delayed tick may advance the target by at most two control periods.
        return self._advance_target(sample, min(dt, 2 / HZ), scale, values)

    def _update_clutch(self, sample, dt):
        """Return whether this tick may advance a target. Input loss requires a new release."""
        if sample is None or not 0.0 <= dt <= MAX_TICK_INTERVAL:
            self.ready_after_release = False
            return False
        if not sample.enabled:
            self.ready_after_release = True
            return False
        return self.ready_after_release

    def _advance_target(self, sample, dt, scale, values):
        starting = not self.active
        if starting:
            if self.joint_space:
                self.joint_target = self.measured_joints.copy()
            else:
                self.target = self.measured

        if values is not None:
            values = np.array(values, dtype=float, copy=True)
            if values.shape != (6,) or not np.all(np.isfinite(values)):
                raise ValueError("Expected six finite command values")

        delta = np.zeros(6)
        if not starting and not self.absolute:
            translation = bounded_vector(sample.translation, MAX_LINEAR_SPEED * dt)
            rotation = bounded_vector(sample.rotation, MAX_ANGULAR_SPEED * dt)
            delta = scale * np.r_[translation, rotation]

        numerical = self.absolute or values is not None or self.goal is not None or sample.joint_drag is not None
        if self.joint_space:
            notice = self._advance_joints(delta, values, dt, sample.joint_drag)
            if notice:
                return notice
        elif not starting or values is not None:
            self._advance_tcp(delta, values, dt)

        # A numerical target still needs to settle after its input ramp finishes.
        # A centered device supplies no new target to replace it.
        if numerical:
            self.target_from_device = False
        elif starting or np.any(delta):
            self.target_from_device = True

    def _send_target(self):
        """Dispatch a calculated target. Both laws share the same limits and clutch."""
        if self.joint_space:
            if self.control_law == "impedance":
                self.backend.command_joint_impedance(self.joint_target)
            else:
                self.backend.command_joints(self.joint_target)
        elif self.control_law == "impedance":
            self.backend.command_cartesian_impedance(*self.target)
        else:
            self.backend.command(*self.target)

    def _advance_tcp(self, delta, values, dt):
        xyz, rotation = self.target
        if values is not None:
            requested_rotation = Rotation.from_rotvec(values[3:])
            if self.absolute:
                self.goal = values[:3], requested_rotation
            else:
                self.goal = xyz + values[:3], requested_rotation * rotation
        if self.goal is not None:
            goal_xyz, goal_rot = self.goal
            delta = np.r_[goal_xyz - xyz, (goal_rot * rotation.inv()).as_rotvec()]
        xyz = xyz + bounded_vector(delta[:3], MAX_LINEAR_SPEED * dt)
        rotation_step = bounded_vector(delta[3:], MAX_ANGULAR_SPEED * dt)
        rotation = Rotation.from_rotvec(rotation_step) * rotation
        if not self.dry_run:
            measured_xyz, measured_rot = self.measured
            xyz = measured_xyz + bounded_vector(xyz - measured_xyz, MAX_POSITION_LEAD)
            lead = bounded_vector((rotation * measured_rot.inv()).as_rotvec(), MAX_ROTATION_LEAD)
            rotation = Rotation.from_rotvec(lead) * measured_rot
        self.target = xyz, rotation
        if self.goal is not None and not self.absolute:
            translation_error = np.linalg.norm(goal_xyz - xyz)
            rotation_error = (goal_rot * rotation.inv()).magnitude()
            if translation_error < GOAL_TOLERANCE and rotation_error < GOAL_TOLERANCE:
                self.goal = None

    def _advance_joints(self, delta, values, dt, drag):
        if values is not None:
            requested = values if self.absolute else self.joint_target + values
            if self.joint_limits is not None:
                lower, upper = self.joint_limits
                if np.any(requested < lower) or np.any(requested > upper):
                    self.goal = None
                    return "입력한 각도가 시뮬레이터의 Joint Limits를 벗어납니다."
            self.goal = requested
        if drag is not None:
            if drag.joint not in range(6) or not np.isfinite(drag.angle):
                raise ValueError("Expected a selected UR5e joint and a finite ring angle.")
            self.goal = None
            step = np.zeros(6)
            step[drag.joint] = drag.angle - self.joint_target[drag.joint]
        elif self.goal is None:
            step = np.r_[JOINT_RAD_PER_METRE * delta[:3], delta[3:]]
        else:
            step = self.goal - self.joint_target
        target = self.joint_target + np.clip(step, -MAX_JOINT_SPEED * dt, MAX_JOINT_SPEED * dt)
        if not self.dry_run:
            target = np.clip(target, self.measured_joints - MAX_JOINT_LEAD, self.measured_joints + MAX_JOINT_LEAD)
        if self.joint_limits is not None:
            target = np.clip(target, *self.joint_limits)
        self.joint_target = target
        if self.goal is not None and not self.absolute:
            if np.max(np.abs(self.goal - target)) < GOAL_TOLERANCE:
                self.goal = None
