"""Clutched EEF and joint commands shared by the UR5e teleoperation backends."""

import numpy as np
from scipy.spatial.transform import Rotation


HZ = 50
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
    """Anchor clutched targets to feedback and dispatch one control law per session."""

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

    def _reset_targets(self):
        self.ready_after_release = False
        self.target = self.joint_target = self.goal = self.last_tick = None

    def set_command_mode(self, command_mode, *, simulation, sample, device=None):
        if command_mode not in COMMAND_MODES:
            return "지원하지 않는 명령 모드입니다."
        if command_mode == self.command_mode:
            return ""
        if sample is None:
            if (self.active or not (simulation or self.dry_run) or
                    getattr(device, "clutch_pressed", None) is True or
                    getattr(device, "gripper_chord", None) is True):
                return "입력 장치 연결을 확인하고 클러치를 놓은 뒤 명령 모드를 다시 선택하세요."
        elif sample.enabled or not sample.buttons_released:
            return "클러치와 그리퍼 버튼을 놓은 뒤 명령 모드를 다시 선택하세요."

        joint_space = command_mode.startswith("joint-")
        if not simulation and self.control_law == "impedance" and self.joint_space != joint_space:
            # The TCP and joint torque loops use different robot-side programs.
            if not self.dry_run:
                return "실기 임피던스 제어에서 TCP와 관절을 바꾸려면 DRY RUN으로 전환한 뒤 선택하세요."
            self.backend.set_joint_impedance_writer(joint_space)
        self.backend.hold()
        self._reset_targets()
        self.measured = self.measured_joints = None
        self.command_mode = command_mode
        return ""

    def set_mode(self, dry_run, *, simulation, sample, device=None):
        if dry_run == self.dry_run:
            return ""
        if not dry_run and not simulation and (sample is None or sample.enabled or not sample.buttons_released):
            if hasattr(device, "clutch_pressed"):
                if device.status:
                    return "SpaceMouse를 USB에 연결한 뒤 다시 시도하세요."
                if device.clutch_pressed is None:
                    return f"SpaceMouse {device.button}번 버튼을 한 번 눌렀다 놓은 뒤 실기 구동을 다시 선택하세요."
                return "SpaceMouse의 모든 버튼을 놓은 뒤 실기 구동을 다시 선택하세요."
            return "컨트롤러의 추적 상태를 확인하고 클러치를 놓은 뒤 실기 구동을 다시 선택하세요."
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
                return f"실기 제어 연결 해제 · 정지 응답을 확인하지 못했습니다. 펜던트를 확인하세요. ({error})"
        self.dry_run = dry_run
        return ""

    @property
    def active(self):
        return self.target is not None or self.joint_target is not None

    def _advance_eef(self, delta, values, dt):
        xyz, rotation = self.target
        if values is not None:
            requested = Rotation.from_rotvec(values[3:])
            self.goal = ((values[:3], requested) if self.absolute else
                         (xyz + values[:3], requested * rotation))
        if self.goal is not None:
            goal_xyz, goal_rot = self.goal
            delta = np.r_[goal_xyz - xyz, (goal_rot * rotation.inv()).as_rotvec()]
        xyz = xyz + bounded_vector(delta[:3], MAX_LINEAR_SPEED * dt)
        rotation = Rotation.from_rotvec(bounded_vector(delta[3:], MAX_ANGULAR_SPEED * dt)) * rotation
        if not self.dry_run:
            measured_xyz, measured_rot = self.measured
            xyz = measured_xyz + bounded_vector(xyz - measured_xyz, MAX_POSITION_LEAD)
            lead = bounded_vector((rotation * measured_rot.inv()).as_rotvec(), MAX_ROTATION_LEAD)
            rotation = Rotation.from_rotvec(lead) * measured_rot
        self.target = xyz, rotation
        if self.goal is not None and not self.absolute:
            if np.linalg.norm(goal_xyz - xyz) < 1e-8 and (goal_rot * rotation.inv()).magnitude() < 1e-8:
                self.goal = None

    def _advance_joints(self, delta, values, dt):
        if values is not None:
            requested = values if self.absolute else self.joint_target + values
            if self.joint_limits is not None:
                lower, upper = self.joint_limits
                if np.any(requested < lower) or np.any(requested > upper):
                    self.goal = None
                    return "목표가 시뮬레이터의 관절 범위를 벗어납니다."
            self.goal = requested
        if self.goal is None:
            step = np.r_[JOINT_RAD_PER_METRE * delta[:3], delta[3:]]
        else:
            step = self.goal - self.joint_target
        target = self.joint_target + np.clip(step, -MAX_JOINT_SPEED * dt, MAX_JOINT_SPEED * dt)
        if not self.dry_run:
            target = np.clip(target, self.measured_joints - MAX_JOINT_LEAD, self.measured_joints + MAX_JOINT_LEAD)
        if self.joint_limits is not None:
            target = np.clip(target, *self.joint_limits)
        self.joint_target = target
        if self.goal is not None and not self.absolute and np.max(np.abs(self.goal - target)) < 1e-8:
            self.goal = None

    def update(self, sample, now, *, scale=1.0, values=None):
        """Apply one device tick plus an optional one-shot six-axis numerical command (SI)."""
        dt = 0.0 if self.last_tick is None else now - self.last_tick
        self.last_tick = now
        self.measured = self.backend.pose()
        if self.joint_space:
            self.measured_joints = self.backend.joints()
        if sample is None or not 0.0 <= dt <= 0.12:
            self.ready_after_release = False
        elif not sample.enabled:
            self.ready_after_release = True
        elif self.ready_after_release:
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
            dt = min(dt, 2 / HZ)
            delta = np.zeros(6)
            if not starting and not self.absolute:
                delta = scale * np.r_[bounded_vector(sample.translation, MAX_LINEAR_SPEED * dt),
                                      bounded_vector(sample.rotation, MAX_ANGULAR_SPEED * dt)]
            if self.joint_space:
                message = self._advance_joints(delta, values, dt)
                if not self.dry_run:
                    if self.control_law == "impedance":
                        self.backend.command_joint_impedance(self.joint_target)
                    else:
                        self.backend.command_joints(self.joint_target)
                return message
            if not starting or values is not None:
                self._advance_eef(delta, values, dt)
            if not self.dry_run:
                if self.control_law == "impedance":
                    self.backend.command_cartesian_impedance(*self.target)
                else:
                    self.backend.command(*self.target)
            return
        self.target = self.joint_target = self.goal = None
        self.backend.hold()
        if values is not None:
            return "클러치를 놓았다 다시 누른 상태에서 목표 적용을 누르세요."
