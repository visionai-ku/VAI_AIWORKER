"""UR5e base-frame TCP teleoperation over ur_rtde. The default session is receive-only."""

from contextlib import ExitStack
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .impedance import TCP_SPEED_LIMITS, ImpedanceSettings, cartesian_wrench


# RTDE getRobotMode()/getSafetyMode() return UR's numeric modes.
_ROBOT_MODES = {
    -1: "NO_CONTROLLER", 0: "DISCONNECTED", 1: "CONFIRM_SAFETY", 2: "BOOTING",
    3: "POWER_OFF", 4: "POWER_ON", 5: "IDLE", 6: "BACKDRIVE", 7: "RUNNING", 8: "UPDATING_FIRMWARE",
}
_SAFETY_MODES = {
    1: "NORMAL", 2: "REDUCED", 3: "PROTECTIVE_STOP", 4: "RECOVERY", 5: "SAFEGUARD_STOP",
    6: "SYSTEM_EMERGENCY_STOP", 7: "ROBOT_EMERGENCY_STOP", 8: "VIOLATION", 9: "FAULT",
    10: "VALIDATE_JOINT_ID", 11: "UNDEFINED_SAFETY_MODE",
}


class RobotControlStopped(RuntimeError):
    """The motion writer has stopped while the RTDE reader can still be healthy."""


class RealUR5e:
    """One control loop owns this session and calls pose() even while idle."""

    WATCHDOG_HZ = 5.0
    FEEDBACK_TIMEOUT = 0.15
    MAX_POSITION_LEAD = 0.03  # m, relative to the measured TCP
    MAX_ANGLE_LEAD = 0.15  # rad, relative to the measured TCP
    MAX_JOINT_LEAD = 0.15  # rad per joint
    FORCE_SPEED_LIMITS = TCP_SPEED_LIMITS
    RTDE_FIELDS = ("timestamp", "actual_TCP_pose", "actual_TCP_speed", "actual_q",
                   "runtime_state", "robot_mode", "safety_mode")

    def __init__(self, host, *, enable_motion=False, frequency=50.0, joint_impedance=False):
        if not host or not host.strip():
            raise ValueError("Provide the UR5e IP address or hostname.")
        if not np.isfinite(frequency) or not 10 <= frequency <= 500:
            raise ValueError("RTDE teleoperation frequency must be between 10 and 500 Hz.")
        self.host = host
        self._start_with_motion = enable_motion
        self._joint_impedance = joint_impedance
        self._gains = ImpedanceSettings()
        self.period = 1.0 / frequency
        self._receive = None
        self._control = None
        self._lost_script = False
        self._active_mode = None
        self._timestamp = None
        self._updated_at = 0.0
        self._heartbeat_at = 0.0

    def __enter__(self):
        if self._receive is not None:
            raise RuntimeError("This RTDE session is already open.")
        try:
            from rtde_receive import RTDEReceiveInterface
        except ImportError as exc:
            raise RuntimeError("Install ur-rtde in the teleoperation Python environment.") from exc

        try:
            self._receive = RTDEReceiveInterface(self.host, -1.0, list(self.RTDE_FIELDS))
            self.pose()
            if self._start_with_motion:
                self.set_motion_enabled(True)
            return self
        except BaseException:
            self.close()
            raise

    @property
    def motion_enabled(self):
        return self._control is not None

    def joints(self):
        """Measured shoulder-to-wrist joint angles in the controller's native order."""
        q = np.asarray(self._receive.getActualQ(), dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise RuntimeError("UR5e returned invalid joint feedback.")
        return q

    def set_joint_impedance_writer(self, required):
        """Select the program for the next REAL RUN without disturbing the RTDE reader."""
        if self.motion_enabled and required != self._joint_impedance:
            raise RuntimeError("Switch to DRY RUN before changing the robot-side impedance program.")
        self._joint_impedance = required

    def get_impedance_gains(self, space):
        return self._gains.snapshot(space)

    def set_impedance_gains(self, space, gains):
        if (space, self._active_mode) in (("eef", "force"), ("joint", "joint_impedance")):
            raise ValueError("Release the clutch before changing active impedance gains.")
        self._gains.set(space, gains)

    def set_motion_enabled(self, enabled):
        """Acquire or release the motion writer while the feedback stream stays open."""
        if enabled == self.motion_enabled:
            return
        if enabled:
            self._acquire_motion_writer()
        else:
            self._release_motion_writer()

    def _acquire_motion_writer(self):
        self.pose()
        self._check_safety_mode()
        # ur_rtde uploads its own control script and can stop a running one.
        if self._receive.getRuntimeState() != 1:  # STOPPED
            raise RuntimeError("The robot already has an active or paused program.")
        if self._joint_impedance:
            from .joint_writer import JointImpedanceWriter

            self._control = JointImpedanceWriter.open(self.host, self._receive, self.MAX_JOINT_LEAD)
        else:
            from rtde_control import RTDEControlInterface

            self._control = RTDEControlInterface(self.host)
        self._lost_script = False
        try:
            if not self._joint_impedance:
                self._require_command(self._control.setWatchdog, self.WATCHDOG_HZ,
                                      reason="UR5e did not enable the RTDE command watchdog.")
            self._heartbeat_at = time.monotonic()
            self.pose()  # Check fresh feedback and script status after the upload.
        except BaseException:
            self._release_motion_writer()
            raise

    def _release_motion_writer(self):
        try:
            with ExitStack() as cleanup:
                cleanup.callback(self._control.disconnect)
                cleanup.callback(self._stop_owned_script)
                self.hold()
        finally:
            self._control = None
            self._active_mode = None

    def _motion_denied(self, reason, modes=None):
        if modes is None:
            modes = self._receive.getRobotMode(), self._receive.getSafetyMode()
        robot, safety = modes
        runtime = self._receive.getRuntimeState()
        return RobotControlStopped(
            f"{reason} robot mode={_ROBOT_MODES.get(robot, 'UNKNOWN')}({robot}), "
            f"safety mode={_SAFETY_MODES.get(safety, 'UNKNOWN')}({safety}), runtime_state={runtime}"
        )

    def _check_safety_mode(self):
        robot, safety = self._receive.getRobotMode(), self._receive.getSafetyMode()
        if robot != 7 or safety not in (1, 2):
            raise self._motion_denied("UR5e must be running in normal or reduced safety mode.", (robot, safety))

    def _require_command(self, operation, *args, reason):
        try:
            accepted = operation(*args)
        except (RuntimeError, OSError) as error:
            raise self._motion_denied(f"{reason} {error}") from error
        if not accepted:
            raise self._motion_denied(reason)

    def _owned_program_running(self):
        if self._lost_script:
            return False
        try:
            running = self._control.isConnected() and self._control.isProgramRunning()
        except (RuntimeError, OSError) as error:
            raise self._motion_denied(f"UR5e writer status unavailable: {error}") from error
        if not running:
            # A program started after we saw our script stop belongs to someone else.
            self._lost_script = True
        return running

    def pose(self):
        """Return measured TCP position in metres and orientation relative to the UR base."""
        receiver = self._receive
        if receiver is None or not receiver.isConnected():
            raise RuntimeError("UR5e RTDE feedback is disconnected.")
        now = time.monotonic()
        self._check_feedback_timestamp(receiver.getTimestamp(), now)
        raw = np.asarray(receiver.getActualTCPPose(), dtype=float)
        if raw.shape != (6,) or not np.all(np.isfinite(raw)):
            raise RuntimeError("UR5e returned an invalid TCP pose.")
        if self.motion_enabled:
            self._service_motion_writer(now)
        return raw[:3], Rotation.from_rotvec(raw[3:])

    def _check_feedback_timestamp(self, timestamp, now):
        if not np.isfinite(timestamp) or timestamp <= 0:
            raise RuntimeError("UR5e feedback has an invalid timestamp.")
        if self._timestamp is not None and timestamp < self._timestamp:
            raise RuntimeError("UR5e controller clock restarted.")
        if timestamp != self._timestamp:
            self._timestamp, self._updated_at = timestamp, now
        if now - self._updated_at >= self.FEEDBACK_TIMEOUT:
            raise RuntimeError("UR5e feedback is stale.")

    def _service_motion_writer(self, now):
        running = self._owned_program_running()
        self._check_safety_mode()
        if not running:
            raise self._motion_denied("UR5e teleoperation script stopped or its writer disconnected.")
        # Active servo commands refresh the watchdog. While idle, the feedback loop does it.
        if self._active_mode is None and now - self._heartbeat_at >= 0.05:
            self._require_command(self._control.kickWatchdog, reason="UR5e rejected the watchdog heartbeat.")
            self._heartbeat_at = now

    def _stop_other_mode(self, next_mode):
        if self._joint_impedance != (next_mode == "joint_impedance"):
            raise ValueError("The selected robot-side writer does not support this control law.")
        if self._active_mode is not None and self._active_mode != next_mode:
            self.hold()

    def _check_lead(self, exceeds_limit, reason):
        if exceeds_limit:
            if self.motion_enabled:
                raise self._motion_denied(reason)
            raise ValueError(reason)

    def _check_target(self, operation, target, reason):
        checked_at = time.monotonic()
        self._require_command(operation, target, reason=reason)
        if time.monotonic() - checked_at >= self.FEEDBACK_TIMEOUT:
            raise self._motion_denied("UR5e target validation timed out.")

    def _checked_tcp(self, xyz, rotation):
        measured_xyz, measured_rotation = self.pose()
        xyz = np.asarray(xyz, dtype=float)
        rotvec = rotation.as_rotvec()
        if (xyz.shape != (3,) or rotvec.shape != (3,) or
                not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(rotvec))):
            raise ValueError("A TCP target needs three finite position and rotation coordinates.")
        self._check_lead(np.linalg.norm(xyz - measured_xyz) > self.MAX_POSITION_LEAD or
                         (rotation * measured_rotation.inv()).magnitude() > self.MAX_ANGLE_LEAD,
                         "TCP target exceeds the allowed lead from the measured pose.")
        target = np.concatenate((xyz, rotvec)).tolist()
        if self.motion_enabled:
            self._check_target(self._control.isPoseWithinSafetyLimits, target,
                               "Target exceeds the robot's configured safety limits or IK reach.")
        return target, measured_xyz, measured_rotation

    def command(self, xyz, rotation):
        """Apply a small absolute TCP target. Dry-run validates it using measured feedback."""
        self._stop_other_mode("servo")
        target, _, _ = self._checked_tcp(xyz, rotation)
        if self.motion_enabled:
            self._active_mode = "servo"  # A failed acknowledgement may follow a transmitted command.
            self._require_command(self._control.servoL, target, 0.10, 0.25, self.period, 0.10, 300,
                                  reason="UR5e rejected the Cartesian servo command.")

    def _checked_joints(self, q):
        self.pose()  # Fresh RTDE data, safety state and script ownership.
        measured = self.joints()
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("A joint target needs six finite angles in radians.")
        self._check_lead(np.max(np.abs(q - measured)) > self.MAX_JOINT_LEAD,
                         "Joint target exceeds the allowed lead from the measured joints.")
        return q.tolist()

    def command_joints(self, q):
        """Apply six absolute joint angles in native order, in radians."""
        self._stop_other_mode("servo")
        target = self._checked_joints(q)
        if self.motion_enabled:
            self._check_target(self._control.isJointsWithinSafetyLimits, target,
                               "Joint target exceeds the robot's configured safety limits.")
            self._active_mode = "servo"
            self._require_command(self._control.servoJ, target, 0.10, 0.25, self.period, 0.10, 300,
                                  reason="UR5e rejected the joint servo command.")

    def command_joint_impedance(self, q):
        """Update the equilibrium of the robot-side 500 Hz joint torque loop."""
        self._stop_other_mode("joint_impedance")
        target = self._checked_joints(q)
        if self.motion_enabled:
            self._active_mode = "joint_impedance"
            self._require_command(self._control.command, target, self._gains.coefficients("joint"),
                                  reason="Joint impedance target rejected or acknowledgement lost.")

    def command_cartesian_impedance(self, xyz, rotation):
        """Drive a bounded base-frame spring and damper at the measured TCP."""
        self._stop_other_mode("force")
        _, measured_xyz, measured_rotation = self._checked_tcp(xyz, rotation)
        if self.motion_enabled:
            twist = np.asarray(self._receive.getActualTCPSpeed(), dtype=float)
            if twist.shape != (6,) or not np.isfinite(twist).all():
                raise self._motion_denied("UR5e returned invalid TCP velocity feedback.")
            wrench = cartesian_wrench(xyz, rotation, measured_xyz, measured_rotation, twist,
                                      **self._gains.coefficients("eef"))
            # Type 2 keeps the axes base-aligned. Locate the origin at the TCP
            # because the wrench's moment is computed about that point.
            task_frame = [*measured_xyz.tolist(), 0.0, 0.0, 0.0]
            self._active_mode = "force"
            self._require_command(self._control.forceMode, task_frame, [1] * 6,
                                  wrench.tolist(), 2, list(self.FORCE_SPEED_LIMITS),
                                  reason="UR5e rejected the Cartesian impedance wrench.")

    def hold(self):
        """Stop servoing once on clutch release; remain in the current physical pose."""
        mode, self._active_mode = self._active_mode, None
        if mode is not None and self._owned_program_running():
            if mode == "force":
                self._require_command(self._control.forceModeStop, reason="UR5e did not acknowledge forceModeStop.")
            elif mode == "joint_impedance":
                self._require_command(self._control.hold, reason="UR5e did not acknowledge joint impedance stop.")
            else:
                self._require_command(self._control.servoStop, reason="UR5e did not acknowledge servoStop.")

    def _stop_owned_script(self):
        if self._owned_program_running() and self._control.stopScript() is False:
            # ur_rtde 1.6 returns None on success; report an explicit negative acknowledgement.
            raise RuntimeError("UR5e did not acknowledge stopScript.")

    def close(self):
        try:
            self.set_motion_enabled(False)
        finally:
            receiver, self._receive = self._receive, None
            self._timestamp = None
            if receiver is not None:
                receiver.disconnect()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
