"""UR5e base-frame TCP teleoperation over ur_rtde. The default session is receive-only."""

from contextlib import ExitStack
import time

import numpy as np
from scipy.spatial.transform import Rotation


class RealUR5e:
    """One control loop owns this session and calls pose() even while idle."""

    WATCHDOG_HZ = 5.0
    FEEDBACK_TIMEOUT = 0.15
    MAX_POSITION_LEAD = 0.03  # m, relative to the measured TCP
    MAX_ANGLE_LEAD = 0.15  # rad, relative to the measured TCP
    RTDE_FIELDS = ("timestamp", "actual_TCP_pose", "actual_q", "runtime_state", "robot_mode", "safety_mode")

    def __init__(self, host, *, enable_motion=False, frequency=50.0):
        if not host or not host.strip():
            raise ValueError("Provide the UR5e IP address or hostname.")
        if not np.isfinite(frequency) or not 10 <= frequency <= 500:
            raise ValueError("RTDE teleoperation frequency must be between 10 and 500 Hz.")
        self.host = host
        self._start_with_motion = enable_motion
        self.period = 1.0 / frequency
        self._receive = None
        self._control = None
        self._servo_active = False
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

    def set_motion_enabled(self, enabled):
        """Acquire or release the motion writer while the feedback stream stays open."""
        if enabled == self.motion_enabled:
            return
        if not enabled:
            try:
                with ExitStack() as cleanup:
                    cleanup.callback(self._control.disconnect)
                    cleanup.callback(self._stop_owned_script)
                    self.hold()
            finally:
                self._control = None
                self._servo_active = False
            return

        self.pose()
        self._check_safety_mode()
        # ur_rtde uploads its own control script and can stop a running one.
        if self._receive.getRuntimeState() != 1:  # STOPPED
            raise RuntimeError("The robot already has an active or paused program.")
        from rtde_control import RTDEControlInterface

        self._control = RTDEControlInterface(self.host)
        try:
            if not self._control.setWatchdog(self.WATCHDOG_HZ):
                raise RuntimeError("UR5e did not enable the RTDE command watchdog.")
            self._heartbeat_at = time.monotonic()
            self.pose()  # Check fresh feedback and script status after the upload.
        except BaseException:
            self.set_motion_enabled(False)
            raise

    def _check_safety_mode(self):
        # UR robot mode 7 is RUNNING; safety modes 1 and 2 are NORMAL and REDUCED.
        if self._receive.getRobotMode() != 7 or self._receive.getSafetyMode() not in (1, 2):
            raise RuntimeError("UR5e must be running in normal or reduced safety mode.")

    def pose(self):
        """Return measured TCP position in metres and orientation relative to the UR base."""
        receiver = self._receive
        if receiver is None or not receiver.isConnected():
            raise RuntimeError("UR5e RTDE feedback is disconnected.")
        now = time.monotonic()
        timestamp = receiver.getTimestamp()
        if not np.isfinite(timestamp) or timestamp <= 0:
            raise RuntimeError("UR5e feedback has an invalid timestamp.")
        if self._timestamp is not None and timestamp < self._timestamp:
            raise RuntimeError("UR5e controller clock restarted.")
        if timestamp != self._timestamp:
            self._timestamp, self._updated_at = timestamp, now
        if now - self._updated_at >= self.FEEDBACK_TIMEOUT:
            raise RuntimeError("UR5e feedback is stale.")
        raw = np.asarray(receiver.getActualTCPPose(), dtype=float)
        if raw.shape != (6,) or not np.all(np.isfinite(raw)):
            raise RuntimeError("UR5e returned an invalid TCP pose.")
        if self._control is not None:
            self._check_safety_mode()
            if not self._control.isConnected() or not self._control.isProgramRunning():
                raise RuntimeError("The UR5e teleoperation control script is no longer running.")
            # Active servo commands themselves update the watchdog register. An
            # idle receive loop must keep it alive after servoStop or before a grip.
            if not self._servo_active and now - self._heartbeat_at >= 0.05:
                if not self._control.kickWatchdog():
                    raise RuntimeError("UR5e rejected the watchdog heartbeat.")
                self._heartbeat_at = now
        return raw[:3], Rotation.from_rotvec(raw[3:])

    def command(self, xyz, rotation):
        """Apply a small absolute TCP target. Dry-run validates it using measured feedback."""
        measured_xyz, measured_rotation = self.pose()
        xyz = np.asarray(xyz, dtype=float)
        rotvec = rotation.as_rotvec()
        if (xyz.shape != (3,) or rotvec.shape != (3,) or
                not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(rotvec))):
            raise ValueError("A TCP target needs three finite position and rotation coordinates.")
        if (np.linalg.norm(xyz - measured_xyz) > self.MAX_POSITION_LEAD or
                (rotation * measured_rotation.inv()).magnitude() > self.MAX_ANGLE_LEAD):
            raise ValueError("TCP target exceeds the allowed lead from the measured pose.")
        if self._control is None:
            return
        target = np.concatenate((xyz, rotvec)).tolist()
        checked_at = time.monotonic()
        if not self._control.isPoseWithinSafetyLimits(target):
            raise RuntimeError("Target exceeds the robot's configured safety limits or IK reach.")
        if time.monotonic() - checked_at >= self.FEEDBACK_TIMEOUT:
            raise RuntimeError("UR5e target validation timed out.")
        self._servo_active = True  # A failed acknowledgement may follow a transmitted command.
        if not self._control.servoL(target, 0.10, 0.25, self.period, 0.10, 300):
            raise RuntimeError("UR5e rejected the Cartesian servo command.")

    def hold(self):
        """Stop servoing once on clutch release; remain in the current physical pose."""
        if self._servo_active:
            if self._control.isConnected() and self._control.isProgramRunning():
                if not self._control.servoStop():
                    raise RuntimeError("UR5e did not acknowledge servoStop.")
            self._servo_active = False

    def _stop_owned_script(self):
        if self._control.isConnected() and self._control.isProgramRunning():
            self._control.stopScript()

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
