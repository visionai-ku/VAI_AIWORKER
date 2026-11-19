"""Exercise the RTDE boundary without loading a driver or opening a socket."""

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from vai_robotics.ur5e.real import RealUR5e, RobotControlStopped


class RealUR5eTests(unittest.TestCase):
    def setUp(self):
        self.receiver = MagicMock()
        self.receiver.isConnected.return_value = True
        self.receiver.getTimestamp.return_value = 1.0
        self.receiver.getActualTCPPose.return_value = [0.3, 0.1, 0.4, 0, 0, 0]
        self.receiver.getRobotMode.return_value = 7
        self.receiver.getSafetyMode.return_value = 1
        self.receiver.getRuntimeState.return_value = 1
        self.control = MagicMock()
        self.control_cls = MagicMock(return_value=self.control)
        self.enterContext(patch.dict(sys.modules, {
            "rtde_receive": SimpleNamespace(RTDEReceiveInterface=MagicMock(return_value=self.receiver)),
            "rtde_control": SimpleNamespace(RTDEControlInterface=self.control_cls),
        }))
        self.clock = self.enterContext(patch("vai_robotics.ur5e.real.time.monotonic", return_value=1.0))

    def test_dry_run_only_opens_receive(self):
        with RealUR5e("test") as robot:
            xyz, rot = robot.pose()
            np.testing.assert_allclose(xyz, [0.3, 0.1, 0.4])
            robot.command(xyz, rot)
            robot.hold()
        self.control_cls.assert_not_called()
        self.receiver.disconnect.assert_called_once()

    def test_dry_view_receives_six_joints_without_control_connection(self):
        self.receiver.getActualQ.return_value = [-1.0, -1.5, 1.2, -1.5, -1.5, 0.1]
        with RealUR5e("test") as robot:
            np.testing.assert_array_equal(robot.joints(), self.receiver.getActualQ.return_value)
            self.receiver.getActualQ.return_value = [0.0] * 5
            with self.assertRaisesRegex(RuntimeError, "joint feedback") as caught:
                robot.joints()
            self.assertNotIsInstance(caught.exception, RobotControlStopped)
        self.control_cls.assert_not_called()

    def test_joint_servo_checks_angles_feedback_and_robot_limits(self):
        measured = np.array([0.4, -1.1, 1.8, -1.4, -1.3, 0.2])
        self.receiver.getActualQ.return_value = measured
        target = measured + [0.02, 0, 0, 0, -0.01, 0]
        with RealUR5e("test") as robot:
            robot.command_joints(target)
            robot.set_motion_enabled(True)
            for invalid in ([0] * 5, [np.nan] * 6):
                with self.assertRaises(ValueError):
                    robot.command_joints(invalid)
            with self.assertRaisesRegex(RobotControlStopped, "lead"):
                robot.command_joints(target + [0, 0, 0.16, 0, 0, 0])
            self.control.isJointsWithinSafetyLimits.assert_not_called()
            self.control.isJointsWithinSafetyLimits.return_value = False
            with self.assertRaisesRegex(RobotControlStopped, "safety limits"):
                robot.command_joints(target)
            self.control.servoJ.assert_not_called()
            self.control.isJointsWithinSafetyLimits.return_value = True
            robot.command_joints(target)
            self.control.servoJ.assert_called_once_with(target.tolist(), 0.10, 0.25, 0.02, 0.10, 300)
        self.control.servoStop.assert_called_once()

    def test_cartesian_impedance_wrench_and_mode_handoff(self):
        from vai_robotics.ur5e.impedance import cartesian_wrench

        measured = Rotation.from_euler("z", 90, degrees=True)
        xyz = np.array([0.3, 0.1, 0.4])
        self.receiver.getActualTCPPose.return_value = [*xyz, *measured.as_rotvec()]
        twist = np.array([0.01, -0.03, 0, 0.02, 0, 0])
        self.receiver.getActualTCPSpeed.return_value = twist
        target = xyz + [0.01, 0, 0]
        rot = Rotation.from_rotvec([0.05, 0, 0]) * measured
        with RealUR5e("test") as robot:
            robot.command_cartesian_impedance(target, rot)
            self.control_cls.assert_not_called()
            robot.set_motion_enabled(True)
            robot.command(target, rot)
            robot.command_cartesian_impedance(target, rot)
            self.control.servoStop.assert_called_once()
            frame, axes, wrench, kind, limits = self.control.forceMode.call_args.args
            np.testing.assert_allclose(frame, [*xyz, 0, 0, 0])
            np.testing.assert_allclose(wrench, cartesian_wrench(target, rot, xyz, measured, twist), atol=1e-12)
            self.assertEqual((axes, kind, limits), ([1] * 6, 2, [0.06] * 3 + [0.25] * 3))
            robot.command_cartesian_impedance(target, rot)
            self.control.forceModeStop.assert_not_called()
            robot.command(target, rot)
            self.control.forceModeStop.assert_called_once()
        self.assertEqual(self.control.servoStop.call_count, 2)

    def test_refused_wrench_stops_force_mode_once_and_invalid_speed_never_starts_it(self):
        with RealUR5e("test", enable_motion=True) as robot:
            xyz, rot = robot.pose()
            self.receiver.getActualTCPSpeed.return_value = [np.nan] * 6
            with self.assertRaisesRegex(RobotControlStopped, "invalid TCP velocity"):
                robot.command_cartesian_impedance(xyz, rot)
            self.control.forceMode.assert_not_called()
            self.receiver.getActualTCPSpeed.return_value = [0] * 6
            self.control.forceMode.return_value = False
            with self.assertRaisesRegex(RobotControlStopped, "impedance wrench"):
                robot.command_cartesian_impedance(xyz, rot)
            robot.hold()
            robot.set_motion_enabled(False)
        self.control.forceModeStop.assert_called_once()
        self.control.servoStop.assert_not_called()

    def test_live_clutch_and_cleanup(self):
        with RealUR5e("test", enable_motion=True) as robot:
            xyz, rot = robot.pose()
            robot.command(xyz + [0.001, 0, 0], rot)
            target = [0.301, 0.1, 0.4, 0, 0, 0]
            self.control.isPoseWithinSafetyLimits.assert_called_once_with(target)
            self.control.servoL.assert_called_once_with(target, 0.10, 0.25, 0.02, 0.10, 300)
            robot.hold()
            robot.hold()
            self.clock.return_value += 0.06
            robot.pose()
            self.control.kickWatchdog.assert_called_once()
        self.control.setWatchdog.assert_called_once_with(5.0)
        self.control.servoStop.assert_called_once()
        self.control.stopScript.assert_called_once()
        self.control.disconnect.assert_called_once()

    def test_existing_program_is_preserved(self):
        for state in (2, 4):
            self.receiver.getRuntimeState.return_value = state
            with self.assertRaisesRegex(RuntimeError, "program"):
                with RealUR5e("test", enable_motion=True):
                    self.fail("unexpected motion session")
        self.control_cls.assert_not_called()

    def test_toggle_keeps_feedback_and_releases_the_writer_once(self):
        second_writer = MagicMock()
        self.control_cls.side_effect = [self.control, second_writer]
        with RealUR5e("test") as robot:
            self.assertFalse(robot.motion_enabled)
            robot.set_motion_enabled(True)
            robot.set_motion_enabled(True)
            robot.command(*robot.pose())
            self.control.servoL.assert_called_once()
            robot.set_motion_enabled(False)
            robot.set_motion_enabled(False)
            self.assertFalse(robot.motion_enabled)
            robot.command(*robot.pose())
            self.control.servoL.assert_called_once()
            self.control.servoStop.assert_called_once()
            self.control.stopScript.assert_called_once()
            self.control.disconnect.assert_called_once()
            self.receiver.disconnect.assert_not_called()
            robot.set_motion_enabled(True)
            self.assertTrue(robot.motion_enabled)
            second_writer.setWatchdog.assert_called_once()
        second_writer.disconnect.assert_called_once()
        self.receiver.disconnect.assert_called_once()

    def test_panel_enable_refusals_leave_receive_only_session(self):
        with RealUR5e("test") as robot:
            self.receiver.getSafetyMode.return_value = 3
            with self.assertRaisesRegex(RobotControlStopped, r"RUNNING\(7\), safety mode=PROTECTIVE_STOP\(3\)"):
                robot.set_motion_enabled(True)
            self.receiver.getSafetyMode.return_value = 1
            self.clock.return_value += robot.FEEDBACK_TIMEOUT + 0.01
            with self.assertRaisesRegex(RuntimeError, "stale"):
                robot.set_motion_enabled(True)
            self.control_cls.assert_not_called()
            self.receiver.getTimestamp.return_value += 0.2
            self.control.setWatchdog.return_value = False
            with self.assertRaisesRegex(RuntimeError, "watchdog"):
                robot.set_motion_enabled(True)
            self.assertFalse(robot.motion_enabled)
            self.control.stopScript.assert_called_once()
            self.control.disconnect.assert_called_once()
            robot.command(*robot.pose())
            self.control.servoL.assert_not_called()
            self.receiver.disconnect.assert_not_called()

    def test_stop_failure_detaches_writer_before_further_commands(self):
        with RealUR5e("test", enable_motion=True) as robot:
            robot.command(*robot.pose())
            self.control.servoStop.return_value = False
            with self.assertRaisesRegex(RuntimeError, "servoStop"):
                robot.set_motion_enabled(False)
            self.assertFalse(robot.motion_enabled)
            self.control.stopScript.assert_called_once()
            self.control.disconnect.assert_called_once()
            robot.command(*robot.pose())
            self.control.servoL.assert_called_once()

    def test_safety_and_lead_limits(self):
        with RealUR5e("test", enable_motion=True) as robot:
            xyz, rot = robot.pose()
            with self.assertRaisesRegex(RobotControlStopped, "lead"):
                robot.command(xyz + [0, 0, 0.04], rot)
            with self.assertRaisesRegex(RobotControlStopped, "lead"):
                robot.command(xyz, Rotation.from_rotvec([0, 0, 0.2]))
            self.control.isPoseWithinSafetyLimits.return_value = False
            with self.assertRaisesRegex(RobotControlStopped, "safety limits"):
                robot.command(xyz, rot)
            self.receiver.getSafetyMode.return_value = 3
            self.receiver.getRobotMode.return_value = 5
            self.receiver.getRuntimeState.return_value = 4
            with self.assertRaisesRegex(RobotControlStopped, r"IDLE\(5\), safety mode=PROTECTIVE_STOP\(3\), runtime_state=4"):
                robot.command(xyz, rot)
        self.control.servoL.assert_not_called()

    def test_refused_servo_stops_and_disconnects(self):
        self.control.servoL.return_value = False
        with self.assertRaisesRegex(RobotControlStopped, "servo command"):
            with RealUR5e("test", enable_motion=True) as robot:
                robot.command(*robot.pose())
        self.control.servoStop.assert_called_once()
        self.control.disconnect.assert_called_once()
        self.receiver.disconnect.assert_called_once()

    def test_stale_or_invalid_feedback_stops_servo(self):
        with self.assertRaisesRegex(RuntimeError, "stale") as caught:
            with RealUR5e("test", enable_motion=True) as robot:
                robot.command(*robot.pose())
                self.clock.return_value += 0.3
                robot.pose()
        self.assertNotIsInstance(caught.exception, RobotControlStopped)
        self.control.servoStop.assert_called_once()
        self.receiver.getTimestamp.return_value = float("nan")
        with self.assertRaisesRegex(RuntimeError, "timestamp"):
            with RealUR5e("test"):
                self.fail("accepted invalid timestamp")

    def test_script_exit_keeps_measured_pose_available_after_writer_release(self):
        with RealUR5e("test", enable_motion=True) as robot:
            self.control.isProgramRunning.return_value = False
            with self.assertRaises(RobotControlStopped):
                robot.pose()
            self.control.isProgramRunning.return_value = True  # Another program started before cleanup.
            robot.set_motion_enabled(False)
            xyz, _ = robot.pose()
            np.testing.assert_array_equal(xyz, [0.3, 0.1, 0.4])
            self.control.disconnect.assert_called_once()
            self.control.stopScript.assert_not_called()
            self.receiver.disconnect.assert_not_called()

    def test_unacknowledged_hold_attempted_once_then_script_stopped_and_feedback_available(self):
        with RealUR5e("test", enable_motion=True) as robot:
            robot.command(*robot.pose())
            self.control.servoStop.side_effect = RuntimeError("stop command failed")
            with self.assertRaisesRegex(RobotControlStopped, "servoStop.*stop command failed"):
                robot.hold()
            robot.set_motion_enabled(False)
            robot.pose()
            self.control.servoStop.assert_called_once()
            self.control.stopScript.assert_called_once()
            self.control.disconnect.assert_called_once()
            self.receiver.disconnect.assert_not_called()

    def test_negative_acknowledgement_of_script_stop_is_reported_after_disconnect(self):
        with RealUR5e("test", enable_motion=True) as robot:
            self.control.stopScript.return_value = False
            with self.assertRaisesRegex(RuntimeError, "stopScript") as caught:
                robot.set_motion_enabled(False)
            self.assertNotIsInstance(caught.exception, RobotControlStopped)
            self.assertFalse(robot.motion_enabled)
            self.control.stopScript.assert_called_once()
            self.control.disconnect.assert_called_once()

    def test_watchdog_rejection_and_target_validation_timeout_stop_control(self):
        with RealUR5e("test", enable_motion=True) as robot:
            self.clock.return_value += 0.06
            self.control.kickWatchdog.return_value = False
            with self.assertRaisesRegex(RobotControlStopped, "watchdog heartbeat"):
                robot.pose()
            robot.set_motion_enabled(False)
            robot.set_motion_enabled(True)

            def slow_validation(target):
                self.clock.return_value += robot.FEEDBACK_TIMEOUT + 0.01
                return True

            self.control.isPoseWithinSafetyLimits.side_effect = slow_validation
            with self.assertRaisesRegex(RobotControlStopped, "target validation timed out"):
                robot.command(*robot.pose())
            self.control.servoL.assert_not_called()

    def test_reader_errors_stay_fatal_during_motion_session(self):
        with RealUR5e("test", enable_motion=True) as robot:
            self.receiver.getActualTCPPose.return_value = [float("nan")] * 6
            with self.assertRaisesRegex(RuntimeError, "invalid TCP pose") as caught:
                robot.pose()
            self.assertNotIsInstance(caught.exception, RobotControlStopped)
            self.receiver.isConnected.return_value = False
            with self.assertRaisesRegex(RuntimeError, "feedback is disconnected") as caught:
                robot.pose()
            self.assertNotIsInstance(caught.exception, RobotControlStopped)

    def test_watchdog_is_required_and_disconnected_on_partial_start(self):
        self.control.setWatchdog.return_value = False
        with self.assertRaisesRegex(RuntimeError, "watchdog"):
            with RealUR5e("test", enable_motion=True):
                self.fail("unexpected motion session")
        self.control.servoL.assert_not_called()
        self.control.stopScript.assert_called_once()
        self.control.disconnect.assert_called_once()
        self.receiver.disconnect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
