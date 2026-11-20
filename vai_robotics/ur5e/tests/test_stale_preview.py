import unittest
from unittest.mock import Mock, call

import numpy as np
from scipy.spatial.transform import Rotation

from vai_robotics.ur5e.real import RealUR5e, RobotControlStopped, StaleRobotFeedback
from vai_robotics.ur5e.teleop_session import TeleopSession
from vai_robotics.ur5e.teleop_types import InputSample


class StalePreviewTests(unittest.TestCase):
    def setUp(self):
        self.now = 1.
        self.backend = Mock()
        self.backend.pose.return_value = self.pose = (np.zeros(3), Rotation.identity())
        self.device = Mock(spec=["read"])
        self.panel = Mock(is_open=True, scale=1., open_width_mm=75.)
        for method in ("poll", "take_command_mode", "take_control_law", "take_target", "take_impedance_gains"):
            getattr(self.panel, method).return_value = None
        self.gripper = Mock(open_width_mm=75., width_mm=50., status="RG2")
        self.view = Mock()
        self.backend.payload.return_value = None
        self.backend.tcp_speed_feedback.return_value = None
        self.backend.joints.return_value = np.zeros(6)
        self.stale = StaleRobotFeedback("UR5e feedback is stale.")
        self.session = TeleopSession(self.device, self.backend, simulation=False, dry_run=True,
                                    panel=self.panel, gripper=self.gripper, gripper_force_n=40.,
                                    command_mode="eef-delta", control_law="position",
                                    monotonic=lambda: self.now)

    def tick(self, *, held=False, run_request=None):
        self.now += .02
        self.device.read.return_value = InputSample.stationary(enabled=held)
        self.panel.poll.return_value = run_request
        return self.session.step(self.view, self.now)

    def test_frozen_dry_disarms_and_ignores_requests_through_first_fresh_frame(self):
        self.tick()
        self.tick(held=True)
        self.assertTrue(self.session.controller.active)
        self.view.update.reset_mock()
        self.backend.pose.side_effect = self.stale
        self.tick(held=True)
        self.tick(run_request=False)
        self.assertFalse(self.session.controller.active)
        self.assertIsNone(self.panel.refresh.call_args.kwargs["measured"])
        self.assertEqual(self.panel.refresh.call_args.kwargs["operator_state"], "waiting-feedback")
        self.view.update.assert_not_called()
        self.panel.mode_result.assert_called_once()
        self.assertTrue(self.panel.mode_result.call_args.kwargs["dry_run"])

        self.backend.pose.side_effect = None
        self.backend.pose.return_value = (np.ones(3), Rotation.identity())
        self.tick(run_request=False)  # Even released input cannot execute a queued REAL request.
        np.testing.assert_array_equal(self.view.update.call_args.kwargs["measured_tcp"][0], np.ones(3))
        self.tick(held=True)
        self.assertFalse(self.session.controller.active)
        self.tick()
        self.tick(held=True)
        self.assertTrue(self.session.controller.active)
        self.assertEqual(set(c.args for c in self.gripper.update.call_args_list), {(None,)})
        self.backend.set_motion_enabled.assert_not_called()
        self.backend.command.assert_not_called()

    def test_stale_while_live_detaches_writer_and_only_resumes_in_dry_run(self):
        self.session.controller.dry_run = False
        self.backend.pose.side_effect = self.stale
        self.assertTrue(self.tick())
        self.assertTrue(self.session.controller.dry_run)
        self.backend.set_motion_enabled.assert_called_once_with(False)
        self.backend.pose.side_effect = None
        self.backend.pose.return_value = self.pose
        self.tick(run_request=False)
        self.assertTrue(self.session.controller.dry_run)
        self.backend.set_motion_enabled.assert_called_once_with(False)

    def test_stale_after_writer_stop_pauses_until_fresh(self):
        self.session.controller.dry_run = False
        self.backend.pose.side_effect = [RobotControlStopped("writer rejected target"), self.stale, self.pose]
        self.assertTrue(self.tick())
        self.assertTrue(self.session.controller.dry_run)
        self.assertEqual(self.backend.set_motion_enabled.call_args_list, [call(False)])
        self.assertTrue(self.tick())

    def test_disconnected_receiver_and_backwards_clock_remain_fatal_while_paused(self):
        self.backend.pose.side_effect = self.stale
        self.tick()
        for reason in ("feedback is disconnected", "controller clock restarted"):
            self.backend.pose.side_effect = RuntimeError(reason)
            with self.assertRaisesRegex(RuntimeError, reason):
                self.tick()

    def test_only_unchanged_timestamps_raise_transient_exception(self):
        robot = RealUR5e("test")
        robot._check_feedback_timestamp(10., 3.)
        with self.assertRaises(StaleRobotFeedback):
            robot._check_feedback_timestamp(10., 4.)
        for stamp in (float("nan"), 9.):
            with self.assertRaises(RuntimeError) as caught:
                robot._check_feedback_timestamp(stamp, 4.)
            self.assertNotIsInstance(caught.exception, StaleRobotFeedback)
