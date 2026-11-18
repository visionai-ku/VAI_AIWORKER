import unittest
from unittest.mock import Mock, call, patch

import numpy as np
from scipy.spatial.transform import Rotation

from vai_robotics.ur5e.real import RobotControlStopped
from vai_robotics.ur5e.teleop import run_teleop
from vai_robotics.ur5e.teleop_types import InputSample


class RealViewerSessionTest(unittest.TestCase):
    def setUp(self):
        self.backend = Mock()
        self.device = Mock(spec=["read"])
        self.panel = Mock(scale=1.0, is_open=True)
        self.released = InputSample(np.zeros(3), np.zeros(3), False)
        self.held = InputSample(np.array([0.001, 0, 0]), np.zeros(3), True)
        self.frame = -1

        self.viewer_type = self.enterContext(patch("vai_robotics.ur5e.real_viewer.RealViewer"))
        self.viewer = self.viewer_type.return_value.__enter__.return_value
        self.viewer.is_running.return_value = True
        clock = self.enterContext(patch("vai_robotics.ur5e.teleop.time"))
        clock.monotonic.side_effect = lambda: self.frame / 50
        self.backend.pose.side_effect = lambda: (np.array([0.4, self.frame / 1000, 0.3]), Rotation.identity())
        self.backend.joints.side_effect = lambda: np.arange(6) + self.frame / 100

    def run_frames(self, *, dry_run, requests, samples):
        def poll():
            self.frame += 1
            self.panel.is_open = self.frame < len(requests)
            if self.panel.is_open:
                return requests[self.frame]

        self.panel.poll.side_effect = poll
        self.device.read.side_effect = lambda: samples[self.frame]
        run_teleop(self.device, self.backend, dry_run=dry_run, panel=self.panel)
        self.viewer_type.assert_called_once()
        self.viewer_type.return_value.__exit__.assert_called_once()
        updates = [entry.kwargs for entry in self.viewer.update.call_args_list]
        self.assertEqual(len(updates), len(requests))
        for frame, update in enumerate(updates):
            np.testing.assert_array_equal(update["q"], np.arange(6) + frame / 100)
            np.testing.assert_array_equal(update["measured_tcp"][0], [0.4, frame / 1000, 0.3])
        return updates

    def test_one_viewer_previews_in_dry_and_tracks_feedback_through_real_and_back(self):
        updates = self.run_frames(
            dry_run=True,
            requests=[None, None, False, None, None, True],
            samples=[self.released, self.held, self.released, self.released, self.held, self.released],
        )
        self.assertEqual([state["dry_run"] for state in updates], [True, True, False, False, False, True])
        self.assertIsNotNone(updates[1]["target"])
        self.assertTrue(all(state["target"] is None for state in updates[2:]))
        self.backend.set_motion_enabled.assert_has_calls([call(True), call(False)])
        self.backend.command.assert_called_once()

    def test_direct_real_start_opens_measured_mirror(self):
        shutdown = []
        self.backend.set_motion_enabled.side_effect = lambda enabled: shutdown.append(("motion", enabled))
        self.viewer_type.return_value.__exit__.side_effect = lambda *exc: shutdown.append("viewer exit")
        updates = self.run_frames(dry_run=False, requests=[None, None], samples=[self.released, self.held])
        for state in updates:
            self.assertFalse(state["dry_run"])
            self.assertIsNone(state["target"])
        self.backend.command.assert_called_once()
        self.assertEqual(shutdown, [("motion", False), "viewer exit"])

    def test_stopped_control_script_with_healthy_feedback_keeps_same_viewer_in_dry(self):
        measured_pose = self.backend.pose.side_effect
        stopped = False

        def pose():
            nonlocal stopped
            if self.frame == 2 and not stopped:
                stopped = True
                raise RobotControlStopped("UR5e control script stopped")
            return measured_pose()

        self.backend.pose.side_effect = pose
        updates = self.run_frames(dry_run=False, requests=[None] * 4,
                                  samples=[self.released, self.held, self.held, self.held])
        self.assertEqual([frame["dry_run"] for frame in updates], [False, False, True, True])
        self.assertTrue(all(frame["target"] is None for frame in updates))
        self.assertTrue(self.panel.refresh.call_args_list[2].kwargs["message"])
        self.backend.command.assert_called_once()
        self.assertNotIn(call(True), self.backend.set_motion_enabled.call_args_list)

    def test_unavailable_pose_feedback_ends_real_session(self):
        for errors in ([RuntimeError("feedback is stale")],
                       [RobotControlStopped("script stopped"), RuntimeError("feedback is stale")]):
            with self.subTest(errors=errors):
                self.panel.poll.return_value = None
                self.device.read.return_value = self.released
                self.backend.pose.side_effect = errors
                with self.assertRaisesRegex(RuntimeError, "feedback is stale"):
                    run_teleop(self.device, self.backend, dry_run=False, panel=self.panel)
                self.backend.command.assert_not_called()
                self.viewer.update.assert_not_called()

    def test_no_panel_skips_viewer_and_joint_reads(self):
        self.device.read.side_effect = [self.released, KeyboardInterrupt()]
        with self.assertRaises(KeyboardInterrupt):
            run_teleop(self.device, self.backend, dry_run=False, panel=None)
        self.viewer_type.assert_not_called()
        self.backend.pose.assert_called_once()
        self.backend.joints.assert_not_called()
