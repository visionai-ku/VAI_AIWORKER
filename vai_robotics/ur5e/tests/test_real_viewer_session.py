import unittest
from dataclasses import replace
from unittest.mock import Mock, PropertyMock, call, patch

import numpy as np
from scipy.spatial.transform import Rotation

from vai_robotics.ur5e.real import RobotControlStopped
from vai_robotics.ur5e.teleop import run_teleop
from vai_robotics.ur5e.teleop_types import InputSample


class RealViewerSessionTest(unittest.TestCase):
    def setUp(self):
        self.backend = Mock()
        self.device = Mock(spec=["read"])
        self.panel = Mock(scale=1.0, is_open=True, open_width_mm=75.0)
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

    def run_frames(self, *, dry_run, requests, samples, gripper=None, simulation=False):
        def poll():
            self.frame += 1
            self.panel.is_open = self.frame < len(requests)
            if self.panel.is_open:
                return requests[self.frame]

        self.panel.poll.side_effect = poll
        self.device.read.side_effect = lambda: samples[self.frame]
        run_teleop(self.device, self.backend, dry_run=dry_run, panel=self.panel, gripper=gripper, sim=simulation)
        if simulation:
            self.viewer_type.assert_not_called()
            return [call.kwargs for call in self.backend.tick.call_args_list]
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

    def test_real_run_rejects_unconfirmed_or_held_spacemouse_until_explicit_retry(self):
        self.device.button = 1
        type(self.device).status = PropertyMock(side_effect=lambda: "SpaceMouse USB 연결 대기" if self.frame == 0 else "")
        # USB unplugged, connected but unknown, clutch held, button 0 alone, then both up.
        clutch_states = [None, None, True, False, False, False]
        type(self.device).clutch_pressed = PropertyMock(side_effect=lambda: clutch_states[self.frame])
        button_zero_only = replace(self.released, buttons_released=False)
        updates = self.run_frames(
            dry_run=True,
            requests=[False, False, False, False, None, False],
            samples=[None, None, None, button_zero_only, self.released, self.released],
        )

        self.assertEqual([frame["dry_run"] for frame in updates], [True, True, True, True, True, False])
        results = [entry.kwargs for entry in self.panel.mode_result.call_args_list]
        self.assertEqual([result["dry_run"] for result in results], [True, True, True, True, False])
        self.assertIn("USB", results[0]["message"])
        self.assertIn("1번 버튼을 한 번 눌렀다 놓", results[1]["message"])
        for result in results[2:4]:
            self.assertIn("버튼을 놓", result["message"])
        self.assertEqual(results[-1]["message"], "")
        self.assertEqual(self.backend.set_motion_enabled.call_args_list, [call(True), call(False)])
        self.backend.command.assert_not_called()

    def test_gripper_chord_is_virtual_in_dry_and_real_view_shows_feedback(self):
        opened = replace(self.released, gripper_chord=False)
        close_edge = replace(self.held, buttons_released=False, gripper_chord=True, gripper_command=True)
        open_edge = replace(opened, gripper_command=False)
        # A workpiece keeps the real finger width at 42 mm even with a close command.
        gripper = Mock(width_mm=42.0, status="RG2 42 mm")
        levels = []
        gripper.update.side_effect = lambda level: levels.append((self.frame, level))
        updates = self.run_frames(dry_run=True, gripper=gripper,
                                  requests=[None, None, None, False, None, None, None, True],
                                  samples=[opened, close_edge, open_edge, opened, opened, close_edge, open_edge, opened])
        widths = [u["gripper_width_mm"] for u in updates]
        self.assertEqual(widths[0], 42)
        self.assertTrue(39.3 < widths[1] < 40.0)
        np.testing.assert_allclose(widths[2:], [42] * 6)
        self.assertEqual([level for frame, level in levels if frame < 3], [None, None, None])
        self.assertEqual([level for frame, level in levels if 4 <= frame <= 6], [False, True, False])
        self.assertEqual(levels[-1][1], None)

    def test_dry_release_with_arm_still_clutched_snaps_to_current_then_tracks_new_measurements(self):
        idle = replace(self.released, gripper_chord=False)
        close = replace(self.held, buttons_released=False, gripper_command=True, gripper_chord=True)
        held = replace(close, gripper_command=None)
        # The operator releases button 0 and continues holding arm clutch button 1.
        release = replace(self.held, buttons_released=False, gripper_command=False, gripper_chord=False)
        right_only = replace(release, gripper_command=None)
        measured = [4.0, 4.0, 58.0, 58.0, 62.0, 65.0, 39.0]
        gripper = Mock(status="RG2 connected")
        type(gripper).width_mm = PropertyMock(side_effect=lambda: measured[self.frame])
        updates = self.run_frames(dry_run=True, gripper=gripper, requests=[None] * len(measured),
                                  samples=[idle, close, held, held, held, release, right_only])
        widths = [u["gripper_width_mm"] for u in updates]
        self.assertEqual(widths[0], 4.0)
        self.assertTrue(1 < widths[1] < 3)
        np.testing.assert_allclose(widths[2:], [0, 0, 0, 65, 39], atol=1e-9)
        self.assertTrue(all(call.args == (None,) for call in gripper.update.call_args_list))

    def test_dry_sim_preview_restores_saved_sim_width_and_sim_release_opens_gradually(self):
        idle = replace(self.released, gripper_chord=False)
        close = replace(self.held, buttons_released=False, gripper_command=True, gripper_chord=True)
        held = replace(close, gripper_command=None)
        release = replace(idle, gripper_command=False)
        samples = [idle, close, held, held, held, idle, idle, close, held, release,
                   close, held, idle, idle, close, held, release, idle]
        requests = [None] * len(samples)
        requests[5], requests[12] = True, False
        updates = self.run_frames(dry_run=False, simulation=True, requests=requests, samples=samples)
        widths = [u["gripper_width_mm"] for u in updates]
        baseline = widths[4]
        self.assertLess(baseline, 110.0)
        np.testing.assert_allclose(widths[5:7], [baseline, baseline])
        self.assertLess(widths[8], baseline)
        self.assertAlmostEqual(widths[9], baseline)
        self.assertLess(widths[11], baseline)
        np.testing.assert_allclose(widths[12:14], [baseline, baseline])
        self.assertLess(widths[15], widths[14])
        self.assertTrue(widths[15] < widths[16] < widths[17] < 110)
        self.assertLess(max(np.abs(np.diff(widths[14:18]))), 133 / 50)

    def test_real_width_mirrors_fast_changes_and_missing_feedback(self):
        idle = replace(self.released, gripper_chord=False)
        close = replace(self.held, buttons_released=False, gripper_command=True, gripper_chord=True)
        release = replace(idle, gripper_command=False)
        measured = [74.0, 35.0, None, 50.0]
        gripper = Mock(status="RG2")
        type(gripper).width_mm = PropertyMock(side_effect=lambda: measured[self.frame])
        updates = self.run_frames(dry_run=False, gripper=gripper, requests=[None] * len(measured),
                                  samples=[idle, close, replace(close, gripper_command=None), release])
        self.assertEqual([u["gripper_width_mm"] for u in updates], measured)

    def test_panel_open_target_reaches_gripper_without_moving_it_or_clipping_feedback(self):
        gripper = Mock(width_mm=110.6, open_width_mm=75.0, status="RG2 110.6 mm")
        type(self.panel).open_width_mm = PropertyMock(side_effect=lambda: 75.0 if self.frame == 0 else 62.5)
        updates = self.run_frames(dry_run=True, gripper=gripper, requests=[None] * 3,
                                  samples=[self.released] * 3)
        gripper.set_open_width.assert_called_once_with(62.5)
        self.assertEqual([u["gripper_width_mm"] for u in updates], [110.6] * 3)
        self.assertTrue(all(c.args == (None,) for c in gripper.update.call_args_list))

    def test_sim_release_uses_panel_open_target(self):
        self.panel.open_width_mm = 35.0
        idle = replace(self.released, gripper_chord=False)
        close = replace(self.held, gripper_command=True, gripper_chord=True)
        held = replace(close, gripper_command=None)
        release = replace(idle, gripper_command=False)
        samples = [idle, close] + [held] * 50 + [release] + [idle] * 50
        frames = self.run_frames(dry_run=False, simulation=True, requests=[None] * len(samples), samples=samples)
        self.assertEqual(frames[0]["gripper_width_mm"], 75.0)
        self.assertEqual(frames[51]["gripper_width_mm"], 0.0)
        self.assertEqual(frames[-1]["gripper_width_mm"], 35.0)

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
