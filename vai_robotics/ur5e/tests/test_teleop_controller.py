import unittest
from unittest.mock import Mock

import numpy as np
from scipy.spatial.transform import Rotation

from vai_robotics.ur5e.teleop import TeleopController, clutch_text
from vai_robotics.ur5e.teleop_types import InputSample


class TeleopControllerTest(unittest.TestCase):
    def setUp(self):
        self.backend = Mock()
        self.start_rot = Rotation.from_euler("y", 60, degrees=True)
        self.backend.pose.return_value = np.array([0.4, 0.0, 0.3]), self.start_rot
        self.controller = TeleopController(self.backend)
        self.held = InputSample(np.array([0.001, 0, 0]), np.array([0, 0, 0.01]), True)
        self.rest = InputSample(np.zeros(3), np.zeros(3), False)

    def test_releases_and_reanchors_after_loss_or_loop_delay(self):
        self.controller.update(self.held, 0)
        self.backend.command.assert_not_called()
        self.controller.update(self.rest, 0.02)
        self.controller.update(self.held, 0.04)
        self.controller.update(self.held, 0.06)
        xyz, rot = self.backend.command.call_args.args
        np.testing.assert_allclose(xyz, [0.401, 0, 0.3])
        np.testing.assert_allclose((rot * self.start_rot.inv()).as_rotvec(), [0, 0, 0.01], atol=1e-10)
        self.controller.update(None, 0.08)
        self.controller.update(self.held, 0.10)
        self.assertIsNone(self.controller.target)
        self.assertEqual(self.backend.command.call_count, 2)
        self.backend.pose.return_value = np.array([0.45, 0.2, 0.4]), self.start_rot
        self.controller.update(self.rest, 0.12)
        self.controller.update(self.held, 0.14)
        np.testing.assert_array_equal(self.controller.target[0], self.backend.pose.return_value[0])
        self.controller.update(self.held, 1.0)
        self.controller.update(self.held, 1.02)
        self.assertIsNone(self.controller.target)

    def test_speed_and_lead_are_norm_limits_in_base_frame(self):
        self.controller.update(self.rest, 0)
        self.controller.update(self.held, 0.02)
        sample = InputSample(np.ones(3), np.ones(3), True)
        self.controller.update(sample, 0.04)
        origin = self.backend.pose.return_value[0]
        xyz, rot = self.controller.target
        self.assertAlmostEqual(np.linalg.norm(xyz - origin), 0.002)
        self.assertAlmostEqual((rot * self.start_rot.inv()).magnitude(), 0.012)
        for i in range(3, 50):
            self.controller.update(sample, i / 50)
        xyz, rot = self.controller.target
        self.assertAlmostEqual(np.linalg.norm(xyz - origin), 0.025)
        self.assertAlmostEqual((rot * self.start_rot.inv()).magnitude(), 0.12)

    def test_dry_run_previews_scaled_input_and_discards_it_on_mode_change(self):
        self.controller.set_mode(True, simulation=True, sample=None)
        self.controller.update(self.rest, 0)
        self.controller.update(self.held, 0.02)
        for i in range(2, 102):
            self.controller.update(self.held, i / 50, scale=0.5)
        origin, rotation = self.backend.pose.return_value
        xyz, rot = self.controller.target
        np.testing.assert_allclose(xyz, origin + [0.050, 0, 0])
        self.assertAlmostEqual((rot * rotation.inv()).magnitude(), 0.50)
        self.backend.command.assert_not_called()
        self.controller.set_mode(False, simulation=True, sample=self.held)
        self.controller.update(self.held, 2.04)
        self.backend.command.assert_not_called()
        self.controller.update(self.rest, 2.06)
        self.controller.update(self.held, 2.08)
        np.testing.assert_array_equal(self.backend.command.call_args.args[0], origin)

    def test_dry_regrip_starts_at_latest_robot_pose(self):
        controller = TeleopController(self.backend, dry_run=True)
        controller.update(self.rest, 0)
        controller.update(self.held, 0.02)
        controller.update(self.held, 0.04)
        controller.update(self.rest, 0.06)
        self.assertIsNone(controller.target)
        xyz = np.array([0.3, 0.1, 0.2])
        rot = Rotation.from_euler("x", 20, degrees=True)
        self.backend.pose.return_value = xyz, rot
        controller.update(self.held, 0.08)
        np.testing.assert_array_equal(controller.target[0], xyz)
        np.testing.assert_array_equal(controller.target[1].as_matrix(), rot.as_matrix())

    def test_numeric_goals_are_one_shot_in_both_spaces_and_laws(self):
        angles = np.array([0.35, -1.5, 1.2, -1.5, -1.5, 0.1])
        self.backend.joints.return_value = angles
        xyz, rot = self.backend.pose.return_value
        held_still = InputSample(np.zeros(3), np.zeros(3), True)
        base_turn = Rotation.from_rotvec([0.018, 0.0, 0.0])
        for mode in ("eef-delta", "eef-absolute", "joint-delta", "joint-absolute"):
            for law in ("position", "impedance"):
                with self.subTest(mode=mode, law=law):
                    self.backend.reset_mock()
                    controller = TeleopController(self.backend, command_mode=mode, control_law=law)
                    if controller.joint_space:
                        end = angles + 0.018
                        values = end if controller.absolute else np.full(6, 0.018)
                    else:
                        end = xyz + [0.008, 0, 0], base_turn * rot
                        values = np.r_[end[0], end[1].as_rotvec()] if controller.absolute else [0.008, 0, 0, 0.018, 0, 0]
                    self.assertTrue(controller.update(self.rest, 0, values=values))
                    controller.update(held_still, 0.02)
                    controller.update(held_still, 0.04, values=values)
                    for tick in range(3, 12):
                        controller.update(held_still, tick * 0.02)
                    if controller.joint_space:
                        command = self.backend.command_joints if law == "position" else self.backend.command_joint_impedance
                        np.testing.assert_allclose(command.call_args.args[0], end)
                    else:
                        command = self.backend.command if law == "position" else self.backend.command_cartesian_impedance
                        np.testing.assert_allclose(command.call_args.args[0], end[0])
                        np.testing.assert_allclose(command.call_args.args[1].as_matrix(), end[1].as_matrix(), atol=1e-12)
                    controller.update(self.rest, 0.24)
                    self.assertFalse(controller.active)
                    self.assertIsNone(controller.goal)

    def test_joint_jog_dry_limit_and_absolute_ignores_device_axes(self):
        self.backend.joints.return_value = np.zeros(6)
        limits = np.full(6, -0.5), np.full(6, 0.5)
        c = TeleopController(self.backend, dry_run=True, command_mode="joint-delta", joint_limits=limits)
        c.update(self.rest, 0)
        c.update(self.held, 0.02)
        c.update(self.held, 0.04)
        np.testing.assert_allclose(c.joint_target, [0.005, 0, 0, 0, 0, 0.01])
        self.assertIn("Joint Limits", c.update(self.held, 0.06, values=np.ones(6)))
        self.backend.command_joints.assert_not_called()
        absolute = TeleopController(self.backend, command_mode="joint-absolute")
        absolute.update(self.rest, 0)
        absolute.update(self.held, 0.02)
        absolute.update(self.held, 0.04)
        np.testing.assert_array_equal(absolute.joint_target, self.backend.joints.return_value)

    def test_numeric_delta_completes_despite_device_jitter_then_returns_to_jog(self):
        controller = TeleopController(self.backend, dry_run=True)
        xyz = self.backend.pose.return_value[0]
        controller.update(self.rest, 0)
        controller.update(self.held, 0.02)
        controller.update(self.held, 0.04, values=[0, 0.01, 0, 0, 0, 0])
        for tick in range(3, 7):
            controller.update(self.held, tick * 0.02)
        np.testing.assert_allclose(controller.target[0], xyz + [0, 0.01, 0])
        controller.update(InputSample.stationary(enabled=True), 0.14)
        self.assertFalse(controller.target_from_device)
        controller.update(self.held, 0.16)
        self.assertTrue(controller.target_from_device)
        np.testing.assert_allclose(controller.target[0], xyz + [0.001, 0.01, 0])
        self.backend.command.assert_not_called()

    def test_spacemouse_label_reports_selected_physical_button(self):
        device = Mock(button=1, clutch_pressed=True)
        self.assertEqual(clutch_text(device, self.held, True), "버튼 1: 누름")
        self.assertIn("놓았다가", clutch_text(device, None, False))
        device.button = 0
        device.clutch_pressed = False
        self.assertEqual(clutch_text(device, self.rest, False), "버튼 0: 놓음")

    def test_real_run_requires_released_input_and_successful_writer_attachment(self):
        self.controller = TeleopController(self.backend, dry_run=True)
        for sample in (None, self.held):
            self.assertTrue(self.controller.set_mode(False, simulation=False, sample=sample))
        self.backend.set_motion_enabled.assert_not_called()
        self.backend.set_motion_enabled.side_effect = RuntimeError("program active")
        self.assertIn("program active", self.controller.set_mode(False, simulation=False, sample=self.rest))
        self.assertTrue(self.controller.dry_run)
        self.backend.set_motion_enabled.side_effect = None
        self.assertFalse(self.controller.set_mode(False, simulation=False, sample=self.rest))
        self.assertFalse(self.controller.dry_run)
        self.controller.set_mode(True, simulation=False, sample=self.held)
        self.assertTrue(self.controller.dry_run)
        self.backend.set_motion_enabled.assert_called_with(False)
