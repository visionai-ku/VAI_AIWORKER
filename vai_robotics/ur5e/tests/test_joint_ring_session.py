import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from vai_robotics.ur5e.inputs import SpaceMouseInput
from vai_robotics.ur5e.real import TargetValidationExpired
from vai_robotics.ur5e.teleop_session import TeleopSession
from vai_robotics.ur5e.teleop_types import InputSample, JointDrag


class JointRingSessionTests(unittest.TestCase):
    def start(self, *, sim=True, dry=False, mode="joint-delta", law="position"):
        self.time = 10.
        self.device = SpaceMouseInput()
        self.device.read = Mock()
        self.view = Mock()
        self.view.set_joint_ring_epoch.side_effect = lambda epoch: setattr(self.view.joint_drag, "return_value", None)
        self.backend = Mock()
        self.q = np.arange(6) / 10
        self.backend.pose.return_value = np.zeros(3), Rotation.identity()
        self.backend.joints.return_value = self.q.copy()
        self.backend.joint_limits.return_value = np.full(6, -2.), np.full(6, 2.)
        self.backend.payload.return_value = self.backend.tcp_speed_feedback.return_value = None
        self.panel = Mock(scale=1., open_width_mm=75., is_open=True)
        for method in ("poll", "take_command_mode", "take_control_law", "take_impedance_gains", "take_target"):
            getattr(self.panel, method).return_value = None
        self.session = TeleopSession(self.device, self.backend, simulation=sim, dry_run=dry, panel=self.panel,
                                    gripper=None, gripper_force_n=40, command_mode=mode, control_law=law,
                                    monotonic=lambda: self.time)
        self.step(None)

    def step(self, drag, physical=None):
        self.time += .02
        self.view.joint_drag.return_value = drag
        self.device.read.return_value = physical
        self.assertTrue(self.session.step(self.view, self.time))
        return self.session.controller.joint_target

    def test_mouse_alone_drives_one_joint_in_both_modes_and_laws(self):
        for mode in ("joint-delta", "joint-absolute"):
            for law in ("position", "impedance"):
                with self.subTest(mode=mode, law=law):
                    self.start(mode=mode, law=law)
                    self.step(JointDrag(0, None))
                    target = self.step(JointDrag(1, 2, self.q[2] + 1.))
                    expected = self.q.copy()
                    expected[2] += .01  # 0.5 rad/s for one 20 ms interval
                    np.testing.assert_allclose(target, expected)
                    writer = self.backend.command_joints if law == "position" else self.backend.command_joint_impedance
                    np.testing.assert_allclose(writer.call_args.args[0], expected)
                    self.assertIsNone(self.step(JointDrag(1, None)))

    def test_real_clutch_does_not_arm_an_old_mouse_press_or_resume_after_input_loss(self):
        self.start(sim=False)
        cap = InputSample(np.ones(3), np.ones(3), True)
        self.step(JointDrag(0, None), InputSample.stationary())
        press = JointDrag(1, 1, 1.)
        self.assertIsNone(self.step(press))
        self.assertIsNone(self.step(press, cap))  # New epoch at physical clutch press.
        self.assertIsNone(self.step(press, cap))
        self.assertIsNone(self.step(JointDrag(1, None), cap))
        self.backend.command_joints.assert_not_called()
        self.assertIsNotNone(self.step(JointDrag(2, 1, 1.), cap))
        self.assertIsNone(self.step(None, cap))  # Stale viewer, still physically clutched.
        self.assertIsNone(self.step(JointDrag(2, 1, 1.), cap))
        self.step(JointDrag(2, None), cap)
        self.assertIsNotNone(self.step(JointDrag(3, 1, 1.), cap))
        self.assertIsNone(self.step(JointDrag(3, 1, 1.), InputSample.stationary()))
        self.assertIsNone(self.step(JointDrag(3, 1, 1.), cap))
        self.assertEqual(self.backend.command_joints.call_count, 2)

    def test_connected_dry_run_uses_shadow_and_numeric_targets_still_require_clutch(self):
        self.start(sim=False, dry=True)
        self.step(JointDrag(0, None))
        target = self.step(JointDrag(1, 4, 1.))
        np.testing.assert_allclose(self.view.update.call_args.kwargs["joint_target"], target)
        self.assertIsNone(self.step(JointDrag(1, None)))
        self.panel.take_target.return_value = [.1, 0., 0., 0., 0., 0.]
        self.assertIsNone(self.step(None))
        self.step(None, InputSample.stationary())
        target = self.step(None, InputSample.stationary(enabled=True))
        self.assertGreater(target[0], self.q[0])
        self.assertEqual(target[4], self.q[4])
        self.backend.command_joints.assert_not_called()

    def test_desktop_joint_session_can_use_mouse_without_optional_hid_library(self):
        with patch("vai_robotics.ur5e.inputs.spacemouse._load_spacemouse", side_effect=RuntimeError("no HIDAPI")):
            with SpaceMouseInput(allow_mouse_only=True) as device:
                self.assertIsNone(device.read())
                self.assertEqual(device.status, "no HIDAPI")
                self.assertFalse(device.connected)

    def test_late_joint_validation_requires_a_new_physical_clutch_and_ring(self):
        self.start(sim=False)
        idle = JointDrag(0, None)
        held = InputSample.stationary(enabled=True)
        self.step(idle, InputSample.stationary())
        self.step(idle, held)
        self.step(idle, held)
        self.backend.command_joints.side_effect = TargetValidationExpired("Joint target discarded")
        self.assertIsNone(self.step(JointDrag(1, 0, .05), held))
        self.assertFalse(self.session.controller.dry_run)
        self.assertEqual(self.panel.refresh.call_args.kwargs["operator_state"], "release-clutch")
        self.backend.command_joints.side_effect = None
        for ring in (None, JointDrag(1, None), JointDrag(2, 0, .05)):
            self.assertIsNone(self.step(ring, held))
        self.assertEqual(self.backend.command_joints.call_count, 1)
        self.step(JointDrag(2, None), InputSample.stationary())
        self.step(JointDrag(2, None), held)
        self.step(JointDrag(2, None), held)
        self.assertIsNotNone(self.step(JointDrag(3, 1, .15), held))
        self.assertEqual(self.backend.command_joints.call_count, 2)
        self.backend.set_motion_enabled.assert_not_called()

    def test_mount_is_forwarded_to_local_and_connected_views(self):
        for sim in (False, True):
            with self.subTest(sim=sim):
                self.start(sim=sim, dry=True)
                self.panel.rg2_mount = (0., 10., 15., 0., 90., 35.)
                self.step(None)
                self.view.set_rg2_mount.assert_called_with(self.panel.rg2_mount)
                self.backend.command_joints.assert_not_called()

    def test_busy_hid_device_does_not_block_mouse_only_preview_and_can_reconnect(self):
        driver = Mock()
        driver.get_connected_devices.return_value = ["SpaceMouse"]
        driver.open.side_effect = [PermissionError("busy"), Mock()]
        with patch("vai_robotics.ur5e.inputs.spacemouse._load_spacemouse", return_value=(driver, OSError)):
            with SpaceMouseInput(allow_mouse_only=True, wait_for_device=True) as device:
                self.assertFalse(device.connected)
                self.assertIn("busy", device.status)
                device.retry_at = 0.
                device.read()
                self.assertTrue(device.connected)
