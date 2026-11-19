import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from vai_robotics.ur5e.controllers import TeleopController
from vai_robotics.ur5e.real import RealUR5e
from vai_robotics.ur5e.teleop import run_teleop
from vai_robotics.ur5e.teleop_types import InputSample


class LiveCommandModeTest(unittest.TestCase):
    def setUp(self):
        self.rest = InputSample(np.zeros(3), np.zeros(3), False)
        self.held = replace(self.rest, enabled=True, buttons_released=False)
        self.backend = Mock()
        self.backend.pose.return_value = np.array([0.4, 0.0, 0.3]), Rotation.identity()
        self.backend.joints.return_value = np.full(6, 0.2)
        self.backend.joint_limits.return_value = np.full(6, -1.0), np.full(6, 1.0)

    def test_disconnected_preview_switch_reanchors_only_after_new_release(self):
        controller = TeleopController(self.backend, dry_run=True)
        self.assertFalse(controller.set_command_mode("joint-delta", simulation=False, sample=None))
        self.assertEqual(controller.command_mode, "joint-delta")
        controller.update(self.held, 0.00)
        self.assertFalse(controller.active)
        controller.update(self.rest, 0.02)
        self.backend.joints.return_value = np.linspace(-0.4, 0.1, 6)
        controller.update(self.held, 0.04)
        np.testing.assert_array_equal(controller.joint_target, self.backend.joints.return_value)
        self.backend.command_joints.assert_not_called()

        for sample in (self.held, replace(self.rest, buttons_released=False)):
            with self.subTest(sample=sample):
                self.assertTrue(controller.set_command_mode("eef-absolute", simulation=False, sample=sample))
                self.assertEqual(controller.command_mode, "joint-delta")

    def test_impedance_writer_changes_only_while_real_motion_is_detached(self):
        # The receive session and both writer implementations are fakes.
        robot = RealUR5e("test")
        old_writer = robot._control = Mock()
        robot._receive = Mock()
        robot._receive.getRuntimeState.return_value = 1
        controller = TeleopController(robot, control_law="impedance")
        self.assertTrue(controller.set_command_mode("eef-absolute", simulation=False, sample=None))
        self.assertEqual(controller.command_mode, "eef-delta")
        self.assertFalse(controller.set_command_mode("eef-absolute", simulation=False, sample=self.rest))
        self.assertTrue(controller.set_command_mode("joint-absolute", simulation=False, sample=self.rest))
        self.assertEqual(controller.command_mode, "eef-absolute")
        old_writer.disconnect.assert_not_called()

        self.enterContext(patch.object(robot, "pose"))
        self.enterContext(patch.object(robot, "_check_safety_mode"))
        rtde_control = SimpleNamespace(RTDEControlInterface=Mock())
        self.enterContext(patch.dict(sys.modules, rtde_control=rtde_control))
        open_joint_writer = self.enterContext(patch("vai_robotics.ur5e.joint_writer.JointImpedanceWriter.open"))
        self.assertFalse(controller.set_mode(True, simulation=False, sample=None))
        old_writer.disconnect.assert_called_once()
        self.assertFalse(controller.set_command_mode("joint-absolute", simulation=False, sample=self.rest))
        open_joint_writer.assert_not_called()
        self.assertFalse(controller.set_mode(False, simulation=False, sample=self.rest))
        open_joint_writer.assert_called_once()
        rtde_control.RTDEControlInterface.assert_not_called()
        self.assertTrue(controller.set_command_mode("eef-delta", simulation=False, sample=self.rest))
        self.assertEqual(controller.command_mode, "joint-absolute")
        self.assertFalse(controller.set_mode(True, simulation=False, sample=None))
        self.assertFalse(controller.set_command_mode("eef-delta", simulation=False, sample=self.rest))
        self.assertFalse(controller.set_mode(False, simulation=False, sample=self.rest))
        rtde_control.RTDEControlInterface.assert_called_once_with(robot.host)
        robot.close()

    def test_runner_confirms_requested_mode_and_discards_stale_numeric_targets(self):
        panel = Mock(is_open=True, scale=1.0, open_width_mm=75.0)
        device = Mock(spec=["read"])
        samples = [self.rest, self.held, self.held, self.rest, self.rest, self.held,
                   self.rest, self.held, self.held, self.held]
        # A held request is refused. The same request is accepted after the clutch is released.
        requests = {2: "joint-absolute", 4: "joint-absolute"}
        numeric_targets = {2: [0.9] * 6, 4: [0.9] * 6, 9: [2.0] * 6}
        frame = -1

        def poll():
            nonlocal frame
            frame += 1
            panel.is_open = frame < len(samples)

        panel.poll.side_effect = poll
        panel.take_command_mode.side_effect = lambda: requests.get(frame)
        panel.take_target.side_effect = lambda: numeric_targets.get(frame)
        device.read.side_effect = lambda: samples[frame]
        self.backend.joints.side_effect = lambda: np.full(6, frame / 100)
        joint_commands = []
        self.backend.command_joints.side_effect = lambda q: joint_commands.append((frame, q.copy()))
        with patch("vai_robotics.ur5e.teleop.time") as clock:
            clock.monotonic.side_effect = lambda: frame / 50
            run_teleop(device, self.backend, sim=True, panel=panel)

        receipts = [call.kwargs for call in panel.command_mode_result.call_args_list]
        self.assertEqual([r["command_mode"] for r in receipts], ["eef-delta", "joint-absolute"])
        self.assertTrue(receipts[0]["message"])
        self.assertFalse(receipts[1]["message"])
        self.assertEqual([tick for tick, _ in joint_commands], [7, 8, 9])
        for _, q in joint_commands:
            np.testing.assert_array_equal(q, np.full(6, 0.07))
        self.assertEqual(self.backend.tick.call_count, len(samples))
        self.assertIn("관절 범위", panel.refresh.call_args.kwargs["message"])
