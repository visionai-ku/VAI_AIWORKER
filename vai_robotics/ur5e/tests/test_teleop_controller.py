import unittest
from unittest.mock import Mock

import numpy as np
from scipy.spatial.transform import Rotation

from vai_robotics.ur5e.teleop import TeleopController
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
