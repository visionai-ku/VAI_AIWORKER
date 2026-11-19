import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from vai_robotics.ur5e.impedance import cartesian_wrench, joint_torque


class ImpedanceTest(unittest.TestCase):
    def test_force_and_rotation_are_in_base_axes_and_oppose_velocity(self):
        measured = Rotation.from_euler("y", 90, degrees=True)
        target = Rotation.from_rotvec([0.1, 0, 0]) * measured
        wrench = cartesian_wrench([0, .01, 0], target, np.zeros(3), measured, [0, .01, 0, .2, 0, 0])
        np.testing.assert_allclose(wrench, [0, 2.7, 0, .5, 0, 0], atol=1e-12)

    def test_saturation_and_invalid_inputs(self):
        r = Rotation.identity()
        wrench = cartesian_wrench([1., 1., 0.], Rotation.from_rotvec([2., 0, 0]), np.zeros(3), r, np.zeros(6))
        self.assertAlmostEqual(np.linalg.norm(wrench[:3]), 15.0)
        self.assertAlmostEqual(np.linalg.norm(wrench[3:]), 2.0)
        torque = joint_torque(np.zeros(6), np.zeros(6), np.full(6, 100.0))
        np.testing.assert_array_equal(torque, [-15, -15, -12, -4, -4, -3])
        with self.assertRaises(ValueError):
            cartesian_wrench(np.zeros(3), r, np.zeros(3), r, np.full(6, np.nan))
        with self.assertRaises(ValueError):
            joint_torque(np.zeros(6), np.zeros(6), np.zeros(3))
