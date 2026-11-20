import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from vai_robotics.ur5e.impedance import cartesian_wrench, joint_torque
from vai_robotics.ur5e.impedance_gains import default_impedance_gains, max_impedance_gains, validate_impedance_gains
from vai_robotics.ur5e.real import RealUR5e
from vai_robotics.ur5e.sim_teleop import SimBackend


class ImpedanceTest(unittest.TestCase):
    def test_gain_profiles_are_bounded_per_axis_and_atomic(self):
        backend = SimBackend(headless=True)
        self.assertEqual(backend.get_impedance_gains("eef"), {
            "stiffness": [3000.] * 3 + [100.] * 3,
            "damping": [400.] * 3 + [22.] * 3,
        })
        for space in ("eef", "joint"):
            maxima = max_impedance_gains(space)
            backend.set_impedance_gains(space, maxima)
            for bad in (True, np.bool_(False), "1", -0.01, np.nan, np.inf, maxima["damping"][5] + .01):
                with self.subTest(space=space, bad=bad):
                    values = [0] * 5 + [bad]
                    with self.assertRaises(ValueError):
                        backend.set_impedance_gains(space, {"stiffness": [0] * 6, "damping": values})
                    self.assertEqual(backend.get_impedance_gains(space), maxima)
            saved = backend.get_impedance_gains(space)
            saved["damping"][0] = -1
            self.assertEqual(backend.get_impedance_gains(space), maxima)
        for invalid in ({}, {"stiffness": [0] * 6, "damping": [1]}, None):
            with self.assertRaises(ValueError):
                validate_impedance_gains("joint", invalid)
        with self.assertRaises(ValueError):
            backend.set_impedance_gains("unknown", default_impedance_gains("joint"))

    def test_per_axis_laws_match_rtde_wrench_and_preserve_output_limits(self):
        kp = np.array([20., 40., 60., 2., 4., 6.])
        kd = np.array([1., 2., 3., .1, .2, .3])
        gains = {"stiffness": kp, "damping": kd}
        error = np.array([.01, -.01, .02, .03, -.04, .02])
        velocity = np.array([.02, -.01, .01, .12, .04, -.1])
        measured = Rotation.from_euler("xy", [50, 20], degrees=True)
        target = Rotation.from_rotvec(error[3:]) * measured
        expected = kp * error - kd * velocity
        np.testing.assert_allclose(joint_torque(error, np.zeros(6), velocity, **gains), expected)

        backend = RealUR5e("test")
        backend._control = Mock()
        backend._receive = Mock()
        backend._receive.getActualTCPSpeed.return_value = velocity
        backend.set_impedance_gains("eef", gains)
        backend._control.forceMode.assert_not_called()
        with patch.object(backend, "_checked_tcp", return_value=(None, np.zeros(3), measured)):
            backend.command_cartesian_impedance(error[:3], target)
        wrench = backend._control.forceMode.call_args.args[2]
        np.testing.assert_allclose(wrench, expected, atol=1e-12)
        with self.assertRaises(ValueError):
            backend.set_impedance_gains("eef", default_impedance_gains("eef"))
        capped = cartesian_wrench([1, 1, 1], target, np.zeros(3), measured, [100] * 6, **gains)
        self.assertAlmostEqual(np.linalg.norm(capped[:3]), 15)
        self.assertAlmostEqual(np.linalg.norm(capped[3:]), 2)

    def test_headless_simulation_uses_each_spaces_selected_stiffness(self):
        for space in ("eef", "joint"):
            displacements = []
            for stiffness_scale in (0, 1):
                with SimBackend(headless=True) as sim:
                    gains = sim.get_impedance_gains(space)
                    gains["stiffness"][0] *= stiffness_scale
                    sim.set_impedance_gains(space, gains)
                    xyz, rotation = sim.pose()
                    q = sim.joints()
                    if space == "eef":
                        sim.command_cartesian_impedance(xyz + [.005, 0, 0], rotation)
                    else:
                        sim.command_joint_impedance(q + [.02, 0, 0, 0, 0, 0])
                    for _ in range(10):
                        sim.tick()
                    displacement = sim.pose()[0][0] - xyz[0] if space == "eef" else sim.joints()[0] - q[0]
                    displacements.append(displacement)
                    with self.assertRaises(ValueError):
                        sim.set_impedance_gains(space, gains)
                    sim.hold()
                    sim.set_impedance_gains(space, gains)
            with self.subTest(space=space):
                self.assertAlmostEqual(displacements[0], 0, places=5)
                self.assertGreater(displacements[1], .001)

    def test_force_and_rotation_are_in_base_axes_and_oppose_velocity(self):
        measured = Rotation.from_euler("y", 90, degrees=True)
        target = Rotation.from_rotvec([0.016, 0, 0]) * measured
        wrench = cartesian_wrench([0, .004, 0], target, np.zeros(3), measured, [0, .01, 0, .05, 0, 0])
        np.testing.assert_allclose(wrench, [0, 8., 0, .5, 0, 0], atol=1e-12)

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
