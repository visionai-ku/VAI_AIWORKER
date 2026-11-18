import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from vai_robotics.ur5e.sim_teleop import SimBackend


class SimBackendTest(unittest.TestCase):
    def test_pose_tracks_translation_and_world_frame_rotation(self):
        with SimBackend(headless=True) as backend:
            position, rotation = backend.pose()
            # Nominal UR base-frame flange pose at the bundled "home" joint angles.
            np.testing.assert_allclose(position, [-0.134, 0.492, 0.488], atol=1e-5)
            home_rotation = Rotation.from_euler("x", 180, degrees=True)
            self.assertLess((rotation * home_rotation.inv()).magnitude(), 1e-5)

            target = position + [0.040, -0.020, 0.035]
            target_rotation = Rotation.from_rotvec([0.15, -0.01, 0.08]) * rotation
            for _ in range(100):
                backend.command(target, target_rotation)
                backend.tick()
            measured, measured_rotation = backend.pose()
            np.testing.assert_allclose(measured, target, atol=2e-4)
            self.assertLess((target_rotation * measured_rotation.inv()).magnitude(), 1e-3)

    def test_release_latches_current_joint_positions(self):
        with SimBackend(headless=True) as backend:
            position, rotation = backend.pose()
            initial = backend.sim.data.ctrl.copy()
            backend.command(position + [1, 1, 1], rotation)
            self.assertLessEqual(np.max(abs(backend.sim.data.ctrl - initial)), 0.031)
            backend.tick()
            released = backend.sim.data.qpos.copy()
            backend.hold()
            np.testing.assert_array_equal(backend.sim.data.ctrl, released)
            for _ in range(50):
                backend.hold()
                backend.tick()
            np.testing.assert_array_equal(backend.sim.data.ctrl, released)
            np.testing.assert_allclose(backend.sim.data.qpos, released, atol=1e-3)
        self.assertFalse(backend.is_running())
