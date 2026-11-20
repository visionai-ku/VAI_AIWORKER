import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from vai_robotics.ur5e.sim_teleop import SimBackend


class SimBackendTest(unittest.TestCase):
    def test_joint_commands_and_joint_dry_preview_have_separate_state(self):
        with SimBackend(headless=True) as backend:
            q = backend.joints()
            requested = q + [0.15, -0.06, 0, 0.10, 0, 0.02]
            backend.tick(dry_run=True, joint_target=requested)
            np.testing.assert_allclose(backend._shadow.qpos, requested)
            np.testing.assert_array_equal(backend.joints(), q)
            self.assertEqual(backend.sim.data.time, 0)
            backend.tick(dry_run=True)
            np.testing.assert_array_equal(backend._shadow.qpos, q)
            for _ in range(100):
                backend.command_joints(requested)
                backend.tick()
            np.testing.assert_allclose(backend.joints(), requested, atol=2e-4)
            for bad in (np.zeros(5), np.full(6, np.nan), np.full(6, 9.0)):
                for command in (backend.command_joints, backend.command_joint_impedance):
                    with self.subTest(command=command, bad=bad), self.assertRaises(ValueError):
                        command(bad)

    def test_joint_impedance_yields_to_external_torque_and_release_latches_new_pose(self):
        with SimBackend(headless=True) as backend:
            q = backend.joints()
            backend.command_joint_impedance(q)
            applied = backend.sim.data.qfrc_applied
            applied[0] = 1.2  # 1.2 Nm / 40 Nm/rad = 0.03 rad spring deflection
            for _ in range(150):
                backend.tick()
            self.assertEqual(applied[0], 1.2)
            displaced = backend.joints()
            self.assertAlmostEqual(displaced[0] - q[0], .030, delta=.002)
            backend.hold()
            applied[0] = 0.0
            for _ in range(60):
                backend.tick()
            np.testing.assert_allclose(backend.joints(), displaced, atol=2e-4)

    def test_cartesian_impedance_deflects_under_load_and_returns_to_fixed_equilibrium(self):
        with SimBackend(headless=True) as backend:
            p, rotation = backend.pose()
            backend.command_cartesian_impedance(p, rotation)
            applied = backend.sim.data.xfrc_applied[backend.sim.model.body("wrist_3_link").id]
            applied[0] = 4.0
            expected_dx = applied[0] / backend.get_impedance_gains("eef")["stiffness"][0]
            for _ in range(150):
                backend.tick()
            np.testing.assert_allclose(backend.pose()[0] - p, [expected_dx, 0, 0], atol=.00025)
            self.assertEqual(applied[0], 4.0)
            applied[0] = 0.0
            for _ in range(180):
                backend.tick()
            np.testing.assert_allclose(backend.pose()[0], p, atol=.0002)

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

    def test_live_gripper_mirrors_measured_width_and_keeps_it_on_data_loss(self):
        with SimBackend(headless=True) as backend:
            q = backend.sim.data.qpos.copy()
            tcp = backend.pose()
            for dry, width, expected in ((False, 50.6, 50.6), (True, 0, 0), (False, None, 50.6)):
                backend.tick(dry_run=dry, measured_q=q, measured_tcp=tcp, gripper_width_mm=width)
                displayed = backend._shadow if dry else backend.sim.data
                gap = displayed.site("rg2_left_pad").xpos - displayed.site("rg2_right_pad").xpos
                self.assertAlmostEqual(1000 * np.linalg.norm(gap), expected, places=4)

    def test_dry_arm_moves_separately_and_releases_to_live_pose_in_one_tick(self):
        with SimBackend(headless=True) as backend:
            origin, rotation = backend.pose()
            live_joints = backend.sim.data.qpos.copy()
            live_ctrl = backend.sim.data.ctrl.copy()
            target = origin + [0.04, -0.02, 0.035], Rotation.from_rotvec([0.1, 0, 0.04]) * rotation
            backend.tick(dry_run=True)
            for _ in range(30):
                backend.tick(dry_run=True, target=target)
            preview_xyz, preview_rotation = backend.preview_pose()
            np.testing.assert_allclose(preview_xyz, target[0], atol=1e-5)
            self.assertLess((preview_rotation * target[1].inv()).magnitude(), 1e-5)
            backend.tick(dry_run=True)
            np.testing.assert_array_equal(backend.preview_pose()[0], origin)
            np.testing.assert_array_equal(backend.preview_pose()[1].as_matrix(), rotation.as_matrix())
            np.testing.assert_array_equal(backend.sim.data.qpos, live_joints)
            np.testing.assert_array_equal(backend.sim.data.ctrl, live_ctrl)
            self.assertEqual(backend.sim.data.time, 0.0)
            backend.tick()
            self.assertIsNone(backend.preview_pose())
            np.testing.assert_allclose(backend.pose()[0], origin, atol=1e-4)
            self.assertAlmostEqual(backend.sim.data.time, backend.period)
            backend.tick(dry_run=True)
            np.testing.assert_allclose(backend.preview_pose()[0], backend.pose()[0], atol=1e-5)

    def test_mirrors_measured_joints_and_previews_the_configured_tcp(self):
        with SimBackend(headless=True) as backend:
            q = backend.sim.data.qpos.copy()
            q[0] += 0.2
            backend.tick(measured_q=q)
            xyz, rot = backend.pose()
            tcp = xyz + rot.apply([0.04, 0.0, 0.12]), rot * Rotation.from_euler("z", 35, degrees=True)
            backend.tick(dry_run=True, measured_q=q, measured_tcp=tcp)
            np.testing.assert_allclose(backend.preview_pose()[0], tcp[0], atol=1e-10)
            np.testing.assert_allclose(backend.preview_pose()[1].as_matrix(), tcp[1].as_matrix(), atol=1e-10)
            target = tcp[0] + [0.02, 0.02, -0.01], Rotation.from_euler("x", 12, degrees=True) * tcp[1]
            for _ in range(35):
                backend.tick(dry_run=True, target=target, measured_q=q, measured_tcp=tcp)
            np.testing.assert_allclose(backend.preview_pose()[0], target[0], atol=1e-5)
            self.assertLess((backend.preview_pose()[1] * target[1].inv()).magnitude(), 1e-5)
            np.testing.assert_array_equal(backend.sim.data.qpos, q)
            # The renderer missed the release and first sees the next press.
            backend.tick(dry_run=True, target=tcp, measured_q=q, measured_tcp=tcp, reset_preview=True)
            np.testing.assert_allclose(backend._shadow.qpos, q, atol=1e-10)
            for delta in (0.15, 0.3):
                # Rotate the entire arm around base Z while the clutch is up.
                new_q = q + [delta, 0, 0, 0, 0, 0]
                base_turn = Rotation.from_rotvec([0, 0, delta])
                new_tcp = base_turn.apply(tcp[0]), base_turn * tcp[1]
                backend.tick(dry_run=True, measured_q=new_q, measured_tcp=new_tcp)
                np.testing.assert_array_equal(backend._shadow.qpos, new_q)
                np.testing.assert_allclose(backend.preview_pose()[0], new_tcp[0], atol=1e-10)
                np.testing.assert_allclose(backend.preview_pose()[1].as_matrix(),
                                           new_tcp[1].as_matrix(), atol=1e-10)
            backend.tick(dry_run=True, target=new_tcp, measured_q=new_q, measured_tcp=new_tcp)
            np.testing.assert_allclose(backend._shadow.qpos, new_q, atol=1e-10)
            backend.tick(measured_q=new_q)
            np.testing.assert_array_equal(backend.sim.data.qpos, new_q)
            self.assertIsNone(backend.preview_pose())
