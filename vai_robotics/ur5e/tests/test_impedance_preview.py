"""Exercise the impedance preview without a robot, an input device or a viewer."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from vai_robotics.ur5e.impedance_gains import default_impedance_gains, max_impedance_gains
from vai_robotics.ur5e.real import RealUR5e
from vai_robotics.ur5e.sim_teleop import SimBackend


class ImpedancePreviewTest(unittest.TestCase):
    def setUp(self):
        self.backend = self.enterContext(SimBackend(headless=True))

    def preview(self, **kwargs):
        self.backend.tick(dry_run=True, preview_impedance=True, **kwargs)

    def step_response(self, space, stiffness, damping):
        self.preview()  # Release before editing gains and start from the same measured pose.
        gains = default_impedance_gains(space)
        gains["stiffness"][0] = stiffness
        gains["damping"][0] = damping
        self.backend.set_impedance_gains(space, gains)
        position, rotation = self.backend.pose()
        angles = self.backend.joints()
        if space == "eef":
            command = {"target": (position + [.005, 0, 0], rotation)}
        else:
            command = {"joint_target": angles + [.02, 0, 0, 0, 0, 0]}
        for _ in range(8):
            self.preview(**command)
        if space == "eef":
            return self.backend.preview_pose()[0][0] - position[0]
        return self.backend._shadow.qpos[0] - angles[0]

    def test_cartesian_and_joint_preview_respond_to_both_gains(self):
        for space in ("eef", "joint"):
            with self.subTest(space=space):
                stiffness = default_impedance_gains(space)["stiffness"][0]
                high_damping = max_impedance_gains(space)["damping"][0]
                no_spring = self.step_response(space, 0, high_damping)
                damped = self.step_response(space, stiffness, high_damping)
                undamped = self.step_response(space, stiffness, 0)
                self.assertAlmostEqual(no_spring, 0.0, delta=1e-5)
                self.assertGreater(damped, no_spring + 1e-4)
                self.assertGreater(undamped, damped + 1e-5)

    def test_preview_scales_with_elapsed_time_and_freezes_live_sim(self):
        sim = self.backend.sim
        initial_q = self.backend.joints()
        initial_ctrl = sim.data.ctrl.copy()
        initial_v = sim.data.qvel.copy()
        initial_groups = sim.model.opt.disableactuator
        position, rotation = self.backend.pose()
        target = position + [.006, 0, 0], rotation
        positions = []
        for dt, frames in ((0.02, 6), (0.04, 3)):
            self.preview()
            self.preview(target=(position, rotation))
            for _ in range(frames):
                self.preview(target=target, preview_dt=dt)
            positions.append(self.backend.preview_pose()[0])
        self.assertGreater(positions[0][0], position[0] + .0002)
        np.testing.assert_allclose(positions[0], positions[1], atol=.0001)
        np.testing.assert_array_equal(self.backend.joints(), initial_q)
        np.testing.assert_array_equal(sim.data.ctrl, initial_ctrl)
        np.testing.assert_array_equal(sim.data.qvel, initial_v)
        self.assertEqual(sim.data.time, 0.0)
        self.assertEqual(sim.model.opt.disableactuator, initial_groups)

    def test_spacemouse_reversal_drops_excess_target_lead(self):
        origin, orientation = self.backend.pose()
        target = origin.copy()
        self.preview(target=(target.copy(), orientation), preview_delta=True)
        # Full deflection advances the raw target by 80 mm/s, faster than the
        # force-mode limit of 60 mm/s. Only the allowed lead should accumulate.
        for _ in range(200):
            target[0] += .08 * self.backend.period
            self.preview(target=(target.copy(), orientation), preview_delta=True)
        positions = []
        for _ in range(24):
            target[0] -= .08 * self.backend.period
            self.preview(target=(target.copy(), orientation), preview_delta=True)
            positions.append(self.backend.preview_pose()[0][0])
        self.assertGreater(target[0], positions[-1])
        self.assertLess(positions[-1], positions[-2])

    def test_numeric_delta_starts_at_limited_jog_target_and_reaches_new_goal(self):
        xyz, rotation = self.backend.pose()
        for space in ("eef", "joint"):
            with self.subTest(space=space):
                self.preview()
                joint = space == "joint"
                raw = self.backend.joints() if joint else xyz.copy()
                if joint:
                    gains = default_impedance_gains("joint")
                    gains["stiffness"][0] = 10
                    self.backend.set_impedance_gains("joint", gains)
                command = {"joint_target": raw} if joint else {"target": (raw, rotation)}
                for _ in range(150):
                    raw[0] += (.5 if joint else .08) * self.backend.period
                    self.preview(**command, preview_delta=True)
                equilibrium = self.backend._impedance_preview.equilibrium
                last_jog = equilibrium[0] if joint else equilibrium[0][0]
                self.assertGreater(raw[0] - last_jog, .01)
                delta = .01 if joint else .003
                raw[0] += delta
                for _ in range(250):
                    self.preview(**command, preview_delta=False)
                final = self.backend._shadow.qpos[0] if joint else self.backend.preview_pose()[0][0]
                self.assertAlmostEqual(final, last_jog + delta, delta=.001)

    def test_offset_tcp_stays_within_force_mode_speed_limits_and_reanchors(self):
        q = self.backend.joints()
        origin, orientation = self.backend.pose()
        tcp = origin + orientation.apply([.06, -.04, .14]), orientation * Rotation.from_euler("y", 25, degrees=True)
        self.preview(measured_q=q, measured_tcp=tcp)
        position, rotation = self.backend.preview_pose()
        period = self.backend.period
        max_linear_step = np.array(RealUR5e.FORCE_SPEED_LIMITS[:3]) * period
        max_angular_step = np.array(RealUR5e.FORCE_SPEED_LIMITS[3:]) * period
        for frame in range(1, 41):
            target_xyz = tcp[0] + [frame * period * .08, 0, 0]
            target_rot = Rotation.from_rotvec([0, frame * period * .5, 0]) * tcp[1]
            self.preview(measured_q=q, measured_tcp=tcp, target=(target_xyz, target_rot))
            next_position, next_rotation = self.backend.preview_pose()
            self.assertTrue(np.all(np.abs(next_position - position) <= max_linear_step + 2e-5))
            rotation_step = (next_rotation * rotation.inv()).as_rotvec()
            self.assertTrue(np.all(np.abs(rotation_step) <= max_angular_step + 1e-4))
            position, rotation = next_position, next_rotation
        self.assertGreater(position[0] - tcp[0][0], .005)
        self.assertGreater((rotation * tcp[1].inv()).as_rotvec()[1], .01)

        # A renderer can miss the release while the physical arm is repositioned.
        new_q = q + [.15, 0, 0, 0, 0, 0]
        turn = Rotation.from_rotvec([0, 0, .15])
        new_tcp = turn.apply(tcp[0]), turn * tcp[1]
        self.preview(measured_q=new_q, measured_tcp=new_tcp, target=new_tcp, reset_preview=True)
        np.testing.assert_allclose(self.backend.preview_pose()[0], new_tcp[0], atol=1e-7)
        self.assertLess((self.backend.preview_pose()[1] * new_tcp[1].inv()).magnitude(), 1e-6)
        self.preview(measured_q=q, measured_tcp=tcp)
        np.testing.assert_allclose(self.backend.preview_pose()[0], tcp[0], atol=1e-10)
        self.assertLess((self.backend.preview_pose()[1] * tcp[1].inv()).magnitude(), 1e-8)
