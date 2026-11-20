import unittest
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from vai_robotics.ur5e.payload import ToolPayload
from vai_robotics.ur5e.rg2_visual import DEFAULT_VISUAL_MOUNT, OPEN_WIDTH_MM, RG2Preview, set_rg2_width
from vai_robotics.ur5e.sim_teleop import SimBackend


class RG2PreviewTest(unittest.TestCase):
    def test_full_stroke_is_timed_and_stops_at_each_endpoint(self):
        for hz in (25, 50, 100):
            with self.subTest(hz=hz):
                preview = RG2Preview()
                preview.update(0, None, True, True)
                widths = [OPEN_WIDTH_MM] + [preview.update(i / hz, None, None, True)
                                           for i in range(1, 3 * hz + 1)]
                self.assertEqual(widths[-1], 0.0)
                self.assertTrue(np.all(np.diff(widths) <= 0))
                speeds = -np.diff(widths) * hz
                midstroke = next(i for i, width in enumerate(widths) if width <= 50)
                # The manufacturer's 40 N curve reaches about 117 mm/s near
                # 50 mm width, and drops to about 38 mm/s at the open stop.
                self.assertTrue(30 < speeds[0] < 60)
                self.assertTrue(105 < speeds[midstroke] < 133)
                self.assertLessEqual(speeds.max(), 133)
                opened = [preview.update(3 + i / hz, None, False if i == 1 else None, False,
                                         open_width_mm=OPEN_WIDTH_MM)
                          for i in range(1, 3 * hz + 1)]
                self.assertTrue(np.all(np.diff(opened) >= 0))
                self.assertEqual(opened[-1], OPEN_WIDTH_MM)

    def test_default_open_goal_from_below_and_above(self):
        for measured in (38.0, 110.6):
            with self.subTest(measured=measured):
                preview = RG2Preview()
                self.assertEqual(preview.update(0, measured, None, False), measured)
                first = preview.update(0.02, measured, False, False)
                self.assertTrue(min(measured, 75.0) < first < max(measured, 75.0))
                for frame in range(2, 101):
                    width = preview.update(frame / 50, measured, None, False)
                self.assertEqual(width, 75.0)

    def test_edit_takes_effect_on_next_open_command(self):
        preview = RG2Preview()
        preview.update(0, 30.0, False, False, open_width_mm=60.11)
        for frame in range(1, 101):
            width = preview.update(frame / 50, None, None, False, open_width_mm=89.0)
        self.assertEqual(width, 60.1)
        preview.update(2.02, None, True, True, open_width_mm=89.0)
        preview.update(2.04, None, False, False, open_width_mm=89.0)
        for frame in range(103, 151):
            width = preview.update(frame / 50, None, None, False)
        self.assertEqual(width, 89.0)

    def test_40n_preview_closes_from_half_open_in_about_four_tenths_of_a_second(self):
        preview = RG2Preview()
        preview.update(0, 50.6, True, True)
        for frame in range(1, 51):
            if preview.update(frame / 50, 50.6, None, True) == 0.0:
                break
        self.assertTrue(0.36 <= frame / 50 <= 0.48)

    def test_halving_commanded_force_reduces_aperture_speed_near_midstroke(self):
        speeds = []
        for force_n in (20.0, 40.0):
            preview = RG2Preview(force_n=force_n)
            preview.update(0, 50.0, True, True)
            width = preview.update(0.02, 50.0, None, True)
            speeds.append((50.0 - width) / 0.02)
        self.assertTrue(50 < speeds[0] < 70)
        self.assertTrue(105 < speeds[1] < 133)

    def test_reverses_in_flight_and_does_not_replay_elapsed_time(self):
        preview = RG2Preview()
        preview.update(0, 50.6, None, False)
        closed = preview.update(0.02, 50.6, True, True)
        self.assertTrue(47.9 < closed < 48.5)
        narrower = preview.update(0.04, 50.6, None, True)
        self.assertLess(narrower, closed)
        opening = preview.update(0.06, 50.6, False, False)
        self.assertTrue(narrower < opening < 50.6)
        self.assertEqual(preview.update(0.08, 50.6, None, None), opening)
        after_delay = preview.update(2.0, 50.6, None, False)
        self.assertTrue(opening < after_delay <= opening + 133 * .05)
        preview.reset()
        self.assertEqual(preview.update(2.02, 38.0, None, False), 38.0)
        preview.reset(24.0)
        self.assertEqual(preview.update(2.04, None, None, False), 24.0)


class RG2VisualTest(unittest.TestCase):
    def setUp(self):
        scene = Path(__file__).resolve().parents[1] / "assets" / "scene.xml"
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        mujoco.mj_forward(self.model, self.data)

    def gap_mm(self):
        left = self.data.site("rg2_left_pad").xpos
        right = self.data.site("rg2_right_pad").xpos
        return 1000 * np.linalg.norm(left - right)

    def test_model_mount_follows_wrist_without_adding_dynamics(self):
        model, data = self.model, self.data
        self.assertEqual((model.nq, model.nv, model.nu), (6, 6, 6))
        self.assertAlmostEqual(sum(model.body_mass), 20.9949, places=5)
        for shoulder, wrist in ((0, 0), (0.3, -0.5)):
            data.joint("shoulder_pan_joint").qpos[0] = shoulder
            data.joint("wrist_3_joint").qpos[0] = wrist
            mujoco.mj_forward(model, data)
            flange, base = data.site("attachment_site"), data.body("rg2_base")
            np.testing.assert_allclose(base.xpos, flange.xpos, atol=1e-10)
            np.testing.assert_allclose(base.xmat, flange.xmat, atol=1e-10)

    def test_open_closed_and_partial_apertures_preserve_parallel_pads(self):
        self.assertAlmostEqual(self.gap_mm(), 104.0693, places=3)
        for width, expected_gap in ((0, 0), (55, 55), (100, 100), (OPEN_WIDTH_MM, 104.0693), (110.6, 104.0693)):
            set_rg2_width(self.model, width)
            mujoco.mj_forward(self.model, self.data)
            self.assertAlmostEqual(self.gap_mm(), expected_gap, places=3)
            left = self.data.site("rg2_left_pad").xmat.reshape(3, 3)
            right = self.data.site("rg2_right_pad").xmat.reshape(3, 3)
            np.testing.assert_allclose(left[:, 1], -right[:, 1], atol=1e-10)


class RG2MountTest(unittest.TestCase):
    def test_mount_offsets_move_the_visual_assembly_in_live_and_shadow(self):
        view = self.enterContext(SimBackend(headless=True))
        model, data = view.sim.model, view.sim.data
        q = view.joints()
        flange_xyz, flange_rot = view.pose()
        tcp = flange_xyz + flange_rot.apply([.01, 0, .13]), flange_rot
        load = ToolPayload(0.91, (0.001, -0.005, 0.083))
        view.set_payload(load)
        view.tick(dry_run=True, measured_q=q, measured_tcp=tcp, joint_target=q + .1, gripper_width_mm=55.)
        q_before = data.qpos.copy(), view._shadow.qpos.copy()
        cog_before = data.body("rg2_base").xipos.copy()
        tcp_before = view.preview_pose()
        inertia_before = data.qM.copy()
        cases = (
            (90., (0, 0, 0, 0, 0, 90)),  # Legacy flange clocking.
            ((10, -20, 45, 90, 30, -40), (10, -20, 45, 90, 30, -40)),
            (DEFAULT_VISUAL_MOUNT, DEFAULT_VISUAL_MOUNT),
        )
        for setting, (x, y, z, roll, pitch, yaw) in cases:
            view.set_rg2_mount(setting)
            flange_relative_rotation = (
                Rotation.from_euler("z", yaw, degrees=True)
                * Rotation.from_euler("y", pitch, degrees=True)
                * Rotation.from_euler("x", roll, degrees=True)
            )
            for state, q_expected in zip((data, view._shadow), q_before):
                flange = state.site("attachment_site")
                base = state.body("rg2_visual_mount")
                flange_axes = flange.xmat.reshape(3, 3)
                expected = flange_axes @ flange_relative_rotation.as_matrix()
                np.testing.assert_allclose(base.xmat.reshape(3, 3), expected, atol=1e-12)
                expected_xyz = flange.xpos + flange_axes @ (np.array([x, y, z]) / 1000)
                np.testing.assert_allclose(base.xpos, expected_xyz, atol=1e-12)
                separation = state.site("rg2_right_pad").xpos - state.site("rg2_left_pad").xpos
                np.testing.assert_allclose(separation, .055 * expected[:, 1], atol=1e-10)
                np.testing.assert_array_equal(state.qpos, q_expected)
            np.testing.assert_allclose(data.body("rg2_base").xipos, cog_before, atol=1e-12)
            np.testing.assert_allclose(data.qM, inertia_before, atol=1e-12)
            np.testing.assert_allclose(view.preview_pose()[0], tcp_before[0], atol=1e-12)
            self.assertLess((view.preview_pose()[1] * tcp_before[1].inv()).magnitude(), 1e-12)

    def test_invalid_mount_preserves_previous_transform(self):
        view = self.enterContext(SimBackend(headless=True))
        view.set_rg2_mount((5, 10, 15, 20, 30, 40))
        mount = view.sim.model.body("rg2_visual_mount")
        pos, quat = mount.pos.copy(), mount.quat.copy()
        for invalid in ([1, 2, 3], np.ones((2, 3)), [1, 2, 3, 4, np.nan, 0], np.inf):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                view.set_rg2_mount(invalid)
            np.testing.assert_array_equal(mount.pos, pos)
            np.testing.assert_array_equal(mount.quat, quat)
