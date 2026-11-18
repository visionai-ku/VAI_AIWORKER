import unittest
from pathlib import Path

import mujoco
import numpy as np

from vai_robotics.ur5e.rg2_visual import OPEN_WIDTH_MM, RG2Preview, set_rg2_width


class RG2PreviewTest(unittest.TestCase):
    def test_full_stroke_is_timed_and_stops_at_each_endpoint(self):
        for hz in (25, 50, 100):
            with self.subTest(hz=hz):
                preview = RG2Preview()
                preview.update(0, None, True, True)
                for i in range(1, 3 * hz + 1):
                    width = preview.update(i / hz, None, None, True)
                    self.assertAlmostEqual(width, max(0, 110 - 40 * i / hz))
                for i in range(1, 3 * hz + 1):
                    width = preview.update(3 + i / hz, None, False if i == 1 else None, False)
                    self.assertAlmostEqual(width, min(110, 40 * i / hz))

    def test_reverses_in_flight_and_does_not_replay_elapsed_time(self):
        preview = RG2Preview()
        preview.update(0, 50.6, None, False)
        self.assertAlmostEqual(preview.update(0.02, 50.6, True, True), 49.8)
        self.assertAlmostEqual(preview.update(0.04, 50.6, None, True), 49.0)
        self.assertAlmostEqual(preview.update(0.06, 50.6, False, False), 49.8)
        self.assertAlmostEqual(preview.update(0.08, 50.6, None, None), 49.8)
        self.assertAlmostEqual(preview.update(2.0, 50.6, None, False), 51.8)
        preview.reset()
        self.assertEqual(preview.update(2.02, 38.0, None, False), 38.0)


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
        for width, expected_gap in ((0, 0), (55, 55), (100, 100), (OPEN_WIDTH_MM, 104.0693)):
            set_rg2_width(self.model, width)
            mujoco.mj_forward(self.model, self.data)
            self.assertAlmostEqual(self.gap_mm(), expected_gap, places=3)
            left = self.data.site("rg2_left_pad").xmat.reshape(3, 3)
            right = self.data.site("rg2_right_pad").xmat.reshape(3, 3)
            np.testing.assert_allclose(left[:, 1], -right[:, 1], atol=1e-10)
