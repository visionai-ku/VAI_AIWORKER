import unittest
from pathlib import Path

import mujoco
import numpy as np

from vai_robotics.ur5e.rg2_visual import OPEN_WIDTH_MM, set_rg2_width


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
