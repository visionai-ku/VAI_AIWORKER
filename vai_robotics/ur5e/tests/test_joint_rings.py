import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import glfw
import mujoco
import numpy as np

from vai_robotics.ur5e.joint_rings import JointRings, _nearest_ring
from vai_robotics.ur5e.sim import load_simulation
from vai_robotics.ur5e.teleop_types import JointDrag


class JointRingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = load_simulation()

    def setUp(self):
        self.data = self.sim.data
        eye = SimpleNamespace(pos=np.array([0., 0., 2.]), frustum_center=0.)
        self.sim.viewer = Mock(window=object(), scn=SimpleNamespace(camera=[eye, eye]), vopt=mujoco.MjvOption())
        self.sim.plot_cylinder_fr2to = Mock()
        self.sim.viewer_text_overlay = Mock()
        self.cursor = np.array([400., 300.])
        self.button = glfw.RELEASE
        self.focused = True
        self.old_button = Mock()
        callbacks = {
            "set_mouse_button_callback": {"return_value": self.old_button},
            "set_window_focus_callback": {"return_value": None},
            "set_cursor_enter_callback": {"return_value": None},
            "get_window_size": {"return_value": (800, 600)},
            "get_framebuffer_size": {"return_value": (1600, 1200)},
            "get_window_attrib": {"side_effect": lambda *args: self.focused},
            "get_cursor_pos": {"side_effect": lambda *_: self.cursor},
            "get_mouse_button": {"side_effect": lambda *_: self.button},
        }
        for name, options in callbacks.items():
            self.enterContext(patch.object(glfw, name, **options))
        self.rings = JointRings(self.sim)
        camera = dict(W=1600, H=1200, fwd=np.array([0., 0., -1.]), up=np.array([0., 1., 0.]),
                      right=np.array([1., 0., 0.]), near=.1, top=.05, bottom=-.05, half_w=.067)
        self.rings.camera._begin_frame = lambda: dict(camera)
        self.assertEqual(self.rings.update(self.data, 1), JointDrag(0, None))

    def edge(self, action):
        self.button = action
        self.rings._button(self.rings.window, glfw.MOUSE_BUTTON_LEFT, action, 0)

    def point_at_ring(self, joint):
        for pixel in self.rings._pixels[joint]:
            if self.rings._pick(pixel) == joint:
                self.cursor = pixel / 2
                return
        self.fail(f"No pickable point for ring {joint}")

    def test_displayed_joint_axis_drag_is_exclusive_and_never_writes_joint_state(self):
        q_before = self.data.qpos.copy()
        for axis, center, ring in zip(self.data.xaxis, self.data.xanchor, self.rings._points):
            np.testing.assert_allclose((ring - ring.mean(axis=0)) @ axis, 0., atol=1e-12)
            np.testing.assert_allclose(np.cross(ring.mean(axis=0) - center, axis), 0., atol=1e-12)
        self.point_at_ring(0)
        self.edge(glfw.PRESS)
        first = self.rings.update(self.data, 1)
        self.assertEqual(first, JointDrag(1, 0, q_before[0]))
        center = self.rings._center / 2.
        x, y = self.cursor - center
        self.cursor = center + [-y, x]
        moved = self.rings.update(self.data, 1)
        self.assertAlmostEqual(moved.angle, q_before[0] - math.pi / 2)
        self.rings.draw()
        self.assertIn("shoulder_pan_joint", self.sim.viewer_text_overlay.call_args.args[0])
        self.edge(glfw.RELEASE)
        self.assertEqual(self.rings.update(self.data, 1), JointDrag(1, None))
        self.old_button.assert_not_called()
        np.testing.assert_array_equal(self.data.qpos, q_before)

    def test_epoch_focus_and_cursor_exit_block_held_press_and_regrips_have_unique_serials(self):
        self.point_at_ring(0)
        self.edge(glfw.PRESS)
        self.assertIsNone(self.rings.update(self.data, 2))
        self.edge(glfw.RELEASE)
        self.edge(glfw.PRESS)
        self.assertEqual(self.rings.update(self.data, 2).serial, 2)
        self.rings._enter(self.rings.window, False)  # Exit and return between two renders.
        self.rings._enter(self.rings.window, True)
        self.assertIsNone(self.rings.update(self.data, 2))
        self.edge(glfw.RELEASE)
        self.edge(glfw.PRESS)
        self.rings._focus(self.rings.window, False)
        self.assertIsNone(self.rings.update(self.data, 2))
        self.edge(glfw.RELEASE)
        self.edge(glfw.PRESS)
        self.assertEqual(self.rings.update(self.data, 2).serial, 4)
        self.assertIsNone(self.rings.update(self.data, 0))

    def test_press_off_ring_cannot_be_stolen_from_camera(self):
        self.cursor[:] = (2., 2.)
        self.edge(glfw.PRESS)
        self.point_at_ring(0)
        self.assertIsNone(self.rings.update(self.data, 1))
        self.old_button.assert_called_once()
        self.edge(glfw.RELEASE)
        self.assertEqual(self.rings.update(self.data, 1), JointDrag(0, None))

    def test_joint_limit_clips_drag_and_reversing_moves_off_limit(self):
        self.point_at_ring(0)
        self.rings._q[0] = self.rings.limits[0, 1] - .01
        self.edge(glfw.PRESS)
        center = self.rings._center / 2
        x, y = self.cursor - center
        self.cursor = center + [y, -x]
        self.assertAlmostEqual(self.rings.update(self.data, 1).angle, self.rings.limits[0, 1])
        self.cursor = center + [x, y]
        self.assertAlmostEqual(self.rings.update(self.data, 1).angle, self.rings.limits[0, 1] - math.pi / 2)

    def test_all_six_rings_can_be_picked_from_the_default_teleop_camera(self):
        model = self.sim.model
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.azimuth, cam.elevation, cam.distance, cam.lookat[:] = 120., -20., 1.9, [0., .17, .35]
        self.sim.viewer = SimpleNamespace(window=object(), viewport=mujoco.MjrRect(0, 0, 1600, 1200),
                                          model=model, data=self.data,
                                          cam=cam, scn=mujoco.MjvScene(model, maxgeom=1000),
                                          vopt=mujoco.MjvOption(), pert=mujoco.MjvPerturb(), _scale=2.)
        self.rings = JointRings(self.sim)
        self.assertEqual(self.rings.update(self.data, 1), JointDrag(0, None))
        for joint in range(6):
            self.point_at_ring(joint)
            self.edge(glfw.PRESS)
            self.assertEqual(self.rings.update(self.data, 1).joint, joint)
            self.edge(glfw.RELEASE)

    def test_pick_nearest_drawn_segment_with_front_depth_tiebreak(self):
        square = np.array([[10., 10.], [20., 10.], [20., 20.], [10., 20.]])
        rings = np.array([square, square, square + 2])
        depth = np.array([3., 2., 1.])
        self.assertEqual(_nearest_ring(rings, (15, 10), depth, 1.), 1)
        self.assertEqual(_nearest_ring(rings, (15, 10), depth, 1.,
                                       visible=lambda j, segment, t: j == 0), 0)
        self.assertEqual(_nearest_ring(rings, (15, 12), depth, 1.), 2)
        self.assertIsNone(_nearest_ring(rings, (0, 0), depth, 1.))
