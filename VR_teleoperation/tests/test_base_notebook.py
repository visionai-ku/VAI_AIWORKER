"""Real base input gating and notebook loop, without GUI or robot sockets."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import glfw
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "vendor"):
    sys.path.insert(0, str(path))

from package.base_control import SG2_BASE_SPEED_SCALES, SG2BaseInputGate, SG2BaseTeleop


def base_notebook_path():
    """Support the merged layout and the original repository's retained path."""
    candidates = [
        ROOT / "notebooks/real/real_base_controller_sg2.ipynb",
        ROOT.parent / "ri_motion_v5_VR/project/ffw_sg2_vr_teleoperation/real_notebook"
        / "real_base_controller_sg2.ipynb",
    ]
    return next(path for path in candidates if path.is_file())


def notebook_loop():
    path = base_notebook_path()
    notebook = json.loads(path.read_text())
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code"
                  and "while env.is_viewer_alive()" in "".join(cell["source"]))
    return compile(source, str(path), "exec")


class Keyboard:
    def __init__(self, *keys):
        self.keys = set(keys)

    def is_key_pressed_repeat(self, *, key):
        return key in self.keys

    def is_key_pressed_once(self, *, key):
        return False


class BaseInputGateTest(unittest.TestCase):
    def setUp(self):
        self.gate = SG2BaseInputGate()
        self.teleop = SG2BaseTeleop()
        self.keyboard = Keyboard(glfw.KEY_W)

    def read(self, *, enabled=True, active=True):
        self.gate.read_keys(self.teleop, self.keyboard, enabled=enabled, active=active)

    def assert_stopped(self):
        np.testing.assert_array_equal(self.teleop.v_target, np.zeros(3))
        np.testing.assert_array_equal(self.teleop.v_cmd, np.zeros(3))

    def arm_and_release(self):
        self.read()
        self.keyboard.keys.clear()
        self.read()

    def test_enabling_with_held_keys_requires_release_then_fresh_input(self):
        self.read()
        self.read()
        self.assert_stopped()
        self.keyboard.keys.clear()
        self.read()
        self.assert_stopped()
        self.keyboard.keys.add(glfw.KEY_W)
        self.read()
        np.testing.assert_allclose(self.teleop.v_target, [.3, 0., 0.])

    def test_off_and_reenable_do_not_reuse_a_held_motion_key(self):
        self.arm_and_release()
        self.keyboard.keys.add(glfw.KEY_W)
        self.read()
        self.teleop.update(.1)
        self.assertGreater(self.teleop.v_cmd[0], 0.)
        self.read(enabled=False)
        self.assert_stopped()
        self.read()
        self.read()
        self.assert_stopped()

    def test_inactive_or_fault_session_stops_and_requires_a_new_release(self):
        self.arm_and_release()
        self.keyboard.keys.add(glfw.KEY_W)
        self.read()
        self.teleop.update(.1)
        self.read(active=False)
        self.assert_stopped()
        self.read(active=True)
        self.assert_stopped()
        self.keyboard.keys.clear()
        self.read()
        self.keyboard.keys.add(glfw.KEY_W)
        self.read()
        self.assertGreater(self.teleop.v_target[0], 0.)

    def test_diagonal_motion_uses_planar_norm_limit_at_each_speed(self):
        self.arm_and_release()
        self.keyboard.keys.update((glfw.KEY_W, glfw.KEY_A, glfw.KEY_Q))
        for scale in SG2_BASE_SPEED_SCALES.values():
            with self.subTest(scale=scale):
                self.teleop.scale = scale
                self.read()
                self.assertAlmostEqual(np.linalg.norm(self.teleop.v_target[:2]), .3 * scale)
                self.assertAlmostEqual(self.teleop.v_target[0], self.teleop.v_target[1])
                self.assertAlmostEqual(self.teleop.v_target[2], .6 * scale)


class BaseNotebookLoopTest(unittest.TestCase):
    def run_loop(self, frames, *, close_error=False):
        events = []
        teleop = SG2BaseTeleop()
        gate = SG2BaseInputGate()
        frame_index = -1
        options = {"Robot command": "ON", "Speed": "FAST", "Camera": "FREE"}

        def next_frame():
            nonlocal frame_index
            frame_index += 1
            return frame_index < len(frames)

        def set_option(label, value):
            options[label] = value
            events.append((frame_index, "option", label, value))

        def step(target, *, enabled):
            events.append((frame_index, "step", enabled, np.asarray(target).copy()))
            state = frames[frame_index].get("state", "ACTIVE") if enabled else "OFF"
            return {"state": state, "reason": "stub", "acknowledged_seq": 1}

        def close_client():
            events.append((frame_index, "client_close"))
            if close_error:
                raise RuntimeError("client close failed")

        client = SimpleNamespace(step=step, close=close_client)
        env = SimpleNamespace(
            is_viewer_alive=next_frame,
            is_key_pressed_repeat=lambda *, key: key in frames[frame_index].get("keys", ()),
            is_key_pressed_once=lambda *, key: (
                key == glfw.KEY_SPACE and frames[frame_index].get("space", False)),
            reset_wall_time=Mock(), reset_sim_time=Mock(),
            viewer=SimpleNamespace(set_cam_info=Mock()),
            plot_global_coordinate_axes=Mock(), viewer_text_overlay=Mock(), render=Mock(),
            close_viewer=lambda: events.append((frame_index, "viewer_close")),
        )
        widget = SimpleNamespace(
            isVisible=lambda: True,
            get_button_option=lambda *, label_text: options[label_text],
            set_button_option=set_option,
            close=lambda: events.append((frame_index, "widget_close")),
        )
        namespace = dict(
            glfw=glfw, teleop=teleop, input_gate=gate, env=env, ctrl_radio=widget,
            app=SimpleNamespace(processEvents=Mock()),
            SimpleTimer=lambda **kw: SimpleNamespace(start=Mock(), do_run=lambda: True, end=Mock()),
            SG2BaseCommandClient=lambda **kw: client, SG2_BASE_SPEED_SCALES=SG2_BASE_SPEED_SCALES,
            ROBOT_IP="192.0.2.1", SUB_FROM_SG2_PORT=5560, PUB_TO_SG2_PORT=5561, SEND_HZ=30,
            base_z=.01,
            apply_sg2_base_to_env=lambda env, current, base_z: events.append(
                (frame_index, "motion_tick", current.v_cmd.copy())),
            draw_sg2_base=Mock(),
            make_sg2_base_overlay_text=lambda *args, **kw: "offline test",
        )
        if close_error:
            with self.assertRaisesRegex(RuntimeError, "client close failed"):
                exec(notebook_loop(), namespace)
        else:
            exec(notebook_loop(), namespace)
        return events, teleop, options

    def test_space_switches_off_and_sends_disabled_zero_before_next_motion_tick(self):
        events, teleop, options = self.run_loop([
            {}, {}, {"keys": (glfw.KEY_W,)}, {"keys": (glfw.KEY_W,), "space": True},
            {"keys": (glfw.KEY_W,)},
        ])
        moving = [event for event in events if event[0] == 2 and event[1] == "step"]
        self.assertEqual(len(moving), 1)
        self.assertTrue(moving[0][2])
        self.assertGreater(moving[0][3][0], 0.)
        stopped = [event for event in events if event[0] == 3]
        self.assertEqual(stopped[0], (3, "option", "Robot command", "OFF"))
        self.assertEqual(stopped[1][:3], (3, "step", False))
        self.assertEqual(stopped[2][1], "motion_tick")
        np.testing.assert_array_equal(stopped[2][2], np.zeros(3))
        for event in events:
            if event[0] >= 3 and event[1] == "step":
                self.assertFalse(event[2])
                np.testing.assert_array_equal(event[3], np.zeros(3))
        self.assertEqual(options["Robot command"], "OFF")
        np.testing.assert_array_equal(teleop.v_cmd, np.zeros(3))

    def test_fault_switches_off_and_following_held_keys_cannot_send_motion(self):
        events, teleop, options = self.run_loop([
            {}, {}, {"keys": (glfw.KEY_W,), "state": "FAULT"}, {"keys": (glfw.KEY_W,)},
        ])
        self.assertIn((2, "option", "Robot command", "OFF"), events)
        for event in events:
            if event[0] == 3 and event[1] == "step":
                self.assertFalse(event[2])
                np.testing.assert_array_equal(event[3], np.zeros(3))
        self.assertEqual(options["Robot command"], "OFF")
        np.testing.assert_array_equal(teleop.v_cmd, np.zeros(3))

    def test_widget_and_viewer_close_even_when_client_close_raises(self):
        events, teleop, _ = self.run_loop([], close_error=True)
        self.assertEqual([event[1] for event in events],
                         ["client_close", "widget_close", "viewer_close"])
        np.testing.assert_array_equal(teleop.v_cmd, np.zeros(3))


if __name__ == "__main__":
    unittest.main()
