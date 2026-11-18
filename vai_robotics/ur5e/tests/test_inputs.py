import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from vai_robotics.ur5e.inputs import QuestInput, SpaceMouseInput


def quest_frame(seq, now, grip, x=0.0, angle=0.0, origin=0, tracked=True):
    quaternion = Rotation.from_rotvec([0, angle, 0]).as_quat().tolist()
    hand = dict(grip=grip, trigger=0, position_m=[x, 0, 0], quaternion_xyzw=quaternion,
                valid=tracked, emulated=False)
    return dict(version=1, session_id="headset", seq=seq, origin_id=origin,
                received_at=now, sample_time_ms=now * 1000, visibility="visible",
                head=dict(quaternion_xyzw=[0, 0, 0, 1]), hands={"left": {"valid": False}, "right": hand})


class QuestInputTests(unittest.TestCase):
    def setUp(self):
        self.device = QuestInput("localhost")
        self.device.receiver = Mock()

    def sample(self, frame, now=None):
        self.device.receiver.latest.return_value = frame
        with patch("time.monotonic", return_value=now or frame["received_at"]):
            return self.device.read()

    def test_one_hand_and_repeated_frame(self):
        self.assertIsNone(self.sample(quest_frame(0, 1, 1)))
        self.assertFalse(self.sample(quest_frame(1, 1.01, 0)).enabled)
        first = self.sample(quest_frame(2, 1.02, 1))
        self.assertTrue(first.enabled)
        np.testing.assert_array_equal(first.translation, 0)
        frame = quest_frame(3, 1.03, 1, x=0.02, angle=0.12)
        moved = self.sample(frame)
        np.testing.assert_allclose(moved.translation, [0, 0.02, 0], atol=1e-12)
        np.testing.assert_allclose(moved.rotation, [0, 0, 0.12], atol=1e-12)
        duplicate = self.sample(frame, now=1.04)
        self.assertTrue(duplicate.enabled)
        np.testing.assert_allclose(duplicate.translation, 0, atol=1e-12)
        np.testing.assert_allclose(duplicate.rotation, 0, atol=1e-12)
        self.assertIsNone(self.sample(frame, now=1.25))
        self.assertIsNone(self.sample(quest_frame(4, 1.26, 1, x=0.5)))
        self.assertFalse(self.sample(quest_frame(5, 1.27, 0, x=0.5)).enabled)
        regripped = self.sample(quest_frame(6, 1.28, 1, x=0.5))
        np.testing.assert_array_equal(regripped.translation, 0)

    def test_reference_reset_loss_and_long_receive_gap_require_release(self):
        self.sample(quest_frame(0, 1, 0))
        self.sample(quest_frame(1, 1.01, 1))
        self.assertIsNone(self.sample(quest_frame(2, 1.21, 1)))
        self.assertFalse(self.sample(quest_frame(3, 1.22, 0)).enabled)
        self.assertTrue(self.sample(quest_frame(4, 1.23, 1)).enabled)
        self.assertIsNone(self.sample(quest_frame(5, 1.24, 1, origin=1)))
        self.assertIsNone(self.sample(quest_frame(6, 1.25, 1, origin=1)))
        self.assertFalse(self.sample(quest_frame(7, 1.26, 0, origin=1)).enabled)
        self.assertIsNone(self.sample(quest_frame(8, 1.27, 1, origin=1, tracked=False)))
        self.assertIsNone(self.sample(quest_frame(9, 1.28, 1, origin=1)))


def hid_report(t, button=0, x=0, y=0, pitch=0, button_event=True):
    return SimpleNamespace(t=t, buttons=[button, 0], button_event=button_event,
                           x=x, y=y, z=0, roll=0, pitch=pitch, yaw=0)


class SpaceMouseTests(unittest.TestCase):
    def setUp(self):
        self.hid = Mock()
        self.driver = Mock()
        self.driver.get_connected_devices.return_value = ["SpaceMouse Compact"]
        self.driver.open.return_value = self.hid
        self.enterContext(patch("vai_robotics.ur5e.inputs._load_spacemouse",
                                return_value=(self.driver, ConnectionError)))
        self.device = self.enterContext(SpaceMouseInput())
        self.button_callback = self.driver.open.call_args.kwargs["button_callback"]

    def sample(self, report, now, earlier=()):
        queue = list(earlier) + [report]

        def read():
            if queue:
                item = queue.pop(0)
                if item.button_event and item.t != self.device.report_time:
                    self.button_callback(item, item.buttons)
                return item
            return report

        self.hid.read.side_effect = read
        with patch("time.perf_counter", return_value=now):
            return self.device.read()

    def test_velocity_to_position_and_rotation_at_different_rates(self):
        self.assertFalse(self.sample(hid_report(1), 1).enabled)
        for dt in (0.01, 0.02):
            now = self.device.last_tick + dt
            motion = self.sample(hid_report(now, button=1, x=1, y=0.5, pitch=0.5), now)
            np.testing.assert_allclose(motion.translation / dt, [-0.04, 0.08, 0])
            np.testing.assert_allclose(motion.rotation / dt, [-0.25, 0, 0])

    def test_cached_active_report_expires_and_release_is_processed_immediately(self):
        self.assertIsNone(self.sample(hid_report(1, button=1), 1))
        self.assertFalse(self.sample(hid_report(1.01), 1.01).enabled)
        held = hid_report(1.02, button=1, x=1)
        self.assertTrue(self.sample(held, 1.02).enabled)
        self.assertIsNone(self.sample(held, 1.18))
        self.assertIsNone(self.sample(hid_report(1.19, button=1), 1.19))
        released = self.sample(hid_report(1.20), 1.20, earlier=[hid_report(1.195, button=1)])
        self.assertFalse(released.enabled)
        self.assertTrue(self.sample(hid_report(1.21, button=1), 1.21).enabled)

    def test_synthetic_initial_button_state_and_release_between_ticks(self):
        self.assertIsNone(self.sample(hid_report(-1, button_event=False), 1))
        self.assertIsNone(self.sample(hid_report(1.01, x=0.9, button_event=False), 1.01))
        self.assertIsNone(self.sample(hid_report(1.02, button=1), 1.02))
        self.assertFalse(self.sample(hid_report(1.03), 1.03).enabled)
        self.assertTrue(self.sample(hid_report(1.04, button=1), 1.04).enabled)
        pressed = hid_report(1.06, button=1)
        observed = self.sample(pressed, 1.06, earlier=[hid_report(1.05)])
        self.assertFalse(observed.enabled)
        self.assertTrue(self.sample(pressed, 1.07).enabled)

    def test_usb_wait_throttles_retries_and_unplug_requires_fresh_release(self):
        self.driver.reset_mock()
        self.driver.get_connected_devices.return_value = []
        clock = self.enterContext(patch("time.monotonic", return_value=0.0))
        self.device = self.enterContext(SpaceMouseInput(wait_for_device=True))
        self.assertIn("대기", self.device.status)
        for _ in range(5):
            self.assertIsNone(self.device.read())
        self.driver.get_connected_devices.assert_called_once()
        self.driver.get_connected_devices.return_value = ["SpaceMouse Compact"]
        clock.return_value = 1.1
        self.assertIsNone(self.device.read())
        self.assertFalse(self.device.status)
        self.button_callback = self.driver.open.call_args.kwargs["button_callback"]
        self.assertFalse(self.sample(hid_report(2), 2).enabled)
        self.assertTrue(self.sample(hid_report(2.01, button=1), 2.01).enabled)
        self.hid.read.side_effect = ConnectionError("device removed")
        self.assertIsNone(self.device.read())
        self.hid.close.assert_called_once()
        self.assertIn("대기", self.device.status)
        clock.return_value = 2.2
        self.assertIsNone(self.device.read())
        self.assertIsNone(self.sample(hid_report(3, button=1), 3))
        self.assertFalse(self.sample(hid_report(3.01), 3.01).enabled)
        self.assertTrue(self.sample(hid_report(3.02, button=1), 3.02).enabled)

    def test_missing_device_fails_fast_without_panel_and_permissions_fail_with_panel(self):
        self.driver.get_connected_devices.return_value = []
        with self.assertRaisesRegex(RuntimeError, "No connected SpaceMouse"):
            with SpaceMouseInput():
                self.fail("started without a device")
        self.driver.get_connected_devices.return_value = ["SpaceMouse Compact"]
        self.driver.open.side_effect = PermissionError("access denied")
        with self.assertRaisesRegex(RuntimeError, "access denied"):
            with SpaceMouseInput(wait_for_device=True):
                self.fail("silently waiting for a connected but inaccessible device")
