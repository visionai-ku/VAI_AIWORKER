import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from vai_robotics.ur5e.controllers import TeleopController
from vai_robotics.ur5e.inputs import QuestInput, SpaceMouseInput
from vai_robotics.ur5e.teleop import operator_state, run_teleop
from vai_robotics.ur5e.teleop_types import InputSample
from vai_robotics.ur5e.tests.test_inputs import quest_frame


class OperatorStatusTest(unittest.TestCase):
    def setUp(self):
        self.backend = Mock()
        self.backend.pose.return_value = np.zeros(3), Rotation.identity()

    def test_runner_reports_usb_connection_initialization_release_and_actual_engagement(self):
        panel = Mock(is_open=True, scale=1.0, open_width_mm=75.0)
        panel.poll.return_value = panel.take_target.return_value = panel.take_command_mode.return_value = None
        device = SpaceMouseInput()
        handle = object()
        released = InputSample(np.zeros(3), np.zeros(3), False)
        held = InputSample(np.zeros(3), np.zeros(3), True)
        # The HID button level can remain known when no valid arm sample is available.
        observations = iter([(None, None, None), (handle, None, None), (handle, True, None),
                             (handle, False, released), (handle, True, held), (handle, True, None),
                             (handle, False, released), (None, None, None)])
        expected = ["waiting-device", "waiting-clutch", "release-clutch", "ready", "active",
                    "release-clutch", "ready", "waiting-device"]

        def read():
            device.device, device._clutch_pressed, sample = next(observations)
            return sample

        self.backend.is_running.side_effect = [True] * len(expected) + [False]
        with patch.object(device, "read", side_effect=read), patch("vai_robotics.ur5e.teleop.time") as clock:
            clock.monotonic.side_effect = lambda: panel.refresh.call_count * 0.02
            run_teleop(device, self.backend, sim=True, panel=panel, dry_run=True)
        self.assertEqual([call.kwargs["operator_state"] for call in panel.refresh.call_args_list], expected)
        self.backend.command.assert_not_called()

    def test_quest_release_interlock_and_tracking_loss_have_distinct_states(self):
        device = QuestInput("localhost")
        device.receiver = Mock()
        controller = TeleopController(self.backend, dry_run=True)
        held_frame = quest_frame(2, 1.04, 1)
        for now, frame, expected in (
            (0.90, None, "waiting-input"),
            (1.00, quest_frame(0, 1.00, 1), "release-clutch"),
            (1.02, quest_frame(1, 1.02, 0), "ready"),
            (1.04, held_frame, "active"),
            (1.20, held_frame, "waiting-input"),
            (1.22, quest_frame(3, 1.22, 1), "release-clutch"),
            (1.24, quest_frame(4, 1.24, 0), "ready"),
        ):
            with self.subTest(now=now):
                device.receiver.latest.return_value = frame
                with patch("time.monotonic", return_value=now):
                    sample = device.read()
                controller.update(sample, now)
                self.assertEqual(operator_state(device, sample, controller), expected)
