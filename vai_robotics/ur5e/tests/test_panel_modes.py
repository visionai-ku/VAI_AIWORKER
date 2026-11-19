import io
import json
import math
import os
from queue import Queue
import time
import unittest
from unittest.mock import Mock, patch

from vai_robotics.ur5e.teleop_panel import TeleopPanel, _PanelWindow


class CommandModeProtocolTest(unittest.TestCase):
    def test_only_targets_for_the_acknowledged_mode_reach_control(self):
        panel = TeleopPanel.__new__(TeleopPanel)
        panel._messages = Queue()
        panel._open = True
        panel._process = Mock()
        panel._process.poll.return_value = None
        panel._process.stdin = io.StringIO()
        panel._can_send = False  # A telemetry frame is still in flight.
        panel._command_mode = "eef-delta"
        panel._control_law = "position"
        target = dict(command_mode="eef-delta", control_law="position", values=[0.001] * 6, time=time.monotonic())

        # A previous click and a picker edit can arrive in a single polling cycle.
        panel._messages.put({"target": target})
        panel._messages.put({"command_mode": "joint-absolute"})
        self.assertIsNone(panel.poll())
        self.assertEqual(panel.take_command_mode(), "joint-absolute")
        self.assertIsNone(panel.take_command_mode())
        self.assertIsNone(panel.take_target())

        panel.command_mode_result(command_mode="joint-absolute")
        self.assertEqual(json.loads(panel._process.stdin.getvalue()),
                         {"command_mode_result": {"command_mode": "joint-absolute", "message": ""}})
        for mode in (None, "eef-delta", "joint-absolute"):
            with self.subTest(mode=mode):
                event = {key: value for key, value in target.items() if key != "command_mode"}
                if mode is not None:
                    event["command_mode"] = mode
                panel._messages.put({"target": event})
                panel.poll()
                self.assertEqual(panel.take_target(), event["values"] if mode == "joint-absolute" else None)

        # With the same coordinate mode, targets must still match the newly confirmed law.
        panel._messages.put({"target": dict(target, command_mode="joint-absolute")})
        panel._messages.put({"control_law": "impedance"})
        self.assertIsNone(panel.poll())
        self.assertEqual(panel.take_control_law(), "impedance")
        self.assertIsNone(panel.take_control_law())
        self.assertIsNone(panel.take_target())
        panel.control_law_result(control_law="impedance")
        self.assertEqual(json.loads(panel._process.stdin.getvalue().splitlines()[-1]),
                         {"control_law_result": {"control_law": "impedance", "message": ""}})
        for law in (None, "position", "impedance"):
            with self.subTest(law=law):
                event = dict(target, command_mode="joint-absolute", time=time.monotonic())
                if law is None:
                    del event["control_law"]
                else:
                    event["control_law"] = law
                panel._messages.put({"target": event})
                panel.poll()
                self.assertEqual(panel.take_target(), event["values"] if law == "impedance" else None)


class CommandPickerTest(unittest.TestCase):
    def setUp(self):
        with patch.dict(os.environ, QT_QPA_PLATFORM="offscreen"):
            self.panel = _PanelWindow(simulation=True, device_name="spacemouse", dry_run=True,
                                      open_width_mm=75.0, command_mode="eef-delta")
        self.addCleanup(self.panel.close)
        self.joints = [5.0, -96.0, 37.0, 16.0, 24.0, 370.0]
        self.pose = ([154.0, 212.0, -32.0], [5.0, -3.0, 7.0])

    def refresh(self):
        self.panel.refresh(dry_run=True, clutch="놓음", measured=self.pose,
                           measured_joints=self.joints, target=None)

    def select_mode(self, command_mode):
        picker = self.panel.command_picker
        index = picker.findData(command_mode)
        self.assertGreaterEqual(index, 0)
        picker.setCurrentIndex(index)
        self.assertEqual(self.panel.take_command_mode(), command_mode)
        self.assertIsNone(self.panel.take_command_mode())

    def test_switch_to_joint_absolute_clears_old_target_and_reloads_new_units(self):
        panel = self.panel
        self.refresh()
        panel.target_inputs[0].setValue(12.5)
        panel.apply_target.click()
        self.select_mode("joint-absolute")
        self.assertIsNone(panel.take_target_event())
        self.refresh()  # Feedback queued before the controller commits the switch.
        self.assertFalse(panel.apply_target.isEnabled())
        panel.command_mode_result(command_mode="joint-absolute")
        self.assertEqual(panel.command_picker.currentData(), "joint-absolute")
        self.assertFalse(panel.apply_target.isEnabled())
        self.refresh()
        self.assertTrue(panel.load_target.isVisible())
        self.assertFalse(panel.apply_target.isEnabled())
        panel.load_target.click()
        self.assertEqual([field.value() for field in panel.target_inputs], self.joints)
        panel.apply_target.click()
        event = panel.take_target_event()
        self.assertEqual(event["command_mode"], "joint-absolute")
        for value, degrees in zip(event["values"], self.joints):
            self.assertAlmostEqual(value, math.radians(degrees))

        self.select_mode("eef-delta")
        panel.command_mode_result(command_mode="eef-delta")
        self.refresh()
        self.assertFalse(panel.load_target.isVisible())
        self.assertEqual([field.value() for field in panel.target_inputs], [0.0] * 6)
        self.assertEqual(panel.measured_cells[0].text(), "154.0")
        self.assertIsNone(panel.take_target_event())

    def test_rejected_switch_restores_confirmed_picker_and_displays_reason(self):
        self.refresh()
        self.select_mode("joint-delta")
        self.panel.command_mode_result(command_mode="eef-delta", message="클러치를 놓으세요.")
        self.assertEqual(self.panel.command_picker.currentData(), "eef-delta")
        self.assertTrue(self.panel.command_picker.isEnabled())
        self.assertEqual(self.panel.message_label.text(), "클러치를 놓으세요.")
        self.select_mode("eef-absolute")

    def test_law_picker_waits_for_ack_and_requires_new_absolute_measurement_on_success(self):
        panel = self.panel
        self.select_mode("joint-absolute")
        panel.command_mode_result(command_mode="joint-absolute")
        self.refresh()
        panel.load_target.click()
        panel.target_inputs[0].setValue(60)
        picker = panel.control_law_picker
        for confirmed in ("position", "impedance"):
            panel.apply_target.click()
            picker.setCurrentIndex(picker.findData("impedance"))
            self.assertEqual(panel.take_control_law(), "impedance")
            self.assertIsNone(panel.take_control_law())
            self.assertIsNone(panel.take_target_event())
            self.assertEqual(picker.currentData(), "position")
            self.refresh()  # Old-law feedback can arrive during the request.
            self.assertIn("전환 중", panel.state_label.text())
            for control in (picker, panel.command_picker, panel.run_button, panel.apply_target, panel.load_target):
                self.assertFalse(control.isEnabled())
            refusal = "클러치를 놓으세요." if confirmed == "position" else ""
            panel.control_law_result(control_law=confirmed, message=refusal)
            self.assertEqual(picker.currentData(), confirmed)
            self.assertTrue(picker.isEnabled())
            if refusal:
                self.assertEqual(panel.message_label.text(), refusal)
                self.assertEqual(panel.target_inputs[0].value(), 60)
                self.assertTrue(panel.apply_target.isEnabled())

        self.assertEqual(panel.command_picker.currentData(), "joint-absolute")
        self.assertEqual(panel.measured_cells[0].text(), "—")
        self.assertEqual([field.value() for field in panel.target_inputs], [0.0] * 6)
        self.assertFalse(panel.load_target.isEnabled())
        self.refresh()
        self.assertTrue(panel.load_target.isEnabled())
        self.assertFalse(panel.apply_target.isEnabled())
        panel.load_target.click()
        panel.apply_target.click()
        target = panel.take_target_event()
        self.assertEqual(target["command_mode"], "joint-absolute")
        self.assertEqual(target["control_law"], "impedance")
        self.assertAlmostEqual(target["values"][0], math.radians(self.joints[0]))
