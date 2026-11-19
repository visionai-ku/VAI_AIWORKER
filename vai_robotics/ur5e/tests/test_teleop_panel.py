import os
from queue import Queue
import time
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from vai_robotics.ur5e.rg2 import DEFAULT_OPEN_WIDTH_MM
from vai_robotics.ur5e.teleop_panel import TeleopPanel, _display_pose, _PanelWindow


class TeleopPanelTest(unittest.TestCase):
    def make_panel(self, simulation=False, open_width_mm=DEFAULT_OPEN_WIDTH_MM, **settings):
        with patch.dict(os.environ, QT_QPA_PLATFORM="offscreen"):
            panel = _PanelWindow(simulation=simulation, device_name="spacemouse", dry_run=True,
                                 open_width_mm=open_width_mm, **settings)
        self.addCleanup(panel.close)
        return panel

    def test_real_button_stays_pending_until_controller_accepts_or_refuses(self):
        panel = self.make_panel()
        panel.run_button.click()
        self.assertIs(panel.take_request(), False)
        self.assertTrue(panel.dry_button.isChecked())
        self.assertIsNone(panel.take_request())
        self.assertIn("전환 중", panel.message_label.text())
        panel.refresh(dry_run=True, clutch="클러치 해제", measured=None, target=None, message="실기 연결 대기")
        self.assertIn("전환 중", panel.message_label.text())
        self.assertTrue(panel.dry_button.isChecked())
        panel.run_button.click()
        self.assertIsNone(panel.take_request())
        panel.mode_result(dry_run=True, message="클러치를 놓으세요")
        self.assertTrue(panel.dry_button.isChecked())
        self.assertEqual(panel.message_label.text(), "클러치를 놓으세요")

        panel.run_button.click()
        self.assertIs(panel.take_request(), False)
        panel.mode_result(dry_run=False)
        self.assertTrue(panel.run_button.isChecked())
        self.assertFalse(panel.message_label.isVisible())
        panel.dry_button.click()
        self.assertIs(panel.take_request(), True)

    def test_sim_input_gain_tcp_units_and_window_close(self):
        panel = self.make_panel(simulation=True)
        self.assertIn("SIM RUN", panel.run_button.text())
        self.assertEqual(panel.scale, 1)
        panel.gain.setValue(35)
        self.assertAlmostEqual(panel.scale, 0.35)
        measured = np.array([0.1, 0.2, 0.3]), Rotation.from_euler("z", 30, degrees=True)
        panel.refresh(dry_run=True, clutch="클러치 해제", measured=_display_pose(measured), target=None)
        self.assertEqual([cell.text() for cell in panel.measured_cells[:3]], ["100.0", "200.0", "300.0"])
        self.assertEqual(panel.measured_cells[5].text(), "30.0")
        self.assertEqual(panel.target_cells[0].text(), "—")
        self.assertTrue(panel.is_open)
        panel.close()
        self.assertFalse(panel.is_open)

    def test_numeric_target_is_sent_once_only_on_apply_and_in_si_units(self):
        for space in ("eef", "joint"):
            for style in ("delta", "absolute"):
                with self.subTest(space=space, style=style):
                    panel = self.make_panel(command_mode=f"{space}-{style}", control_law="impedance")
                    self.assertFalse(panel.apply_target.isEnabled())
                    measured_tcp = ([120, -45, 300], [5, -15, 20])
                    measured_q = [-10, -90, 100, -35, 45, 370]
                    panel.refresh(dry_run=True, clutch="누름", measured=measured_tcp, target=None,
                                  measured_joints=measured_q)
                    if style == "absolute":
                        self.assertFalse(panel.apply_target.isEnabled())
                        panel.load_target.click()
                        expected = measured_q if space == "joint" else [*measured_tcp[0], *measured_tcp[1]]
                        self.assertEqual([box.value() for box in panel.target_inputs], expected)
                    else:
                        self.assertEqual([box.value() for box in panel.target_inputs], [0] * 6)
                    for box, value in zip(panel.target_inputs, (1, 2, 3, 4, 5, 6)):
                        box.setValue(value)
                    self.assertIsNone(panel.take_target_event())
                    panel.apply_target.click()
                    values = panel.take_target_event()["values"]
                    expected = np.deg2rad(np.arange(1, 7))
                    if space == "eef":
                        expected[:3] = [0.001, 0.002, 0.003]
                    np.testing.assert_allclose(values, expected)
                    self.assertIsNone(panel.take_target_event())
                    panel.apply_target.click()
                    panel.run_button.click()
                    self.assertIsNone(panel.take_target_event())
                    self.assertFalse(panel.apply_target.isEnabled())
                    panel.mode_result(dry_run=False)
                    self.assertFalse(panel.apply_target.isEnabled())
                    panel.close()

    def test_control_loop_drops_stale_targets_and_targets_during_mode_switch(self):
        panel = TeleopPanel.__new__(TeleopPanel)
        panel._messages = Queue()
        panel._open = True
        event = dict(time=9.95, values=[0.001] * 6)
        with patch("vai_robotics.ur5e.teleop_panel.time.monotonic", return_value=10.0):
            panel._messages.put({"target": dict(event, time=9.0)})
            panel.poll()
            self.assertIsNone(panel.take_target())
            panel._messages.put({"target": event})
            panel._messages.put({"mode": False})
            self.assertIs(panel.poll(), False)
            self.assertIsNone(panel.take_target())
            panel._messages.put({"target": event})
            panel.poll()
            self.assertEqual(panel.take_target(), event["values"])
            self.assertIsNone(panel.take_target())

    def test_open_width_edits_commit_in_tenths_of_a_millimeter(self):
        from PyQt5.QtCore import Qt
        from PyQt5.QtTest import QTest

        panel = self.make_panel(open_width_mm=81.3)
        control = panel.open_width
        self.assertEqual(control.accessibleName(), "열림 목표 폭")
        self.assertEqual(control.suffix(), " mm")
        self.assertEqual(panel.open_width_mm, 81.3)
        control.setFocus()
        control.selectAll()
        QTest.keyClicks(control, "54.7")
        self.assertEqual(panel.open_width_mm, 81.3)
        QTest.keyClick(control, Qt.Key_Return)
        self.assertEqual(panel.open_width_mm, 54.7)
        control.stepUp()
        self.assertEqual(panel.open_width_mm, 54.8)
        control.setValue(-0.1)
        self.assertEqual(panel.open_width_mm, 0.0)
        control.setValue(110.1)
        self.assertEqual(panel.open_width_mm, 110.0)

    def test_separate_panel_process_consumes_frame_and_exits_on_pipe_eof(self):
        with patch.dict(os.environ, QT_QPA_PLATFORM="offscreen"):
            panel = TeleopPanel(simulation=True, device_name="SpaceMouse", dry_run=True)
        self.addCleanup(panel.close)
        self.assertEqual(panel.open_width_mm, 75.0)
        if os.name == "posix":
            self.assertNotEqual(os.getpgid(panel._process.pid), os.getpgrp())
        panel.refresh(dry_run=True, clutch="입력 대기", measured=None, target=None)
        self.assertFalse(panel._can_send)
        deadline = time.monotonic() + 2
        while not panel._can_send and time.monotonic() < deadline:
            panel.poll()
            time.sleep(0.01)
        self.assertTrue(panel._can_send)
        panel._process.stdin.close()
        self.assertEqual(panel._process.wait(timeout=2), 0)
        self.assertFalse(panel.is_open)
