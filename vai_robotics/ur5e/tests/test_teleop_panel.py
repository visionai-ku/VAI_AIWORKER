import os
import time
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from vai_robotics.ur5e.teleop_panel import TeleopPanel, _display_pose, _PanelWindow


class TeleopPanelTest(unittest.TestCase):
    def make_panel(self, simulation=False):
        with patch.dict(os.environ, QT_QPA_PLATFORM="offscreen"):
            panel = _PanelWindow(simulation=simulation, device_name="spacemouse", dry_run=True)
        self.addCleanup(panel.close)
        return panel

    def test_real_button_requests_mode_and_only_feedback_marks_it_applied(self):
        panel = self.make_panel()
        panel.run_button.click()
        self.assertIs(panel.take_request(), False)
        self.assertTrue(panel.dry_button.isChecked())
        self.assertIsNone(panel.take_request())

        panel.refresh(dry_run=True, clutch="클러치 해제", measured=None, target=None, message="실기 연결 대기")
        self.assertTrue(panel.dry_button.isChecked())
        self.assertEqual(panel.message_label.text(), "실기 연결 대기")
        panel.refresh(dry_run=False, clutch="클러치 해제", measured=None, target=None)
        self.assertTrue(panel.run_button.isChecked())
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
        self.assertEqual([cell.text() for cell in panel.pose_rows[0]], ["100.0", "200.0", "300.0"])
        self.assertEqual(panel.pose_rows[2][2].text(), "30.0")
        self.assertEqual(panel.pose_rows[1][0].text(), "—")
        self.assertTrue(panel.is_open)
        panel.close()
        self.assertFalse(panel.is_open)

    def test_separate_panel_process_consumes_frame_and_exits_on_pipe_eof(self):
        with patch.dict(os.environ, QT_QPA_PLATFORM="offscreen"):
            panel = TeleopPanel(simulation=True, device_name="SpaceMouse", dry_run=True)
        self.addCleanup(panel.close)
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
