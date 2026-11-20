import io
import json
import os
import unittest
from unittest.mock import Mock, patch

from PyQt5.QtCore import QRect
from PyQt5.QtGui import QScreen

from vai_robotics.ur5e.teleop_panel import TeleopPanel, _PanelWindow


class PanelTCPSpeedTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, QT_QPA_PLATFORM="offscreen"))
        self.enterContext(patch.object(QScreen, "availableGeometry", return_value=QRect(0, 0, 375, 740)))
        self.telemetry = dict(dry_run=False, clutch="Pressed", measured=None, target=None)
        self.live = dict(phase="live", real_mm_s=12.6, dry_mm_s=27.2, real_deg_s=5.8, dry_deg_s=7.1)
        self.payload = dict(mass_kg=0.91, cog_m=[0.001, -0.005, 0.083])

    def open_window(self, simulation=False):
        window = _PanelWindow(simulation=simulation, device_name="spacemouse", dry_run=True, open_width_mm=75)
        self.addCleanup(window.close)
        return window

    def test_control_pipe_displays_paired_speeds_and_flange_cog(self):
        panel = TeleopPanel.__new__(TeleopPanel)
        panel._open = panel._can_send = True
        panel._next_update = 0.0
        panel._process = Mock(stdin=io.StringIO())
        panel._process.poll.return_value = None
        panel.refresh(**self.telemetry, speed_comparison=self.live, tool_payload=self.payload)
        window = self.open_window()
        pane = window.speed_comparison
        window.refresh(**json.loads(panel._process.stdin.getvalue()))
        for _ in range(3):
            window.app.processEvents()
        self.assertTrue(pane.section.isVisible())
        self.assertEqual(pane.status.text(), "Live")
        self.assertEqual(pane.values["real_mm_s"].text(), "12.6")
        self.assertEqual(pane.values["dry_deg_s"].text(), "7.1")
        self.assertIn("0.91 kg", pane.payload.text())
        self.assertIn("[1, -5, 83] mm", pane.payload.text())
        self.assertEqual(window.scroll.horizontalScrollBar().maximum(), 0)
        window._feedback_timer.timeout.emit()
        self.assertEqual(pane.values["real_mm_s"].text(), "—")
        self.assertIn("unavailable", pane.payload.text())

    def test_live_reply_after_mode_change_is_hidden_and_last_mean_is_labeled(self):
        window = self.open_window()
        pane = window.speed_comparison
        telemetry = dict(self.telemetry, dry_run=True)
        window.refresh(**telemetry, speed_comparison=self.live, tool_payload=self.payload)
        self.assertEqual(pane.values["real_mm_s"].text(), "—")
        window.refresh(**telemetry, speed_comparison=dict(self.live, phase="last"), tool_payload=self.payload)
        self.assertIn("Mean", pane.status.text())
        self.assertEqual(pane.values["real_mm_s"].text(), "12.6")
        window.refresh(**telemetry, speed_comparison={"phase": "unavailable"})
        self.assertEqual(pane.values["real_mm_s"].text(), "—")

    def test_standalone_sim_omits_connected_robot_telemetry(self):
        self.assertIsNone(self.open_window(simulation=True).speed_comparison)


if __name__ == "__main__":
    unittest.main()
