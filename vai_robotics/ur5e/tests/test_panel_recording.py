import io
import json
import os
from queue import Queue
import unittest
from unittest.mock import Mock, patch

from PyQt5.QtCore import QRect
from PyQt5.QtGui import QScreen

from vai_robotics.ur5e.teleop_panel import TeleopPanel, _PanelWindow, _WindowConnection


class PanelRecordingTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, QT_QPA_PLATFORM="offscreen"))
        self.enterContext(patch.object(QScreen, "availableGeometry", return_value=QRect(0, 0, 375, 740)))
        self.window = _PanelWindow(simulation=False, device_name="SpaceMouse", dry_run=True, open_width_mm=75.0)
        self.addCleanup(self.window.close)
        self.pane = self.window.recording
        self.telemetry = dict(dry_run=False, clutch="놓음", measured=([1, 2, 3], [0, 0, 0]), target=None)
        self.idle = dict(state="idle", frames=0, episodes=3, root="/tmp/demos/ur5e", task="Pick up a block",
                         message="")

    def test_buttons_follow_controller_results_and_reject_duplicate_clicks(self):
        self.assertFalse(self.pane.section.isVisible())
        self.window.refresh(**self.telemetry, recording=self.idle)
        self.assertTrue(self.pane.buttons["start"].isEnabled())
        self.assertFalse(self.pane.buttons["discard"].isEnabled())
        self.pane.buttons["start"].click()
        self.assertEqual(self.window.take_operator_events(), {"record": "start"})

        # An older telemetry frame can arrive before the result for this click.
        self.window.refresh(**self.telemetry, recording=self.idle)
        self.pane.buttons["start"].click()
        self.assertEqual(self.window.take_operator_events(), {})
        self.window.recording_result(recording=dict(self.idle, state="preparing"))
        self.assertTrue(self.pane.buttons["discard"].isEnabled())
        self.assertFalse(self.pane.buttons["start"].isEnabled())

        self.window.refresh(**self.telemetry, recording=dict(self.idle, state="recording", frames=18))
        self.assertIn("18 frames", self.pane.status.text())
        self.pane.buttons["save"].click()
        self.pane.buttons["discard"].click()
        self.assertEqual(self.window.take_operator_events(), {"record": "save"})
        self.window.recording_result(recording=dict(self.idle, state="saving", frames=18))
        self.assertFalse(any(button.isEnabled() for button in self.pane.buttons.values()))
        self.window.refresh(**self.telemetry, recording=dict(self.idle, episodes=4))
        self.assertIn("4 saved", self.pane.status.text())
        self.assertTrue(self.pane.buttons["start"].isEnabled())

    def test_omits_saved_count_until_existing_dataset_has_been_read(self):
        self.window.refresh(**self.telemetry, recording=dict(self.idle, episodes=0, episodes_known=False))
        self.assertEqual(self.pane.status.text(), "Idle · 0 frames")
        self.window.recording_result(recording=dict(self.idle, state="recording", episodes_known=True))
        self.assertIn("3 saved", self.pane.status.text())

    def test_record_requests_travel_alongside_targets_and_ack_independently(self):
        # Use both ends of the production IPC protocol in this test.
        panel = TeleopPanel.__new__(TeleopPanel)
        panel._process = Mock(stdin=io.StringIO())
        panel._process.poll.return_value = None
        panel._open = True
        panel._command_mode = "eef-delta"
        panel._control_law = "position"
        panel._messages = Queue()
        updates = Queue()
        with patch("vai_robotics.ur5e.teleop_panel._incoming", return_value=updates):
            connection = _WindowConnection(self.window)
        self.window.refresh(**self.telemetry, recording=self.idle)
        self.window.apply_target.click()
        self.pane.buttons["start"].click()
        panel._messages.put(connection._take_operator_events())
        panel.poll()
        self.assertEqual(panel.take_target(), [0.] * 6)
        panel.poll()  # An unrelated telemetry poll does not consume the recording request.
        self.assertEqual(panel.take_record_request(), "start")
        self.assertIsNone(panel.take_record_request())
        panel.recording_result(dict(self.idle, state="recording", frames=1))
        updates.put(json.loads(panel._process.stdin.getvalue()))
        self.assertEqual(connection._receive_updates(), {})
        self.pane.buttons["discard"].click()
        self.assertEqual(connection._take_operator_events(), {"record": "discard"})
        panel._messages.put({"record": "save"})
        panel._messages.put(None)  # End of the Qt pipe arrived in the same control tick.
        panel.poll()
        self.assertFalse(panel.is_open)
        self.assertEqual(panel.take_record_request(), "save")

    def test_unavailable_recording_and_long_dataset_path_fit_narrow_panel(self):
        root = "/Users/operator/datasets/" + "a" * 120
        message = "SIM RUN 또는 REAL RUN에서 Record를 누르세요."
        self.window.refresh(**self.telemetry, recording=dict(self.idle, can_start=False, root=root, message=message))
        for _ in range(3):
            self.window.app.processEvents()
        self.assertFalse(self.pane.buttons["start"].isEnabled())
        self.assertEqual(self.pane.message.text(), message)
        self.assertIn(root, self.pane.destination.toolTip())
        self.assertEqual(self.window.scroll.horizontalScrollBar().maximum(), 0)

    def test_save_message_fits_roomy_desktop_and_respects_manual_resize(self):
        self.enterContext(patch.object(QScreen, "availableGeometry", return_value=QRect(0, 0, 1440, 1280)))
        window = _PanelWindow(simulation=True, device_name="SpaceMouse", dry_run=False, open_width_mm=75.0)
        self.addCleanup(window.close)

        def refresh(message):
            window.refresh(**self.telemetry, recording=dict(self.idle, message=message))
            for _ in range(3):
                window.app.processEvents()

        refresh("")
        self.assertEqual(window.scroll.verticalScrollBar().maximum(), 0)
        refresh("Saved episode 0")
        self.assertEqual(window.scroll.verticalScrollBar().maximum(), 0)
        window.window.resize(430, 600)
        manual_size = window.window.size()
        refresh("Dataset is already recording. " * 6)
        self.assertEqual(window.window.size(), manual_size)


if __name__ == "__main__":
    unittest.main()
