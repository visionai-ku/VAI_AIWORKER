import io
import json
import os
from queue import Queue
import unittest
from unittest.mock import Mock, patch

from PyQt5.QtCore import QPoint, QRect
from PyQt5.QtGui import QScreen
from PyQt5.QtWidgets import QPushButton

from vai_robotics.ur5e.impedance_gains import default_impedance_gains
from vai_robotics.ur5e.teleop_panel import TeleopPanel, _PanelWindow


class ImpedanceGainPanelTest(unittest.TestCase):
    def open_panel(self, **settings):
        with patch.dict(os.environ, QT_QPA_PLATFORM="offscreen"):
            panel = _PanelWindow(simulation=True, dry_run=True, device_name="spacemouse",
                                 open_width_mm=75, command_mode="eef-absolute", **settings)
        self.addCleanup(panel.close)
        return panel

    def test_only_confirmed_changes_rearm_absolute_targets_and_preserve_gains_per_space(self):
        panel = self.open_panel(control_law="position")
        editor = panel.gain_editor
        self.assertFalse(editor.toggle.isVisible())
        panel.control_law_result(control_law="impedance")
        self.assertTrue(editor.toggle.isVisible())
        self.assertFalse(editor.body.isVisible())
        editor.toggle.click()

        panel.refresh(dry_run=True, clutch="놓음", measured=([1, 2, 3], [4, 5, 6]), target=None)
        panel.load_target.click()
        panel.apply_target.click()
        original = default_impedance_gains("eef")
        self.assertFalse(editor.apply.isEnabled())
        self.assertIn("N/m", editor.inputs["stiffness"][0].accessibleName())
        self.assertIn("N·m·s/rad", editor.inputs["damping"][5].accessibleName())
        editor.inputs["stiffness"][0].setValue(157.5)
        editor.inputs["damping"][5].setValue(2.1)
        self.assertIsNone(panel.take_impedance_gains())
        self.assertEqual(editor.confirmed["eef"], original)
        for refuse in (True, False):
            editor.apply.click()
            request = panel.take_impedance_gains()
            self.assertEqual(request, {"space": "eef", "gains": editor.values()})
            self.assertIsNone(panel.take_impedance_gains())
            self.assertIsNone(panel.take_target_event())
            self.assertEqual(panel.state_label.text(), "강성·감쇠 적용 중")
            for control in (panel.control_law_picker, panel.command_picker, panel.run_button,
                            editor.body, panel.apply_target):
                self.assertFalse(control.isEnabled())
            if refuse:
                panel.impedance_gains_result(space="eef", gains=original, message="클러치를 놓으세요.")
                self.assertTrue(editor.apply.isEnabled())
                self.assertEqual(panel.message_label.text(), "클러치를 놓으세요.")
                self.assertEqual(editor.confirmed["eef"], original)
                self.assertTrue(panel.apply_target.isEnabled())
            else:
                panel.impedance_gains_result(**request)
                self.assertFalse(editor.apply.isEnabled())
                self.assertFalse(panel.apply_target.isEnabled())
                self.assertFalse(panel.load_target.isEnabled())
                self.assertEqual(editor.confirmed["eef"], request["gains"])
                self.assertEqual(panel.measured_cells[0].text(), "—")

        panel.command_mode_result(command_mode="joint-absolute")
        self.assertIn("관절", editor.toggle.text())
        self.assertEqual(editor.values(), default_impedance_gains("joint"))
        self.assertEqual(editor.inputs["stiffness"][0].accessibleName(), "J1 강성 (N·m/rad)")
        editor.inputs["damping"][0].setValue(5.1)
        panel.command_mode_result(command_mode="eef-delta")
        self.assertEqual(editor.values(), request["gains"])
        panel.command_mode_result(command_mode="joint-delta")
        self.assertEqual(editor.inputs["damping"][0].value(), 5.1)
        self.assertTrue(editor.apply.isEnabled())
        self.assertIsNone(panel.take_impedance_gains())
        panel.command_mode_result(command_mode="eef-delta")
        defaults, = (b for b in editor.body.findChildren(QPushButton) if b.text() == "기본값 불러오기")
        defaults.click()
        self.assertEqual(editor.values(), original)
        self.assertEqual(editor.confirmed["eef"], request["gains"])
        self.assertTrue(editor.apply.isEnabled())
        self.assertIsNone(panel.take_impedance_gains())

    def test_expanded_gain_table_fits_narrow_viewport(self):
        with patch.object(QScreen, "availableGeometry", return_value=QRect(0, 0, 375, 740)):
            panel = self.open_panel(control_law="impedance")
            editor = panel.gain_editor
            editor.toggle.click()
            for mode in ("eef-delta", "joint-delta"):
                panel.command_mode_result(command_mode=mode)
                field = editor.inputs["damping"][-1]
                for _ in range(3):
                    panel.app.processEvents()
                panel.scroll.ensureWidgetVisible(field)
                panel.app.processEvents()
                view = panel.scroll.viewport()
                field_rect = QRect(field.mapTo(view, QPoint()), field.size())
                self.assertTrue(field.isVisible() and view.rect().contains(field_rect), field_rect)
                self.assertEqual(panel.scroll.horizontalScrollBar().maximum(), 0)
                for group in editor.headers:
                    for header in group:
                        if header.isVisible():
                            width = max(header.fontMetrics().horizontalAdvance(s) for s in header.text().splitlines())
                            self.assertGreaterEqual(header.contentsRect().width(), width)

    def test_gain_request_and_late_target_after_ack(self):
        panel = TeleopPanel.__new__(TeleopPanel)
        panel._messages = Queue()
        panel._process = Mock()
        panel._process.poll.return_value = None
        panel._process.stdin = io.StringIO()
        panel._open = True
        panel._command_mode, panel._control_law = "eef-delta", "impedance"
        request = {"space": "eef", "gains": default_impedance_gains("eef")}
        target = dict(time=20.01, values=[0] * 6, command_mode="eef-delta", control_law="impedance")
        with patch("vai_robotics.ur5e.teleop_panel.time.monotonic", return_value=20.02):
            panel._messages.put({"target": target, "impedance_gains": request})
            panel.poll()
            self.assertIsNone(panel.take_target())
            self.assertEqual(panel.take_impedance_gains(), request)
            self.assertIsNone(panel.take_impedance_gains())
            panel.impedance_gains_result(**request)
            self.assertEqual(json.loads(panel._process.stdin.getvalue()),
                             {"impedance_gains_result": {**request, "message": ""}})
        with patch("vai_robotics.ur5e.teleop_panel.time.monotonic", return_value=20.08):
            for timestamp, accepted in ((20.01, False), (20.06, True)):
                panel._messages.put({"target": {**target, "time": timestamp}})
                panel.poll()
                self.assertEqual(panel.take_target(), target["values"] if accepted else None)
