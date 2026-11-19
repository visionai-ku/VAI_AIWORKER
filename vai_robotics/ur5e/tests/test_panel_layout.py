import os
import unittest
from unittest.mock import patch

from PyQt5.QtCore import QPoint, QRect
from PyQt5.QtGui import QFont, QScreen
from PyQt5.QtWidgets import QApplication, QDoubleSpinBox, QLabel, QPushButton, QScrollArea, QStyle, QStyleOptionComboBox

from vai_robotics.ur5e.teleop_panel import _PanelWindow


class PanelLayoutTest(unittest.TestCase):
    def setUp(self):
        platform = patch.dict(os.environ, QT_QPA_PLATFORM="offscreen")
        platform.start()
        self.addCleanup(platform.stop)

    def open_panel(self, width=1440, height=1000, font_scale=1):
        app = QApplication.instance() or QApplication([])
        if font_scale != 1:
            original_font = app.font()
            font = QFont(original_font)
            if font.pointSizeF() > 0:
                font.setPointSizeF(font.pointSizeF() * font_scale)
            else:
                font.setPixelSize(round(font.pixelSize() * font_scale))
            app.setFont(font)
            self.addCleanup(app.setFont, original_font)
        self.desktop = QRect(0, 0, width, height)
        screen = patch.object(QScreen, "availableGeometry", return_value=self.desktop)
        screen.start()
        self.addCleanup(screen.stop)
        self.panel = _PanelWindow(simulation=False, device_name="spacemouse", dry_run=True,
                                  open_width_mm=75, robot_ip="192.168.0.252",
                                  command_mode="eef-absolute", control_law="impedance")
        self.addCleanup(self.panel.close)
        self.window = self.panel.window
        self.scroll = self.window.findChild(QScrollArea)
        self.settle()

    def settle(self):
        # Showing wrapped labels posts layout requests from their parents to the scroll area.
        for _ in range(3):
            self.panel.app.processEvents()

    def populate(self, **overrides):
        pose = ([-412.5, -152.3, 514.2], [-121.5, 121.5, 0.0])
        telemetry = dict(dry_run=True, measured=pose, target=pose,
                         clutch="버튼 1: 상태 확인 대기 · 한 번 눌렀다 놓으세요",
                         gripper="버튼 0+1: 놓음 · RG2 75.0 mm · 미리보기 75.0 mm · 설정 파지력 40 N")
        self.panel.refresh(**{**telemetry, **overrides})
        self.settle()

    def assert_inside(self, widget, container):
        rect = QRect(widget.mapTo(container, QPoint()), widget.size())
        self.assertTrue(container.rect().contains(rect), f"{widget} occupies {rect} within {container.rect()}")

    def assert_text_fits(self, label):
        if label.wordWrap():
            self.assertGreaterEqual(label.height(), label.heightForWidth(label.width()))
        else:
            metrics = label.fontMetrics()
            self.assertGreaterEqual(label.contentsRect().height(), metrics.height())
            self.assertGreaterEqual(label.contentsRect().width(), metrics.horizontalAdvance(label.text()))

    def test_first_telemetry_fits_on_roomy_desktop(self):
        self.open_panel()
        for populated in (False, True):
            with self.subTest(populated=populated):
                if populated:
                    self.populate()
                self.assertLessEqual(self.window.frameGeometry().height(), self.desktop.height())
                self.assertEqual(self.scroll.verticalScrollBar().maximum(), 0)
                self.assertEqual(self.scroll.horizontalScrollBar().maximum(), 0)
                for field in self.panel.target_inputs:
                    self.assert_inside(field, self.scroll.viewport())

    def test_verbose_status_wraps_without_clipping_or_horizontal_scroll(self):
        self.open_panel()
        self.window.resize(max(self.window.minimumWidth(), 440), 600)
        self.settle()
        narrow_width = self.window.width()
        self.populate(clutch="버튼 1: SpaceMouse USB 연결을 확인하고 클러치 버튼을 한 번 눌렀다 놓으세요. " * 3,
                      gripper="버튼 0+1: 놓음 · RG2 실측 폭 대기 · 미리보기 75.0 mm · 설정 파지력 40 N. " * 3,
                      message="RTDE 연결 대기 중입니다. 로봇의 원격 제어와 컨트롤러의 프로그램 상태를 확인하세요. " * 3)
        self.assertEqual(self.window.width(), narrow_width)
        self.assertEqual(self.scroll.horizontalScrollBar().maximum(), 0)
        for label in (self.panel.clutch_label, self.panel.gripper_label, self.panel.message_label):
            with self.subTest(label=label.text()):
                self.assertTrue(label.isVisible())
                self.assertTrue(label.wordWrap())
                required_height = label.heightForWidth(label.width())
                self.assertGreater(required_height, label.fontMetrics().height())
                self.assertGreaterEqual(label.height(), required_height)

    def test_scroll_reaches_last_input_with_session_controls_visible(self):
        self.open_panel()
        self.populate()
        self.window.resize(self.window.width(), self.window.height() // 2)
        self.settle()
        self.assertGreater(self.scroll.verticalScrollBar().maximum(), 0)
        body = self.scroll.widget()
        last_input = max((box for box in body.findChildren(QDoubleSpinBox) if box.isVisible()),
                         key=lambda box: box.mapTo(body, box.rect().bottomLeft()).y())
        self.scroll.ensureWidgetVisible(last_input)
        self.settle()
        self.assert_inside(last_input, self.scroll.viewport())
        self.assertGreater(self.scroll.verticalScrollBar().value(), 0)
        exit_button, = [button for button in self.window.findChildren(QPushButton) if "종료" in button.text()]
        for control in (self.panel.state_label, self.panel.dry_button, self.panel.run_button, exit_button):
            self.assertFalse(self.scroll.isAncestorOf(control))
            self.assertTrue(control.isVisible())
            self.assert_inside(control, self.window)

    def test_status_header_represents_pending_run_change_until_acknowledged(self):
        self.open_panel(width=375, height=740)
        self.assertEqual(self.window.windowTitle(), "UR5e 조작 패널")
        label = self.panel.state_label
        self.assertEqual(label.text(), "상태 수신 대기")
        self.populate(operator_state="waiting-device")
        self.assertEqual(label.text(), "입력 장치 연결 대기")
        self.assert_text_fits(label)
        self.panel.run_button.click()
        self.settle()
        self.assertEqual(label.text(), "실행 모드 전환 중")
        self.assert_inside(label, self.window)
        self.assert_text_fits(label)
        self.populate(operator_state="ready")
        self.assertEqual(label.text(), "실행 모드 전환 중")
        self.panel._feedback_timer.timeout.emit()
        self.assertEqual(label.text(), "실행 모드 전환 중")
        self.panel.mode_result(dry_run=False)
        self.assertEqual(label.text(), "상태 수신 대기")
        self.populate(dry_run=False, operator_state="active")
        self.assertEqual(label.text(), "실기 조작 중")
        self.panel._feedback_timer.timeout.emit()
        self.assertEqual(label.text(), "상태 수신 대기")

    def test_header_tracks_command_switch_and_new_preview_feedback(self):
        self.open_panel()
        label = self.panel.state_label
        self.populate(operator_state="ready")
        self.assertEqual(label.text(), "조작 준비 완료")
        self.panel.command_picker.setCurrentIndex(self.panel.command_picker.findData("joint-delta"))
        self.populate(operator_state="ready")
        self.assertEqual(label.text(), "명령 모드 전환 중")
        self.panel.command_mode_result(command_mode="joint-delta")
        self.assertEqual(label.text(), "상태 수신 대기")
        self.populate(operator_state="active", measured_joints=[0.0] * 6)
        self.assertEqual(label.text(), "미리보기 조작 중")

    def test_narrow_screen_contains_frame_and_numeric_columns_after_status_update(self):
        self.open_panel(width=375, height=740)
        self.populate()
        diagnostic = "ur_rtde::RTDEControlInterface::isPoseWithinSafetyLimits(std::vector<double>)"
        self.populate(message=diagnostic)
        self.assertTrue(self.desktop.contains(self.window.frameGeometry()))
        self.assertEqual(self.scroll.horizontalScrollBar().maximum(), 0)
        picker = self.panel.control_law_picker
        self.scroll.ensureWidgetVisible(picker)
        self.settle()
        self.assert_inside(picker, self.scroll.viewport())
        option = QStyleOptionComboBox()
        picker.initStyleOption(option)
        field = picker.style().subControlRect(QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxEditField, picker)
        self.assertGreaterEqual(field.width(), picker.fontMetrics().horizontalAdvance(picker.currentText()))
        label = self.panel.message_label
        self.assertGreater(label.contentsRect().height(), label.fontMetrics().height())
        self.assertGreaterEqual(label.height(), label.heightForWidth(label.width()))
        self.scroll.ensureWidgetVisible(self.panel.target_inputs[-1])
        self.settle()
        self.assert_inside(self.panel.target_inputs[-1], self.scroll.viewport())

    def test_system_font_enlargement_fits_full_text_on_roomy_screen(self):
        self.open_panel(height=1400, font_scale=1.5)
        self.populate()
        self.assertTrue(self.desktop.contains(self.window.frameGeometry()))
        self.assertEqual(self.scroll.verticalScrollBar().maximum(), 0)
        self.assertEqual(self.scroll.horizontalScrollBar().maximum(), 0)
        for label in self.window.findChildren(QLabel):
            if not label.text() or not label.isVisible():
                continue
            with self.subTest(label=label.text()):
                self.assert_text_fits(label)
