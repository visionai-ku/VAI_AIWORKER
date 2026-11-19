"""UR5e operator window, telemetry display and confirmed control selections."""

import math
import os
from pathlib import Path
import sys
import time

from PyQt5.QtCore import QCoreApplication, QEvent, QSignalBlocker, Qt, QTimer
from PyQt5.QtGui import QFontDatabase
from PyQt5.QtWidgets import (
    QApplication, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QPushButton,
    QScrollArea, QSlider, QStyle, QVBoxLayout, QWidget,
)

from .impedance_editor import ImpedanceEditor
from .widgets import ModePicker, add_divider, add_section, make_label, set_status_text


_PENDING_TITLES = {
    "mode": "실행 모드 전환 중",
    "command_mode": "명령 모드 전환 중",
    "control_law": "제어 방식 전환 중",
    "impedance_gains": "강성·감쇠 적용 중",
}

_STATE_LABELS = {
    "waiting-device": ("입력 장치 연결 대기", "idle"),
    "waiting-input": ("입력 신호 대기", "idle"),
    "waiting-clutch": ("클러치 확인 대기", "attention"),
    "release-clutch": ("클러치를 놓으세요", "attention"),
    "ready": ("조작 준비 완료", "ready"),
}


def _panel_application():
    app = QApplication.instance()
    if app is not None:
        return app
    # Linux OpenCV wheels can point Qt at their own incompatible xcb plugin.
    if sys.platform.startswith("linux"):
        for key in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
            if "cv2" in Path(os.environ.get(key, "")).parts:
                os.environ.pop(key, None)
    return QApplication([])


class PanelWindow:
    """Show the active configuration until the control loop confirms a request."""

    def __init__(self, *, simulation, device_name, dry_run, open_width_mm, robot_ip=None,
                 command_mode="eef-delta", control_law="position"):
        self.app = _panel_application()
        self._simulation = simulation
        self._operator_state = None
        self._applied_dry_run = dry_run
        self._control_law = control_law
        self._pending_switch = None
        self._switch_request = {}
        self._last_measured = None
        self._target_event = None
        self._command_mode = None
        self._loaded_absolute = False
        self._first_frame = True

        self.window = QWidget()
        self._feedback_timer = QTimer(self.window)
        self._feedback_timer.setSingleShot(True)
        self._feedback_timer.setInterval(1500)  # Display frames normally arrive at 10 Hz.
        self._feedback_timer.timeout.connect(self._clear_operator_state)

        self._build_window(device_name, open_width_mm, robot_ip)
        self._show_mode(dry_run)
        self._update_state_header()
        self._show_command_mode(command_mode)
        self._open_window()

    def _build_window(self, device_name, open_width_mm, robot_ip):
        self.window.setObjectName("panel")
        self.window.setAttribute(Qt.WA_QuitOnClose, False)
        self.window.setWindowTitle("UR5e 조작 패널")
        outer = QVBoxLayout(self.window)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self._build_header(outer)
        self._build_body(outer, device_name, open_width_mm)
        self._build_footer(outer, robot_ip)
        self.window.setStyleSheet(Path(__file__).with_name("style.qss").read_text(encoding="utf-8"))

    def _build_body(self, outer, device_name, open_width_mm):
        self.scroll = QScrollArea()
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setWidgetResizable(True)
        outer.addWidget(self.scroll, 1)
        self.body = QWidget()
        self.body.setObjectName("body")
        layout = QVBoxLayout(self.body)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        self.message_label = make_label("", wrap=True)
        self.message_label.setObjectName("message")
        self.message_label.hide()
        layout.addWidget(self.message_label)
        self._build_inputs(add_section(layout), device_name, open_width_mm)
        self._build_targets(add_section(layout))
        layout.addStretch()
        self.scroll.setWidget(self.body)

    def _open_window(self):
        self.window.ensurePolished()
        self.window.layout().activate()
        # Obtain native frame margins without flashing a default-sized window.
        self.window.winId()
        self._fit_window(initial=True)
        self.window.show()
        self._automatic_size = self.window.size()
        QTimer.singleShot(0, self._fit_window)

    def _build_header(self, outer):
        header = QFrame()
        header.setObjectName("header")
        layout = QVBoxLayout(header)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(9)
        title_row = QHBoxLayout()
        self.state_dot = QFrame()
        self.state_dot.setObjectName("stateDot")
        self.state_dot.setFixedSize(8, 8)
        title_row.addWidget(self.state_dot, 0, Qt.AlignVCenter)
        self.state_label = make_label("", wrap=True, emphasis=6)
        self.state_label.setObjectName("title")
        self.state_label.setAccessibleDescription("현재 조작 상태")
        title_row.addWidget(self.state_label, 1)
        layout.addLayout(title_row)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.dry_button = QPushButton("DRY RUN · 미리보기")
        self.run_button = QPushButton("SIM RUN · 시뮬레이션" if self._simulation else "REAL RUN · 실기 구동")
        for button, dry, name in ((self.dry_button, True, "dry"),
                                  (self.run_button, False, "sim" if self._simulation else "real")):
            button.setObjectName(name)
            button.setProperty("role", "mode")
            button.setAttribute(Qt.WA_LayoutUsesWidgetRect)
            button.setCheckable(True)
            button.clicked.connect(lambda checked, requested=dry: self._request_mode(requested))
            row.addWidget(button, 1)
        layout.addLayout(row)
        outer.addWidget(header)

    def _state_presentation(self):
        if self._pending_switch:
            return _PENDING_TITLES[self._pending_switch], "attention"
        if self._operator_state != "active":
            return _STATE_LABELS.get(self._operator_state, ("상태 수신 대기", "idle"))
        if self._applied_dry_run:
            return "미리보기 조작 중", "preview"
        if self._simulation:
            return "시뮬레이션 조작 중", "active"
        return "실기 조작 중", "real"

    def _update_state_header(self):
        title, color = self._state_presentation()
        self.state_label.setText(title)
        if self.state_dot.property("tone") != color:
            self.state_dot.setProperty("tone", color)
            self.state_dot.style().unpolish(self.state_dot)
            self.state_dot.style().polish(self.state_dot)
            self.state_dot.update()

    def _clear_operator_state(self):
        self._operator_state = None
        self._update_state_header()

    def _build_inputs(self, layout, device_name, open_width_mm):
        device = {"spacemouse": "SpaceMouse", "quest": "Quest 3"}.get(device_name, device_name)
        top = QHBoxLayout()
        top.addWidget(make_label("입력 장치", emphasis=2))
        top.addStretch()
        top.addWidget(make_label(device, role="muted", wrap=True))
        layout.addLayout(top)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 1)
        self.clutch_label = make_label("입력 대기", wrap=True)
        self.gripper_label = make_label("상태 수신 대기", wrap=True)
        for row, name, status in ((0, "클러치", self.clutch_label), (1, "그리퍼", self.gripper_label)):
            grid.addWidget(make_label(name, role="muted"), row, 0, Qt.AlignTop)
            grid.addWidget(status, row, 1, 1, 2)
        self._add_scale_control(grid, row=2)
        self._add_open_width_control(grid, row=3, width_mm=open_width_mm)
        layout.addLayout(grid)

    def _add_scale_control(self, grid, row):
        self.gain = QSlider(Qt.Horizontal)
        self.gain.setAccessibleName("입력 배율")
        self.gain.setRange(10, 100)
        self.gain.setValue(100)
        gain_label = make_label("입력 배율", role="muted")
        gain_label.setBuddy(self.gain)
        gain_value = make_label("100%")
        gain_value.setMinimumWidth(gain_value.sizeHint().width())
        gain_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.gain.valueChanged.connect(lambda value: gain_value.setText(f"{value}%"))
        grid.addWidget(gain_label, row, 0)
        grid.addWidget(self.gain, row, 1)
        grid.addWidget(gain_value, row, 2)

    def _add_open_width_control(self, grid, row, width_mm):
        width_label = make_label("열림 목표 폭", role="muted")
        self.open_width = QDoubleSpinBox()
        width_label.setBuddy(self.open_width)
        self.open_width.setAccessibleName(width_label.text())
        self.open_width.setDecimals(1)
        self.open_width.setRange(0.0, 110.0)
        self.open_width.setSingleStep(0.1)
        self.open_width.setSuffix(" mm")
        self.open_width.setKeyboardTracking(False)
        self.open_width.setValue(width_mm)
        grid.addWidget(width_label, row, 0)
        grid.addWidget(self.open_width, row, 1, 1, 2, Qt.AlignLeft)

    def _build_targets(self, layout):
        layout.addWidget(make_label("목표 제어", emphasis=2))
        self._build_mode_selectors(layout)
        self.gain_editor = ImpedanceEditor(layout, self._request_impedance_gains)
        self.gain_editor.toggle.toggled.connect(lambda expanded: self._fit_window(shrink=not expanded))
        self.coordinate_label = make_label("", role="muted", wrap=True)
        layout.addWidget(self.coordinate_label)
        add_divider(layout)
        self._build_target_table(layout)

        layout.addWidget(make_label("클러치를 누른 상태에서 목표를 적용하세요.", role="muted", wrap=True))
        actions = QHBoxLayout()
        actions.setSpacing(6)
        actions.addStretch()
        self.load_target = QPushButton("실측값 불러오기")
        self.load_target.clicked.connect(self._load_measured)
        actions.addWidget(self.load_target)
        self.apply_target = QPushButton("목표 적용")
        self.apply_target.setObjectName("apply")
        self.apply_target.clicked.connect(self._submit_target)
        actions.addWidget(self.apply_target)
        layout.addLayout(actions)

    def _build_mode_selectors(self, layout):
        self.command_picker = ModePicker()
        for space, name in (("eef", "TCP"), ("joint", "관절")):
            self.command_picker.addItem(f"{name} · 상대 변화량", f"{space}-delta")
            self.command_picker.addItem(f"{name} · 절대 목표", f"{space}-absolute")
        self.command_picker.currentIndexChanged.connect(self._request_command_mode)
        self.control_law_picker = ModePicker()
        self.control_law_picker.addItem("위치 제어", "position")
        self.control_law_picker.addItem("임피던스 제어", "impedance")
        self.control_law_picker.setCurrentIndex(self.control_law_picker.findData(self._control_law))
        self.control_law_picker.currentIndexChanged.connect(self._request_control_law)
        config = QGridLayout()
        config.setColumnStretch(1, 1)
        for row, (name, picker) in enumerate((("명령 모드", self.command_picker),
                                              ("제어 방식", self.control_law_picker))):
            caption = make_label(name, role="muted")
            caption.setBuddy(picker)
            picker.setAccessibleName(name)
            config.addWidget(caption, row, 0)
            config.addWidget(picker, row, 1)
        layout.addLayout(config)

    def _build_target_table(self, layout):
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(5)
        grid.setVerticalSpacing(3)
        self.entry_heading = make_label("", role="column")
        for column, heading in enumerate((make_label("축", role="column"), make_label("실측", role="column"),
                                          make_label("현재 목표", role="column"), self.entry_heading)):
            if column:
                heading.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(heading, 0, column)
            grid.setColumnStretch(column, (1, 2, 2, 3)[column])
        self.axis_labels, self.measured_cells, self.target_cells, self.target_inputs = [], [], [], []
        fixed_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        fixed_font.setPointSizeF(self.window.font().pointSizeF())
        for row in range(1, 7):
            self._add_target_axis(grid, row, fixed_font)
        layout.addLayout(grid)

    def _add_target_axis(self, grid, row, font):
        axis = make_label("")
        self.axis_labels.append(axis)
        grid.addWidget(axis, row, 0)
        for column, cells in enumerate((self.measured_cells, self.target_cells), 1):
            cell = make_label("—")
            cell.setFont(font)
            cell.setMinimumWidth(cell.fontMetrics().horizontalAdvance("-2000.0"))
            cell.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(cell, row, column)
            cells.append(cell)
        box = QDoubleSpinBox()
        box.setFont(font)
        box.setAlignment(Qt.AlignRight)
        box.setDecimals(2)
        box.setSingleStep(1.0)
        box.setKeyboardTracking(False)
        axis.setBuddy(box)
        self.target_inputs.append(box)
        grid.addWidget(box, row, 3)

    def _build_footer(self, outer, robot_ip):
        footer = QFrame()
        footer.setObjectName("footer")
        row = QHBoxLayout(footer)
        row.setContentsMargins(16, 6, 16, 6)
        if self._simulation:
            row.addStretch(1)
        else:
            source = f"RTDE · {robot_ip}" if robot_ip else "RTDE"
            row.addWidget(make_label(f"데이터 소스   {source}", role="muted", wrap=True), 1)
        exit_button = QPushButton("세션 종료")
        exit_button.clicked.connect(self.close)
        row.addWidget(exit_button)
        outer.addWidget(footer)

    def _fit_window(self, *, initial=False, shrink=False):
        # Nested layouts post requests when text or visibility changes.
        QCoreApplication.sendPostedEvents(self.body, QEvent.LayoutRequest)
        window = self.window
        if not initial and window.size() != self._automatic_size:
            return  # Preserve the operator's own window size.
        handle = window.windowHandle()
        frame = handle.frameMargins()
        area = handle.screen().availableGeometry()
        available_width = area.width() - frame.left() - frame.right()
        available_height = area.height() - frame.top() - frame.bottom()
        body = self.body.layout()
        body.activate()
        scrollbar = self.scroll.style().pixelMetric(QStyle.PM_ScrollBarExtent)
        default_width = max(460, window.fontMetrics().horizontalAdvance("0") * 52,
                            body.minimumSize().width() + scrollbar)
        width = min(available_width, default_width) if initial else window.width()
        # QScrollArea.sizeHint() does not account for text wrapped at viewport width.
        header, footer = window.layout().itemAt(0), window.layout().itemAt(2)
        chrome_height = sum(
            item.heightForWidth(width) if item.hasHeightForWidth() else item.sizeHint().height()
            for item in (header, footer)
        )
        body_height = max(body.minimumSize().height(), body.heightForWidth(width))
        height = min(available_height, body_height + chrome_height)
        window.resize(width, height if initial or shrink else max(height, window.height()))
        if initial:
            x = max(area.left(), area.right() + 1 - window.width() - frame.right() - frame.left() - 12)
            window.move(x, area.top())
        self._automatic_size = window.size()

    @staticmethod
    def _select_confirmed(picker, value):
        with QSignalBlocker(picker):
            picker.setCurrentIndex(picker.findData(value))

    def _show_command_mode(self, mode):
        self._select_confirmed(self.command_picker, mode)
        if self._command_mode == mode:
            return
        self._command_mode = mode
        self._joint_space = mode.startswith("joint-")
        self._absolute = mode.endswith("-absolute")
        command = "절대 목표" if self._absolute else "상대 변화량"
        self.entry_heading.setText("절대값 입력" if self._absolute else "변화량 입력")
        self.coordinate_label.setText("관절각 · J1–J6" if self._joint_space else "베이스 좌표계 · Rx/Ry/Rz 회전벡터")
        axes = [f"J{i + 1}" for i in range(6)] if self._joint_space else ("X", "Y", "Z", "Rx", "Ry", "Rz")
        for i, (axis, box) in enumerate(zip(axes, self.target_inputs)):
            linear = not self._joint_space and i < 3
            unit = "mm" if linear else "°"
            self.axis_labels[i].setText(f"{axis} ({unit})")
            box.setAccessibleName(f"{axis} {command} ({unit})")
        self.load_target.setVisible(self._absolute)
        self.gain_editor.configure("joint" if self._joint_space else "eef", self._control_law == "impedance")
        self._reset_target_editor()
        self._enable_target_controls()

    def _reset_target_editor(self):
        self._last_measured = self._target_event = None
        self._loaded_absolute = False
        for i, box in enumerate(self.target_inputs):
            limit = 2000.0 if not self._joint_space and i < 3 else 360.0
            box.setRange(-limit, limit)
            box.setValue(0)
            self.measured_cells[i].setText("—")
            self.target_cells[i].setText("—")

    def _request_command_mode(self, index):
        requested = self.command_picker.itemData(index)
        self._select_confirmed(self.command_picker, self._command_mode)
        if requested is not None and requested != self._command_mode:
            self._begin_switch("command_mode", requested, self.command_picker.itemText(index))

    def _request_control_law(self, index):
        requested = self.control_law_picker.itemData(index)
        self._select_confirmed(self.control_law_picker, self._control_law)
        if requested is not None and requested != self._control_law:
            self._begin_switch("control_law", requested, self.control_law_picker.itemText(index))

    def _request_impedance_gains(self, space, gains):
        self._begin_switch("impedance_gains", {"space": space, "gains": gains}, "강성·감쇠")

    def _begin_switch(self, kind, value, caption):
        if self._pending_switch is not None:
            return
        self._pending_switch = kind
        self._switch_request[kind] = value
        self._target_event = None
        self._enable_target_controls()
        self._update_state_header()
        self._set_message(f"{caption} 적용 중…" if kind == "impedance_gains" else f"{caption} 전환 중…")

    def _finish_switch(self, kind, message):
        if self._pending_switch == kind:
            self._pending_switch = None
            self._switch_request.pop(kind, None)
        self._enable_target_controls()
        self._clear_operator_state()
        if self._pending_switch is None:
            self._set_message(message, reveal=True)

    def take_command_mode(self):
        return self._switch_request.pop("command_mode", None)

    def take_control_law(self):
        return self._switch_request.pop("control_law", None)

    def take_impedance_gains(self):
        return self._switch_request.pop("impedance_gains", None)

    def command_mode_result(self, *, command_mode, message=""):
        changed = command_mode != self._command_mode
        self._show_command_mode(command_mode)
        self._finish_switch("command_mode", message)
        if changed:
            self._fit_window()

    def control_law_result(self, *, control_law, message=""):
        self._select_confirmed(self.control_law_picker, control_law)
        if self._control_law != control_law:
            self._control_law = control_law
            self.gain_editor.configure("joint" if self._joint_space else "eef", control_law == "impedance")
            self._reset_target_editor()
        self._finish_switch("control_law", message)
        self._fit_window()

    def impedance_gains_result(self, *, space, gains, message=""):
        accepted = not message and gains is not None
        self.gain_editor.result(space, gains, accepted)
        if accepted:
            self._reset_target_editor()
        self._finish_switch("impedance_gains", message)

    def _set_message(self, message, *, reveal=False):
        set_status_text(self.message_label, message)
        self.message_label.setVisible(bool(message))
        if message and reveal:
            self.scroll.verticalScrollBar().setValue(0)

    @property
    def is_open(self):
        return self.window.isVisible()

    @property
    def scale(self):
        return self.gain.value() / 100

    @property
    def open_width_mm(self):
        return round(self.open_width.value(), self.open_width.decimals())

    def _show_mode(self, dry_run):
        if dry_run != self._applied_dry_run:
            self._target_event = self._last_measured = None
            self._loaded_absolute = False
        self._applied_dry_run = dry_run
        self.dry_button.setChecked(dry_run)
        self.run_button.setChecked(not dry_run)

    def _request_mode(self, dry_run):
        self._show_mode(self._applied_dry_run)
        if dry_run != self._applied_dry_run:
            button = self.dry_button if dry_run else self.run_button
            self._begin_switch("mode", dry_run, button.text())

    def take_request(self):
        return self._switch_request.pop("mode", None)

    def _enable_target_controls(self):
        can_edit = self._pending_switch is None
        for widget in (self.run_button, self.dry_button, self.command_picker, self.control_law_picker,
                       self.gain_editor.toggle, self.gain_editor.body):
            widget.setEnabled(can_edit)
        ready = can_edit and self._last_measured is not None
        self.load_target.setEnabled(ready)
        self.apply_target.setEnabled(ready and (not self._absolute or self._loaded_absolute))

    def _load_measured(self):
        for box, value in zip(self.target_inputs, self._last_measured):
            # A configured TCP offset or a multi-turn joint can lie beyond the
            # initial entry range. Loading its measurement must never clip it.
            box.setRange(min(box.minimum(), value), max(box.maximum(), value))
            box.setValue(value)
        self._loaded_absolute = True
        self._enable_target_controls()

    def _submit_target(self):
        display = [box.value() for box in self.target_inputs]
        if self._joint_space:
            values = [math.radians(degrees) for degrees in display]
        else:
            position = [millimeters / 1000 for millimeters in display[:3]]
            rotation_vector = [math.radians(degrees) for degrees in display[3:]]
            values = position + rotation_vector
        self._target_event = {"time": time.monotonic(), "values": values,
                              "command_mode": self._command_mode, "control_law": self._control_law}

    def take_target_event(self):
        event, self._target_event = self._target_event, None
        return event

    def take_operator_events(self):
        """Consume edits for one IPC frame. A configuration change supersedes a target."""
        events, self._switch_request = self._switch_request, {}
        target = self.take_target_event()
        if not events and target is not None:
            events["target"] = target
        return events

    def mode_result(self, *, dry_run, message=""):
        self._show_mode(dry_run)
        self._finish_switch("mode", message)

    def refresh(self, *, dry_run, clutch, measured, target, message="", gripper="",
                measured_joints=None, target_joints=None, operator_state="waiting-input"):
        self._feedback_timer.start()
        if self._pending_switch is None:
            self._show_mode(dry_run)
            self._set_message(message)
            self._operator_state = operator_state
            self._update_state_header()
        set_status_text(self.clutch_label, clutch)
        set_status_text(self.gripper_label, gripper)
        if self._joint_space:
            measured, target = measured_joints, target_joints
        else:
            measured = None if measured is None else [*measured[0], *measured[1]]
            target = None if target is None else [*target[0], *target[1]]
        self._update_target_table(measured, target)
        if self._first_frame:
            self._first_frame = False
            QTimer.singleShot(0, self._fit_window)

    def _update_target_table(self, measured, target):
        self._last_measured = measured
        if measured is None:
            self._loaded_absolute = False
        for cells, values in ((self.measured_cells, measured), (self.target_cells, target)):
            for i, cell in enumerate(cells):
                cell.setText("—" if values is None else f"{values[i]:.1f}")
        self._enable_target_controls()

    def close(self):
        self.window.close()
