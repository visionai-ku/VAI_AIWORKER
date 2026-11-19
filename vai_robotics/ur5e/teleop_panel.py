"""Operator controls for a running UR5e teleoperation session."""

import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import re
import subprocess
import sys
from threading import Thread
import time


# A clicked target is an input event, just like a sample from the operator device.
TARGET_TIMEOUT = 0.15


def _incoming(stream):
    """Keep blocking pipe reads off the control and Qt event loops."""
    messages = Queue()
    reader_fd = os.dup(stream.fileno())

    def read():
        try:
            # A daemon reader must not hold Python's sys.stdin lock at exit.
            with os.fdopen(reader_fd, "rb") as reader:
                for line in reader:
                    messages.put(json.loads(line))
        finally:
            messages.put(None)  # EOF, including an unexpected peer exit

    Thread(target=read, daemon=True).start()
    return messages


def _drain(messages):
    while True:
        try:
            yield messages.get_nowait()
        except Empty:
            return


def _send(stream, message):
    stream.write(json.dumps(message, ensure_ascii=False) + "\n")
    stream.flush()


def _display_pose(pose):
    if pose is None:
        return None
    xyz, rotation = pose
    return (xyz * 1000).tolist(), rotation.as_rotvec(degrees=True).tolist()


class TeleopPanel:
    """Qt owns the Cocoa event loop in a separate process from the MuJoCo viewer."""

    def __init__(self, *, simulation, device_name, dry_run, robot_ip=None, open_width_mm=None,
                 command_mode="eef-delta", control_law="position"):
        from .rg2 import DEFAULT_OPEN_WIDTH_MM, normalize_open_width_mm

        if open_width_mm is None:
            open_width_mm = DEFAULT_OPEN_WIDTH_MM
        self.open_width_mm = normalize_open_width_mm(open_width_mm)
        settings = dict(simulation=simulation, device_name=device_name, dry_run=dry_run, robot_ip=robot_ip,
                        open_width_mm=self.open_width_mm, command_mode=command_mode, control_law=control_law)
        self._process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), json.dumps(settings)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8",
            # The controller handles terminal Ctrl+C. Qt exits on pipe EOF after cleanup.
            start_new_session=(os.name == "posix"),
        )
        self._messages = _incoming(self._process.stdout)
        self._open = True
        self._command_mode = command_mode
        self._control_law = control_law
        self._command_mode_request = None
        self._control_law_request = None
        self._target_event = None
        self.scale = 1.0
        self._can_send = True
        self._next_update = 0.0
        try:
            if self._messages.get(timeout=10) != "ready":
                raise RuntimeError("UR5e 조작 패널을 열 수 없습니다.")
        except Empty:
            self.close()
            raise RuntimeError("UR5e 조작 패널이 시작 시간 안에 응답하지 않았습니다.") from None
        except BaseException:
            self.close()
            raise

    @property
    def is_open(self):
        return self._open and self._process.poll() is None

    def poll(self):
        request = None
        self._command_mode_request = None
        self._control_law_request = None
        self._target_event = None
        for event in _drain(self._messages):
            if event is None:
                self._open = False
                break
            if "mode" in event:
                request = event["mode"]
            if "command_mode" in event:
                self._command_mode_request = event["command_mode"]
            if "control_law" in event:
                self._control_law_request = event["control_law"]
            if "target" in event:
                self._target_event = event["target"]
            if "scale" in event:
                self.scale = event["scale"]
            if "open_width_mm" in event:
                self.open_width_mm = event["open_width_mm"]
            if "ack" in event:
                self._can_send = True
        if not self._open:
            self._command_mode_request = None
            self._control_law_request = None
        if (not self._open or request is not None or self._command_mode_request is not None
                or self._control_law_request is not None):
            self._target_event = None
        return request

    def take_command_mode(self):
        request, self._command_mode_request = self._command_mode_request, None
        return request

    def take_control_law(self):
        request, self._control_law_request = self._control_law_request, None
        return request

    def take_target(self):
        event, self._target_event = self._target_event, None
        if (event is None or event.get("command_mode") != self._command_mode
                or event.get("control_law") != self._control_law):
            return None
        if 0 <= time.monotonic() - event["time"] <= TARGET_TIMEOUT:
            return event["values"]
        return None

    def refresh(self, *, dry_run, clutch, measured, target, message="", gripper="",
                measured_joints=None, target_joints=None, operator_state="waiting-input"):
        now = time.monotonic()
        if not self.is_open or not self._can_send or now < self._next_update:
            return
        # At most one small display frame can be outstanding, even if Qt stalls.
        state = dict(dry_run=dry_run, clutch=clutch, operator_state=operator_state, measured=_display_pose(measured),
                     target=_display_pose(target), message=message[:256], gripper=gripper[:256],
                     measured_joints=None if measured_joints is None else list(map(math.degrees, measured_joints)),
                     target_joints=None if target_joints is None else list(map(math.degrees, target_joints)))
        try:
            _send(self._process.stdin, state)
        except (BrokenPipeError, OSError):
            self._open = False
            return
        self._can_send = False
        self._next_update = now + 0.1

    def mode_result(self, *, dry_run, message=""):
        self._send_mode_result("mode_result", message, dry_run=dry_run)

    def command_mode_result(self, *, command_mode, message=""):
        self._command_mode = command_mode
        self._send_mode_result("command_mode_result", message, command_mode=command_mode)

    def control_law_result(self, *, control_law, message=""):
        self._control_law = control_law
        self._send_mode_result("control_law_result", message, control_law=control_law)

    def _send_mode_result(self, key, message, **state):
        """Acknowledge a switch independently of the telemetry frame rate."""
        if self.is_open:
            try:
                _send(self._process.stdin, {key: {**state, "message": message[:256]}})
            except (BrokenPipeError, OSError):
                self._open = False

    def close(self):
        self._open = False
        try:
            self._process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._process.stdout.close()


def _label(text, *, role=None, wrap=False, emphasis=0):
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QLabel, QSizePolicy

    label = QLabel(text)
    label.setTextFormat(Qt.PlainText)
    if role:
        label.setProperty("role", role)
    if wrap:
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    if emphasis:
        font = label.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() + emphasis)
        label.setFont(font)
    return label


def _set_status(label, text):
    # QLabel only wraps at word boundaries. Driver names and addresses can be
    # longer than the whole status column on a small screen.
    def break_token(match):
        token = match.group()
        return "\u200b".join(token[i:i + 12] for i in range(0, len(token), 12))

    display = re.sub(r"\S{20,}", break_token, text)
    label.setText(display)
    label.setToolTip(text if display != text else "")


def _section(parent_layout):
    from PyQt5.QtWidgets import QFrame, QVBoxLayout

    frame = QFrame()
    frame.setObjectName("section")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(11, 11, 11, 12)
    layout.setSpacing(6)
    parent_layout.addWidget(frame)
    return layout


def _rule(layout):
    from PyQt5.QtWidgets import QFrame

    rule = QFrame()
    rule.setObjectName("rule")
    rule.setFixedHeight(1)
    layout.addWidget(rule)


class _PanelWindow:
    """Show the active configuration until the control loop confirms a request."""

    def __init__(self, *, simulation, device_name, dry_run, open_width_mm, robot_ip=None,
                 command_mode="eef-delta", control_law="position"):
        from PyQt5.QtCore import Qt, QTimer
        from PyQt5.QtWidgets import QApplication, QFrame, QScrollArea, QVBoxLayout, QWidget

        app = QApplication.instance()
        if app is None:
            # Linux OpenCV wheels can point Qt at their own incompatible xcb plugin.
            if sys.platform.startswith("linux"):
                for key in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
                    if "cv2" in Path(os.environ.get(key, "")).parts:
                        os.environ.pop(key, None)
            app = QApplication([])
        self.app = app
        self._simulation = simulation
        self._operator_state = None
        self._applied_dry_run = dry_run
        self._control_law = control_law
        self._pending_switch = None
        self._switch_request = {}
        self._last_measured = self._target_event = self._command_mode = None
        self._loaded_absolute = False
        self._first_frame = True

        self.window = QWidget()
        self.window.setObjectName("panel")
        self.window.setAttribute(Qt.WA_QuitOnClose, False)
        self.window.setWindowTitle("UR5e 조작 패널")
        self._feedback_timer = QTimer(self.window)
        self._feedback_timer.setSingleShot(True)
        self._feedback_timer.setInterval(1500)  # Display frames normally arrive at 10 Hz.
        self._feedback_timer.timeout.connect(self._clear_operator_state)
        outer = QVBoxLayout(self.window)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self._build_header(outer, simulation)
        self._show_mode(dry_run)
        self._update_state_header()

        self.scroll = QScrollArea()
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setWidgetResizable(True)
        outer.addWidget(self.scroll, 1)
        self.body = QWidget()
        self.body.setObjectName("body")
        layout = QVBoxLayout(self.body)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        self.message_label = _label("", wrap=True)
        self.message_label.setObjectName("message")
        self.message_label.hide()
        layout.addWidget(self.message_label)
        self._build_inputs(_section(layout), device_name, open_width_mm)
        self._build_targets(_section(layout))
        layout.addStretch()
        self.scroll.setWidget(self.body)
        self._build_footer(outer, simulation, robot_ip)
        self.window.setStyleSheet(Path(__file__).with_suffix(".qss").read_text(encoding="utf-8"))
        self._show_command_mode(command_mode)

        self.window.ensurePolished()
        self.window.layout().activate()
        # Obtain native frame margins without flashing a default-sized window.
        self.window.winId()
        self._fit_window(initial=True)
        self.window.show()
        self._automatic_size = self.window.size()
        QTimer.singleShot(0, self._fit_window)

    def _build_header(self, outer, simulation):
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QFrame, QHBoxLayout, QPushButton, QVBoxLayout

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
        self.state_label = _label("", wrap=True, emphasis=6)
        self.state_label.setObjectName("title")
        self.state_label.setAccessibleDescription("현재 조작 상태")
        title_row.addWidget(self.state_label, 1)
        layout.addLayout(title_row)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.dry_button = QPushButton("DRY RUN · 미리보기")
        self.run_button = QPushButton("SIM RUN · 시뮬레이션" if simulation else "REAL RUN · 실기 구동")
        for button, dry, name in ((self.dry_button, True, "dry"),
                                  (self.run_button, False, "sim" if simulation else "real")):
            button.setObjectName(name)
            button.setProperty("role", "mode")
            button.setAttribute(Qt.WA_LayoutUsesWidgetRect)
            button.setCheckable(True)
            button.clicked.connect(lambda checked, requested=dry: self._request_mode(requested))
            row.addWidget(button, 1)
        layout.addLayout(row)
        outer.addWidget(header)

    def _update_state_header(self):
        if self._pending_switch:
            title = {"mode": "실행 모드 전환 중", "command_mode": "명령 모드 전환 중",
                     "control_law": "제어 방식 전환 중"}[self._pending_switch]
            color = "attention"
        elif self._operator_state == "active":
            if self._applied_dry_run:
                title, color = "미리보기 조작 중", "preview"
            elif self._simulation:
                title, color = "시뮬레이션 조작 중", "active"
            else:
                title, color = "실기 조작 중", "real"
        else:
            title, color = {
                "waiting-device": ("입력 장치 연결 대기", "idle"),
                "waiting-input": ("입력 신호 대기", "idle"),
                "waiting-clutch": ("클러치 확인 대기", "attention"),
                "release-clutch": ("클러치를 놓으세요", "attention"),
                "ready": ("조작 준비 완료", "ready"),
            }.get(self._operator_state, ("상태 수신 대기", "idle"))
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
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QDoubleSpinBox, QGridLayout, QHBoxLayout, QSlider

        device = {"spacemouse": "SpaceMouse", "quest": "Quest 3"}.get(device_name, device_name)
        top = QHBoxLayout()
        top.addWidget(_label("입력 장치", emphasis=2))
        top.addStretch()
        top.addWidget(_label(device, role="muted", wrap=True))
        layout.addLayout(top)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 1)
        self.clutch_label = _label("입력 대기", wrap=True)
        self.gripper_label = _label("상태 수신 대기", wrap=True)
        for row, name, status in ((0, "클러치", self.clutch_label), (1, "그리퍼", self.gripper_label)):
            grid.addWidget(_label(name, role="muted"), row, 0, Qt.AlignTop)
            grid.addWidget(status, row, 1, 1, 2)
        self.gain = QSlider(Qt.Horizontal)
        self.gain.setAccessibleName("입력 배율")
        self.gain.setRange(10, 100)
        self.gain.setValue(100)
        gain_label = _label("입력 배율", role="muted")
        gain_label.setBuddy(self.gain)
        gain_value = _label("100%")
        gain_value.setMinimumWidth(gain_value.sizeHint().width())
        gain_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.gain.valueChanged.connect(lambda value: gain_value.setText(f"{value}%"))
        grid.addWidget(gain_label, 2, 0)
        grid.addWidget(self.gain, 2, 1)
        grid.addWidget(gain_value, 2, 2)

        width_label = _label("열림 목표 폭", role="muted")
        self.open_width = QDoubleSpinBox()
        width_label.setBuddy(self.open_width)
        self.open_width.setAccessibleName(width_label.text())
        self.open_width.setDecimals(1)
        self.open_width.setRange(0.0, 110.0)
        self.open_width.setSingleStep(0.1)
        self.open_width.setSuffix(" mm")
        self.open_width.setKeyboardTracking(False)
        self.open_width.setValue(open_width_mm)
        grid.addWidget(width_label, 3, 0)
        grid.addWidget(self.open_width, 3, 1, 1, 2, Qt.AlignLeft)
        layout.addLayout(grid)

    def _build_targets(self, layout):
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QFontDatabase
        from PyQt5.QtWidgets import QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QPushButton

        class ModePicker(QComboBox):
            def wheelEvent(self, event):
                # Scrolling the panel should never request a controller change.
                event.ignore()

        layout.addWidget(_label("목표 제어", emphasis=2))
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
            caption = _label(name, role="muted")
            caption.setBuddy(picker)
            picker.setAccessibleName(name)
            config.addWidget(caption, row, 0)
            config.addWidget(picker, row, 1)
        layout.addLayout(config)
        self.coordinate_label = _label("", role="muted", wrap=True)
        layout.addWidget(self.coordinate_label)
        _rule(layout)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(5)
        grid.setVerticalSpacing(3)
        self.entry_heading = _label("", role="column")
        for column, heading in enumerate((_label("축", role="column"), _label("실측", role="column"),
                                          _label("현재 목표", role="column"), self.entry_heading)):
            if column:
                heading.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(heading, 0, column)
            grid.setColumnStretch(column, (1, 2, 2, 3)[column])
        self.axis_labels, self.measured_cells, self.target_cells, self.target_inputs = [], [], [], []
        fixed_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        fixed_font.setPointSizeF(self.window.font().pointSizeF())
        for row in range(1, 7):
            axis = _label("")
            self.axis_labels.append(axis)
            grid.addWidget(axis, row, 0)
            for column, cells in enumerate((self.measured_cells, self.target_cells), 1):
                cell = _label("—")
                cell.setFont(fixed_font)
                cell.setMinimumWidth(cell.fontMetrics().horizontalAdvance("-2000.0"))
                cell.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                grid.addWidget(cell, row, column)
                cells.append(cell)
            box = QDoubleSpinBox()
            box.setFont(fixed_font)
            box.setAlignment(Qt.AlignRight)
            box.setDecimals(2)
            box.setSingleStep(1.0)
            box.setKeyboardTracking(False)
            axis.setBuddy(box)
            self.target_inputs.append(box)
            grid.addWidget(box, row, 3)
        layout.addLayout(grid)

        layout.addWidget(_label("클러치를 누른 상태에서 목표를 적용하세요.", role="muted", wrap=True))
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

    def _build_footer(self, outer, simulation, robot_ip):
        from PyQt5.QtWidgets import QFrame, QHBoxLayout, QPushButton

        footer = QFrame()
        footer.setObjectName("footer")
        row = QHBoxLayout(footer)
        row.setContentsMargins(16, 6, 16, 6)
        if simulation:
            row.addStretch(1)
        else:
            source = f"RTDE · {robot_ip}" if robot_ip else "RTDE"
            row.addWidget(_label(f"데이터 소스   {source}", role="muted", wrap=True), 1)
        exit_button = QPushButton("세션 종료")
        exit_button.clicked.connect(self.close)
        row.addWidget(exit_button)
        outer.addWidget(footer)

    def _fit_window(self, *, initial=False):
        from PyQt5.QtCore import QCoreApplication, QEvent
        from PyQt5.QtWidgets import QStyle

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
        outer = window.layout()
        # QScrollArea.sizeHint() does not account for text wrapped at viewport width.
        chrome_height = sum(outer.itemAt(i).heightForWidth(width) if outer.itemAt(i).hasHeightForWidth()
                            else outer.itemAt(i).sizeHint().height() for i in (0, 2))
        body_height = max(body.minimumSize().height(), body.heightForWidth(width))
        height = min(available_height, body_height + chrome_height)
        window.resize(width, height if initial else max(height, window.height()))
        if initial:
            x = max(area.left(), area.right() + 1 - window.width() - frame.right() - frame.left() - 12)
            window.move(x, area.top())
        self._automatic_size = window.size()

    @staticmethod
    def _select_confirmed(picker, value):
        from PyQt5.QtCore import QSignalBlocker

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

    def _begin_switch(self, kind, value, caption):
        if self._pending_switch is not None:
            return
        self._pending_switch = kind
        self._switch_request[kind] = value
        self._target_event = None
        self._enable_target_controls()
        self._update_state_header()
        self._set_message(f"{caption} 전환 중…")

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
            self._reset_target_editor()
        self._finish_switch("control_law", message)

    def _set_message(self, message, *, reveal=False):
        _set_status(self.message_label, message)
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
        idle = self._pending_switch is None
        self.run_button.setEnabled(idle)
        self.dry_button.setEnabled(idle)
        self.command_picker.setEnabled(idle)
        self.control_law_picker.setEnabled(idle)
        ready = idle and self._last_measured is not None
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
        values = ([math.radians(v) for v in display] if self._joint_space else
                  [v / 1000 for v in display[:3]] + [math.radians(v) for v in display[3:]])
        self._target_event = {"time": time.monotonic(), "values": values,
                              "command_mode": self._command_mode, "control_law": self._control_law}

    def take_target_event(self):
        event, self._target_event = self._target_event, None
        return event

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
        _set_status(self.clutch_label, clutch)
        _set_status(self.gripper_label, gripper)
        if self._joint_space:
            measured, target = measured_joints, target_joints
        else:
            measured = None if measured is None else [*measured[0], *measured[1]]
            target = None if target is None else [*target[0], *target[1]]
        self._last_measured = measured
        if measured is None:
            self._loaded_absolute = False
        for cells, values in ((self.measured_cells, measured), (self.target_cells, target)):
            for i, cell in enumerate(cells):
                cell.setText("—" if values is None else f"{values[i]:.1f}")
        self._enable_target_controls()
        if self._first_frame:
            from PyQt5.QtCore import QTimer

            self._first_frame = False
            QTimer.singleShot(0, self._fit_window)

    def close(self):
        self.window.close()


def _run_window(settings):
    from PyQt5.QtCore import QTimer

    window = _PanelWindow(**settings)
    messages = _incoming(sys.stdin)
    previous_scale = window.scale
    previous_width = window.open_width_mm

    def exchange():
        nonlocal previous_scale, previous_width
        if not window.is_open:
            window.app.quit()
            return
        event = {}
        for state in _drain(messages):
            if state is None:
                window.app.quit()
                return
            if "mode_result" in state:
                window.mode_result(**state["mode_result"])
            elif "command_mode_result" in state:
                window.command_mode_result(**state["command_mode_result"])
            elif "control_law_result" in state:
                window.control_law_result(**state["control_law_result"])
            else:
                window.refresh(**state)
                event["ack"] = True
        requested = window.take_request()
        if requested is not None:
            event["mode"] = requested
        command_request = window.take_command_mode()
        if command_request is not None:
            event["command_mode"] = command_request
        law_request = window.take_control_law()
        if law_request is not None:
            event["control_law"] = law_request
        target = window.take_target_event()
        if target is not None and requested is None and command_request is None and law_request is None:
            event["target"] = target
        if window.scale != previous_scale:
            previous_scale = window.scale
            event["scale"] = previous_scale
        if window.open_width_mm != previous_width:
            previous_width = window.open_width_mm
            event["open_width_mm"] = previous_width
        if event:
            try:
                _send(sys.stdout, event)
            except (BrokenPipeError, OSError):
                window.app.quit()

    timer = QTimer()
    timer.timeout.connect(exchange)
    timer.start(16)
    _send(sys.stdout, "ready")
    window.app.exec_()
    window.close()


if __name__ == "__main__":
    _run_window(json.loads(sys.argv[1]))
