"""Operator controls for a running UR5e teleoperation session."""

import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
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
        self._target_event = None
        for event in _drain(self._messages):
            if event is None:
                self._open = False
                break
            if "mode" in event:
                request = event["mode"]
            if "target" in event:
                self._target_event = event["target"]
            if "scale" in event:
                self.scale = event["scale"]
            if "open_width_mm" in event:
                self.open_width_mm = event["open_width_mm"]
            if "ack" in event:
                self._can_send = True
        if request is not None or not self._open:
            self._target_event = None
        return request

    def take_target(self):
        event, self._target_event = self._target_event, None
        if event is not None and 0 <= time.monotonic() - event["time"] <= TARGET_TIMEOUT:
            return event["values"]
        return None

    def refresh(self, *, dry_run, clutch, measured, target, message="", gripper="",
                measured_joints=None, target_joints=None):
        now = time.monotonic()
        if not self.is_open or not self._can_send or now < self._next_update:
            return
        # At most one small display frame can be outstanding, even if Qt stalls.
        state = dict(dry_run=dry_run, clutch=clutch, measured=_display_pose(measured),
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
        """Acknowledge one requested switch, independently of the telemetry frame rate."""
        if self.is_open:
            try:
                _send(self._process.stdin, {"mode_result": {"dry_run": dry_run, "message": message[:256]}})
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


class _PanelWindow:
    """The operator's selection is a request until confirmed by the control loop."""

    def __init__(self, *, simulation, device_name, dry_run, open_width_mm, robot_ip=None,
                 command_mode="eef-delta", control_law="position"):
        from PyQt5.QtCore import Qt
        from PyQt5.QtGui import QFontDatabase, QFontMetrics
        from PyQt5.QtWidgets import (
            QApplication, QDoubleSpinBox, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton,
            QScrollArea, QSlider, QVBoxLayout, QWidget,
        )

        app = QApplication.instance()
        if app is None:
            # Linux OpenCV wheels can point Qt at their own incompatible xcb plugin.
            if sys.platform.startswith("linux"):
                for key in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
                    if "cv2" in Path(os.environ.get(key, "")).parts:
                        os.environ.pop(key, None)
            app = QApplication([])
        self.app = app
        self._applied_dry_run = dry_run
        self._requested_dry_run = None
        self._awaiting_mode_result = False
        self._joint_space = command_mode.startswith("joint-")
        self._absolute = command_mode.endswith("-absolute")
        self._last_measured = None
        self._loaded_absolute = False
        self._target_event = None

        self.window = QWidget()
        self.window.setAttribute(Qt.WA_QuitOnClose, False)
        self.window.setWindowTitle("UR5e 조작 패널")
        content = QWidget()
        scroll = QScrollArea()
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        outer = QVBoxLayout(self.window)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 14)
        layout.setSpacing(10)
        device_label = {"spacemouse": "SpaceMouse", "quest": "Quest 3"}.get(device_name, device_name)
        title = QLabel(f"UR5e {'시뮬레이션' if simulation else '실기'} · {device_label}")
        font = title.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() + 2)
        title.setFont(font)
        layout.addWidget(title)

        modes = QGroupBox("실행 모드")
        row = QHBoxLayout(modes)
        self.dry_button = QPushButton("DRY RUN · 미리보기")
        self.run_button = QPushButton("SIM RUN · 시뮬레이션" if simulation else "REAL RUN · 실기 구동")
        for button, value, color in (
            (self.dry_button, True, "#275671"),
            (self.run_button, False, "#30654f" if simulation else "#994729"),
        ):
            button.setCheckable(True)
            button.setMinimumHeight(32)
            selected_font = button.font()
            selected_font.setBold(True)
            button.setMinimumWidth(QFontMetrics(selected_font).horizontalAdvance(button.text()) + 16)
            button.setStyleSheet(
                f"QPushButton:checked {{ background-color: {color}; color: white; "
                "border-radius: 5px; font-weight: bold; }"
            )
            button.clicked.connect(lambda checked, requested=value: self._request_mode(requested))
            row.addWidget(button)
        self._show_mode(dry_run)
        layout.addWidget(modes)

        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        self.message_label.setTextFormat(Qt.PlainText)
        self.message_label.hide()
        layout.addWidget(self.message_label)

        gain_row = QHBoxLayout()
        gain_row.addWidget(QLabel("입력 배율"))
        self.gain = QSlider(Qt.Horizontal)
        self.gain.setAccessibleName("입력 배율")
        self.gain.setRange(10, 100)
        self.gain.setValue(100)
        gain_row.addWidget(self.gain, 1)
        gain_value = QLabel("100%")
        gain_value.setMinimumWidth(gain_value.sizeHint().width())
        gain_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.gain.valueChanged.connect(lambda value: gain_value.setText(f"{value}%"))
        gain_row.addWidget(gain_value)
        layout.addLayout(gain_row)

        self.clutch_label = QLabel("입력 대기")
        layout.addWidget(self.clutch_label)
        width_row = QHBoxLayout()
        width_label = QLabel("열림 목표 폭")
        width_row.addWidget(width_label)
        self.open_width = QDoubleSpinBox()
        width_label.setBuddy(self.open_width)
        self.open_width.setAccessibleName(width_label.text())
        self.open_width.setDecimals(1)
        self.open_width.setRange(0.0, 110.0)
        self.open_width.setSingleStep(0.1)
        self.open_width.setSuffix(" mm")
        self.open_width.setKeyboardTracking(False)
        self.open_width.setValue(open_width_mm)
        width_row.addWidget(self.open_width)
        width_row.addStretch(1)
        layout.addLayout(width_row)
        self.gripper_label = QLabel()
        self.gripper_label.setWordWrap(True)
        self.gripper_label.setTextFormat(Qt.PlainText)
        layout.addWidget(self.gripper_label)
        source = "MuJoCo" if simulation else "RTDE"
        if robot_ip and not simulation:
            source += f" · {robot_ip}"
        coordinate = "관절" if self._joint_space else "TCP (베이스 좌표계)"
        command = "절대 목표" if self._absolute else "상대 변화량"
        law = "위치 제어" if control_law == "position" else "임피던스 제어"
        targets = QGroupBox(f"{coordinate} · {command} · {law}")
        grid = QGridLayout(targets)
        self.measured_cells = []
        self.target_cells = []
        self.target_inputs = []
        fixed_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        for column, name in enumerate(("축", "실측", "목표", command)):
            heading = QLabel(name)
            grid.addWidget(heading, 0, column)
        axes = [f"J{i + 1}" for i in range(6)] if self._joint_space else ("X", "Y", "Z", "Rx", "Ry", "Rz")
        for i, axis in enumerate(axes):
            linear = not self._joint_space and i < 3
            unit = "mm" if linear else "°"
            axis_label = QLabel(f"{axis} ({unit})")
            grid.addWidget(axis_label, i + 1, 0)
            for column, cells in enumerate((self.measured_cells, self.target_cells), 1):
                cell = QLabel("—")
                cell.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                cell.setFont(fixed_font)
                grid.addWidget(cell, i + 1, column)
                cells.append(cell)
            box = QDoubleSpinBox()
            # UI units only; each backend owns reach, joint and slew limits.
            limit = 2000.0 if linear else 360.0
            box.setRange(-limit, limit)
            box.setDecimals(2)
            box.setSingleStep(1.0)
            box.setKeyboardTracking(False)
            box.setAccessibleName(f"{axis} {command} ({unit})")
            axis_label.setBuddy(box)
            self.target_inputs.append(box)
            grid.addWidget(box, i + 1, 3)
        readout = source if self._joint_space else f"{source} · Rx/Ry/Rz는 회전벡터"
        grid.addWidget(QLabel(readout), 7, 0, 1, 4)
        self.load_target = QPushButton("실측값 불러오기")
        self.load_target.setVisible(self._absolute)
        self.load_target.clicked.connect(self._load_measured)
        self.apply_target = QPushButton("목표 적용")
        self.apply_target.clicked.connect(self._submit_target)
        buttons = QHBoxLayout()
        if self._absolute:
            buttons.addWidget(self.load_target)
        buttons.addWidget(self.apply_target)
        grid.addLayout(buttons, 8, 0, 1, 4)
        grid.addWidget(QLabel("클러치를 누른 상태에서 목표를 적용하세요."), 9, 0, 1, 4)
        self._enable_target_controls()
        layout.addWidget(targets)

        exit_button = QPushButton("세션 종료")
        exit_button.clicked.connect(self.close)
        layout.addWidget(exit_button)
        screen = app.primaryScreen()
        size = content.sizeHint()
        # Reserve the scrollbar width as mode errors can wrap onto several lines.
        min_width = content.minimumSizeHint().width() + scroll.verticalScrollBar().sizeHint().width()
        self.window.setMinimumWidth(max(350, min_width))
        self.window.resize(max(self.window.minimumWidth(), size.width()), size.height())
        if screen:
            area = screen.availableGeometry()
            self.window.resize(self.window.width(), min(size.height(), max(1, area.height() - 64)))
            self.window.move(max(area.left(), area.right() - self.window.width() - 16),
                             area.top() + 32)
        self.window.show()

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
        if self._awaiting_mode_result or dry_run == self._applied_dry_run:
            return
        self._requested_dry_run = dry_run
        self._awaiting_mode_result = True
        self._target_event = None
        self._enable_target_controls()
        self.run_button.setEnabled(False)
        self.dry_button.setEnabled(False)
        button = self.dry_button if dry_run else self.run_button
        self.message_label.setText(f"{button.text()} 전환 중…")
        self.message_label.show()

    def take_request(self):
        request, self._requested_dry_run = self._requested_dry_run, None
        return request

    def _enable_target_controls(self):
        ready = not self._awaiting_mode_result and self._last_measured is not None
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
        self._target_event = {"time": time.monotonic(), "values": values}

    def take_target_event(self):
        event, self._target_event = self._target_event, None
        return event

    def mode_result(self, *, dry_run, message=""):
        self._awaiting_mode_result = False
        self.run_button.setEnabled(True)
        self.dry_button.setEnabled(True)
        self._show_mode(dry_run)
        self._enable_target_controls()
        self.message_label.setText(message)
        self.message_label.setVisible(bool(message))

    def refresh(self, *, dry_run, clutch, measured, target, message="", gripper="",
                measured_joints=None, target_joints=None):
        if not self._awaiting_mode_result:
            self._show_mode(dry_run)
            self.message_label.setText(message)
            self.message_label.setVisible(bool(message))
        self.clutch_label.setText(clutch)
        self.gripper_label.setText(gripper)
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
            else:
                window.refresh(**state)
                event["ack"] = True
        requested = window.take_request()
        if requested is not None:
            event["mode"] = requested
        target = window.take_target_event()
        if target is not None:
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
