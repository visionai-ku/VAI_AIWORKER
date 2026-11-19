"""Pipe protocol between the UR5e control loop and the separate Qt process."""

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
_CONFIG_REQUESTS = ("command_mode", "control_law", "impedance_gains")


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

    _targets_valid_since = -math.inf

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
        self._requests = {}
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
        """Drain pending UI events once per control tick, coalescing edits by kind."""
        run_mode = None
        self._requests = {}
        self._target_event = None
        for event in _drain(self._messages):
            if event is None:
                self._open = False
                break
            if "mode" in event:
                run_mode = event["mode"]
            for name in _CONFIG_REQUESTS:
                if name in event:
                    self._requests[name] = event[name]
            if "target" in event:
                self._target_event = event["target"]
            if "scale" in event:
                self.scale = event["scale"]
            if "open_width_mm" in event:
                self.open_width_mm = event["open_width_mm"]
            if "ack" in event:
                self._can_send = True
        if not self._open:
            self._requests.clear()
        # A target captured before a switch must never drive the new configuration.
        pending_configuration = any(value is not None for value in self._requests.values())
        if not self._open or run_mode is not None or pending_configuration:
            self._target_event = None
        return run_mode

    def take_command_mode(self):
        return self._requests.pop("command_mode", None)

    def take_control_law(self):
        return self._requests.pop("control_law", None)

    def take_impedance_gains(self):
        return self._requests.pop("impedance_gains", None)

    def take_target(self):
        event, self._target_event = self._target_event, None
        if (event is None or event.get("command_mode") != self._command_mode
                or event.get("control_law") != self._control_law):
            return None
        if event["time"] >= self._targets_valid_since and 0 <= time.monotonic() - event["time"] <= TARGET_TIMEOUT:
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

    def impedance_gains_result(self, *, space, gains, message=""):
        if not message and gains is not None:
            # A late target tagged with the same mode and law can predate these gains.
            self._targets_valid_since = time.monotonic()
            self._target_event = None
        self._send_mode_result("impedance_gains_result", message, space=space, gains=gains)

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


def __getattr__(name):
    # Compatibility for callers importing the window here. Headless controllers
    # can import TeleopPanel without importing Qt or loading a platform plugin.
    if name == "_PanelWindow":
        from .panel.window import PanelWindow

        return PanelWindow
    raise AttributeError(name)


def _run_window(settings):
    from PyQt5.QtCore import QTimer

    from vai_robotics.ur5e.panel.window import PanelWindow

    window = PanelWindow(**settings)
    results = {
        "mode_result": window.mode_result,
        "command_mode_result": window.command_mode_result,
        "control_law_result": window.control_law_result,
        "impedance_gains_result": window.impedance_gains_result,
    }
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
            result = next((name for name in results if name in state), None)
            if result is not None:
                results[result](**state[result])
            else:
                window.refresh(**state)
                event["ack"] = True
        event.update(window.take_operator_events())
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
    # The controller launches this file by absolute path, including from notebooks
    # outside the repository. Give that child the same checkout's package root.
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    _run_window(json.loads(sys.argv[1]))
