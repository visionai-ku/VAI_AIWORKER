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


def _start_window_process(**settings):
    return subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), json.dumps(settings)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        # The controller handles terminal Ctrl+C. Qt exits on pipe EOF after cleanup.
        start_new_session=(os.name == "posix"),
    )


class TeleopPanel:
    """Qt owns the Cocoa event loop in a separate process from the MuJoCo viewer."""

    _targets_valid_since = -math.inf

    def __init__(self, *, simulation, device_name, dry_run, robot_ip=None, open_width_mm=None,
                 command_mode="eef-delta", control_law="position", joint_rings=False):
        from .rg2 import DEFAULT_OPEN_WIDTH_MM, normalize_open_width_mm

        if open_width_mm is None:
            open_width_mm = DEFAULT_OPEN_WIDTH_MM
        self.open_width_mm = normalize_open_width_mm(open_width_mm)
        self._process = _start_window_process(
            simulation=simulation,
            device_name=device_name,
            dry_run=dry_run,
            robot_ip=robot_ip,
            open_width_mm=self.open_width_mm,
            command_mode=command_mode,
            control_law=control_law,
            joint_rings=joint_rings,
        )
        self._messages = _incoming(self._process.stdout)
        self._open = True
        self._command_mode = command_mode
        self._control_law = control_law
        self._requests = {}
        self._record_request = None
        self._target_event = None
        self.scale = 1.0
        self._can_send = True
        self._next_update = 0.0
        self._wait_until_ready()

    def _wait_until_ready(self):
        try:
            hello = self._messages.get(timeout=10)
            if not isinstance(hello, dict) or hello.get("ready") is not True:
                raise RuntimeError("UR5e 조작 패널을 열 수 없습니다.")
            self.rg2_mount = tuple(hello["rg2_mount"])
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
            if event.get("record") in ("start", "save", "discard"):
                # The panel permits one recording request until an explicit result arrives.
                self._record_request = event["record"]
            if "scale" in event:
                self.scale = event["scale"]
            if "open_width_mm" in event:
                self.open_width_mm = event["open_width_mm"]
            if "rg2_mount" in event:
                self.rg2_mount = tuple(event["rg2_mount"])
            if "ack" in event:
                self._can_send = True
        if not self._open:
            self._requests.clear()
            if self._record_request != "save":
                self._record_request = None
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

    def take_record_request(self):
        request, self._record_request = self._record_request, None
        return request

    def take_target(self):
        event, self._target_event = self._target_event, None
        if (event is None or event.get("command_mode") != self._command_mode
                or event.get("control_law") != self._control_law):
            return None
        if event["time"] >= self._targets_valid_since and 0 <= time.monotonic() - event["time"] <= TARGET_TIMEOUT:
            return event["values"]
        return None

    def refresh(self, *, dry_run, clutch, measured, target, message="", gripper="",
                measured_joints=None, target_joints=None, operator_state="waiting-input", recording=None,
                speed_comparison=None, tool_payload=None):
        now = time.monotonic()
        if not self.is_open or not self._can_send or now < self._next_update:
            return
        # At most one small display frame can be outstanding, even if Qt stalls.
        state = dict(dry_run=dry_run, clutch=clutch, operator_state=operator_state, measured=_display_pose(measured),
                     target=_display_pose(target), message=message[:256], gripper=gripper[:256],
                     measured_joints=None if measured_joints is None else list(map(math.degrees, measured_joints)),
                     target_joints=None if target_joints is None else list(map(math.degrees, target_joints)))
        if recording is not None:
            state["recording"] = recording
        if speed_comparison is not None:
            state["speed_comparison"] = speed_comparison
        if tool_payload is not None:
            state["tool_payload"] = tool_payload
        try:
            _send(self._process.stdin, state)
        except (BrokenPipeError, OSError):
            self._open = False
            return
        self._can_send = False
        self._next_update = now + 0.1

    def mode_result(self, *, dry_run, message=""):
        self._send_result("mode_result", message, dry_run=dry_run)

    def command_mode_result(self, *, command_mode, message=""):
        self._command_mode = command_mode
        self._send_result("command_mode_result", message, command_mode=command_mode)

    def control_law_result(self, *, control_law, message=""):
        self._control_law = control_law
        self._send_result("control_law_result", message, control_law=control_law)

    def impedance_gains_result(self, *, space, gains, message=""):
        if not message and gains is not None:
            # A late target tagged with the same mode and law can predate these gains.
            self._targets_valid_since = time.monotonic()
            self._target_event = None
        self._send_result("impedance_gains_result", message, space=space, gains=gains)

    def recording_result(self, recording):
        self._send_result("recording_result", recording=recording)

    def _send_result(self, key, message=None, **state):
        """Acknowledge an operator request independently of the telemetry frame rate."""
        if self.is_open:
            if message is not None:
                state["message"] = message[:256]
            try:
                _send(self._process.stdin, {key: state})
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


class _WindowConnection:
    """Qt side of the pipe. Each timer tick consumes updates before sending edits."""

    def __init__(self, window):
        self.window = window
        self.messages = _incoming(sys.stdin)
        self.result_handlers = {
            "mode_result": window.mode_result,
            "command_mode_result": window.command_mode_result,
            "control_law_result": window.control_law_result,
            "impedance_gains_result": window.impedance_gains_result,
            "recording_result": window.recording_result,
        }
        self.previous_scale = window.scale
        self.previous_width = window.open_width_mm
        self.previous_mount = window.rg2_mount

    def exchange(self):
        if not self.window.is_open:
            self.window.app.quit()
            return
        reply = self._receive_updates()
        if reply is None:
            self.window.app.quit()
            return
        reply.update(self._take_operator_events())
        if reply:
            try:
                _send(sys.stdout, reply)
            except (BrokenPipeError, OSError):
                self.window.app.quit()

    def _receive_updates(self):
        """Apply controller messages, returning the telemetry acknowledgement or None on EOF."""
        reply = {}
        for state in _drain(self.messages):
            if state is None:
                return None
            for name, handler in self.result_handlers.items():
                if name in state:
                    handler(**state[name])
                    break
            else:
                self.window.refresh(**state)
                reply["ack"] = True
        return reply

    def _take_operator_events(self):
        events = self.window.take_operator_events()
        if self.window.scale != self.previous_scale:
            self.previous_scale = self.window.scale
            events["scale"] = self.previous_scale
        if self.window.open_width_mm != self.previous_width:
            self.previous_width = self.window.open_width_mm
            events["open_width_mm"] = self.previous_width
        if self.window.rg2_mount != self.previous_mount:
            self.previous_mount = self.window.rg2_mount
            events["rg2_mount"] = self.previous_mount
        return events


def _run_window(settings):
    from PyQt5.QtCore import QTimer

    from vai_robotics.ur5e.panel.window import PanelWindow

    window = PanelWindow(**settings)
    connection = _WindowConnection(window)
    timer = QTimer()
    timer.timeout.connect(connection.exchange)
    timer.start(16)
    _send(sys.stdout, {"ready": True, "rg2_mount": window.rg2_mount})
    window.app.exec_()
    # Flush a Save click if the window closed before the 16 ms exchange timer.
    if window.recording.take_request() == "save":
        try:
            _send(sys.stdout, {"record": "save"})
        except (BrokenPipeError, OSError):
            pass
    window.close()


if __name__ == "__main__":
    # The controller launches this file by absolute path, including from notebooks
    # outside the repository. Give that child the same checkout's package root.
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    _run_window(json.loads(sys.argv[1]))
