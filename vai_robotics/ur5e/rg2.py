"""OnRobot RG2 on a single Quick Changer, via its Compute Box Modbus TCP server."""

import math
import threading
import time


DEFAULT_GRIP_FORCE_N = 40.0
DEFAULT_OPEN_WIDTH_MM = 75.0


def normalize_open_width_mm(width_mm):
    """Return an RG2 target width at the Compute Box's 0.1 mm resolution."""
    if not math.isfinite(width_mm) or not 0 <= width_mm <= 110.0:
        raise ValueError("RG2 open width must be between 0 and 110 mm.")
    return round(width_mm * 10) / 10


class RG2:
    """Poll measured aperture and handle button edges outside the arm's RTDE loop."""

    UNIT = 65
    # Encoder feedback can overshoot the nominal 110 mm travel at the end stop.
    # This plausibility check rejects grossly invalid words such as 0xffff.
    MAX_FEEDBACK_WIDTH_MM = 120.0
    INPUT_TIMEOUT = 0.15
    FEEDBACK_TIMEOUT = 0.30

    def __init__(self, host, *, port=502, force_n=DEFAULT_GRIP_FORCE_N, open_width_mm=DEFAULT_OPEN_WIDTH_MM):
        if not host or not host.strip():
            raise ValueError("Provide the RG2 Compute Box address.")
        if not 1 <= port <= 65535 or not math.isfinite(force_n) or not 3 <= force_n <= 40:
            raise ValueError("RG2 requires a valid TCP port and a gripping force from 3 to 40 N.")
        self.host, self.port = host, port
        self._force = round(force_n * 10)
        self._lock = threading.Lock()
        self.set_open_width(open_width_mm)
        self._write_gate = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._client = None
        self._width = None
        self._feedback_at = self._input_at = -math.inf
        self._status = "RG2 연결 대기"
        self._safety = False
        self._reset_input()

    def _reset_input(self):
        self._chord = None
        self._pending = None
        self._issued = None
        self._stopping = False
        self._own_motion = False

    def __enter__(self):
        if self._thread is not None:
            raise RuntimeError("RG2 session is already open.")
        try:
            from pymodbus.client import ModbusTcpClient
        except ImportError as exc:
            raise RuntimeError("Install vai_robotics/requirements-teleop.txt for RG2 support.") from exc
        self._client = ModbusTcpClient(self.host, port=self.port, timeout=0.20, retries=0)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="OnRobot RG2", daemon=True)
        self._thread.start()
        return self

    @property
    def open_width_mm(self):
        with self._lock:
            return self._open_width_mm

    def set_open_width(self, width_mm):
        """Set the target for the next open command, in 0.1 mm increments."""
        width_mm = normalize_open_width_mm(width_mm)
        with self._lock:
            self._open_width_mm = width_mm

    @property
    def width_mm(self):
        with self._lock:
            if time.monotonic() - self._feedback_at < self.FEEDBACK_TIMEOUT:
                return self._width
            return None

    @property
    def status(self):
        with self._lock:
            if self._width is not None and time.monotonic() - self._feedback_at >= self.FEEDBACK_TIMEOUT:
                return "RG2 피드백 지연"
            return self._status

    def update(self, chord):
        """Supply both-button state every tick in REAL, or None to revoke control."""
        now = time.monotonic()
        with self._lock:
            if (chord is None or self._safety or self._stop.is_set() or
                    now - self._feedback_at >= self.FEEDBACK_TIMEOUT):
                self._reset_input()
                return
            if now - self._input_at >= self.INPUT_TIMEOUT:
                self._reset_input()
            self._input_at = now
            if chord and self._chord is False:
                self._pending = (True, 0)
            elif not chord and self._chord is True:
                # Release opens only after this session actually sent a close.
                self._pending = (False, round(self._open_width_mm * 10)) if self._issued is True else None
            self._chord = bool(chord)

    def wait_for_writes(self):
        """After update(None), drain an in-flight command during a mode change."""
        with self._write_gate:
            pass

    def _command_for_feedback(self, width, flags):
        busy = bool(flags & 1)
        now = time.monotonic()
        with self._lock:
            self._width = width / 10
            self._feedback_at = now
            self._safety = bool(flags & 0x7c)  # S1/S2 pushed, triggered, or safety error.
            self._status = f"RG2 {self._width:.1f} mm"
            if self._safety:
                self._status += f" · 안전 스위치 (0x{flags:04x})"
            if self._safety or self._stop.is_set() or now - self._input_at >= self.INPUT_TIMEOUT:
                self._reset_input()
                return
            if not busy:
                self._own_motion = self._stopping = False
            if self._pending is None:
                return
            if busy:
                # A busy RG2 ignores new grip targets. Stop only our own travel,
                # then send the latest requested direction once idle.
                if not self._own_motion or self._stopping:
                    return
                self._stopping = True
                return 2, [8]
            self._issued, target_width = self._pending
            self._pending = None
            self._own_motion = True
            # Force (0.1 N), width (0.1 mm), grip in the same reference as 267.
            return 0, [self._force, target_width, 1]

    def _poll(self):
        # FC3, actual width (aluminum finger spacing in tenths of mm) and status.
        feedback = self._client.read_holding_registers(267, count=2, device_id=self.UNIT)
        if feedback.isError() or len(feedback.registers) != 2:
            raise OSError(f"RG2 feedback failed: {feedback}")
        width, flags = feedback.registers
        if not 0 <= width <= 10 * self.MAX_FEEDBACK_WIDTH_MM:
            raise OSError(f"Invalid RG2 actual width: {width / 10:.1f} mm")
        with self._write_gate:
            command = self._command_for_feedback(width, flags)
            if command is not None:
                address, values = command
                result = self._client.write_registers(address, values, device_id=self.UNIT)
                if result.isError():
                    raise OSError(f"RG2 command failed: {result}")

    def _disconnect(self, reason):
        with self._lock:
            self._reset_input()
            self._width = None
            self._feedback_at = -math.inf
            self._status = reason
        self._client.close()

    def _run(self):
        from pymodbus.exceptions import ModbusException

        try:
            while not self._stop.is_set():
                try:
                    self._poll()
                except (OSError, ModbusException) as error:
                    # Never replay an unacknowledged write after reconnect.
                    self._disconnect(f"RG2 연결 오류 · {error}")
                    self._stop.wait(1.0)
                else:
                    self._stop.wait(0.04)
        finally:
            self._disconnect("RG2 연결 종료")

    def close(self):
        if self._thread is not None:
            self._stop.set()
            self.update(None)
            self._thread.join()
            self._thread = None

    def __exit__(self, *exc):
        self.close()
