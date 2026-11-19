"""Read SpaceMouse HID reports and map the cap and buttons to teleop input."""

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..teleop_types import INPUT_TIMEOUT, InputSample


def _load_spacemouse():
    if sys.platform == "darwin":
        # easyhid 0.0.10 tests for "Darwin" in platform.platform(). Modern
        # Python reports "macOS", so pre-load hidapi symbols before importing it.
        from ctypes import CDLL, RTLD_GLOBAL
        from ctypes.util import find_library

        library = find_library("hidapi")
        if not library:
            for prefix in ("/opt/homebrew/lib", "/usr/local/lib"):
                candidate = Path(prefix) / "libhidapi.dylib"
                if candidate.is_file():
                    library = str(candidate)
                    break
        if not library:
            raise RuntimeError("Install the SpaceMouse HID library with: brew install hidapi")
        CDLL(library, mode=RTLD_GLOBAL)
    try:
        import pyspacemouse
        from easyhid import HIDException
    except ImportError as exc:
        raise RuntimeError("Install the optional SpaceMouse dependency: pip install pyspacemouse==2.1.0") from exc
    return pyspacemouse, HIDException


class SpaceMouseInput:
    """Integrate normalized SpaceMouse velocity while the clutch button is held."""

    def __init__(self, button=1, linear_speed=0.08, angular_speed=0.5, *, wait_for_device=False):
        if button < 0 or not all(np.isfinite(v) and v > 0 for v in (linear_speed, angular_speed)):
            raise ValueError("Use a nonnegative button index and positive, finite input speeds")
        self.button = button
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.wait_for_device = wait_for_device
        self.device = None
        self._clutch_pressed = None
        self._reset_gripper_buttons()

    def _reset_gripper_buttons(self):
        self._gripper_chord = None
        self._gripper_armed = False
        self._gripper_engaged = False
        self._buttons_released = False

    def __enter__(self):
        self.driver, self.hid_error = _load_spacemouse()
        self._connect()
        return self

    @property
    def connected(self):
        return self.device is not None

    @property
    def status(self):
        return "" if self.connected else "SpaceMouse USB 연결 대기"

    @property
    def clutch_pressed(self):
        """Last selected-button HID observation; None before confirmation or after unplug."""
        return self._clutch_pressed if self.device is not None else None

    @property
    def gripper_chord(self):
        """Confirmed button 0+1 state, independent of the six-axis stream."""
        return self._gripper_chord if self.device is not None and self._gripper_armed else None

    def _connect(self):
        self.retry_at = time.monotonic() + 1.0
        devices = self.driver.get_connected_devices()
        if not devices:
            if self.wait_for_device:
                return
            raise RuntimeError("No connected SpaceMouse found. Connect a supported USB device.")
        self.last_tick = None
        self.report_time = -1.0
        self.motion_time = -1.0
        self._clutch_pressed = None
        self._reset_gripper_buttons()
        self.saw_release = False
        self.release_pending = False
        try:
            self.device = self.driver.open(device=devices[0], nonblocking=True,
                                           axis_convention=self.driver.AxisConvention.HID_Z_UP,
                                           button_callback=self._buttons, dof_callback=self._motion)
        except (RuntimeError, OSError, self.hid_error) as exc:
            # A removal between enumeration and open is a normal hot-plug race.
            if self.wait_for_device and devices[0] not in self.driver.get_connected_devices():
                return
            raise RuntimeError(f"Cannot open SpaceMouse: {exc} Check HID permissions and other drivers.") from exc

    def __exit__(self, *exc):
        if self.device is not None:
            self.device.close()
            self.device = None

    def _buttons(self, state, buttons):
        # Axis-only HID reports retain the driver's synthetic initial button
        # zeros. Only a real button report can establish a released clutch.
        if self.button < len(buttons):
            self._clutch_pressed = bool(buttons[self.button])
            self._buttons_released = not self._clutch_pressed and not any(buttons[:2])
            if not self._clutch_pressed:
                self.saw_release = True
                self.release_pending = True
        if len(buttons) >= 2:
            self._gripper_chord = bool(buttons[0] and buttons[1])
            if not any(buttons[:2]):
                self._gripper_armed = True

    def _motion(self, state):
        self.motion_time = state.t

    def read(self):
        if self.device is None:
            if time.monotonic() >= self.retry_at:
                self._connect()
            return None

        state = self._latest_report()
        if state is None:
            return None
        now = time.perf_counter()  # Same clock as pyspacemouse HID reports.
        dt = 0 if self.last_tick is None else now - self.last_tick
        self.last_tick = now
        if self.report_time <= 0:
            return None

        arm = self._arm_sample(state, now, dt)
        command = self._gripper_command(dt)
        if arm is None and command is None:
            return None
        if arm is None:
            arm = InputSample.stationary()
        return replace(arm, gripper_command=command, buttons_released=self._buttons_released,
                       gripper_chord=self.gripper_chord)

    def _latest_report(self):
        # Some models send separate reports for translation, rotation and buttons.
        # Drain the nonblocking HID queue so a queued release takes effect this tick.
        for _ in range(64):
            try:
                state = self.device.read()
            except (self.hid_error, OSError):
                if not self.wait_for_device:
                    raise
                self.device.close()
                self.device = None
                self._reset_gripper_buttons()
                self.retry_at = time.monotonic() + 1.0
                return None
            if self.button >= len(state.buttons):
                raise ValueError(f"Button {self.button} unavailable; device has {len(state.buttons)} buttons.")
            if state.t == self.report_time:
                return state
            self.report_time = state.t

        # Queue overrun: wait for a fresh release before accepting motion.
        self.saw_release = False
        self.release_pending = False
        self._reset_gripper_buttons()
        self.last_tick = None
        return None

    def _gripper_command(self, dt):
        # Coalesce fast edges within one poll to the operator's final intent.
        # A held button at startup or a delayed new close cannot acquire a grasp.
        if self._gripper_chord and not self._gripper_engaged and dt > INPUT_TIMEOUT:
            self._gripper_armed = False
        engaged = self._gripper_armed and self._gripper_chord is True
        if engaged == self._gripper_engaged:
            return None
        self._gripper_engaged = engaged
        return engaged

    def _arm_sample(self, state, now, dt):
        if self.release_pending:
            # Deliver an actual release even if a new press is already queued.
            self.release_pending = False
            return InputSample.stationary()
        if not self._clutch_pressed:
            # An idle, released SpaceMouse may send no further reports.
            return InputSample.stationary() if self.saw_release else None
        # HID_Z_UP is (right, away, up). Facing the robot, UR base axes are
        # (toward the user, right, up), matching the calibrated Quest mapping.
        axes = np.array([-state.y, state.x, state.z, -state.pitch, state.roll, state.yaw])
        if not np.isfinite(axes).all() or dt > INPUT_TIMEOUT:
            self.saw_release = False
        axes = np.clip(axes, -1, 1)
        axes[np.abs(axes) < 0.05] = 0
        # HID is event-driven. Silence while the cap is centered is expected
        # even with the button down; only a nonzero cached velocity can go stale.
        if axes.any() and now - self.motion_time > INPUT_TIMEOUT:
            self.saw_release = False
        if not self.saw_release:
            return None
        return InputSample(axes[:3] * self.linear_speed * dt,
                           axes[3:] * self.angular_speed * dt, True)
