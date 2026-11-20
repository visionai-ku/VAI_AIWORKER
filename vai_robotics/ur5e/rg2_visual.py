"""Pose the OnRobot RG2 visual linkage from the fingertip aperture in millimetres."""

import math
from numbers import Real

import numpy as np

from .rg2 import DEFAULT_GRIP_FORCE_N, DEFAULT_OPEN_WIDTH_MM, normalize_open_width_mm


OPEN_WIDTH_MM = 110.0
DEFAULT_VISUAL_MOUNT = (0.0,) * 6


def normalize_visual_mount(value):
    """Return flange-frame (x, y, z) in mm and fixed-axis (roll, pitch, yaw) in degrees."""
    if isinstance(value, Real):
        # Older sessions store a single rotation about flange +Z.
        value = (0., 0., 0., 0., 0., value)
    try:
        pose = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Expected a finite six-axis RG2 visual mount pose.") from exc
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("Expected a finite six-axis RG2 visual mount pose.")
    return tuple(pose.tolist())


# Approximate aperture rates from OnRobot RG2 Datasheet v1.8, pp. 2–3.
# Each row is (gap mm, speed mm/s at 3, 20, 40 N). Cap at the specified 127 mm/s.
_SPEED_GRAPH = np.array([
    (0,   13, 66, 127), (20, 13, 66, 127), (40, 12, 63, 123),
    (60,  11, 57, 112), (80,  9.5, 49, 95), (90, 8, 42, 82),
    (100, 6.5, 34, 65), (105, 5.5, 28, 54), (108, 4.5, 23, 45), (110, 4, 20, 38),
])


class RG2Preview:
    """Advance the simulated finger gap in millimetres from the last displayed pose."""

    def __init__(self, force_n=DEFAULT_GRIP_FORCE_N):
        if not math.isfinite(force_n) or not 3 <= force_n <= 40:
            raise ValueError("RG2 gripping force must be between 3 and 40 N.")
        force_n = round(force_n * 10) / 10  # Match the resolution of the Compute Box.
        self._speeds = [np.interp(force_n, (3, 20, 40), row[1:]) for row in _SPEED_GRAPH]
        self.width_mm = OPEN_WIDTH_MM
        self.reset()

    def reset(self, width_mm=None):
        if width_mm is not None:
            self.width_mm = width_mm
        self._closing = None
        self._target_width_mm = None
        self._last_tick = None

    def update(self, now, measured_width, command, chord, *, open_width_mm=DEFAULT_OPEN_WIDTH_MM):
        dt = 0.0 if self._last_tick is None else now - self._last_tick
        self._last_tick = now
        if self._closing is None and measured_width is not None:
            self.width_mm = measured_width
        if command is not None:
            self._target_width_mm = 0.0 if command else normalize_open_width_mm(open_width_mm)
            self._closing = command
        if self._closing is not None and chord == self._closing:
            # On a delayed control tick, resume from the visible width without catching up.
            speed_mm_s = float(np.interp(self.width_mm, _SPEED_GRAPH[:, 0], self._speeds))
            step = speed_mm_s * min(0.05, max(dt, 0.0))
            if self.width_mm > self._target_width_mm:
                self.width_mm = max(self._target_width_mm, self.width_mm - step)
            else:
                self.width_mm = min(self._target_width_mm, self.width_mm + step)
        return self.width_mm

    def commanded_width(self, chord, previous_width):
        """Return this tick's simulated setpoint, including holds on lost input."""
        if self._closing is not None and self._closing == chord:
            return self._target_width_mm
        return previous_width


def set_rg2_width(model, width_mm):
    """Set the six visual link orientations before the caller's next MuJoCo forward."""
    if not math.isfinite(width_mm):
        raise ValueError("RG2 aperture must be finite")
    # In the RG2 URDF each outer link connects its root to the finger pivot
    # at (y=-39.592, z=38.177) mm. The counter-rotating pad remains parallel.
    # Its inside face is 18.950 mm inboard of the finger pivot; the outer
    # link root is 17.178 mm outboard of the gripper centre.
    width_mm = max(0.0, min(OPEN_WIDTH_MM, width_mm))
    projection = (width_mm / 2 + 18.950 - 17.178) / math.hypot(39.592, 38.177)
    angle = math.acos(min(1.0, projection)) - math.atan2(38.177, 39.592)
    angle = max(-0.558505, angle)  # The source model's open stop.
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    for part in ("outer", "inner"):
        model.body(f"rg2_left_{part}").quat[:] = (c, -s, 0, 0)
        model.body(f"rg2_right_{part}").quat[:] = (0, 0, -s, c)
    for side in ("left", "right"):
        model.body(f"rg2_{side}_finger").quat[:] = (c, s, 0, 0)
