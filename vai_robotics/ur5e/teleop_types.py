"""One operator input in the UR base frame, measured since the previous sample."""

from dataclasses import dataclass

import numpy as np


INPUT_TIMEOUT = 0.15  # seconds without a valid pose or active HID report


@dataclass(frozen=True)
class JointDrag:
    """One viewer drag in radians. A new press gets a new serial; release clears joint."""

    serial: int
    joint: int | None
    angle: float = 0.0


@dataclass(frozen=True)
class InputSample:
    translation: np.ndarray  # metres
    rotation: np.ndarray     # rotation vector, radians
    enabled: bool            # physical clutch/deadman is held
    gripper_command: bool | None = None  # one-shot close/open on a physical chord edge
    buttons_released: bool = True        # selected arm button and both gripper buttons are up
    gripper_chord: bool | None = None    # observed level of SpaceMouse buttons 0 and 1
    joint_drag: JointDrag | None = None  # absolute angle of the joint selected in the viewer

    @classmethod
    def stationary(cls, *, enabled=False, joint_drag=None):
        """Report a known clutch state with zero translation and rotation."""
        return cls(np.zeros(3), np.zeros(3), enabled, joint_drag=joint_drag)
