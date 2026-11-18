"""One operator input in the UR base frame, measured since the previous sample."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InputSample:
    translation: np.ndarray  # metres
    rotation: np.ndarray     # rotation vector, radians
    enabled: bool            # physical clutch/deadman is held
    gripper_command: bool | None = None  # one-shot close/open on a physical chord edge
    buttons_released: bool = True        # selected arm button and both gripper buttons are up
    gripper_chord: bool | None = None    # observed level of SpaceMouse buttons 0 and 1
