"""One operator input in the UR base frame, measured since the previous sample."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InputSample:
    translation: np.ndarray  # metres
    rotation: np.ndarray     # rotation vector, radians
    enabled: bool            # physical clutch/deadman is held
