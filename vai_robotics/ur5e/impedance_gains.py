"""Session gain defaults and editing ranges shared by the Qt panel and controllers."""

from collections.abc import Mapping
from numbers import Real


# Cartesian entries use base-frame X/Y/Z translation then X/Y/Z rotation.
# Joint entries follow shoulder pan through wrist 3.
TCP_STIFFNESS = (300., 300., 300., 8., 8., 8.)    # N/m, Nm/rad
TCP_DAMPING = (30., 30., 30., 1.5, 1.5, 1.5)      # N s/m, Nm s/rad
JOINT_STIFFNESS = (40., 45., 35., 12., 10., 8.)    # Nm/rad
JOINT_DAMPING = (10., 10., 8., 2.5, 2., 1.6)       # Nm s/rad

_DEFAULTS = {
    "eef": {"stiffness": TCP_STIFFNESS, "damping": TCP_DAMPING},
    "joint": {"stiffness": JOINT_STIFFNESS, "damping": JOINT_DAMPING},
}


def default_impedance_gains(space):
    if space not in ("eef", "joint"):
        raise ValueError("Expected 'eef' or 'joint' impedance gains.")
    return {name: list(values) for name, values in _DEFAULTS[space].items()}


def max_impedance_gains(space):
    return {name: [2 * value for value in values]
            for name, values in default_impedance_gains(space).items()}


def validate_impedance_gains(space, gains):
    """Return independent six-axis vectors suitable for JSON and the control loop."""
    limits = max_impedance_gains(space)
    if not isinstance(gains, Mapping) or gains.keys() != limits.keys():
        raise ValueError("Provide both stiffness and damping.")
    result = {}
    for name, upper in limits.items():
        values = gains[name]
        # Six indexable, ordered values. Also accept numeric arrays from Python callers.
        if not hasattr(values, "__getitem__") or isinstance(values, (Mapping, str, bytes, bytearray)):
            raise ValueError(f"Expected six {name} values.")
        try:
            length = len(values)
        except TypeError:
            raise ValueError(f"Expected six {name} values.") from None
        if length != 6:
            raise ValueError(f"Expected six {name} values.")
        vector = []
        for i, (value, maximum) in enumerate(zip(values, upper), 1):
            if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value <= maximum:
                raise ValueError(f"{name} axis {i} must be between 0 and {maximum:g}.")
            vector.append(float(value))
        result[name] = vector
    return result
