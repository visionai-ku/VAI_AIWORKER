"""UR5e spring-damper laws for a stationary equilibrium in base or joint coordinates."""

import numpy as np
from scipy.spatial.transform import Rotation

from .impedance_gains import (
    JOINT_DAMPING, JOINT_STIFFNESS, TCP_DAMPING, TCP_STIFFNESS,
    default_impedance_gains, validate_impedance_gains,
)

JOINT_TORQUE_LIMIT = (15., 15., 12., 4., 4., 3.)       # Nm, excluding gravity compensation
TCP_SPEED_LIMITS = (0.06, 0.06, 0.06, 0.25, 0.25, 0.25)  # m/s and rad/s per compliant axis


class ImpedanceSettings:
    """Keep validated, preconverted coefficients per backend and control space."""

    def __init__(self):
        self._profiles = {}
        for space in ("eef", "joint"):
            self.set(space, default_impedance_gains(space))

    def set(self, space, gains):
        validated = validate_impedance_gains(space, gains)
        self._profiles[space] = {name: np.asarray(values) for name, values in validated.items()}

    def coefficients(self, space):
        if space not in ("eef", "joint"):
            raise ValueError("Expected 'eef' or 'joint' impedance gains.")
        return self._profiles[space]

    def snapshot(self, space):
        return {name: values.tolist() for name, values in self.coefficients(space).items()}


def _vector(value, length):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f"Expected {length} finite coordinates.")
    return vector


def _limit_norm(vector, limit):
    magnitude = np.linalg.norm(vector)
    return vector if magnitude <= limit else vector * (limit / magnitude)


def cartesian_wrench(xyz, rotation, measured_xyz, measured_rotation, twist, *,
                     stiffness=TCP_STIFFNESS, damping=TCP_DAMPING):
    """Return [force, moment] at the TCP, in base axes (N, Nm); twist is in m/s and rad/s."""
    if not all(isinstance(r, Rotation) and r.single and np.isfinite(r.as_quat()).all()
               for r in (rotation, measured_rotation)):
        raise ValueError("Expected one rotation for each TCP orientation.")
    position_error = _vector(xyz, 3) - _vector(measured_xyz, 3)
    angular_error = (rotation * measured_rotation.inv()).as_rotvec()
    velocity = _vector(twist, 6)
    force = stiffness[:3] * position_error - damping[:3] * velocity[:3]
    moment = stiffness[3:] * angular_error - damping[3:] * velocity[3:]
    return np.r_[_limit_norm(force, 15.0), _limit_norm(moment, 2.0)]


def joint_torque(q_target, q, qd, *, stiffness=JOINT_STIFFNESS, damping=JOINT_DAMPING):
    """Return capped spring-damper torques in Nm. The backend handles gravity."""
    error = _vector(q_target, 6) - _vector(q, 6)
    torque = stiffness * error - damping * _vector(qd, 6)
    return np.clip(torque, -np.array(JOINT_TORQUE_LIMIT), JOINT_TORQUE_LIMIT)
