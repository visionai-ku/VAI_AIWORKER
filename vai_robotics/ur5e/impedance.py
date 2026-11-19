"""UR5e spring-damper laws for a stationary equilibrium in base or joint coordinates."""

import numpy as np
from scipy.spatial.transform import Rotation


# Joint order: shoulder pan, shoulder lift, elbow, wrist 1, wrist 2, wrist 3.
JOINT_STIFFNESS = (40., 45., 35., 12., 10., 8.)       # Nm/rad
JOINT_DAMPING = (10., 10., 8., 2.5, 2., 1.6)          # Nm s/rad
JOINT_TORQUE_LIMIT = (15., 15., 12., 4., 4., 3.)       # Nm, excluding gravity compensation


def _vector(value, length):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f"Expected {length} finite coordinates.")
    return vector


def _limit_norm(vector, limit):
    magnitude = np.linalg.norm(vector)
    return vector if magnitude <= limit else vector * (limit / magnitude)


def cartesian_wrench(xyz, rotation, measured_xyz, measured_rotation, twist):
    """Return [force, moment] at the TCP, in base axes (N, Nm); twist is in m/s and rad/s."""
    if not all(isinstance(r, Rotation) and r.single and np.isfinite(r.as_quat()).all()
               for r in (rotation, measured_rotation)):
        raise ValueError("Expected one rotation for each TCP orientation.")
    position_error = _vector(xyz, 3) - _vector(measured_xyz, 3)
    angular_error = (rotation * measured_rotation.inv()).as_rotvec()
    velocity = _vector(twist, 6)
    force = 300.0 * position_error - 30.0 * velocity[:3]
    moment = 8.0 * angular_error - 1.5 * velocity[3:]
    return np.r_[_limit_norm(force, 15.0), _limit_norm(moment, 2.0)]


def joint_torque(q_target, q, qd):
    """Return capped spring-damper torques in Nm. The backend handles gravity."""
    error = _vector(q_target, 6) - _vector(q, 6)
    torque = JOINT_STIFFNESS * error - JOINT_DAMPING * _vector(qd, 6)
    return np.clip(torque, -np.array(JOINT_TORQUE_LIMIT), JOINT_TORQUE_LIMIT)
