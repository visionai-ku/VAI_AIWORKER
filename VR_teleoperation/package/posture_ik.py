"""Full-pose SG2 IK with a persistent measured posture and elbow-side reference.

The six wrist coordinates are equality constraints.  Joint posture is optimized
only among configurations satisfying those constraints, joint bounds and the
elbow-side constraint; it cannot purchase a lower cost by relaxing orientation.
All kinematics run on private MjData.  This module neither commands a robot nor
advances its physics, and callers must retain their rate and collision guards.
"""
from __future__ import annotations

import time

import mujoco
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .utils import SG2_ARM_JOINT_NAMES, SG2_WRIST_BODY_NAMES, SG2_WRIST_P_LOCAL, SG2_WRIST_R_LOCAL


class _SolveBudgetExceeded(Exception):
    pass


def _skew(vector):
    x, y, z = vector
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def _right_log_jacobian_inverse(vector):
    """Derivative of log(R exp(delta)) at delta=0, away from the pi cut."""
    angle = float(np.linalg.norm(vector))
    matrix = _skew(vector)
    coefficient = (1. / 12. + angle * angle / 720. if angle < 1e-4 else
                   (1. - .5 * angle / np.tan(.5 * angle)) / (angle * angle))
    return np.eye(3) + .5 * matrix + coefficient * matrix @ matrix


class SG2PostureIK:
    """Bounded local seven-joint optimizer with a fixed measured reference.

    A fresh instance has no reference.  Capture a complete measured qpos once
    at engagement.  Subsequent solves and clutch releases never replace it.
    ``reset_reference`` is an explicit lifecycle operation, not an IK recovery.
    """

    def __init__(self, env, hand, *, position_tolerance=.001,
                 rotation_tolerance=np.deg2rad(.5), elbow_limit_rad=np.deg2rad(85),
                 max_iterations=24, max_solve_time_s=.04,
                 max_joint_step=np.deg2rad(20)):
        if hand not in SG2_ARM_JOINT_NAMES:
            raise ValueError("hand must be 'left' or 'right'")
        for name, value in (("position_tolerance", position_tolerance),
                            ("rotation_tolerance", rotation_tolerance),
                            ("elbow_limit_rad", elbow_limit_rad),
                            ("max_solve_time_s", max_solve_time_s),
                            ("max_joint_step", max_joint_step)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("%s must be positive and finite" % name)
        if not 0 < elbow_limit_rad < np.pi / 2:
            raise ValueError("elbow_limit_rad must be less than 90 degrees")
        if isinstance(max_iterations, bool) or int(max_iterations) != max_iterations or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self.model = env.model
        self.hand = hand
        self.joints_use = list(SG2_ARM_JOINT_NAMES[hand])
        self.position_tolerance = float(position_tolerance)
        self.rotation_tolerance = float(rotation_tolerance)
        self.elbow_limit_rad = float(elbow_limit_rad)
        self.max_iterations = int(max_iterations)
        self.max_solve_time_s = float(max_solve_time_s)
        self.max_joint_step = float(max_joint_step)
        self.posture_weights = np.array([.25, .25, 4., .25, 4., 2.5, 2.5])
        self.elbow_weight = 4.
        self._data = mujoco.MjData(self.model)
        self._q_indices = np.array([int(self.model.joint(name).qposadr[0]) for name in self.joints_use])
        self._dof_indices = np.array([int(self.model.joint(name).dofadr[0]) for name in self.joints_use])
        joint_ids = [int(self.model.joint(name).id) for name in self.joints_use]
        ranges = self.model.jnt_range[joint_ids]
        limited = self.model.jnt_limited[joint_ids].astype(bool)
        self._lower = np.where(limited, ranges[:, 0], -np.inf)
        self._upper = np.where(limited, ranges[:, 1], np.inf)
        prefix = "arm_l" if hand == "left" else "arm_r"
        self._shoulder_id = int(self.model.body(prefix + "_link2").id)
        self._elbow_id = int(self.model.body(prefix + "_link4").id)
        self._wrist_id = int(self.model.body(SG2_WRIST_BODY_NAMES[hand]).id)
        self._base_id = int(self.model.body("arm_base_link").id)
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))
        self.reset_reference()

    @property
    def has_reference(self):
        return self._reference_qpos is not None

    @property
    def reference_qpos(self):
        return None if self._reference_qpos is None else self._reference_qpos.copy()

    @property
    def reference_joint_positions(self):
        return None if not self.has_reference else self._reference_qpos[self._q_indices].copy()

    def reset_reference(self):
        self._reference_qpos = None
        self._reference_reach = None
        self._reference_elbow = None

    def _validate_qpos(self, q_full):
        qpos = np.asarray(q_full, dtype=np.float64)
        if qpos.shape != (self.model.nq,) or not np.isfinite(qpos).all():
            raise ValueError("q_full must contain one finite value per model qpos")
        return qpos.copy()

    def _forward(self, q_full):
        self._data.qpos[:] = q_full
        # These are the stages required for positions, orientations and mj_jac;
        # collision detection and dynamics are intentionally not evaluated.
        mujoco.mj_kinematics(self.model, self._data)
        mujoco.mj_comPos(self.model, self._data)

    def _points(self):
        wrist_rotation = self._data.xmat[self._wrist_id].reshape(3, 3)
        wrist = self._data.xpos[self._wrist_id] + wrist_rotation @ SG2_WRIST_P_LOCAL[self.hand]
        return (self._data.xpos[self._shoulder_id].copy(),
                self._data.xpos[self._elbow_id].copy(), wrist,
                wrist_rotation @ SG2_WRIST_R_LOCAL[self.hand])

    @staticmethod
    def _elbow_projection(shoulder, elbow, wrist):
        reach = wrist - shoulder
        length = float(np.linalg.norm(reach))
        if length < 1e-6:
            return None, None, "shoulder and wrist coincide"
        direction = reach / length
        radial = elbow - shoulder
        radial = radial - direction * (radial @ direction)
        radius = float(np.linalg.norm(radial))
        # At a straight elbow, its side is undefined; never silently accept a
        # branch crossing through that degenerate geometry.
        if radius < .002:
            return None, None, "elbow direction undefined near straight arm"
        return direction, radial / radius, None

    def capture_reference(self, q_full):
        qpos = self._validate_qpos(q_full)
        q_arm = qpos[self._q_indices]
        if np.any(q_arm < self._lower - 1e-8) or np.any(q_arm > self._upper + 1e-8):
            raise ValueError("reference arm joints exceed model limits")
        self._forward(qpos)
        shoulder, elbow, wrist, _ = self._points()
        reach, radial, reason = self._elbow_projection(shoulder, elbow, wrist)
        if reason is not None:
            raise ValueError("cannot capture elbow reference: " + reason)
        base_rotation = self._data.xmat[self._base_id].reshape(3, 3)
        self._reference_qpos = qpos
        self._reference_reach = base_rotation.T @ reach
        self._reference_elbow = base_rotation.T @ radial

    def _elbow_geometry(self, shoulder, elbow, wrist, base_rotation):
        direction, radial, reason = self._elbow_projection(shoulder, elbow, wrist)
        if reason is not None:
            return np.nan, -1., reason
        initial_direction = base_rotation @ self._reference_reach
        initial_radial = base_rotation @ self._reference_elbow
        cosine = float(np.clip(initial_direction @ direction, -1., 1.))
        if cosine < -1. + 1e-5:
            return np.nan, -1., "elbow reference transport undefined for opposite reach"
        transported = initial_radial - ((initial_radial @ direction) / (1. + cosine)) * (initial_direction + direction)
        # Remove round-off along the reach axis before evaluating the angle.
        transported -= direction * (transported @ direction)
        norm = float(np.linalg.norm(transported))
        if norm < 1e-8:
            return np.nan, -1., "elbow reference transport is degenerate"
        transported /= norm
        cosine = float(np.clip(transported @ radial, -1., 1.))
        sine = float(direction @ np.cross(transported, radial))
        return float(np.arctan2(sine, cosine)), cosine, None

    def _elbow_cosine_gradient(self, shoulder, elbow, wrist, base_rotation,
                               shoulder_jacobian, elbow_jacobian, wrist_jacobian):
        """Differentiate the transported elbow geometry using point Jacobians."""
        reach = wrist - shoulder
        length = np.linalg.norm(reach)
        direction = reach / length
        du = (np.eye(3) - np.outer(direction, direction)) @ (wrist_jacobian - shoulder_jacobian) / length
        arm = elbow - shoulder
        da = elbow_jacobian - shoulder_jacobian
        radial = arm - direction * (arm @ direction)
        radius = np.linalg.norm(radial)
        radial /= radius
        dh = da - (arm @ direction) * du - np.outer(direction, arm @ du + direction @ da)
        dr = (np.eye(3) - np.outer(radial, radial)) @ dh / radius
        initial_direction = base_rotation @ self._reference_reach
        initial_radial = base_rotation @ self._reference_elbow
        denominator = 1. + initial_direction @ direction
        numerator = initial_radial @ direction
        fraction = numerator / denominator
        fraction_gradient = ((initial_radial @ du) * denominator
                             - numerator * (initial_direction @ du)) / denominator ** 2
        transported = initial_radial - fraction * (initial_direction + direction)
        dt = -np.outer(initial_direction + direction, fraction_gradient) - fraction * du
        return radial @ dt + transported @ dr

    def configuration_info(self, q_full):
        """Evaluate actual emitted joints against the fixed elbow reference."""
        qpos = self._validate_qpos(q_full)
        if not self.has_reference:
            return dict(valid=False, reason="posture reference not captured",
                        elbow_angle_rad=np.nan, posture_cost=np.inf)
        self._forward(qpos)
        shoulder, elbow, wrist, _ = self._points()
        angle, cosine, reason = self._elbow_geometry(
            shoulder, elbow, wrist, self._data.xmat[self._base_id].reshape(3, 3))
        q_arm = qpos[self._q_indices]
        offset = q_arm - self._reference_qpos[self._q_indices]
        cost = .5 * float((self.posture_weights * offset) @ offset) + self.elbow_weight * (1. - cosine)
        if reason is None and (np.any(q_arm < self._lower - 1e-8) or np.any(q_arm > self._upper + 1e-8)):
            reason = "arm joint limit exceeded"
        if reason is None and abs(angle) > self.elbow_limit_rad + 1e-8:
            reason = "elbow crossed initial-side limit"
        return dict(valid=reason is None, reason=reason or "ok",
                    elbow_angle_rad=angle, posture_cost=cost)

    def _point_jacobian(self, point, body):
        mujoco.mj_jac(self.model, self._data, self._jacp, self._jacr, point, body)
        return self._jacp[:, self._dof_indices].copy(), self._jacr[:, self._dof_indices].copy()

    def solve(self, env, T_target):
        """Return a locally continuous full-pose solution without changing env."""
        started = time.perf_counter()
        cpu_started = time.thread_time()
        if env.model is not self.model:
            raise ValueError("solver and environment must share the same MuJoCo model")
        target = np.asarray(T_target, dtype=np.float64)
        if (target.shape != (4, 4) or not np.isfinite(target).all()
                or not np.allclose(target[3], [0., 0., 0., 1.], atol=1e-8)
                or not np.allclose(target[:3, :3].T @ target[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(target[:3, :3]), 1., atol=1e-6)):
            raise ValueError("T_target must be a finite rigid transform")
        context = self._validate_qpos(env.data.qpos)
        start = context[self._q_indices].copy()
        if np.any(start < self._lower - 1e-8) or np.any(start > self._upper + 1e-8):
            raise ValueError("initial arm joints exceed model limits")
        if not self.has_reference:
            return dict(qpos_used_best=start, success=False, position_error_m=np.inf,
                        rotation_error_rad=np.inf, posture_cost=np.inf,
                        elbow_angle_rad=np.nan, iterations=0,
                        elapsed_s=time.perf_counter() - started,
                        reason="posture reference not captured", budget_exhausted=False)
        lower = np.maximum(self._lower, start - self.max_joint_step)
        upper = np.minimum(self._upper, start + self.max_joint_step)
        if np.any(lower > upper):
            return dict(qpos_used_best=start, success=False, position_error_m=np.inf,
                        rotation_error_rad=np.inf, posture_cost=np.inf,
                        elbow_angle_rad=np.nan, iterations=0,
                        elapsed_s=time.perf_counter() - started,
                        reason="initial arm joints exceed model limits", budget_exhausted=False)
        reference = self._reference_qpos[self._q_indices]
        candidates = []
        cache = {}
        evaluations = 0
        iterations = 0

        def evaluate(q):
            nonlocal evaluations
            if evaluations and time.perf_counter() - started >= self.max_solve_time_s:
                raise _SolveBudgetExceeded()
            if "q" in cache and np.array_equal(q, cache["q"]):
                return cache
            context[self._q_indices] = q
            self._forward(context)
            shoulder, elbow, wrist, rotation = self._points()
            base_rotation = self._data.xmat[self._base_id].reshape(3, 3).copy()
            position_error = target[:3, 3] - wrist
            rotation_error = Rotation.from_matrix(target[:3, :3] @ rotation.T).as_rotvec()
            wrist_jacobian, angular_jacobian = self._point_jacobian(wrist, self._wrist_id)
            shoulder_jacobian, _ = self._point_jacobian(shoulder, self._shoulder_id)
            elbow_jacobian, _ = self._point_jacobian(elbow, self._elbow_id)
            angle, cosine, geometry_reason = self._elbow_geometry(shoulder, elbow, wrist, base_rotation)
            cosine_gradient = np.zeros(7)
            if geometry_reason is None:
                cosine_gradient = self._elbow_cosine_gradient(
                    shoulder, elbow, wrist, base_rotation,
                    shoulder_jacobian, elbow_jacobian, wrist_jacobian)
            offset = q - reference
            cost = .5 * float((self.posture_weights * offset) @ offset) + self.elbow_weight * (1. - cosine)
            gradient = self.posture_weights * offset - self.elbow_weight * cosine_gradient
            pose_error = np.hstack((position_error / .1, rotation_error))
            pose_jacobian = np.vstack((-wrist_jacobian / .1,
                                      -_right_log_jacobian_inverse(rotation_error) @ angular_jacobian))
            cache.clear()
            cache.update(q=np.array(q, copy=True), cost=cost, gradient=gradient,
                         pose_error=pose_error, pose_jacobian=pose_jacobian,
                         elbow_margin=cosine - np.cos(self.elbow_limit_rad),
                         elbow_jacobian=cosine_gradient, elbow_angle=angle,
                         position_error=float(np.linalg.norm(position_error)),
                         rotation_error=float(np.linalg.norm(rotation_error)),
                         geometry_reason=geometry_reason)
            candidates.append({name: value.copy() if isinstance(value, np.ndarray) else value
                               for name, value in cache.items()})
            evaluations += 1
            return cache

        def callback(_):
            nonlocal iterations
            iterations += 1
            if time.perf_counter() - started >= self.max_solve_time_s:
                raise _SolveBudgetExceeded()

        budget_exhausted = False
        optimizer_message = ""
        try:
            evaluate(np.clip(start, lower, upper))
            # Do not enter SLSQP after an initial evaluation already consumed
            # the budget (for example, a GC/scheduling pause inside FK).
            if time.perf_counter() - started >= self.max_solve_time_s:
                raise _SolveBudgetExceeded()
            result = minimize(lambda q: evaluate(q)["cost"], np.clip(start, lower, upper),
                              jac=lambda q: evaluate(q)["gradient"], method="SLSQP",
                              bounds=list(zip(lower, upper)), callback=callback,
                              constraints=[
                                  {"type": "eq", "fun": lambda q: evaluate(q)["pose_error"],
                                   "jac": lambda q: evaluate(q)["pose_jacobian"]},
                                  {"type": "ineq", "fun": lambda q: evaluate(q)["elbow_margin"],
                                   "jac": lambda q: evaluate(q)["elbow_jacobian"]}],
                              options={"maxiter": self.max_iterations, "ftol": 1e-9, "disp": False})
            optimizer_message = str(result.message)
            iterations = int(result.nit)
            evaluate(result.x)
            budget_exhausted = result.status == 9
        except _SolveBudgetExceeded:
            optimizer_message = "IK computation time budget reached"
            budget_exhausted = True

        feasible = [candidate for candidate in candidates
                    if candidate["geometry_reason"] is None
                    and candidate["elbow_margin"] >= -1e-8
                    and candidate["position_error"] <= self.position_tolerance
                    and candidate["rotation_error"] <= self.rotation_tolerance
                    and np.all(candidate["q"] >= lower - 1e-8)
                    and np.all(candidate["q"] <= upper + 1e-8)]
        if feasible:
            # Prefer candidates that actually solved the pose equalities; the
            # configured tolerances are a bounded fallback, never a cost weight.
            exact = [candidate for candidate in feasible
                     if candidate["position_error"] <= min(self.position_tolerance, 1e-5)
                     and candidate["rotation_error"] <= min(self.rotation_tolerance, 1e-4)]
            best = min(exact or feasible, key=lambda candidate: candidate["cost"])
            reason = "ok" if not budget_exhausted else "feasible pose at computation budget"
        else:
            best = min(candidates, key=lambda candidate: candidate["position_error"] / self.position_tolerance
                       + candidate["rotation_error"] / self.rotation_tolerance)
            reason = (best["geometry_reason"] or
                      ("elbow crossed initial-side limit" if best["elbow_margin"] < -1e-8 else
                       "full wrist pose unavailable within local joint bounds"))
        return dict(qpos_used_best=best["q"].copy(), success=bool(feasible),
                    position_error_m=best["position_error"], rotation_error_rad=best["rotation_error"],
                    posture_cost=best["cost"], elbow_angle_rad=best["elbow_angle"],
                    iterations=iterations, elapsed_s=time.perf_counter() - started,
                    thread_cpu_s=time.thread_time() - cpu_started,
                    reason=reason, budget_exhausted=budget_exhausted,
                    optimizer_message=optimizer_message, evaluations=evaluations)
