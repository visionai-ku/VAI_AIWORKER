"""Advance an isolated MuJoCo arm with the selected UR5e impedance gains."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .controllers import MAX_JOINT_LEAD, MAX_POSITION_LEAD, MAX_ROTATION_LEAD, MAX_TICK_INTERVAL, bounded_vector
from .impedance import TCP_SPEED_LIMITS, cartesian_wrench, joint_torque


class ImpedancePreview:
    def __init__(self, model, shadow, site_name, gains):
        self.model = model
        self.data = shadow
        self.gains = gains
        self.site = shadow.site(site_name)
        self.tool_body = model.site_bodyid[self.site.id]
        self.root_body = model.body("base").id
        self.com_jacobian = np.empty((3, model.nv))
        self.tool_jacobian = np.empty((6, model.nv))
        self.reset()

    def reset(self):
        self.elapsed = 0.0
        self.last_target = None
        self.equilibrium = None
        self.reference = None

    def tcp_pose(self, flange_to_tcp):
        offset, tool_rotation = flange_to_tcp
        flange_rotation = Rotation.from_matrix(self.site.xmat.reshape(3, 3))
        return self.site.xpos + flange_rotation.apply(offset), flange_rotation * tool_rotation

    def _tcp_equilibrium(self, target, measured, base):
        xyz, rotation = target
        if base is not None:
            last_xyz, last_rotation = self.last_target
            xyz = base[0] + (xyz - last_xyz)
            rotation = (rotation * last_rotation.inv()) * base[1]
        measured_xyz, measured_rotation = measured
        lead = bounded_vector(xyz - measured_xyz, MAX_POSITION_LEAD)
        angle = bounded_vector((rotation * measured_rotation.inv()).as_rotvec(), MAX_ROTATION_LEAD)
        limited = measured_xyz + lead, Rotation.from_rotvec(angle) * measured_rotation
        return (xyz, rotation), limited

    def _joint_equilibrium(self, target, base):
        q = target
        if base is not None:
            q = base + (q - self.last_target)
        limited = np.clip(q, self.data.qpos - MAX_JOINT_LEAD, self.data.qpos + MAX_JOINT_LEAD)
        return q, limited

    def _tcp_torque(self, flange_to_tcp):
        xyz, rotation = self.tcp_pose(flange_to_tcp)
        jac = self.tool_jacobian
        # Use the measured tool offset for both TCP twist and the wrench moment arm.
        mujoco.mj_jac(self.model, self.data, jac[:3], jac[3:], xyz, self.tool_body)
        wrench = cartesian_wrench(
            *self.equilibrium, xyz, rotation, jac @ self.data.qvel, **self.gains.coefficients("eef"),
        )
        return jac.T @ wrench

    def _step(self, joint, flange_to_tcp):
        model, data = self.model, self.data
        mujoco.mj_step1(model, data)
        mujoco.mj_jacSubtreeCom(model, data, self.com_jacobian, self.root_body)
        gravity = model.body_subtreemass[self.root_body] * model.opt.gravity
        data.qfrc_applied[:] = -self.com_jacobian.T @ gravity
        if joint:
            data.qfrc_applied[:] += joint_torque(
                self.equilibrium, data.qpos, data.qvel, **self.gains.coefficients("joint"),
            )
        else:
            data.qfrc_applied[:] += self._tcp_torque(flange_to_tcp)

        old_q = data.qpos.copy()
        mujoco.mj_step2(model, data)
        if joint:
            # The UR joint writer monitors this speed on each joint.
            speed_ratio = np.max(np.abs(data.qvel)) / 0.75
        else:
            speed_ratio = np.max(np.abs(self.tool_jacobian @ data.qvel) / TCP_SPEED_LIMITS)
        if speed_ratio > 1:
            # This model has six hinge joints. Limit the integrated displacement
            # as well as velocity so the pose rendered this step obeys the cap.
            data.qpos[:] = old_q + (data.qpos - old_q) / speed_ratio
            data.qvel[:] /= speed_ratio

    def advance(self, target, joint_target, flange_to_tcp, duration, incremental):
        # Only a numerical goal can retain motion beyond the current lead. A
        # new jog starts at the limited equilibrium, even after a numerical goal.
        base = self.equilibrium if incremental else self.reference
        joint = joint_target is not None
        if joint:
            target = np.array(joint_target, dtype=float, copy=True)
            if target.shape != (6,) or not np.isfinite(target).all():
                raise ValueError("Expected six finite preview joint angles.")
            desired, self.equilibrium = self._joint_equilibrium(target, base)
        else:
            target = np.array(target[0], dtype=float, copy=True), target[1]
            desired, self.equilibrium = self._tcp_equilibrium(target, self.tcp_pose(flange_to_tcp), base)
        self.reference = self.equilibrium if incremental else desired
        self.last_target = target

        # Accumulate fractional substeps between display frames. A long viewer
        # stall must not replay its whole backlog as a single jump.
        self.elapsed += np.clip(duration, 0.0, MAX_TICK_INTERVAL)
        nstep = round((self.elapsed - self.data.time) / self.model.opt.timestep)
        actuator_groups = self.model.opt.disableactuator
        try:
            self.model.opt.disableactuator |= 1  # Disable the six position actuators in group zero.
            for _ in range(nstep):
                self._step(joint, flange_to_tcp)
            mujoco.mj_forward(self.model, self.data)
        finally:
            self.model.opt.disableactuator = actuator_groups
