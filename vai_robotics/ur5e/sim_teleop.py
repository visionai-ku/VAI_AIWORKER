"""Joint and Cartesian teleoperation of the UR5e in the bundled MuJoCo scene."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .impedance import ImpedanceSettings, cartesian_wrench, joint_torque
from .rg2_visual import OPEN_WIDTH_MM, set_rg2_width
from .sim import load_simulation


class SimBackend:
    """Servo the bare tool flange in the UR base frame (the scene's world frame)."""

    site_name = "attachment_site"
    joint_speed = 1.5  # rad/s, maximum slew of each position actuator's target

    def __init__(self, *, hz=50, headless=False):
        if not np.isfinite(hz) or hz <= 0:
            raise ValueError("hz must be finite and positive")
        self.period = 1.0 / hz
        self.headless = headless
        self._gains = ImpedanceSettings()
        self.sim = None
        self.active = False
        self._equilibrium = None
        self.tick_count = 0
        self._shadow = None
        self._previewing = False
        self._preview_clutched = False
        self._last_measured_width = OPEN_WIDTH_MM

    def __enter__(self):
        self.sim = load_simulation()
        try:
            # Pose the same meshes with separate joint state during DRY RUN.
            self._shadow = mujoco.MjData(self.sim.model)
            if self.period < self.sim.dt:
                raise ValueError("teleop frequency exceeds the MuJoCo physics rate")
            if not self.headless:
                import glfw

                if not glfw.init() or not glfw.get_primary_monitor():
                    raise RuntimeError("MuJoCo needs a local display. Use --headless for simulation only.")
                self.sim.init_viewer(title="UR5e teleoperation", width=960, height=720)
                self.sim.viewer.set_cam_info(azimuth=120, elevation=-20, distance=1.9, lookat=[0, 0.17, 0.35])
        except BaseException:
            self.close()
            raise
        self.active = False
        self._equilibrium = None
        self.tick_count = 0
        self._last_measured_width = OPEN_WIDTH_MM
        return self

    def _flange_pose(self, data):
        site = data.site(self.site_name)
        return site.xpos.copy(), Rotation.from_matrix(site.xmat.reshape(3, 3))

    def pose(self):
        return self._flange_pose(self.sim.data)

    def joints(self):
        return self.sim.data.qpos.copy()

    def joint_limits(self):
        lower, upper = self.sim.model.actuator_ctrlrange.T
        return lower.copy(), upper.copy()

    def preview_pose(self):
        """The displayed shadow TCP in UR base coordinates during DRY RUN."""
        if not self._previewing:
            return None
        xyz, rotation = self._flange_pose(self._shadow)
        offset, tool_rotation = self._flange_to_tcp
        return xyz + rotation.apply(offset), rotation * tool_rotation

    def _joint_target(self, data, xyz, rotation, previous_target):
        # Loaded by load_simulation from the same bundled motion library as the SG2.
        from ri_motion_v5_package.mujoco_sim.mujoco_ik import solve_dls

        sim = self.sim
        xyz = np.asarray(xyz, dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all() or not rotation.single:
            raise ValueError("expected a finite xyz position and one orientation")
        current_xyz, current_rot = self._flange_pose(data)
        error = np.r_[xyz - current_xyz, (rotation * current_rot.inv()).as_rotvec()]
        jacobian = self._jacobian(data)
        q_target = data.qpos + solve_dls(jacobian, error, lam=0.002**2)
        max_step = self.joint_speed * self.period
        change = q_target - previous_target
        # Scale all six joints together to preserve the Cartesian direction.
        change *= max_step / max(max_step, np.max(np.abs(change)))
        lower, upper = sim.model.actuator_ctrlrange.T
        return np.clip(previous_target + change, lower, upper)

    def command(self, xyz, rotation):
        """Apply one bounded differential IK update to the six live position actuators."""
        if self._equilibrium is not None:
            self.hold()
        data = self.sim.data
        data.ctrl[:] = self._joint_target(data, xyz, rotation, data.ctrl)
        self.active = True

    def _jacobian(self, data):
        jacobian = np.empty((6, self.sim.model.nv))
        mujoco.mj_jacSite(self.sim.model, data, jacobian[:3], jacobian[3:], data.site(self.site_name).id)
        return jacobian

    def _checked_joints(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("Expected six finite target joint angles in radians.")
        lower, upper = self.sim.model.actuator_ctrlrange.T
        if np.any(q < lower) or np.any(q > upper):
            raise ValueError("Target joint angles exceed the MuJoCo joint limits.")
        return q

    def command_joints(self, q):
        """Slew the six position servo references toward an absolute joint target."""
        q = self._checked_joints(q)
        if self._equilibrium is not None:
            self.hold()
        ctrl = self.sim.data.ctrl
        delta = q - ctrl
        max_step = self.joint_speed * self.period
        ctrl[:] += delta * (max_step / max(max_step, np.max(np.abs(delta))))
        self.active = True

    def command_joint_impedance(self, q):
        self._equilibrium = self._checked_joints(q).copy()
        self.active = True

    def get_impedance_gains(self, space):
        return self._gains.snapshot(space)

    def set_impedance_gains(self, space, gains):
        if self._equilibrium is not None:
            active_space = "eef" if isinstance(self._equilibrium, tuple) else "joint"
            if space == active_space:
                raise ValueError("Release the clutch before changing active impedance gains.")
        self._gains.set(space, gains)

    def command_cartesian_impedance(self, xyz, rotation):
        xyz = np.asarray(xyz, dtype=float)
        if (xyz.shape != (3,) or not np.isfinite(xyz).all()
                or not isinstance(rotation, Rotation) or not rotation.single or not np.isfinite(rotation.as_quat()).all()):
            raise ValueError("Expected a finite TCP position and a single orientation.")
        self._equilibrium = xyz.copy(), rotation
        self.active = True

    def _advance(self, nstep):
        sim = self.sim
        model, data = sim.model, sim.data
        # The parser writes gravity compensation into qfrc_applied. Preserve
        # torques applied by the caller (contacts and mouse forces use separate fields).
        external = data.qfrc_applied.copy()
        original_groups = model.opt.disableactuator
        try:
            if self._equilibrium is None:
                sim.compensate_gravity(["base"])
                data.qfrc_applied[:] += external
                sim.step(nstep=nstep)
                return

            # All six position actuators in ur5e.xml are in group zero. Only
            # the impedance torque law should supply feedback during these steps.
            model.opt.disableactuator |= 1
            for _ in range(nstep):
                mujoco.mj_step1(model, data)
                if isinstance(self._equilibrium, tuple):
                    jacobian = self._jacobian(data)
                    wrench = cartesian_wrench(*self._equilibrium, *self._flange_pose(data), jacobian @ data.qvel,
                                              **self._gains.coefficients("eef"))
                    torque = jacobian.T @ wrench
                else:
                    torque = joint_torque(self._equilibrium, data.qpos, data.qvel,
                                          **self._gains.coefficients("joint"))
                sim.compensate_gravity(["base"])
                data.qfrc_applied[:] += external + torque
                mujoco.mj_step2(model, data)
            sim.increase_wall_time()
        finally:
            data.qfrc_applied[:] = external
            model.opt.disableactuator = original_groups

    def hold(self):
        """Latch measured joint positions on release; keep holding on subsequent ticks."""
        if self.active:
            self.sim.data.ctrl[:] = self.sim.data.qpos
            self._equilibrium = None
            self.active = False

    def _update_shadow(self, target, joint_target, measured_tcp):
        clutched = target is not None or joint_target is not None
        if not clutched or not self._preview_clutched:
            self._shadow.qpos[:] = self.sim.data.qpos
            mujoco.mj_forward(self.sim.model, self._shadow)
            # A real TCP can be offset and rotated relative to this model's flange.
            if measured_tcp is None:
                self._flange_to_tcp = np.zeros(3), Rotation.identity()
            else:
                xyz, rotation = self._flange_pose(self._shadow)
                tcp_xyz, tcp_rotation = measured_tcp
                self._flange_to_tcp = rotation.inv().apply(tcp_xyz - xyz), rotation.inv() * tcp_rotation
        if joint_target is not None:
            q = np.asarray(joint_target, dtype=float)
            if q.shape != (6,) or not np.isfinite(q).all():
                raise ValueError("Expected six finite preview joint angles.")
            self._shadow.qpos[:] = q
            mujoco.mj_forward(self.sim.model, self._shadow)
        elif target is not None:
            tcp_xyz, tcp_rotation = target
            offset, tool_rotation = self._flange_to_tcp
            flange_rotation = tcp_rotation * tool_rotation.inv()
            flange_xyz = tcp_xyz - flange_rotation.apply(offset)
            self._shadow.qpos[:] = self._joint_target(
                self._shadow, flange_xyz, flange_rotation, self._shadow.qpos)
            mujoco.mj_forward(self.sim.model, self._shadow)
        self._preview_clutched = clutched

    def tick(self, *, dry_run=False, target=None, joint_target=None, measured_q=None, measured_tcp=None,
             reset_preview=False, gripper_width_mm=None):
        """Advance live physics, mirror RTDE joints, or articulate the independent preview."""
        sim = self.sim
        # Width is visual geometry; all six RTDE joints still address the arm.
        if measured_q is not None and not dry_run and gripper_width_mm is not None:
            self._last_measured_width = gripper_width_mm
        set_rg2_width(sim.model, self._last_measured_width if gripper_width_mm is None else gripper_width_mm)
        if measured_q is not None:
            q = np.asarray(measured_q, dtype=float)
            if q.shape != (sim.model.nq,) or not np.isfinite(q).all():
                raise ValueError("expected six finite measured joint angles")
            sim.data.qpos[:] = q
            mujoco.mj_forward(sim.model, sim.data)
        elif not dry_run:
            self.tick_count += 1
            nstep = round((self.tick_count * self.period - sim.data.time) / sim.dt)
            self._advance(nstep)
        self._previewing = dry_run
        if dry_run:
            # A renderer can skip every released frame between two clutches.
            if reset_preview:
                self._preview_clutched = False
            self._update_shadow(target, joint_target, measured_tcp)
        else:
            self._preview_clutched = False
        if not self.headless and self.is_running():
            sim.viewer.data = self._shadow if dry_run else sim.data
            if measured_q is not None:
                mode, source = ("DRY RUN", "Preview") if dry_run else ("REAL RUN", "Measured")
                sim.viewer_text_overlay(mode, source, loc="top left")
                if gripper_width_mm is None:
                    sim.viewer_text_overlay("RG2", "Width unavailable", loc="top left")
            display_tcp = target if dry_run else measured_tcp
            if dry_run and joint_target is not None:
                display_tcp = self.preview_pose()
            if display_tcp is not None:
                xyz, rotation = display_tcp
                sim.plot_T(p=xyz, R=rotation.as_matrix(), axis_len=0.08, axis_width=0.004)
            sim.render()

    def is_running(self):
        return self.sim is not None and (self.headless or self.sim.is_viewer_alive())

    def close(self):
        if self.sim is not None:
            if not self.headless:
                self.sim.close_viewer()
            self.sim = None
        self._shadow = None
        self._previewing = self._preview_clutched = False

    def __exit__(self, *exc):
        self.close()
