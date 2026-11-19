"""Joint and Cartesian teleoperation of the UR5e in the bundled MuJoCo scene."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .impedance import ImpedanceSettings, cartesian_wrench, joint_torque
from .impedance_preview import ImpedancePreview
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
        self._impedance_preview = None
        self._previewing = False
        self._preview_mode = None
        self._last_measured_width = OPEN_WIDTH_MM

    def __enter__(self):
        self.sim = load_simulation()
        try:
            # Pose the same meshes with separate joint state during DRY RUN.
            self._shadow = mujoco.MjData(self.sim.model)
            self._impedance_preview = ImpedancePreview(self.sim.model, self._shadow, self.site_name, self._gains)
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
        return self._impedance_preview.tcp_pose(self._flange_to_tcp)

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

    def _impedance_torque(self, data):
        if isinstance(self._equilibrium, tuple):
            jacobian = self._jacobian(data)
            wrench = cartesian_wrench(
                *self._equilibrium, *self._flange_pose(data), jacobian @ data.qvel,
                **self._gains.coefficients("eef"),
            )
            return jacobian.T @ wrench
        return joint_torque(
            self._equilibrium, data.qpos, data.qvel, **self._gains.coefficients("joint"),
        )

    def _advance_physics(self):
        sim = self.sim
        model, data = sim.model, sim.data
        self.tick_count += 1
        nstep = round((self.tick_count * self.period - data.time) / sim.dt)
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
                torque = self._impedance_torque(data)
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

    def _reset_shadow_to_arm(self, measured_tcp):
        self._impedance_preview.reset()
        mujoco.mj_resetData(self.sim.model, self._shadow)
        self._shadow.qpos[:] = self.sim.data.qpos
        mujoco.mj_forward(self.sim.model, self._shadow)
        # A real TCP can be offset and rotated relative to this model's flange.
        if measured_tcp is None:
            self._flange_to_tcp = np.zeros(3), Rotation.identity()
        else:
            xyz, rotation = self._flange_pose(self._shadow)
            tcp_xyz, tcp_rotation = measured_tcp
            inverse_rotation = rotation.inv()
            self._flange_to_tcp = inverse_rotation.apply(tcp_xyz - xyz), inverse_rotation * tcp_rotation

    def _update_shadow(self, target, joint_target, measured_tcp, reset_preview, impedance, dt, incremental):
        if target is None and joint_target is None:
            self._reset_shadow_to_arm(measured_tcp)
            self._preview_mode = None
            return
        mode = ("joint" if joint_target is not None else "eef", impedance)
        # A renderer can skip every released frame between two clutches.
        if reset_preview or mode != self._preview_mode:
            self._reset_shadow_to_arm(measured_tcp)
        self._preview_mode = mode
        if impedance:
            self._impedance_preview.advance(target, joint_target, self._flange_to_tcp, dt, incremental)
        elif joint_target is not None:
            q = np.asarray(joint_target, dtype=float)
            if q.shape != (6,) or not np.isfinite(q).all():
                raise ValueError("Expected six finite preview joint angles.")
            self._shadow.qpos[:] = q
            mujoco.mj_forward(self.sim.model, self._shadow)
        else:
            tcp_xyz, tcp_rotation = target
            offset, tool_rotation = self._flange_to_tcp
            flange_rotation = tcp_rotation * tool_rotation.inv()
            flange_xyz = tcp_xyz - flange_rotation.apply(offset)
            self._shadow.qpos[:] = self._joint_target(
                self._shadow, flange_xyz, flange_rotation, self._shadow.qpos)
            mujoco.mj_forward(self.sim.model, self._shadow)

    def _update_gripper_width(self, width_mm, *, measured):
        # The RG2 is posed via its geometry, independently of the six RTDE joints.
        if width_mm is None:
            width_mm = self._last_measured_width
        elif measured:
            self._last_measured_width = width_mm
        set_rg2_width(self.sim.model, width_mm)

    def _mirror_joints(self, measured_q):
        sim = self.sim
        q = np.asarray(measured_q, dtype=float)
        if q.shape != (sim.model.nq,) or not np.isfinite(q).all():
            raise ValueError("expected six finite measured joint angles")
        sim.data.qpos[:] = q
        mujoco.mj_forward(sim.model, sim.data)

    def _render_frame(self, display_tcp, *, mirroring_robot, width_available):
        sim = self.sim
        sim.viewer.data = self._shadow if self._previewing else sim.data
        if mirroring_robot:
            mode, source = ("DRY RUN", "Preview") if self._previewing else ("REAL RUN", "Measured")
            sim.viewer_text_overlay(mode, source, loc="top left")
            if not width_available:
                sim.viewer_text_overlay("RG2", "Width unavailable", loc="top left")
        if display_tcp is not None:
            xyz, rotation = display_tcp
            sim.plot_T(p=xyz, R=rotation.as_matrix(), axis_len=0.08, axis_width=0.004)
        sim.render()

    def tick(self, *, dry_run=False, target=None, joint_target=None, measured_q=None, measured_tcp=None,
             reset_preview=False, gripper_width_mm=None, preview_impedance=False, preview_dt=None,
             preview_delta=False):
        """Advance live physics, mirror RTDE joints, or articulate the independent preview."""
        mirroring_robot = measured_q is not None
        self._update_gripper_width(gripper_width_mm, measured=mirroring_robot and not dry_run)
        if mirroring_robot:
            self._mirror_joints(measured_q)
        elif not dry_run:
            self._advance_physics()

        self._previewing = dry_run
        if dry_run:
            dt = self.period if preview_dt is None else preview_dt
            self._update_shadow(target, joint_target, measured_tcp, reset_preview, preview_impedance, dt, preview_delta)
        else:
            self._preview_mode = None

        if self.headless or not self.is_running():
            return
        display_tcp = measured_tcp
        if dry_run:
            display_tcp = target
            if joint_target is not None or (preview_impedance and target is not None):
                display_tcp = self.preview_pose()
        self._render_frame(
            display_tcp, mirroring_robot=mirroring_robot, width_available=gripper_width_mm is not None,
        )

    def is_running(self):
        return self.sim is not None and (self.headless or self.sim.is_viewer_alive())

    def close(self):
        if self.sim is not None:
            if not self.headless:
                self.sim.close_viewer()
            self.sim = None
        self._shadow = None
        self._impedance_preview = None
        self._previewing = False
        self._preview_mode = None

    def __exit__(self, *exc):
        self.close()
