"""Cartesian teleoperation of the UR5e arm in the bundled MuJoCo scene."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

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
        self.sim = None
        self.active = False
        self.tick_count = 0
        self._shadow = None
        self._previewing = False
        self._preview_clutched = False

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
        self.tick_count = 0
        return self

    def _flange_pose(self, data):
        site = data.site(self.site_name)
        return site.xpos.copy(), Rotation.from_matrix(site.xmat.reshape(3, 3))

    def pose(self):
        return self._flange_pose(self.sim.data)

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
        jacobian = np.empty((6, sim.model.nv))
        mujoco.mj_jacSite(sim.model, data, jacobian[:3], jacobian[3:], data.site(self.site_name).id)
        q_target = data.qpos + solve_dls(jacobian, error, lam=0.002**2)
        max_step = self.joint_speed * self.period
        change = q_target - previous_target
        # Scale all six joints together to preserve the Cartesian direction.
        change *= max_step / max(max_step, np.max(np.abs(change)))
        lower, upper = sim.model.actuator_ctrlrange.T
        return np.clip(previous_target + change, lower, upper)

    def command(self, xyz, rotation):
        """Apply one bounded differential IK update to the six live position actuators."""
        data = self.sim.data
        data.ctrl[:] = self._joint_target(data, xyz, rotation, data.ctrl)
        self.active = True

    def hold(self):
        """Latch measured joint positions on release; keep holding on subsequent ticks."""
        if self.active:
            self.sim.data.ctrl[:] = self.sim.data.qpos
            self.active = False

    def _update_shadow(self, target, measured_tcp):
        if target is None or not self._preview_clutched:
            self._shadow.qpos[:] = self.sim.data.qpos
            mujoco.mj_forward(self.sim.model, self._shadow)
            # A real TCP can be offset and rotated relative to this model's flange.
            if measured_tcp is None:
                self._flange_to_tcp = np.zeros(3), Rotation.identity()
            else:
                xyz, rotation = self._flange_pose(self._shadow)
                tcp_xyz, tcp_rotation = measured_tcp
                self._flange_to_tcp = rotation.inv().apply(tcp_xyz - xyz), rotation.inv() * tcp_rotation
        if target is not None:
            tcp_xyz, tcp_rotation = target
            offset, tool_rotation = self._flange_to_tcp
            flange_rotation = tcp_rotation * tool_rotation.inv()
            flange_xyz = tcp_xyz - flange_rotation.apply(offset)
            self._shadow.qpos[:] = self._joint_target(
                self._shadow, flange_xyz, flange_rotation, self._shadow.qpos)
            mujoco.mj_forward(self.sim.model, self._shadow)
        self._preview_clutched = target is not None

    def tick(self, *, dry_run=False, target=None, measured_q=None, measured_tcp=None, reset_preview=False):
        """Advance live physics, mirror RTDE joints, or articulate the independent preview."""
        sim = self.sim
        if measured_q is not None:
            q = np.asarray(measured_q, dtype=float)
            if q.shape != (sim.model.nq,) or not np.isfinite(q).all():
                raise ValueError("expected six finite measured joint angles")
            sim.data.qpos[:] = q
            mujoco.mj_forward(sim.model, sim.data)
        elif not dry_run:
            self.tick_count += 1
            nstep = round((self.tick_count * self.period - sim.data.time) / sim.dt)
            sim.compensate_gravity(["base"])
            sim.step(nstep=nstep)
        self._previewing = dry_run
        if dry_run:
            # A renderer can skip every released frame between two clutches.
            if reset_preview:
                self._preview_clutched = False
            self._update_shadow(target, measured_tcp)
        else:
            self._preview_clutched = False
        if not self.headless and self.is_running():
            sim.viewer.data = self._shadow if dry_run else sim.data
            if measured_q is not None:
                mode, source = ("DRY RUN", "Preview") if dry_run else ("REAL RUN", "Measured")
                sim.viewer_text_overlay(mode, source, loc="top left")
            display_tcp = target if dry_run else measured_tcp
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
