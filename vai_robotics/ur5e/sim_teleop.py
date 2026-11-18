"""Cartesian teleoperation of the UR5e arm in the bundled MuJoCo scene."""

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

    def __enter__(self):
        self.sim = load_simulation()
        try:
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

    def pose(self):
        site = self.sim.data.site(self.site_name)
        return site.xpos.copy(), Rotation.from_matrix(site.xmat.reshape(3, 3))

    def command(self, xyz, rotation):
        """Apply one bounded differential IK update to the six position actuators."""
        # Loaded by load_simulation from the same bundled motion library as the SG2.
        from ri_motion_v5_package.mujoco_sim.mujoco_ik import solve_dls

        sim = self.sim
        xyz = np.asarray(xyz, dtype=float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all() or not rotation.single:
            raise ValueError("expected a finite xyz position and one orientation")
        current_xyz, current_rot = self.pose()
        error = np.r_[xyz - current_xyz, (rotation * current_rot.inv()).as_rotvec()]
        jacobian = sim.get_J_site(self.site_name, return_type="full")
        q_target = sim.data.qpos + solve_dls(jacobian, error, lam=0.002**2)
        max_step = self.joint_speed * self.period
        change = q_target - sim.data.ctrl
        # Scale all six joints together to preserve the Cartesian direction.
        change *= max_step / max(max_step, np.max(np.abs(change)))
        lower, upper = sim.model.actuator_ctrlrange.T
        sim.data.ctrl[:] = np.clip(sim.data.ctrl + change, lower, upper)
        self.active = True

    def hold(self):
        """Latch measured joint positions on release; keep holding on subsequent ticks."""
        if self.active:
            self.sim.data.ctrl[:] = self.sim.data.qpos
            self.active = False

    def tick(self):
        sim = self.sim
        self.tick_count += 1
        nstep = round((self.tick_count * self.period - sim.data.time) / sim.dt)
        sim.compensate_gravity(["base"])
        sim.step(nstep=nstep)
        if not self.headless and self.is_running():
            sim.render()

    def is_running(self):
        return self.sim is not None and (self.headless or self.sim.is_viewer_alive())

    def close(self):
        if self.sim is not None:
            if not self.headless:
                self.sim.close_viewer()
            self.sim = None

    def __exit__(self, *exc):
        self.close()
