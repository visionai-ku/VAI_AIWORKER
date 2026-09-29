"""
SG2 swerve base keyboard teleop (sim + real).

Keys (MuJoCo viewer window focused):
    W / S : forward / backward      (+x / -x, base_link)
    A / D : left / right            (+y / -y)
    Q / E : turn left / turn right  (+wz / -wz)
    SPACE : stop immediately

The same SG2BaseTeleop drives the MuJoCo model in sim and produces body velocity
targets. The real notebook sends these through SG2BaseCommandClient's leased
JSON protocol; inbound.py publishes accepted targets to /cmd_vel.
Swerve geometry matches ffw_sg2_follower_ai_hardware_controller.yaml.
"""
import glfw
import numpy as np

from ri_motion_v5_package.kinematics import rpy2r
from ri_motion_v5_package.qt import MultiRadioQtWidget


SG2_BASE_STEER_JOINT_NAMES = ["left_wheel_steer_joint", "right_wheel_steer_joint", "rear_wheel_steer_joint"]
SG2_BASE_DRIVE_JOINT_NAMES = ["left_wheel_drive_joint", "right_wheel_drive_joint", "rear_wheel_drive_joint"]
SG2_BASE_MODULE_X = np.array([0.1371, 0.1371, -0.2899])   # [m]
SG2_BASE_MODULE_Y = np.array([0.2554, -0.2554, 0.0])      # [m]
SG2_BASE_WHEEL_RADIUS = 0.0865                            # [m]
SG2_BASE_STEER_LIMIT = 1.58                               # [rad], steer joint range in ffw_sg2.xml

# inbound.py clamps to the same limits
SG2_BASE_MAX_LIN = 0.3   # [m/s]
SG2_BASE_MAX_ANG = 0.6   # [rad/s]

SG2_BASE_KEYMAP = {
    glfw.KEY_W: (1.0, 0.0, 0.0),
    glfw.KEY_S: (-1.0, 0.0, 0.0),
    glfw.KEY_A: (0.0, 1.0, 0.0),
    glfw.KEY_D: (0.0, -1.0, 0.0),
    glfw.KEY_Q: (0.0, 0.0, 1.0),
    glfw.KEY_E: (0.0, 0.0, -1.0),
}
SG2_BASE_KEY_NAMES = {glfw.KEY_W: "W", glfw.KEY_S: "S", glfw.KEY_A: "A", glfw.KEY_D: "D",
                      glfw.KEY_Q: "Q", glfw.KEY_E: "E"}
SG2_BASE_SPEED_SCALES = {"SLOW": 0.3, "NORMAL": 0.6, "FAST": 1.0}


def wrap_to_pi(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def sg2_swerve_ik(v, prev_steer, eps=1e-4, steer_limit=SG2_BASE_STEER_LIMIT):
    """
    Body twist [vx, vy, wz] -> per-module steering angle [rad] and wheel speed [rad/s].
    Picks the steering direction closest to prev_steer (flip + reverse wheel), like the
    real controller's enabled_steering_flip, and flips again if that leaves the steer range.
    """
    vx, vy, wz = v
    steer = np.array(prev_steer, dtype=float)
    wheel_speed = np.zeros(3)
    for i in range(3):
        mvx = vx - wz * SG2_BASE_MODULE_Y[i]
        mvy = vy + wz * SG2_BASE_MODULE_X[i]
        s = np.hypot(mvx, mvy)
        if s < eps:
            continue
        d = wrap_to_pi(np.arctan2(mvy, mvx) - steer[i])
        if abs(d) > np.pi / 2:
            d = wrap_to_pi(d + np.pi)
            s = -s
        a = wrap_to_pi(steer[i] + d)
        if abs(a) > steer_limit:
            a = wrap_to_pi(a + np.pi)
            s = -s
        steer[i] = a
        wheel_speed[i] = s / SG2_BASE_WHEEL_RADIUS
    return steer, wheel_speed


class SG2BaseTeleop:
    """Keyboard -> smoothed body twist -> open-loop pose + swerve module states."""

    def __init__(self, max_lin=SG2_BASE_MAX_LIN, max_ang=SG2_BASE_MAX_ANG, acc_lin=0.5, acc_ang=1.0):
        self.max_lin, self.max_ang = float(max_lin), float(max_ang)
        self.acc = np.array([acc_lin, acc_lin, acc_ang])
        self.scale = 1.0
        self.v_target = np.zeros(3)
        self.v_cmd = np.zeros(3)
        self.pose = np.zeros(3)   # x, y, yaw (integrated command, open loop)
        self.steer = np.zeros(3)
        self.wheel_speed = np.zeros(3)
        self.wheel_angle = np.zeros(3)
        self.trail = []
        self.keys_held = []

    def read_keys(self, env):
        """Held keys set the target twist; SPACE stops at once."""
        if env.is_key_pressed_once(key=glfw.KEY_SPACE):
            self.stop()
            return
        d = np.zeros(3)
        self.keys_held = []
        for key, direction in SG2_BASE_KEYMAP.items():
            if env.is_key_pressed_repeat(key=key):
                d += direction
                self.keys_held.append(SG2_BASE_KEY_NAMES[key])
        d = np.clip(d, -1.0, 1.0)
        self.v_target = d * np.array([self.max_lin, self.max_lin, self.max_ang]) * self.scale

    def stop(self):
        self.v_target[:] = 0.0
        self.v_cmd[:] = 0.0

    def update(self, dt):
        dv = np.clip(self.v_target - self.v_cmd, -self.acc * dt, self.acc * dt)
        self.v_cmd = self.v_cmd + dv
        self.v_cmd[np.abs(self.v_cmd) < 1e-6] = 0.0

        vx, vy, wz = self.v_cmd
        c, s = np.cos(self.pose[2]), np.sin(self.pose[2])
        self.pose += dt * np.array([c * vx - s * vy, s * vx + c * vy, wz])
        self.pose[2] = float(wrap_to_pi(self.pose[2]))

        self.steer, self.wheel_speed = sg2_swerve_ik(self.v_cmd, self.steer)
        self.wheel_angle = wrap_to_pi(self.wheel_angle + self.wheel_speed * dt)

        if not self.trail or np.hypot(*(self.pose[:2] - self.trail[-1])) > 0.02:
            self.trail = (self.trail + [self.pose[:2].copy()])[-300:]

    def base_msg(self):
        return {"base": [float(v) for v in self.v_cmd]}

    def is_moving(self):
        return bool(np.any(self.v_cmd != 0.0) or np.any(self.v_target != 0.0))


class SG2BaseInputGate:
    """Require released motion keys after enabling or losing the active session.

    The real notebook handles SPACE by switching Robot command OFF before this
    gate runs. Simulation keeps its existing read_keys() behavior.
    """

    def __init__(self):
        self.waiting_for_release = True
        self.enabled_before = False

    def read_keys(self, teleop, env, *, enabled, active):
        rising = bool(enabled) and not self.enabled_before
        self.enabled_before = bool(enabled)
        if rising or not enabled or not active:
            self.waiting_for_release = True
            teleop.stop()
            return
        if self.waiting_for_release:
            held = any(env.is_key_pressed_repeat(key=key) for key in SG2_BASE_KEYMAP)
            self.waiting_for_release = held
            teleop.stop()
            return
        teleop.read_keys(env)
        # The strict robot gate limits the planar norm, including diagonal keys.
        speed = np.linalg.norm(teleop.v_target[:2])
        if speed > teleop.max_lin * teleop.scale:
            teleop.v_target[:2] *= teleop.max_lin * teleop.scale / speed


def get_sg2_base_z(env):
    jntadr = env.model.body("base_link").jntadr[0]
    return float(env.data.qpos[env.model.jnt_qposadr[jntadr] + 2])


def apply_sg2_base_to_env(env, teleop, base_z):
    """Write the base pose (free joint) and swerve steer/drive joints into the MuJoCo model."""
    x, y, yaw = teleop.pose
    env.set_pR("base_link", "base", (x, y, base_z), rpy2r(np.array([0.0, 0.0, yaw])), forward=False)
    env.forward(
        q=np.concatenate([teleop.steer, teleop.wheel_angle]),
        joint_names=SG2_BASE_STEER_JOINT_NAMES + SG2_BASE_DRIVE_JOINT_NAMES,
    )


def draw_sg2_base(env, teleop, base_z, arrow_gain=1.5):
    x, y, yaw = teleop.pose
    p = np.array([x, y, 0.02])
    R = rpy2r(np.array([0.0, 0.0, yaw]))
    for a, b in zip(teleop.trail[:-1], teleop.trail[1:]):
        env.plot_line_fr2to(np.r_[a, 0.005], np.r_[b, 0.005], rgba=[0.2, 0.6, 1.0, 0.8])
    v_world = R @ np.array([teleop.v_cmd[0], teleop.v_cmd[1], 0.0])
    if np.linalg.norm(v_world) > 1e-3:
        env.plot_arrow_fr2to(p, p + arrow_gain * v_world, r=0.01, rgba=[1.0, 0.4, 0.1, 0.9])
    if abs(teleop.v_cmd[2]) > 1e-3:
        env.plot_arrow_fr2to(p, p + R @ np.array([0.0, 0.0, arrow_gain * teleop.v_cmd[2]]), r=0.01,
                             rgba=[0.9, 0.2, 0.9, 0.9])


def make_sg2_base_overlay_text(teleop, extra=""):
    vx, vy, wz = teleop.v_cmd
    x, y, yaw = teleop.pose
    lines = [
        "W/S fwd/back  A/D left/right  Q/E turn  SPACE stop  (click this window first)",
        "keys: %s" % (" ".join(teleop.keys_held) or "-"),
        "speed x%.1f  (max %.2fm/s, %.2frad/s)" % (teleop.scale, teleop.max_lin * teleop.scale, teleop.max_ang * teleop.scale),
        "cmd  vx %+.2f  vy %+.2f  wz %+.2f" % (vx, vy, wz),
        "pose x %+.2f  y %+.2f  yaw %+.0fdeg (open loop)" % (x, y, np.degrees(yaw)),
        "steer [deg] " + " ".join("%+4.0f" % np.degrees(a) for a in teleop.steer),
    ]
    if extra:
        lines.append(extra)
    return "\n".join(lines)


def create_sg2_base_command_widget(real=False):
    labels, options, initial = ["Speed"], [list(SG2_BASE_SPEED_SCALES)], ["NORMAL"]
    if real:
        labels.insert(0, "Robot command")
        options.insert(0, ["OFF", "ON"])
        initial.insert(0, "OFF")
    labels.append("Camera")
    options.append(["FOLLOW", "FREE"])
    initial.append("FOLLOW")
    return MultiRadioQtWidget(
        title          = "SG2 Base Command",
        x_offset       = 0.00,
        y_offset       = 0.00,
        window_width   = 0.22,
        window_height  = 0.16 if real else 0.12,
        label_texts    = labels,
        option_texts   = options,
        initial_texts  = initial,)
