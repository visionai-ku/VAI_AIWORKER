"""Run a trained LeRobot policy on the real SG2 robot.

Builds the same env/guard/planner and SG2FixedQuestReal session teleop uses
(see intent_teleop.build_models/build_planner and fixed_quest_real.tick's
action="policy" mode), but drives it from PolicyRuntime predictions -- read
robot state + camera frames over the same ZMQ endpoints episode_recorder.py
reads, predict, clip_to_joint_limits(), send -- instead of a VR frame. No
Quest controller server, no Qt window: this is a headless deployment loop.

Control frequency: one fixed hz for both reading observations and predicting
(matching the dataset's own recording rate, the same pattern lerobot's own
record_loop() uses). ACTPolicy.select_action() already queues a chunk
internally and only calls the model again once it is consumed, so a single
hz is enough for a first deployment; split it into separate inference/control
rates (see the ROBOTIS AIWorker inference docs) only if per-tick latency at
this hz ever falls behind.

Session control: intentionally bare. Call start()/step()/stop() (or the
blocking run()) from a notebook cell, a keypress loop, whatever the caller
wants -- there is no VR-button UI here, since a policy session has no VR
controller.
"""
from __future__ import annotations

import pickle
import time

import numpy as np
import zmq

from ri_motion_v5_package.mujoco_sim import MuJoCoParser

from package.episode_recorder import DEFAULT_CAMERAS, _recv_latest, _sub, decode_frame
from package.fixed_quest_real import SG2FixedQuestReal
from package.fixed_quest_sim import reset_sg2_sim
from package.fixed_quest_transport import SG2RobotCommandPublisher, SG2RobotStateReceiver
from package.intent_teleop import IntentTeleopSettings, build_models, build_planner
from package.policy_inference import DEFAULT_TASK, PolicyRuntime, clip_to_joint_limits
from package.safety_box import wrist_safety_box_violations
from package.worker.fixed_quest_protocol import FIXED_QUEST_VR_BRIDGE_LIMITS, JOINT_NAMES


class PolicyDeployment:
    """One checkpoint, one robot session. Build once; start()/stop() may repeat."""

    def __init__(self, checkpoint_path, robot_ip="192.168.6.2", cameras=None, hz=30,
                 task=None, device=None, dry_run=True):
        if not 0 < hz <= 200:
            raise ValueError("hz must be in (0, 200]")
        self.cfg = IntentTeleopSettings(backend="real", ROBOT_IP=robot_ip, DRY_RUN=dry_run)
        self.models = build_models(self.cfg)
        _, self.planner = build_planner(self.models, self.cfg)
        self.publisher = SG2RobotCommandPublisher(enabled=not dry_run, port=self.cfg.ROBOT_COMMAND_PORT)
        self.real = SG2FixedQuestReal(
            self.planner, self.models["env_measured"], self.models["q_msg_layout"], self.publisher,
            dry_run=dry_run, max_joint_lag=self.cfg.MAX_JOINT_LAG, max_lift_lag=self.cfg.MAX_LIFT_LAG,
            command_ttl_s=self.cfg.COMMAND_TTL_S, max_home_joint_speed=self.cfg.MAX_HOME_JOINT_SPEED,
            max_arm_joint_lag=self.cfg.MAX_ARM_JOINT_LAG, teleop_ramp_s=self.cfg.TELEOP_RAMP_S,
            required_bridge_limits=FIXED_QUEST_VR_BRIDGE_LIMITS, command_arm_speed_limit=True,
            tracking_pause_s=self.cfg.TRACKING_PAUSE_S)
        self.receiver = SG2RobotStateReceiver(robot_ip, self.cfg.ROBOT_STATE_PORT, self.cfg.FEEDBACK_TIMEOUT_S).start()
        self.robot_ip, self.cameras = robot_ip, dict(DEFAULT_CAMERAS if cameras is None else cameras)
        self.policy = PolicyRuntime(checkpoint_path, device=device, task=task or DEFAULT_TASK)
        self.hz, self.dt = float(hz), 1. / float(hz)
        self.dry_run = dry_run
        self.running = False
        self._cam_ctx, self._cam_socks, self._last_frame = None, {}, {}
        self.last_result = None
        self.last_clipped = {}
        # Dedicated FK-only scratch env for the Cartesian safety box below --
        # never shared with the planner/guard/measured envs in self.models.
        self.env_safety_check = MuJoCoParser(rel_xml_path=self.models["xml_path"], verbose=False)
        reset_sg2_sim(self.env_safety_check)
        self.last_safety_box_violations = {}

    # ---------- cameras (same ZMQ pattern as episode_recorder.py) ----------
    def _open_cameras(self):
        self._cam_ctx = zmq.Context()
        self._cam_socks = {cam: _sub(self._cam_ctx, "tcp://%s:%d" % (self.robot_ip, port))
                           for cam, port in self.cameras.items()}

    def _close_cameras(self):
        for sock in self._cam_socks.values():
            sock.close(linger=0)
        if self._cam_ctx is not None:
            self._cam_ctx.term()
        self._cam_ctx, self._cam_socks = None, {}

    def _poll_cameras(self):
        for cam, sock in self._cam_socks.items():
            raw = _recv_latest(sock)
            if raw is None:
                continue
            bgr = decode_frame(pickle.loads(raw))
            if bgr is not None:
                self._last_frame[cam] = bgr
        return dict(self._last_frame)

    # ---------- session lifecycle ----------
    def start(self):
        if self.running:
            return
        self._open_cameras()
        self.policy.reset()
        self.last_result, self.last_clipped, self.last_safety_box_violations = None, {}, {}
        self.running = True

    def stop(self):
        self.running = False
        self.real.stop()
        self._close_cameras()
        self._last_frame = {}

    def close(self):
        """Full teardown: release the session and every socket. Build a new
        PolicyDeployment to run again."""
        self.stop()
        self.receiver.close()
        self.publisher.close()

    # ---------- control ----------
    def step(self, now=None):
        """One control tick. Call at self.hz from run(), or drive it yourself."""
        if not self.running:
            raise RuntimeError("call start() before step()")
        now = time.monotonic() if now is None else float(now)
        feedback = self.receiver.latest(now)
        frames = self._poll_cameras()
        ready = feedback.get("ready") and feedback.get("q") is not None and len(frames) == len(self.cameras)
        if not ready:
            result = self.real.tick(None, feedback, enabled=False, now=now)
        else:
            q = np.asarray(feedback["q"], dtype=float)
            raw_action = self.policy.predict(q, frames)
            q_target, exceeded = clip_to_joint_limits(raw_action)
            self.last_clipped = exceeded
            if exceeded:
                print("[policy] clipped %d/%d joints to JOINT_LIMITS: %s"
                      % (len(exceeded), len(JOINT_NAMES), ", ".join(exceeded)))
            violations = wrist_safety_box_violations(self.env_safety_check, q_target, JOINT_NAMES)
            self.last_safety_box_violations = violations
            if violations:
                print("[policy] safety box violation, holding position: %s" % {
                    hand: {"requested_m": requested.round(3).tolist(), "box_limit_m": box.round(3).tolist()}
                    for hand, (requested, box) in violations.items()
                })
                result = self.real.tick(None, feedback, enabled=False, now=now)
            else:
                result = self.real.tick(None, feedback, enabled=True, now=now, action="policy", q_target=q_target)
        self.last_result = result
        return result

    def run(self, duration_s=None):
        """Blocking fixed-hz loop. Stops on FAULT, on duration_s elapsing, or
        on KeyboardInterrupt -- always releases the session on the way out."""
        self.start()
        try:
            t_next = time.monotonic()
            end = None if duration_s is None else t_next + duration_s
            while end is None or time.monotonic() < end:
                result = self.step()
                if result["state"] == "FAULT":
                    print("[policy] FAULT:", result["reason"])
                    break
                t_next += self.dt
                sleep_for = t_next - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    t_next = time.monotonic()  # fell behind this tick; resync instead of catching up
        except KeyboardInterrupt:
            print("[policy] interrupted")
        finally:
            self.stop()
        return self.last_result
