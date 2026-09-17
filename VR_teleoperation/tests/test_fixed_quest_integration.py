"""Actual PC adapter/cache and robot gate, with in-memory followers.

This verifies message/state integration, not physical actuator performance.
No command socket or ROS publisher is constructed.
"""
import copy
from pathlib import Path
import runpy
import tempfile
import unittest

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
runpy.run_path(str(PROJECT_DIR / "package/init_project.py"))

from ri_motion_v5_package.mujoco_sim import MuJoCoParser, BodyPositionIKSolverRevPriBase
from utils import create_sg2_arm_ik_solvers, build_q_msg_layout, get_sg2_wrist_control_T
from package.controller_mapping import ControllerMapper
from package.fixed_quest_scene import SG2CommandGuard, build_fixed_quest_scene
from package.fixed_quest_sim import SG2FixedQuestSimulation, SyntheticControllerSource, reset_sg2_sim
from package.fixed_quest_real import SG2FixedQuestReal
from package.fixed_quest_transport import SG2RobotFeedbackCache
from package.worker.fixed_quest_protocol import JOINT_GROUPS, JOINT_NAMES, JointStateTracker, StrictCommandGate, parse_joint_groups


class MemoryBridgePublisher:
    enabled = True

    def __init__(self, gate, q_initial):
        self.gate, self.q = gate, np.asarray(q_initial).copy()
        self.now = 1.
        self.receive_offset_s = 0.
        self.drop_commands = False
        self.drop_windows = []
        self.follow_speed = None
        self.follow_time_constant = None
        self.target = self.q.copy()
        self.messages, self.outcomes = [], []
        self.deliveries, self.dropped = [], []

    def advance(self, now):
        if self.follow_speed is not None:
            dt = max(0., now - self.now)
            step = np.asarray(self.follow_speed) * dt
            delta = self.target - self.q
            if self.follow_time_constant is not None:
                delta *= 1. - np.exp(-dt / self.follow_time_constant)
            self.q += np.clip(delta, -step, step)
        self.now = now

    def apply_target(self, joints):
        self.target = np.asarray(parse_joint_groups(joints))
        if self.follow_speed is None:
            self.q = self.target.copy()

    def send(self, packet):
        self.messages.append(copy.deepcopy(packet))
        dropping = self.drop_commands or any(start <= self.now < end for start, end in self.drop_windows)
        if dropping and packet["mode"] == "command":
            self.dropped.append(copy.deepcopy(packet))
            return True  # Local enqueue success is not robot acknowledgment.
        received_at = self.now + self.receive_offset_s
        outcome = self.gate.receive(packet, received_at)
        self.outcomes.append(outcome)
        self.deliveries.append((received_at, copy.deepcopy(packet), copy.deepcopy(outcome)))
        if outcome["publish_joints"] is not None:
            self.apply_target(outcome["publish_joints"])
        return True


class FixedQuestIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache_dir = tempfile.TemporaryDirectory(dir=PROJECT_DIR / ".cache")
        xml = build_fixed_quest_scene(PROJECT_DIR / "xml/ffw_sg2.xml", PROJECT_DIR / "xml/floor_simple_white.xml",
                                      Path(cls.cache_dir.name) / "scene.xml")
        cls.env, cls.env_ik, cls.env_check, cls.env_measured = [MuJoCoParser(rel_xml_path=xml, verbose=False) for _ in range(4)]
        for model_env in (cls.env, cls.env_ik, cls.env_check, cls.env_measured):
            reset_sg2_sim(model_env)
        cls.solvers, cls.arms = create_sg2_arm_ik_solvers(cls.env_ik, MuJoCoParser, BodyPositionIKSolverRevPriBase)
        cls.guard = SG2CommandGuard(cls.env_check, JOINT_NAMES)
        cls.layout = build_q_msg_layout(cls.env, {group: {"prefixes": names} for group, names in JOINT_GROUPS.items()},
                                        use_joint_names=list(JOINT_NAMES))

    @classmethod
    def tearDownClass(cls):
        cls.cache_dir.cleanup()

    def setUp(self):
        for model_env in (self.env, self.env_ik, self.env_check, self.env_measured):
            reset_sg2_sim(model_env)
        self.initial = self.env.get_qpos(joint_names=JOINT_NAMES).copy()
        self.initial[0], self.initial[1], self.initial[3] = -.1, .05, .04
        self.tracker = JointStateTracker()
        self.gate = StrictCommandGate(self.tracker)
        self.feedback_cache = SG2RobotFeedbackCache()
        self.publisher = MemoryBridgePublisher(self.gate, self.initial)
        self.planner = SG2FixedQuestSimulation(self.env, self.env_ik, self.solvers, self.arms, JOINT_NAMES,
                                              ControllerMapper(scale=.25), self.guard, max_joint_speed=np.deg2rad(60),
                                              max_command_dt=.02)
        self.planner.speed_limits[JOINT_NAMES.index("lift_joint")] = .01
        self.real = SG2FixedQuestReal(self.planner, self.env_measured, self.layout, self.publisher,
                                      dry_run=False, max_home_joint_speed=np.deg2rad(10))
        self.heartbeat = None
        self.heartbeat_history = []
        self.heartbeat_period_steps = 1
        self.heartbeat_delay_s = 0.
        self.receive_offsets_s = (0.,)

    def inputs(self, index, grip=False, height=0.):
        robot_now, pc_now = 1. + index * .02, 101. + index * .02
        self.publisher.advance(robot_now)
        self.publisher.receive_offset_s = self.receive_offsets_s[index % len(self.receive_offsets_s)]
        self.tracker.update(JOINT_NAMES, self.publisher.q.tolist(), round(robot_now * 1e9), robot_now)
        expired = self.gate.expire(robot_now)
        if expired["publish_joints"] is not None:
            self.publisher.apply_target(expired["publish_joints"])
        if index % self.heartbeat_period_steps == 0 or self.heartbeat is None:
            self.heartbeat_history.append(self.gate.status(robot_now))  # Nominal bridge interval is 20 ms.
        available = [status for status in self.heartbeat_history
                     if status["status_time_s"] <= robot_now - self.heartbeat_delay_s + 1e-12]
        self.heartbeat = available[-1] if available else self.heartbeat_history[0]
        payload = self.tracker.snapshot(robot_now)
        payload.update(bridge=copy.deepcopy(self.heartbeat), bridge_age_s=robot_now - self.heartbeat["status_time_s"])
        self.assertTrue(self.feedback_cache.ingest(payload, received_at=pc_now))
        feedback = self.feedback_cache.latest(pc_now)
        self.assertTrue(feedback["ready"], feedback["reason"])
        frame = SyntheticControllerSource().latest(pc_now)
        frame.update(seq=index, sample_time_ms=pc_now * 1000, received_at=pc_now)
        for hand in frame["hands"].values():
            hand.update(grip=float(grip), trigger=0.)
        frame["hands"]["left"]["position_m"] = [.2, 1.1 + height, -.6]
        return frame, feedback, pc_now

    def tick(self, index, enabled=True, grip=False, height=0., action=None):
        frame, feedback, pc_now = self.inputs(index, grip=grip, height=height)
        return self.real.tick(frame, feedback, enabled=enabled, now=pc_now, action=action)

    def tick_grip(self, index, grips=(False, False), height=0., action=None,
                  frame_available=True, right_valid=True, seq=None):
        frame, feedback, pc_now = self.inputs(index, height=height)
        for hand, grip in zip(("left", "right"), grips):
            frame["hands"][hand]["grip"] = float(grip)
        frame["hands"]["right"]["valid"] = right_valid
        if seq is not None:
            frame["seq"] = seq
        return self.real.tick_grip(frame if frame_available else None, feedback, now=pc_now, action=action)

    def arm(self):
        self.tick(0, enabled=False)
        for index in range(1, 6):
            result = self.tick(index)
        self.assertTrue(result["armed"], result["reason"])
        self.assertEqual(result["state"], "READY")
        return result

    def test_delayed_arm_ack_then_both_hands_use_real_gate(self):
        self.heartbeat_period_steps = 5
        self.arm()
        modes = [packet["mode"] for packet in self.publisher.messages]
        self.assertEqual(modes, ["arm"] * 4 + ["command"])
        for index in range(6, 31):
            result = self.tick(index, grip=True, height=max(0, index - 6) * .0002)
            self.assertNotEqual(result["state"], "FAULT", result["reason"])
        self.assertEqual(set(result["active"]), {"left", "right"})
        self.assertTrue(all(outcome["accepted"] for outcome in self.publisher.outcomes), self.publisher.outcomes)
        self.assertEqual(self.gate.last_command_seq, self.publisher.messages[-1]["seq"])
        self.assertEqual(self.gate.owner_session_id, self.real.session_id)
        self.assertGreater(np.max(np.abs(self.publisher.q - self.initial)), 1e-4)

    def test_session_start_return_is_accepted_by_robot_rate_gate(self):
        self.tick(0, enabled=False)
        self.publisher.q[3] += .02
        self.tick(1, enabled=False)
        for index in range(2, 6):
            self.tick(index)
        self.assertTrue(self.real.armed)
        previous = self.publisher.q.copy()
        home_limits = self.real.home_speed_limits.copy()
        for index in range(6, 61):
            result = self.tick(index, action="home" if index == 6 else None)
            self.assertNotEqual(result["state"], "FAULT", result["reason"])
            self.assertTrue(np.all(np.abs(result["q_command"] - previous) <= home_limits * .02 + 1e-10))
            previous = result["q_command"].copy()
            if result["state"] == "HOME":
                break
        self.assertEqual(result["state"], "HOME")
        self.assertTrue(result["needs_disable"])
        np.testing.assert_allclose(result["q_command"], self.initial, atol=1e-9)
        self.assertTrue(all(outcome["accepted"] for outcome in self.publisher.outcomes), self.publisher.outcomes)
        self.assertEqual(self.publisher.messages[-1]["mode"], "stop")
        self.assertEqual(self.gate.state, "STOPPED")

    def configure_adjustable_profile(self):
        from package.worker.fixed_quest_protocol import FIXED_QUEST_VR_GATE_OPTIONS, FIXED_QUEST_VR_BRIDGE_LIMITS
        self.gate = StrictCommandGate(self.tracker, **FIXED_QUEST_VR_GATE_OPTIONS)
        self.publisher.gate = self.gate
        mask = np.array([name.startswith("arm_") for name in JOINT_NAMES])
        self.planner.speed_limits[mask] = np.deg2rad(120)
        self.planner.ik_period = .02
        self.real = SG2FixedQuestReal(self.planner, self.env_measured, self.layout, self.publisher,
            dry_run=False, max_arm_joint_lag=.48, teleop_ramp_s=8.,
            required_bridge_limits=FIXED_QUEST_VR_BRIDGE_LIMITS, command_arm_speed_limit=True)

    def test_adjustable_profile_existing_ik_exceeds_old_cap_and_home_stays_slow(self):
        self.configure_adjustable_profile()
        mask = self.real.arm_mask
        self.arm()
        self.real.teleop_started_at -= 8.  # Ramp arithmetic is tested separately.
        previous = self.tick(6, grip=True)["q_command"].copy()
        peak = 0.
        for index in range(7, 19):
            result = self.tick(index, grip=True, height=.16)
            self.assertEqual(result["state"], "ACTIVE", result["reason"])
            speed = np.abs(result["q_command"] - previous) / .02
            self.assertTrue(np.all(speed <= self.real.teleop_speed_limits + 1e-9))
            peak = max(peak, np.max(speed[mask]))
            previous = result["q_command"].copy()
        self.assertGreater(peak, np.deg2rad(60))
        for index in range(19, 160):
            result = self.tick(index, action="home" if index == 19 else None)
            self.assertNotEqual(result["state"], "FAULT", result["reason"])
            self.assertTrue(np.all(np.abs(result["q_command"] - previous) <= self.real.home_speed_limits * .02 + 1e-9))
            previous = result["q_command"].copy()
            if result["state"] == "HOME":
                break
        self.assertEqual(result["state"], "HOME")
        self.assertTrue(all(outcome["accepted"] for outcome in self.publisher.outcomes), self.publisher.outcomes)

    def test_live_speed_decrease_drains_ack_before_lower_packet_cap(self):
        self.configure_adjustable_profile()
        self.arm()
        self.real.teleop_started_at -= 8.
        self.tick(6, grip=True)
        self.heartbeat_period_steps = 3
        pending_seen = applied_seen = False
        session = self.real.session_id
        for index in range(7, 50):
            if index == 8:
                self.assertTrue(self.real.set_teleop_speed(10.))
            result = self.tick(index, grip=True, height=.16)
            self.assertEqual(result["state"], "ACTIVE", result["reason"])
            self.assertEqual(self.real.session_id, session)
            pending_seen |= result["speed_change_pending"]
            if np.isclose(result["teleop_speed_deg_s"], 10.):
                applied_seen = True
                self.assertLessEqual(self.publisher.messages[-1]["arm_speed_rad_s"], np.deg2rad(10) + 1e-12)
        self.assertTrue(pending_seen)
        self.assertTrue(applied_seen)
        self.assertTrue(all(outcome["accepted"] for outcome in self.publisher.outcomes), self.publisher.outcomes)

    def test_adjustable_profile_slow_follower_jitter_and_one_grip_release(self):
        self.configure_adjustable_profile()
        self.publisher.follow_speed = self.real.teleop_speed_limits.copy()
        self.publisher.follow_time_constant = .2
        self.receive_offsets_s = (.01, 0.)  # Alternating 30/10 ms delivery intervals.
        self.heartbeat_period_steps = 2
        self.tick_grip(0)
        for index in range(1, 10):
            result = self.tick_grip(index, grips=(True, True))
        self.assertEqual(result["state"], "ACTIVE", result["reason"])
        self.real.teleop_started_at -= 8.
        start = self.publisher.q.copy()
        for index in range(10, 160):
            result = self.tick_grip(index, grips=(True, True), height=min(.5, (index - 10) * .02))
            self.assertNotEqual(result["state"], "FAULT", result["reason"])
            self.assertTrue(np.all(np.abs(result["q_command"] - result["robot_q"]) <= self.real.lag_limits + 1e-9))
        self.assertGreater(np.max(np.abs(self.publisher.q - start)), .1)
        result = self.tick_grip(160, grips=(True, False), height=.5)
        self.assertEqual(result["state"], "WAIT_GRIP", result["reason"])
        self.assertEqual(self.publisher.messages[-1]["mode"], "stop")
        self.assertEqual(self.gate.state, "STOPPED")

    def test_teleop_can_exceed_home_speed_and_return_is_slow_from_first_step(self):
        self.arm()
        anchored = self.tick(6, grip=True)
        previous = anchored["q_command"].copy()
        teleop_speeds = []
        for index in range(7, 19):
            # A valid 16 cm controller translation becomes a 4 cm EEF target.
            # Actual existing IK, the joint limiter and both guards process it.
            result = self.tick(index, grip=True, height=.16)
            self.assertEqual(result["state"], "ACTIVE", result["reason"])
            speed = np.abs(result["q_command"] - previous) / .02
            self.assertTrue(np.all(speed <= self.real.teleop_speed_limits + 1e-9), speed)
            teleop_speeds.append(speed[1:])
            previous = result["q_command"].copy()
            self.assertEqual(self.publisher.messages[-1]["motion"], "teleop")
        self.assertGreater(np.max(teleop_speeds), np.deg2rad(30) + 1e-4)
        self.assertLessEqual(np.max(teleop_speeds), np.deg2rad(60) + 1e-9)
        self.assertGreater(np.max(np.abs(previous - self.initial)), .01)

        home_limits = self.real.home_speed_limits.copy()
        start_message = len(self.publisher.messages)
        home_steps = []
        arming_ticks = 0
        for index in range(19, 120):
            result = self.tick(index, grip=True, height=.16, action="home" if index == 19 else None)
            if not home_steps and result["state"] == "ARMING":
                # The teleop lease is released for the whole-path check, then
                # the return re-arms at the measured pose without moving.
                arming_ticks += 1
                self.assertFalse(any(packet["mode"] == "command"
                                     for packet in self.publisher.messages[start_message:]))
                previous = result["q_command"].copy()
                continue
            self.assertIn(result["state"], ("HOMING", "HOME"), result["reason"])
            delta = np.abs(result["q_command"] - previous)
            home_steps.append(delta)
            self.assertTrue(np.all(delta <= home_limits * .02 + 1e-10), delta)
            previous = result["q_command"].copy()
            self.assertEqual(result["active"], [])
            if result["state"] == "HOME":
                break
        self.assertEqual(result["state"], "HOME")
        self.assertGreaterEqual(arming_ticks, 1)
        self.assertEqual(self.publisher.messages[start_message]["mode"], "stop")
        self.assertGreater(np.max(np.max(home_steps, axis=0)), 0.)
        self.assertLessEqual(np.max(home_steps[0][1:]), np.deg2rad(10) * .02 + 1e-10)
        home_packets = [packet for packet in self.publisher.messages[start_message:] if packet["mode"] == "command"]
        self.assertGreater(len(home_packets), 1)
        self.assertTrue(all(packet["motion"] == "home" for packet in home_packets))
        np.testing.assert_allclose(result["q_command"], self.initial, atol=1e-9)
        self.assertTrue(all(outcome["accepted"] for outcome in self.publisher.outcomes), self.publisher.outcomes)
        self.assertEqual(self.gate.state, "STOPPED")

    def test_lost_packets_expire_actual_gate_and_require_new_session(self):
        self.arm()
        old_session = self.real.session_id
        self.publisher.drop_commands = True
        for index in range(6, 31):
            result = self.tick(index)
            if result["needs_disable"]:
                break
        self.assertEqual(result["state"], "FAULT")
        self.assertIn("lease lost", result["reason"])
        self.assertIn(old_session, self.gate.expired_sessions)
        self.assertIsNone(self.real.session_id)
        self.publisher.drop_commands = False
        self.tick(index + 1, enabled=False)
        for index in range(index + 2, index + 12):
            result = self.tick(index)
            if result["armed"]:
                break
        self.assertTrue(result["armed"], result["reason"])
        self.assertNotEqual(old_session, self.real.session_id)
        self.assertEqual(self.gate.owner_session_id, self.real.session_id)

    def test_slow_following_paces_teleop_and_release_discards_pending_motion(self):
        # A first-order follower with a 10 deg/s ceiling deliberately lags the
        # 60 deg/s command preview. This is a deterministic plant, not physics.
        self.publisher.follow_speed = np.full(len(JOINT_NAMES), np.deg2rad(10))
        self.publisher.follow_speed[0] = .01
        self.publisher.follow_time_constant = .8
        for index in range(6):
            result = self.tick_grip(index, grips=(True, True))
        previous = result["q_command"].copy()
        start_q = previous.copy()
        paced, peak_lag, peak_speed = False, 0., 0.
        for index in range(6, 181):
            result = self.tick_grip(index, grips=(True, True), height=.16)
            self.assertEqual(result["state"], "ACTIVE", result["reason"])
            speed = np.abs(result["q_command"] - previous) / .02
            self.assertTrue(np.all(speed <= self.real.teleop_speed_limits + 1e-9), speed)
            lag = np.abs(result["q_command"] - result["robot_q"])
            self.assertTrue(np.all(lag <= self.real.lag_limits + 1e-12), lag)
            paced |= bool(result.get("tracking_paced", False))
            peak_lag = max(peak_lag, np.max(lag[1:]))
            peak_speed = max(peak_speed, np.max(speed[1:]))
            previous = result["q_command"].copy()
        self.assertTrue(paced, "The slow follower must exercise tracking pacing")
        self.assertGreater(peak_lag, np.deg2rad(3))
        self.assertGreater(peak_speed, np.deg2rad(10))
        self.assertGreater(np.max(np.abs(self.publisher.q - start_q)), np.deg2rad(3))
        self.assertTrue(all(outcome["accepted"] for outcome in self.publisher.outcomes), self.publisher.outcomes)
        self.assertGreater(np.max(np.abs(self.publisher.target - self.publisher.q)), 1e-5)

        # Releasing one grip must replace the outstanding target with the fresh
        # measured hold, even while the other grip and old IK target remain held.
        count = len(self.publisher.messages)
        stopped = self.tick_grip(181, grips=(False, True), height=.16)
        self.assertEqual(stopped["active"], [])
        self.assertFalse(stopped["armed"])
        self.assertEqual([packet["mode"] for packet in self.publisher.messages[count:]], ["stop"])
        held = self.publisher.q.copy()
        np.testing.assert_allclose(self.publisher.target, held, atol=1e-12)
        for index in range(182, 193):
            self.tick_grip(index, grips=(False, True), height=.18)
            np.testing.assert_allclose(self.publisher.q, held, atol=1e-12)
        self.assertEqual(len(self.publisher.messages), count + 1)

    def test_measured_jump_beyond_hard_tracking_limit_still_stops(self):
        for index in range(6):
            self.tick_grip(index, grips=(True, True))
        self.publisher.q[3] -= np.deg2rad(5.1)
        result = self.tick_grip(6, grips=(True, True))
        self.assertEqual(result["state"], "FAULT")
        self.assertIn("tracking difference exceeds limit", result["reason"])
        self.assertEqual(result["active"], [])
        self.assertFalse(result["armed"])
        self.assertEqual(self.publisher.messages[-1]["mode"], "stop")
        self.assertIsNone(self.gate.owner_session_id)

    def test_home_recovers_command_drops_and_delayed_ack_without_rate_rejections(self):
        self.heartbeat_delay_s = .1
        self.tick_grip(0, frame_available=False)
        self.publisher.q[3] += np.deg2rad(8)
        displaced = self.publisher.q.copy()
        self.tick_grip(1, action="home", frame_available=False)
        # Each 80 ms loss burst is shorter than the unchanged 200 ms lease.
        # Robot bridge heartbeats additionally arrive 100 ms after generation.
        self.publisher.drop_windows = [(1.32, 1.40), (1.72, 1.80), (2.12, 2.20)]
        previous = displaced.copy()
        ack_paced = False
        for index in range(2, 601):
            result = self.tick_grip(index, frame_available=False)
            self.assertIn(result["state"], ("ARMING", "HOMING", "HOME"), result["reason"])
            delta = np.abs(result["q_command"] - previous)
            self.assertTrue(np.all(delta <= self.real.home_speed_limits * .02 + 1e-10), delta)
            previous = result["q_command"].copy()
            ack_paced |= bool(result.get("ack_paced", False))
            if result["state"] == "HOME":
                break
        self.assertEqual(result["state"], "HOME", result["reason"])
        self.assertTrue(ack_paced, "Delayed acknowledgment must exercise the command budget")
        self.assertGreater(len(self.publisher.dropped), 0)
        self.assertTrue(all(outcome["accepted"] for outcome in self.publisher.outcomes), self.publisher.outcomes)
        self.assertEqual(self.gate.expired_sessions, set())
        commands = [(time, packet) for time, packet, outcome in self.publisher.deliveries
                    if packet["mode"] == "command" and outcome["accepted"]]
        self.assertGreater(len(commands), 10)
        self.assertTrue(all(packet["motion"] == "home" for _, packet in commands))
        last_drop_end = max(end for _, end in self.publisher.drop_windows)
        self.assertTrue(any(time > last_drop_end for time, _ in commands))
        np.testing.assert_allclose(result["q_command"], self.initial, atol=1e-9)
        self.assertTrue(np.all(np.abs(result["robot_q"] - self.initial) <= self.real.home_tolerance))
        self.assertTrue(np.all(np.abs(self.publisher.q - self.initial) <= self.real.home_tolerance))
        self.assertEqual(self.publisher.messages[-1]["mode"], "stop")
        self.assertEqual(self.gate.state, "STOPPED")
        self.assertIsNone(self.gate.owner_session_id)

    def test_home_receive_jitter_recovers_after_actual_gate_rate_rejection(self):
        # A 50 Hz sender reaches the gate at alternating 30/10 ms intervals.
        # The unchanged robot gate may reject a legitimate 0.2 degree home
        # increment on a 10 ms interval; later targets must recover from its ACK.
        self.receive_offsets_s = (0., .01)
        self.tick_grip(0, frame_available=False)
        self.publisher.q[3] += np.deg2rad(8)
        previous = self.publisher.q.copy()
        self.tick_grip(1, action="home", frame_available=False)
        for index in range(2, 401):
            result = self.tick_grip(index, frame_available=False)
            self.assertIn(result["state"], ("ARMING", "HOMING", "HOME"), result["reason"])
            delta = np.abs(result["q_command"] - previous)
            self.assertTrue(np.all(delta <= self.real.home_speed_limits * .02 + 1e-10), delta)
            previous = result["q_command"].copy()
            if result["state"] == "HOME":
                break
        self.assertEqual(result["state"], "HOME", result["reason"])
        rejected = [(time, packet, outcome) for time, packet, outcome in self.publisher.deliveries
                    if not outcome["accepted"]]
        self.assertGreater(len(rejected), 0, "Jitter must exercise the actual rate gate")
        self.assertEqual({outcome["reason"] for _, _, outcome in rejected}, {"joint_rate_limit"})
        self.assertEqual(self.gate.expired_sessions, set())
        for rejected_at, rejected_packet, _ in rejected:
            recovered = [(time, packet) for time, packet, outcome in self.publisher.deliveries
                         if time > rejected_at and packet["seq"] > rejected_packet["seq"]
                         and packet["mode"] == "command" and outcome["accepted"]]
            if recovered:
                self.assertLess(recovered[0][0] - rejected_at, self.gate.timeout_s)
            else:
                # A final rejected increment may immediately stop only when
                # fresh measured joints have already reached the documented
                # 0.5 degree / 1 mm home tolerance. No lease waits on that ACK.
                stopped_at, stopped_packet, stopped_outcome = self.publisher.deliveries[-1]
                self.assertEqual(rejected_at, stopped_at)
                self.assertEqual(stopped_packet["mode"], "stop")
                self.assertTrue(stopped_outcome["accepted"])
                self.assertTrue(np.all(np.abs(result["robot_q"] - self.initial) <= self.real.home_tolerance))
        np.testing.assert_allclose(result["q_command"], self.initial, atol=1e-9)
        self.assertTrue(np.all(np.abs(result["robot_q"] - self.initial) <= self.real.home_tolerance))
        self.assertTrue(np.all(np.abs(self.publisher.q - self.initial) <= self.real.home_tolerance))
        self.assertEqual(self.publisher.messages[-1]["mode"], "stop")
        self.assertEqual(self.gate.state, "STOPPED")
        self.assertIsNone(self.gate.owner_session_id)

    def test_grip_idle_and_one_hand_never_arm_robot(self):
        for index in range(10):
            grips = (False, False) if index < 5 else (True, False)
            result = self.tick_grip(index, grips=grips)
            self.assertEqual(result["active"], [])
            self.assertFalse(result["armed"])
        self.assertEqual(self.publisher.messages, [])
        self.assertIsNone(self.gate.owner_session_id)
        np.testing.assert_array_equal(self.publisher.q, self.initial)

    def test_initial_both_held_arms_then_reanchors_at_ack(self):
        self.heartbeat_period_steps = 5
        result = self.tick_grip(0, grips=(True, True))
        self.assertEqual(result["state"], "ARMING", result["reason"])
        self.assertEqual(result["active"], [])
        self.assertEqual(self.publisher.messages[-1]["mode"], "arm")
        # Both physical and controller poses change during delayed acknowledgment.
        self.publisher.q[3] += .005
        for index in range(1, 5):
            result = self.tick_grip(index, grips=(True, True), height=.02 * index)
            self.assertEqual(result["state"], "ARMING", result["reason"])
            self.assertEqual(result["active"], [])
            self.assertTrue(all(packet["mode"] == "arm" for packet in self.publisher.messages))
        before = self.publisher.q.copy()
        result = self.tick_grip(5, grips=(True, True), height=.10)
        self.assertTrue(result["armed"], result["reason"])
        self.assertEqual(set(result["active"]), {"left", "right"})
        np.testing.assert_allclose(result["q_command"], before, atol=1e-6)
        for hand, target in result["targets"].items():
            np.testing.assert_allclose(target, get_sg2_wrist_control_T(self.env_measured, hand), atol=1e-10)
        self.assertEqual(self.publisher.messages[-1]["mode"], "command")
        self.assertTrue(all(outcome["accepted"] for outcome in self.publisher.outcomes), self.publisher.outcomes)

    def test_either_grip_release_stops_both_and_regrip_starts_new_session(self):
        for index in range(6):
            result = self.tick_grip(index, grips=(True, True))
        self.assertEqual(set(result["active"]), {"left", "right"})
        next_index = 6
        for released_grips in ((False, True), (True, False)):
            old_session = self.real.session_id
            count = len(self.publisher.messages)
            result = self.tick_grip(next_index, grips=released_grips, height=.1)
            self.assertEqual(result["active"], [])
            self.assertFalse(result["armed"])
            self.assertEqual([packet["mode"] for packet in self.publisher.messages[count:]], ["stop"])
            held = self.publisher.q.copy()
            self.assertIsNone(self.gate.owner_session_id)
            for index in range(next_index + 1, next_index + 4):
                result = self.tick_grip(index, grips=released_grips, height=.15)
                self.assertEqual(result["active"], [])
                np.testing.assert_array_equal(self.publisher.q, held)
            self.assertEqual(len(self.publisher.messages), count + 1)
            for index in range(next_index + 4, next_index + 14):
                result = self.tick_grip(index, grips=(True, True), height=.15)
                if result["armed"]:
                    break
            self.assertEqual(set(result["active"]), {"left", "right"}, result["reason"])
            self.assertNotEqual(self.real.session_id, old_session)
            np.testing.assert_allclose(result["q_command"], held, atol=1e-6)
            next_index = index + 1
        self.assertTrue(all(outcome["accepted"] for outcome in self.publisher.outcomes), self.publisher.outcomes)

    def test_release_while_arming_cannot_activate_on_delayed_ack(self):
        self.heartbeat_period_steps = 5
        first = self.tick_grip(0, grips=(True, True))
        self.assertEqual(first["state"], "ARMING", first["reason"])
        session = self.real.session_id
        result = self.tick_grip(1, grips=(False, True))
        self.assertEqual(result["active"], [])
        self.assertFalse(result["armed"])
        self.assertEqual([packet["mode"] for packet in self.publisher.messages], ["arm", "stop"])
        for index in range(2, 7):
            result = self.tick_grip(index, grips=(False, True))
            self.assertEqual(result["active"], [])
            self.assertFalse(result["armed"])
        self.assertEqual(len(self.publisher.messages), 2)
        for index in range(7, 16):
            result = self.tick_grip(index, grips=(True, True))
            if result["armed"]:
                break
        self.assertEqual(set(result["active"]), {"left", "right"}, result["reason"])
        self.assertNotEqual(session, self.real.session_id)
        np.testing.assert_allclose(result["q_command"], self.initial, atol=1e-6)
        self.assertTrue(all(outcome["accepted"] for outcome in self.publisher.outcomes), self.publisher.outcomes)

    def test_tracking_loss_requires_fresh_both_release_before_held_recovery(self):
        for index in range(6):
            self.tick_grip(index, grips=(True, True))
        count = len(self.publisher.messages)
        result = self.tick_grip(6, grips=(True, True), right_valid=False)
        self.assertEqual(result["active"], [])
        self.assertFalse(result["armed"])
        self.assertEqual([packet["mode"] for packet in self.publisher.messages[count:]], ["stop"])
        held_count = len(self.publisher.messages)
        self.tick_grip(7, grips=(True, True))
        # Mutating buttons on an old frame cannot count as a new release.
        self.tick_grip(8, grips=(False, False), seq=5)
        self.tick_grip(9, grips=(True, True))
        self.tick_grip(10, grips=(False, True))
        self.assertEqual(len(self.publisher.messages), held_count)
        result = self.tick_grip(11, grips=(False, False))
        self.assertEqual(result["active"], [])
        self.assertEqual(len(self.publisher.messages), held_count)
        for index in range(12, 21):
            result = self.tick_grip(index, grips=(True, True))
            if result["armed"]:
                break
        self.assertEqual(set(result["active"]), {"left", "right"}, result["reason"])
        self.assertTrue(all(outcome["accepted"] for outcome in self.publisher.outcomes), self.publisher.outcomes)

    def test_home_without_quest_has_priority_and_held_grips_cannot_restart(self):
        self.tick_grip(0, frame_available=False)
        self.assertEqual(self.publisher.messages, [])
        self.publisher.q[3] += .03
        result = self.tick_grip(1, action="home", frame_available=False)
        self.assertEqual(result["state"], "ARMING", result["reason"])
        for index in range(2, 61):
            result = self.tick_grip(index, grips=(True, True), height=.16)
            self.assertEqual(result["active"], [])
            self.assertNotEqual(result["state"], "FAULT", result["reason"])
            if result["state"] == "HOME":
                break
        self.assertEqual(result["state"], "HOME")
        commands = [packet for packet in self.publisher.messages if packet["mode"] == "command"]
        self.assertGreater(len(commands), 1)
        self.assertTrue(all(packet["motion"] == "home" for packet in commands))
        np.testing.assert_allclose(result["q_command"], self.initial, atol=1e-9)
        count = len(self.publisher.messages)
        for held_index in range(index + 1, index + 6):
            result = self.tick_grip(held_index, grips=(True, True), height=.16)
            self.assertEqual(result["active"], [])
            self.assertFalse(result["armed"])
        self.assertEqual(len(self.publisher.messages), count)
        self.tick_grip(held_index + 1, grips=(False, False))
        for restart_index in range(held_index + 2, held_index + 12):
            result = self.tick_grip(restart_index, grips=(True, True))
            if result["armed"]:
                break
        self.assertEqual(set(result["active"]), {"left", "right"}, result["reason"])
        self.assertTrue(all(outcome["accepted"] for outcome in self.publisher.outcomes), self.publisher.outcomes)


if __name__ == "__main__":
    unittest.main()
