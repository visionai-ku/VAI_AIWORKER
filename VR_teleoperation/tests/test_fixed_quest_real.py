"""Real adapter transitions against actual SG2 FK/IK and in-memory transport."""

import copy
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
runpy.run_path(str(PROJECT_DIR / "package/init_project.py"))

from ri_motion_v5_package.mujoco_sim import MuJoCoParser, BodyPositionIKSolverRevPriBase
from utils import (SG2_REAL_COMMAND_JOINT_NAMES, SG2_GRIPPER_DRIVER_JOINT_NAMES,
                   create_sg2_arm_ik_solvers, get_sg2_wrist_control_T)
from package.controller_mapping import ControllerMapper
from package.fixed_quest_scene import SG2CommandGuard, build_fixed_quest_scene
from package.fixed_quest_sim import (SG2FixedQuestSimulation, SyntheticControllerSource,
                                     reset_sg2_sim, IKTargetUnreachable)
from package.fixed_quest_real import SG2FixedQuestReal
from package.intent_sim_bridge import IntentSimBridge
from package.worker.fixed_quest_protocol import parse_joint_groups


class MemoryPublisher:
    """No sockets: capture exactly what would reach the command bridge."""

    def __init__(self, enabled):
        self.enabled = enabled
        self.messages = []
        self.closed = False
        self.close_count = 0

    def send(self, message):
        if self.closed:
            raise RuntimeError("publisher closed")
        self.messages.append(copy.deepcopy(message))
        return True

    def close(self):
        self.closed = True
        self.close_count += 1


class FixedQuestRealTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (PROJECT_DIR / ".cache").mkdir(exist_ok=True)
        cls.cache = tempfile.TemporaryDirectory(dir=PROJECT_DIR / ".cache")
        xml = build_fixed_quest_scene(
            PROJECT_DIR / "xml/ffw_sg2.xml", PROJECT_DIR / "xml/floor_simple_white.xml",
            Path(cls.cache.name) / "scene.xml")
        cls.env, cls.env_ik, cls.env_check, cls.env_measured = [
            MuJoCoParser(rel_xml_path=xml, verbose=False) for _ in range(4)]
        cls.names = list(SG2_REAL_COMMAND_JOINT_NAMES) + list(SG2_GRIPPER_DRIVER_JOINT_NAMES.values())
        for env in (cls.env, cls.env_ik, cls.env_check, cls.env_measured):
            reset_sg2_sim(env)
        cls.solvers, cls.arms = create_sg2_arm_ik_solvers(
            cls.env_ik, MuJoCoParser, BodyPositionIKSolverRevPriBase)
        cls.guard = SG2CommandGuard(cls.env_check, cls.names)
        groups = {
            "lift": ["lift_joint"], "head": ["head_joint1", "head_joint2"],
            "arm_l": ["arm_l_joint%d" % index for index in range(1, 8)],
            "arm_r": ["arm_r_joint%d" % index for index in range(1, 8)],
            "gripper_l": ["gripper_l_joint1"], "gripper_r": ["gripper_r_joint1"],
        }
        cls.layout = {
            "use_joint_names": cls.names, "q_msg_keys": list(groups),
            "qmsgkeys_to_usejointnames_dict": groups,
            "qmsgkeys_to_usejointidxs_dict": {
                group: [cls.names.index(name) for name in names] for group, names in groups.items()},
        }

    @classmethod
    def tearDownClass(cls):
        cls.cache.cleanup()

    def setUp(self):
        for env in (self.env, self.env_ik, self.env_check, self.env_measured):
            reset_sg2_sim(env)
        self.sim_initial = self.env.get_qpos(joint_names=self.names).copy()
        self.robot_initial = self.sim_initial.copy()
        self.robot_initial[self.names.index("arm_l_joint1")] = .04
        self.robot_initial[self.names.index("lift_joint")] = -.10
        self.robot_initial[self.names.index("head_joint1")] = .05
        self.planner = SG2FixedQuestSimulation(
            self.env, self.env_ik, self.solvers, self.arms, self.names,
            ControllerMapper(), self.guard, mode="kinematic", max_joint_speed=np.deg2rad(5))
        self.publisher = None
        self.real = None
        self.generation = 0
        self.created_publishers = []

    def publisher_factory(self, *, enabled):
        publisher = MemoryPublisher(enabled=enabled)
        self.created_publishers.append(publisher)
        return publisher

    def make_real(self, dry_run=False, **options):
        self.publisher = MemoryPublisher(enabled=not dry_run)
        self.real = SG2FixedQuestReal(self.planner, self.env_measured, self.layout,
                                    self.publisher, dry_run=dry_run, **options)
        return self.real

    def test_official_arm_ramp_and_home_limits(self):
        self.configure_teleop_speed(np.rad2deg(4.8))
        real = self.make_real(max_arm_joint_lag=.48, teleop_ramp_s=8.)
        for now, scale in ((10., 0.), (14., .5), (18., 1.), (20., 1.)):
            real._set_motion_speed(False, now=now)
            np.testing.assert_allclose(self.planner.speed_limits[real.arm_mask], 4.8 * scale)
            self.assertAlmostEqual(real.teleop_ramp_scale, scale)
        real._set_motion_speed(True, now=20.)
        np.testing.assert_allclose(self.planner.speed_limits[real.arm_mask], np.deg2rad(10))
        np.testing.assert_allclose(real.lag_limits[real.arm_mask], np.deg2rad(5))
        real.stop()
        real._set_motion_speed(False, now=30.)
        self.assertEqual(real.teleop_ramp_scale, 0.)

    def test_official_lead_allows_tracking_past_old_four_degree_bottleneck(self):
        self.configure_teleop_speed(np.rad2deg(4.8))
        real = self.make_real(max_arm_joint_lag=.48)
        previous = self.robot_initial.copy(); previous[3] += .15
        candidate = previous.copy(); candidate[3] += .048
        real._ack_travel = np.zeros(len(self.names))
        paced, info = real._pace_candidate(previous, candidate, self.robot_initial)
        np.testing.assert_allclose(paced, candidate)
        self.assertFalse(info["tracking_paced"])
        previous[3] = self.robot_initial[3] + .38
        candidate = previous.copy(); candidate[3] += .048
        paced, info = real._pace_candidate(previous, candidate, self.robot_initial)
        self.assertTrue(info["tracking_paced"])
        self.assertAlmostEqual(paced[3] - self.robot_initial[3], .48 * .8)

    def test_old_worker_profile_prevents_any_movement(self):
        real = self.make_real(required_bridge_limits={"teleop_arm_joint_speed_rad_s": 4.8})
        result = real.tick_grip(self.frame(0, grip=True), self.feedback(0.), now=0.)
        self.assertEqual(result["state"], "FAULT", result["reason"])
        self.assertIn("worker speed profile mismatch", result["reason"])
        self.assertEqual(self.publisher.messages, [])

    def configure_teleop_speed(self, degrees_per_s):
        self.planner.speed_limits = np.asarray([
            .01 if name == "lift_joint" else np.deg2rad(degrees_per_s if name.startswith("arm_") else min(60, degrees_per_s)) for name in self.names])

    def feedback(self, now, q=None, ready=True, bridge_state="DISARMED", owner=None):
        return {
            "ready": ready, "q": self.robot_initial.copy() if q is None else np.array(q, dtype=float),
            "generation": self.generation, "bridge_id": "test-bridge",
            "bridge_state": bridge_state, "owner_session_id": owner,
            "last_command_seq": -1, "state": {"source_time_s": now},
        }

    @staticmethod
    def frame(index, grip=False, height=0.):
        now = index / 100
        frame = SyntheticControllerSource().latest(now)
        frame.update(seq=index, sample_time_ms=now * 1000, received_at=now)
        for hand in frame["hands"].values():
            hand.update(grip=float(grip), trigger=0.)
        frame["hands"]["left"]["position_m"][1] += height
        return frame

    def armed_feedback(self, now, q=None):
        feedback = self.feedback(now, q=q, bridge_state="ACTIVE", owner=self.real.session_id)
        feedback["last_command_seq"] = self.real.sequence - 1
        return feedback

    def arm(self):
        self.make_real()
        self.real.tick(self.frame(0), self.feedback(0.), enabled=False, now=0.)
        first = self.real.tick(self.frame(1), self.feedback(.01), enabled=True, now=.01)
        self.assertEqual(first["state"], "ARMING")
        self.assertEqual([message["mode"] for message in self.publisher.messages], ["arm"])
        ready = self.real.tick(self.frame(2), self.armed_feedback(.02), enabled=True, now=.02)
        self.assertEqual(ready["state"], "READY", ready["reason"])
        self.assertTrue(ready["armed"])
        self.assertEqual(self.publisher.messages[-1]["mode"], "command")
        return ready

    def start_grip(self, dry_run=False):
        self.make_real(dry_run=dry_run)
        result = self.real.tick_grip(self.frame(0, grip=True), self.feedback(0.), now=0.)
        if not dry_run:
            self.assertEqual(result["state"], "ARMING")
            result = self.real.tick_grip(self.frame(1, grip=True), self.armed_feedback(.01), now=.01)
        self.assertEqual(result["state"], "ACTIVE", result["reason"])
        self.assertEqual(result["active"], ["left", "right"])
        return result

    def test_grip_startup_waits_without_arm_then_initial_held_starts_at_actual_pose(self):
        for dry_run in (False, True):
            with self.subTest(dry_run=dry_run):
                real = self.make_real(dry_run=dry_run)
                self.assertEqual(real.tick_grip(None, self.feedback(0., ready=False), now=0.)["state"], "WAIT_ROBOT")
                self.assertEqual(real.tick_grip(None, self.feedback(.01), now=.01)["state"], "WAIT_INPUT")
                one_hand = self.frame(2, grip=True)
                one_hand["hands"]["right"]["grip"] = 0.
                idle = real.tick_grip(one_hand, self.feedback(.02), now=.02)
                self.assertEqual(idle["state"], "WAIT_GRIP")
                self.assertEqual(self.publisher.messages, [])
                result = real.tick_grip(self.frame(3, grip=True, height=.1), self.feedback(.03), now=.03)
                self.assertEqual(result["state"], "ACTIVE" if dry_run else "ARMING")
                np.testing.assert_array_equal(result["q_command"], self.robot_initial)
                if not dry_run:
                    self.assertEqual(result["targets"], {})
                    measured = self.robot_initial.copy()
                    measured[self.names.index("arm_l_joint1")] += .01
                    result = real.tick_grip(self.frame(4, grip=True, height=.15),
                                            self.armed_feedback(.04, q=measured), now=.04)
                    self.assertEqual(result["state"], "ACTIVE")
                    np.testing.assert_array_equal(result["q_command"], measured)
                else:
                    self.assertEqual(self.publisher.messages, [])
                for hand in ("left", "right"):
                    np.testing.assert_allclose(result["targets"][hand],
                                               get_sg2_wrist_control_T(self.env_measured, hand), atol=1e-12)

    def test_tracking_pause_holds_command_and_resumes_existing_clutch(self):
        self.start_grip()
        self.real.tracking_pause_s = .1
        session = self.real.session_id
        command = self.planner.q_command.copy()
        anchors = copy.deepcopy(self.planner.mapper._anchors)
        for i in range(2, 11):
            bad = self.frame(i, grip=True)
            bad["hands"]["left"].update(valid=False, emulated=True, position_m=[10., 10., 10.])
            result = self.real.tick_grip(bad, self.armed_feedback(i/100), now=i/100)
            self.assertEqual(result["state"], "TRACKING_PAUSED", result["reason"])
            self.assertEqual(result["targets"], {})
            np.testing.assert_array_equal(result["q_command"], command)
            np.testing.assert_array_equal(parse_joint_groups(self.publisher.messages[-1]["joints"]), command)
            self.assertEqual(self.real.session_id, session)
        restored = self.real.tick_grip(self.frame(11, grip=True), self.armed_feedback(.11), now=.11)
        self.assertEqual(restored["state"], "ACTIVE", restored["reason"])
        self.assertEqual(self.real.session_id, session)
        for hand in ("left", "right"):
            np.testing.assert_array_equal(self.planner.mapper._anchors[hand]["eef_T"], anchors[hand]["eef_T"])

    def test_ik_wait_keeps_strict_bridge_watchdog_session_and_ramp_alive(self):
        self.planner.ik_recovery_timeout_s = 1.
        self.planner.ik_period = .02
        self.configure_teleop_speed(60.)
        bridge = IntentSimBridge(self.robot_initial)
        real = SG2FixedQuestReal(self.planner, self.env_measured, self.layout, bridge,
                                dry_run=False, teleop_ramp_s=8., command_arm_speed_limit=True)
        first = real.tick_grip(self.frame(0, grip=True), bridge.latest(0.), now=0.)
        self.assertEqual(first["state"], "ARMING", first["reason"])
        active = real.tick_grip(self.frame(1, grip=True), bridge.latest(.01), now=.01)
        self.assertEqual(active["state"], "ACTIVE", active["reason"])
        real.tick_grip(self.frame(2, grip=True), bridge.latest(.02), now=.02)
        session, ramp_start = real.session_id, real.teleop_started_at
        held = self.planner.q_command.copy()
        anchors = copy.deepcopy(self.planner.mapper._anchors)
        previous_ramp = real.teleop_ramp_scale
        with patch.object(self.planner, "_solve", side_effect=IKTargetUnreachable("left unavailable")):
            # Twice the robot's 200 ms lease: every held command must be accepted.
            for index in range(3, 43):
                now = index / 100
                frame = self.frame(index, grip=True)
                for hand in frame["hands"].values():
                    hand["trigger"] = 1.
                result = real.tick_grip(frame, bridge.latest(now), now=now)
                self.assertEqual(result["state"], "IK_WAIT", result["reason"])
                self.assertTrue(result["transmitted"])
                self.assertFalse(result["needs_disable"])
                self.assertFalse(result["grip_rearm_required"])
                self.assertTrue(bridge.last_result["accepted"], bridge.last_result)
                self.assertEqual(bridge.gate.state, "ACTIVE")
                self.assertEqual(real.session_id, session)
                self.assertEqual(real.teleop_started_at, ramp_start)
                self.assertGreaterEqual(result["teleop_ramp_scale"], previous_ramp)
                previous_ramp = result["teleop_ramp_scale"]
                np.testing.assert_array_equal(result["q_command"], held)
                np.testing.assert_array_equal(bridge.q, held)
        recovered = real.tick_grip(self.frame(43, grip=True), bridge.latest(.43), now=.43)
        self.assertEqual(recovered["state"], "ACTIVE", recovered["reason"])
        self.assertEqual(recovered["ik_recovery"]["status"], "recovered")
        self.assertTrue(bridge.last_result["accepted"], bridge.last_result)
        self.assertEqual(real.session_id, session)
        self.assertEqual(real.teleop_started_at, ramp_start)
        for hand in ("left", "right"):
            np.testing.assert_array_equal(self.planner.mapper._anchors[hand]["eef_T"], anchors[hand]["eef_T"])

    def test_ik_wait_stops_immediately_on_either_grip_release_or_stale_feedback(self):
        for stop in ("left_grip", "right_grip", "stale_feedback"):
            with self.subTest(stop=stop):
                self.setUp()
                self.planner.ik_recovery_timeout_s = 1.
                self.planner.ik_period = .02
                self.start_grip()
                held = self.planner.q_command.copy()
                with patch.object(self.planner, "_solve", side_effect=IKTargetUnreachable("left unavailable")):
                    waiting = self.real.tick_grip(self.frame(3, grip=True), self.armed_feedback(.03), now=.03)
                    self.assertEqual(waiting["state"], "IK_WAIT", waiting["reason"])
                    frame, feedback = self.frame(4, grip=True), self.armed_feedback(.04)
                    if stop == "stale_feedback":
                        feedback.update(ready=False, reason="stale_joint_state")
                    else:
                        frame["hands"][stop.split("_")[0]]["grip"] = 0.
                    stopped = self.real.tick_grip(frame, feedback, now=.04)
                self.assertNotIn(stopped["state"], ("IK_WAIT", "ACTIVE"))
                self.assertFalse(stopped["armed"])
                self.assertEqual(stopped["active"], [])
                self.assertEqual(stopped["ik_recovery"]["status"], "cancelled")
                self.assertGreaterEqual(stopped["ik_recovery"]["elapsed_s"], .01 - 1e-9)
                self.assertEqual(self.publisher.messages[-1]["mode"], "stop")
                np.testing.assert_array_equal(stopped["q_command"], held)
                if stop == "stale_feedback":
                    self.assertEqual(stopped["state"], "WAIT_ROBOT")
                    self.assertEqual(stopped["reason"], "stale_joint_state")

    def test_home_preempts_ik_wait_without_quest_and_finish_or_cancel_requires_release(self):
        for finish in (True, False):
            with self.subTest(finish=finish):
                self.setUp()
                self.planner.ik_recovery_timeout_s = 1.
                self.planner.ik_period = .02
                self.start_grip()
                with patch.object(self.planner, "_solve", side_effect=IKTargetUnreachable("left unavailable")) as solve:
                    waiting = self.real.tick_grip(self.frame(3, grip=True), self.armed_feedback(.03), now=.03)
                    self.assertEqual(waiting["state"], "IK_WAIT", waiting["reason"])
                    home = self.real.tick_grip(None, self.armed_feedback(.04), now=.04, action="home")
                    self.assertEqual(home["state"], "ARMING", home["reason"])
                    self.assertTrue(home["home_pending"])
                    self.assertTrue(home["home_plan"]["ok"])
                    self.assertEqual(self.planner.ik_recovery_info(.04)["status"], "idle")
                    arming = self.real.tick_grip(None, self.feedback(.045), now=.045)
                    self.assertEqual(arming["state"], "ARMING", arming["reason"])
                    result = self.real.tick_grip(None, self.armed_feedback(.05), now=.05,
                                                 action=None if finish else "cancel_home")
                    end_index = 5
                    if finish:
                        self.assertEqual(result["state"], "HOMING", result["reason"])
                        result = self.real.tick_grip(None, self.armed_feedback(.06), now=.06)
                        end_index = 6
                    self.assertEqual(result["state"], "HOME" if finish else "HOLD", result["reason"])
                    self.assertEqual(solve.call_count, 1, "HOME must not retry the controller IK goal")
                self.assertFalse(result["homing"])
                self.assertFalse(result["home_pending"])
                self.assertFalse(result["armed"])
                self.assertTrue(result["grip_rearm_required"])
                for index in range(end_index + 1, end_index + 4):
                    held = self.real.tick_grip(self.frame(index, grip=True), self.feedback(index / 100), now=index / 100)
                    self.assertEqual(held["state"], "WAIT_RELEASE")
                    self.assertEqual(held["active"], [])
                index = end_index + 4
                self.real.tick_grip(self.frame(index, grip=False), self.feedback(index / 100), now=index / 100)
                index += 1
                rearmed = self.real.tick_grip(self.frame(index, grip=True), self.feedback(index / 100), now=index / 100)
                self.assertEqual(rearmed["state"], "ARMING", rearmed["reason"])

    def test_ik_wait_timeout_survives_tracking_pause_and_real_hold_diagnostics(self):
        self.planner.ik_recovery_timeout_s = .1
        self.planner.ik_period = .02
        self.start_grip()
        self.real.tracking_pause_s = .1
        held = self.planner.q_command.copy()
        with patch.object(self.planner, "_solve", side_effect=IKTargetUnreachable("right unavailable")):
            waiting = self.real.tick_grip(self.frame(3, grip=True), self.armed_feedback(.03), now=.03)
            self.assertEqual(waiting["state"], "IK_WAIT", waiting["reason"])
            for index in range(4, 11):
                frame = self.frame(index, grip=True)
                frame["hands"]["left"].update(valid=False, emulated=True)
                paused = self.real.tick_grip(frame, self.armed_feedback(index / 100), now=index / 100)
                self.assertEqual(paused["state"], "TRACKING_PAUSED", paused["reason"])
            for index in (11, 12):
                resumed = self.real.tick_grip(self.frame(index, grip=True),
                                             self.armed_feedback(index / 100), now=index / 100)
                self.assertEqual(resumed["state"], "IK_WAIT", resumed["reason"])
                self.assertGreaterEqual(resumed["ik_recovery"]["elapsed_s"], .08 - 1e-9)
            timeout = self.real.tick_grip(self.frame(13, grip=True), self.armed_feedback(.13), now=.13)
        self.assertEqual(timeout["state"], "HOLD", timeout["reason"])
        self.assertEqual(timeout["ik_recovery"]["status"], "timeout")
        self.assertGreaterEqual(timeout["ik_recovery"]["elapsed_s"], .1 - 1e-9)
        self.assertIn("right unavailable", timeout["ik_recovery"]["reason"])
        self.assertTrue(timeout["grip_rearm_required"])
        self.assertFalse(timeout["armed"])
        self.assertEqual(self.publisher.messages[-1]["mode"], "stop")
        np.testing.assert_array_equal(timeout["q_command"], held)
        still_held = self.real.tick_grip(self.frame(14, grip=True), self.feedback(.14), now=.14)
        self.assertEqual(still_held["state"], "WAIT_RELEASE")

    def test_tracking_pause_timeout_requires_release(self):
        self.start_grip()
        self.real.tracking_pause_s = .1
        for i in range(2, 13):
            bad = self.frame(i, grip=True)
            bad["hands"]["left"].update(valid=False, emulated=True)
            result = self.real.tick_grip(bad, self.armed_feedback(i/100), now=i/100)
        self.assertEqual(result["state"], "HOLD")
        self.assertIn("pause exceeded", result["reason"])
        self.assertEqual(self.publisher.messages[-1]["mode"], "stop")
        valid = self.real.tick_grip(self.frame(13, grip=True), self.feedback(.13), now=.13)
        self.assertEqual(valid["state"], "WAIT_RELEASE")

    def test_tracking_pause_duplicate_cannot_change_buttons_or_extend_deadline(self):
        self.start_grip()
        self.real.tracking_pause_s = .1
        bad = self.frame(2, grip=True)
        bad["hands"]["left"].update(valid=False, emulated=True)
        self.real.tick_grip(bad, self.armed_feedback(.02), now=.02)
        duplicate = copy.deepcopy(bad)
        duplicate["hands"]["right"]["grip"] = 0.
        for i in range(3, 12):
            duplicate["received_at"] = i/100
            result = self.real.tick_grip(duplicate, self.armed_feedback(i/100), now=i/100)
            self.assertEqual(result["state"], "TRACKING_PAUSED")
        result = self.real.tick_grip(duplicate, self.armed_feedback(.12), now=.12)
        self.assertEqual(result["state"], "HOLD")
        self.assertIn("pause exceeded", result["reason"])

    def test_tracking_pause_release_and_feedback_failure_still_stop(self):
        for cause in ("release", "feedback"):
            with self.subTest(cause=cause):
                self.start_grip()
                self.real.tracking_pause_s = .1
                bad = self.frame(2, grip=True)
                bad["hands"]["left"].update(valid=False, emulated=True)
                self.real.tick_grip(bad, self.armed_feedback(.02), now=.02)
                bad = self.frame(3, grip=True)
                bad["hands"]["left"].update(valid=False, emulated=True)
                feedback = self.armed_feedback(.03)
                if cause == "release":bad["hands"]["right"]["grip"] = 0.
                else:feedback.update(ready=False, reason="stale_joint_state")
                result = self.real.tick_grip(bad, feedback, now=.03)
                self.assertIn(result["state"], ("HOLD", "WAIT_ROBOT"))
                self.assertEqual(self.publisher.messages[-1]["mode"], "stop")
                self.assertTrue(result["grip_rearm_required"])

    def test_tracking_pause_does_not_accept_emulated_start_or_reference_change(self):
        real = self.make_real(tracking_pause_s=.1)
        bad = self.frame(0, grip=True)
        bad["hands"]["left"].update(valid=False, emulated=True)
        result = real.tick_grip(bad, self.feedback(0.), now=0.)
        self.assertEqual(result["state"], "WAIT_INPUT")
        self.assertEqual(self.publisher.messages, [])
        self.start_grip()
        self.real.tracking_pause_s = .1
        changed = self.frame(2, grip=True)
        changed["origin_id"] += 1
        result = self.real.tick_grip(changed, self.armed_feedback(.02), now=.02)
        self.assertEqual(result["state"], "HOLD")
        self.assertEqual(self.publisher.messages[-1]["mode"], "stop")

    def test_selected_long_tracking_pause_keeps_command_lease_then_holds(self):
        for bad in (True, "0.5", -.01, 1.01, float("nan")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.make_real(tracking_pause_s=bad)
        self.start_grip()
        self.real.tracking_pause_s = .5
        session = self.real.session_id
        command = self.planner.q_command.copy()
        for i in range(2, 52):
            bad = self.frame(i, grip=True)
            bad["hands"]["left"].update(valid=False, emulated=True)
            result = self.real.tick_grip(bad, self.armed_feedback(i/100), now=i/100)
            self.assertEqual(result["state"], "TRACKING_PAUSED", (i, result["reason"]))
            self.assertEqual(self.publisher.messages[-1]["mode"], "command")
            np.testing.assert_array_equal(result["q_command"], command)
        self.assertEqual(self.real.session_id, session)
        bad = self.frame(52, grip=True)
        bad["hands"]["left"].update(valid=False, emulated=True)
        result = self.real.tick_grip(bad, self.armed_feedback(.52), now=.52)
        self.assertEqual(result["state"], "HOLD")
        self.assertEqual(result["reason"], "controller tracking pause exceeded 500 ms")
        self.assertEqual(self.publisher.messages[-1]["mode"], "stop")

    def test_tracking_loss_mode_switches_both_readers_only_between_clutches(self):
        real = self.make_real(tracking_pause_s=.1)
        with self.assertRaises(ValueError):
            real.set_tracking_loss_mode(.5, False, now=0.)
        real.tick_grip(self.frame(0), self.feedback(0.), now=0.)
        for pause, emulated in ((True, False), (1.01, False), (.5, 1), (.5, None)):
            with self.subTest(pause=pause, emulated=emulated), self.assertRaises(ValueError):
                real.set_tracking_loss_mode(pause, emulated, now=0.)
        self.assertEqual(real.tracking_loss_mode, {"tracking_pause_s": .1, "accept_emulated": False})
        self.assertEqual(real.set_tracking_loss_mode(1., True, now=0.),
                         {"tracking_pause_s": 1., "accept_emulated": True})
        self.assertTrue(self.planner.mapper.accept_emulated)
        self.assertTrue(real._grip_reader.accept_emulated)
        result = real.tick_grip(self.frame(1, grip=True), self.feedback(.01), now=.01)
        self.assertEqual(result["state"], "ARMING")
        self.assertEqual(result["tracking_loss_mode"], {"tracking_pause_s": 1., "accept_emulated": True})
        with self.assertRaises(ValueError):
            real.set_tracking_loss_mode(.1, False, now=.01)
        self.assertEqual(real.tracking_loss_mode, {"tracking_pause_s": 1., "accept_emulated": True})

    def test_emulated_mode_follows_estimated_pose_in_real_and_still_stops_on_jump(self):
        real = self.make_real()
        real.tick_grip(self.frame(0), self.feedback(0.), now=0.)
        real.set_tracking_loss_mode(.1, True, now=0.)
        estimated = self.frame(1, grip=True)
        estimated["hands"]["left"].update(valid=False, emulated=True)
        self.assertEqual(real.tick_grip(estimated, self.feedback(.01), now=.01)["state"], "ARMING")
        result = real.tick_grip(self.frame(2, grip=True), self.armed_feedback(.02), now=.02)
        self.assertEqual(result["state"], "ACTIVE", result["reason"])
        session = real.session_id
        for i in range(3, 30):
            estimated = self.frame(i, grip=True, height=.001 * (i - 2))
            estimated["hands"]["left"].update(valid=False, emulated=True)
            result = real.tick_grip(estimated, self.armed_feedback(i/100), now=i/100)
            self.assertEqual(result["state"], "ACTIVE", (i, result["reason"]))
            self.assertEqual(self.publisher.messages[-1]["mode"], "command")
        self.assertEqual(real.session_id, session)
        self.assertTrue(result["mapping_stages"]["left"]["input_emulated"])
        self.assertFalse(result["mapping_stages"]["right"]["input_emulated"])
        jump = self.frame(30, grip=True, height=.25)
        jump["hands"]["left"].update(valid=False, emulated=True)
        result = real.tick_grip(jump, self.armed_feedback(.30), now=.30)
        self.assertEqual(result["state"], "HOLD")
        self.assertEqual(result["reason"], "pose_jump_left")
        self.assertEqual(self.publisher.messages[-1]["mode"], "stop")

    def test_manual_position_gains_idle_preserve_home_speed_and_comfortable_anchor(self):
        real = self.make_real(dry_run=True)
        real.tick_grip(self.frame(0), self.feedback(0.), now=0.)
        home = self.planner.q_home.copy()
        speeds = self.planner.speed_limits.copy()
        scales = real.set_position_gains(.8, .6, now=0.)
        for hand in ("left", "right"):
            np.testing.assert_array_equal(scales[hand], [.5, .8, .6])
        np.testing.assert_array_equal(self.planner.q_home, home)
        np.testing.assert_array_equal(self.planner.speed_limits, speeds)
        result = real.tick_grip(self.frame(1, grip=True, height=.15), self.feedback(.01), now=.01)
        self.assertEqual(result["state"], "ACTIVE")
        for hand in ("left", "right"):
            np.testing.assert_allclose(result["targets"][hand],
                                       get_sg2_wrist_control_T(self.env_measured, hand), atol=1e-12)
        self.assertEqual(self.publisher.messages, [])
        with self.assertRaises(ValueError):
            real.set_position_gains(1., 1., now=.01)
        for gains in self.planner.mapper.position_scales.values():
            np.testing.assert_array_equal(gains, [.5, .8, .6])

    def test_manual_position_gains_require_both_released_and_recent_valid_input(self):
        real = self.make_real()
        with self.assertRaises(ValueError):
            real.set_position_gains(.8, .6, now=0.)
        one_grip = self.frame(0)
        one_grip["hands"]["left"]["grip"] = 1.
        real.tick_grip(one_grip, self.feedback(0.), now=0.)
        with self.assertRaises(ValueError):
            real.set_position_gains(.8, .6, now=0.)
        real.tick_grip(self.frame(1), self.feedback(.01), now=.01)
        for value in (True, "1", -3.1, 3.1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                real.set_position_gains(.8, value, now=.01)
        with self.assertRaises(ValueError):
            real.set_position_gains(.8, .6, now=.5)
        for gains in self.planner.mapper.position_scales.values():
            np.testing.assert_array_equal(gains, [.5, .5, .5])
        self.assertEqual(self.publisher.messages, [])

    def test_manual_xyz_signed_z_validates_atomically_and_preserves_home(self):
        real = self.make_real(dry_run=True)
        real.tick_grip(self.frame(0), self.feedback(0.), now=0.)
        home, command = self.planner.q_home.copy(), self.planner.q_command.copy()
        speeds = self.planner.speed_limits.copy()
        for z in (-3., -.25, 0., 3.):
            result = real.set_position_gains(.8, z, now=0., forward=2.)
            for hand in ("left", "right"):
                np.testing.assert_array_equal(result[hand], [2., .8, z])
        for bad in (True, "1", 0., -.5, 3.1, float("nan"), float("inf")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                real.set_position_gains(1., -.5, now=0., forward=bad)
        for hand in ("left", "right"):
            np.testing.assert_array_equal(self.planner.mapper.position_scales[hand], [2., .8, 3.])
        np.testing.assert_array_equal(self.planner.q_home, home)
        np.testing.assert_array_equal(self.planner.q_command, command)
        np.testing.assert_array_equal(self.planner.speed_limits, speeds)
        self.assertEqual(self.publisher.messages, [])

    def test_manual_position_gains_after_release_send_no_motion_and_reject_home(self):
        self.start_grip()
        real = self.real
        real.tick_grip(self.frame(2), self.armed_feedback(.02), now=.02)
        count = len(self.publisher.messages)
        real.set_position_gains(.7, .4, now=.02)
        self.assertEqual(len(self.publisher.messages), count)
        self.assertEqual(self.publisher.messages[-1]["mode"], "stop")
        real.tick_grip(self.frame(3), self.feedback(.03), now=.03, action="home")
        with self.assertRaises(ValueError):
            real.set_position_gains(.8, .6, now=.03)

    def test_either_grip_release_stops_both_and_fresh_regrip_starts_new_lease(self):
        for released_hand in ("left", "right"):
            with self.subTest(released_hand=released_hand):
                self.start_grip()
                previous_session, count = self.real.session_id, len(self.publisher.messages)
                released = self.frame(2, grip=True)
                released["hands"][released_hand]["grip"] = 0.
                result = self.real.tick_grip(released, self.armed_feedback(.02), now=.02)
                self.assertEqual(result["state"], "WAIT_GRIP")
                self.assertEqual(result["active"], [])
                self.assertEqual(result["targets"], {})
                self.assertIsNone(self.real.session_id)
                self.assertEqual([m["mode"] for m in self.publisher.messages[count:]], ["stop"])
                count = len(self.publisher.messages)
                self.real.tick_grip(released, self.feedback(.025), now=.025)
                self.assertEqual(len(self.publisher.messages), count)
                restarted = self.real.tick_grip(self.frame(3, grip=True, height=.1), self.feedback(.03), now=.03)
                self.assertEqual(restarted["state"], "ARMING")
                self.assertNotEqual(self.real.session_id, previous_session)
                activated = self.real.tick_grip(self.frame(4, grip=True, height=.12), self.armed_feedback(.04), now=.04)
                self.assertEqual(activated["active"], ["left", "right"])
                np.testing.assert_allclose(activated["targets"]["left"],
                                           get_sg2_wrist_control_T(self.env_measured, "left"), atol=1e-12)

    def test_grip_release_during_arming_cancels_before_any_command(self):
        real = self.make_real()
        real.tick_grip(self.frame(0, grip=True), self.feedback(0.), now=0.)
        released = self.frame(1, grip=True)
        released["hands"]["left"]["grip"] = 0.
        result = real.tick_grip(released, self.armed_feedback(.01), now=.01)
        self.assertEqual(result["state"], "WAIT_GRIP")
        self.assertEqual([m["mode"] for m in self.publisher.messages], ["arm", "stop"])
        self.assertEqual(result["targets"], {})

    def test_grip_faults_require_new_both_release_then_new_press(self):
        for fault in ("tracking", "feedback", "guard", "reference", "robot_identity"):
            with self.subTest(fault=fault):
                self.start_grip()
                frame, feedback = self.frame(2, grip=True), self.armed_feedback(.02)
                if fault == "tracking":
                    frame["hands"]["left"].update(valid=False, emulated=True)
                elif fault == "feedback":
                    feedback["ready"] = False
                elif fault == "reference":
                    frame["origin_id"] += 1
                elif fault == "robot_identity":
                    self.generation += 1
                    feedback["generation"] = self.generation
                if fault == "guard":
                    with patch.object(self.guard, "check", return_value={"accepted": False, "reason": "test guard fault"}):
                        failed = self.real.tick_grip(frame, feedback, now=.02)
                else:
                    failed = self.real.tick_grip(frame, feedback, now=.02)
                self.assertTrue(failed["grip_rearm_required"])
                self.assertFalse(failed["armed"])
                self.assertEqual(self.publisher.messages[-1]["mode"], "stop")
                count = len(self.publisher.messages)
                held = self.frame(3, grip=True)
                if fault == "reference":
                    held["origin_id"] = frame["origin_id"]
                waiting = self.real.tick_grip(held, self.feedback(.03), now=.03)
                self.assertEqual(waiting["state"], "WAIT_RELEASE")
                self.assertEqual(len(self.publisher.messages), count)
                release = self.frame(4)
                release["origin_id"] = held["origin_id"]
                self.assertEqual(self.real.tick_grip(release, self.feedback(.04), now=.04)["state"], "WAIT_GRIP")
                duplicate = copy.deepcopy(release)
                for hand in duplicate["hands"].values():
                    hand["grip"] = 1.
                self.assertEqual(self.real.tick_grip(duplicate, self.feedback(.045), now=.045)["state"], "WAIT_GRIP")
                next_press = self.frame(5, grip=True)
                next_press["origin_id"] = held["origin_id"]
                self.assertEqual(self.real.tick_grip(next_press, self.feedback(.05), now=.05)["state"], "ARMING")

    def test_delayed_release_from_before_fault_does_not_authorize_restart(self):
        self.start_grip()
        self.real.tick_grip(None, self.armed_feedback(.02), now=.02)
        delayed = self.frame(3)
        delayed["sample_time_ms"] = 15.
        result = self.real.tick_grip(delayed, self.feedback(.03), now=.03)
        self.assertEqual(result["state"], "WAIT_RELEASE")
        self.assertEqual(self.real.tick_grip(self.frame(4, grip=True), self.feedback(.04), now=.04)["state"], "WAIT_RELEASE")
        self.assertEqual(self.real.tick_grip(self.frame(5), self.feedback(.05), now=.05)["state"], "WAIT_GRIP")
        self.assertEqual(self.real.tick_grip(self.frame(6, grip=True), self.feedback(.06), now=.06)["state"], "ARMING")

    def test_duplicate_ack_packet_cannot_change_activation_pose_or_grips(self):
        real = self.make_real()
        original = self.frame(0, grip=True)
        real.tick_grip(original, self.feedback(0.), now=0.)
        altered = copy.deepcopy(original)
        altered["received_at"] = .01
        altered["hands"]["left"]["position_m"][1] += .15
        altered["hands"]["left"]["grip"] = 0.
        altered["head"]["quaternion_xyzw"] = [0., 1., 0., 0.]
        result = real.tick_grip(altered, self.armed_feedback(.01), now=.01)
        self.assertEqual(result["state"], "ACTIVE")
        anchored = {hand: pose.copy() for hand, pose in result["targets"].items()}
        result = real.tick_grip(self.frame(2, grip=True), self.armed_feedback(.02), now=.02)
        self.assertEqual(result["state"], "ACTIVE")
        for hand in ("left", "right"):
            np.testing.assert_allclose(result["targets"][hand], anchored[hand], atol=1e-12)
        np.testing.assert_allclose(result["q_command"], self.robot_initial, atol=1e-9)

    def test_stale_and_out_of_order_grips_do_not_start_or_release_rearm(self):
        real = self.make_real()
        released = self.frame(0)
        real.tick_grip(released, self.feedback(0.), now=0.)
        changed_duplicate = copy.deepcopy(released)
        for hand in changed_duplicate["hands"].values():
            hand["grip"] = 1.
        self.assertEqual(real.tick_grip(changed_duplicate, self.feedback(.01), now=.01)["state"], "WAIT_GRIP")
        stale = self.frame(1, grip=True)
        result = real.tick_grip(stale, self.feedback(.3), now=.3)
        self.assertEqual(result["state"], "WAIT_INPUT")
        self.assertEqual(self.publisher.messages, [])
        held = self.frame(31, grip=True)
        real.tick_grip(held, self.feedback(.31), now=.31)
        real.tick_grip(self.frame(32, grip=True), self.armed_feedback(.32), now=.32)
        real.tick_grip(None, self.armed_feedback(.33), now=.33)
        replayed_release = self.frame(31)
        self.assertEqual(real.tick_grip(replayed_release, self.feedback(.34), now=.34)["state"], "WAIT_INPUT")
        self.assertEqual(real.tick_grip(self.frame(35, grip=True), self.feedback(.35), now=.35)["state"], "WAIT_RELEASE")

    def test_grip_mode_change_requires_release_and_home_overrides_previous_fault(self):
        self.start_grip()
        self.real.tick_grip(None, self.armed_feedback(.02), now=.02)
        self.planner.q_home[self.names.index("arm_l_joint1")] += .02
        home = self.real.tick_grip(None, self.feedback(.03), now=.03, action="home")
        self.assertEqual(home["state"], "ARMING")
        self.assertTrue(home["home_pending"])
        self.real.tick_grip(None, self.feedback(.035), now=.035)
        home = self.real.tick_grip(None, self.armed_feedback(.04), now=.04)
        self.assertEqual(home["state"], "HOMING")
        self.real.set_dry_run(True, self.publisher_factory)
        result = self.real.tick_grip(self.frame(5, grip=True), self.feedback(.05), now=.05)
        self.assertEqual(result["state"], "WAIT_RELEASE")
        self.assertFalse(result["home_pending"])
        self.assertFalse(result["homing"])
        self.real.tick_grip(self.frame(6), self.feedback(.06), now=.06)
        active = self.real.tick_grip(self.frame(7, grip=True), self.feedback(.07), now=.07)
        self.assertEqual(active["state"], "ACTIVE")
        self.assertEqual(self.real.publisher.messages, [])

    def test_mode_change_rejects_release_queued_between_last_tick_and_switch(self):
        self.start_grip()
        self.real.set_dry_run(True, self.publisher_factory)
        queued = self.frame(3)
        queued["sample_time_ms"] = 15.
        result = self.real.tick_grip(queued, self.feedback(.03), now=.03)
        self.assertEqual(result["state"], "WAIT_RELEASE")
        self.assertEqual(self.real.tick_grip(self.frame(4, grip=True), self.feedback(.04), now=.04)["state"], "WAIT_RELEASE")
        self.assertEqual(self.real.tick_grip(self.frame(5), self.feedback(.05), now=.05)["state"], "WAIT_GRIP")
        self.assertEqual(self.real.tick_grip(self.frame(6, grip=True), self.feedback(.06), now=.06)["state"], "ACTIVE")

    def test_explicit_facing_rotation_allows_grip_start_without_head_calibration(self):
        self.planner.mapper = ControllerMapper(R_BQ=np.eye(3))
        real = self.make_real(dry_run=True)
        sample = self.frame(0, grip=True)
        sample["head"] = None
        result = real.tick_grip(sample, self.feedback(0.), now=0.)
        self.assertEqual(result["state"], "ACTIVE")
        self.assertEqual(self.publisher.messages, [])

    def test_home_finish_and_cancel_require_new_release_and_have_no_control_button_reason(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                real = self.make_real(dry_run=True)
                real.tick_grip(None, self.feedback(0.), now=0.)
                result = real.tick_grip(None, self.feedback(.01), now=.01, action="home")
                if cancel:
                    result = real.tick_grip(None, self.feedback(.02), now=.02, action="cancel_home")
                else:
                    result = real.tick_grip(None, self.feedback(.02), now=.02)
                    self.assertEqual(result["state"], "HOME")
                self.assertTrue(result["grip_rearm_required"])
                self.assertNotIn("control OFF", result["reason"])
                self.assertEqual(real.tick_grip(self.frame(3, grip=True), self.feedback(.03), now=.03)["state"], "WAIT_RELEASE")
                self.assertEqual(real.tick_grip(self.frame(4), self.feedback(.04), now=.04)["state"], "WAIT_GRIP")
                self.assertEqual(real.tick_grip(self.frame(5, grip=True), self.feedback(.05), now=.05)["state"], "ACTIVE")

    def test_dry_run_synchronizes_actual_start_and_never_sends(self):
        real = self.make_real(dry_run=True)
        result = real.tick(self.frame(0), self.feedback(0.), enabled=False, now=0.)
        np.testing.assert_array_equal(result["q_command"], self.robot_initial)
        np.testing.assert_array_equal(self.planner.q_home, self.robot_initial)
        self.assertFalse(np.array_equal(self.robot_initial, self.sim_initial))
        moved = self.robot_initial.copy()
        moved[self.names.index("arm_l_joint1")] += .01
        result = real.tick(self.frame(1), self.feedback(.01, q=moved), enabled=False, now=.01)
        np.testing.assert_array_equal(result["q_command"], moved)
        np.testing.assert_array_equal(self.planner.q_home, self.robot_initial)
        real.tick(self.frame(2), self.feedback(.02, q=moved), enabled=True, now=.02)
        real.tick(self.frame(3, grip=True), self.feedback(.03, q=moved), enabled=True, now=.03)
        real.tick(self.frame(4, grip=True, height=.002), self.feedback(.04, q=moved), enabled=True, now=.04)
        real.tick(None, self.feedback(.05, q=moved), enabled=True, now=.05, action="home")
        real.tick(None, self.feedback(.06, ready=False), enabled=True, now=.06)
        real.stop()
        self.assertEqual(self.publisher.messages, [], "Shadow paths must never call send, including stop")

    def test_live_first_feedback_while_on_requires_off_before_arming(self):
        real = self.make_real()
        result = real.tick(self.frame(0), self.feedback(0.), enabled=True, now=0.)
        self.assertTrue(result["needs_disable"])
        np.testing.assert_array_equal(result["q_command"], self.robot_initial)
        result = real.tick(self.frame(1), self.feedback(.01), enabled=True, now=.01)
        self.assertTrue(result["needs_disable"])
        self.assertEqual(self.publisher.messages, [])
        real.tick(self.frame(2), self.feedback(.02), enabled=False, now=.02)
        result = real.tick(self.frame(3), self.feedback(.03), enabled=True, now=.03)
        self.assertEqual(result["state"], "ARMING")

    def test_startup_input_wait_recovers_only_after_release_at_latest_measured_pose(self):
        for dry_run in (True, False):
            with self.subTest(dry_run=dry_run):
                real = self.make_real(dry_run=dry_run)
                real.tick(self.frame(0), self.feedback(0.), enabled=False, now=0.)
                home = self.planner.q_home.copy()
                invalid = self.frame(1, grip=True)
                invalid["hands"]["left"].update(valid=False, emulated=True)
                result = real.tick(invalid, self.feedback(.01), enabled=True, now=.01)
                self.assertEqual(result["state"], "WAIT_INPUT")
                self.assertTrue(result["enabled"])
                self.assertFalse(result["needs_disable"])
                self.assertFalse(result["control_has_started"])
                self.assertEqual(result["targets"], {})
                self.assertEqual(self.publisher.messages, [])

                moved = self.robot_initial.copy()
                moved[self.names.index("arm_l_joint1")] += .01
                invalid = self.frame(2)
                invalid["hands"]["left"]["valid"] = False
                waiting = real.tick(invalid, self.feedback(.02, q=moved), enabled=True, now=.02)
                self.assertEqual(waiting["state"], "WAIT_INPUT")
                np.testing.assert_array_equal(waiting["q_command"], moved)
                np.testing.assert_array_equal(self.planner.q_goal, moved)

                held = real.tick(self.frame(3, grip=True, height=.1), self.feedback(.03, q=moved),
                                 enabled=True, now=.03)
                self.assertEqual(held["state"], "WAIT_RELEASE")
                self.assertFalse(held["needs_disable"])
                self.assertEqual(held["targets"], {})
                self.assertEqual(self.publisher.messages, [])
                moved[self.names.index("arm_l_joint1")] += .01
                released = real.tick(self.frame(4), self.feedback(.04, q=moved), enabled=True, now=.04)
                self.assertEqual(released["state"], "READY" if dry_run else "ARMING")
                self.assertFalse(released["needs_disable"])
                np.testing.assert_array_equal(released["q_command"], moved)
                np.testing.assert_array_equal(self.planner.q_home, home)
                if dry_run:
                    self.assertEqual(self.publisher.messages, [])
                    active = real.tick(self.frame(5, grip=True), self.feedback(.05, q=moved),
                                       enabled=True, now=.05)
                    self.assertEqual(active["state"], "ACTIVE")
                    self.assertTrue(active["control_has_started"])
                    expected = get_sg2_wrist_control_T(self.env_measured, "left")
                    np.testing.assert_allclose(active["targets"]["left"], expected, atol=1e-12)
                else:
                    self.assertTrue(released["control_has_started"])
                    self.assertEqual([m["mode"] for m in self.publisher.messages], ["arm"])
                    np.testing.assert_array_equal(parse_joint_groups(self.publisher.messages[-1]["joints"]), moved)
                    ready = real.tick(self.frame(5), self.armed_feedback(.05, q=moved), enabled=True, now=.05)
                    self.assertEqual(ready["state"], "READY")

    def test_known_startup_input_unavailability_waits_without_latching(self):
        cases = {
            "no_frame": None,
            "stale_pose": lambda frame: frame.update(received_at=-1.),
            "session_not_visible": lambda frame: frame.update(visibility="hidden"),
            "emulated_pose_left": lambda frame: frame["hands"]["left"].update(emulated=True),
            "tracking_lost_right": lambda frame: frame["hands"]["right"].update(valid=False),
            "head_calibration_unavailable": lambda frame: frame.update(head=None),
        }
        for reason, mutate in cases.items():
            with self.subTest(reason=reason):
                real = self.make_real()
                real.tick(self.frame(0), self.feedback(0.), enabled=False, now=0.)
                frame = self.frame(1) if mutate else None
                if mutate:
                    mutate(frame)
                result = real.tick(frame, self.feedback(.01), enabled=True, now=.01)
                self.assertEqual((result["state"], result["reason"]), ("WAIT_INPUT", reason))
                self.assertFalse(result["needs_disable"])
                self.assertEqual(self.publisher.messages, [])

    def test_dry_ready_then_input_loss_can_wait_but_active_loss_requires_off(self):
        real = self.make_real(dry_run=True)
        real.tick(self.frame(0), self.feedback(0.), enabled=False, now=0.)
        ready = real.tick(self.frame(1), self.feedback(.01), enabled=True, now=.01)
        self.assertEqual(ready["state"], "READY")
        waiting = real.tick(None, self.feedback(.02), enabled=True, now=.02)
        self.assertEqual(waiting["state"], "WAIT_INPUT")
        held = real.tick(self.frame(3, grip=True), self.feedback(.03), enabled=True, now=.03)
        self.assertEqual(held["state"], "WAIT_RELEASE")
        real.tick(self.frame(4), self.feedback(.04), enabled=True, now=.04)
        real.tick(self.frame(5, grip=True), self.feedback(.05), enabled=True, now=.05)
        failed = real.tick(None, self.feedback(.06), enabled=True, now=.06)
        self.assertEqual(failed["state"], "HOLD")
        self.assertTrue(failed["needs_disable"])
        self.assertTrue(failed["control_has_started"])
        recovered = real.tick(self.frame(7), self.feedback(.07), enabled=True, now=.07)
        self.assertTrue(recovered["needs_disable"])
        real.tick(self.frame(8), self.feedback(.08), enabled=False, now=.08)
        waiting = real.tick(None, self.feedback(.09), enabled=True, now=.09)
        self.assertEqual(waiting["state"], "WAIT_INPUT")
        self.assertFalse(waiting["control_has_started"])
        self.assertEqual(self.publisher.messages, [])

    def test_input_loss_after_first_arm_request_stops_even_before_ack(self):
        for acknowledged in (False, True):
            with self.subTest(acknowledged=acknowledged):
                real = self.make_real()
                real.tick(self.frame(0), self.feedback(0.), enabled=False, now=0.)
                real.tick(self.frame(1), self.feedback(.01), enabled=True, now=.01)
                if acknowledged:
                    real.tick(self.frame(2), self.armed_feedback(.02), enabled=True, now=.02)
                count = len(self.publisher.messages)
                feedback = self.armed_feedback(.03) if acknowledged else self.feedback(.03)
                failed = real.tick(None, feedback, enabled=True, now=.03)
                self.assertEqual(failed["state"], "HOLD")
                self.assertTrue(failed["needs_disable"])
                self.assertTrue(failed["control_has_started"])
                self.assertEqual([m["mode"] for m in self.publisher.messages[count:]], ["stop"])
                self.assertIsNone(real.session_id)

    def test_startup_robot_staleness_still_requires_off(self):
        real = self.make_real()
        real.tick(self.frame(0), self.feedback(0.), enabled=False, now=0.)
        real.tick(None, self.feedback(.01), enabled=True, now=.01)
        result = real.tick(None, self.feedback(.02, ready=False), enabled=True, now=.02)
        self.assertEqual(result["state"], "WAIT_ROBOT")
        self.assertTrue(result["needs_disable"])
        recovered = real.tick(self.frame(3), self.feedback(.03), enabled=True, now=.03)
        self.assertTrue(recovered["needs_disable"])
        self.assertEqual(self.publisher.messages, [])

    def test_startup_malformed_pose_and_ik_fault_are_not_input_waits(self):
        real = self.make_real(dry_run=True)
        real.tick(self.frame(0), self.feedback(0.), enabled=False, now=0.)
        invalid = self.frame(1)
        invalid["hands"]["left"]["quaternion_xyzw"] = [float("nan"), 0., 0., 1.]
        result = real.tick(invalid, self.feedback(.01), enabled=True, now=.01)
        self.assertEqual(result["state"], "HOLD")
        self.assertTrue(result["needs_disable"])
        real.tick(self.frame(2), self.feedback(.02), enabled=False, now=.02)
        real.tick(self.frame(3), self.feedback(.03), enabled=True, now=.03)
        with patch.object(self.planner, "_solve", side_effect=FloatingPointError("invalid test IK")):
            result = real.tick(self.frame(4, grip=True), self.feedback(.04), enabled=True, now=.04)
        self.assertEqual(result["state"], "FAULT")
        self.assertTrue(result["needs_disable"])
        self.assertEqual(self.publisher.messages, [])

    def test_dry_home_counts_as_control_started_and_mode_switch_resets_it(self):
        real = self.make_real(dry_run=True)
        real.tick(self.frame(0), self.feedback(0.), enabled=False, now=0.)
        self.planner.q_home[self.names.index("arm_l_joint1")] += .02
        result = real.tick(None, self.feedback(.01), enabled=True, now=.01, action="home")
        self.assertEqual(result["state"], "HOMING")
        self.assertTrue(result["control_has_started"])
        real.set_dry_run(False, self.publisher_factory)
        self.assertFalse(real.control_has_started)
        self.assertIsNotNone(real.latched_reason)

    def test_release_and_matching_owner_ack_are_required_before_command(self):
        real = self.make_real()
        real.tick(self.frame(0), self.feedback(0.), enabled=False, now=0.)
        held = real.tick(self.frame(1, grip=True), self.feedback(.01), enabled=True, now=.01)
        self.assertEqual(held["state"], "WAIT_RELEASE")
        self.assertEqual(self.publisher.messages, [])
        result = real.tick(self.frame(2), self.feedback(.02), enabled=True, now=.02)
        self.assertEqual(result["state"], "ARMING")
        packet = self.publisher.messages[-1]
        np.testing.assert_array_equal(parse_joint_groups(packet["joints"]), self.robot_initial)
        self.assertEqual(packet["bridge_id"], "test-bridge")
        self.assertEqual(packet["feedback_time_s"], .02)
        self.assertGreater(packet["valid_for_s"], 0.)
        self.assertLessEqual(packet["valid_for_s"], .2)
        wrong_ack = self.feedback(.03, bridge_state="ACTIVE", owner="another-notebook")
        result = real.tick(self.frame(3), wrong_ack, enabled=True, now=.03)
        self.assertEqual(result["state"], "ARMING")
        self.assertTrue(all(message["mode"] == "arm" for message in self.publisher.messages))
        result = real.tick(self.frame(4), self.armed_feedback(.04), enabled=True, now=.04)
        self.assertEqual(result["state"], "READY")
        self.assertEqual(self.publisher.messages[-1]["mode"], "command")
        self.assertEqual([message["seq"] for message in self.publisher.messages], [0, 1, 2])

    def test_clutch_anchors_follow_measured_wrist_not_command_preview(self):
        self.arm()
        measured = self.robot_initial.copy()
        measured[self.names.index("arm_l_joint1")] += .005
        self.env_measured.forward(q=measured, joint_names=self.names)
        expected = get_sg2_wrist_control_T(self.env_measured, "left").copy()
        preview = get_sg2_wrist_control_T(self.planner.env, "left").copy()
        self.assertGreater(np.linalg.norm(expected - preview), 1e-3)
        result = self.real.tick(self.frame(3, grip=True), self.armed_feedback(.03, q=measured),
                                enabled=True, now=.03)
        self.assertEqual(result["state"], "ACTIVE", result["reason"])
        np.testing.assert_allclose(result["targets"]["left"], expected, atol=1e-12)
        np.testing.assert_array_equal(result["robot_q"], measured)

    def test_stale_feedback_stops_once_and_latches_until_off(self):
        self.arm()
        count = len(self.publisher.messages)
        result = self.real.tick(self.frame(3, grip=True), self.feedback(.03, ready=False),
                                enabled=True, now=.03)
        self.assertTrue(result["needs_disable"])
        self.assertFalse(result["armed"])
        self.assertEqual([m["mode"] for m in self.publisher.messages[count:]], ["stop"])
        count = len(self.publisher.messages)
        result = self.real.tick(self.frame(4), self.feedback(.04), enabled=True, now=.04)
        self.assertTrue(result["needs_disable"])
        self.assertEqual(len(self.publisher.messages), count)
        result = self.real.tick(self.frame(5), self.feedback(.05), enabled=False, now=.05)
        self.assertFalse(result["needs_disable"])
        result = self.real.tick(self.frame(6), self.feedback(.06), enabled=True, now=.06)
        self.assertEqual(result["state"], "ARMING")

    def test_lost_command_owner_stops_and_requires_off(self):
        self.arm()
        result = self.real.tick(self.frame(3), self.feedback(.03, bridge_state="ACTIVE", owner="new-owner"),
                                enabled=True, now=.03)
        self.assertEqual(result["state"], "FAULT")
        self.assertTrue(result["needs_disable"])
        self.assertEqual(self.publisher.messages[-1]["mode"], "stop")
        self.assertIsNone(self.real.session_id)

    def test_tracking_difference_blocks_new_commands(self):
        self.arm()
        moved = self.robot_initial.copy()
        moved[self.names.index("arm_l_joint1")] += np.deg2rad(6)
        count = len(self.publisher.messages)
        result = self.real.tick(self.frame(3, grip=True), self.armed_feedback(.03, q=moved),
                                enabled=True, now=.03)
        self.assertEqual(result["state"], "FAULT")
        self.assertIn("tracking difference", result["reason"])
        self.assertTrue(result["needs_disable"])
        self.assertEqual([m["mode"] for m in self.publisher.messages[count:]], ["stop"])

    def test_measured_to_command_collision_sweep_blocks_publish(self):
        self.arm()
        measured = self.robot_initial.copy()
        measured[self.names.index("arm_l_joint1")] += .005
        reject = {"accepted": False, "reason": "test measured sweep collision", "min_distance_m": -.001}
        count = len(self.publisher.messages)
        with patch.object(self.guard, "check", return_value=reject) as check:
            result = self.real.tick(self.frame(3), self.armed_feedback(.03, q=measured),
                                    enabled=True, now=.03)
        self.assertEqual(check.call_count, 1, "READY must check the measured path even with no IK motion")
        np.testing.assert_array_equal(check.call_args.args[0], measured)
        self.assertEqual(result["state"], "FAULT")
        self.assertIn("measured path rejected", result["reason"])
        self.assertEqual([m["mode"] for m in self.publisher.messages[count:]], ["stop"])

    def test_home_is_gated_and_returns_to_actual_start_with_limited_steps(self):
        self.make_real()
        self.real.tick(self.frame(0), self.feedback(0.), enabled=False, now=0.)
        moved = self.robot_initial.copy()
        moved[self.names.index("arm_l_joint1")] += np.deg2rad(2)
        result = self.real.tick(None, self.feedback(.01, q=moved), enabled=False, now=.01, action="home")
        self.assertFalse(result["homing"])
        np.testing.assert_array_equal(result["q_command"], moved)
        self.assertEqual(self.publisher.messages, [])
        self.real.tick(self.frame(2), self.feedback(.02, q=moved), enabled=True, now=.02)
        self.real.tick(self.frame(3), self.armed_feedback(.03, q=moved), enabled=True, now=.03)
        previous = moved.copy()
        for index in range(4, 105):
            now = index / 100
            result = self.real.tick(None, self.armed_feedback(now, q=previous), enabled=True, now=now,
                                    action="home" if index == 4 else None)
            self.assertNotEqual(result["state"], "FAULT", result["reason"])
            self.assertEqual(result["active"], [])
            self.assertTrue(np.all(np.abs(result["q_command"] - previous) <= self.planner.speed_limits * .01 + 1e-12))
            previous = result["q_command"].copy()
            if result["state"] == "HOME":
                break
        self.assertEqual(result["state"], "HOME", result["reason"])
        np.testing.assert_allclose(result["q_command"], self.robot_initial, atol=1e-9)
        self.assertTrue(result["needs_disable"])
        self.assertEqual(self.publisher.messages[-1]["mode"], "stop")

    def test_explicit_home_from_off_ignores_quest_but_waits_for_live_ack(self):
        for dry_run in (True, False):
            for input_kind in ("missing", "emulated_held"):
                with self.subTest(dry_run=dry_run, input_kind=input_kind):
                    real = self.make_real(dry_run=dry_run)
                    real.tick(None, self.feedback(0.), enabled=False, now=0.)
                    home = self.planner.q_home.copy()
                    moved = home.copy()
                    moved[self.names.index("arm_l_joint1")] += np.deg2rad(2)
                    real.tick(None, self.feedback(.01, q=moved), enabled=False, now=.01)
                    frame = None
                    if input_kind == "emulated_held":
                        frame = self.frame(2, grip=True)
                        for hand in frame["hands"].values():
                            hand.update(valid=False, emulated=True)
                    result = real.tick(frame, self.feedback(.02, q=moved), enabled=True,
                                       now=.02, action="home")
                    self.assertEqual(result["state"], "HOMING" if dry_run else "ARMING")
                    self.assertEqual(result["home_pending"], not dry_run)
                    self.assertEqual(result["homing"], dry_run)
                    self.assertFalse(result["needs_disable"])
                    self.assertEqual(result["active"], [])
                    np.testing.assert_array_equal(result["q_command"], moved)
                    np.testing.assert_array_equal(self.planner.q_home, home)
                    if not dry_run:
                        # The path check ran on the click tick; the lease is
                        # requested on the next one with fresh feedback.
                        self.assertEqual(self.publisher.messages, [])
                        self.assertTrue(result["home_plan"]["ok"])
                        waiting = real.tick(None, self.feedback(.03, q=moved, bridge_state="ACTIVE", owner="other"),
                                            enabled=True, now=.03)
                        self.assertEqual(waiting["state"], "ARMING")
                        self.assertTrue(waiting["home_pending"])
                        self.assertFalse(waiting["homing"])
                        np.testing.assert_array_equal(waiting["q_command"], moved)
                        self.assertEqual([m["mode"] for m in self.publisher.messages], ["arm"])
                        for message in self.publisher.messages:
                            np.testing.assert_array_equal(parse_joint_groups(message["joints"]), moved)
                        ack = real.tick(None, self.armed_feedback(.04, q=moved), enabled=True, now=.04)
                        self.assertEqual(ack["state"], "HOMING")
                        self.assertFalse(ack["home_pending"])
                        self.assertTrue(ack["homing"])
                        np.testing.assert_array_equal(ack["q_command"], moved)
                    previous = moved.copy()
                    for index in range(5, 100):
                        now = index / 100
                        feedback = self.feedback(now, q=previous) if dry_run else self.armed_feedback(now, q=previous)
                        result = real.tick(None, feedback, enabled=True, now=now)
                        self.assertIn(result["state"], ("HOMING", "HOME"), result["reason"])
                        self.assertFalse(result["home_pending"])
                        self.assertEqual(result["targets"], {})
                        self.assertTrue(np.all(np.abs(result["q_command"] - previous)
                                               <= self.planner.speed_limits * result["command_dt"] + 1e-12))
                        previous = result["q_command"].copy()
                        if result["state"] == "HOME":
                            break
                    self.assertEqual(result["state"], "HOME")
                    self.assertTrue(result["needs_disable"])
                    np.testing.assert_allclose(result["q_command"], home, atol=1e-9)
                    if dry_run:
                        self.assertEqual(self.publisher.messages, [])
                    else:
                        self.assertEqual(self.publisher.messages[-1]["mode"], "stop")
                    real.tick(None, self.feedback(now + .01, q=home), enabled=False, now=now + .01)
                    held = real.tick(self.frame(index + 2, grip=True), self.feedback(now + .02, q=home),
                                     enabled=True, now=now + .02)
                    self.assertEqual(held["state"], "WAIT_RELEASE")

    def test_explicit_home_cancel_and_off_clear_pending_and_active_return(self):
        for acknowledged in (False, True):
            for cancel in (False, True):
                with self.subTest(acknowledged=acknowledged, cancel=cancel):
                    real = self.make_real()
                    real.tick(None, self.feedback(0.), enabled=False, now=0.)
                    self.planner.q_home[self.names.index("arm_l_joint1")] += .02
                    real.tick(None, self.feedback(.01), enabled=True, now=.01, action="home")
                    real.tick(None, self.feedback(.015), enabled=True, now=.015)
                    if acknowledged:
                        real.tick(None, self.armed_feedback(.02), enabled=True, now=.02)
                    count = len(self.publisher.messages)
                    feedback = self.armed_feedback(.03) if acknowledged else self.feedback(.03)
                    result = real.tick(None, feedback, enabled=cancel, now=.03,
                                       action="cancel_home" if cancel else None)
                    self.assertFalse(result["home_pending"])
                    self.assertFalse(result["homing"])
                    self.assertEqual([m["mode"] for m in self.publisher.messages[count:]], ["stop"])
                    self.assertIsNone(real.session_id)
                    real.tick(None, self.feedback(.04), enabled=False, now=.04)
                    count = len(self.publisher.messages)
                    next_on = real.tick(None, self.feedback(.05), enabled=True, now=.05)
                    self.assertEqual(next_on["state"], "WAIT_INPUT")
                    self.assertFalse(next_on["home_pending"])
                    self.assertFalse(next_on["homing"])
                    self.assertEqual(len(self.publisher.messages), count)

    def test_home_click_during_teleop_arming_retains_lease_and_latest_measured_pose(self):
        real = self.make_real()
        real.tick(self.frame(0), self.feedback(0.), enabled=False, now=0.)
        real.tick(self.frame(1), self.feedback(.01), enabled=True, now=.01)
        session, started, home = real.session_id, real.arm_started, self.planner.q_home.copy()
        moved = self.robot_initial.copy()
        moved[self.names.index("arm_l_joint1")] += .02
        for index in (2, 3):
            now = index / 100
            result = real.tick(None, self.feedback(now, q=moved), enabled=True, now=now, action="home")
            self.assertEqual(result["state"], "ARMING")
            self.assertTrue(result["home_pending"])
            self.assertFalse(result["homing"])
            self.assertEqual((real.session_id, real.arm_started), (session, started))
            np.testing.assert_array_equal(result["q_command"], moved)
            np.testing.assert_array_equal(parse_joint_groups(self.publisher.messages[-1]["joints"]), moved)
        result = real.tick(None, self.armed_feedback(.04, q=moved), enabled=True, now=.04)
        self.assertEqual(result["state"], "HOMING")
        self.assertFalse(result["home_pending"])
        self.assertEqual(result["targets"], {})
        np.testing.assert_array_equal(self.planner.q_home, home)
        np.testing.assert_array_equal(result["q_command"], moved)

    def test_pending_home_feedback_failure_and_identity_change_cannot_resume(self):
        for changed in (False, True):
            with self.subTest(changed=changed):
                real = self.make_real()
                real.tick(None, self.feedback(0.), enabled=False, now=0.)
                real.tick(None, self.feedback(.01), enabled=True, now=.01, action="home")
                real.tick(None, self.feedback(.015), enabled=True, now=.015)
                if changed:
                    self.generation += 1
                failed_feedback = self.feedback(.02, ready=changed)
                failed = real.tick(None, failed_feedback, enabled=True, now=.02)
                self.assertTrue(failed["needs_disable"])
                self.assertFalse(failed["home_pending"])
                self.assertFalse(failed["homing"])
                self.assertEqual(self.publisher.messages[-1]["mode"], "stop")
                count = len(self.publisher.messages)
                recovered = real.tick(None, self.feedback(.03), enabled=True, now=.03)
                self.assertTrue(recovered["needs_disable"])
                self.assertEqual(len(self.publisher.messages), count)
                real.tick(None, self.feedback(.04), enabled=False, now=.04)
                waiting = real.tick(None, self.feedback(.05), enabled=True, now=.05)
                self.assertEqual(waiting["state"], "WAIT_INPUT")
                self.assertFalse(waiting["home_pending"])

    def test_home_guard_fault_before_and_after_ack_clears_request(self):
        reject = {"accepted": False, "reason": "test return collision", "min_distance_m": -.001}
        for acknowledged in (False, True):
            with self.subTest(acknowledged=acknowledged):
                real = self.make_real()
                real.tick(None, self.feedback(0.), enabled=False, now=0.)
                self.planner.q_home[self.names.index("arm_l_joint1")] += .02
                if acknowledged:
                    real.tick(None, self.feedback(.01), enabled=True, now=.01, action="home")
                    real.tick(None, self.armed_feedback(.02), enabled=True, now=.02)
                with patch.object(self.guard, "check", return_value=reject):
                    feedback = self.armed_feedback(.03) if acknowledged else self.feedback(.03)
                    result = real.tick(None, feedback, enabled=True, now=.03,
                                       action=None if acknowledged else "home")
                self.assertEqual(result["state"], "FAULT")
                self.assertTrue(result["needs_disable"])
                self.assertFalse(result["home_pending"])
                self.assertFalse(result["homing"])
                if acknowledged:
                    self.assertEqual(self.publisher.messages[-1]["mode"], "stop")
                else:
                    self.assertEqual(self.publisher.messages, [])
                recovered = real.tick(None, self.feedback(.04), enabled=True, now=.04)
                self.assertTrue(recovered["needs_disable"])
                self.assertFalse(recovered["home_pending"])

    def test_pending_home_mode_switch_and_stop_do_not_resume_it(self):
        for mode_change in (False, True):
            with self.subTest(mode_change=mode_change):
                real = self.make_real()
                real.tick(None, self.feedback(0.), enabled=False, now=0.)
                real.tick(None, self.feedback(.01), enabled=True, now=.01, action="home")
                real.tick(None, self.feedback(.015), enabled=True, now=.015)
                if mode_change:
                    real.set_dry_run(True, self.publisher_factory)
                else:
                    real.stop()
                self.assertFalse(real.home_pending)
                self.assertFalse(self.planner.homing)
                self.assertIsNone(real.session_id)
                self.assertEqual(self.publisher.messages[-1]["mode"], "stop")
                real.tick(None, self.feedback(.02), enabled=False, now=.02)
                result = real.tick(None, self.feedback(.03), enabled=True, now=.03)
                self.assertEqual(result["state"], "WAIT_INPUT")
                self.assertFalse(result["home_pending"])
                self.assertFalse(result["homing"])

    def test_explicit_home_ack_timeout_and_active_delay_still_stop(self):
        for acknowledged in (False, True):
            with self.subTest(acknowledged=acknowledged):
                real = self.make_real()
                real.tick(None, self.feedback(0.), enabled=False, now=0.)
                self.planner.q_home[self.names.index("arm_l_joint1")] += .02
                real.tick(None, self.feedback(.01), enabled=True, now=.01, action="home")
                real.tick(None, self.feedback(.02), enabled=True, now=.02)
                if acknowledged:
                    homing = real.tick(None, self.armed_feedback(.03), enabled=True, now=.03)
                    self.assertEqual(homing["state"], "HOMING", homing["reason"])
                    result = real.tick(None, self.armed_feedback(.21), enabled=True, now=.21)
                else:
                    result = real.tick(None, self.feedback(.53), enabled=True, now=.53)
                self.assertEqual(result["state"], "FAULT")
                self.assertTrue(result["needs_disable"])
                self.assertFalse(result["home_pending"])
                self.assertFalse(result["homing"])
                self.assertEqual(self.publisher.messages[-1]["mode"], "stop")

    def test_off_cancels_home_and_resynchronizes_measured_state(self):
        self.arm()
        self.planner.q_home[self.names.index("arm_l_joint1")] += .02
        result = self.real.tick(None, self.armed_feedback(.03), enabled=True, now=.03, action="home")
        # An armed request releases its lease before the path check.
        self.assertEqual(result["state"], "ARMING")
        self.assertEqual(self.publisher.messages[-1]["mode"], "stop")
        self.real.tick(None, self.feedback(.035), enabled=True, now=.035)
        result = self.real.tick(None, self.armed_feedback(.037), enabled=True, now=.037)
        self.assertEqual(result["state"], "HOMING")
        result = self.real.tick(None, self.armed_feedback(.04), enabled=False, now=.04)
        self.assertFalse(result["homing"])
        np.testing.assert_array_equal(result["q_command"], self.robot_initial)
        self.assertEqual(self.publisher.messages[-1]["mode"], "stop")

    def test_home_waits_for_measured_arrival_before_releasing_lease(self):
        self.make_real()
        self.real.tick(self.frame(0), self.feedback(0.), enabled=False, now=0.)
        measured = self.robot_initial.copy()
        measured[self.names.index("arm_l_joint1")] += np.deg2rad(2)
        self.real.tick(self.frame(1), self.feedback(.01, q=measured), enabled=False, now=.01)
        self.real.tick(self.frame(2), self.feedback(.02, q=measured), enabled=True, now=.02)
        self.real.tick(self.frame(3), self.armed_feedback(.03, q=measured), enabled=True, now=.03)
        for index in range(4, 65):
            now = index / 100
            result = self.real.tick(None, self.armed_feedback(now, q=measured), enabled=True, now=now,
                                    action="home" if index == 4 else None)
            if index <= 5:
                # Lease released for the path check, then requested again.
                self.assertEqual(result["state"], "ARMING", result["reason"])
                continue
            self.assertEqual(result["state"], "HOMING", "Command arrival must not cancel physical return")
        np.testing.assert_allclose(result["q_command"], self.robot_initial, atol=1e-9)
        np.testing.assert_allclose(result["home_error"], measured - self.robot_initial, atol=1e-9)
        self.assertEqual(self.publisher.messages[-1]["mode"], "command")
        result = self.real.tick(None, self.armed_feedback(.65), enabled=True, now=.65)
        self.assertEqual(result["state"], "HOME")
        self.assertEqual(self.publisher.messages[-1]["mode"], "stop")

    def test_home_settling_timeout_stops_an_unresponsive_robot(self):
        self.arm()
        self.planner.q_home[self.names.index("arm_l_joint1")] += np.deg2rad(2)
        for index in range(3, 405):
            now = index / 100
            result = self.real.tick(None, self.armed_feedback(now), enabled=True, now=now,
                                    action="home" if index == 3 else None)
            if result["state"] == "FAULT":
                break
            self.assertEqual(result["state"], "ARMING" if index <= 4 else "HOMING", result["reason"])
        self.assertEqual(result["state"], "FAULT")
        self.assertIn("settling time", result["reason"])
        self.assertTrue(result["needs_disable"])
        self.assertFalse(result["homing"])
        self.assertGreater(now, 3.)
        self.assertEqual(self.publisher.messages[-1]["mode"], "stop")

    def test_blocked_home_path_sends_nothing_and_next_click_checks_again(self):
        self.make_real()
        self.real.tick_grip(None, self.feedback(0.), now=0.)
        self.planner.q_home[self.names.index("arm_l_joint1")] += .02
        reject = {"accepted": False, "reason": "collision_clearance", "min_distance_m": .004,
                  "pair": {"geom_ids": (1, 2), "body_names": ("arm_l_link5", "gripper_r_r1")},
                  "samples_checked": 5, "sweep_fraction": .4}
        with patch.object(self.guard, "check", return_value=reject):
            blocked = self.real.tick_grip(None, self.feedback(.01), now=.01, action="home")
        self.assertEqual(blocked["state"], "FAULT")
        self.assertIn("initial pose path blocked", blocked["reason"])
        self.assertIn("arm_l_link5-gripper_r_r1", blocked["reason"])
        self.assertFalse(blocked["home_plan"]["ok"])
        self.assertEqual(len(blocked["home_plan"]["attempts"]), 5)
        self.assertFalse(blocked["guard"]["accepted"])
        self.assertFalse(blocked["home_pending"])
        self.assertFalse(blocked["homing"])
        self.assertEqual(self.publisher.messages, [], "A blocked return must not even request a lease")
        for index in range(2, 5):
            held = self.real.tick_grip(None, self.feedback(index / 100), now=index / 100)
            self.assertNotIn(held["state"], ("ARMING", "HOMING"))
        self.assertEqual(self.publisher.messages, [])
        retry = self.real.tick_grip(None, self.feedback(.05), now=.05, action="home")
        self.assertEqual(retry["state"], "ARMING", retry["reason"])
        self.assertEqual(retry["home_plan"]["strategy"], "direct")
        arming = self.real.tick_grip(None, self.feedback(.06), now=.06)
        self.assertEqual(arming["state"], "ARMING")
        self.assertEqual([m["mode"] for m in self.publisher.messages], ["arm"])
        homing = self.real.tick_grip(None, self.armed_feedback(.07), now=.07)
        self.assertEqual(homing["state"], "HOMING", homing["reason"])

    def test_live_home_follows_staged_checked_path_to_measured_start(self):
        self.make_real()
        self.real.tick(None, self.feedback(0.), enabled=False, now=0.)
        home = self.planner.q_home.copy()
        j1, j3 = self.names.index("arm_r_joint1"), self.names.index("arm_r_joint3")
        start = home.copy()
        start[j1] -= np.deg2rad(2)
        start[j3] += np.deg2rad(3)
        self.real.tick(None, self.feedback(.01, q=start), enabled=False, now=.01)
        original, calls = self.guard.check, []
        def direct_blocked(q_start, q_end):
            calls.append(1)
            if len(calls) == 1:
                return {"accepted": False, "reason": "collision_clearance", "min_distance_m": .004,
                        "pair": {"geom_ids": (1, 2), "body_names": ("arm_l_link5", "gripper_r_r1")},
                        "samples_checked": 3, "sweep_fraction": .5}
            return original(q_start, q_end)
        with patch.object(self.guard, "check", side_effect=direct_blocked):
            click = self.real.tick_grip(None, self.feedback(.02, q=start), now=.02, action="home")
        self.assertEqual(click["state"], "ARMING", click["reason"])
        self.assertEqual(click["home_plan"]["strategy"], "unwind_j3j5_first")
        self.real.tick_grip(None, self.feedback(.03, q=start), now=.03)
        measured, stages = start.copy(), []
        for index in range(4, 400):
            now = index / 100
            result = self.real.tick_grip(None, self.armed_feedback(now, q=measured), now=now)
            self.assertIn(result["state"], ("HOMING", "HOME"), result["reason"])
            q = result["q_command"]
            if q[j3] != home[j3]:
                self.assertEqual(q[j1], start[j1], "joint 1 must wait until joint 3 is unwound")
            else:
                self.assertEqual(q[j3], home[j3])
            if result.get("home_path"):
                stages.append(result["home_path"]["stage"])
            measured = q.copy()
            if result["state"] == "HOME":
                break
        self.assertEqual(result["state"], "HOME")
        np.testing.assert_allclose(measured, home, atol=1e-9)
        self.assertEqual(sorted(set(stages)), [1, 2])
        commands = [m for m in self.publisher.messages if m["mode"] == "command"]
        self.assertTrue(commands)
        self.assertTrue(all(m["motion"] == "home" for m in commands))
        self.assertEqual(self.publisher.messages[-1]["mode"], "stop")

    def test_home_check_time_is_not_a_loop_delay_or_stale_arm(self):
        for dry_run in (True, False):
            with self.subTest(dry_run=dry_run):
                real = self.make_real(dry_run=dry_run)
                real.tick_grip(None, self.feedback(0.), now=0.)
                self.planner.q_home[self.names.index("arm_l_joint1")] += .02
                real.tick_grip(None, self.feedback(.01), now=.01, action="home")
                # Simulate a path check that took 0.6 s before the next tick.
                result = real.tick_grip(None, self.feedback(.61), now=.61)
                self.assertEqual(result["state"], "HOMING" if dry_run else "ARMING", result["reason"])
                if not dry_run:
                    self.assertEqual(self.publisher.messages[-1]["feedback_time_s"], .61)
                    result = real.tick_grip(None, self.armed_feedback(.62), now=.62)
                    self.assertEqual(result["state"], "HOMING", result["reason"])
                result = real.tick_grip(None, (self.feedback if dry_run else self.armed_feedback)(.63), now=.63)
                self.assertEqual(result["state"], "HOMING", result["reason"])
                # Only the planning gap is excused; a later stall still stops.
                result = real.tick_grip(None, (self.feedback if dry_run else self.armed_feedback)(.9), now=.9)
                self.assertEqual(result["state"], "FAULT")

    def test_bridge_restart_keeps_first_measured_home(self):
        real = self.make_real()
        real.tick(None, self.feedback(0.), enabled=False, now=0.)
        np.testing.assert_array_equal(self.planner.q_home, self.robot_initial)
        wound = self.robot_initial.copy()
        wound[self.names.index("arm_r_joint3")] += 1.
        real.tick(None, self.feedback(.01, q=wound), enabled=False, now=.01)
        self.generation += 1
        result = real.tick(None, self.feedback(.02, q=wound), enabled=False, now=.02)
        self.assertEqual(real.generation, self.generation)
        np.testing.assert_array_equal(result["q_command"], wound)
        np.testing.assert_array_equal(self.planner.q_home, self.robot_initial)

    def test_invalid_measured_state_never_arms(self):
        self.make_real()
        for index, bad in enumerate((np.full(len(self.names), np.nan), self.robot_initial[:-1])):
            result = self.real.tick(self.frame(index), self.feedback(index / 100, q=bad),
                                    enabled=True, now=index / 100)
            self.assertEqual(result["state"], "FAULT")
            self.assertTrue(result["needs_disable"])
        self.assertEqual(self.publisher.messages, [])

    def test_bridge_generation_change_revokes_old_session(self):
        self.arm()
        previous_session = self.real.session_id
        self.generation += 1
        changed = self.feedback(.03)
        changed["bridge_id"] = "restarted-bridge"
        result = self.real.tick(self.frame(3), changed, enabled=True, now=.03)
        self.assertTrue(result["needs_disable"])
        self.assertFalse(result["armed"])
        self.assertEqual(self.publisher.messages[-1]["mode"], "stop")
        self.assertEqual(self.publisher.messages[-1]["session_id"], previous_session)

    def test_arm_ack_timeout_cannot_fall_through_to_commands(self):
        self.make_real()
        self.real.tick(self.frame(0), self.feedback(0.), enabled=False, now=0.)
        for index in range(1, 54):
            now = index / 100
            result = self.real.tick(self.frame(index), self.feedback(now), enabled=True, now=now)
            if result["needs_disable"]:
                break
        self.assertEqual(result["state"], "FAULT")
        self.assertIn("acknowledge", result["reason"])
        self.assertEqual(self.publisher.messages[-1]["mode"], "stop")
        self.assertTrue(all(m["mode"] == "arm" for m in self.publisher.messages[:-1]))

    def test_failed_command_enqueue_revokes_lease(self):
        self.arm()
        with patch.object(self.publisher, "send", side_effect=lambda packet: self.publisher.messages.append(copy.deepcopy(packet)) or False):
            result = self.real.tick(self.frame(3), self.armed_feedback(.03), enabled=True, now=.03)
        self.assertEqual(result["state"], "FAULT")
        self.assertTrue(result["needs_disable"])
        self.assertFalse(result["transmitted"])
        self.assertFalse(result["armed"])
        self.assertEqual([m["mode"] for m in self.publisher.messages[-2:]], ["command", "stop"])

    def test_real_wrist_intersection_rejects_model_only_range(self):
        self.make_real()
        for index, joint in enumerate(("arm_l_joint7", "arm_r_joint7")):
            measured = self.robot_initial.copy()
            measured[self.names.index(joint)] = -1.7 if joint.startswith("arm_l") else 1.7
            result = self.real.tick(self.frame(index), self.feedback(index / 100, q=measured),
                                    enabled=True, now=index / 100)
            self.assertEqual(result["state"], "FAULT")
            self.assertIn("verified real limits", result["reason"])
        self.assertEqual(self.publisher.messages, [])

    def test_constructor_rejects_nonfinite_safety_bounds(self):
        for option in ("max_joint_lag", "max_lift_lag", "command_ttl_s", "max_home_joint_speed"):
            for value in (float("inf"), float("nan")):
                with self.subTest(option=option, value=value), self.assertRaises(ValueError):
                    SG2FixedQuestReal(self.planner, self.env_measured, self.layout,
                                     MemoryPublisher(enabled=False), dry_run=True, **{option: value})

    def test_live_teleop_reaches_sixty_degrees_with_existing_ik_and_guard(self):
        self.configure_teleop_speed(60)
        self.arm()
        previous = self.planner.q_command.copy()
        peak = 0.
        for index in range(3, 20):
            now = index / 100
            result = self.real.tick(self.frame(index, grip=True, height=0. if index == 3 else .08),
                                    self.armed_feedback(now, q=previous), enabled=True, now=now)
            self.assertEqual(result["state"], "ACTIVE", result["reason"])
            self.assertTrue(result["guard"]["accepted"])
            delta = np.abs(result["q_command"] - previous)
            self.assertTrue(np.all(delta <= self.real.teleop_speed_limits * result["command_dt"] + 1e-12))
            if result["command_dt"] > 0:
                peak = max(peak, float(np.max(delta[1:] / result["command_dt"])))
            previous = result["q_command"].copy()
        self.assertAlmostEqual(np.rad2deg(peak), 60., places=6)
        self.assertTrue(all(packet["motion"] == "teleop" for packet in self.publisher.messages))
        self.assertEqual(self.real.teleop_speed_limits[0], .01)
        self.assertEqual(self.real.home_speed_limits[0], .01)

    def test_live_speed_input_boundaries_change_only_arms(self):
        self.configure_teleop_speed(60)
        real = self.make_real(dry_run=True)
        auxiliary = real.teleop_speed_limits[~real.arm_mask].copy()
        home = real.home_speed_limits.copy()
        for speed in (np.int64(10), np.float32(120), 37.5):
            with self.subTest(speed=speed):
                self.assertTrue(real.set_teleop_speed(speed))
                self.assertAlmostEqual(real.teleop_speed_deg_s, speed)
                self.assertAlmostEqual(real.requested_teleop_speed_deg_s, speed)
                np.testing.assert_allclose(real.teleop_speed_limits[real.arm_mask], np.deg2rad(speed))
                np.testing.assert_array_equal(real.teleop_speed_limits[~real.arm_mask], auxiliary)
                np.testing.assert_array_equal(real.home_speed_limits, home)
                self.assertFalse(real.set_teleop_speed(speed))
        result = real.tick_grip(None, self.feedback(0.), now=0.)
        self.assertAlmostEqual(result["teleop_speed_deg_s"], 37.5)
        self.assertAlmostEqual(result["requested_teleop_speed_deg_s"], 37.5)
        self.assertFalse(result["speed_change_pending"])
        self.assertEqual(self.publisher.messages, [])

    def test_live_speed_rejects_invalid_input_without_changing_limits(self):
        self.configure_teleop_speed(60)
        real = self.make_real(dry_run=True)
        original = real.teleop_speed_limits.copy()
        for invalid in (True, False, np.bool_(True), None, "60", "", [], 60 + 0j, np.complex128(60), float("nan"),
                        float("inf"), -float("inf"), 0., 9.999, 120.001):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                real.set_teleop_speed(invalid)
            np.testing.assert_array_equal(real.teleop_speed_limits, original)
            self.assertAlmostEqual(real.teleop_speed_deg_s, 60.)
            self.assertAlmostEqual(real.requested_teleop_speed_deg_s, 60.)
            self.assertIsNone(real._pending_teleop_speed)

    def test_live_speed_increase_preserves_active_pose_session_and_ack_history(self):
        self.configure_teleop_speed(60)
        self.start_grip()
        real = self.real
        real.tick_grip(self.frame(2, grip=True, height=.08), self.armed_feedback(.02), now=.02)
        command, goal = self.planner.q_command.copy(), self.planner.q_goal.copy()
        home, limits = self.planner.q_home.copy(), real.home_speed_limits.copy()
        session, sequence, ack_seq = real.session_id, real.sequence, real._last_ack_seq
        sent, ack = real._sent_travel.copy(), real._ack_travel.copy()
        records = copy.deepcopy(real._sent_records)
        count = len(self.publisher.messages)
        self.assertTrue(real.set_teleop_speed(120.))
        self.assertTrue(real.armed)
        self.assertTrue(real._grip_engaged)
        self.assertEqual((real.session_id, real.sequence, real._last_ack_seq), (session, sequence, ack_seq))
        self.assertEqual(len(self.publisher.messages), count)
        np.testing.assert_array_equal(self.planner.q_command, command)
        np.testing.assert_array_equal(self.planner.q_goal, goal)
        np.testing.assert_array_equal(self.planner.q_home, home)
        np.testing.assert_array_equal(real.home_speed_limits, limits)
        np.testing.assert_array_equal(real._sent_travel, sent)
        np.testing.assert_array_equal(real._ack_travel, ack)
        self.assertEqual(real._sent_records.keys(), records.keys())
        for seq, record in records.items():
            self.assertEqual(real._sent_records[seq]["mode"], record["mode"])
            np.testing.assert_array_equal(real._sent_records[seq]["travel"], record["travel"])

    def test_live_speed_decrease_holds_until_previous_movement_is_acknowledged(self):
        self.configure_teleop_speed(120)
        self.start_grip()
        real = self.real
        self.planner.ik_period = .001
        target = self.planner.q_command.copy()
        joint = self.names.index("arm_l_joint1")
        target[joint] += .1
        with patch.object(self.planner, "_solve", return_value=target):
            moved = real.tick_grip(self.frame(2, grip=True), self.armed_feedback(.02), now=.02)
            previous = moved["q_command"].copy()
            self.assertGreater(real._sent_travel[joint] - real._ack_travel[joint], np.deg2rad(10) * .04)
            session = real.session_id
            self.assertTrue(real.set_teleop_speed(10.))
            self.assertAlmostEqual(real.teleop_speed_deg_s, 120.)
            self.assertAlmostEqual(real.requested_teleop_speed_deg_s, 10.)
            stale_ack = self.armed_feedback(.03, q=previous)
            stale_ack["last_command_seq"] = 1
            held = real.tick_grip(self.frame(3, grip=True), stale_ack, now=.03)
            self.assertEqual(held["state"], "ACTIVE", held["reason"])
            self.assertTrue(held["speed_change_pending"])
            self.assertTrue(held["speed_change_waiting_ack"])
            np.testing.assert_array_equal(held["q_command"], previous)
            np.testing.assert_array_equal(parse_joint_groups(self.publisher.messages[-1]["joints"]), previous)
            self.assertEqual(real.session_id, session)
            resumed = real.tick_grip(self.frame(4, grip=True), self.armed_feedback(.04, q=previous), now=.04)
        self.assertEqual(resumed["state"], "ACTIVE", resumed["reason"])
        self.assertFalse(resumed["speed_change_pending"])
        self.assertAlmostEqual(resumed["teleop_speed_deg_s"], 10.)
        delta = resumed["q_command"][joint] - previous[joint]
        self.assertGreater(delta, 0.)
        self.assertLessEqual(delta, np.deg2rad(10) * resumed["command_dt"] + 1e-12)
        self.assertEqual(real.session_id, session)

    def test_speed_change_does_not_interfere_with_one_grip_release(self):
        self.configure_teleop_speed(120)
        self.start_grip()
        real = self.real
        real.tick_grip(self.frame(2, grip=True, height=.08), self.armed_feedback(.02), now=.02)
        self.assertTrue(real.set_teleop_speed(10.))
        released = self.frame(3, grip=True)
        released["hands"]["left"]["grip"] = 0.
        result = real.tick_grip(released, self.armed_feedback(.03), now=.03)
        self.assertFalse(result["armed"])
        self.assertFalse(result["transmitted"])
        self.assertEqual(result["active"], [])
        self.assertEqual(self.publisher.messages[-1]["mode"], "stop")
        self.assertAlmostEqual(real.teleop_speed_deg_s, 10.)
        self.assertIsNone(real._pending_teleop_speed)

    def test_speed_change_during_home_preserves_ten_degree_return(self):
        self.configure_teleop_speed(60)
        real = self.make_real(dry_run=True, max_home_joint_speed=np.deg2rad(10))
        real.tick(None, self.feedback(0.), enabled=False, now=0.)
        self.planner.q_home[self.names.index("arm_l_joint1")] += .04
        result = real.tick(None, self.feedback(.01), enabled=True, now=.01, action="home")
        self.assertEqual(result["state"], "HOMING")
        home = real.home_speed_limits.copy()
        previous = result["q_command"].copy()
        for index, speed in enumerate((120., 10., 80.), start=2):
            self.assertTrue(real.set_teleop_speed(speed))
            np.testing.assert_array_equal(real.home_speed_limits, home)
            result = real.tick(None, self.feedback(index / 100), enabled=True, now=index / 100)
            self.assertEqual(result["state"], "HOMING", result["reason"])
            np.testing.assert_array_equal(self.planner.speed_limits, home)
            self.assertTrue(np.all(np.abs(result["q_command"] - previous) <= home * result["command_dt"] + 1e-12))
            previous = result["q_command"].copy()
        real.tick(None, self.feedback(.05), enabled=True, now=.05, action="cancel_home")
        np.testing.assert_allclose(self.planner.speed_limits[real.arm_mask], np.deg2rad(80))

    def gripper_indices(self):
        return [self.names.index(name) for name in ("gripper_l_joint1", "gripper_r_joint1")]

    def test_live_gripper_speed_boundaries_change_only_grippers(self):
        self.configure_teleop_speed(60)
        real = self.make_real(dry_run=True)
        others = real.teleop_speed_limits[~real.gripper_mask].copy()
        home = real.home_speed_limits.copy()
        np.testing.assert_array_equal(np.flatnonzero(real.gripper_mask), self.gripper_indices())
        for speed in (np.int64(10), np.float32(60), 100., 120., 37.5):
            with self.subTest(speed=speed):
                self.assertTrue(real.set_gripper_speed(speed))
                self.assertAlmostEqual(real.gripper_speed_deg_s, speed)
                self.assertAlmostEqual(real.requested_gripper_speed_deg_s, speed)
                np.testing.assert_allclose(real.teleop_speed_limits[real.gripper_mask], np.deg2rad(speed))
                np.testing.assert_array_equal(real.teleop_speed_limits[~real.gripper_mask], others)
                np.testing.assert_array_equal(real.home_speed_limits, home)
                self.assertAlmostEqual(real.teleop_speed_deg_s, 60.)
                self.assertFalse(real.set_gripper_speed(speed))
        result = real.tick_grip(None, self.feedback(0.), now=0.)
        self.assertAlmostEqual(result["gripper_speed_deg_s"], 37.5)
        self.assertAlmostEqual(result["requested_gripper_speed_deg_s"], 37.5)
        self.assertFalse(result["gripper_speed_change_pending"])
        self.assertAlmostEqual(result["teleop_speed_deg_s"], 60.)
        self.assertEqual(self.publisher.messages, [])

    def test_live_gripper_speed_rejects_invalid_input_and_bridge_cap(self):
        self.configure_teleop_speed(60)
        real = self.make_real(dry_run=True)
        original = real.teleop_speed_limits.copy()
        # Preview accepts 120; values beyond 120 remain invalid.
        for invalid in (True, False, np.bool_(True), None, "60", "", [], 60 + 0j, np.complex128(60), float("nan"),
                        float("inf"), -float("inf"), 0., 9.999, 120.001, 121.):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                real.set_gripper_speed(invalid)
            np.testing.assert_array_equal(real.teleop_speed_limits, original)
            self.assertAlmostEqual(real.gripper_speed_deg_s, 60.)
            self.assertIsNone(real._pending_gripper_speed)

    def test_live_gripper_speed_decrease_holds_until_previous_movement_is_acknowledged(self):
        self.configure_teleop_speed(60)
        self.start_grip()
        real = self.real
        self.planner.ik_period = .001
        joint = self.names.index("gripper_l_joint1")
        opening = self.frame(2, grip=True)
        opening["hands"]["left"]["trigger"] = 1.
        moved = real.tick_grip(opening, self.armed_feedback(.02), now=.02)
        previous = moved["q_command"].copy()
        self.assertGreater(real._sent_travel[joint] - real._ack_travel[joint], np.deg2rad(10) * .04)
        session = real.session_id
        self.assertTrue(real.set_gripper_speed(10.))
        self.assertAlmostEqual(real.gripper_speed_deg_s, 60.)
        self.assertAlmostEqual(real.requested_gripper_speed_deg_s, 10.)
        stale_ack = self.armed_feedback(.03, q=previous)
        stale_ack["last_command_seq"] = 1
        still_opening = self.frame(3, grip=True)
        still_opening["hands"]["left"]["trigger"] = 1.
        held = real.tick_grip(still_opening, stale_ack, now=.03)
        self.assertEqual(held["state"], "ACTIVE", held["reason"])
        self.assertTrue(held["gripper_speed_change_pending"])
        self.assertTrue(held["speed_change_waiting_ack"])
        self.assertFalse(held["speed_change_pending"])
        np.testing.assert_array_equal(held["q_command"], previous)
        self.assertEqual(real.session_id, session)
        resumed_frame = self.frame(4, grip=True)
        resumed_frame["hands"]["left"]["trigger"] = 1.
        resumed = real.tick_grip(resumed_frame, self.armed_feedback(.04, q=previous), now=.04)
        self.assertEqual(resumed["state"], "ACTIVE", resumed["reason"])
        self.assertFalse(resumed["gripper_speed_change_pending"])
        self.assertAlmostEqual(resumed["gripper_speed_deg_s"], 10.)
        self.assertAlmostEqual(resumed["teleop_speed_deg_s"], 60.)
        delta = resumed["q_command"][joint] - previous[joint]
        self.assertGreater(delta, 0.)
        self.assertLessEqual(delta, np.deg2rad(10) * resumed["command_dt"] + 1e-12)
        self.assertEqual(real.session_id, session)

    def test_gripper_increase_does_not_force_a_pending_arm_decrease(self):
        self.configure_teleop_speed(120)
        self.planner.speed_limits[self.gripper_indices()] = np.deg2rad(30)
        self.start_grip()
        real = self.real
        self.planner.ik_period = .001
        target = self.planner.q_command.copy()
        joint = self.names.index("arm_l_joint1")
        target[joint] += .1
        with patch.object(self.planner, "_solve", return_value=target):
            real.tick_grip(self.frame(2, grip=True), self.armed_feedback(.02), now=.02)
        self.assertGreater(real._sent_travel[joint] - real._ack_travel[joint], 0.)
        self.assertTrue(real.set_teleop_speed(10.))
        self.assertIsNotNone(real._pending_teleop_speed)
        self.assertTrue(real.set_gripper_speed(60.))
        self.assertAlmostEqual(real.gripper_speed_deg_s, 60.)
        self.assertIsNone(real._pending_gripper_speed)
        # The arm decrease must still wait for its acknowledgment.
        self.assertIsNotNone(real._pending_teleop_speed)
        self.assertAlmostEqual(real.teleop_speed_deg_s, 120.)

    def test_arm_increase_does_not_force_a_pending_gripper_decrease(self):
        self.configure_teleop_speed(60)
        self.start_grip()
        real = self.real
        self.planner.ik_period = .001
        joint = self.names.index("gripper_l_joint1")
        opening = self.frame(2, grip=True)
        opening["hands"]["left"]["trigger"] = 1.
        real.tick_grip(opening, self.armed_feedback(.02), now=.02)
        self.assertGreater(real._sent_travel[joint] - real._ack_travel[joint], 0.)
        self.assertTrue(real.set_gripper_speed(10.))
        self.assertIsNotNone(real._pending_gripper_speed)
        self.assertTrue(real.set_teleop_speed(120.))
        self.assertAlmostEqual(real.teleop_speed_deg_s, 120.)
        self.assertIsNone(real._pending_teleop_speed)
        # The gripper decrease must still wait for its acknowledgment.
        self.assertIsNotNone(real._pending_gripper_speed)
        self.assertAlmostEqual(real.gripper_speed_deg_s, 60.)

    def test_gripper_speed_change_during_home_preserves_ten_degree_return(self):
        self.configure_teleop_speed(60)
        real = self.make_real(dry_run=True, max_home_joint_speed=np.deg2rad(10))
        real.tick(None, self.feedback(0.), enabled=False, now=0.)
        self.planner.q_home[self.names.index("gripper_l_joint1")] += .04
        result = real.tick(None, self.feedback(.01), enabled=True, now=.01, action="home")
        self.assertEqual(result["state"], "HOMING")
        home = real.home_speed_limits.copy()
        np.testing.assert_allclose(home[real.gripper_mask], np.deg2rad(10))
        previous = result["q_command"].copy()
        for index, speed in enumerate((45., 10., 30.), start=2):
            self.assertTrue(real.set_gripper_speed(speed))
            np.testing.assert_array_equal(real.home_speed_limits, home)
            result = real.tick(None, self.feedback(index / 100), enabled=True, now=index / 100)
            self.assertEqual(result["state"], "HOMING", result["reason"])
            np.testing.assert_array_equal(self.planner.speed_limits, home)
            self.assertTrue(np.all(np.abs(result["q_command"] - previous) <= home * result["command_dt"] + 1e-12))
            previous = result["q_command"].copy()
        real.tick(None, self.feedback(.05), enabled=True, now=.05, action="cancel_home")
        np.testing.assert_allclose(self.planner.speed_limits[real.gripper_mask], np.deg2rad(30))

    def test_gripper_speed_is_not_sent_as_the_arm_speed(self):
        self.configure_teleop_speed(60)
        real = self.make_real(command_arm_speed_limit=True)
        real.session_id, real.bridge_id = "test-session", "test-bridge"
        real.set_gripper_speed(10.)
        real._set_motion_speed(False, now=0.)
        real._send("command", self.robot_initial.copy(), self.feedback(0.))
        self.assertAlmostEqual(self.publisher.messages[-1]["arm_speed_rad_s"], np.deg2rad(60))

    def test_requested_arm_speed_packet_uses_effective_ramp_and_excludes_home(self):
        self.configure_teleop_speed(60)
        real = self.make_real(command_arm_speed_limit=True, teleop_ramp_s=8.)
        real.session_id, real.bridge_id = "test-session", "test-bridge"
        q = self.robot_initial.copy()
        real._set_motion_speed(False, now=0.)
        real._send("arm", q, self.feedback(0.))
        self.assertNotIn("arm_speed_rad_s", self.publisher.messages[-1])
        real._send("command", q, self.feedback(0.))
        self.assertGreater(self.publisher.messages[-1]["arm_speed_rad_s"], 0.)
        self.assertLessEqual(self.publisher.messages[-1]["arm_speed_rad_s"], 1e-12)
        real.set_teleop_speed(120.)
        real._set_motion_speed(False, now=4.)
        real._send("command", q, self.feedback(4.))
        self.assertAlmostEqual(self.publisher.messages[-1]["arm_speed_rad_s"], np.deg2rad(60))
        real._set_motion_speed(True, now=4.01)
        real._send("command", q, self.feedback(4.01), motion="home")
        self.assertNotIn("arm_speed_rad_s", self.publisher.messages[-1])

    def test_unarmed_home_uses_ten_degrees_from_first_step_and_final_home_packet(self):
        self.configure_teleop_speed(60)
        real = self.make_real(max_home_joint_speed=np.deg2rad(10))
        real.tick(None, self.feedback(0.), enabled=False, now=0.)
        home = self.planner.q_home.copy()
        moved = home.copy()
        moved[self.names.index("arm_l_joint1")] += .04
        moved[self.names.index("lift_joint")] -= .001
        real.tick(None, self.feedback(.01, q=moved), enabled=False, now=.01)
        first = real.tick(None, self.feedback(.02, q=moved), enabled=True, now=.02, action="home")
        self.assertEqual(first["state"], "ARMING")
        np.testing.assert_array_equal(self.planner.speed_limits, real.home_speed_limits)
        np.testing.assert_array_equal(first["q_command"], moved)
        real.tick(None, self.feedback(.03, q=moved), enabled=True, now=.03)
        ack = real.tick(None, self.armed_feedback(.04, q=moved), enabled=True, now=.04)
        self.assertEqual(ack["state"], "HOMING")
        np.testing.assert_array_equal(self.planner.speed_limits, real.home_speed_limits)
        np.testing.assert_array_equal(ack["q_command"], moved)
        self.assertEqual(self.publisher.messages[-1]["motion"], "home")
        previous, peak, lift_peak = moved.copy(), 0., 0.
        arm, lift = self.names.index("arm_l_joint1"), self.names.index("lift_joint")
        for index in range(5, 100):
            now = index / 100
            result = real.tick(None, self.armed_feedback(now, q=previous), enabled=True, now=now)
            self.assertIn(result["state"], ("HOMING", "HOME"), result["reason"])
            delta = np.abs(result["q_command"] - previous)
            self.assertTrue(np.all(delta <= real.home_speed_limits * result["command_dt"] + 1e-12))
            # One coordinated limit: the lift and the arm stay on the straight
            # checked segment, arriving together instead of the lift finishing first.
            np.testing.assert_allclose(result["q_command"][[arm, lift]] - moved[[arm, lift]],
                                       (home - moved)[[arm, lift]]
                                       * ((result["q_command"][arm] - moved[arm]) / (home - moved)[arm]),
                                       rtol=0, atol=1e-12)
            peak = max(peak, float(np.max(delta[1:] / result["command_dt"])))
            lift_peak = max(lift_peak, float(delta[0] / result["command_dt"]))
            previous = result["q_command"].copy()
            if result["state"] == "HOME":
                break
        self.assertEqual(result["state"], "HOME")
        self.assertAlmostEqual(np.rad2deg(peak), 10., places=6)
        self.assertLess(lift_peak, .01)
        self.assertGreater(lift_peak, 0.)
        np.testing.assert_allclose(result["q_command"], home, atol=1e-9)
        commands = [packet for packet in self.publisher.messages if packet["mode"] == "command"]
        self.assertTrue(commands)
        self.assertTrue(all(packet["motion"] == "home" for packet in commands))
        self.assertEqual((self.publisher.messages[-2]["mode"], self.publisher.messages[-2]["motion"]),
                         ("command", "home"))
        self.assertEqual(self.publisher.messages[-1]["mode"], "stop")
        np.testing.assert_array_equal(self.planner.speed_limits, real.teleop_speed_limits)

    def test_home_speed_does_not_leak_after_cancel_off_or_mode_switch(self):
        for end in ("cancel", "off", "mode"):
            with self.subTest(end=end):
                self.configure_teleop_speed(60)
                real = self.make_real(dry_run=True, max_home_joint_speed=np.deg2rad(10))
                real.tick(None, self.feedback(0.), enabled=False, now=0.)
                self.planner.q_home[self.names.index("arm_l_joint1")] += .02
                result = real.tick(None, self.feedback(.01), enabled=True, now=.01, action="home")
                self.assertEqual(result["state"], "HOMING")
                np.testing.assert_array_equal(self.planner.speed_limits, real.home_speed_limits)
                if end == "mode":
                    real.set_dry_run(False, self.publisher_factory)
                else:
                    real.tick(None, self.feedback(.02), enabled=end == "cancel", now=.02,
                              action="cancel_home" if end == "cancel" else None)
                np.testing.assert_array_equal(self.planner.speed_limits, real.teleop_speed_limits)
                np.testing.assert_allclose(real.teleop_speed_limits[1:], np.deg2rad(60))
                np.testing.assert_allclose(real.home_speed_limits[1:], np.deg2rad(10))
                self.assertEqual(real.home_speed_limits[0], .01)
                self.assertFalse(real.home_pending)
                real.tick(None, self.feedback(.03), enabled=False, now=.03)
                result = real.tick(self.frame(4), self.feedback(.04), enabled=True, now=.04)
                self.assertEqual(result["state"], "ARMING" if end == "mode" else "READY")
                np.testing.assert_array_equal(self.planner.speed_limits, real.teleop_speed_limits)

    def test_home_speed_cannot_raise_lower_teleop_limit(self):
        real = self.make_real(dry_run=True, max_home_joint_speed=np.deg2rad(10))
        np.testing.assert_array_equal(real.home_speed_limits, real.teleop_speed_limits)
        for speed in (0., -1.):
            with self.assertRaises(ValueError):
                self.make_real(dry_run=True, max_home_joint_speed=speed)

    def test_command_baseline_needs_its_own_ack_and_retries_exact_pose(self):
        self.configure_teleop_speed(60)
        self.start_grip()
        baseline = self.planner.q_command.copy()
        for index in range(2, 9):
            feedback = self.armed_feedback(index / 100)
            feedback["last_command_seq"] = 0  # only the arm has been acknowledged
            result = self.real.tick_grip(self.frame(index, grip=True, height=.08), feedback, now=index / 100)
            self.assertEqual(result["state"], "ACTIVE")
            self.assertTrue(result["ack_paced"])
            self.assertEqual(result["command_scale"], 0.)
            np.testing.assert_array_equal(result["q_command"], baseline)
            np.testing.assert_array_equal(parse_joint_groups(self.publisher.messages[-1]["joints"]), baseline)
        result = self.real.tick_grip(self.frame(9, grip=True, height=.08), self.armed_feedback(.09), now=.09)
        self.assertEqual(result["state"], "ACTIVE")
        self.assertGreater(np.max(np.abs(result["q_command"] - baseline)), 1e-5)
        released = self.frame(10)
        self.real.tick_grip(released, self.armed_feedback(.1), now=.1)
        self.assertEqual(self.real._sent_records, {})
        self.assertIsNone(self.real._ack_travel)
        self.assertIsNone(self.real._last_command_q)
        self.assertTrue(np.all(self.real._sent_travel == 0.))
        restarted = self.real.tick_grip(self.frame(11, grip=True), self.feedback(.11), now=.11)
        self.assertEqual(restarted["state"], "ARMING")
        activated = self.real.tick_grip(self.frame(12, grip=True), self.armed_feedback(.12), now=.12)
        self.assertTrue(activated["ack_paced"], "New lease must not inherit the old command ACK")

    def test_regressed_or_unknown_ack_stops_before_new_motion(self):
        for kind in ("regressed", "unknown"):
            with self.subTest(kind=kind):
                self.start_grip()
                self.real.tick_grip(self.frame(2, grip=True), self.armed_feedback(.02), now=.02)
                feedback = self.armed_feedback(.03)
                feedback["last_command_seq"] = 0 if kind == "regressed" else self.real.sequence + 20
                count = len(self.publisher.messages)
                with patch.object(self.planner, "_solve") as solve:
                    result = self.real.tick_grip(self.frame(3, grip=True, height=.08), feedback, now=.03)
                solve.assert_not_called()
                self.assertEqual(result["state"], "FAULT")
                self.assertIn("acknowledgment", result["reason"])
                self.assertEqual([m["mode"] for m in self.publisher.messages[count:]], ["stop"])
                self.assertEqual(self.real._sent_records, {})

    def test_reversals_consume_cumulative_ack_budget_even_with_zero_net_motion(self):
        self.configure_teleop_speed(60)
        self.start_grip()
        baseline = self.planner.q_command.copy()
        self.planner.ik_period = .001
        joint = self.names.index("arm_l_joint1")
        for index in range(2, 7):
            target = baseline.copy()
            if index % 2 == 0:
                target[joint] += .1
            feedback = self.armed_feedback(index / 100)
            feedback["last_command_seq"] = 1
            with patch.object(self.planner, "_solve", return_value=target):
                result = self.real.tick_grip(self.frame(index, grip=True), feedback, now=index / 100)
            self.assertEqual(result["state"], "ACTIVE", result["reason"])
            if index == 6:
                self.assertTrue(result["ack_paced"])
                self.assertAlmostEqual(result["command_scale"], 0., places=8)
        np.testing.assert_allclose(result["q_command"], baseline, atol=1e-12)
        self.assertAlmostEqual(self.real._sent_travel[joint], np.deg2rad(60) * .04, places=12)
        target = baseline.copy()
        target[joint] += .1
        with patch.object(self.planner, "_solve", return_value=target):
            resumed = self.real.tick_grip(self.frame(7, grip=True), self.armed_feedback(.07), now=.07)
        self.assertGreater(resumed["q_command"][joint], baseline[joint])

    def test_measured_soft_pacing_prevents_runaway_but_preserves_hard_fault(self):
        self.configure_teleop_speed(60)
        self.start_grip()
        self.planner.ik_period = .001
        joint = self.names.index("arm_l_joint1")
        target = self.robot_initial.copy()
        target[joint] += .5
        saw_paced = False
        with patch.object(self.planner, "_solve", return_value=target):
            for index in range(2, 30):
                result = self.real.tick_grip(self.frame(index, grip=True), self.armed_feedback(index / 100), now=index / 100)
                self.assertEqual(result["state"], "ACTIVE", result["reason"])
                self.assertLessEqual(abs(result["q_command"][joint] - self.robot_initial[joint]), np.deg2rad(4) + 1e-12)
                saw_paced |= result["tracking_paced"]
        self.assertTrue(saw_paced)
        np.testing.assert_allclose(self.planner.env.get_qpos(joint_names=self.names), result["q_command"], atol=1e-12)
        measured = result["q_command"].copy()
        measured[joint] -= np.deg2rad(6)
        with patch.object(self.planner, "_solve") as solve:
            failed = self.real.tick_grip(self.frame(30, grip=True), self.armed_feedback(.3, q=measured), now=.3)
        solve.assert_not_called()
        self.assertEqual(failed["state"], "FAULT")
        self.assertIn("tracking difference", failed["reason"])

    def test_same_mode_does_not_interrupt_an_active_lease(self):
        self.arm()
        result = self.real.tick(self.frame(3, grip=True), self.armed_feedback(.03), enabled=True, now=.03)
        self.assertEqual(result["state"], "ACTIVE")
        session, sequence = self.real.session_id, self.real.sequence
        messages = len(self.publisher.messages)
        self.assertFalse(self.real.set_dry_run(False, self.publisher_factory))
        self.assertEqual(self.created_publishers, [])
        self.assertEqual(len(self.publisher.messages), messages)
        self.assertFalse(self.publisher.closed)
        self.assertEqual((self.real.session_id, self.real.sequence), (session, sequence))
        self.assertTrue(self.real.armed)
        self.assertIsNone(self.real.latched_reason)

    def test_active_real_to_dry_releases_before_closing_and_never_sends_shadow(self):
        self.arm()
        self.real.tick(self.frame(3, grip=True), self.armed_feedback(.03), enabled=True, now=.03)
        old = self.publisher
        home, speeds = self.planner.q_home.copy(), self.planner.speed_limits.copy()
        generation, bridge = self.real.generation, self.real.bridge_id
        events = []
        original_send, original_close = old.send, old.close
        def send(packet):
            events.append(packet["mode"])
            return original_send(packet)
        def close():
            events.append("close")
            original_close()
        def factory(*, enabled):
            events.append("factory:%s" % enabled)
            return self.publisher_factory(enabled=enabled)
        with patch.object(old, "send", side_effect=send), patch.object(old, "close", side_effect=close):
            self.assertTrue(self.real.set_dry_run(True, factory))
        self.assertEqual(events, ["stop", "close", "factory:False"])
        self.assertTrue(self.real.dry_run)
        self.assertFalse(self.real.publisher.enabled)
        self.assertTrue(old.closed)
        self.assertIsNone(self.real.session_id)
        self.assertIsNone(self.real.last_sent_q)
        self.assertEqual(self.real.sequence, 0)
        self.assertFalse(self.planner.homing)
        self.assertEqual((self.real.generation, self.real.bridge_id), (generation, bridge))
        np.testing.assert_array_equal(self.planner.q_home, home)
        np.testing.assert_array_equal(self.planner.speed_limits, speeds)
        blocked = self.real.tick(self.frame(4, grip=True), self.feedback(.04), enabled=True, now=.04)
        self.assertTrue(blocked["needs_disable"])
        self.assertEqual(blocked["active"], [])
        moved = self.robot_initial.copy()
        moved[self.names.index("arm_l_joint1")] += .01
        off = self.real.tick(self.frame(5), self.feedback(.05, q=moved), enabled=False, now=.05)
        self.assertFalse(off["needs_disable"])
        np.testing.assert_array_equal(off["q_command"], moved)
        np.testing.assert_array_equal(self.planner.q_home, home)
        self.real.tick(self.frame(6), self.feedback(.06, q=moved), enabled=True, now=.06)
        self.real.tick(self.frame(7, grip=True), self.feedback(.07, q=moved), enabled=True, now=.07)
        self.real.stop()
        self.assertEqual(self.real.publisher.messages, [])

    def test_dry_to_real_requires_off_release_and_new_bridge_lease(self):
        self.arm()
        previous_session = self.real.session_id
        home = self.planner.q_home.copy()
        self.real.set_dry_run(True, self.publisher_factory)
        dry_publisher = self.real.publisher
        self.assertTrue(self.real.set_dry_run(False, self.publisher_factory))
        self.assertTrue(dry_publisher.closed)
        self.assertEqual(dry_publisher.messages, [])
        live = self.real.publisher
        self.assertTrue(live.enabled)
        blocked = self.real.tick(self.frame(3), self.feedback(.03), enabled=True, now=.03)
        self.assertTrue(blocked["needs_disable"])
        self.assertEqual(live.messages, [])
        measured = self.robot_initial.copy()
        measured[self.names.index("arm_l_joint1")] += .015
        self.real.tick(self.frame(4, grip=True), self.feedback(.04, q=measured), enabled=False, now=.04)
        held = self.real.tick(self.frame(5, grip=True), self.feedback(.05, q=measured), enabled=True, now=.05)
        self.assertEqual(held["state"], "WAIT_RELEASE")
        self.assertEqual(live.messages, [])
        arming = self.real.tick(self.frame(6), self.feedback(.06, q=measured), enabled=True, now=.06)
        self.assertEqual(arming["state"], "ARMING")
        self.assertEqual(live.messages[-1]["seq"], 0)
        self.assertNotEqual(self.real.session_id, previous_session)
        np.testing.assert_array_equal(parse_joint_groups(live.messages[-1]["joints"]), measured)
        result = self.real.tick(self.frame(7), self.armed_feedback(.07, q=measured), enabled=True, now=.07)
        self.assertEqual(result["state"], "READY")
        self.assertEqual([packet["mode"] for packet in live.messages], ["arm", "command"])
        np.testing.assert_array_equal(self.planner.q_home, home)

    def test_switch_during_home_cancels_return_without_recapturing_home(self):
        self.arm()
        self.planner.q_home[self.names.index("arm_l_joint1")] += .02
        home = self.planner.q_home.copy()
        self.real.tick(None, self.armed_feedback(.03), enabled=True, now=.03, action="home")
        self.real.tick(None, self.feedback(.035), enabled=True, now=.035)
        result = self.real.tick(None, self.armed_feedback(.037), enabled=True, now=.037)
        self.assertEqual(result["state"], "HOMING")
        self.real.set_dry_run(True, self.publisher_factory)
        self.assertFalse(self.planner.homing)
        self.assertIsNone(self.real.home_wait_started)
        np.testing.assert_array_equal(self.planner.q_goal, self.planner.q_command)
        np.testing.assert_array_equal(self.planner.q_home, home)
        blocked = self.real.tick(None, self.feedback(.04), enabled=True, now=.04)
        self.assertTrue(blocked["needs_disable"])
        self.assertFalse(blocked["homing"])
        self.assertEqual(self.real.publisher.messages, [])

    def test_failed_real_factory_leaves_inert_dry_mode(self):
        self.make_real(dry_run=True)
        self.real.tick(self.frame(0), self.feedback(0.), enabled=False, now=0.)
        old, home = self.publisher, self.planner.q_home.copy()
        calls = []
        def failing_factory(*, enabled):
            calls.append(enabled)
            raise OSError("command port already in use")
        with self.assertRaisesRegex(RuntimeError, "port already in use"):
            self.real.set_dry_run(False, failing_factory)
        self.assertEqual(calls, [True])
        self.assertTrue(old.closed)
        self.assertEqual(old.messages, [])
        self.assertTrue(self.real.dry_run)
        self.assertFalse(self.real.publisher.enabled)
        self.assertIsNot(self.real.publisher, old)
        self.assertIsNone(getattr(self.real.publisher, "_socket", None))
        np.testing.assert_array_equal(self.planner.q_home, home)
        with patch.object(self.real.publisher, "send") as send:
            blocked = self.real.tick(self.frame(1), self.feedback(.01), enabled=True, now=.01)
            self.assertTrue(blocked["needs_disable"])
            self.real.tick(self.frame(2), self.feedback(.02), enabled=False, now=.02)
            self.real.tick(self.frame(3), self.feedback(.03), enabled=True, now=.03)
            self.real.stop()
            send.assert_not_called()

    def test_stop_enqueue_failure_still_closes_old_and_prevents_replacement(self):
        self.arm()
        old = self.publisher
        with patch.object(old, "send", return_value=False) as send:
            with self.assertRaisesRegex(RuntimeError, "stop could not be queued"):
                self.real.set_dry_run(True, self.publisher_factory)
        self.assertEqual(send.call_args.args[0]["mode"], "stop")
        self.assertTrue(old.closed)
        self.assertEqual(self.created_publishers, [])
        self.assertTrue(self.real.dry_run)
        self.assertFalse(self.real.publisher.enabled)
        self.assertIsNone(self.real.session_id)
        self.assertIsNone(self.real.last_sent_q)
        self.assertIsNotNone(self.real.latched_reason)

    def test_stop_exception_still_closes_old_and_cancels_motion(self):
        self.arm()
        with patch.object(self.publisher, "send", side_effect=RuntimeError("socket failed")):
            with self.assertRaisesRegex(RuntimeError, "socket failed"):
                self.real.set_dry_run(True, self.publisher_factory)
        self.assertTrue(self.publisher.closed)
        self.assertTrue(self.real.dry_run)
        self.assertFalse(self.real.armed)
        self.assertFalse(self.planner.homing)
        self.assertEqual(self.created_publishers, [])

    def test_close_failure_does_not_create_a_real_publisher(self):
        self.make_real(dry_run=True)
        with patch.object(self.publisher, "close", side_effect=RuntimeError("close failed")) as close:
            with self.assertRaisesRegex(RuntimeError, "close failed"):
                self.real.set_dry_run(False, self.publisher_factory)
        close.assert_called_once()
        self.assertEqual(self.created_publishers, [])
        self.assertTrue(self.real.dry_run)
        self.assertFalse(self.real.publisher.enabled)
        self.assertIsNotNone(self.real.latched_reason)
        old = self.publisher
        self.assertTrue(self.real.set_dry_run(False, self.publisher_factory))
        self.assertTrue(old.closed, "Failed cleanup must be retried before creating a live transport")
        self.assertEqual(len(self.created_publishers), 1)
        self.assertTrue(self.real.publisher.enabled)

    def test_failed_cleanup_remains_blocking_on_later_real_switch(self):
        self.make_real(dry_run=True)
        with patch.object(self.publisher, "close", side_effect=RuntimeError("close failed")) as close:
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "close failed"):
                    self.real.set_dry_run(False, self.publisher_factory)
            self.assertEqual(close.call_count, 2)
        self.assertTrue(self.real.dry_run)
        self.assertEqual(self.created_publishers, [])
        self.real.publisher.close()
        self.assertTrue(self.publisher.closed, "Final fallback cleanup must retain the original transport")

    def test_wrong_factory_mode_is_closed_before_dry_fallback(self):
        self.make_real()
        candidate = MemoryPublisher(enabled=True)
        with self.assertRaisesRegex(RuntimeError, "enabled state"):
            self.real.set_dry_run(True, lambda enabled: candidate)
        self.assertTrue(candidate.closed)
        self.assertTrue(self.real.dry_run)
        self.assertFalse(self.real.publisher.enabled)
        self.assertIsNot(self.real.publisher, candidate)

    def test_wrong_factory_cleanup_failure_is_retained_for_retry(self):
        self.make_real()
        candidate = MemoryPublisher(enabled=True)
        with patch.object(candidate, "close", side_effect=RuntimeError("candidate close failed")):
            with self.assertRaisesRegex(RuntimeError, "candidate close failed"):
                self.real.set_dry_run(True, lambda enabled: candidate)
        self.assertTrue(self.real.dry_run)
        self.assertFalse(self.real.publisher.enabled)
        self.real.publisher.close()
        self.assertTrue(candidate.closed)


    def test_manual_preview_execute_then_home_preserves_session_home(self):
        self.configure_teleop_speed(60)
        real = self.make_real()
        bridge = IntentSimBridge(self.robot_initial)
        real.publisher = bridge
        real.tick_grip(None, bridge.latest(0.), now=0.)
        home = self.planner.q_home.copy()
        preview = real.preview_joint_adjustment("arm_l_joint1", 5., bridge.latest(.001))
        self.assertEqual(bridge.gate.last_command_seq, -1)
        np.testing.assert_array_equal(self.planner.q_home, home)
        target = preview["q_target"]
        previous = bridge.q.copy()
        for i in range(1, 250):
            t = i * .01
            result = real.tick_grip(None, bridge.latest(t), now=t, action="joint_adjust" if i == 1 else None)
            self.assertNotIn(result["state"], ("FAULT", "HOLD", "WAIT_ROBOT"), result["reason"])
            self.assertTrue(np.all(np.abs(bridge.q - previous) <= real.home_speed_limits * .01 + 1e-10))
            previous = bridge.q.copy()
            if result["state"] == "HOME": break
        self.assertEqual(result["state"], "HOME", result["reason"])
        self.assertTrue(result["manual_adjustment"])
        np.testing.assert_allclose(bridge.q, target, atol=np.deg2rad(.5))
        np.testing.assert_array_equal(self.planner.q_home, home)
        for j in range(1, 250):
            t = (i + j) * .01
            result = real.tick_grip(None, bridge.latest(t), now=t, action="home" if j == 1 else None)
            if result["state"] == "HOME": break
        self.assertEqual(result["state"], "HOME", result["reason"])
        np.testing.assert_allclose(bridge.q, home, atol=np.deg2rad(.5))

    def test_manual_preview_blocked_and_stale_preview_never_arm(self):
        self.configure_teleop_speed(60)
        real = self.make_real()
        feedback = self.feedback(0.)
        real.tick_grip(None, feedback, now=0.)
        with patch.object(self.guard, "check", return_value={"accepted": False, "reason": "collision_clearance", "samples_checked": 1}):
            with self.assertRaisesRegex(ValueError, "blocked"):
                real.preview_joint_adjustment("arm_l_joint1", 5., feedback)
        self.assertIsNone(real._joint_preview)
        self.assertEqual(self.publisher.messages, [])
        real.preview_joint_adjustment("arm_l_joint1", 5., feedback)
        moved = self.robot_initial.copy(); moved[3] += .02
        result = real.tick_grip(None, self.feedback(.01, q=moved), now=.01, action="joint_adjust")
        self.assertEqual(result["state"], "HOLD")
        self.assertIn("preview expired", result["reason"])
        self.assertEqual(self.publisher.messages, [])

    def test_manual_cancel_and_feedback_loss_stop_without_resume(self):
        self.configure_teleop_speed(60)
        real = self.make_real()
        bridge = IntentSimBridge(self.robot_initial); real.publisher = bridge
        real.tick_grip(None, bridge.latest(0.), now=0.)
        real.preview_joint_adjustment("arm_l_joint1", 10., bridge.latest(.001))
        for i in range(1, 10):
            result = real.tick_grip(None, bridge.latest(i*.01), now=i*.01, action="joint_adjust" if i==1 else None)
        self.assertEqual(result["state"], "HOMING")
        result = real.tick_grip(None, bridge.latest(.10), now=.10, action="cancel_home")
        self.assertFalse(real.armed)
        self.assertEqual(bridge.gate.state, "STOPPED")
        q = bridge.q.copy()
        result = real.tick_grip(None, bridge.latest(.11), now=.11)
        np.testing.assert_array_equal(bridge.q, q)
        real.preview_joint_adjustment("arm_l_joint1", 10., bridge.latest(.12))
        for i in range(13, 20):
            real.tick_grip(None, bridge.latest(i*.01), now=i*.01, action="joint_adjust" if i==13 else None)
        bad = bridge.latest(.2); bad.update(ready=False, reason="stale_joint_state")
        result = real.tick_grip(None, bad, now=.2)
        self.assertEqual(result["state"], "WAIT_ROBOT")
        self.assertFalse(real.armed)
        self.assertEqual(bridge.gate.state, "STOPPED")

    def test_gripper_pacing_is_independent_only_outside_home(self):
        self.configure_teleop_speed(60)
        real = self.make_real()
        real._ack_travel = np.zeros(len(self.names))
        measured = self.robot_initial.copy()
        previous = measured.copy(); previous[-1] += np.deg2rad(4)
        candidate = previous.copy(); candidate[3] += .005; candidate[-1] += .005
        paced, info = real._pace_candidate(previous, candidate, measured)
        self.assertAlmostEqual(paced[3], candidate[3])
        self.assertAlmostEqual(paced[-1], previous[-1])
        self.assertEqual(info["gripper_command_scales"][-1], 0.)
        paced_home, _ = real._pace_candidate(previous, candidate, measured, home=True)
        np.testing.assert_allclose(paced_home, previous)
        previous = measured.copy(); candidate = previous.copy(); candidate[3] += .005; candidate[-1] += .005
        real._sent_travel[-1] = real.planner.speed_limits[-1] * real.ack_window_s
        paced, _ = real._pace_candidate(previous, candidate, measured)
        self.assertAlmostEqual(paced[3], candidate[3])
        self.assertAlmostEqual(paced[-1], previous[-1])

    def test_new_gripper_cap_negotiation_and_packet_field(self):
        self.configure_teleop_speed(60)
        real = self.make_real()
        with self.assertRaisesRegex(ValueError, "bridge"):
            real.set_gripper_speed(120.)
        bridge = IntentSimBridge(self.robot_initial)
        real.tick_grip(None, bridge.latest(0.), now=0.)
        self.assertTrue(real.set_gripper_speed(120.))
        real._set_motion_speed(False)
        real.session_id, real.bridge_id = "test", bridge.gate.bridge_id
        real._send("command", self.robot_initial, bridge.latest(.01))
        self.assertAlmostEqual(self.publisher.messages[-1]["gripper_speed_rad_s"], np.deg2rad(120))
        real._send("command", self.robot_initial, bridge.latest(.02), motion="home")
        self.assertNotIn("gripper_speed_rad_s", self.publisher.messages[-1])
        self.assertAlmostEqual(real.teleop_speed_deg_s, 60.)


    def test_old_bridge_still_allows_home_with_high_teleop_gripper_setting(self):
        self.configure_teleop_speed(60)
        self.planner.speed_limits[self.gripper_indices()] = np.deg2rad(120)
        real = self.make_real()
        real.tick_grip(None, self.feedback(0.), now=0.)
        # An old bridge has no gripper capability fields. HOME still uses 10.
        result = real.tick_grip(None, self.feedback(.01), now=.01, action="home")
        self.assertEqual(result["state"], "ARMING", result["reason"])
        self.assertTrue(real.home_pending)
        np.testing.assert_allclose(self.planner.speed_limits[real.gripper_mask], np.deg2rad(10))



if __name__ == "__main__":
    unittest.main()
