"""Robot command gate tests; no ROS, robot socket, MuJoCo or hardware required."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import ModuleType, SimpleNamespace
import xml.etree.ElementTree as ET

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from package.worker.fixed_quest_protocol import (
    JOINT_GROUPS, JOINT_NAMES, JOINT_LIMITS, JointStateTracker, StrictCommandGate,
    load_json_message, make_joint_groups, parse_joint_groups,
)


class FixedQuestProtocolTest(unittest.TestCase):
    def setUp(self):
        self.q = [0.] * len(JOINT_NAMES)
        for name in ("arm_l_joint4", "arm_r_joint4"):
            self.q[JOINT_NAMES.index(name)] = -math.pi / 2
        self.tracker = JointStateTracker(stream_id="test-feedback")
        self.tracker.update(JOINT_NAMES, self.q, 1000000000, 1.)
        self.gate = StrictCommandGate(self.tracker, bridge_id="test-bridge")

    def feedback(self, now, q=None):
        self.tracker.update(JOINT_NAMES, self.q if q is None else q, round(now * 1e9), now)

    def command(self, seq=0, mode="arm", now=1., q=None, session="operator-a", motion=None):
        packet = {"type": "sg2_command", "version": 1, "session_id": session,
                  "seq": seq, "bridge_id": self.gate.bridge_id, "feedback_time_s": now,
                  "valid_for_s": .2, "mode": mode, "joints": make_joint_groups(self.q if q is None else q)}
        if motion is not None:
            packet["motion"] = motion
        return packet

    def test_adjustable_profile_100hz_arm_bound_and_home_aux_caps(self):
        from package.worker.fixed_quest_protocol import FIXED_QUEST_VR_GATE_OPTIONS
        for joint, motion, speed in (("arm_l_joint1", "teleop", math.radians(120)),
                                    ("arm_r_joint1", "teleop", math.radians(120)),
                                    ("arm_l_joint1", "home", math.radians(10)),
                                    ("gripper_r_joint1", "teleop", math.radians(60))):
            for extra, accepted in ((0., True), (.005, False)):
                with self.subTest(joint=joint, motion=motion, extra=extra):
                    self.tracker = JointStateTracker()
                    self.feedback(1.)
                    self.gate = StrictCommandGate(self.tracker, **FIXED_QUEST_VR_GATE_OPTIONS)
                    self.assertTrue(self.gate.receive(self.command(), 1.)["accepted"])
                    q = self.q.copy(); q[JOINT_NAMES.index(joint)] += speed * .01 + extra
                    self.feedback(1.01)
                    result = self.gate.receive(self.command(1, "command", 1.01, q, motion=motion), 1.01)
                    self.assertEqual(result["accepted"], accepted, result)

    def test_official_arm_lead_is_bounded_and_home_keeps_five_degrees(self):
        from package.worker.fixed_quest_protocol import FIXED_QUEST_VR_GATE_OPTIONS
        self.gate = StrictCommandGate(self.tracker, **FIXED_QUEST_VR_GATE_OPTIONS)
        self.gate.receive(self.command(), 1.)
        for seq, value in enumerate((.08, .16, .24, .32, .40, .48, .50), 1):
            now = 1. + seq * .04
            self.feedback(now)
            q = self.q.copy(); q[3] = value
            result = self.gate.receive(self.command(seq, "command", now, q), now)
            self.assertEqual(result["accepted"], value <= .48, result)
        self.assertEqual(result["reason"], "tracking_lag")
        # The same accepted teleop pose cannot bypass the narrower home guard.
        q[3] = .48
        result = self.gate.receive(self.command(8, "command", now, q, motion="home"), now)
        self.assertEqual(result["reason"], "tracking_lag")

    def test_selected_speed_is_enforced_on_compressed_arrival(self):
        from package.worker.fixed_quest_protocol import FIXED_QUEST_VR_GATE_OPTIONS
        self.gate = StrictCommandGate(self.tracker, **FIXED_QUEST_VR_GATE_OPTIONS)
        self.gate.receive(self.command(), 1.)
        q = self.q.copy(); q[3] += math.radians(60) * .02
        packet = self.command(1, "command", 1.01, q)
        packet["arm_speed_rad_s"] = math.radians(60)
        result = self.gate.receive(packet, 1.01)
        self.assertEqual(result["reason"], "joint_rate_limit")
        self.assertAlmostEqual(self.gate.rejection_detail["speed_limit"], math.radians(60))
        packet["arm_speed_rad_s"] = math.radians(120)
        self.assertTrue(self.gate.receive(packet, 1.01)["accepted"])

    def test_selected_speed_range_and_home_cannot_bypass_limit(self):
        from package.worker.fixed_quest_protocol import FIXED_QUEST_VR_GATE_OPTIONS
        self.gate = StrictCommandGate(self.tracker, **FIXED_QUEST_VR_GATE_OPTIONS)
        self.gate.receive(self.command(), 1.)
        for value in (0, -1, True, "1", float("nan"), float("inf"), math.radians(121)):
            with self.subTest(value=value):
                packet = self.command(1, "command", 1.02)
                packet["arm_speed_rad_s"] = value
                result = self.gate.receive(packet, 1.02)
                self.assertFalse(result["accepted"])
                self.assertEqual(self.gate.last_command_seq, 0)
        q = self.q.copy(); q[3] += math.radians(120) * .02
        packet = self.command(1, "command", 1.02, q, motion="home")
        packet["arm_speed_rad_s"] = math.radians(120)
        self.assertEqual(self.gate.receive(packet, 1.02)["reason"], "joint_rate_limit")
        packet["motion"] = "teleop"
        self.assertTrue(self.gate.receive(packet, 1.02)["accepted"])

    def test_ranges_match_model_with_conservative_wrist_intersection(self):
        root = ET.parse(PROJECT_DIR / "xml/ffw_sg2.xml").getroot()
        defaults = {element.get("class"): element.find("joint") for element in root.iter("default")}
        for name, limits in JOINT_LIMITS.items():
            source = defaults["gripper_joint" if name.startswith("gripper_") else name]
            xml_limits = tuple(float(value) for value in source.get("range").split())
            if name.endswith("joint7"):
                self.assertEqual(limits, (-1.5804, 1.5804))
                self.assertGreaterEqual(limits[0], xml_limits[0])
                self.assertLessEqual(limits[1], xml_limits[1])
            else:
                self.assertEqual(limits, xml_limits)
        self.assertEqual(len(JOINT_NAMES), 19)

    def test_feedback_names_define_order_and_groups_are_complete(self):
        groups = make_joint_groups(self.q, named=True)
        groups["arm_l"]["name"].reverse()
        groups["arm_l"]["position"].reverse()
        self.assertEqual(parse_joint_groups(groups, named=True), self.q)
        del groups["lift"]
        with self.assertRaises(ValueError):
            parse_joint_groups(groups, named=True)
        tracker = JointStateTracker()
        tracker.update(["head_joint1"], [.1], 1, 1.)
        snapshot = tracker.snapshot(1.)
        self.assertFalse(snapshot["ready"])
        self.assertEqual(snapshot["joints"], {})
        self.assertIn("lift_joint", snapshot["missing_joints"])

    def test_json_rejects_nonfinite_constants_and_binary_payloads(self):
        self.assertEqual(load_json_message(json.dumps({"version": 1})), {"version": 1})
        for payload in ('{"x":NaN}', '{"x":Infinity}', '[1,2]', b'\x80\x04pickle', 'x' * 65537):
            with self.subTest(payload=str(payload)[:25]), self.assertRaises((ValueError, UnicodeError)):
                load_json_message(payload)

    def test_duplicate_source_stamp_does_not_refresh_feedback(self):
        before = self.tracker.snapshot(1.)
        self.assertFalse(self.tracker.update(JOINT_NAMES, self.q, 1000000000, 1.15))
        after = self.tracker.snapshot(1.21)
        self.assertEqual(after["seq"], before["seq"])
        self.assertEqual(after["source_time_s"], 1.)
        self.assertFalse(after["ready"])
        self.assertEqual(after["state"], "stale_joint_state")

    def test_partial_updates_preserve_oldest_required_joint_age(self):
        self.tracker.update(JOINT_GROUPS["arm_l"], self.q[3:10], 1100000000, 1.1)
        snapshot = self.tracker.snapshot(1.21)
        self.assertEqual(snapshot["seq"], 2)
        self.assertEqual(snapshot["source_time_s"], 1.)
        self.assertFalse(snapshot["ready"])

    def test_bad_feedback_is_atomic_and_zero_stamp_not_fresh(self):
        before = self.tracker.snapshot(1.)
        q = self.q.copy()
        q[-1] = float("nan")
        with self.assertRaises(ValueError):
            self.tracker.update(JOINT_NAMES, q, 1100000000, 1.1)
        with self.assertRaises(ValueError):
            self.tracker.update(JOINT_NAMES, self.q, 0, 1.1)
        after = self.tracker.snapshot(1.)
        diagnostic = after.pop("joint_state_diagnostics")
        before.pop("joint_state_diagnostics")
        # Rejection observations may advance, but valid measurements, sequence
        # and source freshness must remain exactly unchanged.
        self.assertEqual(after, before)
        self.assertEqual(diagnostic["last_callback_result"], "rejected")
        self.assertEqual(diagnostic["last_rejection"]["reason"], "invalid_source_stamp")

    def test_arm_uses_measured_hold_then_accepts_bounded_command(self):
        near = self.q.copy()
        near[3] = .005
        armed = self.gate.receive(self.command(q=near), 1.)
        self.assertTrue(armed["accepted"])
        self.assertEqual(parse_joint_groups(armed["publish_joints"]), self.q)
        self.assertEqual(self.gate.status(1.)["state"], "ACTIVE")
        moved = self.q.copy()
        moved[3] += math.radians(5) * .02
        self.feedback(1.02)
        result = self.gate.receive(self.command(1, "command", 1.02, moved), 1.02)
        self.assertTrue(result["accepted"], result)
        self.assertEqual(parse_joint_groups(result["publish_joints"]), moved)

    def test_repeated_arm_only_during_acknowledgment_and_single_owner(self):
        self.assertTrue(self.gate.receive(self.command(), 1.)["accepted"])
        self.feedback(1.02)
        self.assertTrue(self.gate.receive(self.command(1, now=1.02), 1.02)["accepted"])
        other = self.gate.receive(self.command(0, now=1.02, session="operator-b"), 1.02)
        self.assertEqual(other["reason"], "owner_conflict")
        self.assertEqual(self.gate.owner_session_id, "operator-a")
        self.feedback(1.04)
        self.assertTrue(self.gate.receive(self.command(2, "command", 1.04), 1.04)["accepted"])
        rejected = self.gate.receive(self.command(3, now=1.04), 1.04)
        self.assertEqual(rejected["reason"], "already_armed")

    def test_repeated_command_sequence_never_refreshes_lease(self):
        arm = self.command()
        self.gate.receive(arm, 1.)
        self.assertEqual(self.gate.receive(arm, 1.1)["reason"], "repeated_sequence")
        self.assertEqual(self.gate.deadline, 1.2)

    def test_malformed_or_out_of_range_commands_do_not_publish(self):
        for label, mutate in (
            ("nan", lambda packet: packet["joints"]["arm_l"].__setitem__(0, float("nan"))),
            ("bool", lambda packet: packet["joints"]["arm_l"].__setitem__(0, True)),
            ("limit", lambda packet: packet["joints"]["arm_l"].__setitem__(6, 1.7)),
            ("missing", lambda packet: packet["joints"].pop("head")),
            ("bridge", lambda packet: packet.__setitem__("bridge_id", "old-bridge")),
            ("ttl", lambda packet: packet.__setitem__("valid_for_s", .3)),
        ):
            packet = self.command()
            mutate(packet)
            with self.subTest(label=label):
                result = self.gate.receive(packet, 1.)
                self.assertFalse(result["accepted"])
                self.assertIsNone(result["publish_joints"])
                self.assertIsNone(self.gate.owner_session_id)

    def test_arm_pose_and_feedback_echo_must_match_current_robot(self):
        far = self.q.copy()
        far[3] = .1
        self.assertEqual(self.gate.receive(self.command(q=far), 1.)["reason"], "arm_pose_mismatch")
        self.feedback(1.3)
        self.assertEqual(self.gate.receive(self.command(now=1.), 1.3)["reason"], "feedback_echo_expired")
        self.assertEqual(self.gate.receive(self.command(now=1.31), 1.3)["reason"], "feedback_echo_expired")

    def test_rate_limit_caps_long_tick_and_lift_units(self):
        self.gate.receive(self.command(), 1.)
        self.feedback(1.15)
        for motion, degrees in (("teleop", 60), ("home", 10)):
            for joint, movement in (("arm_l_joint1", math.radians(degrees) * .15), ("lift_joint", -.001)):
                with self.subTest(motion=motion, joint=joint):
                    q = self.q.copy()
                    q[JOINT_NAMES.index(joint)] += movement
                    result = self.gate.receive(self.command(1, "command", 1.15, q, motion=motion), 1.15)
                    self.assertEqual(result["reason"], "joint_rate_limit")
                    self.assertIsNone(result["publish_joints"])

    def test_teleop_60_degree_rate_and_missing_motion_compatibility(self):
        for motion in (None, "teleop"):
            with self.subTest(motion=motion):
                self.gate = StrictCommandGate(self.tracker, bridge_id="test-bridge")
                self.assertTrue(self.gate.receive(self.command(), 1.)["accepted"])
                q = self.q.copy()
                q[3] += math.radians(60) * .02
                result = self.gate.receive(self.command(1, "command", 1.02, q, motion=motion), 1.02)
                self.assertTrue(result["accepted"], result)
                # The existing .001 rad tolerance is additive, so use a clear
                # exceedance rather than assuming 61 degrees/s always rejects.
                q[3] += math.radians(70) * .02
                result = self.gate.receive(self.command(2, "command", 1.04, q, motion=motion), 1.04)
                self.assertEqual(result["reason"], "joint_rate_limit")
                self.assertIsNone(result["publish_joints"])

    def test_home_uses_10_degree_rate_on_first_command_after_arm(self):
        self.assertTrue(self.gate.receive(self.command(), 1.)["accepted"])
        q = self.q.copy()
        q[3] += math.radians(60) * .02
        rejected = self.gate.receive(self.command(1, "command", 1.02, q, motion="home"), 1.02)
        self.assertEqual(rejected["reason"], "joint_rate_limit")
        self.assertEqual(self.gate.last_command_seq, 0)
        q[3] = math.radians(10) * .02
        accepted = self.gate.receive(self.command(1, "command", 1.02, q, motion="home"), 1.02)
        self.assertTrue(accepted["accepted"], accepted)

    def test_motion_switch_changes_each_packet_limit_without_rearming(self):
        self.assertTrue(self.gate.receive(self.command(motion="home"), 1.)["accepted"])
        q = self.q.copy()
        for seq, (motion, degrees) in enumerate((("teleop", 60), ("home", 10), ("teleop", 60)), 1):
            now = 1. + seq * .02
            candidate = q.copy()
            candidate[3] += math.radians(60) * .02
            if motion == "home":
                rejected = self.gate.receive(self.command(seq, "command", now, candidate, motion=motion), now)
                self.assertEqual(rejected["reason"], "joint_rate_limit")
                self.assertEqual(self.gate.last_command_seq, seq - 1)
            candidate[3] = q[3] + math.radians(degrees) * .02
            result = self.gate.receive(self.command(seq, "command", now, candidate, motion=motion), now)
            self.assertTrue(result["accepted"], result)
            q = candidate
        self.assertEqual(self.gate.owner_session_id, "operator-a")

    def test_home_respects_lower_global_speed_and_lift_rate_is_unchanged(self):
        for motion in ("teleop", "home"):
            with self.subTest(motion=motion):
                self.gate = StrictCommandGate(self.tracker, bridge_id="test-bridge", max_joint_speed=math.radians(5))
                self.assertTrue(self.gate.receive(self.command(), 1.)["accepted"])
                q = self.q.copy()
                q[3] += math.radians(10) * .05
                result = self.gate.receive(self.command(1, "command", 1.05, q, motion=motion), 1.05)
                self.assertEqual(result["reason"], "joint_rate_limit")
                q[3] = math.radians(5) * .05
                q[0] = -.01 * .05
                result = self.gate.receive(self.command(1, "command", 1.05, q, motion=motion), 1.05)
                self.assertTrue(result["accepted"], result)
                q[0] -= .001
                result = self.gate.receive(self.command(2, "command", 1.10, q, motion=motion), 1.10)
                self.assertEqual(result["reason"], "joint_rate_limit")

    def test_malformed_motion_never_refreshes_lease_or_publishes(self):
        for mode in ("arm", "command", "stop"):
            for value in (None, True, 1, "", "HOME", "other", [], {}):
                with self.subTest(mode=mode, motion=value):
                    self.gate = StrictCommandGate(self.tracker, bridge_id="test-bridge")
                    self.assertTrue(self.gate.receive(self.command(), 1.)["accepted"])
                    packet = self.command(1, mode, 1.02)
                    packet["motion"] = value
                    result = self.gate.receive(packet, 1.02)
                    self.assertEqual(result["reason"], "invalid_motion")
                    self.assertIsNone(result["publish_joints"])
                    self.assertEqual(self.gate.last_command_seq, 0)
                    self.assertEqual(self.gate.deadline, 1.2)
                    self.assertEqual(self.gate.owner_session_id, "operator-a")

    def test_home_speed_configuration_must_be_positive_and_finite(self):
        for value in (0, -1, None, True, math.nan, math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                StrictCommandGate(self.tracker, max_home_joint_speed=value)

    def test_status_reports_effective_limits_as_detached_numeric_data(self):
        expected = {
            "teleop_joint_speed_rad_s": math.radians(60),
            "teleop_arm_joint_speed_rad_s": math.radians(60), "max_arm_tracking_lag_rad": math.radians(5),
            "command_arm_speed_limit": True,
            "teleop_gripper_joint_speed_rad_s": math.radians(60),
            "command_gripper_speed_limit": True,
            "home_joint_speed_rad_s": math.radians(10), "lift_speed_m_s": .01,
            "max_step_dt_s": .05, "max_tracking_lag_rad": math.radians(5), "max_lift_lag_m": .02,
        }
        status = self.gate.status(1.)
        self.assertEqual(status["limits"], expected)
        self.assertIsNone(status["rejection_detail"])
        self.assertEqual(json.loads(json.dumps(status, allow_nan=False)), status)
        status["limits"]["home_joint_speed_rad_s"] = 999.
        self.assertEqual(self.gate.status(1.)["limits"], expected)
        custom = StrictCommandGate(self.tracker, max_joint_speed=math.radians(4), max_home_joint_speed=math.radians(8),
                                   max_lift_speed=.007, max_step_dt=.03, max_tracking_lag=.04, max_lift_lag=.005)
        self.assertEqual(custom.status(1.)["limits"], {
            "teleop_arm_joint_speed_rad_s": math.radians(4), "max_arm_tracking_lag_rad": .04,
            "command_arm_speed_limit": True,
            "teleop_gripper_joint_speed_rad_s": math.radians(4),
            "command_gripper_speed_limit": True,
            "teleop_joint_speed_rad_s": math.radians(4), "home_joint_speed_rad_s": math.radians(4),
            "lift_speed_m_s": .007, "max_step_dt_s": .03, "max_tracking_lag_rad": .04, "max_lift_lag_m": .005,
        })

    def test_rate_rejection_detail_matches_comparison_without_refreshing_lease(self):
        for motion, joint, elapsed, movement, speed, tolerance in (
            ("teleop", "arm_l_joint1", .15, math.radians(60) * .15, math.radians(60), .001),
            ("home", "arm_r_joint1", .02, -math.radians(60) * .02, math.radians(10), .001),
            ("home", "lift_joint", .15, -.001, .01, .0001),
        ):
            with self.subTest(motion=motion, joint=joint):
                self.gate = StrictCommandGate(self.tracker, bridge_id="test-bridge")
                self.assertTrue(self.gate.receive(self.command(), 1.)["accepted"])
                q = self.q.copy()
                q[JOINT_NAMES.index(joint)] += movement
                result = self.gate.receive(self.command(1, "command", 1. + elapsed, q, motion=motion), 1. + elapsed)
                self.assertEqual(result, {"accepted": False, "reason": "joint_rate_limit", "publish_joints": None})
                detail = self.gate.status(1. + elapsed)["rejection_detail"]
                self.assertEqual(detail["joint_name"], joint)
                self.assertEqual(detail["motion"], motion)
                self.assertEqual(detail["unit"], "m" if joint == "lift_joint" else "rad")
                self.assertEqual((detail["command_seq"], detail["last_accepted_seq"]), (1, 0))
                self.assertAlmostEqual(detail["delta"], abs(movement))
                self.assertAlmostEqual(detail["elapsed_dt_s"], elapsed)
                self.assertAlmostEqual(detail["capped_dt_s"], min(elapsed, .05))
                self.assertAlmostEqual(detail["allowed_step"], speed * min(elapsed, .05) + tolerance)
                self.assertEqual(detail["speed_limit"], speed)
                self.assertEqual(detail["tolerance"], tolerance)
                self.assertGreater(detail["delta"], detail["allowed_step"])
                self.assertEqual(self.gate.owner_session_id, "operator-a")
                self.assertEqual(self.gate.last_command_seq, 0)
                self.assertEqual(self.gate.last_q, self.q)
                self.assertEqual(self.gate.last_command_at, 1.)
                self.assertEqual(self.gate.deadline, 1.2)
                detail["joint_name"] = "mutated"
                self.assertEqual(self.gate.status(1. + elapsed)["rejection_detail"]["joint_name"], joint)

    def test_latest_rate_detail_survives_other_reject_and_clears_on_accept(self):
        self.gate.receive(self.command(), 1.)
        q = self.q.copy()
        q[3], q[10] = .03, -.03
        self.assertEqual(self.gate.receive(self.command(1, "command", 1.02, q), 1.02)["reason"], "joint_rate_limit")
        first = self.gate.status(1.02)["rejection_detail"]
        self.assertEqual(first["joint_name"], "arm_l_joint1")  # First failing joint in protocol order.
        invalid = self.command(1, "command", 1.03)
        invalid["motion"] = "invalid"
        self.assertEqual(self.gate.receive(invalid, 1.03)["reason"], "invalid_motion")
        self.assertEqual(self.gate.status(1.03)["rejection_detail"], first)
        q[3] = 0.
        self.assertEqual(self.gate.receive(self.command(2, "command", 1.04, q, motion="home"), 1.04)["reason"], "joint_rate_limit")
        latest = self.gate.status(1.04)["rejection_detail"]
        self.assertEqual((latest["joint_name"], latest["command_seq"], latest["motion"]), ("arm_r_joint1", 2, "home"))
        # Rejected packet sequences remain reusable; acceptance clears diagnostics.
        self.assertTrue(self.gate.receive(self.command(1, "command", 1.05), 1.05)["accepted"])
        self.assertIsNone(self.gate.status(1.05)["rejection_detail"])

    def test_hold_and_expiry_clear_rate_detail_without_changing_hold_behavior(self):
        for stop in (False, True):
            with self.subTest(stop=stop):
                self.tracker = JointStateTracker(stream_id="test-feedback")
                self.tracker.update(JOINT_NAMES, self.q, 1000000000, 1.)
                self.gate = StrictCommandGate(self.tracker, bridge_id="test-bridge")
                self.gate.receive(self.command(), 1.)
                q = self.q.copy()
                q[3] = .08
                self.gate.receive(self.command(1, "command", 1.15, q), 1.15)
                self.assertIsNotNone(self.gate.status(1.15)["rejection_detail"])
                self.feedback(1.19)
                if stop:
                    result = self.gate.receive(self.command(1, "stop", 0.), 1.19)
                    self.assertTrue(result["accepted"])
                    self.assertEqual(result["reason"], "stopped")
                else:
                    result = self.gate.expire(1.21)
                    self.assertFalse(result["accepted"])
                    self.assertEqual(result["reason"], "command_expired")
                    self.assertIsNone(self.gate.expire(1.22)["publish_joints"])
                self.assertEqual(parse_joint_groups(result["publish_joints"]), self.q)
                self.assertIsNone(self.gate.owner_session_id)
                self.assertIsNone(self.gate.status(1.21)["rejection_detail"])

    def test_persistent_measured_lag_blocks_even_small_command_steps(self):
        self.gate.receive(self.command(), 1.)
        for seq in range(1, 46):
            now = 1. + seq * .02
            self.feedback(now)
            q = self.q.copy()
            q[3] += seq * .002
            result = self.gate.receive(self.command(seq, "command", now, q), now)
            if not result["accepted"]:
                break
        self.assertEqual(result["reason"], "tracking_lag")
        self.assertIsNone(result["publish_joints"])

    def test_expiry_holds_once_and_requires_new_explicit_arm_session(self):
        self.gate.receive(self.command(), 1.)
        self.feedback(1.19)
        expired = self.gate.expire(1.21)
        self.assertEqual(expired["reason"], "command_expired")
        self.assertEqual(parse_joint_groups(expired["publish_joints"]), self.q)
        self.assertIsNone(self.gate.expire(1.22)["publish_joints"])
        self.assertEqual(self.gate.state, "EXPIRED")
        self.assertEqual(self.gate.receive(self.command(1, "command", 1.19), 1.22)["reason"], "explicit_arm_required")
        self.assertEqual(self.gate.receive(self.command(2, "arm", 1.19), 1.22)["reason"], "new_session_required")
        self.assertTrue(self.gate.receive(self.command(0, "arm", 1.19, session="new-session"), 1.22)["accepted"])

    def test_expiry_without_fresh_measurement_sends_nothing(self):
        self.gate.receive(self.command(), 1.)
        expired = self.gate.expire(1.21)
        self.assertIsNone(expired["publish_joints"])
        self.assertEqual(self.gate.state, "EXPIRED")

    def test_owner_stop_accepts_stale_echo_and_releases_ownership(self):
        self.gate.receive(self.command(), 1.)
        packet = self.command(1, "stop", 0.)
        result = self.gate.receive(packet, 1.1)
        self.assertTrue(result["accepted"])
        self.assertEqual(parse_joint_groups(result["publish_joints"]), self.q)
        self.assertIsNone(self.gate.owner_session_id)
        self.assertEqual(self.gate.state, "STOPPED")

    @staticmethod
    def load_worker_with_ros_stubs(name):
        class Message:
            def __init__(self, **kwargs):
                self.header = SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0))
                self.__dict__.update(kwargs)
        modules = {}
        for module_name in ("rclpy", "rclpy.node", "rclpy.qos", "builtin_interfaces", "builtin_interfaces.msg",
                            "trajectory_msgs", "trajectory_msgs.msg", "sensor_msgs", "sensor_msgs.msg", "std_msgs", "std_msgs.msg"):
            modules[module_name] = ModuleType(module_name)
        modules["rclpy.node"].Node = object
        modules["rclpy.qos"].qos_profile_sensor_data = object()
        for module_name, symbols in (("builtin_interfaces.msg", ["Duration"]),
                                     ("trajectory_msgs.msg", ["JointTrajectory", "JointTrajectoryPoint"]),
                                     ("sensor_msgs.msg", ["JointState"]), ("std_msgs.msg", ["String"])):
            for symbol in symbols:
                setattr(modules[module_name], symbol, Message)
        path = PROJECT_DIR / "package/worker" / (name + ".py")
        spec = importlib.util.spec_from_file_location("package.worker._test_" + name, path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)
        return module

    def test_inbound_strict_json_routes_existing_topics_and_expires_once(self):
        module = self.load_worker_with_ros_stubs("inbound")
        node = module.SG2ZMQRosBridge.__new__(module.SG2ZMQRosBridge)
        node.strict, node.gate, node.tracker = True, self.gate, self.tracker
        node.report_warning = lambda message: None
        published = {group: [] for group in module.CONTROLLER_CONFIG}
        node.traj_pubs = {group: SimpleNamespace(publish=rows.append) for group, rows in published.items()}
        node.socket = SimpleNamespace(recv=lambda **kwargs: json.dumps(self.command()).encode())
        with patch.object(module.time, "monotonic", return_value=1.):
            node.on_timer()
        sent_names = [name for rows in published.values() for name in rows[0].joint_names]
        self.assertEqual(set(sent_names), set(JOINT_NAMES))
        self.assertEqual(len(sent_names), 19)
        self.assertEqual([len(rows[0].joint_names) for rows in published.values()], [8, 8, 2, 1])
        self.feedback(1.19)
        def empty_socket(**kwargs):
            raise module.zmq.Again()
        node.socket = SimpleNamespace(recv=empty_socket)
        for now in (1.21, 1.22):
            with patch.object(module.time, "monotonic", return_value=now):
                node.on_timer()
        self.assertEqual([len(rows) for rows in published.values()], [2, 2, 2, 2])

    def test_outbound_keeps_source_age_and_forwards_bridge_unchanged(self):
        module = self.load_worker_with_ros_stubs("outbound")
        node = module.SG2OutboundZMQ.__new__(module.SG2OutboundZMQ)
        node.strict, node.tracker = True, self.tracker
        node.bridge = self.gate.status(1.)
        sent = []
        node.socket = SimpleNamespace(send_json=lambda data, **kwargs: sent.append(copy.deepcopy(data)))
        for now in (1.1, 1.3):
            with patch.object(module.time, "monotonic", return_value=now):
                node.timer_callback()
        self.assertEqual(sent[0]["seq"], sent[1]["seq"])
        self.assertEqual(sent[1]["source_time_s"], 1.)
        self.assertFalse(sent[1]["ready"])
        self.assertEqual(sent[1]["bridge"], node.bridge)
        self.assertAlmostEqual(sent[1]["bridge_age_s"], .3)


if __name__ == "__main__":
    unittest.main()
