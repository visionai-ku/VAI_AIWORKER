"""Offline feedback protocol diagnostics; never deserialize network pickle."""
from collections import deque
import json
from pathlib import Path
import pickle
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import zmq

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from package.fixed_quest_transport import SG2RobotStateReceiver, sample_control_inputs
from package.worker.fixed_quest_protocol import JOINT_NAMES, JointStateTracker, StrictCommandGate


class _ByteSocket:
    def __init__(self):
        self.packets = deque()

    def recv(self, **kwargs):
        if self.packets:
            return self.packets.popleft()
        raise zmq.Again()

    def recv_pyobj(self, **kwargs):
        raise AssertionError("legacy feedback must never be deserialized")


class FeedbackProtocolDetectionTest(unittest.TestCase):
    def setUp(self):
        self.receiver = SG2RobotStateReceiver("192.0.2.1")
        # Install an in-memory byte queue without constructing a ZMQ context or
        # connecting to any address.
        self.socket = self.receiver._socket = _ByteSocket()
        self.tracker = JointStateTracker()
        self.gate = StrictCommandGate(self.tracker)

    def strict_packet(self, robot_now=1.):
        self.tracker.update(JOINT_NAMES, [0.] * len(JOINT_NAMES), round(robot_now * 1e9), robot_now)
        payload = self.tracker.snapshot(robot_now)
        payload.update(bridge=self.gate.status(robot_now), bridge_age_s=0.)
        return json.dumps(payload, allow_nan=False).encode("utf-8")

    def test_binary_pickle_reports_restart_instructions_without_deserializing(self):
        for protocol in (2, 3, 4, 5):
            with self.subTest(protocol=protocol):
                raw = pickle.dumps({"joints": {}}, protocol=protocol)
                self.socket.packets.append(raw)
                with patch.object(pickle, "loads", side_effect=AssertionError("unsafe loads")) as loads, \
                        patch.object(pickle, "load", side_effect=AssertionError("unsafe load")) as load:
                    result = self.receiver.latest(now=101.)
                self.assertFalse(result["ready"])
                self.assertIsNone(result["q"])
                self.assertTrue(result["reason"].startswith("protocol_mismatch_legacy_pickle:"))
                self.assertIn("SG2_FIXED_QUEST=1", result["reason"])
                self.assertIn("outbound.py", result["reason"])
                self.assertIn("inbound.py", result["reason"])
                loads.assert_not_called()
                load.assert_not_called()

    def test_unrecognized_or_truncated_binary_remains_invalid_json(self):
        for raw in (b"\xff", b"\x80", b"\x80\x7f", b'{"type":'):
            with self.subTest(raw=raw):
                self.socket.packets.append(raw)
                result = self.receiver.latest(now=101.)
                self.assertFalse(result["ready"])
                self.assertTrue(result["reason"].startswith("invalid_json_feedback:"))

    def test_valid_strict_json_is_ready_without_opening_socket(self):
        self.socket.packets.append(self.strict_packet())
        result = self.receiver.latest(now=101.)
        self.assertTrue(result["ready"], result["reason"])
        self.assertEqual(result["reason"], "ready")
        self.assertEqual(len(result["q"]), len(JOINT_NAMES))
        self.assertIsNone(self.receiver._context)

    def test_protocol_mismatch_invalidates_prior_state_then_strict_feedback_recovers(self):
        self.socket.packets.append(self.strict_packet())
        self.assertTrue(self.receiver.latest(now=101.)["ready"])
        self.socket.packets.append(pickle.dumps({"joints": {}}, protocol=4))
        invalid = self.receiver.latest(now=101.02)
        self.assertFalse(invalid["ready"])
        self.assertTrue(invalid["reason"].startswith("protocol_mismatch_legacy_pickle:"))
        # No permissive fallback: a repeat read remains blocked until a fresh
        # strict JSON packet passes the existing state and bridge validation.
        self.assertFalse(self.receiver.latest(now=101.03)["ready"])
        self.socket.packets.append(self.strict_packet(1.04))
        recovered = self.receiver.latest(now=101.04)
        self.assertTrue(recovered["ready"], recovered["reason"])
        self.assertEqual(recovered["reason"], "ready")

    def test_input_snapshot_ages_feedback_across_a_slow_quest_copy(self):
        self.socket.packets.append(self.strict_packet())
        clock = [101.]

        def quest_latest():
            clock[0] += .3  # e.g. GC or scheduler pause during input sampling
            return {"received_at": clock[0]}

        with patch("package.fixed_quest_transport.time", SimpleNamespace(monotonic=lambda: clock[0])):
            feedback, frame, now = sample_control_inputs(SimpleNamespace(latest=quest_latest), self.receiver)
        self.assertEqual(frame["received_at"], now)
        self.assertFalse(feedback["ready"])
        self.assertEqual(feedback["reason"], "stale_joint_state")
        self.assertAlmostEqual(feedback["age_s"], .3)
        self.assertEqual(len(self.socket.packets), 0)

    def test_sim_snapshot_uses_post_copy_time_without_real_polling(self):
        observed = []
        simulated = SimpleNamespace(latest=lambda now: observed.append(now) or {"ready": True})
        with patch("package.fixed_quest_transport.time", SimpleNamespace(monotonic=lambda: 42.)):
            feedback, frame, now = sample_control_inputs(
                SimpleNamespace(latest=lambda: {"received_at": 41.999}), simulated, simulated=True)
        self.assertTrue(feedback["ready"])
        self.assertEqual(observed, [42.])
        self.assertLessEqual(frame["received_at"], now)


if __name__ == "__main__":
    unittest.main()
