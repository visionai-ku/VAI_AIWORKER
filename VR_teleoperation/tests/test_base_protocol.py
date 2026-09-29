"""Strict base leases and ROS routing, without sockets or hardware."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from package.worker.fixed_quest_protocol import (
    JOINT_NAMES, JointStateTracker, StrictCommandGate, make_joint_groups,
)
import test_fixed_quest_protocol as joint_tests


class BaseProtocolTest(unittest.TestCase):
    def setUp(self):
        self.q = [0.] * len(JOINT_NAMES)
        self.tracker = JointStateTracker()
        self.tracker.update(JOINT_NAMES, self.q, 1, 1.)
        self.gate = StrictCommandGate(self.tracker, bridge_id="test-bridge")

    def packet(self, seq=0, mode="arm", now=1., base=None, session="base-owner"):
        return dict(type="sg2_base_command", version=1, session_id=session, seq=seq,
                    bridge_id=self.gate.bridge_id, feedback_time_s=now, valid_for_s=.2,
                    mode=mode, base=[0., 0., 0.] if base is None else base)

    def joint_packet(self, seq=0, mode="arm", now=1., session="joint-owner"):
        packet = self.packet(seq, mode, now, session=session)
        packet.update(type="sg2_command", joints=make_joint_groups(self.q))
        del packet["base"]
        return packet

    def arm(self):
        result = self.gate.receive(self.packet(), 1.)
        self.assertTrue(result["accepted"], result)
        self.assertIsNone(result["publish_joints"])
        self.assertEqual(result["publish_base"], [0., 0., 0.])

    def moving(self):
        self.arm()
        result = self.gate.receive(self.packet(1, "command", 1.01, [.18, .24, -.6]), 1.01)
        self.assertTrue(result["accepted"], result)
        self.assertIsNone(result["publish_joints"])
        self.assertEqual(result["publish_base"], [.18, .24, -.6])

    def test_capability_and_command_domain_are_advertised(self):
        status = self.gate.status(1.)
        self.assertIs(status["capabilities"]["base_json"], True)
        self.assertIsNone(status["command_kind"])
        self.assertEqual(status["limits"]["base_linear_speed_m_s"], .3)
        self.assertEqual(status["limits"]["base_angular_speed_rad_s"], .6)
        status["capabilities"]["base_json"] = False
        self.assertIs(self.gate.status(1.)["capabilities"]["base_json"], True)
        self.arm()
        self.assertEqual(self.gate.status(1.)["command_kind"], "base")

    def test_arm_requires_zero_fresh_feedback_and_command_requires_arm(self):
        result = self.gate.receive(self.packet(base=[.01, 0., 0.]), 1.)
        self.assertEqual(result["reason"], "base_arm_requires_zero")
        result = self.gate.receive(self.packet(mode="command", base=[.1, 0., 0.]), 1.)
        self.assertEqual(result["reason"], "explicit_arm_required")
        result = self.gate.receive(self.packet(now=1.21), 1.21)
        self.assertEqual(result["reason"], "stale_joint_state")
        self.assertIsNone(self.gate.owner_session_id)
        self.assertIsNone(result["publish_base"])

    def test_repeated_arm_remains_zero_until_first_command(self):
        self.arm()
        result = self.gate.receive(self.packet(1, "arm", 1.02), 1.02)
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["publish_base"], [0., 0., 0.])
        self.assertIsNone(result["publish_joints"])
        self.assertEqual(self.gate.last_command_seq, 1)

    def test_bad_velocities_never_publish_acknowledge_or_renew(self):
        self.arm()
        for base in (None, [], [0., 0.], [0., 0., 0., 0.], "0,0,0",
                     [True, 0., 0.], ["0", 0., 0.], [float("nan"), 0., 0.],
                     [0., float("inf"), 0.], [.301, 0., 0.], [.3, .01, 0.],
                     [0., 0., -.601]):
            with self.subTest(base=base):
                packet = self.packet(1, "command", 1.02)
                packet["base"] = base
                result = self.gate.receive(packet, 1.02)
                self.assertFalse(result["accepted"], result)
                self.assertIsNone(result["publish_base"])
                self.assertIsNone(result["publish_joints"])
                self.assertEqual(self.gate.last_command_seq, 0)
                self.assertEqual(self.gate.deadline, 1.2)

    def test_replay_bad_envelope_and_mixed_payload_do_not_refresh_lease(self):
        self.arm()
        cases = [("seq", 0), ("seq", True), ("bridge_id", "other"),
                 ("valid_for_s", .201), ("feedback_time_s", .5),
                 ("feedback_time_s", 2.), ("version", True), ("mode", "move"),
                 ("motion", "home"), ("joints", make_joint_groups(self.q))]
        for key, value in cases:
            with self.subTest(field=key):
                packet = self.packet(1, "command", 1.02, [.1, 0., 0.])
                packet[key] = value
                result = self.gate.receive(packet, 1.02)
                self.assertFalse(result["accepted"], result)
                self.assertIsNone(result["publish_base"])
                self.assertEqual(self.gate.last_command_seq, 0)
                self.assertEqual(self.gate.deadline, 1.2)

    def test_stop_requires_zero_and_accepts_stale_echo(self):
        self.moving()
        result = self.gate.receive(self.packet(2, "arm", 1.02), 1.02)
        self.assertEqual(result["reason"], "already_armed")
        result = self.gate.receive(self.packet(2, "stop", 0., [.1, 0., 0.]), 1.03)
        self.assertEqual(result["reason"], "base_stop_requires_zero")
        self.assertEqual(self.gate.last_command_seq, 1)
        result = self.gate.receive(self.packet(2, "stop", 0.), 1.04)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["publish_base"], [0., 0., 0.])
        self.assertIsNone(result["publish_joints"])
        self.assertIsNone(self.gate.command_kind)
        self.assertIsNone(self.gate.owner_session_id)
        result = self.gate.receive(self.packet(3, "command", 1.05, [.1, 0., 0.]), 1.05)
        self.assertEqual(result["reason"], "explicit_arm_required")

    def test_joint_and_base_share_owner_and_cannot_switch_domain_while_armed(self):
        for base_first in (True, False):
            with self.subTest(base_first=base_first):
                self.gate = StrictCommandGate(self.tracker, bridge_id="test-bridge")
                first = self.packet if base_first else self.joint_packet
                second = self.joint_packet if base_first else self.packet
                self.assertTrue(self.gate.receive(first(session="owner"), 1.)["accepted"])
                for mode in ("arm", "command", "stop"):
                    for session, reason in (("other", "owner_conflict"),
                                            ("owner", "command_kind_conflict")):
                        result = self.gate.receive(second(1, mode, 1.01, session=session), 1.01)
                        self.assertEqual(result["reason"], reason)
                        self.assertIsNone(result["publish_base"])
                        self.assertIsNone(result["publish_joints"])
                        self.assertEqual(self.gate.deadline, 1.2)
                self.assertTrue(self.gate.receive(first(1, "stop", 0., session="owner"), 1.02)["accepted"])
                self.assertTrue(self.gate.receive(second(0, "arm", 1.03, session="new-owner"), 1.03)["accepted"])

    def test_base_expiry_always_zeroes_once_and_requires_new_session(self):
        for feedback_stale in (False, True):
            with self.subTest(feedback_stale=feedback_stale):
                self.setUp()
                self.moving()
                if not feedback_stale:
                    self.tracker.update(JOINT_NAMES, self.q, 2, 1.2)
                expired = self.gate.expire(1.22)
                self.assertEqual(expired["reason"], "command_expired")
                self.assertEqual(expired["publish_base"], [0., 0., 0.])
                self.assertIsNone(expired["publish_joints"])
                self.assertIsNone(self.gate.expire(1.23)["publish_base"])
                self.tracker.update(JOINT_NAMES, self.q, 3, 1.24)
                result = self.gate.receive(self.packet(2, "arm", 1.24), 1.24)
                self.assertEqual(result["reason"], "new_session_required")
                result = self.gate.receive(self.packet(0, "arm", 1.24, session="new-base"), 1.24)
                self.assertTrue(result["accepted"], result)

    def test_feedback_loss_zeroes_base_before_command_lease_deadline(self):
        self.tracker.timeout_s = .05
        self.moving()
        result = self.gate.expire(1.06)
        self.assertEqual(result["reason"], "feedback_expired")
        self.assertEqual(result["publish_base"], [0., 0., 0.])
        self.assertIsNone(result["publish_joints"])

    def test_expiry_zero_survives_same_receive_new_joint_arm_or_bad_message(self):
        for valid in (False, True):
            with self.subTest(valid=valid):
                self.setUp()
                self.moving()
                self.tracker.update(JOINT_NAMES, self.q, 2, 1.2)
                packet = self.joint_packet(now=1.22) if valid else {"type": "invalid"}
                result = self.gate.receive(packet, 1.22)
                self.assertEqual(result["accepted"], valid)
                self.assertEqual(result["publish_base"], [0., 0., 0.])
                if valid:
                    self.assertEqual(result["publish_joints"], make_joint_groups(self.q))

    def test_expired_joint_hold_survives_same_receive_base_arm_or_stop(self):
        for mode in ("arm", "stop"):
            with self.subTest(mode=mode):
                self.setUp()
                self.assertTrue(self.gate.receive(self.joint_packet(), 1.)["accepted"])
                self.tracker.update(JOINT_NAMES, self.q, 2, 1.2)
                result = self.gate.receive(self.packet(mode=mode, now=1.21), 1.21)
                self.assertTrue(result["accepted"], result)
                self.assertEqual(result["publish_base"], [0., 0., 0.])
                self.assertEqual(result["publish_joints"], make_joint_groups(self.q))
                self.assertIsNone(self.gate.expire(1.22)["publish_joints"])

    def worker(self):
        module = joint_tests.FixedQuestProtocolTest.load_worker_with_ros_stubs("inbound")
        module.Twist = lambda: SimpleNamespace(linear=SimpleNamespace(x=0., y=0., z=0.),
                                               angular=SimpleNamespace(x=0., y=0., z=0.))
        node = module.SG2ZMQRosBridge.__new__(module.SG2ZMQRosBridge)
        node.strict, node.gate, node.tracker = True, self.gate, self.tracker
        node.report_warning = lambda message: None
        self.sent = []
        node.cmd_vel_pub = SimpleNamespace(publish=lambda msg: self.sent.append(
            ("base", [msg.linear.x, msg.linear.y, msg.angular.z])))
        node.traj_pubs = {group: SimpleNamespace(publish=lambda msg: self.sent.append(
            ("joint", copy.deepcopy(msg)))) for group in module.CONTROLLER_CONFIG}
        return module, node

    def tick(self, module, node, now, packet=None):
        def receive(**kwargs):
            if packet is None:
                raise module.zmq.Again()
            return json.dumps(packet).encode()
        node.socket = SimpleNamespace(recv=receive)
        with patch.object(module.time, "monotonic", return_value=now):
            node.on_timer()

    def test_worker_routes_base_only_and_zeroes_without_incoming_packets(self):
        module, node = self.worker()
        self.tick(module, node, 1., self.packet())
        self.tick(module, node, 1.01, self.packet(1, "command", 1.01, [.18, .24, -.6]))
        self.tick(module, node, 1.02, self.packet(2, "command", 1.02, [.3, .3, 0.]))
        self.tick(module, node, 1.22)
        self.tick(module, node, 1.23)
        self.assertEqual(self.sent, [("base", [0., 0., 0.]), ("base", [.18, .24, -.6]),
                                    ("base", [0., 0., 0.])])

    def test_worker_stops_before_new_joint_arm_in_same_timer_tick(self):
        module, node = self.worker()
        self.tick(module, node, 1., self.packet())
        self.tick(module, node, 1.01, self.packet(1, "command", 1.01, [.1, 0., 0.]))
        self.tracker.update(JOINT_NAMES, self.q, 2, 1.2)
        self.sent.clear()
        self.tick(module, node, 1.22, self.joint_packet(now=1.22))
        self.assertEqual(self.sent[0], ("base", [0., 0., 0.]))
        self.assertEqual([kind for kind, _ in self.sent], ["base", "joint", "joint", "joint", "joint"])

    def test_worker_stop_with_stale_feedback_and_graceful_shutdown_zeroes_base(self):
        module, node = self.worker()
        self.tick(module, node, 1., self.packet())
        self.tick(module, node, 1.01, self.packet(1, "command", 1.01, [.1, 0., 0.]))
        self.tracker.timeout_s = .05
        self.sent.clear()
        self.tick(module, node, 1.06, self.packet(2, "stop", 0.))
        self.assertTrue(self.sent)
        self.assertTrue(all(row == ("base", [0., 0., 0.]) for row in self.sent))
        self.assertEqual(self.gate.state, "STOPPED")

        self.tracker.update(JOINT_NAMES, self.q, 2, 1.1)
        self.tick(module, node, 1.1, self.packet(0, "arm", 1.1, session="new-owner"))
        self.sent.clear()
        node.socket = SimpleNamespace(close=lambda: None)
        node.zmq_ctx = SimpleNamespace(term=lambda: None)
        node.destroy_node = lambda: None
        node.get_logger = lambda: SimpleNamespace(info=lambda message: None)
        with patch.object(module, "SG2ZMQRosBridge", return_value=node), \
                patch.object(module.rclpy, "init", create=True), \
                patch.object(module.rclpy, "spin", side_effect=KeyboardInterrupt, create=True), \
                patch.object(module.rclpy, "shutdown", create=True):
            module.main()
        self.assertEqual(self.sent, [("base", [0., 0., 0.])])


if __name__ == "__main__":
    unittest.main()
