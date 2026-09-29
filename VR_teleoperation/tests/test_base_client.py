"""Base client safety against the real command gate, without sockets or hardware."""
import copy
import unittest
import uuid
from unittest.mock import patch

import numpy as np

from test_robot_client import GateHarness
from package.robot_client import SG2BaseCommandClient, SG2JointCommandClient


class BaseHarness(GateHarness):
    def client(self):
        return SG2BaseCommandClient(receiver=self, publisher_factory=self.publisher)

    def tick(self, client, target=(0., 0., 0.), *, enabled=True, dt=.01,
             source=True, bridge_override=None):
        feedback = self.refresh(dt, source=source, bridge_override=bridge_override)
        return client.step(target, enabled, now=self.now, feedback=feedback)

    def start(self, client, target=(.12, -.08, .2)):
        results = [self.tick(client, target) for _ in range(3)]
        assert [r["state"] for r in results] == ["ARMING", "ARMING", "ACTIVE"], results
        return results


class BaseClientTest(unittest.TestCase):
    def make_client(self):
        rig = BaseHarness()
        client = rig.client()
        self.addCleanup(client.close)
        return rig, client

    def assert_stopped(self, rig, client):
        self.assertIsNone(client.publisher)
        self.assertIsNone(client.session_id)
        self.assertTrue(rig.publishers[-1].closed)
        stop = rig.publishers[-1].packets[-1]
        self.assertEqual((stop["mode"], stop["base"]), ("stop", [0., 0., 0.]))
        np.testing.assert_array_equal(client.base_command, np.zeros(3))

    def test_zero_arm_and_baseline_ack_precede_motion_without_joint_publish(self):
        rig, client = self.make_client()
        measured = rig.q.copy()
        target = [.12, -.08, .2]
        rig.start(client, target)
        packets = rig.publishers[0].packets
        self.assertEqual([p["mode"] for p in packets], ["arm", "command", "command"])
        self.assertEqual([p["base"] for p in packets], [[0., 0., 0.], [0., 0., 0.], target])
        self.assertTrue(all(p["type"] == "sg2_base_command" and "joints" not in p for p in packets))
        self.assertTrue(all(result["accepted"] for _, result in rig.received))
        self.assertTrue(all(result["publish_joints"] is None for _, result in rig.received))
        np.testing.assert_array_equal(rig.q, measured)

    def test_missing_or_nonboolean_capability_does_not_open_publisher(self):
        for capabilities in ({}, {"base_json": False}, {"base_json": 1}):
            with self.subTest(capabilities=capabilities):
                rig, client = self.make_client()
                result = rig.tick(client, bridge_override={"capabilities": capabilities})
                self.assertEqual(result["state"], "FAULT")
                self.assertIn("update worker", result["reason"])
                self.assertEqual(rig.publishers, [])

    def test_unacknowledged_baseline_only_sends_zero(self):
        rig, client = self.make_client()
        self.assertEqual(rig.tick(client)["state"], "ARMING")
        rig.deliver = False
        for _ in range(5):
            result = rig.tick(client, [.3, 0., .6])
            self.assertEqual(result["state"], "ARMING")
        self.assertTrue(all(p["base"] == [0., 0., 0.] for p in rig.publishers[0].packets))
        rig.deliver = True
        self.assertEqual(rig.tick(client, [.3, 0., .6])["state"], "ARMING")
        self.assertEqual(rig.tick(client, [.3, 0., .6])["state"], "ACTIVE")

    def test_velocity_uses_linear_norm_and_angular_ceiling(self):
        rig, client = self.make_client()
        rig.start(client, [3., 4., 6.])
        np.testing.assert_allclose(rig.publishers[-1].packets[-1]["base"], [.18, .24, .6])
        limits = rig.gate.status(rig.now)["limits"]
        limits.update(base_linear_speed_m_s=.1, base_angular_speed_rad_s=.2)
        result = rig.tick(client, [3., 4., -6.], bridge_override={"limits": limits})
        self.assertEqual(result["state"], "ACTIVE")
        np.testing.assert_allclose(result["base_command"], [.06, .08, -.2])

    def test_off_releases_without_reading_a_broken_receiver(self):
        rig, client = self.make_client()
        rig.start(client)
        old_session = client.session_id
        old_packet = copy.deepcopy(rig.publishers[0].packets[-1])
        with patch.object(rig, "latest", side_effect=OSError("receiver failed")) as latest:
            result = client.step([.3, 0., 0.], enabled=False)
        latest.assert_not_called()
        self.assertEqual(result["state"], "OFF")
        self.assert_stopped(rig, client)
        stale = rig.gate.receive(old_packet, rig.now)
        self.assertFalse(stale["accepted"])
        self.assertIsNone(stale["publish_base"])
        self.assertEqual(rig.tick(client, [.3, 0., 0.])["state"], "ARMING")
        self.assertNotEqual(client.session_id, old_session)
        self.assertEqual(rig.publishers[-1].packets[-1]["base"], [0., 0., 0.])
        self.assertEqual(rig.tick(client, [.3, 0., 0.])["state"], "ARMING")
        self.assertEqual(rig.publishers[-1].packets[-1]["base"], [0., 0., 0.])

    def test_loop_stall_fault_latches_until_off(self):
        rig, client = self.make_client()
        rig.start(client)
        result = rig.tick(client, [.3, 0., 0.], dt=.11)
        self.assertEqual(result["state"], "FAULT")
        self.assertIn("loop stalled", result["reason"])
        self.assert_stopped(rig, client)
        count = len(rig.publishers)
        self.assertEqual(rig.tick(client)["state"], "FAULT")
        self.assertEqual(len(rig.publishers), count)
        self.assertEqual(rig.tick(client, enabled=False)["state"], "OFF")
        self.assertEqual(rig.tick(client)["state"], "ARMING")

    def test_stalled_joint_source_stops_with_fresh_heartbeats(self):
        rig, client = self.make_client()
        rig.start(client)
        for _ in range(4):
            result = rig.tick(client, [.1, 0., 0.], dt=.06, source=False)
        self.assertEqual(result["state"], "FAULT")
        self.assert_stopped(rig, client)
        self.assertEqual(rig.received[-1][1]["publish_base"], [0., 0., 0.])

    def test_ack_stall_faults_despite_fresh_feedback_and_lease(self):
        rig, client = self.make_client()
        rig.start(client)
        frozen_ack = client.ack_seq
        for _ in range(5):
            result = rig.tick(client, [.1, 0., 0.], dt=.05,
                              bridge_override={"last_command_seq": frozen_ack})
        self.assertEqual(result["state"], "FAULT")
        self.assertIn("acknowledgment stalled", result["reason"])
        self.assert_stopped(rig, client)

    def test_bridge_restart_requires_off_and_zero_rearming(self):
        rig, client = self.make_client()
        rig.start(client)
        rig.gate.bridge_id = str(uuid.uuid4())
        result = rig.tick(client)
        self.assertEqual(result["state"], "FAULT")
        self.assertIn("restarted", result["reason"])
        self.assert_stopped(rig, client)
        self.assertEqual(rig.tick(client, enabled=False)["state"], "OFF")
        # A real worker restart drops ownership as well as changing its ID.
        rig.gate.owner_session_id = None
        rig.gate.command_kind = None
        self.assertEqual(rig.tick(client)["state"], "ARMING")
        self.assertEqual(rig.publishers[-1].packets[-1]["base"], [0., 0., 0.])

    def test_joint_and_base_clients_cannot_take_each_others_lease(self):
        for first_kind in ("base", "joints"):
            with self.subTest(first_kind=first_kind):
                rig = BaseHarness()
                base = rig.client()
                joint = SG2JointCommandClient(receiver=rig, publisher_factory=rig.publisher)
                self.addCleanup(base.close)
                self.addCleanup(joint.close)
                if first_kind == "base":
                    rig.start(base)
                    result = GateHarness.tick(rig, joint)
                else:
                    GateHarness.tick(rig, joint)
                    result = rig.tick(base)
                self.assertEqual(result["state"], "FAULT")
                self.assertIn("another controller", result["reason"])
                self.assertEqual(len(rig.publishers), 1)

    def test_invalid_targets_are_never_sent_and_stop_an_active_session(self):
        for target in ([1., 2.], [[0., 0., 0.]], [np.nan, 0., 0.], [0., np.inf, 0.], None):
            with self.subTest(target=target):
                rig, client = self.make_client()
                result = rig.tick(client, target)
                self.assertEqual(result["state"], "FAULT")
                self.assertEqual(rig.publishers, [])
                rig.tick(client, enabled=False)
                rig.start(client)
                before = len(rig.publishers[-1].packets)
                self.assertEqual(rig.tick(client, target)["state"], "FAULT")
                self.assertEqual([p["mode"] for p in rig.publishers[-1].packets[before:]], ["stop"])
                self.assert_stopped(rig, client)

    def test_feedback_read_exception_and_slow_poll_cannot_send_motion(self):
        for slow in (False, True):
            with self.subTest(slow=slow):
                rig, client = self.make_client()
                rig.start(client)
                before = len(rig.publishers[-1].packets)

                def latest():
                    if not slow:
                        raise OSError("receiver failed")
                    old = rig.cache.latest(rig.now)
                    rig.now += .21
                    return old

                with patch.object(rig, "latest", side_effect=latest), \
                        patch("package.robot_client.time.monotonic", side_effect=lambda: rig.now):
                    result = client.step([.3, 0., 0.], enabled=True)
                self.assertEqual(result["state"], "FAULT")
                self.assertEqual([p["mode"] for p in rig.publishers[-1].packets[before:]], ["stop"])
                self.assert_stopped(rig, client)

    def test_close_stops_and_closed_client_cannot_reopen(self):
        rig, client = self.make_client()
        rig.start(client)
        client.close()
        self.assert_stopped(rig, client)
        self.assertTrue(rig.closed)
        self.assertEqual(rig.tick(client)["state"], "FAULT")
        self.assertEqual(len(rig.publishers), 1)

    def test_failed_enqueue_closes_and_failed_close_is_retried(self):
        rig, client = self.make_client()
        rig.start(client)
        rig.enqueue = False
        result = rig.tick(client)
        self.assertEqual(result["state"], "FAULT")
        self.assertIn("could not be queued", result["reason"])
        self.assert_stopped(rig, client)

        rig, client = self.make_client()
        rig.start(client)
        publisher = rig.publishers[-1]
        with patch.object(publisher, "close", side_effect=OSError("close failed")):
            self.assertEqual(rig.tick(client, enabled=False)["state"], "FAULT")
        client.close()
        self.assertTrue(publisher.closed)
        self.assertTrue(rig.closed)


if __name__ == "__main__":
    unittest.main()
