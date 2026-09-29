"""Offline shared-client integration with the real robot gate and state cache."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from package.fixed_quest_transport import SG2RobotFeedbackCache
from package.robot_client import SG2JointCommandClient, SG2RobotStateSubscriber
from package.worker.fixed_quest_protocol import (
    FIXED_QUEST_VR_GATE_OPTIONS, JOINT_NAMES, JointStateTracker,
    StrictCommandGate, make_joint_groups, parse_joint_groups,
)


class GateHarness:
    """No sockets: a PUB enqueue can be delivered, dropped or rejected locally."""

    def __init__(self, **gate_options):
        self.now = 10.
        self.q = np.zeros(len(JOINT_NAMES))
        self.q[:3] = [-.1, .3, .15]
        for name in ("arm_l_joint4", "arm_r_joint4"):
            self.q[JOINT_NAMES.index(name)] = -np.pi / 2
        self.tracker = JointStateTracker()
        self.gate = StrictCommandGate(self.tracker, **(FIXED_QUEST_VR_GATE_OPTIONS | gate_options))
        self.cache = SG2RobotFeedbackCache()
        self.stamp = 0
        self.publishers = []
        self.deliver = True
        self.enqueue = True
        self.follow = True
        self.closed = False
        self.received = []
        self.refresh()

    def refresh(self, dt=0., *, source=True, bridge_override=None):
        self.now += dt
        if source:
            self.stamp += 1
            self.tracker.update(JOINT_NAMES, self.q.tolist(), self.stamp, self.now)
        self.gate.expire(self.now)
        state = self.tracker.snapshot(self.now)
        state["bridge"] = self.gate.status(self.now)
        if bridge_override:
            state["bridge"].update(bridge_override)
        state["bridge_age_s"] = 0.
        assert self.cache.ingest(state, received_at=self.now)
        return self.cache.latest(self.now)

    def latest(self):
        return self.cache.latest(self.now)

    def close(self):
        self.closed = True
        self.cache.reject("receiver_closed")

    def publisher(self):
        rig = self

        class Publisher:
            def __init__(self):
                self.packets = []
                self.closed = False

            def send(self, packet):
                assert not self.closed
                self.packets.append(copy.deepcopy(packet))
                if not rig.enqueue:
                    return False
                if rig.deliver:
                    result = rig.gate.receive(packet, rig.now)
                    rig.received.append((copy.deepcopy(packet), result))
                    if result["publish_joints"] is not None and rig.follow:
                        rig.q = np.asarray(parse_joint_groups(result["publish_joints"]))
                return True

            def close(self):
                self.closed = True

        pub = Publisher()
        self.publishers.append(pub)
        return pub

    def client(self, **kwargs):
        return SG2JointCommandClient(receiver=self, publisher_factory=self.publisher, **kwargs)

    def tick(self, client, target=None, *, enabled=True, dt=.01, source=True, bridge_override=None):
        feedback = self.refresh(dt, source=source, bridge_override=bridge_override)
        return client.step(make_joint_groups(self.q if target is None else target),
                           enabled, now=self.now, feedback=feedback)

    def start(self, client, target=None):
        first = self.tick(client, target)
        second = self.tick(client, target)
        third = self.tick(client, target)
        assert first["state"] == "ARMING", first
        assert second["state"] == third["state"] == "ACTIVE", (second, third)
        return first, second, third


class RobotClientTest(unittest.TestCase):
    def compressed_gripper_rejection(self):
        """Queue two poses, then deliver them one millisecond apart."""
        rig = GateHarness()
        client = rig.client()
        self.addCleanup(client.close)
        target = rig.q.copy()
        target[JOINT_NAMES.index("gripper_r_joint1")] = .5
        rig.start(client, target)
        rig.deliver = False
        rig.tick(client, target, dt=.02)
        earlier = copy.deepcopy(rig.publishers[-1].packets[-1])
        rig.tick(client, target, dt=.02)
        rejected = copy.deepcopy(rig.publishers[-1].packets[-1])
        accepted = rig.gate.receive(earlier, rig.now)
        self.assertTrue(accepted["accepted"], accepted)
        rig.q = np.asarray(parse_joint_groups(accepted["publish_joints"]))
        rig.refresh(.001)
        outcome = rig.gate.receive(rejected, rig.now)
        self.assertFalse(outcome["accepted"])
        self.assertEqual(outcome["reason"], "joint_rate_limit")
        self.assertEqual(rig.gate.rejection_detail["joint_name"], "gripper_r_joint1")
        rig.refresh()
        return rig, client, target, rejected

    def test_compressed_gripper_arrivals_retry_frozen_pose_until_acknowledged(self):
        rig, client, target, rejected = self.compressed_gripper_rejection()
        frozen = client.q_command.copy()
        travel = client.travel.copy()
        target[-1] = .9  # New input cannot advance an unaccepted command.
        rig.deliver = True
        first = rig.tick(client, target, dt=.001)
        first_retry = copy.deepcopy(rig.publishers[-1].packets[-1])
        self.assertEqual(first["state"], "ACTIVE", first)
        self.assertEqual(rig.gate.reason, "joint_rate_limit")
        np.testing.assert_array_equal(first["q_command"], frozen)
        np.testing.assert_array_equal(client.travel, travel)
        self.assertEqual(first_retry["joints"], rejected["joints"])
        self.assertGreater(first_retry["seq"], rejected["seq"])
        self.assertGreater(first_retry["feedback_time_s"], rejected["feedback_time_s"])

        # An accepted status for an older pose is not proof that the frozen
        # latest pose was accepted. Keep retrying until its sequence is covered.
        rig.deliver = False
        older_ack = rig.tick(client, target, dt=.001,
                             bridge_override={"reason": "accepted"})
        self.assertLess(rig.gate.last_command_seq, rejected["seq"])
        self.assertEqual(older_ack["state"], "ACTIVE", older_ack)
        np.testing.assert_array_equal(older_ack["q_command"], frozen)
        np.testing.assert_array_equal(client.travel, travel)
        self.assertIsNotNone(client.rate_recovery_seq)
        rig.deliver = True

        # The same pose becomes legal once robot-side elapsed time catches up.
        second = rig.tick(client, target, dt=.03)
        self.assertEqual(second["state"], "ACTIVE", second)
        self.assertEqual(rig.gate.reason, "accepted")
        self.assertEqual(rig.publishers[-1].packets[-1]["joints"], rejected["joints"])
        np.testing.assert_array_equal(client.travel, travel)
        self.assertIsNotNone(client.rate_recovery_seq)

        resumed = rig.tick(client, target)
        self.assertEqual(resumed["state"], "ACTIVE", resumed)
        self.assertIsNone(client.rate_recovery_seq)
        self.assertGreater(resumed["q_command"][-1], frozen[-1])
        self.assertEqual(rig.gate.reason, "accepted")
        self.assertFalse(rig.publishers[-1].closed)

    def test_unacknowledged_rate_recovery_still_expires_and_closes(self):
        rig, client, target, rejected = self.compressed_gripper_rejection()
        first_packet = len(rig.publishers[-1].packets)
        started = rig.now
        # Drop all retries: fresh telemetry continues to report the real
        # rejection, but no accepted command renews the lease or acknowledgment.
        for _ in range(25):
            result = rig.tick(client, target)
            if result["state"] == "FAULT":
                break
        self.assertEqual(result["state"], "FAULT", result)
        self.assertLessEqual(rig.now - started, .25)
        self.assertTrue(any(reason in result["reason"] for reason in
                            ("lease lost", "acknowledgment stalled")), result)
        retries = [packet for packet in rig.publishers[-1].packets[first_packet:]
                   if packet["mode"] == "command"]
        self.assertTrue(retries)
        self.assertTrue(all(packet["joints"] == rejected["joints"] for packet in retries))
        self.assertTrue(rig.publishers[-1].closed)

    def test_rate_recovery_preserves_off_and_other_rejection_faults(self):
        for disable in (False, True):
            with self.subTest(disable=disable):
                rig, client, target, _ = self.compressed_gripper_rejection()
                result = rig.tick(client, target)
                self.assertEqual(result["state"], "ACTIVE", result)
                before = len(rig.publishers[-1].packets)
                result = rig.tick(client, target, enabled=not disable,
                                  bridge_override={"reason": "tracking_lag"})
                self.assertEqual(result["state"], "OFF" if disable else "FAULT", result)
                if not disable:
                    self.assertIn("robot rejected command: tracking_lag", result["reason"])
                self.assertEqual([packet["mode"] for packet in rig.publishers[-1].packets[before:]],
                                 ["stop"])
                self.assertTrue(rig.publishers[-1].closed)

    def test_initial_enable_and_command_baseline_use_measured_pose(self):
        rig = GateHarness()
        initial = rig.q.copy()
        target = initial.copy()
        target[3] += .5
        client = rig.client()
        self.addCleanup(client.close)
        self.assertEqual(rig.tick(client, target, enabled=False)["state"], "OFF")
        self.assertFalse(rig.publishers)
        first, baseline, active = rig.start(client, target)
        self.assertTrue(first["synchronize"])
        np.testing.assert_array_equal(first["q_command"], initial)
        np.testing.assert_array_equal(baseline["q_command"], initial)
        self.assertGreater(active["q_command"][3], initial[3])
        self.assertTrue(all(result["accepted"] for _, result in rig.received))

    def test_eef_subset_preserves_measured_head_and_lift(self):
        rig = GateHarness()
        initial = rig.q.copy()
        controlled = [name for name in JOINT_NAMES if name.startswith(("arm_", "gripper_"))]
        client = rig.client(controlled_joint_names=controlled)
        self.addCleanup(client.close)
        target = initial.copy()
        target[:3] = 0.
        target[3] = .4
        rig.start(client, target)
        for _ in range(20):
            result = rig.tick(client, target)
            self.assertEqual(result["state"], "ACTIVE", result)
            np.testing.assert_array_equal(result["q_command"][:3], initial[:3])
        self.assertGreater(rig.q[3], initial[3])

    def test_dropped_frames_are_bounded_and_resume_with_latest_pose(self):
        rig = GateHarness()
        client = rig.client()
        self.addCleanup(client.close)
        target = rig.q.copy()
        target[3] = 1.
        rig.start(client, target)
        rig.tick(client, target)
        acknowledged = rig.q.copy()
        rig.deliver = False
        for _ in range(10):
            result = rig.tick(client, target)
            self.assertEqual(result["state"], "ACTIVE", result)
        self.assertLessEqual(result["q_command"][3] - acknowledged[3], np.deg2rad(60) * .04 + 1e-12)
        rig.deliver = True
        for _ in range(10):
            result = rig.tick(client, target)
            self.assertEqual(result["state"], "ACTIVE", result)
        self.assertTrue(all(result["accepted"] for _, result in rig.received))
        self.assertGreater(rig.q[3], acknowledged[3])

    def test_100hz_commands_and_50hz_bridge_feedback_with_drop_and_jitter(self):
        # Advance a deterministic network clock in 1 ms steps. Both robot state
        # and inbound timers run at 100 Hz; bridge heartbeats run at 50 Hz. The
        # PC ticks halfway between robot ticks. Ready inbound packets conflate.
        cases = [(joint, delays) for joint in ("arm_l_joint1", "gripper_r_joint1")
                 for delays in ((0,), (0, 0, None, None, 0), (1, 14, 0, 3, 27, 0))]
        for joint, delays in cases:
            with self.subTest(joint=joint, delays=delays):
                rig = GateHarness()
                rig.deliver = False
                pending = []
                delivered = []
                queue_count = 0
                last_arrival = rig.now

                def factory():
                    publisher = rig.publisher()
                    original_send = publisher.send

                    def send(packet):
                        nonlocal queue_count, last_arrival
                        original_send(packet)
                        delay = delays[queue_count % len(delays)]
                        queue_count += 1
                        if delay is not None:
                            # TCP preserves order even when one frame is held.
                            last_arrival = max(last_arrival, rig.now + delay / 1000.)
                            pending.append((last_arrival, copy.deepcopy(packet)))
                        return True

                    publisher.send = send
                    return publisher

                client = SG2JointCommandClient(receiver=rig, publisher_factory=factory)
                self.addCleanup(client.close)
                target = rig.q.copy()
                joint_index = JOINT_NAMES.index(joint)
                target[joint_index] = .5
                initial = rig.q.copy()
                last_bridge = rig.gate.status(rig.now)
                fault_at = None
                for ms in range(1, 1301):
                    rig.now = 10. + ms / 1000.
                    if ms % 10 == 0:
                        rig.stamp += 1
                        rig.tracker.update(JOINT_NAMES, rig.q.tolist(), rig.stamp, rig.now)
                        rig.gate.expire(rig.now)
                        arrived = [(index, packet) for index, (at, packet) in enumerate(pending)
                                   if at <= rig.now + 1e-12]
                        if arrived:
                            packet = arrived[-1][1]
                            arrival_indices = {index for index, _ in arrived}
                            pending[:] = [item for index, item in enumerate(pending) if index not in arrival_indices]
                            accepted = rig.gate.receive(packet, rig.now)
                            delivered.append((packet, accepted))
                            if accepted["publish_joints"] is not None:
                                rig.q = np.asarray(parse_joint_groups(accepted["publish_joints"]))
                        if ms % 20 == 0:
                            last_bridge = rig.gate.status(rig.now)
                        state = rig.tracker.snapshot(rig.now)
                        state["bridge"] = last_bridge
                        state["bridge_age_s"] = rig.now - last_bridge["status_time_s"]
                        self.assertTrue(rig.cache.ingest(state, received_at=rig.now))
                    if ms % 10 == 5:
                        result = client.step(make_joint_groups(target), True, now=rig.now,
                                             feedback=rig.cache.latest(rig.now))
                        if result["state"] == "FAULT":
                            fault_at = rig.now
                            break
                rejected = [(packet, outcome) for packet, outcome in delivered if not outcome["accepted"]]
                self.assertIsNone(fault_at, result)
                self.assertFalse(rig.publishers[-1].closed)
                self.assertEqual(result["state"], "ACTIVE", result)
                self.assertEqual(rig.gate.reason, "accepted")
                self.assertAlmostEqual(rig.q[joint_index], target[joint_index])
                self.assertTrue(all(outcome["reason"] == "joint_rate_limit"
                                    for _, outcome in rejected), rejected)
                if joint == "gripper_r_joint1" and delays == (1, 14, 0, 3, 27, 0):
                    self.assertTrue(rejected, "Exercise actual rate rejection and recovery")
                np.testing.assert_array_equal(rig.q[:3], initial[:3])

    def test_ack_stall_faults_while_fresh_bridge_heartbeats_continue(self):
        rig = GateHarness()
        client = rig.client()
        self.addCleanup(client.close)
        rig.start(client)
        frozen = rig.gate.last_command_seq
        for _ in range(24):
            result = rig.tick(client, bridge_override={"last_command_seq": frozen})
            if result["state"] == "FAULT":
                break
        self.assertEqual(result["state"], "FAULT", result)
        self.assertIn("acknowledgment stalled", result["reason"])
        self.assertTrue(rig.publishers[-1].closed)
        self.assertIsNone(rig.gate.owner_session_id)

    def test_unknown_and_regressed_acks_fault_without_further_motion(self):
        for bad_ack in (-1, 100000):
            with self.subTest(ack=bad_ack):
                rig = GateHarness()
                client = rig.client()
                self.addCleanup(client.close)
                rig.start(client)
                result = rig.tick(client, bridge_override={"last_command_seq": bad_ack})
                self.assertEqual(result["state"], "FAULT")
                self.assertIn("acknowledgment", result["reason"])
                self.assertTrue(rig.publishers[-1].closed)

    def test_off_releases_port_and_next_on_creates_new_session(self):
        rig = GateHarness()
        client = rig.client()
        self.addCleanup(client.close)
        rig.start(client)
        previous = client.session_id
        self.assertEqual(rig.tick(client, enabled=False)["state"], "OFF")
        self.assertTrue(rig.publishers[0].closed)
        self.assertIsNone(rig.gate.owner_session_id)
        rig.q[1:3] = [.4, -.1]
        measured = rig.q.copy()
        first, _, _ = rig.start(client)
        self.assertNotEqual(client.session_id, previous)
        np.testing.assert_array_equal(first["q_command"], measured)
        self.assertEqual(len(rig.publishers), 2)

    def test_source_or_bridge_restart_requires_off_on(self):
        for source_restart in (True, False):
            with self.subTest(source_restart=source_restart):
                rig = GateHarness()
                client = rig.client()
                self.addCleanup(client.close)
                rig.start(client)
                if source_restart:
                    rig.tracker = JointStateTracker()
                    rig.gate.tracker = rig.tracker
                else:
                    rig.gate = StrictCommandGate(rig.tracker, **FIXED_QUEST_VR_GATE_OPTIONS)
                result = rig.tick(client)
                self.assertEqual(result["state"], "FAULT")
                self.assertIn("restarted", result["reason"])
                self.assertEqual(rig.tick(client)["state"], "FAULT")
                self.assertEqual(rig.tick(client, enabled=False)["state"], "OFF")
                self.assertEqual(rig.tick(client)["state"], "ARMING")

    def test_stale_source_and_stalled_control_loop_close_command_port(self):
        for source_stale in (True, False):
            with self.subTest(source_stale=source_stale):
                rig = GateHarness()
                client = rig.client()
                self.addCleanup(client.close)
                rig.start(client)
                if source_stale:
                    for _ in range(22):
                        result = rig.tick(client, source=False)
                        if result["state"] == "FAULT":
                            break
                    self.assertIn("stale", result["reason"])
                else:
                    result = rig.tick(client, dt=.11)
                    self.assertIn("loop stalled", result["reason"])
                self.assertEqual(result["state"], "FAULT")
                self.assertTrue(rig.publishers[-1].closed)

    def test_invalid_targets_are_never_transmitted(self):
        for case in ("missing", "nonfinite", "bool", "wrist_bounds"):
            with self.subTest(case=case):
                rig = GateHarness()
                client = rig.client()
                self.addCleanup(client.close)
                rig.start(client)
                target = make_joint_groups(rig.q)
                if case == "missing":
                    del target["head"]
                elif case == "wrist_bounds":
                    target["arm_l"][-1] = 1.6  # MuJoCo permits this; strict URDF intersection does not.
                else:
                    target["arm_l"][0] = float("nan") if case == "nonfinite" else True
                feedback = rig.refresh(.01)
                before = len(rig.publishers[-1].packets)
                result = client.step(target, True, now=rig.now, feedback=feedback)
                self.assertEqual(result["state"], "FAULT")
                self.assertEqual([p["mode"] for p in rig.publishers[-1].packets[before:]], ["stop"])

    def test_speed_caps_use_bridge_head_gripper_lift_and_step_limits(self):
        rig = GateHarness(max_step_dt=.005, max_lift_speed=.003,
                          max_joint_speed=float(np.deg2rad(20)), max_gripper_joint_speed=float(np.deg2rad(120)))
        client = rig.client(joint_speed=np.deg2rad(120), lift_speed=.1)
        self.addCleanup(client.close)
        target = rig.q.copy()
        target[[0, 1, 3, 17]] += [.03, .1, .5, .5]
        rig.start(client, target)
        for _ in range(20):
            result = rig.tick(client, target)
            self.assertEqual(result["state"], "ACTIVE", result)
        self.assertTrue(all(result["accepted"] for _, result in rig.received), rig.received[-1])

    def test_rejected_local_enqueue_faults_and_closed_client_cannot_reopen(self):
        rig = GateHarness()
        client = rig.client()
        rig.start(client)
        rig.enqueue = False
        result = rig.tick(client)
        self.assertEqual(result["state"], "FAULT")
        self.assertIn("could not be queued", result["reason"])
        self.assertTrue(rig.publishers[-1].closed)
        client.close()
        self.assertTrue(rig.closed)
        self.assertIn("closed", rig.tick(client)["reason"])
        self.assertEqual(len(rig.publishers), 1)

    def test_invalid_configuration_does_not_open_a_receiver(self):
        with patch("package.robot_client.SG2RobotStateReceiver") as receiver:
            for kwargs in ({"controlled_joint_names": ["absent"]}, {"joint_speed": -1}):
                with self.assertRaises(ValueError):
                    SG2JointCommandClient(**kwargs)
            receiver.assert_not_called()

    def test_feedback_exception_closes_active_publisher(self):
        rig = GateHarness()
        client = rig.client()
        self.addCleanup(client.close)
        rig.start(client)
        with patch.object(rig, "latest", side_effect=OSError("receiver failed")):
            result = client.step(make_joint_groups(rig.q), True, now=rig.now)
        self.assertEqual(result["state"], "FAULT")
        self.assertTrue(rig.publishers[-1].closed)

    def test_slow_feedback_poll_cannot_authorize_a_stale_command(self):
        rig = GateHarness()
        client = rig.client()
        self.addCleanup(client.close)
        rig.start(client)
        previous = len(rig.publishers[-1].packets)

        def slow_latest():
            snapshot = rig.cache.latest(rig.now)
            rig.now += .21
            return snapshot

        with patch.object(rig, "latest", side_effect=slow_latest), \
                patch("package.robot_client.time.monotonic", side_effect=lambda: rig.now):
            result = client.step(make_joint_groups(rig.q), True)
        self.assertEqual(result["state"], "FAULT")
        self.assertEqual([p["mode"] for p in rig.publishers[-1].packets[previous:]], ["stop"])
        self.assertTrue(rig.publishers[-1].closed)

    def test_cleanup_retries_a_publisher_that_failed_to_close(self):
        rig = GateHarness()
        client = rig.client()
        rig.start(client)
        publisher = rig.publishers[-1]
        with patch.object(publisher, "close", side_effect=OSError("close failed")):
            result = rig.tick(client, enabled=False)
            self.assertEqual(result["state"], "FAULT")
        client.close()
        self.assertTrue(publisher.closed)
        self.assertIsNone(client.publisher)


class RobotStateSubscriberTest(unittest.TestCase):
    def test_only_unique_fresh_measurements_are_returned_and_copy_is_independent(self):
        rig = GateHarness()
        subscriber = SG2RobotStateSubscriber(receiver=rig)
        first = subscriber.recv()
        self.assertEqual(first["joints"]["head"]["position"], [.3, .15])
        self.assertEqual(first["sensors"], {})
        first["joints"]["head"]["position"][0] = 0.
        self.assertEqual(rig.latest()["q"][1], .3)
        self.assertIsNone(subscriber.recv())
        rig.refresh(.01)
        self.assertIsNotNone(subscriber.recv())
        rig.refresh(.21, source=False)
        self.assertIsNone(subscriber.recv())
        self.assertIn("stale", subscriber.reason)
        subscriber.close()
        self.assertTrue(rig.closed)
        self.assertIsNone(subscriber.recv())
        self.assertEqual(subscriber.reason, "receiver_closed")


if __name__ == "__main__":
    unittest.main()
