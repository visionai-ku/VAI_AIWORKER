"""Boundary feedback, capability compatibility and robot-clock rate tests."""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from package.worker.fixed_quest_protocol import (JOINT_NAMES, JointStateTracker, StrictCommandGate,
    FIXED_QUEST_VR_GATE_OPTIONS, make_joint_groups, parse_joint_groups)
from package.fixed_quest_transport import SG2RobotFeedbackCache


class FeedbackGripperUpgradeTest(unittest.TestCase):
    def setUp(self):
        self.q = [0.] * 19
        self.q[6] = self.q[13] = -1.57

    def test_recorded_boundary_samples_remain_fresh_end_to_end(self):
        inbound, outbound = JointStateTracker(), JointStateTracker()
        gate = StrictCommandGate(inbound)
        cache = SG2RobotFeedbackCache()
        for i, value in enumerate((-.000323574072445, -.0000119842249054, -.0000239684498107)):
            q = self.q.copy(); q[4] = value
            t = 1. + i * .3
            for tracker in (inbound, outbound):
                tracker.update(JOINT_NAMES, q, i + 1, t)
            state = outbound.snapshot(t)
            state.update(bridge=gate.status(t), bridge_age_s=0.)
            self.assertTrue(cache.ingest(state, received_at=t + 100))
            self.assertTrue(cache.latest(t + 100)['ready'])
            self.assertEqual(cache.latest(t + 100)['q'][4], 0.)
            adjustment = state['joint_state_diagnostics']['boundary_adjustments']['arm_l_joint2']
            self.assertEqual(adjustment['raw_position'], value)
            with self.assertRaises(ValueError):
                make_joint_groups(q)  # Never extend command limits.
        self.assertFalse(cache.latest(t + 100.21)['ready'])

    def test_excess_or_stalled_timestamp_never_refreshes_state(self):
        tracker = JointStateTracker()
        tracker.update(JOINT_NAMES, self.q, 1, 1.)
        bad = self.q.copy(); bad[4] = -.0006
        with self.assertRaises(ValueError):
            tracker.update(JOINT_NAMES, bad, 2, 1.2)
        tiny = self.q.copy(); tiny[4] = -.00002
        self.assertFalse(tracker.update(JOINT_NAMES, tiny, 1, 1.21))
        self.assertFalse(tracker.snapshot(1.21)['ready'])
        self.assertEqual(tracker.seq, 1)
        bad = self.q.copy(); bad[-1] = -.00002
        with self.assertRaises(ValueError):
            tracker.update(JOINT_NAMES, bad, 3, 1.22)

    def setup_gate(self, gate_options=None):
        tracker = JointStateTracker()
        tracker.update(JOINT_NAMES, self.q, 1, 1.)
        options = FIXED_QUEST_VR_GATE_OPTIONS if gate_options is None else gate_options
        gate = StrictCommandGate(tracker, **options)
        def packet(seq, t, q, mode='command', **extra):
            return dict(type='sg2_command', version=1, session_id='test', seq=seq, bridge_id=gate.bridge_id,
                        mode=mode, motion='teleop', feedback_time_s=t, valid_for_s=.2,
                        joints=make_joint_groups(q), **extra)
        self.assertTrue(gate.receive(packet(0, 1., self.q, mode='arm'), 1.)['accepted'])
        return tracker, gate, packet

    def test_new_speed_legacy_fallback_head_and_home(self):
        tracker, gate, packet = self.setup_gate()
        q = self.q.copy(); q[-1] = math.radians(120) * .02
        self.assertEqual(gate.receive(packet(1, 1.02, q), 1.02)['reason'], 'joint_rate_limit')
        p = packet(1, 1.02, q, gripper_speed_rad_s=math.radians(120))
        self.assertTrue(gate.receive(p, 1.02)['accepted'])
        self.assertAlmostEqual(gate.speed[1], math.radians(60))
        self.assertAlmostEqual(gate.home_speed[-1], math.radians(60))  # B reset after initial arm alignment
        q[-1] += math.radians(120) * .02
        p = packet(2, 1.04, q, gripper_speed_rad_s=math.radians(120)); p['motion'] = 'home'
        self.assertEqual(gate.receive(p, 1.04)['reason'], 'joint_rate_limit')

    def test_reset_home_accepts_sixty_and_rejects_faster_rotary_commands(self):
        for joint in ('arm_l_joint1', 'arm_r_joint1', 'gripper_l_joint1', 'gripper_r_joint1'):
            for speed, accepted in ((60., True), (70., False)):
                with self.subTest(joint=joint, speed=speed):
                    tracker, gate, packet = self.setup_gate()
                    q = self.q.copy()
                    q[JOINT_NAMES.index(joint)] += math.radians(speed) * .02
                    p = packet(1, 1.02, q); p['motion'] = 'home'
                    result = gate.receive(p, 1.02)
                    self.assertEqual(result['accepted'], accepted, result)
                    self.assertAlmostEqual(gate.status(1.02)['limits']['home_joint_speed_rad_s'], math.radians(60))
                    if accepted:
                        self.assertEqual(parse_joint_groups(result['publish_joints']), q)
                    else:
                        self.assertEqual(result['reason'], 'joint_rate_limit')
                        self.assertAlmostEqual(gate.rejection_detail['speed_limit'], math.radians(60))
                        self.assertEqual(gate.last_command_seq, 0)
                        self.assertEqual(gate.deadline, 1.2)

    def test_generic_home_and_lift_keep_existing_limits(self):
        for options in ({}, FIXED_QUEST_VR_GATE_OPTIONS):
            cases = [('lift_joint', -.01, True), ('lift_joint', -.02, False)]
            if not options:
                cases += [('arm_l_joint1', math.radians(10), True),
                          ('arm_l_joint1', math.radians(20), False)]
            for joint, speed, accepted in cases:
                with self.subTest(vr_profile=bool(options), joint=joint, speed=speed):
                    tracker, gate, packet = self.setup_gate(options)
                    q = self.q.copy()
                    q[JOINT_NAMES.index(joint)] += speed * .02
                    p = packet(1, 1.02, q); p['motion'] = 'home'
                    result = gate.receive(p, 1.02)
                    self.assertEqual(result['accepted'], accepted, result)
                    self.assertEqual(gate.home_speed[0], .01)
                    if not options:
                        self.assertAlmostEqual(gate.home_speed[3], math.radians(10))
                    if not accepted:
                        self.assertEqual(result['reason'], 'joint_rate_limit')

    def test_invalid_gripper_speed_and_compressed_arrival_do_not_renew_lease(self):
        tracker, gate, packet = self.setup_gate()
        for value in (True, '2', math.inf, math.nan, -1., 0., math.radians(121)):
            self.assertFalse(gate.receive(packet(1, 1.01, self.q, gripper_speed_rad_s=value), 1.01)['accepted'])
            self.assertEqual(gate.last_command_seq, 0)
            self.assertEqual(gate.deadline, 1.2)
        q = self.q.copy(); q[-1] = math.radians(120) * .02
        p = packet(1, 1.01, q, gripper_speed_rad_s=math.radians(120))
        self.assertEqual(gate.receive(p, 1.01)['reason'], 'joint_rate_limit')
        self.assertTrue(gate.receive(p, 1.02)['accepted'])

    def test_variable_cadence_at_selected_speed_with_fresh_following(self):
        for speed in (60., 90., 100., 120.):
            tracker, gate, packet = self.setup_gate()
            q, t = self.q.copy(), 1.
            for i, dt in enumerate((.01, .02, .03, .01, .03, .02), 1):
                t += dt
                tracker.update(JOINT_NAMES, q, i + 1, t)
                q = q.copy(); q[-1] += math.radians(speed) * dt
                result = gate.receive(packet(i, t, q, gripper_speed_rad_s=math.radians(speed)), t)
                self.assertTrue(result['accepted'], result)
                self.assertEqual(parse_joint_groups(result['publish_joints']), q)


if __name__ == '__main__':
    unittest.main()
