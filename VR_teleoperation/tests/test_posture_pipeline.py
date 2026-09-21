"""Posture IK through measured feedback, clutch lifecycle and strict output gate.

The bridge is an in-memory ideal follower; no sockets, ROS or hardware are used.
The longer solver budget here isolates integration correctness from CPU timing.
"""
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

# Import the module, not its TestCase class: unittest discovery must not collect
# the existing integration suite a second time through this module's globals.
import test_fixed_quest_integration as integration

from package.posture_ik import SG2PostureIK


class _DisabledPublisher:
    enabled = False

    def send(self, packet):
        raise AssertionError("DRY RUN must not publish")

    def close(self):
        pass


class PosturePipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        integration.FixedQuestIntegrationTest.setUpClass()

    @classmethod
    def tearDownClass(cls):
        integration.FixedQuestIntegrationTest.tearDownClass()

    def setUp(self):
        self.flow = integration.FixedQuestIntegrationTest(methodName="runTest")
        self.flow.setUp()
        planner = self.flow.planner
        planner.posture_solvers = {
            hand: SG2PostureIK(self.flow.env_ik, hand, max_solve_time_s=.5)
            for hand in ("left", "right")
        }
        planner.ik_period = .02
        planner.coordinated_arm_rate_limit = True
        planner.retry_ik_near_target = True
        planner.ik_recovery_timeout_s = .4

    def references(self):
        return {hand: solver.reference_qpos
                for hand, solver in self.flow.planner.posture_solvers.items()}

    def assert_references_equal(self, expected):
        for hand, actual in self.references().items():
            np.testing.assert_array_equal(actual, expected[hand], err_msg=hand)

    def start(self):
        self.assertEqual(self.flow.tick_grip(0, grips=(True, True))["state"], "ARMING")
        result = self.flow.tick_grip(1, grips=(True, True))
        self.assertEqual(result["state"], "ACTIVE", result["reason"])
        return result

    def test_delayed_ack_captures_latest_measured_reference_only_at_activation(self):
        f = self.flow
        f.heartbeat_period_steps = 4
        idle = f.tick_grip(0)
        self.assertEqual(idle["state"], "WAIT_GRIP")
        self.assertTrue(all(q is None for q in self.references().values()))
        self.assertEqual(f.publisher.messages, [])
        for index in range(1, 4):
            f.publisher.q[3] += .003
            result = f.tick_grip(index, grips=(True, True), height=.01 * index)
            self.assertEqual(result["state"], "ARMING", result["reason"])
            self.assertTrue(all(q is None for q in self.references().values()))
            np.testing.assert_array_equal(result["q_command"], f.publisher.q)
        measured = f.publisher.q.copy()
        active = f.tick_grip(4, grips=(True, True), height=.08)
        self.assertEqual(active["state"], "ACTIVE", active["reason"])
        self.assertTrue(active["posture_reference_captured"])
        for reference in self.references().values():
            np.testing.assert_array_equal(reference, f.env_measured.data.qpos)
        np.testing.assert_allclose(active["q_command"], measured, atol=1e-12)
        self.assertEqual([packet["mode"] for packet in f.publisher.messages[:3]], ["arm"] * 3)

    def test_reference_survives_release_measured_resync_reset_and_regrip(self):
        f = self.flow
        self.start()
        references = self.references()
        f.tick_grip(2)
        f.publisher.q[3] += .01
        actual = f.publisher.q.copy()
        idle = f.tick_grip(3)
        np.testing.assert_array_equal(idle["q_command"], actual)
        np.testing.assert_array_equal(f.publisher.q, actual)
        f.planner.reset()
        self.assert_references_equal(references)
        self.assertEqual(f.tick_grip(4, grips=(True, True))["state"], "ARMING")
        result = f.tick_grip(5, grips=(True, True))
        self.assertEqual(result["state"], "ACTIVE", result["reason"])
        self.assert_references_equal(references)
        self.assertTrue(all(outcome["accepted"] for outcome in f.publisher.outcomes))

    def test_first_dry_reference_survives_switch_to_live_and_new_lease(self):
        f = self.flow
        f.real = integration.SG2FixedQuestReal(
            f.planner, f.env_measured, f.layout, _DisabledPublisher(), dry_run=True)
        initial = f.tick_grip(0, grips=(True, True))
        self.assertEqual(initial["state"], "ACTIVE", initial["reason"])
        references = self.references()
        self.assertEqual(f.publisher.messages, [])
        f.tick_grip(1)
        f.publisher.q[3] += .01
        self.assertTrue(f.real.set_dry_run(False, lambda enabled: f.publisher))
        f.tick_grip(2)  # mode-change input barrier
        f.tick_grip(3)  # fresh both-released sample after the barrier
        self.assertEqual(f.tick_grip(4, grips=(True, True))["state"], "ARMING")
        active = f.tick_grip(5, grips=(True, True))
        self.assertEqual(active["state"], "ACTIVE", active["reason"])
        self.assert_references_equal(references)
        self.assertTrue(all(outcome["accepted"] for outcome in f.publisher.outcomes))

    def test_failed_reference_capture_is_atomic_and_sends_no_motion(self):
        f = self.flow
        self.assertEqual(f.tick_grip(0, grips=(True, True))["state"], "ARMING")
        with patch.object(f.planner.posture_solvers["right"], "capture_reference",
                          side_effect=ValueError("undefined elbow reference")):
            failed = f.tick_grip(1, grips=(True, True))
        self.assertEqual(failed["state"], "FAULT", failed["reason"])
        self.assertIn("posture reference unavailable", failed["reason"])
        self.assertTrue(all(q is None for q in self.references().values()))
        self.assertNotIn("command", [packet["mode"] for packet in f.publisher.messages])
        np.testing.assert_array_equal(f.publisher.q, f.initial)

    def test_final_post_pacing_elbow_violation_stops_without_publishing_candidate(self):
        f = self.flow
        self.start()
        previous = f.publisher.q.copy()
        left = f.planner.posture_solvers["left"]
        left.elbow_limit_rad = np.deg2rad(1.)
        invalid = None
        for name in left.joints_use:
            for delta in (-.2, .2):
                candidate = previous.copy()
                index = list(integration.JOINT_NAMES).index(name)
                candidate[index] += delta
                check = f.planner.check_posture_command(candidate)
                if not check["accepted"] and "initial-side" in check["reason"]:
                    invalid = candidate
                    break
            if invalid is not None:
                break
        self.assertIsNotNone(invalid, "fixture must supply an actual elbow-side violation")
        count = len(f.publisher.messages)
        clear = {"accepted": True, "reason": "accepted", "min_distance_m": .05}
        with patch.object(f.real, "_pace_candidate", return_value=(invalid, {
                "tracking_paced": True, "ack_paced": False, "command_scale": .5})), \
                patch.object(f.guard, "check", return_value=clear):
            failed = f.tick_grip(2, grips=(True, True))
        self.assertEqual(failed["state"], "FAULT", failed["reason"])
        self.assertIn("posture command rejected", failed["reason"])
        self.assertFalse(failed["posture_command_guard"]["accepted"])
        self.assertIn("attempted_posture_ik", failed)
        self.assertTrue(all(packet["mode"] == "stop" for packet in f.publisher.messages[count:]))
        np.testing.assert_array_equal(f.publisher.q, previous)

    def test_inaccurate_successful_solver_result_holds_all_joints_in_ik_wait(self):
        f = self.flow
        self.start()
        previous = f.publisher.q.copy()
        left = f.planner.posture_solvers["left"]

        def inaccurate(env, target):
            # Claim success while returning the seed, so the planner must verify
            # actual FK error against its 1 mm full-pose acceptance threshold.
            return {"success": True, "qpos_used_best": env.get_qpos(joint_names=left.joints_use).copy(),
                    "reason": "synthetic inaccurate candidate"}

        frame, feedback, now = f.inputs(2, grip=True, height=.04)
        for hand in frame["hands"].values():
            hand["trigger"] = 1.
        with patch.object(left, "solve", side_effect=inaccurate):
            held = f.real.tick_grip(frame, feedback, now=now)
        self.assertEqual(held["state"], "IK_WAIT", held["reason"])
        self.assertGreater(held["ik_errors"]["left"]["position_m"], left.position_tolerance)
        np.testing.assert_array_equal(held["q_command"], previous)
        np.testing.assert_array_equal(f.publisher.q, previous)
        self.assertTrue(held["posture_command_guard"]["accepted"])

    def test_motion_and_rotation_keep_strict_ik_accuracy_and_output_guards(self):
        f = self.flow
        first = self.start()
        previous = first["q_command"].copy()
        references = self.references()
        for index in range(2, 14):
            progress = index - 1
            frame, feedback, now = f.inputs(index, grip=True, height=progress * .001)
            frame["hands"]["left"]["quaternion_xyzw"] = Rotation.from_euler(
                "z", progress * .002).as_quat().tolist()
            result = f.real.tick_grip(frame, feedback, now=now)
            self.assertEqual(result["state"], "ACTIVE", result["reason"])
            for hand, error in result["ik_errors"].items():
                solver = f.planner.posture_solvers[hand]
                self.assertLessEqual(error["position_m"], solver.position_tolerance)
                self.assertLessEqual(error["rotation_rad"], solver.rotation_tolerance)
                self.assertTrue(result["posture_ik"][hand]["success"])
            self.assertTrue(result["guard"]["accepted"])
            self.assertTrue(result["posture_command_guard"]["accepted"])
            self.assertTrue(np.all(np.abs(result["q_command"] - previous)
                                   <= f.planner.speed_limits * .02 + 1e-10))
            previous = result["q_command"].copy()
        self.assertGreater(np.max(np.abs(previous - first["q_command"])), 1e-3)
        self.assertTrue(all(outcome["accepted"] for outcome in f.publisher.outcomes))
        self.assert_references_equal(references)


if __name__ == "__main__":
    unittest.main()
