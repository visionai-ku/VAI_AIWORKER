"""Collision policy at the REAL gate, using only in-memory robot transport."""

import unittest
from unittest.mock import patch

import numpy as np

# Import the module so discovery does not collect its TestCase a second time.
import test_fixed_quest_real as real_fixture

from package.intent_teleop import IntentTeleopSettings, build_planner


COLLISION = {
    "signed_dist": -.001, "is_lower_bound": False,
    "geom1_idx": 1, "geom2_idx": 2,
    "body1_name": "gripper_l_r1", "body2_name": "gripper_r_r1",
}


class TeleopCollisionPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        real_fixture.FixedQuestRealTest.setUpClass()

    @classmethod
    def tearDownClass(cls):
        real_fixture.FixedQuestRealTest.tearDownClass()

    def setUp(self):
        self.flow = real_fixture.FixedQuestRealTest(methodName="runTest")
        self.flow.setUp()
        self.flow.planner.teleop_collision_check = False

    def assert_command_sent(self, result, state):
        self.assertEqual(result["state"], state, result["reason"])
        self.assertTrue(result["transmitted"])
        packet = self.flow.publisher.messages[-1]
        self.assertEqual((packet["mode"], packet["motion"]), ("command", "teleop"))

    def prepare_home(self, *, target_delta=.02):
        f = self.flow
        real = f.make_real()
        real.tick_grip(None, f.feedback(0.), now=0.)
        f.planner.q_home[f.names.index("arm_l_joint1")] += target_delta
        requested = real.tick_grip(None, f.feedback(.01), now=.01, action="home")
        self.assertEqual(requested["state"], "ARMING", requested["reason"])
        self.assertEqual(f.publisher.messages, [])
        return real

    def test_settings_default_and_planner_wiring_keep_home_strict(self):
        self.assertFalse(IntentTeleopSettings().TELEOP_COLLISION_CHECK)
        for invalid in (0, 1, None, "False", np.bool_(False)):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "TELEOP_COLLISION_CHECK must be bool"):
                    IntentTeleopSettings(TELEOP_COLLISION_CHECK=invalid).validate()
        f = self.flow
        models = {
            "env": f.env, "env_ik": f.env_ik, "ik_solvers": f.solvers,
            "arm_joint_names": f.arms, "command_joint_names": f.names, "guard": f.guard,
        }
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                cfg = IntentTeleopSettings(TELEOP_COLLISION_CHECK=enabled).validate()
                _, planner = build_planner(models, cfg)
                self.assertIs(planner.teleop_collision_check, enabled)
                with patch.object(f.guard, "_minimum_distance", return_value=COLLISION):
                    teleop = planner.check_command(f.robot_initial, f.robot_initial)
                    home = planner.check_command(f.robot_initial, f.robot_initial, motion="home")
                self.assertEqual(teleop["accepted"], not enabled)
                self.assertFalse(home["accepted"])
                self.assertEqual(home["reason"], "collision_clearance")

    def test_teleop_arming_and_reclutch_ignore_collision_query(self):
        f = self.flow
        with patch.object(f.guard, "_minimum_distance", return_value=COLLISION) as distance:
            active = f.start_grip()
            self.assert_command_sent(active, "ACTIVE")
            session = f.real.session_id
            f.real.tick_grip(f.frame(2), f.armed_feedback(.02), now=.02)
            arming = f.real.tick_grip(f.frame(3, grip=True), f.feedback(.03), now=.03)
            self.assertEqual(arming["state"], "ARMING", arming["reason"])
            resumed = f.real.tick_grip(f.frame(4, grip=True), f.armed_feedback(.04), now=.04)
            self.assert_command_sent(resumed, "ACTIVE")
            self.assertNotEqual(f.real.session_id, session)
            distance.assert_not_called()

    def test_teleop_paced_and_measured_paths_bypass_collision_only(self):
        f = self.flow
        f.start_grip()
        f.planner.ik_period = .001
        previous = f.planner.q_command.copy()
        goal = previous.copy()
        goal[f.names.index("arm_l_joint1")] += .03
        measured = previous.copy()
        measured[f.names.index("arm_r_joint1")] += .002

        def pace(previous, candidate, measured, **kwargs):
            return (previous + candidate) / 2, {
                "tracking_paced": True, "ack_paced": False, "command_scale": .5,
            }

        with patch.object(f.guard, "_minimum_distance", return_value=COLLISION) as distance, \
                patch.object(f.guard, "check", wraps=f.guard.check) as check, \
                patch.object(f.planner, "_solve", return_value=goal), \
                patch.object(f.real, "_pace_candidate", side_effect=pace):
            result = f.real.tick_grip(f.frame(3, grip=True),
                                      f.armed_feedback(.03, q=measured), now=.03)
        self.assert_command_sent(result, "ACTIVE")
        self.assertFalse(np.array_equal(result["q_command"], previous))
        self.assertTrue(result["tracking_paced"])
        self.assertGreaterEqual(check.call_count, 3)
        for call in check.call_args_list:
            self.assertFalse(call.kwargs.get("check_collision", True))
        np.testing.assert_array_equal(check.call_args_list[-2].args[0], previous)
        np.testing.assert_array_equal(check.call_args_list[-2].args[1], result["q_command"])
        np.testing.assert_array_equal(check.call_args_list[-1].args[0], measured)
        np.testing.assert_array_equal(check.call_args_list[-1].args[1], result["q_command"])
        distance.assert_not_called()

    def test_teleop_tracking_pause_and_ik_wait_keep_collision_bypass(self):
        f = self.flow
        f.start_grip()
        f.real.tracking_pause_s = .1
        f.planner.ik_period = .001
        f.planner.ik_recovery_timeout_s = .2
        with patch.object(f.guard, "_minimum_distance", return_value=COLLISION) as distance:
            lost = f.frame(2, grip=True)
            lost["hands"]["left"].update(valid=False, emulated=True)
            paused = f.real.tick_grip(lost, f.armed_feedback(.02), now=.02)
            self.assert_command_sent(paused, "TRACKING_PAUSED")
            with patch.object(f.planner, "_solve", side_effect=real_fixture.IKTargetUnreachable("unreachable")):
                waiting = f.real.tick_grip(f.frame(3, grip=True), f.armed_feedback(.03), now=.03)
            self.assert_command_sent(waiting, "IK_WAIT")
            distance.assert_not_called()

    def test_collision_bypass_retains_real_feedback_and_paced_joint_limits(self):
        for invalid_stage in ("feedback", "paced"):
            with self.subTest(stage=invalid_stage):
                self.setUp()
                f = self.flow
                f.arm()
                invalid = f.planner.q_command.copy()
                index = f.names.index("arm_l_joint1")
                invalid[index] = f.real.bounds[index, 1] + .01
                count = len(f.publisher.messages)
                with patch.object(f.guard, "_minimum_distance", return_value=COLLISION) as distance:
                    if invalid_stage == "feedback":
                        result = f.real.tick(f.frame(3), f.armed_feedback(.03, q=invalid),
                                             enabled=True, now=.03)
                        self.assertIn("measured joints outside verified real limits", result["reason"])
                    else:
                        with patch.object(f.real, "_pace_candidate", return_value=(invalid, {})):
                            result = f.real.tick(f.frame(3), f.armed_feedback(.03), enabled=True, now=.03)
                        self.assertIn("paced path rejected: joint_limit_end", result["reason"])
                    distance.assert_not_called()
                self.assertEqual(result["state"], "FAULT")
                self.assertEqual([p["mode"] for p in f.publisher.messages[count:]], ["stop"])

    def test_initial_and_reset_pose_collision_block_before_any_arm(self):
        for pose in ("initial", "reset"):
            with self.subTest(pose=pose):
                self.setUp()
                f = self.flow
                real = f.make_real()
                real.tick_grip(None, f.feedback(0.), now=0.)
                target = f.planner.q_home.copy()
                target[f.names.index("arm_l_joint1")] += .02
                if pose == "initial":
                    f.planner.q_home = target
                else:
                    real.request_pose_target(target)
                with patch.object(f.guard, "_minimum_distance", return_value=COLLISION) as distance:
                    result = real.tick_grip(None, f.feedback(.01), now=.01, action="home")
                self.assertEqual(result["state"], "FAULT", result["reason"])
                self.assertFalse(result["home_plan"]["ok"])
                self.assertEqual(result["guard"]["reason"], "collision_clearance")
                self.assertGreater(distance.call_count, 0)
                self.assertEqual(f.publisher.messages, [])

    def test_pending_home_checks_collision_before_first_arm_and_retry(self):
        for retry in (False, True):
            with self.subTest(retry=retry):
                self.setUp()
                f = self.flow
                real = self.prepare_home()
                if retry:
                    real.tick_grip(None, f.feedback(.02), now=.02)
                    self.assertEqual([p["mode"] for p in f.publisher.messages], ["arm"])
                count = len(f.publisher.messages)
                now = .03 if retry else .02
                with patch.object(f.guard, "_minimum_distance", return_value=COLLISION) as distance:
                    result = real.tick_grip(None, f.feedback(now), now=now)
                self.assertEqual(result["state"], "FAULT", result["reason"])
                self.assertEqual(result["guard"]["reason"], "collision_clearance")
                self.assertGreater(distance.call_count, 0)
                self.assertNotIn("arm", [p["mode"] for p in f.publisher.messages[count:]])
                self.assertFalse(real.home_pending)

    def test_home_pacing_rechecks_collision_even_with_teleop_bypass(self):
        f = self.flow
        real = self.prepare_home()
        real.tick_grip(None, f.feedback(.02), now=.02)
        started = real.tick_grip(None, f.armed_feedback(.03), now=.03)
        self.assertEqual(started["state"], "HOMING", started["reason"])
        count = len(f.publisher.messages)
        original_distance = f.guard._minimum_distance
        with patch.object(f.guard, "_minimum_distance", wraps=original_distance) as distance:
            def pace(previous, candidate, measured, **kwargs):
                self.assertTrue(kwargs["home"])
                self.assertFalse(np.array_equal(previous, candidate))
                distance.return_value = COLLISION
                return (previous + candidate) / 2, {"tracking_paced": True}

            with patch.object(real, "_pace_candidate", side_effect=pace):
                result = real.tick_grip(None, f.armed_feedback(.04), now=.04)
        self.assertEqual(result["state"], "FAULT", result["reason"])
        self.assertIn("paced path rejected: collision_clearance", result["reason"])
        self.assertEqual([p["mode"] for p in f.publisher.messages[count:]], ["stop"])

    def test_home_arrival_measured_check_stays_strict_after_planner_clears_homing(self):
        f = self.flow
        real = self.prepare_home(target_delta=0.)
        real.tick_grip(None, f.feedback(.02), now=.02)
        started = real.tick_grip(None, f.armed_feedback(.03), now=.03)
        self.assertEqual(started["state"], "HOMING", started["reason"])
        count = len(f.publisher.messages)
        original_tick = f.planner.tick
        arrivals = []
        with patch.object(f.guard, "_minimum_distance", wraps=f.guard._minimum_distance) as distance:
            def tick(*args, **kwargs):
                result = original_tick(*args, **kwargs)
                if result["state"] == "HOME":
                    arrivals.append(result["state"])
                    self.assertFalse(f.planner.homing)
                    distance.return_value = COLLISION
                return result

            with patch.object(f.planner, "tick", side_effect=tick):
                result = real.tick_grip(None, f.armed_feedback(.04), now=.04)
        self.assertEqual(arrivals, ["HOME"])
        self.assertEqual(result["state"], "FAULT", result["reason"])
        self.assertIn("measured path rejected: collision_clearance", result["reason"])
        self.assertEqual([p["mode"] for p in f.publisher.messages[count:]], ["stop"])


if __name__ == "__main__":
    unittest.main()
