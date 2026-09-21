"""Controller -> existing SG2 IK -> joint limiter -> actual MuJoCo integration."""

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
from package.joint_marker_real import apply_sg2_joint_rate_limit
from package.fixed_quest_scene import SG2CommandGuard, build_fixed_quest_scene
from package.fixed_quest_sim import (SG2FixedQuestSimulation, reset_sg2_sim, SyntheticControllerSource,
                                     ControllerRecorder, ReplayControllerSource, IKTargetUnreachable)


class FixedQuestSimulationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (PROJECT_DIR / ".cache").mkdir(exist_ok=True)
        cls.cache = tempfile.TemporaryDirectory(dir=PROJECT_DIR / ".cache")
        xml = build_fixed_quest_scene(PROJECT_DIR / "xml/ffw_sg2.xml", PROJECT_DIR / "xml/floor_simple_white.xml", Path(cls.cache.name) / "scene.xml")
        cls.env, cls.env_ik, cls.env_check = [MuJoCoParser(rel_xml_path=xml, verbose=False) for _ in range(3)]
        for model_env in (cls.env, cls.env_ik, cls.env_check):
            reset_sg2_sim(model_env)
        cls.names = list(SG2_REAL_COMMAND_JOINT_NAMES) + list(SG2_GRIPPER_DRIVER_JOINT_NAMES.values())
        cls.solvers, cls.arms = create_sg2_arm_ik_solvers(cls.env_ik, MuJoCoParser, BodyPositionIKSolverRevPriBase)
        cls.guard = SG2CommandGuard(cls.env_check, cls.names)

    @classmethod
    def tearDownClass(cls):
        cls.cache.cleanup()

    def setUp(self):
        self.sim = self.make_sim()

    def make_sim(self, mode="kinematic", **options):
        for model_env in (self.env, self.env_ik, self.env_check):
            reset_sg2_sim(model_env)
        self.q_init = self.env.get_qpos(joint_names=self.names).copy()
        return SG2FixedQuestSimulation(self.env, self.env_ik, self.solvers, self.arms,
                                       self.names, ControllerMapper(), self.guard, mode=mode, **options)

    def run_source(self, sim, duration=5.5):
        source = SyntheticControllerSource()
        rows = []
        for index in range(round(duration * 100) + 1):
            t = index / 100
            result = sim.tick(source.latest(t), True, t)
            self.assertNotEqual(result["state"], "FAULT", result)
            rows.append(result)
        return rows

    def offset_home_start(self, sim, angle=np.deg2rad(5)):
        """Place the test fixture away from home before starting its control loop."""
        q = sim.q_home.copy()
        q[self.names.index("arm_l_joint1")] += angle
        self.env.forward(q=q, joint_names=self.names)
        self.env.data.ctrl[self.env.get_idxs_step(joint_names=self.names)] = q
        sim.reset()
        return q

    @staticmethod
    def controller_frame(index, grip):
        t = index / 100
        frame = SyntheticControllerSource().latest(t)
        frame.update(seq=index, sample_time_ms=1000 * t, received_at=t)
        for hand in frame["hands"].values():
            hand.update(grip=float(grip), trigger=0.)
        return frame

    def test_home_without_controllers_returns_in_limited_steps(self):
        q_start = self.offset_home_start(self.sim)
        np.testing.assert_array_equal(self.sim.q_home, self.q_init)
        first = self.sim.tick(None, False, 0., action="home")
        self.assertEqual(first["state"], "HOMING")
        np.testing.assert_array_equal(first["q_command"], q_start)
        rows = [first]
        for index in range(1, 101):
            result = self.sim.tick(None, False, index / 100)
            self.assertNotEqual(result["state"], "FAULT", result["reason"])
            self.assertEqual(result["active"], [])
            rows.append(result)
            if result["state"] == "HOME":
                break
        self.assertEqual(rows[-1]["state"], "HOME")
        self.assertGreater(len(rows), 2)
        self.assertFalse(self.sim.homing)
        self.assertTrue(self.sim.home_rearm_required)
        np.testing.assert_allclose(rows[-1]["q_command"], self.q_init, atol=1e-9)
        q = np.array([row["q_command"] for row in rows])
        self.assertTrue(np.all(np.abs(np.diff(q, axis=0)) <= self.sim.speed_limits * .01 + 1e-12))
        np.testing.assert_allclose(rows[-1]["home_error"], 0., atol=1e-9)

    def test_teleop_collision_policy_is_explicit_and_home_stays_guarded(self):
        self.assertTrue(self.sim.teleop_collision_check)
        q_collision = self.sim.q_home.copy()
        q_collision[self.names.index("arm_l_joint3")] = -2.
        self.assertFalse(self.sim.check_command(q_collision,q_collision)["accepted"])
        self.sim.teleop_collision_check = False
        unchecked = self.sim.check_command(q_collision,q_collision)
        self.assertTrue(unchecked["accepted"],unchecked)
        self.assertFalse(unchecked["collision_check_enabled"])
        home = self.sim.check_command(q_collision,q_collision,motion="home")
        self.assertFalse(home["accepted"],home)
        self.assertTrue(home["collision_check_enabled"])
        with self.assertRaises(ValueError):
            self.sim.check_command(q_collision,q_collision,motion="typo")
        for invalid in (None,0,1,"false"):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                self.make_sim(teleop_collision_check=invalid)

    def test_teleop_near_contact_advances_with_collision_off_and_keeps_rate_limit(self):
        sim = self.make_sim(teleop_collision_check=False)
        q_collision = sim.q_home.copy()
        q_collision[self.names.index("arm_l_joint3")] = -2.
        self.env.forward(q=q_collision,joint_names=self.names)
        sim.reset()
        goal = q_collision.copy()
        goal[self.names.index("arm_l_joint1")] += .1
        sim.tick(self.controller_frame(0,False),True,0.)
        with patch.object(sim,"_solve",return_value=goal), \
                patch.object(sim.guard,"_minimum_distance",side_effect=AssertionError("collision query ran")):
            result = sim.tick(self.controller_frame(1,True),True,.01)
        self.assertEqual(result["state"],"ACTIVE",result["reason"])
        self.assertFalse(result["guard"]["collision_check_enabled"])
        movement = np.abs(result["q_command"]-q_collision)
        self.assertGreater(np.max(movement),0.)
        self.assertTrue(np.all(movement <= sim.speed_limits*.01+1e-12))

    def test_teleop_collision_off_applies_to_dynamic_and_ik_wait_checks(self):
        for mode in ("kinematic","dynamic"):
            with self.subTest(mode=mode):
                sim = self.make_sim(mode,teleop_collision_check=False,ik_recovery_timeout_s=1.)
                sim.tick(self.controller_frame(0,False),True,0.)
                with patch.object(sim.guard,"_minimum_distance",side_effect=AssertionError("collision query ran")), \
                        patch.object(sim.guard,"check",wraps=sim.guard.check) as guard:
                    active = sim.tick(self.controller_frame(1,True),True,.01)
                    self.assertEqual(active["state"],"ACTIVE",active["reason"])
                    self.assertEqual(guard.call_count,2 if mode == "dynamic" else 1)
                    for call in guard.call_args_list:
                        self.assertEqual(call.kwargs,{"check_collision":False})
                    guard.reset_mock()
                    with patch.object(sim,"_solve",side_effect=IKTargetUnreachable("left unavailable")):
                        waiting = sim.tick(self.controller_frame(5,True),True,.05)
                        still_waiting = sim.tick(self.controller_frame(6,True),True,.06)
                    self.assertEqual(waiting["state"],"IK_WAIT",waiting["reason"])
                    self.assertEqual(still_waiting["state"],"IK_WAIT",still_waiting["reason"])
                    self.assertEqual(guard.call_count,4 if mode == "dynamic" else 2)
                    for call in guard.call_args_list:
                        self.assertEqual(call.kwargs,{"check_collision":False})
                    np.testing.assert_array_equal(still_waiting["q_command"],active["q_command"])

    def test_home_final_step_checks_collision_when_teleop_check_is_off(self):
        for mode in ("kinematic","dynamic"):
            with self.subTest(mode=mode):
                sim = self.make_sim(mode,teleop_collision_check=False)
                q_start = self.offset_home_start(sim,angle=np.deg2rad(.1))
                first = sim.tick(None,False,0.,action="home")
                self.assertEqual(first["state"],"HOMING",first["reason"])
                with patch.object(sim.guard,"check",wraps=sim.guard.check) as guard:
                    completed = sim.tick(None,False,.01)
                self.assertEqual(completed["state"],"HOME",completed["reason"])
                self.assertFalse(sim.homing)
                self.assertTrue(completed["guard"]["collision_check_enabled"])
                self.assertEqual(guard.call_count,2 if mode == "dynamic" else 1)
                for call in guard.call_args_list:
                    self.assertEqual(call.kwargs,{})

                sim = self.make_sim(mode,teleop_collision_check=False)
                q_start = self.offset_home_start(sim,angle=np.deg2rad(.1))
                sim.tick(None,False,0.,action="home")
                collision = {"signed_dist":-.001,"is_lower_bound":False,"geom1_idx":1,
                             "geom2_idx":2,"body1_name":"arm_l_link7","body2_name":"arm_r_link7"}
                with patch.object(sim.guard,"_minimum_distance",return_value=collision):
                    rejected = sim.tick(None,False,.01)
                self.assertEqual(rejected["state"],"FAULT",rejected["reason"])
                self.assertTrue(rejected["guard"]["collision_check_enabled"])
                self.assertEqual(rejected["guard"]["reason"],"collision_clearance")
                self.assertTrue(sim.home_failed)
                np.testing.assert_array_equal(rejected["q_command"],q_start)

    def test_coordinated_limiter_preserves_each_arm_ratio_and_auxiliary_limits(self):
        previous = self.q_init.copy()
        goal = previous.copy()
        left = [self.names.index(n) for n in self.arms["left"]]
        right = [self.names.index(n) for n in self.arms["right"]]
        goal[left] += [.06, .03, -.02, .01, .005, -.01, .002]
        goal[right] += [.02, -.01, .005, .005, -.005, .002, .001]
        head = self.names.index("head_joint1")
        goal[head] += .05
        limits = np.ones(len(self.names))
        self.env_ik.forward(q=goal, joint_names=self.names)
        legacy = apply_sg2_joint_rate_limit(self.env_ik, self.names, previous, .01, limits)
        np.testing.assert_allclose(legacy["qpos"], previous + np.clip(goal-previous, -.01, .01))
        self.env_ik.forward(q=goal, joint_names=self.names)
        result = apply_sg2_joint_rate_limit(self.env_ik, self.names, previous, .01, limits,
                                          coordinated_groups=[self.arms["left"], self.arms["right"]])
        np.testing.assert_allclose(result["qpos"][left]-previous[left], (goal-previous)[left]/6)
        np.testing.assert_allclose(result["qpos"][right]-previous[right], (goal-previous)[right]/2)
        self.assertAlmostEqual(result["qpos"][head]-previous[head], .01)
        self.assertTrue(np.all(np.abs(result["qpos"]-previous) <= .01+1e-12))
        np.testing.assert_allclose(result["group_scales"], [1/6, .5])
        self.env_ik.forward(q=goal, joint_names=self.names)
        held = apply_sg2_joint_rate_limit(self.env_ik, self.names, previous, 0., limits,
                                        coordinated_groups=[self.arms["left"], self.arms["right"]])
        np.testing.assert_array_equal(held["qpos"], previous)

    def test_coordinated_option_applies_to_teleop_but_preserves_home_limiter(self):
        self.assertFalse(self.sim.coordinated_arm_rate_limit)
        self.sim.coordinated_arm_rate_limit = True
        self.sim.tick(self.controller_frame(0, False), True, 0.)
        with patch("package.fixed_quest_sim.apply_sg2_joint_rate_limit", wraps=apply_sg2_joint_rate_limit) as limiter:
            result = self.sim.tick(self.controller_frame(1, True), True, .01)
            self.assertEqual(result["state"], "ACTIVE", result["reason"])
            self.assertEqual(limiter.call_args.kwargs["coordinated_groups"],
                             [self.arms["left"], self.arms["right"]])
            self.assertEqual(set(result["arm_rate_scales"]), {"left", "right"})
            self.sim.tick(None, True, .02, action="home")
            home = self.sim.tick(None, True, .03)
            self.assertIn(home["state"], ("HOMING", "HOME"), home["reason"])
            # HOME scales every joint together so the command stays on its
            # checked segment; the teleop arm grouping is not reused.
            self.assertEqual(limiter.call_args.kwargs["coordinated_groups"], [self.names])
            self.assertEqual(home["arm_rate_scales"], {})

    def test_near_ik_retry_halves_target_and_keeps_tighter_error_bound(self):
        self.sim.retry_ik_near_target = True
        targets = {h: get_sg2_wrist_control_T(self.env, h).copy() for h in ("left", "right")}
        targets["right"][0, 3] += .02
        good = {"position_m": 0., "rotation_rad": 0.}
        with patch("package.fixed_quest_sim.wrist_error", side_effect=[good,
                   {"position_m": .01005, "rotation_rad": .012},
                   {"position_m": .00305, "rotation_rad": .003}]):
            self.sim._solve(targets)
        self.assertEqual(self.sim.ik_retries, {"left": 0, "right": 1})
        initial = get_sg2_wrist_control_T(self.env, "right")
        self.assertAlmostEqual(self.sim.ik_targets["right"][0, 3]-initial[0, 3], .01)
        with patch("package.fixed_quest_sim.wrist_error", side_effect=[good,
                   {"position_m": .01005, "rotation_rad": .012},
                   {"position_m": .006, "rotation_rad": .003}]) as errors:
            with self.assertRaises(IKTargetUnreachable):
                self.sim._solve(targets)
            self.assertEqual(errors.call_count, 3)

    def test_cancel_home_holds_partial_command(self):
        self.offset_home_start(self.sim)
        self.sim.tick(None, False, 0., action="home")
        for index in range(1, 6):
            self.sim.tick(None, False, index / 100)
        held = self.sim.q_command.copy()
        self.assertFalse(np.allclose(held, self.sim.q_home))
        result = self.sim.tick(None, False, .06, action="cancel_home")
        self.assertEqual(result["state"], "DISABLED")
        self.assertFalse(self.sim.homing)
        self.assertTrue(self.sim.home_rearm_required)
        for index in range(7, 15):
            result = self.sim.tick(self.controller_frame(index, True), True, index / 100)
            self.assertEqual(result["active"], [])
            np.testing.assert_array_equal(result["q_command"], held)

    def test_home_requires_off_and_new_released_frame_before_vr_resume(self):
        self.offset_home_start(self.sim)
        self.sim.tick(None, False, 0., action="home")
        for index in range(1, 101):
            result = self.sim.tick(self.controller_frame(index, True), False, index / 100)
            self.assertEqual(result["active"], [])
            if result["state"] == "HOME":
                break
        self.assertEqual(result["state"], "HOME")
        held = self.sim.q_command.copy()
        for index in range(index + 1, index + 4):
            result = self.sim.tick(self.controller_frame(index, True), True, index / 100)
            self.assertTrue(self.sim.home_rearm_required)
            self.assertEqual(result["active"], [])
            np.testing.assert_array_equal(result["q_command"], held)
        index += 1
        self.sim.tick(self.controller_frame(index, True), False, index / 100)
        self.assertFalse(self.sim.home_rearm_required)
        index += 1
        result = self.sim.tick(self.controller_frame(index, True), True, index / 100)
        self.assertEqual(result["state"], "WAIT_RELEASE")
        np.testing.assert_array_equal(result["q_command"], held)
        index += 1
        result = self.sim.tick(self.controller_frame(index, False), True, index / 100)
        self.assertEqual(result["state"], "READY")
        before = {hand: get_sg2_wrist_control_T(self.env, hand).copy() for hand in ("left", "right")}
        index += 1
        result = self.sim.tick(self.controller_frame(index, True), True, index / 100)
        self.assertEqual(set(result["active"]), {"left", "right"})
        for hand, target in result["targets"].items():
            np.testing.assert_allclose(target, before[hand], atol=1e-12)

    def test_finite_ik_failure_holds_and_recovers_after_release(self):
        self.sim.tick(self.controller_frame(0, False), True, 0.)
        held = self.sim.q_command.copy()
        with patch.object(self.sim, "_solve", side_effect=IKTargetUnreachable("target unavailable")):
            result = self.sim.tick(self.controller_frame(1, True), True, .01)
        self.assertEqual(result["state"], "HOLD")
        self.assertIsNone(self.sim.fault_reason)
        self.assertEqual(result["active"], [])
        np.testing.assert_array_equal(result["q_command"], held)
        for index in range(2, 5):
            result = self.sim.tick(self.controller_frame(index, True), True, index / 100)
            self.assertEqual(result["state"], "WAIT_RELEASE")
            np.testing.assert_array_equal(result["q_command"], held)
        result = self.sim.tick(self.controller_frame(5, False), True, .05)
        self.assertEqual(result["state"], "READY")
        result = self.sim.tick(self.controller_frame(6, True), True, .06)
        self.assertEqual(set(result["active"]), {"left", "right"})
        self.assertIsNone(self.sim.fault_reason)

    def test_ik_wait_holds_all_joints_and_retries_latest_target_without_reanchoring(self):
        self.sim.ik_recovery_timeout_s = 1.
        self.sim.ik_period = .04
        self.sim.tick(self.controller_frame(0, False), True, 0.)
        self.sim.tick(self.controller_frame(1, True), True, .01)
        held = self.sim.q_command.copy()
        anchors = copy.deepcopy(self.sim.mapper._anchors)
        # An unfinished previous goal must stop advancing as soon as IK fails.
        self.sim.q_goal[self.names.index("arm_l_joint1")] += .01
        self.sim.q_goal[self.sim.gripper_indices["left"]] = .5
        first = self.controller_frame(5, True)
        for hand in first["hands"].values():
            hand["trigger"] = 1.
        solve = self.sim._solve
        with patch.object(self.sim, "_solve", side_effect=IKTargetUnreachable("left unavailable")) as failed:
            result = self.sim.tick(first, True, .05)
            self.assertEqual(result["state"], "IK_WAIT", result["reason"])
            self.assertEqual(result["ik_recovery"]["status"], "waiting")
            for index in (6, 7, 8):
                moving = self.controller_frame(index, True)
                moving["hands"]["left"]["position_m"][1] += .002 * (index - 5)
                for hand in moving["hands"].values():
                    hand["trigger"] = 1.
                result = self.sim.tick(moving, True, index / 100)
                self.assertEqual(result["state"], "IK_WAIT", result["reason"])
                np.testing.assert_array_equal(result["q_command"], held)
                np.testing.assert_array_equal(self.sim.q_goal, held)
            self.assertEqual(failed.call_count, 1)
        latest = self.controller_frame(9, True)
        latest["hands"]["left"]["position_m"][1] += .008
        with patch.object(self.sim, "_solve", wraps=solve) as retry:
            recovered = self.sim.tick(latest, True, .09)
        self.assertEqual(recovered["state"], "ACTIVE", recovered["reason"])
        self.assertEqual(recovered["ik_recovery"]["status"], "recovered")
        self.assertEqual(retry.call_count, 1)
        self.assertTrue(np.all(np.abs(recovered["q_command"] - held)
                               <= self.sim.speed_limits * recovered["command_dt"] + 1e-12))
        np.testing.assert_array_equal(retry.call_args.args[0]["left"], recovered["targets"]["left"])
        self.assertGreater(np.linalg.norm(recovered["targets"]["left"][:3, 3]
                                          - anchors["left"]["eef_T"][:3, 3]), .001)
        for hand in ("left", "right"):
            np.testing.assert_array_equal(self.sim.mapper._anchors[hand]["eef_T"], anchors[hand]["eef_T"])

    def test_ik_wait_deadline_is_checked_even_between_scheduled_retries(self):
        self.sim.ik_recovery_timeout_s = .1
        self.sim.ik_period = .04
        self.sim.tick(self.controller_frame(0, False), True, 0.)
        held = self.sim.q_command.copy()
        with patch.object(self.sim, "_solve", side_effect=IKTargetUnreachable("left unavailable")) as failed:
            for index in range(1, 11):
                result = self.sim.tick(self.controller_frame(index, True), True, index / 100)
                self.assertEqual(result["state"], "IK_WAIT", result["reason"])
                np.testing.assert_array_equal(result["q_command"], held)
            self.assertEqual(failed.call_count, 3)  # .01, .05, .09, not each control tick.
            timeout = self.sim.tick(self.controller_frame(11, True), True, .11)
            self.assertEqual(failed.call_count, 3)
        self.assertEqual(timeout["state"], "HOLD", timeout["reason"])
        self.assertEqual(timeout["ik_recovery"]["status"], "timeout")
        self.assertGreaterEqual(timeout["ik_recovery"]["elapsed_s"], .1 - 1e-9)
        self.assertEqual(timeout["ik_recovery"]["attempts"], 3)
        self.assertEqual(timeout["active"], [])
        self.assertEqual(self.sim.tick(self.controller_frame(12, True), True, .12)["state"], "WAIT_RELEASE")
        np.testing.assert_array_equal(self.sim.q_command, held)

    def test_ik_wait_does_not_absorb_nonfinite_solver_fault_or_collision(self):
        for failure in ("nonfinite", "collision", "collision_between_retries"):
            with self.subTest(failure=failure):
                self.sim = self.make_sim()
                self.sim.ik_recovery_timeout_s = 1.
                self.sim.ik_period = .02
                self.sim.tick(self.controller_frame(0, False), True, 0.)
                held = self.sim.q_command.copy()
                with patch.object(self.sim, "_solve", side_effect=IKTargetUnreachable("left unavailable")):
                    self.assertEqual(self.sim.tick(self.controller_frame(1, True), True, .01)["state"], "IK_WAIT")
                if failure == "nonfinite":
                    with patch.object(self.sim, "_solve", side_effect=ValueError("non-finite IK result")):
                        result = self.sim.tick(self.controller_frame(3, True), True, .03)
                else:
                    reject = {"accepted": False, "reason": "recovery path collision", "min_distance_m": -.001}
                    index = 2 if failure == "collision_between_retries" else 3
                    with patch.object(self.sim.guard, "check", return_value=reject), \
                            patch.object(self.sim, "_solve", wraps=self.sim._solve) as solve:
                        result = self.sim.tick(self.controller_frame(index, True), True, index / 100)
                        self.assertEqual(solve.call_count, 0 if index == 2 else 1)
                self.assertEqual(result["state"], "FAULT", result["reason"])
                np.testing.assert_array_equal(result["q_command"], held)

    def test_nonfinite_solver_output_latches_fault_until_off(self):
        self.sim.tick(self.controller_frame(0, False), True, 0.)
        held = self.sim.q_command.copy()
        invalid = {"qpos_used_best": np.full(len(self.solvers["left"].joints_use), np.nan)}
        with patch.object(self.solvers["left"], "solve_ik", return_value=(None, invalid)):
            result = self.sim.tick(self.controller_frame(1, True), True, .01)
        self.assertEqual(result["state"], "FAULT")
        self.assertIn("non-finite", result["reason"])
        np.testing.assert_array_equal(result["q_command"], held)
        for index, grip in ((2, False), (3, True)):
            result = self.sim.tick(self.controller_frame(index, grip), True, index / 100)
            self.assertEqual(result["state"], "FAULT")
            np.testing.assert_array_equal(result["q_command"], held)
        self.sim.tick(self.controller_frame(4, False), False, .04)
        result = self.sim.tick(self.controller_frame(5, False), True, .05)
        self.assertEqual(result["state"], "READY")
        self.assertIsNone(self.sim.fault_reason)

    def test_home_does_not_bypass_existing_collision(self):
        self.sim.teleop_collision_check = False
        q_collision = self.sim.q_home.copy()
        q_collision[self.names.index("arm_l_joint3")] = -2.
        self.env.forward(q=q_collision, joint_names=self.names)
        self.sim.reset()
        result = self.sim.tick(None, False, 0., action="home")
        self.assertEqual(result["state"], "FAULT")
        self.assertIn("current pose already fails the command guard", result["reason"])
        self.assertFalse(result["home_plan"]["ok"])
        self.assertFalse(result["guard"]["accepted"])
        self.assertLess(result["guard"]["min_distance_m"], 0.)
        self.assertFalse(self.sim.homing)
        np.testing.assert_array_equal(result["q_command"], q_collision)
        result = self.sim.tick(None, False, .01)
        self.assertEqual(result["state"], "FAULT")
        np.testing.assert_array_equal(result["q_command"], q_collision)
        result = self.sim.tick(None, False, .02)
        self.assertEqual(result["state"], "FAULT", "Blocked return must remain visible while SIM control is OFF")
        np.testing.assert_array_equal(result["q_command"], q_collision)
        result = self.sim.tick(None, False, .03, action="cancel_home")
        self.assertEqual(result["state"], "DISABLED")

    def test_home_command_step_caps_short_loop_jitter(self):
        q_start = self.offset_home_start(self.sim)
        self.sim.tick(None, False, 0., action="home")
        result = self.sim.tick(None, False, .04)
        self.assertEqual(result["state"], "HOMING")
        self.assertEqual(result["dt"], .04)
        self.assertEqual(result["command_dt"], .02)
        dq = np.abs(result["q_command"] - q_start)
        self.assertGreater(np.max(dq), 0.)
        self.assertTrue(np.all(dq <= self.sim.speed_limits * .02 + 1e-12))

    def reject_first_home_check(self):
        """Reject only the direct return segment; later candidates use the real guard."""
        original = self.guard.check
        calls = []
        def check(q_start, q_end):
            calls.append((np.array(q_start), np.array(q_end)))
            if len(calls) == 1:
                return {"accepted": False, "reason": "collision_clearance", "min_distance_m": .004,
                        "pair": {"geom_ids": (1, 2), "body_names": ("arm_l_link5", "gripper_r_r1")},
                        "samples_checked": 3, "sweep_fraction": .5}
            return original(q_start, q_end)
        return patch.object(self.guard, "check", side_effect=check), calls

    def test_home_uses_first_fully_checked_path_and_stays_on_its_segments(self):
        q_start = self.sim.q_home.copy()
        j1, j3 = self.names.index("arm_r_joint1"), self.names.index("arm_r_joint3")
        q_start[j1] -= np.deg2rad(4)
        q_start[j3] += np.deg2rad(6)
        self.env.forward(q=q_start, joint_names=self.names)
        self.sim.reset()
        patcher, calls = self.reject_first_home_check()
        with patcher:
            first = self.sim.tick(None, False, 0., action="home")
        self.assertEqual(first["state"], "HOMING", first["reason"])
        plan = first["home_plan"]
        self.assertEqual(plan["strategy"], "unwind_j3j5_first")
        self.assertEqual([a["strategy"] for a in plan["attempts"]], ["direct", "unwind_j3j5_first"])
        np.testing.assert_array_equal(calls[0][1], self.sim.q_home)
        unwound = q_start.copy()
        unwound[j3] = self.sim.q_home[j3]
        np.testing.assert_array_equal(plan["waypoints"][1], unwound)
        np.testing.assert_array_equal(first["q_command"], q_start)
        stages, rows = [], []
        for index in range(1, 400):
            result = self.sim.tick(None, False, index / 100)
            self.assertNotEqual(result["state"], "FAULT", result["reason"])
            rows.append(result)
            if result["state"] == "HOME":
                break
            stages.append(result["home_path"]["stage"])
        self.assertEqual(result["state"], "HOME")
        np.testing.assert_allclose(result["q_command"], self.sim.q_home, atol=1e-9)
        self.assertEqual(sorted(set(stages)), [1, 2])
        self.assertEqual(stages, sorted(stages))
        for row in rows:
            q = row["q_command"]
            if q[j3] != self.sim.q_home[j3]:
                # First segment: only the unwinding joint moves.
                self.assertEqual(q[j1], q_start[j1])
            else:
                # Second segment: joint 3 never leaves its home value again.
                self.assertEqual(q[j3], self.sim.q_home[j3])
        q = np.array([q_start] + [row["q_command"] for row in rows])
        self.assertTrue(np.all(np.abs(np.diff(q, axis=0)) <= self.sim.speed_limits * .01 + 1e-12))

    def test_blocked_home_path_never_moves_and_a_later_request_plans_again(self):
        q_start = self.offset_home_start(self.sim)
        reject = {"accepted": False, "reason": "collision_clearance", "min_distance_m": .004,
                  "pair": {"geom_ids": (1, 2), "body_names": ("arm_l_link7", "arm_r_link7")},
                  "samples_checked": 4, "sweep_fraction": .5}
        with patch.object(self.guard, "check", return_value=reject):
            result = self.sim.tick(None, False, 0., action="home")
        self.assertEqual(result["state"], "FAULT")
        self.assertIn("initial pose path blocked", result["reason"])
        self.assertIn("arm_l_link7-arm_r_link7", result["reason"])
        self.assertEqual([a["strategy"] for a in result["home_plan"]["attempts"]],
                         ["direct", "unwind_j3j5_first", "unwind_forearm_wrist_first",
                          "right_arm_first", "left_arm_first"])
        self.assertFalse(self.sim.homing)
        for index in range(1, 5):
            held = self.sim.tick(None, False, index / 100)
            self.assertEqual(held["state"], "FAULT")
            np.testing.assert_array_equal(held["q_command"], q_start)
        self.assertEqual(self.sim.tick(None, False, .05, action="cancel_home")["state"], "DISABLED")
        retry = self.sim.tick(None, False, .06, action="home")
        self.assertEqual(retry["state"], "HOMING", retry["reason"])
        self.assertEqual(retry["home_plan"]["strategy"], "direct")

    def test_planning_gap_is_excused_for_one_tick_only(self):
        q_start = self.offset_home_start(self.sim)
        self.sim.tick(None, False, 0., action="home")
        result = self.sim.tick(None, False, .5)
        self.assertEqual(result["state"], "HOMING", result["reason"])
        self.assertEqual(result["command_dt"], .02)
        self.assertTrue(np.all(np.abs(result["q_command"] - q_start) <= self.sim.speed_limits * .02 + 1e-12))
        result = self.sim.tick(None, False, 1.)
        self.assertEqual(result["state"], "FAULT")
        self.assertIn("control loop delay", result["reason"])

    def test_dynamic_home_uses_actuators_and_reports_measured_error(self):
        sim = self.make_sim("dynamic")
        self.offset_home_start(sim)
        original_forward, original_step = self.env.forward, self.env.step
        step_count = [0]
        def checked_forward(*args, **kwargs):
            self.assertFalse(args, "Home must not teleport the physics state")
            self.assertIsNone(kwargs.get("q"), "Home must use the existing actuators")
            return original_forward(*args, **kwargs)
        def checked_step(*args, **kwargs):
            step_count[0] += kwargs.get("nstep", 1)
            return original_step(*args, **kwargs)
        self.env.forward, self.env.step = checked_forward, checked_step
        try:
            sim.tick(None, False, 0., action="home")
            home = None
            for index in range(1, 101):
                result = sim.tick(None, False, index / 100)
                self.assertNotEqual(result["state"], "FAULT", result["reason"])
                if result["state"] == "HOME":
                    home = result
            self.assertIsNotNone(home)
            np.testing.assert_allclose(home["q_command"], sim.q_home, atol=1e-9)
            np.testing.assert_array_equal(home["home_error"], home["q_measured"] - sim.q_home)
            self.assertGreater(np.max(np.abs(home["home_error"])), 1e-5)
            np.testing.assert_allclose(result["q_command"], sim.q_home, atol=1e-9)
            np.testing.assert_array_equal(result["home_error"], result["q_measured"] - sim.q_home)
            np.testing.assert_allclose(self.env.data.ctrl[self.env.get_idxs_step(joint_names=self.names)], sim.q_home)
            self.assertGreater(step_count[0], 0)
            self.assertAlmostEqual(self.env.data.time, 1., places=7)
        finally:
            self.env.forward, self.env.step = original_forward, original_step

    def test_existing_ik_tracks_and_inactive_commands_hold(self):
        before = get_sg2_wrist_control_T(self.env, "left").copy()
        rows = self.run_source(self.sim)
        after = get_sg2_wrist_control_T(self.env, "left")
        self.assertAlmostEqual(after[2, 3] - before[2, 3], .02, delta=.001)
        self.assertLess(np.linalg.norm(after[:2, 3] - before[:2, 3]), .001)
        active = [row for row in rows if "left" in row["active"]]
        self.assertLess(active[-1]["eef_errors"]["left"]["position_m"], .001)
        self.assertLess(active[-1]["eef_errors"]["left"]["rotation_rad"], np.deg2rad(1))
        q = np.array([row["q_command"] for row in rows])
        inactive = [self.names.index(name) for name in self.names if name.startswith(("arm_r_", "gripper_r_", "head_", "lift_"))]
        np.testing.assert_allclose(q[:, inactive], np.tile(self.q_init[inactive], (len(q), 1)), atol=1e-12)
        self.assertLessEqual(np.max(np.abs(np.diff(q, axis=0))), np.deg2rad(20) / 100 + 1e-12)
        # Release cancels further IK convergence, including the gripper goal.
        np.testing.assert_array_equal(q[505], q[-1])
        self.assertGreater(q[-1, self.names.index("gripper_l_joint1")], .1)
        self.assertEqual(self.env.data.time, 0.0)

    def test_tracking_loss_cancels_both_hands_and_requires_release(self):
        source = SyntheticControllerSource()
        for index in range(180):
            self.sim.tick(source.latest(index / 100), True, index / 100)
        held = self.sim.q_command.copy()
        bad = copy.deepcopy(source.latest(1.8))
        bad["hands"]["right"]["valid"] = False
        result = self.sim.tick(bad, True, 1.8)
        self.assertEqual(result["state"], "HOLD")
        for index in range(181, 200):
            self.sim.tick(source.latest(index / 100), True, index / 100)
        np.testing.assert_array_equal(self.sim.q_command, held)

    def test_loop_delay_latches_and_nonfinite_time_does_not_mutate(self):
        source = SyntheticControllerSource()
        self.sim.tick(source.latest(0), True, 0)
        self.sim.tick(source.latest(.01), True, .01)
        with self.assertRaises(ValueError):
            self.sim.tick(None, True, float("nan"))
        self.assertEqual(self.sim.last_time, .01)
        result = self.sim.tick(source.latest(.3), True, .3)
        self.assertEqual(result["state"], "FAULT")
        self.sim.tick(source.latest(.31), False, .31)
        result = self.sim.tick(source.latest(.34), True, .34)
        self.assertNotEqual(result["state"], "FAULT")

    def test_dynamic_uses_actuators_without_qpos_teleport(self):
        sim = self.make_sim("dynamic")
        original_forward = self.env.forward
        original_step = self.env.step
        step_count = [0]
        def checked_forward(*args, **kwargs):
            self.assertFalse(args, "No positional qpos writes in physics loop")
            self.assertIsNone(kwargs.get("q"), "No IK qpos writes in physics loop")
            return original_forward(*args, **kwargs)
        def checked_step(*args, **kwargs):
            step_count[0] += kwargs.get("nstep", 1)
            return original_step(*args, **kwargs)
        self.env.forward, self.env.step = checked_forward, checked_step
        try:
            rows = self.run_source(sim)
        finally:
            self.env.forward, self.env.step = original_forward, original_step
        self.assertGreater(step_count[0], 1000)
        self.assertAlmostEqual(self.env.data.time, 5.5, places=7)
        self.assertTrue(np.all(np.isfinite(self.env.data.qpos)))
        np.testing.assert_allclose(self.env.data.ctrl[self.env.get_idxs_step(joint_names=self.names)], sim.q_command)
        # The retained physical model has gravity sag; it must be measured, not hidden.
        final_error = np.max(np.abs(rows[-1]["q_measured"] - rows[-1]["q_command"]))
        self.assertGreater(final_error, 1e-5)
        self.assertLess(final_error, .1)

    def test_record_replay_reproduces_input_timing_and_commands(self):
        path = Path(self.cache.name) / "replay.jsonl"
        recorder = ControllerRecorder(path, self.names, "kinematic")
        source = SyntheticControllerSource()
        commands = []
        try:
            for index in range(220):
                t = index / 100
                frame = source.latest(t)
                # Include an OFF period while a controller is gripped.
                result = self.sim.tick(frame, not 1.5 <= t < 1.8, t)
                self.assertNotEqual(result["state"], "FAULT", result["reason"])
                recorder.write(t, frame, result)
                commands.append(result["q_command"])
        finally:
            recorder.close()
        sim = self.make_sim()
        replay = ReplayControllerSource(path, start_time=10)
        for index, expected in enumerate(commands):
            t = 10 + index / 100
            frame = replay.latest(t)
            self.assertEqual(replay.enabled, not 1.5 <= index / 100 < 1.8)
            result = sim.tick(frame, replay.enabled, t)
            np.testing.assert_allclose(result["q_command"], expected, atol=1e-10)

    def test_replay_home_and_cancel_actions_are_applied_once(self):
        self.offset_home_start(self.sim)
        path = Path(self.cache.name) / "home_replay.jsonl"
        recorder = ControllerRecorder(path, self.names, "kinematic")
        actions = {0: "home", 10: "cancel_home", 20: "home"}
        rows = []
        try:
            for index in range(61):
                t = index / 100
                result = self.sim.tick(None, False, t, action=actions.get(index))
                self.assertNotEqual(result["state"], "FAULT", result["reason"])
                recorder.write(t, None, result)
                rows.append(result)
        finally:
            recorder.close()
        sim = self.make_sim()
        self.offset_home_start(sim)
        replay = ReplayControllerSource(path, start_time=10.)
        for index, expected in enumerate(rows):
            t = 10. + index / 100
            frame = replay.latest(t)
            self.assertEqual(replay.action, actions.get(index))
            result = sim.tick(frame, replay.enabled, t, action=replay.action)
            self.assertEqual(result["state"], expected["state"])
            np.testing.assert_allclose(result["q_command"], expected["q_command"], atol=1e-10)
            replay.latest(t)
            self.assertIsNone(replay.action, "A repeated poll must not repeat a button action")

    def test_both_hands_small_rotation_and_translation_use_existing_ik(self):
        source = SyntheticControllerSource()
        for index in range(301):
            t = index / 100
            frame = copy.deepcopy(source.latest(t))
            frame["seq"], frame["sample_time_ms"], frame["received_at"] = index, t * 1000, t
            u = np.clip((t - .5) / 1.0, 0, 1)
            angle = np.deg2rad(3) * u
            for hand, x in (("left", .2), ("right", -.2)):
                frame["hands"][hand].update(position_m=[x, 1.1 + .01 * u, -.6],
                    quaternion_xyzw=[0, np.sin(angle / 2), 0, np.cos(angle / 2)],
                    grip=float(t >= .5), trigger=0.)
            result = self.sim.tick(frame, True, t)
            self.assertNotEqual(result["state"], "FAULT", result["reason"])
        self.assertEqual(set(result["active"]), {"left", "right"})
        for error in result["eef_errors"].values():
            self.assertLess(error["position_m"], .002)
            self.assertLess(error["rotation_rad"], np.deg2rad(1))


if __name__ == "__main__":
    unittest.main()
