"""Initial pose return path selection, with a stub guard (no MuJoCo)."""

from pathlib import Path
import runpy
import unittest

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
runpy.run_path(str(PROJECT_DIR / "package/init_project.py"))

from package.home_path import (HOME_PATH_STRATEGIES, home_path_duration, home_path_waypoints,
                               home_plan_summary, plan_home_path)
from package.worker.fixed_quest_protocol import JOINT_NAMES


class StubGuard:
    """Reject any segment for which ``blocked(q_start, q_end)`` is true."""

    def __init__(self, blocked=lambda a, b: False, samples=3):
        self.blocked, self.samples, self.calls = blocked, samples, []

    def check(self, q_start, q_end):
        self.calls.append((np.array(q_start), np.array(q_end)))
        if self.blocked(q_start, q_end):
            return {"accepted": False, "reason": "collision_clearance", "min_distance_m": .004,
                    "pair": {"geom_ids": (1, 2), "body_names": ("arm_l_link5", "gripper_r_r1")},
                    "samples_checked": self.samples, "sweep_fraction": 0. if self.samples == 1 else .5}
        return {"accepted": True, "reason": "accepted", "min_distance_m": .02, "samples_checked": self.samples}


class HomePathTest(unittest.TestCase):
    names = list(JOINT_NAMES)
    speeds = np.array([.01 if name == "lift_joint" else np.deg2rad(10) for name in JOINT_NAMES])

    def pose(self, **degrees):
        q = np.zeros(len(self.names))
        q[self.names.index("arm_l_joint4")] = q[self.names.index("arm_r_joint4")] = -np.pi / 2
        for name, value in degrees.items():
            q[self.names.index(name)] = np.deg2rad(value)
        return q

    def test_waypoints_skip_empty_stages_and_end_exactly_at_home(self):
        home = self.pose()
        start = self.pose(arm_r_joint3=120, arm_r_joint5=-150, arm_r_joint1=-50, arm_r_joint4=-10)
        stages = dict(HOME_PATH_STRATEGIES)["unwind_j3j5_first"]
        waypoints = home_path_waypoints(self.names, start, home, stages)
        # Auxiliary joints are already home, so that stage adds no waypoint.
        self.assertEqual(len(waypoints), 3)
        np.testing.assert_array_equal(waypoints[0], start)
        expected = start.copy()
        for name in ("arm_r_joint3", "arm_r_joint5"):
            expected[self.names.index(name)] = 0.
        np.testing.assert_array_equal(waypoints[1], expected)
        np.testing.assert_array_equal(waypoints[-1], home)
        self.assertEqual(len(home_path_waypoints(self.names, home, home, stages)), 1)
        self.assertAlmostEqual(home_path_duration(waypoints, self.speeds), 15. + 8.)

    def test_first_fully_accepted_candidate_wins_in_fixed_order(self):
        home = self.pose()
        start = self.pose(arm_r_joint3=120, arm_r_joint1=-50)
        j1, j3 = self.names.index("arm_r_joint1"), self.names.index("arm_r_joint3")
        # Block any segment that moves joint 1 and joint 3 at the same time.
        guard = StubGuard(lambda a, b: a[j1] != b[j1] and a[j3] != b[j3])
        plan = plan_home_path(guard, self.names, start, home, self.speeds)
        self.assertTrue(plan["ok"])
        self.assertEqual(plan["strategy"], "unwind_j3j5_first")
        self.assertEqual([(a["strategy"], a["accepted"]) for a in plan["attempts"]],
                         [("direct", False), ("unwind_j3j5_first", True)])
        self.assertEqual(plan["attempts"][0]["segment"], 1)
        self.assertEqual(len(guard.calls), 3)
        for (a, b), (wa, wb) in zip(guard.calls[1:], zip(plan["waypoints"][:-1], plan["waypoints"][1:])):
            np.testing.assert_array_equal(a, wa)
            np.testing.assert_array_equal(b, wb)
        self.assertEqual(plan["reason"], "initial pose path verified: unwind_j3j5_first")
        summary = home_plan_summary(plan)
        self.assertEqual(set(summary), {"ok", "strategy", "duration_s", "compute_s", "reason", "attempts", "waypoints"})

    def test_blocked_plan_lists_every_candidate_and_keeps_start(self):
        home, start = self.pose(), self.pose(arm_l_joint3=-40, arm_r_joint3=40)
        plan = plan_home_path(StubGuard(lambda a, b: True), self.names, start, home, self.speeds)
        self.assertFalse(plan["ok"])
        self.assertIsNone(plan["strategy"])
        self.assertEqual(len(plan["waypoints"]), 1)
        np.testing.assert_array_equal(plan["waypoints"][0], start)
        self.assertTrue(plan["reason"].startswith("initial pose path blocked; move the arms apart"))
        for strategy, _ in HOME_PATH_STRATEGIES:
            self.assertIn(strategy + ": collision_clearance arm_l_link5-gripper_r_r1", plan["reason"])

    def test_pose_already_rejected_by_guard_is_reported_once(self):
        home, start = self.pose(), self.pose(arm_l_joint3=-40)
        plan = plan_home_path(StubGuard(lambda a, b: True, samples=1), self.names, start, home, self.speeds)
        self.assertFalse(plan["ok"])
        self.assertIn("current pose already fails the command guard", plan["reason"])
        self.assertEqual(plan["reason"].count("collision_clearance"), 1)

    def test_start_at_home_checks_the_pose_and_returns_single_waypoint(self):
        home = self.pose()
        guard = StubGuard()
        plan = plan_home_path(guard, self.names, home, home, self.speeds)
        self.assertTrue(plan["ok"])
        self.assertEqual(plan["strategy"], "direct")
        self.assertEqual(len(guard.calls), 1)
        self.assertEqual(plan["duration_s"], 0.)


if __name__ == "__main__":
    unittest.main()
