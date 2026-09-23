"""clip_to_joint_limits(): pure-numpy safety clip on predicted policy actions.

No lerobot/torch/robot needed -- PolicyRuntime (which does need them) is not
imported here. See docs on policy_inference.py for why that import is lazy.
"""
from pathlib import Path
import runpy
import unittest

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
runpy.run_path(str(PROJECT_DIR / "package/init_project.py"))

from package.policy_inference import clip_to_joint_limits
from package.worker.fixed_quest_protocol import JOINT_LIMITS, JOINT_NAMES


def mid_q():
    # float32: clip_to_joint_limits round-trips float64 internally but always
    # returns float32 (matching the rest of the codebase's action/state dtype),
    # so an already-float32 input compares exactly equal when untouched.
    return np.array([(lo + hi) / 2. for lo, hi in (JOINT_LIMITS[name] for name in JOINT_NAMES)],
                    dtype=np.float32)


class ClipToJointLimitsTest(unittest.TestCase):
    def test_within_limits_is_unchanged_and_reports_nothing(self):
        q = mid_q()
        clipped, exceeded = clip_to_joint_limits(q)
        np.testing.assert_array_equal(clipped, q)
        self.assertEqual(exceeded, {})

    def test_above_upper_limit_is_clipped_and_reports_positive_excess(self):
        q = mid_q()
        name = "arm_l_joint2"
        index = JOINT_NAMES.index(name)
        _, high = JOINT_LIMITS[name]
        q[index] = high + 1.0
        clipped, exceeded = clip_to_joint_limits(q)
        self.assertEqual(clipped[index], high)
        self.assertEqual(set(exceeded), {name})
        self.assertAlmostEqual(exceeded[name], 1.0, places=4)

    def test_below_lower_limit_is_clipped_and_reports_negative_excess(self):
        q = mid_q()
        name = "lift_joint"
        index = JOINT_NAMES.index(name)
        low, _ = JOINT_LIMITS[name]
        q[index] = low - 0.3
        clipped, exceeded = clip_to_joint_limits(q)
        self.assertEqual(clipped[index], low)
        self.assertEqual(set(exceeded), {name})
        self.assertAlmostEqual(exceeded[name], -0.3, places=4)

    def test_multiple_joints_out_of_range_are_all_reported(self):
        q = mid_q()
        names = ("arm_l_joint2", "gripper_r_joint1")
        for name in names:
            _, high = JOINT_LIMITS[name]
            q[JOINT_NAMES.index(name)] = high + 0.5
        _, exceeded = clip_to_joint_limits(q)
        self.assertEqual(set(exceeded), set(names))

    def test_output_is_float32(self):
        clipped, _ = clip_to_joint_limits(mid_q())
        self.assertEqual(clipped.dtype, np.float32)

    def test_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            clip_to_joint_limits(np.zeros(3))

    def test_rejects_non_finite(self):
        q = mid_q()
        q[0] = float("nan")
        with self.assertRaises(ValueError):
            clip_to_joint_limits(q)
        q[0] = float("inf")
        with self.assertRaises(ValueError):
            clip_to_joint_limits(q)


import os

CHECKPOINT_PATH = os.environ.get("SG2_TEST_CHECKPOINT")


@unittest.skipUnless(CHECKPOINT_PATH, "set SG2_TEST_CHECKPOINT to a pretrained_model dir to run")
class PolicyRuntimeJointSubsetTest(unittest.TestCase):
    """Against a real checkpoint trained on fewer than 19 joints (arm+gripper only,
    lift/head excluded -- see review_dataset.ipynb's arm-only rebuild): joints
    outside that subset must be held at their measured value, not predicted."""

    def setUp(self):
        from package.policy_inference import PolicyRuntime
        self.excluded = ["lift_joint", "head_joint1", "head_joint2"]
        self.included = [n for n in JOINT_NAMES if n not in self.excluded]
        self.runtime = PolicyRuntime(CHECKPOINT_PATH, joint_names=self.included)

    def test_rejects_a_joint_names_length_mismatch(self):
        from package.policy_inference import PolicyRuntime
        with self.assertRaises(ValueError):
            PolicyRuntime(CHECKPOINT_PATH, joint_names=JOINT_NAMES)  # full 19, checkpoint wants 16

    def test_excluded_joints_are_held_at_their_measured_value_not_predicted(self):
        rng = np.random.default_rng(0)
        q = mid_q().astype(np.float64)
        for name in self.excluded:
            q[JOINT_NAMES.index(name)] = rng.uniform(-0.2, 0.2)
        frames = {"wrist_left": np.zeros((240, 424, 3), dtype=np.uint8),
                  "wrist_right": np.zeros((240, 424, 3), dtype=np.uint8)}
        self.runtime.reset()
        full = self.runtime.predict(q, frames)
        self.assertEqual(full.shape, (len(JOINT_NAMES),))
        for name in self.excluded:
            self.assertAlmostEqual(full[JOINT_NAMES.index(name)], q[JOINT_NAMES.index(name)], places=6)
        self.assertTrue(np.all(np.isfinite(full)))


if __name__ == "__main__":
    unittest.main()
