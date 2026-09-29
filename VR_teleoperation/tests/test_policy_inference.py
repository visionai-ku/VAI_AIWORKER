"""Policy clipping and offline LeRobot inference compatibility.

Clipping needs only numpy. The compatibility test uses installed LeRobot with
temporary processor files and a stub network, without a checkpoint or robot.
"""
from importlib.util import find_spec
import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

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


@unittest.skipUnless(find_spec("lerobot"), "requires the LeRobot notebook environment")
class PolicyRuntimeCompatibilityTest(unittest.TestCase):
    def test_saved_cuda_processors_run_on_cpu_and_hold_excluded_joints(self):
        import torch
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.processor import DeviceProcessorStep, PolicyProcessorPipeline
        from package.policy_inference import PolicyRuntime

        joint_names = [name for name in JOINT_NAMES
                       if name not in ("lift_joint", "head_joint1", "head_joint2")]
        cfg = ACTConfig(
            device="cpu",
            input_features={
                "observation.state": PolicyFeature(FeatureType.STATE, (len(joint_names),)),
                "observation.images.wrist_left": PolicyFeature(FeatureType.VISUAL, (3, 2, 2)),
            },
            output_features={"action": PolicyFeature(FeatureType.ACTION, (len(joint_names),))},
        )
        target = torch.arange(len(joint_names), dtype=torch.float32).unsqueeze(0) / 100

        class StubPolicy:
            @classmethod
            def from_pretrained(cls, checkpoint_path, config):
                policy = cls()
                policy.config = config
                return policy

            def select_action(self, observation):
                self.observation = observation
                return target.clone()

        with TemporaryDirectory() as folder:
            checkpoint = Path(folder)
            cfg.save_pretrained(checkpoint)
            # Serialize real processor schemas, then emulate the CUDA setting
            # retained in a checkpoint written on the training machine.
            for name in ("policy_preprocessor", "policy_postprocessor"):
                pipeline = PolicyProcessorPipeline(
                    name=name, steps=[DeviceProcessorStep(device="cpu")])
                saved = pipeline.get_config()
                saved["steps"][0]["config"]["device"] = "cuda"
                (checkpoint / (name + ".json")).write_text(json.dumps(saved), encoding="utf-8")

            for device in (None, "cpu"):
                with self.subTest(device=device), patch(
                        "lerobot.policies.factory.get_policy_class", return_value=StubPolicy):
                    runtime = PolicyRuntime(checkpoint, device=device, joint_names=joint_names)
                    q = mid_q()
                    bgr = np.full((2, 2, 3), (10, 20, 30), dtype=np.uint8)
                    actual = runtime.predict(q, {"wrist_left": bgr})

                    indices = [JOINT_NAMES.index(name) for name in joint_names]
                    expected = q.copy()
                    expected[indices] = target.numpy()[0]
                    np.testing.assert_array_equal(actual, expected)
                    observation = runtime.policy.observation
                    self.assertEqual(observation["observation.state"].device.type, "cpu")
                    np.testing.assert_array_equal(
                        observation["observation.state"].numpy()[0], q[indices])
                    np.testing.assert_allclose(
                        observation["observation.images.wrist_left"].numpy()[0, :, 0, 0],
                        np.array([30, 20, 10], dtype=np.float32) / 255)
                    self.assertEqual(runtime.preprocessor.steps[0].device, "cpu")
                    self.assertEqual(runtime.postprocessor.steps[0].device, "cpu")


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
