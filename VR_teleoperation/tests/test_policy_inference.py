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

from package.policy_inference import AsyncChunkPolicy, clip_to_joint_limits, orient_frame
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


class OrientFrameTest(unittest.TestCase):
    """Live wrist frames are landscape (240x424 on the robot, fingers on the left);
    checkpoints trained on data from 11/10 on expect them turned 90 deg CCW."""

    @staticmethod
    def landscape_wrist(h=4, w=6):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, 0] = 255          # the fingers: the image's left column
        return frame

    def test_wrist_frame_is_turned_ccw_for_a_portrait_checkpoint(self):
        frame = self.landscape_wrist()
        out, rotated = orient_frame("wrist_left", frame, (6, 4))
        self.assertTrue(rotated)
        self.assertEqual(out.shape, (6, 4, 3))
        np.testing.assert_array_equal(out, np.rot90(frame))
        # CCW: the left column (fingers) ends up as the bottom row.
        self.assertTrue(np.all(out[-1] == 255))
        self.assertTrue(np.all(out[:-1] == 0))
        self.assertTrue(out.flags["C_CONTIGUOUS"])

    def test_robot_sized_wrist_frame(self):
        out, rotated = orient_frame("wrist_right", np.zeros((240, 424, 3), np.uint8), (424, 240))
        self.assertTrue(rotated)
        self.assertEqual(out.shape, (424, 240, 3))

    def test_landscape_checkpoint_keeps_the_frame(self):
        # real v0.1 checkpoints (e.g. pi05_full_v01_30k_r2) were trained on the landscape frames
        frame = self.landscape_wrist()
        out, rotated = orient_frame("wrist_left", frame, (4, 6))
        self.assertFalse(rotated)
        self.assertIs(out, frame)

    def test_head_is_never_turned(self):
        frame = np.zeros((4, 6, 3), np.uint8)
        out, rotated = orient_frame("head", frame, (6, 4))
        self.assertFalse(rotated)
        self.assertIs(out, frame)

    def test_unknown_or_unrelated_shape_passes_unchanged(self):
        frame = self.landscape_wrist()
        for expected in (None, (8, 8), (12, 18)):
            with self.subTest(expected=expected):
                out, rotated = orient_frame("wrist_left", frame, expected)
                self.assertFalse(rotated)
                self.assertIs(out, frame)


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

    def test_landscape_wrist_frame_reaches_a_portrait_policy_turned_ccw(self):
        import torch
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.processor import DeviceProcessorStep, PolicyProcessorPipeline
        from package.policy_inference import PolicyRuntime

        cfg = ACTConfig(
            device="cpu",
            input_features={
                "observation.state": PolicyFeature(FeatureType.STATE, (len(JOINT_NAMES),)),
                "observation.images.wrist_left": PolicyFeature(FeatureType.VISUAL, (3, 3, 2)),  # portrait
                "observation.images.head": PolicyFeature(FeatureType.VISUAL, (3, 2, 3)),
            },
            output_features={"action": PolicyFeature(FeatureType.ACTION, (len(JOINT_NAMES),))},
        )

        class StubPolicy:
            @classmethod
            def from_pretrained(cls, checkpoint_path, config):
                policy = cls()
                policy.config = config
                return policy

            def select_action(self, observation):
                self.observation = observation
                return torch.zeros(1, len(JOINT_NAMES))

        with TemporaryDirectory() as folder:
            checkpoint = Path(folder)
            cfg.save_pretrained(checkpoint)
            for name in ("policy_preprocessor", "policy_postprocessor"):
                pipeline = PolicyProcessorPipeline(name=name, steps=[DeviceProcessorStep(device="cpu")])
                (checkpoint / (name + ".json")).write_text(json.dumps(pipeline.get_config()), encoding="utf-8")
            with patch("lerobot.policies.factory.get_policy_class", return_value=StubPolicy):
                runtime = PolicyRuntime(checkpoint)
            wrist = np.zeros((2, 3, 3), np.uint8)     # landscape, as the camera sends it
            wrist[:, 0] = 255                         # fingers on the left
            head = np.zeros((2, 3, 3), np.uint8)
            runtime.predict(mid_q(), {"wrist_left": wrist, "head": head})
            seen = runtime.policy.observation["observation.images.wrist_left"].numpy()[0]   # CHW
            self.assertEqual(seen.shape, (3, 3, 2))
            np.testing.assert_allclose(seen[:, -1], 1.0)   # fingers now along the bottom row
            np.testing.assert_allclose(seen[:, :-1], 0.0)
            self.assertEqual(tuple(runtime.policy.observation["observation.images.head"].shape[2:]), (2, 3))



class GatedChunkRuntime:
    """Stands in for PolicyRuntime: chunk k (1-based) is k*100 + step index on
    every predicted joint, and is only returned once the test release()s it."""

    def __init__(self, chunk_len=10, excluded=("lift_joint", "head_joint1", "head_joint2")):
        import threading
        self.joint_names = [name for name in JOINT_NAMES if name not in excluded]
        self.joint_indices = [JOINT_NAMES.index(name) for name in self.joint_names]
        self.chunk_len, self.calls, self.reset_count, self.error = chunk_len, 0, 0, None
        self._gates, self._lock = {}, threading.Lock()

    def _gate(self, call):
        import threading
        with self._lock:
            return self._gates.setdefault(call, threading.Event())

    def release(self, call):
        self._gate(call).set()

    def predict_chunk(self, q, frames, task=None):
        self.calls += 1
        call = self.calls
        if not self._gate(call).wait(5):
            raise TimeoutError("chunk %d was never released" % call)
        if self.error is not None:
            raise self.error
        steps = call * 100 + np.arange(self.chunk_len, dtype=np.float32)
        return np.repeat(steps[:, None], len(self.joint_names), axis=1)

    def reset(self):
        self.reset_count += 1


class AsyncChunkPolicyTest(unittest.TestCase):
    def make(self, lead=3, chunk_len=10):
        runtime = GatedChunkRuntime(chunk_len)
        policy = AsyncChunkPolicy(runtime, hz=30, lead=lead)
        self.addCleanup(policy.close)
        # Count chunk requests as they are submitted: runtime.calls only moves once
        # the worker thread actually starts one.
        self.requests, submit = 0, policy._executor.submit
        def counting_submit(fn, *args, **kwargs):
            if fn == policy._compute:
                self.requests += 1
            return submit(fn, *args, **kwargs)
        policy._executor.submit = counting_submit
        return runtime, policy

    @staticmethod
    def wait_for_chunk(policy):
        policy._future.result(timeout=5)

    def step(self, policy, q=None):
        action = policy.predict(mid_q() if q is None else q, {"head": np.zeros((2, 2, 3), np.uint8)})
        return None if action is None else float(action[JOINT_NAMES.index("arm_l_joint1")])

    def test_holds_until_the_first_chunk_then_plays_it_from_its_first_step(self):
        runtime, policy = self.make()
        self.assertIsNone(self.step(policy))
        self.assertIsNone(self.step(policy), "must not block or resubmit while computing")
        self.assertEqual(self.requests, 1)
        runtime.release(1)
        self.wait_for_chunk(policy)
        self.assertEqual([self.step(policy) for _ in range(3)], [100, 101, 102])

    def test_next_chunk_is_requested_early_and_skips_the_steps_already_executed(self):
        runtime, policy = self.make(lead=3)
        self.step(policy)
        runtime.release(1)
        self.wait_for_chunk(policy)
        played = [self.step(policy) for _ in range(9)]       # 100..108; chunk 2 requested at 107
        self.assertEqual(played, [100 + i for i in range(9)])
        self.assertEqual(self.requests, 2)
        runtime.release(2)
        self.wait_for_chunk(policy)
        # Chunk 2 saw the state before step 107; 107 and 108 ran since, so it resumes at its index 2.
        self.assertEqual(self.step(policy), 202)

    def test_repeats_the_last_step_while_starved(self):
        runtime, policy = self.make(lead=1, chunk_len=3)
        self.step(policy)
        runtime.release(1)
        self.wait_for_chunk(policy)
        played = [self.step(policy) for _ in range(5)]
        self.assertEqual(played, [100, 101, 102, 102, 102])
        self.assertEqual(policy.starved_steps, 2)

    def test_reset_discards_the_chunk_in_flight_without_blocking(self):
        runtime, policy = self.make()
        self.step(policy)                     # chunk 1 in flight
        policy.reset()
        self.assertIsNone(self.step(policy))  # requests chunk 2, still nothing to play
        runtime.release(1)
        runtime.release(2)
        self.wait_for_chunk(policy)
        self.assertEqual(self.step(policy), 200, "chunk 1 predates the reset and must be dropped")
        self.assertEqual(runtime.reset_count, 1)

    def test_excluded_joints_follow_the_current_measurement(self):
        runtime, policy = self.make()
        self.step(policy)
        runtime.release(1)
        self.wait_for_chunk(policy)
        q = mid_q()
        q[JOINT_NAMES.index("lift_joint")], q[JOINT_NAMES.index("head_joint1")] = .1, .69
        action = policy.predict(q, {"head": np.zeros((2, 2, 3), np.uint8)})
        self.assertAlmostEqual(action[JOINT_NAMES.index("lift_joint")], .1, places=6)
        self.assertAlmostEqual(action[JOINT_NAMES.index("head_joint1")], .69, places=6)
        self.assertEqual(action[JOINT_NAMES.index("arm_r_joint7")], 100)

    def test_a_failed_prediction_is_raised_to_the_control_loop(self):
        runtime, policy = self.make()
        runtime.error = RuntimeError("cuda out of memory")
        self.step(policy)
        runtime.release(1)
        with self.assertRaises(RuntimeError):
            self.wait_for_chunk(policy)
        with self.assertRaises(RuntimeError):
            self.step(policy)

    def test_lead_covers_the_measured_latency_with_margin(self):
        policy = AsyncChunkPolicy(GatedChunkRuntime(), hz=30)
        self.addCleanup(policy.close)
        policy._update_lead(.174, chunk_len=30)   # MolmoAct2 on the RTX 6000 Ada
        self.assertEqual(policy.lead, 10)
        policy._update_lead(2.0, chunk_len=30)
        self.assertEqual(policy.lead, 29, "never more than the chunk minus one step")


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
