"""PolicyDeployment construction/session lifecycle against real ZMQ sockets with
no robot or checkpoint on the other end (never connects, never reads a real
policy). PolicyRuntime is swapped for a stub -- the actual predict/clip/tick
wiring it calls is unit-tested separately in test_policy_inference.py and
test_fixed_quest_real.py.
"""
from pathlib import Path
import runpy
import time
import unittest

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
runpy.run_path(str(PROJECT_DIR / "package/init_project.py"))

import package.policy_deploy as policy_deploy
from package.worker.fixed_quest_protocol import JOINT_NAMES


class StubPolicyRuntime:
    """No-op policy: always predicts the observed state unchanged."""

    def __init__(self, *args, **kwargs):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def predict(self, q, frames, task=None):
        return np.asarray(q, dtype=np.float32)


class PolicyDeploymentTest(unittest.TestCase):
    def setUp(self):
        self._real_runtime = policy_deploy.PolicyRuntime
        policy_deploy.PolicyRuntime = StubPolicyRuntime
        self.addCleanup(setattr, policy_deploy, "PolicyRuntime", self._real_runtime)

    def make_deployment(self, **options):
        options = {"hz": 30, "dry_run": True, **options}
        deployment = policy_deploy.PolicyDeployment(
            "stub-checkpoint", robot_ip="192.168.6.2", cameras={"head": 5570}, **options)
        self.addCleanup(deployment.close)
        return deployment

    def test_construction_builds_a_disarmed_dry_run_session(self):
        deployment = self.make_deployment()
        self.assertFalse(deployment.running)
        self.assertTrue(deployment.dry_run)
        self.assertFalse(deployment.real.armed)
        self.assertIsInstance(deployment.policy, StubPolicyRuntime)

    def test_step_before_start_raises(self):
        deployment = self.make_deployment()
        with self.assertRaises(RuntimeError):
            deployment.step()

    def test_start_resets_policy_and_step_waits_without_robot_or_cameras(self):
        deployment = self.make_deployment()
        deployment.start()
        self.assertTrue(deployment.running)
        self.assertEqual(deployment.policy.reset_count, 1)
        # No robot process and no camera publisher are running on this host,
        # so a tick must hold rather than ever call predict()/send a command.
        result = deployment.step(now=time.monotonic())
        self.assertEqual(result["state"], "WAIT_ROBOT")
        self.assertEqual(result["action"], None)

    def test_stop_releases_session_and_cameras_and_can_restart(self):
        deployment = self.make_deployment()
        deployment.start()
        deployment.stop()
        self.assertFalse(deployment.running)
        self.assertIsNone(deployment._cam_ctx)
        deployment.start()
        self.assertTrue(deployment.running)

    def test_invalid_hz_is_rejected(self):
        with self.assertRaises(ValueError):
            self.make_deployment(hz=0)
        with self.assertRaises(ValueError):
            self.make_deployment(hz=-1)


if __name__ == "__main__":
    unittest.main()
