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
        self.warmup_count = 0
        self.init_kwargs = kwargs

    def reset(self):
        self.reset_count += 1

    def warmup(self):
        self.warmup_count += 1

    def predict(self, q, frames, task=None):
        return np.asarray(q, dtype=np.float32)


class PolicyDeploymentTest(unittest.TestCase):
    def setUp(self):
        self._real_runtime = policy_deploy.PolicyRuntime
        policy_deploy.PolicyRuntime = StubPolicyRuntime
        self.addCleanup(setattr, policy_deploy, "PolicyRuntime", self._real_runtime)

    # Deliberately NOT the real robot's address. These tests assert the "nothing is
    # publishing" states (WAIT_ROBOT), so pointing them at 192.168.6.2 made them pass or
    # fail depending on whether the robot happened to be powered on and on the LAN --
    # with it live the very same tick reports DISABLED (robot_ready=True) instead.
    UNREACHABLE_ROBOT_IP = "192.168.99.250"

    def make_deployment(self, **options):
        options = {"hz": 30, "dry_run": True, **options}
        deployment = policy_deploy.PolicyDeployment(
            "stub-checkpoint", robot_ip=self.UNREACHABLE_ROBOT_IP, cameras={"head": 5570},
            **options)
        self.addCleanup(deployment.close)
        return deployment

    def test_construction_builds_a_disarmed_dry_run_session(self):
        deployment = self.make_deployment()
        self.assertFalse(deployment.running)
        self.assertTrue(deployment.dry_run)
        self.assertFalse(deployment.real.armed)
        self.assertIsInstance(deployment.policy, StubPolicyRuntime)

    def test_start_warms_the_policy_up_before_the_robot_can_be_engaged(self):
        """The first predict() of a process is ~200x slower than the rest, which
        alone exceeds MAX_LOOP_DT_S and faults the session the moment the policy is
        engaged. start() must pay that cost while the robot is still disabled."""
        deployment = self.make_deployment()
        self.assertEqual(deployment.policy.warmup_count, 0)
        deployment.start()
        self.assertEqual(deployment.policy.warmup_count, 1)

    def test_state_joint_names_reach_the_policy_runtime(self):
        names = ["arm_l_joint1", "arm_r_joint1"]
        deployment = self.make_deployment(state_joint_names=names)
        self.assertEqual(deployment.policy.init_kwargs.get("joint_names"), names)

    def test_a_pending_initial_pose_return_keeps_ticking_enabled(self):
        """The return arms the bridge for several ticks before planner.homing flips.
        A disabled tick in that window calls stop(), which disarms and drops the
        request -- the reason pressing 'a' appeared to do nothing."""
        deployment = self.make_deployment()
        deployment.start()
        ticks = []
        deployment.real.tick = lambda frame, feedback, enabled=False, now=None, **kw: (
            ticks.append((enabled, kw.get("action"))) or {"state": "ARMING", "reason": "", "action": None})

        deployment.real.home_pending, deployment.real.planner.homing = True, False
        deployment.step(now=time.monotonic(), enabled=False)
        self.assertEqual(ticks[-1][0], True,
                         "must keep ticking enabled while the return is still arming")

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
