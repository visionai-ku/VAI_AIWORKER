import time
import unittest
from dataclasses import replace
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from vai_robotics.ur5e.demonstrations import DemonstrationRecording
from vai_robotics.ur5e.impedance import ImpedanceSettings
from vai_robotics.ur5e.sim_teleop import SimBackend
from vai_robotics.ur5e.teleop_session import TeleopSession
from vai_robotics.ur5e.teleop_types import InputSample


class RecordingSessionTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.panel = Mock(is_open=True, scale=1.0, open_width_mm=75.0)
        self.panel.take_impedance_gains.return_value = None
        self.panel.take_target.return_value = None
        self.device = Mock(spec=["read"])
        self.released = InputSample.stationary()
        self.held = replace(self.released, enabled=True, buttons_released=False)
        self.elapsed = 0.0

    def open_session(self, backend, *, simulation, dry_run=False, gripper=None):
        recording = DemonstrationRecording(self.root, simulation=simulation, task="Lift block")
        self.addCleanup(recording.close)
        self.session = TeleopSession(
            self.device, backend, simulation=simulation, dry_run=dry_run, gripper=gripper,
            panel=self.panel, gripper_force_n=40, command_mode="joint-delta",
            control_law="position", monotonic=lambda: self.elapsed, recording=recording,
        )

    def tick(self, sample=None, *, record=None, mode=None, law=None, dry_run=None):
        self.panel.poll.return_value = dry_run
        self.panel.take_record_request.return_value = record
        self.panel.take_command_mode.return_value = mode
        self.panel.take_control_law.return_value = law
        self.device.read.return_value = self.released if sample is None else sample
        self.elapsed += 0.02
        self.assertTrue(self.session.step(None, self.elapsed))
        return self.panel.refresh.call_args.kwargs["recording"]

    def wait_for(self, state):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            status = self.tick()
            if status["state"] == state:
                return status
            self.assertNotEqual(status["state"], "error", status["message"])
            time.sleep(0.02)
        self.fail(f"Recorder did not reach {state}: {status}")

    def load_dataset(self, root):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.enterContext(patch("datasets.config.HF_DATASETS_CACHE", self.root / "cache"))
        return LeRobotDataset("local/ur5e", root=Path(root), video_backend="pyav")

    def test_hardware_dry_run_rejects_record_and_keeps_writes_disabled(self):
        backend = Mock()
        backend.pose.return_value = np.array([0.4, 0.0, 0.3]), Rotation.identity()
        backend.joints.return_value = np.arange(6) / 10
        gripper = Mock(open_width_mm=75.0, width_mm=60.0, status="Connected")
        self.open_session(backend, simulation=False, dry_run=True, gripper=gripper)
        status = self.tick(record="start")
        self.assertFalse(status["can_start"])
        self.assertIn("REAL RUN", status["message"])
        self.panel.recording_result.assert_called_once()
        self.tick(self.held)
        self.tick(replace(self.held, translation=np.array([0.004, 0, 0]), gripper_chord=True))
        backend.command_joints.assert_not_called()
        backend.set_motion_enabled.assert_not_called()
        self.assertTrue(all(call.args == (None,) for call in gripper.update.call_args_list))
        self.assertFalse(list(self.root.rglob("*.parquet")))

    @unittest.skipUnless(find_spec("lerobot"), "Install the recording dependencies for the LeRobot round trip")
    def test_saved_simulation_replays_measurement_and_dispatched_joint_target(self):
        backend = self.enterContext(SimBackend(headless=True))
        self.open_session(backend, simulation=True)
        self.tick(record="start")
        self.wait_for("recording")
        self.tick(self.held)
        before_joints = backend.joints()
        before_xyz, before_rotation = backend.pose()
        move = replace(self.held, translation=np.array([0.002, 0, 0]),
                       gripper_command=True, gripper_chord=True)
        status = self.tick(move)
        frame_index = status["frames"] - 1
        target = self.session.controller.joint_target.copy()
        self.assertGreater(target[0] - before_joints[0], 1e-4)
        self.assertGreater(np.linalg.norm(before_joints - backend.joints()), 1e-5)
        self.tick(record="save")
        saved = self.wait_for("idle")
        self.assertEqual(saved["episodes"], 1)
        self.assertEqual(Path(saved["root"]), self.root / "sim" / "joint-delta-position")

        dataset = self.load_dataset(saved["root"])
        frame = dataset[frame_index]
        self.assertEqual(frame["task"], "Lift block")
        self.assertEqual(int(frame["episode_index"]), 0)
        self.assertTrue(bool(frame["control.active"]))
        self.assertAlmostEqual(float(frame["control.elapsed_s"]), frame_index * .02, places=5)
        np.testing.assert_allclose(frame["observation.state"][:6], before_joints, atol=1e-6)
        self.assertAlmostEqual(float(frame["observation.state"][-1]), self.panel.open_width_mm)
        np.testing.assert_allclose(frame["observation.eef_pose"][:3], before_xyz, atol=1e-6)
        quaternion = np.asarray(frame["observation.eef_pose"][3:])
        self.assertAlmostEqual(abs(np.dot(quaternion, before_rotation.as_quat(scalar_first=True))), 1.0, places=6)
        np.testing.assert_allclose(frame["action"][:6], target, atol=1e-6)
        self.assertEqual(float(frame["action"][-1]), 0.0)

    @unittest.skipUnless(find_spec("lerobot"), "Install the recording dependencies for the LeRobot round trip")
    def test_real_tcp_impedance_records_feedback_and_equilibrium_until_return_to_preview(self):
        backend = Mock()
        xyz = np.array([0.38, -0.15, 0.32])
        orientation = Rotation.from_rotvec([0.1, -0.35, 0.42])
        joints = np.linspace(-1, 1, 6)
        backend.pose.return_value = xyz, orientation
        backend.joints.return_value = joints
        backend.get_impedance_gains.side_effect = ImpedanceSettings().snapshot
        self.open_session(backend, simulation=False)
        self.tick(mode="eef-absolute")
        self.tick(law="impedance")
        self.tick(record="start")
        self.wait_for("recording")
        self.tick(self.held)
        advanced_joints = joints + 0.002
        backend.command_cartesian_impedance.side_effect = lambda *args: setattr(
            backend.joints, "return_value", advanced_joints
        )
        self.panel.take_target.return_value = np.r_[xyz + [0, 0.01, 0], orientation.as_rotvec()].tolist()
        status = self.tick(self.held)
        self.panel.take_target.return_value = None
        goal_xyz, goal_orientation = backend.command_cartesian_impedance.call_args.args
        self.assertGreater(goal_xyz[1], xyz[1])
        self.tick(dry_run=True)
        self.wait_for("idle")
        saved_root = self.root / "real" / "eef-absolute-impedance"
        self.assertEqual(Path(status["root"]), saved_root)
        dataset = self.load_dataset(saved_root)
        self.assertEqual(len(dataset), status["frames"])
        frame = dataset[status["frames"] - 1]
        np.testing.assert_allclose(frame["observation.state"], joints, atol=1e-6)
        np.testing.assert_allclose(frame["observation.eef_pose"][:3], xyz, atol=1e-6)
        np.testing.assert_allclose(frame["action"][:3], goal_xyz, atol=1e-6)
        self.assertEqual(len(frame["action"]), 7)
        target_quat = np.asarray(frame["action"][3:])
        self.assertAlmostEqual(abs(np.dot(target_quat, goal_orientation.as_quat(scalar_first=True))), 1.0, places=6)
        gains = backend.get_impedance_gains("eef")
        for key in ("stiffness", "damping"):
            np.testing.assert_allclose(frame[f"control.{key}"], gains[key])

    @unittest.skipUnless(find_spec("lerobot"), "Install the recording dependencies for the LeRobot round trip")
    def test_camera_failure_discards_only_the_current_take_and_allows_retry(self):
        backend = self.enterContext(SimBackend(headless=True))
        self.open_session(backend, simulation=True)
        cameras = self.session.recording.cameras = Mock()
        cameras.snapshot.return_value = {"wrist": np.zeros((32, 32, 3), dtype=np.uint8)}
        self.tick(record="start")
        self.wait_for("recording")
        self.tick(self.held)
        cameras.snapshot.side_effect = RuntimeError("Camera wrist: stale frame")
        failed = self.tick(self.held)
        self.assertEqual(failed["state"], "error")
        self.assertIn("stale frame", failed["message"])
        self.assertTrue(self.session.controller.active)
        root = Path(failed["root"])
        self.assertFalse(root.exists())
        deadline = time.monotonic() + 5
        while not self.tick()["can_start"]:
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.01)
        cameras.snapshot.side_effect = None
        self.tick(record="start")
        self.wait_for("recording")
        self.tick(self.held)
        self.tick(record="save")
        self.wait_for("idle")
        dataset = self.load_dataset(root)
        self.assertEqual(dataset.meta.total_episodes, 1)
        self.assertEqual(tuple(dataset[0]["observation.images.wrist"].shape), (3, 32, 32))

    @unittest.skipUnless(find_spec("lerobot"), "Install the recording dependencies for the LeRobot round trip")
    def test_accepted_mode_change_finishes_episode_without_crossing_boundary(self):
        self.open_session(self.enterContext(SimBackend(headless=True)), simulation=True)
        self.tick(record="start")
        self.wait_for("recording")
        self.tick(self.held)
        refused = self.tick(self.held, mode="eef-delta")
        self.assertEqual(refused["state"], "recording")
        self.assertEqual(self.session.controller.command_mode, "joint-delta")
        before = self.tick()
        self.assertGreater(before["frames"], 0)
        self.tick(mode="eef-delta")
        self.assertEqual(self.session.controller.command_mode, "eef-delta")
        self.session.recording.close()  # Wait for the previous mode's pending save.
        dataset = self.load_dataset(before["root"])
        self.assertEqual(len(dataset), before["frames"])
        self.assertEqual(dataset.meta.total_episodes, 1)
        self.assertEqual(len(dataset[0]["action"]), 7)
