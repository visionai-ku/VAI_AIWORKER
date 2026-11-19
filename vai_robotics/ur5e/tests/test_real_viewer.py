import os
import select
import subprocess
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from vai_robotics.ur5e import real_viewer


class RealViewerTests(unittest.TestCase):
    def setUp(self):
        read_fd, write_fd = os.pipe()
        self.read_fd = read_fd
        self.addCleanup(os.close, read_fd)
        writer = os.fdopen(write_fd, "wb", buffering=0)
        self.process = Mock(stdin=writer)
        self.process.poll.return_value = None
        with patch.object(subprocess, "Popen", return_value=self.process):
            self.viewer = real_viewer.RealViewer().__enter__()
        self.addCleanup(self.viewer.close)
        self.q = np.arange(6, dtype=float) / 10
        self.tcp = np.array([0.2, 0.3, 0.4]), Rotation.from_euler("x", 30, degrees=True)
        self.target = np.array([0.7, 0, 0]), Rotation.from_euler("y", 45, degrees=True)

    def send(self, dry_run, target, width=None):
        self.viewer.update(q=self.q, measured_tcp=self.tcp, dry_run=dry_run, target=target,
                           gripper_width_mm=width)

    def test_draws_latest_measured_joints_and_tcp_in_real_and_reseeds_skipped_grips(self):
        # Run the production consumer against a real pipe. Simulate a renderer that
        # falls behind over multiple mode/clutch changes before servicing its window.
        observed = []
        view = Mock()
        view.is_running.side_effect = lambda: len(observed) < 3

        def draw(**frame):
            observed.append(frame)
            if len(observed) == 1:
                self.send(True, None)
                self.send(True, self.target)
            if len(observed) == 2:
                self.send(False, self.target, 50.6)

        view.tick.side_effect = draw
        now = time.monotonic()
        with patch.object(real_viewer.time, "monotonic", side_effect=lambda: next(times)):
            times = (now + i * 0.021 for i in range(100))
            self.send(True, self.target)
            real_viewer._display(view, self.read_fd, lambda: None)
        first, regrip, live = observed
        for frame in (first, regrip):
            self.assertTrue(frame["dry_run"])
            self.assertTrue(frame["reset_preview"])
            np.testing.assert_array_equal(frame["target"][0], self.target[0])
            np.testing.assert_allclose(frame["target"][1].as_matrix(), self.target[1].as_matrix())
        self.assertFalse(live["dry_run"])
        self.assertAlmostEqual(live["gripper_width_mm"], 50.6)
        self.assertIsNone(first["gripper_width_mm"])
        self.assertIsNone(live["target"])
        np.testing.assert_array_equal(live["measured_q"], self.q)
        np.testing.assert_array_equal(live["measured_tcp"][0], self.tcp[0])
        np.testing.assert_allclose(live["measured_tcp"][1].as_matrix(), self.tcp[1].as_matrix())

    def test_full_pipe_drops_frame_and_renderer_exits_on_parent_eof(self):
        fd = self.process.stdin.fileno()
        self.assertFalse(os.get_blocking(fd))
        with self.assertRaises(BlockingIOError):
            while True:
                os.write(fd, b"\x00" * real_viewer._FRAME.size)
        self.send(False, None)
        self.viewer.close()
        self.assertFalse(list(real_viewer._snapshots(self.read_fd)))
        self.assertFalse(self.viewer.is_running())

    def test_joint_preview_crosses_pipe_without_changing_measured_q(self):
        preview = self.q + 0.2
        self.viewer.update(q=self.q, measured_tcp=self.tcp, dry_run=True, target=None, joint_target=preview)
        view = Mock()
        view.is_running.side_effect = [True, False]
        real_viewer._display(view, self.read_fd, lambda: None)
        frame = view.tick.call_args.kwargs
        self.assertIsNone(frame["target"])
        np.testing.assert_array_equal(frame["joint_target"], preview)
        np.testing.assert_array_equal(frame["measured_q"], self.q)

    def test_updates_are_rate_limited(self):
        with patch.object(real_viewer.time, "monotonic", return_value=10.0):
            for _ in range(4):
                self.send(False, None)
        self.assertEqual(len(os.read(self.read_fd, real_viewer._FRAME.size * 4)), real_viewer._FRAME.size)
        self.assertFalse(select.select([self.read_fd], [], [], 0)[0])
