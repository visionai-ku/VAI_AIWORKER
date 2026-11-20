"""Input-matched DRY versus REAL feedback, without a robot or a desktop."""

from dataclasses import replace
import json
import os
import socket
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from vai_robotics.ur5e import real_viewer
from vai_robotics.ur5e.payload import ToolPayload
from vai_robotics.ur5e.sim_teleop import SimBackend
from vai_robotics.ur5e.speed_comparison import SpeedComparison


def tcp(x, angle):
    return np.array([x, 0., 0.]), Rotation.from_euler("z", angle, degrees=True)


PAYLOAD = ToolPayload(0.91, (.001, -.005, .083))


class SpeedComparisonTests(unittest.TestCase):
    def setUp(self):
        self.view = Mock()
        self.a = real_viewer._Snapshot(
            timestamp=1., grip=1, index=1, dry_run=False, comparing=True, released=False, delta=True,
            q=(0.,) * 6, measured_tcp=tcp(.3, 179), target=tcp(.3, 179), joint_target=None, width_mm=None,
            gains=None, payload=PAYLOAD, feedback_timestamp=8., actual_speed=(0.,) * 6,
        )
        self.b = replace(self.a, index=2, feedback_timestamp=8.02, target=tcp(.31, -178),
                         actual_speed=(.020, 0, 0, 0, 0, np.deg2rad(40)))
        self.end = replace(self.b, comparing=False, released=True, feedback_timestamp=8.06, target=None)
        self.view.start_comparison.return_value = self.a.measured_tcp
        self.view.advance_comparison.side_effect = [tcp(.3002, -179), tcp(.301, -175)]
        self.trial = SpeedComparison(self.view)

    def test_previous_command_and_release_feedback_are_paired_on_rtde_clock(self):
        self.assertEqual(self.trial.update(self.a), {"phase": "waiting"})
        live = self.trial.update(self.b)
        self.assertEqual(live["phase"], "live")
        np.testing.assert_allclose([live[k] for k in ("real_mm_s", "dry_mm_s", "real_deg_s", "dry_deg_s")],
                                   [10., 10., 20., 100.])
        last = self.trial.update(self.end)
        self.assertEqual(last["phase"], "last")
        np.testing.assert_allclose([last[k] for k in ("real_mm_s", "dry_mm_s", "real_deg_s", "dry_deg_s")],
                                   [50 / 3, 50 / 3, 100 / 3, 100.])
        args = self.view.advance_comparison.call_args_list
        self.assertIs(args[0].args[0], self.a.target)
        self.assertIs(args[1].args[0], self.b.target)
        self.assertEqual(self.trial.update(replace(self.end, released=False)), last)

    def test_feedback_or_command_loss_invalidates_entire_stroke_until_new_clutch(self):
        for invalid in (replace(self.b, index=3), replace(self.b, feedback_timestamp=8.),
                        replace(self.b, feedback_timestamp=8.2), replace(self.b, actual_speed=(np.nan,) * 6),
                        replace(self.b, payload=None), replace(self.b, payload=ToolPayload(.5, (0.,) * 3))):
            with self.subTest(invalid=invalid):
                trial = SpeedComparison(self.view)
                trial.update(self.a)
                self.assertEqual(trial.update(invalid)["phase"], "unavailable")
                self.assertEqual(trial.update(self.end)["phase"], "unavailable")
                self.assertEqual(trial.update(replace(self.a, grip=2))["phase"], "waiting")
        self.assertEqual(self.trial.update(self.b)["phase"], "unavailable")  # Missing first sample.
        self.assertEqual(SpeedComparison(self.view).update(self.end)["phase"], "unavailable")

    def test_late_idle_does_not_count_as_release_and_clear_dry_transition_retains_valid_pairs(self):
        self.trial.update(self.a)
        self.assertEqual(self.trial.update(replace(self.end, index=1, released=False))["phase"], "unavailable")
        another = SpeedComparison(self.view)
        another.update(self.a)
        another.update(self.b)
        last = another.update(replace(self.end, dry_run=True, feedback_timestamp=float("nan")))
        self.assertEqual(last["phase"], "last")
        self.assertEqual(another.view.advance_comparison.call_count, 1)


class ComparisonTransportTests(unittest.TestCase):
    def test_child_processes_each_sample_and_renders_only_newest_measured_pose(self):
        sim = self.enterContext(SimBackend(headless=True))
        q, (xyz, rot) = sim.joints(), sim.pose()
        tcp_offset = xyz + rot.apply([.01, .02, .14]), rot
        rfd, wfd = os.pipe()
        self.addCleanup(os.close, rfd)
        process = Mock(stdin=os.fdopen(wfd, "wb", buffering=0))
        with patch.object(real_viewer.subprocess, "Popen", return_value=process):
            parent = self.enterContext(real_viewer.RealViewer())

        for law in ("position", "impedance"):
            for joint in (False, True):
                with self.subTest(law=law, joint=joint):
                    gains = sim.get_impedance_gains("joint" if joint else "eef") if law == "impedance" else None
                    with patch.object(real_viewer.time, "monotonic", return_value=8.):
                        for i in range(4):
                            active = i < 3
                            parent.update(q=q, measured_tcp=tcp_offset, dry_run=False, target=None, payload=PAYLOAD,
                                          comparison_target=(tcp_offset[0] + [.002 * i, 0, 0], rot) if active and not joint else None,
                                          comparison_joints=q + [.01 * i, 0, 0, 0, 0, 0] if active and joint else None,
                                          comparison_gains=gains, comparison_delta=True, actual_tcp_speed=[.03, 0, 0, 0, 0, .1],
                                          feedback_timestamp=10 + .02 * i)
                        batch = next(real_viewer._frame_batches(rfd))
                    self.assertEqual([f.index for f in batch], [1, 2, 3, 3])
                    self.assertEqual(batch[0].payload, PAYLOAD)
                    report = Mock()
                    with patch.object(real_viewer, "_frame_batches", return_value=[batch]), patch.object(sim, "tick", wraps=sim.tick) as draw:
                        real_viewer._display(sim, rfd, lambda: None, report)
                    self.assertEqual(draw.call_count, 1)
                    self.assertFalse(draw.call_args.kwargs["dry_run"])
                    self.assertIsNone(draw.call_args.kwargs["target"])
                    np.testing.assert_array_equal(sim.joints(), q)
                    result = report.call_args.args[2]
                    self.assertEqual(result["phase"], "last")
                    self.assertAlmostEqual(result["real_mm_s"], 30.)
                    self.assertGreater(result["dry_mm_s"], 0.5)

    def test_nonblocking_results_are_scoped_to_current_grip_and_stale_live_is_hidden(self):
        parent = real_viewer.RealViewer()
        read, write = socket.socketpair(type=socket.SOCK_DGRAM)
        self.addCleanup(read.close)
        self.addCleanup(write.close)
        read.setblocking(False)
        parent._results = read
        now = time.monotonic()
        write.send(json.dumps([0, now, {"phase": "live", "real_mm_s": 20.}]).encode())
        self.assertEqual(parent.speed_comparison()["real_mm_s"], 20.)
        with patch.object(real_viewer.time, "monotonic", return_value=now + 1.):
            self.assertEqual(parent.speed_comparison(), {"phase": "unavailable"})
        parent._generation += 1
        self.assertEqual(parent.speed_comparison(), {"phase": "waiting"})


if __name__ == "__main__":
    unittest.main()
