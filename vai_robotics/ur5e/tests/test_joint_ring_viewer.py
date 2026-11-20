import os
import socket
import unittest
from unittest.mock import Mock, patch

from scipy.spatial.transform import Rotation

from vai_robotics.ur5e import real_viewer
from vai_robotics.ur5e.sim_teleop import SimBackend
from vai_robotics.ur5e.teleop_types import JointDrag


class ConnectedJointRingTests(unittest.TestCase):
    def setUp(self):
        reader, writer = os.pipe()
        self.addCleanup(os.close, reader)
        self.input_fd = reader
        process = Mock(stdin=os.fdopen(writer, "wb", buffering=0))
        with patch.object(real_viewer.subprocess, "Popen", return_value=process):
            self.parent = self.enterContext(real_viewer.RealViewer())
        self.parent._results.close()
        self.parent._results, self.sender = socket.socketpair(type=socket.SOCK_DGRAM)
        self.addCleanup(self.sender.close)
        self.parent._results.setblocking(False)
        self.clock = self.enterContext(patch.object(real_viewer.time, "monotonic", return_value=10.0))

    def report(self, epoch, stamp, drag):
        real_viewer._publish_result(self.sender, 0, stamp, {"phase": "waiting"}, epoch, drag)

    def test_rendered_epoch_and_drag_make_round_trip_in_both_modes(self):
        child = Mock()
        for epoch, dry in ((1, True), (2, False)):
            self.parent.set_joint_ring_epoch(epoch)
            self.parent.update(dry_run=dry, q=[0.] * 6, measured_tcp=([0.] * 3, Rotation.identity()), target=None)
            batch = next(real_viewer._frame_batches(self.input_fd))
            self.assertEqual(batch[0].ring_epoch, epoch)
            grabbed = JointDrag(1, 3, .04)
            child.joint_drag.return_value = grabbed
            with patch.object(real_viewer, "_frame_batches", return_value=[batch]):
                real_viewer._display(child, self.input_fd, lambda: None,
                                     lambda *args: real_viewer._publish_result(self.sender, *args))
            child.set_joint_ring_epoch.assert_called_with(epoch)
            self.assertEqual(self.parent.joint_drag(), grabbed)
            self.assertEqual(self.parent.speed_comparison(), {"phase": "waiting"})

    def test_stale_reordered_and_previous_configuration_drags_are_rejected(self):
        released = JointDrag(7, None, 0.)
        active = JointDrag(7, 2, -.08)
        self.parent.set_joint_ring_epoch(8)
        self.report(8, 10., active)
        self.assertEqual(self.parent.joint_drag(), active)
        self.clock.return_value = 10.02
        self.report(8, 10.02, released)
        self.report(8, 10.01, active)
        self.assertEqual(self.parent.joint_drag(), released)
        self.clock.return_value = 10.2
        self.assertIsNone(self.parent.joint_drag())
        self.parent.set_joint_ring_epoch(9)
        self.report(8, 10.2, active)
        self.assertIsNone(self.parent.joint_drag())
        self.report(9, 10.2, None)  # Lost window focus is separate from a confirmed release.
        self.assertIsNone(self.parent.joint_drag())
        self.parent.set_joint_ring_epoch(0)
        self.report(9, 10.21, active)
        self.assertIsNone(self.parent.joint_drag())


class LocalJointRingTests(unittest.TestCase):
    def test_ring_uses_displayed_pose_and_stall_requires_a_new_grab(self):
        sim = SimBackend()
        sim.sim, sim._shadow, rings = Mock(), Mock(), Mock()
        sim._joint_rings = rings
        sim.set_joint_ring_epoch(8)
        rings.update.return_value = JointDrag(2, 3, .2)
        with patch("vai_robotics.ur5e.sim_teleop.time.monotonic", return_value=20.) as clock:
            sim._previewing = True
            sim._render_frame(None, mirroring_robot=False, width_available=True)
            rings.update.assert_called_once_with(sim._shadow, 8)
            rings.draw.assert_called_once_with()
            self.assertEqual(sim.joint_drag(), rings.update.return_value)
            clock.return_value += .2
            self.assertIsNone(sim.joint_drag())
            sim._previewing = False
            sim._render_frame(None, mirroring_robot=True, width_available=True)
            self.assertEqual([c.args for c in rings.update.call_args_list[-2:]],
                             [(sim.sim.data, 0), (sim.sim.data, 8)])
            sim.set_joint_ring_epoch(0)
            self.assertIsNone(sim.joint_drag())
            rings.draw.reset_mock()
            sim._render_frame(None, mirroring_robot=True, width_available=True)
            rings.update.assert_called_with(sim.sim.data, 0)
            rings.draw.assert_not_called()
