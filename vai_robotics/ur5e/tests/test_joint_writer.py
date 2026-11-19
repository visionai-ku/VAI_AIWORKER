"""Check the native torque session's wire protocol without a robot."""

import socket
import struct
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from vai_robotics.ur5e.joint_writer import JointImpedanceWriter, _check_firmware
from vai_robotics.ur5e.real import RealUR5e, RobotControlStopped


class JointWriterTest(unittest.TestCase):
    def setUp(self):
        self.receiver = MagicMock()
        self.receiver.getRuntimeState.return_value = 2
        self.receiver.getRobotMode.return_value = 7
        self.receiver.getSafetyMode.return_value = 1
        self.receiver.getTimestamp.return_value = 1.0
        self.receiver.getActualTCPPose.return_value = [0.3, 0.1, 0.5, 0.0, 0.0, 0.0]
        self.receiver.getActualQ.return_value = [-1.0] * 6
        self.stream = MagicMock()
        self.stream.fileno.return_value = 10
        self.stream.close.side_effect = lambda: setattr(self.stream.fileno, "return_value", -1)
        self.writer = JointImpedanceWriter(self.stream, self.receiver)

    def test_wire_atomic_setpoint_then_idle_and_exit(self):
        self.stream.recv.side_effect = [b"\x00\x00", b"\x00\x01", struct.pack("!i", 2), struct.pack("!i", 3)]
        self.writer.command([0.017, -1.3, 0.2, 0, 0, 0.000001])
        self.writer.hold()
        self.writer.stopScript()
        packets = [struct.unpack("!8i", call.args[0]) for call in self.stream.sendall.call_args_list]
        self.assertEqual(packets, [(1, 1, 17000, -1300000, 200000, 0, 0, 1),
                                   (2, 0, 0, 0, 0, 0, 0, 0), (3, 2, 0, 0, 0, 0, 0, 0)])

    def test_timeout_or_wrong_ack_consumes_session_without_retransmission(self):
        for response in [b"", socket.timeout(), struct.pack("!i", 2)]:
            with self.subTest(response=response):
                stream = MagicMock()
                stream.recv.side_effect = [response]
                with self.assertRaises((OSError, RuntimeError)):
                    JointImpedanceWriter(stream, self.receiver).command([0.00] * 6)
                stream.close.assert_called_once()
                stream.sendall.assert_called_once()

    def test_firmware_and_remote_gate_precede_upload(self):
        dashboard = MagicMock()
        api = SimpleNamespace(DashboardClient=MagicMock(return_value=dashboard))
        with patch.dict(sys.modules, dashboard_client=api):
            for version in ["URSoftware 5.22.4", "unavailable", "10.10.0"]:
                dashboard.polyscopeVersion.return_value = version
                with self.assertRaisesRegex(RuntimeError, "5.23"):
                    _check_firmware("test")
            dashboard.polyscopeVersion.return_value = "URSoftware 5.23.0.12345"
            dashboard.isInRemoteControl.return_value = False
            with self.assertRaisesRegex(RuntimeError, "remote"):
                _check_firmware("test")
            dashboard.isInRemoteControl.return_value = True
            _check_firmware("test")

    def test_upload_requires_stopped_robot_then_owns_stream_and_starts_in_idle(self):
        upload = MagicMock()
        upload.__enter__.return_value = upload
        upload.getsockname.return_value = ("192.0.2.1", 42101)
        upload.getpeername.return_value = ("192.0.2.2", 30002)
        listener = MagicMock()
        listener.__enter__.return_value = listener
        listener.getsockname.return_value = ("192.0.2.1", 42103)
        listener.accept.return_value = self.stream, ("192.0.2.2", 11111)
        self.stream.recv.side_effect = [struct.pack("!i", 123), struct.pack("!i", 1)]
        self.enterContext(patch("vai_robotics.ur5e.joint_writer._check_firmware"))
        self.enterContext(patch("vai_robotics.ur5e.joint_writer.secrets.randbelow", return_value=122))
        self.enterContext(patch("vai_robotics.ur5e.joint_writer.socket.create_connection", return_value=upload))
        self.enterContext(patch("vai_robotics.ur5e.joint_writer.socket.socket", return_value=listener))
        with self.assertRaisesRegex(RuntimeError, "idle"):
            JointImpedanceWriter.open("test", self.receiver, 0.15)
        upload.sendall.assert_not_called()
        self.receiver.getRuntimeState.side_effect = [1, 2]
        JointImpedanceWriter.open("test", self.receiver, 0.15)
        script = upload.sendall.call_args.args[0].decode("ascii")
        self.assertIn('socket_open("192.0.2.1", 42103, "vai")', script)
        self.assertNotIn("$", script)
        self.assertIn("direct_torque(tau)", script)
        self.stream.sendall.assert_called_once_with(struct.pack("!8i", 1, 0, 0, 0, 0, 0, 0, 0))
        # An unrelated peer cannot attach even with an otherwise-valid upload.
        listener.accept.return_value = self.stream, ("192.0.2.3", 11111)
        self.receiver.getRuntimeState.side_effect = [1]
        with self.assertRaisesRegex(RuntimeError, "handshake"):
            JointImpedanceWriter.open("test", self.receiver, 0.15)
        self.stream.close.assert_called()
        self.assertEqual(self.stream.sendall.call_count, 1)

    def test_backend_dry_live_and_release_use_only_owned_joint_channel(self):
        opened = self.enterContext(patch.object(JointImpedanceWriter, "open", return_value=self.writer))
        native = MagicMock()
        self.enterContext(patch.dict(sys.modules, {
            "rtde_receive": SimpleNamespace(RTDEReceiveInterface=MagicMock(return_value=self.receiver)),
            "rtde_control": SimpleNamespace(RTDEControlInterface=native),
        }))
        self.receiver.getRuntimeState.side_effect = [1, 2, 2, 2, 2, 2, 2, 2, 2]
        self.stream.recv.side_effect = [struct.pack("!i", seq) for seq in range(1, 4)]
        with RealUR5e("test", joint_impedance=True) as robot:
            target = np.full(6, -0.99)
            robot.command_joint_impedance(target)
            opened.assert_not_called()
            robot.set_motion_enabled(True)
            with self.assertRaisesRegex(ValueError, "selected robot-side writer"):
                robot.command_joints(target)
            with self.assertRaisesRegex(RobotControlStopped, "lead"):
                robot.command_joint_impedance(target + 1)
            robot.command_joint_impedance(target)
            robot.hold()
            robot.set_motion_enabled(False)
            robot.command_joint_impedance(target)
        self.assertEqual([struct.unpack("!8i", call.args[0])[1] for call in self.stream.sendall.call_args_list],
                         [1, 0, 2])
        native.assert_not_called()
        self.stream.close.assert_called_once()
