"""Connected payload telemetry and MuJoCo load dynamics without a robot."""

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import mujoco
import numpy as np

from vai_robotics.ur5e.payload import ToolPayload
from vai_robotics.ur5e.real import RealUR5e
from vai_robotics.ur5e.sim_teleop import SimBackend


class PayloadFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.receiver = MagicMock()
        self.receiver.getActualTCPPose.return_value = [0.] * 6
        self.receiver.getTimestamp.return_value = 123.0
        self.receiver.getPayload.return_value = 0.91
        self.receiver.getPayloadCog.return_value = [0.001, -0.005, 0.083]
        self.connect = MagicMock(return_value=self.receiver)
        self.enterContext(patch.dict(sys.modules, {
            "rtde_receive": SimpleNamespace(RTDEReceiveInterface=self.connect),
        }))

    def test_tracks_active_settings_including_unloaded_tool(self):
        robot = self.enterContext(RealUR5e("test"))
        fields = self.connect.call_args.args[2]
        self.assertIn("payload", fields)
        self.assertIn("payload_cog", fields)
        self.assertNotIn("payload_inertia", fields)  # Not available on PolyScope 5.9.
        self.assertEqual(robot.payload(), ToolPayload(0.91, (0.001, -0.005, 0.083)))
        self.receiver.getPayload.return_value = 0.
        self.receiver.getPayloadCog.return_value = [0.] * 3
        self.assertEqual(robot.payload(), ToolPayload(0., (0., 0., 0.)))
        for mass, cog in ((-1., [0.] * 3), (np.nan, [0.] * 3), (0.5, [0., 0.]),
                          (0.5, [0., np.inf, 0.])):
            self.receiver.getPayload.return_value = mass
            self.receiver.getPayloadCog.return_value = cog
            self.assertIsNone(robot.payload())
        self.assertFalse(robot.motion_enabled)

    def test_old_controller_retains_required_feedback(self):
        self.connect.side_effect = [RuntimeError("Variables not found: [payload_cog]"), self.receiver]
        robot = self.enterContext(RealUR5e("test"))
        self.assertNotIn("payload_cog", self.connect.call_args.args[2])
        self.assertIsNone(robot.payload())
        robot.pose()
        self.receiver.getPayload.assert_not_called()

    def test_payload_retries_both_fields_on_rollover_and_discards_two_crossings(self):
        robot = self.enterContext(RealUR5e("test"))
        self.receiver.getTimestamp.side_effect = [124., 124.002, 124.002, 124.002]
        self.receiver.getPayload.side_effect = [.91, 1.5]
        self.receiver.getPayloadCog.side_effect = [[.001, -.005, .083], [.03, .01, .12]]
        self.assertEqual(robot.payload(), ToolPayload(1.5, (.03, .01, .12)))
        self.receiver.getTimestamp.side_effect = [125., 125.002, 125.002, 125.004]
        self.receiver.getPayload.side_effect = None
        self.receiver.getPayloadCog.side_effect = None
        self.assertIsNone(robot.payload())

    def test_unrelated_connection_error_is_reported(self):
        self.connect.side_effect = RuntimeError("RTDE connection failed")
        with self.assertRaisesRegex(RuntimeError, "RTDE connection failed"), RealUR5e("test"):
            self.fail("Session opened after a connection error")
        self.connect.assert_called_once()

    def test_speed_sample_carries_device_time_and_skips_invalid_or_crossed_frames(self):
        clock = self.enterContext(patch("vai_robotics.ur5e.real.time.monotonic", return_value=10.))
        robot = self.enterContext(RealUR5e("test"))
        self.receiver.getActualTCPSpeed.return_value = [0.03, 0., 0., 0., 0., 0.]
        stamp, twist = robot.tcp_speed_feedback()
        self.assertEqual(stamp, 123.)
        self.assertAlmostEqual(twist[0], .03)
        clock.return_value += 1.
        self.assertIsNone(robot.tcp_speed_feedback())
        self.receiver.getTimestamp.side_effect = [124., 124.002, 124.002, 124.002]
        self.receiver.getActualTCPSpeed.side_effect = [[.03] * 6, [.05] * 6]
        stamp, twist = robot.tcp_speed_feedback()
        self.assertEqual(stamp, 124.002)
        np.testing.assert_allclose(twist, [.05] * 6)
        self.receiver.getActualTCPSpeed.side_effect = None
        self.receiver.getTimestamp.side_effect = [125., 125.002, 125.002, 125.004]
        self.assertIsNone(robot.tcp_speed_feedback())
        self.receiver.getTimestamp.side_effect = None
        self.receiver.getActualTCPSpeed.return_value = [np.nan] * 6
        self.assertIsNone(robot.tcp_speed_feedback())


class PayloadDynamicsTests(unittest.TestCase):
    def test_inertia_changes_by_flange_point_mass_and_preserves_state(self):
        view = self.enterContext(SimBackend(headless=True))
        model, data = view.sim.model, view.sim.data
        view.tick(dry_run=True)
        xyz, rotation = view.pose()
        view.tick(dry_run=True, target=(xyz + [.003, 0, 0], rotation), preview_impedance=True)
        live_q = data.qpos.copy()
        preview_q, preview_v = view._shadow.qpos.copy(), view._shadow.qvel.copy()
        unloaded = np.empty((model.nv, model.nv))
        mujoco.mj_fullM(model, unloaded, data.qM)
        wrist_mass = model.body("wrist_3_link").mass.copy()
        load = ToolPayload(.91, (.001, -.005, .083))
        view.set_payload(load)
        body = data.body("rg2_base")
        np.testing.assert_allclose(body.xipos, xyz + rotation.apply(load.cog_m))
        jac = np.empty((3, model.nv))
        mujoco.mj_jacBodyCom(model, data, jac, None, body.id)
        loaded = np.empty_like(unloaded)
        mujoco.mj_fullM(model, loaded, data.qM)
        np.testing.assert_allclose(loaded - unloaded, load.mass_kg * jac.T @ jac, atol=1e-12)
        np.testing.assert_array_equal(model.body("wrist_3_link").mass, wrist_mass)
        np.testing.assert_array_equal(data.qpos, live_q)
        np.testing.assert_array_equal(view._shadow.qpos, preview_q)
        np.testing.assert_array_equal(view._shadow.qvel, preview_v)

        with patch.object(mujoco, "mj_setConst") as update:
            view.set_payload(load)
            update.assert_not_called()
        view.set_payload(ToolPayload(0., (0., 0., 0.)))
        mujoco.mj_fullM(model, loaded, data.qM)
        np.testing.assert_allclose(loaded, unloaded, atol=1e-12)
