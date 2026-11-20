import json
import os
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PyQt5.QtCore import QRect, QSettings
from PyQt5.QtGui import QScreen

from vai_robotics.ur5e.rg2_visual import DEFAULT_VISUAL_MOUNT
from vai_robotics.ur5e.teleop_panel import TeleopPanel, _PanelWindow, _WindowConnection


class VisualMountPanelTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, QT_QPA_PLATFORM="offscreen"))
        self.enterContext(patch.object(QScreen, "availableGeometry", return_value=QRect(0, 0, 375, 740)))
        directory = self.enterContext(TemporaryDirectory())
        self.settings = QSettings(str(Path(directory) / "mount.ini"), QSettings.IniFormat)
        self.enterContext(patch("vai_robotics.ur5e.panel.visual_mount.QSettings", return_value=self.settings))

    def window(self, robot_ip=None):
        window = _PanelWindow(simulation=robot_ip is None, robot_ip=robot_ip,
                              device_name="SpaceMouse", dry_run=True, open_width_mm=75.)
        self.addCleanup(window.close)
        return window

    def test_apply_persists_full_pose_per_robot_and_simulation(self):
        window = self.window("192.0.2.10")
        mount = window.visual_mount
        self.assertFalse(mount.body.isVisible())
        mount.toggle.click()
        window.app.processEvents()
        self.assertTrue(mount.body.isVisible())
        self.assertEqual(window.scroll.horizontalScrollBar().maximum(), 0)
        self.assertFalse(mount.apply.isEnabled())
        pose = (10., -2.5, 28., 5., -8., -90.)
        for field, value in zip(mount.fields, pose):
            field.setValue(value)
        self.assertEqual(window.rg2_mount, DEFAULT_VISUAL_MOUNT)
        mount.apply.click()
        self.assertEqual(window.rg2_mount, pose)
        self.assertEqual(self.window("192.0.2.10").rg2_mount, pose)
        self.assertEqual(self.window("192.0.2.11").rg2_mount, DEFAULT_VISUAL_MOUNT)
        self.assertEqual(self.window().rg2_mount, DEFAULT_VISUAL_MOUNT)
        mount.reset.click()
        self.assertEqual(self.window("192.0.2.10").rg2_mount, DEFAULT_VISUAL_MOUNT)
        self.assertEqual(window.rg2_mount, DEFAULT_VISUAL_MOUNT)
        self.assertFalse(mount.apply.isEnabled())
        self.assertFalse(mount.reset.isEnabled())

    def test_saved_jaw_angle_is_preserved_until_pose_is_applied_or_reset(self):
        self.settings.setValue("rg2/simulation/jaw_yaw_deg", "34.5")
        self.settings.setValue("rg2/robots/192.0.2.10/jaw_yaw_deg", "-90")
        sim = self.window()
        self.assertEqual(sim.rg2_mount, (0., 0., 0., 0., 0., 34.5))
        self.assertEqual(self.window("192.0.2.10").rg2_mount, (0., 0., 0., 0., 0., -90.))
        sim.visual_mount.fields[2].setValue(22.5)
        sim.visual_mount.apply.click()
        self.assertEqual(self.window().rg2_mount, (0., 0., 22.5, 0., 0., 34.5))
        sim.visual_mount.reset.click()
        self.assertEqual(self.window().rg2_mount, DEFAULT_VISUAL_MOUNT)

    def test_invalid_settings_restore_default(self):
        for value in ("oops", [1., 2.], [0., 0., 0., 0., 0., "nan"]):
            with self.subTest(value=value):
                self.settings.setValue("rg2/simulation/pose", value)
                self.assertEqual(self.window().rg2_mount, DEFAULT_VISUAL_MOUNT)

    def test_initial_value_and_applied_edits_cross_pipe_separately_from_motion(self):
        sim = self.window()
        sim.visual_mount.fields[5].setValue(30.)
        sim.visual_mount.apply.click()
        reopened = self.window()
        panel = TeleopPanel.__new__(TeleopPanel)
        panel._open = True
        panel._messages = Queue()
        panel._messages.put({"ready": True, "rg2_mount": list(reopened.rg2_mount)})
        panel._wait_until_ready()
        self.assertEqual(panel.rg2_mount, (0., 0., 0., 0., 0., 30.))
        with patch("vai_robotics.ur5e.teleop_panel._incoming", return_value=Queue()):
            connection = _WindowConnection(reopened)
        reopened.visual_mount.fields[2].setValue(18.7)
        reopened.visual_mount.fields[5].setValue(85.)
        self.assertEqual(connection._take_operator_events(), {})
        reopened.visual_mount.apply.click()
        event = json.loads(json.dumps(connection._take_operator_events()))
        self.assertEqual(event, {"rg2_mount": [0., 0., 18.7, 0., 0., 85.]})
        panel._messages.put(event)
        self.assertIsNone(panel.poll())
        self.assertEqual(panel.rg2_mount, (0., 0., 18.7, 0., 0., 85.))
        self.assertEqual(connection._take_operator_events(), {})
