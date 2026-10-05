"""Build actual control widgets with all transports/viewers replaced by mocks."""
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / 'package/init_project.py'))

import numpy as np
from PyQt5.QtWidgets import QApplication, QLineEdit, QPushButton, QComboBox, QLabel
from package import intent_teleop as teleop
from package.intent_calibration import _digest, load_intent_profile


class IntentRuntimeUiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.models = teleop.build_models(teleop.IntentTeleopSettings())

    def test_real_has_arm_speed_and_manual_controls_but_no_tracking_or_gripper_controls(self):
        self.check_ui('real', 60., 60.)
        self.check_ui('real', 100., 120.)

    def test_sim_keeps_optional_controls(self):
        self.check_ui('sim', 60., 60.)

    def check_ui(self, backend, arm_speed, gripper_speed):
        cfg = teleop.IntentTeleopSettings(backend=backend, QUEST_HOST='192.0.2.1',
            VIEWER_MODE='process', CERTFILE='unused', KEYFILE='unused',
            PROFILE_PATH=ROOT / 'calibration/diagnostics/v1.json',
            MAX_JOINT_SPEED=np.deg2rad(arm_speed), MAX_GRIPPER_SPEED=np.deg2rad(gripper_speed),
            TRACKING_PAUSE_S=.1, ACCEPT_EMULATED_POSE=True,
            TRACKING_MODE_BUTTONS=('emulated', 'pause_0.5'))
        windows = []
        original_widget = teleop.MultiRadioQtWidget
        def widget(*args, **kwargs):
            window = original_widget(*args, **kwargs)
            windows.append(window)
            return window
        def recorder(*args, configuration, **kwargs):
            window = windows[-1]
            chooser = window.findChild(QComboBox, 'profile_file_selector')
            self.assertIsNotNone(chooser)
            self.assertEqual(chooser.currentData(), str(cfg.PROFILE_PATH.resolve()))
            self.assertIn('v1.json', window.findChild(QLabel, 'profile_selection_status').text())
            self.assertEqual(float(window.findChild(QLineEdit, 'teleop_speed_input').text()), arm_speed)
            self.assertIsNotNone(window.findChild(QPushButton, 'manual_joint_preview'))
            self.assertIsNotNone(window.findChild(QPushButton, 'manual_joint_apply'))
            for cls, name in [(QLineEdit, 'gripper_speed_input'), (QPushButton, 'gripper_speed_apply'),
                              (QPushButton, 'tracking_mode_emulated'), (QPushButton, 'tracking_mode_pause_0.5')]:
                self.assertEqual(window.findChild(cls, name) is not None, backend == 'sim', name)
            profile, _ = load_intent_profile(cfg.PROFILE_PATH)
            self.assertEqual(configuration['profile_path'], str(cfg.PROFILE_PATH.resolve()))
            self.assertEqual(configuration['profile_sha256'], _digest(profile))
            self.assertEqual(configuration['intent_profile'], profile)
            self.assertEqual(configuration['max_gripper_speed_rad_s'], cfg.MAX_GRIPPER_SPEED)
            self.assertEqual(configuration['accept_emulated_pose'], True)
            self.assertEqual(configuration['tracking_pause_s'], .1)
            self.assertEqual(len(configuration['tracking_mode_buttons']), 2 if backend == 'sim' else 0)
            window.close()  # Stop before a control tick; no actual transport or viewer.
            return Mock()
        server = Mock(port=8443)
        server.start.return_value = server
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(teleop, 'PROJECT_DIR', Path(tmp)), \
             patch.object(teleop, 'build_models', return_value=self.models), \
             patch.object(teleop, 'MultiRadioQtWidget', side_effect=widget), \
             patch.object(teleop, 'MuJoCoRenderProcess'), \
             patch.object(teleop, 'SG2RobotStateReceiver') as receiver, \
             patch.object(teleop, 'SG2RobotCommandPublisher'), \
             patch.object(teleop, 'SG2FixedQuestReal') as runtime, \
             patch.object(teleop, 'IntentSimBridge'), \
             patch.object(teleop, 'QuestControllerServer', return_value=server), \
             patch.object(teleop, 'ControllerRecorder', side_effect=recorder):
            path = teleop.run_intent_teleop(cfg)
            self.assertEqual(path.with_suffix('.status.txt').read_text().strip(), 'PC_STOP_or_control_window_closed')
            self.assertEqual(runtime.call_args.kwargs['tracking_pause_s'], .1)
            runtime.return_value.tick_grip.assert_not_called()
            if backend == 'sim':
                receiver.assert_not_called()
        for window in windows:
            window.deleteLater()
        self.app.processEvents()


if __name__ == '__main__':
    unittest.main()
