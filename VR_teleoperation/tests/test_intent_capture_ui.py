"""Offscreen capture UI checks with synthetic input and no network or robot I/O."""

import json
from datetime import datetime
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "vendor"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QInputMethodEvent
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QLineEdit

from package import intent_capture_ui as capture_ui
from package.intent_calibration import IntentTrajectoryCapture, load_intent_profile
from package.intent_offline import controller_frame, synthetic_capture


class FakeQuest:
    def __init__(self):
        self.port = 8443
        self.seq = 0
        self.closed = False

    def start(self):
        return self

    def latest(self):
        self.seq += 1
        return controller_frame(self.seq, time.monotonic(), grip=0.)

    def close(self):
        self.closed = True


class IntentCaptureUiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_controller_status_distinguishes_hands_and_expired_input(self):
        frame = controller_frame(1, 10., grip=0.)
        frame["hands"]["right"]["valid"] = False
        text = capture_ui.controller_status_text(frame, 10., .2, "tracking_lost_right")
        self.assertIn("왼손: 추적 신호 정상", text)
        self.assertIn("그립 0%", text)
        self.assertIn("오른손: 추적 안 됨", text)
        stale = capture_ui.controller_status_text(frame, 10.3, .2, "stale_pose")
        self.assertEqual(stale.count("입력 만료"), 2)
        self.assertNotIn("추적 신호 정상", stale)
        self.assertEqual(capture_ui.controller_status_text(None, 10., .2, "no_frame").count("입력 대기"), 2)
        frame["hands"]["right"].update(valid=True, emulated=True)
        self.assertIn("오른손: 추정 위치", capture_ui.controller_status_text(frame, 10., .2, "emulated_pose_right"))

    def test_save_names_use_save_date_and_reject_invalid_file_names(self):
        paths = capture_ui.calibration_save_paths('/tmp/capture', ' jaehoon_table A ', datetime(2026, 9, 23))
        self.assertEqual(paths['profile_path'].name, '260923_jaehoon_table A.json')
        self.assertEqual(paths['capture_path'].name, '260923_jaehoon_table A_trajectory.json')
        tomorrow = capture_ui.calibration_save_paths('/tmp/capture', 'jaehoon', datetime(2026, 9, 24))
        self.assertEqual(tomorrow['profile_path'].name, '260924_jaehoon.json')
        for invalid in ('', '  ', '../other', r'folder\name', 'name:one', 'a\nb', 'a' * 41, '한글', 'café', 'Ａlice', 'user１２', '\tname', '\u00a0name'):
            with self.subTest(name=invalid), self.assertRaises(ValueError):
                capture_ui.calibration_save_paths('/tmp/capture', invalid)

    def run_capture(self, root, on_poll):
        """Drive real widgets; bound failures without exceptions escaping Qt callbacks."""
        original_widget = capture_ui.QWidget
        windows, timers, callback_errors = [], [], []
        server = FakeQuest()

        def widget(*args, **kwargs):
            window = original_widget(*args, **kwargs)
            windows.append(window)
            timer = QTimer(window)
            timers.append(timer)
            deadline = time.monotonic() + 5.

            def poll():
                try:
                    self.assertLess(time.monotonic(), deadline, "Capture UI did not reach the expected state")
                    on_poll(window, server)
                except Exception as exc:
                    callback_errors.append(exc)
                    window.close()
                if not window.isVisible():
                    timer.stop()

            timer.timeout.connect(poll)
            timer.start(30)
            return window

        with patch.object(capture_ui, "QWidget", side_effect=widget), \
             patch.object(capture_ui, "QuestControllerServer", return_value=server) as server_factory, \
             patch.object(capture_ui, "create_self_signed_certificate", return_value=("fake-cert", "fake-key")) as certificate:
            try:
                result = capture_ui.run_intent_capture(root, "127.0.0.1", prepare_s=1., capture_s=3.)
            finally:
                for timer in timers:
                    timer.stop()

        self.assertEqual(callback_errors, [])
        self.assertTrue(server.closed)
        self.assertTrue(windows)
        self.assertTrue(all(not window.isVisible() for window in windows))
        certificate.assert_called_once_with(root / ".cache/fixed_quest/tls", "127.0.0.1")
        server_factory.assert_called_once_with(host="0.0.0.0", port=8443, certfile="fake-cert", keyfile="fake-key")
        return result

    @staticmethod
    def button(window, prefix):
        return next(button for button in window.findChildren(QPushButton) if button.text().startswith(prefix))

    def test_partial_capture_is_saved_as_rejected_with_raw_data_and_cleanup(self):
        observed = {}

        def drive(window, server):
            readiness = window.findChild(QLabel, "capture_readiness").text()
            if not observed and "수집 시작 가능" in readiness:
                window.findChild(QLineEdit, 'calibration_name_input').setText('partial')
                observed["signals"] = window.findChild(QLabel, "controller_input_status").text()
                observed["readiness"] = readiness
                self.assertTrue(self.button(window, "선택 동작 5회").isEnabled())
                self.button(window, "선택 동작 5회").click()
            elif readiness == "기록 중":
                observed.setdefault("recording_seq", server.seq)
                if server.seq - observed["recording_seq"] >= 10:
                    self.button(window, "저장·보정").click()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.run_capture(root, drive)
            saved = json.loads(result["capture_path"].read_text(encoding="utf-8"))
            profile = json.loads(result["profile_path"].read_text(encoding="utf-8"))
            self.assertEqual(result["capture_path"].parent, root / "calibration/diagnostics")
            self.assertEqual(result["profile_path"].parent, root / "calibration/diagnostics")
            self.assertEqual(result['profile_path'].name, datetime.now().strftime('%y%m%d') + '_partial.json')
            self.assertEqual(result['capture_path'].stem, result['profile_path'].stem + '_trajectory')
            self.assertEqual(saved["type"], "quest_intent_trajectories")
            self.assertFalse(profile["fit_success"])
            self.assertEqual(profile["reason"], "incomplete_capture")
            self.assertEqual(profile, result["profile"])
            self.assertEqual(len(saved["attempts"]), 1)
            attempt = saved["attempts"][0]
            self.assertFalse(attempt["accepted"])
            self.assertEqual(attempt["reason"], "window_closed")
            self.assertGreaterEqual(len(attempt["raw_samples"]), 10)
            self.assertTrue(all("hands" in sample for sample in attempt["raw_samples"]))
            self.assertFalse((root / "calibration/profiles").exists())
        self.assertIn("왼손: 추적 신호 정상", observed["signals"])
        self.assertIn("오른손: 추적 신호 정상", observed["signals"])
        self.assertIn("수집 시작 가능", observed["readiness"])

    def test_completed_synthetic_capture_saves_loadable_profile_with_provenance(self):
        # Exercise the real fit/save/load path; this fixture is not personal calibration.
        snapshot = synthetic_capture()
        collector = IntentTrajectoryCapture()

        def drive(window, server):
            if "수집 시작 가능" in window.findChild(QLabel, "capture_readiness").text():
                window.findChild(QLineEdit, 'calibration_name_input').setText('jaehoon_tableA')
                self.button(window, "저장·보정").click()

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(capture_ui, "IntentTrajectoryCapture", return_value=collector), \
             patch.object(collector, "snapshot", return_value=snapshot):
            root = Path(tmp)
            result = self.run_capture(root, drive)
            saved = json.loads(result["capture_path"].read_text(encoding="utf-8"))
            self.assertEqual(result["capture_path"].parent, root / "calibration/diagnostics")
            self.assertEqual(result["profile_path"].parent, root / "calibration/profiles")
            self.assertEqual(result['profile_path'].name, datetime.now().strftime('%y%m%d') + '_jaehoon_tableA.json')
            self.assertEqual(result['capture_path'].stem, result['profile_path'].stem + '_trajectory')
            self.assertEqual(list((root / 'calibration/diagnostics').iterdir()), [result['capture_path']])
            self.assertTrue(saved["status"]["complete"])
            self.assertEqual(saved["origin"], "synthetic")
            profile, matrices = load_intent_profile(result["profile_path"])
            self.assertTrue(profile["fit_success"])
            self.assertEqual(profile["origin"], "synthetic")
            self.assertEqual(profile, result["profile"])
            self.assertEqual(set(matrices), {"left", "right"})
            with self.assertRaisesRegex(ValueError, "synthetic_profile_cannot_control_real_robot"):
                load_intent_profile(result["profile_path"], real=True)

    def test_empty_invalid_and_duplicate_names_keep_save_disabled_until_renamed(self):
        def drive(window, server):
            name = window.findChild(QLineEdit, 'calibration_name_input')
            finish = self.button(window, '저장·보정')
            hint = window.findChild(QLabel, 'calibration_save_hint')
            self.assertFalse(finish.isEnabled())
            for invalid in ('한글', 'café', 'Ａlice', 'user１２'):
                name.insert(invalid)
                self.assertEqual(name.text(), '')
            composition = QInputMethodEvent()
            composition.setCommitString('한글')
            QApplication.sendEvent(name, composition)
            self.assertEqual(name.text(), '')
            name.insert('User_01-test A')
            self.assertEqual(name.text(), 'User_01-test A')
            self.assertTrue(finish.isEnabled())
            name.clear()
            name.setText('../invalid')
            self.assertFalse(finish.isEnabled())
            name.setText('duplicate')
            self.assertFalse(finish.isEnabled())
            self.assertIn('다른 이름', hint.text())
            name.setText('new_name')
            self.assertTrue(finish.isEnabled())
            self.assertIn(datetime.now().strftime('%y%m%d') + '_new_name', hint.text())
            finish.click()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing = capture_ui.calibration_save_paths(root, 'duplicate')['profile_path']
            existing.parent.mkdir(parents=True)
            existing.write_text('existing personal calibration', encoding='utf-8')
            result = self.run_capture(root, drive)
            self.assertEqual(existing.read_text(encoding='utf-8'), 'existing personal calibration')
            self.assertEqual(result['profile_path'].name, datetime.now().strftime('%y%m%d') + '_new_name.json')
            self.assertFalse(capture_ui.calibration_save_paths(root, 'duplicate')['capture_path'].exists())

    def test_completed_trial_journal_moves_with_named_capture(self):
        observed = {}

        def drive(window, server):
            readiness = window.findChild(QLabel, 'capture_readiness').text()
            if not observed and '수집 시작 가능' in readiness:
                window.findChild(QLineEdit, 'calibration_name_input').setText('trial_log')
                observed['started'] = True
                self.button(window, '선택 동작 5회').click()
            elif readiness == '기록 중':
                observed['recorded'] = True
            elif observed.get('recorded') and '수집 시작 가능' in readiness:
                self.button(window, '저장·보정').click()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.run_capture(root, drive)
            journal = result['capture_path'].with_suffix('.jsonl')
            rows = [json.loads(line) for line in journal.read_text(encoding='utf-8').splitlines()]
            snapshot = json.loads(result['capture_path'].read_text(encoding='utf-8'))
            self.assertEqual(rows, snapshot['attempts'])
            self.assertEqual(len(rows), 1)
            self.assertGreaterEqual(len(rows[0]['raw_samples']), 60)
            self.assertEqual(list(journal.parent.glob('*.jsonl')), [journal])

    def test_close_without_name_preserves_automatic_recovery_save(self):
        def drive(window, server):
            window.close()

        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_capture(Path(tmp), drive)
            self.assertTrue(result['capture_path'].is_file())
            self.assertTrue(result['profile_path'].is_file())
            self.assertFalse(result['profile']['fit_success'])


if __name__ == "__main__":
    unittest.main()
