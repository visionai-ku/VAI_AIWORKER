"""Calibration compatibility and notebook selection, with no robot/Quest I/O."""
import ast
import copy
import json
import os
from types import SimpleNamespace
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / 'package/init_project.py'))

from package.intent_calibration import (HANDS, LABELS, MAX_PROFILE_BYTES,
    IntentTrajectoryCapture, _digest, fit_intent_profile, load_intent_profile, validate_profile)
from package.intent_offline import synthetic_capture, synthetic_basis
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PyQt5.QtWidgets import QApplication
from package.intent_profile_selector import CommandCenterProfileSelector, profile_options
from package.intent_teleop import IntentTeleopSettings, selected_profile


def sign(profile):
    profile['profile_id'] = _digest({k: v for k, v in profile.items() if k != 'profile_id'})[:16]
    return profile


class IntentProfileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.capture = synthetic_capture()
        cls.fitted = fit_intent_profile(cls.capture)
        cls.legacy_path = ROOT / 'calibration/diagnostics/v1.json'
        cls.legacy = json.loads(cls.legacy_path.read_text())

    def test_twenty_trials_fit_all_five_repeats(self):
        self.assertEqual(LABELS, ('forward', 'left', 'up', 'right'))
        self.assertEqual(len(self.capture['attempts']), 20)
        self.assertTrue(self.capture['status']['complete'])
        profile = self.fitted
        self.assertEqual(profile['version'], 2)
        self.assertTrue(profile['fit_success'])
        self.assertEqual(profile['train_repeats'], [1, 2, 3, 4, 5])
        for key in ('accepted', 'split', 'failed_checks'):
            self.assertNotIn(key, profile)
        for hand, matrix in validate_profile(profile).items():
            np.testing.assert_allclose(matrix @ synthetic_basis(), np.eye(3), atol=.005)
            self.assertNotIn('checks', profile['hands'][hand])
        with self.assertRaises(ValueError):
            IntentTrajectoryCapture().begin_trial('left_up')

    def test_fifth_repeat_changes_fit_for_every_direction(self):
        # Tilt only repeat 5; a 1..4 fitter would return exactly the old matrix.
        for label in LABELS:
            capture = copy.deepcopy(self.capture)
            rotation = np.asarray(capture['reference']['R_OQ'])
            trial = next(a for a in capture['attempts'] if a['label'] == label and a['repeat_index'] == 5)
            axis = {'forward': 0, 'left': 1, 'up': 2, 'right': 1}[label]
            for hand in HANDS:
                start = np.asarray(trial['raw_samples'][0]['hands'][hand]['position_m'])
                for sample in trial['raw_samples']:
                    delta = rotation @ (np.asarray(sample['hands'][hand]['position_m']) - start)
                    delta[(axis + 1) % 3] += .5 * delta[axis]
                    sample['hands'][hand]['position_m'] = (start + rotation.T @ delta).tolist()
            fitted = fit_intent_profile(capture)
            self.assertTrue(fitted['fit_success'], label)
            for hand in HANDS:
                change = np.max(np.abs(np.asarray(fitted['hands'][hand]['correction']) - self.fitted['hands'][hand]['correction']))
                self.assertGreater(change, .01, (label, hand))

    def test_rejected_superseded_invalidated_and_missing_trials_do_not_count(self):
        for flag in ('accepted', 'superseded', 'invalidated_reason', 'missing'):
            capture = copy.deepcopy(self.capture)
            if flag == 'missing':
                capture['attempts'].pop()
            else:
                capture['attempts'][-1][flag] = False if flag == 'accepted' else 'excluded'
            self.assertFalse(fit_intent_profile(capture)['fit_success'], flag)
        capture = copy.deepcopy(self.capture)
        capture['attempts'].append(copy.deepcopy(capture['attempts'][0]))
        self.assertFalse(fit_intent_profile(capture)['fit_success'])

    def test_legacy_profile_loads_exact_stored_matrices_without_review_or_refit(self):
        before = self.legacy_path.read_bytes()
        self.assertFalse(self.legacy['accepted'])
        self.assertEqual(self.legacy['reason'], 'trajectory_validation_failed')
        with patch('package.intent_calibration.fit_intent_profile', side_effect=AssertionError('No refit')):
            for real in (False, True):
                profile, matrices = load_intent_profile(self.legacy_path, real=real)
                self.assertEqual(profile, self.legacy)
                for hand in HANDS:
                    np.testing.assert_array_equal(matrices[hand], self.legacy['hands'][hand]['correction'])
        self.assertEqual(before, self.legacy_path.read_bytes())
        accepted = sign(dict(copy.deepcopy(self.legacy), accepted=True, reason='accepted'))
        validate_profile(accepted)

    def test_legacy_capture_refits_without_diagonals_and_keeps_source_digest(self):
        path = ROOT / 'calibration/diagnostics/v1_trajectory.json'
        before = path.read_bytes()
        capture = json.loads(before)
        profile = fit_intent_profile(capture)
        self.assertTrue(profile['fit_success'], profile['reason'])
        self.assertEqual(profile['capture_sha256'], _digest(capture))
        stripped = copy.deepcopy(capture)
        stripped['attempts'] = [a for a in stripped['attempts'] if a['label'] in LABELS]
        self.assertLess(len(stripped['attempts']), len(capture['attempts']))
        reduced = fit_intent_profile(stripped)
        for hand in HANDS:
            np.testing.assert_array_equal(profile['hands'][hand]['correction'], reduced['hands'][hand]['correction'])
        self.assertNotEqual(profile['capture_sha256'], reduced['capture_sha256'])
        self.assertEqual(before, path.read_bytes())

    def test_invalid_hash_incomplete_and_bad_matrices_are_rejected(self):
        changed = copy.deepcopy(self.legacy)
        changed['hands']['left']['correction'][0][0] += .01
        with self.assertRaisesRegex(ValueError, 'profile_content_changed'):
            validate_profile(changed)
        for version in (1, 2):
            incomplete = copy.deepcopy(self.legacy if version == 1 else self.fitted)
            incomplete.update(reason='incomplete_capture', accepted=False, fit_success=False)
            with self.assertRaisesRegex(ValueError, 'profile_fit_incomplete'):
                validate_profile(sign(incomplete))
        for field, value in [('basis', np.zeros((3, 3)).tolist()), ('correction', [[1, 2, 3]]),
                             ('correction', np.eye(3).tolist())]:
            bad = copy.deepcopy(self.fitted)
            bad['hands']['left'][field] = value
            with self.assertRaisesRegex(ValueError, 'invalid_profile_basis_left'):
                validate_profile(sign(bad))
        bad = copy.deepcopy(self.fitted)
        del bad['hands']['right']
        with self.assertRaisesRegex(ValueError, 'invalid_profile_basis_right'):
            validate_profile(sign(bad))
        bad = copy.deepcopy(self.fitted)
        bad['hands']['left']['correction'][0][0] = float('nan')
        with self.assertRaises(ValueError):
            validate_profile(bad)

    def test_degenerate_capture_cannot_create_usable_profile(self):
        capture = copy.deepcopy(self.capture)
        for sample in capture['attempts'][0]['raw_samples']:
            for hand in HANDS:
                sample['hands'][hand]['position_m'] = [0., 0., 0.]
        self.assertFalse(fit_intent_profile(capture)['fit_success'])

    def test_synthetic_and_oversized_profiles_blocked_for_real(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'synthetic.json'
            path.write_text(json.dumps(self.fitted))
            load_intent_profile(path)
            with self.assertRaisesRegex(ValueError, 'synthetic_profile_cannot_control_real_robot'):
                load_intent_profile(path, real=True)
            path.write_bytes(b' ' * (MAX_PROFILE_BYTES + 1))
            with self.assertRaisesRegex(ValueError, 'profile_too_large'):
                load_intent_profile(path)


class ProfileSelectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.legacy = json.loads((ROOT / 'calibration/diagnostics/v1.json').read_text())
        self.a = self.save('diagnostics/jaehoon.json', self.legacy)
        second = sign(dict(copy.deepcopy(self.legacy), created_at='2026-10-05T00:00:00Z'))
        self.b = self.save('profiles/민수.json', second)
        self.cfg = IntentTeleopSettings(backend='real', PROFILE_PATH=self.a)
        self.selector = CommandCenterProfileSelector(self.root, self.cfg, self.legacy)
        self.addCleanup(self.selector.close)
        self.addCleanup(self.selector.deleteLater)
        self.runtime = SimpleNamespace(_grip_engaged=False, armed=False, _grip_home_active=False,
            home_pending=False, planner=SimpleNamespace(homing=False), set_intent_profile=Mock())

    def save(self, name, data):
        path = self.root / 'calibration' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return path

    def select(self, path):
        index = self.selector.combo.findData(None if path is None else str(path))
        self.assertGreaterEqual(index, 0)
        self.selector.combo.setCurrentIndex(index)
        self.selector.combo.activated[int].emit(index)

    def test_discovery_excludes_captures_reviews_and_oversized_files(self):
        self.save('diagnostics/raw.json', {'type': 'quest_intent_trajectories'})
        self.save('diagnostics/review.json', {'type': 'quest_intent_real_review'})
        huge = self.save('profiles/huge.json', self.legacy)
        huge.write_bytes(b' ' * (MAX_PROFILE_BYTES + 1))
        options = profile_options(self.root)
        self.assertEqual({p for _, p in options}, {str(self.a), str(self.b)})
        self.assertTrue(any(label.startswith('민수.json') for label, _ in options))

    def test_file_click_applies_automatically_and_reports_provenance(self):
        self.select(self.b)
        self.assertEqual(self.cfg.PROFILE_PATH, self.a)  # Tick owns the atomic change.
        event = self.selector.poll(self.runtime, now=0.)
        self.assertTrue(event['accepted'])
        self.assertEqual(self.cfg.PROFILE_PATH, self.b)
        profile, matrices = selected_profile(self.cfg)
        self.assertEqual(event['intent_profile'], profile)
        self.assertEqual(event['profile_sha256'], _digest(profile))
        self.assertEqual(event['profile_path'], str(self.b))
        self.assertIn(profile['profile_id'], self.selector.status.text())
        self.assertIn(self.b.name, self.selector.status.text())
        args = self.runtime.set_intent_profile.call_args.args
        self.assertEqual(args[1], profile['profile_id'])
        for hand in HANDS:
            np.testing.assert_array_equal(args[0][hand], matrices[hand])
        self.assertIsNone(self.selector.poll(self.runtime, now=.01))

    def test_refresh_preserves_active_profile_and_identity_is_explicit(self):
        self.save('profiles/new.json', self.legacy)
        self.selector.refresh_button.click()
        self.assertEqual(self.selector.combo.currentData(), str(self.a))
        self.assertEqual(self.selector.combo.count(), 4)
        self.runtime.set_intent_profile.assert_not_called()
        self.select(None)
        event = self.selector.poll(self.runtime, now=0.)
        self.assertTrue(event['accepted'])
        self.assertIsNone(self.cfg.PROFILE_PATH)
        self.assertEqual(selected_profile(self.cfg), (None, None))
        self.runtime.set_intent_profile.assert_called_once_with(None, None, now=0.)

    def test_identity_and_original_v1_are_distinct_choices_with_exact_stored_matrices(self):
        v1 = self.save('diagnostics/v1.json', self.legacy)
        original = v1.read_bytes()
        self.selector.refresh()
        self.assertEqual(self.selector.combo.itemText(0), '보정 미적용 (identity)')
        self.assertEqual(self.selector.combo.itemText(1), 'v1 원본 (v1.json)')
        self.select(None)
        self.selector.poll(self.runtime, now=0.)
        self.select(v1)
        with patch('package.intent_calibration.fit_intent_profile', side_effect=AssertionError('No refit')):
            event = self.selector.poll(self.runtime, now=.01)
        self.assertTrue(event['accepted'])
        self.assertEqual(event['intent_profile'], self.legacy)
        matrices = self.runtime.set_intent_profile.call_args.args[0]
        for hand in HANDS:
            np.testing.assert_array_equal(matrices[hand], self.legacy['hands'][hand]['correction'])
        self.assertEqual(v1.read_bytes(), original)

    def test_invalid_file_keeps_active_profile_and_reports_failed_filename(self):
        bad = self.save('profiles/incomplete.json', dict(self.legacy, reason='incomplete_capture'))
        self.selector.refresh()
        self.select(bad)
        event = self.selector.poll(self.runtime, now=0.)
        self.assertFalse(event['accepted'])
        self.assertEqual(self.cfg.PROFILE_PATH, self.a)
        self.assertEqual(self.selector.combo.currentData(), str(self.a))
        self.assertIn('incomplete.json', self.selector.status.text())
        self.assertIn('profile_fit_incomplete', self.selector.status.text())
        self.assertIn(self.a.name, self.selector.status.text())
        self.runtime.set_intent_profile.assert_not_called()
        bad.unlink()
        self.select(bad)
        self.assertFalse(self.selector.poll(self.runtime, now=0.)['accepted'])
        self.assertEqual(self.cfg.PROFILE_PATH, self.a)

    def test_busy_click_is_rejected_and_does_not_apply_after_release(self):
        self.select(self.b)
        self.runtime.armed = True
        event = self.selector.poll(self.runtime, now=0.)
        self.assertFalse(event['accepted'])
        self.assertFalse(self.selector.combo.isEnabled())
        self.assertFalse(self.selector.refresh_button.isEnabled())
        self.runtime.armed = False
        self.assertIsNone(self.selector.poll(self.runtime, now=.1))
        self.assertTrue(self.selector.combo.isEnabled())
        self.assertEqual(self.cfg.PROFILE_PATH, self.a)
        self.runtime.set_intent_profile.assert_not_called()
        self.select(self.b)
        self.assertTrue(self.selector.poll(self.runtime, now=.1)['accepted'])

    def test_recording_and_reset_sequence_lock_selection(self):
        record = SimpleNamespace(recorder=SimpleNamespace(recording=False), widget=Mock())
        record.widget.get_button_option.return_value = 'REC'
        buttons = SimpleNamespace(record_control=record, lock_input=False)
        self.selector.poll(self.runtime, buttons, now=0.)
        self.assertFalse(self.selector.combo.isEnabled())
        record.widget.get_button_option.return_value = 'STOP'
        record.recorder.recording = True
        self.selector.poll(self.runtime, buttons, now=0.)
        self.assertFalse(self.selector.combo.isEnabled())
        record.recorder.recording = False
        buttons.lock_input = True
        self.selector.poll(self.runtime, buttons, now=0.)
        self.assertFalse(self.selector.combo.isEnabled())
        buttons.lock_input = False
        self.selector.poll(self.runtime, buttons, now=0.)
        self.assertTrue(self.selector.combo.isEnabled())

    def test_released_idle_guard_error_preserves_active_selection(self):
        self.runtime.set_intent_profile.side_effect = ValueError('release both grips')
        self.select(self.b)
        self.assertFalse(self.selector.poll(self.runtime, now=0.)['accepted'])
        self.assertEqual(self.cfg.PROFILE_PATH, self.a)
        self.assertEqual(self.selector.profile, self.legacy)
        self.assertEqual(self.selector.combo.currentData(), str(self.a))

    def run_cell(self, recording):
        path = ROOT / 'real_notebook' / ('real_vr_teleop_record.ipynb' if recording else 'real_vr_teleop.ipynb')
        return next(''.join(c['source']) for c in json.loads(path.read_text())['cells']
                    if c['cell_type'] == 'code' and 'log_path = run_intent_teleop(' in ''.join(c['source']))

    def test_notebook_runs_without_widgets_and_preserves_recording_order(self):
        for recording in (False, True):
            events = []
            recorder, control = Mock(), Mock()
            recorder.start_thread.side_effect = lambda: events.append('start')
            recorder.close.side_effect = lambda: events.append('save_close')
            control.close.side_effect = lambda: events.append('control_close')
            def validate(cfg):
                events.append('validate')
                return selected_profile(cfg)
            namespace = dict(settings=self.cfg, recorder=recorder, selected_profile=validate,
                RecordControl=lambda r: control, run_intent_teleop=lambda *a, **kw: events.append('run'))
            with patch('package.vr_buttons.ControllerButtonActions', return_value=Mock()):
                exec(self.run_cell(recording), namespace)
            self.assertEqual(events, ['validate', 'start', 'run', 'save_close', 'control_close']
                             if recording else ['validate', 'run'])
            self.assertNotIn('profile_selector', self.run_cell(recording))

    def test_invalid_startup_profile_stops_before_recording_or_teleop(self):
        self.a.write_text('{}')
        for recording in (False, True):
            recorder, run = Mock(), Mock()
            with self.assertRaises(ValueError):
                exec(self.run_cell(recording), dict(settings=self.cfg, selected_profile=selected_profile,
                    recorder=recorder, RecordControl=Mock(), run_intent_teleop=run))
            recorder.start_thread.assert_not_called()
            run.assert_not_called()

    def test_recording_cleanup_still_runs_on_teleop_error(self):
        recorder, control = Mock(), Mock()
        with patch('package.vr_buttons.ControllerButtonActions', return_value=Mock()):
            with self.assertRaisesRegex(RuntimeError, 'run failed'):
                exec(self.run_cell(True), dict(settings=self.cfg, selected_profile=selected_profile,
                    recorder=recorder, RecordControl=lambda r: control,
                    run_intent_teleop=Mock(side_effect=RuntimeError('run failed'))))
        recorder.close.assert_called_once()
        control.close.assert_called_once()

    def test_notebook_control_defaults_remain_unchanged(self):
        for name, arm, gripper, home, stall in [('real_vr_teleop.ipynb', 60., 60., 10., 'off'),
                                                ('real_vr_teleop_record.ipynb', 100., 60., 20., 'observe')]:
            notebook = json.loads((ROOT / 'real_notebook' / name).read_text())
            source = next(''.join(c['source']) for c in notebook['cells'] if 'settings = IntentTeleopSettings(' in ''.join(c['source']))
            tree = ast.parse(source)
            call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                        and n.func.id == 'IntentTeleopSettings')
            config = ast.Expression(body=call)
            cfg = eval(compile(config, name, 'eval'), dict(IntentTeleopSettings=IntentTeleopSettings,
                PROJECT_DIR=ROOT, ROBOT_IP='192.168.6.2', find_lan_ip=lambda _: '192.0.2.1', np=np))
            cfg.validate()
            self.assertAlmostEqual(np.rad2deg(cfg.MAX_JOINT_SPEED), arm)
            self.assertAlmostEqual(np.rad2deg(cfg.MAX_GRIPPER_SPEED), gripper)
            self.assertAlmostEqual(np.rad2deg(cfg.MAX_HOME_JOINT_SPEED), home)
            self.assertEqual(cfg.TRACKING_PAUSE_S, .1)
            self.assertTrue(cfg.ACCEPT_EMULATED_POSE)
            self.assertTrue(cfg.DRY_RUN)
            self.assertFalse(cfg.TELEOP_COLLISION_CHECK)
            self.assertEqual(cfg.IK_RECOVERY_TIMEOUT_S, 2.)
            np.testing.assert_array_equal(cfg.POSITION_SCALES['left'], [.5, .5, .8])
            self.assertEqual(cfg.STALL_LIMIT_MODE, stall)

    def test_home_speed_accepts_up_to_sixty_degrees_per_second(self):
        for degrees in (5., 10., 20., 60.):
            cfg = IntentTeleopSettings(MAX_HOME_JOINT_SPEED=np.deg2rad(degrees)).validate()
            self.assertAlmostEqual(np.rad2deg(cfg.MAX_HOME_JOINT_SPEED), degrees)
        for degrees in (0., -10., 61., np.nan):
            with self.subTest(degrees=degrees), self.assertRaisesRegex(ValueError, 'home speed'):
                IntentTeleopSettings(MAX_HOME_JOINT_SPEED=np.deg2rad(degrees)).validate()

    def test_stall_mode_accepts_only_off_and_observe(self):
        for mode in ('off', 'observe'):
            self.assertEqual(IntentTeleopSettings(STALL_LIMIT_MODE=mode).validate().STALL_LIMIT_MODE, mode)
        for options in ({'STALL_LIMIT_MODE': 'limit'}, {'STALL_LIMIT_MODE': 'OBSERVE'},
                        {'STALL_ENTER_LEAD_DEG': 3.}, {'STALL_WINDOW_S': 0.}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                IntentTeleopSettings(**options).validate()


if __name__ == '__main__':
    unittest.main()
