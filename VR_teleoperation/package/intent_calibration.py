"""Local, full-rank intent axes. No IK, robot transport or implicit height lock.

Lengths remain metres: each learned input basis column is unit length. Numeric
XYZ gains belong to ControllerMapper and are not absorbed into this fit.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from scipy.spatial.transform import Rotation
from .quest_pose_calibration import QuestPoseCalibration

HANDS = ('left', 'right')
AXES = ('forward', 'left', 'up')
TRAIN_LABELS = ('forward', 'left', 'up', 'right')
VALIDATION_LABELS = ('left_up', 'right_up')
LABELS = TRAIN_LABELS + VALIDATION_LABELS
# Desired signs on each labelled axis; right is -Y, right_up is -Y/+Z.
LABEL_COMPONENTS = {
    'forward': ([0], [1]), 'left': ([1], [1]), 'up': ([2], [1]),
    'forward_up': ([0, 2], [1, 1]), 'left_up': ([1, 2], [1, 1]),
    'right': ([1], [-1]), 'right_up': ([1, 2], [-1, 1]),
}
LABEL_TEXT = {
    'forward': '높이를 유지하려는 의도로 앞으로 뻗고 편한 시작 자세로 돌아오기',
    'left': '몸통 고정 · 높이와 앞뒤 거리 유지 · 두 손을 함께 왼쪽(+Y)으로 평행 이동 후 복귀',
    'up': '앞뒤 위치를 유지하려는 의도로 위로 올렸다가 돌아오기',
    'forward_up': '앞으로 가면서 의도적으로 위로 올렸다가 돌아오기 (검증 전용)',
    'left_up': '두 손을 함께 왼쪽 위(+Y/+Z)로 대각선 이동 후 복귀 (검증 전용)',
    'right': '몸통 고정 · 높이와 앞뒤 거리 유지 · 두 손을 함께 오른쪽(-Y)으로 평행 이동 후 복귀',
    'right_up': '두 손을 함께 오른쪽 위(-Y/+Z)로 대각선 이동 후 복귀 (검증 전용)',
}


def _jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def save_json(path, value):
    """Create an output without replacing a user's existing capture/profile."""
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=_jsonable)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as file: file.write(payload + '\n')
    return path


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False, default=_jsonable).encode()).hexdigest()


def _trajectory(samples, hand, rotation):
    p = np.asarray([sample['hands'][hand]['position_m'] for sample in samples], float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError('invalid_trajectory_positions')
    # First/last 0.3s are explicitly stationary in the capture UI.
    count = max(3, min(10, len(p) // 10))
    return (p - np.mean(p[:count], axis=0)) @ np.asarray(rotation, float).T


def _metrics(delta, allowed, signs=None):
    peak = float(np.max(np.linalg.norm(delta, axis=1)))
    other = [axis for axis in range(3) if axis not in allowed]
    leakage = float(np.percentile(np.linalg.norm(delta[:, other], axis=1), 95))
    # Positive means along the labelled intention, including negative robot axes.
    extents = np.percentile(delta[:, allowed] * (1 if signs is None else np.asarray(signs)), 95, axis=0)
    return dict(peak_m=peak, cross_axis_p95_m=leakage,
                cross_axis_ratio=leakage / max(peak, 1e-9),
                intended_positive_extent_m=np.asarray(extents).tolist())


class IntentTrajectoryCapture(QuestPoseCalibration):
    """Reuse validated WebXR/reference tracking; accept motion, not static jitter.

    Five complete repeats per label. Rejected attempts keep raw data and do not
    consume repeats. Reference changes invalidate the set; no robot is commanded.
    """
    _pose_labels = LABELS

    def __init__(self, max_wrist_rotation_deg=30., **options):
        if (isinstance(max_wrist_rotation_deg, bool) or not np.isfinite(max_wrist_rotation_deg)
                or not 0 < max_wrist_rotation_deg <= 180):
            raise ValueError('max_wrist_rotation_deg must be in (0, 180]')
        defaults = dict(repeats=5, min_samples=60, min_duration_s=2.,
                        max_samples_per_trial=1200, max_attempts=100)
        defaults.update(options)
        super().__init__(**defaults)
        self.config['max_wrist_rotation_deg'] = float(max_wrist_rotation_deg)

    def reset(self):
        self._resume_source = None
        return super().reset()

    def restore_capture(self, capture):
        """Revalidate accepted trials only, without merging reference frames.

        Atomic on error. Next live input still has to match the saved session,
        origin and headset reference; rejected/superseded trials are not promoted.
        """
        if self._attempts or self._active_label is not None or self._reference is not None:
            raise ValueError('Restore requires an empty collector')
        if (capture.get('type') != 'quest_intent_trajectories' or capture.get('version') != 1
                or capture.get('origin') != 'quest' or not capture.get('reference')):
            raise ValueError('Resume requires a saved personal trajectory capture with a reference')
        ref = copy.deepcopy(capture['reference'])
        scratch = type(self)(**self.config)
        scratch._reference = ref
        selected = [a for a in capture['attempts'] if a.get('accepted') and not a.get('superseded')
                    and a.get('label') in LABELS]
        for a in selected:
            label = a['label']
            if a['repeat_index'] != len(scratch._trials[label]) + 1:
                raise ValueError('Invalid saved repeat order: ' + label)
            scratch.begin_trial(label)
            for sample in a['raw_samples']:
                frame = dict(version=1, session_id=ref['session_id'], origin_id=ref['origin_id'],
                    seq=sample['seq'], sample_time_ms=sample['sample_time_ms'],
                    received_at=sample['received_at'], visibility='visible', head=copy.deepcopy(sample['head']),
                    hands={h:dict(copy.deepcopy(sample['hands'][h]),valid=True,emulated=False,grip=0.,trigger=0.) for h in HANDS})
                state = scratch.add_frame(frame, now=sample['observed_at'])
                if state['state'] == 'ABORTED':
                    raise ValueError('Saved reference invalid: ' + state['reason'])
            result = scratch.finish_trial()
            if not result['accepted']:
                raise ValueError('Saved trial no longer passes: %s/%s: %s' % (label, a['repeat_index'], result['reason']))
        if not selected:
            raise ValueError('No accepted trials available to resume')
        self._reference = ref
        self._trials, self._attempts = scratch._trials, scratch._attempts
        self._reader = scratch._reader
        self._resume_source = dict(capture_sha256=_digest(capture), created_at=capture.get('created_at'),
            imported_counts={k:len(v) for k,v in self._trials.items()},
            ignored_labels=sorted({a['label'] for a in capture['attempts'] if a['label'] not in LABELS}))
        self._input_valid, self._input_reason = False, 'awaiting_live_reference_check'
        self.state, self.reason = 'IDLE', 'awaiting_live_reference_check'
        return self.status()

    def finish_trial(self):
        if self.state != 'COLLECTING': raise ValueError('No trial is collecting')
        report = dict(label=self._active_label,
            repeat_index=len(self._trials[self._active_label]) + 1,
            sample_count=len(self._samples), duration_s=self._duration(),
            raw_samples=copy.deepcopy(self._samples), hands={}, accepted=False,
            reason=self._trial_error or 'too_few_samples')
        if self._trial_error or len(self._samples) < self.config['min_samples']:
            return self._end_attempt(report)
        if self._duration() < self.config['min_duration_s']:
            report['reason'] = 'capture_duration_too_short'
            return self._end_attempt(report)
        if len(self._samples) >= self.config['max_samples_per_trial']:
            report['reason'] = 'capture_capacity_reached'
            return self._end_attempt(report)
        for hand in HANDS:
            delta = _trajectory(self._samples, hand, self._reference['R_OQ'])
            peak = float(np.max(np.linalg.norm(delta, axis=1)))
            return_error = float(np.linalg.norm(np.mean(delta[-10:], axis=0)))
            rotations = Rotation.from_quat([s['hands'][hand]['quaternion_xyzw'] for s in self._samples])
            wrist = float(np.max((rotations[0].inv() * rotations).magnitude()))
            step = float(np.max(np.linalg.norm(np.diff(delta, axis=0), axis=1)))
            report['hands'][hand] = dict(peak_m=peak, return_error_m=return_error,
                                         wrist_rotation_rad=wrist, max_step_m=step)
            # Cross-axis displacement is intentionally NOT rejected here.
            reason = ('movement_too_small' if peak < .06 else
                      'return_to_start' if return_error > max(.03, .2 * peak) else
                      'wrist_rotated_during_position_capture' if wrist > np.deg2rad(self.config['max_wrist_rotation_deg']) else
                      'position_jump' if step > .15 else None)
            if reason:
                report['reason'] = hand + ':' + reason
                return self._end_attempt(report)
        report.update(accepted=True, reason='accepted')
        return self._end_attempt(report)

    def snapshot(self):
        return dict(type='quest_intent_trajectories', version=1, origin='quest',
            created_at=self.created_at, configuration=copy.deepcopy(self.config),
            reference=copy.deepcopy(self._reference), status=self.status(),
            attempts=copy.deepcopy(self._attempts),
            invalidated_attempts=copy.deepcopy(self._invalidated_attempts),
            resumed_from=copy.deepcopy(self._resume_source))


def fit_intent_profile(capture):
    """Fit repeats 1..4; NEVER refit with repeat 5 or composite validation.

    Returns a rejected report for incomplete/inconsistent data. Thresholds are
    engineering defaults for review, not evidence of physical task success.
    """
    if capture.get('type') != 'quest_intent_trajectories' or capture.get('version') != 1:
        raise ValueError('Expected v2 trajectory capture, not static pose calibration')
    report = dict(type='quest_intent_profile', version=1, accepted=False,
                  origin=capture.get('origin'), capture_sha256=_digest(capture),
                  created_at=datetime.now(timezone.utc).isoformat(),
                  reference=copy.deepcopy(capture.get('reference')), hands={},
                  reason='incomplete_capture', model='normalized_bilateral_linear_basis_inverse',
                  split={'train_repeats': [1, 2, 3, 4], 'held_out_repeats': [5],
                         'training_labels': list(TRAIN_LABELS),
                         'validation_only_labels': list(VALIDATION_LABELS)})
    if report['origin'] not in ('quest', 'synthetic'): raise ValueError('invalid_capture_origin')
    trials = {label: sorted([a for a in capture['attempts'] if a.get('accepted')
                 and not a.get('superseded') and a['label'] == label],
                 key=lambda a: a['repeat_index']) for label in LABELS}
    if any([a['repeat_index'] for a in group] != [1, 2, 3, 4, 5] for group in trials.values()):
        return report
    rotation = np.asarray(capture['reference']['R_OQ'], float)
    if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
        raise ValueError('invalid_capture_rotation')
    failures = []
    for hand in HANDS:
        directions = []
        for axis, label in enumerate(AXES):
            repeat_directions = []
            # Left and right contribute equally to one +Y basis column.
            # Flip right displacement before PCA; normalize each whole repeat
            # so neither reach distance nor sample count gives it extra weight.
            for source_label, sign in ((('left', 1), ('right', -1)) if label == 'left' else ((label, 1),)):
                for trial in trials[source_label][:4]:
                    d = sign * _trajectory(trial['raw_samples'], hand, rotation)
                    _, _, vt = np.linalg.svd(d, full_matrices=False)
                    direction = vt[0] * (1 if vt[0, axis] >= 0 else -1)
                    repeat_directions.append(direction)
            direction = np.mean(repeat_directions, axis=0)
            directions.append(direction / np.linalg.norm(direction))
        basis = np.column_stack(directions)
        condition = float(np.linalg.cond(basis))
        if not np.isfinite(condition) or condition > 5 or np.linalg.det(basis) <= 0:
            report['reason'] = 'ambiguous_or_dependent_axes_' + hand
            return report
        correction = np.linalg.inv(basis)
        checks = []
        for axis, label in enumerate(LABELS):
            allowed, signs = LABEL_COMPONENTS[label]
            for trial in trials[label]:
                d = _trajectory(trial['raw_samples'], hand, rotation)
                corrected = d @ correction.T
                metrics = _metrics(corrected, allowed, signs)
                passed = (metrics['cross_axis_p95_m'] <= .025
                    and metrics['cross_axis_ratio'] <= .20
                    and min(metrics['intended_positive_extent_m']) >= .04)
                check = dict(label=label, repeat_index=trial['repeat_index'],
                    role='train' if label in TRAIN_LABELS and trial['repeat_index'] <= 4 else 'held_out',
                    passed=bool(passed), before=_metrics(d, allowed, signs), after=metrics)
                checks.append(check)
                if not passed: failures.append(hand + ':' + label + ':' + str(trial['repeat_index']))
        report['hands'][hand] = dict(basis=basis.tolist(), correction=correction.tolist(),
                                     condition=condition, checks=checks,
                lateral_review={label: {
                    'passed': all(c['passed'] for c in checks if c['label'] == label and c['role'] == 'held_out'),
                    'held_out_repeats': [c['repeat_index'] for c in checks if c['label'] == label and c['role'] == 'held_out'],
                    'max_cross_axis_p95_m': max(c['after']['cross_axis_p95_m'] for c in checks if c['label'] == label and c['role'] == 'held_out'),
                } for label in ('left', 'right', 'left_up', 'right_up')})
    report.update(accepted=not failures, reason='accepted' if not failures else 'trajectory_validation_failed',
                  failed_checks=failures)
    report['profile_id'] = _digest(report)[:16]
    return report


def load_intent_profile(path, *, real=False, setup_confirmed=False):
    path = Path(path)
    if path.stat().st_size > 5_000_000: raise ValueError('profile_too_large')
    profile = json.loads(path.read_text(encoding='utf-8'))
    matrices = validate_profile(profile)
    if real and profile['origin'] != 'quest': raise ValueError('synthetic_profile_cannot_control_real_robot')
    if not setup_confirmed:
        raise ValueError('Confirm the operator, headset facing direction and workspace setup before applying this profile')
    return profile, matrices


def validate_profile(profile):
    if (profile.get('type') != 'quest_intent_profile' or profile.get('version') != 1
            or profile.get('accepted') is not True or profile.get('origin') not in ('quest', 'synthetic')):
        raise ValueError('profile_not_accepted')
    identifier = profile.get('profile_id')
    unhashed = {k: v for k, v in profile.items() if k != 'profile_id'}
    if identifier != _digest(unhashed)[:16]: raise ValueError('profile_content_changed')
    matrices = {}
    for hand in HANDS:
        b = np.asarray(profile['hands'][hand]['basis'], float)
        c = np.asarray(profile['hands'][hand]['correction'], float)
        if (b.shape != (3, 3) or c.shape != (3, 3) or not np.isfinite(b).all()
                or not np.isfinite(c).all() or not np.allclose(np.linalg.norm(b, axis=0), 1.)
                or np.linalg.det(b) <= 0 or np.linalg.cond(b) > 5
                or not np.allclose(c @ b, np.eye(3), atol=1e-7)):
            raise ValueError('invalid_profile_basis_' + hand)
        matrices[hand] = c.copy()
    return matrices
