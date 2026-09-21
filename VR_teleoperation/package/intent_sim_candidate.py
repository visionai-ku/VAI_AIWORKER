"""Failed candidate integrity checks; REAL additionally requires a bound SIM review."""
import json
from pathlib import Path
import numpy as np
from .intent_calibration import _digest, fit_intent_profile, HANDS


def _load_failed_candidate(profile_path, capture_path, *, setup_confirmed=False):
    if not setup_confirmed:
        raise ValueError('Confirm operator/headset/workspace setup before testing a SIM candidate')
    if capture_path is None:
        raise ValueError('SIM candidate requires its original CAPTURE_PATH')
    profile = json.loads(Path(profile_path).read_text(encoding='utf-8'))
    capture = json.loads(Path(capture_path).read_text(encoding='utf-8'))
    if (profile.get('type') != 'quest_intent_profile' or profile.get('version') != 1
            or profile.get('accepted') is not False or profile.get('reason') != 'trajectory_validation_failed'):
        raise ValueError('Only a complete trajectory-validation-failed candidate can use this SIM path')
    if _digest({k:v for k,v in profile.items() if k!='profile_id'})[:16] != profile.get('profile_id'):
        raise ValueError('Candidate profile content changed')
    if _digest(capture) != profile.get('capture_sha256'):
        raise ValueError('Candidate capture/profile mismatch')
    fitted = fit_intent_profile(capture)
    for key in ('model','origin','reference','split','reason','accepted','failed_checks'):
        if profile.get(key) != fitted.get(key):
            raise ValueError('Candidate does not match recomputed fit: ' + key)
    matrices = {}
    for hand in HANDS:
        candidate = profile.get('hands',{}).get(hand,{})
        expected = fitted.get('hands',{}).get(hand,{})
        b = np.asarray(candidate.get('basis'),float)
        c = np.asarray(candidate.get('correction'),float)
        if (b.shape != (3,3) or c.shape != (3,3) or not np.isfinite(b).all() or not np.isfinite(c).all()
                or np.linalg.det(b) <= 0 or np.linalg.cond(b) > 5
                or not np.allclose(c@b,np.eye(3),atol=1e-7)
                or not np.allclose(b,np.asarray(expected.get('basis'),float),atol=1e-10)
                or not np.allclose(c,np.asarray(expected.get('correction'),float),atol=1e-10)):
            raise ValueError('Invalid candidate basis/correction: '+hand)
        matrices[hand] = c.copy()
    return profile, matrices


def load_sim_candidate(profile_path, capture_path, *, setup_confirmed=False):
    return _load_failed_candidate(profile_path, capture_path, setup_confirmed=setup_confirmed)


def load_reviewed_real_candidate(profile_path, capture_path, review_path, *, setup_confirmed=False):
    review = json.loads(Path(review_path).read_text(encoding='utf-8'))
    if (review.get('type') != 'intent_operator_sim_review' or review.get('version') != 1
            or review.get('decision') != 'allow_real_trial' or not review.get('operator_statement')):
        raise ValueError('Explicit operator SIM review required for this REAL candidate')
    profile, matrices = _load_failed_candidate(profile_path, capture_path, setup_confirmed=setup_confirmed)
    if profile['origin'] != 'quest':
        raise ValueError('Synthetic candidate cannot control real robot')
    if (review.get('profile_sha256') != _digest(profile)
            or review.get('capture_sha256') != profile['capture_sha256']):
        raise ValueError('REAL review does not match this candidate')
    return profile, matrices
