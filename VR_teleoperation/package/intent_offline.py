"""Deterministic fixtures and offline checks; never a personal calibration source."""
import numpy as np
from .controller_mapping import FACING_R_BQ
from .intent_calibration import IntentTrajectoryCapture, LABELS


def synthetic_basis():
    basis = np.array([[1., .05, -.03], [.06, 1., .04], [.18, .10, 1.]])
    return basis / np.linalg.norm(basis, axis=0)


def controller_frame(seq, now, delta=None, grip=1., valid=True, origin_id=0):
    delta = np.zeros(3) if delta is None else np.asarray(delta, float)
    return dict(version=1, session_id='v2-synthetic', origin_id=origin_id, seq=seq,
        sample_time_ms=now * 1000, received_at=now, visibility='visible',
        head={'position_m': [0., 1.6, 0.], 'quaternion_xyzw': [0., 0., 0., 1.]},
        hands={hand: dict(position_m=(np.array([x, 1.1, -.6]) + FACING_R_BQ.T @ delta).tolist(),
            quaternion_xyzw=[0., 0., 0., 1.], valid=valid, emulated=False,
            grip=grip, trigger=0.) for hand, x in (('left', .2), ('right', -.2))})


def synthetic_capture(seed=19, basis=None):
    """Known oblique axes with fresh noise on each of twenty whole repeats."""
    rng = np.random.default_rng(seed)
    basis = synthetic_basis() if basis is None else np.asarray(basis, float)
    capture = IntentTrajectoryCapture()
    directions = {'forward': [1,0,0], 'left': [0,1,0], 'up': [0,0,1],
                  'right': [0,-1,0]}
    seq = 0
    for label in LABELS:
        direction = np.asarray(directions[label], float)
        for repeat in range(5):
            capture.begin_trial(label)
            amplitude = .18 * rng.uniform(.9, 1.1)
            for i in range(181):
                phase = np.clip((i / 180. - .10) / .8, 0, 1)
                progress = np.sin(np.pi * phase) ** 2
                delta = basis @ (direction * amplitude * progress) + rng.normal(0, .00035, 3)
                now = 1. + seq / 30.
                capture.add_frame(controller_frame(seq, now, delta, grip=0.), now=now)
                seq += 1
            result = capture.finish_trial()
            if not result['accepted']: raise AssertionError(result['reason'])
    snapshot = capture.snapshot()
    snapshot['origin'] = 'synthetic'
    return snapshot
