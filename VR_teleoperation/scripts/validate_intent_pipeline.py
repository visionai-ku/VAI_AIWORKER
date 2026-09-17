"""Offline synthetic calibration + MuJoCo FK/IK + in-process strict bridge.

No network sockets, renderer or physical robot. Run from any working directory.
"""
from pathlib import Path
import runpy
import json
from collections import Counter
import argparse
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / 'package/init_project.py'))
from package.intent_offline import synthetic_capture, synthetic_basis, controller_frame
from package.intent_calibration import fit_intent_profile, validate_profile, save_json
from package.intent_teleop import IntentTeleopSettings, build_models, build_planner, execution_diagnostics
from package.intent_sim_bridge import IntentSimBridge
from package.fixed_quest_real import SG2FixedQuestReal
from package.fixed_quest_sim import ControllerRecorder
from package.worker.fixed_quest_protocol import FIXED_QUEST_VR_BRIDGE_LIMITS
from utils import get_sg2_wrist_control_T


def plot_validation(rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    times = [r['t'] for r in rows]
    for hand, color in (('left', 'tab:blue'), ('right', 'tab:orange')):
        for axis, component in zip(axes, (0, 2)):
            axis.plot(times, [1000*r['by_hand'][hand]['measured_delta'][component] for r in rows], color=color, label=hand+' measured FK')
    for axis, component, title in zip(axes, (0, 2), ('Forward X', 'Up Z')):
        axis.plot(times, [1000*r['intent'][component] if r['t'] < 20 else float('nan') for r in rows], 'k--', label='labelled synthetic intent')
        axis.axvspan(20.5, 24, color='grey', alpha=.12, label='Initial joint pose return')
        axis.set_ylabel(title+' (mm)'); axis.legend(loc='upper left'); axis.grid(alpha=.3)
    axes[-1].set_xlabel('Virtual time (s)')
    fig.suptitle('Synthetic intent → existing IK → ideal SIM follower (not hardware)')
    fig.tight_layout(); fig.savefig(out/'trajectory.png', dpi=150); plt.close(fig)


def run_validation(output_dir=None):
    out = Path(output_dir or ROOT / '.cache/validation/intent')
    out.mkdir(parents=True, exist_ok=True)
    capture = synthetic_capture()
    profile = fit_intent_profile(capture)
    assert profile['accepted'], profile['reason']
    matrices = validate_profile(profile)
    cfg = IntentTeleopSettings(POSITION_SCALE=1., POSITION_SCALE_Y=1., POSITION_SCALE_Z=1.).validate()
    models = build_models(cfg)
    mapper, planner = build_planner(models, cfg, profile, matrices)
    bridge = IntentSimBridge(planner.q_command)
    real = SG2FixedQuestReal(planner, models['env_measured'], models['q_msg_layout'], bridge,
        dry_run=False, teleop_ramp_s=cfg.TELEOP_RAMP_S,
        max_arm_joint_lag=cfg.MAX_ARM_JOINT_LAG,
        required_bridge_limits=FIXED_QUEST_VR_BRIDGE_LIMITS,
        command_arm_speed_limit=True, tracking_pause_s=.1)
    initial = {h: get_sg2_wrist_control_T(models['env'], h)[:3, 3].copy() for h in ('left', 'right')}
    rows, states, rejected = [], Counter(), Counter()
    basis = synthetic_basis()
    recorder = ControllerRecorder(out / 'synthetic-sim.jsonl', planner.joint_names, 'intent_sim_offline',
        configuration={'origin': 'synthetic', 'intent_profile': profile, 'gain': [1, 1, 1],
                       'limitations': 'ideal feedback; no actuator dynamics, network or human intent evidence'})
    try:
        for i in range(1201):
            t, now = i * .02, 1. + i * .02
            intended = np.zeros(3)
            if 9 <= t < 12: intended[0] = .035 * np.sin(np.pi * (t - 9) / 3)**2
            elif 12 <= t < 15: intended[2] = .025 * np.sin(np.pi * (t - 12) / 3)**2
            elif 15 <= t < 18: intended[[0, 2]] = .02 * np.sin(np.pi * (t - 15) / 3)**2
            elif t >= 18: intended[0] = .025 * min(1., (t - 18) / 2)
            frame = controller_frame(i, now, basis @ intended, grip=float(.5 <= t < 20))
            if 11 <= t < 11.04:
                frame['hands']['left']['valid'] = False
            if t >= 20.5: frame = None  # HOME must not depend on Quest tracking/grips.
            result = real.tick_grip(frame, bridge.latest(now), now=now, action='home' if i == 1025 else None)
            execution_diagnostics(result, models['env'], models['env_measured'])
            states[result['state']] += 1
            if bridge.last_result and not bridge.last_result['accepted']:
                rejected[bridge.last_result['reason']] += 1
            row = {'t': t, 'intent': intended.tolist(), 'state': result['state'], 'reason': result['reason'],
                   'by_hand': {}}
            for hand in initial:
                stage = result.get('mapping_stages', {}).get(hand)
                row['by_hand'][hand] = dict(
                    measured_delta=(get_sg2_wrist_control_T(models['env_measured'], hand)[:3, 3] - initial[hand]).tolist(),
                    intent_delta=None if stage is None else np.asarray(stage['intent_delta_m']).tolist())
            rows.append(row)
            recorder.write(now, frame, result)
        faults = [(r['t'], r['state'], r['reason']) for r in rows if r['state'] in ('FAULT', 'HOLD')]
        max_up = {hand: max(r['by_hand'][hand]['measured_delta'][2] for r in rows if 12 <= r['t'] < 15)
                  for hand in initial}
        forward_height = {hand: max(abs(r['by_hand'][hand]['measured_delta'][2]) for r in rows if 9 <= r['t'] < 11)
                          for hand in initial}
        returned_error = float(np.max(np.abs(bridge.q - planner.q_home)))
        report = {'origin': 'synthetic', 'profile_accepted': profile['accepted'],
            'basis_inverse_max_error': max(float(np.max(np.abs(c @ basis - np.eye(3)))) for c in matrices.values()),
            'states': dict(states), 'faults': faults, 'bridge_rejections': dict(rejected),
            'forward_height_max_m': forward_height, 'intentional_up_peak_m': max_up,
            'home_joint_max_error': returned_error,
            'home_tolerance_ratio': float(np.max(np.abs(bridge.q - planner.q_home) / real.home_tolerance)),
            'limitations': 'Synthetic labelled trajectories and ideal joint follower; not a personal calibration or real robot result.'}
        report['passed'] = bool(not faults and not rejected and states['ACTIVE'] > 100
            and states['TRACKING_PAUSED'] >= 1 and states['HOME'] >= 1
            and min(max_up.values()) > .01 and report['home_tolerance_ratio'] <= 1
            and report['basis_inverse_max_error'] < .02)
        (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        (out / 'series.json').write_text(json.dumps(rows) + '\n')
        # Synthetic profile stays under cache, never personal profiles/.
        (out / 'synthetic-profile.json').write_text(json.dumps(profile, indent=2) + '\n')
        plot_validation(rows, out)
        return report
    finally:
        real.stop(); bridge.close(); recorder.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = run_validation(args.output)
    print(json.dumps(report, indent=2))
    if not report['passed']: raise SystemExit(1)
