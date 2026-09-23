"""Load a trained LeRobot checkpoint and run one prediction on a synthetic
observation. No robot, camera or Quest connection; no dataset needed.

Verifies the inference path is wired correctly -- the checkpoint loads for its
own policy type (ACT now; others need no changes here), its pre/postprocessor
stats load from the checkpoint, and the predicted action comes back as a finite
(len(JOINT_NAMES),) vector. It does NOT verify the policy is any good: the
synthetic observation (mid-range joints, blank camera frames) is not a real
scene, so a sane-looking action here is not a promise of a sane action on the
real robot. Run this right after training, before ever pointing inference at
hardware.

    python scripts/check_policy_inference.py path/to/checkpoint
    python scripts/check_policy_inference.py path/to/checkpoint --device cpu
"""
from pathlib import Path
import argparse
import runpy

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "package/init_project.py"))

from package.episode_recorder import DEFAULT_CAMERAS
from package.policy_inference import PolicyRuntime, clip_to_joint_limits
from package.worker.fixed_quest_protocol import JOINT_LIMITS, JOINT_NAMES


def synthetic_observation(cameras, height=480, width=640):
    mid_q = np.array([(lo + hi) / 2. for lo, hi in (JOINT_LIMITS[name] for name in JOINT_NAMES)],
                     dtype=np.float32)
    frames = {cam: np.zeros((height, width, 3), dtype=np.uint8) for cam in cameras}
    return mid_q, frames


def check_policy_inference(checkpoint_path, device=None, cameras=None, height=480, width=640,
                           joint_names=None):
    cameras = list(DEFAULT_CAMERAS) if cameras is None else list(cameras)
    runtime = PolicyRuntime(checkpoint_path, device=device, joint_names=joint_names)
    runtime.reset()
    q, frames = synthetic_observation(cameras, height, width)
    raw = runtime.predict(q, frames)
    if raw.shape != (len(JOINT_NAMES),) or not np.all(np.isfinite(raw)):
        raise AssertionError("predicted action has the wrong shape or is not finite: %r" % (raw,))
    clipped, exceeded = clip_to_joint_limits(raw)

    print("PASS: %s policy loaded from %s, predicted one action for %d camera(s)"
          % (runtime.policy_type, checkpoint_path, len(cameras)))
    for name, value in zip(JOINT_NAMES, raw):
        flag = " CLIPPED -> %.4f" % clipped[JOINT_NAMES.index(name)] if name in exceeded else ""
        print("  %-16s %+.4f%s" % (name, value, flag))
    if exceeded:
        print("%d/%d joints exceeded JOINT_LIMITS on this synthetic input (expected: it is not a real scene)"
              % (len(exceeded), len(JOINT_NAMES)))
    return raw, clipped, exceeded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint_path", help="local directory or HF Hub repo id of a trained lerobot policy")
    parser.add_argument("--device", default=None,
                        help="cpu/cuda/mps; defaults to the checkpoint's own saved config.device")
    parser.add_argument("--cameras", nargs="*", default=None,
                        help="camera names; defaults to episode_recorder.DEFAULT_CAMERAS")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--exclude-joints", nargs="*", default=None,
                        help="JOINT_NAMES this checkpoint was NOT trained on (e.g. lift_joint "
                             "head_joint1 head_joint2 for an arm-only checkpoint); they are held at "
                             "the synthetic mid-range value instead of predicted")
    args = parser.parse_args()
    joint_names = ([n for n in JOINT_NAMES if n not in args.exclude_joints]
                   if args.exclude_joints else None)
    check_policy_inference(args.checkpoint_path, args.device, args.cameras, args.height, args.width,
                           joint_names)
