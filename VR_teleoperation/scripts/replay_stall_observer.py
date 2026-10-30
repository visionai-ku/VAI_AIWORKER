"""Replay the observe-only stall detector (package/stall_observer.py) over a teleop log.

Reads an existing `.cache/fixed_quest/recordings/*-intent-real.jsonl` and feeds every
live ACTIVE tick to StallObserver, the same way SG2FixedQuestReal does in
STALL_LIMIT_MODE="observe", then prints how often and where it would suspect a stall.

    python scripts/replay_stall_observer.py .cache/fixed_quest/recordings/<log>.jsonl
    python scripts/replay_stall_observer.py <log>.jsonl --enter-lead-deg 8 --eps-deg .5

Approximations, because older logs do not record everything the live hook sees:
- the rate-limited candidate before pacing is not logged; the sent command stands in
  for it, so the "request does not back off" condition is only approximate;
- sample time is the robot source stamp (source_stamp_ns), sample age is feedback_age_s.
This estimates detection tendencies on recorded motion. It cannot show how the robot
would have reacted to a command change, nor measure force.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from package.stall_observer import StallObserver  # noqa: E402

ARM_INDICES = {"left": list(range(3, 10)), "right": list(range(10, 17))}


def replay(path, observer):
    previous = None
    episodes, open_episodes = [], {}
    active_ticks = invalid_windows = new_samples = 0
    t0 = None
    with open(path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue  # a log still being written may end mid-line
            if d.get("type") == "metadata":
                continue
            r = d["result"]
            t0 = d["recorded_at"] if t0 is None else t0
            diag = r.get("feedback_diagnostics") or {}
            applicable = (r.get("dry_run") is False and r.get("armed") and r.get("transmitted")
                          and r.get("state") == "ACTIVE" and r.get("q_command") is not None
                          and r.get("robot_q") is not None and diag.get("source_stamp_ns") is not None)
            if not applicable:
                for hand, episode in open_episodes.items():
                    episode["end"] = "reset:" + str(r.get("state"))
                    episodes.append(episode)
                open_episodes = {}
                observer.reset("not_applicable")
                previous = None
                continue
            active_ticks += 1
            sent = np.asarray(r["q_command"], float)
            stages, execution = r.get("mapping_stages") or {}, r.get("execution_stages") or {}
            out = observer.update(
                sample_key=(diag.get("state_seq"), diag["source_stamp_ns"]),
                sample_time=diag["source_stamp_ns"] / 1e9, sample_age=r.get("feedback_age_s"),
                q_measured=np.asarray(r["robot_q"], float),
                q_previous=sent if previous is None else previous, q_requested=sent, q_sent=sent,
                tcp_measured={hand: np.asarray(execution[hand]["measured_eef_m"], float) for hand in ARM_INDICES},
                tcp_requested={hand: stage.get("bounded_eef_m") for hand, stage in stages.items()},
                ik_target_limited=r.get("ik_target_limited", False))
            previous = sent
            if out["new_sample"]:
                new_samples += 1
                invalid_windows += not out["window"]["valid"]
            for change in out["transitions"]:
                hand = change["hand"]
                z = execution[hand]["measured_eef_m"][2]
                if change["to"] == "STALL_SUSPECTED":
                    open_episodes[hand] = dict(hand=hand, start_s=d["recorded_at"] - t0, joint=change["joint"],
                                               lead_deg=change["lead_deg"], z_m=z, max_lead_deg=change["lead_deg"])
                elif hand in open_episodes:
                    episode = open_episodes.pop(hand)
                    episode.update(end=change["reason"], duration_s=change["held_s"])
                    episodes.append(episode)
            for hand, episode in open_episodes.items():
                episode["max_lead_deg"] = max(episode["max_lead_deg"], out["hands"][hand]["lead_sent_max_deg"])
                episode["duration_s"] = d["recorded_at"] - t0 - episode["start_s"]
    episodes += list(open_episodes.values())
    return dict(active_ticks=active_ticks, new_samples=new_samples, invalid_windows=invalid_windows,
                episodes=episodes)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=Path)
    parser.add_argument("--enter-lead-deg", type=float, default=6.)
    parser.add_argument("--stall-lead-deg", type=float, default=4.)
    parser.add_argument("--window-s", type=float, default=.08)
    parser.add_argument("--eps-deg", type=float, default=.3)
    parser.add_argument("--tcp-eps-m", type=float, default=.003)
    parser.add_argument("--gap-max-s", type=float, default=.03)
    parser.add_argument("--age-max-s", type=float, default=.03)
    args = parser.parse_args()
    observer = StallObserver(ARM_INDICES, enter_lead_rad=np.deg2rad(args.enter_lead_deg),
                             stall_lead_rad=np.deg2rad(args.stall_lead_deg), window_s=args.window_s,
                             stall_eps_rad=np.deg2rad(args.eps_deg), tcp_eps_m=args.tcp_eps_m,
                             sample_gap_max_s=args.gap_max_s, sample_age_max_s=args.age_max_s)
    summary = replay(args.log, observer)
    episodes = summary["episodes"]
    print("live ACTIVE ticks: %d (%.0f s) | new robot samples: %d | invalid windows: %.1f%%" % (
        summary["active_ticks"], summary["active_ticks"] / 100, summary["new_samples"],
        100 * summary["invalid_windows"] / max(1, summary["new_samples"])))
    print("suspected stall episodes: %d, total %.1f s" % (len(episodes), sum(e.get("duration_s", 0) for e in episodes)))
    if not episodes:
        return
    names = ["lift_joint", "head_joint1", "head_joint2"] + ["arm_l_joint%d" % i for i in range(1, 8)] \
        + ["arm_r_joint%d" % i for i in range(1, 8)] + ["gripper_l_joint1", "gripper_r_joint1"]
    joints, ends = {}, {}
    for e in episodes:
        joints[names[e["joint"]]] = joints.get(names[e["joint"]], 0) + 1
        ends[e.get("end", "open")] = ends.get(e.get("end", "open"), 0) + 1
    z = np.array([e["z_m"] for e in episodes]); durations = np.array([e.get("duration_s", 0) for e in episodes])
    print("  by hand:", {h: sum(e["hand"] == h for e in episodes) for h in ARM_INDICES})
    print("  entry joint:", dict(sorted(joints.items(), key=lambda kv: -kv[1])))
    print("  ended by:", ends)
    print("  wrist z at entry m: p10 %.3f p50 %.3f p90 %.3f" % tuple(np.percentile(z, [10, 50, 90])))
    print("  duration s: p50 %.2f p90 %.2f max %.2f | max lead deg p50 %.1f max %.1f" % (
        np.median(durations), np.percentile(durations, 90), durations.max(),
        np.median([e["max_lead_deg"] for e in episodes]), max(e["max_lead_deg"] for e in episodes)))


if __name__ == "__main__":
    main()
