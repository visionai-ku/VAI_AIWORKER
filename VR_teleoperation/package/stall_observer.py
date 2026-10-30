"""Observe-only arm stall detector for teleoperation (P0 of the contact-lead plan).

It reads measured joints, commands and requested targets and reports, per arm,
whether the arm looks stalled while its command keeps leading. It never changes
a command, a limit or a control state; claude-codex.md tracks the later steps
that would act on it.

Evidence comes only from new robot samples (state seq/stamp progress), and every
timer advances on robot source time, not on PC ticks. A window counts only when
its samples span the whole window without a gap, so a repeated or late sample is
never read as a stall. A break in continuity keeps the state but restarts the
release evidence.
"""
from __future__ import annotations

from collections import deque

import numpy as np

STALL_MODES = ("off", "observe")


class StallObserver:
    def __init__(self, arm_indices, *, enter_lead_rad=np.deg2rad(6.), stall_lead_rad=np.deg2rad(4.),
                 window_s=.08, stall_eps_rad=np.deg2rad(.3), tcp_eps_m=.003,
                 sample_gap_max_s=.03, sample_age_max_s=.03, release_hold_s=.15, release_tcp_m=.01):
        values = dict(enter_lead_rad=enter_lead_rad, stall_lead_rad=stall_lead_rad, window_s=window_s,
                      stall_eps_rad=stall_eps_rad, tcp_eps_m=tcp_eps_m, sample_gap_max_s=sample_gap_max_s,
                      sample_age_max_s=sample_age_max_s, release_hold_s=release_hold_s, release_tcp_m=release_tcp_m)
        for name, value in values.items():
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
                raise ValueError(name + " must be a positive finite number")
        if enter_lead_rad <= stall_lead_rad:
            raise ValueError("enter_lead_rad must exceed stall_lead_rad")
        if sample_gap_max_s >= window_s:
            raise ValueError("sample_gap_max_s must be shorter than window_s")
        self.arm_indices = {hand: list(indices) for hand, indices in arm_indices.items()}
        for name, value in values.items():
            setattr(self, name, float(value))
        self.reset("init")

    @property
    def parameters(self):
        return {name: getattr(self, name) for name in (
            "enter_lead_rad", "stall_lead_rad", "window_s", "stall_eps_rad", "tcp_eps_m",
            "sample_gap_max_s", "sample_age_max_s", "release_hold_s", "release_tcp_m")}

    def reset(self, reason):
        self._last_key = None
        self._samples = deque()
        self._hands = {hand: {"state": "FREE", "since": None, "joint": None, "release_from": None}
                       for hand in self.arm_indices}
        self.last_reset_reason = reason

    def _window(self, sample_age):
        """Samples covering [latest - window, latest], keeping one boundary sample."""
        latest = self._samples[-1][0]
        start = latest - self.window_s
        while len(self._samples) > 1 and self._samples[1][0] <= start:
            self._samples.popleft()
        times = [sample[0] for sample in self._samples]
        gaps = np.diff(times) if len(times) > 1 else np.array([np.inf])
        info = {"samples": len(times), "span_s": latest - times[0],
                "max_gap_s": float(np.max(gaps)), "age_s": sample_age}
        info["valid"] = bool(len(times) > 1 and times[0] <= start + 1e-9
                             and info["max_gap_s"] <= self.sample_gap_max_s
                             and sample_age is not None and sample_age <= self.sample_age_max_s)
        return info

    def update(self, *, sample_key, sample_time, sample_age, q_measured, q_previous, q_requested, q_sent,
               tcp_measured, tcp_requested, ik_target_limited):
        """Feed one applicable control tick; returns a log-sized summary.

        q_requested is the rate-limited candidate before pacing; q_sent is the
        command actually transmitted this tick. tcp_requested is the operator's
        final TCP target (None when a hand has no target).
        """
        q_measured, q_previous = np.asarray(q_measured, float), np.asarray(q_previous, float)
        q_requested, q_sent = np.asarray(q_requested, float), np.asarray(q_sent, float)
        new_sample = sample_key != self._last_key
        if new_sample:
            if self._samples and sample_time <= self._samples[-1][0]:
                # Robot time must advance with seq; treat a regression as a broken window.
                self._samples.clear()
            self._last_key = sample_key
            self._samples.append((float(sample_time), q_measured.copy(),
                                  {hand: np.asarray(p, float).copy() for hand, p in tcp_measured.items()}))
            window = self._window(sample_age)
        else:
            window = None
        transitions = []
        hands = {}
        for hand, indices in self.arm_indices.items():
            status = self._hands[hand]
            lead_sent = (q_sent - q_measured)[indices]
            lead_previous = (q_previous - q_measured)[indices]
            lead_requested = (q_requested - q_measured)[indices]
            request = tcp_requested.get(hand)
            tcp_demand = (None if request is None
                          else float(np.linalg.norm(np.asarray(request, float) - tcp_measured[hand])))
            summary = {
                "lead_sent_max_deg": float(np.rad2deg(np.max(np.abs(lead_sent)))),
                "lead_joint": int(indices[int(np.argmax(np.abs(lead_sent)))]),
                "lead_requested_max_deg": float(np.rad2deg(np.max(np.abs(lead_requested)))),
                "tcp_demand_m": tcp_demand,
                "tcp_demand_z_m": None if request is None else float(request[2] - tcp_measured[hand][2]),
                "ik_target_limited": bool(ik_target_limited),
            }
            if window is not None and window["valid"]:
                joints = np.array([sample[1][indices] for sample in self._samples])
                motion = np.max(joints, axis=0) - np.min(joints, axis=0)
                tcps = np.array([sample[2][hand] for sample in self._samples])
                tcp_move = float(np.max(np.linalg.norm(tcps - tcps[-1], axis=1)))
                summary.update(meas_range_max_deg=float(np.rad2deg(np.max(motion))), tcp_move_m=tcp_move)
                if status["state"] == "FREE":
                    # Lagging joint, a request that does not back off, and no measured motion.
                    candidates = [k for k in range(len(indices))
                                  if abs(lead_sent[k]) >= self.enter_lead_rad
                                  and abs(lead_requested[k]) >= abs(lead_previous[k]) - 1e-9
                                  and motion[k] < self.stall_eps_rad]
                    if candidates and tcp_move < self.tcp_eps_m:
                        worst = max(candidates, key=lambda k: abs(lead_sent[k]))
                        status.update(state="STALL_SUSPECTED", since=sample_time, joint=int(indices[worst]),
                                      release_from=None)
                        transitions.append({"hand": hand, "to": "STALL_SUSPECTED", "joint": int(indices[worst]),
                                            "lead_deg": float(np.rad2deg(abs(lead_sent[worst])))})
                else:
                    # Provisional release (finalized in P2): command back within the
                    # stall lead and the operator target close to the measured TCP.
                    relaxed = (np.max(np.abs(lead_sent)) <= self.stall_lead_rad
                               and tcp_demand is not None and tcp_demand <= self.release_tcp_m)
                    if not relaxed:
                        status["release_from"] = None
                    elif status["release_from"] is None:
                        status["release_from"] = sample_time
                    elif sample_time - status["release_from"] >= self.release_hold_s:
                        transitions.append({"hand": hand, "to": "FREE", "reason": "would_release",
                                            "held_s": float(sample_time - status["since"])})
                        status.update(state="FREE", since=sample_time, joint=None, release_from=None)
            elif window is not None:
                # Broken continuity: keep the state, restart the release evidence.
                status["release_from"] = None
            summary.update(state=status["state"], stall_joint=status["joint"],
                           stall_s=None if status["since"] is None or status["state"] == "FREE"
                           else float(sample_time - status["since"]))
            hands[hand] = summary
        return {"applicable": True, "new_sample": new_sample, "window": window,
                "hands": hands, "transitions": transitions}
