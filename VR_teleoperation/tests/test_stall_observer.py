"""Observe-only stall detector: window rules, entry/release evidence and resets."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from package.stall_observer import StallObserver

LEFT = list(range(3, 10))
RIGHT = list(range(10, 17))


class StallObserverTest(unittest.TestCase):
    def setUp(self):
        self.observer = StallObserver({"left": LEFT, "right": RIGHT})
        self.seq = 0
        self.t = 0.

    def feed(self, measured=None, previous=None, requested=None, sent=None, dt=.01, new=True, age=.005,
             tcp=None, tcp_request=None):
        q = np.zeros(19) if measured is None else np.asarray(measured, float)
        if new:
            self.seq += 1
            self.t += dt
        tcp = {"left": np.array([.5, .2, 1.]), "right": np.array([.5, -.2, 1.])} if tcp is None else tcp
        return self.observer.update(
            sample_key=(0, self.seq, self.seq), sample_time=self.t, sample_age=age,
            q_measured=q, q_previous=q if previous is None else previous,
            q_requested=q if requested is None else requested, q_sent=q if sent is None else sent,
            tcp_measured=tcp, tcp_requested={} if tcp_request is None else tcp_request, ik_target_limited=False)

    def leading(self, degrees, joint=LEFT[3]):
        q = np.zeros(19)
        q[joint] = np.deg2rad(degrees)
        return q

    def test_blocked_arm_with_leading_command_is_suspected(self):
        lead = self.leading(10.)
        states = [self.feed(previous=lead, requested=lead, sent=lead)["hands"]["left"]["state"] for _ in range(10)]
        self.assertIn("STALL_SUSPECTED", states)
        result = self.feed(previous=lead, requested=lead, sent=lead)
        self.assertEqual(result["hands"]["left"]["stall_joint"], LEFT[3])
        self.assertEqual(result["hands"]["right"]["state"], "FREE")

    def test_window_needs_full_time_span_not_sample_count(self):
        lead = self.leading(10.)
        # Five fresh 10 ms samples span only 40 ms of an 80 ms window.
        for _ in range(5):
            result = self.feed(previous=lead, requested=lead, sent=lead)
        self.assertFalse(result["window"]["valid"])
        self.assertEqual(result["hands"]["left"]["state"], "FREE")

    def test_repeated_sample_is_not_new_evidence(self):
        lead = self.leading(10.)
        self.feed(previous=lead, requested=lead, sent=lead)
        for _ in range(20):
            result = self.feed(previous=lead, requested=lead, sent=lead, new=False)
            self.assertFalse(result["new_sample"])
            self.assertIsNone(result["window"])
        self.assertEqual(result["hands"]["left"]["state"], "FREE")

    def test_sample_gap_or_stale_sample_invalidates_window(self):
        lead = self.leading(10.)
        for _ in range(5):
            self.feed(previous=lead, requested=lead, sent=lead, dt=.02)
        self.assertFalse(self.feed(previous=lead, requested=lead, sent=lead, dt=.05)["window"]["valid"])
        for _ in range(9):
            self.feed(previous=lead, requested=lead, sent=lead)
        self.assertFalse(self.feed(previous=lead, requested=lead, sent=lead, age=.05)["window"]["valid"])

    def test_slowly_moving_arm_is_not_a_stall(self):
        lead = self.leading(10.)
        for k in range(15):
            measured = self.leading(.6 * k)  # 60 deg/s: lagging but moving
            result = self.feed(measured=measured, previous=measured + lead, requested=measured + lead,
                               sent=measured + lead)
        self.assertEqual(result["hands"]["left"]["state"], "FREE")

    def test_backing_off_request_and_small_lead_are_not_stalls(self):
        lead, smaller = self.leading(10.), self.leading(9.)
        for _ in range(15):
            result = self.feed(previous=lead, requested=smaller, sent=lead)
        self.assertEqual(result["hands"]["left"]["state"], "FREE")
        self.setUp()
        small = self.leading(5.)
        for _ in range(15):
            result = self.feed(previous=small, requested=small, sent=small)
        self.assertEqual(result["hands"]["left"]["state"], "FREE")

    def test_release_needs_continuous_relaxed_evidence(self):
        lead = self.leading(10.)
        for _ in range(10):
            self.feed(previous=lead, requested=lead, sent=lead)
        near = {"left": np.array([.5, .2, 1.005]), "right": None}
        relaxed = dict(previous=self.leading(3.), requested=self.leading(3.), sent=self.leading(3.), tcp_request=near)
        for _ in range(10):  # 0.10 s < 0.15 s hold
            self.assertEqual(self.feed(**relaxed)["hands"]["left"]["state"], "STALL_SUSPECTED")
        self.feed(**relaxed, dt=.05)  # a gap restarts the release evidence
        # The window stays invalid until it again spans 80 ms, then the 150 ms hold restarts.
        for _ in range(8 + 14):
            self.assertEqual(self.feed(**relaxed)["hands"]["left"]["state"], "STALL_SUSPECTED")
        result = self.feed(**relaxed)
        self.assertEqual(result["hands"]["left"]["state"], "FREE")
        self.assertEqual(result["transitions"], [{"hand": "left", "to": "FREE", "reason": "would_release",
                                                  "held_s": result["transitions"][0]["held_s"]}])

    def test_reset_clears_state_and_window(self):
        lead = self.leading(10.)
        for _ in range(10):
            self.feed(previous=lead, requested=lead, sent=lead)
        self.observer.reset("stop")
        result = self.feed(previous=lead, requested=lead, sent=lead)
        self.assertEqual(result["hands"]["left"]["state"], "FREE")
        self.assertFalse(result["window"]["valid"])

    def test_invalid_parameters_are_rejected(self):
        for options in ({"enter_lead_rad": np.deg2rad(3.)}, {"window_s": 0.}, {"sample_gap_max_s": .1},
                        {"tcp_eps_m": float("nan")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                StallObserver({"left": LEFT}, **options)


if __name__ == "__main__":
    unittest.main()
