"""Compute Box protocol and two-button control without a physical gripper."""

import threading
import unittest
from unittest.mock import Mock, call, patch

from vai_robotics.ur5e.rg2 import RG2


class RG2Test(unittest.TestCase):
    def setUp(self):
        self.clock = self.enterContext(patch("vai_robotics.ur5e.rg2.time.monotonic", return_value=10.0))
        self.gripper = RG2("compute-box")
        self.client = self.gripper._client = Mock()
        self.feedback = self.client.read_holding_registers.return_value
        self.feedback.isError.return_value = False
        self.feedback.registers = [506, 0]
        self.client.write_registers.return_value.isError.return_value = False
        self.gripper._poll()

    def press(self):
        self.gripper.update(False)
        self.gripper.update(True)

    def test_recorded_target_follows_acknowledged_close_stop_and_open(self):
        self.press()
        self.assertEqual(self.gripper.recording_widths(), (50.6, 50.6))
        self.gripper._poll()
        self.assertEqual(self.gripper.recording_widths(), (50.6, 0))
        self.gripper.update(False)
        self.feedback.registers = [420, 1]
        self.gripper._poll()  # Busy, Compute Box acknowledged stop before reversing.
        self.assertEqual(self.gripper.recording_widths(), (42, 42))
        self.feedback.registers = [416, 0]
        self.gripper._poll()
        self.assertEqual(self.gripper.recording_widths(), (41.6, 75))
        self.clock.return_value += 1
        self.assertEqual(self.gripper.recording_widths(), (None, None))

    def test_registers_startup_busy_reversal_and_deduplication(self):
        grip = self.gripper
        self.client.read_holding_registers.assert_called_once_with(267, count=2, device_id=65)
        self.assertEqual(grip.width_mm, 50.6)
        grip.update(True)  # Chord was already down when LIVE became eligible.
        grip._poll()
        self.client.write_registers.assert_not_called()
        self.press()
        grip._poll()
        grip.update(True)
        self.feedback.registers = [400, 1]
        grip._poll()
        grip.update(False)
        grip._poll()
        grip._poll()  # A busy owned move receives only one stop.
        grip.set_open_width(20)  # The pending release already captured 75 mm.
        self.feedback.registers = [350, 0]
        grip._poll()
        grip._poll()
        self.assertEqual(self.client.write_registers.call_args_list, [
            call(0, [400, 0, 1], device_id=65), call(2, [8], device_id=65),
            call(0, [400, 750, 1], device_id=65),
        ])

    def test_custom_force_is_written_in_decinewtons(self):
        self.gripper = RG2("compute-box", force_n=20.0, open_width_mm=110)
        self.gripper._client = self.client
        self.gripper._poll()
        self.press()
        self.gripper._poll()
        self.gripper.update(False)
        self.gripper._poll()
        self.assertEqual(self.client.write_registers.call_args_list, [
            call(0, [200, 0, 1], device_id=65), call(0, [200, 1100, 1], device_id=65),
        ])

    def test_open_stop_feedback_preserves_width_and_release(self):
        self.assertEqual(self.gripper.open_width_mm, 75)
        self.feedback.registers = [1106, 0]  # Observed full-open encoder value.
        self.gripper._poll()
        self.assertEqual(self.gripper.width_mm, 110.6)
        self.assertIn("110.6 mm", self.gripper.status)
        self.press()
        self.gripper._poll()
        self.gripper.update(False)
        self.gripper._poll()
        self.assertEqual(self.client.write_registers.call_args_list, [
            call(0, [400, 0, 1], device_id=65), call(0, [400, 750, 1], device_id=65),
        ])

    def test_setting_open_width_only_affects_future_open_commands(self):
        self.gripper.set_open_width(60.44)
        self.assertEqual(self.gripper.open_width_mm, 60.4)
        self.gripper._poll()
        self.client.write_registers.assert_not_called()
        self.press()
        self.gripper._poll()
        self.gripper.update(False)
        self.gripper._poll()
        self.client.write_registers.assert_called_with(0, [400, 604, 1], device_id=65)
        self.gripper.set_open_width(25)
        self.feedback.registers = [310, 1]  # Opening to 60.4 mm is still in progress.
        self.gripper._poll()
        self.feedback.registers = [604, 0]
        self.gripper._poll()
        self.assertEqual(self.client.write_registers.call_count, 2)
        self.press()
        self.gripper._poll()
        self.gripper.update(False)
        self.gripper._poll()
        self.client.write_registers.assert_called_with(0, [400, 250, 1], device_id=65)
        for invalid in (-0.1, 110.6, float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.gripper.set_open_width(invalid)
        self.assertEqual(self.gripper.open_width_mm, 25.0)

    def test_grossly_invalid_width_never_dispatches_pending_close(self):
        self.press()
        self.feedback.registers = [0xffff, 0]
        with self.assertRaisesRegex(OSError, "Invalid RG2 actual width"):
            self.gripper._poll()
        self.client.write_registers.assert_not_called()

    def test_never_preempts_external_motion_or_replays_canceled_close(self):
        self.feedback.registers = [560, 1]
        self.press()
        self.gripper._poll()
        self.gripper.update(False)
        self.feedback.registers = [470, 0]
        self.gripper._poll()
        self.client.write_registers.assert_not_called()

    def test_input_loss_timeout_and_safety_switches_require_a_fresh_edge(self):
        grip = self.gripper
        for fault in ("missing", "timeout", "safety"):
            with self.subTest(fault=fault):
                self.press()
                if fault == "missing":
                    grip.update(None)
                if fault == "timeout":
                    self.clock.return_value += 0.18
                if fault == "safety":
                    self.feedback.registers = [509, 8]
                grip._poll()
                if fault == "safety":
                    self.assertIn("안전", grip.status)
                self.feedback.registers = [509, 0]
                grip.update(True)
                grip._poll()
                self.client.write_registers.assert_not_called()
        self.press()
        grip._poll()
        self.client.write_registers.assert_called_once()
        self.clock.return_value += 0.5
        self.assertIsNone(grip.width_mm)

    def test_release_during_unrelated_read_cancels_before_write_gate(self):
        self.press()
        reading, allow_read = threading.Event(), threading.Event()

        def delayed_feedback(*args, **kwargs):
            reading.set()
            allow_read.wait(timeout=2)
            return self.feedback

        self.client.read_holding_registers.side_effect = delayed_feedback
        worker = threading.Thread(target=self.gripper._poll)
        worker.start()
        try:
            self.assertTrue(reading.wait(timeout=1))
            self.gripper.update(None)
            self.gripper.wait_for_writes()
        finally:
            allow_read.set()
            worker.join(timeout=1)
        self.assertFalse(worker.is_alive())
        self.client.write_registers.assert_not_called()

    def test_mode_barrier_waits_for_committed_write_but_updates_never_wait(self):
        self.press()
        writing, acknowledge, drained = threading.Event(), threading.Event(), threading.Event()

        def delayed_ack(*args, **kwargs):
            writing.set()
            acknowledge.wait(timeout=2)
            return Mock(isError=lambda: False)

        def drain():
            self.gripper.wait_for_writes()
            drained.set()

        self.client.write_registers.side_effect = delayed_ack
        writer = threading.Thread(target=self.gripper._poll)
        barrier = threading.Thread(target=drain)
        writer.start()
        try:
            self.assertTrue(writing.wait(timeout=1))
            self.gripper.update(None)
            barrier.start()
            self.assertFalse(drained.wait(timeout=0.03))
        finally:
            acknowledge.set()
            writer.join(timeout=1)
            if barrier.ident is not None:
                barrier.join(timeout=1)
        self.assertTrue(drained.is_set())
        self.gripper.update(True)
        self.gripper._poll()
        self.client.write_registers.assert_called_once()
