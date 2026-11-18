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
        self.feedback.registers = [350, 0]
        grip._poll()
        grip._poll()
        self.assertEqual(self.client.write_registers.call_args_list, [
            call(0, [200, 0, 1], device_id=65), call(2, [8], device_id=65),
            call(0, [200, 1100, 1], device_id=65),
        ])

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
