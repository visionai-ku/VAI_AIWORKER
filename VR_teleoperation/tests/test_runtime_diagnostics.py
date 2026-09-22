"""Timing probes and in-memory Quest ingestion; no sockets or robot commands."""
import asyncio
import gc
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from package.runtime_diagnostics import GCPauseMonitor
from package.quest_controller_server import QuestControllerServer
from test_controller_mapping import frame


class RuntimeDiagnosticsTest(unittest.TestCase):
    def test_monitor_is_bounded_drains_and_unregisters_without_changing_gc(self):
        before = list(gc.callbacks)
        enabled, thresholds = gc.isenabled(), gc.get_threshold()
        monitor = GCPauseMonitor().start().start()
        try:
            self.assertEqual(gc.callbacks, before + [monitor._callback])
            gc.collect(0)
            self.assertTrue(any(event["generation"] == 0 for event in monitor.take_events()))
            monitor.close()  # Drive the bounded buffer without automatic GC interleaving.
            clock = iter([1., 1.025] * 40)
            with patch("package.runtime_diagnostics.time", SimpleNamespace(monotonic=lambda: next(clock))):
                for _ in range(40):
                    monitor._callback("start", {"generation": 2})
                    monitor._callback("stop", {"generation": 2})
            events = monitor.take_events()
            self.assertEqual(len(events), 32)
            self.assertEqual(events[0]["generation"], 2)
            self.assertAlmostEqual(events[0]["elapsed_ms"], 25.)
            self.assertEqual(monitor.take_events(), [])
        finally:
            monitor.close()
            monitor.close()
        self.assertEqual(gc.callbacks, before)
        self.assertEqual(gc.isenabled(), enabled)
        self.assertEqual(gc.get_threshold(), thresholds)

    def test_receiver_preserves_sample_time_and_records_trusted_ingress_gaps(self):
        server = QuestControllerServer(host="127.0.0.1")
        samples = [frame(1, 100.), frame(2, 100.02), frame(3, 100.04, origin_id=1)]
        for sample in samples:
            sample["received_at"] = -999
            sample["transport"] = {"receive_gap_ms": -999}
        snapshots = []

        class Socket:
            def __aiter__(self):
                self.index = 0
                return self

            async def __anext__(self):
                if self.index:
                    snapshots.append(server.latest())
                if self.index == len(samples):
                    raise StopAsyncIteration
                raw = json.dumps(samples[self.index])
                self.index += 1
                return raw

        clock = iter([100., 100.005, 100.25, 100.251, 100.26, 100.262])
        with patch("package.quest_controller_server.time", SimpleNamespace(monotonic=lambda: next(clock))):
            asyncio.run(server._receive(Socket()))
        first, delayed, changed = snapshots
        self.assertEqual(first["received_at"], 100.)
        self.assertAlmostEqual(first["transport"]["decode_ms"], 5.)
        self.assertIsNone(first["transport"]["receive_gap_ms"])
        self.assertEqual(delayed["sample_time_ms"], samples[1]["sample_time_ms"])
        self.assertAlmostEqual(delayed["transport"]["receive_gap_ms"], 250.)
        self.assertAlmostEqual(delayed["transport"]["sample_gap_ms"], 20.)
        self.assertEqual(delayed["transport"]["sequence_gap"], 1)
        self.assertEqual(delayed["transport"]["receive_stall_count"], 1)
        self.assertIsNone(changed["transport"]["receive_gap_ms"])
        self.assertEqual(changed["transport"]["last_receive_stall"]["seq"], 2)
        self.assertEqual(changed["transport"]["receive_stall_count"], 1)
        self.assertIsNone(server.latest())  # disconnect still invalidates input


if __name__ == "__main__":
    unittest.main()
