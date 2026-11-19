import json
import tempfile
import time
import unittest
from importlib.util import find_spec
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch

import numpy as np

from vai_robotics.ur5e.recording import EpisodeRecorder


FEATURES = {"action": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]}}
VIDEO = {"dtype": "video", "shape": (32, 32, 3), "names": ["height", "width", "channels"]}


class EpisodeRecorderTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "demo"

    def recorder(self, **kwargs):
        recorder = EpisodeRecorder(self.root, **kwargs)
        self.addCleanup(recorder.close)
        return recorder

    def wait_state(self, recorder, expected):
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            status = recorder.status
            if status["state"] == expected:
                return
            if status["state"] == "error":
                self.fail(status["message"])
            time.sleep(0.01)
        self.fail(f"Expected {expected}, got {recorder.status}")

    def test_queue_overflow_and_competing_writer_do_not_publish_partial_data(self):
        entered, release = Event(), Event()
        dataset = Mock()
        dataset.meta.total_episodes = 0

        def stall(_frame):
            entered.set()
            release.wait(timeout=10)

        dataset.add_frame.side_effect = stall
        with patch("vai_robotics.ur5e.recording._open_staged_dataset", return_value=dataset):
            first, second = self.recorder(), self.recorder()
            self.assertFalse(first.status["episodes_known"])
            self.assertTrue(first.status["can_start"])
            first.start(FEATURES)
            self.wait_state(first, "recording")
            self.assertTrue(first.status["episodes_known"])
            second.start(FEATURES)
            self.wait_state(second, "error")
            self.assertIn("already recording", second.status["message"])
            self.assertTrue(first.append({}))
            self.assertTrue(entered.wait(2))
            for _ in range(30):
                if not first.append({}):
                    break
            self.assertIn("could not keep up", first.status["message"])
            self.assertFalse(first.status["can_start"])
            release.set()
            first.close()
            self.assertFalse(self.root.exists())
            dataset.save_episode.assert_not_called()
            dataset.finalize.assert_called_once()

    @unittest.skipUnless(find_spec("lerobot"), "requires optional LeRobot recorder")
    def test_video_dataset_is_immediately_readable_and_resumes_across_sessions(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        camera = "observation.images.wrist"
        features = {**FEATURES, camera: VIDEO}
        for episode in range(2):
            recorder = self.recorder(task=f"task {episode}")
            self.assertTrue(recorder.start(features))
            self.wait_state(recorder, "recording")
            for frame in range(4):
                self.assertTrue(recorder.append({
                    "action": np.array([episode, frame], dtype=np.float32),
                    camera: np.full((32, 32, 3), 80 * episode + 20 * frame, dtype=np.uint8),
                }))
            self.assertTrue(recorder.save())
            recorder.close()  # Closing immediately must finish the requested save.
            self.assertEqual(recorder.status["state"], "idle", recorder.status)
            self.assertEqual(recorder.status["episodes"], episode + 1)
            dataset = LeRobotDataset("local/ur5e", root=self.root, video_backend="pyav")
            self.assertEqual(len(dataset), 4 * (episode + 1))
            row = dataset[len(dataset) - 1]
            self.assertEqual(row["task"], f"task {episode}")
            np.testing.assert_array_equal(row["action"].numpy(), [episode, 3])
            self.assertAlmostEqual(float(row[camera].mean()), (80 * episode + 60) / 255, delta=0.03)

    @unittest.skipUnless(find_spec("lerobot"), "requires optional LeRobot recorder")
    def test_discard_errors_and_orphan_shards_preserve_saved_dataset(self):
        recorder = self.recorder(fps=2, max_seconds=1)
        recorder.start(FEATURES)
        self.wait_state(recorder, "recording")
        frame = {"action": np.zeros(2, dtype=np.float32)}
        recorder.append(frame)
        recorder.append(frame)
        self.wait_state(recorder, "idle")  # duration limit saves automatically
        meta = self.root / "meta" / "info.json"
        original = meta.read_bytes()
        for operation in (recorder.discard, lambda: recorder.fail("missing feedback")):
            recorder.start(FEATURES)
            self.wait_state(recorder, "recording")
            self.assertTrue(recorder.append(frame))
            operation()
            self.wait_state(recorder, "idle" if operation == recorder.discard else "error")
            self.assertEqual(meta.read_bytes(), original)
        recorder.close()

        first_shard, = (self.root / "data").rglob("*.parquet")
        orphan = first_shard.with_name("file-001.parquet")
        orphan.write_bytes(b"keep me")
        recorder = self.recorder(fps=2)
        recorder.start(FEATURES)
        self.wait_state(recorder, "recording")
        recorder.append(frame)
        recorder.save()
        self.wait_state(recorder, "error")
        self.assertEqual(orphan.read_bytes(), b"keep me")
        self.assertEqual(meta.read_bytes(), original)

        invalid_info = json.loads(original)
        invalid_info["data_path"] = "../elsewhere/file-{file_index:03d}.parquet"
        meta.write_text(json.dumps(invalid_info))
        recorder = self.recorder(fps=2)
        recorder.start(FEATURES)
        self.wait_state(recorder, "error")
        self.assertIn("shard paths", recorder.status["message"])
