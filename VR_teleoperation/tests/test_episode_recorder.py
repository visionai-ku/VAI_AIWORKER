"""Dataset root handling for the LeRobot episode recorder (no ZMQ, no Qt window)."""
from pathlib import Path
import runpy
import shutil
import tempfile
import unittest

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
runpy.run_path(str(PROJECT_DIR / "package/init_project.py"))

from package.episode_recorder import (VREpisodeRecorder, build_features, dataset_status,
                                      list_episodes, rejected_episodes, ROOT_ABSENT, ROOT_EMPTY,
                                      ROOT_FOREIGN, ROOT_READY, ROOT_UNFINALIZED)
from package.worker.fixed_quest_protocol import JOINT_NAMES


CAMERA_SHAPES = {"head": (24, 32, 3)}


class DatasetRootTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.root = self.tmp / "pick_and_place"

    def recorder(self, root=None):
        return VREpisodeRecorder(robot_ip="127.0.0.1", repo_id="local/test",
                                 root=self.root if root is None else root, hz=10,
                                 cameras={"head": 5570})

    def write_episodes(self, count, frames=4):
        """A real dataset at self.root, written the way the recorder writes one."""
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset.create("local/test", 10, build_features(CAMERA_SHAPES),
                                        root=self.root, use_videos=True, vcodec="h264",
                                        video_backend="pyav")
        for _ in range(count):
            for _ in range(frames):
                dataset.add_frame({
                    "task": "unit test",
                    "observation.state": np.zeros(len(JOINT_NAMES), dtype=np.float32),
                    "action": np.zeros(len(JOINT_NAMES), dtype=np.float32),
                    "observation.images.head": np.zeros(CAMERA_SHAPES["head"], dtype=np.uint8)})
            dataset.save_episode()
        dataset.finalize()

    def test_missing_root_is_created(self):
        self.recorder()._open_dataset(CAMERA_SHAPES)
        self.assertTrue((self.root / "meta" / "info.json").exists())

    def test_empty_leftover_root_is_reused(self):
        # Deleting every episode leaves the directory behind, and LeRobotDataset.create()
        # refuses to write into a directory that already exists.
        self.root.mkdir(parents=True)
        recorder = self.recorder()
        recorder._open_dataset(CAMERA_SHAPES)
        self.assertTrue((self.root / "meta" / "info.json").exists())
        self.assertEqual(recorder.dataset.meta.total_episodes, 0)

    def test_existing_dataset_is_appended_to_not_recreated(self):
        self.write_episodes(2)
        recorder = self.recorder()
        recorder._open_dataset(CAMERA_SHAPES)
        self.assertEqual(recorder.dataset.meta.total_episodes, 2)
        self.assertEqual(len(list_episodes(self.root)), 2)

    def test_root_holding_other_files_is_refused_at_construction(self):
        self.root.mkdir(parents=True)
        (self.root / "episode_000").mkdir()          # the pre-LeRobot layout
        with self.assertRaises(FileExistsError) as caught:
            self.recorder()
        self.assertIn("episode_000", str(caught.exception))

    def unfinalized_root(self):
        """A dataset whose session died before finalize: info.json written, meta/episodes not."""
        self.write_episodes(1)
        shutil.rmtree(self.root / "meta" / "episodes")
        return self.root

    def test_dataset_status_classifies_each_root(self):
        self.assertEqual(dataset_status(self.root), ROOT_ABSENT)
        self.root.mkdir(parents=True)
        self.assertEqual(dataset_status(self.root), ROOT_EMPTY)
        (self.root / "stray.txt").write_text("x")
        self.assertEqual(dataset_status(self.root), ROOT_FOREIGN)
        shutil.rmtree(self.root)
        self.write_episodes(1)
        self.assertEqual(dataset_status(self.root), ROOT_READY)
        shutil.rmtree(self.root / "meta" / "episodes")
        self.assertEqual(dataset_status(self.root), ROOT_UNFINALIZED)

    def test_present_but_footerless_episode_metadata_counts_as_unfinalized(self):
        # The file exists, so an existence check would call the dataset ready and then blow up
        # with ArrowInvalid on the first read.
        self.write_episodes(1)
        target = next(iter((self.root / "meta" / "episodes").rglob("*.parquet")))
        target.write_bytes(b"PAR1" + b"\x00" * 64)        # footer 없는 파일
        self.assertEqual(dataset_status(self.root), ROOT_UNFINALIZED)
        self.assertEqual(list_episodes(self.root), [], "크래시 대신 빈 목록")
        with self.assertRaises(FileExistsError):
            self.recorder()

    def test_unfinalized_root_reports_locally_instead_of_hitting_the_hub(self):
        # LeRobotDataset() would fall back to downloading repo_id and raise a confusing
        # "Repository Not Found" 404 for what is purely a local problem.
        self.unfinalized_root()
        with self.assertRaises(FileExistsError) as caught:
            self.recorder()
        message = str(caught.exception)
        self.assertIn("finalize", message)
        self.assertIn("meta/episodes", message)
        self.assertNotIn("404", message)

    def test_list_episodes_counts_saved_episodes_only(self):
        self.assertEqual(list_episodes(self.root), [])
        self.write_episodes(3, frames=5)
        episodes = list_episodes(self.root)
        self.assertEqual([e["episode_index"] for e in episodes], [0, 1, 2])
        self.assertEqual([e["length"] for e in episodes], [5, 5, 5])


    def test_y_marks_and_unmarks_the_last_saved_episode(self):
        # A saved episode cannot be deleted in place (shared parquet/mp4 files), so Y records
        # a reversible mark that review_dataset.ipynb honours.
        self.write_episodes(2)
        recorder = self.recorder()
        recorder._open_dataset(CAMERA_SHAPES)
        recorder.last_saved_episode = 1

        recorder.discard_or_delete_last()
        self.assertEqual(rejected_episodes(self.root), [1])
        self.assertIn("폐기 표시", recorder.msg)

        recorder.discard_or_delete_last()                 # 같은 에피소드 -> 표시 해제
        self.assertEqual(rejected_episodes(self.root), [])
        self.assertIn("해제", recorder.msg)

    def test_y_while_recording_discards_the_running_episode_only(self):
        recorder = self.recorder()
        recorder._open_dataset(CAMERA_SHAPES)
        recorder.is_recording, recorder.last_saved_episode = True, 0
        recorder.discard_or_delete_last()
        self.assertFalse(recorder.recording)
        self.assertEqual(rejected_episodes(self.root), [], "저장된 에피소드를 건드리면 안 된다")

    def test_rejected_episodes_tolerates_a_missing_or_broken_file(self):
        self.assertEqual(rejected_episodes(self.root), [])
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "rejected_episodes.json").write_text("not json")
        self.assertEqual(rejected_episodes(self.root), [])

    def record_episode(self, recorder, frames=4):
        """Drive the recorder's own start/stop path (no ZMQ thread, frames fed directly)."""
        recorder.is_recording, recorder.ep_started_at, recorder.ep_frames = True, 0., 0
        for _ in range(frames):
            recorder.dataset.add_frame({
                "task": "unit test",
                "observation.state": np.zeros(len(JOINT_NAMES), dtype=np.float32),
                "action": np.zeros(len(JOINT_NAMES), dtype=np.float32),
                "observation.images.head": np.zeros(CAMERA_SHAPES["head"], dtype=np.uint8)})
            recorder.ep_frames += 1
        return recorder.stop()

    def test_each_saved_episode_survives_a_session_that_never_closes(self):
        """A killed kernel must not take finished episodes with it.

        LeRobotDataset keeps one parquet writer open across episodes, so stop() finalizes and
        re-opens per episode; otherwise a session that dies before close() leaves every
        episode of that session without a parquet footer (unreadable).
        """
        import pyarrow.parquet as pq
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        recorder = self.recorder()
        recorder._open_dataset(CAMERA_SHAPES)
        self.record_episode(recorder, frames=4)
        self.record_episode(recorder, frames=5)
        del recorder                      # 커널이 죽은 상황: close()/finalize() 없음

        for path in sorted(self.root.rglob("*.parquet")):
            pq.ParquetFile(path)          # footer 없으면 ArrowInvalid
        reloaded = LeRobotDataset("local/test", root=self.root, vcodec="h264",
                                  video_backend="pyav")
        self.assertEqual(reloaded.meta.total_episodes, 2)
        self.assertEqual([e["length"] for e in list_episodes(self.root)], [4, 5])
        self.assertEqual(len(reloaded), 9)

    def test_restarting_the_recorder_appends_to_the_existing_dataset(self):
        recorder = self.recorder()
        recorder._open_dataset(CAMERA_SHAPES)
        self.record_episode(recorder, frames=4)
        del recorder

        again = self.recorder()           # 노트북을 다시 실행한 상황
        again._open_dataset(CAMERA_SHAPES)
        self.assertEqual(again.dataset.meta.total_episodes, 1, "기존 에피소드를 이어받아야 한다")
        self.record_episode(again, frames=6)
        self.assertEqual([e["length"] for e in list_episodes(self.root)], [4, 6])

    def test_close_finalizes_even_when_stopping_raises(self):
        """An interrupt while the last episode is still encoding must not skip finalize.

        LeRobotDataset keeps one parquet writer open across episodes, so finalize() is what
        writes the footers: skipping it leaves every episode of the session unreadable.
        """
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        recorder = self.recorder()
        recorder._open_dataset(CAMERA_SHAPES)
        for _ in range(4):                       # one completed episode to protect
            recorder.dataset.add_frame({
                "task": "unit test",
                "observation.state": np.zeros(len(JOINT_NAMES), dtype=np.float32),
                "action": np.zeros(len(JOINT_NAMES), dtype=np.float32),
                "observation.images.head": np.zeros(CAMERA_SHAPES["head"], dtype=np.uint8)})
        recorder.dataset.save_episode()

        recorder.is_recording, recorder.ep_started_at = True, 0.
        def interrupted():
            raise KeyboardInterrupt("interrupted while encoding")
        recorder.stop = interrupted
        with self.assertRaises(KeyboardInterrupt):
            recorder.close()

        self.assertIn("finalized", recorder.msg)
        reloaded = LeRobotDataset("local/test", root=self.root, vcodec="h264", video_backend="pyav")
        self.assertEqual(reloaded.meta.total_episodes, 1)
        self.assertEqual(len(reloaded), 4)


if __name__ == "__main__":
    unittest.main()
