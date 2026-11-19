"""Local, episode-at-a-time LeRobot recording outside the UR5e control loop."""

import ctypes
import fcntl
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from queue import Empty, Full, Queue
from tempfile import TemporaryDirectory
from threading import Lock, Thread


@contextmanager
def _dataset_lock(root):
    root.parent.mkdir(parents=True, exist_ok=True)
    # Keep the inode across sessions so competing writers always lock the same file.
    with root.with_name(f".{root.name}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Dataset is already recording: {root}") from None
        yield


def _publish(staged, root):
    """Publish a finalized dataset without a gap for readers of an existing root."""
    if not root.exists():
        os.rename(staged, root)
        return
    # A resumed LeRobot writer needs only metadata to create its new shards.
    # Link previous shards *after* finalize. A name collision aborts the take
    # without ever opening an original data or video file for writing.
    for name in ("data", "videos"):
        if (root / name).is_dir():
            shutil.copytree(root / name, staged / name, dirs_exist_ok=True, copy_function=os.link)
    # POSIX rename cannot replace a nonempty directory. Darwin and Linux both
    # support an atomic exchange on the same filesystem (our stage is a sibling).
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        result = libc.renamex_np(os.fsencode(staged), os.fsencode(root), 2)  # RENAME_SWAP
    else:
        result = libc.renameat2(-100, os.fsencode(staged), -100, os.fsencode(root), 2)  # RENAME_EXCHANGE
    if result != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno), root)


def _validate_existing(root, features, fps, robot_type):
    from lerobot.datasets.utils import DEFAULT_DATA_PATH, DEFAULT_VIDEO_PATH
    from lerobot.utils.constants import DEFAULT_FEATURES

    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise ValueError(f"Dataset directory is occupied: {root}")
    info = json.loads(info_path.read_text())
    if info.get("codebase_version") != "v3.0" or info["fps"] != fps or info.get("robot_type") != robot_type:
        raise ValueError(f"Incompatible dataset version, FPS or robot type in {root}")
    if info.get("data_path") != DEFAULT_DATA_PATH or info.get("video_path") not in (None, DEFAULT_VIDEO_PATH):
        raise ValueError(f"Incompatible dataset shard paths in {root}")
    stored = {k: v for k, v in info["features"].items() if k not in DEFAULT_FEATURES}
    if stored.keys() != features.keys() or any(
        (stored[k]["dtype"], tuple(stored[k]["shape"]), stored[k].get("names"))
        != (v["dtype"], tuple(v["shape"]), v.get("names"))
        for k, v in features.items()
    ):
        raise ValueError(f"Dataset features do not match this recording: {root}")
    if (not info.get("total_episodes")
            or not (root / "meta/tasks.parquet").is_file()
            or not (root / "meta/stats.json").is_file()
            or not any((root / "meta/episodes").rglob("*.parquet"))
            or not any((root / "data").rglob("*.parquet"))):
        raise ValueError(f"Dataset is incomplete: {root}")


def _open_staged_dataset(root, staged, features, fps, robot_type):
    try:
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise RuntimeError(f"Recording requires LeRobot 0.6.1 and its dependencies: {error}") from error

    repo_id = "local/ur5e"
    options = dict(root=staged, video_backend="pyav", rgb_encoder=RGBEncoderConfig(vcodec="h264"))
    if not root.exists() or not any(root.iterdir()):
        has_video = any(feature["dtype"] == "video" for feature in features.values())
        return LeRobotDataset.create(repo_id, fps, features, robot_type=robot_type, use_videos=has_video, **options)

    _validate_existing(root, features, fps, robot_type)
    shutil.copytree(root, staged, ignore=lambda src, names: {"data", "videos"} if Path(src) == root else set())
    return LeRobotDataset.resume(repo_id, **options)


def _verify_camera_frames(dataset, expected_frames):
    # LeRobot's PNG writer logs individual write failures. Verify each newly
    # encoded MP4 before publishing it alongside the action rows.
    if dataset.meta.video_keys:
        import av

        for camera in dataset.meta.video_keys:
            paths = list((dataset.root / "videos" / camera).rglob("*.mp4"))
            if len(paths) != 1:
                raise ValueError(f"Missing video for {camera}")
            with av.open(str(paths[0])) as video:
                count = video.streams.video[0].frames
                if count != expected_frames:
                    raise ValueError(f"{camera} video has {count} frames, expected {expected_frames}")


class EpisodeRecorder:
    """Only the worker touches LeRobot and disk. Save publishes a complete episode."""

    def __init__(self, root, *, fps=50, task="UR5e teleoperation", max_seconds=120):
        if not isinstance(fps, int) or fps <= 0 or max_seconds <= 0 or not task.strip():
            raise ValueError("Recording requires positive FPS, duration and a nonempty task")
        self.root = Path(root).expanduser().absolute()
        self.fps = fps
        self.task = task
        self._max_frames = max(1, int(max_seconds * fps))
        self._lock = Lock()
        self._thread = None
        self._state = "idle"
        self._message = ""
        self._frames = 0
        self._episodes = 0
        self._episodes_known = False
        self._closed = False

    @property
    def status(self):
        with self._lock:
            return dict(state=self._state, frames=self._frames, episodes=self._episodes,
                        can_start=not self._closed and self._thread is None, episodes_known=self._episodes_known,
                        root=str(self.root), task=self.task, message=self._message)

    @property
    def recording(self):
        with self._lock:
            return self._state == "recording"

    def start(self, features, *, robot_type="ur5e"):
        with self._lock:
            if self._closed or self._thread is not None:
                return False
            self._queue = Queue(maxsize=16)
            self._frames = 0
            self._state, self._message = "preparing", "Opening dataset"
            self._thread = Thread(target=self._run, args=(features, robot_type), name="UR5e recording", daemon=False)
            self._thread.start()
            return True

    def append(self, frame):
        """Transfer ownership of a fresh observation/action snapshot, without waiting."""
        with self._lock:
            if self._state != "recording":
                return False
            try:
                self._queue.put_nowait(frame)
            except Full:
                self._state, self._message = "error", "Recording aborted: writer could not keep up"
                return False
            self._frames += 1
            if self._frames >= self._max_frames:
                self._state, self._message = "saving", "Episode duration reached"
            return True

    def save(self):
        with self._lock:
            if self._state != "recording":
                return False
            if not self._frames:
                self._message = "Waiting for the first frame"
                return False
            self._state, self._message = "saving", "Finalizing episode"
            return True

    def discard(self):
        with self._lock:
            if self._state not in ("preparing", "recording"):
                return False
            self._state, self._message = "discarding", "Discarding episode"
            return True

    def fail(self, message):
        with self._lock:
            if self._state in ("preparing", "recording"):
                self._state, self._message = "error", str(message)

    def close(self):
        """Discard an uncommitted take and wait for an explicitly requested save."""
        with self._lock:
            self._closed = True
            if self._state in ("preparing", "recording"):
                self._state, self._message = "discarding", "Discarding unfinished episode"
            thread = self._thread
        if thread is not None:
            thread.join()

    def _consume(self, dataset):
        with self._lock:
            self._episodes = dataset.meta.total_episodes
            self._episodes_known = True
            if self._state != "preparing":
                return False
            self._state, self._message = "recording", ""
        while True:
            with self._lock:
                if self._state not in ("recording", "saving"):
                    return False
                if self._state == "saving" and self._queue.empty():
                    break
            try:
                frame = self._queue.get(timeout=0.05)
            except Empty:
                continue
            dataset.add_frame({**frame, "task": self.task})
        dataset.save_episode(parallel_encoding=False)
        return True

    def _run(self, features, robot_type):
        try:
            root = self.root.resolve()
            with _dataset_lock(root), TemporaryDirectory(prefix=f".{root.name}.take-", dir=root.parent) as temp:
                staged = Path(temp) / "dataset"
                dataset = _open_staged_dataset(root, staged, features, self.fps, robot_type)
                try:
                    save = self._consume(dataset)
                finally:
                    dataset.finalize()
                if save:
                    _verify_camera_frames(dataset, self._frames)
                    _publish(staged, root)
                    with self._lock:
                        self._episodes = dataset.meta.total_episodes
                        self._message = f"Saved episode {self._episodes - 1}"
        except Exception as error:
            with self._lock:
                self._state, self._message = "error", str(error)
        finally:
            with self._lock:
                if self._state == "discarding":
                    self._frames, self._message = 0, "Episode discarded"
                if self._state != "error":
                    self._state = "idle"
                self._queue = None
                self._thread = None
