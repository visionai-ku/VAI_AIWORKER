"""
Episode recorder for SG2 VR teleop (strict JSON workers), writing directly to a
LeRobot v3.0 dataset (https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3)
via the `lerobot` package's LeRobotDataset.

Runs next to run_intent_teleop() without changing it:
  - a background thread receives ZMQ messages and calls dataset.add_frame() /
    save_episode() (video encoding never blocks the control loop)
  - a small Qt window (Record: STOP / REC / DEL) driven by a QTimer, which fires inside the
    teleop loop's app.processEvents()

Sources (all ZMQ SUB, extra subscribers do not affect the teleop):
  robot state   tcp://<robot>:5560   outbound.py (strict JSON "sg2_state", or legacy pickle)
  cameras       tcp://<robot>:5570/5571/5572   camera_outbound.py / outbound.py --meta (pickle)
  teleop action tcp://127.0.0.1:5561 this PC's own command publisher (strict JSON "sg2_command")

The LeRobot dataset schema (action / observation.state / observation.images.<cam>) is fixed
for the whole dataset root at creation time, so recording cannot start until at least one
frame has arrived from every camera in `cameras` -- that is what fixes each camera's
resolution. Pass a smaller `cameras` dict (e.g. {"head": 5570}) if some cameras are not wired
up. `action`/`observation.state` follow JOINT_NAMES order; camera frames are converted from
the wire's BGR to RGB (the convention LeRobot's own camera backends use). With
record_eef=True the schema also has observation.eef_pose / action.eef_pose (package/eef_pose.py:
FK of the measured / commanded joints, per hand xyz + quat wxyz in base_link).

An existing `root` (one with meta/info.json) is resumed and appended to with
LeRobotDataset.resume(), matching lerobot-record --resume; a fresh `root` is created with
LeRobotDataset.create(). (Since lerobot 0.5 the plain LeRobotDataset(...) constructor is
read-only -- only create()/resume() return a dataset that accepts add_frame().) The default
`vcodec="h264"` targets a software libx264 encoder available on stock ffmpeg builds; the
LeRobotDataset default ("libsvtav1") needs an ffmpeg built with libsvtav1, which is not
guaranteed to be installed. It is passed to lerobot as RGBEncoderConfig(vcodec=...), whose
other defaults (yuv420p, g=2, crf=30) match what lerobot 0.4's `vcodec=` argument used.
"""
import json
import pickle
import platform
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import zmq
from PyQt5.QtCore import QTimer

from ri_motion_v5_package.qt import MultiRadioQtWidget
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from package.worker.fixed_quest_protocol import JOINT_NAMES, parse_joint_groups


DEFAULT_CAMERAS = {"head": 5570, "wrist_left": 5571, "wrist_right": 5572}
ACTION_FRESH_S = 0.2
DEFAULT_TASK = "SG2 VR teleoperation"
DEFAULT_VCODEC = "h264"
REJECTED_PATH = "rejected_episodes.json"


def find_lan_ip(robot_ip="192.168.6.2"):
    """
    This PC's IP on the robot's subnet, checking every interface (Wi-Fi en0, USB/Thunderbolt
    Ethernet en5/en7/..., Linux eth/wlan). get_ip_address() only looks at en0.
    """
    ips = []
    if platform.system() == "Darwin":
        ifaces = subprocess.run(["ifconfig", "-l"], capture_output=True, text=True).stdout.split()
        for iface in ifaces:
            r = subprocess.run(["ipconfig", "getifaddr", iface], capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip():
                ips.append(r.stdout.strip())
    else:
        ips = subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()
    prefix = robot_ip.rsplit(".", 1)[0] + "."
    same_subnet = [ip for ip in ips if ip.startswith(prefix)]
    if same_subnet:
        return same_subnet[0]
    raise RuntimeError("로봇(%s)과 같은 네트워크(%s*)의 IP가 없습니다. 연결을 확인하세요. 찾은 IP: %s"
                       % (robot_ip, prefix, ips or "없음"))


def _sub(ctx, endpoint, conflate=True):
    s = ctx.socket(zmq.SUB)
    if conflate:
        s.setsockopt(zmq.CONFLATE, 1)
    s.setsockopt(zmq.SUBSCRIBE, b"")
    s.setsockopt(zmq.LINGER, 0)
    s.connect(endpoint)
    return s


def _recv_latest(sock):
    """Drain the socket and return the newest raw message (or None)."""
    last = None
    while True:
        try:
            last = sock.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return last


def decode_message(raw):
    """Strict workers send JSON, legacy ones pickle."""
    if raw is None:
        return None
    if raw[:1] == b"{":
        return json.loads(raw.decode("utf-8"))
    return pickle.loads(raw)


def rejected_episodes(root):
    """Episode indices the operator marked bad while recording (the Y button).

    A saved episode cannot be removed in place -- LeRobot v3 packs many of them into shared
    parquet/mp4 files -- so Y records a mark here instead. It is reversible (pressing Y again
    clears it) and review_dataset.ipynb drops the marked episodes when it builds the final
    dataset with delete_episodes().
    """
    path = Path(root) / REJECTED_PATH
    if not path.exists():
        return []
    try:
        return sorted({int(index) for index in json.loads(path.read_text())})
    except (TypeError, ValueError):
        return []


def parse_robot_state(msg):
    """-> q in JOINT_NAMES order, or None when the state is not ready / incomplete."""
    if not isinstance(msg, dict) or msg.get("ready") is False:
        return None
    joints = msg.get("joints") or {}
    by_name = {}
    for grp, st in joints.items():
        for name, pos in zip(st["name"], st["position"]):
            by_name[name] = float(pos)
    if not all(name in by_name for name in JOINT_NAMES):
        return None
    return np.asarray([by_name[name] for name in JOINT_NAMES], dtype=np.float64)


def parse_command(msg):
    """-> (q in JOINT_NAMES order, mode, motion) or None."""
    if not isinstance(msg, dict) or msg.get("type") != "sg2_command":
        return None
    try:
        q = np.asarray(parse_joint_groups(msg.get("joints")), dtype=np.float64)
    except (ValueError, TypeError, KeyError):
        q = np.full(len(JOINT_NAMES), np.nan)
    return q, str(msg.get("mode")), str(msg.get("motion", "teleop"))


def decode_frame(msg):
    """-> HWC BGR uint8 (OpenCV convention; callers convert to RGB for LeRobot)."""
    img = cv2.imdecode(np.frombuffer(msg["data"], dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img[:, :img.shape[1] // 2, :] if msg.get("stereo", True) else img


ROOT_ABSENT, ROOT_EMPTY, ROOT_READY, ROOT_UNFINALIZED, ROOT_FOREIGN = (
    "absent", "empty", "ready", "unfinalized", "foreign")


def readable_episode_files(root):
    """meta/episodes parquet files that actually have a footer, oldest first.

    A file written by a session that died is present but unreadable, so existence alone is
    not enough: pyarrow raises ArrowInvalid("Parquet magic bytes not found in footer").
    ParquetFile() only reads the footer, so this stays cheap.
    """
    files = []
    for path in sorted((Path(root) / "meta" / "episodes").rglob("*.parquet")):
        try:
            pq.ParquetFile(path)
        except Exception:
            continue
        files.append(path)
    return files


def dataset_status(root):
    """How `root` can be used, decided from local files only (never hits the Hub).

    LeRobotDataset() falls back to downloading `repo_id` when local metadata is incomplete,
    so an interrupted session surfaces as a confusing "Repository Not Found" 404 instead of
    the local problem it is.
    """
    root = Path(root)
    if not root.exists():
        return ROOT_ABSENT
    if not any(root.iterdir()):
        return ROOT_EMPTY
    if not (root / "meta" / "info.json").exists():
        return ROOT_FOREIGN
    if readable_episode_files(root):
        return ROOT_READY
    return ROOT_UNFINALIZED


def _root_error(root, status):
    root = Path(root)
    if status == ROOT_FOREIGN:
        return FileExistsError(
            "%s 는 LeRobot 데이터셋이 아닌데 (meta/info.json 없음) 내용이 있습니다: %s. "
            "다른 이름을 쓰거나, 이 디렉터리를 비우거나 옮기세요."
            % (root, ", ".join(sorted(p.name for p in list(root.iterdir())[:5]))))
    videos = len(list((root / "videos").rglob("*.mp4")))
    return FileExistsError(
        "%s 는 finalize 되지 않은 데이터셋입니다 (meta/episodes 없음). 이전 세션이 finalize 전에 "
        "끝나서 data/*.parquet 에 footer가 없고 관절·action 을 읽을 수 없습니다"
        "%s. 이 디렉터리를 옮기거나 지운 뒤 새 이름으로 시작하세요."
        % (root, " (videos/*.mp4 %d개는 온전할 수 있습니다)" % videos if videos else ""))


def list_episodes(root):
    """Saved episode records of the LeRobot dataset at `root`, oldest first.

    Empty when `root` holds no dataset yet, so `len(list_episodes(root))` still answers "how
    many episodes so far" the way it did when each episode was its own directory. Each record
    is the dataset's own episode row (episode_index, length, tasks, ...). Reads
    meta/episodes/*.parquet directly, so counting never opens a second writer on the dataset.
    LeRobotDataset buffers episode metadata, so an episode shows up here once save_episode()
    has flushed it -- VREpisodeRecorder.close() (finalize) guarantees that.
    """
    root = Path(root)
    if not (root / "meta" / "info.json").exists():
        return []
    files = readable_episode_files(root)
    if not files:
        return []
    rows = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return rows.sort_values("episode_index").to_dict("records")


EEF_OBS_KEY, EEF_ACTION_KEY = "observation.eef_pose", "action.eef_pose"


def build_features(camera_shapes, eef_names=None):
    """LeRobot features dict: action / observation.state (JOINT_NAMES order) + one
    observation.images.<cam> video feature per camera, shape (h, w, c). With `eef_names`,
    also observation.eef_pose / action.eef_pose (FK of the measured / commanded joints)."""
    features = {
        "action": {"dtype": "float32", "shape": (len(JOINT_NAMES),), "names": list(JOINT_NAMES)},
        "observation.state": {"dtype": "float32", "shape": (len(JOINT_NAMES),), "names": list(JOINT_NAMES)},
    }
    if eef_names:
        for key in (EEF_OBS_KEY, EEF_ACTION_KEY):
            features[key] = {"dtype": "float32", "shape": (len(eef_names),), "names": list(eef_names)}
    for cam, shape in camera_shapes.items():
        features["observation.images.%s" % cam] = {
            "dtype": "video", "shape": tuple(shape), "names": ["height", "width", "channels"],
        }
    return features


class VREpisodeRecorder:
    def __init__(self, robot_ip, repo_id, root, hz=30, max_sec=120, cameras=None,
                 state_port=5560, command_endpoint="tcp://127.0.0.1:5561",
                 task=DEFAULT_TASK, vcodec=DEFAULT_VCODEC, record_eef=False):
        self.robot_ip = robot_ip
        self.eef_names, self._eef_obs, self._eef_action = None, None, None
        if record_eef:
            from package.eef_pose import SG2EEFPose, EEF_POSE_NAMES
            # Separate instances: each keeps its own quaternion-sign continuity.
            self.eef_names = list(EEF_POSE_NAMES)
            self._eef_obs, self._eef_action = SG2EEFPose(), SG2EEFPose()
        self.repo_id, self.root = repo_id, Path(root)
        self.hz, self.max_sec = int(hz), float(max_sec)
        self.cameras = dict(DEFAULT_CAMERAS if cameras is None else cameras)
        self.state_endpoint = "tcp://%s:%d" % (robot_ip, state_port)
        self.command_endpoint = command_endpoint
        self.task, self.vcodec = str(task), vcodec
        # Fail here rather than inside the receive thread: `root` either holds a finalized
        # LeRobot dataset to append to, or must be free for LeRobotDataset.create().
        status = dataset_status(self.root)
        if status in (ROOT_FOREIGN, ROOT_UNFINALIZED):
            raise _root_error(self.root, status)

        self.lock = threading.Lock()
        self.dataset = None                          # created once every camera has a frame
        self.is_recording, self.ep_started_at, self.ep_frames = False, None, 0
        self.last_saved_episode, self.msg = None, "카메라 대기 중"
        self.rates = {"joint": 0.0, "action": 0.0, **{c: 0.0 for c in self.cameras}}
        self.error = None
        self._stop_evt = threading.Event()
        self._thread = None

    # ---------- control (main / Qt thread) ----------
    @property
    def ready(self):
        return self.dataset is not None

    @property
    def recording(self):
        return self.is_recording

    def start_thread(self):
        self._thread = threading.Thread(target=self._run, name="vr-episode-recorder", daemon=True)
        self._thread.start()
        return self

    def start(self):
        with self.lock:
            if self.dataset is None:
                self.msg = "녹화 불가: 카메라 프레임 대기 중 (%d/%d)" % (0, len(self.cameras))
            elif not self.is_recording:
                if self.eef_names:
                    self._eef_obs.reset()
                    self._eef_action.reset()
                self.is_recording, self.ep_started_at, self.ep_frames = True, time.time(), 0
                self.msg = "recording episode_%06d" % self.dataset.meta.total_episodes
        print("[rec]", self.msg)

    def stop(self):
        with self.lock:
            if not self.is_recording:
                return None
            self.is_recording = False
            episode_index, n_frames = self.dataset.meta.total_episodes, self.ep_frames
            duration = time.time() - self.ep_started_at
            dataset = self.dataset
        # Video encoding happens here; kept outside the lock so status() stays responsive.
        dataset.save_episode()
        # Durability: LeRobotDataset keeps its parquet writers open across episodes, so a
        # session that dies before finalize() leaves *every* episode it recorded unreadable
        # (no footer). Finalize each episode instead, then re-open: a finalized-but-reused
        # object would truncate meta/episodes on its next flush, while a fresh load recomputes
        # the chunk/file indices from disk and rolls to new files. Only the episode being
        # recorded can now be lost.
        dataset.finalize()
        reopened = self._resume_dataset()
        meta = {"episode_index": episode_index, "duration_sec": duration, "n_frames": n_frames}
        with self.lock:
            self.dataset = reopened
            self.last_saved_episode = episode_index
            self.msg = "saved episode_%06d (%.1fs, %d frames) · 디스크 기록 완료" % (
                episode_index, duration, n_frames)
        print("[rec]", self.msg)
        return meta

    def discard_or_delete_last(self):
        """Throw away the running episode, else toggle the reject mark on the last saved one.

        A saved episode cannot be deleted in place (see rejected_episodes), so it is marked
        instead; pressing again clears the mark.
        """
        with self.lock:
            if self.is_recording:
                self.is_recording, dataset, target = False, self.dataset, None
            else:
                dataset, target = None, self.last_saved_episode
        if dataset is not None:
            dataset.clear_episode_buffer()
            message = "discarded (recording)"
        elif target is None:
            message = "삭제할 녹화 없음"
        else:
            marks = set(rejected_episodes(self.root))
            if target in marks:
                marks.discard(target)
                message = "episode_%06d 폐기 표시 해제" % target
            else:
                marks.add(target)
                message = "episode_%06d 폐기 표시 · 최종 데이터셋에서 제외됩니다" % target
            (self.root / REJECTED_PATH).write_text(json.dumps(sorted(marks)))
        with self.lock:
            self.msg = message
        print("[rec]", message)

    def status(self):
        hz = " ".join("%s %.0f" % (k, v) for k, v in self.rates.items())
        if self.error:
            return "REC ERROR: %s" % self.error
        n_done = self.dataset.meta.total_episodes if self.dataset is not None else 0
        if self.is_recording:
            return "● REC episode_%06d %.1fs | saved %d | Hz %s" % (
                n_done, time.time() - self.ep_started_at, n_done, hz)
        return "IDLE | saved %d | Hz %s | %s" % (n_done, hz, self.msg)

    def close(self):
        """Stop recording and finalize the dataset.

        finalize() is what writes the parquet footers and flushes meta/episodes; until it
        runs, every episode saved in this session is unreadable (LeRobotDataset keeps one
        writer open across episodes by design). So it runs in a finally, and a
        KeyboardInterrupt while the last episode is still encoding video cannot skip it --
        interrupting again during "finalizing" is what loses a session's joint data.
        """
        try:
            with self.lock:
                was_recording = self.is_recording
            if was_recording:
                self.stop()
        finally:
            self._stop_evt.set()
            if self._thread is not None:
                self._thread.join(timeout=5.0)
            if self.dataset is not None:
                with self.lock:
                    self.msg = "finalizing (중단하지 마세요)"
                print("[rec]", self.msg)
                self.dataset.finalize()
                with self.lock:
                    self.msg = "finalized · %d episodes in %s" % (
                        self.dataset.meta.total_episodes, self.root)
                print("[rec]", self.msg)

    # ---------- receive / write (background thread) ----------
    def _build_frame(self, state_q, action_q, frames_bgr):
        frame = {
            "task": self.task,
            "observation.state": np.asarray(state_q, dtype=np.float32),
            "action": np.asarray(action_q, dtype=np.float32),
        }
        if self.eef_names:
            frame[EEF_OBS_KEY] = self._eef_obs(state_q)
            frame[EEF_ACTION_KEY] = self._eef_action(action_q)
        for cam, bgr in frames_bgr.items():
            frame["observation.images.%s" % cam] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return frame

    def _resume_dataset(self):
        """The finalized dataset at `root`, opened for appending (write mode)."""
        return LeRobotDataset.resume(self.repo_id, root=self.root,
                                     rgb_encoder=RGBEncoderConfig(vcodec=self.vcodec),
                                     image_writer_threads=4, video_backend="pyav")

    def _open_dataset(self, camera_shapes):
        # video_backend="pyav": this machine's torchcodec build (LeRobotDataset's own
        # default) fails to load its ffmpeg shared libraries; pyav works against the
        # system ffmpeg. Pass the same video_backend when loading this dataset later
        # (e.g. for training) instead of relying on the default.
        status = dataset_status(self.root)
        if status in (ROOT_FOREIGN, ROOT_UNFINALIZED):
            raise _root_error(self.root, status)
        if status == ROOT_READY:
            self.dataset = self._resume_dataset()
            has_eef = EEF_OBS_KEY in self.dataset.meta.features
            if has_eef != bool(self.eef_names):
                self.dataset = None
                raise ValueError(
                    "%s 는 EEF pose %s 데이터셋입니다. record_eef=%s 로는 이어서 녹화할 수 없으니 "
                    "다른 이름(root)을 쓰세요." % (self.root, "포함" if has_eef else "없는", bool(self.eef_names)))
        else:
            # LeRobotDataset.create() creates `root` itself and refuses an existing directory,
            # so a leftover empty one (deleting every episode leaves it behind) has to go first.
            if status == ROOT_EMPTY:
                self.root.rmdir()
            self.dataset = LeRobotDataset.create(
                self.repo_id, self.hz, build_features(camera_shapes, self.eef_names), root=self.root,
                use_videos=True, rgb_encoder=RGBEncoderConfig(vcodec=self.vcodec),
                image_writer_threads=4, video_backend="pyav")
        self.msg = "ready (%d cameras, %d saved)" % (len(camera_shapes), self.dataset.meta.total_episodes)
        print("[rec]", self.msg)

    def _run(self):
        ctx = zmq.Context()
        state_sock = _sub(ctx, self.state_endpoint)
        cmd_sock = _sub(ctx, self.command_endpoint, conflate=False)
        cam_socks = {c: _sub(ctx, "tcp://%s:%d" % (self.robot_ip, p)) for c, p in self.cameras.items()}
        last_state, last_cmd, last_cmd_t, last_frame = None, None, -np.inf, {}
        counts = {k: 0 for k in self.rates}
        t_rate, dt, t_next = time.time(), 1.0 / self.hz, time.time()
        try:
            while not self._stop_evt.is_set():
                now = time.time()
                if now < t_next:
                    time.sleep(min(t_next - now, 2e-3))
                    continue
                t_next = max(t_next + dt, now)

                state = parse_robot_state(decode_message(_recv_latest(state_sock)))
                if state is not None:
                    last_state = state
                cmd_raw = _recv_latest(cmd_sock)
                if cmd_raw is not None:
                    parsed = parse_command(decode_message(cmd_raw))
                    if parsed is not None:
                        last_cmd, last_cmd_t = parsed, now
                for cam, sock in cam_socks.items():
                    raw = _recv_latest(sock)
                    if raw is None:
                        continue
                    bgr = decode_frame(pickle.loads(raw))
                    if bgr is not None:
                        last_frame[cam] = bgr

                if self.dataset is None and len(last_frame) == len(self.cameras):
                    self._open_dataset({cam: img.shape for cam, img in last_frame.items()})

                valid = last_cmd is not None and last_cmd[1] == "command" and now - last_cmd_t <= ACTION_FRESH_S
                ready = (self.dataset is not None and last_state is not None
                         and len(last_frame) == len(self.cameras))
                with self.lock:
                    recording, dataset = self.is_recording, self.dataset
                    if recording and ready:
                        action_q = last_cmd[0] if last_cmd is not None else np.full(len(JOINT_NAMES), np.nan)
                        dataset.add_frame(self._build_frame(last_state, action_q, last_frame))
                        self.ep_frames += 1
                    timeout = recording and self.ep_started_at is not None and now - self.ep_started_at > self.max_sec

                counts["joint"] += state is not None
                counts["action"] += bool(valid)
                for cam in last_frame:
                    counts[cam] += 1
                if now - t_rate >= 1.0:
                    self.rates = {k: v / (now - t_rate) for k, v in counts.items()}
                    counts = {k: 0 for k in counts}
                    t_rate = now
                if timeout:
                    self.stop()
                    self.msg = "timeout → " + self.msg
        except Exception as exc:
            self.error = repr(exc)
            print("[rec] recorder thread stopped:", self.error)
        finally:
            for s in [state_sock, cmd_sock, *cam_socks.values()]:
                s.close(linger=0)
            ctx.term()


class RecordControl:
    """Qt window + QTimer. Keep a reference to this object while the teleop runs."""

    def __init__(self, recorder, period_ms=100, x_offset=0.0, y_offset=0.72, width=0.28, height=0.12):
        self.recorder = recorder
        self.widget = MultiRadioQtWidget(
            title="SG2 Record", x_offset=x_offset, y_offset=y_offset,
            window_width=width, window_height=height,
            label_texts=["Record"], option_texts=[["STOP", "REC", "DEL"]], initial_texts=["STOP"])
        self.prev = "STOP"
        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(period_ms)

    def tick(self):
        opt = self.widget.get_button_option(label_text="Record")
        if opt != self.prev:
            if opt == "REC":
                self.recorder.start()
                if not self.recorder.recording:    # dataset not ready yet (camera warm-up)
                    self.widget.set_button_option(label_text="Record", option_text="STOP")
                    opt = "STOP"
            elif opt == "STOP":
                self.recorder.stop()
            elif opt == "DEL":
                self.recorder.discard_or_delete_last()
                self.widget.set_button_option(label_text="Record", option_text="STOP")
                opt = "STOP"
            self.prev = opt
        if self.prev == "REC" and not self.recorder.recording:   # stopped by timeout
            self.widget.set_button_option(label_text="Record", option_text="STOP")
            self.prev = "STOP"
        self.widget.setWindowTitle(self.recorder.status())

    def close(self):
        self.timer.stop()
        self.widget.close()
