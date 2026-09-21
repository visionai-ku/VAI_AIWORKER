"""
Episode recorder for SG2 VR teleop (strict JSON workers).

Runs next to run_intent_teleop() without changing it:
  - a background thread receives and writes data (camera decode / video write never block the control loop)
  - a small Qt window (Record: STOP / REC / DEL) driven by a QTimer, which fires inside the
    teleop loop's app.processEvents()

Sources (all ZMQ SUB, extra subscribers do not affect the teleop):
  robot state   tcp://<robot>:5560   outbound.py (strict JSON "sg2_state", or legacy pickle)
  cameras       tcp://<robot>:5570/5571/5572   camera_outbound.py / outbound.py --meta (pickle)
  teleop action tcp://127.0.0.1:5561 this PC's own command publisher (strict JSON "sg2_command")

Saved per episode (<task_dir>/episode_NNN/):
  joints.npz        t, q, names                           robot state
  action.npz        t, q, names, valid, mode, motion      last teleop command (valid = fresh "command")
  rgb_<cam>.mp4     per-camera video
  rgb_<cam>_t.npy   per-camera frame time [s] since episode start
  meta.json
"""
import json
import pickle
import platform
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import zmq
from PyQt5.QtCore import QTimer

from ri_motion_v5_package.qt import MultiRadioQtWidget
from package.worker.fixed_quest_protocol import JOINT_NAMES, parse_joint_groups


DEFAULT_CAMERAS = {"head": 5570, "wrist_left": 5571, "wrist_right": 5572}
ACTION_FRESH_S = 0.2


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


def parse_robot_state(msg):
    """-> (names, q) or None when the state is not ready / incomplete."""
    if not isinstance(msg, dict) or msg.get("ready") is False:
        return None
    joints = msg.get("joints") or {}
    names, pos = [], []
    for grp, st in joints.items():
        names += list(st["name"])
        pos += [float(p) for p in st["position"]]
    if not names:
        return None
    return names, np.asarray(pos, dtype=np.float64)


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
    img = cv2.imdecode(np.frombuffer(msg["data"], dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img[:, :img.shape[1] // 2, :] if msg.get("stereo", True) else img


def list_episodes(task_dir):
    return sorted(p for p in Path(task_dir).glob("episode_*") if (p / "meta.json").exists())


def next_episode_dir(task_dir):
    idx = 0
    while (Path(task_dir) / ("episode_%03d" % idx)).exists():
        idx += 1
    return Path(task_dir) / ("episode_%03d" % idx)


class EpisodeBuffer:
    def __init__(self, ep_dir, hz):
        self.ep_dir, self.hz = ep_dir, hz
        self.t0 = time.time()
        self.jt, self.jq, self.jnames = [], [], None
        self.at, self.aq, self.avalid, self.amode, self.amotion = [], [], [], [], []
        self.rt = {}
        self.writers = {}
        ep_dir.mkdir(parents=True)

    def elapsed(self):
        return time.time() - self.t0

    def add_joint(self, names, q):
        if self.jnames is None:
            self.jnames = list(names)
        self.jt.append(self.elapsed())
        self.jq.append(q)

    def add_action(self, q, valid, mode, motion):
        self.at.append(self.elapsed())
        self.aq.append(q)
        self.avalid.append(bool(valid))
        self.amode.append(mode)
        self.amotion.append(motion)

    def add_frame(self, cam, bgr):
        if cam not in self.writers:
            h, w = bgr.shape[:2]
            self.writers[cam] = cv2.VideoWriter(
                str(self.ep_dir / ("rgb_%s.mp4" % cam)), cv2.VideoWriter_fourcc(*"mp4v"), self.hz, (w, h))
            self.rt[cam] = []
        self.writers[cam].write(bgr)
        self.rt[cam].append(self.elapsed())

    def _release(self):
        for w in self.writers.values():
            w.release()

    def stop(self, **meta_extra):
        self._release()
        np.savez(self.ep_dir / "joints.npz", t=np.asarray(self.jt), q=np.asarray(self.jq),
                 names=np.asarray(self.jnames or []))
        np.savez(self.ep_dir / "action.npz", t=np.asarray(self.at), q=np.asarray(self.aq),
                 names=np.asarray(JOINT_NAMES), valid=np.asarray(self.avalid, dtype=bool),
                 mode=np.asarray(self.amode), motion=np.asarray(self.amotion))
        for cam, ts in self.rt.items():
            np.save(self.ep_dir / ("rgb_%s_t.npy" % cam), np.asarray(ts))
        meta = dict(hz=self.hz, duration_sec=self.elapsed(),
                    n_joint_samples=len(self.jt), n_action_samples=len(self.at),
                    n_action_valid=int(np.sum(self.avalid)),
                    n_frames={c: len(ts) for c, ts in self.rt.items()}, cameras=list(self.rt),
                    joint_names=self.jnames, action_names=list(JOINT_NAMES),
                    recorded_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        meta.update(meta_extra)
        (self.ep_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return meta

    def discard(self):
        self._release()
        shutil.rmtree(self.ep_dir, ignore_errors=True)


class VREpisodeRecorder:
    def __init__(self, robot_ip, task_dir, hz=30, max_sec=120, cameras=None,
                 state_port=5560, command_endpoint="tcp://127.0.0.1:5561", meta=None):
        self.robot_ip, self.task_dir = robot_ip, Path(task_dir)
        self.hz, self.max_sec = float(hz), float(max_sec)
        self.cameras = dict(DEFAULT_CAMERAS if cameras is None else cameras)
        self.state_endpoint = "tcp://%s:%d" % (robot_ip, state_port)
        self.command_endpoint = command_endpoint
        self.meta = dict(meta or {})
        self.task_dir.mkdir(parents=True, exist_ok=True)

        self.lock = threading.Lock()
        self.buf, self.last_saved, self.msg = None, None, "ready"
        self.rates = {"joint": 0.0, "action": 0.0, **{c: 0.0 for c in self.cameras}}
        self.error = None
        self._stop_evt = threading.Event()
        self._thread = None

    # ---------- control (main / Qt thread) ----------
    @property
    def recording(self):
        return self.buf is not None

    def start_thread(self):
        self._thread = threading.Thread(target=self._run, name="vr-episode-recorder", daemon=True)
        self._thread.start()
        return self

    def start(self):
        with self.lock:
            if self.buf is None:
                self.buf = EpisodeBuffer(next_episode_dir(self.task_dir), self.hz)
                self.msg = "recording %s" % self.buf.ep_dir.name
        print("[rec]", self.msg)

    def stop(self):
        with self.lock:
            if self.buf is None:
                return None
            meta = self.buf.stop(robot_ip=self.robot_ip, **self.meta)
            self.last_saved, self.buf = self.buf.ep_dir, None
        nf = "/".join(str(v) for v in meta["n_frames"].values()) or "0"
        self.msg = "saved %s (%.1fs, %d joint / %d action(valid %d) / %s frames)" % (
            self.last_saved.name, meta["duration_sec"], meta["n_joint_samples"],
            meta["n_action_samples"], meta["n_action_valid"], nf)
        print("[rec]", self.msg)
        return meta

    def discard_or_delete_last(self):
        with self.lock:
            if self.buf is not None:
                name = self.buf.ep_dir.name
                self.buf.discard()
                self.buf = None
                self.msg = "discarded (recording) %s" % name
            elif self.last_saved is not None and self.last_saved.exists():
                shutil.rmtree(self.last_saved)
                self.msg = "deleted %s" % self.last_saved.name
                self.last_saved = None
            else:
                self.msg = "nothing to delete"
        print("[rec]", self.msg)

    def status(self):
        n_done = len(list_episodes(self.task_dir))
        hz = " ".join("%s %.0f" % (k, v) for k, v in self.rates.items())
        if self.error:
            return "REC ERROR: %s" % self.error
        buf = self.buf
        if buf is not None:
            return "● REC %s %.1fs | saved %d | Hz %s" % (buf.ep_dir.name, buf.elapsed(), n_done, hz)
        return "IDLE | saved %d | Hz %s | %s" % (n_done, hz, self.msg)

    def close(self):
        if self.buf is not None:
            self.stop()
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ---------- receive / write (background thread) ----------
    def _run(self):
        ctx = zmq.Context()
        state_sock = _sub(ctx, self.state_endpoint)
        cmd_sock = _sub(ctx, self.command_endpoint, conflate=False)
        cam_socks = {c: _sub(ctx, "tcp://%s:%d" % (self.robot_ip, p)) for c, p in self.cameras.items()}
        last_cmd, last_cmd_t = None, -np.inf
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
                cmd_raw = _recv_latest(cmd_sock)
                if cmd_raw is not None:
                    parsed = parse_command(decode_message(cmd_raw))
                    if parsed is not None:
                        last_cmd, last_cmd_t = parsed, now
                frames = {}
                for cam, sock in cam_socks.items():
                    raw = _recv_latest(sock)
                    if raw is None:
                        continue
                    bgr = decode_frame(pickle.loads(raw))
                    if bgr is not None:
                        frames[cam] = bgr

                valid = last_cmd is not None and last_cmd[1] == "command" and now - last_cmd_t <= ACTION_FRESH_S
                with self.lock:
                    buf = self.buf
                    if buf is not None:
                        if state is not None:
                            buf.add_joint(*state)
                        if last_cmd is not None:
                            buf.add_action(last_cmd[0], valid, last_cmd[1], last_cmd[2])
                        else:
                            buf.add_action(np.full(len(JOINT_NAMES), np.nan), False, "none", "none")
                        for cam, bgr in frames.items():
                            buf.add_frame(cam, bgr)
                timeout = buf is not None and buf.elapsed() > self.max_sec

                counts["joint"] += state is not None
                counts["action"] += bool(valid)
                for cam in frames:
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
