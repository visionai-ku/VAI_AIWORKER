"""
Episode recorder for real SG2 data collection.

Receives robot joint state (ZMQ :5560, outbound.py) and camera frames
(ZMQ :5570/5571/5572, camera_outbound.py) and saves one episode per folder:

    <task_dir>/episode_NNN/
        joints.npz           t, q, names                 (robot state)
        action.npz           t, q, names, on             (teleop command, optional)
        rgb_<cam>.mp4        per-camera video
        rgb_<cam>_t.npy      per-camera frame time [s] since episode start
        meta.json
"""
import json
import shutil
import time

import cv2
import numpy as np
import zmq


DEFAULT_CAMERAS = {"head": 5570, "wrist_left": 5571, "wrist_right": 5572}


def open_camera_sub(ip, port):
    s = zmq.Context.instance().socket(zmq.SUB)
    s.setsockopt(zmq.CONFLATE, 1)
    s.setsockopt(zmq.SUBSCRIBE, b"")
    s.setsockopt(zmq.RCVTIMEO, 1)
    s.connect("tcp://%s:%d" % (ip, port))
    return s


def recv_camera(sock):
    try:
        return sock.recv_pyobj()
    except zmq.Again:
        return None


def decode_frame(msg):
    img = cv2.imdecode(np.frombuffer(msg["data"], dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img[:, :img.shape[1] // 2, :] if msg.get("stereo", True) else img


def open_camera_subs(ip, cameras=None, probe_sec=1.5, verbose=True):
    """Connect to each camera port and keep only the ones that deliver frames."""
    cameras = DEFAULT_CAMERAS if cameras is None else cameras
    subs = {}
    for cam, port in cameras.items():
        s = open_camera_sub(ip, port)
        t0, n, last = time.time(), 0, None
        while time.time() - t0 < probe_sec:
            m = recv_camera(s)
            if m is not None:
                n += 1
                last = m
            time.sleep(1e-3)
        if last is None:
            s.close()
            if verbose:
                print("[camera] %-11s 없음: tcp://%s:%d → 제외" % (cam, ip, port))
        else:
            subs[cam] = s
            if verbose:
                print("[camera] %-11s ≈%.0f Hz, %s" % (cam, n / probe_sec, decode_frame(last).shape))
    if verbose:
        print("[camera] 사용: %s" % (list(subs) or "없음 (관절만 저장)"))
    return subs


def flatten_joints(robot_state):
    names, pos = [], []
    for grp, st in robot_state["joints"].items():
        names += list(st["name"])
        pos += [float(p) for p in st["position"]]
    return names, np.asarray(pos, dtype=np.float64)


def list_episodes(task_dir):
    return sorted(p for p in task_dir.glob("episode_*") if (p / "meta.json").exists())


def next_episode_dir(task_dir):
    idx = 0
    while (task_dir / ("episode_%03d" % idx)).exists():
        idx += 1
    return task_dir / ("episode_%03d" % idx)


class EpisodeBuffer:
    """관절/액션은 메모리에, 카메라는 카메라별 mp4 로 바로 쓰고 stop() 에서 나머지를 저장."""

    def __init__(self, ep_dir, hz, cam_names, action_names=None):
        self.ep_dir, self.hz = ep_dir, hz
        self.t0 = time.time()
        self.jt, self.jq, self.jnames = [], [], None
        self.at, self.aq, self.aon = [], [], []
        self.action_names = list(action_names) if action_names is not None else None
        self.rt = {c: [] for c in cam_names}
        self.writers = {}
        ep_dir.mkdir(parents=True)

    def elapsed(self):
        return time.time() - self.t0

    def add_joint(self, robot_state):
        names, q = flatten_joints(robot_state)
        if self.jnames is None:
            self.jnames = names
        self.jt.append(self.elapsed())
        self.jq.append(q)

    def add_action(self, q, on=True):
        self.at.append(self.elapsed())
        self.aq.append(np.asarray(q, dtype=np.float64).copy())
        self.aon.append(bool(on))

    def add_frame(self, cam, bgr):
        if cam not in self.writers:
            h, w = bgr.shape[:2]
            self.writers[cam] = cv2.VideoWriter(
                str(self.ep_dir / ("rgb_%s.mp4" % cam)), cv2.VideoWriter_fourcc(*"mp4v"), self.hz, (w, h)
            )
        self.writers[cam].write(bgr)
        self.rt[cam].append(self.elapsed())

    def _release(self):
        for w in self.writers.values():
            w.release()

    def stop(self, **meta_extra):
        self._release()
        np.savez(self.ep_dir / "joints.npz", t=np.asarray(self.jt), q=np.asarray(self.jq),
                 names=np.asarray(self.jnames or []))
        if self.at:
            np.savez(self.ep_dir / "action.npz", t=np.asarray(self.at), q=np.asarray(self.aq),
                     on=np.asarray(self.aon), names=np.asarray(self.action_names or []))
        for cam, ts in self.rt.items():
            if ts:
                np.save(self.ep_dir / ("rgb_%s_t.npy" % cam), np.asarray(ts))
        meta = dict(hz=self.hz, duration_sec=self.elapsed(),
                    n_joint_samples=len(self.jt), n_action_samples=len(self.at),
                    n_frames={c: len(ts) for c, ts in self.rt.items()}, cameras=list(self.rt),
                    joint_names=self.jnames, action_names=self.action_names,
                    recorded_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        meta.update(meta_extra)
        (self.ep_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return meta

    def discard(self):
        self._release()
        shutil.rmtree(self.ep_dir, ignore_errors=True)


class EpisodeRecorder:
    """
    텔레옵 루프에 붙이는 녹화기. 매 tick 마다 poll() 을 부르면
    관절/카메라를 받아서 녹화 중일 때만 버퍼에 쌓는다.
    """

    def __init__(self, sub_joint, sub_cams, task_dir, hz=30, max_sec=120, action_names=None, meta=None):
        self.sub_joint, self.sub_cams = sub_joint, sub_cams
        self.task_dir, self.hz, self.max_sec = task_dir, hz, max_sec
        self.action_names = action_names
        self.meta = dict(meta or {})
        self.buf, self.last_saved, self.msg = None, None, "ready"
        self.frames = {}
        task_dir.mkdir(parents=True, exist_ok=True)

    @property
    def recording(self):
        return self.buf is not None

    def start(self):
        if self.buf is None:
            self.buf = EpisodeBuffer(next_episode_dir(self.task_dir), self.hz, list(self.sub_cams), self.action_names)
            self.msg = "recording %s" % self.buf.ep_dir.name
            print("[rec] start", self.buf.ep_dir.name)

    def stop(self):
        if self.buf is None:
            return None
        meta = self.buf.stop(**self.meta)
        self.last_saved, self.buf = self.buf.ep_dir, None
        nf = "/".join(str(v) for v in meta["n_frames"].values())
        self.msg = "saved %s (%.1fs, %d j / %d a / %s f)" % (
            self.last_saved.name, meta["duration_sec"], meta["n_joint_samples"], meta["n_action_samples"], nf)
        print("[rec]", self.msg)
        return meta

    def discard_or_delete_last(self):
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

    def poll(self, action_q=None, action_on=True):
        m = self.sub_joint.recv()
        if m is not None and self.buf is not None:
            self.buf.add_joint(m)
        for cam, sock in self.sub_cams.items():
            cm = recv_camera(sock)
            if cm is None:
                continue
            bgr = decode_frame(cm)
            if bgr is None:
                continue
            self.frames[cam] = bgr
            if self.buf is not None:
                self.buf.add_frame(cam, bgr)
        if self.buf is not None:
            if action_q is not None:
                self.buf.add_action(action_q, action_on)
            if self.buf.elapsed() > self.max_sec:
                self.stop()
                self.msg = "timeout → " + self.msg

    def status(self):
        n_done = len(list_episodes(self.task_dir))
        if self.buf is not None:
            return "● REC %s %.1fs | saved %d" % (self.buf.ep_dir.name, self.buf.elapsed(), n_done)
        return "IDLE | saved %d | %s" % (n_done, self.msg)

    def close(self):
        if self.buf is not None:
            self.stop()
        for s in self.sub_cams.values():
            s.close()
