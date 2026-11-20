"""Send UR feedback to MuJoCo and return comparison readings without blocking RTDE."""

from dataclasses import dataclass
import errno
from functools import partial
import json
import math
import os
import select
import socket
import struct
import subprocess
import sys
import time

from .payload import ToolPayload
from .rg2_visual import DEFAULT_VISUAL_MOUNT, normalize_visual_mount
from .teleop_types import JointDrag


# Each write is atomic even on a pipe with POSIX minimum PIPE_BUF (512 bytes).
_FRAME = struct.Struct("<dQI7? 6d 6d 6d d 12d 4d d 6d Q 6d")
_HZ = 50
_MAX_FRAME_AGE = 0.15
_ZERO_AXES = (0.0,) * 6


@dataclass
class _Snapshot:
    timestamp: float
    grip: int
    index: int
    dry_run: bool
    comparing: bool
    released: bool
    delta: bool
    q: tuple
    measured_tcp: tuple
    target: tuple | None
    joint_target: tuple | None
    width_mm: float | None
    gains: tuple | None
    payload: ToolPayload | None
    feedback_timestamp: float
    actual_speed: tuple
    ring_epoch: int = 0
    rg2_mount: tuple = DEFAULT_VISUAL_MOUNT

    @classmethod
    def decode(cls, data):
        from scipy.spatial.transform import Rotation

        now, grip, index, dry, clutched, joints, impedance, delta, comparing, released = data[:10]
        q, measured, preview = data[10:16], data[16:22], data[22:28]
        width, gains = data[28], data[29:41]
        mass, cog, feedback_timestamp, speed = data[41], data[42:45], data[45], data[46:52]
        target = (preview[:3], Rotation.from_rotvec(preview[3:])) if clutched and not joints else None
        return cls(now, grip, index, dry, comparing, released, delta, q, (measured[:3], Rotation.from_rotvec(measured[3:])),
                   target, preview if joints else None, None if math.isnan(width) else width,
                   gains if impedance else None, None if math.isnan(mass) else ToolPayload(mass, cog),
                   feedback_timestamp, speed, data[52], data[53:59])


class RealViewer:
    def __init__(self):
        self._process = None
        self._results = None
        self._latest_result = None
        self._next_frame = 0.0
        self._generation = 0
        self._index = 0
        self._preview_mode = None
        self._ring_epoch = 0
        self._rg2_mount = DEFAULT_VISUAL_MOUNT
        self._drag = None
        self._drag_at = float("-inf")

    def __enter__(self):
        self._results, sender = socket.socketpair(type=socket.SOCK_DGRAM)
        try:
            self._results.setblocking(False)
            with sender:
                self._process = subprocess.Popen(
                    [sys.executable, "-m", "vai_robotics.ur5e.real_viewer", str(sender.fileno())],
                    stdin=subprocess.PIPE, bufsize=0, start_new_session=True, pass_fds=(sender.fileno(),),
                )
            os.set_blocking(self._process.stdin.fileno(), False)
        except BaseException:
            self.close()
            raise
        return self

    def set_joint_ring_epoch(self, epoch):
        if self._ring_epoch != epoch:
            self._ring_epoch = epoch
            self._drag = None
            self._drag_at = float("-inf")
            self._next_frame = 0.0  # Send the new input configuration on the next tick.

    def set_rg2_mount(self, mount):
        self._rg2_mount = normalize_visual_mount(mount)

    def joint_drag(self):
        self._poll_feedback()
        if not self._ring_epoch or not 0 <= time.monotonic() - self._drag_at <= _MAX_FRAME_AGE:
            return None
        return self._drag

    def update(self, *, q, measured_tcp, dry_run, target, joint_target=None, gripper_width_mm=None,
               preview_gains=None, preview_delta=False, payload=None, comparison_target=None,
               comparison_joints=None, comparison_gains=None, comparison_delta=False,
               actual_tcp_speed=None, feedback_timestamp=None):
        comparing = not dry_run and (comparison_target is not None or comparison_joints is not None)
        if not dry_run:
            target, joint_target = comparison_target, comparison_joints
            preview_gains, preview_delta = comparison_gains, comparison_delta
        joints = joint_target is not None
        clutched = target is not None or joints
        impedance = clutched and preview_gains is not None
        mode = (dry_run, joints, impedance) if clutched else None
        released = self._preview_mode is not None and mode is None
        if clutched and self._preview_mode != mode:
            self._generation += 1
            self._index = 0
        self._preview_mode = mode
        if comparing:
            self._index += 1  # A failed nonblocking write is detectable by the consumer.
        now = time.monotonic()
        if not (comparing or released) and now < self._next_frame:
            return
        self._next_frame = now + 1 / _HZ
        xyz, rotation = measured_tcp
        preview = _ZERO_AXES
        if joints:
            preview = joint_target
        elif clutched:
            preview = (*target[0], *target[1].as_rotvec())
        stiffness = preview_gains["stiffness"] if impedance else _ZERO_AXES
        damping = preview_gains["damping"] if impedance else _ZERO_AXES
        load = (payload.mass_kg, *payload.cog_m) if payload is not None else (math.nan,) * 4
        speed = actual_tcp_speed if actual_tcp_speed is not None else (math.nan,) * 6
        frame = _FRAME.pack(now, self._generation, self._index, dry_run, clutched, joints, impedance,
                            impedance and preview_delta, comparing, released, *q, *xyz, *rotation.as_rotvec(), *preview,
                            math.nan if gripper_width_mm is None else gripper_width_mm, *stiffness, *damping,
                            *load, math.nan if feedback_timestamp is None else feedback_timestamp, *speed,
                            self._ring_epoch, *self._rg2_mount)
        try:
            os.write(self._process.stdin.fileno(), frame)
        except (BlockingIOError, BrokenPipeError):
            pass

    def _poll_feedback(self):
        if self._results is None:
            return
        while True:
            try:
                packet = self._results.recv(1024)
            except (BlockingIOError, ConnectionResetError, ConnectionRefusedError):
                break
            if not packet:
                break
            fields = json.loads(packet)
            self._latest_result = fields[:3]
            if len(fields) == 5:
                stamp, epoch, drag = fields[1], fields[3], fields[4]
                if epoch == self._ring_epoch and stamp > self._drag_at:
                    self._drag_at = stamp
                    self._drag = None if drag is None else JointDrag(*drag)

    def speed_comparison(self):
        self._poll_feedback()
        if self._latest_result is None:
            return None
        grip, timestamp, result = self._latest_result
        if grip != self._generation:
            return {"phase": "waiting"}
        if result["phase"] == "live" and time.monotonic() - timestamp > _MAX_FRAME_AGE:
            return {"phase": "unavailable"}
        return result

    def is_running(self):
        return self._process is not None and self._process.poll() is None

    def close(self):
        self.set_joint_ring_epoch(0)
        if self._results is not None:
            self._results.close()
            self._results = None
        if self._process is not None:
            self._process.stdin.close()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None

    def __exit__(self, *exc):
        self.close()


def _frame_batches(fd):
    """Read all fresh control samples. Only the newest needs to be rendered."""
    os.set_blocking(fd, False)
    pending = bytearray()
    while True:
        readable, _, _ = select.select([fd], [], [], 1 / _HZ)
        if not readable:
            yield []
            continue
        while True:
            try:
                block = os.read(fd, 65536)
            except BlockingIOError:
                break
            if not block:
                return
            pending.extend(block)
        size = len(pending) // _FRAME.size * _FRAME.size
        if size:
            now = time.monotonic()
            frames = (_FRAME.unpack_from(pending, offset) for offset in range(0, size, _FRAME.size))
            batch = [_Snapshot.decode(frame) for frame in frames if now - frame[0] < _MAX_FRAME_AGE]
            del pending[:size]
            yield batch


def _snapshots(fd):
    """Latest display frame (or None while idle)."""
    for batch in _frame_batches(fd):
        yield batch[-1] if batch else None


def _configure_preview(view, frame, applied_gains):
    if frame.payload is not None:
        view.set_payload(frame.payload)
    if frame.gains is not None:
        space = "joint" if frame.joint_target is not None else "eef"
        if applied_gains.get(space) != frame.gains:
            view.set_impedance_gains(space, {"stiffness": frame.gains[:6], "damping": frame.gains[6:]})
            applied_gains[space] = frame.gains


def _display(view, fd, poll_events, report=None):
    from .speed_comparison import SpeedComparison

    comparison = SpeedComparison(view)
    previous_grip = None
    previous_time = 0.0
    applied_gains = {}
    for batch in _frame_batches(fd):
        if not view.is_running():
            break
        if not batch:
            poll_events()
            continue
        for frame in batch:
            if frame.comparing:
                _configure_preview(view, frame, applied_gains)
            result = comparison.update(frame)
        frame = batch[-1]
        if not frame.comparing:
            _configure_preview(view, frame, applied_gains)

        regripped = frame.grip != previous_grip
        elapsed = frame.timestamp - previous_time
        dt = elapsed if not regripped and 0 < elapsed <= _MAX_FRAME_AGE else 1 / _HZ
        view.set_joint_ring_epoch(frame.ring_epoch)
        view.set_rg2_mount(frame.rg2_mount)
        view.tick(dry_run=frame.dry_run, target=frame.target if frame.dry_run else None,
                  joint_target=frame.joint_target if frame.dry_run else None,
                  measured_q=frame.q, measured_tcp=frame.measured_tcp, gripper_width_mm=frame.width_mm,
                  reset_preview=regripped, preview_impedance=frame.dry_run and frame.gains is not None,
                  preview_delta=frame.dry_run and frame.delta, preview_dt=dt, comparing=frame.comparing)
        if report is not None:
            if frame.ring_epoch:
                report(frame.grip, frame.timestamp, result, frame.ring_epoch, view.joint_drag())
            else:
                report(frame.grip, frame.timestamp, result)
        previous_grip, previous_time = frame.grip, frame.timestamp


def _publish_result(results, grip, timestamp, result, ring_epoch=0, drag=None):
    packet = [grip, timestamp, result]
    if ring_epoch:
        state = None if drag is None else (drag.serial, drag.joint, drag.angle)
        packet.extend((ring_epoch, state))
    try:
        results.send(json.dumps(packet, allow_nan=False).encode())
    except OSError as error:
        # macOS may return ENOBUFS when a nonblocking datagram socket is full.
        if error.errno not in (errno.EAGAIN, errno.EWOULDBLOCK, errno.ENOBUFS, errno.ECONNRESET,
                               errno.ECONNREFUSED, errno.ENOTCONN, errno.EPIPE):
            raise


def _run_viewer(result_fd=None):
    import glfw
    from .sim_teleop import SimBackend

    with SimBackend(hz=_HZ) as viewer:
        if result_fd is None:
            _display(viewer, sys.stdin.fileno(), glfw.poll_events)
            return
        with socket.socket(fileno=result_fd) as results:
            results.setblocking(False)
            _display(viewer, sys.stdin.fileno(), glfw.poll_events, partial(_publish_result, results))


if __name__ == "__main__":
    _run_viewer(int(sys.argv[1]) if len(sys.argv) > 1 else None)
