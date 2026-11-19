"""Send UR feedback to a local MuJoCo window without blocking the RTDE loop."""

import math
import os
import select
import struct
import subprocess
import sys
import time


# Monotonic timestamp, grip generation, dry/preview flags, measured joints,
# measured TCP and preview (TCP pose or joint angles according to the last flag).
# One frame fits in POSIX PIPE_BUF, so the nonblocking pipe writes are atomic.
_FRAME = struct.Struct("<dQ??19d?")  # Last double is RG2 width in mm, NaN if unavailable.
_HZ = 50
_ZERO_TCP = (0.0,) * 6


class RealViewer:
    def __init__(self):
        self._process = None
        self._next_frame = 0.0
        self._generation = 0
        self._clutched = False

    def __enter__(self):
        self._process = subprocess.Popen(
            [sys.executable, "-m", "vai_robotics.ur5e.real_viewer"],
            stdin=subprocess.PIPE, bufsize=0, start_new_session=True,
        )
        try:
            os.set_blocking(self._process.stdin.fileno(), False)
        except BaseException:
            self.close()
            raise
        return self

    def update(self, *, q, measured_tcp, dry_run, target, joint_target=None, gripper_width_mm=None):
        preview_joints = dry_run and joint_target is not None
        clutched = dry_run and (target is not None or preview_joints)
        if clutched and not self._clutched:
            self._generation += 1
        self._clutched = clutched

        now = time.monotonic()
        if now < self._next_frame:
            return
        self._next_frame = now + 1 / _HZ
        xyz, rotation = measured_tcp
        preview = _ZERO_TCP
        if preview_joints:
            preview = joint_target
        elif clutched:
            preview = (*target[0], *target[1].as_rotvec())
        frame = _FRAME.pack(now, self._generation, dry_run, clutched, *q, *xyz,
                            *rotation.as_rotvec(), *preview,
                            math.nan if gripper_width_mm is None else gripper_width_mm, preview_joints)
        try:
            os.write(self._process.stdin.fileno(), frame)
        except (BlockingIOError, BrokenPipeError):
            # A slow/closing viewer can drop a snapshot without delaying a robot tick.
            pass

    def is_running(self):
        return self._process is not None and self._process.poll() is None

    def close(self):
        if self._process is None:
            return
        self._process.stdin.close()
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._process = None

    def __exit__(self, *exc):
        self.close()


def _snapshots(fd):
    """Yield the newest complete frame, or None to pump window events while idle."""
    os.set_blocking(fd, False)
    pending = bytearray()
    while True:
        readable, _, _ = select.select([fd], [], [], 1 / _HZ)
        if not readable:
            yield None
            continue
        while True:
            try:
                block = os.read(fd, 65536)
            except BlockingIOError:
                break
            if not block:  # Parent closed the session, including on unexpected exit.
                return
            pending.extend(block)
        size = len(pending) // _FRAME.size * _FRAME.size
        if size:
            frame = _FRAME.unpack_from(pending, size - _FRAME.size)
            del pending[:size]
            # Discard old feedback after a delayed launch or a window-system stall.
            yield frame if time.monotonic() - frame[0] < 0.15 else None


def _display(view, fd, poll_events):
    from scipy.spatial.transform import Rotation

    previous_grip = None
    for frame in _snapshots(fd):
        if not view.is_running():
            break
        if frame is None:
            poll_events()
            continue
        _, grip, dry_run, clutched = frame[:4]
        measured_tcp = frame[10:13], Rotation.from_rotvec(frame[13:16])
        preview_joints = frame[23]
        target = (frame[16:19], Rotation.from_rotvec(frame[19:22])) if clutched and not preview_joints else None
        view.tick(dry_run=dry_run, target=target, joint_target=frame[16:22] if preview_joints else None,
                  measured_q=frame[4:10],
                  measured_tcp=measured_tcp, reset_preview=grip != previous_grip,
                  gripper_width_mm=None if math.isnan(frame[22]) else frame[22])
        previous_grip = grip


if __name__ == "__main__":
    import glfw

    from .sim_teleop import SimBackend

    with SimBackend(hz=_HZ) as viewer:
        _display(viewer, sys.stdin.fileno(), glfw.poll_events)
