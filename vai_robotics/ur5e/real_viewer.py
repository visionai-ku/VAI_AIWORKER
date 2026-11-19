"""Send UR feedback to a local MuJoCo window without blocking the RTDE loop."""

from dataclasses import dataclass
import math
import os
import select
import struct
import subprocess
import sys
import time


# Each frame carries its own configuration: a full pipe may drop any write.
# Even with all twelve gains the frame fits in the POSIX minimum PIPE_BUF (512 bytes).
_FRAME = struct.Struct("<dQ5? 6d 6d 6d d 12d")
_HZ = 50
_MAX_FRAME_AGE = 0.15
_ZERO_AXES = (0.0,) * 6


@dataclass
class _Snapshot:
    timestamp: float
    grip: int
    dry_run: bool
    delta: bool
    q: tuple
    measured_tcp: tuple
    target: tuple | None
    joint_target: tuple | None
    width_mm: float | None
    gains: tuple | None

    @classmethod
    def decode(cls, values):
        from scipy.spatial.transform import Rotation

        timestamp, grip, dry_run, clutched, joints, impedance, delta = values[:7]
        q, measured, preview = values[7:13], values[13:19], values[19:25]
        width_mm, gains = values[25], values[26:]
        measured_tcp = measured[:3], Rotation.from_rotvec(measured[3:])
        target = (preview[:3], Rotation.from_rotvec(preview[3:])) if clutched and not joints else None
        return cls(timestamp, grip, dry_run, delta, q, measured_tcp, target, preview if joints else None,
                   None if math.isnan(width_mm) else width_mm, gains if impedance else None)


class RealViewer:
    def __init__(self):
        self._process = None
        self._next_frame = 0.0
        self._generation = 0
        self._preview_mode = None

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

    def update(self, *, q, measured_tcp, dry_run, target, joint_target=None, gripper_width_mm=None,
               preview_gains=None, preview_delta=False):
        preview_joints = dry_run and joint_target is not None
        clutched = dry_run and (target is not None or preview_joints)
        impedance = clutched and preview_gains is not None
        mode = (preview_joints, impedance) if clutched else None
        if clutched and self._preview_mode != mode:
            self._generation += 1
        self._preview_mode = mode

        now = time.monotonic()
        if now < self._next_frame:
            return
        self._next_frame = now + 1 / _HZ
        xyz, rotation = measured_tcp
        preview = _ZERO_AXES
        if preview_joints:
            preview = joint_target
        elif clutched:
            preview = (*target[0], *target[1].as_rotvec())
        stiffness = preview_gains["stiffness"] if impedance else _ZERO_AXES
        damping = preview_gains["damping"] if impedance else _ZERO_AXES
        frame = _FRAME.pack(now, self._generation, dry_run, clutched, preview_joints, impedance,
                            impedance and preview_delta, *q, *xyz, *rotation.as_rotvec(), *preview,
                            math.nan if gripper_width_mm is None else gripper_width_mm, *stiffness, *damping)
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
            yield _Snapshot.decode(frame) if time.monotonic() - frame[0] < _MAX_FRAME_AGE else None


def _display(view, fd, poll_events):
    previous_grip = None
    previous_time = 0.0
    applied_gains = {}
    for frame in _snapshots(fd):
        if not view.is_running():
            break
        if frame is None:
            poll_events()
            continue
        if frame.gains is not None:
            space = "joint" if frame.joint_target is not None else "eef"
            if applied_gains.get(space) != frame.gains:
                view.set_impedance_gains(space, {"stiffness": frame.gains[:6], "damping": frame.gains[6:]})
                applied_gains[space] = frame.gains

        regripped = frame.grip != previous_grip
        # Integrate skipped display frames during a grip, but discard long pauses.
        elapsed = frame.timestamp - previous_time
        dt = elapsed if not regripped and 0 < elapsed <= _MAX_FRAME_AGE else 1 / _HZ
        view.tick(dry_run=frame.dry_run, target=frame.target, joint_target=frame.joint_target,
                  measured_q=frame.q, measured_tcp=frame.measured_tcp, gripper_width_mm=frame.width_mm,
                  reset_preview=regripped, preview_impedance=frame.gains is not None,
                  preview_delta=frame.delta, preview_dt=dt)
        previous_grip, previous_time = frame.grip, frame.timestamp


if __name__ == "__main__":
    import glfw

    from .sim_teleop import SimBackend

    with SimBackend(hz=_HZ) as viewer:
        _display(viewer, sys.stdin.fileno(), glfw.poll_events)
