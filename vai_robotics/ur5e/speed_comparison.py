"""Compare one REAL clutch to its input-matched MuJoCo preview on the RTDE clock."""

import math

import numpy as np

from .controllers import MAX_TICK_INTERVAL


SPEED_FIELDS = ("real_mm_s", "dry_mm_s", "real_deg_s", "dry_deg_s")


def _speeds(previous, current, before, after, dt):
    # RTDE reports instantaneous base-frame twist. Average its two endpoints
    # over the same interval as the simulated TCP pose difference.
    real_linear = (math.hypot(*previous[:3]) + math.hypot(*current[:3])) / 2
    real_angular = (math.hypot(*previous[3:]) + math.hypot(*current[3:])) / 2
    dry_linear = np.linalg.norm(after[0] - before[0]) / dt
    dry_angular = (after[1] * before[1].inv()).magnitude() / dt
    return np.array((real_linear * 1000, dry_linear * 1000,
                     math.degrees(real_angular), math.degrees(dry_angular)))


class SpeedComparison:
    """Process every active control frame, independently of the viewer's render rate."""

    def __init__(self, view):
        self.view = view
        self.grip = None
        self.finished_grip = None
        self.previous = None
        self.pose = None
        self.elapsed = 0.
        self.integral = np.zeros(4)
        self.result = {"phase": "waiting"}

    def _invalidate(self):
        self.previous = None
        self.result = {"phase": "unavailable"}

    def _pair(self, frame):
        previous = self.previous
        dt = frame.feedback_timestamp - previous.feedback_timestamp
        if not 0 < dt <= MAX_TICK_INTERVAL or previous.payload != frame.payload:
            self._invalidate()
            return
        # Feedback on this tick reflects the previous command. Advancing the
        # new command here would give the simulation a one-sample head start.
        pose = self.view.advance_comparison(previous.target, previous.joint_target,
                                            impedance=previous.gains is not None,
                                            incremental=previous.delta, dt=dt)
        speeds = _speeds(previous.actual_speed, frame.actual_speed, self.pose, pose, dt)
        if not np.isfinite(speeds).all():
            self._invalidate()
            return
        self.integral += speeds * dt
        self.elapsed += dt
        self.pose = pose
        self.previous = frame
        self.result = {"phase": "live", **dict(zip(SPEED_FIELDS, speeds))}

    def _finish(self, frame):
        if self.grip is None:
            if frame.index and self.finished_grip != frame.grip:
                # Even the first active frame can be lost in a stalled viewer.
                self._invalidate()
                self.finished_grip = frame.grip
            return
        if self.previous is None or frame.grip != self.grip or frame.index != self.previous.index:
            self._invalidate()
        elif not frame.dry_run:
            if frame.released and self._has_feedback(frame):
                self._pair(frame)
            else:
                self._invalidate()
        if self.previous is not None and self.elapsed > 0:
            self.result = {"phase": "last", **dict(zip(SPEED_FIELDS, self.integral / self.elapsed))}
        else:
            self._invalidate()
        self.finished_grip = self.grip
        self.grip = None
        self.previous = None

    @staticmethod
    def _has_feedback(frame):
        return (frame.payload is not None and math.isfinite(frame.feedback_timestamp)
                and all(map(math.isfinite, frame.actual_speed)))

    def update(self, frame):
        if not frame.comparing:
            self._finish(frame)
            return self.result

        new_grip = frame.grip != self.grip
        if new_grip:
            self.grip = frame.grip
            self.previous = None
            self.elapsed = 0.
            self.integral.fill(0.)
        if not self._has_feedback(frame):
            self._invalidate()
        elif new_grip and frame.index == 1:
            self.pose = self.view.start_comparison(frame.q, frame.measured_tcp, joint=frame.joint_target is not None,
                                                   impedance=frame.gains is not None)
            self.previous = frame
            self.result = {"phase": "waiting"}
        elif self.previous is None or frame.index != self.previous.index + 1:
            # Partial gestures do not have a reproducible initial condition.
            self._invalidate()
        else:
            self._pair(frame)
        return self.result
