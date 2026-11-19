"""Encode matched teleoperation observations and setpoints as LeRobot episodes."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .controllers import HZ
from .recording import EpisodeRecorder


JOINTS = ("shoulder_pan_rad", "shoulder_lift_rad", "elbow_rad", "wrist_1_rad", "wrist_2_rad", "wrist_3_rad")
TCP = ("x_m", "y_m", "z_m", "qw", "qx", "qy", "qz")


def _feature(names):
    return {"dtype": "float32", "shape": (len(names),), "names": list(names)}


def _vector(values, length, name):
    vector = np.array(values, dtype=np.float32, copy=True)
    if vector.shape != (length,) or not np.isfinite(vector).all():
        raise ValueError(f"Missing or invalid {name}")
    return vector


@dataclass(frozen=True)
class _Observation:
    eef_mode_joints: np.ndarray | None
    gripper_width: float | None
    images: dict


class _EpisodeFrames:
    """Fix the action space and camera layout for the lifetime of a take."""

    def __init__(self, arm, with_gripper, images):
        self.joint_space = arm.joint_space
        self.with_gripper = with_gripper
        width = ("rg2_width_mm",) if with_gripper else ()
        action = JOINTS if self.joint_space else TCP
        self.features = {
            "observation.state": _feature(JOINTS + width),
            "observation.eef_pose": _feature(TCP),
            "action": _feature(action + width),
            "control.active": _feature(["clutch_active"]),
            "control.elapsed_s": _feature(["elapsed_s"]),
        }
        self.impedance = arm.control_law == "impedance"
        if self.impedance:
            axes = ("J1", "J2", "J3", "J4", "J5", "J6") if self.joint_space else ("X", "Y", "Z", "RX", "RY", "RZ")
            for gain in ("stiffness", "damping"):
                self.features[f"control.{gain}"] = _feature(axes)
        self.image_shapes = {name: image.shape for name, image in images.items()}
        for name, shape in self.image_shapes.items():
            self.features[f"observation.images.{name}"] = {
                "dtype": "video", "shape": shape, "names": ["height", "width", "channels"],
            }
        self.first_tick = None
        self.previous_quat = {}

    def _pose(self, pose, field):
        xyz, rotation = pose
        quat = rotation.as_quat()[[3, 0, 1, 2]]  # RTDE rotation vector -> scalar-first quaternion.
        previous = self.previous_quat.get(field, self.previous_quat.get("measured"))
        if previous is not None and np.dot(previous, quat) < 0:
            quat = -quat
        self.previous_quat[field] = quat
        return _vector(np.r_[xyz, quat], 7, "TCP pose")

    def encode(self, arm, observation, gripper, tick):
        q = arm.measured_joints if self.joint_space else observation.eef_mode_joints
        state = _vector(q, 6, "joint observation")
        measured = self._pose(arm.measured, "measured")
        if self.joint_space:
            action = _vector(arm.joint_target if arm.active else state, 6, "joint setpoint")
        else:
            action = self._pose(arm.target if arm.active else arm.measured, "target")
        if self.with_gripper:
            target = gripper.commanded_width(observation.gripper_width)
            width, target_width = _vector([observation.gripper_width, target], 2, "RG2 feedback or target width")
            state = np.append(state, width)
            action = np.append(action, target_width)
        if self.first_tick is None:
            self.first_tick = tick
        frame = {
            "observation.state": state, "observation.eef_pose": measured, "action": action,
            "control.active": np.array([arm.active], dtype=np.float32),
            "control.elapsed_s": _vector([tick - self.first_tick], 1, "tick timestamp"),
        }
        if self.impedance:
            gains = arm.backend.get_impedance_gains("joint" if self.joint_space else "eef")
            for gain in ("stiffness", "damping"):
                frame[f"control.{gain}"] = _vector(gains[gain], 6, gain)
        if observation.images.keys() != self.image_shapes.keys():
            raise ValueError("Camera set changed during recording")
        for name, image in observation.images.items():
            if image.shape != self.image_shapes[name]:
                raise ValueError(f"Camera resolution changed during recording: {name}")
            frame[f"observation.images.{name}"] = image
        return frame


class DemonstrationRecording:
    """Route panel requests to the dataset for the confirmed run and controller."""

    def __init__(self, base_dir, *, simulation, task="UR5e teleoperation", cameras=None, max_seconds=120):
        if not task.strip():
            raise ValueError("Recording requires a nonempty task")
        self.base_dir = Path(base_dir).expanduser().absolute() / ("sim" if simulation else "real")
        self.task = task
        self.cameras = cameras
        self.max_seconds = max_seconds
        self._writers = {}
        self._current = None
        self._frames = None
        self._start_errors = {}

    def _images(self):
        return self.cameras.snapshot() if self.cameras is not None else {}

    def observe(self, arm, gripper, request):
        """Sample auxiliary sensors before dispatch. Joint mode reads its own q during control."""
        if (arm.dry_run or request is not None
                or self._current is None or not self._current.recording):
            return None
        try:
            joints = None if arm.joint_space else arm.backend.joints()
            return _Observation(joints, gripper.measured_width(), self._images())
        except Exception as error:
            # Apply the failure only if this tick belongs to the same episode after dispatch.
            return error

    def _start(self, destination, arm, gripper):
        self._start_errors.pop(destination, None)
        frames = _EpisodeFrames(arm, gripper.has_recorded_width, self._images())
        writer = self._writers.get(destination)
        if writer is None:
            writer = EpisodeRecorder(destination, fps=HZ, task=self.task, max_seconds=self.max_seconds)
            self._writers[destination] = writer
        if writer.start(frames.features):
            self._current = writer
            self._frames = frames

    def _finish_on_switch(self, request):
        writer = self._current
        state = writer.status
        if request != "discard" and state["state"] == "recording" and state["frames"]:
            writer.save()
        else:
            writer.discard()

    def _can_open_destination(self, destination):
        if self._current is None or self._current.root == destination:
            return True
        return self._current.status["state"] in ("idle", "error")

    def _status(self, destination, dry_run):
        writer = self._writers.get(destination)
        status = (writer.status if writer else
                  dict(state="idle", root=str(destination), task=self.task, frames=0,
                       episodes=0, episodes_known=False, message=""))
        ready = self._can_open_destination(destination)
        if not ready:
            # Still display the old path until its requested save has settled.
            status = self._current.status
        elif destination in self._start_errors:
            status.update(state="error", message=self._start_errors[destination])
        elif self._current and self._current is not writer and self._current.status["state"] == "error":
            previous = self._current.status
            status["message"] = f"Previous dataset {previous['root']}: {previous['message']}"
        elif dry_run and status["state"] == "idle":
            status["message"] = "Switch to SIM RUN or REAL RUN to record a demonstration."
        status["can_start"] = status.get("can_start", True) and ready and not dry_run
        return status

    def fail(self, message):
        if self._current is not None:
            self._current.fail(message)

    def save_pending(self):
        if self._current is not None:
            self._current.save()

    def update(self, request, arm, gripper, observation, tick):
        destination = self.base_dir / f"{arm.command_mode}-{arm.control_law}"
        changed = self._current is not None and (arm.dry_run or destination != self._current.root)
        if changed:
            self._finish_on_switch(request)
        writer = self._writers.get(destination)
        if request == "start" and not arm.dry_run and self._can_open_destination(destination):
            try:
                self._start(destination, arm, gripper)
                writer = self._writers.get(destination)
            except Exception as error:
                self._start_errors[destination] = str(error)
        elif request == "discard" and writer is not None:
            writer.discard()
        elif request == "save" and writer is not None:
            writer.save()

        capturing = writer is not None and writer is self._current and writer.recording and not arm.dry_run
        if observation is not None and capturing:
            try:
                if isinstance(observation, Exception):
                    raise observation
                frame = self._frames.encode(arm, observation, gripper, tick)
                writer.append(frame)
            except Exception as error:
                # A failed sensor invalidates only the take. Teleop continues.
                writer.fail(str(error))
        return self._status(destination, arm.dry_run)

    def close(self):
        for writer in self._writers.values():
            writer.close()
