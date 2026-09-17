"""Small MuJoCo control replay recorder.

The class records the minimum state needed to replay a MuJoCo control
trajectory on an environment that is initialized the same way for collect and
replay.
"""

from datetime import datetime
from pathlib import Path

import numpy as np


class MuJoCoReplay:
    """
    Record and replay MuJoCo control trajectories.

    Usage:
        obj = MuJoCoReplay(...)
    """

    schema_version = "mujoco_replay_v1"

    def __init__(
            self,
            env,
            sim_nstep     = 1,
            record_hz     = None,
            save_qpos_ref = True,
            save_qvel_ref = False,
        ):
        """
        Initialize MuJoCoReplay.

        Parameters:
            env (Any): Input env.
            sim_nstep (int): Input sim nstep. Defaults to `1`.
            record_hz (Any | None): Input record hz. Defaults to `None`.
            save_qpos_ref (bool): Input save qpos ref. Defaults to `True`.
            save_qvel_ref (bool): Input save qvel ref. Defaults to `False`.

        Returns:
            None: Initializes the instance in place.
        """
        self.env = env
        self.sim_nstep = int(sim_nstep)
        self.record_hz = (
            float(record_hz)
            if record_hz is not None
            else 1.0 / (float(env.dt) * float(sim_nstep))
        )
        self.save_qpos_ref = bool(save_qpos_ref)
        self.save_qvel_ref = bool(save_qvel_ref)
        self.path = None
        self.clear()

    def clear(self):
        """
        Clear all recorded data and reset state. Call this before starting a new
        recording.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.initial_qpos = None
        self.initial_qvel = None
        self.initial_ctrl = None
        self.t0 = None
        self.t = []
        self.sim_time = []
        self.ctrl = []
        self.qpos_ref = []
        self.qvel_ref = []
        self.play_idx = 0
        self.frame_idx = 0
        self.is_playing = False
        self.qpos_err = None

    def get_start_record_dict(self, env=None, ctrl=None):
        """
        Return the initial-state payload that `start_record` will store.

        Parameters:
            env (Any | None): Input env. Defaults to `None`.
            ctrl (Any | None): Input ctrl. Defaults to `None`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        env = self.env if env is None else env
        return {
            "initial_qpos": np.asarray(env.data.qpos, dtype=np.float64).copy(),
            "initial_qvel": np.asarray(env.data.qvel, dtype=np.float64).copy(),
            "initial_ctrl": self._get_ctrl(ctrl=ctrl,env=env).copy(),
            "t0": self._get_sim_time(env=env),
        }

    def get_append_record_dict(self, env=None, ctrl=None):
        """
        Return the per-frame payload that `append` will store.

        Parameters:
            env (Any | None): Input env. Defaults to `None`.
            ctrl (Any | None): Input ctrl. Defaults to `None`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        self._assert_started()
        env = self.env if env is None else env
        sim_time = self._get_sim_time(env=env)
        record_dict = {
            "t": sim_time - self.t0,
            "sim_time": sim_time,
            "ctrl": self._get_ctrl(ctrl=ctrl,env=env).copy(),
        }
        if self.save_qpos_ref:
            record_dict["qpos_ref"] = np.asarray(env.data.qpos, dtype=np.float64).copy()
        if self.save_qvel_ref:
            record_dict["qvel_ref"] = np.asarray(env.data.qvel, dtype=np.float64).copy()
        return record_dict

    def summarize_record_dict(self, record_dict, max_items=6):
        """
        Return a compact printable summary of a replay record dict.

        Parameters:
            record_dict (Any): Input record dict.
            max_items (int): Input max items. Defaults to `6`.

        Returns:
            Any: Result computed from the current inputs.
        """
        summary = {}
        for key,value in record_dict.items():
            arr = np.asarray(value)
            if arr.ndim == 0:
                summary[key] = arr.item()
            else:
                flat = arr.reshape(-1)
                summary[key] = {
                    "shape": arr.shape,
                    "dtype": str(arr.dtype),
                    "head": flat[:max_items].tolist(),
                }
        return summary

    def set_start_record_dict(self, record_dict):
        """
        Store an initial-state record dict on this replay object.

        Parameters:
            record_dict (Any): Input record dict.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.initial_qpos = record_dict["initial_qpos"]
        self.initial_qvel = record_dict["initial_qvel"]
        self.initial_ctrl = record_dict["initial_ctrl"]
        self.t0 = record_dict["t0"]

    def append_record_dict(self, record_dict):
        """
        Append one per-frame record dict to this replay object.

        Parameters:
            record_dict (Any): Input record dict.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.t.append(record_dict["t"])
        self.sim_time.append(record_dict["sim_time"])
        self.ctrl.append(record_dict["ctrl"])
        if self.save_qpos_ref:
            self.qpos_ref.append(record_dict["qpos_ref"])
        if self.save_qvel_ref:
            self.qvel_ref.append(record_dict["qvel_ref"])

    def start_record(self, ctrl=None, env=None, record_dict=None, verbose=False):
        """
        Snapshot the initial state and start a fresh recording buffer.

        Parameters:
            ctrl (Any | None): Input ctrl. Defaults to `None`.
            env (Any | None): Input env. Defaults to `None`.
            record_dict (Any | None): Input record dict. Defaults to `None`.
            verbose (bool): Input verbose. Defaults to `False`.

        Returns:
            Any: Result computed from the current inputs.
        """
        self.clear()
        if record_dict is None:
            record_dict = self.get_start_record_dict(env=env,ctrl=ctrl)
        self.set_start_record_dict(record_dict)
        print("[MuJoCoReplay] start record")
        if verbose:
            print(self.summarize_record_dict(record_dict))
        return record_dict

    def append(self, ctrl=None, env=None, record_dict=None, verbose=False):
        """
        Append one frame using the current env state as reference.

        Parameters:
            ctrl (Any | None): Input ctrl. Defaults to `None`.
            env (Any | None): Input env. Defaults to `None`.
            record_dict (Any | None): Input record dict. Defaults to `None`.
            verbose (bool): Input verbose. Defaults to `False`.

        Returns:
            Any: Result computed from the current inputs.
        """
        if record_dict is None:
            record_dict = self.get_append_record_dict(env=env,ctrl=ctrl)
        self.append_record_dict(record_dict)
        if verbose:
            print(self.summarize_record_dict(record_dict))
        return record_dict

    def save(self, path_or_dir):
        """
        Save the replay data to an `.npz` file.

        Parameters:
            path_or_dir (str | Path): Input path or dir.

        Returns:
            Any: Result computed from the current inputs.
        """
        self._assert_started()
        if len(self.ctrl) == 0:
            print("[MuJoCoReplay] no frames to save")
            return None

        path = Path(path_or_dir)
        if path.suffix == "":
            path.mkdir(parents=True, exist_ok=True)
            path = path / ("%s.npz" % datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)

        payload = self._make_payload()
        np.savez_compressed(path, **payload)
        self.path = path
        print("[MuJoCoReplay] saved %d frames -> %s" % (len(self.ctrl), path))
        return path

    @classmethod
    def load(cls, env, path, strict=True):
        """
        Load replay data and validate it against `env`.

        Parameters:
            env (Any): Input env.
            path (str | Path): Input path.
            strict (bool): Input strict. Defaults to `True`.

        Returns:
            Any: Result computed from the current inputs.
        """
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        sim_nstep = int(data["sim_nstep"])
        record_hz = float(data["record_hz"])
        obj = cls(
            env           = env,
            sim_nstep     = sim_nstep,
            record_hz     = record_hz,
            save_qpos_ref = "qpos_ref" in data.files,
            save_qvel_ref = "qvel_ref" in data.files,
        )
        obj.path = path
        obj.initial_qpos = np.asarray(data["initial_qpos"], dtype=np.float64).copy()
        obj.initial_qvel = np.asarray(data["initial_qvel"], dtype=np.float64).copy()
        obj.initial_ctrl = np.asarray(data["initial_ctrl"], dtype=np.float64).copy()
        obj.t = list(np.asarray(data["t"], dtype=np.float64))
        obj.sim_time = list(np.asarray(data["sim_time"], dtype=np.float64))
        obj.ctrl = [row.copy() for row in np.asarray(data["ctrl"], dtype=np.float64)]
        obj.qpos_ref = (
            [row.copy() for row in np.asarray(data["qpos_ref"], dtype=np.float64)]
            if "qpos_ref" in data.files
            else []
        )
        obj.qvel_ref = (
            [row.copy() for row in np.asarray(data["qvel_ref"], dtype=np.float64)]
            if "qvel_ref" in data.files
            else []
        )
        obj.t0 = 0.0
        if strict:
            obj.validate_data(data)
        print("[MuJoCoReplay] loaded %d frames <- %s" % (obj.n_frame, path))
        return obj

    @property
    def n_frame(self):
        """
        Number of recorded control frames.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        return len(self.ctrl)

    def has_initial_state(self):
        """
        Return whether the replay has an initial qpos, qvel, and ctrl.

        Parameters:
            None.

        Returns:
            bool: Boolean result for the requested condition.
        """
        return (
            self.initial_qpos is not None
            and self.initial_qvel is not None
            and self.initial_ctrl is not None
        )

    def has_frames(self):
        """
        Return whether at least one control frame is recorded.

        Parameters:
            None.

        Returns:
            bool: Boolean result for the requested condition.
        """
        return self.n_frame > 0

    def is_playback_at_start(self):
        """
        Return whether playback is positioned at the first frame.

        Parameters:
            None.

        Returns:
            bool: Boolean result for the requested condition.
        """
        return self.play_idx == 0 and self.frame_idx == 0

    def reset(self):
        """
        Restore the recorded initial qpos, qvel, and ctrl.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self._assert_loaded_or_started()
        self.env.data.qpos[:] = self.initial_qpos
        self.env.data.qvel[:] = self.initial_qvel
        self.env.set_ctrl(self.initial_ctrl)
        self.env.forward(clip_position=False)

    def reset_playback(self, ref_env=None, set_qvel=False):
        """
        Reset replay state, current env, and optional target/reference env.

        Parameters:
            ref_env (Any | None): Input ref env. Defaults to `None`.
            set_qvel (bool): Input set qvel. Defaults to `False`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.reset()
        self.play_idx = 0
        self.frame_idx = 0
        self.is_playing = False
        if ref_env is not None:
            self.set_ctrl_target_frame(0, env=ref_env)
        self.qpos_err = self.get_qpos_error(0)
        return self.frame_idx

    def start_playback(self, ref_env=None, set_qvel=False, reset_at_start=True):
        """
        Start playback, restoring the initial state at the first play.

        Parameters:
            ref_env (Any | None): Input ref env. Defaults to `None`.
            set_qvel (bool): Input set qvel. Defaults to `False`.
            reset_at_start (bool): Input reset at start. Defaults to `True`.

        Returns:
            Any: Result computed from the current inputs.
        """
        self._assert_loaded_or_started()
        if reset_at_start and self.is_playback_at_start():
            self.reset_playback(ref_env=ref_env, set_qvel=set_qvel)
        self.is_playing = True
        return self.is_playing

    def toggle_playback(self, ref_env=None, set_qvel=False, reset_at_start=True):
        """
        Toggle playback state and return the new state.

        Parameters:
            ref_env (Any | None): Input ref env. Defaults to `None`.
            set_qvel (bool): Input set qvel. Defaults to `False`.
            reset_at_start (bool): Input reset at start. Defaults to `True`.

        Returns:
            Any: Result computed from the current inputs.
        """
        if self.is_playing:
            self.is_playing = False
        else:
            self.start_playback(
                ref_env        = ref_env,
                set_qvel       = set_qvel,
                reset_at_start = reset_at_start,
            )
        return self.is_playing

    def step_playback(self, ref_env=None, gravity_bodies=None, set_qvel=False):
        """
        Step one replay frame and advance the internal playback cursor.

        Parameters:
            ref_env (Any | None): Input ref env. Defaults to `None`.
            gravity_bodies (Any | None): Input gravity bodies. Defaults to `None`.
            set_qvel (bool): Input set qvel. Defaults to `False`.

        Returns:
            Any: Result computed from the current inputs.
        """
        frame_idx,ctrl = self.get_playback_ctrl()
        if gravity_bodies is not None and hasattr(self.env, "compensate_gravity"):
            self.env.compensate_gravity(list(gravity_bodies))
        self.env.step(ctrl=ctrl, nstep=self.sim_nstep)
        self.update_playback_after_step(frame_idx,ref_env=ref_env)
        return ctrl

    def step(self, frame_idx):
        """
        Replay one frame by stepping with recorded control.

        Parameters:
            frame_idx (int): Input frame idx.

        Returns:
            Any: Result computed from the current inputs.
        """
        frame_idx = self._clip_frame_idx(frame_idx)
        ctrl = self.get_ctrl(frame_idx)
        self.env.step(ctrl=ctrl, nstep=self.sim_nstep)
        return ctrl

    def get_ctrl(self, frame_idx):
        """
        Return a copy of the recorded control for one frame.

        Parameters:
            frame_idx (int): Input frame idx.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        frame_idx = self._clip_frame_idx(frame_idx)
        return np.asarray(self.ctrl[frame_idx], dtype=np.float64).copy()

    def get_playback_ctrl(self):
        """
        Return the current playback frame index and control vector.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        frame_idx = self._clip_frame_idx(self.play_idx)
        return frame_idx,self.get_ctrl(frame_idx)

    def update_playback_after_step(self, frame_idx, ref_env=None):
        """
        Update reference overlay, errors, and cursor after an external step.

        Parameters:
            frame_idx (int): Input frame idx.
            ref_env (Any | None): Input ref env. Defaults to `None`.

        Returns:
            Any: Result computed from the current inputs.
        """
        frame_idx = self._clip_frame_idx(frame_idx)
        if ref_env is not None:
            self.set_ctrl_target_frame(frame_idx,env=ref_env)
        self.qpos_err = self.get_qpos_error(frame_idx)
        self.frame_idx = frame_idx
        if frame_idx < self.n_frame - 1:
            self.play_idx = frame_idx + 1
        else:
            self.play_idx = frame_idx
            self.is_playing = False
        return self.frame_idx

    def set_ref_frame(self, frame_idx, env=None, set_qvel=False):
        """
        Set an env to the recorded reference frame for visualization.

        Parameters:
            frame_idx (int): Input frame idx.
            env (Any | None): Input env. Defaults to `None`.
            set_qvel (bool): Input set qvel. Defaults to `False`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if len(self.qpos_ref) == 0:
            raise RuntimeError("Replay has no qpos_ref")
        frame_idx = self._clip_frame_idx(frame_idx)
        env = self.env if env is None else env
        env.data.qpos[:] = self.qpos_ref[frame_idx]
        if set_qvel and len(self.qvel_ref) > 0:
            env.data.qvel[:] = self.qvel_ref[frame_idx]
        else:
            env.data.qvel[:] = 0.0
        env.forward(clip_position=False)

    def set_ctrl_target_frame(self, frame_idx, env=None, clip_position=True):
        """
        Set an env to recorded state, then overlay recorded control targets.

        Parameters:
            frame_idx (int): Input frame idx.
            env (Any | None): Input env. Defaults to `None`.
            clip_position (bool): Input clip position. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        frame_idx = self._clip_frame_idx(frame_idx)
        env = self.env if env is None else env
        if len(self.qpos_ref) > 0:
            env.data.qpos[:] = self.qpos_ref[frame_idx]
        else:
            env.data.qpos[:] = self.initial_qpos
        env.data.qvel[:] = 0.0
        ctrl = self.get_ctrl(frame_idx)
        qpos_trgt,joint_names_trgt = self.env.get_qpos_from_ctrl(
            ctrl,
            self.env.ctrl_names,
            clip_position = clip_position,
        )
        env.forward(qpos_trgt,joint_names_trgt)

    def get_qpos_error(self, frame_idx, ord=None):
        """
        Return norm between current env qpos and recorded qpos reference.

        Parameters:
            frame_idx (int): Input frame idx.
            ord (Any | None): Input ord. Defaults to `None`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if len(self.qpos_ref) == 0:
            return None
        frame_idx = self._clip_frame_idx(frame_idx)
        return float(np.linalg.norm(self.env.data.qpos - self.qpos_ref[frame_idx], ord=ord))

    def get_qvel_error(self, frame_idx, ord=None):
        """
        Return norm between current env qvel and recorded qvel reference.

        Parameters:
            frame_idx (int): Input frame idx.
            ord (Any | None): Input ord. Defaults to `None`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if len(self.qvel_ref) == 0:
            return None
        frame_idx = self._clip_frame_idx(frame_idx)
        return float(np.linalg.norm(self.env.data.qvel - self.qvel_ref[frame_idx], ord=ord))

    def validate_data(self, data):
        """
        Validate saved replay metadata against the current env.

        Parameters:
            data (Any): Input data.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        schema = str(np.asarray(data["schema_version"]).item())
        if schema != self.schema_version:
            raise ValueError("Unsupported replay schema: [%s]" % schema)
        checks = [
            ("nq", int(self.env.model.nq)),
            ("nv", int(self.env.model.nv)),
            ("nu", int(self.env.model.nu)),
        ]
        for key, value in checks:
            saved = int(np.asarray(data[key]).item())
            if saved != value:
                raise ValueError("Replay %s mismatch: saved [%d] env [%d]" % (key, saved, value))
        self._validate_names("joint_names", self.env.joint_names, data)
        self._validate_names("ctrl_names", self.env.ctrl_names, data)

    def _make_payload(self):
        """
        Make a dict payload of all replay data for saving to `.npz`.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        payload = {
            "schema_version": np.array(self.schema_version, dtype=str),
            "dt": np.array(float(self.env.dt), dtype=np.float64),
            "sim_nstep": np.array(self.sim_nstep, dtype=np.int32),
            "record_hz": np.array(self.record_hz, dtype=np.float64),
            "nq": np.array(int(self.env.model.nq), dtype=np.int32),
            "nv": np.array(int(self.env.model.nv), dtype=np.int32),
            "nu": np.array(int(self.env.model.nu), dtype=np.int32),
            "joint_names": np.array(self.env.joint_names, dtype=str),
            "ctrl_names": np.array(self.env.ctrl_names, dtype=str),
            "initial_qpos": self.initial_qpos.copy(),
            "initial_qvel": self.initial_qvel.copy(),
            "initial_ctrl": self.initial_ctrl.copy(),
            "t": np.array(self.t, dtype=np.float64),
            "sim_time": np.array(self.sim_time, dtype=np.float64),
            "ctrl": np.stack(self.ctrl, axis=0),
        }
        if self.save_qpos_ref:
            payload["qpos_ref"] = np.stack(self.qpos_ref, axis=0)
        if self.save_qvel_ref:
            payload["qvel_ref"] = np.stack(self.qvel_ref, axis=0)
        return payload

    def _get_ctrl(self, ctrl=None, env=None):
        """
        Return the control vector to record for one frame.

        Parameters:
            ctrl (Any | None): Input ctrl. Defaults to `None`.
            env (Any | None): Input env. Defaults to `None`.

        Returns:
            Any: Result computed from the current inputs.
        """
        env = self.env if env is None else env
        if ctrl is None:
            return np.asarray(env.data.ctrl, dtype=np.float64)
        return np.asarray(ctrl, dtype=np.float64)

    def _get_sim_time(self, env=None):
        """
        Return the simulation time for the current frame.

        Parameters:
            env (Any | None): Input env. Defaults to `None`.

        Returns:
            Any: Result computed from the current inputs.
        """
        env = self.env if env is None else env
        if hasattr(env, "get_sim_time"):
            return float(env.get_sim_time())
        return float(getattr(env.data, "time", 0.0))

    def _clip_frame_idx(self, frame_idx):
        """
        Clip the frame index to be within the valid range of recorded frames.

        Parameters:
            frame_idx (int): Input frame idx.

        Returns:
            Any: Result computed from the current inputs.
        """
        self._assert_loaded_or_started()
        if not self.has_frames():
            raise RuntimeError("Replay has no frames")
        return int(np.clip(int(frame_idx), 0, self.n_frame - 1))

    def _assert_started(self):
        """
        Assert that the replay has been started.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if not self.has_initial_state():
            raise RuntimeError("Call start_record(...) before appending or saving")

    def _assert_loaded_or_started(self):
        """
        Assert that the replay has been loaded or started.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if not self.has_initial_state():
            raise RuntimeError("Replay has no initial state")

    def _validate_names(self, key, env_names, data):
        """
        Validate that the saved names match the current env names for joints or
        controls.

        Parameters:
            key (Any): Input key.
            env_names (list[str]): Input env names.
            data (Any): Input data.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        saved_names = [str(x) for x in data[key]]
        env_names = [str(x) for x in env_names]
        if saved_names != env_names:
            raise ValueError("Replay %s mismatch" % key)
