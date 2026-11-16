"""LeRobot policy inference for SG2: load a trained checkpoint and turn one
(robot state, camera frames) observation into a joint target in JOINT_NAMES order.

Works with any policy type lerobot.policies.factory.get_policy_class knows about
(ACT now; diffusion/pi0/etc. later need no changes here -- the checkpoint's own
config.json picks the class and its saved normalization stats). See
https://github.com/huggingface/lerobot (pinned lerobot==0.6.1 in requirements.txt).

The observation layout matches build_features() in episode_recorder.py exactly, so
a policy trained on a dataset this repo recorded sees the same schema at inference
time: observation.state (JOINT_NAMES order, float32) + one observation.images.<cam>
per camera (RGB HWC uint8). Camera frames arrive over ZMQ as BGR (see decode_frame
in episode_recorder.py) and are converted the same way here.

This module only predicts; it does not talk to the robot or apply the collision /
arming checks the real and sim control loops already enforce on teleop input.
clip_to_joint_limits() guards only against a raw policy output leaving the
verified-safe joint range already used elsewhere (JOINT_LIMITS); it is not the
Cartesian safety box for autonomous rollouts, which is planned separately. A
caller wiring this into fixed_quest_real.py / fixed_quest_sim.py must still run
the predicted (and clipped) action through that loop's own guards before sending
it to the robot.

torch/lerobot are imported lazily inside PolicyRuntime so importing this module,
and unit-testing clip_to_joint_limits(), does not require the notebook's full
dependency stack (this repo's dev machine does not have lerobot/torch installed).
"""
from __future__ import annotations

import numpy as np

from package.worker.fixed_quest_protocol import JOINT_LIMITS, JOINT_NAMES

DEFAULT_TASK = "SG2 VR teleoperation"
DEFAULT_ROBOT_TYPE = "ffw_sg2"

_JOINT_LOWER = np.array([JOINT_LIMITS[name][0] for name in JOINT_NAMES], dtype=np.float64)
_JOINT_UPPER = np.array([JOINT_LIMITS[name][1] for name in JOINT_NAMES], dtype=np.float64)


def orient_frame(cam, frame, expected_hw):
    """Turn a live camera frame into the orientation the checkpoint was trained on.

    The wrist cameras stream landscape 240x424 frames with the gripper fingers on the
    left. Datasets made from 2026-11-10 on (the sim datasets v0.2 / v0.2-tuft and new
    real recordings) store the wrist images turned 90 degrees counter-clockwise -- portrait
    424x240, fingers at the bottom -- with np.rot90(frame) (k=1), the same call as the
    sim export (export_lerobot.py --wrist-rotate ccw). Older checkpoints (real v0.1,
    e.g. pi05_full_v01_30k_r2) were trained on the landscape frames.

    The policy resizes and pads every image to its own input size, so a frame in the
    wrong orientation would not raise; it would just be seen sideways. Hence this
    decides from the checkpoint's own image shape: a wrist frame whose (h, w) is the
    expected (w, h) is turned with np.rot90(frame); anything else, the head camera
    included, passes unchanged.

    cam: camera name (e.g. "wrist_left"); frame: HWC array; expected_hw: (h, w) from
    the checkpoint's observation.images.<cam> feature, or None when unknown.
    -> (frame, rotated)
    """
    if (expected_hw is not None and cam.startswith("wrist") and frame.shape[0] != frame.shape[1]
            and tuple(frame.shape[:2]) == (expected_hw[1], expected_hw[0])):
        return np.ascontiguousarray(np.rot90(frame)), True
    return frame, False


def clip_to_joint_limits(q):
    """Clip a predicted joint target into JOINT_LIMITS (the same bounds
    fixed_quest_real.py already enforces on measured state via self.bounds).

    Returns (clipped, exceeded): `clipped` is a (len(JOINT_NAMES),) float32 array,
    unchanged where `q` was already inside range. `exceeded` maps joint name ->
    signed excess (positive above the upper limit, negative below the lower one)
    for every joint the raw prediction violated; empty when nothing was clipped.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (len(JOINT_NAMES),) or not np.all(np.isfinite(q)):
        raise ValueError("predicted action must be a finite (%d,) vector" % len(JOINT_NAMES))
    clipped = np.clip(q, _JOINT_LOWER, _JOINT_UPPER)
    exceeded = {name: float(q[i] - _JOINT_UPPER[i] if q[i] > _JOINT_UPPER[i] else q[i] - _JOINT_LOWER[i])
                for i, name in enumerate(JOINT_NAMES) if clipped[i] != q[i]}
    return clipped.astype(np.float32), exceeded


class PolicyRuntime:
    """Loads one LeRobot policy checkpoint and predicts joint targets from it.

    joint_names: the JOINT_NAMES subset this checkpoint's observation.state/action
    actually cover, in the order its dataset was built with (e.g. a checkpoint
    trained with lift/head excluded, see review_dataset.ipynb's arm-only rebuild).
    A trained checkpoint's config.json only stores feature shapes, not names, so
    this cannot be recovered from the checkpoint itself -- the caller must know
    which dataset it was trained on. Defaults to the full JOINT_NAMES (every
    joint), matching a checkpoint trained on episode_recorder.py's default
    19-joint schema with no caller-side change needed.
    """

    def __init__(self, checkpoint_path, device=None, task=DEFAULT_TASK, robot_type=DEFAULT_ROBOT_TYPE,
                 joint_names=None):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        from lerobot.utils.device_utils import get_safe_torch_device

        checkpoint_path = str(checkpoint_path)
        cfg = PreTrainedConfig.from_pretrained(checkpoint_path)
        if device is not None:
            # Overrides the checkpoint's own saved device (e.g. "cuda" from the
            # training box) -- from_pretrained() below re-reads cfg.device to
            # decide where to load the weights, so this must happen first.
            cfg.device = str(device)
        policy_cls = get_policy_class(cfg.type)
        self.policy = policy_cls.from_pretrained(checkpoint_path, config=cfg)
        # The saved processors pin their own device (policy_preprocessor.json keeps
        # "cuda" from the training box), so a device override must reach them too,
        # the same way lerobot's own rollout/policy_server do it.
        device_override = {"device": cfg.device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            cfg, pretrained_path=checkpoint_path,
            preprocessor_overrides={"device_processor": device_override},
            postprocessor_overrides={"device_processor": device_override})
        self.device = get_safe_torch_device(cfg.device)
        self.task, self.robot_type = str(task), str(robot_type)
        self.checkpoint_path, self.policy_type = checkpoint_path, cfg.type
        self.joint_names = list(JOINT_NAMES) if joint_names is None else list(joint_names)
        if set(self.joint_names) - set(JOINT_NAMES):
            raise ValueError("joint_names has names outside JOINT_NAMES: %s"
                             % sorted(set(self.joint_names) - set(JOINT_NAMES)))
        self._joint_idx = [JOINT_NAMES.index(name) for name in self.joint_names]
        # (h, w) each camera's images had in training; live frames are turned to match (orient_frame).
        self._image_hw = {key.rsplit(".", 1)[-1]: tuple(feature.shape[1:3])
                          for key, feature in self.policy.config.input_features.items()
                          if key.startswith("observation.images.")}
        self._orient_reported = set()
        state_dim = self.policy.config.input_features["observation.state"].shape[0]
        action_dim = self.policy.config.output_features["action"].shape[0]
        if state_dim != len(self.joint_names) or action_dim != len(self.joint_names):
            raise ValueError(
                "joint_names has %d entries but the checkpoint's observation.state/action are "
                "%d/%d-dimensional -- pass the exact JOINT_NAMES subset (and order) the training "
                "dataset used." % (len(self.joint_names), state_dim, action_dim))

    def reset(self):
        """Call once before each new episode/session: clears ACT's action queue /
        temporal ensemble and any pre/postprocessor running state (e.g. history)."""
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def warmup(self):
        """Run one throwaway prediction on a synthetic observation.

        The very first predict() of a process pays a one-off cost (CUDA kernel
        load / lazy init) that measured ~200ms here against ~1ms steady state.
        Inside the control loop that single spike trips the "control loop delay"
        guard (SG2FixedQuestReal faults when a tick exceeds MAX_LOOP_DT_S=0.10s),
        so the session dies the moment the policy is first engaged. Paying it up
        front, while the robot is still disabled, keeps every in-loop tick fast.
        Cheap to call again; later predicts are already warm.
        """
        shapes = {key: feature.shape
                  for key, feature in self.policy.config.input_features.items()
                  if key.startswith("observation.images.")}
        frames = {key.rsplit(".", 1)[-1]: np.zeros((shape[1], shape[2], 3), dtype=np.uint8)
                  for key, shape in shapes.items()}
        self.predict(np.zeros(len(JOINT_NAMES), dtype=np.float32), frames)
        self.reset()

    def predict(self, q, frames, task=None):
        """q: (len(JOINT_NAMES),) measured state, JOINT_NAMES order.
        frames: {camera_name: HWC BGR uint8}, one entry per camera this checkpoint
        was trained with (see episode_recorder.DEFAULT_CAMERAS / build_features).

        -> raw predicted joint target, JOINT_NAMES order, float32, NOT yet safety
        clipped -- pass it through clip_to_joint_limits() before sending it on.
        Joints outside self.joint_names (e.g. lift/head excluded from a reduced
        checkpoint) are held at their current measured value from `q`, not
        predicted -- the caller's guard/collision pipeline still sees a full
        19-joint target either way.
        """
        from lerobot.common.control_utils import predict_action

        q, observation = self._observation(q, frames)
        action = predict_action(
            observation=observation, policy=self.policy, device=self.device,
            preprocessor=self.preprocessor, postprocessor=self.postprocessor,
            use_amp=bool(self.policy.config.use_amp),
            task=self.task if task is None else str(task), robot_type=self.robot_type)
        action = action.squeeze(0).to("cpu").numpy().astype(np.float32)
        full = q.copy()
        full[self._joint_idx] = action
        return full

    @property
    def joint_indices(self):
        """Positions of self.joint_names inside JOINT_NAMES."""
        return list(self._joint_idx)

    def predict_chunk(self, q, frames, task=None):
        """Predict the policy's whole next action chunk from one observation.

        Same inputs as predict(), but bypasses the policy's own select_action()
        queue: preprocess -> predict_action_chunk() -> postprocess in one call, so
        AsyncChunkPolicy can run it off the control thread while the previous
        chunk is still being executed.

        -> (n_action_steps, len(self.joint_names)) float32, self.joint_names order,
        NOT yet safety clipped. Unlike predict(), joints outside self.joint_names
        are not filled in here -- the caller holds them at the value measured when
        each step is actually used.
        """
        from contextlib import nullcontext
        import torch
        from lerobot.policies import prepare_observation_for_inference

        _, observation = self._observation(q, frames)
        use_amp = bool(self.policy.config.use_amp) and self.device.type == "cuda"
        with torch.inference_mode(), torch.autocast(device_type=self.device.type) if use_amp else nullcontext():
            batch = prepare_observation_for_inference(
                observation, self.device, self.task if task is None else str(task), self.robot_type)
            batch = self.preprocessor(batch)
            chunk = self.policy.predict_action_chunk(batch)[:, : self.policy.config.n_action_steps]
            chunk = self.postprocessor(chunk)
        return chunk.squeeze(0).to("cpu").numpy().astype(np.float32)

    def _observation(self, q, frames):
        import cv2

        q = np.asarray(q, dtype=np.float32)
        if q.shape != (len(JOINT_NAMES),) or not np.all(np.isfinite(q)):
            raise ValueError("q must be a finite (%d,) vector in JOINT_NAMES order" % len(JOINT_NAMES))
        observation = {"observation.state": q[self._joint_idx]}
        for cam, bgr in frames.items():
            in_hw = tuple(bgr.shape[:2])
            bgr, rotated = orient_frame(cam, bgr, self._image_hw.get(cam))
            if (cam, in_hw) not in self._orient_reported:
                # Once per camera and frame size, so the orientation can be checked before a robot run.
                self._orient_reported.add((cam, in_hw))
                print("[policy] %s: frame %dx%d -> %s (checkpoint %s)" % (
                    cam, in_hw[0], in_hw[1],
                    "rotated 90 deg CCW to %dx%d" % bgr.shape[:2] if rotated else "unchanged",
                    "%dx%d" % self._image_hw[cam] if cam in self._image_hw else "has no such camera"))
            observation["observation.images.%s" % cam] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return q, observation


class AsyncChunkPolicy:
    """Runs PolicyRuntime.predict_chunk() on a background thread so the control
    loop never waits on the model.

    PolicyRuntime.predict() goes through the policy's select_action(), which
    computes a whole chunk synchronously whenever its queue runs dry. For ACT that
    is ~1ms, but a VLA policy (pi05, MolmoAct2) measured 140-175ms per chunk on the
    RTX 6000 Ada -- longer than MAX_LOOP_DT_S (0.10s), so the tick that refills the
    queue faults the session ("control loop delay"). Here the next chunk is
    requested while `lead` steps of the current one are still queued, and each
    tick only pops an already computed step.

    When a chunk arrives, its first steps -- those whose time already passed while
    it was being computed, counted in steps executed since its observation was
    taken -- are dropped and the rest replaces the queue. `lead` adapts to the
    measured inference time (see _update_lead). If the queue still runs dry, the
    last step is repeated (the robot holds its last target) until the next chunk
    arrives. Before the first chunk of a session/episode, predict() returns None
    and the caller holds position.

    Same predict()/reset()/warmup() interface as PolicyRuntime, so
    PolicyDeployment uses either one.
    """

    def __init__(self, runtime, hz, lead=None):
        from concurrent.futures import ThreadPoolExecutor

        self.runtime, self.hz = runtime, float(hz)
        self.fixed_lead = lead
        self.lead = 1 if lead is None else int(lead)
        # One worker: chunk requests and reset() run strictly in submission order.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-chunk")
        self._latencies = []
        self._generation = 0
        self._clear()

    def __getattr__(self, name):
        # policy_type, device, checkpoint_path, policy, ... of the wrapped runtime.
        if name == "runtime":
            raise AttributeError(name)
        return getattr(self.runtime, name)

    def _clear(self):
        self._queue, self._last, self._future = [], None, None
        self._executed = 0          # queued steps executed since reset()
        self.starved_steps = 0      # ticks that repeated the last step for lack of a chunk

    def reset(self):
        """Start a new episode: drop the queue and any chunk still being computed.

        Never blocks: the runtime's own reset is queued behind the in-flight chunk
        on the same worker thread, and that chunk's result is discarded."""
        self._generation += 1
        self._executor.submit(self.runtime.reset)
        self._clear()

    def warmup(self):
        """Pay the one-off first-call cost and measure the chunk latency that sets `lead`.

        Blocks until done; runs on the worker thread, behind anything still in flight."""
        self._executor.submit(self._warmup).result()

    def _warmup(self):
        import time

        self.runtime.warmup()
        shapes = {key.rsplit(".", 1)[-1]: feature.shape
                  for key, feature in self.runtime.policy.config.input_features.items()
                  if key.startswith("observation.images.")}
        frames = {cam: np.zeros((shape[1], shape[2], 3), dtype=np.uint8) for cam, shape in shapes.items()}
        start = time.monotonic()
        self.runtime.predict_chunk(np.zeros(len(JOINT_NAMES), dtype=np.float32), frames)
        self._update_lead(time.monotonic() - start, chunk_len=self.runtime.policy.config.n_action_steps)
        self.runtime.reset()

    def _update_lead(self, latency_s, chunk_len):
        self._latencies = (self._latencies + [float(latency_s)])[-10:]
        if self.fixed_lead is None:
            # 1.5x the slowest recent chunk (the control thread competes for the GIL
            # while it runs), plus two ticks of margin.
            steps = int(np.ceil(max(self._latencies) * self.hz * 1.5)) + 2
            self.lead = int(min(max(steps, 1), max(chunk_len - 1, 1)))

    def _compute(self, generation, executed, q, frames, task):
        import time

        start = time.monotonic()
        chunk = self.runtime.predict_chunk(q, frames, task=task)
        return generation, executed, chunk, time.monotonic() - start

    def predict(self, q, frames, task=None):
        """q: (len(JOINT_NAMES),) measured state, JOINT_NAMES order; frames as in
        PolicyRuntime.predict().

        -> next joint target, JOINT_NAMES order, float32, NOT yet safety clipped,
        joints outside runtime.joint_names held at their value in `q`; or None
        while the first chunk of this session/episode is still being computed.
        Raises whatever the background prediction raised.
        """
        q = np.asarray(q, dtype=np.float32)
        if self._future is not None and self._future.done():
            generation, executed, chunk, latency = self._future.result()
            self._future = None
            if generation == self._generation:
                self._update_lead(latency, chunk_len=len(chunk))
                fresh = list(chunk[self._executed - executed:])
                if fresh:
                    self._queue = fresh
        if self._future is None and len(self._queue) <= self.lead:
            snapshot = {cam: np.array(bgr, copy=True) for cam, bgr in frames.items()}
            self._future = self._executor.submit(
                self._compute, self._generation, self._executed, q.copy(), snapshot, task)
        if self._queue:
            self._last = self._queue.pop(0)
            self._executed += 1
        elif self._last is None:
            return None
        else:
            self.starved_steps += 1
        full = q.copy()
        full[self.runtime.joint_indices] = self._last
        return full

    def close(self):
        """Stop the worker thread. An in-flight chunk is left to finish on its own."""
        self._generation += 1
        self._executor.shutdown(wait=False, cancel_futures=True)
