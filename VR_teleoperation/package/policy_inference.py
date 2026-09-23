"""LeRobot policy inference for SG2: load a trained checkpoint and turn one
(robot state, camera frames) observation into a joint target in JOINT_NAMES order.

Works with any policy type lerobot.policies.factory.get_policy_class knows about
(ACT now; diffusion/pi0/etc. later need no changes here -- the checkpoint's own
config.json picks the class and its saved normalization stats). See
https://github.com/huggingface/lerobot (pinned lerobot==0.4.4 in requirements.txt).

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
        from lerobot.utils.utils import get_safe_torch_device

        checkpoint_path = str(checkpoint_path)
        cfg = PreTrainedConfig.from_pretrained(checkpoint_path)
        if device is not None:
            # Overrides the checkpoint's own saved device (e.g. "cuda" from the
            # training box) -- from_pretrained() below re-reads cfg.device to
            # decide where to load the weights, so this must happen first.
            cfg.device = str(device)
        policy_cls = get_policy_class(cfg.type)
        self.policy = policy_cls.from_pretrained(checkpoint_path, config=cfg)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            cfg, pretrained_path=checkpoint_path)
        self.device = get_safe_torch_device(cfg.device)
        self.task, self.robot_type = str(task), str(robot_type)
        self.checkpoint_path, self.policy_type = checkpoint_path, cfg.type
        self.joint_names = list(JOINT_NAMES) if joint_names is None else list(joint_names)
        if set(self.joint_names) - set(JOINT_NAMES):
            raise ValueError("joint_names has names outside JOINT_NAMES: %s"
                             % sorted(set(self.joint_names) - set(JOINT_NAMES)))
        self._joint_idx = [JOINT_NAMES.index(name) for name in self.joint_names]
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
        import cv2
        from lerobot.utils.control_utils import predict_action

        q = np.asarray(q, dtype=np.float32)
        if q.shape != (len(JOINT_NAMES),) or not np.all(np.isfinite(q)):
            raise ValueError("q must be a finite (%d,) vector in JOINT_NAMES order" % len(JOINT_NAMES))
        observation = {"observation.state": q[self._joint_idx]}
        for cam, bgr in frames.items():
            observation["observation.images.%s" % cam] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        action = predict_action(
            observation=observation, policy=self.policy, device=self.device,
            preprocessor=self.preprocessor, postprocessor=self.postprocessor,
            use_amp=bool(self.policy.config.use_amp),
            task=self.task if task is None else str(task), robot_type=self.robot_type)
        action = action.squeeze(0).to("cpu").numpy().astype(np.float32)
        full = q.copy()
        full[self._joint_idx] = action
        return full
