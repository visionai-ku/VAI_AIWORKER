"""Controller-only relative EEF targets for a stationary, facing Quest.

This module maps poses and clutch state only. Robot FK, IK, joint limits, and
simulation remain in the existing SG2/MuJoCo control code. Input quaternions use
WebXR's ``[x, y, z, w]`` order and input positions are metres.
"""

from __future__ import annotations

import time

import numpy as np


HANDS = ("left", "right")
FACING_R_BQ = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])


def _number(value, name):
    if isinstance(value, (bool, str)) or not np.isscalar(value):
        raise ValueError(name)
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(name)
    return value


def _rotation(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("invalid_quaternion")
    norm = np.linalg.norm(q)
    if abs(norm - 1.) > 0.02:
        raise ValueError("invalid_quaternion")
    x, y, z, w = q / norm
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def _check_rotation(rotation):
    rotation = np.asarray(rotation, dtype=float)
    if (rotation.shape != (3, 3) or not np.all(np.isfinite(rotation))
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6)):
        raise ValueError("rotation_must_be_proper")
    return rotation.copy()


def _position_scales(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != set(HANDS):
        raise ValueError("position_scales requires left and right XYZ gains")
    result = {}
    for hand in HANDS:
        raw = np.asarray(value[hand])
        if (raw.shape != (3,) or raw.dtype.kind not in "fiu"
                or not np.all(np.isfinite(raw)) or np.any(raw[:2] <= 0)):
            raise ValueError("position_scales requires positive X/Y and finite signed Z gains")
        result[hand] = raw.astype(float, copy=True)
    return result


def _rotvec(R):
    """Rotation matrix -> rotation vector (axis * angle), numerically safe near 0 and pi."""
    angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    norm = np.linalg.norm(axis)
    if norm < 1e-9:   # angle ~ pi: axis from the symmetric part
        axis = np.sqrt(np.clip((np.diag(R) + 1.0) / 2.0, 0.0, None))
        norm = np.linalg.norm(axis)
    return axis / norm * angle


def _rotmat(v):
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.eye(3)
    k = v / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


class ControllerMapper:
    """Map active grip poses to relative EEF targets, with release-to-rearm.

    With no explicit ``R_BQ``, the first fresh, released frame freezes headset
    yaw. Facing mapping then sends operator forward/left/up to robot
    forward/left/up. Head position and subsequent head motion have no effect.
    ``max_translation_m`` clamps each clutch's robot-space displacement; jump
    thresholds instead reject discontinuities and require both grips released.

    ``received_at`` and ``now`` use the host monotonic clock. Source timestamps
    use a different clock: the minimum receive-minus-source offset estimates
    extra transport delay within a session. Absolute first-packet latency is
    unobservable without clock synchronization. Duplicate sequences never
    refresh either age or button/pose data.
    """

    def __init__(self, scale=0.5, timeout_s=0.2, grip_threshold=0.5,
                 max_translation_m=0.20, R_BQ=None,
                 max_position_step_m=0.20, max_rotation_step_rad=np.pi/2,
                 position_scales=None, intent_matrices=None, intent_profile_id=None,
                 accept_emulated=False, rotation_deadband_rad=0.0):
        # Soft deadband on each clutch's relative controller rotation: rotations up to
        # this angle are ignored, larger ones apply only the excess (continuous, no jump).
        self.rotation_deadband_rad = float(rotation_deadband_rad)
        if not 0.0 <= self.rotation_deadband_rad <= np.pi / 6:
            raise ValueError("rotation_deadband_rad must be in 0..30 degrees")
        if type(accept_emulated) is not bool:
            raise ValueError("accept_emulated must be a bool")
        # Test mode: consume Quest's estimated (emulatedPosition) poses as-is.
        # Jump, workspace and timeout checks still apply to those poses.
        self.accept_emulated = accept_emulated
        self.scale = _number(scale, "invalid_scale")
        self.timeout_s = _number(timeout_s, "invalid_timeout")
        self.grip_threshold = _number(grip_threshold, "invalid_grip_threshold")
        # None disables the per-clutch displacement clamp entirely.
        self.max_translation_m = (np.inf if max_translation_m is None
                                  else _number(max_translation_m, "invalid_workspace"))
        self.max_position_step_m = _number(max_position_step_m, "invalid_jump_limit")
        self.max_rotation_step_rad = _number(max_rotation_step_rad, "invalid_jump_limit")
        if (min(self.scale, self.timeout_s, self.max_translation_m,
                self.max_position_step_m, self.max_rotation_step_rad) <= 0
                or not 0 < self.grip_threshold <= 1):
            raise ValueError("Mapper limits must be positive; grip threshold must be in (0, 1].")
        self._explicit_R_BQ = None if R_BQ is None else _check_rotation(R_BQ)
        # Optional absolute gains in robot forward/left/up axes, AFTER rotation.
        # These replace scale, rather than multiplying it a second time.
        self._position_scales = _position_scales(position_scales)
        self.intent_profile_id = intent_profile_id
        self._intent_matrices = {}
        for hand in HANDS:
            matrix = np.eye(3) if intent_matrices is None else np.asarray(intent_matrices[hand], float)
            if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                    or np.linalg.det(matrix) <= 0 or np.linalg.cond(matrix) > 5):
                raise ValueError("invalid_intent_matrix_" + hand)
            self._intent_matrices[hand] = matrix.copy()
        self.reset()

    @property
    def position_scales(self):
        return {hand: (np.full(3, self.scale) if self._position_scales is None
                       else self._position_scales[hand].copy()) for hand in HANDS}

    def set_position_scales(self, value):
        """Change robot-axis gains only between clutches; never move an anchor."""
        validated = _position_scales(value)
        if self._anchors:
            raise ValueError("Release controller anchors before changing position gains")
        self._position_scales = validated

    def set_intent_profile(self, matrices, profile_id):
        """Replace only calibration matrices between clutches, atomically."""
        if self._anchors:
            raise ValueError("Release controller anchors before changing calibration")
        validated = {}
        for hand in HANDS:
            matrix = np.eye(3) if matrices is None else np.asarray(matrices[hand], float)
            if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                    or np.linalg.det(matrix) <= 0 or np.linalg.cond(matrix) > 5):
                raise ValueError("invalid_intent_matrix_" + hand)
            validated[hand] = matrix.copy()
        self._intent_matrices = validated
        self.intent_profile_id = profile_id

    @property
    def R_BQ(self):
        """Frozen WebXR-reference to robot-base rotation, or None before setup."""
        return None if self._R_BQ is None else self._R_BQ.copy()

    def reset(self):
        self._key = None
        self._sample = None
        self._min_clock_offset = None
        self._R_BQ = None if self._explicit_R_BQ is None else self._explicit_R_BQ.copy()
        self._anchors = {}
        self._needs_release = True
        self._now = None

    def _age(self):
        if self._sample is None:
            return None
        return max(0., self._now - self._sample["received_at"],
                   self._now - self._sample["source_time"] - self._min_clock_offset)

    def _result(self, state, reason, targets=None, triggers=None, clamped=()):
        targets = {} if targets is None else targets
        return {"state": state, "reason": reason, "targets": targets,
                "triggers": {} if triggers is None else triggers,
                "active": tuple(targets), "pose_age_s": self._age(),
                "clamped": tuple(clamped),
                "intent_profile_id": self.intent_profile_id,
                "mapping_stages": getattr(self, "_mapping_stages", {}) if targets else {}}

    def _hold(self, reason):
        self.require_release()
        return self._result("HOLD", reason)

    def require_release(self):
        """Cancel both clutch anchors while preserving the fixed reference axes."""
        self._anchors.clear()
        self._needs_release = True

    def read_intent(self, frame, now=None):
        """Read validated grip intent without EEF targets; use a separate reader.

        This reader retains sequence/clock history independently of a planner's
        resets. Duplicate packets return cached buttons and are never fresh.
        """
        self._now = time.monotonic() if now is None else _number(now, "invalid_now")
        try:
            sample, fresh, changed = self._read_frame(frame)
            if changed:
                raise ValueError("reference_changed")
            if self._R_BQ is None:
                self._freeze_yaw(sample["head"])
        except (TypeError, ValueError, AttributeError, OverflowError) as exc:
            return dict(valid=False, fresh=False, reason=str(exc), pose_age_s=self._age(),
                        both_held=False, both_released=False)
        grips = [sample["hands"][hand]["grip"] >= self.grip_threshold for hand in HANDS]
        return dict(valid=True, fresh=fresh, reason="ready", pose_age_s=self._age(),
                    both_held=all(grips), both_released=not any(grips),
                    effective_sample_at=self._now - self._age())

    def activate_held(self, frame, now=None):
        """Initialize an explicitly authorized both-grip gesture at this pose.

        The caller owns gesture/recovery authorization. Input validation and
        fixed-yaw calibration remain mandatory; update() anchors fresh EEF poses
        on the following call, without synthesizing a released controller frame.
        """
        self._now = time.monotonic() if now is None else _number(now, "invalid_now")
        self.require_release()
        try:
            sample, _, changed = self._read_frame(frame)
            if changed:
                raise ValueError("reference_changed")
            if not all(sample["hands"][hand]["grip"] >= self.grip_threshold for hand in HANDS):
                raise ValueError("both_grips_required")
            if self._R_BQ is None:
                self._freeze_yaw(sample["head"])
        except (TypeError, ValueError, AttributeError, OverflowError) as exc:
            return self._hold(str(exc))
        self._needs_release = False
        return self._result("READY", "both_grips_initialized")

    def _read_frame(self, frame):
        if not isinstance(frame, dict):
            raise ValueError("no_frame")
        if type(frame.get("version")) is not int or frame["version"] != 1:
            raise ValueError("unsupported_version")
        session_id, origin_id, seq = (frame.get(name) for name in ("session_id", "origin_id", "seq"))
        if not isinstance(session_id, str) or not 0 < len(session_id) <= 128:
            raise ValueError("invalid_session")
        if type(origin_id) is not int or origin_id < 0 or type(seq) is not int or seq < 0:
            raise ValueError("invalid_sequence")
        key = (session_id, origin_id)
        if key == self._key:
            if seq < self._sample["seq"]:
                raise ValueError("out_of_order")
            if seq == self._sample["seq"]:
                if self._age() > self.timeout_s:
                    raise ValueError("stale_pose")
                return self._sample, False, False

        received_at = _number(frame.get("received_at"), "invalid_receive_time")
        source_time = _number(frame.get("sample_time_ms"), "invalid_sample_time") / 1000.
        if source_time < 0 or received_at > self._now + 0.01:
            raise ValueError("invalid_sample_time")
        if key == self._key and (source_time <= self._sample["source_time"]
                                 or received_at < self._sample["received_at"]):
            raise ValueError("out_of_order")
        offset = received_at - source_time
        same_session = self._key is not None and session_id == self._key[0]
        min_offset = min(self._min_clock_offset, offset) if same_session else offset
        age = max(0., self._now - received_at, self._now - source_time - min_offset)
        if age > self.timeout_s:
            raise ValueError("stale_pose")
        if frame.get("visibility") != "visible":
            raise ValueError("session_not_visible")

        hands = {}
        for hand in HANDS:
            data = frame.get("hands", {}).get(hand)
            # The Quest client marks estimated poses valid=False, emulated=True.
            estimated = (self.accept_emulated and isinstance(data, dict)
                         and data.get("emulated") is True and type(data.get("valid")) is bool)
            if not estimated:
                if not isinstance(data, dict) or data.get("valid") is not True:
                    raise ValueError("tracking_lost_" + hand)
                if data.get("emulated") is not False:
                    raise ValueError("emulated_pose_" + hand)
            position = np.asarray(data.get("position_m"), dtype=float)
            if (position.shape != (3,) or not np.all(np.isfinite(position))
                    or np.max(np.abs(position)) > 100.):
                raise ValueError("invalid_position_" + hand)
            grip = _number(data.get("grip"), "invalid_grip_" + hand)
            trigger = _number(data.get("trigger"), "invalid_trigger_" + hand)
            if not 0 <= grip <= 1 or not 0 <= trigger <= 1:
                raise ValueError("invalid_button_" + hand)
            hands[hand] = {"p": position.copy(), "R": _rotation(data.get("quaternion_xyzw")),
                           "grip": grip, "trigger": trigger, "emulated": estimated}

        changed = self._key is not None and key != self._key
        sample = {"seq": seq, "received_at": received_at, "source_time": source_time,
                  "hands": hands, "head": frame.get("head")}
        self._key, self._sample, self._min_clock_offset = key, sample, min_offset
        if changed:
            self._R_BQ = None if self._explicit_R_BQ is None else self._explicit_R_BQ.copy()
        return sample, True, changed

    def _freeze_yaw(self, head):
        if not isinstance(head, dict):
            raise ValueError("head_calibration_unavailable")
        rotation = _rotation(head.get("quaternion_xyzw"))
        z_axis = rotation[:, 2].copy()
        z_axis[1] = 0.
        if np.linalg.norm(z_axis) < 1e-4:
            raise ValueError("head_yaw_unavailable")
        z_axis /= np.linalg.norm(z_axis)
        y_axis = np.array([0., 1., 0.])
        yaw_rotation = np.column_stack((np.cross(y_axis, z_axis), y_axis, z_axis))
        self._R_BQ = FACING_R_BQ @ yaw_rotation.T

    def update(self, frame, eef_T_by_hand, enabled, now=None):
        """Return targets for active hands only; never move or solve a robot.

        Call as often as desired with the latest frame. Grip release returns no
        target for that arm, allowing the caller to hold its current joints.
        Fault recovery needs a newer valid sample with both grips released;
        pressing again anchors against the EEF poses supplied on that call.
        """
        self._now = time.monotonic() if now is None else _number(now, "invalid_now")
        self._mapping_stages = {}
        if not enabled:
            self._anchors.clear()
            self._needs_release = True
            return self._result("DISABLED", "disabled")
        previous = self._sample
        try:
            sample, fresh, changed = self._read_frame(frame)
        except (TypeError, ValueError, AttributeError, OverflowError) as exc:
            return self._hold(str(exc))
        if changed:
            return self._hold("reference_changed")
        hands = sample["hands"]
        if self._needs_release:
            if not fresh or any(hands[hand]["grip"] >= self.grip_threshold for hand in HANDS):
                return self._result("WAIT_RELEASE", "release_both_grips")
            if self._R_BQ is None:
                try:
                    self._freeze_yaw(sample["head"])
                except (TypeError, ValueError, OverflowError) as exc:
                    return self._hold(str(exc))
            self._needs_release = False
            return self._result("READY", "ready")

        targets, triggers, clamped = {}, {}, []
        for hand in HANDS:
            data = hands[hand]
            if data["grip"] < self.grip_threshold:
                self._anchors.pop(hand, None)
                continue
            if hand in self._anchors and fresh and previous is not None:
                old = previous["hands"][hand]
                angle = np.arccos(np.clip((np.trace(data["R"] @ old["R"].T) - 1.) / 2., -1., 1.))
                if (np.linalg.norm(data["p"] - old["p"]) > self.max_position_step_m
                        or angle > self.max_rotation_step_rad + 1e-9):
                    return self._hold("pose_jump_" + hand)
            if hand not in self._anchors:
                try:
                    eef_T = np.asarray(eef_T_by_hand[hand], dtype=float)
                    if (eef_T.shape != (4, 4) or not np.all(np.isfinite(eef_T))
                            or not np.allclose(eef_T[3], [0., 0., 0., 1.], atol=1e-6)):
                        raise ValueError("invalid_eef_" + hand)
                    _check_rotation(eef_T[:3, :3])
                except (KeyError, TypeError, ValueError):
                    return self._hold("invalid_eef_" + hand)
                self._anchors[hand] = {"p": data["p"].copy(), "R": data["R"].copy(),
                                       "eef_T": eef_T.copy()}
            anchor = self._anchors[hand]
            gains = self.scale if self._position_scales is None else self._position_scales[hand]
            operator_delta = self._R_BQ @ (data["p"] - anchor["p"])
            intent_delta = self._intent_matrices[hand] @ operator_delta
            requested_delta = gains * intent_delta
            delta = requested_delta.copy()
            length = np.linalg.norm(delta)
            if length > self.max_translation_m:
                delta *= self.max_translation_m / length
                clamped.append(hand)
            target = anchor["eef_T"].copy()
            target[:3, 3] += delta
            R_rel = self._R_BQ @ data["R"] @ anchor["R"].T @ self._R_BQ.T
            rotation = _rotvec(R_rel)
            rotation_angle = float(np.linalg.norm(rotation))
            if self.rotation_deadband_rad > 0:
                applied = max(0.0, rotation_angle - self.rotation_deadband_rad)
                R_rel = _rotmat(rotation * (applied / rotation_angle)) if applied > 0 else np.eye(3)
            target[:3, :3] = R_rel @ anchor["eef_T"][:3, :3]
            targets[hand], triggers[hand] = target, data["trigger"]
            self._mapping_stages[hand] = {
                "controller_delta_quest_m": data["p"] - anchor["p"],
                "operator_delta_m": operator_delta, "intent_delta_m": intent_delta,
                "gains": self.position_scales[hand], "R_BQ": self._R_BQ.copy(),
                "requested_delta_m": requested_delta, "bounded_delta_m": delta.copy(),
                "workspace_scale": min(1., self.max_translation_m / max(length, 1e-12)),
                "anchor_eef_m": anchor["eef_T"][:3, 3].copy(),
                "requested_eef_m": anchor["eef_T"][:3, 3] + requested_delta,
                "bounded_eef_m": target[:3, 3].copy(),
                "input_emulated": data["emulated"],
                "controller_rotation_deg": float(np.degrees(rotation_angle)),
                "rotation_deadband_deg": float(np.degrees(self.rotation_deadband_rad)),
            }
        return self._result("ACTIVE" if targets else "READY",
                            "workspace_clamped" if clamped else ("tracking" if targets else "ready"),
                            targets, triggers, clamped)
