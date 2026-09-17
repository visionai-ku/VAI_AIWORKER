"""Five-pose controller capture and optional constrained position calibration.

No robot commands, FK/IK, body tracking, or skeleton estimates live here. Robot
correspondences are explicit EEF points supplied by a separate FK caller.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .controller_mapping import ControllerMapper


POSE_LABELS = ("neutral", "forward", "side", "retracted", "hands_together")
SNAPSHOT_VERSION = 2
HANDS = ("left", "right")


def _positive(value, name):
    if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")
    return float(value)


def _proper_rotation(value, name):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (3, 3) or not np.all(np.isfinite(matrix))
            or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(matrix), 1., atol=1e-7)):
        raise ValueError(name + " must be a proper rotation")
    return matrix


class QuestPoseCalibration:
    """Collect both controllers for five labels, with equally weighted repeats.

    The caller owns countdowns and capture timing: begin_trial(label), repeatedly
    add_frame(frame, now), then finish_trial(). A rejected or cancelled attempt
    does not consume a repeat. finish_trial() returns accepted/reason/hand stats.
    Session/origin changes invalidate every trial and enter ABORTED until reset().
    Invalidated raw attempts remain in a bounded audit archive across reset().
    Calling add_frame between trials keeps reference and freshness checks live.
    """

    # Subclasses can reuse the same capture/quality checks with another fixed
    # set of labels. The personal-calibration schema and fitter stay five-pose.
    _pose_labels = POSE_LABELS

    def __init__(self, repeats=5, min_samples=20, min_duration_s=.4,
                 timeout_s=.2, trim_fraction=.2, max_samples_per_trial=240,
                 max_attempts=64, max_position_rms_m=.01,
                 max_position_p95_m=.025, max_rotation_rms_rad=np.deg2rad(5),
                 max_rotation_p95_rad=np.deg2rad(10),
                 max_repeat_position_rms_m=.025,
                 max_repeat_rotation_rms_rad=np.deg2rad(10),
                 max_head_translation_m=.025, max_head_rotation_rad=np.deg2rad(5),
                 max_sample_gap_s=.15, R_BQ=None):
        for name, value in (("repeats", repeats), ("min_samples", min_samples),
                            ("max_samples_per_trial", max_samples_per_trial), ("max_attempts", max_attempts)):
            if type(value) is not int or value < 1:
                raise ValueError(name + " must be a positive integer")
        if min_samples < 3 or max_samples_per_trial < min_samples or max_attempts < len(self._pose_labels) * repeats:
            raise ValueError("capture counts cannot cover the requested repeats")
        if isinstance(trim_fraction, bool) or not np.isfinite(trim_fraction) or not 0 <= trim_fraction < .4:
            raise ValueError("trim_fraction must be between 0 and .4")
        self.config = dict(repeats=repeats, min_samples=min_samples, min_duration_s=_positive(min_duration_s, "min_duration_s"),
                           timeout_s=_positive(timeout_s, "timeout_s"), trim_fraction=float(trim_fraction),
                           max_samples_per_trial=max_samples_per_trial, max_attempts=max_attempts)
        for name, value in (("max_position_rms_m", max_position_rms_m), ("max_position_p95_m", max_position_p95_m),
                            ("max_rotation_rms_rad", max_rotation_rms_rad), ("max_rotation_p95_rad", max_rotation_p95_rad),
                            ("max_repeat_position_rms_m", max_repeat_position_rms_m),
                            ("max_repeat_rotation_rms_rad", max_repeat_rotation_rms_rad),
                            ("max_head_translation_m", max_head_translation_m),
                            ("max_head_rotation_rad", max_head_rotation_rad),
                            ("max_sample_gap_s", max_sample_gap_s)):
            self.config[name] = _positive(value, name)
        self._explicit_rotation = None if R_BQ is None else _proper_rotation(R_BQ, "R_BQ").copy()
        self._reader = ControllerMapper(timeout_s=timeout_s, R_BQ=R_BQ)
        self.reset()

    def reset(self):
        self._archive_capture("capture_reset")
        self._reader.reset()
        self._reference = None
        self._trials = {label: [] for label in self._pose_labels}
        self._attempts = []
        self._samples = []
        self._active_label = None
        self._discarded_frames = {}
        self._trial_error = None
        self._input_valid, self._input_reason = False, "no_frame"
        self._fit = None
        self.state, self.reason = "IDLE", "ready_for_trial"
        self.created_at = datetime.now(timezone.utc).isoformat()
        return self.status()

    def _archive_capture(self, reason):
        if not hasattr(self, "_invalidated_attempts"):
            self._invalidated_attempts = []
            return
        attempts = copy.deepcopy(self._attempts)
        if self._active_label is not None:
            attempts.append({"accepted": False, "reason": reason, "label": self._active_label,
                             "repeat_index": len(self._trials[self._active_label]) + 1,
                             "sample_count": len(self._samples), "duration_s": self._duration(),
                             "hands": {}, "raw_samples": copy.deepcopy(self._samples)})
        for attempt in attempts:
            attempt.update(invalidated_reason=reason, reference=copy.deepcopy(self._reference),
                           capture_created_at=self.created_at)
        self._invalidated_attempts.extend(attempts)

    def _abort_reference(self, reason="reference_changed_all_capture_discarded"):
        self._archive_capture(reason)
        self._trials = {label: [] for label in self._pose_labels}
        self._attempts, self._samples = [], []
        self._active_label, self._fit = None, None
        self._reference = None
        self._trial_error = None
        self._input_valid, self._input_reason = False, reason
        self.state, self.reason = "ABORTED", reason

    def begin_trial(self, label):
        if label not in self._pose_labels:
            raise ValueError("Unknown pose label")
        if self.state == "ABORTED":
            raise ValueError("reset is required after a reference change")
        if self.state == "COLLECTING":
            raise ValueError("Finish or cancel the current trial first")
        if len(self._trials[label]) >= self.config["repeats"]:
            raise ValueError("This pose already has all repeats")
        if len(self._attempts) + len(self._invalidated_attempts) >= self.config["max_attempts"]:
            raise ValueError("Attempt limit reached; export and create a new collector")
        self._active_label, self._samples = label, []
        self._trial_error = None
        self.state, self.reason = "COLLECTING", "collecting"
        self._fit = None
        return self.status()

    def clear_pose(self, label):
        """Discard one pose's accepted repeats; preserve other poses and history."""
        if label not in self._pose_labels:
            raise ValueError("Unknown pose label")
        if self._active_label == label:
            self.cancel_trial("pose_cleared")
        self._trials[label], self._fit = [], None
        for attempt in self._attempts:
            if attempt["label"] == label and attempt["accepted"]:
                attempt["superseded"] = True
        if self.state != "ABORTED" and self.state != "COLLECTING":
            self.state, self.reason = "IDLE", "pose_cleared"
        return self.status()

    def _reject_input(self, reason):
        self._discarded_frames[reason] = self._discarded_frames.get(reason, 0) + 1
        self._input_valid, self._input_reason = False, reason
        if self.state == "COLLECTING" and self._trial_error is None:
            self._trial_error = "input_interrupted_" + reason
        self.reason = reason
        return self.status()

    @staticmethod
    def _head_pose(frame):
        head = frame.get("head")
        if not isinstance(head, dict):
            raise ValueError("head_pose_unavailable")
        diagnostic = frame.get("diagnostics", {})
        if not isinstance(diagnostic, dict):
            raise ValueError("invalid_head_diagnostics")
        if ("viewer_pose_available" in diagnostic and diagnostic["viewer_pose_available"] is not True
                or "viewer_pose_emulated" in diagnostic and diagnostic["viewer_pose_emulated"] is not False):
            raise ValueError("head_pose_unavailable")
        position = np.asarray(head.get("position_m"), dtype=float)
        quaternion = np.asarray(head.get("quaternion_xyzw"), dtype=float)
        if (position.shape != (3,) or not np.all(np.isfinite(position)) or np.max(np.abs(position)) > 100
                or quaternion.shape != (4,) or not np.all(np.isfinite(quaternion))
                or abs(np.linalg.norm(quaternion) - 1.) > .02):
            raise ValueError("invalid_head_pose")
        return {"position_m": position.tolist(), "quaternion_xyzw": Rotation.from_quat(quaternion).as_quat().tolist()}

    def add_frame(self, frame, now=None):
        now = time.monotonic() if now is None else float(now)
        if not np.isfinite(now):
            raise ValueError("now must be finite")
        if self.state == "ABORTED":
            return self.status()
        if isinstance(frame, dict) and self._reference is not None:
            key = (frame.get("session_id"), frame.get("origin_id"))
            if isinstance(key[0], str) and type(key[1]) is int and key != (
                    self._reference["session_id"], self._reference["origin_id"]):
                self._abort_reference()
                return self.status()
        intent = self._reader.read_intent(frame, now=now)
        if not intent["valid"]:
            return self._reject_input(intent["reason"])
        if not intent["fresh"]:
            reason = "duplicate_sequence"
            self._discarded_frames[reason] = self._discarded_frames.get(reason, 0) + 1
            return self.status()
        try:
            head = self._head_pose(frame)
        except (ValueError, TypeError, OverflowError):
            return self._reject_input("invalid_head_pose")
        if self._reference is None:
            # The persistent intent reader may have seen a malformed head before
            # this first accepted sample. Freeze axes from this validated head.
            reference_reader = ControllerMapper(timeout_s=self.config["timeout_s"], R_BQ=self._explicit_rotation)
            reference_intent = reference_reader.read_intent(frame, now=now)
            if not reference_intent["valid"]:
                return self._reject_input(reference_intent["reason"])
            self._reference = {"session_id": frame["session_id"], "origin_id": frame["origin_id"],
                               "R_OQ": reference_reader.R_BQ.tolist(), "head_at_reference": head,
                               "first_sequence": frame["seq"], "first_sample_time_ms": frame["sample_time_ms"]}
        else:
            reference_head = self._reference["head_at_reference"]
            translation = np.linalg.norm(np.asarray(head["position_m"]) - reference_head["position_m"])
            angle = (Rotation.from_quat(reference_head["quaternion_xyzw"]).inv()
                     * Rotation.from_quat(head["quaternion_xyzw"])).magnitude()
            if translation > self.config["max_head_translation_m"] or angle > self.config["max_head_rotation_rad"]:
                self._abort_reference("headset_moved_all_capture_discarded")
                return self.status()
        self._input_valid, self._input_reason = True, "ready"
        self.reason = "collecting" if self.state == "COLLECTING" else "ready_for_trial"
        if self.state != "COLLECTING":
            return self.status()
        if len(self._samples) >= self.config["max_samples_per_trial"]:
            self.reason = "sample_capacity_reached_finish_trial"
            return self.status()
        sample = {"seq": frame["seq"], "sample_time_ms": float(frame["sample_time_ms"]),
                  "received_at": float(frame["received_at"]), "observed_at": now,
                  "pose_age_s": float(intent["pose_age_s"]), "head": head, "hands": {}}
        if self._samples and (sample["sample_time_ms"] - self._samples[-1]["sample_time_ms"]) / 1000. > self.config["max_sample_gap_s"]:
            self._trial_error = self._trial_error or "valid_sample_gap_too_large"
        for hand in HANDS:
            data = frame["hands"][hand]
            sample["hands"][hand] = {"position_m": list(map(float, data["position_m"])),
                                      "quaternion_xyzw": Rotation.from_quat(data["quaternion_xyzw"]).as_quat().tolist()}
        self._samples.append(sample)
        self.reason = "collecting"
        return self.status()

    def _hand_stats(self, hand):
        positions = np.asarray([sample["hands"][hand]["position_m"] for sample in self._samples])
        rotations = Rotation.from_quat([sample["hands"][hand]["quaternion_xyzw"] for sample in self._samples])
        distances = np.linalg.norm(positions - np.median(positions, axis=0), axis=1)
        angles = (rotations.mean().inv() * rotations).magnitude()
        scores = distances / self.config["max_position_rms_m"] + angles / self.config["max_rotation_rms_rad"]
        count = max(3, int(np.ceil(len(positions) * (1 - self.config["trim_fraction"]))))
        retained = np.sort(np.argsort(scores, kind="stable")[:count])
        position_mean = positions[retained].mean(axis=0)
        rotation_mean = rotations[retained].mean()
        residuals = np.linalg.norm(positions[retained] - position_mean, axis=1)
        rotation_residuals = (rotation_mean.inv() * rotations[retained]).magnitude()
        return {"position_m": position_mean.tolist(), "quaternion_xyzw": rotation_mean.as_quat().tolist(),
                "retained_sample_indices": retained.tolist(), "retained_count": int(count),
                "position_rms_m": float(np.sqrt(np.mean(residuals**2))),
                "position_p95_m": float(np.percentile(residuals, 95)),
                "rotation_rms_rad": float(np.sqrt(np.mean(rotation_residuals**2))),
                "rotation_p95_rad": float(np.percentile(rotation_residuals, 95)),
                "raw_position_span_m": float(np.linalg.norm(np.ptp(positions, axis=0))),
                "raw_rotation_p95_rad": float(np.percentile(angles, 95))}

    def _pose_mean(self, trials):
        hands = {}
        rotation = np.asarray(self._reference["R_OQ"])
        for hand in HANDS:
            positions = np.asarray([trial["hands"][hand]["position_m"] for trial in trials])
            rotations = Rotation.from_quat([trial["hands"][hand]["quaternion_xyzw"] for trial in trials])
            mean_p, mean_r = positions.mean(axis=0), rotations.mean()
            hands[hand] = {"position_m": mean_p.tolist(), "operator_position_m": (rotation @ mean_p).tolist(),
                           "quaternion_xyzw": mean_r.as_quat().tolist(),
                           "position_repeat_rms_m": float(np.sqrt(np.mean(np.sum((positions - mean_p)**2, axis=1)))),
                           "rotation_repeat_rms_rad": float(np.sqrt(np.mean((mean_r.inv() * rotations).magnitude()**2)))}
        return {"repeat_count": len(trials), "hands": hands}

    def _end_attempt(self, report):
        self._attempts.append(copy.deepcopy(report))
        if report["accepted"]:
            self._trials[self._active_label].append(copy.deepcopy(report))
        self._samples, self._active_label = [], None
        self.state = "COMPLETE" if all(len(v) == self.config["repeats"] for v in self._trials.values()) else "IDLE"
        self.reason = report["reason"]
        return copy.deepcopy(report)

    def finish_trial(self):
        if self.state != "COLLECTING":
            raise ValueError("No trial is collecting")
        report = {"label": self._active_label, "repeat_index": len(self._trials[self._active_label]) + 1,
                  "sample_count": len(self._samples), "duration_s": self._duration(),
                  "raw_samples": copy.deepcopy(self._samples), "hands": {}, "accepted": False, "reason": "too_few_samples"}
        if self._trial_error is not None:
            report["reason"] = self._trial_error
            return self._end_attempt(report)
        if len(self._samples) < self.config["min_samples"]:
            return self._end_attempt(report)
        if self._duration() < self.config["min_duration_s"] - 1e-9:
            report["reason"] = "capture_duration_too_short"
            return self._end_attempt(report)
        report["hands"] = {hand: self._hand_stats(hand) for hand in HANDS}
        for hand, stats in report["hands"].items():
            for metric in ("position_rms_m", "position_p95_m", "rotation_rms_rad", "rotation_p95_rad"):
                if stats[metric] > self.config["max_" + metric]:
                    report["reason"] = "unstable_" + hand + "_" + metric
                    return self._end_attempt(report)
        aggregate = self._pose_mean(self._trials[self._active_label] + [report])
        for hand, stats in aggregate["hands"].items():
            if (stats["position_repeat_rms_m"] > self.config["max_repeat_position_rms_m"]
                    or stats["rotation_repeat_rms_rad"] > self.config["max_repeat_rotation_rms_rad"]):
                report["reason"] = "inconsistent_repeats_" + hand
                return self._end_attempt(report)
        report.update(accepted=True, reason="accepted")
        return self._end_attempt(report)

    def cancel_trial(self, reason="cancelled"):
        if self.state != "COLLECTING":
            return {"accepted": False, "reason": "no_active_trial"}
        return self._end_attempt({"accepted": False, "reason": str(reason), "label": self._active_label,
                                  "repeat_index": len(self._trials[self._active_label]) + 1,
                                  "sample_count": len(self._samples), "duration_s": self._duration(),
                                  "hands": {}, "raw_samples": copy.deepcopy(self._samples)})

    def _duration(self):
        return 0. if len(self._samples) < 2 else (self._samples[-1]["sample_time_ms"] - self._samples[0]["sample_time_ms"]) / 1000.

    def status(self):
        counts = {label: len(self._trials[label]) for label in self._pose_labels}
        return {"state": self.state, "reason": self.reason, "label": self._active_label,
                "repeat_index": None if self._active_label is None else counts[self._active_label] + 1,
                "counts": counts, "repeats": self.config["repeats"], "sample_count": len(self._samples),
                "duration_s": self._duration(), "ready_to_finish": len(self._samples) >= self.config["min_samples"]
                and self._duration() >= self.config["min_duration_s"] - 1e-9,
                "complete": all(count == self.config["repeats"] for count in counts.values()),
                "discarded_frames": dict(self._discarded_frames), "trial_error": self._trial_error,
                "input_valid": self._input_valid, "input_reason": self._input_reason,
                "attempt_count": len(self._attempts) + len(self._invalidated_attempts)}

    def snapshot(self):
        means = {label: self._pose_mean(trials) for label, trials in self._trials.items() if trials}
        if "neutral" in means:
            for pose in means.values():
                for hand in HANDS:
                    pose["hands"][hand]["delta_from_neutral_operator_m"] = (
                        np.asarray(pose["hands"][hand]["operator_position_m"])
                        - means["neutral"]["hands"][hand]["operator_position_m"]).tolist()
        if "hands_together" in means:
            distances = np.asarray([np.linalg.norm(
                np.asarray(trial["hands"]["left"]["position_m"]) - trial["hands"]["right"]["position_m"])
                for trial in self._trials["hands_together"]])
            means["hands_together"]["bilateral"] = {
                "controller_center_distance_m": float(distances.mean()),
                "controller_center_distance_repeat_rms_m": float(np.sqrt(np.mean((distances - distances.mean())**2))),
                "expected_zero": False,
                "note": "Tracked controller origins are separated even when controller bodies touch."}
        return {"type": "quest_pose_calibration", "version": SNAPSHOT_VERSION, "created_at": self.created_at,
                "pose_labels": list(self._pose_labels),
                "configuration": copy.deepcopy(self.config), "status": self.status(),
                "reference": copy.deepcopy(self._reference), "pose_means": means,
                "attempts": copy.deepcopy(self._attempts), "active_samples": copy.deepcopy(self._samples),
                "invalidated_attempts": copy.deepcopy(self._invalidated_attempts),
                "fit": copy.deepcopy(self._fit),
                "provenance": {"input": "validated WebXR controller grip poses; no body or skeleton tracking",
                               "operator_axes": "forward, left, up; rotated WebXR origin, not an anatomical origin",
                               "position_mean": "trimmed samples per trial; equal weight per accepted trial",
                               "orientation_mean": "scipy Rotation.mean per trial and across trial means",
                               "timestamps": "Quest sample_time_ms and PC monotonic received_at/observed_at; no absolute clock synchronization",
                               "limitations": "static controller poses do not determine elbow posture, orientation transfer or dynamic robot lag; vertical gain requires sufficient vertical variation"}}

    def fit_positions(self, robot_points, R_BO=None, **options):
        self._fit = fit_position_calibration(self.snapshot(), robot_points, R_BO=R_BO, **options)
        return copy.deepcopy(self._fit)


def fit_position_calibration(snapshot, robot_points, R_BO=None, min_axis_extent_m=.03,
                             max_condition=20., gain_bounds=(.05, 3.), max_rms_error_m=.02):
    """Fit positive diagonal gains, with fixed proper operator-to-robot axes.

    robot_points is {left/right: {label: [x,y,z] for label in POSE_LABELS}}
    in robot base metres. R_BO defaults to identity; this function never fits
    a free affine matrix, a reflection, robot orientations, or skeletal
    proportions. A version-2 capture must explicitly contain all five labels.
    """
    report = {"accepted": False, "reason": "capture_incomplete", "by_hand": {},
              "model": "robot_neutral + R_BO @ diag(scales) @ controller_delta_operator",
              "pose_labels": list(POSE_LABELS), "scale_axes": ["forward", "left", "up"],
              "error_labels": list(POSE_LABELS[1:])}
    if robot_points is None:
        report["reason"] = "robot_correspondences_missing"
        return report
    if (not isinstance(snapshot, dict) or snapshot.get("type") != "quest_pose_calibration"
            or type(snapshot.get("version")) is not int or snapshot["version"] != SNAPSHOT_VERSION):
        report["reason"] = "unsupported_capture_version"
        return report
    if (snapshot.get("pose_labels") != list(POSE_LABELS)
            or not isinstance(snapshot.get("pose_means"), dict)
            or any(label not in POSE_LABELS for label in snapshot["pose_means"])):
        report["reason"] = "pose_schema_mismatch"
        return report
    status, configuration = snapshot.get("status", {}), snapshot.get("configuration", {})
    if not isinstance(status, dict) or not isinstance(configuration, dict):
        return report
    repeats = configuration.get("repeats")
    counts = status.get("counts")
    if (status.get("complete") is not True or type(repeats) is not int or repeats < 1
            or not isinstance(counts, dict) or set(counts) != set(POSE_LABELS)
            or any(type(counts[label]) is not int or counts[label] != repeats for label in POSE_LABELS)
            or set(snapshot["pose_means"]) != set(POSE_LABELS)
            or any(not isinstance(snapshot["pose_means"][label], dict)
                   or type(snapshot["pose_means"][label].get("repeat_count")) is not int
                   or snapshot["pose_means"][label]["repeat_count"] != repeats for label in POSE_LABELS)):
        return report
    if (not isinstance(robot_points, dict)
            or any(not isinstance(robot_points.get(hand), dict)
                   or set(robot_points[hand]) != set(POSE_LABELS) for hand in HANDS)):
        report["reason"] = "robot_pose_schema_mismatch"
        return report
    rotation = np.eye(3) if R_BO is None else _proper_rotation(R_BO, "R_BO")
    min_axis_extent_m = _positive(min_axis_extent_m, "min_axis_extent_m")
    max_condition = _positive(max_condition, "max_condition")
    max_rms_error_m = _positive(max_rms_error_m, "max_rms_error_m")
    if len(gain_bounds) != 2 or not 0 < gain_bounds[0] < gain_bounds[1] or not np.all(np.isfinite(gain_bounds)):
        raise ValueError("gain_bounds must be finite and positive")
    for hand in HANDS:
        points = np.asarray([robot_points[hand][label] for label in POSE_LABELS], dtype=float)
        source = np.asarray([snapshot["pose_means"][label]["hands"][hand]["operator_position_m"] for label in POSE_LABELS], dtype=float)
        expected_shape = (len(POSE_LABELS), 3)
        if points.shape != expected_shape or source.shape != expected_shape or not np.all(np.isfinite(points)) or not np.all(np.isfinite(source)):
            raise ValueError("Each hand needs a finite controller and robot 3D point for every pose label")
        control = source[1:] - source[0]
        target = (rotation.T @ (points[1:] - points[0]).T).T
        singular = np.linalg.svd(control, compute_uv=False)
        target_singular = np.linalg.svd(target, compute_uv=False)
        condition = float(singular[0] / singular[-1]) if singular[-1] > 1e-12 else None
        axis_extents = np.max(np.abs(control), axis=0)
        unobservable_axes = [axis for axis, extent in zip(report["scale_axes"], axis_extents) if extent < min_axis_extent_m]
        item = {"accepted": False, "reason": "insufficient_axis_coverage", "scales": None,
                "errors_mm": None, "errors_mm_by_pose": None,
                "source_singular_values_m": singular.tolist(), "source_condition": condition,
                "source_axis_extents_m": axis_extents.tolist(), "unobservable_axes": unobservable_axes,
                "source_rank": int(np.linalg.matrix_rank(control)), "robot_rank": int(np.linalg.matrix_rank(target))}
        report["by_hand"][hand] = item
        if unobservable_axes:
            item["reason"] = "unobservable_operator_axis"
            continue
        if singular[-1] < min_axis_extent_m or target_singular[-1] <= 1e-8 or condition is None:
            continue
        if condition > max_condition:
            item["reason"] = "ill_conditioned_source"
            continue
        # Cross-covariance is 3x3 regardless of the number of landmarks. Its
        # handedness detects reflected correspondences without fitting an affine
        # transform or selecting an arbitrary three-landmark subset.
        if np.linalg.slogdet(control.T @ target)[0] <= 0:
            item["reason"] = "reflection_required"
            continue
        gains = np.sum(control * target, axis=0) / np.sum(control**2, axis=0)
        if np.any(gains < gain_bounds[0]) or np.any(gains > gain_bounds[1]):
            item["reason"] = "nonpositive_or_out_of_range_gain"
            continue
        predicted = (rotation @ (control * gains).T).T + points[0]
        errors = np.linalg.norm(predicted - points[1:], axis=1)
        rms = float(np.sqrt(np.mean(errors**2)))
        named_errors = {"neutral": 0., **dict(zip(POSE_LABELS[1:], map(float, errors * 1000)))}
        item.update(errors_mm=(errors * 1000).tolist(), errors_mm_by_pose=named_errors, rms_error_mm=rms * 1000,
                    max_error_mm=float(np.max(errors) * 1000), scales=gains.tolist())
        if rms > max_rms_error_m:
            item["reason"] = "position_residual_too_large"
            continue
        item.update(accepted=True, reason="accepted", scales=gains.tolist(), R_BO=rotation.tolist(),
                    robot_neutral_m=points[0].tolist(), controller_neutral_operator_m=source[0].tolist())
    report["accepted"] = all(item["accepted"] for item in report["by_hand"].values())
    report["reason"] = "accepted" if report["accepted"] else "one_or_more_hands_rejected"
    report["limitations"] = "Position fit only; robot correspondences must be reviewed externally. No elbow/orientation or dynamic calibration."
    return report
