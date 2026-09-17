"""Hardware-free checks for facing coordinates and clutch/fault behavior."""

import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "package" / "controller_mapping.py"
SPEC = importlib.util.spec_from_file_location("controller_mapping", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ControllerMapper = MODULE.ControllerMapper


def frame(seq=0, now=100., **kwargs):
    result = {"version": 1, "session_id": "quest-session", "origin_id": 0,
              "seq": seq, "sample_time_ms": (now - 90.) * 1000., "received_at": now,
              "visibility": "visible", "head": {"position_m": [0., 1.5, 0.],
                                                    "quaternion_xyzw": [0., 0., 0., 1.]},
              "hands": {hand: {"position_m": [0.2 if hand == "left" else -0.2, 1., -0.5],
                                "quaternion_xyzw": [0., 0., 0., 1.], "valid": True,
                                "emulated": False, "grip": 0., "trigger": 0.}
                        for hand in MODULE.HANDS}}
    result.update(kwargs)
    return result


class ControllerMapperTest(unittest.TestCase):
    def setUp(self):
        self.mapper = ControllerMapper(scale=1.)
        self.eef = {hand: np.eye(4) for hand in MODULE.HANDS}
        self.eef["left"][:3, 3] = [0.3, 0.2, 0.9]
        self.eef["right"][:3, 3] = [0.3, -0.2, 0.9]

    def update(self, sample, enabled=True, now=None):
        return self.mapper.update(sample, self.eef, enabled,
                                  sample["received_at"] if now is None else now)

    def arm(self, hand="left"):
        self.assertEqual(self.update(frame())["state"], "READY")
        sample = frame(1, 100.01)
        sample["hands"][hand]["grip"] = 1.
        self.assertEqual(self.update(sample)["state"], "ACTIVE")
        return sample

    def test_facing_axes_and_no_left_right_swap(self):
        for axis, expected in enumerate(([0., 0.1, 0.], [0., 0., 0.1], [0.1, 0., 0.])):
            with self.subTest(axis=axis):
                self.mapper.reset()
                sample = self.arm()
                sample.update(seq=2, received_at=100.02, sample_time_ms=10020.)
                sample["hands"]["left"]["position_m"][axis] += 0.1
                result = self.update(sample)
                self.assertEqual(result["active"], ("left",))
                np.testing.assert_allclose(result["targets"]["left"][:3, 3],
                                           self.eef["left"][:3, 3] + expected)

    def test_relative_rotation_conjugates_axes(self):
        sample = self.arm()
        sample.update(seq=2, received_at=100.02, sample_time_ms=10020.)
        sample["hands"]["left"]["quaternion_xyzw"] = [0., np.sqrt(0.5), 0., np.sqrt(0.5)]
        result = self.update(sample)
        np.testing.assert_allclose(result["targets"]["left"][:3, :3],
                                   [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], atol=1e-12)

    def test_axis_gains_apply_after_facing_rotation_and_replace_scalar(self):
        gains = {"left": [.25, 1.2, .6], "right": [.25, .8, .4]}
        self.mapper = ControllerMapper(scale=.25, position_scales=gains)
        self.update(frame())
        sample = frame(1, 100.01)
        for hand in MODULE.HANDS:
            sample["hands"][hand]["grip"] = 1.
        anchored = self.update(sample)
        for hand in MODULE.HANDS:
            np.testing.assert_allclose(anchored["targets"][hand], self.eef[hand])
        sample.update(seq=2, received_at=100.02, sample_time_ms=10020.)
        for hand in MODULE.HANDS:
            sample["hands"][hand]["position_m"] = (
                np.asarray(sample["hands"][hand]["position_m"]) + [.04, .03, .02]).tolist()
        result = self.update(sample)
        for hand in MODULE.HANDS:
            np.testing.assert_allclose(result["targets"][hand][:3, 3] - self.eef[hand][:3, 3],
                                       np.asarray(gains[hand]) * [.02, .04, .03])
            np.testing.assert_allclose(result["targets"][hand][:3, :3], np.eye(3))
        with self.assertRaises(ValueError):
            self.mapper.set_position_scales({hand: [1., 1., 1.] for hand in MODULE.HANDS})

    def test_axis_gains_keep_workspace_cap_and_do_not_alias_input(self):
        gains = {hand: [.25, 3., 3.] for hand in MODULE.HANDS}
        self.mapper = ControllerMapper(scale=.25, position_scales=gains)
        gains["left"][1] = 0.
        exported = self.mapper.position_scales
        exported["left"][1] = 0.
        sample = self.arm()
        sample.update(seq=2, received_at=100.02, sample_time_ms=10020.)
        sample["hands"]["left"]["position_m"][0] += .1
        result = self.update(sample)
        self.assertEqual(result["clamped"], ("left",))
        np.testing.assert_allclose(result["targets"]["left"][:3, 3] - self.eef["left"][:3, 3],
                                   [0., .2, 0.], atol=1e-12)

    def test_axis_gains_validate_all_axes_before_assignment(self):
        for bad in (None, [1., 0., 1.], [1., np.nan, 1.], [1., 1.], [True, True, True], ["1", "1", "1"]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.mapper.set_position_scales({"left": [1., 1., 1.], "right": bad})
        for gains in self.mapper.position_scales.values():
            np.testing.assert_array_equal(gains, [1., 1., 1.])

    def test_signed_and_zero_z_preserve_forward_axes_and_orientation(self):
        for z, expected_z in ((-.5, -.05), (0., 0.), (.5, .05)):
            with self.subTest(z=z):
                self.mapper = ControllerMapper(position_scales={h: [2., .25, z] for h in MODULE.HANDS})
                sample = self.arm()
                sample.update(seq=2, received_at=100.02, sample_time_ms=10020.)
                sample["hands"]["left"]["position_m"][1] += .1
                sample["hands"]["left"]["position_m"][2] += .02
                result = self.update(sample)
                np.testing.assert_allclose(result["targets"]["left"][:3, 3] - self.eef["left"][:3, 3],
                                           [.04, 0., expected_z], atol=1e-12)
                np.testing.assert_allclose(result["targets"]["left"][:3, :3], np.eye(3))

    def test_negative_z_obeys_workspace_limit(self):
        self.mapper = ControllerMapper(position_scales={h: [2., .25, -3.] for h in MODULE.HANDS})
        sample = self.arm()
        sample.update(seq=2, received_at=100.02, sample_time_ms=10020.)
        sample["hands"]["left"]["position_m"][1] += .1
        result = self.update(sample)
        self.assertEqual(result["clamped"], ("left",))
        np.testing.assert_allclose(result["targets"]["left"][:3, 3] - self.eef["left"][:3, 3],
                                   [0., 0., -.2], atol=1e-12)

    def test_initial_held_grip_requires_release(self):
        sample = frame()
        sample["hands"]["left"]["grip"] = 1.
        self.assertEqual(self.update(sample)["state"], "WAIT_RELEASE")
        self.assertEqual(self.update(frame(1, 100.01))["state"], "READY")

    def test_intent_reader_uses_cached_buttons_and_source_age(self):
        reader = ControllerMapper()
        sample = frame()
        intent = reader.read_intent(sample, now=100.)
        self.assertTrue(intent["fresh"])
        self.assertTrue(intent["both_released"])
        duplicate = copy.deepcopy(sample)
        duplicate["received_at"] = 100.1
        for hand in duplicate["hands"].values():
            hand["grip"] = 1.
        intent = reader.read_intent(duplicate, now=100.1)
        self.assertFalse(intent["fresh"])
        self.assertTrue(intent["both_released"])
        self.assertEqual(intent["effective_sample_at"], 100.)
        delayed = frame(1, 100.15)
        delayed["sample_time_ms"] = 10020.
        intent = reader.read_intent(delayed, now=100.15)
        self.assertTrue(intent["valid"])
        self.assertAlmostEqual(intent["effective_sample_at"], 100.02)
        self.assertEqual(reader.read_intent(delayed, now=100.4)["reason"], "stale_pose")

    def test_authorized_held_activation_anchors_current_eef_without_fake_release(self):
        sample = frame()
        for hand in sample["hands"].values():
            hand["grip"] = 1.
            hand["position_m"][2] -= .15
        self.assertEqual(self.mapper.activate_held(sample, now=100.)["state"], "READY")
        result = self.update(sample)
        self.assertEqual(result["active"], ("left", "right"))
        for hand in MODULE.HANDS:
            np.testing.assert_allclose(result["targets"][hand], self.eef[hand], atol=1e-12)
        next_sample = copy.deepcopy(sample)
        next_sample.update(seq=1, received_at=100.01, sample_time_ms=10010.)
        next_sample["hands"]["left"]["position_m"][0] += .21
        self.assertEqual(self.update(next_sample)["reason"], "pose_jump_left")

    def test_held_activation_rejects_one_grip_untracked_and_stale_input(self):
        self.assertEqual(self.mapper.activate_held(frame(), now=100.)["reason"], "both_grips_required")
        for kind in ("untracked", "stale"):
            with self.subTest(kind=kind):
                self.mapper.reset()
                sample = frame()
                for hand in sample["hands"].values():
                    hand["grip"] = 1.
                if kind == "untracked":
                    sample["hands"]["right"]["emulated"] = True
                result = self.mapper.activate_held(sample, now=100.3 if kind == "stale" else 100.)
                self.assertEqual(result["state"], "HOLD")
                self.assertEqual(result["targets"], {})

    def test_nonidentity_controller_anchor_has_no_initial_rotation_jump(self):
        self.update(frame())
        sample = frame(1, 100.01)
        sample["hands"]["left"]["grip"] = 1.
        h = np.sqrt(0.5)
        sample["hands"]["left"]["quaternion_xyzw"] = [0., 0., h, h]
        np.testing.assert_allclose(self.update(sample)["targets"]["left"], self.eef["left"], atol=1e-12)
        sample.update(seq=2, received_at=100.02, sample_time_ms=10020.)
        s, c = np.sin(np.pi/8), np.cos(np.pi/8)
        # A world-X rotation after a pre-existing Z rotation maps to robot Y.
        sample["hands"]["left"]["quaternion_xyzw"] = [s*h, -s*h, c*h, c*h]
        np.testing.assert_allclose(self.update(sample)["targets"]["left"][:3, :3],
                                   [[h, 0., h], [0., 1., 0.], [-h, 0., h]], atol=1e-12)

    def test_release_stops_target_and_repress_reanchors(self):
        sample = self.arm()
        sample.update(seq=2, received_at=100.02, sample_time_ms=10020.)
        sample["hands"]["left"]["grip"] = 0.
        sample["hands"]["left"]["position_m"][2] += 2.
        self.assertEqual(self.update(sample)["targets"], {})
        self.eef["left"][:3, 3] += [0.05, 0., 0.]
        sample.update(seq=3, received_at=100.03, sample_time_ms=10030.)
        sample["hands"]["left"]["grip"] = 1.
        np.testing.assert_allclose(self.update(sample)["targets"]["left"], self.eef["left"])

    def test_releasing_one_hand_keeps_other_active(self):
        sample = self.arm()
        sample.update(seq=2, received_at=100.02, sample_time_ms=10020.)
        sample["hands"]["right"]["grip"] = 1.
        self.assertEqual(self.update(sample)["active"], ("left", "right"))
        sample.update(seq=3, received_at=100.03, sample_time_ms=10030.)
        sample["hands"]["left"]["grip"] = 0.
        self.assertEqual(self.update(sample)["active"], ("right",))

    def test_duplicates_keep_targets_but_do_not_renew_age_or_buttons(self):
        sample = self.arm()
        duplicate = copy.deepcopy(sample)
        duplicate["received_at"] = 100.1
        duplicate["hands"]["left"]["grip"] = 0.
        self.assertEqual(self.update(duplicate, now=100.1)["active"], ("left",))
        result = self.update(duplicate, now=100.25)
        self.assertEqual(result["state"], "HOLD")
        self.assertEqual(result["reason"], "stale_pose")

    def test_out_of_order_and_stale_new_sequence_hold(self):
        self.arm()
        self.assertEqual(self.update(frame(0, 100.02))["reason"], "out_of_order")
        self.mapper.reset()
        self.assertEqual(self.update(frame(), now=100.3)["reason"], "stale_pose")

    def test_transport_backlog_with_increasing_sequences(self):
        sample = self.arm()
        sample.update(seq=2, received_at=100.4, sample_time_ms=10020.)
        self.assertEqual(self.update(sample)["reason"], "stale_pose")
        sample.update(seq=3, received_at=100.41, sample_time_ms=10030.)
        sample["hands"]["left"]["grip"] = 0.
        self.assertEqual(self.update(sample)["state"], "HOLD")
        self.assertEqual(self.update(frame(4, 100.42))["state"], "READY")

    def test_reference_change_requires_fresh_release(self):
        for key, value in (("origin_id", 1), ("session_id", "new-session")):
            with self.subTest(key=key):
                self.mapper.reset()
                sample = self.arm()
                sample.update(seq=0, received_at=100.02, sample_time_ms=10020.)
                sample[key] = value
                self.assertEqual(self.update(sample)["reason"], "reference_changed")
                sample.update(seq=1, received_at=100.03, sample_time_ms=10030.)
                self.assertEqual(self.update(sample)["state"], "WAIT_RELEASE")
                sample.update(seq=2, received_at=100.04, sample_time_ms=10040.)
                sample["hands"]["left"]["grip"] = 0.
                self.assertEqual(self.update(sample)["state"], "READY")

    def test_tracking_visibility_and_emulation_faults_need_release(self):
        for field, value in (("valid", False), ("emulated", True), ("visibility", "hidden")):
            with self.subTest(field=field):
                self.mapper.reset()
                sample = self.arm()
                sample.update(seq=2, received_at=100.02, sample_time_ms=10020.)
                if field == "visibility":
                    sample[field] = value
                else:
                    sample["hands"]["right"][field] = value
                self.assertEqual(self.update(sample)["state"], "HOLD")
                sample = frame(3, 100.03)
                sample["hands"]["left"]["grip"] = 1.
                self.assertEqual(self.update(sample)["state"], "WAIT_RELEASE")
                self.assertEqual(self.update(frame(4, 100.04))["state"], "READY")

    def test_invalid_numbers_and_missing_hand_hold(self):
        for field, value in (("position_m", [np.nan, 0., 0.]),
                             ("quaternion_xyzw", [0., 0., 0., 0.]),
                             ("trigger", np.inf), ("grip", 1.1)):
            with self.subTest(field=field):
                self.mapper.reset()
                sample = frame()
                sample["hands"]["left"][field] = value
                self.assertEqual(self.update(sample)["state"], "HOLD")
        sample = frame()
        del sample["hands"]["right"]
        self.assertEqual(self.update(sample)["state"], "HOLD")

    def test_jump_requires_release_and_workspace_clamps(self):
        sample = self.arm()
        sample.update(seq=2, received_at=100.02, sample_time_ms=10020.)
        sample["hands"]["left"]["position_m"][0] += 0.5
        self.assertEqual(self.update(sample)["reason"], "pose_jump_left")
        self.mapper = ControllerMapper(scale=1., max_translation_m=0.05)
        sample = self.arm()
        sample.update(seq=2, received_at=100.02, sample_time_ms=10020.)
        sample["hands"]["left"]["position_m"][2] += 0.1
        result = self.update(sample)
        self.assertEqual(result["clamped"], ("left",))
        np.testing.assert_allclose(result["targets"]["left"][:3, 3],
                                   self.eef["left"][:3, 3] + [0.05, 0., 0.])

    def test_rotation_jump_and_invalid_eef_hold(self):
        sample = self.arm()
        sample.update(seq=2, received_at=100.02, sample_time_ms=10020.)
        sample["hands"]["left"]["quaternion_xyzw"] = [1., 0., 0., 0.]
        self.assertEqual(self.update(sample)["reason"], "pose_jump_left")
        self.mapper.reset()
        self.update(frame())
        self.eef["left"][0, 0] = -1.
        sample = frame(1, 100.01)
        sample["hands"]["left"]["grip"] = 1.
        self.assertEqual(self.update(sample)["reason"], "invalid_eef_left")

    def test_head_yaw_is_frozen_and_head_pitch_preserves_up(self):
        sample = frame()
        sample["head"]["quaternion_xyzw"] = [0., np.sqrt(0.5), 0., np.sqrt(0.5)]
        self.update(sample)
        rotation = self.mapper.R_BQ
        np.testing.assert_allclose(rotation @ [1., 0., 0.], [1., 0., 0.], atol=1e-12)
        np.testing.assert_allclose(rotation @ [0., 1., 0.], [0., 0., 1.], atol=1e-12)
        self.update(frame(1, 100.01))
        np.testing.assert_array_equal(self.mapper.R_BQ, rotation)
        self.mapper.reset()
        sample["head"]["quaternion_xyzw"] = [np.sin(0.2), 0., 0., np.cos(0.2)]
        self.update(sample)
        np.testing.assert_allclose(self.mapper.R_BQ, MODULE.FACING_R_BQ, atol=1e-12)

    def test_explicit_rotation_rejects_reflection_and_works_without_head(self):
        with self.assertRaises(ValueError):
            ControllerMapper(R_BQ=np.diag([-1., 1., 1.]))
        self.mapper = ControllerMapper(R_BQ=np.eye(3))
        sample = frame()
        del sample["head"]
        self.assertEqual(self.update(sample)["state"], "READY")

    def test_disabled_discards_anchors_and_trigger_only_when_active(self):
        sample = self.arm()
        sample.update(seq=2, received_at=100.02, sample_time_ms=10020.)
        sample["hands"]["left"]["trigger"] = 0.7
        self.assertEqual(self.update(sample)["triggers"], {"left": 0.7})
        self.assertEqual(self.update(sample, enabled=False)["state"], "DISABLED")
        self.assertEqual(self.update(sample)["state"], "WAIT_RELEASE")
        self.assertEqual(self.update(frame(3, 100.03))["triggers"], {})


if __name__ == "__main__":
    unittest.main()
