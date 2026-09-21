"""Reset stage speeds and completion/failure transitions, without robot commands."""
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

runpy.run_path(str(Path(__file__).resolve().parents[1] / "package/init_project.py"))

from package.vr_buttons import ControllerButtonActions, RETRY_DELAY_S
from package.worker.fixed_quest_protocol import JOINT_NAMES


class ControllerButtonActionsTest(unittest.TestCase):
    def setUp(self):
        self.base_speed = np.full(len(JOINT_NAMES), np.deg2rad(10.))
        self.base_speed[0] = .01
        self.real = SimpleNamespace(
            home_speed_limits=self.base_speed.copy(),
            teleop_speed_limits=np.full(len(JOINT_NAMES), np.deg2rad(120.)),
            planner=SimpleNamespace(joint_names=JOINT_NAMES, homing=False),
            arm_mask=np.array([name.startswith("arm_") for name in JOINT_NAMES]),
            last_feedback={"q": np.zeros(len(JOINT_NAMES)),
                           "bridge": {"limits": {"home_joint_speed_rad_s": np.deg2rad(60.)}}},
            dry_run=False,
            env_measured=None, _grip_home_active=False, home_pending=False,
        )
        self.targets = []
        self.real.request_pose_target = lambda q: self.targets.append(q.copy())
        self.buttons = ControllerButtonActions()
        self.actions = []

    def start(self, lift_ok=True):
        with patch("package.vr_buttons.lifted_arm_pose",
                   return_value=(np.zeros(len(JOINT_NAMES)), lift_ok, .01)):
            self.buttons._start_reset(self.real)
        self.advance("IDLE")

    def advance(self, state, now=1., reason=""):
        self.actions.clear()
        self.buttons._advance(self.real, {"state": state, "reason": reason}, self.actions, now)

    def assert_speed(self, degrees):
        np.testing.assert_allclose(self.real.home_speed_limits[1:], np.deg2rad(degrees))
        self.assertEqual(self.real.home_speed_limits[0], .01)

    def tick(self, now=1., state="READY", **pressed):
        """One on_tick call with the given face buttons down."""
        self.actions.clear()
        frame = {"buttons": {"left": {"x": pressed.get("x", False), "y": pressed.get("y", False)},
                             "right": {"a": pressed.get("a", False), "b": pressed.get("b", False)}}}
        self.buttons(now, frame, {"state": state, "reason": ""}, self.real, self.actions)

    def test_x_and_y_act_on_a_single_press(self):
        # Measured Quest taps last 0.03-0.26 s, so a 1 s hold never fired for these.
        from types import SimpleNamespace
        pressed = []
        widget = SimpleNamespace(
            get_button_option=lambda label_text: "STOP",
            set_button_option=lambda label_text, option_text: pressed.append(option_text))
        self.buttons.record_control = SimpleNamespace(widget=widget)
        self.tick(x=True)
        self.tick(now=1.01)
        self.tick(now=1.02, y=True)
        self.assertEqual(pressed, ["REC", "DEL"])

    def test_a_requests_initial_pose_on_a_single_press(self):
        # Quest taps last well under the 1 s hold used by Y/B: measured presses were
        # 0.00-0.26 s, so A must fire on the press edge like the on-screen button.
        self.tick(a=True)
        self.assertEqual(self.actions, ["home"])
        self.assertIn("Initial pose", self.buttons.status)
        self.assertFalse(self.buttons.lock_input, "the guarded return does not lock teleop input")

    def test_held_a_requests_the_return_only_once(self):
        self.tick(now=1., a=True)
        self.assertEqual(self.actions, ["home"])
        for now in (1.01, 1.5, 3.):
            self.tick(now=now, a=True)
            self.assertEqual(self.actions, [], "holding A must not resend the request")
        self.tick(now=3.1)               # release
        self.tick(now=3.2, a=True)       # press again
        self.assertEqual(self.actions, ["home"])

    def test_a_without_robot_feedback_requests_nothing(self):
        self.real.last_feedback = None
        self.tick(a=True)
        self.assertEqual(self.actions, [])
        self.assertIn("로봇 피드백 없음", self.buttons.status)

    def test_a_is_ignored_while_the_reset_sequence_runs(self):
        self.start(lift_ok=False)
        self.assertTrue(self.buttons.lock_input)
        with patch.object(self.buttons, "_start_home") as start_home:
            self.tick(now=2., a=True)
        start_home.assert_not_called()

    def test_reset_only_accelerates_after_alignment_completes_and_restores_speed(self):
        # A worker ceiling above both stages' nominal speeds keeps the 1x/2x
        # split visible; the default 60 deg/s ceiling (other tests) saturates
        # both stages at the same value instead.
        self.real.last_feedback["bridge"]["limits"]["home_joint_speed_rad_s"] = np.deg2rad(200.)
        self.start()
        self.assert_speed(60)
        self.assertIn("위로", self.buttons.status)
        self.advance("HOME")
        self.assertIn("shutdown1", self.buttons.status)
        self.assert_speed(60)
        self.real.planner.homing = True
        self.advance("HOMING")  # Alignment has not reached its measured target yet.
        self.assertEqual(len(self.targets), 2)
        self.assertFalse(self.actions)
        self.assert_speed(60)
        self.real.planner.homing = False
        for label in ("shutdown2", "shutdown3", "shutdown4", "bringup start",
                      "bringup step1", "bringup step2", "bringup step3", "bringup step4"):
            self.advance("HOME")
            self.assertIn(label, self.buttons.status)
            self.assertEqual(self.actions, ["home"])
            self.assert_speed(120)
        self.advance("HOME")
        self.assertFalse(self.buttons.lock_input)
        self.assertIn("완료", self.buttons.status)
        np.testing.assert_array_equal(self.real.home_speed_limits, self.base_speed)
        self.start()  # A later reset must start slowly again.
        self.assert_speed(60)

    def test_skipped_lift_still_keeps_alignment_speed(self):
        self.start(lift_ok=False)
        self.assertIn("1/9 shutdown1", self.buttons.status)
        self.assert_speed(60)
        self.advance("HOME")
        self.assertIn("2/9 shutdown2", self.buttons.status)
        self.assert_speed(60)

    def test_retry_keeps_each_stage_speed(self):
        self.start(lift_ok=False)
        for label, speed in (("shutdown1", 60), ("shutdown2", 60)):
            target = self.targets[-1].copy()
            self.advance("FAULT", now=2., reason="stale_joint_state")
            self.assertIn("재시도", self.buttons.status)
            self.advance("IDLE", now=2. + RETRY_DELAY_S / 2)
            self.assertFalse(self.actions)
            self.advance("IDLE", now=2. + RETRY_DELAY_S)
            self.assertIn(label, self.buttons.status)
            self.assertEqual(self.actions, ["home"])
            self.assert_speed(speed)
            np.testing.assert_array_equal(self.targets[-1], target)
            self.advance("HOME", now=3.)

    def test_alignment_failure_never_starts_fast_stage(self):
        self.start(lift_ok=False)
        self.advance("FAULT", reason="collision guard blocked")
        self.assertEqual(len(self.targets), 1)
        self.assertFalse(self.buttons.lock_input)
        self.assertFalse(self.actions)
        np.testing.assert_array_equal(self.real.home_speed_limits, self.base_speed)

    def test_fast_stage_failure_restores_normal_home_speed(self):
        self.start(lift_ok=False)
        self.advance("HOME")
        self.assert_speed(60)
        self.advance("FAULT", reason="collision guard blocked")
        self.assertFalse(self.buttons.lock_input)
        np.testing.assert_array_equal(self.real.home_speed_limits, self.base_speed)

    def test_lower_teleop_limits_are_respected(self):
        self.real.teleop_speed_limits[1:] = np.deg2rad(40)
        self.start(lift_ok=False)
        self.assert_speed(40)
        self.advance("HOME")
        self.assert_speed(40)

    def test_missing_feedback_keeps_next_stage_and_its_speed(self):
        self.start(lift_ok=False)
        self.real.last_feedback = None
        self.advance("HOME")
        self.assertEqual(len(self.targets), 1)
        self.assert_speed(60)
        self.real.last_feedback = {"q": self.targets[-1],
                                   "bridge": {"limits": {"home_joint_speed_rad_s": np.deg2rad(60.)}}}
        self.advance("IDLE")
        self.assertIn("shutdown2", self.buttons.status)
        self.assert_speed(60)

    def test_old_worker_caps_every_step_without_skipping_the_fold(self):
        self.real.last_feedback["bridge"]["limits"]["home_joint_speed_rad_s"] = np.deg2rad(30.)
        self.start()
        for completed in range(10):
            self.assert_speed(30)
            self.assertIn("로봇 상한 30", self.buttons.status)
            self.advance("HOME")
        self.assertEqual(len(self.targets), 10)
        self.assertIn("완료", self.buttons.status)
        np.testing.assert_array_equal(self.real.home_speed_limits, self.base_speed)

    def test_missing_or_invalid_real_worker_limit_never_starts_reset(self):
        for cap in (None, True, "60", np.nan, np.inf, 0., -1.):
            with self.subTest(cap=cap):
                self.real.last_feedback["bridge"]["limits"]["home_joint_speed_rad_s"] = cap
                with patch("package.vr_buttons.lifted_arm_pose") as lift:
                    self.buttons._start_reset(self.real)
                lift.assert_not_called()
                self.assertFalse(self.buttons.lock_input)
                self.assertFalse(self.targets)
                self.assertIn("속도 제한 확인 불가", self.buttons.status)

    def test_dry_run_can_preview_fast_stages_without_worker_limit(self):
        self.real.dry_run = True
        self.real.last_feedback.pop("bridge")
        self.start(lift_ok=False)
        self.advance("HOME")
        self.assert_speed(120)


if __name__ == "__main__":
    unittest.main()
