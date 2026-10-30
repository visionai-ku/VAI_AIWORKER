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
                           "bridge": {"limits": {"home_joint_speed_rad_s": np.deg2rad(100.)}}},
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

    def recording_control(self, prev="REC"):
        """A RecordControl stand-in whose prev turns STOP only once its (synchronous) save ran."""
        options = []
        widget = SimpleNamespace(get_button_option=lambda label_text: options[-1] if options else "REC",
                                 set_button_option=lambda label_text, option_text: options.append(option_text))
        self.buttons.record_control = SimpleNamespace(widget=widget, prev=prev)
        return self.buttons.record_control

    def ready_tick(self, now, ready=True, **pressed):
        self.actions.clear()
        frame = {"buttons": {"left": {"x": pressed.get("x", False), "y": False},
                             "right": {"a": pressed.get("a", False), "b": False}}}
        self.buttons(now, frame, {"state": "WAIT_GRIP", "reason": "", "robot_ready": ready}, self.real, self.actions)

    def test_x_stop_returns_to_initial_pose_once_the_save_is_done_and_feedback_is_fresh(self):
        control = self.recording_control()
        self.ready_tick(1., x=True)
        self.assertEqual(self.actions, [])
        self.assertIn("자동 복귀", self.buttons.status)
        self.ready_tick(1.01)
        self.assertEqual(self.actions, [], "the save has not run yet")
        control.prev = "STOP"            # RecordControl.tick returned from save_episode()
        self.ready_tick(20., ready=False)
        self.assertEqual(self.actions, [], "stale feedback right after the save")
        self.ready_tick(20.3)
        self.assertEqual(self.actions, ["home"])
        self.assertIn("녹화 저장 완료", self.buttons.status)
        for now in (20.31, 25.):
            self.ready_tick(now)
            self.assertEqual(self.actions, [], "the return is requested only once")

    def test_auto_return_is_dropped_when_feedback_does_not_come_back(self):
        control = self.recording_control()
        self.ready_tick(1., x=True)
        control.prev = "STOP"
        self.ready_tick(20., ready=False)
        self.ready_tick(22.1, ready=False)
        self.assertIn("자동 복귀 취소", self.buttons.status)
        self.ready_tick(60.)
        self.assertEqual(self.actions, [], "a late recovery must not move the robot unasked")

    def test_x_start_and_disabled_option_never_return(self):
        control = self.recording_control(prev="STOP")
        control.widget.set_button_option(label_text="Record", option_text="STOP")
        self.ready_tick(1., x=True)          # STOP -> REC: a start
        self.ready_tick(2.)
        self.assertEqual(self.actions, [])
        self.buttons = ControllerButtonActions(home_after_record=False)
        control = self.recording_control()
        self.ready_tick(3., x=True)
        control.prev = "STOP"
        self.ready_tick(4.)
        self.assertEqual(self.actions, [])

    def test_a_after_x_requests_a_single_return(self):
        control = self.recording_control()
        self.ready_tick(1., x=True)
        control.prev = "STOP"
        self.ready_tick(20., a=True)
        self.assertEqual(self.actions, ["home"])
        self.ready_tick(20.01)
        self.assertEqual(self.actions, [])

    def test_auto_return_skips_when_a_return_or_reset_already_runs(self):
        control = self.recording_control()
        self.ready_tick(1., x=True)
        control.prev = "STOP"
        self.real.home_pending = True
        self.ready_tick(20.)
        self.real.home_pending = False
        self.ready_tick(21.)
        self.assertEqual(self.actions, [])

    def test_faster_initial_pose_keeps_the_reset_profile(self):
        self.base_speed[1:] = np.deg2rad(20.)
        self.real.home_speed_limits = self.base_speed.copy()
        self.start(lift_ok=False)
        self.assert_speed(30)
        self.advance("HOME")
        np.testing.assert_allclose(self.real.home_speed_limits[self.real.arm_mask], np.deg2rad(100))
        for _ in range(9):
            self.advance("HOME")
        self.assertIn("완료", self.buttons.status)
        np.testing.assert_array_equal(self.real.home_speed_limits, self.base_speed)

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
        self.start()
        self.assert_speed(30)
        self.assertIn("위로", self.buttons.status)
        self.advance("HOME")
        self.assertIn("shutdown1", self.buttons.status)
        self.assert_speed(30)
        self.real.planner.homing = True
        self.advance("HOMING")  # Alignment has not reached its measured target yet.
        self.assertEqual(len(self.targets), 2)
        self.assertFalse(self.actions)
        self.assert_speed(30)
        self.real.planner.homing = False
        for label in ("shutdown2", "shutdown3", "shutdown4", "bringup start",
                      "bringup step1", "bringup step2", "bringup step3", "bringup step4"):
            self.advance("HOME")
            self.assertIn(label, self.buttons.status)
            self.assertEqual(self.actions, ["home"])
            self.assert_speed(100)
        self.advance("HOME")
        self.assertFalse(self.buttons.lock_input)
        self.assertIn("완료", self.buttons.status)
        np.testing.assert_array_equal(self.real.home_speed_limits, self.base_speed)
        self.start()  # A later reset must start slowly again.
        self.assert_speed(30)

    def test_every_reset_stage_keeps_bringup_head_despite_feedback_drift(self):
        pitch = list(JOINT_NAMES).index("head_joint1")
        yaw = list(JOINT_NAMES).index("head_joint2")
        self.start()  # The mocked lift IK returns a full vector with pitch zero.
        self.assertEqual(self.targets[-1][pitch], .69)
        for stage in range(1, 10):
            measured_pitch = .69 - .01 * stage
            self.real.last_feedback["q"][pitch] = measured_pitch
            self.real.last_feedback["q"][yaw] = .05
            self.advance("HOME")
            self.assertEqual(self.targets[-1][pitch], .69, self.buttons.status)
            self.assertEqual(self.targets[-1][yaw], .05)
            self.assertEqual(self.real.last_feedback["q"][pitch], measured_pitch)
        self.assertEqual(len(self.targets), 10)
        self.advance("HOME")
        self.assertIn("완료", self.buttons.status)

    def test_skipped_lift_still_keeps_alignment_speed(self):
        self.start(lift_ok=False)
        self.assertIn("1/9 shutdown1", self.buttons.status)
        self.assert_speed(30)
        self.advance("HOME")
        self.assertIn("2/9 shutdown2", self.buttons.status)
        self.assert_speed(100)

    def test_retry_keeps_each_stage_speed(self):
        self.start(lift_ok=False)
        for label, speed in (("shutdown1", 30), ("shutdown2", 100)):
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
        self.assert_speed(100)
        self.advance("FAULT", reason="collision guard blocked")
        self.assertFalse(self.buttons.lock_input)
        np.testing.assert_array_equal(self.real.home_speed_limits, self.base_speed)

    def test_reset_arm_speed_is_independent_of_teleop_but_auxiliary_caps_are_kept(self):
        self.real.teleop_speed_limits[1:] = np.deg2rad(60)
        teleop = self.real.teleop_speed_limits.copy()
        self.start(lift_ok=False)
        self.assert_speed(30)
        self.advance("HOME")
        np.testing.assert_allclose(self.real.home_speed_limits[self.real.arm_mask], np.deg2rad(100))
        np.testing.assert_allclose(self.real.home_speed_limits[[1, 2, 17, 18]], np.deg2rad(60))
        self.assertEqual(self.real.home_speed_limits[0], .01)
        np.testing.assert_array_equal(self.real.teleop_speed_limits, teleop)

    def test_missing_feedback_keeps_next_stage_and_its_speed(self):
        self.start(lift_ok=False)
        self.real.last_feedback = None
        self.advance("HOME")
        self.assertEqual(len(self.targets), 1)
        self.assert_speed(30)
        self.real.last_feedback = {"q": self.targets[-1],
                                   "bridge": {"limits": {"home_joint_speed_rad_s": np.deg2rad(100.)}}}
        self.advance("IDLE")
        self.assertIn("shutdown2", self.buttons.status)
        self.assert_speed(100)

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
        self.assert_speed(100)


if __name__ == "__main__":
    unittest.main()
