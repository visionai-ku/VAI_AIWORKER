"""run_policy_with_keyboard_recording()'s loop and, above all, its cleanup.

No robot, no checkpoint, no window: PolicyDeployment/VREpisodeRecorder are
stubs, and the status window is disabled (headless), so these tests cover the
control flow only -- which key does what, and what happens on the way out.
"""
from pathlib import Path
import runpy
import unittest

PROJECT_DIR = Path(__file__).resolve().parents[1]
runpy.run_path(str(PROJECT_DIR / "package/init_project.py"))

import package.policy_record_control as prc


class StubPolicy:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


class StubDeploy:
    """Ticks a canned result; records whether stop() ran."""

    def __init__(self, state="WAIT_ROBOT"):
        self.policy = StubPolicy()
        self.cameras = {"wrist_left": 5571}
        self.dt = 0.0
        self.state = state
        self.started = self.stopped = False
        self.steps = 0
        self.last_result = None
        self.actions = []
        self.enabled_log = []

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def step(self, now=None, action=None, enabled=True):
        self.steps += 1
        self.actions.append(action)
        self.enabled_log.append(None if action == "home" else enabled)
        self.last_result = {"state": self.state, "reason": "stub", "action": None}
        return self.last_result

    def _poll_cameras(self):
        return {}


class StubRecorder:
    def __init__(self):
        self.recording = False
        self.closed = False
        self.thread_started = False
        self.started = self.stopped = 0
        self.discarded = 0

    def start_thread(self):
        self.thread_started = True

    def start(self):
        self.started += 1
        self.recording = True

    def stop(self):
        self.stopped += 1
        self.recording = False

    def discard_or_delete_last(self):
        self.discarded += 1

    def status(self):
        return "stub recorder"

    def close(self):
        self.closed = True


class _Keys:
    """Stands in for the cv2 window, handing the loop one scripted key per tick."""

    def __init__(self, keys):
        self.keys = list(keys)
        self.closed = False
        self.shown = 0

    def show(self, lines, frames):
        self.shown += 1
        return self._next()

    def poll_key(self):
        return self._next()

    def _next(self):
        return self.keys.pop(0) if self.keys else None

    def close(self):
        self.closed = True


class KeyboardRecordingLoopTest(unittest.TestCase):
    def run_loop(self, keys, deploy=None, recorder=None, window=None, duration_s=None):
        deploy = deploy or StubDeploy()
        recorder = recorder or StubRecorder()
        window = _Keys(keys) if window is None else window
        self._patch_window(window)
        prc.run_policy_with_keyboard_recording(deploy, recorder, duration_s=duration_s)
        return deploy, recorder, window

    def _patch_window(self, window):
        original = prc._open_status_window
        prc._open_status_window = lambda cameras, **kw: window
        self.addCleanup(setattr, prc, "_open_status_window", original)

    def test_x_starts_then_stops_one_episode_and_resets_the_policy(self):
        deploy, recorder, _ = self.run_loop(["x", None, "x", "q"])
        self.assertEqual((recorder.started, recorder.stopped), (1, 1))
        self.assertEqual(deploy.policy.reset_count, 1, "ACT's action queue must not span episodes")

    def test_y_discards_and_a_requests_the_guarded_return(self):
        deploy, recorder, _ = self.run_loop(["y", "a", None, "q"])
        self.assertEqual(recorder.discarded, 1)
        self.assertIn("home", deploy.actions)

    def test_a_never_ticks_a_disabled_step_before_the_home_request(self):
        """A disabled tick between pressing 'a' and the request disarms the robot and
        drops the return (SG2FixedQuestReal.stop()), which is what made 'a' look dead."""
        deploy, _, _ = self.run_loop(["a", None, "q"])
        home_at = deploy.actions.index("home")
        self.assertEqual(deploy.enabled_log[home_at], None,
                         "the home request must own its tick, not follow a disabled one")

    def test_a_stops_a_running_episode_so_the_return_is_not_recorded(self):
        deploy, recorder, _ = self.run_loop(["x", "a", None, "q"])
        self.assertEqual(recorder.stopped, 1)
        self.assertIn("home", deploy.actions)

    def test_quit_key_ends_the_session(self):
        deploy, _, _ = self.run_loop(["q"])
        self.assertTrue(deploy.stopped)

    def test_fault_breaks_the_loop(self):
        deploy = StubDeploy(state="FAULT")
        deploy, recorder, _ = self.run_loop([None] * 5, deploy=deploy)
        self.assertEqual(deploy.steps, 1, "a FAULT must stop the loop immediately")
        self.assertTrue(recorder.closed)

    def test_recorder_is_finalized_even_when_another_cleanup_step_raises(self):
        """The bug that left eval_ datasets unreadable: recorder.close() writes the
        parquet footers, so it must not be skipped when deploy.stop() blows up --
        the next run would refuse to open the half-written dataset."""
        deploy = StubDeploy()

        def explode():
            raise RuntimeError("stop() failed")

        deploy.stop = explode
        recorder = StubRecorder()
        window = _Keys(["q"])
        self._patch_window(window)
        prc.run_policy_with_keyboard_recording(deploy, recorder)
        self.assertTrue(recorder.closed, "dataset must still be finalized")
        self.assertTrue(window.closed, "window must still be closed")

    def test_duration_none_only_ends_on_quit(self):
        deploy, _, window = self.run_loop([None] * 50 + ["q"], duration_s=None)
        self.assertEqual(deploy.steps, 51, "no idle timeout may cut the session short")

    def test_duration_is_an_idle_timeout_that_recording_holds_off(self):
        """It must not be a hard session cap: an episode left running has to keep the
        session alive past duration_s, instead of being cut off mid-rollout."""
        recorder = StubRecorder()
        recorder.recording = True                      # mid-episode
        deploy, recorder, _ = self.run_loop([None] * 30 + ["q"],
                                            recorder=recorder, duration_s=0.0)
        self.assertEqual(deploy.steps, 31, "recording must postpone the idle timeout")

    def test_idle_timeout_ends_the_session_when_untouched(self):
        deploy, recorder, _ = self.run_loop([None] * 50, duration_s=0.0)
        self.assertEqual(deploy.steps, 0, "an untouched session must stop at the timeout")
        self.assertTrue(recorder.closed, "and still finalize the dataset")

    def test_loop_runs_without_a_window(self):
        self._patch_window(None)
        deploy, recorder = StubDeploy(state="FAULT"), StubRecorder()
        prc.run_policy_with_keyboard_recording(deploy, recorder)
        self.assertTrue(recorder.closed)


if __name__ == "__main__":
    unittest.main()
