"""Keyboard-driven "record one episode, running the policy only for it" session.

Mirrors two things from lerobot (github.com/huggingface/lerobot, pinned 0.4.4,
see also github.com/plaxxma/DiffusionPoicy_UR5e_real's lerobot/ tree):

* lerobot.scripts.lerobot_record's record_loop() -- one control tick gets an
  action (there from teleop or a policy) and sends it, recording the frame
  when a dataset is attached. Here the action always comes from the policy
  (PolicyDeployment.step()); recording is a separate, already-running
  VREpisodeRecorder that independently watches the robot's own state/command
  ZMQ endpoints (episode_recorder.py), so it captures whatever action
  actually reaches the robot -- no add_frame() call needed here.
* lerobot.utils.control_utils.init_keyboard_listener()'s contract: a
  non-blocking pynput listener (None in a headless environment) that sets
  flags on a shared dict, polled once per tick by the control loop.

The specific keys are this repo's own VR button scheme (package/vr_buttons.py)
rather than lerobot's fixed right/left/esc set, since that set has no "start
recording" or "return to initial pose" event and this one needs both:

    x   start this episode: engage the policy AND start recording together
        (resets the policy's internal state -- ACT's action queue/temporal
        ensemble should not carry over a gap). Pressed again: stop the policy
        AND save the episode together, same as releasing teleop's grip while
        pressing REC off. The robot holds position whenever no episode is
        running, exactly like a released VR grip.
    y   discard the running episode, else toggle reject on the last saved
        one (episode_recorder.VREpisodeRecorder.discard_or_delete_last),
        like vr_buttons.py's left Y. Also disengages the policy if an episode
        was running, matching the recorder's own state change.
    a   guarded return to the initial pose, like vr_buttons.py's right A
    q / Esc   stop the whole session

A small floating Qt window shows live status (episode in progress / saved
count / policy state) -- the same idea as episode_recorder.RecordControl's
window (whose title bar shows recorder.status()) for teleop recording, just
read-only here since keyboard already drives start/stop/discard/home. Falls
back to printing the status line in the cell output if a window cannot be
opened (e.g. no display).
"""
from __future__ import annotations

import time


class _StatusWindow:
    """Read-only floating window mirroring episode_recorder.RecordControl's
    window, without its REC/STOP/DEL buttons (keyboard already covers those)."""

    def __init__(self, x_offset=0.0, y_offset=0.72, width=0.28, height=0.12):
        from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
        from ri_motion_v5_package.utility import get_monitor_size

        self.app = QApplication.instance() or QApplication([])
        w_monitor, h_monitor = get_monitor_size()
        self.widget = QWidget()
        self.widget.setWindowTitle("SG2 Policy Recording")
        self.widget.setGeometry(int(x_offset * w_monitor), int(y_offset * h_monitor),
                                int(width * w_monitor), int(height * h_monitor))
        layout = QVBoxLayout(self.widget)
        self.label = QLabel("starting...")
        self.label.setStyleSheet("font-family: monospace; font-size: 13px;")
        self.label.setWordWrap(True)
        layout.addWidget(self.label)
        self.widget.show()

    def set_text(self, text):
        self.label.setText(text)
        self.app.processEvents()

    def close(self):
        self.widget.close()
        self.app.processEvents()


def _open_status_window():
    try:
        return _StatusWindow()
    except Exception as exc:
        print("[record] status window unavailable (%s); printing status in the cell instead." % exc)
        return None


def _make_listener(on_key):
    """Same non-blocking-listener contract as lerobot.utils.control_utils.
    init_keyboard_listener(): a pynput global listener when a display is
    reachable, otherwise None (Ctrl-C is then the only way to stop, e.g. over
    a headless SSH session)."""
    from lerobot.utils.control_utils import is_headless

    if is_headless():
        print("[record] headless environment detected; keyboard controls unavailable (Ctrl-C to stop).")
        return None
    from pynput import keyboard

    def on_press(key):
        try:
            char = key.char.lower() if getattr(key, "char", None) else None
        except Exception:
            char = None
        name = char if char else ("esc" if key == keyboard.Key.esc else None)
        if name:
            on_key(name)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener


def _status_line(recorder, result):
    return "%s | policy: %s (%s)" % (recorder.status(), result["state"], result["reason"])


def run_policy_with_keyboard_recording(deploy, recorder, duration_s=None, status_hz=2.0):
    """Blocking loop: x starts/stops one episode (policy + recording together),
    y/a/q as above. The policy only drives the robot while an episode is
    running (see PolicyDeployment.step(enabled=...)); it holds position the
    rest of the time.

    `recorder` (episode_recorder.VREpisodeRecorder) must already point at the
    same robot_ip as `deploy`. Always releases both sessions on the way out
    (FAULT, duration_s elapsing, 'q'/Esc, or Ctrl-C) -- an episode still being
    recorded when the loop exits is saved, not dropped.
    """
    events = {"toggle_episode": False, "discard": False, "home": False, "quit": False}

    def on_key(name):
        if name == "x":
            events["toggle_episode"] = True
        elif name == "y":
            events["discard"] = True
        elif name == "a":
            events["home"] = True
        elif name in ("q", "esc"):
            events["quit"] = True

    listener = _make_listener(on_key)
    print("[record] x=start/stop episode (policy+recording)  y=discard/reject last  a=Initial pose  q/Esc=stop")
    status_window = _open_status_window()

    deploy.start()
    recorder.start_thread()
    try:
        t_next = time.monotonic()
        end = None if duration_s is None else t_next + duration_s
        last_status_at = 0.0
        while (end is None or time.monotonic() < end) and not events["quit"]:
            now = time.monotonic()

            if events["home"]:
                events["home"] = False
                # One-shot: PolicyDeployment.step() itself keeps stepping the
                # guarded return on every later tick (it checks
                # self.real.planner.homing), so nothing else re-requests it.
                result = deploy.step(now, action="home")
            else:
                result = deploy.step(now, enabled=recorder.recording)

            if events["toggle_episode"]:
                events["toggle_episode"] = False
                if recorder.recording:
                    recorder.stop()
                else:
                    deploy.policy.reset()
                    recorder.start()
            if events["discard"]:
                events["discard"] = False
                recorder.discard_or_delete_last()

            if status_window is not None:
                status_window.app.processEvents()  # keep the window responsive every tick
            if now - last_status_at >= 1.0 / status_hz:
                last_status_at = now
                text = _status_line(recorder, result)
                if status_window is not None:
                    status_window.set_text(text)
                else:
                    print("\r" + text + " " * 10, end="", flush=True)

            if result["state"] == "FAULT":
                print("[policy] FAULT:", result["reason"])
                break

            t_next += deploy.dt
            sleep_for = t_next - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                t_next = time.monotonic()  # fell behind this tick; resync instead of catching up
    except KeyboardInterrupt:
        print("[record] interrupted")
    finally:
        if listener is not None:
            listener.stop()
        if status_window is not None:
            status_window.close()
        deploy.stop()
        recorder.close()
    return deploy.last_result
