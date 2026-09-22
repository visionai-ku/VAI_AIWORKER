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

A one-line status (episode in progress / saved count / policy state) is
printed each tick via IPython.display.clear_output, similar in spirit to the
on-screen status text lerobot's own recording tools show.
"""
from __future__ import annotations

import time


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
    from IPython.display import clear_output

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

            if now - last_status_at >= 1.0 / status_hz:
                last_status_at = now
                clear_output(wait=True)
                print("[record] x=start/stop episode  y=discard/reject last  a=Initial pose  q/Esc=stop")
                print(_status_line(recorder, result))

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
        deploy.stop()
        recorder.close()
    return deploy.last_result
