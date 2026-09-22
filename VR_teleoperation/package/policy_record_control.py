"""Keyboard-driven "record while running a policy" session for the real robot.

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
recording" or "return to initial pose" event and this session needs both:

    x   toggle recording (REC <-> STOP), like vr_buttons.py's left X
    y   discard the running episode, else toggle reject on the last saved
        one (episode_recorder.VREpisodeRecorder.discard_or_delete_last),
        like vr_buttons.py's left Y
    a   guarded return to the initial pose, like vr_buttons.py's right A
    q / Esc   stop the whole session

The policy keeps driving the robot regardless of recording state -- x only
starts/stops capturing the already-running rollout into an episode, exactly
like teleop's REC button does for teleop's own already-running motion.
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


def run_policy_with_keyboard_recording(deploy, recorder, duration_s=None):
    """Blocking loop: the policy always drives the robot once ready (see
    PolicyDeployment.step()); x/y/a/q control recording and homing as above.

    `recorder` (episode_recorder.VREpisodeRecorder) must already point at the
    same robot_ip as `deploy`. Always releases both sessions on the way out
    (FAULT, duration_s elapsing, 'q'/Esc, or Ctrl-C).
    """
    events = {"toggle_record": False, "discard": False, "home": False, "quit": False}

    def on_key(name):
        if name == "x":
            events["toggle_record"] = True
        elif name == "y":
            events["discard"] = True
        elif name == "a":
            events["home"] = True
        elif name in ("q", "esc"):
            events["quit"] = True

    listener = _make_listener(on_key)
    print("[record] x=toggle recording  y=discard/reject last  a=Initial pose  q/Esc=stop")

    deploy.start()
    recorder.start_thread()
    try:
        t_next = time.monotonic()
        end = None if duration_s is None else t_next + duration_s
        while (end is None or time.monotonic() < end) and not events["quit"]:
            now = time.monotonic()

            if events["home"]:
                events["home"] = False
                # One-shot: PolicyDeployment.step() itself keeps stepping the
                # guarded return on every later tick (it checks
                # self.real.planner.homing), so nothing else re-requests it.
                result = deploy.step(now, action="home")
            else:
                result = deploy.step(now)

            if events["toggle_record"]:
                events["toggle_record"] = False
                recorder.stop() if recorder.recording else recorder.start()
            if events["discard"]:
                events["discard"] = False
                recorder.discard_or_delete_last()

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
