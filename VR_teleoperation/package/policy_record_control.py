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
* diffusion_policy's eval_real_robot.py in that same repo: the whole operator
  interface is one cv2.imshow window -- camera view, cv2.putText state on top,
  cv2.pollKey() read once per tick for non-blocking key input. No PyQt5 and no
  global keyboard hook, which is what this module uses too (see _StatusWindow
  for why mixing cv2's Qt with PyQt5 kills the process).

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

One cv2 window shows live status (episode in progress / saved count / policy
state) next to the camera feeds the policy itself sees -- the same idea as
lerobot's own record_loop(display_data=True), which streams observation images
to a live viewer every tick so you can see what the policy is reacting to, not
just whether it's running. Keys are read from that window (it must have focus).
Falls back to printing the status line in the cell output if a window cannot be
opened (e.g. no display), in which case Ctrl-C is the only control.
"""
from __future__ import annotations

import time

CAMERA_ORDER = ("head", "wrist_left", "wrist_right")
# Per-camera tile. Big enough to actually judge what the policy is looking at
# from across the desk; pass view_scale to run_policy_with_keyboard_recording()
# to grow/shrink the whole window.
_THUMB_W, _THUMB_H = 480, 360


class _StatusWindow:
    """One OpenCV window: the camera feeds side by side with the status lines
    drawn on top, and its own keyboard input.

    Deliberately cv2-only, no PyQt5. cv2 here is the GUI build, which points Qt
    at its OWN bundled plugins (QT_QPA_PLATFORM_PLUGIN_PATH); a PyQt5
    QApplication created in the same process then loads cv2's incompatible xcb
    plugin and aborts -- SIGABRT, uncatchable from Python, so the kernel just
    dies with no traceback. Using only cv2 keeps that env var pointing where cv2
    wants it and never mixes the two Qt stacks. This also matches how
    diffusion_policy's eval_real_robot.py (github.com/plaxxma/
    DiffusionPoicy_UR5e_real) drives its real-robot policy loop: one cv2.imshow
    window, cv2.putText for state, cv2.pollKey for non-blocking key input.

    Keys are read from THIS window, so it must have focus -- unlike a global
    hook (pynput), which needs X11 privileges and fires even when you are typing
    somewhere else.
    """

    WINDOW = "SG2 Policy Recording"

    # Fraction of the screen width the whole window should take when view_scale
    # is left on "auto": it grows on the 5120-wide desk monitor instead of turning
    # into a postage stamp, and shrinks on a small laptop panel instead of running
    # off the screen. Clamped so neither extreme gets silly.
    AUTO_WIDTH_FRACTION = 1.0 / 6.0
    # Floor keeps the text readable on a small panel, where a strict sixth of the
    # width would be unusably tiny; pass view_scale to override either way.
    MIN_SCALE, MAX_SCALE = 0.55, 2.0

    def __init__(self, cameras, view_scale=None, x_offset=0.0, y_offset=0.55):
        import cv2

        self.cv2 = cv2
        self.cameras = [cam for cam in CAMERA_ORDER if cam in cameras] or list(cameras)
        self.last_frames = {}
        w_monitor, h_monitor = self._monitor_size()
        if view_scale is None:
            target_w = self.AUTO_WIDTH_FRACTION * w_monitor / max(1, len(self.cameras))
            view_scale = min(self.MAX_SCALE, max(self.MIN_SCALE, target_w / _THUMB_W))
        self.tile_w = int(_THUMB_W * view_scale)
        self.tile_h = int(_THUMB_H * view_scale)
        self.font_scale = 0.75 * view_scale
        self.line_h = int(34 * view_scale)
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_AUTOSIZE)
        self._move(x_offset, y_offset, w_monitor, h_monitor)
        self.show(["starting..."], {})

    @staticmethod
    def _monitor_size():
        """Best-effort screen size; a fixed fallback beats failing to open."""
        try:
            from ri_motion_v5_package.utility import get_monitor_size

            return get_monitor_size()
        except Exception:
            return 1920, 1080

    def _move(self, x_offset, y_offset, w_monitor, h_monitor):
        """Place the window as a fraction of the screen (default: lower-left, so it
        does not sit on top of the notebook)."""
        x = int(x_offset * w_monitor)
        y = int(y_offset * h_monitor)
        # Keep the whole window on screen even when the fraction would push it off.
        y = min(y, max(0, h_monitor - (self.tile_h + 6 * self.line_h)))
        try:
            self.cv2.moveWindow(self.WINDOW, x, y)
        except Exception:
            pass

    def _canvas(self, lines, frames):
        import numpy as np

        cv2 = self.cv2
        for cam, bgr in frames.items():
            if bgr is not None:
                self.last_frames[cam] = bgr
        tiles = []
        for cam in self.cameras:
            bgr = self.last_frames.get(cam)
            tile = (np.zeros((self.tile_h, self.tile_w, 3), np.uint8) if bgr is None
                    else cv2.resize(bgr, (self.tile_w, self.tile_h)))
            tile = tile.copy()
            cv2.putText(tile, cam, (10, int(28 * self.font_scale) + 8), cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale, (0, 255, 255), 2, cv2.LINE_AA)
            if bgr is None:
                cv2.putText(tile, "waiting", (10, self.tile_h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            self.font_scale, (0, 0, 255), 2, cv2.LINE_AA)
            tiles.append(tile)
        strip = (np.hstack(tiles) if tiles
                 else np.zeros((self.tile_h, self.tile_w, 3), np.uint8))

        header = np.zeros((self.line_h * len(lines) + 16, strip.shape[1], 3), np.uint8)
        for i, line in enumerate(lines):
            cv2.putText(header, line, (10, int(self.line_h * (i + 1))), cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale, (255, 255, 255), 2, cv2.LINE_AA)
        return np.vstack([header, strip])

    def poll_key(self):
        """Read a key without redrawing -- cheap enough to call every control tick.

        Redrawing is not: resizing the camera tiles and pushing them through
        imshow costs milliseconds, and a tick over MAX_LOOP_DT_S (0.10s) faults
        the robot session, so the loop draws at status_hz and polls keys here.
        """
        return self._key(self.cv2.pollKey())

    def _key(self, key):
        if key == -1:
            return None
        key &= 0xFF
        if key == 27:
            return "esc"
        return chr(key).lower() if 32 <= key < 127 else None

    def show(self, lines, frames):
        """Draw one frame and return the key pressed this tick (None if none)."""
        self.cv2.imshow(self.WINDOW, self._canvas(lines, frames))
        return self._key(self.cv2.pollKey())

    def close(self):
        try:
            self.cv2.destroyWindow(self.WINDOW)
            self.cv2.pollKey()          # let the window manager actually drop it
        except Exception:
            pass


def _open_status_window(cameras, view_scale=None, x_offset=0.0, y_offset=0.55):
    try:
        return _StatusWindow(cameras, view_scale=view_scale,
                             x_offset=x_offset, y_offset=y_offset)
    except Exception as exc:
        print("[record] status window unavailable (%s); printing status in the cell instead.\n"
              "         keyboard control needs the window -- use Ctrl-C to stop." % exc)
        return None


def _status_line(recorder, result):
    return "%s | policy: %s (%s)" % (recorder.status(), result["state"], result["reason"])


def _status_lines(recorder, result):
    """Same information as _status_line, split for the window's fixed width."""
    recording = "REC" if recorder.recording else "idle"
    return [
        "%s  |  policy: %s" % (recording, result["state"]),
        (result["reason"] or "")[:70],
        recorder.status()[:70],
        "x=start/stop  y=discard  a=initial pose  q=quit",
    ]


def run_policy_with_keyboard_recording(deploy, recorder, duration_s=None, status_hz=2.0,
                                       view_scale=None, x_offset=0.0, y_offset=0.55):
    """Blocking loop: x starts/stops one episode (policy + recording together),
    y/a/q as above. The policy only drives the robot while an episode is
    running (see PolicyDeployment.step(enabled=...)); it holds position the
    rest of the time.

    `recorder` (episode_recorder.VREpisodeRecorder) must already point at the
    same robot_ip as `deploy`. Always releases both sessions on the way out
    (FAULT, idle timeout, 'q'/Esc, or Ctrl-C) -- an episode still being
    recorded when the loop exits is saved, not dropped.

    duration_s is an IDLE timeout: the countdown restarts on every keypress and
    never runs while an episode is recording, so it cannot cut a rollout short.
    None keeps the session open until 'q'/Esc or Ctrl-C.

    view_scale scales the status window (None = auto-fit to ~42% of screen width);
    x_offset/y_offset place it as a fraction of the screen (default lower-left).
    """
    print("[record] click the 'SG2 Policy Recording' window, then: "
          "x=start/stop episode (policy+recording)  y=discard/reject last  a=Initial pose  q/Esc=stop")
    status_window = _open_status_window(deploy.cameras, view_scale=view_scale,
                                        x_offset=x_offset, y_offset=y_offset)

    quit_requested = home_requested = False
    deploy.start()
    recorder.start_thread()
    try:
        t_next = time.monotonic()
        # Idle timeout, not a hard session cap: recording an episode or pressing a key
        # pushes it back, so a long rollout is never cut off mid-episode and the window
        # only closes on its own once it has really been left alone this long.
        idle_deadline = None if duration_s is None else t_next + duration_s
        last_draw = 0.0
        while not quit_requested:
            if (idle_deadline is not None and not recorder.recording
                    and time.monotonic() >= idle_deadline):
                print("[record] %.0fs 동안 조작이 없어 세션을 종료합니다 (duration_s)." % duration_s)
                break
            now = time.monotonic()
            # A pending Initial-pose request owns this tick: ticking it as a normal
            # (disabled) step first would disarm the robot and cancel the return.
            if home_requested:
                home_requested = False
                result = deploy.step(now, action="home")
            else:
                result = deploy.step(now, enabled=recorder.recording)

            # Keys are polled every tick (cheap); the picture is redrawn at status_hz,
            # because resizing/blitting the camera tiles costs enough that doing it every
            # tick can push a tick past MAX_LOOP_DT_S and fault the robot session.
            key = None
            due = now - last_draw >= 1. / status_hz
            if status_window is not None:
                if due:
                    last_draw = now
                    # step() only polls cameras while enabled+ready, so poll here as well to
                    # keep the preview live while the policy is idle between episodes.
                    key = status_window.show(_status_lines(recorder, result), deploy._poll_cameras())
                else:
                    key = status_window.poll_key()
            elif due:
                last_draw = now
                print("\r" + _status_line(recorder, result) + " " * 10, end="", flush=True)

            if key == "x":
                if recorder.recording:
                    recorder.stop()
                else:
                    deploy.policy.reset()
                    recorder.start()
            elif key == "y":
                recorder.discard_or_delete_last()
            elif key == "a":
                # Handled at the top of the next tick; from there PolicyDeployment.step()
                # keeps stepping the guarded return itself until it completes.
                home_requested = True
                if recorder.recording:
                    recorder.stop()     # an Initial-pose return is not part of an episode
            elif key in ("q", "esc"):
                quit_requested = True

            if key is not None or recorder.recording:
                idle_deadline = None if duration_s is None else time.monotonic() + duration_s

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
        # recorder.close() is what finalizes the LeRobot dataset (writes the parquet
        # footers). If an earlier cleanup step raises, skipping it leaves the dataset
        # unreadable -- and the NEXT run then refuses to start on that directory
        # (dataset_status -> ROOT_UNFINALIZED). So each step gets its own guard and the
        # recorder is closed first.
        for step in (recorder.close, deploy.stop,
                     status_window.close if status_window is not None else None):
            if step is None:
                continue
            try:
                step()
            except Exception as exc:
                print("[record] cleanup step %s failed: %r" % (getattr(step, "__name__", step), exc))
    return deploy.last_result
