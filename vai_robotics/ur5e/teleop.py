"""Run Cartesian UR5e teleoperation with a Quest 3 or a 3Dconnexion SpaceMouse."""

import argparse
from contextlib import ExitStack
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation


HZ = 50
MAX_LINEAR_SPEED = 0.10       # m/s
MAX_ANGULAR_SPEED = 0.60      # rad/s
MAX_POSITION_LEAD = 0.025     # m from feedback
MAX_ROTATION_LEAD = 0.12      # rad from feedback


def bounded_vector(vector, limit):
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError("Expected a finite 3D vector")
    norm = np.linalg.norm(vector)
    return vector * (limit / norm) if norm > limit else vector


class TeleopController:
    """Integrate clutched deltas into bounded absolute targets for either backend."""

    def __init__(self, backend, *, dry_run=False):
        self.backend = backend
        self.dry_run = dry_run
        self.measured = None
        self.ready_after_release = False
        self.target = None
        self.last_tick = None

    def set_mode(self, dry_run, *, simulation, sample):
        if dry_run == self.dry_run:
            return ""
        if not dry_run and not simulation and (sample is None or sample.enabled):
            return "입력 장치를 연결하고 클러치를 놓은 뒤 REAL RUN을 선택하세요."
        self.backend.hold()
        self.target = self.last_tick = None
        self.ready_after_release = False
        if not simulation:
            if dry_run:
                self.backend.set_motion_enabled(False)
            else:
                try:
                    self.backend.set_motion_enabled(True)
                except (RuntimeError, OSError, ImportError) as error:
                    return f"REAL RUN 전환 실패 · {error}"
        self.dry_run = dry_run
        return ""

    def update(self, sample, now, *, scale=1.0):
        dt = 0.0 if self.last_tick is None else now - self.last_tick
        self.last_tick = now
        self.measured = self.backend.pose()
        measured_xyz, measured_rot = self.measured
        if sample is None or not 0.0 <= dt <= 0.12:
            self.ready_after_release = False
        elif not sample.enabled:
            self.ready_after_release = True
        elif self.ready_after_release:
            if self.target is None:
                self.target = (measured_xyz.copy(), measured_rot)
            else:
                xyz, rot = self.target
                # Never replay a backlog after a slow frame.
                dt = min(dt, 2 / HZ)
                xyz = xyz + scale * bounded_vector(sample.translation, MAX_LINEAR_SPEED * dt)
                rot = Rotation.from_rotvec(scale * bounded_vector(sample.rotation, MAX_ANGULAR_SPEED * dt)) * rot
                if not self.dry_run:
                    xyz = measured_xyz + bounded_vector(xyz - measured_xyz, MAX_POSITION_LEAD)
                    lead = bounded_vector((rot * measured_rot.inv()).as_rotvec(), MAX_ROTATION_LEAD)
                    rot = Rotation.from_rotvec(lead) * measured_rot
                self.target = xyz, rot
            if not self.dry_run:
                self.backend.command(*self.target)
            return
        self.target = None
        self.backend.hold()


def clutch_text(device, sample, active):
    if hasattr(device, "clutch_pressed"):
        pressed = device.clutch_pressed
        if pressed is None:
            state = device.status or "입력 대기"
        elif not pressed:
            state = "놓음"
        elif active:
            state = "누름"
        else:
            state = "누름 · 놓았다가 다시 누르세요"
        return f"버튼 {device.button}: {state}"
    if sample is None:
        return "입력 대기"
    if active:
        return "누름"
    return "놓음" if not sample.enabled else "클러치를 놓으세요"


def run_teleop(device, backend, *, sim=False, dry_run=False, panel=None):
    controller = TeleopController(backend, dry_run=dry_run)
    last_report = float("-inf")
    message = ""
    active = False
    view = backend if sim else None
    print("Release the device clutch, then hold it to move. Ctrl+C exits.", flush=True)
    with ExitStack() as session:
        preview_resources = session.enter_context(ExitStack())
        session.callback(backend.hold)
        while view is None or view.is_running():
            if not sim and panel is not None and controller.dry_run and view is None:
                from .sim_teleop import SimBackend

                # The desktop preview only owns simulation state. Load/render it
                # while the real backend is receive-only, outside active servoing.
                view = preview_resources.enter_context(SimBackend(hz=HZ))
            started = time.monotonic()
            request = panel.poll() if panel is not None else None
            if panel is not None and not panel.is_open:
                break
            change_mode = request is not None and request != controller.dry_run
            # Going back to preview takes priority over polling USB or network input.
            sample = None if change_mode and request else device.read()
            if change_mode:
                if not sim and not request and view is not None and sample is not None and not sample.enabled:
                    preview_resources.close()
                    view = None
                message = controller.set_mode(request, simulation=sim, sample=sample)
                # Don't use input captured before a transition opened an RTDE session.
                if not message:
                    sample = None
            controller.update(sample, time.monotonic(), scale=panel.scale if panel is not None else 1.0)
            if sim:
                view.tick(dry_run=controller.dry_run, target=controller.target)
            elif view is not None:
                view.tick(dry_run=True, target=controller.target, measured_q=backend.joints(),
                          measured_tcp=controller.measured)
            if (controller.target is not None) != active:
                active = controller.target is not None
                print("Clutch active" if active else "Holding", flush=True)
            if panel is not None:
                panel.refresh(dry_run=controller.dry_run, clutch=clutch_text(device, sample, active),
                              measured=controller.measured,
                              target=controller.target, message=message)
            elif controller.dry_run and active and started - last_report >= 0.5:
                xyz, rot = controller.target
                print("Dry-run target [m, rad]:", np.round(np.r_[xyz, rot.as_rotvec()], 4), flush=True)
                last_report = started
            time.sleep(max(0.0, started + 1 / HZ - time.monotonic()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sim", "real"), required=True)
    parser.add_argument("--input", choices=("quest", "spacemouse"), required=True)
    parser.add_argument("--quest-host", help="operator PC LAN address reachable from the Quest")
    parser.add_argument("--quest-port", type=int, default=8443)
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--button", type=int, default=1, help="SpaceMouse clutch button index (default 1)")
    parser.add_argument("--robot-ip", help="UR controller address for the real backend")
    initial_mode = parser.add_mutually_exclusive_group()
    initial_mode.add_argument("--enable-motion", action="store_true", help="send real robot motion commands")
    initial_mode.add_argument("--dry-run", action="store_true", help="start with the virtual arm preview")
    parser.add_argument("--headless", action="store_true", help="run simulation without a viewer")
    parser.add_argument("--no-panel", action="store_true", help="run without the desktop settings panel")
    args = parser.parse_args(argv)
    if args.backend == "real" and not args.robot_ip:
        parser.error("--backend real requires --robot-ip")
    if args.input == "quest" and not args.quest_host:
        parser.error("--input quest requires --quest-host")
    if args.backend == "sim" and args.enable_motion:
        parser.error("--enable-motion applies only to --backend real")
    if args.backend == "real" and args.headless:
        parser.error("--headless applies only to --backend sim")
    if args.button < 0 or not 1 <= args.quest_port <= 65535:
        parser.error("Invalid button index or Quest port")

    from .inputs import QuestInput, SpaceMouseInput

    simulation = args.backend == "sim"
    dry_run = args.dry_run or (not simulation and not args.enable_motion)
    desktop = sys.platform != "linux" or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    show_panel = desktop and not args.no_panel and not args.headless
    with ExitStack() as stack:
        panel = None
        if show_panel:
            from .teleop_panel import TeleopPanel

            device_name = f"Quest 3 · {args.hand}" if args.input == "quest" else f"SpaceMouse · 버튼 {args.button}"
            panel = TeleopPanel(simulation=simulation, device_name=device_name, dry_run=dry_run,
                                robot_ip=args.robot_ip)
            stack.callback(panel.close)
        if args.input == "quest":
            device = QuestInput(host=args.quest_host, port=args.quest_port, hand=args.hand)
        else:
            device = SpaceMouseInput(button=args.button, wait_for_device=show_panel)
        if simulation:
            from .sim_teleop import SimBackend

            backend = SimBackend(hz=HZ, headless=args.headless)
        else:
            from .real import RealUR5e

            print("Real robot motion ENABLED" if args.enable_motion else "Real robot, receive-only dry run", flush=True)
            backend = RealUR5e(args.robot_ip, enable_motion=args.enable_motion, frequency=HZ)
        stack.enter_context(device)
        stack.enter_context(backend)
        try:
            run_teleop(device, backend, sim=simulation, dry_run=dry_run, panel=panel)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
