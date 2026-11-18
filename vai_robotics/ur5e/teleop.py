"""Run Cartesian UR5e teleoperation with a Quest 3 or a 3Dconnexion SpaceMouse."""

import argparse
from contextlib import ExitStack
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .real import RobotControlStopped
from .rg2 import DEFAULT_GRIP_FORCE_N, DEFAULT_OPEN_WIDTH_MM
from .rg2_visual import OPEN_WIDTH_MM, RG2Preview


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

    def set_mode(self, dry_run, *, simulation, sample, device=None):
        if dry_run == self.dry_run:
            return ""
        if not dry_run and not simulation and (sample is None or sample.enabled or not sample.buttons_released):
            if hasattr(device, "clutch_pressed"):
                if device.status:
                    return "SpaceMouse를 USB에 연결한 뒤 다시 시도하세요."
                if device.clutch_pressed is None:
                    return f"SpaceMouse {device.button}번 버튼을 한 번 눌렀다 놓은 뒤 실기 구동을 다시 선택하세요."
                return "SpaceMouse의 모든 버튼을 놓은 뒤 실기 구동을 다시 선택하세요."
            return "컨트롤러의 추적 상태를 확인하고 클러치를 놓은 뒤 실기 구동을 다시 선택하세요."
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
            state = device.status or "상태 확인 대기 · 한 번 눌렀다 놓으세요"
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


def run_teleop(device, backend, *, sim=False, dry_run=False, panel=None, gripper=None,
               gripper_force_n=DEFAULT_GRIP_FORCE_N):
    controller = TeleopController(backend, dry_run=dry_run)
    last_report = float("-inf")
    message = ""
    active = False
    preview = RG2Preview(gripper_force_n)
    open_width_mm = gripper.open_width_mm if gripper is not None else DEFAULT_OPEN_WIDTH_MM
    baseline_width = open_width_mm if sim else OPEN_WIDTH_MM
    print("Release the device clutch, then hold it to move. Ctrl+C exits.", flush=True)
    with ExitStack() as session:
        view = backend if sim else None
        if not sim and panel is not None:
            from .real_viewer import RealViewer

            view = session.enter_context(RealViewer())
            # Release the RTDE writer before waiting for the GUI to close.
            session.callback(backend.set_motion_enabled, False)
        else:
            session.callback(backend.hold)
        if gripper is not None:
            session.callback(gripper.update, None)
        while view is None or view.is_running():
            started = time.monotonic()
            request = panel.poll() if panel is not None else None
            if panel is not None and not panel.is_open:
                break
            if panel is not None and panel.open_width_mm != open_width_mm:
                open_width_mm = panel.open_width_mm
                if gripper is not None:
                    gripper.set_open_width(open_width_mm)
            change_mode = request is not None and request != controller.dry_run
            # Going back to preview takes priority over polling USB or network input.
            sample = None if change_mode and request else device.read()
            if change_mode:
                if gripper is not None:
                    gripper.update(None)
                message = controller.set_mode(request, simulation=sim, sample=sample, device=device)
                # Don't use input captured before a transition opened an RTDE session.
                if not message:
                    sample = None
                    preview.reset(baseline_width)
                    if request and gripper is not None:
                        # The arm writer is detached. Finish any Modbus write already in flight
                        # before presenting the transition to read-only control as complete.
                        gripper.wait_for_writes()
            try:
                controller.update(sample, time.monotonic(), scale=panel.scale if panel is not None else 1.0)
            except RobotControlStopped as error:
                if sim or panel is None or controller.dry_run:
                    raise
                if gripper is not None:
                    gripper.update(None)
                controller.set_mode(True, simulation=False, sample=None)
                if gripper is not None:
                    gripper.wait_for_writes()
                sample = None
                preview.reset(baseline_width)
                controller.update(sample, time.monotonic())  # Verify feedback before showing a measured pose.
                message = "실기 제어 연결이 중단되어 DRY RUN으로 전환했습니다."
                print(f"{message} {error}", file=sys.stderr, flush=True)
            if request is not None:
                if not change_mode:
                    message = ""
                panel.mode_result(dry_run=controller.dry_run, message=message)
                requested_mode = "DRY RUN" if request else ("SIM RUN" if sim else "REAL RUN")
                print(f"{requested_mode} · {message or '전환 완료'}", flush=True)
            measured_width = None
            chord = sample.gripper_chord if sample is not None else getattr(device, "gripper_chord", None)
            if gripper is not None:
                gripper.update(None if controller.dry_run or change_mode else chord)
                measured_width = gripper.width_mm
                if measured_width is not None:
                    baseline_width = measured_width
            width = measured_width
            if sim or controller.dry_run:
                command = sample.gripper_command if sample is not None else None
                if controller.dry_run and chord is not True:
                    # Released preview mirrors fresh RTDE/Modbus feedback or the frozen SIM RUN pose.
                    preview.reset(baseline_width)
                    command = None
                width = preview.update(time.monotonic(), baseline_width, command, chord,
                                       open_width_mm=open_width_mm)
                if sim and not controller.dry_run:
                    baseline_width = width
            if sim:
                backend.tick(dry_run=controller.dry_run, target=controller.target, gripper_width_mm=width)
            elif view is not None:
                view.update(q=backend.joints(), measured_tcp=controller.measured, dry_run=controller.dry_run,
                            target=controller.target if controller.dry_run else None, gripper_width_mm=width)
            if (controller.target is not None) != active:
                active = controller.target is not None
                print("Clutch active" if active else "Holding", flush=True)
            if panel is not None:
                if sim:
                    gripper_text = f"RG2 · 가상 폭 {width:.1f} mm"
                else:
                    gripper_text = gripper.status if gripper is not None else "RG2 · 그리퍼 IP 미설정"
                    if controller.dry_run:
                        gripper_text += f" · 미리보기 {width:.1f} mm"
                gripper_text += f" · 설정 파지력 {gripper_force_n:g} N"
                if hasattr(device, "gripper_chord"):
                    state = {True: "누름", False: "놓음", None: "입력 대기"}[device.gripper_chord]
                    gripper_text = f"버튼 0+1: {state} · {gripper_text}"
                panel.refresh(dry_run=controller.dry_run, clutch=clutch_text(device, sample, active),
                              measured=controller.measured,
                              target=controller.target, message=message, gripper=gripper_text)
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
    parser.add_argument("--gripper-ip", help="OnRobot Compute Box address for RG2 feedback and control")
    parser.add_argument("--gripper-force-n", type=float, default=DEFAULT_GRIP_FORCE_N,
                        help="RG2 grip force for preview and real commands, in newtons (3–40, default 40)")
    initial_mode = parser.add_mutually_exclusive_group()
    initial_mode.add_argument("--enable-motion", action="store_true", help="send real robot motion commands")
    initial_mode.add_argument("--dry-run", action="store_true", help="start with the virtual arm preview")
    parser.add_argument("--headless", action="store_true", help="run simulation without a viewer")
    parser.add_argument("--no-panel", action="store_true", help="run without the desktop settings panel")
    args = parser.parse_args(argv)
    if args.backend == "real" and not args.robot_ip:
        parser.error("--backend real requires --robot-ip")
    if args.gripper_ip and args.backend != "real":
        parser.error("--gripper-ip applies only to --backend real")
    if not 3 <= args.gripper_force_n <= 40:
        parser.error("--gripper-force-n must be between 3 and 40 N")
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
        gripper = None
        if args.gripper_ip:
            from .rg2 import RG2

            gripper = stack.enter_context(RG2(args.gripper_ip, force_n=args.gripper_force_n))
        stack.enter_context(backend)
        try:
            run_teleop(device, backend, sim=simulation, dry_run=dry_run, panel=panel, gripper=gripper,
                       gripper_force_n=args.gripper_force_n)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
