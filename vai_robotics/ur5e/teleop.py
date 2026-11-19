"""Teleoperate the UR5e in EEF or joint space with a Quest 3 or SpaceMouse."""

import argparse
from contextlib import ExitStack
import os
import sys
import time

import numpy as np

from .controllers import COMMAND_MODES, CONTROL_LAWS, HZ, TeleopController
from .real import RobotControlStopped
from .rg2 import DEFAULT_GRIP_FORCE_N, DEFAULT_OPEN_WIDTH_MM
from .rg2_visual import OPEN_WIDTH_MM, RG2Preview


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


def operator_state(device, sample, controller):
    """Summarize the latest control tick, including each input's release interlock."""
    if controller.active:
        return "active"
    if getattr(device, "connected", True) is False:
        return "waiting-device"
    if hasattr(device, "clutch_pressed"):
        if device.clutch_pressed is None:
            return "waiting-clutch"
        if device.clutch_pressed and not controller.ready_after_release:
            return "release-clutch"
    if getattr(device, "waiting_for_release", False) is True or (sample is not None and sample.enabled):
        return "release-clutch"
    if sample is not None and controller.ready_after_release:
        return "ready"
    return "waiting-input"


def run_teleop(device, backend, *, sim=False, dry_run=False, panel=None, gripper=None,
               gripper_force_n=DEFAULT_GRIP_FORCE_N, command_mode="eef-delta", control_law="position"):
    limits = backend.joint_limits() if sim and (panel is not None or command_mode.startswith("joint-")) else None
    controller = TeleopController(backend, dry_run=dry_run, command_mode=command_mode,
                                  control_law=control_law, joint_limits=limits)
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
            command_request = panel.take_command_mode() if panel is not None else None
            values = panel.take_target() if panel is not None else None
            if panel is not None and panel.open_width_mm != open_width_mm:
                open_width_mm = panel.open_width_mm
                if gripper is not None:
                    gripper.set_open_width(open_width_mm)
            change_mode = request is not None and request != controller.dry_run
            # Going back to preview takes priority over polling USB or network input.
            sample = None if change_mode and request else device.read()
            if change_mode:
                values = None
                if gripper is not None:
                    gripper.update(None)
                message = controller.set_mode(request, simulation=sim, sample=sample, device=device)
                # Don't use input captured before a transition opened an RTDE session.
                if controller.dry_run == request:
                    sample = None
                    preview.reset(baseline_width)
                    if request and gripper is not None:
                        # The arm writer is detached. Finish any Modbus write already in flight
                        # before presenting the transition to read-only control as complete.
                        gripper.wait_for_writes()
            try:
                arm_sample = sample
                if command_request is not None:
                    # Serialize execution- and command-mode changes; never interpret an
                    # old target or held device input using the new coordinate system.
                    if request is None:
                        message = controller.set_command_mode(command_request, simulation=sim,
                                                              sample=sample, device=device)
                    arm_sample = values = None
                notice = controller.update(arm_sample, time.monotonic(), values=values,
                                           scale=panel.scale if panel is not None else 1.0)
                if values is not None:
                    message = notice or ""
            except RobotControlStopped as error:
                if sim or panel is None or controller.dry_run:
                    raise
                if gripper is not None:
                    gripper.update(None)
                stop_message = controller.set_mode(True, simulation=False, sample=None)
                if gripper is not None:
                    gripper.wait_for_writes()
                sample = None
                preview.reset(baseline_width)
                controller.update(sample, time.monotonic())  # Verify feedback before showing a measured pose.
                message = f"DRY RUN 전환 · {error}"
                if stop_message:
                    message += f" · {stop_message}"
                print(message, file=sys.stderr, flush=True)
            if request is not None:
                if not change_mode:
                    message = ""
                panel.mode_result(dry_run=controller.dry_run, message=message)
                requested_mode = "DRY RUN" if request else ("SIM RUN" if sim else "REAL RUN")
                print(f"{requested_mode} · {message or '전환 완료'}", flush=True)
            if command_request is not None:
                if request is not None and not message:
                    message = "실행 모드 전환 후 명령 모드를 다시 선택하세요."
                panel.command_mode_result(command_mode=controller.command_mode, message=message)
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
                backend.tick(dry_run=controller.dry_run, target=controller.target,
                             joint_target=controller.joint_target if controller.dry_run else None,
                             gripper_width_mm=width)
            elif view is not None:
                q = controller.measured_joints if controller.joint_space else backend.joints()
                view.update(q=q, measured_tcp=controller.measured, dry_run=controller.dry_run,
                            target=controller.target if controller.dry_run else None,
                            joint_target=controller.joint_target if controller.dry_run else None,
                            gripper_width_mm=width)
            if controller.active != active:
                active = controller.active
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
                              operator_state=operator_state(device, sample, controller),
                              measured=controller.measured,
                              measured_joints=controller.measured_joints, target_joints=controller.joint_target,
                              target=controller.target, message=message, gripper=gripper_text)
            elif controller.dry_run and active and started - last_report >= 0.5:
                if controller.joint_space:
                    print("Dry-run joints [rad]:", np.round(controller.joint_target, 4), flush=True)
                else:
                    xyz, rot = controller.target
                    print("Dry-run target [m, rad]:", np.round(np.r_[xyz, rot.as_rotvec()], 4), flush=True)
                last_report = started
            time.sleep(max(0.0, started + 1 / HZ - time.monotonic()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sim", "real"), required=True)
    parser.add_argument("--input", choices=("quest", "spacemouse"), required=True)
    parser.add_argument("--command-mode", choices=COMMAND_MODES, default="eef-delta")
    parser.add_argument("--control-law", choices=CONTROL_LAWS, default="position")
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
    if args.command_mode.endswith("-absolute") and not show_panel:
        parser.error("Absolute teleoperation requires the target panel. Use TeleopController.update(values=...) for headless targets.")
    with ExitStack() as stack:
        panel = None
        if show_panel:
            from .teleop_panel import TeleopPanel

            device_name = f"Quest 3 · {args.hand}" if args.input == "quest" else f"SpaceMouse · 버튼 {args.button}"
            panel = TeleopPanel(simulation=simulation, device_name=device_name, dry_run=dry_run,
                                robot_ip=args.robot_ip, command_mode=args.command_mode, control_law=args.control_law)
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
            backend = RealUR5e(args.robot_ip, enable_motion=args.enable_motion, frequency=HZ,
                               joint_impedance=args.command_mode.startswith("joint-") and args.control_law == "impedance")
        stack.enter_context(device)
        gripper = None
        if args.gripper_ip:
            from .rg2 import RG2

            gripper = stack.enter_context(RG2(args.gripper_ip, force_n=args.gripper_force_n))
        stack.enter_context(backend)
        try:
            run_teleop(device, backend, sim=simulation, dry_run=dry_run, panel=panel, gripper=gripper,
                       gripper_force_n=args.gripper_force_n, command_mode=args.command_mode,
                       control_law=args.control_law)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
