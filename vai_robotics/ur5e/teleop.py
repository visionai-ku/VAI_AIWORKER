"""Teleoperate the UR5e in EEF or joint space with a Quest 3 or SpaceMouse."""

import argparse
from contextlib import ExitStack
import os
import sys
import time

from .controllers import COMMAND_MODES, CONTROL_LAWS, HZ, TeleopController
from .rg2 import DEFAULT_GRIP_FORCE_N
from .teleop_session import TeleopSession, clutch_text, operator_state


def run_teleop(device, backend, *, sim=False, dry_run=False, panel=None, gripper=None,
               gripper_force_n=DEFAULT_GRIP_FORCE_N, command_mode="eef-delta", control_law="position",
               recording=None):
    session = TeleopSession(device, backend, simulation=sim, dry_run=dry_run, panel=panel, gripper=gripper,
                            gripper_force_n=gripper_force_n, command_mode=command_mode, control_law=control_law,
                            monotonic=time.monotonic, recording=recording)
    print("Release the device clutch, then hold it to move. Ctrl+C exits.", flush=True)
    with ExitStack() as cleanup:
        view = backend if sim else None
        if not sim and panel is not None:
            from .real_viewer import RealViewer

            view = cleanup.enter_context(RealViewer())
            # Release the RTDE writer before waiting for the GUI to close.
            cleanup.callback(backend.set_motion_enabled, False)
        else:
            cleanup.callback(backend.hold)
        if gripper is not None:
            cleanup.callback(gripper.update, None)
        while view is None or view.is_running():
            started = time.monotonic()
            if not session.step(view, started):
                break
            time.sleep(max(0.0, started + 1 / HZ - time.monotonic()))


def _argument_parser():
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
    parser.add_argument("--record-dir", help="base directory for local LeRobot datasets (default: datasets/ur5e)")
    parser.add_argument("--record-task", help="episode task description (default: UR5e teleoperation)")
    parser.add_argument("--record-max-seconds", type=int, help="auto-save after this duration at 50 Hz (default: 120)")
    parser.add_argument("--record-camera", action="append", default=[], metavar="NAME=tcp://HOST:PORT",
                        help="AI Worker ZMQ RGB stream to record; repeat for multiple cameras")
    return parser


def _parse_args(argv):
    parser = _argument_parser()
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

    has_display = sys.platform != "linux" or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    args.show_panel = has_display and not args.no_panel and not args.headless
    if args.command_mode.endswith("-absolute") and not args.show_panel:
        parser.error("Absolute teleoperation requires the target panel. "
                     "Use TeleopController.update(values=...) for headless targets.")
    recording_options = (args.record_dir, args.record_task, args.record_max_seconds)
    if (any(value is not None for value in recording_options) or args.record_camera) and not args.show_panel:
        parser.error("Recording controls require the desktop panel")
    if args.record_max_seconds is not None and args.record_max_seconds < 1:
        parser.error("--record-max-seconds must be positive")
    if args.record_task is not None and not args.record_task.strip():
        parser.error("--record-task must contain a task description")
    from .recording_camera import parse_camera_endpoints

    try:
        args.camera_endpoints = parse_camera_endpoints(args.record_camera)
    except ValueError as error:
        parser.error(str(error))
    return args


def _make_panel(args, dry_run):
    from .teleop_panel import TeleopPanel

    device_name = f"Quest 3 · {args.hand}" if args.input == "quest" else f"SpaceMouse · 버튼 {args.button}"
    return TeleopPanel(
        simulation=args.backend == "sim", device_name=device_name, dry_run=dry_run,
        robot_ip=args.robot_ip, command_mode=args.command_mode, control_law=args.control_law,
        joint_rings=args.input == "spacemouse",
    )


def _make_input(args):
    from .inputs import QuestInput, SpaceMouseInput

    if args.input == "quest":
        return QuestInput(host=args.quest_host, port=args.quest_port, hand=args.hand)
    mouse_only_preview = args.show_panel and args.command_mode.startswith("joint-") and not args.enable_motion
    return SpaceMouseInput(button=args.button, wait_for_device=args.show_panel, allow_mouse_only=mouse_only_preview)


def _make_backend(args):
    if args.backend == "sim":
        from .sim_teleop import SimBackend

        return SimBackend(hz=HZ, headless=args.headless)

    from .real import RealUR5e

    print("Real robot motion ENABLED" if args.enable_motion else "Real robot, receive-only dry run", flush=True)
    return RealUR5e(
        args.robot_ip, enable_motion=args.enable_motion, frequency=HZ,
        joint_impedance=args.command_mode.startswith("joint-") and args.control_law == "impedance",
    )


def main(argv=None):
    args = _parse_args(argv)
    simulation = args.backend == "sim"
    dry_run = args.dry_run or (not simulation and not args.enable_motion)

    with ExitStack() as stack:
        panel = _make_panel(args, dry_run) if args.show_panel else None
        recording = None
        if panel is not None:
            from .demonstrations import DemonstrationRecording
            from .recording_camera import RecordingCameras

            stack.callback(panel.close)
            cameras = stack.enter_context(RecordingCameras(args.camera_endpoints))
            recording = DemonstrationRecording(args.record_dir or "datasets/ur5e", simulation=simulation,
                                               task=args.record_task or "UR5e teleoperation", cameras=cameras,
                                               max_seconds=args.record_max_seconds or 120)
            stack.callback(recording.close)
        device = _make_input(args)
        backend = _make_backend(args)
        stack.enter_context(device)
        gripper = None
        if args.gripper_ip:
            from .rg2 import RG2

            gripper = stack.enter_context(RG2(args.gripper_ip, force_n=args.gripper_force_n))
        stack.enter_context(backend)
        try:
            run_teleop(device, backend, sim=simulation, dry_run=dry_run, panel=panel, gripper=gripper,
                       gripper_force_n=args.gripper_force_n, command_mode=args.command_mode,
                       control_law=args.control_law, recording=recording)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
