"""Run Cartesian UR5e teleoperation with a Quest 3 or a 3Dconnexion SpaceMouse."""

import argparse
from contextlib import ExitStack
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

    def __init__(self, backend):
        self.backend = backend
        self.ready_after_release = False
        self.target = None
        self.last_tick = None

    def update(self, sample, now):
        dt = 0.0 if self.last_tick is None else now - self.last_tick
        self.last_tick = now
        measured_xyz, measured_rot = self.backend.pose()
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
                xyz = xyz + bounded_vector(sample.translation, MAX_LINEAR_SPEED * dt)
                rot = Rotation.from_rotvec(bounded_vector(sample.rotation, MAX_ANGULAR_SPEED * dt)) * rot
                lead = bounded_vector(xyz - measured_xyz, MAX_POSITION_LEAD)
                angular_lead = bounded_vector((rot * measured_rot.inv()).as_rotvec(), MAX_ROTATION_LEAD)
                self.target = measured_xyz + lead, Rotation.from_rotvec(angular_lead) * measured_rot
            self.backend.command(*self.target)
            return
        self.target = None
        self.backend.hold()


def run_teleop(device, backend, *, sim=False, dry_run=False):
    controller = TeleopController(backend)
    last_report = float("-inf")
    active = False
    print("Release the device clutch, then hold it to move. Ctrl+C exits.", flush=True)
    try:
        while not sim or backend.is_running():
            started = time.monotonic()
            controller.update(device.read(), time.monotonic())
            if sim:
                backend.tick()
            if (controller.target is not None) != active:
                active = controller.target is not None
                print("Clutch active" if active else "Holding", flush=True)
            if dry_run and active and started - last_report >= 0.5:
                xyz, rot = controller.target
                print("Dry-run target [m, rad]:", np.round(np.r_[xyz, rot.as_rotvec()], 4), flush=True)
                last_report = started
            time.sleep(max(0.0, started + 1 / HZ - time.monotonic()))
    finally:
        backend.hold()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sim", "real"), required=True)
    parser.add_argument("--input", choices=("quest", "spacemouse"), required=True)
    parser.add_argument("--quest-host", help="operator PC LAN address reachable from the Quest")
    parser.add_argument("--quest-port", type=int, default=8443)
    parser.add_argument("--hand", choices=("left", "right"), default="right")
    parser.add_argument("--button", type=int, default=0, help="SpaceMouse clutch button index (default 0)")
    parser.add_argument("--robot-ip", help="UR controller address for the real backend")
    parser.add_argument("--enable-motion", action="store_true", help="send real robot motion commands")
    parser.add_argument("--headless", action="store_true", help="run simulation without a viewer")
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
    dry_run = not simulation and not args.enable_motion
    with ExitStack() as stack:
        if args.input == "quest":
            device = QuestInput(host=args.quest_host, port=args.quest_port, hand=args.hand)
        else:
            device = SpaceMouseInput(button=args.button)
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
            run_teleop(device, backend, sim=simulation, dry_run=dry_run)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
