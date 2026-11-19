"""Process one operator tick and publish its confirmed UR5e feedback."""

from dataclasses import dataclass
import sys

import numpy as np

from .controllers import TeleopController
from .real import RobotControlStopped
from .rg2 import DEFAULT_OPEN_WIDTH_MM
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
    return "놓음" if not sample.enabled else "Clutch를 놓으세요"


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


@dataclass(frozen=True)
class _PanelRequests:
    """All events received together, before the next operator sample."""

    dry_run: bool | None = None
    command_mode: str | None = None
    control_law: str | None = None
    gains: dict | None = None
    target: list[float] | None = None

    @classmethod
    def poll(cls, panel):
        if panel is None:
            return cls()
        dry_run = panel.poll()
        if not panel.is_open:
            return None
        return cls(dry_run=dry_run, command_mode=panel.take_command_mode(), control_law=panel.take_control_law(),
                   gains=panel.take_impedance_gains(), target=panel.take_target())

    @property
    def config_count(self):
        return sum(value is not None for value in (self.command_mode, self.control_law, self.gains))

    @property
    def gain_space(self):
        space = self.gains.get("space") if isinstance(self.gains, dict) else None
        return space if space in ("eef", "joint") else None

    def configure(self, controller, *, simulation, sample, device):
        """Apply a single settings request only after the execution mode is settled."""
        if self.config_count > 1:
            return "제어 설정은 한 번에 하나씩 변경하세요."
        if self.command_mode is not None:
            return controller.set_command_mode(self.command_mode, simulation=simulation, sample=sample, device=device)
        if self.control_law is not None:
            return controller.set_control_law(self.control_law, simulation=simulation, sample=sample, device=device)
        if self.gain_space is None:
            return "Stiffness / Damping을 적용할 대상(TCP 또는 Joint)을 확인하세요."
        return controller.set_impedance_gains(self.gain_space, self.gains.get("gains"),
                                              simulation=simulation, sample=sample, device=device)

    def acknowledge(self, panel, controller, *, simulation, run_switch_requested, message):
        if self.dry_run is not None:
            if not run_switch_requested:
                message = ""
            panel.mode_result(dry_run=controller.dry_run, message=message)
            run_mode = "DRY RUN" if self.dry_run else ("SIM RUN" if simulation else "REAL RUN")
            print(f"{run_mode} · {message or '전환 완료'}", flush=True)
            if self.config_count and not message:
                message = "실행 모드 전환 후 설정을 다시 선택하세요."
        if self.command_mode is not None:
            panel.command_mode_result(command_mode=controller.command_mode, message=message)
        if self.control_law is not None:
            panel.control_law_result(control_law=controller.control_law, message=message)
        if self.gains is not None:
            space = self.gain_space
            gains = controller.backend.get_impedance_gains(space) if space is not None else None
            panel.impedance_gains_result(space=space, gains=gains, message=message)
        return message


class _SessionGripper:
    """Track the displayed aperture, the dry-run baseline, and the hardware command separately."""

    def __init__(self, hardware, simulation, force_n):
        self.hardware = hardware
        self.simulation = simulation
        self.force_n = force_n
        self.preview = RG2Preview(force_n)
        self.open_width_mm = hardware.open_width_mm if hardware is not None else DEFAULT_OPEN_WIDTH_MM
        self.baseline_mm = self.open_width_mm if simulation else OPEN_WIDTH_MM

    def set_open_width(self, width):
        if width != self.open_width_mm:
            self.open_width_mm = width
            if self.hardware is not None:
                self.hardware.set_open_width(width)

    def update(self, device, sample, *, dry_run, run_switch_requested, monotonic):
        chord = sample.gripper_chord if sample is not None else getattr(device, "gripper_chord", None)
        measured = None
        if self.hardware is not None:
            self.hardware.update(None if dry_run or run_switch_requested else chord)
            measured = self.hardware.width_mm
            if measured is not None:
                self.baseline_mm = measured
        if not self.simulation and not dry_run:
            return measured

        command = sample.gripper_command if sample is not None else None
        if dry_run and chord is not True:
            # Released preview follows real feedback or the frozen SIM RUN state.
            self.preview.reset(self.baseline_mm)
            command = None
        width = self.preview.update(monotonic(), self.baseline_mm, command, chord,
                                    open_width_mm=self.open_width_mm)
        if self.simulation and not dry_run:
            self.baseline_mm = width
        return width

    def status(self, device, width, dry_run):
        if self.simulation:
            text = f"RG2 · Simulated Width {width:.1f} mm"
        else:
            text = self.hardware.status if self.hardware is not None else "RG2 · Gripper IP 미설정"
            if dry_run:
                text += f" · Preview Width {width:.1f} mm"
        text += f" · Grip Force (set) {self.force_n:g} N"
        if hasattr(device, "gripper_chord"):
            state = {True: "누름", False: "놓음", None: "입력 대기"}[device.gripper_chord]
            text = f"버튼 0+1: {state} · {text}"
        return text


class TeleopSession:
    def __init__(self, device, backend, *, simulation, dry_run, panel, gripper,
                 gripper_force_n, command_mode, control_law, monotonic):
        self.device = device
        self.backend = backend
        self.simulation = simulation
        self.panel = panel
        self.monotonic = monotonic
        limits = None
        if simulation and (panel is not None or command_mode.startswith("joint-")):
            limits = backend.joint_limits()
        self.controller = TeleopController(backend, dry_run=dry_run, command_mode=command_mode,
                                          control_law=control_law, joint_limits=limits)
        self.gripper = _SessionGripper(gripper, simulation, gripper_force_n)
        self.message = ""
        self._was_active = False
        self._last_report = float("-inf")

    def step(self, view, started):
        """Return False when the panel closes, before sampling or sending commands."""
        requests = _PanelRequests.poll(self.panel)
        if requests is None:
            return False
        if self.panel is not None:
            self.gripper.set_open_width(self.panel.open_width_mm)

        run_switch = requests.dry_run is not None and requests.dry_run != self.controller.dry_run
        # A return to preview takes priority over polling USB or network input.
        sample = None if run_switch and requests.dry_run else self.device.read()
        if run_switch:
            sample = self._switch_run_mode(requests.dry_run, sample)

        try:
            self._advance_arm(requests, sample, run_switch)
        except RobotControlStopped as error:
            if self.simulation or self.panel is None or self.controller.dry_run:
                raise
            self._recover_to_preview(error)
            sample = None

        self.message = requests.acknowledge(self.panel, self.controller, simulation=self.simulation,
                                            run_switch_requested=run_switch, message=self.message)
        width = self.gripper.update(self.device, sample, dry_run=self.controller.dry_run,
                                    run_switch_requested=run_switch, monotonic=self.monotonic)
        self._render(view, width)
        self._report(sample, width, started)
        return True

    def _switch_run_mode(self, dry_run, sample):
        hardware = self.gripper.hardware
        if hardware is not None:
            hardware.update(None)
        self.message = self.controller.set_mode(dry_run, simulation=self.simulation, sample=sample, device=self.device)
        if self.controller.dry_run != dry_run:
            return sample  # Request refused; the old mode still owns the input.
        self.gripper.preview.reset(self.gripper.baseline_mm)
        if dry_run and hardware is not None:
            # Finish pending Modbus writes before confirming read-only control.
            hardware.wait_for_writes()
        return None  # Input captured before switching writers cannot arm the new mode.

    def _advance_arm(self, requests, sample, run_switch):
        values = None if run_switch else requests.target
        if requests.config_count:
            if requests.dry_run is None:
                self.message = requests.configure(self.controller, simulation=self.simulation,
                                                  sample=sample, device=self.device)
            # Never consume a held sample or target in a different control space.
            sample = values = None
        notice = self.controller.update(sample, self.monotonic(), values=values,
                                        scale=self.panel.scale if self.panel is not None else 1.0)
        if values is not None:
            self.message = notice or ""

    def _recover_to_preview(self, error):
        hardware = self.gripper.hardware
        if hardware is not None:
            hardware.update(None)
        stop_message = self.controller.set_mode(True, simulation=False, sample=None)
        if hardware is not None:
            hardware.wait_for_writes()
        self.gripper.preview.reset(self.gripper.baseline_mm)
        self.controller.update(None, self.monotonic())  # Confirm fresh feedback before rendering.
        self.message = f"DRY RUN 전환 · {error}"
        if stop_message:
            self.message += f" · {stop_message}"
        print(self.message, file=sys.stderr, flush=True)

    def _render(self, view, width):
        arm = self.controller
        joint_preview = arm.joint_target if arm.dry_run else None
        if self.simulation:
            self.backend.tick(dry_run=arm.dry_run, target=arm.target, joint_target=joint_preview,
                              gripper_width_mm=width)
        elif view is not None:
            q = arm.measured_joints if arm.joint_space else self.backend.joints()
            view.update(q=q, measured_tcp=arm.measured, dry_run=arm.dry_run,
                        target=arm.target if arm.dry_run else None, joint_target=joint_preview,
                        gripper_width_mm=width)

    def _report(self, sample, width, started):
        arm = self.controller
        if arm.active != self._was_active:
            self._was_active = arm.active
            print("Clutch active" if arm.active else "Holding", flush=True)
        if self.panel is not None:
            gripper_text = self.gripper.status(self.device, width, arm.dry_run)
            self.panel.refresh(dry_run=arm.dry_run, clutch=clutch_text(self.device, sample, arm.active),
                               operator_state=operator_state(self.device, sample, arm),
                               measured=arm.measured, target=arm.target, measured_joints=arm.measured_joints,
                               target_joints=arm.joint_target, message=self.message,
                               gripper=gripper_text)
        elif arm.dry_run and arm.active and started - self._last_report >= 0.5:
            if arm.joint_space:
                print("Dry-run joints [rad]:", np.round(arm.joint_target, 4), flush=True)
            else:
                xyz, rot = arm.target
                print("Dry-run target [m, rad]:", np.round(np.r_[xyz, rot.as_rotvec()], 4), flush=True)
            self._last_report = started
