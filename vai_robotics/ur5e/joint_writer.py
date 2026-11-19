"""Stream joint equilibria to the UR controller's native 500 Hz torque loop."""

from pathlib import Path
import re
import secrets
import socket
import struct
from string import Template
import time

from .impedance import JOINT_TORQUE_LIMIT
from .impedance_gains import default_impedance_gains, max_impedance_gains, validate_impedance_gains


def _read_int(stream, deadline):
    data = bytearray()
    while len(data) < 4:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Joint impedance acknowledgement timed out.")
        stream.settimeout(remaining)
        chunk = stream.recv(4 - len(data))
        if not chunk:
            raise ConnectionError("UR5e closed the joint impedance session.")
        data.extend(chunk)
    return struct.unpack("!i", data)[0]


def _check_firmware(host):
    from dashboard_client import DashboardClient

    dashboard = DashboardClient(host)
    dashboard.connect(1000)
    try:
        version = dashboard.polyscopeVersion()
        match = re.search(r"\b(\d+)\.(\d+)\.", version)
        if match is None or int(match[1]) != 5 or int(match[2]) < 23:
            raise RuntimeError(f"Joint impedance needs UR5e PolyScope 5.23 or later (reported {version!r}).")
        if not dashboard.isInRemoteControl():
            raise RuntimeError("Joint impedance needs UR5e remote control mode.")
    finally:
        dashboard.disconnect()


class JointImpedanceWriter:
    """A single owner, one outstanding command, no retransmission after faults."""

    def __init__(self, stream, receiver):
        self.stream = stream
        self.receiver = receiver
        self.sequence = 0

    @classmethod
    def open(cls, host, receiver, max_lead):
        _check_firmware(host)
        template = Template(Path(__file__).with_name("joint_impedance.script").read_text())
        with socket.create_connection((host, 30002), timeout=1.0) as upload, socket.socket() as listener:
            local_ip = upload.getsockname()[0]
            robot_ip = upload.getpeername()[0]
            listener.bind((local_ip, 0))
            listener.listen(1)
            listener.settimeout(3.0)
            nonce = secrets.randbelow(2**30) + 1
            upper = max_impedance_gains("joint")
            script = template.substitute(HOST=local_ip, PORT=listener.getsockname()[1], NONCE=nonce,
                                         MAX_STIFFNESS=upper["stiffness"], MAX_DAMPING=upper["damping"],
                                         TORQUE_LIMIT=list(JOINT_TORQUE_LIMIT), MAX_LEAD=max_lead)
            if (receiver.getRuntimeState() != 1 or receiver.getRobotMode() != 7 or
                    receiver.getSafetyMode() not in (1, 2)):
                raise RuntimeError("Joint impedance requires an idle program and motion-ready safety state.")
            upload.sendall(script.encode("ascii"))
            stream, peer = listener.accept()
            try:
                if peer[0] != robot_ip or _read_int(stream, time.monotonic() + 0.5) != nonce:
                    raise RuntimeError("Joint impedance session handshake did not match this upload.")
                stream.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                writer = cls(stream, receiver)
                deadline = time.monotonic() + 1.0
                while True:
                    writer.hold()
                    if receiver.getRuntimeState() == 2:
                        return writer
                    if time.monotonic() >= deadline:
                        raise RuntimeError("UR5e did not report the owned joint impedance program running.")
                    time.sleep(0.01)
            except BaseException:
                stream.close()
                raise

    def _send(self, mode, values=(0.,) * 18):
        # A setpoint and both gain vectors are one sequenced command. All 18
        # payload values use 1e-6 SI units; idle and stop carry zeros.
        packet = struct.pack("!20i", self.sequence + 1, mode, *(round(x * 1_000_000) for x in values))
        self.sequence += 1
        # An explicit stop can need time to decelerate before the acknowledgement.
        timeout = 0.10 if mode == 1 else 0.65
        try:
            self.stream.settimeout(timeout)
            self.stream.sendall(packet)
            accepted = _read_int(self.stream, time.monotonic() + timeout)
            if accepted != self.sequence:
                raise RuntimeError("UR5e did not acknowledge the current joint impedance command.")
        except (OSError, RuntimeError):
            self.disconnect()  # Never replay a target or consume a previous answer as a new one.
            raise
        return True

    def command(self, q, gains=None):
        gains = default_impedance_gains("joint") if gains is None else validate_impedance_gains("joint", gains)
        return self._send(1, (*q, *gains["stiffness"], *gains["damping"]))

    def hold(self):
        return self._send(0)

    def kickWatchdog(self):
        # Only called before engagement or after hold. An idle packet carries no torque.
        return self.hold()

    def isConnected(self):
        return self.stream.fileno() >= 0

    def isProgramRunning(self):
        return self.receiver.getRuntimeState() == 2

    def stopScript(self):
        return self._send(2)

    def disconnect(self):
        self.stream.close()
