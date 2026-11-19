"""Map the tracked Quest controller to clutched UR base-frame deltas."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from vai_robotics.core.quest import ControllerMapper, QuestControllerServer, create_self_signed_certificate
from ..teleop_types import INPUT_TIMEOUT, InputSample


DEFAULT_CERT_DIR = Path(__file__).resolve().parents[3] / ".cache" / "quest"


class QuestInput:
    """One controller; facing yaw is taken from the first tracked, released frame."""

    def __init__(self, host, port=8443, hand="right", cert_dir=None):
        self.host = host
        self.port = port
        self.hand = hand
        self.cert_dir = cert_dir or DEFAULT_CERT_DIR
        self.mapper = ControllerMapper(tracked_hand=hand, timeout_s=INPUT_TIMEOUT,
                                       scale=1.0, max_translation_m=None)
        self.receiver = None
        self.reference = {hand: np.eye(4)}
        self.previous_pose = None
        self.last_received = None
        self.waiting_for_release = False

    def __enter__(self):
        if self.host in ("0.0.0.0", "::", "[::]"):
            raise ValueError("Set --quest-host to the operator PC's LAN IP address.")
        cert, key = create_self_signed_certificate(self.cert_dir, self.host)
        self.receiver = QuestControllerServer(self.host, self.port, cert, key).start()
        self.mapper.reset()
        self.previous_pose = self.last_received = None
        self.waiting_for_release = False
        print(f"Quest Browser: {self.receiver.url}  |  {self.hand} grip  |  certificate: {cert}", flush=True)
        return self

    def __exit__(self, *exc):
        self.receiver.close()

    def read(self):
        frame = self.receiver.latest()
        if frame is not None:
            received = frame["received_at"]
            if self.last_received is not None and received - self.last_received > INPUT_TIMEOUT:
                self.mapper.require_release()
            self.last_received = received
        result = self.mapper.update(frame, self.reference, enabled=True)
        self.waiting_for_release = result["state"] == "WAIT_RELEASE"
        target = result["targets"].get(self.hand)
        if target is None:
            self.previous_pose = None
            return InputSample.stationary() if result["state"] == "READY" else None

        previous = self.previous_pose
        self.previous_pose = target
        if previous is None:
            return InputSample.stationary(enabled=True)
        relative = target[:3, :3] @ previous[:3, :3].T
        return InputSample(target[:3, 3] - previous[:3, 3],
                           Rotation.from_matrix(relative).as_rotvec(), True)
