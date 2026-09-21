"""JSON-only PC transport for the strict SG2 robot bridge.

The receiver never unpickles network data. Joint measurements and bridge
heartbeats have separate freshness checks, even when one envelope repeats the
same measurements. No command socket is allocated unless explicitly enabled.
"""

from __future__ import annotations

import copy
import time
import uuid

import numpy as np
import zmq

from .worker.fixed_quest_protocol import JOINT_NAMES, load_json_message, parse_joint_groups


def _number(value, name):
    if type(value) not in (int, float) or not np.isfinite(value):
        raise ValueError(name)
    return float(value)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name)
    return value


def _identifier(value, name):
    if not isinstance(value, str):
        raise ValueError(name)
    try:
        uuid.UUID(value)
    except (ValueError, AttributeError):
        raise ValueError(name) from None
    return value


class SG2RobotFeedbackCache:
    """Validate robot feedback without opening sockets.

    ``received_at`` and ``now`` are PC monotonic seconds. The minimum observed
    receive-minus-send offset estimates additional transport delay without
    subtracting unaligned clocks. Initial one-way delay is unobservable.
    ``ready`` covers fresh measurements and a fresh operational strict bridge;
    ownership and explicit arming remain the caller's responsibility.
    """

    def __init__(self, timeout_s=.2):
        self.timeout_s = _number(timeout_s, "invalid_feedback_timeout")
        if self.timeout_s <= 0:
            raise ValueError("invalid_feedback_timeout")
        self.reset()

    def reset(self):
        self._state, self._q, self._bridge = None, None, None
        self._stream_id, self._identity = None, None
        self._min_offset, self._received_at = None, None
        self._bridge_received_at, self._bridge_age_at_receive = None, None
        self._stamp_progress_at, self._seq_progress_at = None, None
        self._generation, self._requires_resync = 0, True
        self._error, self._bridge_error = "no_feedback", "bridge_missing"

    def acknowledge_resync(self, generation):
        """Acknowledge exactly the generation used to synchronize robot state."""
        if generation != self._generation or self._identity is None:
            return False
        self._requires_resync = False
        return True

    def reject(self, reason):
        self._error = str(reason)

    def _read_bridge(self, payload, sent_at):
        bridge = payload.get("bridge")
        if not isinstance(bridge, dict):
            raise ValueError("bridge_missing")
        if (bridge.get("type") != "sg2_bridge" or type(bridge.get("version")) is not int
                or bridge["version"] != 1 or bridge.get("strict") is not True):
            raise ValueError("strict_bridge_required")
        _identifier(bridge.get("bridge_id"), "invalid_bridge_id")
        if type(bridge.get("ready")) is not bool or not isinstance(bridge.get("state"), str):
            raise ValueError("invalid_bridge_state")
        status_at = _number(bridge.get("status_time_s"), "invalid_bridge_time")
        bridge_age = _number(payload.get("bridge_age_s"), "invalid_bridge_age")
        if status_at < 0 or status_at > sent_at + 1e-6 or bridge_age < 0:
            raise ValueError("invalid_bridge_time")
        _integer(bridge.get("last_command_seq"), "invalid_bridge_sequence", minimum=-1)
        if bridge.get("owner_session_id") is not None:
            _identifier(bridge["owner_session_id"], "invalid_bridge_owner")
        if (self._bridge is not None and bridge["bridge_id"] == self._bridge["bridge_id"]
                and status_at < self._bridge["status_time_s"]):
            raise ValueError("out_of_order_bridge")
        return copy.deepcopy(bridge)

    def ingest(self, payload, received_at=None):
        """Accept a decoded sg2_state envelope; malformed data makes ready false."""
        received_at = time.monotonic() if received_at is None else _number(received_at, "invalid_receive_time")
        try:
            if not isinstance(payload, dict) or payload.get("type") != "sg2_state":
                raise ValueError("legacy_or_unknown_feedback")
            if type(payload.get("version")) is not int or payload["version"] != 1:
                raise ValueError("unsupported_feedback_version")
            stream_id = _identifier(payload.get("stream_id"), "invalid_state_stream")
            seq = _integer(payload.get("seq"), "invalid_state_sequence")
            sent_at = _number(payload.get("sent_at_s"), "invalid_state_send_time")
            if sent_at < 0 or type(payload.get("ready")) is not bool:
                raise ValueError("invalid_state_status")
            if not isinstance(payload.get("state"), str):
                raise ValueError("invalid_state_status")
            missing = payload.get("missing_joints")
            if not isinstance(missing, list) or any(name not in JOINT_NAMES for name in missing):
                raise ValueError("invalid_missing_joints")
            source_at = payload.get("source_time_s")
            stamp = payload.get("source_stamp_ns")
            if source_at is not None:
                source_at = _number(source_at, "invalid_state_source_time")
                if not 0 <= source_at <= sent_at:
                    raise ValueError("invalid_state_source_time")
                stamp = _integer(stamp, "invalid_ros_source_stamp", minimum=1)
            elif stamp is not None or payload["ready"]:
                raise ValueError("missing_state_source_time")
            q = None if missing else np.asarray(parse_joint_groups(payload.get("joints"), named=True), dtype=float)
            if payload["ready"] and (missing or q is None):
                raise ValueError("incomplete_joint_state")

            same_stream = stream_id == self._stream_id
            previous = self._state if same_stream else None
            if previous is not None:
                if (seq < previous["seq"] or sent_at < previous["sent_at_s"]
                        or received_at < self._received_at):
                    raise ValueError("out_of_order_state")
                if previous["source_time_s"] is not None:
                    if (source_at is None or source_at < previous["source_time_s"]
                            or stamp < previous["source_stamp_ns"]):
                        raise ValueError("out_of_order_source")
                if seq == previous["seq"] and (
                        source_at != previous["source_time_s"] or stamp != previous["source_stamp_ns"]
                        or not np.array_equal(q, self._q)):
                    raise ValueError("state_changed_without_sequence")
            offset = received_at - sent_at
            self._min_offset = min(self._min_offset, offset) if same_stream else offset
            if previous is None or stamp != previous["source_stamp_ns"]:
                self._stamp_progress_at = received_at
            if previous is None or seq != previous["seq"]:
                self._seq_progress_at = received_at
            self._stream_id, self._state, self._q = stream_id, copy.deepcopy(payload), q
            self._received_at = received_at
            self._error = None
        except (TypeError, ValueError, KeyError, OverflowError) as exc:
            self._error = str(exc)
            return False

        # A repeated measurement may still carry a new watchdog heartbeat.
        try:
            bridge = self._read_bridge(payload, sent_at)
            self._bridge, self._bridge_error = bridge, None
            self._bridge_received_at = received_at
            self._bridge_age_at_receive = float(payload["bridge_age_s"])
            identity = (stream_id, bridge["bridge_id"])
            if identity != self._identity:
                self._identity = identity
                self._generation += 1
                self._requires_resync = True
        except (TypeError, ValueError, KeyError, OverflowError) as exc:
            self._bridge_error = str(exc)
        return self._error is None and self._bridge_error is None

    def latest(self, now=None):
        now = time.monotonic() if now is None else _number(now, "invalid_now")
        state, bridge = self._state, self._bridge
        age = None
        bridge_age = None
        reason = self._error
        if state is not None:
            robot_now = now - self._min_offset
            if now < self._received_at - 1e-6:
                reason = "non_monotonic_receiver_clock"
            if state["source_time_s"] is not None:
                age = max(0., robot_now - state["source_time_s"], now - self._received_at,
                          now - self._seq_progress_at, now - self._stamp_progress_at)
            if reason is None and (not state["ready"] or self._q is None):
                reason = state["state"] or "robot_state_not_ready"
            if reason is None and (age is None or age > self.timeout_s):
                reason = "stale_joint_state"
            if bridge is not None:
                bridge_age = max(0., robot_now - bridge["status_time_s"],
                                 now - self._bridge_received_at + self._bridge_age_at_receive)
            if reason is None and self._bridge_error is not None:
                reason = self._bridge_error
            if reason is None and (bridge_age is None or bridge_age > self.timeout_s):
                reason = "stale_bridge_status"
            if reason is None and not bridge["ready"]:
                reason = "bridge_not_ready:" + bridge["state"]
        return {
            "ready": reason is None, "reason": "ready" if reason is None else reason,
            "q": None if self._q is None else self._q.copy(), "age_s": age,
            "bridge_age_s": bridge_age, "generation": self._generation,
            "requires_resync": self._requires_resync, "stream_id": self._stream_id,
            "bridge_id": None if bridge is None else bridge["bridge_id"],
            "bridge_state": None if bridge is None else bridge["state"],
            "owner_session_id": None if bridge is None else bridge["owner_session_id"],
            "last_command_seq": None if bridge is None else bridge["last_command_seq"],
            "state": copy.deepcopy(state), "bridge": copy.deepcopy(bridge),
            "diagnostics": {
                "packet_age_s": None if self._received_at is None else max(0., now - self._received_at),
                "source_age_estimate_s": age,
                "bridge_age_s": bridge_age,
                "sequence_progress_age_s": None if self._seq_progress_at is None else max(0., now - self._seq_progress_at),
                "stamp_progress_age_s": None if self._stamp_progress_at is None else max(0., now - self._stamp_progress_at),
                "state_seq": None if state is None else state["seq"],
                "source_stamp_ns": None if state is None else state["source_stamp_ns"],
                "sender_ready": None if state is None else state["ready"],
                "sender_state": None if state is None else state["state"],
                "outbound_joint_state": None if state is None else copy.deepcopy(state.get("joint_state_diagnostics")),
                "inbound_joint_state": None if bridge is None else copy.deepcopy(bridge.get("joint_state_diagnostics")),
            },
        }


class SG2RobotStateReceiver:
    """Poll the robot's JSON PUB stream from the notebook, without another thread."""

    def __init__(self, ip, port=5560, timeout_s=.2):
        if not isinstance(ip, str) or not ip or any(c.isspace() for c in ip) or "://" in ip:
            raise ValueError("invalid_robot_ip")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("invalid_feedback_port")
        self.ip, self.port = ip, port
        host = "[" + ip + "]" if ":" in ip else ip
        self.endpoint = "tcp://%s:%d" % (host, port)
        self.cache = SG2RobotFeedbackCache(timeout_s)
        self._context, self._socket = None, None

    def start(self):
        if self._socket is not None:
            return self
        self._context = zmq.Context()
        try:
            self._socket = self._context.socket(zmq.SUB)
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.setsockopt(zmq.RCVHWM, 64)
            self._socket.setsockopt(zmq.SUBSCRIBE, b"")
            self._socket.connect(self.endpoint)
        except Exception:
            self.close()
            raise
        return self

    def acknowledge_resync(self, generation):
        return self.cache.acknowledge_resync(generation)

    def latest(self, now=None):
        if self._socket is not None:
            for _ in range(128):
                try:
                    raw = self._socket.recv(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                # Legacy outbound.send_pyobj starts with pickle's binary PROTO
                # marker. Inspect only those two bytes; never deserialize it.
                if isinstance(raw, bytes) and len(raw) >= 2 and raw[0] == 0x80 and 2 <= raw[1] <= 5:
                    self.cache.reject(
                        "protocol_mismatch_legacy_pickle: robot feedback is legacy pickle; "
                        "restart both outbound.py and inbound.py with SG2_FIXED_QUEST=1")
                    continue
                try:
                    payload = load_json_message(raw)
                except (TypeError, ValueError, UnicodeDecodeError) as exc:
                    self.cache.reject("invalid_json_feedback:" + str(exc))
                    continue
                self.cache.ingest(payload, received_at=time.monotonic() if now is None else now)
        return self.cache.latest(now)

    def close(self):
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None
        self.cache.reject("receiver_closed")

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()


class SG2RobotCommandPublisher:
    """Explicitly enabled JSON PUB; send success only means local socket acceptance."""

    def __init__(self, enabled=False, port=5561, bind_ip="*"):
        if type(enabled) is not bool or type(port) is not int or not 0 <= port <= 65535:
            raise ValueError("invalid_command_transport_configuration")
        self.enabled, self.port, self.bind_ip = enabled, port, bind_ip
        self._context, self._socket = None, None
        self.endpoint = None
        if enabled:
            self.start()

    def start(self):
        if not self.enabled or self._socket is not None:
            return self
        self._context = zmq.Context()
        try:
            self._socket = self._context.socket(zmq.PUB)
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.setsockopt(zmq.SNDHWM, 1)
            if self.port == 0:
                self.port = self._socket.bind_to_random_port("tcp://" + self.bind_ip)
            else:
                self._socket.bind("tcp://%s:%d" % (self.bind_ip, self.port))
            self.endpoint = self._socket.getsockopt(zmq.LAST_ENDPOINT).decode("utf-8")
        except Exception:
            self.close()
            raise
        return self

    def send(self, payload):
        if not self.enabled:
            return False
        if self._socket is None:
            raise RuntimeError("Command publisher is closed")
        if (not isinstance(payload, dict) or payload.get("type") != "sg2_command"
                or type(payload.get("version")) is not int or payload["version"] != 1):
            raise ValueError("strict_command_envelope_required")
        try:
            self._socket.send_json(payload, flags=zmq.NOBLOCK, allow_nan=False)
        except zmq.Again:
            return False
        return True

    def close(self):
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
