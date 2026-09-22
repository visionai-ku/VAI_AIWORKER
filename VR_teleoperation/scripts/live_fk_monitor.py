"""Live FK monitor for safety-box boundary discovery.

Connects to the robot's outbound joint-state stream (package/worker/
outbound.py, either strict JSON or legacy pickle -- auto-detected, same as
package/episode_recorder.py does) as a SUB client, runs MuJoCo forward
kinematics on every received pose and continuously prints both wrist tip
positions in the world/base frame. Move the real robot by hand (free-drive)
and read the x/y/z off the terminal at whatever pose you want to record --
no need to copy joint values anywhere.

    python scripts/live_fk_monitor.py --robot-ip 192.168.6.2
    python scripts/live_fk_monitor.py --robot-ip 192.168.6.2 --rate 10

This never publishes anything and never opens the command socket; it only
subscribes to :5560 (default) and reads.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import runpy
import time

import mujoco
import numpy as np
import zmq

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "package/init_project.py"))

from package.utils import SG2_WRIST_BODY_NAMES, SG2_WRIST_P_LOCAL, SG2_XML_PATH
from package.worker.fixed_quest_protocol import JOINT_NAMES, load_json_message


def build_model():
    model = mujoco.MjModel.from_xml_path(str(SG2_XML_PATH))
    data = mujoco.MjData(model)
    qpos_indices = np.array([int(model.joint(name).qposadr[0]) for name in JOINT_NAMES])
    wrist_body_ids = {hand: int(model.body(name).id) for hand, name in SG2_WRIST_BODY_NAMES.items()}
    return model, data, qpos_indices, wrist_body_ids


def wrist_tip_positions(model, data, wrist_body_ids):
    positions = {}
    for hand, body_id in wrist_body_ids.items():
        rotation = data.xmat[body_id].reshape(3, 3)
        positions[hand] = data.xpos[body_id] + rotation @ SG2_WRIST_P_LOCAL[hand]
    return positions


def open_state_socket(robot_ip, port):
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.CONFLATE, 1)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect("tcp://%s:%d" % (robot_ip, port))
    return ctx, sock


def recv_latest_raw(sock):
    """Drain the socket and keep only the newest message."""
    raw = None
    while True:
        try:
            raw = sock.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return raw


def decode_message(raw):
    """Strict outbound.py (SG2_FIXED_QUEST=1) sends JSON; legacy sends pickle.

    Both come from this robot's own trusted outbound socket on the local
    network -- the same trust boundary package/episode_recorder.py's
    decode_message() already relies on for this exact stream.
    """
    if raw is None:
        return None
    if raw[:1] == b"{":
        return load_json_message(raw)
    return pickle.loads(raw)


def parse_robot_state(msg):
    """-> q in JOINT_NAMES order, or None when incomplete/not ready."""
    if not isinstance(msg, dict) or msg.get("ready") is False:
        return None
    joints = msg.get("joints") or {}
    by_name = {}
    for group in joints.values():
        for name, position in zip(group["name"], group["position"]):
            by_name[name] = float(position)
    if not all(name in by_name for name in JOINT_NAMES):
        return None
    return np.asarray([by_name[name] for name in JOINT_NAMES], dtype=np.float64)


def run(robot_ip, port=5560, rate=5.0):
    model, data, qpos_indices, wrist_body_ids = build_model()
    period = 1.0 / float(rate)
    ctx, sock = open_state_socket(robot_ip, port)
    print("Subscribed to tcp://%s:%d -- Ctrl+C to stop." % (robot_ip, port))
    last_q = None
    try:
        next_print = time.monotonic()
        while True:
            try:
                msg = decode_message(recv_latest_raw(sock))
                q = parse_robot_state(msg)
            except (pickle.UnpicklingError, EOFError, ValueError, KeyError, TypeError, UnicodeDecodeError):
                q = None
            if q is not None:
                last_q = q
                data.qpos[qpos_indices] = q
                mujoco.mj_kinematics(model, data)

            now = time.monotonic()
            if now >= next_print:
                next_print = now + period
                if last_q is None:
                    print("\rwaiting for joint feedback...%-20s" % "", end="", flush=True)
                else:
                    tips = wrist_tip_positions(model, data, wrist_body_ids)
                    left, right = tips["left"], tips["right"]
                    print(
                        "\rL x=%+.4f y=%+.4f z=%+.4f | R x=%+.4f y=%+.4f z=%+.4f"
                        % (left[0], left[1], left[2], right[0], right[1], right[2]),
                        end="", flush=True,
                    )
            time.sleep(0.005)
    except KeyboardInterrupt:
        print()
    finally:
        sock.close(linger=0)
        ctx.term()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", required=True, help="Robot outbound host, e.g. 192.168.6.2")
    parser.add_argument("--port", type=int, default=5560)
    parser.add_argument("--rate", type=float, default=5.0, help="Print refresh rate in Hz")
    args = parser.parse_args()
    run(robot_ip=args.robot_ip, port=args.port, rate=args.rate)


if __name__ == "__main__":
    main()
