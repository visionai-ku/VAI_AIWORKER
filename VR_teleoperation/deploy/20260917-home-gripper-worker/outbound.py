import os
import platform
import subprocess
import time
import traceback
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from rclpy.qos import qos_profile_sensor_data
import zmq

try:
    from .fixed_quest_protocol import JointStateTracker, load_json_message
except ImportError:
    from fixed_quest_protocol import JointStateTracker, load_json_message


JOINT_GROUPS = {
    "lift": ["lift_joint"],
    "head": ["head_joint1", "head_joint2"],
    "arm_l": ["arm_l_joint%d" % idx for idx in range(1, 8)],
    "arm_r": ["arm_r_joint%d" % idx for idx in range(1, 8)],
    "gripper_l": ["gripper_l_joint1"],
    "gripper_r": ["gripper_r_joint1"],
}

ALL_JOINT_NAMES = [
    joint_name
    for joint_names in JOINT_GROUPS.values()
    for joint_name in joint_names
]

ZMQ_PUB_PORT = 5560
JOINT_STATE_TOPIC = "/joint_states"
QOS_DEPTH = 10
PUBLISH_HZ = 100.0


def get_ip_address():
    system_name = platform.system()
    if system_name == "Darwin":
        result = subprocess.run(
            ["ipconfig", "getifaddr", "en0"],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    if system_name == "Linux":
        result = subprocess.run(
            ["hostname", "-I"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None
        ip_list = result.stdout.strip().split()
        return ip_list[0] if ip_list else None

    return None


class SG2OutboundZMQ(Node):
    def __init__(self):
        super().__init__("sg2_outbound_zmq_node")

        self.strict = os.environ.get("SG2_FIXED_QUEST", "0") == "1"
        self.tracker = JointStateTracker() if self.strict else None
        self.bridge = None
        self._last_warning_at = -float("inf")

        self.joint_positions = {}
        self.joint_state_received = False

        ip_addr = get_ip_address()
        pid = os.getpid()

        self.zmq_ctx = zmq.Context()
        self.socket = self.zmq_ctx.socket(zmq.PUB)
        self.socket.set_hwm(1)
        self.socket.bind("tcp://*:%d" % ZMQ_PUB_PORT)
        self.get_logger().info(
            "[PUB] PID %d, IP %s, PORT %d" % (pid, ip_addr, ZMQ_PUB_PORT)
        )
        time.sleep(1.0)

        self.create_subscription(
            JointState,
            JOINT_STATE_TOPIC,
            self.joint_state_callback,
            qos_profile_sensor_data if self.strict else QOS_DEPTH,
        )

        if self.strict:
            self.bridge_sub = self.create_subscription(String, "/fixed_quest/bridge_status", self.bridge_callback, 10)

        self.timer = self.create_timer(1.0 / PUBLISH_HZ, self.timer_callback)
        self.get_logger().info("SG2 Outbound ZMQ ready (%.0fHz)" % PUBLISH_HZ)
        self.get_logger().info("Protocol: %s" % ("strict JSON with source freshness" if self.strict else "legacy pickle"))

    def report_warning(self, message):
        now = time.monotonic()
        if now - self._last_warning_at >= 2.:
            self.get_logger().warn(message)
            self._last_warning_at = now

    def bridge_callback(self, msg):
        try:
            bridge = load_json_message(msg.data)
            status_time = bridge.get("status_time_s")
            if (bridge.get("type") != "sg2_bridge" or type(bridge.get("version")) is not int
                    or bridge["version"] != 1 or bridge.get("strict") is not True
                    or type(status_time) not in (int, float) or not math.isfinite(status_time)
                    or status_time > time.monotonic()):
                raise ValueError("invalid_bridge_status")
            if self.bridge is not None and bridge.get("bridge_id") == self.bridge.get("bridge_id") and status_time <= self.bridge["status_time_s"]:
                return
            self.bridge = bridge
        except (ValueError, TypeError, UnicodeError) as exc:
            self.report_warning("Strict bridge status rejected: %s" % exc)

    def joint_state_callback(self, msg):
        if self.strict:
            try:
                self.tracker.update(list(msg.name), list(msg.position),
                                    int(msg.header.stamp.sec) * 1000000000 + int(msg.header.stamp.nanosec), time.monotonic())
            except (ValueError, TypeError, OverflowError) as exc:
                self.report_warning("Strict feedback rejected: %s" % exc)
            return
        self.joint_positions.update(dict(zip(msg.name, msg.position)))
        self.joint_state_received = True

    def timer_callback(self):
        if self.strict:
            now = time.monotonic()
            state = self.tracker.snapshot(now)
            state["bridge"] = self.bridge
            state["bridge_age_s"] = None if self.bridge is None else max(0., now - self.bridge["status_time_s"])
            try:
                self.socket.send_json(state, flags=zmq.NOBLOCK, allow_nan=False)
            except zmq.Again:
                pass
            return
        if not self.joint_state_received:
            return

        joints_dict = {}
        for group_name, joint_names in JOINT_GROUPS.items():
            joints_dict[group_name] = {
                "name": joint_names,
                "position": [
                    self.joint_positions.get(joint_name, 0.0)
                    for joint_name in joint_names
                ],
            }

        state = {
            "joints": joints_dict,
            "sensors": {},
        }

        try:
            self.socket.send_pyobj(state, flags=zmq.NOBLOCK)
        except zmq.Again:
            pass


def main():
    rclpy.init()
    node = SG2OutboundZMQ()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print("Unexpected error in main: %s" % exc)
        print("Traceback: %s" % traceback.format_exc())
    finally:
        node.get_logger().info("Shutting down SG2 outbound ZMQ...")
        node.socket.close()
        node.destroy_node()
        rclpy.shutdown()
        node.zmq_ctx.term()


if __name__ == "__main__":
    main()
