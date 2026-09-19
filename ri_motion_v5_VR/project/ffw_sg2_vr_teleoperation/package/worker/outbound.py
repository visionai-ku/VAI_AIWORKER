import argparse
import os
import platform
import subprocess
import time
import traceback

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
import zmq


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
            QOS_DEPTH,
        )

        self.timer = self.create_timer(1.0 / PUBLISH_HZ, self.timer_callback)
        self.get_logger().info("SG2 Outbound ZMQ ready (%.0fHz)" % PUBLISH_HZ)

    def joint_state_callback(self, msg):
        self.joint_positions.update(dict(zip(msg.name, msg.position)))
        self.joint_state_received = True

    def timer_callback(self):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", action="store_true",
                        help="also publish cameras (camera_outbound.py, :5570/5571/5572)")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = SG2OutboundZMQ()
    nodes = [node]
    if args.meta:
        from camera_outbound import SG2CameraOutboundZMQ
        nodes.append(SG2CameraOutboundZMQ())

    executor = MultiThreadedExecutor()
    for n in nodes:
        executor.add_node(n)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print("Unexpected error in main: %s" % exc)
        print("Traceback: %s" % traceback.format_exc())
    finally:
        node.get_logger().info("Shutting down SG2 outbound ZMQ...")
        node.socket.close()
        for n in nodes[1:]:
            for socket in n.sockets.values():
                socket.close()
        for n in nodes:
            n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for n in nodes:
            n.zmq_ctx.term()


if __name__ == "__main__":
    main()
