import traceback
import os

from builtin_interfaces.msg import Duration
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import zmq


CONTROLLER_CONFIG = {
    "arm_l": {
        "topic": "/leader/joint_trajectory_command_broadcaster_left/joint_trajectory",
        "groups": [
            ("arm_l", ["arm_l_joint%d" % idx for idx in range(1, 8)]),
            ("gripper_l", ["gripper_l_joint1"]),
        ],
    },
    "arm_r": {
        "topic": "/leader/joint_trajectory_command_broadcaster_right/joint_trajectory",
        "groups": [
            ("arm_r", ["arm_r_joint%d" % idx for idx in range(1, 8)]),
            ("gripper_r", ["gripper_r_joint1"]),
        ],
    },
    "head": {
        "topic": "/leader/joystick_controller_left/joint_trajectory",
        "groups": [
            ("head", ["head_joint1", "head_joint2"]),
        ],
    },
    "lift": {
        "topic": "/leader/joystick_controller_right/joint_trajectory",
        "groups": [
            ("lift", ["lift_joint"]),
        ],
    },
}

ZMQ_SUB_IP = os.environ.get("SG2_ZMQ_SUB_IP", "192.168.6.102")
ZMQ_SUB_PORT = 5561
QOS_DEPTH = 10
PUBLISH_HZ = 100.0


class SG2ZMQRosBridge(Node):
    def __init__(self):
        super().__init__("sg2_zmq_ros2_bridge_node")

        self.zmq_ctx = zmq.Context()
        self.socket = self.zmq_ctx.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.set_hwm(1)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.connect("tcp://%s:%d" % (ZMQ_SUB_IP, ZMQ_SUB_PORT))
        self.get_logger().info(
            "[SUB] Connected to tcp://%s:%d" % (ZMQ_SUB_IP, ZMQ_SUB_PORT)
        )

        self.traj_pubs = {}
        for module, cfg in CONTROLLER_CONFIG.items():
            self.traj_pubs[module] = self.create_publisher(
                JointTrajectory,
                cfg["topic"],
                QOS_DEPTH,
            )
            self.get_logger().info(
                "  Publisher: %s -> %s" % (module, cfg["topic"])
            )

        self.timer = self.create_timer(1.0 / PUBLISH_HZ, self.on_timer)
        self.get_logger().info("SG2 ZMQ ROS2 Bridge ready (%.0fHz)" % PUBLISH_HZ)

    def collect_controller_command(self, module, cfg, data):
        joint_names = []
        positions = []

        for group_name, group_joint_names in cfg["groups"]:
            group_positions = data.get(group_name)
            if group_positions is None:
                continue

            if len(group_positions) != len(group_joint_names):
                self.get_logger().warn(
                    "Skip %s/%s: got %d positions, expected %d"
                    % (
                        module,
                        group_name,
                        len(group_positions),
                        len(group_joint_names),
                    )
                )
                return None, None

            joint_names.extend(group_joint_names)
            positions.extend(float(position) for position in group_positions)

        if len(joint_names) == 0:
            return None, None
        return joint_names, positions

    def on_timer(self):
        try:
            data = self.socket.recv_pyobj(flags=zmq.NOBLOCK)
        except zmq.Again:
            return
        except Exception as exc:
            self.get_logger().error("ZMQ recv error: %s" % exc)
            return

        for module, cfg in CONTROLLER_CONFIG.items():
            joint_names, positions = self.collect_controller_command(module, cfg, data)
            if joint_names is None:
                continue

            msg = JointTrajectory()
            msg.header.stamp.sec = 0
            msg.header.stamp.nanosec = 0
            msg.joint_names = joint_names

            point = JointTrajectoryPoint()
            point.positions = positions
            point.time_from_start = Duration(sec=0, nanosec=0)
            msg.points = [point]

            self.traj_pubs[module].publish(msg)


def main():
    rclpy.init()
    node = SG2ZMQRosBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print("Unexpected error in main: %s" % exc)
        print("Traceback: %s" % traceback.format_exc())
    finally:
        node.get_logger().info("Shutting down SG2 ZMQ ROS2 bridge...")
        node.socket.close()
        node.destroy_node()
        rclpy.shutdown()
        node.zmq_ctx.term()


if __name__ == "__main__":
    main()
