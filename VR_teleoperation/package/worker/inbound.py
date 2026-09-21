import traceback
import os
import json
import math
import time

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from rclpy.qos import qos_profile_sensor_data
import zmq

try:
    from .fixed_quest_protocol import JointStateTracker, StrictCommandGate, load_json_message, FIXED_QUEST_VR_GATE_OPTIONS
except ImportError:  # Also allow the existing `python inbound.py` launch style.
    from fixed_quest_protocol import JointStateTracker, StrictCommandGate, load_json_message, FIXED_QUEST_VR_GATE_OPTIONS


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

ZMQ_SUB_IP = os.environ.get("SG2_ZMQ_SUB_IP", "192.168.6.100")
ZMQ_SUB_PORT = 5561
CMD_VEL_TOPIC = "/cmd_vel"   # swerve_drive_controller (cmd_vel_timeout 1.0s)
BASE_MAX_LIN = 0.3           # [m/s]
BASE_MAX_ANG = 0.6           # [rad/s]
QOS_DEPTH = 10
PUBLISH_HZ = 100.0


class SG2ZMQRosBridge(Node):
    def __init__(self):
        super().__init__("sg2_zmq_ros2_bridge_node")
        self.strict = os.environ.get("SG2_FIXED_QUEST", "0") == "1"
        self._last_warning_at = -float("inf")

        self.zmq_ctx = zmq.Context()
        self.socket = self.zmq_ctx.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.set_hwm(1)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        if self.strict:
            self.socket.setsockopt(zmq.MAXMSGSIZE, 65536)
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

        self.cmd_vel_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, QOS_DEPTH)
        self.get_logger().info(
            "  Publisher: base -> %s (|v|<=%.2fm/s, |w|<=%.2frad/s)" % (CMD_VEL_TOPIC, BASE_MAX_LIN, BASE_MAX_ANG)
        )

        self.timer = self.create_timer(1.0 / PUBLISH_HZ, self.on_timer)
        if self.strict:
            self.tracker = JointStateTracker()
            self.gate = StrictCommandGate(self.tracker, **FIXED_QUEST_VR_GATE_OPTIONS)
            self.state_sub = self.create_subscription(JointState, "/joint_states", self.on_joint_state, qos_profile_sensor_data)
            self.status_pub = self.create_publisher(String, "/fixed_quest/bridge_status", 10)
            # Acknowledge at the PC's command cadence so it can bound commands
            # still in flight without advancing beyond the last accepted pose.
            self.status_timer = self.create_timer(.02, self.publish_status)
            self.get_logger().info(
                "Strict limits: arm teleop %.1f deg/s, gripper %.1f deg/s, home %.1f deg/s, "
                "lift %.3f m/s; acknowledgment 50Hz" % (
                    math.degrees(self.gate.speed[3]),
                    math.degrees(self.gate.speed[-1]),
                    math.degrees(self.gate.home_speed[1]), self.gate.speed[0]))
        self.get_logger().info("SG2 ZMQ ROS2 Bridge ready (%.0fHz)" % PUBLISH_HZ)
        self.get_logger().info("Protocol: %s" % ("strict JSON; explicit arm required" if self.strict else "legacy pickle"))

    def report_warning(self, message):
        now = time.monotonic()
        if now - self._last_warning_at >= 2.:
            self.get_logger().warn(message)
            self._last_warning_at = now

    def on_joint_state(self, msg):
        try:
            self.tracker.update(list(msg.name), list(msg.position),
                                int(msg.header.stamp.sec) * 1000000000 + int(msg.header.stamp.nanosec), time.monotonic())
        except (ValueError, TypeError, OverflowError) as exc:
            self.report_warning("Strict feedback rejected: %s" % exc)

    def publish_status(self):
        msg = String()
        msg.data = json.dumps(self.gate.status(time.monotonic()), allow_nan=False)
        self.status_pub.publish(msg)

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
        if self.strict:
            now = time.monotonic()
            expired = self.gate.expire(now)
            if expired["publish_joints"] is not None:
                self.publish_joint_groups(expired["publish_joints"])
            try:
                message = load_json_message(self.socket.recv(flags=zmq.NOBLOCK))
                result = self.gate.receive(message, time.monotonic())
                if result["publish_joints"] is not None:
                    self.publish_joint_groups(result["publish_joints"])
                if not result["accepted"]:
                    self.report_warning("Strict command rejected: %s" % result["reason"])
            except zmq.Again:
                pass
            except (ValueError, TypeError, UnicodeError, zmq.ZMQError) as exc:
                self.report_warning("Strict command decode rejected: %s" % exc)
            return
        try:
            data = self.socket.recv_pyobj(flags=zmq.NOBLOCK)
        except zmq.Again:
            return
        except Exception as exc:
            self.get_logger().error("ZMQ recv error: %s" % exc)
            return

        self.publish_joint_groups(data)
        if isinstance(data, dict) and data.get("base") is not None:
            self.publish_base(data["base"])

    def publish_base(self, base):
        """base = [vx, vy, wz] in base_link frame [m/s, m/s, rad/s]."""
        try:
            vx, vy, wz = (float(v) for v in base)
        except (TypeError, ValueError):
            self.report_warning("Skip base: expected [vx, vy, wz], got %r" % (base,))
            return
        if not all(math.isfinite(v) for v in (vx, vy, wz)):
            self.report_warning("Skip base: non-finite command %r" % (base,))
            return
        msg = Twist()
        msg.linear.x = max(-BASE_MAX_LIN, min(BASE_MAX_LIN, vx))
        msg.linear.y = max(-BASE_MAX_LIN, min(BASE_MAX_LIN, vy))
        msg.angular.z = max(-BASE_MAX_ANG, min(BASE_MAX_ANG, wz))
        self.cmd_vel_pub.publish(msg)

    def publish_joint_groups(self, data):
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
