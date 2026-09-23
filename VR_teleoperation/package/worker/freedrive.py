"""Manually toggle Dynamixel torque for freedrive (hand-guiding) via the
existing `dynamixel_hardware_interface/set_dxl_torque` service.

NOTE ON SCOPE: this service is a single on/off switch for the whole
"follower" ros2_control hardware interface, which covers arm_l, arm_r,
gripper_l, gripper_r, lift, head AND the mobile base steering joints
together (see ffw_sg2_follower.ros2_control.xacro). There is no
per-joint-group torque service in the current stack, so turning torque
off releases all of those joints at once, not just the arms/grippers.
Support the arm and lift physically before running `off`.

Usage:
    python3 freedrive.py off   # release torque -> free to hand-guide
    python3 freedrive.py on    # re-engage torque
"""

import argparse
import sys

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool


SERVICE_NAME = "/dynamixel_hardware_interface/set_dxl_torque"
SERVICE_WAIT_TIMEOUT_S = 5.0
CALL_TIMEOUT_S = 5.0


class TorqueToggleClient(Node):
    def __init__(self):
        super().__init__("sg2_freedrive_torque_toggle_node")
        self.client = self.create_client(SetBool, SERVICE_NAME)

    def set_torque(self, enable: bool):
        if not self.client.wait_for_service(timeout_sec=SERVICE_WAIT_TIMEOUT_S):
            self.get_logger().error(
                "Service %s not available after %.1fs "
                "(is the robot controller stack running?)"
                % (SERVICE_NAME, SERVICE_WAIT_TIMEOUT_S)
            )
            return False

        request = SetBool.Request()
        request.data = enable

        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=CALL_TIMEOUT_S)

        if not future.done():
            self.get_logger().error("Service call timed out")
            return False

        response = future.result()
        if response is None:
            self.get_logger().error("Service call failed: %s" % future.exception())
            return False

        self.get_logger().info(
            "set_dxl_torque(%s) -> success=%s message=%r"
            % (enable, response.success, response.message)
        )
        return response.success


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", choices=["on", "off"], help="torque on (locked) or off (freedrive)")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = TorqueToggleClient()

    if args.state == "off":
        node.get_logger().warn(
            "Disabling torque for ALL follower joints (arms, grippers, lift, "
            "head, base steering) -- support the robot by hand before this "
            "call returns."
        )

    ok = node.set_torque(args.state == "on")

    node.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
