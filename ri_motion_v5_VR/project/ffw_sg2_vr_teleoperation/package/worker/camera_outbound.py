import os
import time
import traceback
from functools import partial

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
import zmq

from outbound import get_ip_address


# camera name -> (compressed image topic, ZMQ PUB port)
CAMERAS = {
    # "head": ("/zed/zed_node/left/image_rect_color/compressed", 5570),  # ZED-M head (replaced)
    "head": ("/camera_head/camera_head/color/image_rect_raw/compressed", 5570),  # D405 head, 640x480@30
    "wrist_left": ("/camera_left/camera_left/color/image_rect_raw/compressed", 5571),
    "wrist_right": ("/camera_right/camera_right/color/image_rect_raw/compressed", 5572),
}
PUBLISH_HZ = 30.0


class SG2CameraOutboundZMQ(Node):
    def __init__(self):
        super().__init__("sg2_camera_outbound_zmq_node")

        self.min_dt = 1.0 / PUBLISH_HZ
        self.last_pub = {name: 0.0 for name in CAMERAS}
        self.n_pub = {name: 0 for name in CAMERAS}

        ip_addr = get_ip_address()
        pid = os.getpid()

        self.zmq_ctx = zmq.Context()
        self.sockets = {}
        for name, (topic, port) in CAMERAS.items():
            socket = self.zmq_ctx.socket(zmq.PUB)
            socket.set_hwm(1)
            socket.bind("tcp://*:%d" % port)
            self.sockets[name] = socket
            self.get_logger().info(
                "[PUB] PID %d, IP %s, PORT %d <- %s (%s)" % (pid, ip_addr, port, topic, name)
            )
        time.sleep(1.0)

        for name, (topic, _) in CAMERAS.items():
            self.create_subscription(
                CompressedImage,
                topic,
                partial(self.image_callback, name),
                qos_profile_sensor_data,
            )
        self.create_timer(5.0, self.log_rate)
        self.get_logger().info("SG2 Camera Outbound ZMQ ready (max %.0fHz)" % PUBLISH_HZ)

    def image_callback(self, name, msg):
        now = time.time()
        if now - self.last_pub[name] < self.min_dt:
            return
        self.last_pub[name] = now

        # msg.data is already JPEG (image_transport compressed); forward as-is.
        frame = {
            "cam": name,
            "data": bytes(msg.data),
            "format": msg.format,
            "stereo": False,
            "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
        }
        try:
            self.sockets[name].send_pyobj(frame, flags=zmq.NOBLOCK)
            self.n_pub[name] += 1
        except zmq.Again:
            pass

    def log_rate(self):
        self.get_logger().info(
            "published " + "  ".join("%s %.1fHz" % (n, c / 5.0) for n, c in self.n_pub.items())
        )
        self.n_pub = {name: 0 for name in CAMERAS}


def main():
    rclpy.init()
    node = SG2CameraOutboundZMQ()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print("Unexpected error in main: %s" % exc)
        print("Traceback: %s" % traceback.format_exc())
    finally:
        node.get_logger().info("Shutting down SG2 camera outbound ZMQ...")
        for socket in node.sockets.values():
            socket.close()
        node.destroy_node()
        rclpy.shutdown()
        node.zmq_ctx.term()


if __name__ == "__main__":
    main()
