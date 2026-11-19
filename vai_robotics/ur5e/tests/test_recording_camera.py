import io
import pickle
import time
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
from PIL import Image

from vai_robotics.ur5e.recording_camera import RecordingCameras, parse_camera_endpoints


def color_packet(*, stereo):
    pixels = np.zeros((4, 8, 3), dtype=np.uint8)
    pixels[:, :4, 0] = 255
    pixels[:, 4:, 2] = 255
    encoded = io.BytesIO()
    Image.fromarray(pixels).save(encoded, "PNG")
    return pickle.dumps({"cam": "wrist", "data": encoded.getvalue(), "format": "png",
                         "stereo": stereo, "stamp": 5.0})


class RecordingCameraTests(unittest.TestCase):
    def test_optional_sources_and_validation(self):
        with RecordingCameras(parse_camera_endpoints([])) as cameras:
            self.assertEqual(cameras.snapshot(), {})
            self.assertIsNone(cameras._thread)
        self.assertEqual(parse_camera_endpoints(["wrist=tcp://cam:5571"]), {"wrist": "tcp://cam:5571"})
        for specs in (["/tmp=tcp://cam:1"], ["wrist="], ["wrist=tcp://cam"], ["wrist=rtsp://cam:5"],
                      ["wrist=tcp://a:1", "wrist=tcp://b:2"]):
            with self.subTest(specs=specs), self.assertRaises(ValueError):
                parse_camera_endpoints(specs)

    def test_full_set_rgb_stereo_staleness_and_recovery(self):
        cameras = RecordingCameras({"left": "unused", "right": "unused"})
        with patch("vai_robotics.ur5e.recording_camera.time.monotonic", return_value=10) as clock:
            cameras._receive("left", color_packet(stereo=False))
            with self.assertRaisesRegex(RuntimeError, "Waiting for camera right"):
                cameras.snapshot()
            cameras._receive("right", color_packet(stereo=True))
            frames = cameras.snapshot()
            self.assertEqual(frames["left"].shape, (4, 8, 3))
            self.assertEqual(frames["right"].shape, (4, 4, 3))
            np.testing.assert_array_equal(frames["left"][0, 7], [0, 0, 255])
            np.testing.assert_array_equal(frames["right"][0, 0], [255, 0, 0])
            self.assertFalse(frames["right"].flags.writeable)
            clock.return_value += 1
            with self.assertRaisesRegex(RuntimeError, "stale"):
                cameras.snapshot()
            cameras._receive("left", pickle.dumps({"data": bytes}))
            with self.assertRaisesRegex(RuntimeError, "unsupported camera packet encoding"):
                cameras.snapshot()
            for name in cameras.endpoints:
                cameras._receive(name, color_packet(stereo=False))
            self.assertEqual(set(cameras.snapshot()), {"left", "right"})

    def test_receiver_with_in_process_ai_worker_publisher(self):
        import zmq

        endpoint = "inproc://demo-camera"
        with zmq.Context() as context, context.socket(zmq.PUB) as publisher:
            publisher.bind(endpoint)
            with patch.object(zmq, "Context", return_value=nullcontext(context)):
                with RecordingCameras({"wrist": endpoint}) as cameras:
                    packet = color_packet(stereo=False)
                    deadline = time.monotonic() + 2
                    while True:
                        publisher.send(packet)
                        time.sleep(.01)
                        try:
                            images = cameras.snapshot()
                        except RuntimeError:
                            if time.monotonic() >= deadline:
                                raise
                        else:
                            break
                    np.testing.assert_array_equal(images["wrist"][0, -1], [0, 0, 255])
                self.assertFalse(cameras._thread.is_alive())
