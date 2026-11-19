"""Receive optional AI Worker color streams for UR5e demonstrations."""

import io
import pickle
import pickletools
import re
import threading
import time
from contextlib import ExitStack
from urllib.parse import urlsplit

import numpy as np


def parse_camera_endpoints(specs):
    """Parse repeated NAME=tcp://HOST:PORT options; camera names become dataset feature names."""
    endpoints = {}
    for spec in specs:
        name, separator, endpoint = spec.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
            raise ValueError(f"Expected CAMERA_NAME=tcp://HOST:PORT, got {spec!r}")
        address = urlsplit(endpoint)
        if (address.scheme != "tcp" or not address.hostname or not address.port or
                address.path or address.query or address.fragment or address.username or address.password):
            raise ValueError(f"Expected tcp://HOST:PORT for camera {name!r}")
        if name in endpoints:
            raise ValueError(f"Duplicate recording camera: {name}")
        endpoints[name] = endpoint
    return endpoints


# camera_outbound.py sends a plain dict with bytes/string/number/bool fields via
# ZMQ send_pyobj(). Accept its data-only representation, without allowing object
# construction, imports, function calls or extension lookup from network data.
_CAMERA_OPCODES = frozenset("""
    PROTO FRAME EMPTY_DICT MEMOIZE MARK SHORT_BINUNICODE BINUNICODE SHORT_BINBYTES BINBYTES
    BINGET NEWTRUE NEWFALSE BINFLOAT BININT BININT1 BININT2 SETITEM SETITEMS STOP
""".split())
_MAX_PACKET_BYTES = 16 * 1024 * 1024


class _DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("camera packets cannot contain Python objects")


def _decode_rgb(packet):
    from PIL import Image

    if len(packet) > _MAX_PACKET_BYTES:
        raise ValueError("oversize camera packet")
    for index, (opcode, _, _) in enumerate(pickletools.genops(packet)):
        if index >= 128 or opcode.name not in _CAMERA_OPCODES:
            raise ValueError("unsupported camera packet encoding")
    message = _DataOnlyUnpickler(io.BytesIO(packet)).load()
    if not isinstance(message, dict) or not isinstance(message.get("data"), bytes):
        raise ValueError("camera packet has no compressed color image")
    stereo = message.get("stereo", True)  # Legacy SG2 cameras publish a side-by-side pair.
    if type(stereo) is not bool:
        raise ValueError("invalid stereo flag")
    with Image.open(io.BytesIO(message["data"]), formats=("JPEG", "PNG")) as image:
        if image.width * image.height > 32 * 1024 * 1024:
            raise ValueError("oversize camera image")
        if stereo:
            if image.width < 2 or image.width % 2:
                raise ValueError("invalid stereo image width")
            image = image.crop((0, 0, image.width // 2, image.height))
        rgb = np.ascontiguousarray(image.convert("RGB"))
    rgb.setflags(write=False)
    return rgb


class RecordingCameras:
    """The control loop only copies references to immutable, recently received RGB frames."""

    MAX_AGE_S = 0.25

    def __init__(self, endpoints):
        self.endpoints = dict(endpoints)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frames = {}
        self._errors = {}
        self._thread = None
        self._failure = None

    def __enter__(self):
        if self.endpoints:
            self._thread = threading.Thread(target=self._run, name="recording-cameras", daemon=True)
            self._thread.start()
        return self

    def snapshot(self):
        """Return all configured cameras or raise RuntimeError if the set is incomplete."""
        with self._lock:
            if self._failure:
                raise RuntimeError(self._failure)
            now = time.monotonic()
            images = {}
            for name in self.endpoints:
                if name in self._errors:
                    raise RuntimeError(f"Camera {name}: {self._errors[name]}")
                if name not in self._frames:
                    raise RuntimeError(f"Waiting for camera {name}")
                received, image = self._frames[name]
                if now - received > self.MAX_AGE_S:
                    raise RuntimeError(f"Camera {name}: stale frame")
                images[name] = image
            return images

    def _run(self):
        try:
            import zmq

            with ExitStack() as stack:
                context = stack.enter_context(zmq.Context())
                poller = zmq.Poller()
                names = {}
                for name, endpoint in self.endpoints.items():
                    sock = stack.enter_context(context.socket(zmq.SUB))
                    sock.setsockopt(zmq.LINGER, 0)
                    sock.setsockopt(zmq.CONFLATE, 1)
                    sock.setsockopt(zmq.MAXMSGSIZE, _MAX_PACKET_BYTES)
                    sock.setsockopt(zmq.SUBSCRIBE, b"")
                    sock.connect(endpoint)
                    poller.register(sock, zmq.POLLIN)
                    names[sock] = name
                while not self._stop.is_set():
                    for sock, _ in poller.poll(50):
                        try:
                            packet = sock.recv(zmq.NOBLOCK)
                        except zmq.Again:
                            continue
                        self._receive(names[sock], packet)
        except Exception as exc:
            with self._lock:
                self._failure = f"Recording cameras unavailable: {exc}"

    def _receive(self, name, packet):
        received = time.monotonic()
        try:
            image = _decode_rgb(packet)
        except Exception as exc:
            with self._lock:
                self._errors[name] = str(exc) or "invalid image"
            return
        with self._lock:
            self._frames[name] = received, image
            self._errors.pop(name, None)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        with self._lock:
            self._frames.clear()

    def __exit__(self, *exc):
        self.close()
