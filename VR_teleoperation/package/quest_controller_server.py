"""Serve the controller-only Quest page and receive its latest frame in one thread.

The notebook remains the only control process. This receiver doesn't solve IK
or publish robot commands. Install ``websockets>=14`` in the notebook kernel.
Non-loopback listeners require TLS; the certificate must be trusted by Quest
Browser before WebXR is available.
"""

import asyncio
import copy
from http import HTTPStatus
import ipaddress
import json
import math
from pathlib import Path
import re
import ssl
import subprocess
import threading
import time
from urllib.parse import urlsplit


def create_self_signed_certificate(output_dir, host):
    """Create a local development certificate for a PC's LAN IP or DNS name.

    Returns ``(certfile, keyfile)`` as strings. Existing files are reused. This
    does not install trust certificates or change browser/headset settings.
    """
    host = str(host).strip("[]")
    try:
        address = ipaddress.ip_address(host)
        san = f"IP:{address}"
    except ValueError:
        if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?", host):
            raise ValueError("host must be a LAN IP address or DNS name") from None
        san = f"DNS:{host}"
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "quest-" + re.sub(r"[^A-Za-z0-9.-]", "_", host)
    certfile, keyfile = output_dir / f"{stem}.crt", output_dir / f"{stem}.key"
    if certfile.is_file() and keyfile.is_file():
        return str(certfile), str(keyfile)
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-sha256",
         "-days", "30", "-keyout", str(keyfile), "-out", str(certfile),
         "-subj", f"/CN={host}", "-addext",
         f"subjectAltName={san},DNS:localhost,IP:127.0.0.1"],
        check=True, capture_output=True, text=True,
    )
    keyfile.chmod(0o600)
    return str(certfile), str(keyfile)


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _pose_is_valid(value):
    if not isinstance(value, dict):
        return False
    for key, length in (("position_m", 3), ("quaternion_xyzw", 4)):
        vector = value.get(key)
        if not isinstance(vector, list) or len(vector) != length:
            return False
        if not all(_finite_number(item) for item in vector):
            return False
    return True


def _frame_is_valid(frame):
    """Check the transport envelope; controller mapping owns motion validation."""
    if not isinstance(frame, dict) or type(frame.get("version")) is not int or frame["version"] != 1:
        return False
    if not isinstance(frame.get("session_id"), str) or not 1 <= len(frame["session_id"]) <= 128:
        return False
    for key in ("origin_id", "seq"):
        if type(frame.get(key)) is not int or frame[key] < 0:
            return False
    if not _finite_number(frame.get("sample_time_ms")) or frame["sample_time_ms"] < 0:
        return False
    if frame.get("visibility") not in ("visible", "visible-blurred", "hidden"):
        return False
    if not _pose_is_valid(frame.get("head")):
        return False
    if not isinstance(frame.get("hands"), dict):
        return False
    for hand in ("left", "right"):
        value = frame["hands"].get(hand)
        if not _pose_is_valid(value):
            return False
        if type(value.get("valid")) is not bool or type(value.get("emulated")) is not bool:
            return False
        for button in ("grip", "trigger"):
            if not _finite_number(value.get(button)) or not 0 <= value[button] <= 1:
                return False
    return True


class QuestControllerServer:
    """A latest-only HTTPS/WebSocket receiver with deterministic cleanup.

    ``latest()`` returns a copied frame with trusted ``received_at`` in Python
    monotonic seconds, or ``None`` before input / after disconnect or malformed
    data. Browser ``sample_time_ms`` has a different clock origin. Only one
    pose sender is accepted. Opening a page/socket doesn't reserve that role.
    A reconnect of the same XR session must advance its sequence and origin;
    a different session cannot replace the active sender.
    """

    def __init__(self, host="0.0.0.0", port=8443, certfile=None, keyfile=None):
        if bool(certfile) != bool(keyfile):
            raise ValueError("Provide both certfile and keyfile")
        if certfile is None and host not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("A LAN listener requires certfile and keyfile for HTTPS/WebXR")
        self.host, self.port = host, int(port)
        self.certfile, self.keyfile = certfile, keyfile
        self._lock = threading.Lock()
        self._latest = None
        self._thread = None
        self._loop = None
        self._stop = None
        self._sender = None
        self._ready = threading.Event()
        self._error = None
        self._assets = {}

    @property
    def url(self):
        """Page URL; replace a wildcard listener host with the PC's LAN address."""
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{'https' if self.certfile else 'http'}://{host}:{self.port}/"

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return self
        from websockets.asyncio.server import serve  # Fail in the caller if missing.

        client_dir = Path(__file__).resolve().parents[1] / "quest_client"
        self._assets = {
            "/": ((client_dir / "index.html").read_bytes(), "text/html; charset=utf-8"),
            "/client.js": ((client_dir / "client.js").read_bytes(), "text/javascript; charset=utf-8"),
        }
        self._ready.clear()
        self._error = None
        self._thread = threading.Thread(target=self._run, args=(serve,), name="Quest input", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=5):
            self.close()
            raise TimeoutError("Quest receiver did not start within 5 seconds")
        if self._error is not None:
            self.close()
            raise RuntimeError("Quest receiver failed to start") from self._error
        return self

    def latest(self):
        with self._lock:
            return copy.deepcopy(self._latest)

    def close(self):
        if self._loop is not None and self._stop is not None and not self._loop.is_closed():
            try:
                self._loop.call_soon_threadsafe(self._stop.set)
            except RuntimeError:  # The event loop may already have finished.
                pass
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                raise RuntimeError("Quest receiver thread did not stop")
        with self._lock:
            self._latest = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    def _run(self, serve):
        try:
            asyncio.run(self._serve(serve))
        except Exception as exc:
            self._error = exc
        finally:
            with self._lock:
                self._latest = None
            self._ready.set()

    async def _serve(self, serve):
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        context = None
        if self.certfile:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(self.certfile, self.keyfile)
        async with serve(
            self._receive, self.host, self.port, ssl=context,
            process_request=self._http_request, max_size=16_384, max_queue=1,
            compression=None, close_timeout=1, ping_interval=5, ping_timeout=5,
        ) as server:
            self.port = server.sockets[0].getsockname()[1]
            self._ready.set()
            await self._stop.wait()

    def _http_request(self, connection, request):
        path = urlsplit(request.path).path
        if request.headers.get("Upgrade", "").lower() == "websocket":
            if path != "/ws":
                return connection.respond(HTTPStatus.NOT_FOUND, "WebSocket endpoint: /ws\n")
            # Same-origin pages only. Native test/recording clients may omit Origin.
            origin = request.headers.get("Origin")
            expected = f"{'https' if self.certfile else 'http'}://{request.headers.get('Host', '')}"
            if origin is not None and origin != expected:
                return connection.respond(HTTPStatus.FORBIDDEN, "Use the receiver's own page\n")
            return None
        asset = self._assets.get(path)
        if asset is None:
            return connection.respond(HTTPStatus.NOT_FOUND, "Not found\n")
        response = connection.respond(HTTPStatus.OK, asset[0].decode("utf-8"))
        del response.headers["Content-Type"]
        response.headers["Content-Type"] = asset[1]
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Permissions-Policy"] = "xr-spatial-tracking=(self)"
        return response

    async def _receive(self, websocket):
        from websockets.exceptions import ConnectionClosed

        try:
            async for message in websocket:
                try:
                    frame = json.loads(message) if isinstance(message, str) else None
                except (ValueError, TypeError):
                    frame = None
                valid = _frame_is_valid(frame)
                if not valid:
                    if self._sender is websocket:
                        with self._lock:
                            self._latest = None
                    await websocket.close(code=1003, reason="Invalid controller frame")
                    break
                previous_sender = None
                if self._sender is not None and self._sender is not websocket:
                    with self._lock:
                        previous_frame = self._latest
                    # The existing browser reconnects with the same XR UUID,
                    # an advanced sequence and a changed origin. This is a
                    # lifecycle identity, not device authentication. Advancing
                    # origin also forces the mapper to release/re-anchor.
                    resumed = (previous_frame is not None
                               and frame["session_id"] == previous_frame["session_id"]
                               and frame["seq"] > previous_frame["seq"]
                               and frame["origin_id"] > previous_frame["origin_id"])
                    if not resumed:
                        await websocket.close(code=1008, reason="One Quest sender is already connected")
                        return
                    previous_sender = self._sender
                # No await between choosing the owner and publishing the frame.
                # Idle browser tabs cannot reserve input or clear another pose.
                self._sender = websocket
                with self._lock:
                    frame["received_at"] = time.monotonic()
                    self._latest = frame
                if previous_sender is not None:
                    await previous_sender.close(code=1008, reason="Quest connection resumed in another socket")
        except ConnectionClosed:
            pass
        finally:
            if self._sender is websocket:
                self._sender = None
                with self._lock:
                    self._latest = None
