"""Bounded, observational timing probes; never change GC or motion policy."""
from collections import deque
import gc
import threading
import time


class GCPauseMonitor:
    """Report completed collections since the preceding control tick.

    GC may start in any Python thread. Keep its thread and monotonic interval so
    a long IK wall time can be compared with a collection instead of guessed.
    The callback neither collects nor disables GC and retains at most 32 events.
    """

    def __init__(self):
        self._started = {}
        self._events = deque(maxlen=32)
        self._registered = False

    def start(self):
        if not self._registered:
            gc.callbacks.append(self._callback)
            self._registered = True
        return self

    def _callback(self, phase, info):
        key = (threading.get_ident(), info["generation"])
        now = time.monotonic()
        if phase == "start":
            self._started[key] = now
        elif phase == "stop":
            started = self._started.pop(key, None)
            if started is not None:
                self._events.append({"generation": key[1], "thread_id": key[0],
                                     "started_at": started, "ended_at": now,
                                     "elapsed_ms": (now - started) * 1000})

    def take_events(self):
        # popleft is atomic; do not copy then clear (a receiver-thread event
        # arriving between those operations would disappear).
        events = []
        for _ in range(len(self._events)):
            events.append(self._events.popleft())
        return events

    def close(self):
        if self._registered:
            gc.callbacks.remove(self._callback)
            self._registered = False
        self._started.clear()
        self._events.clear()
