#!/usr/bin/env python3
"""Headless helper: keep left-clicking inside an X display (Xvfb) so that ViSP apps that
wait on vpDisplay::getClick() (e.g. visp-calibrate-camera) run to completion without a user.

Usage (normally called from sg2_calib_compute.sh):
    DISPLAY=:99 python3 xauto_click.py [interval_s]
Moves the pointer to a few spots near the top-left (where windows open without a window
manager) and clicks there periodically until killed.
"""
import ctypes
import sys
import time

interval = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
x11 = ctypes.cdll.LoadLibrary("libX11.so.6")
xtst = ctypes.cdll.LoadLibrary("libXtst.so.6")
x11.XOpenDisplay.restype = ctypes.c_void_p
x11.XFlush.argtypes = [ctypes.c_void_p]
xtst.XTestFakeMotionEvent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_ulong]
xtst.XTestFakeButtonEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]

dpy = x11.XOpenDisplay(None)
if not dpy:
    sys.exit("xauto_click: cannot open DISPLAY")

spots = [(60, 60), (320, 240), (700, 300), (1000, 400)]
i = 0
while True:
    x, y = spots[i % len(spots)]
    xtst.XTestFakeMotionEvent(dpy, -1, x, y, 0)
    xtst.XTestFakeButtonEvent(dpy, 1, 1, 0)
    xtst.XTestFakeButtonEvent(dpy, 1, 0, 0)
    x11.XFlush(dpy)
    i += 1
    time.sleep(interval)
