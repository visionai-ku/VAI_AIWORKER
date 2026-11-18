"""Pose the OnRobot RG2 visual linkage from the fingertip aperture in millimetres."""

import math


OPEN_WIDTH_MM = 110.0


def set_rg2_width(model, width_mm):
    """Set the six visual link orientations before the caller's next MuJoCo forward."""
    if not math.isfinite(width_mm):
        raise ValueError("RG2 aperture must be finite")
    # In the RG2 URDF each outer link connects its root to the finger pivot
    # at (y=-39.592, z=38.177) mm. The counter-rotating pad remains parallel.
    # Its inside face is 18.950 mm inboard of the finger pivot; the outer
    # link root is 17.178 mm outboard of the gripper centre.
    width_mm = max(0.0, min(OPEN_WIDTH_MM, width_mm))
    projection = (width_mm / 2 + 18.950 - 17.178) / math.hypot(39.592, 38.177)
    angle = math.acos(min(1.0, projection)) - math.atan2(38.177, 39.592)
    angle = max(-0.558505, angle)  # The source model's open stop.
    c, s = math.cos(angle / 2), math.sin(angle / 2)
    for part in ("outer", "inner"):
        model.body(f"rg2_left_{part}").quat[:] = (c, -s, 0, 0)
        model.body(f"rg2_right_{part}").quat[:] = (0, 0, -s, c)
    for side in ("left", "right"):
        model.body(f"rg2_{side}_finger").quat[:] = (c, s, 0, 0)
