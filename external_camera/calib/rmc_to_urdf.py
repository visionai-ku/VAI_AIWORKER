#!/usr/bin/env python3
"""Convert ViSP's eye-to-hand result rMc (head_link2 -> camera optical frame, vpPoseVector YAML)
into URDF / Isaac Sim numbers, and sanity-check it against the old ZED-M left optical frame.

Usage: python3 rmc_to_urdf.py sg2_rPc.yaml
"""
import sys

import numpy as np
from scipy.spatial.transform import Rotation


def load_pose_yaml(path):
    vals = []
    in_data = False
    for line in open(path):
        s = line.strip()
        if s.startswith("data:"):
            in_data = True
            continue
        if in_data and s.startswith("- ["):
            vals.append(float(s[3:].rstrip("]")))
    if len(vals) != 6:
        sys.exit("expected 6 values in %s, got %d" % (path, len(vals)))
    return np.array(vals[:3]), Rotation.from_rotvec(vals[3:])


t, R = load_pose_yaml(sys.argv[1])
rpy = R.as_euler("xyz")              # URDF rpy = fixed-axis roll, pitch, yaw
qx, qy, qz, qw = R.as_quat()
R_usd = R * Rotation.from_euler("x", np.pi)   # ROS optical (z fwd, y down) -> USD camera (-z fwd, y up)
ux, uy, uz, uw = R_usd.as_quat()

print("head_link2 -> camera_head_color_optical_frame")
print("  xyz [m]        : %.5f %.5f %.5f" % tuple(t))
print("  rpy [rad]      : %.5f %.5f %.5f   (deg: %.2f %.2f %.2f)" % (*rpy, *np.degrees(rpy)))
print("  quat xyzw (ROS optical axes)  : %.6f %.6f %.6f %.6f" % (qx, qy, qz, qw))
print("  quat wxyz (ROS optical axes)  : %.6f %.6f %.6f %.6f   -> Isaac Camera set_local_pose(..., camera_axes='ros')"
      % (qw, qx, qy, qz))
print("  quat wxyz (USD camera axes)   : %.6f %.6f %.6f %.6f   -> USD xformOp:orient of the Camera prim"
      % (uw, ux, uy, uz))
print()
print("URDF snippet:")
print('  <link name="camera_head_color_optical_frame"/>')
print('  <joint name="camera_head_color_optical_joint" type="fixed">')
print('    <parent link="head_link2"/>')
print('    <child link="camera_head_color_optical_frame"/>')
print('    <origin xyz="%.5f %.5f %.5f" rpy="%.5f %.5f %.5f"/>' % (*t, *rpy))
print("  </joint>")

# Old ZED-M left optical frame in head_link2, from ffw_sg2_follower.urdf:
# zed_joint xyz(0.0238122 -0.00651797 -0.0242094) + center z 0.01325 + left y 0.0315,
# optical rpy(-pi/2, 0, -pi/2)
t_zed = np.array([0.0238122, -0.00651797 + 0.0315, -0.0242094 + 0.01325])
R_zed = Rotation.from_euler("xyz", [-np.pi / 2, 0.0, -np.pi / 2])
d = np.linalg.norm(t - t_zed)
ang = np.degrees(np.linalg.norm((R_zed.inv() * R).as_rotvec()))  # scipy<1.4 has no magnitude()
print()
print("sanity vs old ZED-M left optical frame: offset %.1f mm, rotation diff %.1f deg" % (d * 1e3, ang))
print("  (the D405 sits where the ZED was: expect a few cm / a few-to-tens deg; far more = suspect)")
