#!/usr/bin/env python3
"""
Turn a sg2_head_cam_snap.py capture folder (board fixed on the table, HEAD moved) into a ViSP
eye-in-hand dataset: rPe = base_link -> head_link2, computed by FK from the joint_states saved
with each capture and the robot's URDF.

  # URDF from the running robot_state_publisher (inside ai_worker, ROS env sourced)
  ros2 param get /robot_state_publisher robot_description --hide-type > sg2_urdf.xml
  # optional: FK vs live TF (needs ROS + bringup)
  python3 sg2_eih_prepare.py --urdf sg2_urdf.xml --validate
  # build the dataset (renumbered 1..N, only --keep captures)
  python3 sg2_eih_prepare.py --urdf sg2_urdf.xml --src data-snap-1110 --out data-snap-1110-eih \
      --keep 3,4,6,7,8,9,10,15,17,18,20,21

Writes into --out: sg2_image-<n>.png, sg2_pose_rPe_<n>.yaml (vpPoseVector), sg2_camera.xml,
README_eih.json (new index -> original index).
"""
import argparse
import json
import os
import shutil
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R


def load_chain(urdf, base, tip):
    root = ET.fromstring(open(urdf).read().strip())
    by_child = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        ax = j.find("axis")
        axis = [float(v) for v in ax.get("xyz").split()] if ax is not None else [1.0, 0.0, 0.0]
        by_child[j.find("child").get("link")] = {
            "name": j.get("name"), "type": j.get("type"), "parent": j.find("parent").get("link"),
            "xyz": np.array(xyz), "R": R.from_euler("xyz", rpy).as_matrix(), "axis": np.array(axis)}
    chain, link = [], tip
    while link != base:
        if link not in by_child:
            sys.exit("no joint chain %s -> %s in the URDF" % (base, tip))
        chain.append(by_child[link])
        link = by_child[link]["parent"]
    return chain[::-1]


def fk(chain, q):
    M = np.eye(4)
    for c in chain:
        A = np.eye(4)
        A[:3, :3], A[:3, 3] = c["R"], c["xyz"]
        M = M @ A
        B = np.eye(4)
        if c["type"] in ("revolute", "continuous"):
            B[:3, :3] = R.from_rotvec(c["axis"] * q[c["name"]]).as_matrix()
        elif c["type"] == "prismatic":
            B[:3, 3] = c["axis"] * q[c["name"]]
        M = M @ B
    return M


def validate(chain, base, tip):
    import rclpy
    import tf2_ros
    from sensor_msgs.msg import JointState
    rclpy.init()
    n = rclpy.create_node("sg2_eih_fk_check")
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, n)
    js = []
    n.create_subscription(JointState, "/joint_states", js.append, 10)
    t0 = time.time()
    while time.time() - t0 < 3.0:
        rclpy.spin_once(n, timeout_sec=0.1)
    if not js:
        sys.exit("no /joint_states - ROS env (ROS_DOMAIN_ID / RMW) sourced?")
    M = fk(chain, dict(zip(js[-1].name, js[-1].position)))
    tf = buf.lookup_transform(base, tip, rclpy.time.Time())
    t, q = tf.transform.translation, tf.transform.rotation
    dt = np.linalg.norm(M[:3, 3] - [t.x, t.y, t.z]) * 1e3
    dr = np.degrees(np.linalg.norm((R.from_matrix(M[:3, :3]).inv() * R.from_quat([q.x, q.y, q.z, q.w])).as_rotvec()))
    print("FK vs TF %s -> %s: %.3f mm, %.4f deg  (%s)" % (base, tip, dt, dr, "OK" if dt < 0.5 and dr < 0.05 else "MISMATCH"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--src", help="sg2_head_cam_snap.py output folder")
    ap.add_argument("--out", help="new eye-in-hand dataset folder")
    ap.add_argument("--keep", default="", help="comma list of capture indices to use (default: all)")
    ap.add_argument("--base", default="base_link")
    ap.add_argument("--tip", default="head_link2")
    ap.add_argument("--validate", action="store_true", help="compare FK with live TF and exit")
    a = ap.parse_args()

    chain = load_chain(a.urdf, a.base, a.tip)
    print("chain:", " -> ".join("%s(%s)" % (c["name"], c["type"]) for c in chain))
    if a.validate:
        return validate(chain, a.base, a.tip)
    if not (a.src and a.out):
        sys.exit("--src and --out are required")
    if os.path.exists(a.out):
        sys.exit("%s already exists - pick a new --out" % a.out)
    caps = json.load(open(os.path.join(a.src, "capture_meta.json")))["captures"]
    keep = [int(v) for v in a.keep.split(",") if v.strip()] or sorted(int(k) for k in caps)
    os.makedirs(a.out)
    shutil.copy(os.path.join(a.src, "sg2_camera.xml"), a.out)
    mapping = {}
    for n, i in enumerate(keep, 1):
        c = caps[str(i)]
        if "joint_states" not in c:
            sys.exit("capture %d has no joint_states" % i)
        M = fk(chain, c["joint_states"])
        shutil.copy(os.path.join(a.src, "sg2_image-%d.png" % i), os.path.join(a.out, "sg2_image-%d.png" % n))
        with open(os.path.join(a.out, "sg2_pose_rPe_%d.yaml" % n), "w") as f:
            f.write("rows: 6\ncols: 1\ndata:\n")
            for v in list(M[:3, 3]) + list(R.from_matrix(M[:3, :3]).as_rotvec()):
                f.write("  - [%.12g]\n" % v)
        mapping[n] = i
        q = c["joint_states"]
        print("%2d <- #%-2d  %s" % (n, i, "  ".join("%s=%.3f" % (ch["name"], q[ch["name"]]) for ch in chain)))
    json.dump({"new_to_orig": mapping, "rPe": "%s -> %s (URDF FK from joint_states)" % (a.base, a.tip),
               "src": os.path.abspath(a.src)}, open(os.path.join(a.out, "README_eih.json"), "w"), indent=1)
    print("wrote %d poses to %s" % (len(keep), a.out))


if __name__ == "__main__":
    main()
