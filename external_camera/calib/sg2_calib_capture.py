#!/usr/bin/env python3
"""
SG2 head D405 calibration data capture, written in ViSP's file format.

Replaces ViSP's visp-acquire-universal-robots-calib-data for the SG2: images come from the
bringup's head camera topic (the exact 640x480 stream used for recording), robot poses come
from TF (robot_state_publisher), so the ViSP compute apps can be used unchanged:

  visp-calibrate-camera                 (intrinsic, mode "intrinsic" or "handeye" images)
  visp-compute-chessboard-poses         (cPo per image)
  visp-compute-eye-to-hand-calibration  (rMc = head_link2 -> camera optical, eMo)

Eye-to-hand setup: the camera is fixed on the head, the chessboard is fixed on the LEFT
gripper body. The reference frame r is head_link2 (not base_link), so the result rMc is
directly the head_link2 -> camera_head_color_optical_frame mount transform, valid for any
head/lift pose.

Files written to --out (index starts at 1, existing captures are kept and continued):
  sg2_image-<i>.png          color image (BGR PNG), as received
  sg2_pose_rPe_<i>.yaml      vpPoseVector [tx ty tz tux tuy tuz] of ee in ref (handeye mode)
  sg2_camera.xml             factory intrinsics from camera_info (ViSP format, 2 models)
  preview/sg2_image-<i>.jpg  detected corners (first corner = red end of the colored lines)
  capture_meta.json          args, camera_info, per-capture stamp/joint_states/TF

Run inside the ai_worker container while worker_bringup is running:
  python3 sg2_calib_capture.py --mode intrinsic --out <dir>
  python3 sg2_calib_capture.py --mode handeye   --out <dir>
Keys: Enter = capture, d = delete last, l = list, q = quit.
"""
import argparse
import glob
import json
import os
import re
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
import tf2_ros
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, JointState

PREFIX = "sg2"
STILL_DT = 0.5          # s, look back this far in TF to check the arm is not moving
STILL_TRANS = 0.001     # m
STILL_ROT_DEG = 0.2     # deg
MIN_DIVERSITY_DEG = 5.0  # warn if a new pose is this close (rotation) to an existing one


def image_to_bgr(msg):
    h, w = msg.height, msg.width
    buf = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, msg.step)
    if msg.encoding in ("rgb8", "bgr8"):
        img = buf[:, :w * 3].reshape(h, w, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == "rgb8" else img.copy()
    if msg.encoding == "mono8":
        return cv2.cvtColor(buf[:, :w].copy(), cv2.COLOR_GRAY2BGR)
    raise ValueError("unsupported image encoding: %s" % msg.encoding)


def tf_to_pose(tf):
    t = tf.transform.translation
    q = tf.transform.rotation
    trans = np.array([t.x, t.y, t.z])
    rot = Rotation.from_quat([q.x, q.y, q.z, q.w])
    return trans, rot


def write_pose_yaml(path, trans, rotvec):
    with open(path, "w") as f:
        f.write("rows: 6\ncols: 1\ndata:\n")
        for v in list(trans) + list(rotvec):
            f.write("  - [%.12g]\n" % v)


def write_camera_xml(path, info):
    """Same conversion as ViSP's vpRealSense2::getCameraParameters (kdu = k1, kud = -k1)."""
    k = info.k
    px, py, u0, v0 = k[0], k[4], k[2], k[5]
    k1 = info.d[0] if len(info.d) else 0.0
    cam = """  <camera>
    <name>Camera</name>
    <image_width>%d</image_width>
    <image_height>%d</image_height>
    <model>
%s
    </model>
  </camera>
"""
    m0 = ("      <type>perspectiveProjWithoutDistortion</type>\n"
          "      <px>%.12g</px>\n      <py>%.12g</py>\n      <u0>%.12g</u0>\n      <v0>%.12g</v0>"
          % (px, py, u0, v0))
    m1 = ("      <type>perspectiveProjWithDistortion</type>\n"
          "      <px>%.12g</px>\n      <py>%.12g</py>\n      <u0>%.12g</u0>\n      <v0>%.12g</v0>\n"
          "      <kud>%.12g</kud>\n      <kdu>%.12g</kdu>" % (px, py, u0, v0, -k1, k1))
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n<root>\n  <!-- SG2 head D405 factory intrinsics from '
                'camera_info (%s, D=%s) -->\n' % (info.distortion_model, list(info.d)))
        f.write(cam % (info.width, info.height, m0))
        f.write(cam % (info.width, info.height, m1))
        f.write("</root>\n")


class Capture(Node):
    def __init__(self, args):
        super().__init__("sg2_calib_capture")
        self.args = args
        self.lock = threading.Lock()
        self.last_img = None
        self.last_info = None
        self.last_js = None
        self.create_subscription(Image, args.image_topic, self._on_img, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, args.info_topic, self._on_info, qos_profile_sensor_data)
        self.create_subscription(JointState, "/joint_states", self._on_js, 50)
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def _on_img(self, msg):
        with self.lock:
            self.last_img = msg

    def _on_info(self, msg):
        with self.lock:
            self.last_info = msg

    def _on_js(self, msg):
        with self.lock:
            self.last_js = msg

    def snapshot(self):
        with self.lock:
            return self.last_img, self.last_info, self.last_js

    def lookup(self, stamp):
        return self.tf_buffer.lookup_transform(self.args.ref_frame, self.args.ee_frame, stamp,
                                               timeout=Duration(seconds=0.5))


def existing_indices(out):
    idx = []
    for p in glob.glob(os.path.join(out, PREFIX + "_image-*.png")):
        m = re.search(r"_image-(\d+)\.png$", p)
        if m:
            idx.append(int(m.group(1)))
    return sorted(idx)


def chown_like_parent(path, ref):
    try:
        st = os.stat(ref)
        os.chown(path, st.st_uid, st.st_gid)
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["intrinsic", "handeye"], required=True)
    ap.add_argument("--out", required=True, help="output folder (created if missing)")
    ap.add_argument("--image-topic", default="/camera_head/camera_head/color/image_rect_raw")
    ap.add_argument("--info-topic", default="/camera_head/camera_head/color/camera_info")
    ap.add_argument("--ref-frame", default="head_link2", help="ViSP 'r' frame (camera is fixed in it)")
    ap.add_argument("--ee-frame", default="gripper_l_rh_p12_rn_base",
                    help="ViSP 'e' frame (chessboard is fixed in it)")
    ap.add_argument("--target", choices=["chessboard", "apriltag"], default="chessboard")
    ap.add_argument("--board", default="9x6", help="inner corners WxH (OpenCV_Chessboard.pdf = 9x6)")
    ap.add_argument("--tag-size", type=float, default=0.0,
                    help="apriltag: black-square edge [m] (NOT the paper size); used only for the "
                         "tilt check here, pass the same value to sg2_calib_compute.sh")
    ap.add_argument("--force", action="store_true", help="save even if the arm seems to move")
    args = ap.parse_args()
    bw, bh = (int(v) for v in args.board.lower().split("x"))
    if args.target == "apriltag":
        tag_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        tag_params = cv2.aruco.DetectorParameters_create()
        tag_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    out = os.path.abspath(args.out)
    os.makedirs(os.path.join(out, "preview"), exist_ok=True)
    owner_ref = os.path.dirname(out)
    for p in (out, os.path.join(out, "preview")):
        chown_like_parent(p, owner_ref)
    meta_path = os.path.join(out, "capture_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"captures": {}}
    meta["args"] = vars(args)

    rclpy.init()
    node = Capture(args)
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()

    print("[wait] image %s ..." % args.image_topic)
    t0 = time.time()
    while True:
        img, info, _ = node.snapshot()
        if img is not None and info is not None:
            break
        if time.time() - t0 > 15:
            sys.exit("no image/camera_info received in 15 s - is worker_bringup (camera_head) running?")
        time.sleep(0.2)
    print("[ok] image %dx%d %s, fx=%.3f fy=%.3f cx=%.3f cy=%.3f" % (
        info.width, info.height, img.encoding, info.k[0], info.k[4], info.k[2], info.k[5]))
    xml_path = os.path.join(out, PREFIX + "_camera.xml")
    write_camera_xml(xml_path, info)
    chown_like_parent(xml_path, owner_ref)
    meta["camera_info"] = {"width": info.width, "height": info.height, "K": list(info.k),
                           "D": list(info.d), "distortion_model": info.distortion_model,
                           "frame_id": info.header.frame_id}

    if args.mode == "handeye":
        try:
            node.lookup(Time())
            print("[ok] TF %s -> %s" % (args.ref_frame, args.ee_frame))
        except Exception as e:  # noqa: BLE001
            sys.exit("TF %s -> %s not available: %s" % (args.ref_frame, args.ee_frame, e))

    poses = {}  # index -> Rotation, for diversity warnings
    for i_str, c in meta["captures"].items():
        if "rotvec" in c:
            poses[int(i_str)] = Rotation.from_rotvec(c["rotvec"])
    last_dir = None

    def save_meta():
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=1)
        chown_like_parent(meta_path, owner_ref)

    print("\nmode=%s  out=%s  existing=%d" % (args.mode, out, len(existing_indices(out))))
    print("Enter = capture | d = delete last | l = list | q = quit\n")
    while True:
        try:
            cmd = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            cmd = "q"
        if cmd == "q":
            break
        if cmd == "l":
            print("captures:", existing_indices(out))
            continue
        if cmd == "d":
            idx = existing_indices(out)
            if not idx:
                print("nothing to delete")
                continue
            i = idx[-1]
            for p in (PREFIX + "_image-%d.png" % i, PREFIX + "_pose_rPe_%d.yaml" % i,
                      "preview/" + PREFIX + "_image-%d.jpg" % i):
                if os.path.exists(os.path.join(out, p)):
                    os.remove(os.path.join(out, p))
            meta["captures"].pop(str(i), None)
            poses.pop(i, None)
            save_meta()
            print("deleted #%d" % i)
            continue
        if cmd != "":
            print("unknown command")
            continue

        img, info, js = node.snapshot()
        age = (node.get_clock().now() - Time.from_msg(img.header.stamp)).nanoseconds * 1e-9
        if age > 1.0:
            print("[reject] image is %.1f s old - camera stalled?" % age)
            continue
        bgr = image_to_bgr(img)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        warn = []
        if args.target == "apriltag":
            tcorners, ids, _ = cv2.aruco.detectMarkers(gray, tag_dict, parameters=tag_params)
            if ids is None or len(ids) != 1:
                print("[reject] need exactly 1 tag36h11, found %d" % (0 if ids is None else len(ids)))
                continue
            corners = tcorners[0].reshape(-1, 1, 2)
            c0, c_end = corners[0, 0], corners[1, 0]
            direction = np.degrees(np.arctan2(c_end[1] - c0[1], c_end[0] - c0[0]))
            side_px = float(np.mean([np.linalg.norm(corners[k, 0] - corners[(k + 1) % 4, 0])
                                     for k in range(4)]))
            if side_px < 60:
                warn.append("tag only %.0f px wide - bring it closer (pose gets noisy)" % side_px)
            if args.tag_size > 0:
                K = np.array(info.k).reshape(3, 3)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    tcorners, args.tag_size, K, np.array(info.d))
                Rt, _ = cv2.Rodrigues(rvecs[0])
                tilt = float(np.degrees(np.arccos(abs(Rt[2, 2]))))
                if tilt < 15:
                    warn.append("tag almost facing the camera (tilt %.0f deg) - single-tag pose is "
                                "ambiguous here, tilt it >= 20 deg" % tilt)
                print("  tag %d: %.0f px, dist %.2f m, tilt %.0f deg"
                      % (int(ids[0][0]), side_px, float(np.linalg.norm(tvecs[0])), tilt))
        else:
            found, corners = cv2.findChessboardCorners(
                gray, (bw, bh), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            if not found:
                print("[reject] chessboard %dx%d not found (whole board visible? glare? blur?)" % (bw, bh))
                continue
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
            c0, c_end = corners[0, 0], corners[bw - 1, 0]
            direction = np.degrees(np.arctan2(c_end[1] - c0[1], c_end[0] - c0[0]))
            if last_dir is not None:
                d = (direction - last_dir + 180) % 360 - 180
                if abs(d) > 90:
                    warn.append("corner order flipped vs previous shot (in-plane rotation %.0f deg) - "
                                "rotate the board less in the image plane, or delete (d)" % d)

        rec = {"stamp": img.header.stamp.sec + img.header.stamp.nanosec * 1e-9,
               "corner0_px": [float(c0[0]), float(c0[1])], "row_dir_deg": float(direction)}
        if js is not None:
            rec["joint_states"] = dict(zip(js.name, [float(p) for p in js.position]))

        if args.mode == "handeye":
            stamp = Time.from_msg(img.header.stamp)
            try:
                trans, rot = tf_to_pose(node.lookup(stamp))
            except Exception as e:  # noqa: BLE001
                print("[reject] TF lookup at image stamp failed: %s" % e)
                continue
            try:
                trans0, rot0 = tf_to_pose(node.lookup(stamp - Duration(seconds=STILL_DT)))
                dt_m = float(np.linalg.norm(trans - trans0))
                dr_deg = float(np.degrees((rot0.inv() * rot).magnitude()))
                if (dt_m > STILL_TRANS or dr_deg > STILL_ROT_DEG) and not args.force:
                    print("[reject] arm moving (%.1f mm / %.2f deg in %.1f s) - wait until still"
                          % (dt_m * 1e3, dr_deg, STILL_DT))
                    continue
            except Exception:  # noqa: BLE001
                warn.append("could not check stillness (TF history too short)")
            if poses:
                nearest = min(np.degrees((r.inv() * rot).magnitude()) for r in poses.values())
                if nearest < MIN_DIVERSITY_DEG:
                    warn.append("only %.1f deg from an existing pose - vary the orientation more"
                                % nearest)
            rec.update({"trans": trans.tolist(), "rotvec": rot.as_rotvec().tolist(),
                        "rpy_deg": rot.as_euler("xyz", degrees=True).tolist()})

        idx = existing_indices(out)
        i = (idx[-1] + 1) if idx else 1
        img_path = os.path.join(out, PREFIX + "_image-%d.png" % i)
        cv2.imwrite(img_path, bgr)
        chown_like_parent(img_path, owner_ref)
        prev = bgr.copy()
        if args.target == "apriltag":
            cv2.polylines(prev, [corners.reshape(-1, 2).astype(np.int32)], True, (0, 255, 0), 2)
        else:
            cv2.drawChessboardCorners(prev, (bw, bh), corners, True)
        cv2.circle(prev, (int(c0[0]), int(c0[1])), 8, (0, 0, 255), 2)
        cv2.putText(prev, "#%d" % i, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        prev_path = os.path.join(out, "preview", PREFIX + "_image-%d.jpg" % i)
        cv2.imwrite(prev_path, prev)
        chown_like_parent(prev_path, owner_ref)
        if args.mode == "handeye":
            pose_path = os.path.join(out, PREFIX + "_pose_rPe_%d.yaml" % i)
            write_pose_yaml(pose_path, trans, rot.as_rotvec())
            chown_like_parent(pose_path, owner_ref)
            poses[i] = rot
        meta["captures"][str(i)] = rec
        save_meta()
        last_dir = direction

        msg = "[saved #%d] total=%d" % (i, len(existing_indices(out)))
        if args.mode == "handeye":
            msg += "  t=[%.3f %.3f %.3f] m  rpy=[%.1f %.1f %.1f] deg" % (
                *trans, *rot.as_euler("xyz", degrees=True))
        print(msg)
        for w in warn:
            print("  [warn] " + w)

    n = len(existing_indices(out))
    need = 15 if args.mode == "handeye" else 10
    print("\n%d captures in %s%s" % (n, out, "" if n >= need else "  (recommend >= %d)" % need))
    ex.shutdown()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
