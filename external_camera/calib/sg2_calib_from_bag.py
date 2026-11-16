#!/usr/bin/env python3
"""
Pick calibration frames out of a rosbag of the SG2 head camera and write them in the same ViSP
format as sg2_calib_capture.py, so sg2_calib_compute.sh / sg2_handeye_check.py work unchanged.

Record (inside the ai_worker container, ROS_DOMAIN_ID=30 RMW_IMPLEMENTATION=rmw_zenoh_cpp),
move the board / arm slowly and HOLD each pose ~1-2 s, then Ctrl+C:
  ros2 bag record -o <bag_dir> /camera_head/camera_head/color/image_rect_raw \\
      /camera_head/camera_head/color/camera_info /tf /tf_static /joint_states

Extract:
  python3 sg2_calib_from_bag.py <bag_dir> --mode intrinsic --out <dir>
  python3 sg2_calib_from_bag.py <bag_dir> --mode handeye   --out <dir>
  python3 sg2_calib_from_bag.py <bag_dir> --mode handeye   --out <dir> --target apriltag --tag-size 0.096

Selection:
  1. detect the target in every --step'th frame,
  2. keep frames that are STILL: within +-still_sec all frames are detected and no corner moves more
     than still_px (and, in handeye mode, ref->ee TF moves < 1 mm / 0.2 deg),
  3. group consecutive still frames into holds, take the middle frame of each hold,
  4. if there are more holds than --n, farthest-point sampling on board pose (rotation, image
     position, size) so the kept frames are as different as possible.
Chessboard corner-order flips (OpenCV may start from either end) are tracked through the video;
in handeye mode the minority orientation is dropped.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import rosbag2_py
import tf2_ros
from rclpy.duration import Duration
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from rosidl_runtime_py.utilities import get_message
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sg2_calib_capture import PREFIX, image_to_bgr, tf_to_pose, write_camera_xml, write_pose_yaml  # noqa: E402

STILL_TRANS = 0.001
STILL_ROT_DEG = 0.2


def open_reader(bag, topics):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id=""),
           rosbag2_py.ConverterOptions(input_serialization_format="", output_serialization_format=""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    missing = [t for t in topics if t not in types]
    r.set_filter(rosbag2_py.StorageFilter(topics=[t for t in topics if t in types]))
    return r, types, missing


def stamp_sec(st):
    return st.sec + st.nanosec * 1e-9


class Detector:
    def __init__(self, args):
        self.args = args
        if args.target == "apriltag":
            self.tag_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
            self.params = cv2.aruco.DetectorParameters_create()
            self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            s = args.tag_size / 2.0
            self.obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float64)
        else:
            self.bw, self.bh = (int(v) for v in args.board.lower().split("x"))
            self.obj = np.array([[x, y, 0] for y in range(self.bh) for x in range(self.bw)], np.float64)

    def __call__(self, gray):
        if self.args.target == "apriltag":
            c, ids, _ = cv2.aruco.detectMarkers(gray, self.tag_dict, parameters=self.params)
            if ids is None or len(ids) != 1:
                return None
            return c[0].reshape(-1, 2)
        f = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
        ok, c = cv2.findChessboardCorners(gray, (self.bw, self.bh), f)
        if not ok:
            return None
        # refinement window must stay inside one square, or corners snap to neighbours (5 px jitter
        # with the usual 11x11 window when squares are ~10 px)
        win = int(np.clip(0.5 * self.min_square(c.reshape(-1, 2)), 3, 11))
        c = cv2.cornerSubPix(gray, c, (win, win), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
        return c.reshape(-1, 2)

    def min_square(self, c):
        if self.args.target == "apriltag":
            return float(min(np.linalg.norm(c[k] - c[(k + 1) % 4]) for k in range(4)) / 8.0)
        g = c.reshape(self.bh, self.bw, 2)
        return float(min(np.linalg.norm(np.diff(g, axis=1), axis=2).min(),
                         np.linalg.norm(np.diff(g, axis=0), axis=2).min()))

    def row_dir(self, c):
        end = c[1] if self.args.target == "apriltag" else c[self.bw - 1]
        return float(np.degrees(np.arctan2(end[1] - c[0][1], end[0] - c[0][0])))


def ang_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("--mode", choices=["intrinsic", "handeye"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", choices=["chessboard", "apriltag"], default="chessboard")
    ap.add_argument("--board", default="9x6", help="inner corners WxH")
    ap.add_argument("--tag-size", type=float, default=0.0, help="apriltag black-square edge [m]")
    ap.add_argument("--n", type=int, default=25, help="max frames to keep")
    ap.add_argument("--step", type=int, default=1, help="detect in every k-th frame (speed)")
    ap.add_argument("--still-sec", type=float, default=0.3, help="half window for the stillness check")
    ap.add_argument("--still-px", type=float, default=1.0, help="max corner motion inside the window")
    ap.add_argument("--min-square-px", type=float, default=20.0,
                    help="chessboard: drop frames whose smallest square is below this; ViSP refines "
                         "corners with an 11x11 window (23 px) and drifts on smaller squares")
    ap.add_argument("--image-topic", default="/camera_head/camera_head/color/image_rect_raw")
    ap.add_argument("--info-topic", default="/camera_head/camera_head/color/camera_info")
    ap.add_argument("--ref-frame", default="head_link2")
    ap.add_argument("--ee-frame", default="gripper_l_rh_p12_rn_base")
    args = ap.parse_args()
    if args.target == "apriltag" and args.tag_size <= 0:
        sys.exit("--tag-size is required with --target apriltag")

    out = os.path.abspath(args.out)
    if os.path.isdir(out) and any(f.startswith(PREFIX + "_image-") for f in os.listdir(out)):
        sys.exit("%s already has %s_image-*.png - use a new --out" % (out, PREFIX))
    os.makedirs(os.path.join(out, "preview"), exist_ok=True)
    det = Detector(args)

    # ---- pass 1: camera_info, TF buffer, detections -------------------------------------------
    topics = [args.image_topic, args.info_topic]
    if args.mode == "handeye":
        topics += ["/tf", "/tf_static"]
    reader, types, missing = open_reader(args.bag, topics)
    if args.image_topic in missing:
        sys.exit("bag has no %s" % args.image_topic)
    if args.mode == "handeye" and "/tf" in missing:
        sys.exit("handeye mode needs /tf (and /tf_static) in the bag")
    msg_cls = {t: get_message(types[t]) for t in topics if t in types}
    tfbuf = tf2_ros.Buffer(cache_time=Duration(seconds=24 * 3600))
    info = None
    frames = []  # dicts: t, corners, dir
    n_img = 0
    print("[pass 1] reading %s ..." % args.bag)
    while reader.has_next():
        topic, data, _ = reader.read_next()
        msg = deserialize_message(data, msg_cls[topic])
        if topic == "/tf":
            for tf in msg.transforms:
                tfbuf.set_transform(tf, "bag")
        elif topic == "/tf_static":
            for tf in msg.transforms:
                tfbuf.set_transform_static(tf, "bag")
        elif topic == args.info_topic:
            info = info or msg
        elif topic == args.image_topic:
            n_img += 1
            if (n_img - 1) % args.step:
                continue
            gray = cv2.cvtColor(image_to_bgr(msg), cv2.COLOR_BGR2GRAY)
            c = det(gray)
            frames.append({"t": stamp_sec(msg.header.stamp), "stamp": msg.header.stamp, "c": c,
                           "dir": det.row_dir(c) if c is not None else None})
            if n_img % 300 == 0:
                print("  %d images, %d detected" % (n_img, sum(f["c"] is not None for f in frames)))
    del reader
    if info is None:
        sys.exit("bag has no %s" % args.info_topic)
    n_det = sum(f["c"] is not None for f in frames)
    dur = frames[-1]["t"] - frames[0]["t"] if len(frames) > 1 else 0.0
    fps = (len(frames) - 1) / dur if dur > 0 else 30.0 / args.step
    print("[pass 1] %d images (%.1f s, %.1f fps processed), target detected in %d" % (n_img, dur, fps, n_det))
    if n_det == 0:
        sys.exit("target never detected - board too small/oblique, wrong --board/--target?")

    K = np.array(info.k).reshape(3, 3)
    D = np.array(info.d)
    W, H = info.width, info.height

    # ---- chessboard corner-order parity (tracked through the video) ---------------------------
    parity, prev = 0, None
    for f in frames:
        if f["c"] is None:
            continue
        if prev is not None and abs(ang_diff(f["dir"], prev["dir"])) > 90 and det.args.target == "chessboard":
            parity ^= 1
        f["parity"] = parity
        prev = f
    par_count = np.bincount([f["parity"] for f in frames if f["c"] is not None], minlength=2)
    major = int(np.argmax(par_count))

    # ---- stillness ----------------------------------------------------------------------------
    times = np.array([f["t"] for f in frames])
    need = max(2, int(0.7 * 2 * args.still_sec * fps))
    stats = {"not_still": 0, "flip": 0, "tf": 0, "small": 0}
    still = []
    for i, f in enumerate(frames):
        if f["c"] is None:
            continue
        lo, hi = np.searchsorted(times, f["t"] - args.still_sec), np.searchsorted(times, f["t"] + args.still_sec, "right")
        win = [g for g in frames[lo:hi] if g["c"] is not None and g["parity"] == f["parity"]]
        if len(win) < need or max(np.abs(g["c"] - f["c"]).max() for g in win) > args.still_px:
            stats["not_still"] += 1
            continue
        if args.target == "chessboard" and det.min_square(f["c"]) < args.min_square_px:
            stats["small"] += 1
            continue
        if args.mode == "handeye" and f["parity"] != major:
            stats["flip"] += 1
            continue
        if args.mode == "handeye":
            try:
                st = Time.from_msg(f["stamp"])
                tr, rot = tf_to_pose(tfbuf.lookup_transform(args.ref_frame, args.ee_frame, st))
                moved = False
                for d in (-args.still_sec, args.still_sec):
                    tr2, rot2 = tf_to_pose(tfbuf.lookup_transform(
                        args.ref_frame, args.ee_frame, st + Duration(seconds=d)))
                    if (np.linalg.norm(tr - tr2) > STILL_TRANS
                            or np.degrees((rot.inv() * rot2).magnitude()) > STILL_ROT_DEG):
                        moved = True
                if moved:
                    stats["tf"] += 1
                    continue
                f["trans"], f["rot"] = tr, rot
            except Exception as e:  # noqa: BLE001
                stats["tf"] += 1
                f["tf_err"] = str(e)
                continue
        ok, rvec, tvec = cv2.solvePnP(det.obj, f["c"].astype(np.float64), K, D)
        f["brot"] = Rotation.from_rotvec(rvec.ravel())
        f["center"] = f["c"].mean(axis=0)
        f["size"] = float(np.sqrt(cv2.contourArea(cv2.convexHull(f["c"].astype(np.float32)))))
        f["tilt"] = float(np.degrees(np.arccos(abs(cv2.Rodrigues(rvec)[0][2, 2]))))
        f["i"] = i
        still.append(f)
    print("[still] %d usable still frames  (rejected: moving %d, squares < %.0f px %d, "
          "minority corner order %d, TF %d)" % (len(still), stats["not_still"], args.min_square_px,
                                                 stats["small"], stats["flip"], stats["tf"]))
    if stats["small"] and not still:
        print("  board too small in the image - bring it closer (or --min-square-px lower, at a cost)")
    if stats["tf"] and not still:
        errs = [f.get("tf_err") for f in frames if f.get("tf_err")]
        print("  TF error example:", errs[0] if errs else "arm moving in every hold")
    if not still:
        sys.exit("no usable frames - hold each pose ~1 s, board large in the image")

    # ---- holds: consecutive still frames -> middle one ----------------------------------------
    holds, cur = [], [still[0]]
    for f in still[1:]:
        p = cur[-1]
        if f["t"] - p["t"] < 3.0 / fps and np.abs(f["c"] - p["c"]).max() < 2 * args.still_px:
            cur.append(f)
        else:
            holds.append(cur)
            cur = [f]
    holds.append(cur)
    cands = [h[len(h) // 2] for h in holds]
    print("[holds] %d distinct holds" % len(cands))

    # ---- farthest-point sampling on board pose ------------------------------------------------
    def dist(a, b):
        return (np.degrees((a["brot"].inv() * b["brot"]).magnitude()) / 15.0
                + np.linalg.norm(a["center"] - b["center"]) / (0.15 * W)
                + abs(np.log(a["size"] / b["size"])) / 0.3)

    if len(cands) > args.n:
        first = max(range(len(cands)), key=lambda k: cands[k]["tilt"])
        sel = [first]
        dmin = np.array([dist(cands[first], c) for c in cands])
        while len(sel) < args.n:
            k = int(np.argmax(dmin))
            sel.append(k)
            dmin = np.minimum(dmin, [dist(cands[k], c) for c in cands])
        chosen = sorted((cands[k] for k in sel), key=lambda f: f["t"])
    else:
        chosen = cands
    want = {f["i"]: n + 1 for n, f in enumerate(chosen)}

    # ---- pass 2: write the chosen frames ------------------------------------------------------
    write_camera_xml(os.path.join(out, PREFIX + "_camera.xml"), info)
    reader, _, _ = open_reader(args.bag, [args.image_topic])
    img_cls = get_message(types[args.image_topic])
    k, idx = 0, -1
    meta = {"args": vars(args), "bag": os.path.abspath(args.bag),
            "camera_info": {"width": W, "height": H, "K": list(info.k), "D": list(info.d),
                            "distortion_model": info.distortion_model, "frame_id": info.header.frame_id},
            "captures": {}}
    by_i = {f["i"]: f for f in chosen}
    while reader.has_next() and len(meta["captures"]) < len(chosen):
        _, data, _ = reader.read_next()
        k += 1
        if (k - 1) % args.step:
            continue
        idx += 1
        if idx not in want:
            continue
        n, f = want[idx], by_i[idx]
        bgr = image_to_bgr(deserialize_message(data, img_cls))
        cv2.imwrite(os.path.join(out, PREFIX + "_image-%d.png" % n), bgr)
        prev = bgr.copy()
        if args.target == "apriltag":
            cv2.polylines(prev, [f["c"].astype(np.int32)], True, (0, 255, 0), 2)
        else:
            cv2.drawChessboardCorners(prev, (det.bw, det.bh), f["c"].reshape(-1, 1, 2).astype(np.float32), True)
        cv2.circle(prev, tuple(int(v) for v in f["c"][0]), 8, (0, 0, 255), 2)
        cv2.putText(prev, "#%d t=%.1f tilt=%.0f" % (n, f["t"] - frames[0]["t"], f["tilt"]), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(out, "preview", PREFIX + "_image-%d.jpg" % n), prev)
        rec = {"stamp": f["t"], "t_rel": f["t"] - frames[0]["t"], "corner0_px": f["c"][0].tolist(),
               "row_dir_deg": f["dir"], "tilt_deg": f["tilt"], "size_px": f["size"]}
        if args.mode == "handeye":
            write_pose_yaml(os.path.join(out, PREFIX + "_pose_rPe_%d.yaml" % n), f["trans"], f["rot"].as_rotvec())
            rec.update({"trans": f["trans"].tolist(), "rotvec": f["rot"].as_rotvec().tolist()})
        meta["captures"][str(n)] = rec
    del reader
    with open(os.path.join(out, "capture_meta.json"), "w") as fp:
        json.dump(meta, fp, indent=1)

    # ---- summary ------------------------------------------------------------------------------
    cx = np.array([f["center"] for f in chosen])
    grid = np.zeros((3, 3), int)
    for x, y in cx:
        grid[min(2, int(3 * y / H)), min(2, int(3 * x / W))] += 1
    tilts = [f["tilt"] for f in chosen]
    rel = [np.degrees((a["brot"].inv() * b["brot"]).magnitude()) for a in chosen for b in chosen]
    print("\n[saved] %d frames -> %s" % (len(chosen), out))
    print("  tilt deg: min %.0f  median %.0f  max %.0f   (want a spread up to 30-45)"
          % (min(tilts), np.median(tilts), max(tilts)))
    print("  max relative rotation between frames: %.0f deg" % max(rel))
    print("  board-center coverage (3x3 image grid):")
    for row in grid:
        print("    " + " ".join("%3d" % v for v in row))
    recommend = 15 if args.mode == "handeye" else 10
    if len(chosen) < recommend:
        print("  [warn] only %d frames (recommend >= %d): record more holds" % (len(chosen), recommend))
    if args.mode == "intrinsic" and (grid == 0).sum() > 3:
        print("  [warn] board never reached %d of 9 image regions - corners/edges matter for distortion"
              % (grid == 0).sum())
    print("next: check preview/, then on vai_4090: sg2_calib_compute.sh %s <dir> <square_size_m>" % args.mode)


if __name__ == "__main__":
    main()
