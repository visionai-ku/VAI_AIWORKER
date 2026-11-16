#!/usr/bin/env python3
"""
SG2 head camera live view + snapshot button, in the browser.

Run inside the ai_worker container (worker_bringup running, ROS_DOMAIN_ID=30,
RMW_IMPLEMENTATION=rmw_zenoh_cpp), then open http://<robot>:8097/ on the Mac:
  python3 sg2_head_cam_snap.py --out data-snap-1110 [--port 8097] [--tag-size 0.096]

Each click on "Capture" (or Space) saves the latest RAW frame plus the robot pose at that frame's
stamp, in ViSP's file layout (same as sg2_calib_capture.py):
  sg2_image-<i>.png        raw color frame (BGR PNG)
  sg2_pose_rPe_<i>.yaml    vpPoseVector of --ee-frame in --ref-frame at the image stamp (if TF ok)
  sg2_camera.xml           factory intrinsics from camera_info
  capture_meta.json        stamps, joint_states, warnings
The chessboard/AprilTag overlay is only a guide; nothing is rejected.
"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
import tf2_ros
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sg2_calib_capture import (PREFIX, chown_like_parent, existing_indices, image_to_bgr,  # noqa: E402
                               tf_to_pose, write_camera_xml, write_pose_yaml)

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>SG2 head cam</title>
<style>
body{margin:0;background:#111;color:#ddd;font:15px system-ui,sans-serif}
#wrap{display:flex;flex-wrap:wrap;gap:12px;padding:12px}
#live{max-width:min(960px,100%);width:100%;border:1px solid #333}
#side{flex:1;min-width:260px}
button{font-size:18px;padding:12px 20px;margin:4px 4px 4px 0;border:0;border-radius:6px;cursor:pointer}
#cap{background:#2e7d32;color:#fff}#del{background:#555;color:#fff}
#log{white-space:pre-wrap;font:13px ui-monospace,monospace;background:#1b1b1b;padding:8px;
     max-height:50vh;overflow:auto;margin-top:8px}
#last{max-width:100%;margin-top:8px;border:1px solid #333}
#n{font-size:28px;font-weight:600}
</style></head><body><div id="wrap">
<img id="live" src="/stream">
<div id="side">
 <div>saved: <span id="n">-</span></div>
 <div id="dir" style="font-size:12px;color:#999"></div>
 <button id="cap" onclick="cap()">Capture (Space)</button><button id="del" onclick="del_()">Delete last</button>
 <div id="log"></div><img id="last">
</div></div>
<script>
const log=t=>{const l=document.getElementById('log');l.textContent=t+"\\n"+l.textContent};
async function post(u){const r=await fetch(u,{method:'POST'});return r.json()}
async function cap(){const b=document.getElementById('cap');b.disabled=true;
 try{const j=await post('/capture');log(j.msg);upd(j);if(j.ok)document.getElementById('last').src='/last?'+Date.now()}
 catch(e){log('error '+e)} b.disabled=false}
async function del_(){const j=await post('/delete');log(j.msg);upd(j)}
function upd(j){document.getElementById('n').textContent=j.count;document.getElementById('dir').textContent=j.out}
document.addEventListener('keydown',e=>{if(e.code==='Space'){e.preventDefault();cap()}});
fetch('/status').then(r=>r.json()).then(upd);
</script></body></html>"""


class Cam(Node):
    def __init__(self, a):
        super().__init__("sg2_head_cam_snap")
        self.a = a
        self.lock = threading.Lock()
        self.img = self.info = self.js = None
        self.jpg, self.seq = None, 0
        self.last_view = 0.0
        ns = a.camera_ns
        self.create_subscription(Image, ns + "image_rect_raw", self.on_img, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, ns + "camera_info", self.on_info, qos_profile_sensor_data)
        self.create_subscription(JointState, "/joint_states", self.on_js, 50)
        self.tf = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self.tfl = tf2_ros.TransformListener(self.tf, self)
        self.tag_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.tag_params = cv2.aruco.DetectorParameters_create()
        self.bw, self.bh = (int(v) for v in a.board.lower().split("x"))

    def on_info(self, m):
        with self.lock:
            self.info = m

    def on_js(self, m):
        with self.lock:
            self.js = m

    def on_img(self, m):
        with self.lock:
            self.img = m
        now = time.time()
        if now - self.last_view < 1.0 / 15:
            return
        self.last_view = now
        bgr = image_to_bgr(m)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        txt = []
        ok, cb = cv2.findChessboardCorners(gray, (self.bw, self.bh), cv2.CALIB_CB_ADAPTIVE_THRESH
                                           | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
        if ok:
            cv2.drawChessboardCorners(bgr, (self.bw, self.bh), cb, True)
            g = cb.reshape(self.bh, self.bw, 2)
            sq = min(np.linalg.norm(np.diff(g, axis=1), axis=2).min(),
                     np.linalg.norm(np.diff(g, axis=0), axis=2).min())
            txt.append(("board %dx%d  min square %.0f px" % (self.bw, self.bh, sq),
                        (0, 200, 0) if sq >= 20 else (0, 165, 255)))
        c, ids, _ = cv2.aruco.detectMarkers(gray, self.tag_dict, parameters=self.tag_params)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(bgr, c, ids)
            q = c[0].reshape(-1, 2)
            side = float(np.mean([np.linalg.norm(q[k] - q[(k + 1) % 4]) for k in range(4)]))
            t = "tag %d  %.0f px" % (int(ids[0][0]), side)
            if self.a.tag_size > 0 and self.info is not None:
                K, D = np.array(self.info.k).reshape(3, 3), np.array(self.info.d)
                r, tv, _ = cv2.aruco.estimatePoseSingleMarkers(c[:1], self.a.tag_size, K, D)
                tilt = float(np.degrees(np.arccos(abs(cv2.Rodrigues(r[0])[0][2, 2]))))
                t += "  %.2f m  tilt %.0f deg" % (float(np.linalg.norm(tv[0])), tilt)
            txt.append((t, (0, 200, 0)))
        if not txt:
            txt.append(("no board / tag", (0, 0, 255)))
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 30 * len(txt) + 6), (0, 0, 0), -1)
        for k, (s, col) in enumerate(txt):
            cv2.putText(bgr, s, (8, 24 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
        ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self.lock:
                self.jpg, self.seq = jpg.tobytes(), self.seq + 1


class Store:
    def __init__(self, a, node):
        self.a, self.node = a, node
        self.out = os.path.abspath(a.out)
        os.makedirs(self.out, exist_ok=True)
        self.owner = os.path.dirname(self.out)
        chown_like_parent(self.out, self.owner)
        self.meta_path = os.path.join(self.out, "capture_meta.json")
        self.meta = json.load(open(self.meta_path)) if os.path.exists(self.meta_path) else {"captures": {}}
        self.meta["args"] = vars(a)
        self.lock = threading.Lock()
        self.last_png = None

    def _w(self, path):
        chown_like_parent(path, self.owner)

    def status(self, ok=True, msg=""):
        return {"ok": ok, "msg": msg, "count": len(existing_indices(self.out)), "out": self.out}

    def capture(self):
        n = self.node
        with n.lock:
            img, info, js = n.img, n.info, n.js
        if img is None or info is None:
            return self.status(False, "no image / camera_info yet")
        age = (n.get_clock().now() - Time.from_msg(img.header.stamp)).nanoseconds * 1e-9
        if age > 1.0:
            return self.status(False, "image is %.1f s old - camera stalled?" % age)
        with self.lock:
            xml = os.path.join(self.out, PREFIX + "_camera.xml")
            if not os.path.exists(xml):
                write_camera_xml(xml, info)
                self._w(xml)
            idx = existing_indices(self.out)
            i = (idx[-1] + 1) if idx else 1
            bgr = image_to_bgr(img)
            p = os.path.join(self.out, PREFIX + "_image-%d.png" % i)
            cv2.imwrite(p, bgr)
            self._w(p)
            self.last_png = p
            rec = {"stamp": img.header.stamp.sec + img.header.stamp.nanosec * 1e-9}
            if js is not None:
                rec["joint_states"] = dict(zip(js.name, [float(v) for v in js.position]))
            msg = "#%d saved" % i
            try:
                st = Time.from_msg(img.header.stamp)
                tr, rot = tf_to_pose(n.tf.lookup_transform(self.a.ref_frame, self.a.ee_frame, st,
                                                           timeout=Duration(seconds=0.5)))
                pp = os.path.join(self.out, PREFIX + "_pose_rPe_%d.yaml" % i)
                write_pose_yaml(pp, tr, rot.as_rotvec())
                self._w(pp)
                rec.update({"trans": tr.tolist(), "rotvec": rot.as_rotvec().tolist()})
                msg += "  t=[%.3f %.3f %.3f]" % tuple(tr)
                try:
                    tr0, rot0 = tf_to_pose(n.tf.lookup_transform(
                        self.a.ref_frame, self.a.ee_frame, st - Duration(seconds=0.5)))
                    dm, dr = np.linalg.norm(tr - tr0) * 1e3, np.degrees((rot0.inv() * rot).magnitude())
                    if dm > 1.0 or dr > 0.2:
                        msg += "\n  [warn] arm moving (%.1f mm / %.2f deg in 0.5 s) - delete & retake" % (dm, dr)
                        rec["moving"] = True
                except Exception:  # noqa: BLE001
                    pass
            except Exception as e:  # noqa: BLE001
                msg += "  (no pose: %s)" % str(e).splitlines()[0][:80]
            self.meta["captures"][str(i)] = rec
            json.dump(self.meta, open(self.meta_path, "w"), indent=1)
            self._w(self.meta_path)
        print(msg, flush=True)
        return self.status(True, msg)

    def delete(self):
        with self.lock:
            idx = existing_indices(self.out)
            if not idx:
                return self.status(False, "nothing to delete")
            i = idx[-1]
            for f in (PREFIX + "_image-%d.png" % i, PREFIX + "_pose_rPe_%d.yaml" % i):
                fp = os.path.join(self.out, f)
                if os.path.exists(fp):
                    os.remove(fp)
            self.meta["captures"].pop(str(i), None)
            json.dump(self.meta, open(self.meta_path, "w"), indent=1)
        print("#%d deleted" % i, flush=True)
        return self.status(True, "#%d deleted" % i)


def make_handler(node, store):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, d):
            self._send(200, "application/json", json.dumps(d).encode())

        def do_POST(self):
            if self.path == "/capture":
                self._json(store.capture())
            elif self.path == "/delete":
                self._json(store.delete())
            else:
                self._send(404, "text/plain", b"404")

        def do_GET(self):
            if self.path == "/status":
                return self._json(store.status())
            if self.path.startswith("/last"):
                if store.last_png and os.path.exists(store.last_png):
                    ok, jpg = cv2.imencode(".jpg", cv2.imread(store.last_png))
                    return self._send(200, "image/jpeg", jpg.tobytes())
                return self._send(404, "text/plain", b"none")
            if self.path != "/stream":
                return self._send(200, "text/html; charset=utf-8", PAGE.encode())
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            seen = -1
            try:
                while True:
                    with node.lock:
                        jpg, seq = node.jpg, node.seq
                    if jpg is None or seq == seen:
                        time.sleep(0.02)
                        continue
                    seen = seq
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                    self.wfile.write(jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output folder (existing captures are continued)")
    ap.add_argument("--port", type=int, default=8097)
    ap.add_argument("--camera-ns", default="/camera_head/camera_head/color/")
    ap.add_argument("--ref-frame", default="head_link2")
    ap.add_argument("--ee-frame", default="gripper_l_rh_p12_rn_base")
    ap.add_argument("--board", default="9x6", help="chessboard inner corners for the overlay")
    ap.add_argument("--tag-size", type=float, default=0.0, help="tag black-square edge [m] for the overlay")
    a = ap.parse_args()

    rclpy.init()
    node = Cam(a)
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    store = Store(a, node)
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(node, store))
    print("open http://<robot>:%d/   saving to %s (existing %d)"
          % (a.port, store.out, len(existing_indices(store.out))), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        ex.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
