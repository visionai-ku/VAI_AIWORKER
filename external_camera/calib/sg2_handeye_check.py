#!/usr/bin/env python3
"""
Diagnostics for an SG2 eye-to-hand dataset (run after sg2_calib_compute.sh handeye, which
writes sg2_pose_cPo_<i>.yaml next to sg2_pose_rPe_<i>.yaml).

1. Per-pose consistency: for every pair (i, j) the relative rotation angle of the robot
   (rPe_i^-1 rPe_j) must equal that of the board seen by the camera (cPo_i^-1 cPo_j).
   This is independent of the unknown mount, the square size and the solver.
   Good data: median mismatch < ~0.5 deg. A pose far above the rest is corrupted.
2. Motion diversity: largest relative rotation and spread of relative-rotation axes
   (one dominant singular value = rotations about one axis = translation along it unobservable).
3. Optional --prune: drop the worst pose (by ViSP's per-pose residual) one at a time and re-run
   visp-compute-eye-to-hand-calibration, to see whether xyz is stable.

Usage:
  python3 sg2_handeye_check.py <data_dir> [--exclude 1,5] [--prune] [--min-poses 10]
  python3 sg2_handeye_check.py <data_dir> --exclude 1 --export <new_dir>   # renumbered copy
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

VISP = os.environ.get("VISP_APPS", os.path.expanduser("~/Desktop/yujin/visp/build/apps/calibration"))


def load(path):
    v = [float(l.strip()[3:-1]) for l in open(path) if l.strip().startswith("- [")]
    return np.array(v[:3]), Rotation.from_rotvec(v[3:])


def angle_deg(r):
    return float(np.degrees(np.linalg.norm(r.as_rotvec())))  # scipy<1.4 has no magnitude()


def indices(d):
    idx = []
    for p in glob.glob(os.path.join(d, "sg2_pose_rPe_*.yaml")):
        i = int(re.search(r"_(\d+)\.yaml$", p).group(1))
        if os.path.exists(os.path.join(d, "sg2_pose_cPo_%d.yaml" % i)):
            idx.append(i)
    return sorted(idx)


def consistency(d, idx):
    E = {i: load(os.path.join(d, "sg2_pose_rPe_%d.yaml" % i)) for i in idx}
    C = {i: load(os.path.join(d, "sg2_pose_cPo_%d.yaml" % i)) for i in idx}
    print("per-pose median |robot rel-angle - camera rel-angle| [deg] (good < ~0.5):")
    row = []
    for i in idx:
        e = [abs(angle_deg(E[i][1].inv() * E[j][1]) - angle_deg(C[i][1].inv() * C[j][1]))
             for j in idx if j != i]
        row.append("%d:%.2f" % (i, np.median(e)))
    print("  " + "  ".join(row))
    rel, axes = [], []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            rv = (E[idx[a]][1].inv() * E[idx[b]][1]).as_rotvec()
            rel.append(np.degrees(np.linalg.norm(rv)))
            if np.linalg.norm(rv) > np.radians(3):
                axes.append(rv / np.linalg.norm(rv))
    s = np.linalg.svd(np.array(axes), compute_uv=False) if axes else np.zeros(3)
    print("max relative robot rotation: %.1f deg (want >= ~60)" % max(rel))
    print("relative-rotation axis spread (want 3 similar values): %s" % np.round(s / s[0], 3))
    print("board distance from camera [m]: %s" % " ".join("%.2f" % np.linalg.norm(C[i][0]) for i in idx))


def solve(d, idx):
    tmp = tempfile.mkdtemp(prefix="sg2_he_")
    try:
        for j, i in enumerate(idx, 1):
            for k in ("rPe", "cPo"):
                shutil.copy(os.path.join(d, "sg2_pose_%s_%d.yaml" % (k, i)),
                            os.path.join(tmp, "p_%s_%d.yaml" % (k, j)))
        out = subprocess.run([os.path.join(VISP, "hand-eye", "visp-compute-eye-to-hand-calibration"),
                              "--data-path", tmp, "--rPe", "p_rPe_%d.yaml", "--cPo", "p_cPo_%d.yaml",
                              "--output-rPc", tmp + "/rPc.yaml", "--output-ePo", tmp + "/ePo.yaml"],
                             capture_output=True, text=True).stdout
        if not os.path.exists(tmp + "/rPc.yaml"):
            return None
        th = [float(x) for x in re.findall(r"Distance theta between rMo/rMc\(\d+\) and mean \(deg\) = (\S+)", out)]
        dd = [float(x) for x in re.findall(r"Distance d between rMo/rMc\(\d+\) and mean \(m\) = (\S+)", out)]
        mr = float(re.search(r"Mean residual.*rotation \(deg\) = (\S+)", out).group(1))
        mt = float(re.search(r"Mean residual.*translation \(m\) = (\S+)", out).group(1))
        t, r = load(tmp + "/rPc.yaml")
        return th, dd, mr, mt, t, r
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_dir")
    ap.add_argument("--exclude", default="", help="comma-separated pose indices to drop, e.g. 1,5")
    ap.add_argument("--prune", action="store_true", help="greedily drop worst poses and re-solve")
    ap.add_argument("--min-poses", type=int, default=10)
    ap.add_argument("--export", help="write the kept poses (images+rPe, renumbered 1..N) to this dir")
    a = ap.parse_args()
    excl = {int(x) for x in a.exclude.split(",") if x.strip()}
    idx = [i for i in indices(a.data_dir) if i not in excl]
    print("poses: %s%s" % (idx, "  (excluded %s)" % sorted(excl) if excl else ""))
    if not idx:
        raise SystemExit("no sg2_pose_cPo_*.yaml - run sg2_calib_compute.sh handeye first")
    consistency(a.data_dir, idx)

    if a.prune:
        print("\ngreedy prune (worst = largest d/5mm + theta/deg):")
        keep = list(idx)
        while len(keep) >= a.min_poses:
            res = solve(a.data_dir, keep)
            if res is None:
                print("  n=%d  solver failed" % len(keep))
                break
            th, dd, mr, mt, t, r = res
            w = max(range(len(dd)), key=lambda k: dd[k] / 0.005 + th[k])
            print("  n=%2d  rot %.2f deg  trans %5.1f mm  xyz=(%.3f %.3f %.3f)  rpy=(%.1f %.1f %.1f) deg  worst=%d"
                  % (len(keep), mr, mt * 1e3, *t, *r.as_euler("xyz", degrees=True), keep[w]))
            keep.pop(w)
        print("stable xyz/rpy across rows = trustworthy; a component that jumps is unobservable.")

    if a.export:
        os.makedirs(a.export, exist_ok=True)
        shutil.copy(os.path.join(a.data_dir, "sg2_camera.xml"), a.export)
        for j, i in enumerate(idx, 1):
            shutil.copy(os.path.join(a.data_dir, "sg2_image-%d.png" % i), os.path.join(a.export, "sg2_image-%d.png" % j))
            shutil.copy(os.path.join(a.data_dir, "sg2_pose_rPe_%d.yaml" % i),
                        os.path.join(a.export, "sg2_pose_rPe_%d.yaml" % j))
        print("\nexported %d poses to %s (run sg2_calib_compute.sh handeye on it)" % (len(idx), a.export))


if __name__ == "__main__":
    main()
