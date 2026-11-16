#!/usr/bin/env bash
# SG2 head D405 calibration - compute step (runs on vai_4090, where ViSP is built).
#
#   ./sg2_calib_compute.sh intrinsic <data_dir> <square_size_m>
#   ./sg2_calib_compute.sh handeye   <data_dir> <square_size_m> [intrinsic_xml]
#
# intrinsic: visp-calibrate-camera (headless via Xvfb + auto-click), initialised from the
#            factory intrinsics sg2_camera.xml, then compares result vs factory.
# handeye  : visp-compute-chessboard-poses -> visp-compute-eye-to-hand-calibration
#            -> rmc_to_urdf.py (URDF / Isaac numbers). Uses <data_dir>/sg2_camera.xml
#            (factory intrinsics) unless [intrinsic_xml] is given.
# Board inner corners default to 9x6 (OpenCV_Chessboard.pdf); override with BOARD_W / BOARD_H.
# AprilTag target (handeye only): TARGET=apriltag ./sg2_calib_compute.sh handeye <data_dir> <tag_size_m>
#   tag_size_m = edge of the BLACK square (36h11: 8/10 of the full tag incl. white border).
set -euo pipefail

VISP=${VISP_APPS:-$HOME/Desktop/yujin/visp/build/apps/calibration}
HERE=$(cd "$(dirname "$0")" && pwd)
BW=${BOARD_W:-9}
BH=${BOARD_H:-6}
TARGET=${TARGET:-chessboard}

[ $# -ge 3 ] || { sed -n 2,13p "$0"; exit 1; }
mode=$1
data=$(cd "$2" && pwd)
sq=$3
n=$(ls "$data"/sg2_image-*.png 2>/dev/null | wc -l)
echo "data: $data  images: $n  board: ${BW}x${BH}  square: $sq m"

case $mode in
intrinsic)
  [ "$TARGET" = chessboard ] || { echo "intrinsic mode needs the chessboard"; exit 1; }
  cfg=$data/sg2_chessboard.cfg
  out=$data/sg2_camera_visp.xml
  cat > "$cfg" <<EOF
BoardSize_Width: $BW
BoardSize_Height: $BH
Square_Size: $sq
Calibrate_Pattern: CHESSBOARD
Input: $data/sg2_image-%d.png
Tempo: 0.1
EOF
  rm -f "$out"
  cd "$data"
  xvfb-run -a -s "-screen 0 1920x1080x24" bash -c "
    python3 '$HERE/xauto_click.py' 0.3 & C=\$!
    '$VISP/intrinsic/visp-calibrate-camera' '$cfg' --init-from-xml '$data/sg2_camera.xml' \
        --camera-name Camera --output '$out'; r=\$?
    kill \$C; exit \$r" > "$data/intrinsic_log.txt" 2>&1 || true
  grep -E "reprojection error|Global reprojection|failed|Unable" "$data/intrinsic_log.txt" || true
  [ -f "$out" ] || { echo "calibration failed, see $data/intrinsic_log.txt"; exit 1; }
  python3 - "$data/sg2_camera.xml" "$out" <<'EOF'
import sys, xml.etree.ElementTree as ET
def models(p):
    r = {}
    for cam in ET.parse(p).getroot().iter("camera"):
        m = cam.find("model")
        r[m.findtext("type")] = {k: float(m.findtext(k)) for k in ("px", "py", "u0", "v0", "kud") if m.find(k) is not None}
    return r
fac, vis = models(sys.argv[1]), models(sys.argv[2])
for t in ("perspectiveProjWithoutDistortion", "perspectiveProjWithDistortion"):
    print("\n" + t)
    for k in fac.get(t, {}):
        a, b = fac[t][k], vis.get(t, {}).get(k, float("nan"))
        rel = "" if k == "kud" else "  (%+.2f%%)" % (100 * (b - a) / a)
        print("  %-3s factory %10.4f   visp %10.4f%s" % (k, a, b, rel))
print("\nOK if: global reprojection error < ~0.5 px and px/py/u0/v0 within ~1-2 % of factory.")
EOF
  ;;
handeye)
  intr=${4:-$data/sg2_camera.xml}
  echo "intrinsics: $intr"
  cd "$data"
  rm -f sg2_pose_cPo_*.yaml
  if [ "$TARGET" = apriltag ]; then
    "$VISP/hand-eye/visp-compute-apriltag-poses" --tag-size "$sq" \
        --input "$data/sg2_image-%d.png" --intrinsic "$intr" --camera-name Camera \
        --output "$data/sg2_pose_cPo_%d.yaml" --no-interactive > "$data/cpo_log.txt" 2>&1 || true
  else
    "$VISP/hand-eye/visp-compute-chessboard-poses" -w "$BW" -h "$BH" --square-size "$sq" \
        --input "$data/sg2_image-%d.png" --intrinsic "$intr" --camera-name Camera \
        --output "$data/sg2_pose_cPo_%d.yaml" --no-interactive > "$data/cpo_log.txt" 2>&1 || true
  fi
  nc=$(ls sg2_pose_cPo_*.yaml 2>/dev/null | wc -l)
  echo "$TARGET poses: $nc / $n"
  [ "$nc" -gt 0 ] || { tail -20 "$data/cpo_log.txt"; exit 1; }
  "$VISP/hand-eye/visp-compute-eye-to-hand-calibration" --data-path "$data" \
      --rPe "sg2_pose_rPe_%d.yaml" --cPo "sg2_pose_cPo_%d.yaml" \
      --output-rPc "$data/sg2_rPc.yaml" --output-ePo "$data/sg2_ePo.yaml" > "$data/handeye_log.txt" 2>&1 || true
  grep -E "Distance (theta|d) between|Mean residual|failed|Check your" "$data/handeye_log.txt" || true
  [ -f "$data/sg2_rPc.yaml" ] || { echo "eye-to-hand failed, see $data/handeye_log.txt"; exit 1; }
  echo
  python3 "$HERE/rmc_to_urdf.py" "$data/sg2_rPc.yaml" | tee "$data/sg2_head_camera_extrinsic.txt"
  echo
  echo "OK if: mean residual rotation < ~1 deg and translation < ~5 mm; drop outlier poses and rerun."
  ;;
*)
  echo "unknown mode: $mode"; exit 1 ;;
esac
