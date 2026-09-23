#!/usr/bin/env bash
# Ubuntu setup for ri_motion_v5_VR and VR_teleoperation notebooks.
#
#   bash setup_ubuntu_env.sh [env_name]      (default: ri_motion_v5_py312)
#
# 1. Creates a Python 3.12 conda env (lerobot 0.6.x requires >=3.12) and installs
#    CUDA torch + the notebook requirements. The older Python 3.10 / lerobot 0.4.4 env
#    (ri_motion_v5_env) is left alone; this one gets a different name.
# 2. Adds conda activate/deactivate hooks that drop the system library
#    directories (/usr/lib/x86_64-linux-gnu, /lib/x86_64-linux-gnu) from
#    LD_LIBRARY_PATH. When they are listed there (e.g. by ~/.bashrc for CUDA or
#    ROS), PyQt5 loads the system Qt 5.12 instead of its bundled Qt 5.15 and
#    aborts with `Could not load the Qt platform plugin "xcb"`.
# 3. Registers a Jupyter kernel that uses the same cleaned LD_LIBRARY_PATH.
set -euo pipefail

ENV_NAME="${1:-ri_motion_v5_py312}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

eval "$(conda shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -y -n "$ENV_NAME" python=3.12 pip
fi
conda activate "$ENV_NAME"
# Same torch build as the training env (lerobot_act), so checkpoints load identically.
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url "$TORCH_INDEX_URL"
python -m pip install -r "$REPO_DIR/VR_teleoperation/requirements-notebook.txt"
# lerobot pulls opencv-python-headless, which overwrites cv2/ with a build that has no GUI
# (cv2.imshow fails). Put the GUI build back last; see requirements.txt.
python -m pip install --force-reinstall --no-deps opencv-python==4.12.0.88

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/ri_motion_ld_library_path.sh" <<'EOF'
export _RI_MOTION_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
export LD_LIBRARY_PATH="$(printf '%s' "${LD_LIBRARY_PATH-}" | tr ':' '\n' \
    | grep -vxE '/(usr/)?lib/x86_64-linux-gnu/?' | grep -v '^$' | paste -sd: -)"
EOF
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/ri_motion_ld_library_path.sh" <<'EOF'
export LD_LIBRARY_PATH="${_RI_MOTION_OLD_LD_LIBRARY_PATH-}"
unset _RI_MOTION_OLD_LD_LIBRARY_PATH
EOF
# Apply the hook to this shell as well.
source "$CONDA_PREFIX/etc/conda/activate.d/ri_motion_ld_library_path.sh"

python -m ipykernel install --user --name "$ENV_NAME" --display-name "$ENV_NAME" \
    --env LD_LIBRARY_PATH "$LD_LIBRARY_PATH"

echo
echo "Done. Next:"
echo "  conda activate $ENV_NAME"
echo "  jupyter lab    # select the [$ENV_NAME] kernel"
