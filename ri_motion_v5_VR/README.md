# ri_motion_v5_junha

Integrated `ri_motion_v5` workspace for robotics simulation, notebook
experiments, teleoperation projects, shared assets, and reusable Python code.

This repository keeps project-specific work in this repository and tracks the
shared `ri_motion_v5` folders as Git submodules. Clone with submodules enabled
to get the package, notebooks, and assets at the versions used by this
workspace.

## Layout

```text
ri_motion_v5_junha/
├── ri_motion_v5_package/    reusable Python package submodule
├── ri_motion_v5_notebook/   general notebooks/examples submodule
├── ri_motion_v5_asset/      shared asset submodule
├── project/                 project-specific notebooks, assets, and scripts
├── bvh/                     motion-capture datasets
└── code_convention.md       local coding convention notes
```

Important project folders:

- `project/T-FR_various_teleoperation_system`: SH5 teleoperation prototypes,
  including Leap Motion, MANUS, MetaQuest3, VADER, slider, simulation, real
  robot publishing, and render notebooks.
- `project/RLWRLD_2026_BMW_TMP`: BMW demo integration notebooks and assets.
- `project/KT_2026_FFWSH5_DEMO`: FFW SH5 demo notebooks.
- `project/force_aware_manipulation`: force-aware manipulation experiments.
- `project/safe_constraint_patching`: safety/constraint patching experiments.

## Clone

Clone this repository with submodules:

```bash
git clone --recurse-submodules https://github.com/Junha02/ri_motion_junha.git
cd ri_motion_junha
```

If the repository was already cloned without submodules:

```bash
git submodule update --init --recursive
```

Update the shared `ri_motion_v5` submodules later with:

```bash
git submodule update --remote ri_motion_v5_package ri_motion_v5_asset ri_motion_v5_notebook
git add ri_motion_v5_package ri_motion_v5_asset ri_motion_v5_notebook
git commit -m "Update ri_motion submodules"
```

## Setup

Create an environment and install the local package in editable mode:

```bash
cd /path/to/ri_motion_junha

conda create -n ri_motion_v5_env python=3.10 pip
conda activate ri_motion_v5_env

pip install -r ri_motion_v5_package/requirements.txt
pip install -e ri_motion_v5_package
```

Then start Jupyter from the workspace:

```bash
jupyter lab
```

Use `ri_motion_v5_env` as the notebook kernel.

### Ubuntu

The `ffw_sg2_vr_teleoperation` notebooks only need the smaller requirement set
used by `VR_teleoperation`. From the repository root (`VAI_AIWORKER`):

```bash
bash setup_ubuntu_env.sh          # creates ri_motion_v5_py312 (Python 3.12) + Jupyter kernel
conda activate ri_motion_v5_py312
jupyter lab
```

- The script removes `/usr/lib/x86_64-linux-gnu` from `LD_LIBRARY_PATH` inside
  the env and kernel. If it is listed there (e.g. by `~/.bashrc` for CUDA/ROS),
  PyQt5 loads the system Qt 5.12 and aborts with
  `Could not load the Qt platform plugin "xcb"`.
- `build_sg2_collision_aware_ik_backend(...)` compiles the native IK with `g++`
  (`build-essential`) into `mujoco_sim/so/*-linux-gnu.so`. The file name carries the
  Python ABI (`cpython-310`, `cpython-312`, ...), so a new env just builds its own copy
  next to the old one on first use; pass `clean=True` to force a rebuild.

## Notebook Usage

Most notebooks use relative paths to the workspace folders. Keep the repository
layout unchanged:

```text
project/
ri_motion_v5_asset/
ri_motion_v5_notebook/
ri_motion_v5_package/
```

Project notebooks usually bootstrap local project modules with a cell like:

```python
%run ../package/init_project.py
```

General notebooks import reusable modules from `ri_motion_v5_package` and load
shared robot/object assets from `ri_motion_v5_asset`.

## Teleoperation Notes

The T-FR SH5 workflows under `project/T-FR_various_teleoperation_system` include
simulation and real-robot notebooks. Real notebooks typically follow this split:

- teleop or homing notebooks publish robot qpos commands to port `5561`
- render preview messages are published to port `5556`
- robot state is received from port `5560`

The Leap Motion + MANUS SH5 notebooks include wrist attach and wrist-distance
safety logic. Arm IK starts only after the Leap wrist target attaches, and real
robot publishing is gated by the command toggle, attach state, and wrist jump
rejection state.

## Repository Notes

- `ri_motion_v5_package`, `ri_motion_v5_asset`, and `ri_motion_v5_notebook` are
  Git submodules.
- Project folders under `project/` are tracked directly by this repository.
- Large generated outputs, local environments, caches, and runtime artifacts
  should be kept out of future commits when possible.
- See the folder-level README files for more focused details:
  - `ri_motion_v5_package/README.md`
  - `ri_motion_v5_notebook/README.md`
  - `ri_motion_v5_asset/README.md`
  - `project/T-FR_various_teleoperation_system/README.md`
