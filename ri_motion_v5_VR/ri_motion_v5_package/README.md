# `ri_motion_v5_package`

Reusable Python package code for the `ri_motion_v5` workspace.

## Workspace Layout

This repository is expected to live next to the notebook and asset repositories:

```text
ri_motion_v5/
├── project/
├── ri_motion_v5_asset/
├── ri_motion_v5_notebook/
└── ri_motion_v5_package/
```

The package is installed from this repository and then used by notebooks in
`../ri_motion_v5_notebook` and project-specific workflows under `../project`.
Those notebooks and projects usually load simulation assets from
`../ri_motion_v5_asset`.

## Repository Role

Use this repository for reusable Python modules:

- MuJoCo parsing, simulation, rendering, drawing, and IK helpers
- process-separated MuJoCo rendering helpers for notebook workflows
- optional nanobind C++ extension build/import helpers for MuJoCo hot paths
- kinematics, transforms, rotations, and chain utilities
- Qt widgets and interactive viewer tools
- motion capture and motion retargeting helpers
- trajectory sampling and optimization utilities
- filtering, visualization, timers, and general utilities
- optional hardware/model wrappers such as Leap Motion, VADER, HF models, and
  LLM clients

Notebook experiments belong in `../ri_motion_v5_notebook`. Shared robot,
object, camera, and scene assets belong in `../ri_motion_v5_asset`.

## Style Reference

`SOUL.md` is the canonical coding and notebook style reference for this
workspace. It is stored in this package repository so the rules are visible in
git and can be pushed with package changes. The workspace root may keep a
symlink to this file for convenience, but this copy is the source of truth.

## Sibling Repositories

- `ri_motion_v5_notebook`: runnable notebooks and interactive experiments
- `ri_motion_v5_asset`: shared robots, MJCF/XML files, meshes, objects, cameras,
  fixtures, and floor models
- `project`: project-specific notebooks, scenes, and local workflows

## Setup

Install this package in editable mode from the workspace:

```bash
cd ri_motion_v5_package
conda create -n ri_motion_v5_env python=3.10 pip
conda activate ri_motion_v5_env
pip install -r requirements.txt
pip install -e .
```

Use the same `ri_motion_v5_env` for notebooks and terminal checks.

## Contents

```text
dpp/                 DPP sampling helpers
gaussian_process/    Gaussian process helpers
hf_models/           local model wrappers
init_scripts/        notebook/IPython bootstrap scripts
kinematics/          transforms, rotations, and kinematic chains
leap2/               Leap Motion handlers
llm_clients/         API client helpers
mocap/               motion-capture conversion utilities
motion_retarget/     retargeting utilities
mujoco_sim/          MuJoCo parser, viewer, IK, drawing, and XML helpers
qt/                  Qt widgets
traj_optimization/   trajectory optimization helpers
utility/             timers, filters, visualization, sampling, and misc tools
vader5/              VADER handlers
```

## Native And Render Helpers

`mujoco_sim` now contains two optional acceleration/interaction layers:

- Native C++ bind helpers:
  - C++ source and headers live under `mujoco_sim/cpp/`
  - compiled extension modules live under `mujoco_sim/so/`
  - `build_native(...)` and `clean_native(...)` manage notebook-selected build
    inputs
  - `get_native_backend(..., module_name=...)` imports the compiled backend as
    an explicit `be` handle for notebook examples
- Process-separated rendering:
  - `MuJoCoRenderProcess` starts a normal Python subprocess with its own
    `MuJoCoParser`, `MjModel`, `MjData`, viewer, and render loop
  - the notebook process sends JSON-safe state, overlay, and draw commands
    through ZMQ
  - this is intended for notebook responsiveness experiments, not as a
    replacement for ordinary single-process viewer notebooks

## Maintainer

Sungjoon Choi  
Korea University  
sungjoon-choi at korea dot ac dot kr
