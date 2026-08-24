# `ri_motion_v5_package.traj_optimization`

Trajectory shaping, interpolation, smoothing, and utility helpers.

## Modules

- `traj_shaper.py`
  - One-dimensional trajectory shaping and remainder-style helpers
- `traj_utils.py`
  - Shared trajectory interpolation and smoothing utilities

## Public API Snapshot

- `traj_shaper.py`: `try_cvxpy_solver(...)`, limit/endpoint violation checks, `traj_1d_shaper(...)`, `traj_1d_remainer(...)`, and corresponding plotting helpers
- `traj_utils.py`: jerk/finite-difference matrix helpers and `get_interp_const_vel_traj_nd(...)`

## Import

```python
from ri_motion_v5_package.traj_optimization import *
```

## Notes

- Notebook examples live in `../../ri_motion_v5_notebook/31_traj_optimization/`
- Keep experiment-specific durations, limits, and plotting settings in notebooks
