# `ri_motion_v5_package.dpp`

Determinantal point process helpers for sampling diverse subsets and trajectories.

## Modules

- `common.py`
  - Shared matrix, kernel, and probability utilities used by DPP variants
- `dpp.py`
  - Core DPP, k-DPP, iterative k-DPP, temporal iterative k-DPP, and MCMC-style routines
- `approx.py`
  - Approximate or lightweight helper routines for DPP-style workflows

## Public API Snapshot

- `common.py`: `grid_2d(...)`, `grid_2d_uniform(...)`, `add_colorbar(...)`, `gmm(...)`, `QualityMapClass`
- `dpp.py`: `kernel_se(...)`, `block_mtx(...)`, `inv_inc(...)`, `det_inc(...)`, `kdpp(...)`, `ikdpp(...)`, plus shared grid/probability helpers
- `approx.py`: `get_color_2d(...)`, `get_dpp_subset_from_K(...)`, `get_dpp_subset(...)`, `get_rand_subset(...)`

## Import

```python
from ri_motion_v5_package.dpp import *
```

## Notes

- Keep standalone DPP notebooks in `../../ri_motion_v5_notebook/32_dpp_sampling/`
- Prefer adding reusable algorithm code here instead of duplicating it across notebooks
