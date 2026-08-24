# `ri_motion_v5_package.gaussian_process`

Gaussian process utilities and kernel functions used by sampling and trajectory experiments.

## Modules

- `kernels.py`
  - Kernel functions for Gaussian process and sampling workflows
- `gp_utils.py`
  - Utility functions for GP-style prediction, conditioning, and visualization support

## Public API Snapshot

- `kernels.py`: `brownian_kernel(...)`, `brownian_bridge_kernel(...)`, `rbf_kernel(...)`, `rbf_bridge_kernel(...)`, `linear_kernel(...)`, `polynomial_kernel(...)`, `matern_kernel(...)`, `periodic_kernel(...)`, `rational_quadratic_kernel(...)`
- `gp_utils.py`: `get_K_chol(...)`, `get_schur_K_chol(...)`, `safe_matmul(...)`, `sample_trajs(...)`, `sample_trajs_multidim(...)`

## Import

```python
from ri_motion_v5_package.gaussian_process import *
```

## Notes

- Keep notebook-specific hyperparameters in notebooks
- Move only reusable GP logic into this package folder
