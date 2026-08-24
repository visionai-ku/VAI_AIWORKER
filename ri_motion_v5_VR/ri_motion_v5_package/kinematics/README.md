# `ri_motion_v5_package.kinematics`

Spatial algebra, transform, and kinematic-chain utilities.

## Modules

- `transforms.py`
  - Rigid transforms, rotations, quaternions, RPY conversions, adjoints, and screw helpers
- `screw_chain.py`
  - Screw-axis based serial-chain utilities
- `numpy_chain.py`
  - Numpy-based kinematic chain utilities for motion-capture and retargeting workflows

## Public API Snapshot

- `transforms.py`: transform extractors/builders such as `t2p(...)`, `t2r(...)`, `pr2t(...)`, `rpy2r(...)`, `r2rpy(...)`, quaternion helpers, adjoint/frame-view helpers, screw helpers, and vector/plane helpers
- `screw_chain.py`: `ScrewChain` with screw-axis registration, forward kinematics, and MuJoCo plotting helpers
- `numpy_chain.py`: `NumpyChain` with joint lookup, tree traversal, FK/update, plotting, and transform access helpers

## Import

```python
from ri_motion_v5_package.kinematics import *
```

## Notes

- Prefer quantity-first names such as `p_cam`, `R_cam`, `T_cam`, and `T_hand`
- Keep reusable frame math here rather than duplicating helper functions in notebooks
