# `ri_motion_v5_package.motion_retarget`

Motion retargeting helpers for JOI, SEG, CMU, and KAPEX-style workflows.

## Modules

- `joi_seg.py`
  - JOI and SEG retargeting utilities
- `joi_cmu.py`
  - JOI and CMU motion-capture retargeting utilities
- `joi_kapex.py`
  - JOI and KAPEX retargeting utilities

## Public API Snapshot

- `joi_seg.py`: `get_chain_pose(...)`, `get_T_joi_of_chain_seg(...)`
- `joi_cmu.py`: `get_chain_pose(...)`, `get_T_joi_of_chain_cmu(...)`
- `joi_kapex.py`: `get_joi_site_name_of_kapex(...)`, `get_T_joi_of_kapex(...)`

## Import

```python
from ri_motion_v5_package.motion_retarget import *
```

## Notes

- Notebook examples live in `../../ri_motion_v5_notebook/11_motion_retarget/`
- Keep dataset-specific paths and experiment settings in notebooks or project folders
