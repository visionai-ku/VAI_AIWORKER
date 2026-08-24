# `ri_motion_v5_package.mocap`

Motion-capture parsing and chain-conversion helpers.

## Modules

- `bvh_to_chain.py`
  - BVH parsing and conversion helpers for kinematic-chain workflows

## Public API Snapshot

- `bvh_to_chain.py`: BVH parser classes `Node`, `BvhReader`, `Joint`, `ReadBVH`, `Skeleton`
- `bvh_to_chain.py`: conversion helpers `process_bvhfile(...)`, `get_skeleton_from_bvh(...)`, `get_chains_from_skeleton(...)`, `get_chains_translated(...)`, `get_chains_from_bvh(...)`

## Import

```python
from ri_motion_v5_package.mocap import *
```

## Notes

- Notebook examples live in `../../ri_motion_v5_notebook/10_motion_capture/`
- Keep large motion data under workspace asset/data folders rather than inside this package
