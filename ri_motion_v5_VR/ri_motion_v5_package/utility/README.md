# `ri_motion_v5_package.utility`

General-purpose utilities, filters, and visualization helpers used across the workspace.

## Modules

- `utils.py`
  - Timing, indexing, filesystem, sampling, and small helper functions
- `filters.py`
  - Filtering utilities for signals and state streams
- `viz.py`
  - Matplotlib and image visualization helpers, color utilities, and print helpers

## Public API Snapshot

- `utils.py`: indexing/list helpers, XML indentation, monitor/render wait
  helpers, notebook helpers such as `sleep(...)` and `close_objects(...)`,
  scaling/statistics helpers, trajectory sampling helpers, camera ray helper
  `xy2xyz(...)`, point sampling, repo-tree printing, `TicToc`, `TicTocs`,
  `ZMQPubSub`, and `SimpleTimer`
- `filters.py`: `VectorizedKalmanFilters`, `VectorizedKalmanFiltersFast`
- `viz.py`: `get_colors(...)`, image/depth helpers, `imshow(...)`, `play_frames(...)`, colored print helpers, and `meters2xyz(...)`

## Import

```python
from ri_motion_v5_package.utility import *
```

## Notes

- Keep broadly reusable helpers here
- Prefer domain-specific helpers in their domain package instead of expanding this folder unnecessarily
- `SimpleTimer.get_lag_info()` reports schedule lag as missed ticks, milliseconds, and seconds; pair it with `env.plot_tmr_lag(tmr)` to overlay both lag and the timer run time instead of creating notebook-local lag helpers
- `print_repo_tree_extension(...)` reports `.py`, `.ipynb`, `.cpp`, and `.hpp`
  counts by default. When line counting is enabled, it prints comma-formatted
  line totals including `Total .cpp+.hpp lines`.
