# `ri_motion_v5_package.vader5`

VADER5 gamepad/input-device helpers.

## Modules

- `vader5_handler.py`
  - Higher-level VADER5 input handling and state mapping
- `vader5_utils.py`
  - Lower-level VADER5 utility functions

## Public API Snapshot

- `vader5_handler.py`: `Vader5Handler`
- `Vader5Handler`: optional hardware availability/status handling, motion-memory reset, state updates, raw/processed state access, and rumble support
- `vader5_utils.py`: currently contains no public helper functions

## Import

```python
from ri_motion_v5_package.vader5 import *
```

## Notes

- Notebook examples live in `../../ri_motion_v5_notebook/43_vader/`
- `Vader5Handler(required=False)` is the default behavior: missing `pygame` or missing joystick hardware is reported through `is_available()` and `get_status_msg()` while neutral inputs keep notebook loops running
- Use `Vader5Handler(required=True)` when a script must fail fast without controller input
- Keep robot-specific teleoperation mappings in notebooks or project-specific helpers until they are stable
