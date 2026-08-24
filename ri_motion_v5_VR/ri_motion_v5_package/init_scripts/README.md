# `ri_motion_v5_package.init_scripts`

Notebook bootstrap scripts for the interactive `ri_motion_v5` environment.

## Modules

- `init_ipython_setup.py`
  - IPython/Jupyter setup, autoreload-style workflow support, plotting defaults, and Qt event-loop bootstrap
- `init_qt.py`
  - Qt application/widget setup for notebook-driven tools
- `init_leap2.py`
  - Leap Motion binding setup for notebooks that use `leap2`

## Public API Snapshot

- `init_qt.py`: `get_qt_app(...)`
- `init_ipython_setup.py`: notebook/IPython side-effect setup for imports, plotting defaults, and Qt event-loop bootstrap
- `init_leap2.py`: Leap Motion import-path/bootstrap setup for notebooks

## Notebook Contract

Interactive notebooks commonly use:

```python
from ri_motion_v5_package.init_scripts.init_ipython_setup import *
from ri_motion_v5_package.init_scripts.init_qt import *
```

Keep `%gui qt` behavior active inside IPython/Jupyter sessions. Do not force `matplotlib.use("Agg")` in notebook sessions unless the behavior is gated for non-interactive execution.

## Notes

- Changes here can affect MuJoCo viewer shutdown, Qt widgets, and render-window behavior
- Re-test at least one standard interactive notebook after editing these scripts
