from __future__ import annotations

import importlib
from pathlib import Path
import sys

THIS_FILE = Path(__file__).resolve()
PROJECT_DIR = THIS_FILE.parents[1]
PACKAGE_DIR = THIS_FILE.parent
REPO_ROOT = PROJECT_DIR.parents[1]

for _path in [REPO_ROOT, PROJECT_DIR, PACKAGE_DIR]:
    _path_str = str(_path)
    if _path_str in sys.path:
        sys.path.remove(_path_str)
    sys.path.insert(0, _path_str)

importlib.invalidate_caches()
sys.modules.pop("utils", None)

_utils = importlib.import_module("utils")
for _name in getattr(_utils, "__all__", []):
    globals()[_name] = getattr(_utils, _name)

print("\n[ffw_sg2_vr_teleoperation] initialized.")
print(" - PROJECT_DIR: %s" % str(PROJECT_DIR))

