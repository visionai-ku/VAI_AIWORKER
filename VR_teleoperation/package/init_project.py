from __future__ import annotations

import importlib
from pathlib import Path
import sys

THIS_FILE = Path(__file__).resolve()
PROJECT_DIR = THIS_FILE.parents[1]
PACKAGE_DIR = THIS_FILE.parent
REPO_ROOT = PROJECT_DIR
VENDOR_DIR = PROJECT_DIR / "vendor"

# A live kernel can otherwise keep using modules imported from the v1 tree.
# Do not unload potentially running controllers; require a fresh kernel instead.
for _module_name, _module in list(sys.modules.items()):
    if not any(_module_name == _root or _module_name.startswith(_root + ".")
               for _root in ("utils", "package", "ri_motion_v5_package")):
        continue
    _locations = list(getattr(_module, "__path__", ()))
    if getattr(_module, "__file__", None):
        _locations.append(_module.__file__)
    if any(not Path(_location).resolve().is_relative_to(PROJECT_DIR) for _location in _locations):
        raise RuntimeError(
            "An earlier project module is already loaded (%s). "
            "Restart this notebook's kernel before using VR_teleoperation." % _module_name)

for _path in [VENDOR_DIR, PROJECT_DIR, PACKAGE_DIR]:
    _path_str = str(_path)
    if _path_str in sys.path:
        sys.path.remove(_path_str)
    sys.path.insert(0, _path_str)

importlib.invalidate_caches()
sys.modules.pop("utils", None)

_utils = importlib.import_module("utils")
__all__ = list(getattr(_utils, "__all__", []))
for _name in __all__:
    globals()[_name] = getattr(_utils, _name)

print("\n[VR_teleoperation / ffw_sg2_vr_teleoperation] initialized.")
print(" - PROJECT_DIR: %s" % str(PROJECT_DIR))
