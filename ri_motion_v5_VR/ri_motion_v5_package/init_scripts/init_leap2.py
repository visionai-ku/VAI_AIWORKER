from __future__ import annotations

from pathlib import Path
import sys

# Get directory where this script lives
THIS_FILE = Path(__file__).resolve()
PACKAGE_ROOT = THIS_FILE.parent.parent  # package/

# Set paths and import modules
sys.path.append(str(PACKAGE_ROOT / 'leap2')) # for 'leap2'
from leap2_handler import * # type: ignore

# Print
print ("[leap2] package initialized.")
print (" from leap2_handler import *")