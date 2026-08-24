"""
init_qt.py
"""
from __future__ import annotations

import sys
from typing import Optional

from ri_motion_v5_package.qt.qt_widgets import * # type: ignore

def get_qt_app(argv: Optional[list[str]] = None,style: str = "Fusion"):
    """
    Get or create a QApplication instance (safe to call repeatedly).

    Parameters:
        argv (Optional[list[str]]): Input argv. Defaults to `None`.
        style (str): Input style. Defaults to `'Fusion'`.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        if argv is None:
            argv = sys.argv
        app = QApplication(argv)

    if style:
        try:
            app.setStyle(style)
        except Exception:
            pass

    return app

def update_qt():
    """
    Process pending Qt events in the current Python process.

    Parameters:
        None.

    Returns:
        Any: QApplication instance when available, otherwise None.
    """
    try:
        app = get_qt_app()
        app.processEvents()
        return app
    except Exception:
        return None

try:
    get_qt_app(style="Fusion")
except Exception:
    pass

print("[qt] package initialized.")
