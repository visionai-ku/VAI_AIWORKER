"""
qt_widgets.py
"""

from __future__ import annotations

import os
import re
import sys
import numbers
import time
from typing import Any, Optional, Sequence, Union, List, Tuple

import numpy as np
import pyqtgraph as pg
import matplotlib.pyplot as plt

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor,QFont
from PyQt5.QtWidgets import (
    QWidget,
    QLabel,
    QSlider,
    QScrollArea,
    QVBoxLayout,
    QGridLayout,
    QRadioButton,
    QButtonGroup,
    QSizePolicy,
)

# -----------------------------------------------------------------------------
# Shared helpers (screen-aware geometry)
# -----------------------------------------------------------------------------
def use_pyqt_platform_plugins() -> None:
    """
    Drop Qt plugin paths that `opencv-python` points at its own bundled Qt.

    On Linux, `import cv2` sets `QT_QPA_PLATFORM_PLUGIN_PATH` to `cv2/qt/plugins`.
    A PyQt5 `QApplication` created afterwards then loads cv2's incompatible
    `xcb` plugin and aborts. Call this right before creating a `QApplication`.

    Parameters:
        None.

    Returns:
        None.
    """
    if not sys.platform.startswith("linux"):
        return
    for key in ("QT_QPA_PLATFORM_PLUGIN_PATH","QT_QPA_FONTDIR"):
        value = os.environ.get(key,"")
        if "cv2" in value.split(os.sep):
            os.environ.pop(key)

def get_monitor_size_fallback() -> Tuple[int, int]:
    """
    Returns (screen_w, screen_h) via pyautogui.size().

    Parameters:
        None.

    Returns:
        Tuple[int, int]: Requested value computed from the current inputs.
    """
    try:
        import pyautogui  # type: ignore
        w, h = pyautogui.size()
        return int(w), int(h)
    except Exception:
        return 1920, 1080

def resolve_screen_value(value: Union[int, float], screen_size: int) -> int:
    """
    If numeric and 0 < value <= 1, interpret it as a ratio of screen_size.

    Parameters:
        value (Union[int, float]): Input value.
        screen_size (int): Input screen size.

    Returns:
        int: Result computed from the current inputs.
    """
    if isinstance(value, (numbers.Number, np.floating)):
        v = float(value)
        if 0.0 < v <= 1.0:
            return int(round(v * float(screen_size)))
        return int(round(v))
    return int(round(float(value)))

def is_number(x: Any) -> bool:
    """
    Return whether a value can be converted to float.

    Parameters:
        x (Any): Input x.

    Returns:
        bool: Boolean result for the requested condition.
    """
    try:
        float(x)
        return True
    except Exception:
        return False

def to_255(v: float) -> int:
    """
    Convert a 0..1 or 0..255 channel value to uint8 range.

    Parameters:
        v (float): Input v.

    Returns:
        int: Result computed from the current inputs.
    """
    vv = float(v)
    if 0.0 <= vv <= 1.0:
        vv *= 255.0
    return max(0,min(255,int(round(vv))))

def rgb_hex(r: int,g: int,b: int) -> str:
    """
    Format RGB channels as an opaque #RRGGBB string.

    Parameters:
        r (int): Input r.
        g (int): Input g.
        b (int): Input b.

    Returns:
        str: Result computed from the current inputs.
    """
    return "#%02X%02X%02X"%(r,g,b)

def force_opaque_from_hex(s: str) -> Optional[str]:
    """
    Convert #RRGGBB or #AARRGGBB into opaque #RRGGBB.

    Parameters:
        s (str): Input s.

    Returns:
        Optional[str]: Result computed from the current inputs.
    """
    s2 = s.strip()
    s2 = re.sub(r"[\s\-]","",s2)
    m6 = re.fullmatch(r"#?([0-9A-Fa-f]{6})",s2)
    if m6:
        return "#" + m6.group(1).upper()
    m8 = re.fullmatch(r"#?([0-9A-Fa-f]{8})",s2)
    if m8:
        h = m8.group(1).upper()
        return "#" + h[2:]
    return None

def qcolor_to_hex_opaque(s: str) -> Optional[str]:
    """
    Resolve a QColor-compatible string to opaque #RRGGBB.

    Parameters:
        s (str): Input s.

    Returns:
        Optional[str]: Result computed from the current inputs.
    """
    q = QColor(s)
    if q.isValid():
        return rgb_hex(q.red(),q.green(),q.blue())
    return None

# -----------------------------------------------------------------------------
# MultiSliderQtWidget
# -----------------------------------------------------------------------------
class MultiSliderQtWidget(QWidget):
    """
    Multi sliders using Qt.

    Usage:
        obj = MultiSliderQtWidget(...)
    """

    def __init__(
            self,
            x_offset: Union[int, float] = 0,
            y_offset: Union[int, float] = 0,
            window_width: Union[int, float] = 500,
            window_height: Union[int, float] = 300,
            title: str = "Sliders",
            n_slider: Optional[int] = None,
            label_width: Optional[int] = None,
            slider_width: Optional[int] = None,
            label_texts: Optional[Sequence[Any]] = None,
            label_colors: Optional[Sequence[Any]] = None,
            slider_colors: Optional[Sequence[Any]] = None,
            slider_mins: Optional[Sequence[float]] = None,
            slider_maxs: Optional[Sequence[float]] = None,
            slider_steps: int = 1000001,
            slider_vals: Optional[Sequence[float]] = None,
            resolutions: Optional[Sequence[float]] = None,
            fontsize: int = 10,
            verbose: bool = False,
        ):
        """
        Initialize MultiSliderQtWidget.

        Parameters:
            x_offset (Union[int, float]): Input x offset. Defaults to `0`.
            y_offset (Union[int, float]): Input y offset. Defaults to `0`.
            window_width (Union[int, float]): Input window width. Defaults to `500`.
            window_height (Union[int, float]): Input window height. Defaults to `300`.
            title (str): Input title. Defaults to `'Sliders'`.
            n_slider (Optional[int]): Input n slider. Defaults to `None`.
            label_width (Optional[int]): Input label width. Defaults to `None`.
            slider_width (Optional[int]): Input slider width. Defaults to `None`.
            label_texts (Optional[Sequence[Any]]): Input label texts. Defaults to
                                                   `None`.
            label_colors (Optional[Sequence[Any]]): Input label colors. Defaults to
                                                    `None`.
            slider_colors (Optional[Sequence[Any]]): Input slider colors. Defaults to
                                                     `None`.
            slider_mins (Optional[Sequence[float]]): Input slider mins. Defaults to
                                                     `None`.
            slider_maxs (Optional[Sequence[float]]): Input slider maxs. Defaults to
                                                     `None`.
            slider_steps (int): Input slider steps. Defaults to `1000001`.
            slider_vals (Optional[Sequence[float]]): Input slider vals. Defaults to
                                                     `None`.
            resolutions (Optional[Sequence[float]]): Input resolutions. Defaults to
                                                     `None`.
            fontsize (int): Input fontsize. Defaults to `10`.
            verbose (bool): Input verbose. Defaults to `False`.

        Returns:
            None: Initializes the instance in place.
        """
        super().__init__()

        # Infer n_slider
        if n_slider is None:
            if label_texts is None:
                raise ValueError("Either n_slider or label_texts must be provided.")
            n_slider = len(label_texts)

        self.title = str(title)
        self.n_slider = int(n_slider)

        self.label_width = None if label_width is None else int(label_width)
        self.slider_width = None if slider_width is None else int(slider_width)
        self.fontsize = int(fontsize)
        self.verbose = bool(verbose)

        # Texts
        if label_texts is None:
            label_texts = list(np.arange(0, self.n_slider, 1))
        if len(label_texts) != self.n_slider:
            raise ValueError(f"label_texts length ({len(label_texts)}) != n_slider ({self.n_slider}).")
        self.label_texts = list(label_texts)

        # Color map (short keys)
        self.color_map = {
            "r": "red",
            "g": "green",
            "b": "blue",
            "c": "cyan",
            "m": "magenta",
            "y": "yellow",
            "k": "black",
            "w": "white",
        }

        # Colors
        if label_colors is None:
            label_colors = ["w"] * self.n_slider
        if len(label_colors) != self.n_slider:
            raise ValueError(f"label_colors length ({len(label_colors)}) != n_slider ({self.n_slider}).")
        self.label_colors = list(label_colors)

        # Slider colors
        if slider_colors is not None and len(slider_colors) != self.n_slider:
            raise ValueError(f"slider_colors length ({len(slider_colors)}) != n_slider ({self.n_slider}).")
        self.slider_colors = None if slider_colors is None else list(slider_colors)

        # Slider ranges / values
        if slider_mins is None:
            slider_mins = 0.0 * np.ones(self.n_slider)
        if slider_maxs is None:
            slider_maxs = 100.0 * np.ones(self.n_slider)
        if slider_vals is None:
            slider_vals = 0.0 * np.ones(self.n_slider)

        self.slider_mins=np.asarray(slider_mins,dtype=float).copy()
        self.slider_maxs=np.asarray(slider_maxs,dtype=float).copy()
        self.slider_vals=np.asarray(slider_vals,dtype=float).copy()

        if self.slider_mins.shape[0] != self.n_slider:
            raise ValueError("slider_mins length mismatch with n_slider.")
        if self.slider_maxs.shape[0] != self.n_slider:
            raise ValueError("slider_maxs length mismatch with n_slider.")
        if self.slider_vals.shape[0] != self.n_slider:
            raise ValueError("slider_vals length mismatch with n_slider.")

        self.slider_steps = int(slider_steps)
        if self.slider_steps <= 0:
            raise ValueError("slider_steps must be positive.")

        # Resolution
        if resolutions is None:
            resolutions = (self.slider_maxs - self.slider_mins) / float(self.slider_steps)
            resolutions = np.maximum(resolutions, 1e-9)
        self.resolutions = np.asarray(resolutions, dtype=float)
        if self.resolutions.shape[0] != self.n_slider:
            raise ValueError("resolutions length mismatch with n_slider.")
        self.resolutions = np.maximum(self.resolutions, 1e-12)

        # Geometry (screen-aware)
        screen_w, screen_h = get_monitor_size_fallback()
        self.x_offset = resolve_screen_value(x_offset, screen_w)
        self.y_offset = resolve_screen_value(y_offset, screen_h)
        self.window_width = resolve_screen_value(window_width, screen_w)
        self.window_height = resolve_screen_value(window_height, screen_h)

        # Buffers
        self.sliders: List[QSlider] = []
        self.labels: List[QLabel] = []

        # UI
        self.init_ui()

        if self.verbose:
            print(f"Initialized MultiSliderQtWidget with {self.n_slider} sliders.")

    # -------------------------
    # Color utilities
    # -------------------------
    def resolve_color(self, key: Any) -> str:
        """
        Resolve a color key to an opaque '#RRGGBB' string whenever possible.

        Parameters:
            key (Any): Input key.

        Returns:
            str: Result computed from the current inputs.
        """
        # Sequence of numbers (R,G,B[,A])
        if isinstance(key, (list, tuple, np.ndarray)):
            try:
                vals = list(key.tolist()) if hasattr(key, "tolist") else list(key)
            except Exception:
                vals = []
            if len(vals) >= 3 and all(is_number(v) for v in vals[:3]):
                r, g, b = to_255(vals[0]), to_255(vals[1]), to_255(vals[2])
                return rgb_hex(r,g,b)

        # Scalar number -> grayscale
        if is_number(key) and not isinstance(key, str):
            v = to_255(float(key))
            return rgb_hex(v,v,v)

        # String
        if isinstance(key, str):
            k_raw = key.strip()
            k_lower = k_raw.lower()

            if k_lower in self.color_map:
                mapped = str(self.color_map[k_lower]).strip()
                hex_ok = force_opaque_from_hex(mapped)
                if hex_ok is not None:
                    return hex_ok
                q_ok = qcolor_to_hex_opaque(mapped)
                if q_ok is not None:
                    return q_ok
                return mapped

            hex_ok = force_opaque_from_hex(k_raw)
            if hex_ok is not None:
                return hex_ok

            # keep original behavior: strip spaces/hyphens before Qt parsing
            k2 = re.sub(r"[\s\-]", "", k_raw)
            q_ok = qcolor_to_hex_opaque(k2)
            if q_ok is not None:
                return q_ok

            return k_raw

        return str(key)

    # -------------------------
    # UI
    # -------------------------
    def init_ui(self) -> None:
        """
        Initialize ui.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        main_layout = QVBoxLayout(self)

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)

        scroll_widget = QWidget()
        scroll_layout = QGridLayout(scroll_widget)

        # Make slider column expand by default, label column stay minimal
        scroll_layout.setColumnStretch(0, 0)
        scroll_layout.setColumnStretch(1, 1)

        for s_idx in range(self.n_slider):
            # Slider
            slider = QSlider(Qt.Horizontal, scroll_widget)

            if self.slider_width is not None:
                slider.setFixedWidth(self.slider_width)
            else:
                slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

            resolution = float(self.resolutions[s_idx])
            slider_min_f = float(self.slider_mins[s_idx])
            slider_max_f = float(self.slider_maxs[s_idx])
            slider_val_f = float(self.slider_vals[s_idx])

            slider.setMinimum(0)
            slider.setMaximum(
                self.float_to_int(
                    val_float=slider_max_f,
                    min_val_float=slider_min_f,
                    resolution=resolution,
                )
            )
            slider.setValue(
                self.float_to_int(
                    val_float=slider_val_f,
                    min_val_float=slider_min_f,
                    resolution=resolution,
                )
            )
            slider.setSingleStep(1)

            slider.valueChanged.connect(lambda value, idx=s_idx: self.cb_value_changed(idx, value))

            if self.slider_colors is not None:
                slider_color = self.resolve_color(self.slider_colors[s_idx])
                slider.setStyleSheet(self.get_slider_stylesheet(slider_color))

            self.sliders.append(slider)

            # Label
            label_text = self.label_texts[s_idx]
            label = QLabel(f"{label_text}:{slider_val_f:.2f}", scroll_widget)

            if self.label_width is not None:
                label.setFixedWidth(self.label_width)
            else:
                label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

            label_color = self.resolve_color(self.label_colors[s_idx])
            label.setStyleSheet(f"color: {label_color}; font-size: {self.fontsize}px;")

            self.labels.append(label)

            # Layout
            scroll_layout.addWidget(label, s_idx, 0)
            scroll_layout.addWidget(slider, s_idx, 1)

        # After creating all labels/sliders, freeze label width once if label_width is None
        if self.label_width is None:
            # Ensure Qt has computed size hints
            self.repaint()

            w_max=0
            for lb in self.labels:
                w_max=max(w_max,lb.sizeHint().width())

            # small margin
            w_fix=int(w_max+6)

            for lb in self.labels:
                lb.setFixedWidth(w_fix)
                lb.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed)

        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)

        main_layout.addWidget(scroll_area)

        self.setLayout(main_layout)
        self.setWindowTitle(self.title)
        self.setGeometry(self.x_offset, self.y_offset, self.window_width, self.window_height)
        self.show()

    def show(self) -> None:
        """
        Show the slider window and request foreground focus.

        Parameters:
            None.

        Returns:
            None: Shows the widget in place.

        Notes:
            `raise_()` and `activateWindow()` are focus requests to the window
            manager. macOS or other desktop environments may ignore them in
            some cases, but keeping them here lets notebooks call only
            `sliders.show()`.
        """
        super().show()
        self.raise_()
        self.activateWindow()

    @staticmethod
    def get_slider_stylesheet(color: str) -> str:
        """
        Return a compact horizontal QSlider stylesheet using the given color.

        Parameters:
            color (str): Input color.

        Returns:
            str: Requested value computed from the current inputs.
        """
        return """
            QSlider::groove:horizontal {
                border: 1px solid #555555;
                height: 6px;
                background: #333333;
                margin: 2px 0;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: %s;
                border: 1px solid #555555;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: %s;
                border: 1px solid #222222;
                width: 12px;
                margin: -5px 0;
                border-radius: 6px;
            }
        """%(color,color)

    # -------------------------
    # Public helpers
    # -------------------------
    def reset(self, n_reset: int = 1) -> None:
        """
        Reset internal state.

        Parameters:
            n_reset (int): Input n reset. Defaults to `1`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        for _ in range(int(n_reset)):
            self.resize(self.width(), self.height())
            self.repaint()

    def cb_value_changed(self, idx: int, value: int) -> None:
        """
        Handle a slider value-change callback.

        Parameters:
            idx (int): Input idx.
            value (int): Input value.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        value_float = self.int_to_float(
            val_int=value,
            min_val_float=float(self.slider_mins[idx]),
            resolution=float(self.resolutions[idx]),
        )
        self.slider_vals[idx] = value_float
        self.labels[idx].setText(f"{self.label_texts[idx]}:{value_float:.2f}")

        if self.verbose:
            print(f"slider index:[{idx}] value:[{value}] value_float:[{value_float:.6f}]")

    @staticmethod
    def float_to_int(val_float: float, min_val_float: float, resolution: float) -> int:
        """
        Convert a float slider value to an integer tick.

        Parameters:
            val_float (float): Input val float.
            min_val_float (float): Input min val float.
            resolution (float): Input resolution.

        Returns:
            int: Result computed from the current inputs.
        """
        return int(np.round((float(val_float) - float(min_val_float)) / float(resolution)))

    @staticmethod
    def int_to_float(val_int: int, min_val_float: float, resolution: float) -> float:
        """
        Convert an integer slider tick to a float value.

        Parameters:
            val_int (int): Input val int.
            min_val_float (float): Input min val float.
            resolution (float): Input resolution.

        Returns:
            float: Result computed from the current inputs.
        """
        return float(val_int) * float(resolution) + float(min_val_float)

    def get_values(self) -> np.ndarray:
        """
        Get current slider values as a numpy array of floats.

        Parameters:
            None.

        Returns:
            np.ndarray: Requested value computed from the current inputs.
        """
        values = np.zeros(self.n_slider, dtype=float)
        for s_idx in range(self.n_slider):
            values[s_idx] = self.int_to_float(
                val_int=int(self.sliders[s_idx].value()),
                min_val_float=float(self.slider_mins[s_idx]),
                resolution=float(self.resolutions[s_idx]),
            )
        return values

    def get_value_from_label(self,label:Any)->float:
        """
        Get slider value from label text.

        Parameters:
            label (Any): Input label.

        Returns:
            float: Requested value computed from the current inputs.
        """
        label_str = str(label)

        matched_idxs = []
        for s_idx,label_text in enumerate(self.label_texts):
            if str(label_text) == label_str:
                matched_idxs.append(s_idx)

        if len(matched_idxs) == 0:
            raise KeyError("Label [%s] not found." % (label_str))
        if len(matched_idxs) > 1:
            raise ValueError("Label [%s] is duplicated. Matched indices: %s" % (label_str,matched_idxs))

        s_idx = matched_idxs[0]
        return float(self.get_values()[s_idx])

    def set_value(self, s_idx: int, value: float) -> None:
        """
        Set value.

        Parameters:
            s_idx (int): Input s idx.
            value (float): Input value.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        s_idx = int(s_idx)
        if not (0 <= s_idx < self.n_slider):
            raise IndexError(f"s_idx out of range: {s_idx}")

        val_int = self.float_to_int(
            val_float=float(value),
            min_val_float=float(self.slider_mins[s_idx]),
            resolution=float(self.resolutions[s_idx]),
        )
        self.sliders[s_idx].setValue(val_int)

        actual = self.int_to_float(
            val_int=val_int,
            min_val_float=float(self.slider_mins[s_idx]),
            resolution=float(self.resolutions[s_idx]),
        )
        self.slider_vals[s_idx] = actual
        self.labels[s_idx].setText(f"{self.label_texts[s_idx]}:{actual:.2f}")

    def set_values(self,values:Sequence[float],idxs:Optional[Sequence[int]]=None)->None:
        """
        Set multiple slider values.

        Parameters:
            values (Sequence[float]): Input values.
            idxs (Optional[Sequence[int]]): Input idxs. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if idxs is None:
            if len(values)!=self.n_slider:
                raise ValueError(
                    "values length (%d) != n_slider (%d)." % (len(values),self.n_slider)
                )
            for s_idx in range(self.n_slider):
                self.set_value(s_idx=s_idx,value=float(values[s_idx]))
        else:
            if len(values)!=len(idxs):
                raise ValueError(
                    "values length (%d) != idxs length (%d)." % (len(values),len(idxs))
                )
            for s_idx,val in zip(idxs,values):
                if s_idx<0 or s_idx>=self.n_slider:
                    raise IndexError(
                        "s_idx %d out of range [0,%d]." % (s_idx,self.n_slider-1)
                    )
                self.set_value(s_idx=s_idx,value=float(val))

    def close(self) -> None:
        """
        Close the widget and release Qt resources.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        super().close()

# -----------------------------------------------------------------------------
# SingleSliderQtWidget
# -----------------------------------------------------------------------------
class SingleSliderQtWidget(object):
    """
    Thin wrapper around `MultiSliderQtWidget` for a single scalar slider.

    Usage:
        obj = SingleSliderQtWidget(...)
    """

    def __init__(
            self,
            x_offset: Union[int, float] = 0,
            y_offset: Union[int, float] = 0,
            window_width: Union[int, float] = 500,
            window_height: Union[int, float] = 80,
            title: str = "Slider",
            label_text: Optional[Any] = None,
            slider_min: float = 0.0,
            slider_max: float = 1.0,
            slider_val: float = 0.0,
            slider_steps: int = 1000001,
            resolution: Optional[float] = None,
            fontsize: int = 10,
            verbose: bool = False,
        ):
        """
        Initialize SingleSliderQtWidget.

        Parameters:
            x_offset (Union[int, float]): Input x offset. Defaults to `0`.
            y_offset (Union[int, float]): Input y offset. Defaults to `0`.
            window_width (Union[int, float]): Input window width. Defaults to `500`.
            window_height (Union[int, float]): Input window height. Defaults to `80`.
            title (str): Input title. Defaults to `'Slider'`.
            label_text (Optional[Any]): Input label text. Defaults to `None`.
            slider_min (float): Input slider min. Defaults to `0.0`.
            slider_max (float): Input slider max. Defaults to `1.0`.
            slider_val (float): Input slider val. Defaults to `0.0`.
            slider_steps (int): Input slider steps. Defaults to `1000001`.
            resolution (Optional[float]): Input resolution. Defaults to `None`.
            fontsize (int): Input fontsize. Defaults to `10`.
            verbose (bool): Input verbose. Defaults to `False`.

        Returns:
            None: Initializes the instance in place.
        """
        if label_text is None:
            label_text = title

        resolutions = None if resolution is None else [float(resolution)]
        self.slider = MultiSliderQtWidget(
            x_offset      = x_offset,
            y_offset      = y_offset,
            window_width  = window_width,
            window_height = window_height,
            title         = title,
            label_texts   = [label_text],
            slider_mins   = [float(slider_min)],
            slider_maxs   = [float(slider_max)],
            slider_vals   = [float(slider_val)],
            slider_steps  = slider_steps,
            resolutions   = resolutions,
            fontsize      = fontsize,
            verbose       = verbose,
        )

    def get_value(self) -> float:
        """
        Return value.

        Parameters:
            None.

        Returns:
            float: Requested value computed from the current inputs.
        """
        return float(self.slider.get_values()[0])

    def set_value(self, value: float) -> None:
        """
        Set value.

        Parameters:
            value (float): Input value.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.slider.set_value(s_idx=0, value=float(value))

    def close(self) -> None:
        """
        Close the widget and release Qt resources.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.slider.close()

# -----------------------------------------------------------------------------
# MultiRadioQtWidget
# -----------------------------------------------------------------------------
class MultiRadioQtWidget(QWidget):
    """
    Multi radio buttons using Qt.

    Usage:
        obj = MultiRadioQtWidget(...)
    """

    def __init__(
            self,
            x_offset: Union[int, float] = 0,
            y_offset: Union[int, float] = 0,
            window_width: Union[int, float] = 500,
            window_height: Union[int, float] = 300,
            title: str = "Multi Radio Buttons",
            label_texts: Optional[Sequence[str]] = None,
            option_texts: Optional[Sequence[Sequence[str]]] = None,
            fontsize: int = 10,
            initial_texts: Optional[Sequence[Optional[str]]] = None,
            verbose: bool = False,
        ):
        """
        Initialize MultiRadioQtWidget.

        Parameters:
            x_offset (Union[int, float]): Input x offset. Defaults to `0`.
            y_offset (Union[int, float]): Input y offset. Defaults to `0`.
            window_width (Union[int, float]): Input window width. Defaults to `500`.
            window_height (Union[int, float]): Input window height. Defaults to `300`.
            title (str): Input title. Defaults to `'Multi Radio Buttons'`.
            label_texts (Optional[Sequence[str]]): Input label texts. Defaults to
                                                   `None`.
            option_texts (Optional[Sequence[Sequence[str]]]): Input option texts.
                                                              Defaults to `None`.
            fontsize (int): Input fontsize. Defaults to `10`.
            initial_texts (Optional[Sequence[Optional[str]]]): Input initial texts.
                                                               Defaults to `None`.
            verbose (bool): Input verbose. Defaults to `False`.

        Returns:
            None: Initializes the instance in place.
        """
        super().__init__()

        self.verbose = bool(verbose)
        self.title = str(title)
        self.fontsize = int(fontsize)

        # -------------------------
        # Normalize label_texts
        # -------------------------
        if not label_texts:
            label_texts = ["option 0"]

        self.label_texts: List[str] = [str(x) for x in label_texts]
        n_row = len(self.label_texts)

        # -------------------------
        # Normalize option_texts
        # -------------------------
        if not option_texts:
            option_texts = [["none"] for _ in range(n_row)]

        # Ensure each row has at least one option
        norm_option_texts: List[List[str]] = []
        for i in range(n_row):
            if i < len(option_texts) and option_texts[i]:
                norm_option_texts.append([str(x) for x in option_texts[i]])
            else:
                norm_option_texts.append(["none"])

        self.option_texts = norm_option_texts
        self.n_row = n_row

        # -------------------------
        # Normalize initial_texts
        # -------------------------
        if initial_texts is None:
            self.initial_texts = [None] * self.n_row
        else:
            self.initial_texts = list(initial_texts)[: self.n_row]
            if len(self.initial_texts) < self.n_row:
                self.initial_texts += [None] * (self.n_row - len(self.initial_texts))

        # -------------------------
        # Geometry (screen-aware)
        # -------------------------
        screen_w, screen_h = get_monitor_size_fallback()
        self.window_width = resolve_screen_value(window_width, screen_w)
        self.window_height = resolve_screen_value(window_height, screen_h)
        self.x_offset = resolve_screen_value(x_offset, screen_w)
        self.y_offset = resolve_screen_value(y_offset, screen_h)

        # -------------------------
        # State
        # -------------------------
        self.button_groups: List[QButtonGroup] = []
        self.prev_idx_list: List[Optional[int]] = [None]*self.n_row

        self.init_ui()

    # -------------------------
    # UI
    # -------------------------
    def init_ui(self) -> None:
        """
        Initialize ui.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        main_layout = QVBoxLayout(self)

        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)

        scroll_widget = QWidget()
        scroll_layout = QGridLayout(scroll_widget)

        for row in range(self.n_row):
            # Row label
            label = QLabel(self.label_texts[row], self)
            label.setStyleSheet(f"font-size: {self.fontsize}px;")
            scroll_layout.addWidget(label, row, 0)

            # Buttons for this row
            button_group = QButtonGroup(self)
            self.button_groups.append(button_group)

            init_text = self.initial_texts[row]
            init_set = False

            for col, text in enumerate(self.option_texts[row]):
                radio = QRadioButton(text, self)
                radio.setStyleSheet(f"font-size: {self.fontsize}px;")
                button_group.addButton(radio, col)
                scroll_layout.addWidget(radio, row, col + 1)

                if (init_text is not None) and (text == init_text):
                    radio.setChecked(True)
                    init_set = True

            if (init_text is not None) and (not init_set) and self.verbose:
                print(f"[Warning] initial_text '{init_text}' not found in row '{self.label_texts[row]}'.")

        scroll_widget.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area)

        self.setLayout(main_layout)
        self.setWindowTitle(self.title)
        self.setGeometry(self.x_offset, self.y_offset, self.window_width, self.window_height)
        self.show()

    # -------------------------
    # Query helpers
    # -------------------------
    def get_button_indices(self) -> List[int]:
        """
        Return button indices.

        Parameters:
            None.

        Returns:
            List[int]: Requested value computed from the current inputs.
        """
        selected: List[int] = []
        for group in self.button_groups:
            checked = group.checkedId()
            selected.append(checked if checked >= 0 else -1)
        return selected

    def get_button_option(self,label_text:str,return_toggled:bool=False) -> Union[Optional[str],tuple[Optional[str],bool]]:
        """
        Return button option.

        Parameters:
            label_text (str): Input label text.
            return_toggled (bool): Input return toggled. Defaults to `False`.

        Returns:
            Union[Optional[str], tuple[Optional[str], bool]]: Requested value computed
                                                              from the current inputs.
        """
        for row,row_label in enumerate(self.label_texts):
            if row_label != label_text:
                continue

            checked_idx = self.button_groups[row].checkedId()
            selected    = None

            if checked_idx >= 0:
                selected = self.option_texts[row][checked_idx]

            if return_toggled:
                return selected,self.is_toggled(label_text=label_text)
            return selected

        if return_toggled:
            return None,False
        return None

    def get_option(self,label_text: str) -> Optional[str]:
        """
        Alias for get_button_option.

        Parameters:
            label_text (str): Input label text.

        Returns:
            Optional[str]: Requested value computed from the current inputs.
        """
        return self.get_button_option(label_text)

    # -------------------------
    # Change detection
    # -------------------------
    def is_toggled(self,label_text: str,option_text: str = None) -> bool:
        """
        - option_text is given:.

        Parameters:
            label_text (str): Input label text.
            option_text (str): Input option text. Defaults to `None`.

        Returns:
            bool: Boolean result for the requested condition.
        """
        for row,row_label in enumerate(self.label_texts):
            if row_label != label_text:
                continue

            cur  = self.button_groups[row].checkedId()  # -1 or index
            prev = self.prev_idx_list[row]

            # First-time sync: do NOT treat as an event
            if prev is None:
                self.prev_idx_list[row] = cur
                return False

            if option_text is None:
                toggled_now              = (cur != prev) and (cur != -1)
                self.prev_idx_list[row]  = cur
                return toggled_now

            if option_text not in self.option_texts[row]:
                if self.verbose:
                    print("[Warning] option_text '%s' not found in row '%s'" % (option_text,label_text))
                return False

            col                     = self.option_texts[row].index(option_text)
            toggled_now              = (cur != prev) and (cur == col)
            self.prev_idx_list[row]  = cur
            return toggled_now

        if self.verbose:
            print("[Warning] label_text '%s' not found" % (label_text))
        return False


    # -------------------------
    # Setters
    # -------------------------
    def set_button_option(self, label_text: str, option_text: str) -> bool:
        """
        Select option_text in the row identified by label_text.

        Parameters:
            label_text (str): Input label text.
            option_text (str): Input option text.

        Returns:
            bool: Result computed from the current inputs.
        """
        for row, row_label in enumerate(self.label_texts):
            if row_label != label_text:
                continue

            if option_text not in self.option_texts[row]:
                if self.verbose:
                    print(f"[Warning] option_text '{option_text}' not found in row '{label_text}'.")
                return False

            col = self.option_texts[row].index(option_text)
            btn = self.button_groups[row].button(col)
            if btn is None:
                if self.verbose:
                    print(f"[Warning] button not found for option '{option_text}' in row '{label_text}'.")
                return False

            btn.setChecked(True)
            return True

        if self.verbose:
            print(f"[Warning] label_text '{label_text}' not found.")
        return False

# -----------------------------------------------------------------------------
# LivePlotWidget
# -----------------------------------------------------------------------------
class LivePlotWidget(pg.PlotWidget):
    """
    Live time-series plot widget (PyQtGraph).

    Usage:
        obj = LivePlotWidget(...)
    """
    def __init__(
            self,
            labels,
            max_points       = 1000,
            title            = 'Live Plot',
            colors           = None,
            styles           = None,
            parent           = None,
            x_offset         = 0,
            y_offset         = 0,
            window_width     = 500,
            window_height    = 300,
            x_label          = 'time (sec)',
            legend_offset    = (10,10),
            symbol_size      = 6,
            pen_width        = 1,
            axis_font_size   = 10,
            auto_range       = True,
            auto_range_every = 1,
        ):
        """
        Initialize LivePlotWidget.

        Parameters:
            labels (Any): Input labels.
            max_points (int): Input max points. Defaults to `1000`.
            title (str): Input title. Defaults to `'Live Plot'`.
            colors (Any | None): Input colors. Defaults to `None`.
            styles (Any | None): Input styles. Defaults to `None`.
            parent (Any | None): Input parent. Defaults to `None`.
            x_offset (int): Input x offset. Defaults to `0`.
            y_offset (int): Input y offset. Defaults to `0`.
            window_width (int): Input window width. Defaults to `500`.
            window_height (int): Input window height. Defaults to `300`.
            x_label (str): Input x label. Defaults to `'time (sec)'`.
            legend_offset (tuple): Input legend offset. Defaults to `(10, 10)`.
            symbol_size (int): Input symbol size. Defaults to `6`.
            pen_width (int): Input pen width. Defaults to `1`.
            axis_font_size (int): Input axis font size. Defaults to `10`.
            auto_range (bool): Input auto range. Defaults to `True`.
            auto_range_every (int): Input auto range every. Defaults to `1`.

        Returns:
            None: Initializes the instance in place.
        """
        super().__init__(parent)

        if max_points is None or max_points is np.inf:
            raise ValueError("LivePlotWidget requires finite max_points (ring buffer).")

        self.labels         = labels
        self.n              = len(labels)
        self.max_points     = int(max_points)
        self.symbol_size    = symbol_size
        self.pen_width      = pen_width
        self.axis_font_size = axis_font_size

        # Window geometry
        # -------------------------------------------------
        # Window geometry (ratio [0,1] or absolute pixels)
        # -------------------------------------------------
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtGui import QGuiApplication

        # Ensure QApplication exists
        app = QApplication.instance()
        if app is None:
            use_pyqt_platform_plugins()
            app = QApplication([])

        # Get monitor size
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            geom = screen.availableGeometry()
            monitor_w = geom.width()
            monitor_h = geom.height()
        else:
            monitor_w = 1920
            monitor_h = 1080

        # Resolve window size
        if 0.0 <= window_width <= 1.0:
            win_w = int(monitor_w*float(window_width))
        else:
            win_w = int(window_width)

        if 0.0 <= window_height <= 1.0:
            win_h = int(monitor_h*float(window_height))
        else:
            win_h = int(window_height)

        # Minimum sanity
        win_w = max(50,win_w)
        win_h = max(50,win_h)

        # Default position = center
        x_px = int((monitor_w-win_w)*0.5)
        y_px = int((monitor_h-win_h)*0.5)

        # Override offsets if provided (ratio [0,1] or absolute pixels)
        if x_offset is not None:
            if 0.0 <= x_offset <= 1.0:
                x_px = int((monitor_w-win_w)*float(x_offset))
            else:
                x_px = int(x_offset)

        if y_offset is not None:
            if 0.0 <= y_offset <= 1.0:
                y_px = int((monitor_h-win_h)*float(y_offset))
            else:
                y_px = int(y_offset)

        # Clamp to screen
        x_px = max(0,min(x_px,monitor_w-win_w))
        y_px = max(0,min(y_px,monitor_h-win_h))

        # Apply
        self.setGeometry(x_px,y_px,win_w,win_h)
        self.setWindowTitle(title)


        # Buffers (ring)
        self.times=np.zeros(self.max_points,dtype=np.float64)
        self.data=np.zeros((self.n,self.max_points),dtype=np.float64)
        self.ptr=0
        self.full=False
        self._start=time.time()

        # Plot styling
        self.plotItem.setLabel('bottom',x_label)
        self.plotItem.showGrid(x=True,y=True)

        axis_style={'tickFont':QFont("Arial",axis_font_size)}
        self.plotItem.getAxis('bottom').setStyle(**axis_style)
        self.plotItem.getAxis('left').setStyle(**axis_style)

        # Legend
        self.legend=pg.LegendItem(offset=legend_offset)
        self.legend.setParentItem(self.plotItem)

        # Curves
        if colors is None:
            colors=['w']*self.n
        if styles is None:
            styles=['-']*self.n

        self.curves=[]
        for i,lbl in enumerate(labels):
            curve=self._add_curve(lbl,colors[i],styles[i])
            self.curves.append(curve)

        # Auto-range
        self.auto_range_enabled=bool(auto_range)
        self.auto_range_every=max(1,int(auto_range_every))
        self._auto_range_tick=0
        if self.auto_range_enabled:
            self.plotItem.autoRange()

        self.show()

    def _mk_color(self,color):
        """
        Build mk color.

        Parameters:
            color (Any): Input color.

        Returns:
            Any: Result computed from the current inputs.
        """
        if isinstance(color,(tuple,list,np.ndarray)) and len(color) in [3,4]:
            arr=np.asarray(color)
            if np.issubdtype(arr.dtype,np.floating) and arr.max()<=1.0:
                arr=(255.0*arr).astype(np.float64)
            return pg.mkColor(tuple(int(x) for x in arr[:4])) if len(arr)==4 else pg.mkColor(tuple(int(x) for x in arr[:3]))
        return pg.mkColor(color)

    def _parse_style(self,style_str,color):
        """
        Parse parse style.

        Parameters:
            style_str (Any): Input style str.
            color (Any): Input color.

        Returns:
            Any: Result computed from the current inputs.
        """
        c=self._mk_color(color)
        style_kwargs={}

        if 'o' in style_str:
            style_kwargs['symbol']='o'
            style_kwargs['symbolSize']=self.symbol_size
            style_kwargs['symbolPen']=c
            style_kwargs['symbolBrush']=c

        if '-' in style_str:
            style_kwargs['pen']=pg.mkPen(color=c,width=self.pen_width)
        else:
            style_kwargs['pen']=None

        return style_kwargs

    def _add_curve(self,label,color,style_str):
        """
        Add curve.

        Parameters:
            label (Any): Input label.
            color (Any): Input color.
            style_str (Any): Input style str.

        Returns:
            Any: Result computed from the current inputs.
        """
        style_kwargs=self._parse_style(style_str,color)
        curve=self.plotItem.plot(name=label,**style_kwargs)
        self.legend.addItem(curve,label)
        return curve

    # -------------------------
    # Auto-range controls
    # -------------------------
    def set_auto_range(self,flag=True):
        """
        Set auto range.

        Parameters:
            flag (bool): Input flag. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.auto_range_enabled = bool(flag)
        if self.auto_range_enabled:
            self.plotItem.autoRange()

    def toggle_auto_range(self):
        """
        Toggle plot axis auto-ranging.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.set_auto_range(not self.auto_range_enabled)

    def auto_range(self):
        """
        Return whether plot axes are currently auto-ranging.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.plotItem.autoRange()

    # -------------------------
    # Data push
    # -------------------------
    def push(self,x,timestamp=None,update_curves=True):
        """
        Push one sample.

        Parameters:
            x (Any): Input x.
            timestamp (Any | None): Input timestamp. Defaults to `None`.
            update_curves (bool): Input update curves. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        t=timestamp if timestamp is not None else (time.time()-self._start)
        x=np.asarray(x).reshape(-1,)
        if x.shape[0]!=self.n:
            raise ValueError(f"x must have length {self.n}, got {x.shape[0]}")

        self.times[self.ptr]=t
        self.data[:,self.ptr]=x

        self.ptr+=1
        if self.ptr>=self.max_points:
            self.ptr=0
            self.full=True

        if update_curves:
            self.redraw()

        if self.auto_range_enabled:
            self._auto_range_tick+=1
            if (self._auto_range_tick%self.auto_range_every)==0:
                self.plotItem.autoRange()

    def redraw(self):
        """
        Redraw curves from current buffers.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if self.full:
            t_disp=np.roll(self.times,-self.ptr)
            for i in range(self.n):
                y_disp=np.roll(self.data[i],-self.ptr)
                self.curves[i].setData(t_disp,y_disp)
        else:
            t_disp=self.times[:self.ptr]
            for i in range(self.n):
                self.curves[i].setData(t_disp,self.data[i,:self.ptr])

    # -------------------------
    # Optional: safe style update (no curve recreation)
    # -------------------------
    def set_style(self,index,style_str):
        """
        Update pen/symbol of an existing curve without recreating it (safer than
        remove+add).

        Parameters:
            index (Any): Input index.
            style_str (Any): Input style str.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if index<0 or index>=len(self.curves):
            raise IndexError("Curve index out of range")

        curve=self.curves[index]

        # Infer color from current pen if available
        pen=curve.opts.get('pen',None)
        if pen is not None:
            c=pen.color()
        else:
            # fallback: try symbolPen, else white
            spen=curve.opts.get('symbolPen',None)
            c=spen if spen is not None else pg.mkColor('w')

        # Apply new style
        if 'o' in style_str:
            curve.setSymbol('o')
            curve.setSymbolSize(self.symbol_size)
            curve.setSymbolPen(c)
            curve.setSymbolBrush(c)
        else:
            curve.setSymbol(None)

        if '-' in style_str:
            curve.setPen(pg.mkPen(color=c,width=self.pen_width))
        else:
            curve.setPen(None)

# -----------------------------------------------------------------------------
# get click position in plot coordinates (optional utility)
# -----------------------------------------------------------------------------
def pick_pos_neg_points_qt(
        img_np,
        title             = "Point Picker",
        x_offset          = 0,
        y_offset          = 0,
        window_width      = 0.4,
        window_height     = 0.4,
        fit_to_window     = True,
        allow_upscale     = True,
        max_display_width = None,
        fontsize          = 10,
        verbose           = True,
        show_image        = True,
        fig_w             = 4,
        dpi               = 120,
        pad_px            = 0,
        title_src         = None,
        title_fs          = 8,
        title_overlay     = True,
    ):
    """
    Display an image and allow user to pick positive/negative points by clicking.

    Parameters:
        img_np (Any): Input img np.
        title (str): Input title. Defaults to `'Point Picker'`.
        x_offset (int): Input x offset. Defaults to `0`.
        y_offset (int): Input y offset. Defaults to `0`.
        window_width (float): Input window width. Defaults to `0.4`.
        window_height (float): Input window height. Defaults to `0.4`.
        fit_to_window (bool): Input fit to window. Defaults to `True`.
        allow_upscale (bool): Input allow upscale. Defaults to `True`.
        max_display_width (Any | None): Input max display width. Defaults to
                                        `None`.
        fontsize (int): Input fontsize. Defaults to `10`.
        verbose (bool): Input verbose. Defaults to `True`.
        show_image (bool): Input show image. Defaults to `True`.
        fig_w (int): Input fig w. Defaults to `4`.
        dpi (int): Input dpi. Defaults to `120`.
        pad_px (int): Input pad px. Defaults to `0`.
        title_src (Any | None): Input title src. Defaults to `None`.
        title_fs (int): Input title fs. Defaults to `8`.
        title_overlay (bool): Input title overlay. Defaults to `True`.

    Returns:
        Any: Result computed from the current inputs.
    """
    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    from PyQt5.QtCore import Qt,QPoint,QTimer
    from PyQt5.QtGui import QPixmap,QPainter,QPen,QFont,QImage
    from PyQt5.QtWidgets import QApplication,QWidget,QLabel,QVBoxLayout,QSizePolicy

    def _np_to_qimage_rgb(x):
        """
        Convert an RGB numpy image to a Qt image.

        Parameters:
            x (Any): Input x.

        Returns:
            Any: Result computed from the current inputs.
        """
        x=np.asarray(x)
        if x.dtype!=np.uint8:
            x=np.clip(x,0,255).astype(np.uint8)
        if x.ndim==2:
            x=np.stack([x,x,x],axis=-1)
        if (x.ndim==3) and (x.shape[-1]==4):
            x=x[...,:3]
        if (x.ndim!=3) or (x.shape[-1]!=3):
            raise ValueError("img_np must be HxWx3 (or convertible),got shape=%s"%(str(x.shape),))
        h,w=x.shape[0],x.shape[1]
        bpl=3*w
        return QImage(x.data,w,h,bpl,QImage.Format_RGB888).copy()

    def _resolve_screen_value(v,full):
        """
        Resolve a fractional or absolute screen value.

        Parameters:
            v (Any): Input v.
            full (Any): Input full.

        Returns:
            Any: Result computed from the current inputs.
        """
        v=float(v)
        if (0.0 < v) and (v <= 1.0):
            return int(round(v*float(full)))
        return int(round(v))

    def _get_monitor_size_fallback():
        """
        Return the primary monitor size from Qt.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        app=QApplication.instance()
        if app is None:
            return (1920,1080)
        try:
            scr=app.primaryScreen()
            if scr is None:
                return (1920,1080)
            g=scr.availableGeometry()
            return (int(g.width()),int(g.height()))
        except Exception:
            return (1920,1080)

    def _imshow_fixed_axes(img,fig_w,fig_h=None,title=None,title_fs=8,axis_off=True,
                          cmap="viridis",colorbar=False,cbar_fs=6,caxis=None,
                          dpi=120,pad_px=0,title_overlay=True,cbar_inset=True):
        """
        Render an image with fixed Matplotlib axes.

        Parameters:
            img (Any): Input img.
            fig_w (Any): Input fig w.
            fig_h (Any | None): Input fig h. Defaults to `None`.
            title (Any | None): Input title. Defaults to `None`.
            title_fs (int): Input title fs. Defaults to `8`.
            axis_off (bool): Input axis off. Defaults to `True`.
            cmap (str): Input cmap. Defaults to `'viridis'`.
            colorbar (bool): Input colorbar. Defaults to `False`.
            cbar_fs (int): Input cbar fs. Defaults to `6`.
            caxis (Any | None): Input caxis. Defaults to `None`.
            dpi (int): Input dpi. Defaults to `120`.
            pad_px (int): Input pad px. Defaults to `0`.
            title_overlay (bool): Input title overlay. Defaults to `True`.
            cbar_inset (bool): Input cbar inset. Defaults to `True`.

        Returns:
            Any: Result computed from the current inputs.
        """
        if img is None:
            print("[imshow] 'img' is None. Nothing to display.")
            return

        img=np.asarray(img)
        h,w=img.shape[:2]

        if fig_h is None:
            fig_h=float(fig_w)*float(h)/max(1.0,float(w))
        else:
            fig_h=float(fig_h)

        pad_px=int(max(0,pad_px))
        fig=plt.figure(figsize=(float(fig_w),float(fig_h)),dpi=int(dpi),constrained_layout=False)
        fig.set_constrained_layout(False)

        if pad_px>0:
            fw_px=float(fig.get_figwidth()*fig.dpi)
            fh_px=float(fig.get_figheight()*fig.dpi)
            l=pad_px/fw_px
            b=pad_px/fh_px
            r=1.0-pad_px/fw_px
            t=1.0-pad_px/fh_px
            ax=fig.add_axes([l,b,max(0.0,r-l),max(0.0,t-b)])
        else:
            ax=fig.add_axes([0.0,0.0,1.0,1.0])

        if img.ndim==2:
            if caxis is not None:
                im=ax.imshow(img,cmap=cmap,vmin=caxis[0],vmax=caxis[1])
            else:
                im=ax.imshow(img,cmap=cmap)
        else:
            im=ax.imshow(img)

        if bool(axis_off):
            ax.set_axis_off()

        if title is not None:
            if bool(title_overlay):
                ax.text(
                    0.01,0.99,str(title),
                    transform=ax.transAxes,
                    ha="left",va="top",
                    fontsize=int(title_fs),
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.2",facecolor="black",alpha=0.35,edgecolor="none"),
                )
            else:
                fig.text(0.01,0.99,str(title),ha="left",va="top",fontsize=int(title_fs))

        if bool(colorbar) and (img.ndim==2):
            if bool(cbar_inset):
                try:
                    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                    cax=inset_axes(ax,width="3.5%",height="45%",loc="lower right",borderpad=0.8)
                    cbar=fig.colorbar(im,cax=cax)
                    cbar.ax.tick_params(labelsize=int(cbar_fs))
                except Exception as e:
                    print("[imshow] warning: inset colorbar failed (%s)"%str(e))
            else:
                cbar=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
                cbar.ax.tick_params(labelsize=int(cbar_fs))

        return fig,ax

    img_np=np.asarray(img_np)
    H0,W0=img_np.shape[0],img_np.shape[1]

    class _Win(QWidget):
        """
        Show the interactive point-picking Qt window.

        Usage:
            Used internally by this module.
        """
        def __init__(self):
            """
            Initialize _Win.

            Parameters:
                None.

            Returns:
                None: Initializes the instance in place.
            """
            super(_Win,self).__init__()
            self.setWindowTitle(str(title))
            self.pos=[]
            self.neg=[]
            self.done=False

            self._font=QFont("Menlo",int(fontsize))

            qimg0=_np_to_qimage_rgb(img_np)
            self._pm0=QPixmap.fromImage(qimg0)

            self._sx=1.0
            self._sy=1.0
            self._pm_disp=self._pm0

            self._label=QLabel(self)
            self._label.setAlignment(Qt.AlignLeft|Qt.AlignTop)
            self._label.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Ignored)
            self._label.setMinimumSize(1,1)

            layout=QVBoxLayout(self)
            layout.setContentsMargins(0,0,0,0)
            layout.addWidget(self._label)
            self.setLayout(layout)

            screen_w,screen_h=_get_monitor_size_fallback()
            x_px=_resolve_screen_value(x_offset,screen_w)
            y_px=_resolve_screen_value(y_offset,screen_h)
            w_px=_resolve_screen_value(window_width,screen_w)
            h_px=_resolve_screen_value(window_height,screen_h)
            self.setGeometry(x_px,y_px,w_px,h_px)

            if verbose:
                print("[pick_pos_neg_points_qt] Left=pos,Right=neg,Enter/Esc=done,Backspace=undo")

            QTimer.singleShot(0,self._update_pixmap)

        def _target_disp_size(self):
            """
            Return target disp size.

            Parameters:
                None.

            Returns:
                Any: Result computed from the current inputs.
            """
            r=self._label.contentsRect()
            w_av=max(1,int(r.width()))
            h_av=max(1,int(r.height()))

            if max_display_width is not None:
                w_av=min(w_av,int(max_display_width))

            if not fit_to_window:
                return self._pm0.width(),self._pm0.height()

            s=min(float(w_av)/float(W0),float(h_av)/float(H0))
            if not allow_upscale:
                s=min(1.0,s)

            w_disp=max(1,int(round(W0*s)))
            h_disp=max(1,int(round(H0*s)))
            return w_disp,h_disp

        def _update_pixmap(self):
            """
            Update update pixmap.

            Parameters:
                None.

            Returns:
                None: This function mutates state, renders output, or performs side
                      effects in place.
            """
            w_disp,h_disp=self._target_disp_size()
            self._pm_disp=self._pm0.scaled(w_disp,h_disp,Qt.KeepAspectRatio,Qt.SmoothTransformation)
            self._label.setPixmap(self._pm_disp)

            self._sx=float(W0)/float(self._pm_disp.width())
            self._sy=float(H0)/float(self._pm_disp.height())
            self._redraw()

        def resizeEvent(self,event):
            """
            Handle a Qt resize event.

            Parameters:
                event (Any): Input event.

            Returns:
                None: This function mutates state, renders output, or performs side
                      effects in place.
            """
            super(_Win,self).resizeEvent(event)
            self._update_pixmap()

        def _redraw(self):
            """
            Redraw the current display.

            Parameters:
                None.

            Returns:
                None: This function mutates state, renders output, or performs side
                      effects in place.
            """
            pm=QPixmap(self._pm_disp)
            p=QPainter(pm)
            p.setFont(self._font)

            pen_pos=QPen(Qt.red,2)
            pen_neg=QPen(Qt.blue,2)

            for (x0,y0) in self.pos:
                xd=int(round(float(x0)/self._sx))
                yd=int(round(float(y0)/self._sy))
                p.setPen(pen_pos)
                p.drawEllipse(QPoint(xd,yd),6,6)
                p.drawText(xd+8,yd+8,"(%d,%d)"%(x0,y0))

            for (x0,y0) in self.neg:
                xd=int(round(float(x0)/self._sx))
                yd=int(round(float(y0)/self._sy))
                p.setPen(pen_neg)
                p.drawLine(xd-6,yd-6,xd+6,yd+6)
                p.drawLine(xd-6,yd+6,xd+6,yd-6)
                p.drawText(xd+8,yd+8,"(%d,%d)"%(x0,y0))

            p.end()
            self._label.setPixmap(pm)

        def _event_to_img_xy(self,event):
            """
            Convert a Qt mouse event to image pixel coordinates.

            Parameters:
                event (Any): Input event.

            Returns:
                Any: Result computed from the current inputs.
            """
            gp=self._label.mapFrom(self,QPoint(int(event.x()),int(event.y())))
            xd=int(gp.x())
            yd=int(gp.y())

            if xd<0 or yd<0 or xd>=self._pm_disp.width() or yd>=self._pm_disp.height():
                return None

            x0=int(round(float(xd)*self._sx))
            y0=int(round(float(yd)*self._sy))
            x0=max(0,min(W0-1,x0))
            y0=max(0,min(H0-1,y0))
            return (x0,y0)

        def mousePressEvent(self,event):
            """
            Handle a Qt mouse-press event.

            Parameters:
                event (Any): Input event.

            Returns:
                None: This function mutates state, renders output, or performs side
                      effects in place.
            """
            xy=self._event_to_img_xy(event)
            if xy is None:
                return
            x0,y0=xy

            if event.button()==Qt.LeftButton:
                self.pos.append([x0,y0])
                if verbose:
                    print("pos:(%d,%d)"%(x0,y0))
            elif event.button()==Qt.RightButton:
                self.neg.append([x0,y0])
                if verbose:
                    print("neg:(%d,%d)"%(x0,y0))

            self._redraw()

        def keyPressEvent(self,event):
            """
            Handle a Qt key-press event.

            Parameters:
                event (Any): Input event.

            Returns:
                None: This function mutates state, renders output, or performs side
                      effects in place.
            """
            k=event.key()
            if (k==Qt.Key_Return) or (k==Qt.Key_Enter) or (k==Qt.Key_Escape):
                self.done=True
                if verbose:
                    print("done. pos=%d neg=%d"%(len(self.pos),len(self.neg)))
                self.close()
                return
            if k==Qt.Key_Backspace:
                if len(self.neg)>0:
                    self.neg.pop()
                elif len(self.pos)>0:
                    self.pos.pop()
                self._redraw()

    app=QApplication.instance()
    created=False
    if app is None:
        use_pyqt_platform_plugins()
        app=QApplication(sys.argv)
        created=True

    win=_Win()
    win.show()
    app.exec_()

    pos=list(win.pos)
    neg=list(win.neg)

    if created:
        try: app.quit()
        except Exception:
            pass

    # Plot (fixed-axes, fig_w preserved)
    if show_image:
        img_show=np.asarray(img_np)
        h,w=img_show.shape[:2]
        fig_h=float(fig_w)*float(h)/max(1.0,float(w))

        fig,ax=_imshow_fixed_axes(
            img_show,
            fig_w=float(fig_w),
            fig_h=float(fig_h),
            title=title_src,
            title_fs=int(title_fs),
            axis_off=True,
            cmap="viridis",
            colorbar=False,
            dpi=int(dpi),
            pad_px=int(pad_px),
            title_overlay=bool(title_overlay),
        )

        marker_size = 50
        text_offset = [10,0]  # (dx,dy)

        for (x,y) in pos:
            x1=float(x); y1=float(y)
            ax.scatter([x1],[y1],s=float(marker_size),marker="o",facecolors="none",edgecolors="r",lw=2)

            txt="(%d,%d)"%(int(x),int(y))
            ax.text(
                x1+float(text_offset[0]),
                y1+float(text_offset[1]),
                txt,
                fontsize = 8,
                ha       = "left",
                va       = "top",
                bbox     = dict(
                    boxstyle  = "square,pad=0.05",   # 핵심: pad를 아주 작게 (0도 가능)
                    facecolor = "white",
                    alpha     = 0.7,
                    edgecolor = "none",
                ),
            )

        for (x,y) in neg:
            x1=float(x); y1=float(y)
            ax.scatter([x1],[y1],s=float(marker_size),marker="x",color="b",lw=2)

            txt="(%d,%d)"%(int(x),int(y))
            ax.text(
                x1+float(text_offset[0]),
                y1+float(text_offset[1]),
                txt,
                fontsize = 8,
                ha       = "left",
                va       = "top",
                bbox     = dict(
                    boxstyle  = "square,pad=0.05",
                    facecolor = "white",
                    alpha     = 0.7,
                    edgecolor = "none",
                ),
            )

        plt.show()

    return pos,neg

# -----------------------------------------------------------------------------
