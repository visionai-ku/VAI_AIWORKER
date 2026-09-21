"""
utils.py
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import platform
import subprocess
import zmq
import os
import json
import inspect
from pathlib import Path
from xml.dom import minidom
from IPython.display import display,Javascript
from scipy.spatial.distance import cdist
from termcolor import colored

def sleep(sec):
    """
    Sleep for a short interval.

    Parameters:
        sec (float): Seconds to sleep.

    Returns:
        None.
    """
    time.sleep(float(sec))

def close_objects(*objects, scope=None, verbose=False):
    """
    Close existing objects with a `.close()` method.

    Parameters:
        *objects (Any): Objects to close, or variable names to resolve from the
            caller scope.
        scope (dict | None): Optional namespace used when an item is a string.
            If None, string items are resolved from the caller's local and
            global variables.
        verbose (bool): If True, print close failures instead of staying quiet.

    Returns:
        None.

    Notes:
        This helper is mainly for rerunnable notebooks. It lets a cell clean up
        objects from a previous run with `close_objects("rp","sliders")`
        without writing repeated `if name in globals(): ...` blocks.
    """
    # Resolve caller scope if needed
    caller_scope = {}
    if scope is None and any(isinstance(obj,str) for obj in objects):
        frame = inspect.currentframe()
        if frame is not None and frame.f_back is not None: # get caller frame
            caller_scope.update(frame.f_back.f_globals)
            caller_scope.update(frame.f_back.f_locals)
    elif scope is not None:
        caller_scope = scope

    # Close each object
    for i,obj in enumerate(objects):
        obj_close = caller_scope.get(obj,None) if isinstance(obj,str) else obj
        if obj_close is None:
            if verbose:
                print("[%d] object [%s] not found in scope"%(i,obj))
            continue
        close_fn = getattr(obj_close,"close",None) # get the close method
        if not callable(close_fn): # if no close method, skip
            continue
        try:
            close_fn() # call the close method
            if verbose:
                print("[%d] closed [%s]"%(i,obj_close))
        except Exception as err:
            if verbose:
                print("[%d] close failed [%s]: %s"%(i,obj,err)) # print error if verbose

def xyz2str(
        xyz,
        fmt     = "%.3f",
        bracket = "[]",
        space   = True,
    ):
    """
    Format an xyz vector as a compact string.

    Parameters:
        xyz (Any): Input xyz.
        fmt (str): Input fmt. Defaults to `'%.3f'`.
        bracket (str): Input bracket. Defaults to `'[]'`.
        space (bool): Input space. Defaults to `True`.

    Returns:
        Any: Result computed from the current inputs.
    """
    xyz = np.asarray(xyz,dtype=float).reshape(-1)
    if xyz.size != 3:
        raise ValueError("xyz should contain exactly three values.")
    if isinstance(fmt,int):
        fmt = "%%.%df"%fmt
    if bracket is None or bracket == "":
        bracket_l,bracket_r = "",""
    elif isinstance(bracket,str):
        if bracket == "[]":
            bracket_l,bracket_r = "[","]"
        elif bracket == "()":
            bracket_l,bracket_r = "(",")"
        else:
            raise ValueError("bracket should be '[]', '()', '', None, or a pair.")
    else:
        if len(bracket) != 2:
            raise ValueError("bracket pair should contain two strings.")
        bracket_l,bracket_r = bracket
    return "%s%s%s"%(
        bracket_l,
        (", " if space else ",").join([fmt%float(v) for v in xyz]),
        bracket_r,
    )

def get_idxs(query_list, domain_list, strict=True):
    """
    Get the corresponding indices in list_query for the items present in
    list_domain.

    Parameters:
        query_list (np.ndarray): Input query list.
        domain_list (Any): Input domain list.
        strict (bool): Input strict. Defaults to `True`.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    if not isinstance(query_list, list) or not isinstance(domain_list, list):
        raise TypeError("[get_idxs] Both inputs should be lists.")
    idx_map = {item:idx for idx,item in enumerate(query_list)}
    missing = [item for item in domain_list if item not in idx_map]
    if strict and len(missing) > 0:
        raise ValueError("[get_idxs] missing items: %s"%(missing))
    return [idx_map[item] for item in domain_list if item in idx_map]

def get_items_by_prefix(items, prefix):
    """
    Return the strings in `items` that start with `prefix`.

    Parameters:
        items (Any): Input items.
        prefix (Any): Input prefix.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    if not isinstance(items, list):
        raise TypeError("[get_items_by_prefix] items should be a list.")
    if not isinstance(prefix, str):
        raise TypeError("[get_items_by_prefix] prefix should be a string.")
    if not all([isinstance(item, str) for item in items]):
        raise TypeError("[get_items_by_prefix] items should contain strings only.")
    items = [item for item in items if item.startswith(prefix)]
    # Return
    return sorted(items) # sort for consistent order

def group_by_prefix(items, prefix_dict, rest_key="rest", verbose=False):
    """
    Group strings by prefix rules and collect unmatched items into a rest bucket.

    Parameters:
        items (Any): Input items.
        prefix_dict (Any): Input prefix dict.
        rest_key (str): Input rest key. Defaults to `'rest'`.
        verbose (bool): Input verbose. Defaults to `False`.

    Returns:
        Any: Result computed from the current inputs.
    """
    if not isinstance(items,list):
        raise TypeError("[group_by_prefix] items should be a list.")
    if not isinstance(prefix_dict,dict):
        raise TypeError("[group_by_prefix] prefix_dict should be a dict.")

    grouped = {key: [] for key in prefix_dict.keys()}
    grouped[rest_key] = []

    prefix_dict_norm = {}
    for key, prefixes in prefix_dict.items():
        if isinstance(prefixes,str):
            prefixes = [prefixes]
        elif isinstance(prefixes,list):
            if not all([isinstance(prefix,str) for prefix in prefixes]):
                raise TypeError("[group_by_prefix] prefix list should contain strings only.")
        else:
            raise TypeError(
                "[group_by_prefix] prefix_dict values should be str or list of str."
            )
        prefix_dict_norm[key] = prefixes

    for item in items:
        if not isinstance(item,str):
            raise TypeError("[group_by_prefix] items should contain strings only.")

        is_matched = False
        for key, prefixes in prefix_dict_norm.items():
            if any([item.startswith(prefix) for prefix in prefixes]):
                grouped[key].append(item)
                is_matched = True
                break
        if not is_matched:
            grouped[rest_key].append(item)

    # Verbose print
    if verbose:
        i = 0
        for key, group_items in grouped.items():
            print ("[%d/%d] group:[%s] n_items:[%d]"%(i,len(grouped),key,len(group_items)))
            for j,item in enumerate(group_items):
                print (" [%02d/%02d] [%s]"%(j,len(group_items),item))
            i += 1

    # Return
    return grouped

def sample_path_list(
        root_dir,
        glob_pattern,
        n_sample   = None,
        seed       = None,
        replace    = False,
        sort_paths = True,
        verbose    = False,
    ):
    """
    Collect paths under a root directory and optionally sample from them.

    Parameters:
        root_dir (str | Path): Input root dir.
        glob_pattern (Any): Input glob pattern.
        n_sample (Any | None): Input n sample. Defaults to `None`.
        seed (Any | None): Input seed. Defaults to `None`.
        replace (bool): Input replace. Defaults to `False`.
        sort_paths (bool): Input sort paths. Defaults to `True`.
        verbose (bool): Input verbose. Defaults to `False`.

    Returns:
        Any: Result computed from the current inputs.
    """
    root_dir = Path(root_dir)
    path_list = list(root_dir.glob(glob_pattern))
    if sort_paths:
        path_list = sorted(path_list)

    if n_sample is not None:
        assert len(path_list) > 0, "[sample_path_list] No matched paths found."
        if replace:
            assert n_sample >= 0, "[sample_path_list] n_sample should be non-negative."
        else:
            assert len(path_list) >= n_sample, "[sample_path_list] Not enough matched paths."
        rng = np.random.default_rng(seed)
        sampled_idxs = rng.choice(len(path_list), size=n_sample, replace=replace)
        path_list = [path_list[idx] for idx in sampled_idxs]

    # Convert to string paths
    path_list = [str(path) for path in path_list]

    # Print verbose info
    if verbose:
        print ("[sample_path_list] \nroot_dir:[%s] \nglob_pattern:[%s] \nn_matched:[%d] n_sampled:[%d] seed:[%s]"%
               (str(root_dir), glob_pattern, len(list(root_dir.glob(glob_pattern))), len(path_list), seed))
        for i,path in enumerate(path_list):
            print (" [%02d/%02d] [%s]"%(i,len(path_list),path))

    return path_list

def check_path_list(
        root_dir,
        glob_pattern,
        sort_paths = True,
        verbose    = False,
    ):
    """
    Collect paths under a root directory, print all matched paths, and return
    them.

    Parameters:
        root_dir (str | Path): Input root dir.
        glob_pattern (Any): Input glob pattern.
        sort_paths (bool): Input sort paths. Defaults to `True`.
        verbose (bool): Input verbose. Defaults to `False`.

    Returns:
        Any: Result computed from the current inputs.
    """
    root_dir = Path(root_dir)
    path_list = list(root_dir.glob(glob_pattern))
    if sort_paths:
        path_list = sorted(path_list)

    # Convert to string paths
    path_list = [str(path) for path in path_list]

    # Print summary
    if verbose:
        print ("[check_path_list] \nroot_dir:[%s] \nglob_pattern:[%s] \nn_matched:[%d]"%
               (str(root_dir), glob_pattern, len(path_list)))

    # Print all matched paths
    for i,path in enumerate(path_list):
        print (" [%02d/%02d] [%s]"%(i,len(path_list),path))

    return path_list

def indent_xml(elem):
    """
    Convert an XML Element into a neatly formatted, indented XML string.

    Parameters:
        elem (Any): Input elem.

    Returns:
        Any: Result computed from the current inputs.
    """
    rough_string = ET.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="    ")
    # Remove empty lines
    lines = [line for line in pretty_xml.splitlines() if line.strip()]
    return "\n".join(lines)

def get_monitor_size():
    """
    Retrieve the current monitor size.

    Parameters:
        None.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    # pyautogui connects to X11 at import time on Linux, so import it lazily
    # and fall back to a common size when no display is available.
    try:
        import pyautogui
        w,h = pyautogui.size()
    except Exception:
        w,h = 1920,1080
    return w,h

def wait_frontend_render():
    """
    Wait for the frontend to finish rendering (e.g., MathJax).

    Parameters:
        None.

    Returns:
        None: This function mutates state, renders output, or performs side
              effects in place.
    """
    display(Javascript(r"""
    (async () => {
      for (let i=0;i<3;i++){
        await new Promise(r=>requestAnimationFrame(r));
        await new Promise(r=>setTimeout(r,50));
      }
      if (window.MathJax && MathJax.typesetPromise){
        try { await MathJax.typesetPromise(); } catch (e) {}
      }
      await new Promise(r=>requestAnimationFrame(r));
    })();
    """))

class TicToc(object):
    """
    Simple timer for measuring elapsed wall time.

    Usage:
        obj = TicToc(...)
    """
    def __init__(self,name="tictoc"):
        """
        Initialize TicToc.

        Parameters:
            name (str): Input name. Defaults to `'tictoc'`.

        Returns:
            None: Initializes the instance in place.
        """
        self.name         = name
        self.time_start   = time.time()
        self.time_elapsed = 0.0
        self.cnt          = 0

    def tic(self):
        """
        Start or restart the timer.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.time_start = time.time()

    def toc(self,label=None,cnt=None,print_every=None,verbose=False):
        """
        Stop the timer and return the elapsed time.

        Parameters:
            label (Any | None): Input label. Defaults to `None`.
            cnt (Any | None): Input cnt. Defaults to `None`.
            print_every (Any | None): Input print every. Defaults to `None`.
            verbose (bool): Input verbose. Defaults to `False`.

        Returns:
            Any: Result computed from the current inputs.
        """
        t = time.time()
        self.time_elapsed = t - self.time_start

        if print_every is not None and verbose:
            if self.time_elapsed < 1.0:
                time_show = self.time_elapsed * 1000.0
                unit = "ms"
            elif self.time_elapsed < 60.0:
                time_show = self.time_elapsed
                unit = "s"
            else:
                time_show = self.time_elapsed / 60.0
                unit = "min"

            if cnt is not None:
                self.cnt = cnt

            if (self.cnt % print_every) == 0:
                prefix = self.name if label is None else label
                print(f"{prefix} Elapsed time:[{time_show:.2f}]{unit}")

        self.cnt += 1
        return self.time_elapsed

class TicTocs(object):
    """
    Small named collection of `TicToc` timers.

    Usage:
        tt = TicTocs("load","solve","render")
        tt.tic("solve")
        ...
        tt.toc("solve")
        solve_ms = tt.ms("solve")
    """
    def __init__(self,*names):
        """
        Initialize named timers.

        Parameters:
            *names (str): Timer names.

        Returns:
            None: Initializes the instance in place.
        """
        if len(names) == 1 and isinstance(names[0],(list,tuple)):
            names = tuple(names[0])
        self.names = tuple(str(name) for name in names)
        self.tt = {name:TicToc(name=name) for name in self.names}

    def __contains__(self,name):
        """
        Check whether a timer name exists.

        Parameters:
            name (str): Timer name.

        Returns:
            bool: True if the timer exists.
        """
        return str(name) in self.tt

    def __getitem__(self,name):
        """
        Return the underlying `TicToc` timer.

        Parameters:
            name (str): Timer name.

        Returns:
            TicToc: Underlying timer.
        """
        return self.tt[str(name)]

    def tic(self,name):
        """
        Start or restart one named timer.

        Parameters:
            name (str): Timer name.

        Returns:
            TicTocs: This object.
        """
        self.tt[str(name)].tic()
        return self

    def toc(self,name,**kwargs):
        """
        Stop one named timer and return elapsed seconds.

        Parameters:
            name (str): Timer name.
            **kwargs: Forwarded to `TicToc.toc(...)`.

        Returns:
            float: Elapsed seconds.
        """
        return self.tt[str(name)].toc(**kwargs)

    def sec(self,name):
        """
        Return the last elapsed seconds for one timer.

        Parameters:
            name (str): Timer name.

        Returns:
            float: Last elapsed seconds.
        """
        return float(self.tt[str(name)].time_elapsed)

    def ms(self,name):
        """
        Return the last elapsed milliseconds for one timer.

        Parameters:
            name (str): Timer name.

        Returns:
            float: Last elapsed milliseconds.
        """
        return 1000.0*self.sec(name)

    def secs(self):
        """
        Return all last elapsed seconds.

        Parameters:
            None.

        Returns:
            dict[str, float]: Elapsed seconds keyed by timer name.
        """
        return {name:self.sec(name) for name in self.names}

    def msecs(self):
        """
        Return all last elapsed milliseconds.

        Parameters:
            None.

        Returns:
            dict[str, float]: Elapsed milliseconds keyed by timer name.
        """
        return {name:self.ms(name) for name in self.names}

def trim_scale(x,th):
    """
    Trim the scale of the input array such that its maximum absolute value does
    not exceed a given threshold.

    Parameters:
        x (Any): Input x.
        th (Any): Input th.

    Returns:
        Any: Result computed from the current inputs.
    """
    x         = np.copy(x)
    x_abs_max = np.abs(x).max()
    if x_abs_max > th:
        x = x*th/x_abs_max
    return x

def fix_scale(x,th):
    """
    Fix the scale of the input array to a given threshold based on its maximum
    absolute value.

    Parameters:
        x (Any): Input x.
        th (Any): Input th.

    Returns:
        Any: Result computed from the current inputs.
    """
    x         = np.copy(x)
    x_abs_max = np.abs(x).max()
    if x_abs_max > 0.0:
        x = x*th/x_abs_max
    return x

def softmax(x,axis=-1,temperature=1.0,eps=1e-12):
    """
    Compute softmax values for each set of scores in x.

    Parameters:
        x (Any): Input x.
        axis (Any): Input axis. Defaults to `-1`.
        temperature (float): Input temperature. Defaults to `1.0`.
        eps (float): Input eps. Defaults to `1e-12`.

    Returns:
        Any: Result computed from the current inputs.
    """
    x = np.asarray(x,dtype=np.float64)
    T = float(temperature)
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("softmax: temperature must be finite and > 0 (got %s)" % str(temperature))
    z    = x / T
    zmax = np.max(z,axis=axis,keepdims=True)
    z    = z - zmax
    ez   = np.exp(z)
    den  = np.sum(ez,axis=axis,keepdims=True)
    den  = np.maximum(den,eps)
    # Return
    return ez / den

def standardize(
        x,
        eps          = 1e-8,
        center       = True,
        scale        = True,
        return_stats = False,
    ):
    """
    Standardize values: mean 0 and/or std 1.

    Parameters:
        x (Any): Input x.
        eps (float): Input eps. Defaults to `1e-08`.
        center (bool): Input center. Defaults to `True`.
        scale (bool): Input scale. Defaults to `True`.
        return_stats (bool): Input return stats. Defaults to `False`.

    Returns:
        Any: Result computed from the current inputs.
    """
    x = np.asarray(x,dtype=np.float64)

    mu = float(np.mean(x)) if center else 0.0
    y  = x - mu

    sd = float(np.std(y)) if scale else 1.0
    if scale:
        y = y / (sd + eps)

    if return_stats:
        return y,mu,sd
    return y

def power_transform(
        x,
        T          = 1.0,
        p          = 1.0,
        mode       = "signed",  # "signed" | "abs" | "positive" | "exp"
        eps        = 0.0,
    ):
    """
    Elementwise power transform.

    Parameters:
        x (Any): Input x.
        T (float): Input T. Defaults to `1.0`.
        p (float): Input p. Defaults to `1.0`.
        mode (str): Input mode. Defaults to `'signed'`.
        eps (float): Input eps. Defaults to `0.0`.

    Returns:
        Any: Result computed from the current inputs.
    """
    x = np.asarray(x,dtype=np.float64)

    if mode == "signed":
        return np.sign(x) * (T*np.abs(x) + eps)**p
    elif mode == "abs":
        return (T*np.abs(x) + eps)**p
    elif mode == "positive":
        return (T*x + eps)**p
    elif mode == "exp":
        return np.exp(T*x+eps)**p
    else:
        raise ValueError("Invalid mode: %s"%mode)

def get_trajs_topk(
        trajs,
        scores,
        k,
        largest = True,   # False: smaller score is better (default)
    ):
    """
    Select top-k trajectories according to scores.

    Parameters:
        trajs (Any): Input trajs.
        scores (Any): Input scores.
        k (Any): Input k.
        largest (bool): Input largest. Defaults to `True`.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    trajs = np.asarray(trajs)
    scores = np.asarray(scores)

    if trajs.ndim != 2:
        raise ValueError("trajs must have shape (L,n_traj)")
    if scores.ndim != 1:
        raise ValueError("scores must have shape (n_traj,)")

    n_traj = scores.shape[0]
    if k > n_traj:
        raise ValueError("k must be <= number of trajectories")

    # argsort: ascending
    order = np.argsort(scores)

    if largest:
        idx_topk = order[-k:][::-1]   # largest first
    else:
        idx_topk = order[:k]          # smallest first

    trajs_topk = trajs[:,idx_topk]
    scores_topk = scores[idx_topk]

    return trajs_topk,scores_topk

def sample_xyzs(
        n_sample   = 1,
        x_range    = (-1,-1),
        y_range    = (-1,+1),
        z_range    = (0,0),
        min_dist   = 0.1,
        xy_margin  = 0.0,
        seed       = None,
        return_ranges = False,
    ):
    """
    Sample 3D points within specified ranges ensuring a minimum distance between
    points.

    Parameters:
        n_sample (int): Input n sample. Defaults to `1`.
        x_range (tuple): Input x range. Defaults to `(-1, -1)`.
        y_range (tuple): Input y range. Defaults to `(-1, +1)`.
        z_range (tuple): Input z range. Defaults to `(0, 0)`.
        min_dist (float): Input min dist. Defaults to `0.1`.
        xy_margin (float): Input xy margin. Defaults to `0.0`.
        seed (Any | None): Input seed. Defaults to `None`.
        return_ranges (bool): Input return ranges. Defaults to `False`.

    Returns:
        Any: Result computed from the current inputs.
    """
    rng = np.random.default_rng(seed)
    ranges = {
        "x_range": (x_range[0]+xy_margin,x_range[1]-xy_margin),
        "y_range": (y_range[0]+xy_margin,y_range[1]-xy_margin),
        "z_range": (z_range[0],z_range[1]),
    }
    xyzs = np.zeros((n_sample,3)) # x,y,z for each row
    # Sample points
    for p_idx in range(n_sample): # for each point
        while True:
            x_rand = rng.uniform(low=ranges["x_range"][0],high=ranges["x_range"][1])
            y_rand = rng.uniform(low=ranges["y_range"][0],high=ranges["y_range"][1])
            z_rand = rng.uniform(low=ranges["z_range"][0],high=ranges["z_range"][1])
            xyz = np.array([x_rand,y_rand,z_rand])
            if p_idx == 0: break
            devc = cdist(xyz.reshape((-1,3)),xyzs[:p_idx,:].reshape((-1,3)),"euclidean")
            if devc.min() > min_dist: break # minimum distance between objects
        xyzs[p_idx,:] = xyz
    # Return
    if return_ranges:
        return xyzs,ranges
    return xyzs

def get_p_cam_orbit(
        theta_cam,
        radius,
        height,
        p_target,
    ):
    """
    Compute a camera position on a horizontal circular orbit around a target.

    Parameters:
        theta_cam (Any): Input theta cam.
        radius (Any): Input radius.
        height (Any): Input height.
        p_target (np.ndarray): Input p target.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    p_target = np.asarray(p_target,dtype=np.float64).reshape(3,)
    p_cam = p_target + np.array([
        radius*np.cos(theta_cam),
        radius*np.sin(theta_cam),
        height-p_target[2],
    ],dtype=np.float64)
    return p_cam

def xy2xyz(xy,z=0.0):
    """
    Convert 2D coordinates to 3D by appending a constant z-value.

    Parameters:
        xy (Any): Input xy.
        z (float): Input z. Defaults to `0.0`.

    Returns:
        Any: Result computed from the current inputs.
    """
    xy = np.asarray(xy)
    if xy.ndim == 1:
        if xy.shape[0] != 2:
            raise ValueError("1D input must have shape (2,).")
        return np.concatenate((xy,np.array([z],dtype=xy.dtype)))
    elif xy.ndim == 2:
        if xy.shape[1] != 2:
            raise ValueError("2D input must have shape (L,2).")
        z_col = np.full((xy.shape[0],1),z,dtype=xy.dtype)
        return np.concatenate((xy,z_col),axis=1)
    else:
        raise ValueError("Input must be a 1D or 2D array.")

def normalize_angle(angle):
    """
    Normalize angle to [-pi, pi], preserving +pi if the original angle.

    Parameters:
        angle (Any): Input angle.

    Returns:
        Any: Result computed from the current inputs.
    """
    angle = np.asarray(angle)
    is_scalar = angle.ndim == 0

    norm = (angle + np.pi) % (2 * np.pi) - np.pi

    # Preserve +pi if the original angle was equivalent to +pi
    mask = (norm == -np.pi) & (np.mod(angle, 2 * np.pi) == np.pi)

    if is_scalar:
        return float(np.pi) if mask.item() else float(norm)

    norm = norm.astype(float)
    norm[mask] = np.pi
    return norm

def soft_tanh_margin(z,margin):
    """
    Apply a margin-scaled tanh soft threshold.

    Parameters:
        z (Any): Input z.
        margin (Any): Input margin.

    Returns:
        Any: Result computed from the current inputs.
    """
    return margin*np.tanh(z/margin)

def soft_squash(x,x_min=-1.0,x_max=+1.0,margin=0.1):
    """
    Softly squash x into [x_min,x_max] with given margin.

    Parameters:
        x (Any): Input x.
        x_min (Any): Input x min. Defaults to `-1.0`.
        x_max (Any): Input x max. Defaults to `+1.0`.
        margin (float): Input margin. Defaults to `0.1`.

    Returns:
        Any: Result computed from the current inputs.
    """
    x_in   = np.asarray(x,dtype=np.float64).copy()
    x_min  = float(x_min)
    x_max  = float(x_max)
    margin = float(margin)

    if x_min >= x_max:
        raise ValueError("[soft_squash] require x_min<x_max")
    if margin <= 0.0:
        return np.clip(x_in,x_min,x_max)

    half_range = 0.5*(x_max-x_min)
    if margin > half_range:
        margin = half_range

    upper_th = x_max - margin
    lower_th = x_min + margin

    idxs_upper = np.where(x_in>upper_th)
    x_in[idxs_upper] = soft_tanh_margin(x_in[idxs_upper]-upper_th,margin) + upper_th

    idxs_lower = np.where(x_in<lower_th)
    x_in[idxs_lower] = soft_tanh_margin(x_in[idxs_lower]-lower_th,margin) + lower_th

    # Return
    return x_in # shape: same as x

def soft_squash_multidim(x,x_min,x_max,margin=0.1,dim_axis=-1):
    """
    Softly squash each dimension of x into [x_min,x_max] with given margin.

    Parameters:
        x (Any): Input x.
        x_min (Any): Input x min.
        x_max (Any): Input x max.
        margin (float): Input margin. Defaults to `0.1`.
        dim_axis (Any): Input dim axis. Defaults to `-1`.

    Returns:
        Any: Result computed from the current inputs.
    """
    x_in = np.asarray(x,dtype=np.float64)
    if x_in.ndim < 1:
        raise ValueError("[soft_squash_multidim] require x.ndim>=1")

    dim_axis = int(dim_axis)
    if dim_axis < 0:
        dim_axis = x_in.ndim + dim_axis

    d = int(x_in.shape[dim_axis])

    x_min_arr = np.asarray(x_min,dtype=np.float64)
    x_max_arr = np.asarray(x_max,dtype=np.float64)

    if x_min_arr.ndim == 0:
        x_min_arr = np.full((d,),float(x_min_arr))
    if x_max_arr.ndim == 0:
        x_max_arr = np.full((d,),float(x_max_arr))

    if (x_min_arr.shape[0] != d) or (x_max_arr.shape[0] != d):
        raise ValueError("[soft_squash_multidim] require x_min,x_max shape (d,)")

    x_perm  = np.moveaxis(x_in,dim_axis,-1)          # (...,d)
    x_out   = x_perm.copy()

    for k in range(d):
        x_out[...,k] = soft_squash(x=x_out[...,k],x_min=x_min_arr[k],x_max=x_max_arr[k],margin=margin)

    x_out = np.moveaxis(x_out,-1,dim_axis)

    # Return
    return x_out # shape: same as x

def hex_to_rgba(hex_color: str, alpha: float = 1.0):
    """
    Convert a hex color string to an RGBA color string.

    Parameters:
        hex_color (str): Input hex color.
        alpha (float): Input alpha. Defaults to `1.0`.

    Returns:
        Any: Result computed from the current inputs.
    """
    hex_color = hex_color.lstrip("#")
    # Extract RGB components
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return [r, g, b, alpha]

def farthest_point_sampling(pcd, n_sample, seed=None, rng=None):
    """
    Perform farthest point sampling on a point cloud.

    Parameters:
        pcd (Any): Input pcd.
        n_sample (int): Input n sample.
        seed (Any | None): Input seed. Defaults to `None`.
        rng (Any | None): Input rng. Defaults to `None`.

    Returns:
        Any: Result computed from the current inputs.
    """

    assert len(pcd.shape) == 2, "Input pcd must be a 2D array."
    N, D = pcd.shape

    # If the number of points is less than or equal to the number of samples, include all points and sample additional ones
    if n_sample >= N:
        rng = np.random.default_rng(seed) if rng is None else rng
        additional_indices = rng.choice(N, n_sample - N, replace=True)
        return np.concatenate([np.arange(N, dtype=np.int32), additional_indices])

    # Initialize arrays
    sampled_indices = []  # List to store the indices of sampled points
    distances = np.full(N,np.inf,dtype=np.float64)  # squared distances

    # Randomly pick the first point
    rng = np.random.default_rng(seed) if rng is None else rng
    first_index = int(rng.integers(0,N))
    sampled_indices.append(first_index)

    for _ in range(n_sample - 1):
        # Get the last sampled point
        last_sampled_point = pcd[sampled_indices[-1]].reshape(1, -1)

        # Compute distances from the last sampled point to all other points
        diff = pcd-last_sampled_point
        current_distances = np.einsum("ij,ij->i",diff,diff)

        # Update the minimum distances for each point
        distances = np.minimum(distances, current_distances)

        # Select the farthest point from the already sampled points
        next_index = np.argmax(distances)
        sampled_indices.append(next_index)

    return np.array(sampled_indices, dtype=np.int32)

def get_ip_address():
    """
    Get the local IP address of the machine.

    Parameters:
        None.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    system_name = platform.system()

    if system_name == "Darwin":  # macOS
        result = subprocess.run(["ipconfig", "getifaddr", "en0"], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    elif system_name == "Linux":  # Ubuntu or other Linux
        result = subprocess.run(["hostname", "-I"], capture_output=True, text=True)
        if result.returncode != 0:
            return None
        ip_list = result.stdout.strip().split()
        return ip_list[0] if ip_list else None

    else:
        print(f"Unsupported system: {system_name}")
        return None

class ZMQPubSub(object):
    """
    A class for ZeroMQ publisher and subscriber sockets.

    Usage:
        obj = ZMQPubSub(...)
    """
    def __init__(
            self,
            role     = "pub",
            ip       = None,
            port     = 5556,
            bind_ip  = "0.0.0.0",
            hwm      = 1,
            conflate = True,
            verbose  = True,
        ):
        """
        Initialize ZMQPubSub.

        Parameters:
            role (str): Input role. Defaults to `'pub'`.
            ip (Any | None): Input ip. Defaults to `None`.
            port (int): Input port. Defaults to `5556`.
            bind_ip (str): Input bind ip. Defaults to `'0.0.0.0'`.
            hwm (int): Input hwm. Defaults to `1`.
            conflate (bool): Input conflate. Defaults to `True`.
            verbose (bool): Input verbose. Defaults to `True`.

        Returns:
            None: Initializes the instance in place.
        """
        role = role.lower()
        assert role in ("pub", "sub"), "role must be 'pub' or 'sub'"
        self.context  = zmq.Context()
        self.socket   = None
        self.role     = role
        self.port     = port
        if ip is None: ip = get_ip_address()
        self.ip       = ip
        self.bind_ip  = bind_ip
        self.hwm      = hwm
        self.conflate = conflate
        self.verbose  = verbose
        self._pending_msg = None  # subscriber internal buffer
        self._pub_closed  = False # <-- 추가: publisher 종료 신호 수신 여부

        if self.role == "pub":
            self._open_pub()
        elif self.role == "sub":
            self._open_sub()
        else:
            raise ValueError("Invalid role: [%s]" % (self.role))

        if self.verbose:
            print ("[ZMQ] role:[%s] ip:[%s] port:[%d] hwm:[%d] conflate:[%s]"%
                   (self.role, self.ip, self.port, self.hwm, str(self.conflate)))

    def _open_pub(self):
        """
        Open a ZMQ publisher socket.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.socket = self.context.socket(zmq.PUB)
        self.socket.set_hwm(self.hwm)
        self.socket.setsockopt(zmq.LINGER, 0) # prevent hanging on close
        endpoint = "tcp://%s:%d" % (self.bind_ip, self.port)
        self.socket.bind(endpoint)
        pid = os.getpid()
        print("[ZMQ Publisher] pid:[%d] endpoint:[%s]" % (pid, endpoint))

    def _open_sub(self):
        """
        Open a ZMQ subscriber socket.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.socket = self.context.socket(zmq.SUB)
        self.socket.set_hwm(self.hwm)
        self.socket.setsockopt(zmq.LINGER, 0) # prevent hanging on close
        if self.conflate:
            self.socket.setsockopt(zmq.CONFLATE, 1)  # 가장 최근 메시지만 유지
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")   # 모든 토픽 구독
        endpoint = "tcp://%s:%d" % (self.ip, self.port)
        self.socket.connect(endpoint)
        pid = os.getpid()
        print("[ZMQ Subscriber] pid:[%d] endpoint:[%s]" % (pid, endpoint))

    def send(self, obj):
        """
        Send a Python object (publisher only).

        Parameters:
            obj (Any): Input obj.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        assert self.role == "pub", "Socket is not a publisher."
        self.socket.send_pyobj(obj)

    def recv(self, nonblock=True):
        """
        Receive a Python object (subscriber only).

        Parameters:
            nonblock (bool): Input nonblock. Defaults to `True`.

        Returns:
            Any: Result computed from the current inputs.
        """
        assert self.role == "sub", "Socket is not a subscriber."
        if self._pending_msg is not None:
            msg = self._pending_msg
            self._pending_msg = None
            return msg
        try:
            flags = zmq.NOBLOCK if nonblock else 0
            msg = self.socket.recv_pyobj(flags=flags)
            if isinstance(msg,dict) and msg.get("__ctrl__")=="EXIT":
                # EXIT message is not returned to user
                self._pub_closed = True   # <-- 추가: 종료 플래그 세팅
                self._pending_msg = None
                return None
            return msg
        except zmq.Again:
            return None

    def is_pub_closed(self, nonblock=True):
        """
        Check if the publisher has sent a close signal.

        Parameters:
            nonblock (bool): Input nonblock. Defaults to `True`.

        Returns:
            bool: Boolean result for the requested condition.
        """
        assert self.role == "sub", "is_pub_closed() can only be used by subscriber."
        # 먼저 내부 플래그 확인
        if self._pub_closed:
            return True
        try:
            flags = zmq.NOBLOCK if nonblock else 0
            msg = self.socket.recv_pyobj(flags=flags)
            if isinstance(msg,dict) and msg.get("__ctrl__")=="EXIT":
                print("[ZMQ Subscriber] pid:[%d] Publisher closed detected."%(os.getpid()))
                self._pub_closed = True   # <-- 추가: 종료 플래그 세팅
                return True
            else:
                self._pending_msg = msg   # 일반 메시지는 버퍼에 저장
                return False
        except zmq.Again:
            return False

    def close(self):
        """
        Close the ZMQ socket and terminate the context.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if self.socket is not None:
            if self.role == "pub":
                try:
                    # send termination control message
                    self.socket.send_pyobj({"__ctrl__": "EXIT"})
                    print("[ZMQ Publisher] pid:[%d] sent close signal."%(os.getpid()))
                    time.sleep(0.01)  # <-- EXIT가 전달될 시간을 약간 확보
                except Exception as e:
                    print("[ZMQ Publisher] pid:[%d] failed to send close signal: %s"%(os.getpid(),str(e)))
            self.socket.close()
            self.socket = None
        if self.context is not None:
            self.context.term()
        pid = os.getpid()
        print("[ZMQ %s] pid:[%d] Socket closed." % (self.role.upper(), pid))

def repo_tree_safe_listdir(path):
    """
    Return sorted directory entries, or an empty list on failure.

    Parameters:
        path (str | Path): Input path.

    Returns:
        Any: Result computed from the current inputs.
    """
    try:
        return sorted(os.listdir(path))
    except Exception:
        return []

def repo_tree_keep_file(name,exts):
    """
    Return whether a filename has one of the requested extensions.

    Parameters:
        name (str): Input name.
        exts (Any): Input exts.

    Returns:
        Any: Result computed from the current inputs.
    """
    return any(name.lower().endswith(e) for e in exts)

def repo_tree_count_text_lines(path):
    """
    Count text lines in a file.

    Parameters:
        path (str | Path): Input path.

    Returns:
        Any: Result computed from the current inputs.
    """
    try:
        with open(path,"r",encoding="utf-8",errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        return None

def repo_tree_count_ipynb_source_lines(path):
    """
    Count source lines in notebook cells.

    Parameters:
        path (str | Path): Input path.

    Returns:
        Any: Result computed from the current inputs.
    """
    try:
        with open(path,"r",encoding="utf-8",errors="ignore") as f:
            nb = json.load(f)
        n = 0
        for cell in nb.get("cells",[]):
            src = cell.get("source","")
            if isinstance(src,list):
                n += sum(s.count("\n")+1 for s in src if isinstance(s,str) and s!="")
            elif isinstance(src,str) and src!="":
                n += src.count("\n")+1
        return n
    except Exception:
        return repo_tree_count_text_lines(path)

def repo_tree_format_int(value):
    """
    Format an integer with thousands separators for tree summaries.

    Parameters:
        value (int): Input value.

    Returns:
        str: Requested value computed from the current inputs.
    """
    return format(int(value),",d")

def repo_tree_visible_entries(path,exts):
    """
    Return visible directories and selected files.

    Parameters:
        path (str | Path): Input path.
        exts (Any): Input exts.

    Returns:
        Any: Result computed from the current inputs.
    """
    out = []
    for name in repo_tree_safe_listdir(path):
        if name.startswith(".") or name=="__pycache__":
            continue
        full = os.path.join(path,name)
        if os.path.isdir(full) or repo_tree_keep_file(name,exts):
            out.append(name)
    return out

def repo_tree_scan(path,exts,show_py_lines,totals):
    """
    Accumulate file counts and line counts.

    Parameters:
        path (str | Path): Input path.
        exts (Any): Input exts.
        show_py_lines (Any): Input show py lines.
        totals (Any): Input totals.

    Returns:
        None: This function mutates state, renders output, or performs side
              effects in place.
    """
    for name in repo_tree_safe_listdir(path):
        if name.startswith(".") or name=="__pycache__":
            continue
        full = os.path.join(path,name)
        if os.path.isdir(full):
            repo_tree_scan(full,exts,show_py_lines,totals)
        elif repo_tree_keep_file(name,exts):
            low = name.lower()
            if low.endswith(".py"):
                totals["py_files"] += 1
                if show_py_lines:
                    n = repo_tree_count_text_lines(full)
                    if n is not None:
                        totals["py_lines"] += n
            elif low.endswith(".ipynb"):
                totals["ipynb_files"] += 1
                if show_py_lines:
                    n = repo_tree_count_ipynb_source_lines(full)
                    if n is not None:
                        totals["ipynb_lines"] += n
            elif low.endswith(".cpp"):
                totals["cpp_files"] += 1
                if show_py_lines:
                    n = repo_tree_count_text_lines(full)
                    if n is not None:
                        totals["cpp_lines"] += n
            elif low.endswith(".hpp"):
                totals["hpp_files"] += 1
                if show_py_lines:
                    n = repo_tree_count_text_lines(full)
                    if n is not None:
                        totals["hpp_lines"] += n

def repo_tree_colorize(text,is_dir,top,top_colors):
    """
    Color a tree entry if its top-level folder has a configured color.

    Parameters:
        text (Any): Input text.
        is_dir (bool): Input is dir.
        top (Any): Input top.
        top_colors (Any): Input top colors.

    Returns:
        Any: Result computed from the current inputs.
    """
    color = top_colors.get(top,None)
    if color is None:
        return text
    return colored(text,color,attrs=["bold"]) if is_dir else colored(text,color)

def repo_tree_walk(path,exts,top_colors,show_py_lines,print_excludes,prefix="",top=None):
    """
    Print a repository tree.

    Parameters:
        path (str | Path): Input path.
        exts (Any): Input exts.
        top_colors (Any): Input top colors.
        show_py_lines (Any): Input show py lines.
        print_excludes (Any): Input print excludes.
        prefix (str): Input prefix. Defaults to `''`.
        top (Any | None): Input top. Defaults to `None`.

    Returns:
        None: This function mutates state, renders output, or performs side
              effects in place.
    """
    entries = repo_tree_visible_entries(path,exts)
    for i,name in enumerate(entries):
        full     = os.path.join(path,name)
        is_dir   = os.path.isdir(full)
        last     = (i==len(entries)-1)
        branch   = "└── " if last else "├── "
        curr_top = name if (top is None and is_dir) else top
        disp     = name

        if is_dir and (name in print_excludes):
            disp = "%s (collapsed)" % name

        elif not is_dir and show_py_lines:
            if name.lower().endswith(".py"):
                n = repo_tree_count_text_lines(full)
                if n is not None:
                    disp = "%s (%s lines)" % (name,repo_tree_format_int(n))
            elif name.lower().endswith(".ipynb"):
                n = repo_tree_count_ipynb_source_lines(full)
                if n is not None:
                    disp = "%s (%s lines)" % (name,repo_tree_format_int(n))
            elif name.lower().endswith(".cpp") or name.lower().endswith(".hpp"):
                n = repo_tree_count_text_lines(full)
                if n is not None:
                    disp = "%s (%s lines)" % (name,repo_tree_format_int(n))

        print(prefix+branch+repo_tree_colorize(disp,is_dir,curr_top,top_colors))

        if is_dir and (name not in print_excludes):
            repo_tree_walk(
                path           = full,
                exts           = exts,
                top_colors     = top_colors,
                show_py_lines  = show_py_lines,
                print_excludes = print_excludes,
                prefix         = prefix+("    " if last else "│   "),
                top            = curr_top,
            )

def print_repo_tree_extension(
        root           = ".",
        exts           = [".py",".ipynb",".cpp",".hpp"],
        top_colors     = None,
        show_py_lines  = True,
        print_excludes = None,
    ):
    """
    Print the directory tree of a repository with selected source extensions.

    Parameters:
        root (str): Input root. Defaults to `'.'`.
        exts (list): Input exts. Defaults to
            `['.py', '.ipynb', '.cpp', '.hpp']`.
        top_colors (Any | None): Input top colors. Defaults to `None`.
        show_py_lines (bool): Input show py lines. Defaults to `True`.
        print_excludes (Any | None): Input print excludes. Defaults to `None`.

    Returns:
        None: This function mutates state, renders output, or performs side
              effects in place.
    """
    top_colors        = {} if top_colors is None else top_colors
    print_excludes    = set() if print_excludes is None else set(print_excludes)
    root              = os.path.abspath(root)
    exts              = tuple(e.lower() if e.startswith(".") else "."+e.lower() for e in exts)
    totals            = {
        "py_files"    : 0,
        "ipynb_files" : 0,
        "cpp_files"   : 0,
        "hpp_files"   : 0,
        "py_lines"    : 0,
        "ipynb_lines" : 0,
        "cpp_lines"   : 0,
        "hpp_lines"   : 0,
    }

    repo_tree_scan(root,exts,show_py_lines,totals)

    print(".")
    repo_tree_walk(
        path           = root,
        exts           = exts,
        top_colors     = top_colors,
        show_py_lines  = show_py_lines,
        print_excludes = print_excludes,
    )

    print("")
    print("Summary")
    print("  #Python files:[%s]" % repo_tree_format_int(totals["py_files"]))
    print("  #Jupyter notebooks:[%s]" % repo_tree_format_int(totals["ipynb_files"]))
    print("  #C++ source files:[%s]" % repo_tree_format_int(totals["cpp_files"]))
    print("  #C++ header files:[%s]" % repo_tree_format_int(totals["hpp_files"]))
    if show_py_lines:
        print("  Total .py lines:[%s]" % repo_tree_format_int(totals["py_lines"]))
        print("  Total .ipynb lines:[%s]" % repo_tree_format_int(totals["ipynb_lines"]))
        print("  Total .py+.ipynb lines:[%s]" % repo_tree_format_int(totals["py_lines"]+totals["ipynb_lines"]))
        print("  Total .cpp lines:[%s]" % repo_tree_format_int(totals["cpp_lines"]))
        print("  Total .hpp lines:[%s]" % repo_tree_format_int(totals["hpp_lines"]))
        print("  Total .cpp+.hpp lines:[%s]" % repo_tree_format_int(totals["cpp_lines"]+totals["hpp_lines"]))

def list_exclude(a,b):
    """
    Return a list of items in a that are not in b.

    Parameters:
        a (Any): Input a.
        b (Any): Input b.

    Returns:
        list: Requested value computed from the current inputs.
    """
    b_set = set(b)
    return [x for x in a if x not in b_set]

def list_merge(a,b):
    """
    Return a list that starts with a, then appends items from b.

    Parameters:
        a (Any): Input a.
        b (Any): Input b.

    Returns:
        list: Requested value computed from the current inputs.
    """
    c    = list(a)
    seen = set(a)

    for x in b:
        if x not in seen:
            c.append(x)
            seen.add(x)

    return c

class SimpleTimer(object):
    """
    Simple fixed-rate timer for loop execution.

    Usage:
        obj = SimpleTimer(...)
    """
    def __init__(
            self,
            name      = "timer",
            Hz        = 10,
            max_sec   = np.inf,
            verbose   = True,
            sleep_sec = 0.0,
            first_run = True,
        ):
        """
        Initialize SimpleTimer.

        Parameters:
            name (str): Input name. Defaults to `'timer'`.
            Hz (int): Input Hz. Defaults to `10`.
            max_sec (Any): Input max sec. Defaults to `np.inf`.
            verbose (bool): Input verbose. Defaults to `True`.
            sleep_sec (float): Input sleep sec. Defaults to `0.0`.
            first_run (bool): If True, the first do_run() after start() always
                returns True. Defaults to `True`.

        Returns:
            None: Initializes the instance in place.
        """
        if Hz <= 0:
            raise ValueError("Hz must be positive.")

        self.name           = name
        self.Hz             = Hz
        self.dt             = 1.0/Hz
        self.max_sec        = max_sec
        self.verbose        = verbose
        self.sleep_sec      = float(sleep_sec)
        self.first_run      = bool(first_run)

        self.sec_period     = 1.0/self.Hz
        self.sec_next       = 0.0

        self.sec_elps       = 0.0
        self.sec_elps_prev  = 0.0
        self.sec_elps_diff  = 0.0
        self.sec_elps_loop  = None
        self.sec_elps_loop_prev = None

        self.tick           = 0
        self.force_finish   = False
        self.delayed_flag   = False
        self.first_run_flag = self.first_run

        self.time_start     = None
        self.time_run_start = None

        if self.verbose:
            print("[%s] Initialized with [%.1f]HZ. MAX_SEC:[%.1f]sec."
                  % (self.name,self.Hz,self.max_sec))

    def start(self):
        """
        Start the timer and reset runtime states.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.time_start     = time.perf_counter()
        self.time_run_start = None

        self.sec_next       = 0.0
        self.sec_elps       = 0.0
        self.sec_elps_prev  = 0.0
        self.sec_elps_diff  = 0.0
        self.sec_elps_loop  = None
        self.sec_elps_loop_prev = None

        self.tick           = 0
        self.force_finish   = False
        self.delayed_flag   = False
        self.first_run_flag = self.first_run

        if self.verbose:
            print("[%s] Start ([%.1f]HZ. MAX_SEC:[%.1f]sec)."
                  % (self.name,self.Hz,self.max_sec))

    def finish(self):
        """
        Force finish the timer.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.force_finish = True

    def is_finished(self):
        """
        Check whether the timer has finished.

        Parameters:
            None.

        Returns:
            bool: Boolean result for the requested condition.
        """
        if self.force_finish:
            return True

        self.sec_elps = time.perf_counter() - self.time_start
        return (self.sec_elps > self.max_sec)

    def is_notfinished(self):
        """
        Check whether the timer is still running.

        Parameters:
            None.

        Returns:
            bool: Boolean result for the requested condition.
        """
        return not self.is_finished()

    def do_run(self):
        """
        Check whether this tick should run now.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        if self.sleep_sec > 0.0:
            time.sleep(self.sleep_sec)
        self.sec_elps = time.perf_counter() - self.time_start

        if self.first_run_flag:
            self.first_run_flag = False
            self.sec_next       = self.sec_elps + self.sec_period
            self.tick           = self.tick + 1

            self.sec_elps_diff = self.sec_elps - self.sec_elps_prev
            self.sec_elps_prev = self.sec_elps
            self.delayed_flag  = False

            self.time_run_start = time.perf_counter()
            return True

        if self.sec_elps > self.sec_next:
            # Intentionally keep catch-up behavior:
            # advance only by one period even if we were delayed.
            self.sec_next      = self.sec_next + self.sec_period
            self.tick          = self.tick + 1

            self.sec_elps_diff = self.sec_elps - self.sec_elps_prev
            self.sec_elps_prev = self.sec_elps

            if self.sec_elps_diff > self.sec_period*1.1:
                self.delayed_flag = True
                if self.verbose:
                    run_ms = self.get_elapsed_run_time("ms",default=0.0)
                    print("[%s][%d][%.1fs] delayed! T:[%.1f]ms interval:[%.1f]ms prev_run:[%.1f]ms."
                          % (self.name,
                             self.tick,
                             self.sec_elps,
                             self.sec_period*1000.0,
                             self.sec_elps_diff*1000.0,
                             run_ms))
            else:
                self.delayed_flag = False

            # Start measuring only this block's execution time.
            # If end() is not called, scheduling still works fine.
            self.time_run_start = time.perf_counter()
            return True

        self.delayed_flag = False
        return False

    def end(self):
        """
        Mark the end of the current run block and record its execution time.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        if self.time_run_start is None:
            return None

        self.sec_elps_loop_prev = self.sec_elps_loop
        self.sec_elps_loop = time.perf_counter() - self.time_run_start

        if (self.sec_elps_loop > self.sec_period) and (self.Hz != 0):
            if self.verbose:
                print("[%s][%d] run delayed! T:[%.1f]ms run:[%.1f]ms]"
                    % (self.name,self.tick,self.sec_period*1000.0,self.sec_elps_loop*1000.0))

        return self.sec_elps_loop

    def get_elapsed_run_time(self,unit="s",default=0.0):
        """
        Get the most recently recorded execution time between do_run() and end().

        Parameters:
            unit (str): Input unit. Defaults to `'s'`.
            default (float): Input default. Defaults to `0.0`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        sec_elps_loop = self.sec_elps_loop
        if sec_elps_loop is None:
            sec_elps_loop = self.sec_elps_loop_prev
        if sec_elps_loop is None:
            return default
        if unit == "s":
            return sec_elps_loop
        elif unit == "ms":
            return sec_elps_loop * 1000.0
        elif unit == "us":
            return sec_elps_loop * 1e6
        else:
            raise ValueError("Unsupported unit: %s" % unit)

    def get_ems(self):
        """
        Get recent execution time in milliseconds.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        return self.get_elapsed_run_time("ms")

    def get_tms(self):
        """
        Get target timer period in milliseconds.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        return 1000.0/self.Hz

    def get_Hz(self):
        """
        Get target timer frequency in Hz.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        return self.Hz

    def get_lag_info(self):
        """
        Return how far this timer is behind its ideal fixed-rate schedule.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if self.time_start is None:
            return {"tick":0,"ms":0.0,"sec":0.0}
        sec_elps = time.perf_counter() - self.time_start
        # do_run() schedules the first runnable tick at start time, so include tick 0 in the ideal count.
        expected_tick = int(np.floor(sec_elps / self.sec_period)) + 1
        lag_tick = max(0,expected_tick - self.tick)
        lag_sec = lag_tick * self.sec_period
        return {"tick":lag_tick,"ms":1000.0*lag_sec,"sec":lag_sec}

def deadzone(x,th=0.05,center=0.0,rescale=False):
    """
    Apply deadzone to a scalar analog input.

    Parameters:
        x (Any): Input x.
        th (float): Input th. Defaults to `0.05`.
        center (float): Input center. Defaults to `0.0`.
        rescale (bool): Input rescale. Defaults to `False`.

    Returns:
        Any: Result computed from the current inputs.
    """
    x            = float(x)
    th           = float(th)
    center = float(center)

    # Distance from the chosen center
    dx     = x-center
    dx_abs = abs(dx)

    # Inside deadzone -> snap to center
    if dx_abs < th:
        return center

    # Outside deadzone,keep original value if no rescaling is requested
    if not rescale:
        return x

    # Rescale outside the deadzone back to the original range.
    # This implementation assumes the input range is [-1,1].
    if dx >= 0.0:
        span = 1.0-center
        if span <= th:
            return center
        dx_out = (dx_abs-th)/(span-th)*span
        return center+dx_out
    else:
        span = center-(-1.0)
        if span <= th:
            return center
        dx_out = (dx_abs-th)/(span-th)*span
        return center-dx_out

# ------------------------------------------------------------------------
