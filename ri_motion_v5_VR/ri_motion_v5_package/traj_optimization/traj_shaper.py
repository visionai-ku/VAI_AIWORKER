import time
import warnings
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# from traj_utils import (
#     get_A_vel_acc_jerk_same_length,
# )
from ri_motion_v5_package.traj_optimization.traj_utils import (
    get_A_vel_acc_jerk_same_length,
)

def try_cvxpy_solver(prob,name,solver_kwargs):
    """
    Try one cvxpy solver name and return whether it solved the problem.

    Parameters:
        prob (Any): Input prob.
        name (str): Input name.
        solver_kwargs (Any): Input solver kwargs.

    Returns:
        Any: Result computed from the current inputs.
    """
    if not isinstance(name,str):
        return False
    name = name.upper()
    if not hasattr(cp,name):
        return False
    try:
        prob.solve(solver=getattr(cp,name),**solver_kwargs)
        return prob.status in ("optimal","optimal_inaccurate")
    except Exception:
        return False

def max_pos_lower_violation(z,x_min):
    """
    Return max lower-bound position violation.

    Parameters:
        z (Any): Input z.
        x_min (Any): Input x min.

    Returns:
        Any: Result computed from the current inputs.
    """
    if x_min is None:
        return 0.0
    return float(np.maximum(0.0,x_min - z).max())

def max_pos_upper_violation(z,x_max):
    """
    Return max upper-bound position violation.

    Parameters:
        z (Any): Input z.
        x_max (Any): Input x max.

    Returns:
        Any: Result computed from the current inputs.
    """
    if x_max is None:
        return 0.0
    return float(np.maximum(0.0,z - x_max).max())

def max_abs_limit_violation(x,x_max):
    """
    Return max absolute-limit violation.

    Parameters:
        x (Any): Input x.
        x_max (Any): Input x max.

    Returns:
        Any: Result computed from the current inputs.
    """
    if x_max is None:
        return 0.0
    return float((np.abs(x) - x_max).clip(min=0).max())

def endpoint_value_violation(x,value,idx):
    """
    Return endpoint equality violation.

    Parameters:
        x (Any): Input x.
        value (Any): Input value.
        idx (int): Input idx.

    Returns:
        Any: Result computed from the current inputs.
    """
    if value is None:
        return 0.0
    return float(abs(x[idx] - float(value)))

def get_traj_plot_colors(k=4):
    """
    Return distinct plotting colors.

    Parameters:
        k (int): Input k. Defaults to `4`.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    cmap = plt.get_cmap("tab10")
    return [cmap(i) for i in range(k)]

def as_array_or_none(val,n):
    """
    Convert scalar or vector plotting input into a flat array.

    Parameters:
        val (Any): Input val.
        n (int): Input n.

    Returns:
        Any: Result computed from the current inputs.
    """
    if val is None:
        return None
    arr = np.asarray(val)
    if arr.shape == ():
        return np.full(n,float(arr))
    return arr.ravel()

def shade_remain_segments(ax,t,remain_segs,remain_val):
    """
    Shade remain segments in a trajectory plot.

    Parameters:
        ax (Any): Input ax.
        t (Any): Input t.
        remain_segs (Any): Input remain segs.
        remain_val (Any): Input remain val.

    Returns:
        None: This function mutates state, renders output, or performs side
              effects in place.
    """
    if not remain_segs:
        return
    seg_labeled = False
    anchor_labeled = False
    for seg in remain_segs:
        idx = np.asarray(seg,dtype=int).ravel()
        if idx.size == 0:
            continue
        t0,t1 = t[idx[0]],t[idx[-1]]
        ax.axvspan(t0,t1,alpha=0.10,label=None if seg_labeled else "remain segment")
        seg_labeled = True
        if remain_val is not None:
            ax.hlines(
                remain_val,t0,t1,
                linestyles = "dashed",
                linewidth  = 1.0,
                label      = None if anchor_labeled else "remain anchor",
            )
            anchor_labeled = True

def traj_1d_shaper(
        t: np.ndarray,
        x_ref: np.ndarray,
        *,
        # Constraints (None → disabled)
        x_min: float | None = None,
        x_max: float | None = None,
        v_max: float | None = None,
        a_max: float | None = None,
        idxs_eq: np.ndarray | None = None,
        vals_eq: np.ndarray | None = None,
        hard_eq: bool = False,
        # New: endpoint position constraints (None → disabled)
        x_init: float | None = None,
        x_final: float | None = None,
        # Endpoint velocity constraints (None → disabled)
        v_init: float | None = None,
        v_final: float | None = None,
        # Hyperparameters
        p_norm: int = 2,
        lambda_j: float = 1e-6,
        rho_eq: float = 1e4,
        # Solver options
        prefer_solver: str | None = "CLARABEL",
        solver_kwargs: dict | None = None,
        # Verbosity
        verbose: bool = False,
    ):
    """
    Given a 1D reference trajectory x_ref sampled at times t, optimize a new
    trajectory z that tracks x_ref.

    Parameters:
        t (np.ndarray): Input t.
        x_ref (np.ndarray): Input x ref.
        x_min (float | None): Input x min. Defaults to `None`.
        x_max (float | None): Input x max. Defaults to `None`.
        v_max (float | None): Input v max. Defaults to `None`.
        a_max (float | None): Input a max. Defaults to `None`.
        idxs_eq (np.ndarray | None): Input idxs eq. Defaults to `None`.
        vals_eq (np.ndarray | None): Input vals eq. Defaults to `None`.
        hard_eq (bool): Input hard eq. Defaults to `False`.
        x_init (float | None): Input x init. Defaults to `None`.
        x_final (float | None): Input x final. Defaults to `None`.
        v_init (float | None): Input v init. Defaults to `None`.
        v_final (float | None): Input v final. Defaults to `None`.
        p_norm (int): Input p norm. Defaults to `2`.
        lambda_j (float): Input lambda j. Defaults to `1e-06`.
        rho_eq (float): Input rho eq. Defaults to `10000.0`.
        prefer_solver (str | None): Input prefer solver. Defaults to `'CLARABEL'`.
        solver_kwargs (dict | None): Input solver kwargs. Defaults to `None`.
        verbose (bool): Input verbose. Defaults to `False`.

    Returns:
        Any: Result computed from the current inputs.
    """
    time_start = time.time()

    # --- Input checks ---
    t     = np.asarray(t).ravel()
    x_ref = np.asarray(x_ref).ravel()
    if t.shape != x_ref.shape:
        raise ValueError("t and x_ref must have the same shape.")
    n = len(t)
    if n < 2:
        raise ValueError("t must have at least 2 samples.")

    # Robust dt checks (prevents inf/overflow in finite differences)
    dts = np.diff(t)
    if not np.all(np.isfinite(dts)) or not np.all(dts > 0.0):
        raise ValueError("t must be strictly increasing and finite.")
    dt_min = float(np.min(dts))
    dt_max = float(np.max(dts))
    if dt_min < 1e-9:
        raise ValueError("dt too small (min_dt=%g)." % dt_min)
    if (dt_max/dt_min) > 1.01:
        raise ValueError("t not uniform (min_dt=%g,max_dt=%g). Resample or use non-uniform operators." % (dt_min,dt_max))
    dt = float(np.mean(dts))

    if (idxs_eq is None) ^ (vals_eq is None):
        raise ValueError("idxs_eq and vals_eq must be provided together or both None.")
    if idxs_eq is not None:
        idxs_eq = np.asarray(idxs_eq,dtype=int).ravel()
        vals_eq = np.asarray(vals_eq).ravel()
        if idxs_eq.size != vals_eq.size:
            raise ValueError("idxs_eq and vals_eq must have the same length.")
        if idxs_eq.size > 0 and (idxs_eq.min() < 0 or idxs_eq.max() >= n):
            raise IndexError("idxs_eq contains out-of-range indices.")

    # --- Difference operators and reference diffs (for plots) ---
    A_vel,A_acc,A_jerk = get_A_vel_acc_jerk_same_length(n,dt)
    v_ref = A_vel@x_ref
    a_ref = A_acc@x_ref
    j_ref = A_jerk@x_ref

    # --- Optimization variables & objective ---
    z = cp.Variable(n)

    # Exclude equality indices from tracking term
    if idxs_eq is not None and idxs_eq.size > 0:
        track_mask = np.ones(n,dtype=float)
        track_mask[idxs_eq] = 0.0
    else:
        track_mask = np.ones(n,dtype=float)

    if p_norm == 1:
        obj_terms = [cp.norm1(cp.multiply(track_mask,z - x_ref))]
    elif p_norm == 2:
        obj_terms = [cp.sum_squares(cp.multiply(track_mask,z - x_ref))]
    else:
        raise ValueError("p_norm must be 1 or 2.")
    obj_terms.append(lambda_j*cp.sum_squares(A_jerk@z))

    constraints = []

    # Position bounds
    if x_min is not None:
        constraints.append(z >= float(x_min))
    if x_max is not None:
        constraints.append(z <= float(x_max))

    # New: endpoint position constraints
    if x_init is not None:
        constraints.append(z[0] == float(x_init))
    if x_final is not None:
        constraints.append(z[-1] == float(x_final))

    # Equality (hard or soft)
    if idxs_eq is not None and idxs_eq.size > 0:
        if hard_eq:
            constraints.append(z[idxs_eq] == vals_eq)
        else:
            s_eq = cp.Variable(idxs_eq.size)
            constraints.append(z[idxs_eq] + s_eq == vals_eq)
            obj_terms.append(rho_eq*cp.sum_squares(s_eq))

    # Velocity / acceleration limits
    if v_max is not None:
        constraints.append(cp.abs(A_vel@z) <= float(v_max))
    if a_max is not None:
        constraints.append(cp.abs(A_acc@z) <= float(a_max))

    # Endpoint velocity constraints
    if v_init is not None:
        constraints.append((A_vel@z)[0] == float(v_init))
    if v_final is not None:
        constraints.append((A_vel@z)[-1] == float(v_final))

    prob = cp.Problem(cp.Minimize(sum(obj_terms)),constraints)
    solver_kwargs = {} if solver_kwargs is None else dict(solver_kwargs)

    solved = False
    status = None
    if isinstance(prefer_solver,str):
        solved = try_cvxpy_solver(prob,prefer_solver,solver_kwargs)
        status = prob.status

    if not solved:
        for cand in ("CLARABEL","OSQP","ECOS","SCS"):
            if prefer_solver and cand == prefer_solver.upper():
                continue
            if try_cvxpy_solver(prob,cand,solver_kwargs):
                solved = True
                status = prob.status
                break

    if not solved:
        raise RuntimeError("Optimization failed (status=%s)." % prob.status)

    # --- Retrieve solution ---
    z_opt = z.value
    v_opt = A_vel@z_opt
    a_opt = A_acc@z_opt
    j_opt = A_jerk@z_opt

    # --- Violations ---
    viol_lb     = max_pos_lower_violation(z_opt,x_min)
    viol_ub     = max_pos_upper_violation(z_opt,x_max)
    viol_v      = max_abs_limit_violation(v_opt,v_max)
    viol_a      = max_abs_limit_violation(a_opt,a_max)
    viol_x_init = endpoint_value_violation(z_opt,x_init,0)
    viol_x_fin  = endpoint_value_violation(z_opt,x_final,-1)
    viol_v_init = endpoint_value_violation(v_opt,v_init,0)
    viol_v_fin  = endpoint_value_violation(v_opt,v_final,-1)

    time_elapsed = time.time() - time_start
    if verbose:
        print("Optimization results:")
        print("  It took [%.2f]ms for solving [%d] variables"%(time_elapsed*1000,n))
        print("  Status: [%s]" % status)
        print("  Optimal objective: [%.6f]" % float(prob.value))
        print("  max LB viol: [%.3e], max UB viol: [%.3e]" % (viol_lb,viol_ub))
        print("  max |v| viol: [%.3e], max |a| viol: [%.3e]" % (viol_v,viol_a))
        if x_init is not None:
            print("  x_init target: [%.6f], achieved: [%.6f], abs err: [%.3e]" % (float(x_init),float(z_opt[0]),viol_x_init))
        if x_final is not None:
            print("  x_final target: [%.6f], achieved: [%.6f], abs err: [%.3e]" % (float(x_final),float(z_opt[-1]),viol_x_fin))
        if v_init is not None:
            print("  v_init target: [%.6f], achieved: [%.6f], abs err: [%.3e]" % (float(v_init),float(v_opt[0]),viol_v_init))
        if v_final is not None:
            print("  v_final target: [%.6f], achieved: [%.6f], abs err: [%.3e]" % (float(v_final),float(v_opt[-1]),viol_v_fin))

    # --- Pack arrays for plotting convenience ---
    if x_min is not None:
        idxs_lb = np.arange(n)
        vals_lb = np.full(n,float(x_min))
    else:
        idxs_lb,vals_lb = None,None

    if x_max is not None:
        idxs_ub = np.arange(n)
        vals_ub = np.full(n,float(x_max))
    else:
        idxs_ub,vals_ub = None,None

    return {
        # core
        "status": status,
        "objective": float(prob.value),
        "t": t,
        "x_ref": x_ref,
        "z": z_opt,
        # bounds
        "x_min": x_min,"x_max": x_max,
        "v_max": v_max,"a_max": a_max,
        "idxs_lb": idxs_lb,"vals_lb": vals_lb,
        "idxs_ub": idxs_ub,"vals_ub": vals_ub,
        # equality
        "idxs_eq": idxs_eq,"vals_eq": vals_eq,"hard_eq": hard_eq,
        # endpoint targets
        "x_init": x_init,"x_final": x_final,
        "v_init": v_init,"v_final": v_final,
        # operators and signals
        "A_vel": A_vel,"A_acc": A_acc,"A_jerk": A_jerk,
        "v_ref": v_ref,"a_ref": a_ref,"j_ref": j_ref,
        "v": v_opt,"a": a_opt,"j": j_opt,
        # penalties
        "lambda_j": lambda_j,"rho_eq": rho_eq,
        # violations
        "violations": {
            "pos_lb": viol_lb,
            "pos_ub": viol_ub,
            "vel":    viol_v,
            "acc":    viol_a,
            "x_init": viol_x_init,
            "x_final": viol_x_fin,
            "v_init": viol_v_init,
            "v_final": viol_v_fin,
        },
    }


def plot_traj_1d_shaping_result(
        res: dict,
        plot_vel: bool  = True,
        plot_acc: bool  = True,
        plot_jerk: bool = True,
        figsize         = (6,3),
    ):
    """
    Visualize the output of `traj_1d_shaper`.

    Parameters:
        res (dict): Input res.
        plot_vel (bool): Input plot vel. Defaults to `True`.
        plot_acc (bool): Input plot acc. Defaults to `True`.
        plot_jerk (bool): Input plot jerk. Defaults to `True`.
        figsize (tuple): Input figsize. Defaults to `(6, 3)`.

    Returns:
        None: This function mutates state, renders output, or performs side
              effects in place.
    """
    # --- Unpack (with safety for optional fields) ---
    t       = res["t"]
    x_ref   = res["x_ref"]
    z_opt   = res["z"]

    idxs_lb = res.get("idxs_lb", None)
    vals_lb = res.get("vals_lb", None)
    idxs_ub = res.get("idxs_ub", None)
    vals_ub = res.get("vals_ub", None)

    idxs_eq = res.get("idxs_eq", None)
    vals_eq = res.get("vals_eq", None)

    v_ref   = res.get("v_ref", None)
    a_ref   = res.get("a_ref", None)
    j_ref   = res.get("j_ref", None)

    v_opt   = res.get("v", None)
    a_opt   = res.get("a", None)
    j_opt   = res.get("j", None)

    v_max   = res.get("v_max", None)
    a_max   = res.get("a_max", None)
    x_min   = res.get("x_min", None)
    x_max   = res.get("x_max", None)

    lambda_j = res.get("lambda_j", None)

    colors = get_traj_plot_colors(4)

    # 1) Position
    fig, ax = plt.subplots(figsize=figsize)

    # draw low-priority items first (bounds, targets)
    if idxs_lb is not None and vals_lb is not None:
        ax.plot(t[idxs_lb], vals_lb, '--', color=colors[0], lw=1.0,
                label='lower bounds', alpha=0.7, zorder=1)

    if idxs_ub is not None and vals_ub is not None:
        ax.plot(t[idxs_ub], vals_ub, '--', color=colors[1], lw=1.0,
                label='upper bounds', alpha=0.7, zorder=1)

    if idxs_eq is not None and vals_eq is not None and len(idxs_eq) > 0:
        # plot eq targets with lower zorder than main lines
        ax.plot(t[idxs_eq], vals_eq, 'o', color=colors[2], ms=3, mfc='none', mew=0.5, alpha=0.5,
                label='equality targets', zorder=2)
        ax.plot(t[idxs_eq], x_ref[idxs_eq], 'o', color=colors[3], ms=3, mfc='none', mew=0.5, alpha=0.5,
                label='points to be modified', zorder=2)

    # draw high-priority items last (main lines on top)
    ref_line, = ax.plot(t, x_ref, '-', color='k', lw=0.5, label='reference', zorder=9)
    opt_line, = ax.plot(t, z_opt, '-', color='b', lw=1.5, label='optimized', zorder=10)

    ax.set_title('Trajectory', fontsize=9)
    ax.set_xlabel('Time [s]', fontsize=8)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    # 2) Velocity
    if plot_vel:
        if v_ref is not None and v_opt is not None:
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(t, v_ref, '-', color='k', lw=0.5, label='velocity (ref)')
            ax.plot(t, v_opt, '-', color='b', lw=1.5, label='velocity (opt)')
            if v_max is not None:
                ax.fill_between(t, -v_max, v_max, alpha=0.08, label='±v_max band')
                ax.hlines([v_max, -v_max], t[0], t[-1], colors=['r','r'], linestyles='dashed', lw=1)
            ax.set_title('Velocity constraint', fontsize=9)
            ax.set_xlabel('Time [s]', fontsize=8)
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
            plt.tight_layout(); plt.show()

    # 3) Acceleration
    if plot_acc:
        if a_ref is not None and a_opt is not None:
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(t, a_ref, '-', color='k', lw=0.5, label='acceleration (ref)')
            ax.plot(t, a_opt, '-', color='b', lw=1.5, label='acceleration (opt)')
            if a_max is not None:
                ax.fill_between(t, -a_max, a_max, alpha=0.08, label='±a_max band')
                ax.hlines([a_max, -a_max], t[0], t[-1], colors=['r','r'], linestyles='dashed', lw=1)
            ax.set_title('Acceleration constraint', fontsize=9)
            ax.set_xlabel('Time [s]', fontsize=8)
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
            plt.tight_layout(); plt.show()

    # 4) Jerk
    if plot_jerk:
        if j_ref is not None and j_opt is not None:
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(t, j_ref, '-', color='k', lw=0.5, label='jerk (ref)')
            ax.plot(t, j_opt, '-', color='b', lw=1.5, label='jerk (opt)')
            if lambda_j is not None:
                ax.text(0.01, 0.95, f'jerk reg λ={lambda_j:g}', transform=ax.transAxes,
                        ha='left', va='top', fontsize=8)
            ax.set_title('Jerk (for smoothness)', fontsize=9)
            ax.set_xlabel('Time [s]', fontsize=8)
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
            plt.tight_layout(); plt.show()


def traj_1d_remainer(
        secs: np.ndarray,
        x_ref: np.ndarray,
        remain_segs: list[np.ndarray],
        *,
        v_max: float | None = None,
        a_max: float | None = None,
        p_norm: int = 2,
        lambda_j: float = 1e-6,
        rho_remain: float = 1e4,
        remain_val: float | None = None,   # None → piecewise-constant
        prefer_solver: str | None = "CLARABEL",
        solver_kwargs: dict | None = None,
        verbose: bool = False,
    ):
    """
    This function is called "remainer" because it encourages specified segments of
    the trajectory to "remain" constant.

    Parameters:
        secs (np.ndarray): Input secs.
        x_ref (np.ndarray): Input x ref.
        remain_segs (list[np.ndarray]): Input remain segs.
        v_max (float | None): Input v max. Defaults to `None`.
        a_max (float | None): Input a max. Defaults to `None`.
        p_norm (int): Input p norm. Defaults to `2`.
        lambda_j (float): Input lambda j. Defaults to `1e-06`.
        rho_remain (float): Input rho remain. Defaults to `10000.0`.
        remain_val (float | None): Input remain val. Defaults to `None`.
        prefer_solver (str | None): Input prefer solver. Defaults to `'CLARABEL'`.
        solver_kwargs (dict | None): Input solver kwargs. Defaults to `None`.
        verbose (bool): Input verbose. Defaults to `False`.

    Returns:
        Any: Result computed from the current inputs.
    """
    time_start = time.time()

    secs  = np.asarray(secs).ravel()
    x_ref = np.asarray(x_ref).ravel()
    if secs.shape != x_ref.shape:
        raise ValueError("secs and x_ref must have the same shape.")
    n = len(secs)
    if n < 2:
        raise ValueError("Need at least 2 samples.")
    dt = float(secs[1] - secs[0])

    # Difference operators
    A_vel,A_acc,A_jerk = get_A_vel_acc_jerk_same_length(n,dt)

    # First-difference matrix (n-1,n): (z[i+1]-z[i]) without /dt (penalty only)
    D1 = np.zeros((n-1,n))
    for i in range(n-1):
        D1[i,i]   = -1.0
        D1[i,i+1] = +1.0

    # Decision var
    z = cp.Variable(n)

    # Cache expressions (smaller DCP graph)
    vz = A_vel@z
    az = A_acc@z
    jz = A_jerk@z

    # Objective: tracking + jerk
    if p_norm == 1:
        obj_terms = [cp.norm1(z - x_ref)]
    elif p_norm == 2:
        obj_terms = [cp.sum_squares(z - x_ref)]
    else:
        raise ValueError("p_norm must be 1 or 2.")
    obj_terms.append(lambda_j*cp.sum_squares(jz))

    cons = []
    if v_max is not None:
        cons.append(cp.abs(vz) <= float(v_max))
    if a_max is not None:
        cons.append(cp.abs(az) <= float(a_max))

    # remain segments (soft penalties only; no extra vars/constraints)
    if remain_segs:
        for seg in remain_segs:
            idx = np.asarray(seg,dtype=int).ravel()
            if idx.size == 0:
                continue

            t0 = int(idx[0])
            t1 = int(idx[-1])

            if not (0 <= t0 <= t1 < n):
                raise IndexError("remain segment out of range: (%d,%d) with n=%d" % (t0,t1,n))

            if remain_val is None:
                # piecewise-constant encouragement inside [t0,t1]:
                # penalize first differences z[t0+1]-z[t0],...,z[t1]-z[t1-1]
                if t1 - t0 >= 1:
                    rows  = np.arange(t0,t1)     # D1 rows corresponding to diffs within segment
                    D_seg = D1[rows,:]           # (t1-t0,n)
                    obj_terms.append(rho_remain*cp.sum_squares(D_seg@z))
            else:
                # stay near remain_val inside [t0,t1]
                idxs = np.arange(t0,t1+1,dtype=int)
                obj_terms.append(rho_remain*cp.sum_squares(z[idxs] - float(remain_val)))

    # Solve (fallback chain)
    prob = cp.Problem(cp.Minimize(sum(obj_terms)),cons)
    solver_kwargs = {} if solver_kwargs is None else dict(solver_kwargs)

    solved = False
    if isinstance(prefer_solver,str):
        solved = try_cvxpy_solver(prob,prefer_solver,solver_kwargs)

    if not solved:
        for cand in ("CLARABEL","OSQP","ECOS","SCS"):
            if isinstance(prefer_solver,str) and cand == prefer_solver.upper():
                continue
            if try_cvxpy_solver(prob,cand,solver_kwargs):
                solved = True
                break

    if prob.status not in ("optimal","optimal_inaccurate"):
        raise RuntimeError("Optimization failed (status=%s)." % prob.status)

    z_opt = np.asarray(z.value).ravel()

    # Signals
    v_opt = (A_vel@z_opt)
    a_opt = (A_acc@z_opt)
    j_opt = (A_jerk@z_opt)

    v_ref = (A_vel@x_ref)
    a_ref = (A_acc@x_ref)
    j_ref = (A_jerk@x_ref)

    if verbose:
        st = prob.solver_stats
        solve_time = getattr(st,"solve_time",None)
        time_elapsed = time.time() - time_start
        print("Status: %s, obj=%.6f, solve_time=%s, wall=%.2fms" % (
            prob.status,float(prob.value),str(solve_time),time_elapsed*1000.0
        ))

    return {
        "status": prob.status,
        "objective": float(prob.value),
        "t": secs,
        "x_ref": x_ref,
        "z": z_opt,
        "v": v_opt,
        "a": a_opt,
        "j": j_opt,
        "lambda_j": lambda_j,
        "rho_remain": rho_remain,
        "remain_val": remain_val,
        "remain_segs": remain_segs,
        "v_max": v_max,
        "a_max": a_max,
        "A_vel": A_vel,
        "A_acc": A_acc,
        "A_jerk": A_jerk,
        "v_ref": v_ref,
        "a_ref": a_ref,
        "j_ref": j_ref,
        "dt": dt,
    }


def plot_traj_1d_remaining_result(
        res: dict,
        plot_vel: bool  = True,
        plot_acc: bool  = True,
        plot_jerk: bool = True,
        figsize         = (6, 3),
    ):
    """
    Visualize the output of `traj_1d_remainer`.

    Parameters:
        res (dict): Input res.
        plot_vel (bool): Input plot vel. Defaults to `True`.
        plot_acc (bool): Input plot acc. Defaults to `True`.
        plot_jerk (bool): Input plot jerk. Defaults to `True`.
        figsize (tuple): Input figsize. Defaults to `(6, 3)`.

    Returns:
        None: This function mutates state, renders output, or performs side
              effects in place.
    """

    # ---------- Unpack (with fallbacks) ----------
    t     = np.asarray(res["t"]).ravel()
    x_ref = np.asarray(res["x_ref"]).ravel()
    z_opt = np.asarray(res["z"]).ravel()

    remain_segs = res.get("remain_segs", None)
    remain_val  = res.get("remain_val", None)

    v_ref = res.get("v_ref", None); a_ref = res.get("a_ref", None); j_ref = res.get("j_ref", None)
    v_opt = res.get("v", None);     a_opt = res.get("a", None);     j_opt = res.get("j", None)
    A_vel  = res.get("A_vel", None); A_acc  = res.get("A_acc", None); A_jerk = res.get("A_jerk", None)

    if v_ref is None and A_vel  is not None: v_ref = A_vel  @ x_ref
    if a_ref is None and A_acc  is not None: a_ref = A_acc  @ x_ref
    if j_ref is None and A_jerk is not None: j_ref = A_jerk @ x_ref
    if v_opt is None and A_vel  is not None: v_opt = A_vel  @ z_opt
    if a_opt is None and A_acc  is not None: a_opt = A_acc  @ z_opt
    if j_opt is None and A_jerk is not None: j_opt = A_jerk @ z_opt

    v_max    = res.get("vmax", res.get("v_max", None))
    a_max    = res.get("a_max", res.get("amax", None))
    lambda_j = res.get("lambda_j", None)
    rho_rem  = res.get("rho_remain", None)

    # ---------- 1) Position ----------
    fig, ax = plt.subplots(figsize=figsize)
    shade_remain_segments(ax,t,remain_segs,remain_val)
    ax.plot(t, x_ref, "-", linewidth=0.9, label="reference", zorder=9)
    ax.plot(t, z_opt, "-", linewidth=1.5, label="optimized", zorder=10)
    if rho_rem is not None:
        ax.text(0.01, 0.95, f"ρ_remain={rho_rem:g}", transform=ax.transAxes,
                ha="left", va="top", fontsize=8)
    ax.set_title("Trajectory", fontsize=9)
    ax.set_xlabel("Time [s]", fontsize=8)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    # ---------- 2) Velocity ----------
    if plot_vel and (v_ref is not None or v_opt is not None):
        fig, ax = plt.subplots(figsize=figsize)
        if v_ref is not None:
            ax.plot(t, v_ref, "-", linewidth=0.9, label="velocity (ref)")
        if v_opt is not None:
            ax.plot(t, v_opt, "-", linewidth=1.5, label="velocity (opt)")
        if v_max is not None:
            vmax_arr = as_array_or_none(v_max,len(t))
            ax.fill_between(t, -vmax_arr, vmax_arr, alpha=0.08, label="±v_max band")
            # --- dashed lines for ±v_max ---
            ax.plot(t,  vmax_arr, "--", linewidth=1.0, label="v_max")
            ax.plot(t, -vmax_arr, "--", linewidth=1.0, label="-v_max")
        ax.set_title("Velocity", fontsize=9)
        ax.set_xlabel("Time [s]", fontsize=8)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

    # ---------- 3) Acceleration ----------
    if plot_acc and (a_ref is not None or a_opt is not None):
        fig, ax = plt.subplots(figsize=figsize)
        if a_ref is not None:
            ax.plot(t, a_ref, "-", linewidth=0.9, label="acceleration (ref)")
        if a_opt is not None:
            ax.plot(t, a_opt, "-", linewidth=1.5, label="acceleration (opt)")
        if a_max is not None:
            amax_arr = as_array_or_none(a_max,len(t))
            ax.fill_between(t, -amax_arr, amax_arr, alpha=0.08, label="±a_max band")
            # --- dashed lines for ±a_max ---
            ax.plot(t,  amax_arr, "--", linewidth=1.0, label="a_max")
            ax.plot(t, -amax_arr, "--", linewidth=1.0, label="-a_max")
        ax.set_title("Acceleration", fontsize=9)
        ax.set_xlabel("Time [s]", fontsize=8)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

    # ---------- 4) Jerk ----------
    if plot_jerk and (j_ref is not None or j_opt is not None):
        fig, ax = plt.subplots(figsize=figsize)
        if j_ref is not None:
            ax.plot(t, j_ref, "-", linewidth=0.9, label="jerk (ref)")
        if j_opt is not None:
            ax.plot(t, j_opt, "-", linewidth=1.5, label="jerk (opt)")
        if lambda_j is not None:
            ax.text(0.01, 0.95, f"jerk reg λ={lambda_j:g}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=8)
        ax.set_title("Jerk", fontsize=9)
        ax.set_xlabel("Time [s]", fontsize=8)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()
