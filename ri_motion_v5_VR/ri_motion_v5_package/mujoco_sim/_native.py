"""
Optional native backend helpers for mujoco_sim.
"""
from __future__ import annotations


import importlib
import sys
import warnings

import mujoco
import numpy as np


_mujoco_native = None
_native_modules = {}
_native_import_errors = {}
HAS_NATIVE = False
NATIVE_IMPORT_ERROR = None


def _native_platform_supported():
    """
    Return True when native extension modules are enabled on this platform.

    Parameters:
        None.

    Returns:
        bool: True on macOS and Linux, False elsewhere.
    """
    return sys.platform == "darwin" or sys.platform.startswith("linux")


def _warn_unsupported_native_platform(action,module_name):
    """
    Warn that native extension support is currently macOS/Linux-only.

    Parameters:
        action (str): Action being skipped.
        module_name (str): Native extension module basename.

    Returns:
        None.
    """
    warnings.warn(
        (
            "mujoco_sim native extension support is currently enabled only on "
            "macOS and Linux. Skipping %s for module [%s] on platform [%s]."
        )%(action,module_name,sys.platform),
        RuntimeWarning,
        stacklevel=2,
    )


def _native_module_import_name(module_name):
    """
    Return the fully qualified import path for a native backend module.

    Parameters:
        module_name (str): Native extension module basename.

    Returns:
        str: Fully qualified module import path.
    """
    return "ri_motion_v5_package.mujoco_sim.so.%s" % module_name


def _import_native_backend(module_name="_mujoco_native"):
    """
    Import or refresh the compiled native backend module.

    Parameters:
        module_name (str): Native extension module basename.

    Returns:
        module | None: Native backend module or None.
    """
    global _mujoco_native,HAS_NATIVE,NATIVE_IMPORT_ERROR

    try:
        backend = importlib.import_module(_native_module_import_name(module_name))
        _native_modules[module_name] = backend
        _native_import_errors.pop(module_name,None)
        if module_name == "_mujoco_native":
            _mujoco_native = backend
            HAS_NATIVE = True
            NATIVE_IMPORT_ERROR = None
        return backend
    except Exception as exc:  # pragma: no cover - exercised when extension is absent
        _native_modules.pop(module_name,None)
        _native_import_errors[module_name] = exc
        if module_name == "_mujoco_native":
            _mujoco_native = None
            HAS_NATIVE = False
            NATIVE_IMPORT_ERROR = exc
        return None


def get_native_backend(required=False,module_name="_mujoco_native"):
    """
    Return the optional native backend module.

    Parameters:
        required (bool): If True, raise when the backend is unavailable.
        module_name (str): Native extension module basename.

    Returns:
        module | None: Native backend module or None.
    """
    if not _native_platform_supported():
        _warn_unsupported_native_platform("get_native_backend",module_name)
        return None

    if module_name in _native_modules:
        return _native_modules[module_name]
    if module_name == "_mujoco_native" and HAS_NATIVE and _mujoco_native is not None:
        return _mujoco_native
    backend = _import_native_backend(module_name=module_name)
    if backend is not None:
        return backend
    if required:
        import_error = _native_import_errors.get(module_name,NATIVE_IMPORT_ERROR)
        raise RuntimeError(
            "mujoco_sim native backend is not available. "
            "Build %s first." % module_name
        ) from import_error
    return None


def _mujoco_addresses(model,data):
    """
    Return integer addresses for MuJoCo model/data objects.

    Parameters:
        model (mujoco.MjModel): MuJoCo model object.
        data (mujoco.MjData): MuJoCo data object.

    Returns:
        tuple[int, int]: Raw `mjModel` and `mjData` addresses.
    """
    try:
        model_address = int(model._address)
        data_address = int(data._address)
    except AttributeError as exc:
        raise TypeError("model and data must be MuJoCo objects with `_address`.") from exc
    if model_address == 0 or data_address == 0:
        raise ValueError("MuJoCo model/data address must be non-zero.")
    return model_address,data_address


def echo_model_data_native(model,data,module_name="_mujoco_native_echo"):
    """
    Echo selected mjModel and mjData fields through the native backend.

    Parameters:
        model (mujoco.MjModel): MuJoCo model object.
        data (mujoco.MjData): MuJoCo data object.
        module_name (str): Native extension module basename.

    Returns:
        dict: Copied scalar and array fields read by native code.
    """
    backend = get_native_backend(required=True,module_name=module_name)
    if backend is None:
        warnings.warn(
            "native echo backend is unavailable; returning None.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    model_address,data_address = _mujoco_addresses(model,data)
    return backend.echo_model_data(model_address,data_address)


def _signed_distance_arrays_from_backend(
        backend,
        env,
        geom_pairs,
        distmax          = np.inf,
        topk             = None,
        forward          = False,
        sort             = True,
        sanity_check     = True,
        lower_bound_tol  = 1e-6,
        fromto_dist_tol  = 1e-4,
    ):
    """
    Compute signed geom-pair distances through an already imported backend.

    Parameters:
        backend (module): Compiled extension module.
        env (MuJoCoParser): Environment that owns the MuJoCo model and data.
        geom_pairs (array-like): Candidate geom pairs with shape `(N, 2)`.
        distmax (float): Maximum distance query radius.
        topk (int | None): Number of closest signed-distance records to keep.
        forward (bool): If True, call `env.forward()` before querying.
        sort (bool): If True, sort output records by signed distance.
        sanity_check (bool): If True, skip inconsistent native distance records.
        lower_bound_tol (float): Tolerance for lower-bound sanity checks.
        fromto_dist_tol (float): Tolerance for closest-point sanity checks.

    Returns:
        dict: Compact NumPy arrays read and computed by native code.
    """
    if forward:
        env.forward()

    geom_pairs = np.ascontiguousarray(geom_pairs,dtype=np.int32).reshape(-1,2)
    native_topk = -1 if topk is None else max(0,int(topk))
    model_address,data_address = _mujoco_addresses(env.model,env.data)
    return backend.signed_distance_arrays(
        model_address,
        data_address,
        geom_pairs,
        float(distmax),
        native_topk,
        bool(sort),
        bool(sanity_check),
        float(lower_bound_tol),
        float(fromto_dist_tol),
    )


def _signed_distance_infos_from_arrays(env,arrays):
    """
    Convert compact signed-distance arrays to parser-compatible dictionaries.

    Parameters:
        env (MuJoCoParser): Environment that owns geom/body name lists.
        arrays (dict): Compact native signed-distance arrays.

    Returns:
        list[dict]: Signed-distance records compatible with existing parser
            plotting and depenetration helpers.
    """
    dist_values = np.asarray(arrays["dist"],dtype=np.float64)
    fromto_values = np.asarray(arrays["fromto"],dtype=np.float64)
    normal_values = np.asarray(arrays["normal"],dtype=np.float64)
    geom_pair_values = np.asarray(arrays["geom_pair"],dtype=np.int32)
    body_pair_values = np.asarray(arrays["body_pair"],dtype=np.int32)
    source_values = np.asarray(arrays["source"],dtype=np.int32)
    contact_idx_values = np.asarray(arrays["contact_idx"],dtype=np.int32)

    infos = []
    for idx,dist in enumerate(dist_values):
        g1 = int(geom_pair_values[idx,0])
        g2 = int(geom_pair_values[idx,1])
        b1 = int(body_pair_values[idx,0])
        b2 = int(body_pair_values[idx,1])
        fromto = fromto_values[idx,:].copy()
        p1 = fromto[:3].copy()
        p2 = fromto[3:].copy()
        point = 0.5*(p1+p2)
        is_contact = bool(source_values[idx] == 1)
        info = {
            "dist": float(dist),
            "signed_dist": float(dist),
            "source": "contact" if is_contact else "geom_distance",
            "is_contact": is_contact,
            "fromto": fromto,
            "p1": p1,
            "p2": p2,
            "point": point,
            "geom1_idx": g1,
            "geom2_idx": g2,
            "geom1_name": env.geom_names[g1],
            "geom2_name": env.geom_names[g2],
            "body1_idx": b1,
            "body2_idx": b2,
            "body1_name": env.body_names[b1],
            "body2_name": env.body_names[b2],
        }
        if is_contact:
            info["contact_idx"] = int(contact_idx_values[idx])
            info["p_contact"] = point.copy()
            info["normal"] = normal_values[idx,:].copy()
        infos.append(info)

    return infos


def _signed_distance_infos_from_backend(
        backend,
        env,
        geom_pairs,
        distmax          = np.inf,
        topk             = None,
        forward          = False,
        sort             = True,
        sanity_check     = True,
        lower_bound_tol  = 1e-6,
        fromto_dist_tol  = 1e-4,
    ):
    """
    Return parser-compatible signed-distance dictionaries from a backend.

    Parameters:
        backend (module): Compiled extension module.
        env (MuJoCoParser): Environment that owns the MuJoCo model and data.
        geom_pairs (array-like): Candidate geom pairs with shape `(N, 2)`.
        distmax (float): Maximum distance query radius.
        topk (int | None): Number of closest signed-distance records to keep.
        forward (bool): If True, call `env.forward()` before querying.
        sort (bool): If True, sort output records by signed distance.
        sanity_check (bool): If True, skip inconsistent native distance records.
        lower_bound_tol (float): Tolerance for lower-bound sanity checks.
        fromto_dist_tol (float): Tolerance for closest-point sanity checks.

    Returns:
        list[dict]: Signed-distance records compatible with parser helpers.
    """
    arrays = _signed_distance_arrays_from_backend(
        backend,
        env,
        geom_pairs,
        distmax         = distmax,
        topk            = topk,
        forward         = forward,
        sort            = sort,
        sanity_check    = sanity_check,
        lower_bound_tol = lower_bound_tol,
        fromto_dist_tol = fromto_dist_tol,
    )
    return _signed_distance_infos_from_arrays(env,arrays)


def _body_position_ik_coupling_arrays(solver):
    """
    Return rev_pri joint-coupling metadata as compact arrays.

    Parameters:
        solver (BodyPositionIKSolverRevPriBase): Solver with rev_pri metadata.

    Returns:
        tuple[np.ndarray, ...]: Passive indices, active indices, coefficients,
            active offsets, and passive offsets.
    """
    passive_idxs = []
    active_idxs = []
    coefs = []
    x0_values = []
    y0_values = []

    model = solver.ik_env.model
    for eq_idx in range(int(model.eq_data.shape[0])):
        if model.eq_type[eq_idx] != mujoco.mjtEq.mjEQ_JOINT:
            continue

        passive_joint = model.joint(int(model.eq_obj1id[eq_idx])).name
        active_joint = model.joint(int(model.eq_obj2id[eq_idx])).name
        if (passive_joint not in solver._rev_pri_set) or (active_joint not in solver._rev_pri_set):
            continue

        passive_idxs.append(solver.rev_pri_joint_names.index(passive_joint))
        active_idxs.append(solver.rev_pri_joint_names.index(active_joint))
        coefs.append(np.asarray(model.eq_data[eq_idx,:5],dtype=np.float64))
        x0_values.append(float(model.joint(active_joint).qpos0[0]))
        y0_values.append(float(model.joint(passive_joint).qpos0[0]))

    if len(passive_idxs) == 0:
        return (
            np.zeros((0,),dtype=np.int32),
            np.zeros((0,),dtype=np.int32),
            np.zeros((0,5),dtype=np.float64),
            np.zeros((0,),dtype=np.float64),
            np.zeros((0,),dtype=np.float64),
        )

    return (
        np.asarray(passive_idxs,dtype=np.int32),
        np.asarray(active_idxs,dtype=np.int32),
        np.ascontiguousarray(np.vstack(coefs),dtype=np.float64),
        np.asarray(x0_values,dtype=np.float64),
        np.asarray(y0_values,dtype=np.float64),
    )


def _body_position_solve_ik_from_backend(
        backend,
        solver,
        env                         = None,
        joints_use                  = None,
        joint_limit_handle_flag     = True,
        nullspace_control_flag      = True,
        task_null_mask_flag         = False,
        init_from_median_flag       = False,
        base_control_flag           = False,
        base_stepsize               = 0.5,
        base_pos_th                 = 0.1,
        base_rot_th                 = np.deg2rad(5.0),
    ):
    """
    Solve BodyPositionIKSolverRevPriBase through a compiled backend.

    Parameters:
        backend (module): Compiled extension module.
        solver (BodyPositionIKSolverRevPriBase): Python solver wrapper.
        env (MuJoCoParser | None): Environment to sync before solving.
        joints_use (list[str] | None): Joint names to use.
        joint_limit_handle_flag (bool): If True, enable joint-limit probing.
        nullspace_control_flag (bool): If True, enable home nullspace control.
        task_null_mask_flag (bool): If True, mask task-irrelevant nullspace joints.
        init_from_median_flag (bool): Unsupported in native path for now.
        base_control_flag (bool): Unsupported in native path for now.
        base_stepsize (float): Kept for API compatibility.
        base_pos_th (float): Kept for API compatibility.
        base_rot_th (float): Kept for API compatibility.

    Returns:
        tuple[np.ndarray, dict]: Same return shape as solve_ik(...).
    """
    _ = base_stepsize,base_pos_th,base_rot_th

    if backend is None:
        warnings.warn(
            "IK bind backend is unavailable; falling back to Python solve_ik(...).",
            RuntimeWarning,
            stacklevel=2,
        )
        q,info = solver.solve_ik(
            env                         = env,
            joints_use                  = joints_use,
            joint_limit_handle_flag     = joint_limit_handle_flag,
            nullspace_control_flag      = nullspace_control_flag,
            task_null_mask_flag         = task_null_mask_flag,
            init_from_median_flag       = init_from_median_flag,
            base_control_flag           = base_control_flag,
            base_stepsize               = base_stepsize,
            base_pos_th                 = base_pos_th,
            base_rot_th                 = base_rot_th,
        )
        info["backend"] = "python_fallback"
        info["backend_reason"] = "backend_unavailable"
        return q,info

    if init_from_median_flag or base_control_flag:
        warnings.warn(
            (
                "IK bind currently supports init_from_median_flag=False and "
                "base_control_flag=False only; falling back to Python solve_ik(...)."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        q,info = solver.solve_ik(
            env                         = env,
            joints_use                  = joints_use,
            joint_limit_handle_flag     = joint_limit_handle_flag,
            nullspace_control_flag      = nullspace_control_flag,
            task_null_mask_flag         = task_null_mask_flag,
            init_from_median_flag       = init_from_median_flag,
            base_control_flag           = base_control_flag,
            base_stepsize               = base_stepsize,
            base_pos_th                 = base_pos_th,
            base_rot_th                 = base_rot_th,
        )
        info["backend"] = "python_fallback"
        info["backend_reason"] = "unsupported_flags"
        return q,info

    if env is not None:
        solver.sync_env(env)

    if solver.n_target == 0:
        q,info = solver.solve_ik(
            env                         = None,
            joints_use                  = joints_use,
            joint_limit_handle_flag     = joint_limit_handle_flag,
            nullspace_control_flag      = nullspace_control_flag,
            task_null_mask_flag         = task_null_mask_flag,
            init_from_median_flag       = False,
            base_control_flag           = False,
        )
        info["backend"] = "python_fallback"
        info["backend_reason"] = "no_target"
        return q,info

    if joints_use is not None:
        solver.joints_use = solver.normalize_joints_use(joints_use)
    solver.idxs_rev_pri_use = list(np.asarray(
        [solver.rev_pri_joint_names.index(joint_name) for joint_name in solver.joints_use],
        dtype=np.int32,
    ))

    model_address,data_address = _mujoco_addresses(solver.ik_env.model,solver.ik_env.data)
    rev_pri_qpos_idxs = np.ascontiguousarray(
        solver._get_rev_pri_qpos_idxs(solver.ik_env.model),
        dtype=np.int32,
    )
    idxs_jac_rev_pri = np.ascontiguousarray(solver.idxs_jac_rev_pri,dtype=np.int32)
    idxs_rev_pri_use = np.ascontiguousarray(solver.idxs_rev_pri_use,dtype=np.int32)
    idxs_rev_pri_active = np.ascontiguousarray(solver.idxs_rev_pri_active,dtype=np.int32)
    is_rev_in_rev_pri = np.ascontiguousarray(solver._is_rev_in_rev_pri.astype(np.int32),dtype=np.int32)

    dls = solver._get_dls_damping_use()
    if np.isscalar(dls):
        dls_damping_use = np.full((len(solver.idxs_rev_pri_use),),float(dls),dtype=np.float64)
    else:
        dls_damping_use = np.ascontiguousarray(dls,dtype=np.float64).reshape(-1)

    q_home_rev_pri = (
        np.zeros((0,),dtype=np.float64)
        if solver.q_home_rev_pri is None
        else np.ascontiguousarray(solver.q_home_rev_pri,dtype=np.float64).reshape(-1)
    )

    coupling_passive_idxs,coupling_active_idxs,coupling_coefs,coupling_x0,coupling_y0 = (
        _body_position_ik_coupling_arrays(solver)
    )

    arrays = backend.solve_body_position_ik(
        model_address,
        data_address,
        np.ascontiguousarray(solver.body_idx_list,dtype=np.int32),
        np.ascontiguousarray(np.vstack(solver.p_local_list),dtype=np.float64),
        np.ascontiguousarray(np.vstack(solver.p_trgt_list),dtype=np.float64),
        np.ascontiguousarray(solver.weight_list,dtype=np.float64),
        rev_pri_qpos_idxs,
        idxs_jac_rev_pri,
        idxs_rev_pri_use,
        idxs_rev_pri_active,
        is_rev_in_rev_pri,
        np.ascontiguousarray(solver.rev_pri_joint_mins,dtype=np.float64),
        np.ascontiguousarray(solver.rev_pri_joint_maxs,dtype=np.float64),
        q_home_rev_pri,
        dls_damping_use,
        coupling_passive_idxs,
        coupling_active_idxs,
        coupling_coefs,
        coupling_x0,
        coupling_y0,
        int(solver.max_ik_tick),
        float(solver.ik_stepsize_rev),
        float(solver.ik_stepsize_pri),
        float(solver.ik_update_th_rev),
        float(solver.ik_update_th_pri),
        float(solver.max_probe_rev),
        float(solver.max_probe_pri),
        float(solver.k_null),
        bool(joint_limit_handle_flag),
        bool(nullspace_control_flag),
        bool(task_null_mask_flag),
    )

    q_rev_pri_best = np.asarray(arrays["q_rev_pri_best"],dtype=np.float64).copy()
    ik_info = {
        "ik_err_array":np.asarray(arrays["ik_err_array"],dtype=np.float64).copy(),
        "idx_best":int(arrays["idx_best"]),
        "ik_err_best":float(arrays["ik_err_best"]),
        "elapsed_time":float(arrays["elapsed_time"]),
        "qpos_full_best":np.asarray(arrays["qpos_full_best"],dtype=np.float64).copy(),
        "qpos_used_best":np.asarray(arrays["qpos_used_best"],dtype=np.float64).copy(),
        "joints_use":solver.joints_use,
        "backend":"bind",
        "backend_reason":"native",
    }
    return q_rev_pri_best,ik_info
