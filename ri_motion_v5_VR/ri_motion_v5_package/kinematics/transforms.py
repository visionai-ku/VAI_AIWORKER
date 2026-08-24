from __future__ import annotations

import numpy as np
from typing import Optional
from scipy.linalg import logm, expm

# Constants
D2R = np.pi / 180.0
R2D = 180.0 / np.pi

def t2pr(T):
    """
    Convert a 4x4 transformation matrix to its position vector and rotation
    matrix.

    Parameters:
        T (Any): Input T.

    Returns:
        Any: Result computed from the current inputs.
    """
    p = T[:3,3]
    R = T[:3,:3]
    return p,R

def t2p(T):
    """
    Extract the position vector from a 4x4 transformation matrix.

    Parameters:
        T (Any): Input T.

    Returns:
        Any: Result computed from the current inputs.
    """
    p = T[:3,3]
    return p

def t2x(T,aw=0.1):
    """
    Get a point in the x-axis direction of the transform T for visualization.

    Parameters:
        T (Any): Input T.
        aw (float): Input aw. Defaults to `0.1`.

    Returns:
        Any: Result computed from the current inputs.
    """
    p = T[:3,3]
    R = T[:3,:3]
    x = p + aw*R[:,0] # x-axis direction
    return x

def t2y(T,aw=0.1):
    """
    Get a point in the y-axis direction of the transform T for visualization.

    Parameters:
        T (Any): Input T.
        aw (float): Input aw. Defaults to `0.1`.

    Returns:
        Any: Result computed from the current inputs.
    """
    p = T[:3,3]
    R = T[:3,:3]
    y = p + aw*R[:,1] # y-axis direction
    return y

def t2z(T,aw=0.1):
    """
    Get a point in the z-axis direction of the transform T for visualization.

    Parameters:
        T (Any): Input T.
        aw (float): Input aw. Defaults to `0.1`.

    Returns:
        Any: Result computed from the current inputs.
    """
    p = T[:3,3]
    R = T[:3,:3]
    z = p + aw*R[:,2] # z-axis direction
    return z

def r2x(R,aw=0.1):
    """
    Get a point in the x-axis direction of the rotation R for visualization.

    Parameters:
        R (Any): Input R.
        aw (float): Input aw. Defaults to `0.1`.

    Returns:
        Any: Result computed from the current inputs.
    """
    x = aw*R[:,0] # x-axis direction
    return x

def r2y(R,aw=0.1):
    """
    Get a point in the y-axis direction of the rotation R for visualization.

    Parameters:
        R (Any): Input R.
        aw (float): Input aw. Defaults to `0.1`.

    Returns:
        Any: Result computed from the current inputs.
    """
    y = aw*R[:,1] # y-axis direction
    return y

def r2z(R,aw=0.1):
    """
    Get a point in the z-axis direction of the rotation R for visualization.

    Parameters:
        R (Any): Input R.
        aw (float): Input aw. Defaults to `0.1`.

    Returns:
        Any: Result computed from the current inputs.
    """
    z = aw*R[:,2] # z-axis direction
    return z

def t2r(T):
    """
    Extract the rotation matrix from a 4x4 transformation matrix.

    Parameters:
        T (Any): Input T.

    Returns:
        Any: Result computed from the current inputs.
    """
    R = T[:3,:3]
    return R

def t2rpy(T, unit='rad'):
    """
    Convert a 4x4 transformation matrix to roll, pitch, and yaw angles.

    Parameters:
        T (Any): Input T.
        unit (str): Input unit. Defaults to `'rad'`.

    Returns:
        Any: Result computed from the current inputs.
    """
    R = t2r(T)
    rpy = r2rpy(R, unit=unit)
    return rpy

def t2prpy(T, unit='rad'):
    """
    Convert a 4x4 transformation matrix to its position vector and roll, pitch,
    yaw angles.

    Parameters:
        T (Any): Input T.
        unit (str): Input unit. Defaults to `'rad'`.

    Returns:
        Any: Result computed from the current inputs.
    """
    p = t2p(T)
    rpy = t2rpy(T, unit=unit)
    prpy = np.concatenate([p, rpy]) # shape (6,)
    return prpy

def pr2t(p=(0,0,0),R=np.eye(3)):
    """
    Convert a position vector and rotation matrix into a 4x4 transformation
    matrix.

    Parameters:
        p (tuple): Input p. Defaults to `(0, 0, 0)`.
        R (Any): Input R. Defaults to `np.eye(3)`.

    Returns:
        Any: Result computed from the current inputs.
    """
    p = np.asarray(p, dtype=np.float64)
    p0 = p.ravel() # flatten
    T = np.block([
        [R, p0[:, np.newaxis]],
        [np.zeros(3), 1]
    ])
    return T

def p2t(p=(0,0,0)):
    """
    Convert a position vector into a 4x4 transformation matrix.

    Parameters:
        p (tuple): Input p. Defaults to `(0, 0, 0)`.

    Returns:
        Any: Result computed from the current inputs.
    """
    p = np.asarray(p, dtype=np.float64)
    R = np.eye(3)
    p0 = p.ravel() # flatten
    T = np.block([
        [R, p0[:, np.newaxis]],
        [np.zeros(3), 1]
    ])
    return T

def r2t(R=np.eye(3)):
    """
    Convert a rotation matrix into a 4x4 transformation matrix.

    Parameters:
        R (Any): Input R. Defaults to `np.eye(3)`.

    Returns:
        Any: Result computed from the current inputs.
    """
    p = np.asarray((0,0,0), dtype=np.float64)
    p0 = p.ravel() # flatten
    T = np.block([
        [R, p0[:, np.newaxis]],
        [np.zeros(3), 1]
    ])
    return T

def rpy2r(rpy,unit="rad"):
    """
    Convert roll, pitch, and yaw angles (in radians) to a 3x3 rotation matrix.

    Parameters:
        rpy (Any): Input rpy.
        unit (str): Input unit. Defaults to `'rad'`.

    Returns:
        Any: Result computed from the current inputs.
    """
    if unit == 'rad':
        rpy_rad = rpy
    elif unit == 'deg':
        rpy_rad = np.deg2rad(rpy)
    else:
        raise ValueError("[rpy2r] Unknown unit:[%s]" % unit)

    roll  = rpy_rad[0]
    pitch = rpy_rad[1]
    yaw   = rpy_rad[2]
    Cphi  = np.cos(roll)
    Sphi  = np.sin(roll)
    Cthe  = np.cos(pitch)
    Sthe  = np.sin(pitch)
    Cpsi  = np.cos(yaw)
    Spsi  = np.sin(yaw)
    R     = np.array([
        [Cpsi * Cthe, -Spsi * Cphi + Cpsi * Sthe * Sphi, Spsi * Sphi + Cpsi * Sthe * Cphi],
        [Spsi * Cthe, Cpsi * Cphi + Spsi * Sthe * Sphi, -Cpsi * Sphi + Spsi * Sthe * Cphi],
        [-Sthe, Cthe * Sphi, Cthe * Cphi]
    ])
    assert R.shape == (3, 3)
    return R

def rpy2r_order(r0, order=[0,1,2]):
    """
    Convert roll, pitch, and yaw angles (in radians) to a 3x3 rotation matrix with
    a specified rotation order.

    Parameters:
        r0 (Any): Input r0.
        order (list): Input order. Defaults to `[0, 1, 2]`.

    Returns:
        Any: Result computed from the current inputs.
    """
    c1 = np.cos(r0[0]); c2 = np.cos(r0[1]); c3 = np.cos(r0[2])
    s1 = np.sin(r0[0]); s2 = np.sin(r0[1]); s3 = np.sin(r0[2])
    a1 = np.array([[1,0,0],[0,c1,-s1],[0,s1,c1]])
    a2 = np.array([[c2,0,s2],[0,1,0],[-s2,0,c2]])
    a3 = np.array([[c3,-s3,0],[s3,c3,0],[0,0,1]])
    a_list = [a1,a2,a3]
    a = np.matmul(np.matmul(a_list[order[0]],a_list[order[1]]),a_list[order[2]])
    assert a.shape == (3,3)
    return a

def r2rpy(R,unit='rad'):
    """
    Convert a 3x3 rotation matrix to roll, pitch, and yaw angles.

    Parameters:
        R (Any): Input R.
        unit (str): Input unit. Defaults to `'rad'`.

    Returns:
        Any: Result computed from the current inputs.
    """
    roll  = np.atan2(R[2, 1], R[2, 2])
    pitch = np.atan2(-R[2, 0], (np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
    yaw   = np.atan2(R[1, 0], R[0, 0])
    if unit == 'rad':
        out = np.array([roll, pitch, yaw])
    elif unit == 'deg':
        out = np.array([roll, pitch, yaw])*180/np.pi
    else:
        out = None
        raise Exception("[r2rpy] Unknown unit:[%s]"%(unit))
    return out

def prpy2t(prpy, unit='rad'):
    """
    Convert a position vector and roll, pitch, yaw angles into a 4x4
    transformation matrix.

    Parameters:
        prpy (Any): Input prpy.
        unit (str): Input unit. Defaults to `'rad'`.

    Returns:
        Any: Result computed from the current inputs.
    """
    prpy = np.array(prpy, dtype=np.float64).flatten() # ensure shape (6,)
    p = prpy[:3]
    rpy = prpy[3:]
    if unit == 'rad':
        R = rpy2r(rpy)
    elif unit == 'deg':
        R = rpy2r(rpy*np.pi/180)
    else:
        R = None
        raise Exception("[prpy2t] Unknown unit:[%s]"%(unit))
    T = pr2t(p=p,R=R)
    return T

def get_R_from_twopoints(p_fr,p_to):
    """
    Compute a rotation matrix that aligns a reference direction with the direction
    from one point to another.

    Parameters:
        p_fr (np.ndarray): Input p fr.
        p_to (np.ndarray): Input p to.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    p_a  = np.copy(np.array([1e-10,-1e-10,1.0]))
    if np.linalg.norm(p_to-p_fr) < 1e-8: # if two points are too close
        return np.eye(3)
    p_b  = (p_to-p_fr)/np.linalg.norm(p_to-p_fr)
    v    = np.cross(p_a,p_b)
    S = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    if np.linalg.norm(v) == 0:
        R = np.eye(3,3)
    else:
        R = np.eye(3,3) + S + S@S*(1-np.dot(p_a,p_b))/(np.linalg.norm(v)*np.linalg.norm(v))
    return R

def r2quat(R):
    """
    Convert a 3x3 rotation matrix to a quaternion.

    Parameters:
        R (Any): Input R.

    Returns:
        Any: Result computed from the current inputs.
    """
    R = np.asarray(R, dtype=np.float64)
    Qxx, Qyx, Qzx = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    Qxy, Qyy, Qzy = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    Qxz, Qyz, Qzz = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    # Fill only lower half of symmetric matrix
    K = np.zeros(R.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0
    # TODO: vectorize this -- probably could be made faster
    q = np.empty(K.shape[:-2] + (4,))
    it = np.nditer(q[..., 0], flags=['multi_index'])
    while not it.finished:
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[it.multi_index][0] < 0:
            q[it.multi_index] *= -1
        it.iternext()
    return q

def r2w(R):
    """
    Compute the angular velocity vector from a 3x3 rotation matrix.

    Parameters:
        R (Any): Input R.

    Returns:
        Any: Result computed from the current inputs.
    """
    el = np.array([
            [R[2,1] - R[1,2]],
            [R[0,2] - R[2,0]],
            [R[1,0] - R[0,1]]
        ])
    norm_el = np.linalg.norm(el)
    if norm_el > 1e-10:
        w = np.arctan2(norm_el, np.trace(R)-1) / norm_el * el
    elif R[0,0] > 0 and R[1,1] > 0 and R[2,2] > 0:
        w = np.array([[0, 0, 0]]).T
    else:
        w = np.pi/2 * np.array([[R[0,0]+1], [R[1,1]+1], [R[2,2]+1]])
    return w.flatten()

def rodrigues(a=np.array([1.0,0.0,0.0]),q_rad=0.0):
    """
    Rodrigues formula: R=exp([a]_x*q_rad).

    Parameters:
        a (Any): Input a. Defaults to `np.array([1.0, 0.0, 0.0])`.
        q_rad (float): Input q rad. Defaults to `0.0`.

    Returns:
        Any: Result computed from the current inputs.
    """
    a = np_uv(a)
    a_hat = skew(a)
    c = np.cos(float(q_rad))
    s = np.sin(float(q_rad))
    R = np.eye(3)+a_hat*s+(a_hat@a_hat)*(1.0-c)
    return R


def w2r(w):
    """
    Exponential map on SO(3): R=exp([w]_x), where w=axis*angle.

    Parameters:
        w (Any): Input w.

    Returns:
        Any: Result computed from the current inputs.
    """
    w  = np.asarray(w,dtype=np.float64).reshape(3,)
    th = np.linalg.norm(w)
    if th<1e-12:
        return np.eye(3)
    a = w/th
    return rodrigues(a,q_rad=th)

def skew(x):
    """
    Compute the skew-symmetric matrix of a 3-element vector.

    Parameters:
        x (Any): Input x.

    Returns:
        Any: Result computed from the current inputs.
    """
    x_hat = np.array([[0,-x[2],x[1]],[x[2],0,-x[0]],[-x[1],x[0],0]])
    return x_hat

def unskew(x_hat):
    """
    Compute the vector from a skew-symmetric matrix.

    Parameters:
        x_hat (Any): Input x hat.

    Returns:
        Any: Result computed from the current inputs.
    """
    x = np.array([x_hat[2,1], x_hat[0,2], x_hat[1,0]])
    return x

def get_adjoint(T=None, p=(0,0,0), R=np.eye(3)):
    """
    Compute the 6x6 adjoint matrix of a rigid transform in SE(3).

    Parameters:
        T (Any | None): Input T. Defaults to `None`.
        p (tuple): Input p. Defaults to `(0, 0, 0)`.
        R (Any): Input R. Defaults to `np.eye(3)`.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    if T is not None:
        T = np.asarray(T, dtype=np.float64)
        if T.shape != (4,4):
            raise ValueError("T must be (4,4), got %s" % (str(T.shape)))
        p = t2p(T)
        R = t2r(T)
    else:
        p = np.asarray(p, dtype=np.float64).reshape(3,)
        R = np.asarray(R, dtype=np.float64).reshape(3,3)
    Ad_T = np.block([
        [R, np.zeros((3,3), dtype=np.float64)],
        [skew(p) @ R, R],
    ])
    return Ad_T

def adjoint_twist(V, T=None, p=(0,0,0), R=np.eye(3)):
    """
    Transform a twist by the adjoint of a rigid transform.

    Parameters:
        V (Any): Input V.
        T (Any | None): Input T. Defaults to `None`.
        p (tuple): Input p. Defaults to `(0, 0, 0)`.
        R (Any): Input R. Defaults to `np.eye(3)`.

    Returns:
        Any: Result computed from the current inputs.
    """
    V = np.asarray(V, dtype=np.float64).reshape(6,)
    return get_adjoint(T=T, p=p, R=R) @ V

def view_in_local(
        p: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        T: Optional[np.ndarray] = None,
        T_wl: Optional[np.ndarray] = None,
    ):
    """
    View quantities (p, R, or T) defined in the world frame (W).

    Parameters:
        p (Optional[np.ndarray]): Input p. Defaults to `None`.
        R (Optional[np.ndarray]): Input R. Defaults to `None`.
        T (Optional[np.ndarray]): Input T. Defaults to `None`.
        T_wl (Optional[np.ndarray]): Input T wl. Defaults to `None`.

    Returns:
        Any: Result computed from the current inputs.
    """
    cnt = 0
    if p is not None:
        cnt += 1
    if R is not None:
        cnt += 1
    if T is not None:
        cnt += 1
    if cnt != 1:
        raise ValueError("Exactly one of (p, R, T) must be provided, got %d" % (cnt))

    if T_wl is None:
        raise ValueError("T_wl must be provided")
    T_wl = np.asarray(T_wl)
    if T_wl.shape != (4, 4):
        raise ValueError("T_wl must be (4,4), got %s" % (str(T_wl.shape)))

    R_wl = T_wl[:3, :3]
    t_wl = T_wl[:3, 3]

    # Inverse pose: T_lw = inv(T_wl) = [R^T, -R^T t]
    R_lw = R_wl.T
    t_lw = -R_lw @ t_wl

    if p is not None:
        p = np.asarray(p)
        if p.shape[-1] != 3:
            raise ValueError("p must end with 3, got %s" % (str(p.shape)))
        p_flat = p.reshape(-1, 3)
        p_l = (p_flat @ R_lw.T + t_lw).reshape(p.shape)
        return p_l

    if R is not None:
        R = np.asarray(R)
        if R.shape[-2:] != (3, 3):
            raise ValueError("R must end with (3,3), got %s" % (str(R.shape)))
        R_flat = R.reshape(-1, 3, 3)
        R_l = np.einsum("ij,njk->nik", R_lw, R_flat).reshape(R.shape)
        return R_l

    T = np.asarray(T)
    if T.shape[-2:] != (4, 4):
        raise ValueError("T must end with (4,4), got %s" % (str(T.shape)))
    T_flat = T.reshape(-1, 4, 4)

    # Lift R_lw, t_lw to homogeneous, then left-multiply.
    T_lw = np.eye(4, dtype=T_wl.dtype)
    T_lw[:3, :3] = R_lw
    T_lw[:3, 3] = t_lw

    T_l = (T_lw @ T_flat).reshape(T.shape)
    return T_l # (...,4,4)

def view_in_world(
        p: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        T: Optional[np.ndarray] = None,
        T_wl: Optional[np.ndarray] = None,
    ):
    """
    View (i.e., re-express) quantities (p, R, or T) expressed in the local frame
    (L).

    Parameters:
        p (Optional[np.ndarray]): Input p. Defaults to `None`.
        R (Optional[np.ndarray]): Input R. Defaults to `None`.
        T (Optional[np.ndarray]): Input T. Defaults to `None`.
        T_wl (Optional[np.ndarray]): Input T wl. Defaults to `None`.

    Returns:
        Any: Result computed from the current inputs.
    """
    cnt = 0
    if p is not None:
        cnt += 1
    if R is not None:
        cnt += 1
    if T is not None:
        cnt += 1
    if cnt != 1:
        raise ValueError("Exactly one of (p, R, T) must be provided, got %d" % (cnt))

    if T_wl is None:
        raise ValueError("T_wl must be provided")
    T_wl = np.asarray(T_wl)
    if T_wl.shape != (4, 4):
        raise ValueError("T_wl must be (4,4), got %s" % (str(T_wl.shape)))

    R_wl = T_wl[:3, :3]
    t_wl = T_wl[:3, 3]

    if p is not None:
        p = np.asarray(p)
        if p.shape[-1] != 3:
            raise ValueError("p must end with 3, got %s" % (str(p.shape)))
        p_flat = p.reshape(-1, 3)
        p_w = (p_flat @ R_wl.T + t_wl).reshape(p.shape)
        return p_w

    if R is not None:
        R = np.asarray(R)
        if R.shape[-2:] != (3, 3):
            raise ValueError("R must end with (3,3), got %s" % (str(R.shape)))
        R_flat = R.reshape(-1, 3, 3)
        R_w = np.einsum("ij,njk->nik", R_wl, R_flat).reshape(R.shape)
        return R_w

    T = np.asarray(T)
    if T.shape[-2:] != (4, 4):
        raise ValueError("T must end with (4,4), got %s" % (str(T.shape)))
    T_flat = T.reshape(-1, 4, 4)
    T_w = (T_wl @ T_flat).reshape(T.shape)
    return T_w

def tf_global(T_in, p_global=(0, 0, 0), R_global=np.eye(3), T_global=None):
    """
    Applies a global transformation to the given transformation matrix T_in.

    Parameters:
        T_in (Any): Input T in.
        p_global (tuple): Input p global. Defaults to `(0, 0, 0)`.
        R_global (Any): Input R global. Defaults to `np.eye(3)`.
        T_global (Any | None): Input T global. Defaults to `None`.

    Returns:
        Any: Result computed from the current inputs.
    """
    if T_global is None:
        T_global = pr2t(p=p_global, R=R_global)
    return T_global @ T_in

def tf_local(T_in, p_local=(0, 0, 0), R_local=np.eye(3), T_local=None):
    """
    Applies a local transformation to the given transformation matrix T_in.

    Parameters:
        T_in (Any): Input T in.
        p_local (tuple): Input p local. Defaults to `(0, 0, 0)`.
        R_local (Any): Input R local. Defaults to `np.eye(3)`.
        T_local (Any | None): Input T local. Defaults to `None`.

    Returns:
        Any: Result computed from the current inputs.
    """
    if T_local is None:
        T_local = pr2t(p=p_local, R=R_local)
    return T_in @ T_local

def get_T_global_indep(T_0,dT):
    """
    Update pose by applying translation and rotation independently in the global
    frame.

    Parameters:
        T_0 (Any): Input T 0.
        dT (float): Input dT.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    T = pr2t(
        p = t2p(T_0)+t2p(dT),
        R = t2r(dT)@t2r(T_0),
    )
    return T

def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Compute the conjugate of a quaternion.

    Parameters:
        q (np.ndarray): Input q.

    Returns:
        np.ndarray: Result computed from the current inputs.
    """
    q = np.asarray(q, dtype=np.float64).reshape(4,)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions.

    Parameters:
        q1 (np.ndarray): Input q1.
        q2 (np.ndarray): Input q2.

    Returns:
        np.ndarray: Result computed from the current inputs.
    """
    q1 = np.asarray(q1, dtype=np.float64).reshape(4,)
    q2 = np.asarray(q2, dtype=np.float64).reshape(4,)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=np.float64)

def rotate_vector_by_quat(vec: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """
    Rotate a vector by a quaternion.

    Parameters:
        vec (np.ndarray): Input vec.
        quat (np.ndarray): Input quat.

    Returns:
        np.ndarray: Result computed from the current inputs.
    """
    v = np.asarray(vec, dtype=np.float64).reshape(3,)
    q = np.asarray(quat, dtype=np.float64).reshape(4,)
    vq = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    out = quat_multiply(quat_multiply(q, vq), quat_conjugate(q))[1:]
    return out

def np_uv(vec, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize a vector to have unit length.

    Parameters:
        vec (Any): Input vec.
        eps (float): Input eps. Defaults to `1e-12`.

    Returns:
        np.ndarray: Result computed from the current inputs.
    """
    x = np.asarray(vec, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(x)
    if n < eps:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return x / n

def project_onto_plane(v, normal):
    """
    Project vector v onto the plane orthogonal to the given normal vector.

    Parameters:
        v (Any): Input v.
        normal (Any): Input normal.

    Returns:
        Any: Result computed from the current inputs.
    """
    return v - np.dot(v, normal) * normal

def get_angle_about_axis(uv_fr, uv_to, axis):
    """
    Compute the signed rotation angle (radians) from uv_fr to uv_to.

    Parameters:
        uv_fr (Any): Input uv fr.
        uv_to (Any): Input uv to.
        axis (Any): Input axis.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    axis = np_uv(axis)
    v1 = np_uv(project_onto_plane(uv_fr, axis))
    v2 = np_uv(project_onto_plane(uv_to, axis))

    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    magnitude = np.arccos(dot)
    sign = np.sign(np.dot(np.cross(v1, v2), axis))
    angle = sign*magnitude # angle in [-pi, pi]
    # if angle < 0.0: angle = angle + 2*np.pi # angle in [0,2pi]
    return angle

def axis_angle2quat(axis, angle):
    """
    Convert an axis-angle representation to a unit quaternion.

    Parameters:
        axis (Any): Input axis.
        angle (Any): Input angle.

    Returns:
        Any: Result computed from the current inputs.
    """
    axis = np_uv(axis)
    half = 0.5 * float(angle)
    s = np.sin(half)
    return np.array([np.cos(half), s*axis[0], s*axis[1], s*axis[2]], dtype=np.float64)

def quat2axis_angle(quat):
    """
    Extract the rotation axis and angle from a quaternion.

    Parameters:
        quat (np.ndarray): Input quat.

    Returns:
        Any: Result computed from the current inputs.
    """
    quat = np_uv(quat)
    w, x, y, z = quat

    # Ensure w is positive to get the shorter rotation
    if w < 0:
        w, x, y, z = -w, -x, -y, -z

    # Compute angle
    angle = 2 * np.arccos(np.clip(w, -1.0, 1.0))

    # Handle near-zero rotation case
    if angle < 1e-6:
        return np.array([1.0, 0.0, 0.0]), 0.0  # arbitrary axis, zero angle

    # Compute axis
    sin_half_angle = np.sin(angle / 2)
    axis = np.array([x, y, z]) / sin_half_angle
    axis = np_uv(axis)

    return axis, angle

def R_from_axis(w,a=np.array([1,0,0])):
    """
    Create a rotation matrix from an angular velocity vector (axis-angle
    representation).

    Parameters:
        w (Any): Input w.
        a (Any): Input a. Defaults to `np.array([1, 0, 0])`.

    Returns:
        Any: Result computed from the current inputs.
    """
    if np.linalg.norm(w) < 1e-6:
        return np.eye(3)  # Return identity if a is zero vector

    # z-axis of the rotation
    w = np.array(w, dtype=np.float64)
    ez = w / np.linalg.norm(w) # w to be the z-axis of the rotation

    # x-axis of the rotation
    u = a - ez * (a.dot(ez))
    ex = u / np.linalg.norm(u)

    # y-axis of the rotation
    ey = np.cross(ez, ex)

    # Return the rotation matrix
    R = np.stack([ex, ey, ez], axis=1)
    return R

def screw_to_se3(w, q):
    """
    Convert screw axis (w, q) to a 4x4 Lie algebra se(3) matrix.

    Parameters:
        w (Any): Input w.
        q (np.ndarray): Input q.

    Returns:
        Any: Result computed from the current inputs.
    """
    w_hat = skew(w) # (3x3)
    v     = -np.cross(w, q) # (3,)
    se3   = np.zeros((4, 4)) # (4x4)
    se3[:3,:3] = w_hat
    se3[:3,3]  = v
    se3[3, 3] = 0.0  # stays 0, because this is se(3), not SE(3)
    return se3 # (4x4)

def get_plane_basis_from_normal(normal):
    """
    Get two orthonormal basis vectors on the plane orthogonal to a given normal.

    Parameters:
        normal (Any): Input normal.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    normal = np_uv(normal)
    temp = np.array([1, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    v1 = np_uv(np.cross(normal, temp))
    v2 = np_uv(np.cross(normal, v1))
    return v1, v2

def get_quat_between_vectors(uv_fr, uv_to):
    """
    Compute the quaternion that rotates 'uv_fr' to 'uv_to'.

    Parameters:
        uv_fr (Any): Input uv fr.
        uv_to (Any): Input uv to.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    uv_fr = np_uv(uv_fr)
    uv_to = np_uv(uv_to)

    dot = np.clip(np.dot(uv_fr, uv_to), -1.0, 1.0)

    # Check if vectors are parallel (same direction)
    if dot > 0.99999:
        return np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion

    # Check if vectors are anti-parallel (opposite direction)
    if dot < -0.99999:
        # Find a perpendicular axis
        axis = np.cross(uv_fr, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(uv_fr, np.array([0.0, 1.0, 0.0]))
        axis = np_uv(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]])  # 180-degree rotation

    # General case
    cross = np.cross(uv_fr, uv_to)
    w = 1.0 + dot
    quat = np.array([w, cross[0], cross[1], cross[2]])
    return np_uv(quat)
