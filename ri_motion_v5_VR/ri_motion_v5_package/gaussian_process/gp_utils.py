import numpy as np
import warnings

def get_K_chol(
        t,
        kernel_fn,
        hyp,
        eps      = 1e-6,
        return_K = False,
    ):
    """
    Compute Cholesky factor of the covariance matrix K using a generic kernel
    function.

    Parameters:
        t (Any): Input t.
        kernel_fn (Any): Input kernel fn.
        hyp (Any): Input hyp.
        eps (float): Input eps. Defaults to `1e-06`.
        return_K (bool): Input return K. Defaults to `False`.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    # Convert input to numpy array
    t = np.asarray(t)

    # Get size
    N  = int(t.shape[0])

    # Compute kernel matrix
    K      = kernel_fn(t,t,hyp)
    K      = np.asarray(K,dtype=np.float64)
    I_N    = np.eye(N,dtype=np.float64)
    K_chol = np.linalg.cholesky(K + eps*I_N)

    # Return
    if return_K:
        return K_chol, K
    return K_chol

def get_schur_K_chol(
        t_in,
        t_out,
        kernel_fn,
        hyp,
        eps      = 1e-6,
        return_K = False,
        upper    = False,
    ):
    """
    Compute Cholesky factor of the Schur-complement covariance:.

    Parameters:
        t_in (Any): Input t in.
        t_out (Any): Input t out.
        kernel_fn (Any): Input kernel fn.
        hyp (Any): Input hyp.
        eps (float): Input eps. Defaults to `1e-06`.
        return_K (bool): Input return K. Defaults to `False`.
        upper (bool): Input upper. Defaults to `False`.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    # Convert inputs to numpy arrays
    t_out = np.asarray(t_out)
    t_in  = np.asarray(t_in)
    if t_out.ndim!=1 or t_in.ndim!=1:
        raise ValueError("t_out and t_in must be 1D arrays")

    # Get sizes
    L  = int(t_out.shape[0])
    M  = int(t_in.shape[0])
    if L<=0 or M<=0:
        raise ValueError("Empty t_out or t_in")

    # Compute kernel matrices
    K_oo    = kernel_fn(t_out,t_out,hyp)
    K_oi    = kernel_fn(t_out,t_in,hyp)
    K_ii    = kernel_fn(t_in,t_in,hyp)
    K_oo    = np.asarray(K_oo,dtype=np.float64)
    K_oi    = np.asarray(K_oi,dtype=np.float64)
    K_ii    = np.asarray(K_ii,dtype=np.float64)

    I_M     = np.eye(M,dtype=np.float64)
    A       = K_ii + eps*I_M

    # Schur complement term: K_oi @ inv(A) @ K_oi^T  ==  K_oi @ solve(A,K_oi^T)
    with np.errstate(divide='ignore',over='ignore',invalid='ignore'):
        X = np.linalg.solve(A,K_oi.T)     # (M,L)
        T = K_oi @ X                      # (L,L)
        K = K_oo - T

    K = 0.5*(K + K.T) # ensure symmetry
    I_L    = np.eye(L,dtype=np.float64)
    K_chol = np.linalg.cholesky(K+eps*I_L,upper=upper)

    # Return
    if return_K:
        return K_chol, K
    return K_chol

def safe_matmul(A,B):
    """
    Matrix multiplication with RuntimeWarning suppressed.

    Parameters:
        A (Any): Input A.
        B (Any): Input B.

    Returns:
        Any: Result computed from the current inputs.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',category=RuntimeWarning)
        return A@B

def sample_trajs(K_chol,n_traj=100,use_uniform=False):
    """
    Sample trajectories from a Gaussian process with given Cholesky factor of
    covariance.

    Parameters:
        K_chol (Any): Input K chol.
        n_traj (int): Input n traj. Defaults to `100`.
        use_uniform (bool): Input use uniform. Defaults to `False`.

    Returns:
        Any: Result computed from the current inputs.
    """
    L = K_chol.shape[0]

    if use_uniform:
        U = np.random.uniform(low=0.0,high=1.0,size=(L,n_traj))
        Z = np.sqrt(12.0)*(U - 0.5)  # mean 0, var 1
    else:
        Z = np.random.randn(L,n_traj)

    trajs = safe_matmul(K_chol,Z) # shape: (L,n_traj)
    # Return
    return trajs # shape: (L,n_traj)

def sample_trajs_multidim(K_chol,n_traj=100,d=3,use_uniform=False,seed=None):
    """
    Sample multi-dimensional trajectories from a Gaussian process with given
    Cholesky factor of covariance.

    Parameters:
        K_chol (Any): Input K chol.
        n_traj (int): Input n traj. Defaults to `100`.
        d (int): Input d. Defaults to `3`.
        use_uniform (bool): Input use uniform. Defaults to `False`.
        seed (Any | None): Input seed. Defaults to `None`.

    Returns:
        Any: Result computed from the current inputs.
    """
    # Set seed
    if seed is not None:
        np.random.seed(seed=seed)

    # Get size
    L = K_chol.shape[0]

    if use_uniform:
        U = np.random.uniform(low=0.0,high=1.0,size=(L,n_traj,d))
        Z = np.sqrt(12.0)*(U - 0.5)
    else:
        Z = np.random.randn(L,n_traj,d)

    trajs = np.zeros((L,n_traj,d))
    for k in range(d):
        trajs[:,:,k] = safe_matmul(K_chol,Z[:,:,k]) # shape: (L,n_traj)

    return trajs # shape: (L,n_traj,d)

