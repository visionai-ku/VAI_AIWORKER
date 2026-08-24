import numpy as np

def brownian_kernel(X1,X2,params=None):
    """
    Brownian motion kernel.

    Parameters:
        X1 (Any): Input X1.
        X2 (Any): Input X2.
        params (Any | None): Input params. Defaults to `None`.

    Returns:
        Any: Result computed from the current inputs.
    """
    X1       = np.asarray(X1,dtype=np.float64)
    X2       = np.asarray(X2,dtype=np.float64)
    params   = {} if params is None else params
    variance = float(params.get('variance',1.0))

    if X1.ndim==1: X1=np.atleast_2d(X1).T
    if X2.ndim==1: X2=np.atleast_2d(X2).T

    if X1.shape[1]!=1:
        raise ValueError("X1 must have exactly one feature for brownian_kernel.")
    if X2.shape[1]!=1:
        raise ValueError("X2 must have exactly one feature for brownian_kernel.")

    t = X1[:,0].reshape(-1,1)
    s = X2[:,0].reshape(1,-1)

    K = np.minimum(t,s)
    return variance*K


def brownian_bridge_kernel(X1,X2,params=None):
    """
    Brownian bridge kernel.

    Parameters:
        X1 (Any): Input X1.
        X2 (Any): Input X2.
        params (Any | None): Input params. Defaults to `None`.

    Returns:
        Any: Result computed from the current inputs.
    """
    X1       = np.asarray(X1,dtype=np.float64)
    X2       = np.asarray(X2,dtype=np.float64)
    params   = {} if params is None else params
    T        = float(params.get('T',1.0))
    variance = float(params.get('variance',1.0))

    if T<=0.0:
        raise ValueError("T must be > 0.")

    if X1.ndim==1: X1=np.atleast_2d(X1).T
    if X2.ndim==1: X2=np.atleast_2d(X2).T

    if X1.shape[1]!=1:
        raise ValueError("X1 must have exactly one feature for brownian_bridge_kernel.")
    if X2.shape[1]!=1:
        raise ValueError("X2 must have exactly one feature for brownian_bridge_kernel.")

    t = X1[:,0].reshape(-1,1)
    s = X2[:,0].reshape(1,-1)

    K = np.minimum(t,s)-(t*s)/T
    return variance*K

def rbf_kernel(X1,X2,params=None):
    """
    Radial Basis Function (RBF) kernel (Gaussian kernel).

    Parameters:
        X1 (Any): Input X1.
        X2 (Any): Input X2.
        params (Any | None): Input params. Defaults to `None`.

    Returns:
        Any: Result computed from the current inputs.
    """
    X1           = np.asarray(X1,dtype=np.float64)
    X2           = np.asarray(X2,dtype=np.float64)
    params       = {} if params is None else params
    length_scale = float(params.get('length_scale',1.0))
    variance     = float(params.get('variance',1.0))

    if length_scale<=0.0:
        raise ValueError("length_scale must be > 0.")

    if X1.ndim==1: X1=np.atleast_2d(X1).T
    if X2.ndim==1: X2=np.atleast_2d(X2).T

    sqdist = np.sum(X1*X1,axis=1).reshape(-1,1)+np.sum(X2*X2,axis=1)-2.0*np.dot(X1,X2.T)
    sqdist = np.maximum(sqdist,0.0)

    return variance*np.exp(-0.5*sqdist/(length_scale*length_scale))

def rbf_bridge_kernel(X1,X2,params=None):
    """
    RBF bridge kernel by conditioning an RBF GP on f(0)=0 and f(T)=0.

    Parameters:
        X1 (Any): Input X1.
        X2 (Any): Input X2.
        params (Any | None): Input params. Defaults to `None`.

    Returns:
        Any: Result computed from the current inputs.
    """
    X1           = np.asarray(X1,dtype=np.float64)
    X2           = np.asarray(X2,dtype=np.float64)
    params       = {} if params is None else params
    length_scale = float(params.get('length_scale',1.0))
    variance     = float(params.get('variance',1.0))
    T            = float(params.get('T',1.0))
    eps          = float(params.get('eps',1e-12))

    if length_scale<=0.0:
        raise ValueError("length_scale must be > 0.")
    if T<=0.0:
        raise ValueError("T must be > 0.")

    if X1.ndim==1: X1=np.atleast_2d(X1).T
    if X2.ndim==1: X2=np.atleast_2d(X2).T

    if X1.shape[1]!=1:
        raise ValueError("X1 must have exactly one feature for rbf_bridge_kernel.")
    if X2.shape[1]!=1:
        raise ValueError("X2 must have exactly one feature for rbf_bridge_kernel.")

    Z    = np.array([[0.0],[T]],dtype=np.float64)
    hyp  = {'length_scale':length_scale,'variance':variance}

    K12  = rbf_kernel(X1,X2,hyp)
    K1Z  = rbf_kernel(X1,Z,hyp)
    KZZ  = rbf_kernel(Z,Z,hyp)
    KZ2  = rbf_kernel(Z,X2,hyp)

    KZZ_inv = np.linalg.inv(KZZ+eps*np.eye(KZZ.shape[0]))
    K       = K12-K1Z@KZZ_inv@KZ2
    return K

def linear_kernel(X1,X2,params=None):
    """
    Linear kernel.

    Parameters:
        X1 (Any): Input X1.
        X2 (Any): Input X2.
        params (Any | None): Input params. Defaults to `None`.

    Returns:
        Any: Result computed from the current inputs.
    """
    X1       = np.asarray(X1,dtype=np.float64)
    X2       = np.asarray(X2,dtype=np.float64)
    params   = {} if params is None else params
    variance = float(params.get('variance',1.0))

    if X1.ndim==1: X1=np.atleast_2d(X1).T
    if X2.ndim==1: X2=np.atleast_2d(X2).T

    return variance*np.dot(X1,X2.T)

def polynomial_kernel(X1,X2,params=None):
    """
    Polynomial kernel.

    Parameters:
        X1 (Any): Input X1.
        X2 (Any): Input X2.
        params (Any | None): Input params. Defaults to `None`.

    Returns:
        Any: Result computed from the current inputs.
    """
    params   = {} if params is None else params
    degree   = int(params.get('degree',3))
    coef0    = float(params.get('coef0',1.0))
    variance = float(params.get('variance',1.0))

    if degree<0:
        raise ValueError("degree must be >= 0.")

    if X1.ndim==1: X1=np.atleast_2d(X1).T
    if X2.ndim==1: X2=np.atleast_2d(X2).T

    return variance*(np.dot(X1,X2.T)+coef0)**degree


def matern_kernel(X1,X2,params=None):
    """
    Matern kernel.

    Parameters:
        X1 (Any): Input X1.
        X2 (Any): Input X2.
        params (Any | None): Input params. Defaults to `None`.

    Returns:
        Any: Result computed from the current inputs.
    """
    params       = {} if params is None else params
    length_scale = float(params.get('length_scale',1.0))
    variance     = float(params.get('variance',1.0))
    nu           = float(params.get('nu',1.5))

    if length_scale<=0.0:
        raise ValueError("length_scale must be > 0.")
    if nu<=0.0:
        raise ValueError("nu must be > 0.")

    if X1.ndim==1: X1=np.atleast_2d(X1).T
    if X2.ndim==1: X2=np.atleast_2d(X2).T

    sqdist = np.sum(X1*X1,axis=1).reshape(-1,1)+np.sum(X2*X2,axis=1)-2.0*np.dot(X1,X2.T)
    sqdist = np.maximum(sqdist,0.0)
    r      = np.sqrt(sqdist)

    if nu==0.5:
        k = variance*np.exp(-r/length_scale)

    elif nu==1.5:
        sqrt3 = np.sqrt(3.0)
        a     = sqrt3*r/length_scale
        k     = variance*(1.0+a)*np.exp(-a)

    elif nu==2.5:
        sqrt5 = np.sqrt(5.0)
        a     = sqrt5*r/length_scale
        k     = variance*(1.0+a+(a*a)/3.0)*np.exp(-a)

    else:
        try:
            from scipy.special import gamma,kv
        except Exception as e:
            raise ImportError("matern_kernel with nu not in {0.5,1.5,2.5} requires scipy (scipy.special.gamma,kv).") from e

        z = np.sqrt(2.0*nu)*r/length_scale

        # For z=0, define k=variance (limit value)
        k = np.empty_like(z,dtype=np.float64)
        mask0 = (z==0.0)
        mask1 = ~mask0

        # k = variance * 2^{1-nu}/Gamma(nu) * z^nu * K_nu(z)
        c = (2.0**(1.0-nu))/gamma(nu)
        k[mask0] = variance
        if np.any(mask1):
            zz       = z[mask1]
            k[mask1] = variance*c*(zz**nu)*kv(nu,zz)

    return k


def periodic_kernel(X1,X2,params=None):
    """
    Periodic kernel (using Euclidean distance in input space).

    Parameters:
        X1 (Any): Input X1.
        X2 (Any): Input X2.
        params (Any | None): Input params. Defaults to `None`.

    Returns:
        Any: Result computed from the current inputs.
    """
    params       = {} if params is None else params
    length_scale = float(params.get('length_scale',1.0))
    variance     = float(params.get('variance',1.0))
    period       = float(params.get('period',1.0))

    if length_scale<=0.0:
        raise ValueError("length_scale must be > 0.")
    if period<=0.0:
        raise ValueError("period must be > 0.")

    if X1.ndim==1: X1=np.atleast_2d(X1).T
    if X2.ndim==1: X2=np.atleast_2d(X2).T

    sqdist = np.sum(X1*X1,axis=1).reshape(-1,1)+np.sum(X2*X2,axis=1)-2.0*np.dot(X1,X2.T)
    sqdist = np.maximum(sqdist,0.0)
    r      = np.sqrt(sqdist)

    s = np.sin(np.pi*r/period)
    return variance*np.exp(-2.0*(s*s)/(length_scale*length_scale))


def rational_quadratic_kernel(X1,X2,params=None):
    """
    Rational Quadratic kernel.

    Parameters:
        X1 (Any): Input X1.
        X2 (Any): Input X2.
        params (Any | None): Input params. Defaults to `None`.

    Returns:
        Any: Result computed from the current inputs.
    """
    params       = {} if params is None else params
    length_scale = float(params.get('length_scale',1.0))
    variance     = float(params.get('variance',1.0))
    alpha        = float(params.get('alpha',1.0))

    if length_scale<=0.0:
        raise ValueError("length_scale must be > 0.")
    if alpha<=0.0:
        raise ValueError("alpha must be > 0.")

    if X1.ndim==1: X1=np.atleast_2d(X1).T
    if X2.ndim==1: X2=np.atleast_2d(X2).T

    sqdist = np.sum(X1*X1,axis=1).reshape(-1,1)+np.sum(X2*X2,axis=1)-2.0*np.dot(X1,X2.T)
    sqdist = np.maximum(sqdist,0.0)

    return variance*(1.0+sqdist/(2.0*alpha*length_scale*length_scale))**(-alpha)
