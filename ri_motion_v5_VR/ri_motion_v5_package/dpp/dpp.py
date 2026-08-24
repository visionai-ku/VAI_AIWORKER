import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
from scipy.interpolate import bisplrep,bisplev

def grid_2d(x0_min,x0_max,x1_min,x1_max,x0_step,x1_step):
    """
    Build a 2-D dense grid and return both flattened and meshgrid views.

    Parameters:
        x0_min (Any): Input x0 min.
        x0_max (Any): Input x0 max.
        x1_min (Any): Input x1 min.
        x1_max (Any): Input x1 max.
        x0_step (Any): Input x0 step.
        x1_step (Any): Input x1 step.

    Returns:
        Any: Result computed from the current inputs.
    """
    x0s,x1s = np.meshgrid(np.arange(x0_min,x0_max+x0_step,x0_step),
                          np.arange(x1_min,x1_max+x1_step,x1_step))
    xs = np.dstack([x0s,x1s]).reshape(-1,2)
    return xs,x0s,x1s

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """
    Append a compact vertical colorbar to an existing image axis.

    Parameters:
        im (Any): Input im.
        aspect (int): Input aspect. Defaults to `20`.
        pad_fraction (float): Input pad fraction. Defaults to `0.5`.
        **kwargs (Any): Additional keyword arguments.

    Returns:
        Any: Result computed from the current inputs.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits import axes_grid1
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

def gmm(x,weights,mus,sigmas):
    """
    Evaluate a Gaussian mixture model on a set of 2-D query points.

    Parameters:
        x (Any): Input x.
        weights (Any): Input weights.
        mus (Any): Input mus.
        sigmas (Any): Input sigmas.

    Returns:
        Any: Result computed from the current inputs.
    """
    x = np.asarray(x,dtype=np.float64)
    prob = np.zeros((x.shape[0],),dtype=np.float64)
    weight_sum = float(np.asarray(weights,dtype=np.float64).sum())
    for weight,mu,sigma in zip(weights,mus,sigmas):
        prob += float(weight)/weight_sum*multivariate_normal.pdf(x,mean=mu,cov=sigma)
    prob = prob.reshape(-1,1)
    return prob

class QualityMapClass(object):
    """
    Store and evaluate 2D quality-map samples.

    Usage:
        obj = QualityMapClass(...)
    """
    def __init__(self,
                 name        = 'QM',
                 axis_info   = [-1.5,1.5,-1,1],
                 step_coarse = 0.05,
                 step_fine   = 0.01,
                 gmm_weights = [1,2,3],
                 gmm_mus     = [
                     np.array([-1.0,-0.5]),
                     np.array([0.0,0.0]),
                     np.array([1.0,0.5])
                     ],
                 gmm_sigmas  = [
                     0.1*np.array([[1.0,0.0],[0.0,1.0]]),
                     0.1*np.array([[1.0,0.0],[0.0,1.0]]),
                     0.1*np.array([[1.0,0.0],[0.0,1.0]])
                     ]
                ):
        """
        Initialize QualityMapClass.

        Parameters:
            name (str): Input name. Defaults to `'QM'`.
            axis_info (list): Input axis info. Defaults to `[-1.5, 1.5, -1, 1]`.
            step_coarse (float): Input step coarse. Defaults to `0.05`.
            step_fine (float): Input step fine. Defaults to `0.01`.
            gmm_weights (list): Input gmm weights. Defaults to `[1, 2, 3]`.
            gmm_mus (list): Input gmm mus. Defaults to `[np.array([-1.0, -0.5]),
                            np.array([0.0, 0.0]), np.array([1.0, 0.5])]`.
            gmm_sigmas (list): Input gmm sigmas. Defaults to `[0.1 * np.array([[1.0,
                               0.0], [0.0, 1.0]]), 0.1 * np.array([[1.0, 0.0], [0.0,
                               1.0]]), 0.1 * np.array([[1.0, 0.0], [0.0, 1.0]])]`.

        Returns:
            None: Initializes the instance in place.
        """
        self.name        = name
        self.axis_info   = axis_info
        self.step_coarse = step_coarse
        self.step_fine   = step_fine
        self.gmm_weights = gmm_weights
        self.gmm_mus     = gmm_mus
        self.gmm_sigmas  = gmm_sigmas

        # Build a coarse GMM surface first, then spline-fit it for fast lookup.
        xs,x0s,x1s = grid_2d(x0_min  = self.axis_info[0],
                             x0_max  = self.axis_info[1],
                             x1_min  = self.axis_info[2],
                             x1_max  = self.axis_info[3],
                             x0_step = self.step_coarse,
                             x1_step = self.step_coarse)
        self.xs_coarse  = xs
        self.x0s_coarse = x0s
        self.x1s_coarse = x1s
        self.prob = gmm(
            x       = self.xs_coarse,
            weights = self.gmm_weights,
            mus     = self.gmm_mus,
            sigmas  = self.gmm_sigmas)
        self.tck_coarse = bisplrep(
            x  = self.xs_coarse[:,0],
            y  = self.xs_coarse[:,1],
            z  = self.prob/max(self.prob),
            s  = 0.001,
            kx = 3,
            ky = 3)
        # Evaluate a denser map from the coarse spline for plotting and sampling.
        xs,x0s,x1s = grid_2d(x0_min  = self.axis_info[0],
                             x0_max  = self.axis_info[1],
                             x1_min  = self.axis_info[2],
                             x1_max  = self.axis_info[3],
                             x0_step = self.step_fine,
                             x1_step = self.step_fine)
        self.xs_fine  = xs
        self.x0s_fine = x0s
        self.x1s_fine = x1s
        self.Q    = bisplev(
            np.arange(self.axis_info[0],self.axis_info[1]+self.step_fine,self.step_fine),
            np.arange(self.axis_info[2],self.axis_info[3]+self.step_fine,self.step_fine),
            self.tck_coarse).T

    def get_meas(self,n_meas=100):
        """
        Sample positions uniformly in the map bounds and query quality values.

        Parameters:
            n_meas (int): Input n meas. Defaults to `100`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        x_meas = np.array([self.axis_info[0],self.axis_info[2]]) \
            + np.array([self.axis_info[1]-self.axis_info[0],
                        self.axis_info[3]-self.axis_info[2]])*np.random.rand(n_meas,2)
        y_meas = np.zeros(shape=(n_meas))
        for i_idx in range(n_meas):
            y_meas[i_idx] = bisplev(
                x=x_meas[i_idx,0],y=x_meas[i_idx,1],tck=self.tck_coarse)
        return x_meas,y_meas

def kernel_se(X1,X2,hyp={'g':1.0,'l':1.0}):
    """
    Evaluate the squared-exponential kernel matrix between two point sets.

    Parameters:
        X1 (Any): Input X1.
        X2 (Any): Input X2.
        hyp (dict): Input hyp. Defaults to `{'g': 1.0, 'l': 1.0}`.

    Returns:
        Any: Result computed from the current inputs.
    """
    K = hyp['g']*np.exp(-cdist(X1,X2,'sqeuclidean')/(2*hyp['l']*hyp['l']))
    return K

def block_mtx(M11,M12,M21,M22):
    """
    Assemble a block matrix from four submatrices.

    Parameters:
        M11 (Any): Input M11.
        M12 (Any): Input M12.
        M21 (Any): Input M21.
        M22 (Any): Input M22.

    Returns:
        Any: Result computed from the current inputs.
    """
    M_upper = np.concatenate((M11,M12),axis=1)
    M_lower = np.concatenate((M21,M22),axis=1)
    M = np.concatenate((M_upper,M_lower),axis=0)
    return M

def inv_inc(inv_A,b,c):
    """
    Update the inverse of a kernel matrix after appending one sample.

    Parameters:
        inv_A (Any): Input inv A.
        b (Any): Input b.
        c (Any): Input c.

    Returns:
        Any: Result computed from the current inputs.
    """
    k   = c - b.T @ inv_A @ b
    M11 = inv_A + 1/k * inv_A @ b @ b.T @ inv_A
    M12 = -1/k * inv_A @ b
    M21 = -1/k * b.T @ inv_A
    M22 = 1/k
    M   = block_mtx(M11=M11,M12=M12,M21=M21,M22=M22)
    return M

def det_inc(det_A,inv_A,b,c):
    """
    Update the determinant of a matrix after appending one sample.

    Parameters:
        det_A (Any): Input det A.
        inv_A (Any): Input inv A.
        b (Any): Input b.
        c (Any): Input c.

    Returns:
        Any: Result computed from the current inputs.
    """
    out = det_A * (c - b.T @ inv_A @ b)
    return out

def kdpp(
    xs_total,
    qs_total = None,
    n_select = 10,
    hyp      = {'g':1.0,'l':1.0}
    ):
    """
    Greedily select a diverse subset by maximizing kernel determinants.

    Parameters:
        xs_total (Any): Input xs total.
        qs_total (Any | None): Input qs total. Defaults to `None`.
        n_select (int): Input n select. Defaults to `10`.
        hyp (dict): Input hyp. Defaults to `{'g': 1.0, 'l': 1.0}`.

    Returns:
        Any: Result computed from the current inputs.
    """
    n_total     = xs_total.shape[0]
    idxs_remain = np.arange(0,n_total,1,dtype=np.int32)
    idxs_select = []
    for i_idx in range(n_select+1): # loop
        n_remain = len(idxs_remain)
        if i_idx == 0: # random first
            idx_select = np.random.permutation(n_total)[0]
        else:
            xs_select = xs_total[idxs_select,:]
            # Test each remaining point by appending it to the current subset.
            dets = np.zeros(shape=n_remain)
            for r_idx in range(n_remain): # for the remained indices
                idx_check  = idxs_remain[r_idx]
                idxs_check = idxs_select + [idx_check]
                xs_check   = xs_total[idxs_check,:]
                # Larger determinant implies a more diverse-and-spread subset.
                K_check    = kernel_se(xs_check,xs_check,hyp=hyp)
                if qs_total is not None:
                    K_check = K_check + np.diag(qs_total[idxs_check])
                det_check  = np.linalg.det(K_check)
                # Append the determinant
                dets[r_idx] = det_check
            # Get the index with the highest determinant
            idx_select = idxs_remain[np.where(dets == np.amax(dets))[0][0]]
        # Remove currently selected index from 'idxs_remain'
        idxs_remain = idxs_remain[idxs_remain != idx_select]
        # Append currently selected index to 'idxs_select'
        idxs_select.append(idx_select)
    # Select the subset from 'xs_total' with removing the first sample
    idxs_select = idxs_select[1:] # excluding the first one
    idxs_kdpp   = np.array(idxs_select)
    xs_kdpp     = xs_total[idxs_kdpp]
    return xs_kdpp,idxs_kdpp

def ikdpp(
    xs_total,
    qs_total = None,
    n_select = 10,
    n_trunc  = np.inf,
    hyp      = {'g':1.0,'l':1.0}
    ):
    """
    Greedy k-DPP selection using incremental determinant/inverse updates.

    Parameters:
        xs_total (Any): Input xs total.
        qs_total (Any | None): Input qs total. Defaults to `None`.
        n_select (int): Input n select. Defaults to `10`.
        n_trunc (int): Input n trunc. Defaults to `np.inf`.
        hyp (dict): Input hyp. Defaults to `{'g': 1.0, 'l': 1.0}`.

    Returns:
        Any: Result computed from the current inputs.
    """
    n_total     = xs_total.shape[0]
    idxs_remain = np.arange(0,n_total,1,dtype=np.int32)
    idxs_select = []
    for i_idx in range(n_select+1): # loop
        n_remain = len(idxs_remain)
        if i_idx == 0: # random first
            idx_select = np.random.permutation(n_total)[0]
            if qs_total is not None:
                q = 1.0+qs_total[idx_select]
            else:
                q = 1.0
            det_K_prev = q
            K_inv_prev = 1/q*np.ones(shape=(1,1))
        else:
            xs_select = xs_total[idxs_select,:]
            # Only a truncated random subset of candidates is checked when
            # `n_trunc` is finite, which trades optimality for speed.
            dets = np.zeros(shape=n_remain)
            for r_idx in np.random.permutation(n_remain)[:min(n_remain,n_trunc)]:
                # Reuse the previous inverse/determinant instead of rebuilding
                # the whole kernel matrix for each candidate.
                k_vec     = kernel_se(
                    X1  = xs_select,
                    X2  = xs_total[idxs_remain[r_idx],:].reshape(1,-1),
                    hyp = hyp)
                q = 1.0+qs_total[idxs_remain[r_idx]] if qs_total is not None else 1.0
                det_check = det_inc(
                    det_A = det_K_prev,
                    inv_A = K_inv_prev,
                    b     = k_vec,
                    c     = q)
                # Append the determinant
                dets[r_idx] = det_check
            # Get the index with the highest determinant
            idx_temp   = np.where(dets == np.amax(dets))[0][0]
            idx_select = idxs_remain[idx_temp]

            # Compute 'det_K_prev' and 'K_inv_prev'
            det_K_prev = dets[idx_temp]
            k_vec      = kernel_se(
                xs_select,
                xs_total[idx_select,:].reshape(1,-1),
                hyp=hyp)
            if qs_total is not None:
                q = 1+qs_total[idx_select]
            else:
                q = 1.0
            K_inv_prev = inv_inc(
                inv_A = K_inv_prev,
                b     = k_vec,
                c     = q)
        # Remove currently selected index from 'idxs_remain'
        idxs_remain = idxs_remain[idxs_remain != idx_select]
        # Append currently selected index to 'idxs_select'
        idxs_select.append(idx_select)
    # Select the subset from 'xs_total' with removing the first sample
    idxs_select = idxs_select[1:] # excluding the first one
    idxs_ikdpp  = np.array(idxs_select)
    xs_ikdpp    = xs_total[idxs_ikdpp]
    return xs_ikdpp,idxs_ikdpp
