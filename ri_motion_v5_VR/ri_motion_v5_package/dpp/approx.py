import numpy as np
from scipy.spatial import distance


def get_color_2d(x):
    """
    Map 2-D coordinates to simple RGB colors for visualization.

    Parameters:
        x (Any): Input x.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    c = np.concatenate((x[:,0:1],x[:,1:2]),axis=1)
    c = (c-np.min(x,axis=0))/(np.max(x,axis=0)-np.min(x,axis=0))
    r,g,b = 1.0-c[:,1:2],c[:,0:1],0.5-np.zeros_like(c[:,0:1])
    c = np.concatenate((r,g,b),axis=1)
    return c


def get_dpp_subset_from_K(K,n_sel):
    """
    Pick a subset greedily from a precomputed similarity matrix.

    Parameters:
        K (Any): Input K.
        n_sel (int): Input n sel.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    n_total     = K.shape[0]
    remain_idxs = np.arange(n_total)
    sub_idx     = np.zeros((n_sel))
    sum_K_vec   = np.zeros(n_total)
    for i_idx in range(n_sel):
        if i_idx == 0:
            sel_idx = np.random.randint(n_total)
        else:
            curr_K_vec = K[int(sub_idx[i_idx-1]),:]
            sum_K_vec  = sum_K_vec + curr_K_vec
            k_vals     = sum_K_vec[remain_idxs]
            min_idx    = np.argmin(k_vals)
            sel_idx    = remain_idxs[min_idx]
        sub_idx[i_idx] = int(sel_idx)
        remain_idxs = np.delete(remain_idxs,np.argwhere(remain_idxs==sel_idx))
    sub_idx = sub_idx.astype(np.int32)
    return sub_idx


def get_dpp_subset(x,n_sel,alpha=1000):
    """
    Pick a diverse subset from points using an RBF-style similarity matrix.

    Parameters:
        x (Any): Input x.
        n_sel (int): Input n sel.
        alpha (int): Input alpha. Defaults to `1000`.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    D = distance.cdist(x,x,"sqeuclidean")
    K = np.exp(-alpha*D)
    sub_idx = get_dpp_subset_from_K(K,n_sel)
    x_sel = x[sub_idx,:]
    return x_sel,sub_idx


def get_rand_subset(x,n_sel):
    """
    Draw a random subset of points without replacement.

    Parameters:
        x (Any): Input x.
        n_sel (int): Input n sel.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    n_total = x.shape[0]
    sub_idx = np.random.permutation(n_total)[:n_sel]
    x_sel = x[sub_idx,:]
    return x_sel,sub_idx
