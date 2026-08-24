import numpy as np
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


def grid_2d_uniform(x0_min,x0_max,x1_min,x1_max,n_x0,n_x1):
    """
    Build a uniformly sampled 2-D grid with a fixed number of bins.

    Parameters:
        x0_min (Any): Input x0 min.
        x0_max (Any): Input x0 max.
        x1_min (Any): Input x1 min.
        x1_max (Any): Input x1 max.
        n_x0 (int): Input n x0.
        n_x1 (int): Input n x1.

    Returns:
        Any: Result computed from the current inputs.
    """
    x0s,x1s = np.meshgrid(np.arange(x0_min,x0_max,(x0_max-x0_min)/n_x0),
                          np.arange(x1_min,x1_max,(x1_max-x1_min)/n_x1))
    xs = np.dstack([x0s,x1s]).reshape(-1,2)
    return xs


def add_colorbar(im,aspect=20,pad_fraction=0.5,**kwargs):
    """
    Add a vertical color bar to an image plot.

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
    width = axes_grid1.axes_size.AxesY(im.axes,aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction,width)
    current_ax = plt.gca()
    cax = divider.append_axes("right",size=width,pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im,cax=cax,**kwargs)


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
                 name        = "QM",
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
                 ]):
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

        xs,x0s,x1s = grid_2d(x0_min=self.axis_info[0],
                             x0_max=self.axis_info[1],
                             x1_min=self.axis_info[2],
                             x1_max=self.axis_info[3],
                             x0_step=self.step_coarse,
                             x1_step=self.step_coarse)
        self.xs_coarse  = xs
        self.x0s_coarse = x0s
        self.x1s_coarse = x1s
        self.prob = gmm(
            x       = self.xs_coarse,
            weights = self.gmm_weights,
            mus     = self.gmm_mus,
            sigmas  = self.gmm_sigmas,
        )
        self.tck_coarse = bisplrep(
            x  = self.xs_coarse[:,0],
            y  = self.xs_coarse[:,1],
            z  = self.prob/max(self.prob),
            s  = 0.001,
            kx = 3,
            ky = 3,
        )

        xs,x0s,x1s = grid_2d(x0_min=self.axis_info[0],
                             x0_max=self.axis_info[1],
                             x1_min=self.axis_info[2],
                             x1_max=self.axis_info[3],
                             x0_step=self.step_fine,
                             x1_step=self.step_fine)
        self.xs_fine  = xs
        self.x0s_fine = x0s
        self.x1s_fine = x1s
        self.Q = bisplev(
            np.arange(self.axis_info[0],self.axis_info[1]+self.step_fine,self.step_fine),
            np.arange(self.axis_info[2],self.axis_info[3]+self.step_fine,self.step_fine),
            self.tck_coarse,
        ).T

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
