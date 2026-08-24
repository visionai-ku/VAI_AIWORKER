import numpy as np

class VectorizedKalmanFilters:
    """
    VectorizedKalmanFilters is a class that implements a vectorized Kalman filter.

    Usage:
        obj = VectorizedKalmanFilters(...)
    """
    def __init__(self, shape, dt=0.05, q=0.1, r=1.0):
        """
        Initialize VectorizedKalmanFilters.

        Parameters:
            shape (Any): Input shape.
            dt (float): Input dt. Defaults to `0.05`.
            q (float): Input q. Defaults to `0.1`.
            r (float): Input r. Defaults to `1.0`.

        Returns:
            None: Initializes the instance in place.
        """
        self.shape = shape
        self.N = np.prod(shape)

        self.dt = dt
        self.F = np.array([[1, dt], [0, 1]])  # (2,2)
        self.H = np.array([[1, 0]])           # (1,2)
        self.Q = np.array([[q, 0], [0, q]])
        self.R = np.array([[r]])

        # Broadcast to (N, 2, 2), (N, 2, 1)
        self.Fs = np.broadcast_to(self.F, (self.N, 2, 2))
        self.Hs = np.broadcast_to(self.H, (self.N, 1, 2))
        self.Qs = np.broadcast_to(self.Q, (self.N, 2, 2))
        self.Rs = np.broadcast_to(self.R, (self.N, 1, 1))

        self.x_hat = np.zeros((self.N, 2, 1))                   # [pos, vel]^T
        self.P_hat = np.tile(self.Q[None, :, :], (self.N, 1, 1)) # cov
        self._initialized = False

    def update(self, obs, reset=False):  # obs: self.shape (e.g., (5,4,2,3))
        """
        Update the Kalman filters with new observations.

        Parameters:
            obs (Any): Input obs.
            reset (bool): Input reset. Defaults to `False`.

        Returns:
            Any: Result computed from the current inputs.
        """
        obs_flat_1d = obs.reshape(self.N)  # (N,)

        # Hard reset / first-time init
        if reset or not self._initialized:
            self.x_hat[:, 0, 0] = obs_flat_1d          # pos = obs
            self.x_hat[:, 1, 0] = 0.0                  # vel = 0
            self.P_hat = self.Qs.copy()                # reset covariance
            self._initialized = True
            return self.x_hat[:, 0, 0].reshape(self.shape)

        # ------- Standard predict-update -------
        obs_flat = obs_flat_1d[:, None, None]          # (N,1,1)

        # Predict
        x_hat_m = self.Fs @ self.x_hat
        P_hat_m = self.Fs @ self.P_hat @ self.Fs.transpose(0, 2, 1) + self.Qs

        # Kalman gain
        S = self.Hs @ P_hat_m @ self.Hs.transpose(0, 2, 1) + self.Rs  # (N,1,1)
        K = P_hat_m @ self.Hs.transpose(0, 2, 1) @ np.linalg.inv(S)   # (N,2,1)

        # Update
        y_diff = obs_flat - (self.Hs @ x_hat_m)  # innovation
        self.x_hat = x_hat_m + K @ y_diff
        self.P_hat = P_hat_m - K @ self.Hs @ P_hat_m

        return self.x_hat[:, 0, 0].reshape(self.shape)

class VectorizedKalmanFiltersFast(object):
    """
    Vectorized Kalman filters for many independent 1D signals, using state
    [pos,vel]^T.

    Usage:
        obj = VectorizedKalmanFiltersFast(...)
    """

    def __init__(
            self,
            shape,
            dt     = 0.05,
            q      = 0.1,
            r      = 1.0,
            q_mode = "acc",
            dtype  = np.float32,
            p0     = 10.0,
        ):
        """
        Initialize VectorizedKalmanFiltersFast.

        Parameters:
            shape (Any): Input shape.
            dt (float): Input dt. Defaults to `0.05`.
            q (float): Input q. Defaults to `0.1`.
            r (float): Input r. Defaults to `1.0`.
            q_mode (str): Input q mode. Defaults to `'acc'`.
            dtype (float): Input dtype. Defaults to `np.float32`.
            p0 (float): Input p0. Defaults to `10.0`.

        Returns:
            None: Initializes the instance in place.
        """
        self.shape         = tuple(shape)
        self.N             = int(np.prod(self.shape))

        self.dt            = float(dt)
        self.q             = float(q)
        self.r             = float(r)
        self.q_mode        = str(q_mode)
        self.dtype         = dtype
        self.p0            = float(p0)

        self.F             = np.array([[1.0,self.dt],[0.0,1.0]],dtype=self.dtype)
        self.H             = np.array([[1.0,0.0]],dtype=self.dtype)
        self.R             = np.array([[self.r]],dtype=self.dtype)

        self._refresh_Q()

        self.x_hat         = np.zeros((self.N,2,1),dtype=self.dtype)
        self.P_hat         = np.zeros((self.N,2,2),dtype=self.dtype)
        self.P_hat[:,0,0]  = self.p0
        self.P_hat[:,1,1]  = self.p0

        self._initialized  = False

    def _refresh_Q(self):
        """
        Refresh the process-noise covariance matrix.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        dt = self.dt
        q  = self.q

        if self.q_mode.lower() in ["diag","simple"]:
            self.Q00 = q
            self.Q01 = 0.0
            self.Q10 = 0.0
            self.Q11 = q
        else:
            # white acceleration, q interpreted as sigma_a^2
            self.Q00 = (0.25*(dt**4))*q
            self.Q01 = (0.50*(dt**3))*q
            self.Q10 = self.Q01
            self.Q11 = ((dt**2))*q

        self.Q   = np.array([[self.Q00,self.Q01],[self.Q10,self.Q11]],dtype=self.dtype)

    def set_dt(self,dt):
        """
        Set dt.

        Parameters:
            dt (float): Input dt.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.dt = float(dt)
        self.F  = np.array([[1.0,self.dt],[0.0,1.0]],dtype=self.dtype)
        self._refresh_Q()

    def reset(self,obs,p0=None):
        """
        Reset internal state.

        Parameters:
            obs (Any): Input obs.
            p0 (Any | None): Input p0. Defaults to `None`.

        Returns:
            Any: Result computed from the current inputs.
        """
        obs_flat_1d        = np.asarray(obs,dtype=self.dtype).reshape(self.N)

        self.x_hat[:,0,0]  = obs_flat_1d
        self.x_hat[:,1,0]  = 0.0

        if p0 is None:
            p0 = self.p0
        else:
            p0 = float(p0)

        self.P_hat[:]      = 0.0
        self.P_hat[:,0,0]  = p0
        self.P_hat[:,1,1]  = p0

        self._initialized  = True
        return self.x_hat[:,0,0].reshape(self.shape)

    def update(self,obs,reset=False,dt=None):
        """
        Update the object state from the latest inputs.

        Parameters:
            obs (Any): Input obs.
            reset (bool): Input reset. Defaults to `False`.
            dt (Any | None): Input dt. Defaults to `None`.

        Returns:
            Any: Result computed from the current inputs.
        """
        if dt is not None:
            self.set_dt(dt)

        obs_flat_1d = np.asarray(obs,dtype=self.dtype).reshape(self.N)

        if reset or (not self._initialized):
            self.x_hat[:,0,0] = obs_flat_1d
            self.x_hat[:,1,0] = 0.0
            self._initialized = True
            return self.x_hat[:,0,0].reshape(self.shape)

        # Predict
        x0        = self.x_hat[:,0,0]
        v0        = self.x_hat[:,1,0]

        x_m       = x0 + self.dt*v0
        v_m       = v0

        P00       = self.P_hat[:,0,0]
        P01       = self.P_hat[:,0,1]
        P10       = self.P_hat[:,1,0]
        P11       = self.P_hat[:,1,1]

        dt        = self.dt
        P00_m     = P00 + dt*(P10+P01) + (dt*dt)*P11 + self.Q00
        P01_m     = P01 + dt*P11                 + self.Q01
        P10_m     = P10 + dt*P11                 + self.Q10
        P11_m     = P11                          + self.Q11

        # Update (no inv)
        y         = obs_flat_1d - x_m
        S         = P00_m + self.r
        S_inv     = 1.0 / S

        K0        = P00_m * S_inv
        K1        = P10_m * S_inv

        x_new     = x_m + K0*y
        v_new     = v_m + K1*y

        # Joseph covariance
        a00       = 1.0 - K0
        a01       = 0.0
        a10       = -K1
        a11       = 1.0

        AP00      = a00*P00_m + a01*P10_m
        AP01      = a00*P01_m + a01*P11_m
        AP10      = a10*P00_m + a11*P10_m
        AP11      = a10*P01_m + a11*P11_m

        P00_new   = AP00*a00 + AP01*a01
        P01_new   = AP00*a10 + AP01*a11
        P10_new   = AP10*a00 + AP11*a01
        P11_new   = AP10*a10 + AP11*a11

        Kr        = self.r
        P00_new  += (K0*K0)*Kr
        P01_new  += (K0*K1)*Kr
        P10_new  += (K1*K0)*Kr
        P11_new  += (K1*K1)*Kr

        self.x_hat[:,0,0]  = x_new
        self.x_hat[:,1,0]  = v_new

        self.P_hat[:,0,0]  = P00_new
        self.P_hat[:,0,1]  = P01_new
        self.P_hat[:,1,0]  = P10_new
        self.P_hat[:,1,1]  = P11_new

        return self.x_hat[:,0,0].reshape(self.shape)

    def predict(self,alpha=0.5):
        """
        Predict the next state based on the current state and a given alpha.

        Parameters:
            alpha (float): Input alpha. Defaults to `0.5`.

        Returns:
            Any: Result computed from the current inputs.
        """
        x0 = self.x_hat[:,0,0] # current position
        v0 = self.x_hat[:,1,0] # current velocity
        # Prediction: x_pred = x0 + alpha * v0
        x_pred = x0 + alpha*v0
        # Return the predicted positions reshaped to the original observation shape
        return x_pred.reshape(self.shape)

# ---------------------------------------------------------------------------
