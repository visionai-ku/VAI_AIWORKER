import numpy as np

def set_forward_jerk_row(D,r,c0,scale):
    """
    Fill one forward one-sided jerk row.

    Parameters:
        D (Any): Input D.
        r (Any): Input r.
        c0 (Any): Input c0.
        scale (Any): Input scale.

    Returns:
        None: This function mutates state, renders output, or performs side
              effects in place.
    """
    D[r,c0+0] = -1.0*scale
    D[r,c0+1] =  3.0*scale
    D[r,c0+2] = -3.0*scale
    D[r,c0+3] =  1.0*scale

def set_backward_jerk_row(D,r,c0,scale):
    """
    Fill one backward one-sided jerk row.

    Parameters:
        D (Any): Input D.
        r (Any): Input r.
        c0 (Any): Input c0.
        scale (Any): Input scale.

    Returns:
        None: This function mutates state, renders output, or performs side
              effects in place.
    """
    D[r,c0+0] =  1.0*scale
    D[r,c0+1] = -3.0*scale
    D[r,c0+2] =  3.0*scale
    D[r,c0+3] = -1.0*scale

def diff_matrix_same_length(n:int,dt:float,order:int)->np.ndarray:
    """
    Create a finite-difference differentiation matrix of specified order.

    Parameters:
        n (int): Input n.
        dt (float): Input dt.
        order (int): Input order.

    Returns:
        np.ndarray: Result computed from the current inputs.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be > 0")
    if order not in (1,2,3):
        raise ValueError("order must be 1,2,or3")

    D = np.zeros((n,n),dtype=float)

    if n == 1:
        return D

    h  = float(dt)
    h2 = h*h
    h3 = h2*h

    if order == 1:
        # Row 0: forward [-1,1]/h
        D[0,0] = -1.0/h
        D[0,1] =  1.0/h

        if n == 2:
            # Row 1: backward [-1,1]/h on (0,1)
            D[1,0] = -1.0/h
            D[1,1] =  1.0/h
            return D

        # Rows 1..n-2: central [-1/2,0,1/2]/h
        for i in range(1,n-1):
            D[i,i-1] = -0.5/h
            D[i,i+1] =  0.5/h

        # Row n-1: backward [-1,1]/h on (n-2,n-1)
        D[n-1,n-2] = -1.0/h
        D[n-1,n-1] =  1.0/h
        return D

    if order == 2:
        s = 1.0/h2

        if n >= 4:
            # Row 0: 2nd-order-accurate forward second derivative [2,-5,4,-1]/h^2
            D[0,0] =  2.0*s
            D[0,1] = -5.0*s
            D[0,2] =  4.0*s
            D[0,3] = -1.0*s

            # Row n-1: 2nd-order-accurate backward [-1,4,-5,2]/h^2
            D[n-1,n-4] = -1.0*s
            D[n-1,n-3] =  4.0*s
            D[n-1,n-2] = -5.0*s
            D[n-1,n-1] =  2.0*s

            # Interior rows 1..n-2: central [1,-2,1]/h^2
            for i in range(1,n-1):
                D[i,i-1] =  1.0*s
                D[i,i]   = -2.0*s
                D[i,i+1] =  1.0*s
            return D

        if n == 3:
            # Use simple central everywhere [1,-2,1]/h^2 for rows 0..2
            for i in range(0,3):
                if i == 0:
                    D[i,0] =  1.0*s
                    D[i,1] = -2.0*s
                    D[i,2] =  1.0*s
                elif i == 1:
                    D[i,0] =  1.0*s
                    D[i,1] = -2.0*s
                    D[i,2] =  1.0*s
                else:
                    D[i,0] =  1.0*s
                    D[i,1] = -2.0*s
                    D[i,2] =  1.0*s
            return D

        # n == 2: second derivative not well-defined; return zeros
        return D

    # order == 3
    # For robust small-n behavior, use:
    # - if n >= 5: central 5-point for i=2..n-3, and 4-point one-sided for i=0,1,n-2,n-1
    # - if n == 4: use 4-point one-sided for all rows (forward for first two, backward for last two)
    # - if n <= 3: fall back to repeated first derivative (D1 @ D1 @ D1), which is coarse but safe
    if n <= 3:
        D1 = diff_matrix_same_length(n,h,order=1)
        return D1 @ D1 @ D1

    s_f = 1.0/h3
    s_c = 1.0/(2.0*h3)

    if n == 4:
        # Rows 0,1: forward on (0..3)
        set_forward_jerk_row(D,0,0,s_f)
        set_forward_jerk_row(D,1,0,s_f)
        # Rows 2,3: backward on (0..3)
        set_backward_jerk_row(D,2,0,s_f)
        set_backward_jerk_row(D,3,0,s_f)
        return D

    # n >= 5
    # Rows 0,1: forward starting at 0 and 1
    set_forward_jerk_row(D,0,0,s_f)
    set_forward_jerk_row(D,1,1,s_f)

    # Interior rows: i = 2..n-3 use central 5-point [1,-2,0,2,-1]/(2*h^3)
    for i in range(2,n-2):
        D[i,i-2] =  1.0*s_c
        D[i,i-1] = -2.0*s_c
        # D[i,i] = 0
        D[i,i+1] =  2.0*s_c
        D[i,i+2] = -1.0*s_c

    # Rows n-2,n-1: backward ending at n-2 and n-1
    set_backward_jerk_row(D,n-2,n-5,s_f)  # uses (n-5..n-2)
    set_backward_jerk_row(D,n-1,n-4,s_f)  # uses (n-4..n-1)
    return D

def get_A_vel_acc_jerk_same_length(n:int=100,dt:float=1e-2):
    """
    Get finite-difference differentiation matrices for velocity, acceleration, and
    jerk.

    Parameters:
        n (int): Input n. Defaults to `100`.
        dt (float): Input dt. Defaults to `0.01`.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    A_vel  = diff_matrix_same_length(n,dt,order=1)
    A_acc  = diff_matrix_same_length(n,dt,order=2)
    A_jerk = diff_matrix_same_length(n,dt,order=3)
    return A_vel,A_acc,A_jerk

def get_interp_const_vel_traj_nd(
        anchors,    # [M x D]
        vel      = 1.0,
        Hz       = 100,
        ord      = np.inf,
        min_time = None,
    ):
    """
    Get a piecewise-linear, constant-velocity interpolated trajectory through
    given anchor points.

    Parameters:
        anchors (Any): Input anchors.
        vel (float): Input vel. Defaults to `1.0`.
        Hz (int): Input Hz. Defaults to `100`.
        ord (Any): Input ord. Defaults to `np.inf`.
        min_time (Any | None): Input min time. Defaults to `None`.

    Returns:
        Any: Requested value computed from the current inputs.
    """

    # Input validation
    anchors = np.asarray(anchors,dtype=float)

    # Checks on inputs
    if anchors.ndim != 2:
        raise ValueError("anchors must be 2D [M x D].")

    # Get sizes
    M,D = anchors.shape

    # Input sanity checks
    if M < 1:
        raise ValueError("anchors must have at least 1 point.")
    if vel <= 0.0:
        raise ValueError("vel must be > 0.")
    if Hz <= 0:
        raise ValueError("Hz must be > 0.")
    if min_time is not None and min_time < 0.0:
        raise ValueError("min_time must be >= 0 or None.")

    # Time step
    dt = 1.0/float(Hz)

    # Single anchor
    if M == 1:
        T_use          = 0.0 if min_time is None else float(min_time)
        times_interp   = np.arange(0.0,T_use+0.5*dt,dt)
        anchors_interp = np.repeat(anchors[:1],len(times_interp),axis=0)
        return {
            "times_interp":times_interp,
            "anchors_interp":anchors_interp,
            "times_anchor":np.array([0.0]),
            "idxs_anchor":np.array([0],dtype=int),
        }

    # Segment distances between anchors
    diffs = anchors[1:]-anchors[:-1]
    seg_d = np.linalg.norm(diffs,ord=ord,axis=1)

    # Anchor times (constant-velocity baseline)
    times_anchor0 = np.concatenate(([0.0],np.cumsum(seg_d/vel)))
    T0            = times_anchor0[-1]

    # Degenerate case (all anchors identical)
    if T0 <= 0.0:
        T_use          = T0 if min_time is None else max(T0,float(min_time))
        times_interp   = np.arange(0.0,T_use+0.5*dt,dt)
        anchors_interp = np.repeat(anchors[:1],len(times_interp),axis=0)
        return {
            "times_interp":times_interp,
            "anchors_interp":anchors_interp,
            "times_anchor":times_anchor0,
            "idxs_anchor":np.array([0],dtype=int),
        }

    # Uniform time scaling to satisfy min_time
    if min_time is not None and float(min_time) > T0:
        scale        = float(min_time)/T0
        times_anchor = times_anchor0*scale
    else:
        scale        = 1.0
        times_anchor = times_anchor0

    T_use = times_anchor[-1]

    # Interpolated times (strict Hz grid)
    times_interp = np.arange(0.0,T_use+0.5*dt,dt)
    L            = len(times_interp)

    # Interpolated trajectory (piecewise-linear in time)
    anchors_interp = np.empty((L,D),dtype=float)
    for d in range(D):
        anchors_interp[:,d] = np.interp(times_interp,times_anchor,anchors[:,d])

    # For each anchor time, find the closest interpolated time index
    idx_ai = np.searchsorted(times_interp,times_anchor,side="left")
    idx_ai = np.clip(idx_ai,1,len(times_interp)-1)
    left   = idx_ai-1
    right  = idx_ai
    choose_right = (times_anchor-times_interp[left]) > (times_interp[right]-times_anchor)
    idxs_anchor  = np.where(choose_right,right,left).astype(int)

    # Exact-hit override on samples (no continuity/jerk concerns)
    # - Always force endpoints to match
    anchors_interp[0,:]  = anchors[0,:]
    anchors_interp[-1,:] = anchors[-1,:]

    # - Force all anchors at their nearest sampled indices
    #   If duplicates occur, later anchors overwrite earlier ones.
    for i in range(M):
        anchors_interp[idxs_anchor[i],:] = anchors[i,:]

    return {
        "times_interp":times_interp,      # [L,]
        "anchors_interp":anchors_interp,  # [L x D]
        "times_anchor":times_anchor,      # [M,]
        "idxs_anchor":idxs_anchor,        # [M,]
    }
