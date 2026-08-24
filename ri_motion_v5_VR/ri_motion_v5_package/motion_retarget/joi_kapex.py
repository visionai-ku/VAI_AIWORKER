import numpy as np

# """
# %run ../../package/init_kinematics.py
# """
# from transforms import ( # type: ignore
#     pr2t,
#     t2p,
#     t2r,
# )
from ri_motion_v5_package.kinematics.transforms import (
    pr2t,
    t2p,
    t2r,
)

def get_joi_site_name_of_kapex():
    """
    Map our JOI keys -> MuJoCo XML <site name="..."> for KAPEX.

    Parameters:
        None.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    joi_site_name = {
        'base': 'base',
        # Pelvis / legs
        'lp':   'lp',
        'lk':   'lk',
        'la':   'la',
        'rp':   'rp',
        'rk':   'rk',
        'ra':   'ra',

        # Spine / arms
        'spine':'spine',
        'ls':   'ls',
        'le':   'le',
        'lw':   'lw',
        'rs':   'rs',
        're':   're',
        'rw':   'rw',
    }
    return joi_site_name

def get_T_joi_of_kapex(
        env,
        neck_between_shoulder = True,
        base_between_pelvis   = True,
    ):
    """
    Get JOI transforms from KAPEX MuJoCo environment.

    Parameters:
        env (Any): Input env.
        neck_between_shoulder (bool): Input neck between shoulder. Defaults to
                                      `True`.
        base_between_pelvis (bool): Input base between pelvis. Defaults to `True`.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    joi_site_name = get_joi_site_name_of_kapex()

    T_joi = {}
    for key,name in joi_site_name.items():
        T_joi[key] = env.get_T(name,'site')

    # Neck: midpoint of shoulders, identity orientation
    if neck_between_shoulder:
        p_neck = 0.5*(t2p(T_joi['rs'])+t2p(T_joi['ls']))
        T_joi['neck'] = pr2t(p_neck,np.eye(3))

    # Base: midpoint of pelvis sites, keep current base rotation
    if base_between_pelvis:
        p_base = 0.5*(t2p(T_joi['rp'])+t2p(T_joi['lp']))
        R_base = t2r(T_joi['base'])
        T_joi['base'] = pr2t(p_base,R_base)

    return T_joi