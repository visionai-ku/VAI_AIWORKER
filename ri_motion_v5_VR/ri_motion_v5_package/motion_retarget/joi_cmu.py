import numpy as np

# """
# %run ../../package/init_kinematics.py
# """
# from transforms import ( # type: ignore
#     pr2t,
# )
from ri_motion_v5_package.kinematics.transforms import (
    pr2t,
)

def get_chain_pose(chain,joint_name):
    """
    Get a joint pose from a NumpyChain by name.

    Parameters:
        chain (Any): Input chain.
        joint_name (str): Input joint name.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    return chain.get_pose(joint_name=joint_name)

def get_T_joi_of_chain_cmu(
        chain,
        base_between_pelvis=True,
    ):
    """
    Get joints of interest of CMU mocap chain (NumpyChain version).

    Parameters:
        chain (Any): Input chain.
        base_between_pelvis (bool): Input base between pelvis. Defaults to `True`.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    # -------------------------
    # Core body (CMU naming)
    # -------------------------
    p_base,R_base     = get_chain_pose(chain,'Hips')
    p_spine,R_spine   = get_chain_pose(chain,'Spine')

    p_rs,R_rs         = get_chain_pose(chain,'RightArm')
    p_re,R_re         = get_chain_pose(chain,'RightForeArm')
    p_rw,R_rw         = get_chain_pose(chain,'RightHand')
    p_rh,R_rh         = get_chain_pose(chain,'RightHand')

    p_ls,R_ls         = get_chain_pose(chain,'LeftArm')
    p_le,R_le         = get_chain_pose(chain,'LeftForeArm')
    p_lw,R_lw         = get_chain_pose(chain,'LeftHand')
    p_lh,R_lh         = get_chain_pose(chain,'LeftHand')

    # -------------------------
    # Head / torso
    # -------------------------
    p_neck,R_neck     = get_chain_pose(chain,'Neck')
    p_neck_z          = 0.5*(p_rs+p_ls)
    p_neck            = np.array([p_neck[0],p_neck[1],p_neck_z[2]])

    p_head,R_head     = get_chain_pose(chain,'Head')

    # -------------------------
    # Legs / feet
    # -------------------------
    p_rp,R_rp         = get_chain_pose(chain,'RightUpLeg')
    p_rk,R_rk         = get_chain_pose(chain,'RightLeg')
    p_ra,R_ra         = get_chain_pose(chain,'RightFoot')
    p_rf,R_rf         = get_chain_pose(chain,'RightFoot')

    p_lp,R_lp         = get_chain_pose(chain,'LeftUpLeg')
    p_lk,R_lk         = get_chain_pose(chain,'LeftLeg')
    p_la,R_la         = get_chain_pose(chain,'LeftFoot')
    p_lf,R_lf         = get_chain_pose(chain,'LeftFoot')

    p_r_toe,R_r_toe   = get_chain_pose(chain,'RightToeBase')
    p_l_toe,R_l_toe   = get_chain_pose(chain,'LeftToeBase')

    # -------------------------
    # Base correction
    # -------------------------
    if base_between_pelvis:
        p_base = 0.5*(p_rp+p_lp)

    # -------------------------
    # Return transforms
    # -------------------------
    T_joi = {
        'base' : pr2t(p_base,R_base),
        'spine': pr2t(p_spine,R_spine),

        'rs'   : pr2t(p_rs,R_rs),
        're'   : pr2t(p_re,R_re),
        'rw'   : pr2t(p_rw,R_rw),
        # 'rh'   : pr2t(p_rh,R_rh),

        'ls'   : pr2t(p_ls,R_ls),
        'le'   : pr2t(p_le,R_le),
        'lw'   : pr2t(p_lw,R_lw),
        # 'lh'   : pr2t(p_lh,R_lh),

        'neck' : pr2t(p_neck,R_neck),
        'head' : pr2t(p_head,R_head),

        'rp'   : pr2t(p_rp,R_rp),
        'rk'   : pr2t(p_rk,R_rk),
        'ra'   : pr2t(p_ra,R_ra),
        'rf'   : pr2t(p_rf,R_rf),
        'rtoe' : pr2t(p_r_toe,R_r_toe),

        'lp'   : pr2t(p_lp,R_lp),
        'lk'   : pr2t(p_lk,R_lk),
        'la'   : pr2t(p_la,R_la),
        'lf'   : pr2t(p_lf,R_lf),
        'ltoe' : pr2t(p_l_toe,R_l_toe),
    }
    return T_joi
