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

def get_T_joi_of_chain_seg(
        chain,
        base_between_pelvis=True,
    ):
    """
    Get joints of interest of SeG mocap chain (NumpyChain version).

    Parameters:
        chain (Any): Input chain.
        base_between_pelvis (bool): Input base between pelvis. Defaults to `True`.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    p_base,R_base     = get_chain_pose(chain,'Hips')
    p_spine,R_spine   = get_chain_pose(chain,'Chest2')
    p_rs,R_rs         = get_chain_pose(chain,'RightShoulder')
    p_re,R_re         = get_chain_pose(chain,'RightElbow')
    p_rw,R_rw         = get_chain_pose(chain,'RightWrist')
    p_rh,R_rh         = get_chain_pose(chain,'RightWrist')
    p_ls,R_ls         = get_chain_pose(chain,'LeftShoulder')
    p_le,R_le         = get_chain_pose(chain,'LeftElbow')
    p_lw,R_lw         = get_chain_pose(chain,'LeftWrist')
    p_lh,R_lh         = get_chain_pose(chain,'LeftWrist')

    # -------------------------
    # Right hand
    # -------------------------
    # Thumb: 0,1,2,t -> cmc,mcp,ip,tip
    p_rthumb_cmc,R_rthumb_cmc = get_chain_pose(chain,'RightFinger0')
    p_rthumb_mcp,R_rthumb_mcp = get_chain_pose(chain,'RightFinger01')
    p_rthumb_ip, R_rthumb_ip  = get_chain_pose(chain,'RightFinger02')
    p_rthumb_tip,R_rthumb_tip = get_chain_pose(chain,'RightFinger02End')

    # Index: 0,1,2,t -> mcp,pip,dip,tip
    p_rindex_mcp,R_rindex_mcp = get_chain_pose(chain,'RightFinger1')
    p_rindex_pip,R_rindex_pip = get_chain_pose(chain,'RightFinger11')
    p_rindex_dip,R_rindex_dip = get_chain_pose(chain,'RightFinger12')
    p_rindex_tip,R_rindex_tip = get_chain_pose(chain,'RightFinger12End')

    # Middle
    p_rmiddle_mcp,R_rmiddle_mcp = get_chain_pose(chain,'RightFinger2')
    p_rmiddle_pip,R_rmiddle_pip = get_chain_pose(chain,'RightFinger21')
    p_rmiddle_dip,R_rmiddle_dip = get_chain_pose(chain,'RightFinger22')
    p_rmiddle_tip,R_rmiddle_tip = get_chain_pose(chain,'RightFinger22End')

    # Ring
    p_rring_mcp,R_rring_mcp = get_chain_pose(chain,'RightFinger3')
    p_rring_pip,R_rring_pip = get_chain_pose(chain,'RightFinger31')
    p_rring_dip,R_rring_dip = get_chain_pose(chain,'RightFinger32')
    p_rring_tip,R_rring_tip = get_chain_pose(chain,'RightFinger32End')

    # Little (Pinky)
    p_rlittle_mcp,R_rlittle_mcp = get_chain_pose(chain,'RightFinger4')
    p_rlittle_pip,R_rlittle_pip = get_chain_pose(chain,'RightFinger41')
    p_rlittle_dip,R_rlittle_dip = get_chain_pose(chain,'RightFinger42')
    p_rlittle_tip,R_rlittle_tip = get_chain_pose(chain,'RightFinger42End')

    # -------------------------
    # Left hand
    # -------------------------
    # Thumb: 0,1,2,t -> cmc,mcp,ip,tip
    p_lthumb_cmc,R_lthumb_cmc = get_chain_pose(chain,'LeftFinger0')
    p_lthumb_mcp,R_lthumb_mcp = get_chain_pose(chain,'LeftFinger01')
    p_lthumb_ip, R_lthumb_ip  = get_chain_pose(chain,'LeftFinger02')
    p_lthumb_tip,R_lthumb_tip = get_chain_pose(chain,'LeftFinger02End')

    # Index
    p_lindex_mcp,R_lindex_mcp = get_chain_pose(chain,'LeftFinger1')
    p_lindex_pip,R_lindex_pip = get_chain_pose(chain,'LeftFinger11')
    p_lindex_dip,R_lindex_dip = get_chain_pose(chain,'LeftFinger12')
    p_lindex_tip,R_lindex_tip = get_chain_pose(chain,'LeftFinger12End')

    # Middle
    p_lmiddle_mcp,R_lmiddle_mcp = get_chain_pose(chain,'LeftFinger2')
    p_lmiddle_pip,R_lmiddle_pip = get_chain_pose(chain,'LeftFinger21')
    p_lmiddle_dip,R_lmiddle_dip = get_chain_pose(chain,'LeftFinger22')
    p_lmiddle_tip,R_lmiddle_tip = get_chain_pose(chain,'LeftFinger22End')

    # Ring
    p_lring_mcp,R_lring_mcp = get_chain_pose(chain,'LeftFinger3')
    p_lring_pip,R_lring_pip = get_chain_pose(chain,'LeftFinger31')
    p_lring_dip,R_lring_dip = get_chain_pose(chain,'LeftFinger32')
    p_lring_tip,R_lring_tip = get_chain_pose(chain,'LeftFinger32End')

    # Little
    p_llittle_mcp,R_llittle_mcp = get_chain_pose(chain,'LeftFinger4')
    p_llittle_pip,R_llittle_pip = get_chain_pose(chain,'LeftFinger41')
    p_llittle_dip,R_llittle_dip = get_chain_pose(chain,'LeftFinger42')
    p_llittle_tip,R_llittle_tip = get_chain_pose(chain,'LeftFinger42End')

    # -------------------------
    # Head / torso / legs
    # -------------------------
    p_neck,R_neck     = get_chain_pose(chain,'Neck')
    p_neck_z          = 0.5*(p_rs+p_ls)
    p_neck            = np.array([p_neck[0],p_neck[1],p_neck_z[2]])

    p_head,R_head     = get_chain_pose(chain,'Head')

    p_rp,R_rp         = get_chain_pose(chain,'RightHip')
    p_rk,R_rk         = get_chain_pose(chain,'RightKnee')
    p_ra,R_ra         = get_chain_pose(chain,'RightAnkle')
    p_rf,R_rf         = get_chain_pose(chain,'RightAnkle')

    p_lp,R_lp         = get_chain_pose(chain,'LeftHip')
    p_lk,R_lk         = get_chain_pose(chain,'LeftKnee')
    p_la,R_la         = get_chain_pose(chain,'LeftAnkle')
    p_lf,R_lf         = get_chain_pose(chain,'LeftAnkle')

    p_r_toe,R_r_toe   = get_chain_pose(chain,'RightToe')
    p_l_toe,R_l_toe   = get_chain_pose(chain,'LeftToe')

    if base_between_pelvis:
        p_base = 0.5*(p_rp+p_lp)

    T_joi = {
        'base': pr2t(p_base,R_base),
        'spine': pr2t(p_spine,R_spine),
        'rs': pr2t(p_rs,R_rs),
        're': pr2t(p_re,R_re),
        'rw': pr2t(p_rw,R_rw),
        # 'rh': pr2t(p_rh,R_rh),
        'ls': pr2t(p_ls,R_ls),
        'le': pr2t(p_le,R_le),
        'lw': pr2t(p_lw,R_lw),
        # 'lh': pr2t(p_lh,R_lh),
        'neck': pr2t(p_neck,R_neck),
        'head': pr2t(p_head,R_head),
        'rp': pr2t(p_rp,R_rp),
        'rk': pr2t(p_rk,R_rk),
        'ra': pr2t(p_ra,R_ra),
        'rf': pr2t(p_rf,R_rf),
        'rtoe': pr2t(p_r_toe,R_r_toe),
        'lp': pr2t(p_lp,R_lp),
        'lk': pr2t(p_lk,R_lk),
        'la': pr2t(p_la,R_la),
        'lf': pr2t(p_lf,R_lf),
        'ltoe': pr2t(p_l_toe,R_l_toe),

        # -------------------------
        # Right hand
        # -------------------------
        'rthumb_cmc': pr2t(p_rthumb_cmc,R_rthumb_cmc),
        'rthumb_mcp': pr2t(p_rthumb_mcp,R_rthumb_mcp),
        'rthumb_ip' : pr2t(p_rthumb_ip ,R_rthumb_ip ),
        'rthumb_tip': pr2t(p_rthumb_tip,R_rthumb_tip),

        'rindex_mcp': pr2t(p_rindex_mcp,R_rindex_mcp),
        'rindex_pip': pr2t(p_rindex_pip,R_rindex_pip),
        'rindex_dip': pr2t(p_rindex_dip,R_rindex_dip),
        'rindex_tip': pr2t(p_rindex_tip,R_rindex_tip),

        'rmiddle_mcp': pr2t(p_rmiddle_mcp,R_rmiddle_mcp),
        'rmiddle_pip': pr2t(p_rmiddle_pip,R_rmiddle_pip),
        'rmiddle_dip': pr2t(p_rmiddle_dip,R_rmiddle_dip),
        'rmiddle_tip': pr2t(p_rmiddle_tip,R_rmiddle_tip),

        'rring_mcp': pr2t(p_rring_mcp,R_rring_mcp),
        'rring_pip': pr2t(p_rring_pip,R_rring_pip),
        'rring_dip': pr2t(p_rring_dip,R_rring_dip),
        'rring_tip': pr2t(p_rring_tip,R_rring_tip),

        'rlittle_mcp': pr2t(p_rlittle_mcp,R_rlittle_mcp),
        'rlittle_pip': pr2t(p_rlittle_pip,R_rlittle_pip),
        'rlittle_dip': pr2t(p_rlittle_dip,R_rlittle_dip),
        'rlittle_tip': pr2t(p_rlittle_tip,R_rlittle_tip),

        # -------------------------
        # Left hand
        # -------------------------
        'lthumb_cmc': pr2t(p_lthumb_cmc,R_lthumb_cmc),
        'lthumb_mcp': pr2t(p_lthumb_mcp,R_lthumb_mcp),
        'lthumb_ip' : pr2t(p_lthumb_ip ,R_lthumb_ip ),
        'lthumb_tip': pr2t(p_lthumb_tip,R_lthumb_tip),

        'lindex_mcp': pr2t(p_lindex_mcp,R_lindex_mcp),
        'lindex_pip': pr2t(p_lindex_pip,R_lindex_pip),
        'lindex_dip': pr2t(p_lindex_dip,R_lindex_dip),
        'lindex_tip': pr2t(p_lindex_tip,R_lindex_tip),

        'lmiddle_mcp': pr2t(p_lmiddle_mcp,R_lmiddle_mcp),
        'lmiddle_pip': pr2t(p_lmiddle_pip,R_lmiddle_pip),
        'lmiddle_dip': pr2t(p_lmiddle_dip,R_lmiddle_dip),
        'lmiddle_tip': pr2t(p_lmiddle_tip,R_lmiddle_tip),

        'lring_mcp': pr2t(p_lring_mcp,R_lring_mcp),
        'lring_pip': pr2t(p_lring_pip,R_lring_pip),
        'lring_dip': pr2t(p_lring_dip,R_lring_dip),
        'lring_tip': pr2t(p_lring_tip,R_lring_tip),

        'llittle_mcp': pr2t(p_llittle_mcp,R_llittle_mcp),
        'llittle_pip': pr2t(p_llittle_pip,R_llittle_pip),
        'llittle_dip': pr2t(p_llittle_dip,R_llittle_dip),
        'llittle_tip': pr2t(p_llittle_tip,R_llittle_tip),
    }
    return T_joi
