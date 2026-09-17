"""
mujoco_utils.py
"""
import os
import mujoco
import numpy as np
import xml.etree.ElementTree as ET

# Import utility functions
# """
# Run the following command in a Jupyter notebook to import utility functions:
# %run ../../package/init_utility.py
# """
# from utils import (
#     indent_xml,
# )
from ri_motion_v5_package.utility.utils import (
    indent_xml
)
from ri_motion_v5_package.kinematics.transforms import (
    np_uv,
    pr2t,
)

def normalize_mujoco_joint_types(joint_types):
    """
    Normalize user joint-type input into MuJoCo joint type integers.

    Parameters:
        joint_types (Any): Input joint types.

    Returns:
        Any: Result computed from the current inputs.
    """
    if joint_types is None:
        return None

    if not isinstance(joint_types,(list,tuple,set,np.ndarray)):
        joint_types = [joint_types]

    joint_type_set = set()

    for jt in joint_types:
        if isinstance(jt,str):
            jt_lower = jt.lower()

            if jt_lower in ["rev","revolute","hinge"]:
                joint_type_set.add(int(mujoco.mjtJoint.mjJNT_HINGE))
            elif jt_lower in ["pri","prismatic","slide"]:
                joint_type_set.add(int(mujoco.mjtJoint.mjJNT_SLIDE))
            elif jt_lower in ["ball"]:
                joint_type_set.add(int(mujoco.mjtJoint.mjJNT_BALL))
            elif jt_lower in ["free"]:
                joint_type_set.add(int(mujoco.mjtJoint.mjJNT_FREE))
            else:
                raise ValueError("Unsupported joint type string:[%s]"%(jt))
        else:
            joint_type_set.add(int(jt))

    return joint_type_set

def merge_mjcfs(
        mjcf_model_name     = 'scene',
        included_mjcf_files = ['a.xml', 'b.xml'],
        timestep            = 0.002,
        integrator          = 'implicitfast',
        noslip              = None,  # e.g., 1, 5, 10
        memory              = None,  # "500M", "1G", etc.
        armature            = None,  # e.g., 1.0
        damping             = None,  # e.g., 2.0
        cone                = None,  # e.g., 'elliptic'
        exclude_body_list   = None,   # NEW ARG: [['palm','ff_base'], ...]
        output_xml_path     = 'merged_scene.xml',
        verbose             = True,
    ):
    """
    Merges multiple MJCF files into a single MuJoCo model XML, adds optional
    simulation settings,.

    Parameters:
        mjcf_model_name (str): Input mjcf model name. Defaults to `'scene'`.
        included_mjcf_files (list): Input included mjcf files. Defaults to
                                    `['a.xml', 'b.xml']`.
        timestep (float): Input timestep. Defaults to `0.002`.
        integrator (str): Input integrator. Defaults to `'implicitfast'`.
        noslip (Any | None): Input noslip. Defaults to `None`.
        memory (Any | None): Input memory. Defaults to `None`.
        armature (Any | None): Input armature. Defaults to `None`.
        damping (Any | None): Input damping. Defaults to `None`.
        cone (Any | None): Input cone. Defaults to `None`.
        exclude_body_list (Any | None): Input exclude body list. Defaults to
                                        `None`.
        output_xml_path (str): Input output xml path. Defaults to
                               `'merged_scene.xml'`.
        verbose (bool): Input verbose. Defaults to `True`.

    Returns:
        Any: Result computed from the current inputs.
    """
    # Create root element
    mujoco_elem = ET.Element("mujoco", attrib={"model": mjcf_model_name})

    output_dir = os.path.dirname(os.path.abspath(output_xml_path))

    # Add include elements
    for filepath in included_mjcf_files:
        abs_path = os.path.abspath(filepath)
        rel_path = os.path.relpath(abs_path, output_dir)
        ET.SubElement(mujoco_elem, "include", attrib={"file": rel_path})

    # Add <default> tag if needed
    if armature is not None or damping is not None:
        default_elem = ET.SubElement(mujoco_elem, "default")
        joint_attrs = {}
        if armature is not None:
            joint_attrs["armature"] = str(armature)
        if damping is not None:
            joint_attrs["damping"] = str(damping)
        ET.SubElement(default_elem, "joint", attrib=joint_attrs)

    # Add <option> tag with extra attributes if given
    option_attrs = {
        "timestep": str(timestep),
        "integrator": integrator,
    }
    if noslip is not None:
        option_attrs["noslip_iterations"] = str(int(noslip))
    if cone is not None:
        option_attrs["cone"] = cone
    ET.SubElement(mujoco_elem, "option", attrib=option_attrs)

    # Add <size> tag if memory is given
    if memory is not None:
        mem_value = memory if isinstance(memory, str) else "1G"
        ET.SubElement(mujoco_elem, "size", attrib={"memory": mem_value})

    # Add <contact> tag with <exclude> children if exclude_body_list is given
    if exclude_body_list is not None and len(exclude_body_list) > 0:
        contact_elem = ET.SubElement(mujoco_elem, "contact")
        for pair in exclude_body_list:
            if len(pair) != 2:
                raise ValueError(f"Each exclude pair must have length 2, got: {pair}")
            body1, body2 = pair
            ET.SubElement(contact_elem, "exclude", attrib={
                "body1": str(body1),
                "body2": str(body2),
            })

    # Convert to pretty string
    pretty_xml = indent_xml(mujoco_elem)

    # Ensure the directory exists
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write to file
    with open(output_xml_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

    # Verbose output
    if verbose:
        print("[merge_mjcfs] Merging [%d] MJCF files:"%(len(included_mjcf_files)))
        for f_idx,f in enumerate(included_mjcf_files):
            print("  - [%d] [%s]"%(f_idx,f))
        print(f"[merge_mjcfs] Saved merged XML to:[%s]"%(output_xml_path))

    return output_xml_path

def get_label_colors_from_joint_axis(env,joints_names=None,bright_colors=True):
    """
    Get label colors based on the primary axis of the specified joints.

    Parameters:
        env (Any): Input env.
        joints_names (Any | None): Input joints names. Defaults to `None`.
        bright_colors (bool): Input bright colors. Defaults to `True`.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    # Default to all revolute joints if none specified
    if joints_names is None:
        joints_names = env.rev_joint_names

    if bright_colors:
        rgb_list = ['#FF6666', '#66FF66', '#66CCFF']
    else:
        rgb_list = ['#AA0000', '#00AA00', '#0066AA']

    # Determine label colors based on joint axes
    label_colors = []
    for joint_name in joints_names:
        joint_idx   = env.joint_names.index(joint_name)
        axis_joint  = env.model.jnt_axis[joint_idx]
        if np.allclose(axis_joint, 0):
            label_color = '#888888'  # Default gray color for zero axis
        else:
            label_color = rgb_list[int(np.argmax(np.abs(axis_joint)))]
        label_colors.append(label_color)

    # Return the list of label colors
    return label_colors

def get_hand_coord(
        p_palm,
        p_top    = None,
        p_palmar = None,
        p_front  = None,
        hand_type = "right",
    ):
    """
    Build a hand coordinate frame from palm and at least two direction points.

    Parameters:
        p_palm (np.ndarray): Input p palm.
        p_top (Any | None): Input p top. Defaults to `None`.
        p_palmar (Any | None): Input p palmar. Defaults to `None`.
        p_front (Any | None): Input p front. Defaults to `None`.
        hand_type (str): Input hand type. Defaults to `'right'`.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    hand_type = str(hand_type).lower()
    if hand_type in ["right","r","rh"]:
        is_right = True
    elif hand_type in ["left","l","lh"]:
        is_right = False
    else:
        raise ValueError("Unsupported hand_type:[%s]"%(hand_type))

    p_palm = np.asarray(p_palm,dtype=np.float64).reshape(3)

    ux_hint,uy_hint,uz_hint = None,None,None

    if p_front is not None:
        p_front = np.asarray(p_front,dtype=np.float64).reshape(3)
        ux_hint = np_uv(p_front-p_palm)          # +x: front

    if p_palmar is not None:
        p_palmar = np.asarray(p_palmar,dtype=np.float64).reshape(3)
        if is_right:
            uy_hint = np_uv(p_palmar-p_palm)     # +y: palm -> palmar
        else:
            uy_hint = np_uv(p_palm-p_palmar)     # +y: reversed for left hand

    if p_top is not None:
        p_top = np.asarray(p_top,dtype=np.float64).reshape(3)
        uz_hint = np_uv(p_top-p_palm)            # +z: top

    n_hint = sum(axis is not None for axis in [ux_hint,uy_hint,uz_hint])
    if n_hint < 2:
        raise ValueError("At least two of p_top, p_palmar, and p_front are required.")

    eps = 1e-8
    if ux_hint is not None and uy_hint is not None:
        ux = ux_hint
        uz = np.cross(ux,uy_hint)
        if np.linalg.norm(uz) < eps:
            raise ValueError("p_front and p_palmar define parallel hand directions.")
        uz = np_uv(uz)
        if uz_hint is not None and np.dot(uz,uz_hint) < 0.0:
            uz = -uz
        uy = np_uv(np.cross(uz,ux))
    elif ux_hint is not None and uz_hint is not None:
        ux = ux_hint
        uy = np.cross(uz_hint,ux)
        if np.linalg.norm(uy) < eps:
            raise ValueError("p_front and p_top define parallel hand directions.")
        uy = np_uv(uy)
        if uy_hint is not None and np.dot(uy,uy_hint) < 0.0:
            uy = -uy
        uz = np_uv(np.cross(ux,uy))
        if np.dot(uz,uz_hint) < 0.0:
            uy = -uy
            uz = -uz
    else:
        uy = uy_hint
        ux = np.cross(uy,uz_hint)
        if np.linalg.norm(ux) < eps:
            raise ValueError("p_palmar and p_top define parallel hand directions.")
        ux = np_uv(ux)
        if ux_hint is not None and np.dot(ux,ux_hint) < 0.0:
            ux = -ux
        uz = np_uv(np.cross(ux,uy))
        if np.dot(uz,uz_hint) < 0.0:
            ux = -ux
            uz = -uz
        uy = np_uv(np.cross(uz,ux))

    R_palm = np.column_stack((ux,uy,uz)).reshape(3,3)
    T_palm = pr2t(p=p_palm,R=R_palm)
    return T_palm

def get_head_coord(
        p_head,
        p_top,
        p_front,
    ):
    """
    Build a head coordinate frame from head/top/front points.

    Parameters:
        p_head (np.ndarray): Input p head.
        p_top (np.ndarray): Input p top.
        p_front (np.ndarray): Input p front.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    p_head  = np.asarray(p_head,dtype=np.float64).reshape(3)
    p_top   = np.asarray(p_top,dtype=np.float64).reshape(3)
    p_front = np.asarray(p_front,dtype=np.float64).reshape(3)

    ux = np_uv(p_front-p_head)      # +x: front
    uz_hint = np_uv(p_top-p_head)   # +z: top

    uy = np.cross(uz_hint,ux)
    if np.linalg.norm(uy) < 1e-8:
        raise ValueError("p_front and p_top define parallel head directions.")
    uy = np_uv(uy)

    uz = np_uv(np.cross(ux,uy))
    if np.dot(uz,uz_hint) < 0.0:
        uy = -uy
        uz = -uz

    R_head = np.column_stack((ux,uy,uz)).reshape(3,3)
    T_head = pr2t(p=p_head,R=R_head)
    return T_head

def get_foot_coord(
        p_foot,
        p_top,
        p_front,
        foot_type = "right",
    ):
    """
    Build a foot coordinate frame from foot/top/front points.

    Parameters:
        p_foot (np.ndarray): Input p foot.
        p_top (np.ndarray): Input p top.
        p_front (np.ndarray): Input p front.
        foot_type (str): Input foot type. Defaults to `'right'`.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    foot_type = str(foot_type).lower()
    if foot_type not in ["right","r","rf","left","l","lf"]:
        raise ValueError("Unsupported foot_type:[%s]"%(foot_type))

    p_foot  = np.asarray(p_foot,dtype=np.float64).reshape(3)
    p_top   = np.asarray(p_top,dtype=np.float64).reshape(3)
    p_front = np.asarray(p_front,dtype=np.float64).reshape(3)

    ux = np_uv(p_front-p_foot)      # +x: front
    uz_hint = np_uv(p_top-p_foot)   # +z: top

    # With only front/top points, +x and +z fully determine the right-handed
    # lateral axis. foot_type is validated above to keep lf/rf calls explicit.
    uy = np.cross(uz_hint,ux)
    if np.linalg.norm(uy) < 1e-8:
        raise ValueError("p_front and p_top define parallel foot directions.")
    uy = np_uv(uy)

    uz = np_uv(np.cross(ux,uy))
    if np.dot(uz,uz_hint) < 0.0:
        uy = -uy
        uz = -uz

    R_foot = np.column_stack((ux,uy,uz)).reshape(3,3)
    T_foot = pr2t(p=p_foot,R=R_foot)
    return T_foot

def get_T_hand_from_sites(
        env,
        site_name_common = None,
        hand_type        = "right",
        site_type        = "site",
        postfix_dict     = None,
    ):
    """
    Build hand coordinate frame from four sites:.

    Parameters:
        env (Any): Input env.
        site_name_common (Any | None): Input site name common. Defaults to `None`.
        hand_type (str): Input hand type. Defaults to `'right'`.
        site_type (str): Input site type. Defaults to `'site'`.
        postfix_dict (Any | None): Input postfix dict. Defaults to `None`.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    if site_name_common is None:
        raise ValueError("site_name_common is required.")

    if postfix_dict is None:
        postfix_dict = {
            "palm"   :"",
            "top"    :"_top",
            "palmar" :"_palmar",
            "front"  :"_front",
        }

    # Resolve site names
    site_name_palm   = site_name_common + postfix_dict["palm"]
    site_name_top    = site_name_common + postfix_dict["top"]
    site_name_palmar = site_name_common + postfix_dict["palmar"]
    site_name_front  = site_name_common + postfix_dict["front"]

    # Get points
    p_palm   = env.get_p(site_name_palm,site_type)
    p_top    = env.get_p(site_name_top,site_type)
    p_palmar = env.get_p(site_name_palmar,site_type)
    p_front  = env.get_p(site_name_front,site_type)

    return get_hand_coord(
        p_palm   = p_palm,
        p_top    = p_top,
        p_palmar = p_palmar,
        p_front  = p_front,
        hand_type = hand_type,
    )
