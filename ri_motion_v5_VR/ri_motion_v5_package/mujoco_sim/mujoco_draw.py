"""
mujoco_draw.py
"""
import numpy as np

# """
# Run the following command in a Jupyter notebook to import utility functions:
# %run ../../package/init_utility.py
# """
# from transforms import ( # type: ignore
#     np_uv,
#     get_plane_basis_from_normal,
#     axis_angle2quat,
#     rotate_vector_by_quat,
# )

from ri_motion_v5_package.kinematics.transforms import ( # type: ignore
    np_uv,
    get_plane_basis_from_normal,
    axis_angle2quat,
    rotate_vector_by_quat,
)

def draw_rotation_plane(
        env,
        center,
        axis,
        radius   = 0.3,
        n_circle = 32,
        n_spoke  = 12,
        n_rings  = 3,
        r_line   = 0.004,
        rgba     = (0.7,0.7,0.9,0.4),
        label    = None,
    ):
    """
    Draw a wheel-like rotation plane: outer circle + spokes + concentric rings.

    Parameters:
        env (Any): Input env.
        center (Any): Input center.
        axis (Any): Input axis.
        radius (float): Input radius. Defaults to `0.3`.
        n_circle (int): Input n circle. Defaults to `32`.
        n_spoke (int): Input n spoke. Defaults to `12`.
        n_rings (int): Input n rings. Defaults to `3`.
        r_line (float): Input r line. Defaults to `0.004`.
        rgba (tuple): Input rgba. Defaults to `(0.7, 0.7, 0.9, 0.4)`.
        label (Any | None): Input label. Defaults to `None`.

    Returns:
        None: This function mutates state, renders output, or performs side
              effects in place.
    """
    axis = np_uv(axis)
    v1,v2 = get_plane_basis_from_normal(axis)

    # Outer circle
    angles = np.linspace(0,2*np.pi,n_circle)
    circle_pts = [center + radius*(np.cos(a)*v1 + np.sin(a)*v2) for a in angles]
    for i in range(len(circle_pts)-1):
        env.plot_cylinder_fr2to(circle_pts[i],circle_pts[i+1],r=r_line, rgba=rgba)
    env.plot_cylinder_fr2to(circle_pts[-1], circle_pts[0], r=r_line, rgba=rgba)

    # Spokes
    spoke_angles = np.linspace(0, 2 * np.pi, n_spoke, endpoint=False)
    for a in spoke_angles:
        end = center + radius*(np.cos(a)*v1 + np.sin(a)*v2)
        env.plot_cylinder_fr2to(center, end, r=r_line, rgba=rgba)

    # Inner rings
    for i in range(1, n_rings):
        r_ring = radius * i / n_rings
        ring_pts = [center + r_ring * (np.cos(a) * v1 + np.sin(a) * v2) for a in angles]
        for j in range(len(ring_pts) - 1):
            env.plot_cylinder_fr2to(ring_pts[j], ring_pts[j + 1], r=r_line, rgba=rgba)
        env.plot_cylinder_fr2to(ring_pts[-1], ring_pts[0], r=r_line, rgba=rgba)

    # Label
    if label is not None:
        env.plot_text(center, label)


def draw_rotation_arc(
        env,
        center,
        axis,
        vec_fr,
        angle,
        n_arc       = 32,
        n_spoke     = 10,
        rgba_fr     = (1.0, 0.5, 0.0, 0.8),  # start color
        rgba_to     = (0.2, 0.5, 1.0, 0.8),  # end color
        r_arc       = 0.005,
        r_spoke     = 0.005,
        alpha_arc   = None,
        alpha_spoke = None,
    ):
    """
    Draw a gradient arc that rotates vec_from around axis by angle (rad), from
    center.

    Parameters:
        env (Any): Input env.
        center (Any): Input center.
        axis (Any): Input axis.
        vec_fr (Any): Input vec fr.
        angle (Any): Input angle.
        n_arc (int): Input n arc. Defaults to `32`.
        n_spoke (int): Input n spoke. Defaults to `10`.
        rgba_fr (tuple): Input rgba fr. Defaults to `(1.0, 0.5, 0.0, 0.8)`.
        rgba_to (tuple): Input rgba to. Defaults to `(0.2, 0.5, 1.0, 0.8)`.
        r_arc (float): Input r arc. Defaults to `0.005`.
        r_spoke (float): Input r spoke. Defaults to `0.005`.
        alpha_arc (Any | None): Input alpha arc. Defaults to `None`.
        alpha_spoke (Any | None): Input alpha spoke. Defaults to `None`.

    Returns:
        None: This function mutates state, renders output, or performs side
              effects in place.
    """
    axis = np_uv(axis)
    vec_fr = np.array(vec_fr)
    arc_points = []
    arc_angles = np.linspace(0,angle,n_arc)

    for theta in arc_angles:
        q = axis_angle2quat(axis, theta)
        rotated = rotate_vector_by_quat(vec_fr,q)
        arc_points.append(center + rotated)

    rgba_fr = np.array(rgba_fr)
    rgba_to = np.array(rgba_to)

    for i in range(len(arc_points) - 1):
        t = i / (len(arc_points) - 1)
        color = (1 - t) * rgba_fr + t * rgba_to
        if alpha_arc is not None: color[3] = alpha_arc
        env.plot_cylinder_fr2to(arc_points[i],arc_points[i+1],r=r_arc,rgba=color)

    # Optionally draw spokes
    if n_spoke > 0:
        spoke_indices = np.linspace(0, len(arc_points) - 1, n_spoke, dtype=int)
        for i in range(n_spoke):
            t = i / (n_spoke - 1)
            color = (1 - t) * rgba_fr + t * rgba_to
            if alpha_spoke is not None: color[3] = alpha_spoke
            idx = spoke_indices[i]
            env.plot_cylinder_fr2to(center, arc_points[idx], r=r_spoke, rgba=color)

def draw_numpy_chain(
        env,
        chain,
        plot_link         = True,
        r_link            = 0.002,
        rgba_link         = (0,0,0,0.5),
        plot_joint        = True,
        plot_joint_axis   = True,
        plot_joint_sphere = True,
        plot_joint_name   = True,
        axis_len_joint    = 0.05,
        axis_width_joint  = 0.005,
        r_joint           = 0.01,
        rgba_joint        = (0.1,0.1,0.1,0.9),
        plot_rev_axis     = True,
        r_axis            = 0.015,
        h_axis            = 0.1,
        root_idx          = None,
    ):
    """
    Draw a NumpyChain in the Mujoco environment.

    Parameters:
        env (Any): Input env.
        chain (Any): Input chain.
        plot_link (bool): Input plot link. Defaults to `True`.
        r_link (float): Input r link. Defaults to `0.002`.
        rgba_link (tuple): Input rgba link. Defaults to `(0, 0, 0, 0.5)`.
        plot_joint (bool): Input plot joint. Defaults to `True`.
        plot_joint_axis (bool): Input plot joint axis. Defaults to `True`.
        plot_joint_sphere (bool): Input plot joint sphere. Defaults to `True`.
        plot_joint_name (bool): Input plot joint name. Defaults to `True`.
        axis_len_joint (float): Input axis len joint. Defaults to `0.05`.
        axis_width_joint (float): Input axis width joint. Defaults to `0.005`.
        r_joint (float): Input r joint. Defaults to `0.01`.
        rgba_joint (tuple): Input rgba joint. Defaults to `(0.1, 0.1, 0.1, 0.9)`.
        plot_rev_axis (bool): Input plot rev axis. Defaults to `True`.
        r_axis (float): Input r axis. Defaults to `0.015`.
        h_axis (float): Input h axis. Defaults to `0.1`.
        root_idx (Any | None): Input root idx. Defaults to `None`.

    Returns:
        None: This function mutates state, renders output, or performs side
              effects in place.
    """
    import numpy as np
    from networkx.algorithms.traversal.depth_first_search import dfs_edges

    if root_idx is None:
        root_idx = chain.get_root_idx()

    # Plot link
    if plot_link:
        for _,edge in enumerate(dfs_edges(chain.graph,source=root_idx)):
            joint_fr = chain.get_joint(joint_idx=edge[0])
            joint_to = chain.get_joint(joint_idx=edge[1])
            env.plot_cylinder_fr2to(
                p_fr = joint_fr['p'],
                p_to = joint_to['p'],
                r    = r_link,
                rgba = rgba_link,
            )

    # Plot joint
    if plot_joint:
        for j_idx in range(chain.get_n_joint()):
            joint = chain.get_joint(joint_idx=j_idx)
            joint_name = joint['name'] if plot_joint_name else ''
            env.plot_T(
                p           = joint['p'],
                R           = joint['R'],
                plot_axis   = plot_joint_axis,
                axis_len    = axis_len_joint,
                axis_width  = axis_width_joint,
                plot_sphere = plot_joint_sphere,
                sphere_r    = r_joint,
                sphere_rgba = rgba_joint,
                label       = joint_name,
            )

    # Plot revolute axis
    if plot_rev_axis:
        for j_idx in range(chain.get_n_joint()):
            joint = chain.get_joint(joint_idx=j_idx)
            a = joint['a']
            if np.linalg.norm(a) > 1e-6:
                p,R = joint['p'],joint['R']
                p2  = p + R@a*h_axis
                axis_color = [0,0,0,0.5]
                axis_color[int(np.argmax(a))] = 1
                env.plot_arrow_fr2to(p_fr=p,p_to=p2,r=r_axis,rgba=axis_color)
