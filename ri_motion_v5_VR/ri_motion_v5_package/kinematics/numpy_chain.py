from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_edges

# """
# %run ../../package/init_kinematics.py
# """
# from transforms import ( # type: ignore
#     pr2t,
#     r2rpy,
#     rodrigues,
# )
from ri_motion_v5_package.kinematics.transforms import (
    pr2t,
    r2rpy,
    rodrigues,
)

class NumpyChain(object):
    """
    Numpy Chain class (kinematic chain implemented with numpy and networkx).

    Usage:
        obj = NumpyChain(...)
    """

    def __init__(self,name='Numpy Chain'):
        """
        Initialize NumpyChain.

        Parameters:
            name (str): Input name. Defaults to `'Numpy Chain'`.

        Returns:
            None: Initializes the instance in place.
        """
        self.name            = name
        self.graph           = None
        self.joint_names     = []
        self.rev_joint_names = []
        self.init_chain()

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------
    def _resolve_idx(self,joint_idx=None,joint_name=None):
        """
        Resolve joint index from joint_idx or joint_name.

        Parameters:
            joint_idx (Any | None): Input joint idx. Defaults to `None`.
            joint_name (Any | None): Input joint name. Defaults to `None`.

        Returns:
            Any: Result computed from the current inputs.
        """
        if joint_idx is not None:
            return int(joint_idx)
        if joint_name is not None:
            return self.idx(joint_name)
        raise ValueError("Provide joint_idx or joint_name.")

    # ---------------------------------------------------------------------
    # Graph-level getters
    # ---------------------------------------------------------------------
    def init_chain(self):
        """
        Initialize the kinematic chain graph.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        if self.graph is not None:
            self.graph.clear()
        self.graph = nx.DiGraph(name=self.name)

    def get_n_joint(self):
        """
        Get the number of joints in the kinematic chain.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        return self.graph.number_of_nodes()

    def get_n_rev_joint(self):
        """
        Get the number of revolute joints in the kinematic chain.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        return len(self.rev_joint_names)

    def idx(self,joint_name):
        """
        Get the index of a joint by its name.

        Parameters:
            joint_name (str): Input joint name.

        Returns:
            Any: Result computed from the current inputs.
        """
        return self.joint_names.index(joint_name)

    def idxs(self,joint_names):
        """
        Get the indices of multiple joints by their names.

        Parameters:
            joint_names (list[str]): Input joint names.

        Returns:
            Any: Result computed from the current inputs.
        """
        return [self.idx(nm) for nm in joint_names]

    def get_root_idx(self):
        """
        Get the index of the root joint.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        for node in self.graph.nodes:
            if self.graph.in_degree(node)==0:
                return int(node)
        return 0

    def get_root_name(self):
        """
        Get the name of the root joint.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        return self.joint_names[self.get_root_idx()]

    def get_joint(self,joint_idx=None,joint_name=None):
        """
        Get joint information by index or name.

        Parameters:
            joint_idx (Any | None): Input joint idx. Defaults to `None`.
            joint_name (Any | None): Input joint name. Defaults to `None`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        j = self._resolve_idx(joint_idx=joint_idx,joint_name=joint_name)
        return self.graph.nodes[j]

    # ---------------------------------------------------------------------
    # Joint attribute getters (MuJoCoParser-style)
    # ---------------------------------------------------------------------
    def get_p(self,joint_idx=None,joint_name=None):
        """
        Get the position of a joint by index or name.

        Parameters:
            joint_idx (Any | None): Input joint idx. Defaults to `None`.
            joint_name (Any | None): Input joint name. Defaults to `None`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        return self.get_joint(joint_idx=joint_idx,joint_name=joint_name)['p']

    def get_R(self,joint_idx=None,joint_name=None):
        """
        Get the rotation matrix of a joint by index or name.

        Parameters:
            joint_idx (Any | None): Input joint idx. Defaults to `None`.
            joint_name (Any | None): Input joint name. Defaults to `None`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        return self.get_joint(joint_idx=joint_idx,joint_name=joint_name)['R']

    def get_pose(self,joint_idx=None,joint_name=None):
        """
        Get the position and rotation matrix of a joint by index or name.

        Parameters:
            joint_idx (Any | None): Input joint idx. Defaults to `None`.
            joint_name (Any | None): Input joint name. Defaults to `None`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        j = self.get_joint(joint_idx=joint_idx,joint_name=joint_name)
        return j['p'],j['R']

    def get_T(self,joint_idx=None,joint_name=None):
        """
        Get the homogeneous transformation matrix of a joint by index or name.

        Parameters:
            joint_idx (Any | None): Input joint idx. Defaults to `None`.
            joint_name (Any | None): Input joint name. Defaults to `None`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        p,R = self.get_pose(joint_idx=joint_idx,joint_name=joint_name)
        return pr2t(p,R)

    def get_p_offset(self,joint_idx=None,joint_name=None):
        """
        Get the position offset of a joint by index or name.

        Parameters:
            joint_idx (Any | None): Input joint idx. Defaults to `None`.
            joint_name (Any | None): Input joint name. Defaults to `None`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        return self.get_joint(joint_idx=joint_idx,joint_name=joint_name)['p_offset']

    def get_R_offset(self,joint_idx=None,joint_name=None):
        """
        Get the rotation offset of a joint by index or name.

        Parameters:
            joint_idx (Any | None): Input joint idx. Defaults to `None`.
            joint_name (Any | None): Input joint name. Defaults to `None`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        return self.get_joint(joint_idx=joint_idx,joint_name=joint_name)['R_offset']

    def get_axis(self,joint_idx=None,joint_name=None):
        """
        Get the axis of a joint by index or name.

        Parameters:
            joint_idx (Any | None): Input joint idx. Defaults to `None`.
            joint_name (Any | None): Input joint name. Defaults to `None`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        return self.get_joint(joint_idx=joint_idx,joint_name=joint_name)['a']

    def get_q(self,joint_idx=None,joint_name=None):
        """
        Get the joint variable (q) of a joint by index or name.

        Parameters:
            joint_idx (Any | None): Input joint idx. Defaults to `None`.
            joint_name (Any | None): Input joint name. Defaults to `None`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        return self.get_joint(joint_idx=joint_idx,joint_name=joint_name)['q']

    # ---------------------------------------------------------------------
    # Joint attribute setters (MuJoCoParser-style)
    # ---------------------------------------------------------------------
    def set_p(self,joint_idx=None,joint_name=None,p=None):
        """
        Set the position of a joint by index or name.

        Parameters:
            joint_idx (Any | None): Input joint idx. Defaults to `None`.
            joint_name (Any | None): Input joint name. Defaults to `None`.
            p (Any | None): Input p. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        j = self._resolve_idx(joint_idx=joint_idx,joint_name=joint_name)
        self.graph.nodes[j]['p'] = p

    def set_R(self,joint_idx=None,joint_name=None,R=None):
        """
        Set the rotation matrix of a joint by index or name.

        Parameters:
            joint_idx (Any | None): Input joint idx. Defaults to `None`.
            joint_name (Any | None): Input joint name. Defaults to `None`.
            R (Any | None): Input R. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        j = self._resolve_idx(joint_idx=joint_idx,joint_name=joint_name)
        self.graph.nodes[j]['R'] = R

    def set_T(self,joint_idx=None,joint_name=None,T=None):
        """
        Set the homogeneous transformation matrix of a joint by index or name.

        Parameters:
            joint_idx (Any | None): Input joint idx. Defaults to `None`.
            joint_name (Any | None): Input joint name. Defaults to `None`.
            T (Any | None): Input T. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        j = self._resolve_idx(joint_idx=joint_idx,joint_name=joint_name)
        T = np.asarray(T).reshape(4,4)
        self.graph.nodes[j]['p'] = T[:3,3].copy()
        self.graph.nodes[j]['R'] = T[:3,:3].copy()

    def set_pose(self,joint_idx=None,joint_name=None,p=None,R=None):
        """
        Set pose.

        Parameters:
            joint_idx (Any | None): Input joint idx. Defaults to `None`.
            joint_name (Any | None): Input joint name. Defaults to `None`.
            p (Any | None): Input p. Defaults to `None`.
            R (Any | None): Input R. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if p is not None:
            self.set_p(joint_idx=joint_idx,joint_name=joint_name,p=p)
        if R is not None:
            self.set_R(joint_idx=joint_idx,joint_name=joint_name,R=R)

    def set_p_offset(self,joint_idx=None,joint_name=None,p_offset=None):
        """
        Set the position offset of a joint by index or name.

        Parameters:
            joint_idx (Any | None): Input joint idx. Defaults to `None`.
            joint_name (Any | None): Input joint name. Defaults to `None`.
            p_offset (Any | None): Input p offset. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        j = self._resolve_idx(joint_idx=joint_idx,joint_name=joint_name)
        self.graph.nodes[j]['p_offset'] = p_offset

    def set_R_offset(self,joint_idx=None,joint_name=None,R_offset=None):
        """
        Set the rotation offset of a joint by index or name.

        Parameters:
            joint_idx (Any | None): Input joint idx. Defaults to `None`.
            joint_name (Any | None): Input joint name. Defaults to `None`.
            R_offset (Any | None): Input R offset. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        j = self._resolve_idx(joint_idx=joint_idx,joint_name=joint_name)
        self.graph.nodes[j]['R_offset'] = R_offset

    def set_axis(self,joint_idx=None,joint_name=None,a=None):
        """
        Set the axis of a joint by index or name.

        Parameters:
            joint_idx (Any | None): Input joint idx. Defaults to `None`.
            joint_name (Any | None): Input joint name. Defaults to `None`.
            a (Any | None): Input a. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        j = self._resolve_idx(joint_idx=joint_idx,joint_name=joint_name)
        self.graph.nodes[j]['a'] = a

    def set_q(self,joint_idx=None,joint_name=None,q=0.0):
        """
        Set the joint variable (q) of a joint by index or name.

        Parameters:
            joint_idx (Any | None): Input joint idx. Defaults to `None`.
            joint_name (Any | None): Input joint name. Defaults to `None`.
            q (float): Input q. Defaults to `0.0`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        j = self._resolve_idx(joint_idx=joint_idx,joint_name=joint_name)
        self.graph.nodes[j]['q'] = float(q)

    # Root helpers
    def get_root_pose(self):
        """
        Get the position and rotation matrix of the root joint.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        root_name = self.get_root_name()
        return self.get_pose(joint_name=root_name)

    def set_root_pose(self,p=None,R=None):
        """
        Set the position and rotation matrix of the root joint.

        Parameters:
            p (Any | None): Input p. Defaults to `None`.
            R (Any | None): Input R. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        root_name = self.get_root_name()
        self.set_pose(joint_name=root_name,p=p,R=R)

    # ---------------------------------------------------------------------
    # qpos helpers (hinge scalar only, for now)
    # ---------------------------------------------------------------------
    def get_qpos(self,joint_names=None,flatten=True):
        """
        Get the joint variables (q) for multiple joints.

        Parameters:
            joint_names (Any | None): Input joint names. Defaults to `None`.
            flatten (bool): Input flatten. Defaults to `True`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if joint_names is None:
            joint_names = self.joint_names
        qs = [float(self.get_q(joint_name=nm)) for nm in joint_names]
        qs = np.asarray(qs,dtype=np.float64)
        if flatten:
            return qs
        else:
            return qs.reshape(-1,1)

    def set_qpos(self,joint_names,qs):
        """
        Set the joint variables (q) for multiple joints.

        Parameters:
            joint_names (list[str]): Input joint names.
            qs (np.ndarray): Input qs.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        qs = np.asarray(qs).reshape(-1)
        if len(joint_names)!=len(qs):
            raise ValueError("len(joint_names)=[%d] != len(qs)=[%d]"%(len(joint_names),len(qs)))
        for nm,q in zip(joint_names,qs):
            self.set_q(joint_name=nm,q=float(q))

    # ---------------------------------------------------------------------
    # Batch offset setters
    # ---------------------------------------------------------------------
    def set_R_offset_list(self,joint_name_list,R_offset_list):
        """
        Set rotation offsets for multiple joints.

        Parameters:
            joint_name_list (Any): Input joint name list.
            R_offset_list (Any): Input R offset list.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        for nm,Roff in zip(joint_name_list,R_offset_list):
            self.set_R_offset(joint_name=nm,R_offset=Roff)

    def set_p_offset_list(self,joint_name_list,p_offset_list):
        """
        Set position offsets for multiple joints.

        Parameters:
            joint_name_list (Any): Input joint name list.
            p_offset_list (np.ndarray): Input p offset list.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        for nm,poff in zip(joint_name_list,p_offset_list):
            self.set_p_offset(joint_name=nm,p_offset=poff)

    # ---------------------------------------------------------------------
    # Build chain
    # ---------------------------------------------------------------------
    def add_joint(
            self,
            name        = '',
            p_offset    = np.zeros(3),
            R_offset    = np.eye(3),
            a           = np.array([0,0,0]),
            p           = np.zeros(3),
            R           = np.eye(3),
            parent_name = None,
        ):
        """
        Add a joint to the kinematic chain.

        Parameters:
            name (str): Input name. Defaults to `''`.
            p_offset (np.ndarray): Input p offset. Defaults to `np.zeros(3)`.
            R_offset (Any): Input R offset. Defaults to `np.eye(3)`.
            a (Any): Input a. Defaults to `np.array([0, 0, 0])`.
            p (np.ndarray): Input p. Defaults to `np.zeros(3)`.
            R (Any): Input R. Defaults to `np.eye(3)`.
            parent_name (Any | None): Input parent name. Defaults to `None`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        new_joint_idx = self.get_n_joint()
        self.graph.add_node(new_joint_idx)

        p = np.asarray(p).reshape(3)
        R = np.asarray(R).reshape(3,3)
        a = np.asarray(a).reshape(3)
        p_offset = np.asarray(p_offset).reshape(3)
        R_offset = np.asarray(R_offset).reshape(3,3)
        joint_info = {
            'name':name,
            'p':p,
            'R':R,
            'q':0.0,
            'a':a,
            'p_offset':p_offset,
            'R_offset':R_offset,
            'parent':None,
            'childs':[],
        }
        self.graph.update(nodes=[(new_joint_idx,joint_info)]) # Add joint info

        # Record joint name
        self.joint_names.append(name)

        # Record revolute joint names
        if np.allclose(np.linalg.norm(a),1.0,atol=1e-8):
            self.rev_joint_names.append(name)

        if parent_name is not None:
            parent_idx = self.idx(parent_name)
            self.graph.nodes[new_joint_idx]['parent'] = parent_idx
            self.graph.add_edge(u_of_edge=parent_idx,v_of_edge=new_joint_idx)
            # Append to parent's children (FIXED)
            self.graph.nodes[parent_idx]['childs'].append(new_joint_idx)

    # ---------------------------------------------------------------------
    # FK
    # ---------------------------------------------------------------------
    def forward_kinematics(self,root_idx=0):
        """
        Perform forward kinematics to update positions and rotations of all joints.

        Parameters:
            root_idx (int): Input root idx. Defaults to `0`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        for _,edge in enumerate(dfs_edges(self.graph,source=root_idx)):
            idx_fr = edge[0]
            idx_to = edge[1]
            j_fr   = self.graph.nodes[idx_fr]
            j_to   = self.graph.nodes[idx_to]

            # p
            p = j_fr['R']@j_to['p_offset'] + j_fr['p']
            j_to['p'] = p

            # R
            a = j_to['a']
            if abs(np.linalg.norm(a)-1.0) < 1e-6:
                q = j_to['q']
                R = j_fr['R']@j_to['R_offset']@rodrigues(a=a,q_rad=q)
            else:
                R = j_fr['R']@j_to['R_offset']
            j_to['R'] = R

    # ---------------------------------------------------------------------
    # Debug / visualization (matplotlib)
    # ---------------------------------------------------------------------
    def print_chain_info(self):
        """
        Print the position and orientation (in RPY degrees) of each joint in the
        kinematic chain.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        n_joint = self.get_n_joint()
        for j_idx in range(n_joint):
            j = self.get_joint(joint_idx=j_idx)
            print("[%d/%d] joint name:[%s] p:%s rpy_deg:%s"%
                  (j_idx,n_joint,j['name'],j['p'],np.degrees(r2rpy(j['R']))))

    def plot_chain_graph(
            self,
            align           = 'horizontal',
            figsize         = (5,3),
            node_size       = 300,
            font_size_node  = 10,
            node_colors     = None,
            font_size_title = 10,
            root_on_top     = True,
        ):
        """
        Plot the kinematic chain graph using matplotlib.

        Parameters:
            align (str): Input align. Defaults to `'horizontal'`.
            figsize (tuple): Input figsize. Defaults to `(5, 3)`.
            node_size (int): Input node size. Defaults to `300`.
            font_size_node (int): Input font size node. Defaults to `10`.
            node_colors (Any | None): Input node colors. Defaults to `None`.
            font_size_title (int): Input font size title. Defaults to `10`.
            root_on_top (bool): Input root on top. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        n_joint = self.get_n_joint()
        tree    = self.graph

        for layer,nodes in enumerate(nx.topological_generations(tree)):
            for node in nodes:
                tree.nodes[node]['layer'] = layer

        pos = nx.multipartite_layout(
            tree,
            align      = align,
            scale      = 1.0,
            subset_key = 'layer',
        )

        if root_on_top:
            pos = {node:(x,-y) for node,(x,y) in pos.items()}

        fig,ax = plt.subplots(figsize=figsize)

        if node_colors is None:
            node_colors = []
            for j_idx in range(n_joint):
                a = self.get_axis(joint_idx=j_idx)
                if np.linalg.norm(a) < 1e-6:
                    node_color = (1,1,1,0.5)
                else:
                    node_color = [0,0,0,0.5]
                    node_color[np.argmax(a)] = 1
                    node_color = tuple(node_color)
                node_colors.append(node_color)

        nx.draw_networkx(
            tree,
            pos         = pos,
            ax          = ax,
            with_labels = True,
            node_size   = node_size,
            font_size   = font_size_node,
            node_color  = node_colors,
            linewidths  = 1,
            edgecolors  = 'k',
        )

        ax.set_title('%s'%(tree.name),fontsize=font_size_title)
        fig.tight_layout()
        plt.show()