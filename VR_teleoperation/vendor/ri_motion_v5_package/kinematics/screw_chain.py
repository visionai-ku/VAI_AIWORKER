import numpy as np

from scipy.linalg import logm, expm

from ri_motion_v5_package.kinematics.transforms import ( # type: ignore
    pr2t,
    t2pr,
    t2r,
    R_from_axis,
    screw_to_se3,
)
from ri_motion_v5_package.utility.viz import (
    get_colors,
)

class ScrewChain(object):
    """
    Screw Chain.

    Usage:
        obj = ScrewChain(...)
    """
    def __init__(self,name='Screw Chain',verbose=True):
        """
        Initialize ScrewChain.

        Parameters:
            name (str): Input name. Defaults to `'Screw Chain'`.
            verbose (bool): Input verbose. Defaults to `True`.

        Returns:
            None: Initializes the instance in place.
        """
        # Parse arguments
        self.name    = name
        self.verbose = verbose

        # Buffers
        self.n = 0 # number of screws
        self.parent_name_list = [] # parent names
        self.name_list        = [] # screw names
        self.rev_name_list    = [] # revolute screw names

        # Screw parameters
        self.q_list           = [] # screw position in global coordinates (3)
        self.w_list           = [] # axis of rotation in global coordinates (3)
        self.M_list           = [] # transformation matrix in global coordinates (4x4)
        self.type_list        = [] # type of screw ('fixed' or 'revolute')

        # Data structures
        self.children         = {}
        self.root_name        = None # name of the root screw
        self.root_idx         = None # index of the root screw

        if self.verbose:
            print ("[%s] Initialized."%(self.name))

    def add_screw(self,parent_name=None,name=None,q=(0,0,0),w=(0,0,0)):
        """
        Add a screw to the chain.

        Parameters:
            parent_name (Any | None): Input parent name. Defaults to `None`.
            name (Any | None): Input name. Defaults to `None`.
            q (tuple): Input q. Defaults to `(0, 0, 0)`.
            w (tuple): Input w. Defaults to `(0, 0, 0)`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.n += 1 # increment the number of screws

        # Append name
        self.parent_name_list.append(parent_name)
        if parent_name is None: # if no parent, this is the root
            self.root_name = name
            self.root_idx  = self.n - 1 # index of the root screw
        self.name_list.append(name)
        if np.linalg.norm(w) > 1e-6: # if w is not zero, it is a revolute screw
            self.rev_name_list.append(name)

        # Append screw parameters
        self.q_list.append(np.array(q))
        self.w_list.append(np.array(w))
        self.M_list.append(pr2t(p=np.array(q),R=R_from_axis(np.array(w))))
        screw_type = 'revolute' if np.linalg.norm(w) > 1e-6 else 'fixed'
        self.type_list.append(screw_type)

        # Build child map
        if parent_name not in self.children:
            self.children[parent_name] = []
        self.children[parent_name].append(name)

    def get_idx_fr_to_list(self, name=None):
        """
        Recursively collect all (idx_fr,idx_to) tuples starting from `name`.

        Parameters:
            name (Any | None): Input name. Defaults to `None`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if name is None: # if no name is given, start from the root
            name = self.root_name
        idx = self.name_list.index(name)
        idx_fr_to_list = []
        for child_name in self.children.get(name, []):
            child_idx = self.name_list.index(child_name)
            idx_fr_to_list.append((idx, child_idx))
            idx_fr_to_list.extend(self.get_idx_fr_to_list(child_name))
        return idx_fr_to_list

    def forward_kinematics(self,screw_names,thetas):
        """
        Forward kinematics of the screw chain.

        Parameters:
            screw_names (list[str]): Input screw names.
            thetas (Any): Input thetas.

        Returns:
            Any: Result computed from the current inputs.
        """
        # Update twist exponentials
        theta_map = dict(zip(screw_names,thetas))
        T_list = []
        for idx in range(self.n): # for each screw
            name = self.name_list[idx]
            theta = theta_map.get(name,0.0)
            w = self.w_list[idx] # (3)
            q = self.q_list[idx] # (3)
            T = expm(screw_to_se3(w=w,q=q)*theta) # (4x4)
            T_list.append(T)

        # PoE
        M_fk_list  = ['']*self.n # fk results, (4x4) list
        T_bar_list = ['']*self.n # cumprod of T, (4x4) list
        M_fk_list[self.root_idx] = self.M_list[self.root_idx]
        T_bar_list[self.root_idx] = T_list[self.root_idx]
        for (idx_fr,idx_to) in self.get_idx_fr_to_list():
            # print ("idx_fr:[%d] idx_to:[%d]"%(idx_fr,idx_to))
            M_fk_list[idx_to] = T_bar_list[idx_fr] @ self.M_list[idx_to]
            T_bar_list[idx_to] = T_bar_list[idx_fr] @ T_list[idx_to]

        # Extract screw information
        q_fk_list,R_fk_list,w_fk_list = [],[],[]
        for idx in range(self.n):
            q_fk,R_fk = t2pr(M_fk_list[idx]) # (3),(3x3)
            w_fk = R_fk[:,2] # (3)
            screw_type = self.type_list[idx]
            if screw_type == 'fixed': w_fk = np.zeros(3)
            q_fk_list.append(q_fk)
            R_fk_list.append(R_fk)
            w_fk_list.append(w_fk)

        # Return
        return q_fk_list,w_fk_list,M_fk_list

    def plot_chain_mujoco(
            self,
            env,
            q_list    = None,
            w_list    = None,
            M_list    = None,
            r_sphere  = 0.05,
            r_axis    = 0.01,
            axis_len  = 0.1,
            r_link    = 0.01,
            alpha     = 0.2,
            plot_text = False,
            cmap_name = 'gist_rainbow',
        ):
        """
        Plot the screw chain in MuJoCo.

        Parameters:
            env (Any): Input env.
            q_list (Any | None): Input q list. Defaults to `None`.
            w_list (Any | None): Input w list. Defaults to `None`.
            M_list (Any | None): Input M list. Defaults to `None`.
            r_sphere (float): Input r sphere. Defaults to `0.05`.
            r_axis (float): Input r axis. Defaults to `0.01`.
            axis_len (float): Input axis len. Defaults to `0.1`.
            r_link (float): Input r link. Defaults to `0.01`.
            alpha (float): Input alpha. Defaults to `0.2`.
            plot_text (bool): Input plot text. Defaults to `False`.
            cmap_name (str): Input cmap name. Defaults to `'gist_rainbow'`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if q_list is None: q_list = self.q_list
        if w_list is None: w_list = self.w_list
        if M_list is None: M_list = self.M_list

        # Screws
        colors = get_colors(n_color=self.n,cmap_name=cmap_name,alpha=alpha)
        for idx in range(self.n):
            q = q_list[idx]
            w = w_list[idx]
            M = M_list[idx]
            R = t2r(M)
            rgba = colors[idx]
            # Axis
            plot_axis = True if np.linalg.norm(w)>1e-6 else False
            if plot_axis:
                p_fr = q - 0.5*axis_len*w
                p_to = q + 0.5*axis_len*w
                env.plot_cylinder_fr2to(
                    p_fr  = p_fr,
                    p_to  = p_to,
                    r     = r_axis,
                    rgba  = rgba,
                    label = '',
                )
            else:
                # Sphere
                env.plot_sphere(p=q,r=r_sphere,rgba=rgba)
            # Text
            if plot_text:
                env.plot_text(q, self.name_list[idx])

        # Links
        for (idx_fr,idx_to) in self.get_idx_fr_to_list():
            q_fr = q_list[idx_fr]
            q_to = q_list[idx_to]
            env.plot_cylinder_fr2to(q_fr, q_to, r_link, (0,0,0,alpha))

    def plot_init_chain_mujoco(self,env,**cfgs):
        """
        Plot the initial screw chain in MuJoCo.

        Parameters:
            env (Any): Input env.
            **cfgs (Any): Additional keyword arguments.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.plot_chain_mujoco(
            env,
            q_list = self.q_list,
            w_list = self.w_list,
            M_list = self.M_list,
            **cfgs
        )
