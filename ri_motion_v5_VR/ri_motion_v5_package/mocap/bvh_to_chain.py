import numpy as np
import matplotlib.pyplot as plt
from math import radians, cos, sin
from collections import deque
from copy import deepcopy

# """
# %run ../../package/init_kinematics.py
# """
# from numpy_chain import NumpyChain # type: ignore
# from transforms import ( # type: ignore
#     rpy2r,
#     rpy2r_order,
#     t2p,
#     t2r,
#     pr2t,
#     t2pr,
# )
from ri_motion_v5_package.kinematics.numpy_chain import (
    NumpyChain,
)
from ri_motion_v5_package.kinematics.transforms import (
    rpy2r,
    rpy2r_order,
    t2p,
    t2r,
    pr2t,
    t2pr,
)

ZEROMAT = np.array(
    [[0., 0., 0., 0.], [0., 0., 0., 0.],
    [0., 0., 0., 0.], [0., 0., 0., 0.]])
IDENTITY = np.array(
    [[1., 0., 0., 0.], [0., 1., 0., 0.],
     [0., 0., 1., 0.], [0., 0., 0., 1.]])

class Node(object):
    """
    Represents a node (joint) in the BVH skeleton hierarchy.

    Usage:
        obj = Node(...)
    """

    def __init__(self, root=False):
        """
        Initialize Node.

        Parameters:
            root (bool): Input root. Defaults to `False`.

        Returns:
            None: Initializes the instance in place.
        """
        self.name = None
        self.channels = []
        self.offset = (0, 0, 0)
        self.children = []
        self._is_root = root

    @property
    def is_root(self):
        """
        Check if this node is the root joint.

        Parameters:
            None.

        Returns:
            bool: Boolean result for the requested condition.
        """
        return self._is_root

    @property
    def is_end_site(self):
        """
        Check if this node is an End Site (i.e., has no children).

        Parameters:
            None.

        Returns:
            bool: Boolean result for the requested condition.
        """
        return len(self.children) == 0

class BvhReader(object):
    """
    A basic BVH file reader that parses the hierarchy and motion data.

    Usage:
        obj = BvhReader(...)
    """

    def __init__(self, filename):
        """
        Initialize BvhReader.

        Parameters:
            filename (str): Input filename.

        Returns:
            None: Initializes the instance in place.
        """
        self.filename = filename
        # A list of unprocessed tokens (strings)
        self._token_list = []
        # The current line number
        self._line_num = 0

        # Root node
        self.root = None
        self._node_stack = []

        # Total number of channels
        self.num_channels = 0

    def on_hierarchy(self, root):
        """
        Callback invoked after reading the skeleton hierarchy.

        Parameters:
            root (str | Path): Input root.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        pass

    def on_motion(self, frames, dt):
        """
        Callback invoked after reading the motion header (frame count and frame time).

        Parameters:
            frames (Any): Input frames.
            dt (float): Input dt.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        pass

    def on_frame(self, values):
        """
        Callback invoked for each frame of motion data, passing a single frame's
        channel values.

        Parameters:
            values (Any): Input values.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        pass

    def read(self):
        """
        Read and parse the BVH file, invoking callbacks for hierarchy and motion data.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        with open(self.filename, 'r') as self._file_handle:
            self.read_hierarchy()
            self.on_hierarchy(self.root)
            self.read_motion()

    def read_motion(self):
        """
        Read the motion data portion of the BVH file, including frame count,.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        # No more tokens (i.e. end of file)? Then just return
        try:
            tok = self.token()
        except StopIteration:
            return

        if tok != "MOTION":
            raise SyntaxError("Syntax error in line %d: 'MOTION' expected, "
                              "got '%s' instead" % (self._line_num, tok))

        # Read the number of frames
        tok = self.token()
        if tok != "Frames:":
            raise SyntaxError("Syntax error in line %d: 'Frames:' expected, "
                              "got '%s' instead" % (self._line_num, tok))

        frames = self.int_token()

        # Read the frame time
        tok = self.token()
        if tok != "Frame":
            raise SyntaxError("Syntax error in line %d: 'Frame Time:' "
                              "expected, got '%s' instead"
                              % (self._line_num, tok))
        tok = self.token()
        if tok != "Time:":
            raise SyntaxError("Syntax error in line %d: 'Frame Time:' "
                              "expected, got 'Frame %s' instead"
                              % (self._line_num, tok))

        dt = self.float_token()

        self.on_motion(frames, dt)

        # Read the channel values
        for i in range(frames):
            s = self.read_line()
            a = s.split()
            if len(a) != self.num_channels:
                raise SyntaxError("Syntax error in line %d: %d float values "
                                  "expected, got %d instead"
                                  % (self._line_num, self.num_channels,
                                     len(a)))
            values = list(map(lambda x: float(x), a))  # In Python 3 map returns map-object, not a list. Can't slice.
            self.on_frame(values)

    def read_hierarchy(self):
        """
        Read the skeleton hierarchy from the BVH file, starting with the ROOT node.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        tok = self.token()
        if tok != "HIERARCHY":
            raise SyntaxError("Syntax error in line %d: 'HIERARCHY' expected, "
                              "got '%s' instead" % (self._line_num, tok))
        tok = self.token()
        if tok != "ROOT":
            raise SyntaxError("Syntax error in line %d: 'ROOT' expected, "
                              "got '%s' instead" % (self._line_num, tok))

        self.root = Node(root=True)
        self._node_stack.append(self.root)
        self.read_node()

    def read_node(self):
        """
        Read a single node (joint) in the BVH hierarchy, including its.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """

        # Read the node name (or the word 'Site' if it was a 'End Site' node)
        name = self.token()
        self._node_stack[-1].name = name

        tok = self.token()
        if tok != "{":
            raise SyntaxError("Syntax error in line %d: '{' expected, "
                              "got '%s' instead" % (self._line_num, tok))

        while 1:
            tok = self.token()
            if tok == "OFFSET":
                x = self.float_token()
                y = self.float_token()
                z = self.float_token()
                self._node_stack[-1].offset = (x, y, z)
            elif tok == "CHANNELS":
                n = self.int_token()
                channels = []
                for i in range(n):
                    tok = self.token()
                    if tok not in ["Xposition", "Yposition", "Zposition",
                                   "Xrotation", "Yrotation", "Zrotation"]:
                        raise SyntaxError("Syntax error in line %d: Invalid "
                                          "channel name: '%s'"
                                          % (self._line_num, tok))
                    channels.append(tok)
                self.num_channels += len(channels)
                self._node_stack[-1].channels = channels
            elif tok == "JOINT":
                node = Node()
                self._node_stack[-1].children.append(node)
                self._node_stack.append(node)
                self.read_node()
            elif tok == "End":
                node = Node()
                self._node_stack[-1].children.append(node)
                self._node_stack.append(node)
                self.read_node()
            elif tok == "}":
                if self._node_stack[-1].is_end_site:
                    self._node_stack[-1].name = "End Site"
                self._node_stack.pop()
                break
            else:
                raise SyntaxError("Syntax error in line %d: Unknown "
                                  "keyword '%s'" % (self._line_num, tok))

    def int_token(self):
        """
        Get the next token and interpret it as an integer.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        tok = self.token()
        try:
            return int(tok)
        except ValueError:
            raise SyntaxError("Syntax error in line %d: Integer expected, "
                              "got '%s' instead" % (self._line_num, tok))

    def float_token(self):
        """
        Get the next token and interpret it as a float.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        tok = self.token()
        try:
            return float(tok)
        except ValueError:
            raise SyntaxError("Syntax error in line %d: Float expected, "
                              "got '%s' instead" % (self._line_num, tok))

    def token(self):
        """
        Return the next token in the BVH file. If no more tokens are available.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        # Are there still some tokens left? then just return the next one
        if self._token_list:
            tok = self._token_list[0]
            self._token_list = self._token_list[1:]
            return tok

        # Read a new line
        s = self.read_line()
        self.create_tokens(s)
        return self.token()

    def read_line(self):
        """
        Read the next line from the file, skipping empty lines. Raises.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        # Discard any remaining tokens
        self._token_list = []
        # Read the next line
        while 1:
            s = self._file_handle.readline()
            self._line_num += 1
            if s == "":
                raise StopIteration
            return s

    def create_tokens(self, s):
        """
        Tokenize the given line by splitting on whitespace.

        Parameters:
            s (Any): Input s.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        s = s.strip()
        a = s.split()
        self._token_list = a


class Joint:
    """
    A class representing a joint in a BVH skeleton hierarchy.

    Usage:
        obj = Joint(...)
    """
    def __init__(self, name):
        """
        Initialize Joint.

        Parameters:
            name (str): Input name.

        Returns:
            None: Initializes the instance in place.
        """
        self.name = name
        self.children = []
        self.channels = []  # Ordered list of channels: each entry is one of [XYZ]position, [XYZ]rotation
        self.hasparent = 0  # flag
        self.parent = 0  # joint.addchild() sets this
        self.strans = np.array([0., 0., 0.])

        # Transformation matrices
        self.stransmat = np.array([[0., 0., 0., 0.], [0., 0., 0., 0.],
                                   [0., 0., 0., 0.], [0., 0., 0., 0.]])

        self.rot = {}
        self.trtr = {}
        self.worldpos = {}

    def info(self):
        """
        Print information about this joint and its children.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        print("Joint name:", self.name)
        print(" %s is connected to " % self.name,)
        if len(self.children) == 0:
            print("nothing")
        else:
            for child in self.children:
                print("%s " % child.name,)
            print()
        for child in self.children:
            child.info()

    def __str__(self):
        """
        Return a readable string representation.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        str2 = self.name + " at strans=" + str(self.strans) + " is connected to "
        if len(self.children) == 0:
            str2 = str2 + "nothing\n"
        else:
            for child in self.children:
                str2 = str2 + child.name + " "
            str2 = str2 + "\n"
        str3 = ""
        for child in self.children:
            str3 = str3 + child.__str__()
        str1 = str2 + str3
        return str1

    def addchild(self, childjoint):
        """
        Add a child joint to this joint, setting parent/child flags appropriately.

        Parameters:
            childjoint (Any): Input childjoint.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.children.append(childjoint)
        childjoint.hasparent = 1
        childjoint.parent = self


class ReadBVH(BvhReader):
    """
    A BVH reader that stores the hierarchy and motion data in attributes.

    Usage:
        obj = ReadBVH(...)
    """
    def on_hierarchy(self, root):
        """
        Store the parsed BVH hierarchy root.

        Parameters:
            root (str | Path): Input root.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.root = root
        self.keyframes = []

    def on_motion(self, frames, dt):
        """
        Store the parsed BVH motion frames and timestep.

        Parameters:
            frames (Any): Input frames.
            dt (float): Input dt.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.frames = frames
        self.dt = dt

    def on_frame(self, values):
        """
        Append one parsed BVH frame.

        Parameters:
            values (Any): Input values.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.keyframes.append(values)


def process_bvhnode(node, parentname='hips'):
    """
    Recursively process a BVH node to create a Joint hierarchy.

    Parameters:
        node (Any): Input node.
        parentname (str): Input parentname. Defaults to `'hips'`.

    Returns:
        Any: Result computed from the current inputs.
    """
    name = node.name
    if (name == "End Site") or (name == "end site"):
        name = parentname + "End"

    b1 = Joint(name)
    b1.channels = node.channels
    b1.strans[0] = node.offset[0]
    b1.strans[1] = node.offset[1]
    b1.strans[2] = node.offset[2]

    b1.stransmat = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                             [0., 0., 1., 0.], [0., 0., 0., 1.]])

    b1.stransmat[0, 3] = b1.strans[0]
    b1.stransmat[1, 3] = b1.strans[1]
    b1.stransmat[2, 3] = b1.strans[2]

    for child in node.children:
        b2 = process_bvhnode(child, name)
        b1.addchild(b2)
    return b1


class Skeleton:
    """
    A container class representing a full skeleton from a BVH file,.

    Usage:
        obj = Skeleton(...)
    """

    def __init__(self, hips, keyframes, frames=0, dt=.033333333, ignore_root_offset=True):
        """
        Initialize Skeleton.

        Parameters:
            hips (Any): Input hips.
            keyframes (Any): Input keyframes.
            frames (int): Input frames. Defaults to `0`.
            dt (float): Input dt. Defaults to `0.033333333`.
            ignore_root_offset (bool): Input ignore root offset. Defaults to `True`.

        Returns:
            None: Initializes the instance in place.
        """
        self.root = hips
        self.keyframes = keyframes
        self.frames = frames
        self.dt = dt
        self.edges = {}

        offset = 0
        for channel in self.root.channels:
            if channel == "Xposition":
                xoffset = offset
            if channel == "Yposition":
                yoffset = offset
            if channel == "Zposition":
                zoffset = offset
            offset += 1

        self.minx = 999999999999
        self.miny = 999999999999
        self.minz = 999999999999
        self.maxx = -999999999999
        self.maxy = -999999999999
        self.maxz = -999999999999

        if ignore_root_offset:
            self.root.strans[0] = 0.0
            self.root.strans[1] = 0.0
            self.root.strans[2] = 0.0
            self.root.stransmat = IDENTITY

        xcorrect = self.root.strans[0]
        ycorrect = self.root.strans[1]
        zcorrect = self.root.strans[2]

        for keyframe in self.keyframes:
            x = keyframe[xoffset] + xcorrect
            y = keyframe[yoffset] + ycorrect
            z = keyframe[zoffset] + zcorrect
            if x < self.minx:
                self.minx = x
            if x > self.maxx:
                self.maxx = x
            if y < self.miny:
                self.miny = y
            if y > self.maxy:
                self.maxy = y
            if z < self.minz:
                self.minz = z
            if z > self.maxz:
                self.maxz = z

    def __str__(self):
        """
        Return a readable string representation.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        str1 = "frames = " + str(self.frames) + ", dt = " + str(self.dt) + "\n"
        str1 = str1 + self.root.__str__()
        return str1

    @staticmethod
    def joint_dfs(root):
        """
        Yield joints in depth-first order.

        Parameters:
            root (str | Path): Input root.

        Returns:
            Any: Result computed from the current inputs.
        """
        nodes = []
        stack = [root]
        while stack:
            cur_node = stack[0]
            stack = stack[1:]
            nodes.append(cur_node)
            for child in cur_node.children:
                stack.insert(0, child)
        return nodes

    def get_frames_worldpos(self, n=None):
        """
        Return frames worldpos.

        Parameters:
            n (Any | None): Input n. Defaults to `None`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        joints = self.joint_dfs(self.root)

        frame_data = []
        if n is None:
            for i in range(len(self.keyframes)):
                t = i * self.dt
                single_frame = [t, ]
                for j in joints:
                    single_frame.extend(j.worldpos[t][:3])
                frame_data.append(single_frame)
        else:
            t = n * self.dt
            single_frame = [t, ]
            for j in joints:
                single_frame.extend(j.worldpos[t][:3])
            frame_data.append(single_frame)

        header = ["{}.{}".format(j.name, thing) for j in joints for thing in ("X", "Y", "Z")]
        header = ["Time", ] + header
        return header, frame_data

    def get_frames_rotations(self, n=None):
        """
        Return frames rotations.

        Parameters:
            n (Any | None): Input n. Defaults to `None`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        joints = self.joint_dfs(self.root)

        frame_data = []
        if n is None:
            for i in range(len(self.keyframes)):
                t = i * self.dt
                single_frame = [t, ]
                for j in joints:
                    if j.rot:
                        rot = j.rot[t]
                    else:
                        rot = [0.0, 0.0, 0.0]
                    single_frame.extend(rot)
                frame_data.append(single_frame)
        else:
            t = n * self.dt
            single_frame = [t, ]
            for j in joints:
                if j.rot:
                    rot = j.rot[t]
                else:
                    rot = [0.0, 0.0, 0.0]
                single_frame.extend(rot)
            frame_data.append(single_frame)

        header = ["{}.{}".format(j.name, thing) for j in joints for thing in ("X", "Y", "Z")]
        header = ["Time", ] + header
        return header, frame_data

    def get_frame(self, f):
        """
        Return frame.

        Parameters:
            f (Any): Input f.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        joints = self.joint_dfs(self.root)
        frame_data = dict()

        t = f * self.dt
        for j in joints:
            frame_data[j.name] = j.rot[t] if t in j.rot else None, j.worldpos[t][:3]
        return frame_data

    def get_offsets(self):
        """
        Return offsets.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        joints = self.joint_dfs(self.root)
        offsets = dict()
        for j in joints:
            offsets[j.name] = j.strans
        return offsets

    def as_dict(self):
        """
        Return the skeleton as a nested dictionary.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        joints = self.joint_dfs(self.root)
        joints_dict = {}

        for j in joints:
            if not j.hasparent:
                type = 'root'
            else:
                type = 'joint'
            if j.name[-3:] == "End":
                type = 'end'

            if j.rot:
                rot_0 = tuple(j.rot[0])
            else:
                rot_0 = None

            joints_dict[j.name] = (j.parent.name if j.hasparent else None,
                                   tuple(j.strans),
                                   rot_0,
                                   type,
                                   [child.name for child in j.children])
        return joints_dict


def process_bvhkeyframe(keyframe, joint, t, DEBUG=0):
    """
    Process bvhkeyframe.

    Parameters:
        keyframe (Any): Input keyframe.
        joint (Any): Input joint.
        t (Any): Input t.
        DEBUG (int): Input DEBUG. Defaults to `0`.

    Returns:
        Any: Result computed from the current inputs.
    """
    counter = 0
    dotrans = 0

    drotmat = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                        [0., 0., 1., 0.], [0., 0., 0., 1.]])

    if DEBUG:
        print(" process_bvhkeyframe: doing joint %s, t=%d" % (joint.name, t))
        print(" keyframe has %d elements in it." % (len(keyframe)))

    has_xrot = False
    has_yrot = False
    has_zrot = False
    for channel in joint.channels:
        keyval = keyframe[counter]
        if channel == "Xposition":
            dotrans = 1
            xpos = keyval
        elif channel == "Yposition":
            dotrans = 1
            ypos = keyval
        elif channel == "Zposition":
            dotrans = 1
            zpos = keyval
        elif channel == "Xrotation":
            has_xrot = True
            xrot = keyval
            theta = radians(xrot)
            mycos = cos(theta)
            mysin = sin(theta)
            drotmat2 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                                 [0., 0., 1., 0.], [0., 0., 0., 1.]])
            drotmat2[1, 1] = mycos
            drotmat2[1, 2] = -mysin
            drotmat2[2, 1] = mysin
            drotmat2[2, 2] = mycos
            drotmat = np.dot(drotmat, drotmat2)

        elif channel == "Yrotation":
            has_yrot = True
            yrot = keyval
            theta = radians(yrot)
            mycos = cos(theta)
            mysin = sin(theta)
            drotmat2 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                                 [0., 0., 1., 0.], [0., 0., 0., 1.]])
            drotmat2[0, 0] = mycos
            drotmat2[0, 2] = mysin
            drotmat2[2, 0] = -mysin
            drotmat2[2, 2] = mycos
            drotmat = np.dot(drotmat, drotmat2)

        elif channel == "Zrotation":
            has_zrot = True
            zrot = keyval
            theta = radians(zrot)
            mycos = cos(theta)
            mysin = sin(theta)
            drotmat2 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                                 [0., 0., 1., 0.], [0., 0., 0., 1.]])
            drotmat2[0, 0] = mycos
            drotmat2[0, 1] = -mysin
            drotmat2[1, 0] = mysin
            drotmat2[1, 1] = mycos
            drotmat = np.dot(drotmat, drotmat2)
        else:
            print("Fatal error in process_bvhkeyframe: illegal channel name ", channel)
            return(0)
        counter += 1

    if has_xrot or has_yrot or has_zrot:
        joint.rot[t] = (xrot, yrot, zrot)

    if dotrans:
        dtransmat = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.],
                              [0., 0., 1., 0.], [0., 0., 0., 1.]])
        dtransmat[0, 3] = xpos
        dtransmat[1, 3] = ypos
        dtransmat[2, 3] = zpos

        if DEBUG:
            print("  Joint %s: xpos ypos zpos is %s %s %s" % (joint.name, xpos, ypos, zpos))
            print("  Joint %s: xrot yrot zrot is %s %s %s" % (joint.name, xrot, yrot, zrot))

    if joint.hasparent:
        parent_trtr = joint.parent.trtr[t]
        localtoworld = np.dot(parent_trtr, joint.stransmat)
    else:
        localtoworld = np.dot(joint.stransmat, dtransmat)

    trtr = np.dot(localtoworld, drotmat)
    joint.trtr[t] = trtr

    worldpos = np.array([localtoworld[0, 3], localtoworld[1, 3],
                         localtoworld[2, 3], localtoworld[3, 3]])
    joint.worldpos[t] = worldpos

    newkeyframe = keyframe[counter:]
    for child in joint.children:
        newkeyframe = process_bvhkeyframe(newkeyframe, child, t, DEBUG=DEBUG)
        if newkeyframe == 0:
            print("Passing up fatal error in process_bvhkeyframe")
            return 0
    return newkeyframe


def process_bvhfile(filename, DEBUG=0):
    """
    Process bvhfile.

    Parameters:
        filename (str): Input filename.
        DEBUG (int): Input DEBUG. Defaults to `0`.

    Returns:
        Any: Result computed from the current inputs.
    """
    if DEBUG:
        print("Reading BVH file...",)
    my_bvh = ReadBVH(filename)
    my_bvh.read()

    hips = process_bvhnode(my_bvh.root)
    if DEBUG:
        print("done")

    if DEBUG:
        print("Building skeleton...",)
    myskeleton = Skeleton(hips, keyframes=my_bvh.keyframes, frames=my_bvh.frames, dt=my_bvh.dt)
    if DEBUG:
        print("done")
        print("skeleton is: ", myskeleton)
    return myskeleton


def get_skeleton_from_bvh(bvh_path):
    """
    Return skeleton from bvh.

    Parameters:
        bvh_path (str | Path): Input bvh path.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    skeleton = process_bvhfile(bvh_path)
    for tick in range(skeleton.frames):
        _ = process_bvhkeyframe(
            skeleton.keyframes[tick],
            skeleton.root,
            skeleton.dt*tick,
        )
    return skeleton


def get_chains_from_skeleton(
        skeleton,
        env              = None,
        rpy_order        = [2,1,0],
        p_rate           = 1.0,
        plot_chain_graph = True,
        plot_init_chain  = True,
        verbose          = True,
        name             = 'mocap',
    ):
    """
    Convert a Skeleton object into a list of NumpyChain objects representing each
    frame.

    Parameters:
        skeleton (Any): Input skeleton.
        env (Any | None): Input env. Defaults to `None`.
        rpy_order (list): Input rpy order. Defaults to `[2, 1, 0]`.
        p_rate (float): Input p rate. Defaults to `1.0`.
        plot_chain_graph (bool): Input plot chain graph. Defaults to `True`.
        plot_init_chain (bool): Input plot init chain. Defaults to `True`.
        verbose (bool): Input verbose. Defaults to `True`.
        name (str): Input name. Defaults to `'mocap'`.

    Returns:
        Any: Requested value computed from the current inputs.
    """

    L = skeleton.frames
    skel_pos_info,skel_pos_time_list = skeleton.get_frames_worldpos()
    skel_rot_info,skel_rot_time_list = skeleton.get_frames_rotations()
    secs = np.array(skel_pos_time_list)[:,0]
    skel_pos_array     = np.array(skel_pos_time_list)[:,1:].reshape(L,-1,3)
    skel_rpy_deg_array = np.array(skel_rot_time_list)[:,1:].reshape(L,-1,3)

    # Initialize kinematic chain
    chain = NumpyChain(name=name)
    node = skeleton.root
    chain.add_joint(
        name=node.name,
        p=np.array([0,0,0]),
        R=rpy2r(np.radians([0,0,0])),
    )
    deq = deque()
    while (len(node.children)>0) or (len(deq)>0):
        if len(deq) > 0:
            node = deq.pop()
            chain.add_joint(
                name        = node.name,
                parent_name = node.parent.name,
                p_offset    = t2p(np.array(node.stransmat))*p_rate,
                R_offset    = t2r(np.array(node.stransmat)),
            )
        for child in node.children:
            deq.append(child)

    # Copy init chain
    chain_init = deepcopy(chain)

    # Root joint name
    root_name = chain.get_root_name()

    # Print joint names
    if verbose:
        print("[Joint names]")
        print("root_name:[%s]"%(root_name))
        for j_idx,joint_name in enumerate(chain.joint_names):
            print("[%02d/%d] joint_name:[%s]"%(j_idx,chain.get_n_joint(),joint_name))

    # Plot chain graph
    if plot_chain_graph:
        chain.plot_chain_graph(
            align           = 'vertical',
            figsize         = (5,2),
            node_size       = 200,
            font_size_node  = 8,
            node_colors     = None,
            font_size_title = 10,
            root_on_top     = True,
        )

    # NOTE:
    # plot_init_chain block still uses chain.plot_chain_mujoco which no longer exists in NumpyChain.
    # If you want to keep this visualization, replace it with draw_numpy_chain(env,chain,...).

    if plot_init_chain:
        env.init_viewer(title="Kinematic Chain", width=1200, height=800, hide_menu=True)
        env.set_viewer(
            azimuth       = 177,
            distance      = 4.0,
            elevation     = -28,
            lookat        = [0.0,0.1,0.65],
            transparent   = False,
            contactpoint  = True,
            contactwidth  = 0.05,
            contactheight = 0.05,
            contactrgba   = np.array([1,0,0,1]),
            joint         = True,
            jointlength   = 0.5,
            jointwidth    = 0.1,
            jointrgba     =[0.2,0.6,0.8,0.6],
        )
        env.reset()
        chain.set_root_pose(p=np.array([0,0,0]),R=rpy2r(np.radians([0,0,0])))
        chain.forward_kinematics()
        while env.is_viewer_alive():
            # chain.plot_chain_mujoco(...)  # removed in new NumpyChain
            env.plot_T(p=np.zeros(3),R=np.eye(3,3),plot_axis=True,axis_len=1.0,axis_width=0.005)
            env.render()
            if (env.render_tick%10)==0:
                scene_img = env.grab_image()
        env.close_viewer()
        plt.figure(figsize=(5,4));plt.imshow(scene_img)
        plt.title("Initial Chain",fontsize=10);plt.axis('off');plt.show()
        print("Done.")

    # Make chains
    chains = []
    for tick in range(L):
        p_root = p_rate*skel_pos_array[tick,0,:]
        R_offset_list = []
        for joint_idx in range(chain.get_n_joint()):
            if joint_idx == 0:
                rpy_deg = np.array([0,0,0])
                rpy_deg_root = skel_rpy_deg_array[tick,joint_idx,:]
            else:
                rpy_deg = skel_rpy_deg_array[tick,joint_idx,:]
            R_offset_list.append(
                rpy2r_order(np.radians(rpy_deg),order=rpy_order)
            )

        R_root = rpy2r_order(np.radians(rpy_deg_root),order=rpy_order)
        chain.set_root_pose(p=p_root,R=R_root)
        chain.set_R_offset_list(chain.joint_names,R_offset_list)
        chain.forward_kinematics()
        chains.append(deepcopy(chain))

    return secs,chains


def get_chains_translated(
        chains,
        T_trans = pr2t(np.array([0,0,0]),rpy2r(np.radians([90,0,0]))),
    ):

    """
    Return chains translated.

    Parameters:
        chains (Any): Input chains.
        T_trans (Any): Input T trans. Defaults to `pr2t(np.array([0, 0, 0]),
                       rpy2r(np.radians([90, 0, 0])))`.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    chains_zup = []
    for tick,chain in enumerate(chains):
        p_root,R_root = chain.get_root_pose()
        T_root = pr2t(p_root,R_root)
        T_root_zup = T_trans @ T_root
        p_root_zup,R_root_zup = t2pr(T_root_zup)

        chain_zup = deepcopy(chain)
        chain_zup.set_root_pose(p=p_root_zup,R=R_root_zup)
        chain_zup.forward_kinematics()
        chains_zup.append(chain_zup)

    return chains_zup


def get_chains_from_bvh(
        bvh_path,
        env              = None,
        rpy_order        = [2,1,0],
        p_rate           = 1.0,
        zup              = True,
        p_offset         = np.array([0,0,0]),
        R_offset         = np.eye(3,3),
        plot_chain_graph = False,
        plot_init_chain  = False,
        verbose          = False,
        name             = 'mocap',
        mocap_type       = None, # 'seg' or 'cmu'
    ):
    """
    Convert a BVH file into a list of NumpyChain objects representing each frame.

    Parameters:
        bvh_path (str | Path): Input bvh path.
        env (Any | None): Input env. Defaults to `None`.
        rpy_order (list): Input rpy order. Defaults to `[2, 1, 0]`.
        p_rate (float): Input p rate. Defaults to `1.0`.
        zup (bool): Input zup. Defaults to `True`.
        p_offset (np.ndarray): Input p offset. Defaults to `np.array([0, 0, 0])`.
        R_offset (Any): Input R offset. Defaults to `np.eye(3, 3)`.
        plot_chain_graph (bool): Input plot chain graph. Defaults to `False`.
        plot_init_chain (bool): Input plot init chain. Defaults to `False`.
        verbose (bool): Input verbose. Defaults to `False`.
        name (str): Input name. Defaults to `'mocap'`.
        mocap_type (Any | None): Input mocap type. Defaults to `None`.

    Returns:
        Any: Requested value computed from the current inputs.
    """

    # Get skeleton from BVH
    skeleton = get_skeleton_from_bvh(bvh_path)

    # Set preset parameters based on mocap_type
    p_offset = np.array(p_offset)
    R_offset = np.array(R_offset)
    if mocap_type is not None:
        assert mocap_type in ['seg','cmu'], "mocap_type must be 'seg' or 'cmu'"
        if mocap_type == 'seg':
            p_rate   = 0.01
            R_offset = rpy2r(np.deg2rad([0,0,90]))
        elif mocap_type == 'cmu':
            p_rate   = 0.056444

    # Get chains from skeleton
    secs,chains = get_chains_from_skeleton(
        skeleton         = skeleton,
        env              = env,
        rpy_order        = rpy_order,
        p_rate           = p_rate,
        plot_chain_graph = plot_chain_graph,
        plot_init_chain  = plot_init_chain,
        verbose          = verbose,
        name             = name,
    )

    # Convert to Z-up
    if zup:
        chains = get_chains_translated(
            chains,
            T_trans = pr2t(np.array([0,0,0]),rpy2r(np.radians([90,0,0]))),
        )

    # Apply final offsets
    chains = get_chains_translated(
        chains,
        T_trans = pr2t(p_offset,R_offset),
    )

    # Return time stamps and chains
    return secs,chains
