import numpy as np
import warnings

try:
    import leap # type: ignore
    LEAP_AVAILABLE = True
    LEAP_IMPORT_ERROR = None
    _LeapListenerBase = leap.Listener
except Exception as e:
    LEAP_AVAILABLE = False
    LEAP_IMPORT_ERROR = e

    class _LeapTrackingModeStub:
        """
        Provide fallback tracking-mode names when Leap is unavailable.

        Usage:
            Used internally by this module.
        """
        Desktop = "Desktop"
        HMD = "HMD"
        ScreenTop = "ScreenTop"

    class _LeapStub:
        """
        Provide a fallback Leap namespace when Leap is unavailable.

        Usage:
            Used internally by this module.
        """
        TrackingMode = _LeapTrackingModeStub

        class LeapCannotOpenDeviceError(Exception):
            """
            Represent failure to open a Leap device.

            Usage:
                Used internally by this module.
            """
            pass

    class _LeapListenerBase(object):
        """
        Provide a no-op listener base when Leap is unavailable.

        Usage:
            Used internally by this module.
        """
        pass

    leap = _LeapStub()

# """
# %run ../../package/init_utility.py
# """
# from utils import ( # type: ignore
#     hex_to_rgba,
# )
# from filters import ( # type: ignore
#     VectorizedKalmanFilters,
# )
# """
# %run ../../package/init_kinematics.py
# """
# from transforms import ( # type: ignore
#     np_uv,
#     pr2t,
# )
from ri_motion_v5_package.utility.utils import (
    hex_to_rgba,
)
from ri_motion_v5_package.utility.filters import (
    VectorizedKalmanFiltersFast,
)
from ri_motion_v5_package.kinematics.transforms import (
    np_uv,
    pr2t,
)

class NullLeapConnection(object):
    """
    No-op connection used when Leap SDK or hardware is unavailable.

    Usage:
        obj = NullLeapConnection(...)
    """
    def __init__(self,status_msg=""):
        """
        Initialize NullLeapConnection.

        Parameters:
            status_msg (str): Input status msg. Defaults to `''`.

        Returns:
            None: Initializes the instance in place.
        """
        self.status_msg = status_msg

    def add_listener(self,listener):
        """
        Add listener.

        Parameters:
            listener (Any): Input listener.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        return None

    def set_tracking_mode(self,tracking_mode):
        """
        Set tracking mode.

        Parameters:
            tracking_mode (Any): Input tracking mode.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        return False

    def open(self):
        """
        Return the no-op connection context.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        return self

    def __enter__(self):
        """
        Enter the context manager.

        Parameters:
            None.

        Returns:
            Any: Result computed from the current inputs.
        """
        return self

    def __exit__(self,exc_type,exc_value,traceback):
        """
        Exit the context manager.

        Parameters:
            exc_type (Any): Input exc type.
            exc_value (Any): Input exc value.
            traceback (Any): Input traceback.

        Returns:
            Any: Result computed from the current inputs.
        """
        return False


class LeapMotion2Handler(_LeapListenerBase):
    """
    LeapMotion2Handler is a class that handles Leap Motion data processing.

    Usage:
        obj = LeapMotion2Handler(...)
    """
    def __init__(self,required=False,verbose=True):
        """
        Initialize LeapMotion2Handler.

        Parameters:
            required (bool): Input required. Defaults to `False`.
            verbose (bool): Input verbose. Defaults to `True`.

        Returns:
            None: Initializes the instance in place.
        """
        super().__init__()
        self.available  = False
        self.status_msg = ""
        self.verbose    = verbose
        self.device_detected = False
        # Tracking modes
        self.TRACKING_MODES = {
            leap.TrackingMode.Desktop: "Desktop",
            leap.TrackingMode.HMD: "HMD",
            leap.TrackingMode.ScreenTop: "ScreenTop",
        }
        # Placeholders
        self.bones          = {'right':None,'left':None}
        self.bones_kf       = {'right':None,'left':None}
        self.visible_times  = {'right':0.0,'left':0.0}
        self.kfs            = {'right':None,'left':None}
        self.scale          = (1.0,1.0,1.0) # positional scaling factor
        self.offset         = (0.0,0.0,0.0) # positional offset

        # Rotation offset to align Leap Motion data (with MuJoCo coordinate system)
        self.R_offset = np.array([
            [ 0, -1,  0],
            [ 0,  0,  1],
            [-1,  0,  0]
        ])

        # Finger colors for visualization (using 'hex_to_rgba')
        self.finger_colors = {
            'right': {
                'Thumb':  '#FF4C4C',  # bright red
                'Index':  '#E03C3C',
                'Middle': '#C02C2C',
                'Ring':   '#A01C1C',
                'Little': '#801010',  # dark red
            },
            'left': {
                'Thumb':  '#4C6CFF',  # bright blue
                'Index':  '#3C5CDB',
                'Middle': '#2C4CB7',
                'Ring':   '#1C3C93',
                'Little': '#102C70',  # dark blue
            }
        }
        # Reset placeholders
        self.reset_visible_times()
        # Set connection
        self.init_connection(required=required)

    def mark_unavailable(self,status_msg,required=False):
        """
        Store a non-fatal unavailable hardware state.

        Parameters:
            status_msg (Any): Input status msg.
            required (bool): Input required. Defaults to `False`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.available  = False
        self.status_msg = status_msg
        self.connection = NullLeapConnection(status_msg=status_msg)
        self.device_detected = False

        if required:
            raise RuntimeError(status_msg)
        if self.verbose:
            print(status_msg)

    def is_available(self):
        """
        Return whether Leap SDK connection was created.

        Parameters:
            None.

        Returns:
            bool: Boolean result for the requested condition.
        """
        return bool(self.available)

    def is_device_detected(self):
        """
        Return whether a Leap device event has been received.

        Parameters:
            None.

        Returns:
            bool: Boolean result for the requested condition.
        """
        return bool(self.device_detected)

    def get_status_msg(self):
        """
        Return the current hardware or SDK status message.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if self.available:
            if not self.device_detected:
                return "[Leap] connection available, but no Leap Motion device event has been received yet."
            return "[Leap] connection available."
        return self.status_msg

    def reset_visible_times(self):
        """
        Reset visible_times for hand data.

        Parameters:
            None.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        for hand_type in ['right','left']:
            self.visible_times[hand_type] = 0.0

    def init_connection(self,required=False):
        """
        Initialize Leap Motion connection.

        Parameters:
            required (bool): Input required. Defaults to `False`.

        Returns:
            Any: Result computed from the current inputs.
        """
        if not LEAP_AVAILABLE:
            self.mark_unavailable(
                "[Leap] Python SDK import failed: [%s]. "
                "Running without Leap Motion hardware."%(LEAP_IMPORT_ERROR),
                required = required,
            )
            return False

        try:
            self.connection = leap.Connection()
            self.connection.add_listener(self)
            self.available = True
            return True
        except Exception as e:
            self.mark_unavailable(
                "[Leap] connection initialization failed: [%s]. "
                "Running without Leap Motion hardware."%(e),
                required = required,
            )
            return False

    def set_kalman_filter(self,dt= 1./120.,q=0.01,r=5.0):
        """
        Set Kalman filter.

        Parameters:
            dt (float): Input dt. Defaults to `1.0 / 120.0`.
            q (float): Input q. Defaults to `0.01`.
            r (float): Input r. Defaults to `5.0`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        for hand_type in ['right','left']:
            self.kfs[hand_type] = VectorizedKalmanFiltersFast(shape=(5,4,2,3),dt=dt,q=q,r=r)

    def set_tracking_mode(self, tracking_mode, verbose=True):
        """
        Set the tracking mode for Leap Motion.

        Parameters:
            tracking_mode (Any): Input tracking mode.
            verbose (bool): Input verbose. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.tracking_mode = tracking_mode
        if not self.available:
            if verbose:
                print(self.get_status_msg())
            return False
        self.connection.set_tracking_mode(self.tracking_mode)
        if verbose:
            print("[Leap] set tracking mode to: [%s]"%(self.TRACKING_MODES[self.tracking_mode]))
        return True

    def set_scale(self, scale=(1.0, 1.0, 1.0)):
        """
        Set the positional scaling factor for Leap Motion data.

        Parameters:
            scale (tuple): Input scale. Defaults to `(1.0, 1.0, 1.0)`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.scale = scale

    def set_offset(self, offset=(0.0, 0.0, 0.0)):
        """
        Set the positional offset for Leap Motion data.

        Parameters:
            offset (tuple): Input offset. Defaults to `(0.0, 0.0, 0.0)`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.offset = offset

    def on_device_event(self, event):
        """
        Callback for device connection events.

        Parameters:
            event (Any): Input event.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        try:
            with event.device.open():
                info = event.device.get_info()
        except leap.LeapCannotOpenDeviceError:
            info = event.device.get_info()
        self.device_detected = True
        print("[Leap] device found: [%s]" % (info.serial))

    def on_tracking_mode_event(self, event):
        """
        Callback for tracking mode change events.

        Parameters:
            event (Any): Input event.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.set_tracking_mode(event.current_tracking_mode)
        print("[Leap] current tracking mode: [%s]"%(self.TRACKING_MODES[event.current_tracking_mode]))

    def on_tracking_event(self, event):
        """
        Callback for tracking events.

        Parameters:
            event (Any): Input event.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.latest_event = event # store the latest event
        self.parse_event(event)

    def parse_event(self, event):
        """
        Parse the Leap Motion event to extract hand and bone data.

        Parameters:
            event (Any): Input event.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        self.event = event
        # Reset visible times
        self.reset_visible_times()

        # Parse event
        for hand in self.event.hands: # for each hand
            # Skip if hand is invalid
            try: hand_type = hand.type.name.lower()  # 'right' or 'left'
            except Exception as e:
                print("[Leap] skipping current hand due to invalid type:[%s]"%(e))
                continue

            if hand_type=='right' or hand_type=='left': # if right or left hand is detected
                self.visible_times[hand_type] = hand.visible_time/1e6 # time in seconds
                self.bones[hand_type] = self.parse_hand(hand) # (5,4,2,3)
                # Apply scaling
                p_ref = self.get_p_from_bones(
                    bones       = self.bones[hand_type],
                    finger_name = 'Middle',
                    joint_name  = 'CMC',
                    ) # (3,)
                p_ref_scale = p_ref * np.array(self.scale) # (3,)
                p_offset    = p_ref_scale - p_ref # (3,)
                self.bones[hand_type] = self.bones[hand_type] + p_offset # (5,4,2,3)
                # Apply offset
                self.bones[hand_type] = self.bones[hand_type] + np.array(self.offset) # (5,4,2,3)
                # Run Kalman filter
                if self.kfs[hand_type] is None: # if not filter, just copy the bones
                    self.bones_kf[hand_type] = self.bones[hand_type].copy() # (5,4,2,3)
                else:
                    kf = self.kfs[hand_type] # Kalman filter object
                    self.bones_kf[hand_type] = kf.update(self.bones[hand_type]) # (5,4,2,3)
            else:
                print ("[Leap] unrecognized hand type:[%s]"%(hand_type))

    @staticmethod
    def xyz(joint):
        """
        Convert a Leap Motion joint object to a numpy array.

        Parameters:
            joint (Any): Input joint.

        Returns:
            Any: Result computed from the current inputs.
        """
        return np.array([joint.x,joint.y,joint.z],dtype=np.float32)

    def parse_hand(self,hand):
        """
        Parse a Leap Motion hand object to extract bone positions.

        Parameters:
            hand (Any): Input hand.

        Returns:
            Any: Result computed from the current inputs.
        """
        bones = np.zeros((5,4,2,3)) # 5 fingers, 4 bones per finger, 2 joints per bone, 3D
        for f_idx in range(5): # for each finger
            for b_idx in range(4): # for each bone
                p_prev = self.xyz(hand.digits[f_idx].bones[b_idx].prev_joint)
                p_next = self.xyz(hand.digits[f_idx].bones[b_idx].next_joint)
                bones[f_idx,b_idx,0,:] = p_prev*0.001 # [m]
                bones[f_idx,b_idx,1,:] = p_next*0.001 # [m]
        # Transform 'bones'
        bones = bones@self.R_offset # (5,4,2,3)
        # Return
        return bones # (5,4,2,3)

    def get_bones(
            self,
            hand_type   = 'right',
            return_mode = 'raw',
        ):
        """
        Get the bone positions of the specified hand.

        Parameters:
            hand_type (str): Input hand type. Defaults to `'right'`.
            return_mode (str): Input return mode. Defaults to `'raw'`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        hand_type = hand_type.lower() # ensure case-insensitive matching
        if return_mode == 'raw':
            return None if self.bones[hand_type] is None else self.bones[hand_type].copy()
        elif return_mode == 'kf':
            return None if self.bones_kf[hand_type] is None else self.bones_kf[hand_type].copy()
        elif return_mode == 'both':
            raw = None if self.bones[hand_type]    is None else self.bones[hand_type].copy()
            kf  = None if self.bones_kf[hand_type] is None else self.bones_kf[hand_type].copy()
            return raw,kf
        else:
            raise ValueError("Invalid return_mode:[%s]" %(return_mode))

    def get_visible_time(self, hand_type='right'):
        """
        Get the visible time of the specified hand.

        Parameters:
            hand_type (str): Input hand type. Defaults to `'right'`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        return self.visible_times[hand_type]

    def is_detected(self, hand_type='right', time_margin=0.0):
        """
        Check if the specified hand is detected based on visible time.

        Parameters:
            hand_type (str): Input hand type. Defaults to `'right'`.
            time_margin (float): Input time margin. Defaults to `0.0`.

        Returns:
            bool: Boolean result for the requested condition.
        """
        return self.visible_times[hand_type] > time_margin

    @staticmethod
    def get_p_from_bones(bones, finger_name='Index', joint_name='MCP'):
        """
        Get the position of a specific joint from the bone positions.

        Parameters:
            bones (Any): Input bones.
            finger_name (str): Input finger name. Defaults to `'Index'`.
            joint_name (str): Input joint name. Defaults to `'MCP'`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        finger_to_i = {'Thumb': 0, 'Index': 1, 'Middle': 2, 'Ring': 3, 'Little': 4}
        i = finger_to_i[finger_name]

        if finger_name == 'Thumb':
            joint_to_jk = {
                'CMC': (0, 0),
                'MCP': (1, 1),
                'IP' : (2, 1),
                'Tip': (3, 1),
            }
        else:
            joint_to_jk = {
                'CMC': (0, 0),
                'MCP': (0, 1),
                'PIP': (1, 1),
                'DIP': (2, 1),
                'Tip': (3, 1),
            }
        j, k = joint_to_jk[joint_name]
        return bones[i, j, k, :]  # shape: (3,)

    def get_p(self, hand_type='right', finger_name='Index', joint_name='MCP', use_kf=False):
        """
        Get the position of a specific joint for the specified hand.

        Parameters:
            hand_type (str): Input hand type. Defaults to `'right'`.
            finger_name (str): Input finger name. Defaults to `'Index'`.
            joint_name (str): Input joint name. Defaults to `'MCP'`.
            use_kf (bool): Input use kf. Defaults to `False`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        bones = self.get_bones(
            hand_type,
            'kf' if use_kf else 'raw'
        ) # (5,4,2,3)
        p = self.get_p_from_bones(bones, finger_name, joint_name) # (3,)
        return p # (3,)

    def get_T_wrist_from_bones(self, bones, hand_type='right'):
        """
        Get the transformation matrix of the wrist from the bone positions.

        Parameters:
            bones (Any): Input bones.
            hand_type (str): Input hand type. Defaults to `'right'`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        # Extract the bone pairs for the right wrist
        p_top    = self.get_p_from_bones(bones, 'Middle', 'MCP')
        p_bottom = self.get_p_from_bones(bones, 'Middle', 'CMC')
        p_left   = self.get_p_from_bones(bones, 'Index', 'CMC')
        p_right  = self.get_p_from_bones(bones, 'Ring', 'CMC')
        # Compute the postion
        p_wrist = 0.5*(p_right + p_left)
        # Compute the rotation matrix
        if hand_type == 'right':
            ux = np_uv(p_bottom - p_top) # (3,) top to bottom vector
            n  = np.cross(p_right - p_top, p_left - p_top)
            n  = n / np.linalg.norm(n)
            v  = np.cross(n, ux)
            uz = v / np.linalg.norm(v)
            uy = np.cross(uz, ux)
        elif hand_type == 'left':
            ux = np_uv(p_top - p_bottom) # (3,) bottom to top vector
            n  = np.cross(p_left - p_top, p_right - p_top)
            n  = n / np.linalg.norm(n)
            v  = np.cross(n, ux)
            uz = v / np.linalg.norm(v)
            uy = np.cross(uz, ux)
        R_wrist = np.stack([ux, uy, uz], axis=-1)
        # Create the transformation matrix
        T_wrist = pr2t(p=p_wrist, R=R_wrist)
        # Return
        return T_wrist  # (4 x 4)

    def get_T_wrist(self, hand_type='right', use_kf=True):
        """
        Get the transformation matrix of the wrist for the specified hand.

        Parameters:
            hand_type (str): Input hand type. Defaults to `'right'`.
            use_kf (bool): Input use kf. Defaults to `True`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        # Extract the bone pairs for the right wrist
        bones = self.get_bones(
            hand_type,
            'kf' if use_kf else 'raw'
        ) # (5,4,2,3)
        T_wrist = self.get_T_wrist_from_bones(bones, hand_type) # (4 x 4)

        # Return
        return T_wrist  # (4 x 4)

    @staticmethod
    def get_finger_names():
        """
        Get the names of the fingers in a structured format.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        return ['Thumb', 'Index', 'Middle', 'Ring', 'Little']

    @staticmethod
    def get_finger_joint_names():
        """
        Get the names of the fingers and joints in a structured format.

        Parameters:
            None.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        names = []
        # Thumb
        thumb_joints = ['CMC', 'MCP', 'IP', 'Tip']
        names += [['Thumb', jt] for jt in thumb_joints]
        # Other fingers
        other_fingers = ['Index', 'Middle', 'Ring', 'Little']
        other_joints  = ['CMC', 'MCP', 'PIP', 'DIP', 'Tip']
        for finger in other_fingers:
            names += [[finger, jt] for jt in other_joints]
        return names

    @staticmethod
    def get_rgba_by_hand_type(hand_type='right', data_type='kf', alpha=None, alpha_raw=0.1, alpha_kf=0.7):
        """
        Get RGBA color based on hand type and data type.

        Parameters:
            hand_type (str): Input hand type. Defaults to `'right'`.
            data_type (str): Input data type. Defaults to `'kf'`.
            alpha (Any | None): Input alpha. Defaults to `None`.
            alpha_raw (float): Input alpha raw. Defaults to `0.1`.
            alpha_kf (float): Input alpha kf. Defaults to `0.7`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if alpha is None:
            if data_type == 'raw':
                alpha = alpha_raw
            elif data_type == 'kf':
                alpha = alpha_kf
            else:
                raise ValueError(f"Invalid data_type: {data_type}")

        if hand_type == 'right':
            return (1, 0, 0, alpha)
        elif hand_type == 'left':
            return (0, 0, 1, alpha)
        else:
            raise ValueError(f"Invalid hand_type: {hand_type}")

    @staticmethod
    def _angle_between(v1,v2,eps=1e-8):
        """
        Compute the angle between two 3D vectors in radians.

        Parameters:
            v1 (Any): Input v1.
            v2 (Any): Input v2.
            eps (float): Input eps. Defaults to `1e-08`.

        Returns:
            Any: Result computed from the current inputs.
        """
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if (n1 < eps) or (n2 < eps):
            return 0.0

        c = np.dot(v1,v2)/(n1*n2)
        c = np.clip(c,-1.0,1.0)
        return float(np.arccos(c))

    @staticmethod
    def _finger_name_to_index(finger_name):
        """
        Convert finger name to finger index.

        Parameters:
            finger_name (str): Input finger name.

        Returns:
            Any: Result computed from the current inputs.
        """
        finger_to_i = {
            'thumb' : 0,
            'index' : 1,
            'middle': 2,
            'ring'  : 3,
            'little': 4,
        }
        finger_name_l = finger_name.lower() # convert to lowercase for case-insensitive matching
        if finger_name_l not in finger_to_i:
            raise ValueError("Invalid finger_name:[%s]"%(finger_name))
        return finger_to_i[finger_name_l]

    def get_flexion_ratio(
            self,
            hand_type    = 'right',
            finger_name  = None,
            use_kf       = True,
            clip         = True,
            return_debug = False,
        ):
        """
        Get flexion ratio in [0,1] from Leap hand bones.

        Parameters:
            hand_type (str): Input hand type. Defaults to `'right'`.
            finger_name (Any | None): Input finger name. Defaults to `None`.
            use_kf (bool): Input use kf. Defaults to `True`.
            clip (bool): Input clip. Defaults to `True`.
            return_debug (bool): Input return debug. Defaults to `False`.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        if hand_type not in ['right','left']:
            raise ValueError("Invalid hand_type:[%s]"%(hand_type))

        if not self.is_detected(hand_type=hand_type,time_margin=0.0):
            warnings.warn(
                "[get_flexion_ratio] Hand [%s] is not currently detected."
                %(hand_type)
            )

        bones = self.get_bones(
            hand_type   = hand_type,
            return_mode = 'kf' if use_kf else 'raw',
        )

        if bones is None:
            raise ValueError(
                "[get_flexion_ratio] No bones available for hand_type=[%s],use_kf=[%s]."
                %(hand_type,use_kf)
            )

        finger_names = ['Thumb','Index','Middle','Ring','Little']
        finger_ratios = {}
        finger_angles = {}

        # Normalize by the sum of three inter-bone angles.
        # Straight finger -> near 0
        # Strongly curled finger -> closer to 1
        # 3*pi/2 is a practical soft upper bound (three 90-degree bends).
        max_angle_sum = 1.5*np.pi

        for fname in finger_names: # for each finger
            f_idx = self._finger_name_to_index(fname)

            v0 = bones[f_idx,0,1,:] - bones[f_idx,0,0,:]
            v1 = bones[f_idx,1,1,:] - bones[f_idx,1,0,:]
            v2 = bones[f_idx,2,1,:] - bones[f_idx,2,0,:]
            v3 = bones[f_idx,3,1,:] - bones[f_idx,3,0,:]

            a01 = self._angle_between(v0,v1)
            a12 = self._angle_between(v1,v2)
            a23 = self._angle_between(v2,v3)

            angle_sum = a01 + a12 + a23
            ratio     = angle_sum/max_angle_sum

            if clip:
                ratio = float(np.clip(ratio,0.0,1.0))
            else:
                ratio = float(ratio)

            finger_ratios[fname] = ratio
            finger_angles[fname] = {
                'a01'      : a01,
                'a12'      : a12,
                'a23'      : a23,
                'angle_sum': angle_sum,
            }

        if finger_name is None:
            ratio = float(np.mean(list(finger_ratios.values())))
        else:
            if finger_name not in finger_ratios:
                raise ValueError("Invalid finger_name:[%s]"%(finger_name))
            ratio = finger_ratios[finger_name]

        if return_debug:
            debug = {
                'hand_type'     : hand_type,
                'use_kf'        : use_kf,
                'finger_ratios' : finger_ratios,
                'finger_angles' : finger_angles,
                'mean_ratio'    : float(np.mean(list(finger_ratios.values()))),
            }
            return ratio,debug
        else:
            return ratio

    def plot_bones_in_mujoco(
            self,
            env,
            bones,
            hand_type    = 'right',
            use_kf       = True,
            r_sphere     = 0.005,
            r_cylinder   = 0.0015,
            rgba         = None,
            alpha        = None,
            print_names  = False,
            colorful     = False,
        ):
        """
        Plot the bones in the MuJoCo environment.

        Parameters:
            env (Any): Input env.
            bones (Any): Input bones.
            hand_type (str): Input hand type. Defaults to `'right'`.
            use_kf (bool): Input use kf. Defaults to `True`.
            r_sphere (float): Input r sphere. Defaults to `0.005`.
            r_cylinder (float): Input r cylinder. Defaults to `0.0015`.
            rgba (Any | None): Input rgba. Defaults to `None`.
            alpha (Any | None): Input alpha. Defaults to `None`.
            print_names (bool): Input print names. Defaults to `False`.
            colorful (bool): Input colorful. Defaults to `False`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']

        if rgba is None:
            if use_kf: alpha_use = 0.7
            else:      alpha_use = 0.1
            rgba = self.get_rgba_by_hand_type(
                hand_type = hand_type,
                data_type = 'kf',
                alpha     = alpha_use,
            )

        for f_idx in range(5): # for each finger
            finger_name = finger_names[f_idx]
            if colorful:
                finger_color = self.finger_colors[hand_type][finger_name]
                rgba = hex_to_rgba(finger_color,alpha=rgba[3])
            if isinstance(rgba, tuple): rgba = list(rgba)  # ensure rgba is a list
            if alpha is not None: rgba[3] = alpha # set the alpha channel

            for b_idx in range(4): # for each bone
                p_fr = bones[f_idx,b_idx,0,:]
                p_to = bones[f_idx,b_idx,1,:]
                # Draw the bone segment
                env.plot_sphere(p_fr, r_sphere, rgba, '')
                env.plot_sphere(p_to, r_sphere, rgba, '')
                env.plot_cylinder_fr2to(p_fr, p_to, r_cylinder, rgba)
                if print_names:
                    label = '[%d]-[%d]'%(f_idx,b_idx)
                else:
                    label = ''
                env.plot_sphere(0.5*(p_fr+p_to), r_cylinder, rgba, label)


