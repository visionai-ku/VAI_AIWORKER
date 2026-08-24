"""
viz.py
"""

import os
import cv2
import tempfile
import numpy as np
import matplotlib.pyplot as plt
try:
    import mediapy as media
except Exception:
    media = None
from termcolor import colored
from IPython.display import Video,display
from PIL import Image

def get_colors(n_color=10, cmap_name='gist_rainbow', alpha=1.0):
    """
    Generate a list of diverse colors using a specified colormap.

    Parameters:
        n_color (int): Input n color. Defaults to `10`.
        cmap_name (str): Input cmap name. Defaults to `'gist_rainbow'`.
        alpha (float): Input alpha. Defaults to `1.0`.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(np.linspace(0, 1, n_color))
    rgba[:, 3] = alpha
    return [tuple(r) for r in rgba]

def depth_to_gray_img(depth_img,max_val=10.0):
    """
    Convert a 1-channel float depth image to a 3-channel uint8 gray image.

    Parameters:
        depth_img (Any): Input depth img.
        max_val (float): Input max val. Defaults to `10.0`.

    Returns:
        Any: Result computed from the current inputs.
    """
    if depth_img is None:
        return None
    depth_clip = np.clip(depth_img,a_min=0.0,a_max=max_val) # float-type
    depth_max = float(depth_clip.max())
    if depth_max <= 0.0:
        return np.zeros((*depth_clip.shape,3),dtype=np.uint8)
    # Normalize depth image
    img = np.tile(255*depth_clip[:,:,np.newaxis]/depth_max,(1,1,3)).astype(np.uint8) # unit8-type
    return img

def imread(
        impath,
        verbose    = True,
        return_np  = True,
        return_pil = False,
        show_image = True,
        fig_w      = 4,
        fig_h      = None,
        uint8      = True,
        out_w      = None,
        out_h      = None,
    ):
    """
    Read an image from a file and convert it to an RGB numpy array.

    Parameters:
        impath (str | Path): Input impath.
        verbose (bool): Input verbose. Defaults to `True`.
        return_np (bool): Input return np. Defaults to `True`.
        return_pil (bool): Input return pil. Defaults to `False`.
        show_image (bool): Input show image. Defaults to `True`.
        fig_w (int): Input fig w. Defaults to `4`.
        fig_h (Any | None): Input fig h. Defaults to `None`.
        uint8 (bool): Input uint8. Defaults to `True`.
        out_w (Any | None): Input out w. Defaults to `None`.
        out_h (Any | None): Input out h. Defaults to `None`.

    Returns:
        Any: Result computed from the current inputs.
    """
    if (not bool(return_np)) and (not bool(return_pil)):
        raise ValueError("At least one of return_np or return_pil must be True.")

    # Load the image using PIL and convert to RGB format
    img_pil = Image.open(impath).convert("RGB") # pil image in RGB format

    # Optionally resize the image
    if (out_w is not None) or (out_h is not None):
        src_w,src_h = img_pil.size

        if out_w is not None:
            out_w = int(out_w)
            if out_w <= 0:
                raise ValueError("out_w must be a positive integer, got %s"%(str(out_w)))

        if out_h is not None:
            out_h = int(out_h)
            if out_h <= 0:
                raise ValueError("out_h must be a positive integer, got %s"%(str(out_h)))

        # Preserve aspect ratio if only one side is specified
        if (out_w is not None) and (out_h is None):
            out_h = max(1,int(round(float(src_h)*float(out_w)/float(src_w))))
        elif (out_w is None) and (out_h is not None):
            out_w = max(1,int(round(float(src_w)*float(out_h)/float(src_h))))

        # PIL compatibility across versions
        try:
            resample = Image.Resampling.BILINEAR
        except AttributeError:
            resample = Image.BILINEAR

        img_pil = img_pil.resize((out_w,out_h),resample=resample)

    img_np = np.array(img_pil) # numpy array in RGB format

    # Convert dtype/range if requested
    if bool(uint8):
        if img_np.dtype != np.uint8:
            x  = img_np.astype(np.float32)
            mx = float(np.max(x)) if x.size > 0 else 0.0
            if mx <= 1.0+1e-6:
                x = x*255.0
            img_np = np.clip(x,0.0,255.0).astype(np.uint8)
    else:
        if img_np.dtype == np.uint8:
            img_np = img_np.astype(np.float32)/255.0
        else:
            x  = img_np.astype(np.float32)
            mx = float(np.max(x)) if x.size > 0 else 0.0
            if mx > 1.0+1e-6:
                x = x/255.0
            img_np = np.clip(x,0.0,1.0).astype(np.float32)

    # Print debug information
    if verbose:
        print("[imread] Loaded image from [%s] with shape %s dtype=%s"%(str(impath),str(img_np.shape),str(img_np.dtype)))

    # Show the image if requested
    if bool(show_image):
        try:
            imshow(img_np,fig_w=fig_w,fig_h=fig_h)
        except Exception as e:
            if verbose:
                print("[imread] warning: show_image failed (%s)"%str(e))

    # Return the image in the requested formats
    if bool(return_np) and bool(return_pil):
        return img_np,img_pil
    elif bool(return_np):
        return img_np
    else:
        return img_pil

def imshow(
        img,
        fig_w         = 4,
        fig_h         = None,
        title         = None,
        title_fs      = 8,
        axis_off      = True,
        cmap          = 'viridis',
        colorbar      = False,
        cbar_fs       = 6,
        caxis         = None,
        dpi           = 120,
        pad_px        = 0,
        title_overlay = True,
        cbar_inset    = True,
    ):
    """
    Display an image using matplotlib with various customization options.

    Parameters:
        img (Any): Input img.
        fig_w (int): Input fig w. Defaults to `4`.
        fig_h (Any | None): Input fig h. Defaults to `None`.
        title (Any | None): Input title. Defaults to `None`.
        title_fs (int): Input title fs. Defaults to `8`.
        axis_off (bool): Input axis off. Defaults to `True`.
        cmap (str): Input cmap. Defaults to `'viridis'`.
        colorbar (bool): Input colorbar. Defaults to `False`.
        cbar_fs (int): Input cbar fs. Defaults to `6`.
        caxis (Any | None): Input caxis. Defaults to `None`.
        dpi (int): Input dpi. Defaults to `120`.
        pad_px (int): Input pad px. Defaults to `0`.
        title_overlay (bool): Input title overlay. Defaults to `True`.
        cbar_inset (bool): Input cbar inset. Defaults to `True`.

    Returns:
        None: This function mutates state, renders output, or performs side
              effects in place.
    """
    if img is None:
        print("[imshow] 'img' is None. Nothing to display.")
        return

    h,w = img.shape[:2]

    if fig_h is None:
        fig_h = float(fig_w)*float(h)/float(w)
    else:
        fig_h = float(fig_h)

    # pixel padding -> normalized padding
    pad_px = int(max(0,pad_px))
    fig    = plt.figure(figsize=(float(fig_w),float(fig_h)),dpi=int(dpi))

    # axes box in figure coords, keeping width fixed
    if pad_px>0:
        fw_px = float(fig.get_figwidth()*fig.dpi)
        fh_px = float(fig.get_figheight()*fig.dpi)
        l     = pad_px/fw_px
        b     = pad_px/fh_px
        r     = 1.0 - pad_px/fw_px
        t     = 1.0 - pad_px/fh_px
        ax    = fig.add_axes([l,b,max(0.0,r-l),max(0.0,t-b)])
    else:
        ax    = fig.add_axes([0.0,0.0,1.0,1.0])

    if img.ndim==2:
        if caxis is not None:
            im = ax.imshow(img,cmap=cmap,vmin=caxis[0],vmax=caxis[1])
        else:
            im = ax.imshow(img,cmap=cmap)
    else:
        im = ax.imshow(img)

    if bool(axis_off):
        ax.set_axis_off()

    # title: overlay so it doesn't affect layout/axes size
    if (title is not None):
        if bool(title_overlay):
            ax.text(
                0.01,0.99,str(title),
                transform=ax.transAxes,
                ha='left',va='top',
                fontsize=int(title_fs),
                color='white',
                bbox=dict(boxstyle='round,pad=0.2',facecolor='black',alpha=0.35,edgecolor='none'),
            )
        else:
            # will not shrink axes here (since axes already fixed), but may be clipped if pad_px=0
            fig.text(0.01,0.99,str(title),ha='left',va='top',fontsize=int(title_fs))

    # colorbar: inset inside axes to keep width unchanged
    if bool(colorbar) and (img.ndim==2):
        if bool(cbar_inset):
            try:
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                cax = inset_axes(
                    ax,
                    width  = "3.5%",
                    height = "45%",
                    loc    = "lower right",
                    borderpad=0.8,
                )
                cbar = fig.colorbar(im,cax=cax)
                cbar.ax.tick_params(labelsize=int(cbar_fs))
            except Exception as e:
                print("[imshow] warning: inset colorbar failed (%s)"%str(e))
        else:
            # this steals width if used normally; included only if you explicitly want it
            cbar = fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
            cbar.ax.tick_params(labelsize=int(cbar_fs))

    plt.show()

def play_frames(frames,fps=30,view_width=None,view_height=None,codec='h264',verbose=True):
    """
    Play a sequence of frames as a video in a Jupyter notebook.

    Parameters:
        frames (Any): Input frames.
        fps (int): Input fps. Defaults to `30`.
        view_width (Any | None): Input view width. Defaults to `None`.
        view_height (Any | None): Input view height. Defaults to `None`.
        codec (str): Input codec. Defaults to `'h264'`.
        verbose (bool): Input verbose. Defaults to `True`.

    Returns:
        None: This function mutates state, renders output, or performs side
              effects in place.
    """
    # Print debug information
    if verbose:
        print(f"[play_frames] Number of frames:[%d] fps:[%d]"%(len(frames),fps))

    # Convert list to np.array if needed
    frames_np = np.asarray(frames)
    if media is None:
        raise ImportError("mediapy is required for play_frames(...).")

    # Create a temporary file for the video
    fd,mp4_path = tempfile.mkstemp(suffix='.mp4')
    os.close(fd) # close the file descriptor

    # Write video to the temporary file
    media.write_video(mp4_path,frames_np,fps=fps,codec=codec)

    # Display the video in the notebook
    display(Video(mp4_path,embed=True,width=view_width,height=view_height))

def print_red(str):
    """
    Print a string in red color.

    Parameters:
        str (Any): Input str.

    Returns:
        None: This function mutates state, renders output, or performs side
              effects in place.
    """
    print (colored(str,'red'))

def print_yellow(str):
    """
    Print a string in yellow color.

    Parameters:
        str (Any): Input str.

    Returns:
        None: This function mutates state, renders output, or performs side
              effects in place.
    """
    print (colored(str,'yellow'))

def print_green(str):
    """
    Print a string in green color.

    Parameters:
        str (Any): Input str.

    Returns:
        None: This function mutates state, renders output, or performs side
              effects in place.
    """
    print (colored(str,'green'))

def print_blue(str):
    """
    Print a string in blue color.

    Parameters:
        str (Any): Input str.

    Returns:
        None: This function mutates state, renders output, or performs side
              effects in place.
    """
    print (colored(str,'blue'))

def print_light_green(str):
    """
    Print a string in light green color.

    Parameters:
        str (Any): Input str.

    Returns:
        None: This function mutates state, renders output, or performs side
              effects in place.
    """
    print (colored(str,'light_green'))

def print_light_blue(str):
    """
    Print a string in light blue color.

    Parameters:
        str (Any): Input str.

    Returns:
        None: This function mutates state, renders output, or performs side
              effects in place.
    """
    print (colored(str,'light_blue'))

def meters2xyz(depth_img,cam_matrix):
    """
    Convert a scaled depth image into a 3D point cloud.

    Parameters:
        depth_img (Any): Input depth img.
        cam_matrix (Any): Input cam matrix.

    Returns:
        Any: Result computed from the current inputs.
    """
    fx = cam_matrix[0][0]
    cx = cam_matrix[0][2]
    fy = cam_matrix[1][1]
    cy = cam_matrix[1][2]

    height = depth_img.shape[0]
    width = depth_img.shape[1]
    indices = np.indices((height, width),dtype=np.float32).transpose(1,2,0)

    z_e = depth_img
    x_e = (indices[..., 1] - cx) * z_e / fx
    y_e = (indices[..., 0] - cy) * z_e / fy

    # Order of y_ e is reversed !
    xyz_img = np.stack([z_e, -x_e, -y_e], axis=-1) # [H x W x 3]
    return xyz_img # [H x W x 3]
