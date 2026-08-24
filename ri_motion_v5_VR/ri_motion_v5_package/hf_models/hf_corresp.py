"""
hf_corresp.py
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from transformers import (
    AutoImageProcessor,
    Dinov2Model,
)

# from hf_common import (
#     ensure_local_ckpt,
#     get_device,
# )
# from hf_viz import (
#     norm_xy,
#     rgb_img_np,
#     viz_make_fig_ax,
#     viz_title,
#     viz_text_bbox,
#     viz_label,
#     viz_close,
# )
from ri_motion_v5_package.hf_models.hf_common import (
    ensure_local_ckpt,
    get_device,
)
from ri_motion_v5_package.hf_models.hf_viz import (
    norm_xy,
    rgb_img_np,
    viz_make_fig_ax,
    viz_title,
    viz_text_bbox,
    viz_label,
    viz_close,
)

def get_fig_h_from_image_shape(fig_w,h,w,override=None):
    """
    Compute figure height from image aspect ratio unless explicitly overridden.

    Parameters:
        fig_w (Any): Input fig w.
        h (Any): Input h.
        w (Any): Input w.
        override (Any | None): Input override. Defaults to `None`.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    if override is None:
        return float(fig_w)*float(h)/max(1.0,float(w))
    return float(override)

class DINOv2Correspondence(object):
    """
    Dense correspondence heatmap via DINOv2 patch tokens.

    Usage:
        obj = DINOv2Correspondence(...)
    """

    def __init__(
            self,
            ckpt              = "facebook/dinov2-base",
            device            = None,
            verbose           = True,
            local_dir         = "./models/dinov2",
            revision          = None,
            img_size          = 448,
            processor_resize  = True,
            processor_crop    = False,
        ):
        """
        Initialize DINOv2Correspondence.

        Parameters:
            ckpt (str): Input ckpt. Defaults to `'facebook/dinov2-base'`.
            device (Any | None): Input device. Defaults to `None`.
            verbose (bool): Input verbose. Defaults to `True`.
            local_dir (str): Input local dir. Defaults to `'./models/dinov2'`.
            revision (Any | None): Input revision. Defaults to `None`.
            img_size (int): Input img size. Defaults to `448`.
            processor_resize (bool): Input processor resize. Defaults to `True`.
            processor_crop (bool): Input processor crop. Defaults to `False`.

        Returns:
            None: Initializes the instance in place.
        """
        self.ckpt    = ensure_local_ckpt(ckpt,local_dir=local_dir,revision=revision,verbose=verbose)
        self.verbose = bool(verbose)
        self.device  = get_device(device=device,verbose=self.verbose)

        self.processor = AutoImageProcessor.from_pretrained(self.ckpt)
        self.model     = Dinov2Model.from_pretrained(self.ckpt).to(self.device)
        self.model.eval()

        self.patch    = int(getattr(self.model.config,"patch_size",14))
        self.img_size = int(img_size)

        try:
            if hasattr(self.processor,"do_resize"):
                self.processor.do_resize = bool(processor_resize)

            if hasattr(self.processor,"size") and bool(processor_resize):
                if isinstance(self.processor.size,dict):
                    if "shortest_edge" in self.processor.size:
                        self.processor.size["shortest_edge"] = int(self.img_size)
                    elif ("height" in self.processor.size) and ("width" in self.processor.size):
                        self.processor.size["height"] = int(self.img_size)
                        self.processor.size["width"]  = int(self.img_size)
                else:
                    self.processor.size = int(self.img_size)

            if hasattr(self.processor,"do_center_crop"):
                self.processor.do_center_crop = bool(processor_crop)
        except Exception:
            pass

        # simple cache for interactive use (same PIL object reused)
        self._feat_cache = {}

        if self.verbose:
            print("[DINOv2Correspondence] ckpt=[%s] device=[%s] patch=%d img_size=%d ready."%
                  (self.ckpt,self.device,self.patch,self.img_size))

    # -------------------------------------------------------------------------
    # Image helpers
    # -------------------------------------------------------------------------
    def _to_pil(self,x):
        """
        Convert an image input to a PIL image.

        Parameters:
            x (Any): Input x.

        Returns:
            Any: Result computed from the current inputs.
        """
        if hasattr(x,"convert"):
            return x.convert("RGB")
        from PIL import Image
        arr = np.asarray(x)
        if arr.ndim == 2:
            arr = np.stack([arr,arr,arr],axis=-1)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    def _resize_square(self,img_pil):
        """
        Resize resize square.

        Parameters:
            img_pil (Any): Input img pil.

        Returns:
            Any: Result computed from the current inputs.
        """
        return img_pil.resize((int(self.img_size),int(self.img_size)))

    def _cache_key(self,img_pil):
        """
        Return cache key.

        Parameters:
            img_pil (Any): Input img pil.

        Returns:
            Any: Result computed from the current inputs.
        """
        try:
            return (id(img_pil),img_pil.size)
        except Exception:
            return id(img_pil)

    # -------------------------------------------------------------------------
    # Feature extraction
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def _feat_grid(self,img_pil):
        # deterministic pre-resize (keeps token grid stable)
        """
        Return feat grid.

        Parameters:
            img_pil (Any): Input img pil.

        Returns:
            Any: Result computed from the current inputs.
        """
        img_pil = self._to_pil(img_pil)
        img_sq  = self._resize_square(img_pil)

        key = self._cache_key(img_sq)
        if key in self._feat_cache:
            return self._feat_cache[key]

        inputs = self.processor(images=img_sq,return_tensors="pt")
        for k in inputs:
            if hasattr(inputs[k],"to"):
                inputs[k] = inputs[k].to(self.device)

        pv = inputs.get("pixel_values",None)
        if pv is None:
            raise RuntimeError("processor did not return pixel_values")
        H_in = int(pv.shape[-2])
        W_in = int(pv.shape[-1])

        out = self.model(**inputs)
        x   = out.last_hidden_state            # (1,1+N,D)
        tok = x[:,1:,:]                        # (1,N,D)

        N   = int(tok.shape[1])
        D   = int(tok.shape[2])

        gh  = int(H_in // self.patch)
        gw  = int(W_in // self.patch)

        if gh*gw != N:
            g = int(round(np.sqrt(float(N))))
            if g*g != N:
                raise RuntimeError("Token/grid mismatch: N=%d,H_in=%d,W_in=%d,patch=%d"%(N,H_in,W_in,self.patch))
            gh = g
            gw = g

        feat = tok.reshape(1,gh,gw,D).permute(0,3,1,2).contiguous()
        feat = F.normalize(feat,dim=1)

        pack = ((H_in,W_in),feat,(gh,gw))
        self._feat_cache[key] = pack
        return pack

    def _xy_to_ij(self,x,y,W_in,H_in,gh,gw):
        """
        Convert xy to ij.

        Parameters:
            x (Any): Input x.
            y (Any): Input y.
            W_in (Any): Input W in.
            H_in (Any): Input H in.
            gh (Any): Input gh.
            gw (Any): Input gw.

        Returns:
            Any: Result computed from the current inputs.
        """
        x = float(x); y = float(y)
        x = max(0.0,min(float(W_in-1),x))
        y = max(0.0,min(float(H_in-1),y))
        j = int(x // int(self.patch))
        i = int(y // int(self.patch))
        i = int(np.clip(i,0,gh-1))
        j = int(np.clip(j,0,gw-1))
        return i,j

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def heatmap(
            self,
            img_src_pil,
            img_trgt_pil,
            xy_src,
            temperature      = 0.07,
            normalize_method = "softmax",
            return_debug     = False,
        ):
        """
        Compute .

        Parameters:
            img_src_pil (Any): Input img src pil.
            img_trgt_pil (Any): Input img trgt pil.
            xy_src (Any): Input xy src.
            temperature (float): Input temperature. Defaults to `0.07`.
            normalize_method (str): Input normalize method. Defaults to `'softmax'`.
            return_debug (bool): Input return debug. Defaults to `False`.

        Returns:
            Any: Result computed from the current inputs.
        """
        img_src_pil  = self._to_pil(img_src_pil)
        img_trgt_pil = self._to_pil(img_trgt_pil)

        WA,HA = img_src_pil.size
        WB,HB = img_trgt_pil.size

        (HAi,WAi),FA,(gha,gwa) = self._feat_grid(img_src_pil)
        (HBi,WBi),FB,(ghb,gwb) = self._feat_grid(img_trgt_pil)

        # canonicalize xy_src but keep as float for mapping
        xy = norm_xy(xy_src)
        xA = float(xy[0]); yA = float(xy[1])

        # IMPORTANT: because we always pre-resize to a square (img_size,img_size),
        # the correct mapping is:
        #   orig -> square(img_size,img_size) -> model_input(H_in,W_in)
        x_sq = xA*(float(self.img_size)/max(1.0,float(WA)))
        y_sq = yA*(float(self.img_size)/max(1.0,float(HA)))

        xAi  = x_sq*(float(WAi)/max(1.0,float(self.img_size)))
        yAi  = y_sq*(float(HAi)/max(1.0,float(self.img_size)))

        ia,ja = self._xy_to_ij(xAi,yAi,WAi,HAi,gha,gwa)

        q = FA[0,:,ia,ja].reshape(1,-1,1,1)
        q = F.normalize(q,dim=1)

        sim = (FB*q).sum(dim=1,keepdim=True)

        nm = str(normalize_method).lower().strip()
        if nm == "softmax":
            t = float(temperature)
            if t <= 0.0:
                t = 0.07
            sim_t = sim/float(t)
            sm    = sim_t.reshape(1,1,ghb*gwb)
            sm    = F.softmax(sm,dim=-1).reshape(1,1,ghb,gwb)
            hm_in = sm
        elif nm == "minmax":
            smin  = float(sim.min().item())
            smax  = float(sim.max().item())
            hm_in = (sim - smin)/max(1e-8,(smax-smin))
        else:
            raise ValueError("normalize_method must be 'softmax' or 'minmax'")

        hm_i = F.interpolate(hm_in,size=(HBi,WBi),mode="bilinear",align_corners=False)
        hm_o = F.interpolate(hm_i ,size=(HB,WB) ,mode="bilinear",align_corners=False)
        hmB  = hm_o[0,0].detach().cpu().numpy().astype(np.float32)

        if return_debug:
            dbg = dict(
                A_orig_hw=(HA,WA),
                B_orig_hw=(HB,WB),
                A_in_hw=(HAi,WAi),
                B_in_hw=(HBi,WBi),
                A_grid=(gha,gwa),
                B_grid=(ghb,gwb),
                ijA=(ia,ja),
                xy_src_sq=(x_sq,y_sq),
                xy_src_in=(xAi,yAi),
                patch=self.patch,
                normalize=nm,
            )
            return hmB,dbg

        return hmB

    @staticmethod
    def plot_heatmap_on_image(
            img_trgt_np,
            hmB,
            alpha   = 0.55,
            fig_w   = 5,
            cmap    = "jet",
            show_cb = True,
        ):
        """
        Plot heatmap on image.

        Parameters:
            img_trgt_np (Any): Input img trgt np.
            hmB (Any): Input hmB.
            alpha (float): Input alpha. Defaults to `0.55`.
            fig_w (int): Input fig w. Defaults to `5`.
            cmap (str): Input cmap. Defaults to `'jet'`.
            show_cb (bool): Input show cb. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        img_trgt = rgb_img_np(img_trgt_np)
        HB,WB    = img_trgt.shape[:2]

        hm = np.asarray(hmB,dtype=np.float32)
        if (hm.shape[0]!=HB) or (hm.shape[1]!=WB):
            ht = torch.from_numpy(hm)[None,None,:,:]
            ht = F.interpolate(ht,size=(HB,WB),mode="bilinear",align_corners=False)
            hm = ht[0,0].numpy()

        fig_h = float(fig_w)*float(HB)/max(1.0,float(WB))
        fig,ax = viz_make_fig_ax(fig_w,fig_h,dpi=120,pad_px=0,w=WB,h=HB)

        ax.imshow(img_trgt)
        im = ax.imshow(hm,alpha=float(alpha),cmap=str(cmap))

        if bool(show_cb):
            # overlay colorbar (doesn't steal width)
            cbar_w = 0.028
            cbar_h = 0.40
            cbar_l = 1.0 - cbar_w - 0.015
            cbar_b = 0.06
            cax    = fig.add_axes([cbar_l,cbar_b,cbar_w,cbar_h])
            vmin   = float(np.nanmin(hm))
            vmax   = float(np.nanmax(hm))
            if not np.isfinite(vmin): vmin = 0.0
            if not np.isfinite(vmax): vmax = 1.0
            ticks  = list(np.linspace(vmin,vmax,5))
            ticks.append(0.0)
            ticks  = np.array(sorted(set([float(t) for t in ticks])))
            cbar   = fig.colorbar(im,cax=cax,ticks=ticks,format="%.2f")
            cbar.ax.tick_params(labelsize=6)
            cax.set_facecolor((0,0,0,0))

        plt.show()
        viz_close(fig)

    @staticmethod
    def peak_xy(hmB):
        """
        Return xy.

        Parameters:
            hmB (Any): Input hmB.

        Returns:
            Any: Result computed from the current inputs.
        """
        hm = np.asarray(hmB)
        y,x = np.unravel_index(int(np.argmax(hm)),hm.shape)
        return [int(x),int(y)],float(hm[y,x])

    @staticmethod
    def plot_corresp(
            img_src_np,
            img_trgt_np,
            xy_src,
            hm_trgt,
            fig_w         = 6,
            fig_h_src     = None,
            fig_h_trgt    = None,
            dpi           = 120,
            pad_px        = 0,
            alpha         = 0.55,
            cmap          = "jet",
            show_cb       = False,
            cbar_fs       = 6,
            cbar_inset    = True,
            s             = 120,
            lw            = 4,
            edge_c        = "r",
            txt_dx        = 5,
            txt_dy        = 5,
            txt_fs        = 9,
            title_src     = None,
            title_trgt    = None,
            title_hm      = None,
            title_fs      = 8,
            title_overlay = True,
        ):
        """
        Plot corresp.

        Parameters:
            img_src_np (Any): Input img src np.
            img_trgt_np (Any): Input img trgt np.
            xy_src (Any): Input xy src.
            hm_trgt (Any): Input hm trgt.
            fig_w (int): Input fig w. Defaults to `6`.
            fig_h_src (Any | None): Input fig h src. Defaults to `None`.
            fig_h_trgt (Any | None): Input fig h trgt. Defaults to `None`.
            dpi (int): Input dpi. Defaults to `120`.
            pad_px (int): Input pad px. Defaults to `0`.
            alpha (float): Input alpha. Defaults to `0.55`.
            cmap (str): Input cmap. Defaults to `'jet'`.
            show_cb (bool): Input show cb. Defaults to `False`.
            cbar_fs (int): Input cbar fs. Defaults to `6`.
            cbar_inset (bool): Input cbar inset. Defaults to `True`.
            s (int): Input s. Defaults to `120`.
            lw (int): Input lw. Defaults to `4`.
            edge_c (str): Input edge c. Defaults to `'r'`.
            txt_dx (int): Input txt dx. Defaults to `5`.
            txt_dy (int): Input txt dy. Defaults to `5`.
            txt_fs (int): Input txt fs. Defaults to `9`.
            title_src (Any | None): Input title src. Defaults to `None`.
            title_trgt (Any | None): Input title trgt. Defaults to `None`.
            title_hm (Any | None): Input title hm. Defaults to `None`.
            title_fs (int): Input title fs. Defaults to `8`.
            title_overlay (bool): Input title overlay. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        src  = rgb_img_np(img_src_np)
        trgt = rgb_img_np(img_trgt_np)

        HS,WS = src.shape[:2]
        HT,WT = trgt.shape[:2]

        hm = np.asarray(hm_trgt,dtype=np.float32)
        if (hm.shape[0]!=HT) or (hm.shape[1]!=WT):
            ht = torch.from_numpy(hm)[None,None,:,:]
            ht = F.interpolate(ht,size=(HT,WT),mode="bilinear",align_corners=False)
            hm = ht[0,0].detach().cpu().numpy().astype(np.float32)

        xy = norm_xy(xy_src)
        xA = float(xy[0]); yA = float(xy[1])

        bbox_txt = viz_text_bbox(facecolor="white",alpha=0.7,pad=0.05,edgecolor="none")

        # (1) src + point
        h1 = get_fig_h_from_image_shape(fig_w,HS,WS,fig_h_src)
        fig,ax = viz_make_fig_ax(fig_w,h1,dpi,pad_px,WS,HS)
        ax.imshow(src)
        ax.scatter([xA],[yA],s=float(s),lw=float(lw),marker="o",facecolors="none",edgecolors=str(edge_c))
        viz_label(ax,xA,yA,"(%d,%d)"%(int(xA),int(yA)),txt_dx,txt_dy,txt_fs,bbox_txt,clip_on=True)
        viz_title(fig,ax,title_src,title_fs,title_overlay)
        plt.show()
        viz_close(fig)

        # (2) trgt
        h2 = get_fig_h_from_image_shape(fig_w,HT,WT,fig_h_trgt)
        fig,ax = viz_make_fig_ax(fig_w,h2,dpi,pad_px,WT,HT)
        ax.imshow(trgt)
        viz_title(fig,ax,title_trgt,title_fs,title_overlay)
        plt.show()
        viz_close(fig)

        # (3) trgt + heatmap
        fig,ax = viz_make_fig_ax(fig_w,h2,dpi,pad_px,WT,HT)
        ax.imshow(trgt)
        im = ax.imshow(hm,alpha=float(alpha),cmap=str(cmap))
        viz_title(fig,ax,title_hm,title_fs,title_overlay)

        if bool(show_cb):
            # overlay colorbar so it doesn't shrink the main axes
            cbar_w = 0.028
            cbar_h = 0.40
            cbar_l = 1.0 - cbar_w - 0.015
            cbar_b = 0.06
            cax    = fig.add_axes([cbar_l,cbar_b,cbar_w,cbar_h])

            vmin = float(np.nanmin(hm))
            vmax = float(np.nanmax(hm))
            if not np.isfinite(vmin): vmin = 0.0
            if not np.isfinite(vmax): vmax = 1.0

            ticks = list(np.linspace(vmin,vmax,5))
            ticks.append(0.0)
            ticks = np.array(sorted(set([float(t) for t in ticks])))

            cbar = fig.colorbar(im,cax=cax,ticks=ticks,format="%.2f")
            cbar.ax.tick_params(labelsize=int(cbar_fs))
            cax.set_facecolor((0,0,0,0))

        plt.show()
        viz_close(fig)

# -------------------------------------------------------------------------
