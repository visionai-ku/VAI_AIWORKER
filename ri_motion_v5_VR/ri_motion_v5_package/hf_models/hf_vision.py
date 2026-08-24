"""
hf_vision.py
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.functional as F

from PIL import Image
from scipy.ndimage import binary_fill_holes,label

from transformers import (
    Sam2Processor,
    Sam2Model,
    Owlv2Processor,
    Owlv2ForObjectDetection,
    AutoImageProcessor,
    AutoModelForDepthEstimation,
)
from diffusers import (
    DiffusionPipeline,
)
from huggingface_hub import snapshot_download

# from hf_common import (
#     ensure_local_ckpt,
#     get_device,
# )
# from hf_viz import (
#     to_int_xy,
#     norm_points_xy_list,
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
    to_int_xy,
    norm_points_xy_list,
    rgb_img_np,
    viz_make_fig_ax,
    viz_title,
    viz_text_bbox,
    viz_label,
    viz_close,
)


class PointSam2(object):
    """
    Point-prompted segmentation using SAM2.

    Usage:
        obj = PointSam2(...)
    """
    def __init__(
            self,
            ckpt      = "facebook/sam2.1-hiera-base-plus",
            device    = None,
            verbose   = True,
            local_dir = "./models/sam2",
            revision  = None,
        ):
        """
        Initialize PointSam2.

        Parameters:
            ckpt (str): Input ckpt. Defaults to `'facebook/sam2.1-hiera-base-plus'`.
            device (Any | None): Input device. Defaults to `None`.
            verbose (bool): Input verbose. Defaults to `True`.
            local_dir (str): Input local dir. Defaults to `'./models/sam2'`.
            revision (Any | None): Input revision. Defaults to `None`.

        Returns:
            None: Initializes the instance in place.
        """
        # SAM2 checkpoints can be large, so we ensure they are downloaded and cached locally.
        self.ckpt = ensure_local_ckpt(
            ckpt,
            local_dir = local_dir,
            revision  = revision,
            verbose   = verbose,
            )
        # Initialize the processor and model from the checkpoint.
        self.processor = Sam2Processor.from_pretrained(self.ckpt)
        self.verbose   = bool(verbose)
        self.device    = get_device(device=device,verbose=self.verbose)
        self.model     = Sam2Model.from_pretrained(self.ckpt).to(self.device)
        self.model.eval()
        if self.verbose:
            print("[PointSam2] ckpt=[%s] ready."%self.ckpt)

    def _keep_largest_cc(self,mask):
        """
        Keep only the largest connected component in a binary mask.

        Parameters:
            mask (Any): Input mask.

        Returns:
            Any: Result computed from the current inputs.
        """
        mask      = np.asarray(mask).astype(bool)
        structure = np.ones((3,3),dtype=np.int8)
        lab,n     = label(mask,structure=structure) # lab: (H,W), n: number of CCs
        if int(n)==0:
            return mask
        sizes     = np.bincount(lab.ravel())
        if sizes.size>0:
            sizes[0]=0
        k         = int(np.argmax(sizes)) if sizes.size>0 else 0
        return (lab==k)

    def _select_candidate_mask(
            self,
            cand_masks,
            pos,
            neg,
            prefer_pos_mask = False,
        ):
        """
        Select one candidate mask from SAM2 outputs.

        Parameters:
            cand_masks (Any): Input cand masks.
            pos (Any): Input pos.
            neg (Any): Input neg.
            prefer_pos_mask (bool): Input prefer pos mask. Defaults to `False`.

        Returns:
            Any: Result computed from the current inputs.
        """
        cand_masks = np.asarray(cand_masks).astype(bool)
        if cand_masks.ndim != 3:
            raise ValueError("cand_masks must have shape (K,H,W).")

        if not bool(prefer_pos_mask):
            cand_i = int(np.argmax(cand_masks.reshape(cand_masks.shape[0],-1).sum(axis=1)))
            return cand_masks[cand_i]

        h,w = cand_masks.shape[1:]
        pos = norm_points_xy_list(pos)
        neg = norm_points_xy_list(neg)
        best_i = 0
        best_key = None

        for i,mask in enumerate(cand_masks):
            pos_hit = 0
            neg_hit = 0
            for x,y in pos:
                if (0 <= x < w) and (0 <= y < h) and bool(mask[y,x]):
                    pos_hit += 1
            for x,y in neg:
                if (0 <= x < w) and (0 <= y < h) and bool(mask[y,x]):
                    neg_hit += 1
            area = int(mask.sum())
            key = (pos_hit,-neg_hit,-area)
            if (best_key is None) or (key > best_key):
                best_key = key
                best_i = i
        return cand_masks[best_i]

    def predict_mask(
            self,
            img_pil,
            pos          = None,
            neg          = None,
            fill_holes   = True,
            keep_largest = True,
            prefer_pos_mask = False,
        ):
        """
        Predict a segmentation mask for the given image based on positive and negative
        points.

        Parameters:
            img_pil (Any): Input img pil.
            pos (Any | None): Input pos. Defaults to `None`.
            neg (Any | None): Input neg. Defaults to `None`.
            fill_holes (bool): Input fill holes. Defaults to `True`.
            keep_largest (bool): Input keep largest. Defaults to `True`.
            prefer_pos_mask (bool): Input prefer pos mask. Defaults to `False`.

        Returns:
            Any: Result computed from the current inputs.
        """
        if pos is None: pos = []
        if neg is None: neg = []
        if (len(pos)==0) and (len(neg)==0):
            raise ValueError("At least one of pos or neg points must be provided.")

        # unify point format: list of [x,y] ints (CRITICAL for Sam2Processor nesting validation)
        pos = norm_points_xy_list(pos)
        neg = norm_points_xy_list(neg)

        # algorithmic core unchanged
        pts          = pos + neg # SAM2 expects all points together, with separate labels for pos/neg
        labels       = [1]*len(pos) + [0]*len(neg)
        input_points = [[pts]]      # [image][object][point][xy]  (4-level, xy is list)
        input_labels = [[labels]]   # [image][object][point]

        # SAM2 processor will handle the rest (including moving tensors to device), and we just need to post-process the output masks.
        inputs = self.processor(
            images         = img_pil,
            input_points   = input_points,
            input_labels   = input_labels,
            return_tensors = "pt",
        )

        # Move inputs to the same device as the model (if they have a .to() method), and run the model to get the raw outputs.
        for k in inputs:
            if hasattr(inputs[k],"to"):
                inputs[k]=inputs[k].to(self.device)

        # Post-process the raw outputs to get the predicted masks, and apply optional hole filling and largest component filtering.
        with torch.no_grad():
            outputs = self.model(**inputs)

        # SAM2's post-processing will give us the predicted masks in the original image size, and we can convert them to boolean numpy arrays for further processing.
        masks      = self.processor.post_process_masks(outputs.pred_masks.cpu(),inputs["original_sizes"])[0]
        cand_masks = masks[0].numpy().astype(bool)  # (3,H,W)
        mask       = self._select_candidate_mask(
            cand_masks       = cand_masks,
            pos              = pos,
            neg              = neg,
            prefer_pos_mask  = prefer_pos_mask,
        )

        # Optionally fill holes and keep only the largest connected component in the predicted mask, and return the final binary mask as a numpy array.
        if bool(fill_holes):
            mask = binary_fill_holes(mask)

        # SAM2 can sometimes produce multiple disconnected components, so we can optionally keep only the largest one to get a cleaner segmentation mask.
        if bool(keep_largest):
            mask = self._keep_largest_cc(mask)

        # Return the final binary mask as a numpy array of type bool, where True indicates the segmented region and False indicates the background.
        return np.asarray(mask).astype(bool)

    def run(
            self,
            img_pil,
            pos          = None,
            neg          = None,
            fill_holes   = True,
            keep_largest = True,
            prefer_pos_mask = False,
        ):
        """
        Run the segmentation mask prediction for the given image and points.

        Parameters:
            img_pil (Any): Input img pil.
            pos (Any | None): Input pos. Defaults to `None`.
            neg (Any | None): Input neg. Defaults to `None`.
            fill_holes (bool): Input fill holes. Defaults to `True`.
            keep_largest (bool): Input keep largest. Defaults to `True`.
            prefer_pos_mask (bool): Input prefer pos mask. Defaults to `False`.

        Returns:
            Any: Result computed from the current inputs.
        """

        # unify point format: list of [x,y] ints (CRITICAL for Sam2Processor nesting validation)
        pos_n = norm_points_xy_list(pos)
        neg_n = norm_points_xy_list(neg)

        # The core prediction logic is unchanged, but we just need to post-process the output masks to get the final binary mask, and optionally fill holes and keep only the largest connected component for cleaner results.
        mask  = self.predict_mask(
            img_pil,
            pos_n,
            neg_n,
            fill_holes   = fill_holes,
            keep_largest = keep_largest,
            prefer_pos_mask = prefer_pos_mask,
        )
        if self.verbose:
            print("[PointSam2] Predicted segmentation mask of size [%d,%d] with [%d] positive and [%d] negative points."%
                  (mask.shape[0],mask.shape[1],len(pos_n),len(neg_n)))
        return mask

    @staticmethod
    def plot_segmask(
            img_np,
            mask          = None,
            pos           = None,
            neg           = None,
            alpha         = 0.5,
            bright        = 0.5,
            mask_color    = (255,0,0),
            fig_w         = 4,
            dpi           = 120,
            pad_px        = 0,
            marker_size   = 60,
            text_offset   = (5,5),
            text_fs       = 8,
            title_fs      = 8,
            title_overlay = True,
            cmap_mask     = "gray",
        ):
        """
        Visualize the segmentation mask along with the original image and
        positive/negative points.

        Parameters:
            img_np (Any): Input img np.
            mask (Any | None): Input mask. Defaults to `None`.
            pos (Any | None): Input pos. Defaults to `None`.
            neg (Any | None): Input neg. Defaults to `None`.
            alpha (float): Input alpha. Defaults to `0.5`.
            bright (float): Input bright. Defaults to `0.5`.
            mask_color (tuple): Input mask color. Defaults to `(255, 0, 0)`.
            fig_w (int): Input fig w. Defaults to `4`.
            dpi (int): Input dpi. Defaults to `120`.
            pad_px (int): Input pad px. Defaults to `0`.
            marker_size (int): Input marker size. Defaults to `60`.
            text_offset (tuple): Input text offset. Defaults to `(5, 5)`.
            text_fs (int): Input text fs. Defaults to `8`.
            title_fs (int): Input title fs. Defaults to `8`.
            title_overlay (bool): Input title overlay. Defaults to `True`.
            cmap_mask (str): Input cmap mask. Defaults to `'gray'`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        img_np = np.asarray(img_np)
        h,w    = img_np.shape[:2]
        fig_h  = float(fig_w)*float(h)/max(1.0,float(w))

        pos = norm_points_xy_list(pos)
        neg = norm_points_xy_list(neg)

        overlay = None
        if mask is not None:
            m    = np.asarray(mask).astype(bool)
            base = rgb_img_np(img_np).astype(np.float32).copy()
            a    = float(alpha);  a=max(0.0,min(1.0,a))
            b    = float(bright); b=max(0.0,min(1.0,b))
            base = base*(1.0-b)+255.0*b
            mc   = np.array(mask_color,dtype=np.float32).reshape((1,1,3))
            base[m] = a*base[m]+(1.0-a)*mc
            overlay = np.clip(base,0,255).astype(np.uint8)

        bbox_xy = viz_text_bbox(facecolor="white",alpha=0.7,pad=0.05,edgecolor="none")
        dx,dy   = float(text_offset[0]),float(text_offset[1])

        # (1) original + pos/neg
        fig,ax = viz_make_fig_ax(fig_w,fig_h,dpi,pad_px,w,h)
        if img_np.ndim==2:
            ax.imshow(img_np,cmap="viridis")
        else:
            ax.imshow(img_np)

        text_offset = [10,0]  # (dx,dy) for point labels to avoid overlap with markers
        for p in pos:
            x,y = int(p[0]),int(p[1])
            ax.scatter([x],[y],s=float(marker_size),marker="o",facecolors="none",edgecolors="r")
            viz_label(ax,x+text_offset[0],y+text_offset[1],"(%d,%d)"%(x,y),dx,dy,text_fs,bbox_xy,clip_on=True)
        for p in neg:
            x,y = int(p[0]),int(p[1])
            ax.scatter([x],[y],s=float(marker_size),marker="x",color="b")
            viz_label(ax,x+text_offset[0],y+text_offset[1],"(%d,%d)"%(x,y),dx,dy,text_fs,bbox_xy,clip_on=True)

        viz_title(fig,ax,"original + pos/neg",title_fs,title_overlay)
        plt.show()
        viz_close(fig)

        # (2) mask
        fig,ax = viz_make_fig_ax(fig_w,fig_h,dpi,pad_px,w,h)
        if mask is None:
            ax.imshow(np.zeros((h,w),dtype=np.uint8),cmap=str(cmap_mask),vmin=0,vmax=255)
            viz_title(fig,ax,"mask (None)",title_fs,title_overlay)
        else:
            mm = np.asarray(mask).astype(np.uint8)*255
            ax.imshow(mm,cmap=str(cmap_mask),vmin=0,vmax=255)
            viz_title(fig,ax,"mask",title_fs,title_overlay)
        plt.show()
        viz_close(fig)

        # (3) overlay
        fig,ax = viz_make_fig_ax(fig_w,fig_h,dpi,pad_px,w,h)
        if overlay is None:
            if img_np.ndim==2:
                ax.imshow(img_np,cmap="viridis")
            else:
                ax.imshow(img_np)
            viz_title(fig,ax,"overlay (None)",title_fs,title_overlay)
        else:
            ax.imshow(overlay)
            viz_title(fig,ax,"overlay",title_fs,title_overlay)
        plt.show()
        viz_close(fig)

class PromptOwlv2(object):
    """
    Text-prompted object detection using OWLv2.

    Usage:
        obj = PromptOwlv2(...)
    """
    def __init__(
            self,
            ckpt      = "google/owlv2-base-patch16",
            device    = None,
            verbose   = True,
            local_dir = "./models/owlv2",
            revision  = None,
        ):
        """
        Initialize PromptOwlv2.

        Parameters:
            ckpt (str): Input ckpt. Defaults to `'google/owlv2-base-patch16'`.
            device (Any | None): Input device. Defaults to `None`.
            verbose (bool): Input verbose. Defaults to `True`.
            local_dir (str): Input local dir. Defaults to `'./models/owlv2'`.
            revision (Any | None): Input revision. Defaults to `None`.

        Returns:
            None: Initializes the instance in place.
        """
        self.ckpt      = ensure_local_ckpt(ckpt,local_dir=local_dir,revision=revision,verbose=verbose)
        self.processor = Owlv2Processor.from_pretrained(self.ckpt)
        self.verbose   = bool(verbose)
        self.device    = get_device(device=device,verbose=self.verbose)
        self.model     = Owlv2ForObjectDetection.from_pretrained(self.ckpt).to(self.device)
        self.model.eval()
        if self.verbose:
            print("[PromptOwlv2] ckpt=[%s] ready."%self.ckpt)

    def predict_boxes(
            self,
            img_pil,
            text_queries,
            threshold=0.1,
        ):
        """
        Predict bounding boxes for the given image and text queries.

        Parameters:
            img_pil (Any): Input img pil.
            text_queries (Any): Input text queries.
            threshold (float): Input threshold. Defaults to `0.1`.

        Returns:
            Any: Result computed from the current inputs.
        """
        if text_queries is None:
            raise ValueError("text_queries must be provided.")
        if isinstance(text_queries,str):
            text_queries=[text_queries]
        if len(text_queries)==0:
            raise ValueError("text_queries must be non-empty.")

        inputs = self.processor(
            images         = img_pil,
            text           = text_queries,
            return_tensors = "pt",
        )
        for k in inputs:
            if hasattr(inputs[k],"to"):
                inputs[k]=inputs[k].to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        h,w          = img_pil.size[1],img_pil.size[0]
        target_sizes = torch.tensor([[h,w]],dtype=torch.int64)

        if hasattr(self.processor,"post_process_object_detection"):
            results = self.processor.post_process_object_detection(
                outputs,
                threshold    = float(threshold),
                target_sizes = target_sizes,
            )[0]
        elif hasattr(self.processor,"image_processor") and hasattr(self.processor.image_processor,"post_process_object_detection"):
            results = self.processor.image_processor.post_process_object_detection(
                outputs,
                threshold    = float(threshold),
                target_sizes = target_sizes,
            )[0]
        else:
            raise AttributeError("No post_process_object_detection found in Owlv2Processor/image_processor. Update transformers.")

        # convert to numpy arrays on CPU for easier downstream processing (and to detach from computation graph just in case)
        boxes  = results["boxes"].detach().cpu().numpy().astype(np.float32)
        scores = results["scores"].detach().cpu().numpy().astype(np.float32)
        labels = results["labels"].detach().cpu().numpy().astype(np.int64)

        # Return boxes, scores, and labels as numpy arrays.
        return boxes,scores,labels

    def plot_dets(
            self,
            img_np,
            boxes,
            scores,
            labels,
            text_queries  = None,
            fig_w         = 4,
            show_scores   = True,
            score_fmt     = "%.2f",
            linewidth     = 2,
            edgecolor     = "r",
            text_offset   = (2,2),
            dpi           = 120,
            pad_px        = 0,
            title         = None,
            title_fs      = 8,
            title_overlay = True,
            cmap          = "viridis",
            clip_text     = False,
            bbox_fc       = "white",
            bbox_alpha    = 0.7,
        ):
        """
        Visualize the detected bounding boxes on the image.

        Parameters:
            img_np (Any): Input img np.
            boxes (Any): Input boxes.
            scores (Any): Input scores.
            labels (Any): Input labels.
            text_queries (Any | None): Input text queries. Defaults to `None`.
            fig_w (int): Input fig w. Defaults to `4`.
            show_scores (bool): Input show scores. Defaults to `True`.
            score_fmt (str): Input score fmt. Defaults to `'%.2f'`.
            linewidth (int): Input linewidth. Defaults to `2`.
            edgecolor (str): Input edgecolor. Defaults to `'r'`.
            text_offset (tuple): Input text offset. Defaults to `(2, 2)`.
            dpi (int): Input dpi. Defaults to `120`.
            pad_px (int): Input pad px. Defaults to `0`.
            title (Any | None): Input title. Defaults to `None`.
            title_fs (int): Input title fs. Defaults to `8`.
            title_overlay (bool): Input title overlay. Defaults to `True`.
            cmap (str): Input cmap. Defaults to `'viridis'`.
            clip_text (bool): Input clip text. Defaults to `False`.
            bbox_fc (str): Input bbox fc. Defaults to `'white'`.
            bbox_alpha (float): Input bbox alpha. Defaults to `0.7`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        img_np = np.asarray(img_np)
        h,w    = img_np.shape[:2]
        fig_h  = float(fig_w)*float(h)/max(1.0,float(w))

        fig,ax = viz_make_fig_ax(fig_w,fig_h,dpi,pad_px,w,h)

        if img_np.ndim==2:
            ax.imshow(img_np,cmap=str(cmap))
        else:
            ax.imshow(img_np)

        n = int(boxes.shape[0]) if boxes is not None else 0

        bbox_txt = viz_text_bbox(facecolor=str(bbox_fc),alpha=float(bbox_alpha),pad=0.05,edgecolor="none")
        dx,dy    = float(text_offset[0]),float(text_offset[1])

        for i in range(n): # iterate over detected boxes
            x1,y1,x2,y2 = boxes[i].tolist()
            x1=float(x1); y1=float(y1); x2=float(x2); y2=float(y2)

            # Draw the bounding box as a rectangle patch on the axis.
            rect = plt.Rectangle(
                (x1,y1),
                x2-x1,
                y2-y1,
                fill      = False,
                linewidth = float(linewidth),
                edgecolor = str(edgecolor),
            )
            ax.add_patch(rect)

            # Determine the label name for the detected box based on the class label and the provided text queries, and format the label text to include the confidence score if show_scores is True.
            lab = int(labels[i]) if labels is not None else -1
            if (text_queries is not None) and (0<=lab<len(text_queries)):
                name = text_queries[lab]
            else:
                name = "q%d"%lab

            # Format the label text to include the confidence score if show_scores is True and scores are available, otherwise just use the name.
            if bool(show_scores) and (scores is not None):
                s   = float(scores[i])
                txt = "%s %s"%(name,(score_fmt%s))
            else:
                txt = "%s"%name

            # Add the label text near the top-left corner of the bounding box, with an optional background for better visibility, and optionally clip the text to the image boundaries if clip_text is True.
            fs = 8 # default font size for labels, can be adjusted as needed
            viz_label(ax,x1,y1,txt,dx,dy,fs,bbox_txt,clip_on=bool(clip_text))

        # Add the title to the visualization using the viz_title helper function, and display the plot.
        viz_title(fig,ax,title,title_fs,title_overlay)
        plt.show()
        viz_close(fig)

    def run(
            self,
            img_pil,
            img_np,
            text_queries,
            threshold = 0.1,
            plot      = False,
            fig_w     = 4,
        ):
        """
        Run the object detection and optionally plot the results.

        Parameters:
            img_pil (Any): Input img pil.
            img_np (Any): Input img np.
            text_queries (Any): Input text queries.
            threshold (float): Input threshold. Defaults to `0.1`.
            plot (bool): Input plot. Defaults to `False`.
            fig_w (int): Input fig w. Defaults to `4`.

        Returns:
            Any: Result computed from the current inputs.
        """
        # Run the box prediction using the predict_boxes method.
        boxes,scores,labels = self.predict_boxes(
            img_pil,
            text_queries,
            threshold=threshold,
        )
        if self.verbose:
            qn = len(text_queries) if not isinstance(text_queries,str) else 1
            print("[PromptOwlv2] det:%d, threshold:%.3f, queries:%d"%(boxes.shape[0],float(threshold),qn))

        if plot:
            tq = text_queries if not isinstance(text_queries,str) else [text_queries]
            self.plot_dets(img_np,boxes,scores,labels,text_queries=tq,fig_w=fig_w)

        return boxes,scores,labels

class PromptOwlv2Sam2(object):
    """
    Text-based segmentation using:.

    Usage:
        obj = PromptOwlv2Sam2(...)
    """
    def __init__(self,owl,sam2,verbose=True):
        """
        Initialize PromptOwlv2Sam2.

        Parameters:
            owl (Any): Input owl.
            sam2 (Any): Input sam2.
            verbose (bool): Input verbose. Defaults to `True`.

        Returns:
            None: Initializes the instance in place.
        """
        self.owl     = owl
        self.sam2    = sam2
        self.verbose = bool(verbose)

    @staticmethod
    def _clip_xy(x,y,w,h):
        """
        Clip a point to image bounds.

        Parameters:
            x (Any): Input x.
            y (Any): Input y.
            w (Any): Input w.
            h (Any): Input h.

        Returns:
            Any: Result computed from the current inputs.
        """
        x = int(np.clip(int(x),0,w-1))
        y = int(np.clip(int(y),0,h-1))
        return [x,y]  # canonical

    def _box_to_points(self,box,w,h,neg_margin=10):
        """
        Convert a bounding box to positive and negative points for segmentation.

        Parameters:
            box (Any): Input box.
            w (Any): Input w.
            h (Any): Input h.
            neg_margin (int): Input neg margin. Defaults to `10`.

        Returns:
            Any: Result computed from the current inputs.
        """
        x1,y1,x2,y2 = [float(v) for v in box]
        cx          = int(round(0.5*(x1+x2)))
        cy          = int(round(0.5*(y1+y2)))
        pos         = [self._clip_xy(cx,cy,w,h)]

        nm  = int(neg_margin)
        neg = [
            self._clip_xy(int(round(x1-nm)),int(round(y1-nm)),w,h),
            self._clip_xy(int(round(x2+nm)),int(round(y1-nm)),w,h),
            self._clip_xy(int(round(x1-nm)),int(round(y2+nm)),w,h),
            self._clip_xy(int(round(x2+nm)),int(round(y2+nm)),w,h),
        ]
        return pos,neg

    def _overlay_mask(self,img_np,mask,alpha=0.5,bright=0.35,mask_color=(255,0,0)):
        """
        Overlay a segmentation mask on the image.

        Parameters:
            img_np (Any): Input img np.
            mask (Any): Input mask.
            alpha (float): Input alpha. Defaults to `0.5`.
            bright (float): Input bright. Defaults to `0.35`.
            mask_color (tuple): Input mask color. Defaults to `(255, 0, 0)`.

        Returns:
            Any: Result computed from the current inputs.
        """
        if mask is None:
            return img_np

        m = np.asarray(mask).astype(bool)

        if np.asarray(img_np).ndim==2:
            out = np.stack([img_np,img_np,img_np],axis=-1).astype(np.float32)
        else:
            out = np.asarray(img_np).astype(np.float32).copy()

        alpha  = float(alpha);  alpha=max(0.0,min(1.0,alpha))
        bright = float(bright); bright=max(0.0,min(1.0,bright))

        out[~m] = out[~m]*(1.0-bright)+255.0*bright
        mc      = np.array(mask_color,dtype=np.float32).reshape((1,1,3))
        out[m]  = alpha*out[m]+(1.0-alpha)*mc

        return np.clip(out,0,255).astype(np.uint8)

    def plot_det_seg(
            self,
            img_np,
            results,
            fig_w          = 4,
            show_scores    = True,
            score_fmt      = "%.2f",
            linewidth      = 2,
            edgecolor      = "r",
            text_offset    = (2,2),
            overlay_alpha  = 0.5,
            overlay_bright = 0.35,
            mask_color     = (255,0,0),
            dpi            = 120,
            pad_px         = 0,
            cmap           = "viridis",
            clip_text      = False,
        ):
        """
        Visualize the detected bounding boxes and segmentation masks on the image.

        Parameters:
            img_np (Any): Input img np.
            results (Any): Input results.
            fig_w (int): Input fig w. Defaults to `4`.
            show_scores (bool): Input show scores. Defaults to `True`.
            score_fmt (str): Input score fmt. Defaults to `'%.2f'`.
            linewidth (int): Input linewidth. Defaults to `2`.
            edgecolor (str): Input edgecolor. Defaults to `'r'`.
            text_offset (tuple): Input text offset. Defaults to `(2, 2)`.
            overlay_alpha (float): Input overlay alpha. Defaults to `0.5`.
            overlay_bright (float): Input overlay bright. Defaults to `0.35`.
            mask_color (tuple): Input mask color. Defaults to `(255, 0, 0)`.
            dpi (int): Input dpi. Defaults to `120`.
            pad_px (int): Input pad px. Defaults to `0`.
            cmap (str): Input cmap. Defaults to `'viridis'`.
            clip_text (bool): Input clip text. Defaults to `False`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        img_np = np.asarray(img_np)
        h,w    = img_np.shape[:2]
        fig_h  = float(fig_w)*float(h)/max(1.0,float(w))

        n = int(len(results)) if results is not None else 0

        bbox_txt = viz_text_bbox(facecolor="white",alpha=0.7,pad=0.05,edgecolor="none")
        dx,dy    = float(text_offset[0]),float(text_offset[1])

        if n==0:
            fig,ax = viz_make_fig_ax(fig_w,fig_h,dpi,pad_px,w,h)
            if img_np.ndim==2:
                ax.imshow(img_np,cmap=str(cmap))
            else:
                ax.imshow(img_np)
            plt.show()
            viz_close(fig)
            return

        for i in range(n):
            r       = results[i]
            fig,ax  = viz_make_fig_ax(fig_w,fig_h,dpi,pad_px,w,h)

            alpha   = float(r.get("alpha",overlay_alpha))
            bright  = float(r.get("bright",overlay_bright))
            mcolor  = r.get("mask_color",mask_color)

            if r.get("mask",None) is None:
                vis = img_np
            else:
                vis = self._overlay_mask(img_np,r["mask"],alpha=alpha,bright=bright,mask_color=mcolor)

            if np.asarray(vis).ndim==2:
                ax.imshow(vis,cmap=str(cmap))
            else:
                ax.imshow(vis)

            x1,y1,x2,y2 = [float(v) for v in r["box"]]
            rect = plt.Rectangle(
                (x1,y1),
                x2-x1,
                y2-y1,
                fill      = False,
                linewidth = float(linewidth),
                edgecolor = str(edgecolor),
            )
            ax.add_patch(rect)

            name = str(r.get("query","q%d"%int(r.get("label",-1))))
            if bool(show_scores):
                s   = float(r.get("score",0.0))
                txt = "%s %s"%(name,(score_fmt%s))
            else:
                txt = "%s"%name

            viz_label(ax,x1,y1,txt,dx,dy,9,bbox_txt,clip_on=bool(clip_text))

            plt.show()
            viz_close(fig)

    def run(
            self,
            img_pil,
            text_queries,
            det_threshold  = 0.1,
            topk_per_query = 1,
            neg_margin     = 10,
            fill_holes     = True,
            keep_largest   = True,
            overlay_alpha  = 0.5,
        ):
        """
        Run the combined OWLv2 detection and SAM2 segmentation for the given image and
        text queries.

        Parameters:
            img_pil (Any): Input img pil.
            text_queries (Any): Input text queries.
            det_threshold (float): Input det threshold. Defaults to `0.1`.
            topk_per_query (int): Input topk per query. Defaults to `1`.
            neg_margin (int): Input neg margin. Defaults to `10`.
            fill_holes (bool): Input fill holes. Defaults to `True`.
            keep_largest (bool): Input keep largest. Defaults to `True`.
            overlay_alpha (float): Input overlay alpha. Defaults to `0.5`.

        Returns:
            Any: Result computed from the current inputs.
        """
        if isinstance(text_queries,str):
            text_queries=[text_queries]
        if (text_queries is None) or (len(text_queries)==0):
            raise ValueError("text_queries must be non-empty.")

        W,H = img_pil.size

        boxes,scores,labels = self.owl.predict_boxes(
            img_pil,
            text_queries,
            threshold=det_threshold,
        )

        results = []
        for qi in range(len(text_queries)):
            idxs = np.where(labels==qi)[0]
            if idxs.size==0:
                continue

            idxs = idxs[np.argsort(scores[idxs])[::-1]]
            idxs = idxs[:int(topk_per_query)]

            for j in idxs.tolist():
                box   = boxes[j].astype(np.float32)
                score = float(scores[j])

                pos,neg = self._box_to_points(box,W,H,neg_margin=neg_margin)

                # unify inter-class point format: list of [x,y] ints
                pos = norm_points_xy_list(pos)
                neg = norm_points_xy_list(neg)

                mask = self.sam2.predict_mask(
                    img_pil,
                    pos          = pos,
                    neg          = neg,
                    fill_holes   = fill_holes,
                    keep_largest = keep_largest,
                )

                results.append(dict(
                    query = text_queries[qi],
                    box   = box,
                    score = score,
                    label = int(qi),
                    pos   = pos,
                    neg   = neg,
                    mask  = np.asarray(mask).astype(bool),
                    alpha = float(overlay_alpha),
                ))

        if self.verbose:
            print("[PromptOwlv2Sam2] queries:%d, det_threshold:%.3f, results:%d"%
                  (len(text_queries),float(det_threshold),len(results)))

        return results

class DepthAnythingV2(object):
    """
    Monocular depth estimation using Depth Anything V2.

    Usage:
        obj = DepthAnythingV2(...)
    """

    def __init__(
            self,
            ckpt      = "depth-anything/Depth-Anything-V2-Small-hf",
            device    = None,
            verbose   = True,
            local_dir = "./models/depth_anything_v2",
            revision  = None,
        ):
        """
        Initialize DepthAnythingV2.

        Parameters:
            ckpt (str): Input ckpt. Defaults to `'depth-anything/Depth-
                        Anything-V2-Small-hf'`.
            device (Any | None): Input device. Defaults to `None`.
            verbose (bool): Input verbose. Defaults to `True`.
            local_dir (str): Input local dir. Defaults to
                             `'./models/depth_anything_v2'`.
            revision (Any | None): Input revision. Defaults to `None`.

        Returns:
            None: Initializes the instance in place.
        """
        self.ckpt = ensure_local_ckpt(
            ckpt,
            local_dir = local_dir,
            revision  = revision,
            verbose   = verbose,
        )
        self.verbose   = bool(verbose)
        self.device    = get_device(device=device,verbose=self.verbose)
        self.processor = AutoImageProcessor.from_pretrained(self.ckpt)
        self.model     = AutoModelForDepthEstimation.from_pretrained(self.ckpt).to(self.device)
        self.model.eval()

        if self.verbose:
            print("[DepthAnythingV2] ckpt=[%s] device=[%s] ready."%(self.ckpt,self.device))

    def _to_pil(self,x):
        """
        Convert input to PIL RGB image.

        Parameters:
            x (Any): Input x.

        Returns:
            Any: Result computed from the current inputs.
        """
        if hasattr(x,"convert"):
            return x.convert("RGB")
        arr = np.asarray(x)
        if arr.ndim==2:
            arr = np.stack([arr,arr,arr],axis=-1)
        if arr.dtype!=np.uint8:
            arr = arr.astype(np.float32)
            if arr.max()<=1.0+1e-6:
                arr = arr*255.0
            arr = np.clip(arr,0,255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    @torch.no_grad()
    def predict_depth(self,img_pil):
        """
        Predict depth map from image.

        Parameters:
            img_pil (Any): Input img pil.

        Returns:
            Any: Result computed from the current inputs.
        """
        img_pil = self._to_pil(img_pil)
        W,H     = img_pil.size

        inputs = self.processor(images=img_pil,return_tensors="pt")
        for k in inputs:
            if hasattr(inputs[k],"to"):
                inputs[k] = inputs[k].to(self.device)

        outputs = self.model(**inputs)
        pred    = outputs.predicted_depth  # (1,H',W')

        pred = F.interpolate(
            pred.unsqueeze(1),
            size=(H,W),
            mode="bilinear",
            align_corners=False,
        )
        depth = pred[0,0].cpu().numpy().astype(np.float32)

        # Normalize to [0, 1]
        d_min = float(np.min(depth))
        d_max = float(np.max(depth))
        if d_max-d_min>1e-8:
            depth = (depth-d_min)/(d_max-d_min)
        else:
            depth = np.zeros_like(depth)

        # Model outputs disparity (high = near), flip to depth (high = far)
        depth = 1.0 - depth

        return depth

    def run(self,img_pil):
        """
        Run depth estimation (alias for predict_depth).

        Parameters:
            img_pil (Any): Input img pil.

        Returns:
            Any: Result computed from the current inputs.
        """
        depth = self.predict_depth(img_pil)
        if self.verbose:
            print("[DepthAnythingV2] Predicted depth map of size [%d,%d]"%
                  (depth.shape[0],depth.shape[1]))
        return depth

    @staticmethod
    def plot_depth(
            img_np,
            depth_map,
            fig_w         = 5,
            dpi           = 120,
            pad_px        = 0,
            cmap          = "inferno",
            show_cb       = True,
            cbar_fs       = 7,
            title_img     = "Input",
            title_depth   = "Depth",
            title_fs      = 9,
            title_overlay = True,
        ):
        """
        Plot original image and depth map separately.

        Parameters:
            img_np (Any): Input img np.
            depth_map (Any): Input depth map.
            fig_w (int): Input fig w. Defaults to `5`.
            dpi (int): Input dpi. Defaults to `120`.
            pad_px (int): Input pad px. Defaults to `0`.
            cmap (str): Input cmap. Defaults to `'inferno'`.
            show_cb (bool): Input show cb. Defaults to `True`.
            cbar_fs (int): Input cbar fs. Defaults to `7`.
            title_img (str): Input title img. Defaults to `'Input'`.
            title_depth (str): Input title depth. Defaults to `'Depth'`.
            title_fs (int): Input title fs. Defaults to `9`.
            title_overlay (bool): Input title overlay. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        img   = rgb_img_np(img_np)
        H,W   = img.shape[:2]
        fig_h = float(fig_w)*float(H)/max(1.0,float(W))

        # (1) Original image
        fig,ax = viz_make_fig_ax(fig_w,fig_h,dpi,pad_px,W,H)
        ax.imshow(img)
        viz_title(fig,ax,title_img,title_fs,title_overlay)
        plt.show()
        viz_close(fig)

        # (2) Depth map
        fig,ax = viz_make_fig_ax(fig_w,fig_h,dpi,pad_px,W,H)
        im     = ax.imshow(depth_map,cmap=str(cmap),vmin=0,vmax=1)
        viz_title(fig,ax,title_depth,title_fs,title_overlay)

        if bool(show_cb):
            cbar_w = 0.028
            cbar_h = 0.40
            cbar_l = 1.0-cbar_w-0.015
            cbar_b = 0.06
            cax    = fig.add_axes([cbar_l,cbar_b,cbar_w,cbar_h])
            ticks  = [0.0,0.25,0.5,0.75,1.0]
            cbar   = fig.colorbar(im,cax=cax,ticks=ticks,format="%.2f")
            cbar.ax.tick_params(labelsize=int(cbar_fs))
            cbar.set_label("near ← → far",fontsize=int(cbar_fs))
            cax.set_facecolor((0,0,0,0))

        plt.show()
        viz_close(fig)

    @staticmethod
    def plot_comparison(
            img_np,
            depth_map,
            fig_w       = 10,
            dpi         = 120,
            cmap        = "inferno",
            title_img   = "Input",
            title_depth = "Depth",
            title_fs    = 10,
        ):
        """
        Plot original image and depth map side by side.

        Parameters:
            img_np (Any): Input img np.
            depth_map (Any): Input depth map.
            fig_w (int): Input fig w. Defaults to `10`.
            dpi (int): Input dpi. Defaults to `120`.
            cmap (str): Input cmap. Defaults to `'inferno'`.
            title_img (str): Input title img. Defaults to `'Input'`.
            title_depth (str): Input title depth. Defaults to `'Depth'`.
            title_fs (int): Input title fs. Defaults to `10`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        img   = rgb_img_np(img_np)
        H,W   = img.shape[:2]
        fig_h = float(fig_w)*0.5*float(H)/max(1.0,float(W))

        fig,axes = plt.subplots(1,2,figsize=(float(fig_w),fig_h),dpi=int(dpi))

        axes[0].imshow(img)
        axes[0].set_title(str(title_img),fontsize=int(title_fs))
        axes[0].axis("off")

        im = axes[1].imshow(depth_map,cmap=str(cmap),vmin=0,vmax=1)
        axes[1].set_title(str(title_depth),fontsize=int(title_fs))
        axes[1].axis("off")

        cbar = fig.colorbar(im,ax=axes[1],fraction=0.046,pad=0.04)
        cbar.set_label("near ← → far",fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        plt.tight_layout()
        plt.show()
        viz_close(fig)

    @staticmethod
    def depth_to_pointcloud(
            depth_map,
            img_np    = None,
            fx        = None,
            fy        = None,
            cx        = None,
            cy        = None,
            max_depth = 10.0,
        ):
        """
        Convert depth map to 3D point cloud.

        Parameters:
            depth_map (Any): Input depth map.
            img_np (Any | None): Input img np. Defaults to `None`.
            fx (Any | None): Input fx. Defaults to `None`.
            fy (Any | None): Input fy. Defaults to `None`.
            cx (Any | None): Input cx. Defaults to `None`.
            cy (Any | None): Input cy. Defaults to `None`.
            max_depth (float): Input max depth. Defaults to `10.0`.

        Returns:
            Any: Result computed from the current inputs.
        """
        H,W = depth_map.shape

        if fx is None: fx = float(W)
        if fy is None: fy = float(W)
        if cx is None: cx = float(W)/2.0
        if cy is None: cy = float(H)/2.0

        u,v = np.meshgrid(np.arange(W),np.arange(H))
        u   = u.astype(np.float32)
        v   = v.astype(np.float32)

        # depth_map is already depth (0=near, 1=far), scale to max_depth
        z = depth_map.astype(np.float32) * float(max_depth)

        x = (u-cx)*z/fx
        y = (v-cy)*z/fy

        points = np.stack([x,y,z],axis=-1).reshape(-1,3)

        colors = None
        if img_np is not None:
            img    = rgb_img_np(img_np)
            colors = img.reshape(-1,3).astype(np.float32)/255.0

        return points,colors

    @staticmethod
    def plot_pointcloud(
            points,
            colors     = None,
            point_size = 2.0,
            width      = 800,
            height     = 600,
            downsample = 4,
            elev       = +10,   # top- down, 0 - front, + up
            azim       = -85,   # left - right, 0 - front, + up
            method     = "auto",
            bg_color   = (0.1,0.1,0.1),
        ):
        """
        Visualize 3D point cloud.

        Parameters:
            points (Any): Input points.
            colors (Any | None): Input colors. Defaults to `None`.
            point_size (float): Input point size. Defaults to `2.0`.
            width (int): Input width. Defaults to `800`.
            height (int): Input height. Defaults to `600`.
            downsample (int): Input downsample. Defaults to `4`.
            elev (Any): Input elev. Defaults to `+10`.
            azim (Any): Input azim. Defaults to `-85`.
            method (str): Input method. Defaults to `'auto'`.
            bg_color (tuple): Input bg color. Defaults to `(0.1, 0.1, 0.1)`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        method = str(method).lower().strip()

        # Try Open3D
        if method in ["auto","open3d"]:
            try:
                import open3d as o3d

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
                if colors is not None:
                    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

                pcd.transform([[1,0,0,0],
                            [0,-1,0,0],
                            [0,0,-1,0],
                            [0,0,0,1]])

                vis = o3d.visualization.Visualizer()
                vis.create_window(window_name="Point Cloud",width=int(width),height=int(height))
                vis.add_geometry(pcd)
                opt = vis.get_render_option()
                opt.point_size = float(point_size)
                opt.background_color = np.array(bg_color)
                vis.run()
                vis.destroy_window()
                return

            except ImportError:
                if method=="open3d":
                    print("[DepthAnythingV2] Open3D not installed.")
                    return
            except Exception as e:
                if method=="open3d":
                    print("[DepthAnythingV2] Open3D error: %s"%str(e))
                    return

        # Matplotlib fallback
        print("[DepthAnythingV2] Using matplotlib 3D visualization...")

        step = max(1,int(downsample))
        idx  = np.arange(0,len(points),step)
        pts  = points[idx]
        clr  = colors[idx] if colors is not None else None

        fig = plt.figure(figsize=(width/100,height/100),dpi=100)
        ax  = fig.add_subplot(111,projection='3d')

        # 이미지와 비슷한 뷰를 위해 좌표 변환
        x,y,z = pts[:,0], pts[:,1], -pts[:,2]

        if clr is not None:
            ax.scatter(x,y,z,c=clr,s=point_size,marker='.')
        else:
            ax.scatter(x,y,z,c=z,cmap='viridis',s=point_size,marker='.')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 이미지와 비슷하게 정면에서 약간만 회전
        ax.view_init(elev=elev, azim=azim)

        plt.tight_layout()
        plt.show()
        viz_close(fig)

    @staticmethod
    def plot_pointcloud_open3d(
            points,
            colors     = None,
            point_size = 2.0,
            bg_color   = (0.1,0.1,0.1),
            width      = 800,
            height     = 600,
        ):
        """
        Visualize 3D point cloud using Open3D.

        Parameters:
            points (Any): Input points.
            colors (Any | None): Input colors. Defaults to `None`.
            point_size (float): Input point size. Defaults to `2.0`.
            bg_color (tuple): Input bg color. Defaults to `(0.1, 0.1, 0.1)`.
            width (int): Input width. Defaults to `800`.
            height (int): Input height. Defaults to `600`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        try:
            import open3d as o3d
        except ImportError:
            print("[DepthAnythingV2] Open3D not installed. Run: pip install open3d")
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

        pcd.transform([[1,0,0,0],
                       [0,-1,0,0],
                       [0,0,-1,0],
                       [0,0,0,1]])

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Point Cloud",width=int(width),height=int(height))
        vis.add_geometry(pcd)
        opt = vis.get_render_option()
        opt.point_size = float(point_size)
        opt.background_color = np.array(bg_color)
        vis.run()
        vis.destroy_window()


# -------------------------------------------------------------------------
