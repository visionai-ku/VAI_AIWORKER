"""
hf_face.py
"""

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# from hf_common import (
#     ensure_pip_packages,
#     hf_hub_download_if_needed,
# )
# from hf_viz import (
#     viz_title,
#     viz_close,
# )
from ri_motion_v5_package.hf_models.hf_common import (
    ensure_pip_packages,
    hf_hub_download_if_needed,
)
from ri_motion_v5_package.hf_models.hf_viz import (
    viz_title,
    viz_close,
)

# Install required packages on demand before importing them.
ensure_pip_packages(
    packages     = [
        "ultralytics",
        "huggingface_hub",
        "opencv-python",
        "insightface",
        "onnxruntime",
    ],
    import_names = {
        "opencv-python":"cv2",
        "huggingface_hub":"huggingface_hub",
        "onnxruntime":"onnxruntime",
        "ultralytics":"ultralytics",
        "insightface":"insightface",
    },
    verbose      = True,
    upgrade      = False,
)

from ultralytics import YOLO
import onnxruntime

class YoloFaceDet(object):
    """
    YOLO-based face detection.

    Usage:
        obj = YoloFaceDet(...)
    """

    def __init__(
            self,
            ckpt_repo   = "AdamCodd/YOLOv11n-face-detection",
            ckpt_file   = "model.pt",
            verbose     = True,
            local_dir   = "./models/yolo_face_det",
            revision    = None,
        ):
        """
        Initialize YoloFaceDet.

        Parameters:
            ckpt_repo (str): Input ckpt repo. Defaults to `'AdamCodd/YOLOv11n-face-
                             detection'`.
            ckpt_file (str): Input ckpt file. Defaults to `'model.pt'`.
            verbose (bool): Input verbose. Defaults to `True`.
            local_dir (str): Input local dir. Defaults to `'./models/yolo_face_det'`.
            revision (Any | None): Input revision. Defaults to `None`.

        Returns:
            None: Initializes the instance in place.
        """
        self.verbose   = bool(verbose)
        self.ckpt_repo = str(ckpt_repo)
        self.ckpt_file = str(ckpt_file)
        self.local_dir = str(local_dir)
        self.revision  = revision

        os.makedirs(self.local_dir,exist_ok=True)

        self.ckpt_path = hf_hub_download_if_needed(
            repo_id   = self.ckpt_repo,
            filename  = self.ckpt_file,
            local_dir = self.local_dir,
            revision  = self.revision,
            verbose   = self.verbose,
        )

        self.model = YOLO(self.ckpt_path)

        if self.verbose:
            print("[YoloFaceDet] ckpt=[%s] ready."%self.ckpt_path)

    @staticmethod
    def _to_bgr(img):
        """
        Convert input image to BGR numpy array.

        Parameters:
            img (Any): Input img.

        Returns:
            Any: Result computed from the current inputs.
        """
        if isinstance(img,str):
            out = cv2.imread(img)
            if out is None:
                raise ValueError("Failed to read image:[%s]"%img)
            return out

        if hasattr(img,"convert"):
            arr = np.asarray(img.convert("RGB"))
            return cv2.cvtColor(arr,cv2.COLOR_RGB2BGR)

        arr = np.asarray(img)
        if arr.ndim==2:
            arr = np.stack([arr,arr,arr],axis=-1)

        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
            if arr.max() <= 1.0 + 1e-6:
                arr = arr * 255.0
            arr = np.clip(arr,0,255).astype(np.uint8)

        return arr

    def predict_boxes(
            self,
            img,
            conf = 0.25,
        ):
        """
        Detect faces and return boxes/confidences.

        Parameters:
            img (Any): Input img.
            conf (float): Input conf. Defaults to `0.25`.

        Returns:
            Any: Result computed from the current inputs.
        """
        img_bgr = self._to_bgr(img)

        results = self.model.predict(
            source = img_bgr,
            conf   = float(conf),
            verbose=False,
        )

        if len(results)==0 or results[0].boxes is None:
            boxes = np.zeros((0,4),dtype=np.int32)
            confs = np.zeros((0,),dtype=np.float32)
            return img_bgr,boxes,confs,results

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(np.int32)
        confs = results[0].boxes.conf.cpu().numpy().astype(np.float32)

        return img_bgr,boxes,confs,results

    @staticmethod
    def crop_faces(
            img_bgr,
            boxes,
            pad = 0,
        ):
        """
        Crop face images from detection boxes.

        Parameters:
            img_bgr (Any): Input img bgr.
            boxes (Any): Input boxes.
            pad (int): Input pad. Defaults to `0`.

        Returns:
            Any: Result computed from the current inputs.
        """
        H,W = img_bgr.shape[:2]
        face_imgs  = []
        face_boxes = []

        for box in boxes:
            x1,y1,x2,y2 = [int(v) for v in box.tolist()]
            x1 = max(0,x1-int(pad))
            y1 = max(0,y1-int(pad))
            x2 = min(W,x2+int(pad))
            y2 = min(H,y2+int(pad))
            if x2 <= x1 or y2 <= y1:
                continue

            cropped = img_bgr[y1:y2,x1:x2].copy()
            face_imgs.append(cropped)
            face_boxes.append([x1,y1,x2,y2])

        return face_imgs,face_boxes

    def run(
            self,
            img,
            conf = 0.25,
            pad  = 0,
        ):
        """
        Run face detection and cropping.

        Parameters:
            img (Any): Input img.
            conf (float): Input conf. Defaults to `0.25`.
            pad (int): Input pad. Defaults to `0`.

        Returns:
            Any: Result computed from the current inputs.
        """
        img_bgr,boxes,confs,results = self.predict_boxes(
            img  = img,
            conf = conf,
        )

        face_imgs,face_boxes = self.crop_faces(
            img_bgr = img_bgr,
            boxes   = boxes,
            pad     = pad,
        )

        out = dict(
            img_bgr    = img_bgr,
            boxes      = boxes,
            confs      = confs,
            face_imgs  = face_imgs,
            face_boxes = face_boxes,
            results    = results,
        )

        if self.verbose:
            print("[YoloFaceDet] det:%d, conf:%.3f"%(len(face_imgs),float(conf)))

        return out

    @staticmethod
    def plot_dets(
            img_bgr,
            boxes,
            confs        = None,
            fig_w        = 5,
            linewidth    = 2,
            edgecolor    = "r",
            title        = "Face Detection",
            title_fs     = 10,
            show_scores  = True,
            score_fmt    = "%.2f",
            text_offset  = (2,2),
            fontsize     = 8,
        ):
        """
        Plot face detection results.

        Parameters:
            img_bgr (Any): Input img bgr.
            boxes (Any): Input boxes.
            confs (Any | None): Input confs. Defaults to `None`.
            fig_w (int): Input fig w. Defaults to `5`.
            linewidth (int): Input linewidth. Defaults to `2`.
            edgecolor (str): Input edgecolor. Defaults to `'r'`.
            title (str): Input title. Defaults to `'Face Detection'`.
            title_fs (int): Input title fs. Defaults to `10`.
            show_scores (bool): Input show scores. Defaults to `True`.
            score_fmt (str): Input score fmt. Defaults to `'%.2f'`.
            text_offset (tuple): Input text offset. Defaults to `(2, 2)`.
            fontsize (int): Input fontsize. Defaults to `8`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
        H,W     = img_rgb.shape[:2]
        fig_h   = float(fig_w)*float(H)/max(1.0,float(W))

        fig,ax = plt.subplots(figsize=(float(fig_w),fig_h),dpi=120)
        ax.imshow(img_rgb)
        ax.axis("off")
        ax.set_title(str(title),fontsize=int(title_fs))

        dx,dy = float(text_offset[0]),float(text_offset[1])

        for i,box in enumerate(boxes):
            x1,y1,x2,y2 = [float(v) for v in box.tolist()]
            rect = plt.Rectangle(
                (x1,y1),
                x2-x1,
                y2-y1,
                fill      = False,
                linewidth = float(linewidth),
                edgecolor = str(edgecolor),
            )
            ax.add_patch(rect)

            if (confs is not None) and bool(show_scores):
                txt = score_fmt%float(confs[i])
                ax.text(
                    x1+dx,
                    y1+dy,
                    txt,
                    fontsize=int(fontsize),
                    color="white",
                    bbox={"facecolor":"red","alpha":0.7,"pad":1,"edgecolor":"none"},
                )

        plt.tight_layout()
        plt.show()
        plt.close(fig)

    @staticmethod
    def plot_faces(
            face_imgs,
            names        = None,
            sims         = None,
            unknown_name = "Unknown",
            ncols        = 4,
            fig_w        = 12,
            score_fmt    = "%.2f",
        ):
        """
        Plot cropped face images.

        Parameters:
            face_imgs (Any): Input face imgs.
            names (Any | None): Input names. Defaults to `None`.
            sims (Any | None): Input sims. Defaults to `None`.
            unknown_name (str): Input unknown name. Defaults to `'Unknown'`.
            ncols (int): Input ncols. Defaults to `4`.
            fig_w (int): Input fig w. Defaults to `12`.
            score_fmt (str): Input score fmt. Defaults to `'%.2f'`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        n = len(face_imgs)
        if n == 0:
            print("[YoloFaceDet] no face images to plot.")
            return

        ncols = max(1,int(ncols))
        nrows = int(np.ceil(float(n)/float(ncols)))

        fig_h = 3.5 * nrows
        fig,axes = plt.subplots(nrows,ncols,figsize=(float(fig_w),fig_h),dpi=120)

        if not isinstance(axes,np.ndarray):
            axes = np.array([axes])
        axes = axes.reshape(-1)

        for i in range(len(axes)):
            ax = axes[i]
            ax.axis("off")

            if i >= n:
                continue

            face_rgb = cv2.cvtColor(face_imgs[i],cv2.COLOR_BGR2RGB)
            ax.imshow(face_rgb)

            if names is None:
                title = "[%d]"%i
                ax.set_title(title)
                continue

            name = str(names[i])
            if sims is not None:
                title = "[%d] %s sim:%s"%(i,name,score_fmt%float(sims[i]))
            else:
                title = "[%d] %s"%(i,name)

            if name != str(unknown_name):
                ax.set_title(title,fontweight="bold",color="blue")
            else:
                ax.set_title(title)

        plt.tight_layout()
        plt.show()
        plt.close(fig)


class FaceEmbedding(object):
    """
    InsightFace-style ONNX face embedding extractor.

    Usage:
        obj = FaceEmbedding(...)
    """

    def __init__(
            self,
            ckpt_repo   = "deepghs/insightface",
            ckpt_file   = "buffalo_l/w600k_r50.onnx",
            providers   = None,
            verbose     = True,
            local_dir   = "./models/face_embedding",
            revision    = None,
        ):
        """
        Initialize FaceEmbedding.

        Parameters:
            ckpt_repo (str): Input ckpt repo. Defaults to `'deepghs/insightface'`.
            ckpt_file (str): Input ckpt file. Defaults to
                             `'buffalo_l/w600k_r50.onnx'`.
            providers (Any | None): Input providers. Defaults to `None`.
            verbose (bool): Input verbose. Defaults to `True`.
            local_dir (str): Input local dir. Defaults to `'./models/face_embedding'`.
            revision (Any | None): Input revision. Defaults to `None`.

        Returns:
            None: Initializes the instance in place.
        """
        self.verbose   = bool(verbose)
        self.ckpt_repo = str(ckpt_repo)
        self.ckpt_file = str(ckpt_file)
        self.local_dir = str(local_dir)
        self.revision  = revision

        os.makedirs(self.local_dir,exist_ok=True)

        self.ckpt_path = hf_hub_download_if_needed(
            repo_id   = self.ckpt_repo,
            filename  = self.ckpt_file,
            local_dir = self.local_dir,
            revision  = self.revision,
            verbose   = self.verbose,
        )

        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.providers = list(providers)

        self.model = onnxruntime.InferenceSession(
            self.ckpt_path,
            providers = self.providers,
        )

        self.input_name = self.model.get_inputs()[0].name

        if self.verbose:
            print("[FaceEmbedding] ckpt=[%s] ready."%self.ckpt_path)
            print("[FaceEmbedding] providers=%s"%str(self.providers))

    @staticmethod
    def _preprocess(face_img_bgr):
        """
        Preprocess BGR face crop into ONNX model input.

        Parameters:
            face_img_bgr (Any): Input face img bgr.

        Returns:
            Any: Result computed from the current inputs.
        """
        face_img = cv2.resize(face_img_bgr,(112,112))
        face_img = cv2.cvtColor(face_img,cv2.COLOR_BGR2RGB)
        face_img = face_img.transpose((2,0,1)).astype(np.float32)
        face_img = (face_img - 127.5) / 128.0
        face_img = np.expand_dims(face_img,axis=0)
        return face_img

    def get_embedding(
            self,
            face_img_bgr,
        ):
        """
        Extract normalized face embedding.

        Parameters:
            face_img_bgr (Any): Input face img bgr.

        Returns:
            Any: Requested value computed from the current inputs.
        """
        x       = self._preprocess(face_img_bgr)
        net_out = self.model.run(None,{self.input_name:x})
        emb     = net_out[0].flatten().astype(np.float32)
        norm    = float(np.linalg.norm(emb))
        emb     = emb / (norm + 1e-6)
        return emb

    def run(
            self,
            face_img_bgr,
        ):
        """
        Alias of get_embedding().

        Parameters:
            face_img_bgr (Any): Input face img bgr.

        Returns:
            Any: Result computed from the current inputs.
        """
        emb = self.get_embedding(face_img_bgr)
        if self.verbose:
            print("[FaceEmbedding] embedding_dim:%d"%int(emb.shape[0]))
        return emb

    @staticmethod
    def compute_similarity(
            feat1,
            feat2,
        ):
        """
        Compute cosine similarity.

        Parameters:
            feat1 (Any): Input feat1.
            feat2 (Any): Input feat2.

        Returns:
            Any: Result computed from the current inputs.
        """
        feat1 = np.asarray(feat1,dtype=np.float32).reshape(-1)
        feat2 = np.asarray(feat2,dtype=np.float32).reshape(-1)

        n1 = float(np.linalg.norm(feat1))
        n2 = float(np.linalg.norm(feat2))

        return float(np.dot(feat1,feat2) / (n1*n2 + 1e-12))

class FaceIdentification(object):
    """
    Face identification pipeline using:.

    Usage:
        obj = FaceIdentification(...)
    """

    def __init__(
            self,
            detector,
            encoder,
            id_threshold = 0.4,
            verbose      = True,
        ):
        """
        Initialize FaceIdentification.

        Parameters:
            detector (Any): Input detector.
            encoder (Any): Input encoder.
            id_threshold (float): Input id threshold. Defaults to `0.4`.
            verbose (bool): Input verbose. Defaults to `True`.

        Returns:
            None: Initializes the instance in place.
        """
        self.detector     = detector
        self.encoder      = encoder
        self.id_threshold = float(id_threshold)
        self.verbose      = bool(verbose)

        self.db = {}

    def build_db_from_dir(
            self,
            db_dir,
            glob_patterns = None,
            return_db     = False,
        ):
        """
        Build face DB from images in a directory.

        Parameters:
            db_dir (str | Path): Input db dir.
            glob_patterns (Any | None): Input glob patterns. Defaults to `None`.
            return_db (bool): Input return db. Defaults to `False`.

        Returns:
            Any: Result computed from the current inputs.
        """

        if glob_patterns is None:
            glob_patterns = ["*.jpg","*.jpeg","*.png","*.bmp","*.webp"]

        paths = []
        for pat in glob_patterns:
            paths.extend(glob.glob(os.path.join(db_dir,pat)))
        paths = sorted(list(set(paths)))

        db = {}
        for img_path in paths:
            img = cv2.imread(img_path)
            if img is None:
                continue

            name = os.path.splitext(os.path.basename(img_path))[0]

            db[name] = {}
            db[name]["image"]     = img
            db[name]["embedding"] = self.encoder.get_embedding(img)
            db[name]["path"]      = img_path

        self.db = db

        if self.verbose:
            print("[FaceIdentification] db_size:[%d], dir:[%s]"%(len(self.db),db_dir))

        if bool(return_db):
            return self.db

    def identify_faces(
            self,
            face_imgs,
        ):
        """
        Identify cropped face images against current DB.

        Parameters:
            face_imgs (Any): Input face imgs.

        Returns:
            Any: Result computed from the current inputs.
        """
        pred_names    = []
        pred_sims     = []
        nearest_names = []

        for face in face_imgs:
            face_emb = self.encoder.get_embedding(face)

            best_sim  = -1.0
            best_name = ""

            for name in self.db.keys():
                sim = self.encoder.compute_similarity(
                    face_emb,
                    self.db[name]["embedding"],
                )
                if sim > best_sim:
                    best_sim  = sim
                    best_name = name

            nearest_names.append(best_name)
            pred_sims.append(float(best_sim))

            if best_sim > float(self.id_threshold):
                pred_names.append(best_name)
            else:
                pred_names.append("Unknown")

        return pred_names,pred_sims,nearest_names

    def run(
            self,
            img,
            conf = 0.25,
            pad  = 0,
        ):
        """
        Full face identification pipeline.

        Parameters:
            img (Any): Input img.
            conf (float): Input conf. Defaults to `0.25`.
            pad (int): Input pad. Defaults to `0`.

        Returns:
            Any: Result computed from the current inputs.
        """
        det_out = self.detector.run(
            img  = img,
            conf = conf,
            pad  = pad,
        )

        pred_names,pred_sims,nearest_names = self.identify_faces(
            face_imgs = det_out["face_imgs"],
        )

        out = dict(
            img_bgr       = det_out["img_bgr"],
            boxes         = det_out["boxes"],
            confs         = det_out["confs"],
            face_imgs     = det_out["face_imgs"],
            face_boxes    = det_out["face_boxes"],
            pred_names    = pred_names,
            pred_sims     = pred_sims,
            nearest_names = nearest_names,
        )

        if self.verbose:
            print("[FaceIdentification] faces:%d, threshold:%.3f"%
                (len(out["face_imgs"]),float(self.id_threshold)))

        return out

    @staticmethod
    def _get_plot_style(
            title_fs = 8,
            label_fs = None,
            name_fs  = None,
            meta_fs  = None,
            alpha    = 0.55,
            pad      = 1.2,
            line_w   = 1.5,
        ):
        """
        Shared plotting style.

        Parameters:
            title_fs (int): Input title fs. Defaults to `8`.
            label_fs (Any | None): Input label fs. Defaults to `None`.
            name_fs (Any | None): Input name fs. Defaults to `None`.
            meta_fs (Any | None): Input meta fs. Defaults to `None`.
            alpha (float): Input alpha. Defaults to `0.55`.
            pad (float): Input pad. Defaults to `1.2`.
            line_w (float): Input line w. Defaults to `1.5`.

        Returns:
            Any: Result computed from the current inputs.
        """
        title_fs = int(title_fs)

        if label_fs is None:
            label_fs = max(4,int(round(title_fs*0.70)))
        if name_fs is None:
            name_fs = max(5,int(round(title_fs*0.95)))
        if meta_fs is None:
            meta_fs = max(4,int(round(title_fs*0.85)))

        style = dict(
            title_fs = int(title_fs),
            label_fs = int(label_fs),
            name_fs  = int(name_fs),
            meta_fs  = int(meta_fs),
            alpha    = float(alpha),
            pad      = float(pad),
            line_w   = float(line_w),
        )
        return style

    """
    Visualization utilities for face identification results.
     - plot_db()
     - plot_detection()
     - plot_faces()
    """

    @staticmethod
    def plot_db(
            db,
            fig_w    = 5.0,
            n_col    = 3,
            title_fs = 8,
            label_fs = None,
            alpha    = 0.55,
            pad      = 1.2,
        ):
        """
        Plot face database.

        Parameters:
            db (Any): Input db.
            fig_w (float): Input fig w. Defaults to `5.0`.
            n_col (int): Input n col. Defaults to `3`.
            title_fs (int): Input title fs. Defaults to `8`.
            label_fs (Any | None): Input label fs. Defaults to `None`.
            alpha (float): Input alpha. Defaults to `0.55`.
            pad (float): Input pad. Defaults to `1.2`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """

        style = FaceIdentification._get_plot_style(
            title_fs = title_fs,
            label_fs = label_fs,
            alpha    = alpha,
            pad      = pad,
        )

        names = sorted(list(db.keys()))
        n     = len(names)

        if n == 0:
            print("[FaceIdentification] DB is empty.")
            return

        n_col = max(1,int(n_col))
        n_row = int(np.ceil(float(n)/float(n_col)))

        margin_x = 0.01
        gap_x    = 0.01

        usable_w = 1.0 - 2.0*margin_x - gap_x*(n_col-1)
        cell_w   = usable_w / float(n_col)

        title_h  = 0.06
        gap_y    = 0.015
        margin_y = 0.01
        cell_h   = cell_w

        usable_h  = n_row*cell_h + (n_row-1)*gap_y
        content_h = title_h + usable_h + margin_y
        fig_h     = float(fig_w) * float(content_h)

        title_h_n  = title_h  / content_h
        gap_y_n    = gap_y    / content_h
        cell_h_n   = cell_h   / content_h

        fig = plt.figure(figsize=(float(fig_w),float(fig_h)),dpi=120)

        ax0 = fig.add_axes([0,0,1,1])
        ax0.set_axis_off()
        viz_title(fig,ax0,"Face Database",style["title_fs"],False)

        y_top0 = 1.0 - title_h_n

        for i in range(n_row*n_col):
            row = i // n_col
            col = i % n_col

            x = margin_x + col*(cell_w + gap_x)
            y = y_top0 - (row+1)*cell_h_n - row*gap_y_n

            ax = fig.add_axes([x,y,cell_w,cell_h_n])
            ax.set_xticks([])
            ax.set_yticks([])

            if i >= n:
                ax.axis("off")
                continue

            name = names[i]
            img  = db[name]["image"]
            rgb  = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            ax.imshow(rgb,interpolation="nearest")
            ax.text(
                0.5,0.97,str(name),
                transform = ax.transAxes,
                ha        = "center",
                va        = "top",
                fontsize  = style["label_fs"],
                color     = "white",
                bbox      = dict(
                    facecolor = "black",
                    alpha     = style["alpha"],
                    pad       = style["pad"],
                    edgecolor = "none",
                ),
            )

        plt.show()
        viz_close(fig)

    @staticmethod
    def plot_detection(
            out,
            fig_w    = 5.0,
            title_fs = 8,
            label_fs = None,
            alpha    = 0.55,
            pad      = 1.2,
            line_w   = 1.5,
            y_offset = 6.0,
        ):
        """
        Plot detection result.

        Parameters:
            out (Any): Input out.
            fig_w (float): Input fig w. Defaults to `5.0`.
            title_fs (int): Input title fs. Defaults to `8`.
            label_fs (Any | None): Input label fs. Defaults to `None`.
            alpha (float): Input alpha. Defaults to `0.55`.
            pad (float): Input pad. Defaults to `1.2`.
            line_w (float): Input line w. Defaults to `1.5`.
            y_offset (float): Input y offset. Defaults to `6.0`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """

        style = FaceIdentification._get_plot_style(
            title_fs = title_fs,
            label_fs = label_fs,
            alpha    = alpha,
            pad      = pad,
            line_w   = line_w,
        )

        img_bgr = out.get("img_bgr",None)
        boxes   = out.get("boxes",[])
        confs   = out.get("confs",None)

        if img_bgr is None:
            raise ValueError("plot_detection() requires out[\"img_bgr\"].")

        img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
        h,w     = img_rgb.shape[:2]

        title_h   = 0.06
        margin_y  = 0.01
        ax_h      = 1.0 * float(h) / max(1.0,float(w))
        content_h = title_h + ax_h + margin_y
        fig_h     = float(fig_w) * float(content_h)

        margin_y_n = margin_y / content_h
        ax_h_n     = ax_h     / content_h

        fig = plt.figure(figsize=(float(fig_w),float(fig_h)),dpi=120)

        ax0 = fig.add_axes([0,0,1,1])
        ax0.set_axis_off()
        viz_title(fig,ax0,"Detection",style["title_fs"],False)

        ax = fig.add_axes([0.0,margin_y_n,1.0,ax_h_n])
        ax.imshow(img_rgb,interpolation="nearest")
        ax.set_xlim(-0.5,float(w)-0.5)
        ax.set_ylim(float(h)-0.5,-0.5)
        ax.set_autoscale_on(False)
        ax.set_axis_off()

        for i,box in enumerate(boxes):
            x1,y1,x2,y2 = [float(v) for v in box.tolist()]

            rect = plt.Rectangle(
                (x1,y1),
                x2-x1,
                y2-y1,
                fill      = False,
                linewidth = style["line_w"],
                edgecolor = "r",
            )
            ax.add_patch(rect)

            txt = "[%d]"%i
            if (confs is not None) and (i < len(confs)):
                txt = "%s %.2f"%(txt,float(confs[i]))

            y_txt = max(y1-float(y_offset),0.0)

            ax.text(
                x1,
                y_txt,
                txt,
                ha       = "left",
                va       = "bottom",
                fontsize = style["label_fs"],
                color    = "white",
                bbox      = dict(
                    facecolor = "black",
                    alpha     = style["alpha"],
                    pad       = style["pad"],
                    edgecolor = "none",
                ),
                clip_on   = False,
            )

        plt.show()
        viz_close(fig)

    @staticmethod
    def plot_faces(
            out,
            fig_w    = 5.0,
            n_col    = 3,
            title_fs = 8,
            name_fs  = None,
            meta_fs  = None,
            alpha    = 0.55,
            pad      = 1.2,
        ):
        """
        Plot cropped face gallery.

        Parameters:
            out (Any): Input out.
            fig_w (float): Input fig w. Defaults to `5.0`.
            n_col (int): Input n col. Defaults to `3`.
            title_fs (int): Input title fs. Defaults to `8`.
            name_fs (Any | None): Input name fs. Defaults to `None`.
            meta_fs (Any | None): Input meta fs. Defaults to `None`.
            alpha (float): Input alpha. Defaults to `0.55`.
            pad (float): Input pad. Defaults to `1.2`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """

        style = FaceIdentification._get_plot_style(
            title_fs = title_fs,
            name_fs  = name_fs,
            meta_fs  = meta_fs,
            alpha    = alpha,
            pad      = pad,
        )

        face_imgs     = out.get("face_imgs",[])
        pred_names    = out.get("pred_names",[])
        pred_sims     = out.get("pred_sims",[])
        nearest_names = out.get("nearest_names",pred_names)

        n = len(face_imgs)
        if n == 0:
            print("[FaceIdentification] no faces to plot.")
            return

        n_col = max(1,int(n_col))
        n_row = int(np.ceil(float(n)/float(n_col)))

        margin_x = 0.01
        gap_x    = 0.01

        title_h  = 0.06
        gap_y    = 0.015
        margin_y = 0.01

        usable_w = 1.0 - 2.0*margin_x - gap_x*(n_col-1)
        cell_w   = usable_w / float(n_col)
        cell_h   = cell_w

        usable_h  = n_row*cell_h + (n_row-1)*gap_y
        content_h = title_h + usable_h + margin_y
        fig_h     = float(fig_w) * float(content_h)

        title_h_n  = title_h  / content_h
        gap_y_n    = gap_y    / content_h
        cell_h_n   = cell_h   / content_h

        fig = plt.figure(figsize=(float(fig_w),float(fig_h)),dpi=120)

        ax0 = fig.add_axes([0,0,1,1])
        ax0.set_axis_off()
        viz_title(fig,ax0,"Faces",style["title_fs"],False)

        y_top0 = 1.0 - title_h_n

        for i in range(n_row*n_col):
            row = i // n_col
            col = i % n_col

            x = margin_x + col*(cell_w + gap_x)
            y = y_top0 - (row+1)*cell_h_n - row*gap_y_n

            ax = fig.add_axes([x,y,cell_w,cell_h_n])
            ax.set_xticks([])
            ax.set_yticks([])

            if i >= n:
                ax.axis("off")
                continue

            face_rgb = cv2.cvtColor(face_imgs[i],cv2.COLOR_BGR2RGB)
            ax.imshow(face_rgb,interpolation="nearest")

            pred_name    = pred_names[i] if i < len(pred_names) else ""
            pred_sim     = pred_sims[i] if i < len(pred_sims) else None
            nearest_name = nearest_names[i] if i < len(nearest_names) else ""

            line1 = "[%d] %s"%(i,str(pred_name))
            if pred_sim is not None:
                line2 = "%s %.2f"%(str(nearest_name),float(pred_sim))
            else:
                line2 = "%s"%str(nearest_name)

            c1 = "white" if str(pred_name) == "Unknown" else "cyan"

            ax.text(
                0.5,0.97,line1,
                transform = ax.transAxes,
                ha        = "center",
                va        = "top",
                fontsize  = style["name_fs"],
                color     = c1,
                bbox      = dict(
                    facecolor = "black",
                    alpha     = style["alpha"],
                    pad       = style["pad"],
                    edgecolor = "none",
                ),
            )
            ax.text(
                0.5,0.85,line2,
                transform = ax.transAxes,
                ha        = "center",
                va        = "top",
                fontsize  = style["meta_fs"],
                color     = "white",
                bbox      = dict(
                    facecolor = "black",
                    alpha     = style["alpha"],
                    pad       = style["pad"],
                    edgecolor = "none",
                ),
            )

        plt.show()
        viz_close(fig)

# -------------------------------------------------------------------------