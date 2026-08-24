"""
hf_llm.py
"""

import os
import numpy as np
import torch

from PIL import Image
from transformers import AutoProcessor

# from hf_common import (
#     ensure_local_ckpt,
#     get_device,
# )
from ri_motion_v5_package.hf_models.hf_common import (
    ensure_local_ckpt,
    get_device,
)

def _to_pil_rgb(image):
    """
    Convert an image to PIL RGB format.

    Parameters:
        image (Any): Input image.

    Returns:
        Any: Result computed from the current inputs.
    """
    if image is None:
        return None
    if isinstance(image,Image.Image):
        return image.convert("RGB")

    x = np.asarray(image)
    if x.ndim==2:
        x = np.stack([x,x,x],axis=-1)
    if (x.ndim==3) and (x.shape[-1]==4):
        x = x[...,:3]
    if (x.ndim!=3) or (x.shape[-1]!=3):
        raise ValueError("image must be HxWx3 (or convertible),got shape=%s"%(str(x.shape),))

    if x.dtype!=np.uint8:
        x = x.astype(np.float32)
        mx = float(np.max(x)) if x.size>0 else 0.0
        if mx<=1.0+1e-6:
            x = x*255.0
        x = np.clip(x,0.0,255.0).astype(np.uint8)

    return Image.fromarray(x,mode="RGB")

def _maybe_show_image_notebook(img_pil,width=480):
    """
    Display an image in a Jupyter notebook if possible.

    Parameters:
        img_pil (Any): Input img pil.
        width (int): Input width. Defaults to `480`.

    Returns:
        None: This function mutates state, renders output, or performs side
              effects in place.
    """
    try:
        from IPython.display import display,Image as IPyImage
        import io
        buf = io.BytesIO()
        img_pil.save(buf,format="PNG")
        display(IPyImage(data=buf.getvalue(),width=int(width)))
    except Exception:
        pass


class HuggingFaceLLM(object):
    """
    HuggingFace Large Language Model (or Vision Language Model) wrapper with
    history stacking.

    Usage:
        obj = HuggingFaceLLM(...)
    """
    def __init__(
            self,
            model                  = "Qwen/Qwen3-VL-4B-Instruct",
            local_dir              = None,
            revision               = None,
            device                 = None,
            init_prompt            = "You are a helpful agent with vision capabilities.",
            verbose                = True,
            notebook               = False,
            torch_dtype            = "auto",
            attn_impl              = "sdpa",
            resume_download        = True,
            local_dir_use_symlinks = False,
        ):
        """
        Initialize HuggingFaceLLM.

        Parameters:
            model (str): Input model. Defaults to `'Qwen/Qwen3-VL-4B-Instruct'`.
            local_dir (Any | None): Input local dir. Defaults to `None`.
            revision (Any | None): Input revision. Defaults to `None`.
            device (Any | None): Input device. Defaults to `None`.
            init_prompt (str): Input init prompt. Defaults to `'You are a helpful
                               agent with vision capabilities.'`.
            verbose (bool): Input verbose. Defaults to `True`.
            notebook (bool): Input notebook. Defaults to `False`.
            torch_dtype (str): Input torch dtype. Defaults to `'auto'`.
            attn_impl (str): Input attn impl. Defaults to `'sdpa'`.
            resume_download (bool): Input resume download. Defaults to `True`.
            local_dir_use_symlinks (bool): Input local dir use symlinks. Defaults to
                                           `False`.

        Returns:
            None: Initializes the instance in place.
        """
        self.instructions = str(init_prompt)
        self.history      = []
        self.notebook     = bool(notebook)
        self.verbose      = bool(verbose)

        # device
        self.device       = get_device(device=device,verbose=self.verbose)

        # resolve ckpt to local path if requested
        self.repo_id      = str(model)
        self.local_dir    = local_dir

        if self.local_dir is not None:
            self.model_id = ensure_local_ckpt(
                self.repo_id,
                local_dir            = self.local_dir,
                revision             = revision,
                verbose              = self.verbose,
                resume_download      = bool(resume_download),
                local_dir_use_symlinks = bool(local_dir_use_symlinks),
            )
        else:
            # no forced local_dir: just pass repo_id (HF cache default)
            self.model_id = self.repo_id

        # Processor
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        # Model load (dense first, then MoE fallback, then text-only fallback)
        self.model = None
        err0       = None
        err1       = None

        try:
            from transformers import Qwen3VLForConditionalGeneration
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device=="cuda" else None,
                attn_implementation=str(attn_impl),
            )
        except Exception as e:
            err0 = e

        if self.model is None:
            try:
                from transformers import Qwen3VLMoeForConditionalGeneration
                self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                    self.model_id,
                    torch_dtype=torch_dtype,
                    device_map="auto" if self.device=="cuda" else None,
                    attn_implementation=str(attn_impl),
                )
            except Exception as e:
                err1 = e

        if self.model is None:
            try:
                from transformers import AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch_dtype,
                    device_map="auto" if self.device=="cuda" else None,
                    attn_implementation=str(attn_impl),
                )
            except Exception as e2:
                raise RuntimeError("HF model load failed. dense_err=[%s], moe_err=[%s], text_err=[%s]"%(str(err0),str(err1),str(e2)))

        if (self.device!="cuda") and hasattr(self.model,"to"):
            self.model = self.model.to(self.device)
        self.model.eval()

        # Optional qwen_vl_utils for some versions
        self._has_qwen_vl_utils = False
        try:
            import qwen_vl_utils  # noqa: F401
            self._has_qwen_vl_utils = True
        except Exception:
            self._has_qwen_vl_utils = False

        if self.verbose:
            self._print_system(self.instructions)
            print("[HuggingFaceVisionLanguageModel] model=[%s] local_dir=[%s] device=[%s] ready."%
                  (str(self.repo_id),str(self.local_dir),str(self.device)))

    def reset_agent(self,content=None,verbose=True):
        """
        Reset the agent's state and optionally update the instructions.

        Parameters:
            content (Any | None): Input content. Defaults to `None`.
            verbose (bool): Input verbose. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if content is not None:
            self.instructions = str(content)
        self.history = []
        if bool(verbose):
            self._print_system(self.instructions)

    def _print_system(self,msg):
        """
        Print a system message.

        Parameters:
            msg (Any): Input msg.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if self.notebook:
            try:
                from IPython.display import display,Markdown
                tag = "<span style='color:#555;background:#f2f2f2;padding:2px 8px;border-radius:8px;font-weight:700;'>System</span>"
                display(Markdown("%s  %s"%(tag,msg)))
                return
            except Exception:
                pass
        print("[System] %s"%msg)

    def _print_user(self,msg):
        """
        Print a user message.

        Parameters:
            msg (Any): Input msg.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if self.notebook:
            try:
                from IPython.display import display,Markdown
                tag = "<span style='color:#0b3d91;background:#e8f0fe;padding:2px 8px;border-radius:8px;font-weight:700;'>User</span>"
                display(Markdown("%s  %s"%(tag,msg)))
                return
            except Exception:
                pass
        print("[User] %s"%msg)

    def _print_assistant(self,msg):
        """
        Print an assistant message.

        Parameters:
            msg (Any): Input msg.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        if self.notebook:
            try:
                from IPython.display import display,Markdown
                tag = "<span style='color:#0f5132;background:#eaf7ee;padding:2px 8px;border-radius:8px;font-weight:700;'>Assistant</span>"
                display(Markdown("%s\n\n%s"%(tag,msg)))
                return
            except Exception:
                pass
        print("[Assistant] %s"%msg)

    def _build_messages(self,text,img_pil):
        """
        Build the messages for the chat.

        Parameters:
            text (Any): Input text.
            img_pil (Any): Input img pil.

        Returns:
            Any: Result computed from the current inputs.
        """
        sys_msg = {"role":"system","content":[{"type":"text","text":self.instructions}]}
        if img_pil is None:
            user_msg = {"role":"user","content":[{"type":"text","text":str(text) if text is not None else ""}]}
        else:
            user_msg = {"role":"user","content":[{"type":"image","image":img_pil},{"type":"text","text":str(text) if text is not None else ""}]}
        return [sys_msg] + list(self.history) + [user_msg]

    def _encode_inputs(self,messages):
        """
        Encode the input messages for the model.

        Parameters:
            messages (Any): Input messages.

        Returns:
            Any: Result computed from the current inputs.
        """
        if self._has_qwen_vl_utils: # some versions of Qwen VL models require special vision input processing
            try:
                from qwen_vl_utils import process_vision_info
                vision_inputs = process_vision_info(messages)
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    **vision_inputs,
                )
            except Exception:
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
        else:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

        if isinstance(inputs,dict) and ("token_type_ids" in inputs):
            inputs.pop("token_type_ids",None)

        for k in inputs:
            if hasattr(inputs[k],"to"):
                inputs[k] = inputs[k].to(self.device)
        return inputs

    def chat(
            self,
            text        = None,
            image       = None,
            temperature = None,
            top_p       = None,
            stack       = True,
            verbose     = True,
            show_image  = True,
            width       = 480,
            max_new_tokens = 256,
        ):
        """
        Chat with the assistant.

        Parameters:
            text (Any | None): Input text. Defaults to `None`.
            image (Any | None): Input image. Defaults to `None`.
            temperature (Any | None): Input temperature. Defaults to `None`.
            top_p (Any | None): Input top p. Defaults to `None`.
            stack (bool): Input stack. Defaults to `True`.
            verbose (bool): Input verbose. Defaults to `True`.
            show_image (bool): Input show image. Defaults to `True`.
            width (int): Input width. Defaults to `480`.
            max_new_tokens (int): Input max new tokens. Defaults to `256`.

        Returns:
            Any: Result computed from the current inputs.
        """
        img_pil  = _to_pil_rgb(image)
        messages = self._build_messages(text,img_pil)
        inputs   = self._encode_inputs(messages)

        gen_kwargs = dict(max_new_tokens=int(max_new_tokens))
        if temperature is not None and float(temperature)>0.0:
            gen_kwargs["do_sample"]   = True
            gen_kwargs["temperature"] = float(temperature)
            if top_p is not None:
                gen_kwargs["top_p"] = float(top_p)

        with torch.no_grad():
            out_ids = self.model.generate(**inputs,**gen_kwargs)

        in_ids   = inputs["input_ids"]
        trimmed  = [o[len(i):] for i,o in zip(in_ids,out_ids)]
        msg      = self.processor.batch_decode(trimmed,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]

        if stack:
            if img_pil is None:
                self.history.append({"role":"user","content":[{"type":"text","text":str(text) if text is not None else ""}]})
            else:
                self.history.append({"role":"user","content":[{"type":"image","image":img_pil},{"type":"text","text":str(text) if text is not None else ""}]})
            self.history.append({"role":"assistant","content":[{"type":"text","text":str(msg)}]})

        if bool(verbose):
            self._print_user(str(text) if text is not None else "")
            if self.notebook and (img_pil is not None) and bool(show_image):
                _maybe_show_image_notebook(img_pil,width=width)
            self._print_assistant(str(msg))

        return msg

