"""
openai_helper.py
"""

import io,base64
import openai
import numpy as np
from openai import OpenAI
from PIL import Image

def print_available_openai_models(
        key_path: str = "../../../rilab_openai_key/rilab_key.txt",
    ):
    """
    Print available OpenAI models.

    Parameters:
        key_path (str): Input key path. Defaults to
                        `'../../../rilab_openai_key/rilab_key.txt'`.

    Returns:
        None: This function mutates state, renders output, or performs side
              effects in place.
    """
    # Read API key from file
    with open(key_path,"r") as f:
        api_key = f.read().strip()

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Existing models
    models = client.models.list()
    for idx,model in enumerate(models.data):
        mid = model.id
        mown = model.owned_by
        print("[%03d/%d] [%s] by [%s]"%(idx,len(models.data),mid,mown))

class OpenAIVisionLanguageModel(object):
    """
    A class representing an OpenAI Vision-Language Model.

    Usage:
        obj = OpenAIVisionLanguageModel(...)
    """
    def __init__(
            self,
            model       = "gpt-5.2-pro-2025-12-11",
            key_path    = "",
            init_prompt = "You are a helpful agent with vision capabilities.",
            verbose     = True,
            notebook    = False,
        ):
        """
        Initialize OpenAIVisionLanguageModel.

        Parameters:
            model (str): Input model. Defaults to `'gpt-5.2-pro-2025-12-11'`.
            key_path (str): Input key path. Defaults to `''`.
            init_prompt (str): Input init prompt. Defaults to `'You are a helpful
                               agent with vision capabilities.'`.
            verbose (bool): Input verbose. Defaults to `True`.
            notebook (bool): Input notebook. Defaults to `False`.

        Returns:
            None: Initializes the instance in place.
        """
        self.model        = model
        self.instructions = init_prompt
        self.history      = []
        self.notebook     = bool(notebook)

        # Read API key from file
        api_key = None
        if key_path:
            with open(key_path,"r") as f:
                api_key = f.read().strip()

        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)

        if verbose:
            self._print_system(self.instructions)

    def reset_agent(self,content=None,verbose=True):
        """
        Reset the agent's state.

        Parameters:
            content (Any | None): Input content. Defaults to `None`.
            verbose (bool): Input verbose. Defaults to `True`.

        Returns:
            None: This function mutates state, renders output, or performs side
                  effects in place.
        """
        # Update instructions if new content is provided
        if content is not None:
            self.instructions = content

        # Reset conversation history
        self.history = []

        # Print instructions if verbose
        if verbose:
            self._print_system(self.instructions)

    def _img_array_to_data_url(self,img_array,fmt="PNG"):
        """
        Convert an image array to a data URL.

        Parameters:
            img_array (Any): Input img array.
            fmt (str): Input fmt. Defaults to `'PNG'`.

        Returns:
            Any: Result computed from the current inputs.
        """
        x = np.asarray(img_array)

        if x.ndim==2:
            x = np.stack([x,x,x],axis=-1)
        if (x.ndim==3) and (x.shape[-1]==4):
            x = x[...,:3]
        if (x.ndim!=3) or (x.shape[-1]!=3):
            raise ValueError("image must be HxWx3 (or convertible),got shape=%s" % (str(x.shape),))

        if x.dtype!=np.uint8:
            x = x.astype(np.float32)
            if x.size>0:
                mx = float(np.max(x))
            else:
                mx = 0.0
            if mx<=1.0+1e-6:
                x = x*255.0
            x = np.clip(x,0.0,255.0).astype(np.uint8)

        img = Image.fromarray(x,mode="RGB")
        buf = io.BytesIO()

        # Determine the image format and save the image to the buffer
        fmt_u = str(fmt).upper()
        if fmt_u in ["JPG","JPEG"]:
            img.save(buf,format="JPEG",quality=90,optimize=True)
            mime = "image/jpeg"
        else:
            img.save(buf,format="PNG")
            mime = "image/png"

        # Encode the image data in base64
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Note: The "data:" URL scheme is used to include data in-line in web pages as if they were external resources. The format is:
        # data:[<mediatype>][;base64],<data>
        return "data:%s;base64,%s" % (mime,b64)

    def _print_system(self,msg):
        """
        Print a system message block.

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
                display(Markdown("%s  %s" % (tag,msg)))
                return
            except Exception:
                pass
        print("[System] %s" % msg)

    def _print_user(self,msg):
        """
        Print a user message block.

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
                display(Markdown("%s  %s" % (tag,msg)))
                return
            except Exception:
                pass
        print("[User] %s" % msg)

    def _print_assistant(self,msg):
        """
        Print an assistant message block.

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
                display(Markdown("%s\n\n%s" % (tag,msg)))
                return
            except Exception:
                pass
        print("[Assistant] %s" % msg)

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

        Returns:
            Any: Result computed from the current inputs.
        """
        content = [{"type":"input_text","text":text if text is not None else ""}]

        if image is not None:
            content.append({"type":"input_image","image_url":self._img_array_to_data_url(image,fmt="PNG")})

        user_item = {"role":"user","content":content}

        req = dict(
            model        = self.model,
            instructions = self.instructions,
            input        = self.history + [user_item],
        )
        if temperature is not None:
            req["temperature"] = temperature
        if top_p is not None:
            req["top_p"] = top_p

        resp = self.client.responses.create(**req)
        msg  = resp.output_text

        if stack:
            self.history.append(user_item)
            self.history.append({"role":"assistant","content":[{"type":"output_text","text":msg}]})

        if verbose:
            self._print_user(text if text is not None else "")

            if self.notebook and (image is not None) and bool(show_image):
                try:

                    from IPython.display import display,Image as IPyImage
                    img = Image.fromarray(np.asarray(image).astype(np.uint8))
                    buf = io.BytesIO()
                    img.save(buf,format="PNG")
                    display(IPyImage(data=buf.getvalue(),width=width))  # width 픽셀

                except Exception:
                    pass

            self._print_assistant(msg)

        return msg
