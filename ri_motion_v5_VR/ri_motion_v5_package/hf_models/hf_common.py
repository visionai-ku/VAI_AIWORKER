"""
hf_common.py
"""

import os
import sys
import torch
import json,re,ast
import subprocess

from huggingface_hub import snapshot_download

def ensure_pip_packages(
        packages,
        verbose      = True,
        upgrade      = False,
        import_names = None,
    ):
    """
    Ensure pip packages are installed.

    Parameters:
        packages (Any): Input packages.
        verbose (bool): Input verbose. Defaults to `True`.
        upgrade (bool): Input upgrade. Defaults to `False`.
        import_names (Any | None): Input import names. Defaults to `None`.

    Returns:
        None: This function mutates state, renders output, or performs side
              effects in place.
    """
    if isinstance(packages,str):
        packages = [packages]
    packages = list(packages)

    if import_names is None:
        import_names = {}

    for pkg in packages:
        mod_name = import_names.get(pkg,pkg)

        is_ok = True
        try:
            __import__(mod_name)
        except Exception:
            is_ok = False

        if is_ok:
            if verbose:
                print("[ensure_pip_packages] already installed:[%s]"%pkg)
            continue

        cmd = [sys.executable,"-m","pip","install"]
        if bool(upgrade):
            cmd.append("--upgrade")
        cmd.append(pkg)

        if verbose:
            print("[ensure_pip_packages] installing:[%s]"%pkg)

        subprocess.check_call(cmd)

    if verbose:
        print("[ensure_pip_packages] done.")

def ensure_local_ckpt(
        ckpt,
        local_dir,
        revision               = None,
        verbose                = True,
        resume_download        = True,
        local_dir_use_symlinks = False,
    ):
    """
    Ensure that the checkpoint is available locally.

    Parameters:
        ckpt (Any): Input ckpt.
        local_dir (str | Path): Input local dir.
        revision (Any | None): Input revision. Defaults to `None`.
        verbose (bool): Input verbose. Defaults to `True`.
        resume_download (bool): Input resume download. Defaults to `True`.
        local_dir_use_symlinks (bool): Input local dir use symlinks. Defaults to
                                       `False`.

    Returns:
        Any: Result computed from the current inputs.
    """
    if isinstance(ckpt,str) and os.path.isdir(ckpt):
        if verbose:
            print("[ensure_local_ckpt] using local ckpt:[%s]"%ckpt)
        return ckpt
    cfg_path = os.path.join(local_dir,"config.json")

    # If config.json doesn't exist, we assume local_dir is not complete and needs download.
    if not os.path.isfile(cfg_path):
        if verbose:
            print("[ensure_local_ckpt] downloading [%s] -> [%s]"%(ckpt,local_dir))
        # HuggingFace snapshot_download will skip download if local_dir already has the same revision (or default branch) downloaded.
        snapshot_download(
            repo_id                = ckpt,
            revision               = revision,
            local_dir              = local_dir,
            local_dir_use_symlinks = bool(local_dir_use_symlinks),
            resume_download        = bool(resume_download),
        )
    else:
        if verbose:
            print("[ensure_local_ckpt] found local_dir:[%s]"%local_dir)

    return local_dir

def hf_hub_download_if_needed(
        repo_id,
        filename,
        local_dir = None,
        revision  = None,
        verbose   = True,
    ):
    """
    Download a single file from Hugging Face Hub and return local path.

    Parameters:
        repo_id (Any): Input repo id.
        filename (str): Input filename.
        local_dir (Any | None): Input local dir. Defaults to `None`.
        revision (Any | None): Input revision. Defaults to `None`.
        verbose (bool): Input verbose. Defaults to `True`.

    Returns:
        Any: Result computed from the current inputs.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id   = repo_id,
        filename  = filename,
        local_dir = local_dir,
        revision  = revision,
    )
    if verbose:
        print("[hf_hub_download_if_needed] repo:[%s] file:[%s]"%(repo_id,filename))
        print("[hf_hub_download_if_needed] path:[%s]"%path)
    return path

def get_device(device=None,verbose=True):
    """
    Get the appropriate device for PyTorch operations.

    Parameters:
        device (Any | None): Input device. Defaults to `None`.
        verbose (bool): Input verbose. Defaults to `True`.

    Returns:
        Any: Requested value computed from the current inputs.
    """
    if device is not None:
        if verbose: print("Using specified device:[%s]"%(device))
        return device
    if torch.cuda.is_available():
        if verbose: print("Using [CUDA] device")
        return "cuda"
    mps_ok = (hasattr(torch.backends,"mps") and torch.backends.mps.is_available())
    if mps_ok:
        if verbose: print("Using [MPS] device")
        return "mps"
    if verbose: print("Using [CPU] device")
    return "cpu"


def extract_objects_json(raw,n_max=3):
    """
    Extract objects from a JSON-like string.

    Parameters:
        raw (Any): Input raw.
        n_max (int): Input n max. Defaults to `3`.

    Returns:
        Any: Result computed from the current inputs.
    """
    s = raw if isinstance(raw,str) else str(raw)
    s = s.strip()

    # Remove common prefixes like "raw:", "response:", "output:"
    s = re.sub(r"^(raw|response|output)\s*:\s*","",s,flags=re.IGNORECASE).strip()

    # Remove code block markers ``` or ```json
    s = re.sub(r"^```(json)?\s*","",s,flags=re.IGNORECASE).strip()
    s = re.sub(r"\s*```$","",s).strip()

    # direct JSON
    try:
        j = json.loads(s)
        if isinstance(j,dict) and isinstance(j.get("objects",None),list):
            return [str(x).strip() for x in j["objects"] if str(x).strip()!=""][:int(n_max)]
    except Exception:
        pass

    # Extract JSON objects greedily (from the first '{' to the last '}')
    m = re.search(r"\{.*\}",s,flags=re.DOTALL)
    if m is not None:
        ss = m.group(0).strip()

        try:
            j = json.loads(ss)
            if isinstance(j,dict) and isinstance(j.get("objects",None),list):
                return [str(x).strip() for x in j["objects"] if str(x).strip()!=""][:int(n_max)]
        except Exception:
            pass

        # Cover Python dict representation as well (just in case)
        try:
            j = ast.literal_eval(ss)
            if isinstance(j,dict) and isinstance(j.get("objects",None),list):
                return [str(x).strip() for x in j["objects"] if str(x).strip()!=""][:int(n_max)]
        except Exception:
            pass

    # Return empty list if no valid objects found
    return []

# -------------------------------------------------------------------------