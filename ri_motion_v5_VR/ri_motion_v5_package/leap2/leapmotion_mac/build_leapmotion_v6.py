#!/usr/bin/env python3
import subprocess
import sys
import pathlib
import os
import shutil

TIMEOUT_PER_EXAMPLE = 3

DEFAULT_LEAPSDK_DIR = "/Applications/Ultraleap Hand Tracking.app/Contents/LeapSDK"
CACHE_ROOT          = pathlib.Path.home() / "Library" / "Caches" / "leap_build_cache"
CACHE_LEAPSDK_DIR   = CACHE_ROOT / "LeapSDK"

def run(cmd,**kwargs):
    print(">>> %s" % " ".join(cmd))
    subprocess.run(cmd,check=True,**kwargs)

def _copytree_overwrite(src_dir,dst_dir):
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(src_dir,dst_dir)

def _ensure_leapsdk_cached():
    src = pathlib.Path(os.environ.get("LEAPSDK_INSTALL_LOCATION",DEFAULT_LEAPSDK_DIR))
    if not src.exists():
        raise RuntimeError("[leap] LeapSDK dir not found: %s" % str(src))

    _copytree_overwrite(src,CACHE_LEAPSDK_DIR)
    os.environ["LEAPSDK_INSTALL_LOCATION"] = str(CACHE_LEAPSDK_DIR)

def _ensure_buildtime_lib5():
    """
    Build scripts in this repo appear to look for libLeapC.5.dylib
    and/or link with -lLeapC.5. Even if runtime wants 6, we keep a
    libLeapC.5.dylib *in the cached SDK* as a symlink to 6.
    """
    d = CACHE_LEAPSDK_DIR / "lib"
    if not d.exists():
        raise RuntimeError("[leap] Cached SDK lib dir not found: %s" % str(d))

    lib6 = d / "libLeapC.6.dylib"
    lib5 = d / "libLeapC.5.dylib"

    if not lib6.exists():
        raise RuntimeError("[leap] Missing %s" % str(lib6))

    if lib5.exists():
        return

    lib5.symlink_to(lib6.name)

def _safe_copy(src_path,dst_path):
    """
    Copy src -> dst, but avoid SameFileError and redundant work.
    - If dst exists, skip.
    - If src and dst resolve to same real file, skip.
    """
    src_path = pathlib.Path(src_path)
    dst_path = pathlib.Path(dst_path)

    if dst_path.exists():
        return False

    try:
        if src_path.resolve() == dst_path.resolve():
            return False
    except Exception:
        pass

    dst_path.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src_path,dst_path)
    return True

def _stage_runtime_dylibs_for_wheel():
    """
    Root fix (version-6 aligned):
    The extension requires @rpath/libLeapC.6.dylib at runtime.
    Therefore the wheel must contain libLeapC.6.dylib under the python package dir.
    We'll stage dylibs into: leapc-cffi/src/leapc_cffi/
    so they get included in sdist/wheel via MANIFEST.in (or package_data).

    We prioritize shipping:
      - libLeapC.6.dylib (required)
      - libLeapC.dylib (nice-to-have)
      - optionally libLeapC.5.dylib (only if you want backward compat)
    """
    src_lib_dir = CACHE_LEAPSDK_DIR / "lib"
    if not src_lib_dir.exists():
        raise RuntimeError("[leap] Cached SDK lib dir not found: %s" % str(src_lib_dir))

    pkg_src = pathlib.Path("leapc-cffi") / "src" / "leapc_cffi"

    required = src_lib_dir / "libLeapC.6.dylib"
    if not required.exists():
        raise RuntimeError("[leap] Missing %s" % str(required))

    copied_any = False
    copied_any = _safe_copy(required,pkg_src/"libLeapC.6.dylib") or copied_any

    opt0 = src_lib_dir / "libLeapC.dylib"
    if opt0.exists():
        copied_any = _safe_copy(opt0,pkg_src/"libLeapC.dylib") or copied_any

    # Keep 5 only if you want; it doesn't hurt, but not necessary for runtime-6.
    opt5 = src_lib_dir / "libLeapC.5.dylib"
    if opt5.exists():
        copied_any = _safe_copy(opt5,pkg_src/"libLeapC.5.dylib") or copied_any

    # ensure the staged runtime requirement exists
    if not (pkg_src / "libLeapC.6.dylib").exists():
        raise RuntimeError("[leap] Failed to stage libLeapC.6.dylib into %s" % str(pkg_src))

def install_requirements():
    run([sys.executable,"-m","pip","install","-r","requirements.txt"])

def build_leapc_cffi():
    _ensure_leapsdk_cached()
    _ensure_buildtime_lib5()
    _stage_runtime_dylibs_for_wheel()
    run([sys.executable,"-m","build","leapc-cffi"])

def install_leapc_cffi():
    dist_dir = pathlib.Path("leapc-cffi/dist")
    whls     = sorted(dist_dir.glob("*.whl"))
    tars     = sorted(dist_dir.glob("*.tar.gz"))
    pkgs     = whls if whls else tars
    if not pkgs:
        raise RuntimeError("[leap] cannot find any .whl or .tar.gz in leapc-cffi/dist")

    latest = max(pkgs,key=lambda p:p.stat().st_mtime)
    run([sys.executable,"-m","pip","install","--force-reinstall",str(latest)])

def install_python_api():
    run([sys.executable,"-m","pip","install","-e","leapc-python-api"])

def _preflight_import():
    print("\n===== Preflight =====")
    print("PYTHON                  : %s" % sys.executable)
    print("LEAPSDK_INSTALL_LOCATION: %s" % os.environ.get("LEAPSDK_INSTALL_LOCATION","<unset>"))

    try:
        import leapc_cffi  # type: ignore
        pkg_dir = pathlib.Path(leapc_cffi.__file__).resolve().parent
        print("Import leapc_cffi       : OK (%s)" % str(pkg_dir))

        so = pkg_dir / "_leapc_cffi.abi3.so"
        if so.exists():
            print("otool -L _leapc_cffi.abi3.so:")
            subprocess.run(["otool","-L",str(so)],check=False)

        print("Package dylibs:")
        for p in sorted(pkg_dir.glob("libLeapC*.dylib")):
            print(" - %s" % str(p.name))
    except Exception as e:
        print("Import leapc_cffi       : FAIL (%s)" % str(e))
    print("=====================\n")

def test_examples(timeout=TIMEOUT_PER_EXAMPLE):
    ex_dir   = pathlib.Path("examples")
    py_files = sorted(ex_dir.glob("*.py"))
    if not py_files:
        print("Warning: cannot find any .py files in examples/")
        return

    results = []
    print("▶ Running example scripts (timeout=%ds each):" % int(timeout))
    for py in py_files:
        cmd  = [sys.executable,str(py)]
        name = py.name
        try:
            print(" Testing %-30s ... " % name,end="",flush=True)
            subprocess.run(cmd,check=True,timeout=timeout)
        except subprocess.TimeoutExpired:
            print("TIMEOUT (OK)")
            results.append((name,True,""))
        except subprocess.CalledProcessError as e:
            print("FAIL")
            results.append((name,False,"Exited with code %d" % e.returncode))
        else:
            print("OK")
            results.append((name,True,""))

    print("\n===== Test Summary =====")
    passed = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    print("✅ Passed:   %d/%d" % (len(passed),len(results)))
    if failed:
        print("❌ Failed:   %d/%d" % (len(failed),len(results)))
        for name,_,reason in failed:
            print("  - %s: %s" % (name,reason))
        sys.exit(1)
    else:
        print("🎉 All examples ran successfully!")

def main():
    install_requirements()
    build_leapc_cffi()
    install_leapc_cffi()
    install_python_api()
    _preflight_import()
    test_examples()

if __name__ == "__main__":
    main()
