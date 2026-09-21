"""
Build helper for the optional mujoco_sim native backend.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
import warnings
from pathlib import Path

import mujoco
import nanobind


def _native_platform_supported():
    """
    Return True when native extension builds are enabled on this platform.

    Parameters:
        None.

    Returns:
        bool: True on macOS and Linux, False elsewhere.
    """
    return sys.platform == "darwin" or sys.platform.startswith("linux")


def _warn_unsupported_native_platform(action,module_name):
    """
    Warn that native extension support is currently macOS/Linux-only.

    Parameters:
        action (str): Action being skipped.
        module_name (str): Native extension module basename.

    Returns:
        None.
    """
    warnings.warn(
        (
            "mujoco_sim native extension support is currently enabled only on "
            "macOS and Linux. Skipping %s for module [%s] on platform [%s]."
        )%(action,module_name,sys.platform),
        RuntimeWarning,
        stacklevel=2,
    )


def _resolve_cxx():
    """
    Return the C++ compiler command as an argument list.

    Parameters:
        None.

    Returns:
        list[str]: Compiler executable followed by its default flags.
    """
    for candidate in (os.environ.get("CXX"),sysconfig.get_config_var("CXX")):
        if not candidate:
            continue
        # sysconfig may hold flags too, e.g. `g++ -pthread -B <conda>/compiler_compat`.
        cxx = shlex.split(candidate)
        if cxx and shutil.which(cxx[0]):
            return cxx
    return ["c++"]


def _mujoco_shared_library():
    """
    Return the MuJoCo shared library shipped inside the `mujoco` Python package.

    Parameters:
        None.

    Returns:
        Path: Path to `libmujoco.so.*`.
    """
    mujoco_dir = Path(mujoco.__file__).resolve().parent
    candidates = sorted(mujoco_dir.glob("libmujoco.so*"))
    if not candidates:
        raise FileNotFoundError("libmujoco.so* not found under %s" % mujoco_dir)
    return candidates[-1]


def _platform_link_flags():
    """
    Return platform-specific flags for linking a Python extension module.

    Parameters:
        None.

    Returns:
        list[str]: Linker flags.
    """
    if sys.platform == "darwin":
        # Python and MuJoCo symbols are resolved from the host process at import time.
        return ["-bundle","-undefined","dynamic_lookup"]
    return ["-shared"]


def _platform_link_libs():
    """
    Return platform-specific libraries linked after the object files.

    Parameters:
        None.

    Returns:
        list[str]: Linker inputs and flags.
    """
    if sys.platform == "darwin":
        return []
    # On Linux the `mujoco` package loads libmujoco with RTLD_LOCAL, so its
    # symbols are not visible to other extensions. Link the same library
    # explicitly; the loader reuses the already loaded copy via its SONAME.
    mujoco_lib = _mujoco_shared_library()
    return [str(mujoco_lib),"-Wl,-rpath,%s" % mujoco_lib.parent]


def _native_paths(module_name="_mujoco_native"):
    """
    Return native backend build paths.

    Parameters:
        module_name (str): Native extension module basename.

    Returns:
        tuple[Path, Path, Path, Path]: Package, C++ source, build, output paths.
    """
    package_dir = Path(__file__).resolve().parents[1]
    cpp_dir = package_dir / "mujoco_sim" / "cpp"
    build_dir = package_dir / "build" / "native"
    out_dir = package_dir / "mujoco_sim" / "so"
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    out_path = out_dir / ("%s%s" % (module_name,ext_suffix))
    return package_dir,cpp_dir,build_dir,out_path


def _resolve_cpp_dir_files(cpp_dir,names,allowed_suffixes,kind):
    """
    Resolve and validate project-local native source/header filenames.

    Parameters:
        cpp_dir (Path): Native C++ directory.
        names (tuple[str, ...]): Filenames under `cpp_dir`.
        allowed_suffixes (tuple[str, ...]): Allowed file suffixes.
        kind (str): Label for error messages.

    Returns:
        list[Path]: Resolved file paths.
    """
    paths = []
    cpp_dir = cpp_dir.resolve()
    for name in names:
        path = (cpp_dir / name).resolve()
        if path.parent != cpp_dir:
            raise ValueError("%s must be a filename under %s: %s" % (kind,cpp_dir,name))
        if path.suffix not in allowed_suffixes:
            raise ValueError(
                "%s has invalid suffix %s, expected one of %s: %s" % (
                    kind,path.suffix,allowed_suffixes,name
                )
            )
        if not path.exists():
            raise FileNotFoundError("%s does not exist: %s" % (kind,path))
        paths.append(path)
    return paths


def _resolve_define_names(names):
    """
    Validate C++ preprocessor define names.

    Parameters:
        names (tuple[str, ...]): Preprocessor names to pass as `-D...`.

    Returns:
        tuple[str, ...]: Validated names.
    """
    out = []
    for name in names:
        if not name:
            raise ValueError("define name must be non-empty")
        if name[0].isdigit():
            raise ValueError("define name must not start with a digit: %s" % name)
        if not name.replace("_","").isalnum():
            raise ValueError("define name must contain only letters, digits, and _: %s" % name)
        out.append(name)
    return tuple(out)


def _validate_module_name(module_name):
    """
    Validate a Python extension module name that is also a C++ identifier.

    Parameters:
        module_name (str): Module basename without extension suffix.

    Returns:
        str: Validated module name.
    """
    if not module_name:
        raise ValueError("module_name must be non-empty")
    if module_name[0].isdigit():
        raise ValueError("module_name must not start with a digit: %s" % module_name)
    if not module_name.replace("_","").isalnum():
        raise ValueError(
            "module_name must contain only letters, digits, and _: %s" % module_name
        )
    return module_name


def _display_path(path,base_dir):
    """
    Return a compact path for notebook-friendly build logs.

    Parameters:
        path (Path): Path to display.
        base_dir (Path): Preferred directory for relative display.

    Returns:
        str: Relative path when possible, otherwise absolute path.
    """
    path = Path(path)
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return str(path)


def _print_numbered_paths(title,paths,base_dir):
    """
    Print a compact numbered path list.

    Parameters:
        title (str): Section label.
        paths (list[Path]): Paths to print.
        base_dir (Path): Preferred directory for relative display.

    Returns:
        None.
    """
    print("  %s:" % title)
    if not paths:
        print("    (none)")
        return
    for idx,path in enumerate(paths,start=1):
        print("    [%02d] %s" % (idx,_display_path(path,base_dir)))


def _print_numbered_names(title,names):
    """
    Print a compact numbered name list.

    Parameters:
        title (str): Section label.
        names (tuple[str, ...]): Names to print.

    Returns:
        None.
    """
    print("  %s:" % title)
    if not names:
        print("    (none)")
        return
    for idx,name in enumerate(names,start=1):
        print("    [%02d] %s" % (idx,name))


def _native_source_label(path,nanobind_source_path,base_dir):
    """
    Return the readable label used in build progress logs.

    Parameters:
        path (Path): Source file path.
        nanobind_source_path (Path): nanobind runtime source path.
        base_dir (Path): Preferred directory for relative display.

    Returns:
        str: Compact source label.
    """
    if Path(path).resolve() == Path(nanobind_source_path).resolve():
        return "nanobind/src/nb_combined.cpp"
    return _display_path(path,base_dir)


def _reset_loaded_native_module():
    """
    Reset Python-side references to the native extension module.

    Parameters:
        None.

    Returns:
        None.
    """
    module_prefixes = (
        "ri_motion_v5_package.mujoco_sim.so._mujoco_native",
        "ri_motion_v5_package.mujoco_sim.so._mujoco_bind",
        "ri_motion_v5_package.mujoco_sim.so._mujoco_signed_distance_bind",
        "ri_motion_v5_package.mujoco_sim.so._mujoco_ik_bind",
        "ri_motion_v5_package.mujoco_sim.so._mujoco_depenetration_bind",
        "ri_motion_v5_package.mujoco_sim._mujoco_native",
        "ri_motion_v5_package.mujoco_sim._mujoco_bind",
        "ri_motion_v5_package.mujoco_sim._mujoco_signed_distance_bind",
        "ri_motion_v5_package.mujoco_sim._mujoco_ik_bind",
        "ri_motion_v5_package.mujoco_sim._mujoco_depenetration_bind",
    )
    for name in list(sys.modules):
        if any(name == prefix or name.startswith("%s_" % prefix) for prefix in module_prefixes):
            sys.modules.pop(name,None)

    native_helper = sys.modules.get("ri_motion_v5_package.mujoco_sim._native")
    if native_helper is not None:
        native_helper._mujoco_native = None
        native_helper.HAS_NATIVE = False
        native_helper.NATIVE_IMPORT_ERROR = None
        for module_name in list(native_helper._native_modules):
            import_name = "ri_motion_v5_package.mujoco_sim.so.%s" % module_name
            if any(import_name == prefix or import_name.startswith("%s_" % prefix) for prefix in module_prefixes):
                native_helper._native_modules.pop(module_name,None)
        for module_name in list(native_helper._native_import_errors):
            import_name = "ri_motion_v5_package.mujoco_sim.so.%s" % module_name
            if any(import_name == prefix or import_name.startswith("%s_" % prefix) for prefix in module_prefixes):
                native_helper._native_import_errors.pop(module_name,None)


def clean_native(module_name=None,verbose=True,return_removed=False):
    """
    Remove generated native extension and object files from disk.

    Parameters:
        module_name (str | None): If provided, remove only this native extension
            module. If None, remove all `_mujoco_native*`, `_mujoco_bind*`,
            and `_mujoco_*_bind*` extension modules.
        verbose (bool): If True, print removed paths.
        return_removed (bool): If True, return removed paths. If False, return
            None so notebooks do not display the same list twice.

    Returns:
        list[Path] | None: Removed file paths when requested.
    """
    package_dir,_,build_dir,out_path = _native_paths(module_name or "_mujoco_native")
    legacy_out_dir = package_dir / "mujoco_sim"
    removed = []

    if module_name is None:
        extension_paths = sorted(
            list(out_path.parent.glob("_mujoco_native*")) +
            list(out_path.parent.glob("_mujoco_bind*")) +
            list(out_path.parent.glob("_mujoco_*_bind*")) +
            list(legacy_out_dir.glob("_mujoco_native*")) +
            list(legacy_out_dir.glob("_mujoco_bind*")) +
            list(legacy_out_dir.glob("_mujoco_*_bind*"))
        )
    else:
        module_name = _validate_module_name(module_name)
        extension_paths = [
            out_path,
            legacy_out_dir / out_path.name,
        ]

    for path in [*extension_paths,*build_dir.glob("*.o")]:
        if path.exists():
            path.unlink()
            removed.append(path)

    _reset_loaded_native_module()
    if verbose:
        print("[clean_native]")
        print("  module : %s" % (module_name or "_mujoco_native* / _mujoco_bind* / _mujoco_*_bind*"))
        if removed:
            _print_numbered_paths("removed",removed,package_dir)
        else:
            print("  removed:")
            print("    (nothing to remove)")
    if return_removed:
        return removed
    return None

def build_native(
        cpp_source_names,
        hpp_header_names = (),
        define_names     = (),
        module_name      = "_mujoco_native",
        verbose          = True,
        show_commands    = False,
        return_output_path = False,
    ):
    """
    Build the selected native backend under `mujoco_sim/so/`.

    Parameters:
        cpp_source_names (tuple[str, ...]): C++ source filenames under
            `mujoco_sim/cpp` to compile into the native backend.
        hpp_header_names (tuple[str, ...]): Header filenames under
            `mujoco_sim/cpp` that belong to this build. Headers are validated
            and printed, but they are included by `.cpp` files rather than
            compiled directly.
        define_names (tuple[str, ...]): C++ preprocessor names to pass as
            `-D...`.
        module_name (str): Python extension module basename to build.
        verbose (bool): If True, print a readable build summary.
        show_commands (bool): If True, print full compile and link commands.
        return_output_path (bool): If True, return the built extension path.
            If False, return None so notebooks do not display the path twice.

    Returns:
        Path | None: Built extension path when requested.
    """
    module_name = _validate_module_name(module_name)
    package_dir,cpp_dir,build_dir,out_path = _native_paths(module_name)

    if not _native_platform_supported():
        if verbose:
            print("[build_native]")
            print("  module      : %s" % module_name)
            print("  platform    : %s" % sys.platform)
            print("  status      : skipped; native builds are currently macOS/Linux-only")
        _warn_unsupported_native_platform("build_native",module_name)
        return None

    build_dir.mkdir(parents=True,exist_ok=True)
    out_path.parent.mkdir(parents=True,exist_ok=True)

    include_dirs = [
        sysconfig.get_paths()["include"],
        nanobind.include_dir(),
        str(Path(nanobind.__file__).resolve().parent / "ext" / "robin_map" / "include"),
        str(Path(mujoco.__file__).resolve().parent / "include"),
        str(cpp_dir),
    ]

    cxx = _resolve_cxx()
    cpp_source_paths = _resolve_cpp_dir_files(
        cpp_dir,cpp_source_names,(".cpp",".cc",".cxx"),"C++ source"
    )
    hpp_header_paths = _resolve_cpp_dir_files(
        cpp_dir,hpp_header_names,(".hpp",".hh",".hxx",".h"),"C++ header"
    )
    define_names = _resolve_define_names(define_names)
    compile_define_names = (*define_names,"RI_NATIVE_MODULE_NAME=%s" % module_name)
    nanobind_source_path = Path(nanobind.__file__).resolve().parent / "src" / "nb_combined.cpp"
    sources = cpp_source_paths + [nanobind_source_path]

    if verbose:
        print("[build_native]")
        print("  module      : %s" % module_name)
        print("  output      : %s" % _display_path(out_path,package_dir))
        print("  compiler    : %s" % " ".join(cxx))
        _print_numbered_paths("project cpp",cpp_source_paths,package_dir)
        _print_numbered_paths("project hpp",hpp_header_paths,package_dir)
        _print_numbered_names("defines",compile_define_names)
        print("  nanobind cpp:")
        print("    [01] nanobind/src/nb_combined.cpp")
        if not show_commands:
            print("  commands    : hidden; pass show_commands=True to print full commands")

    objects = []
    if verbose:
        print("[build_native] compile")
    for src in sources:
        obj = build_dir / ("%s.o" % src.stem)
        cmd = [
            *cxx,
            "-std=c++17",
            "-O2",
            "-fPIC",
            "-fvisibility=hidden",
            "-fno-strict-aliasing",
            "-DNDEBUG",
            "-DNB_COMPACT_ASSERTIONS",
        ]
        for name in compile_define_names:
            cmd.append("-D%s" % name)
        for include_dir in include_dirs:
            cmd.extend(["-I",str(include_dir)])
        cmd.extend(["-c",str(src),"-o",str(obj)])
        if verbose:
            step = len(objects) + 1
            print(
                "  [%02d/%02d] %s -> %s" % (
                    step,
                    len(sources),
                    _native_source_label(src,nanobind_source_path,package_dir),
                    _display_path(obj,package_dir),
                )
            )
        if show_commands:
            print("[compile] %s" % " ".join(shlex.quote(part) for part in cmd))
        subprocess.run(cmd,check=True)
        objects.append(obj)

    link_cmd = [
        *cxx,
        *_platform_link_flags(),
        *[str(obj) for obj in objects],
        "-o",
        str(out_path),
        *_platform_link_libs(),
    ]
    if verbose:
        print("[build_native] link")
        print("  output: %s" % _display_path(out_path,package_dir))
    if show_commands:
        print("[link] %s" % " ".join(shlex.quote(part) for part in link_cmd))
    subprocess.run(link_cmd,check=True)
    if verbose:
        print("[build_native] done")
    if return_output_path:
        return out_path
    return None


def main():
    """
    Build the native backend from the command line with explicit file names.

    Parameters:
        None.

    Returns:
        None.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp",dest="cpp_source_names",action="append",required=True)
    parser.add_argument("--hpp",dest="hpp_header_names",action="append",default=[])
    parser.add_argument("--define",dest="define_names",action="append",default=[])
    parser.add_argument("--module-name",dest="module_name",default="_mujoco_native")
    parser.add_argument("--clean",dest="clean",action="store_true")
    parser.add_argument("--show-commands",dest="show_commands",action="store_true")
    parser.add_argument("--return-output-path",dest="return_output_path",action="store_true")
    args = parser.parse_args()

    if args.clean:
        clean_native(module_name=args.module_name,verbose=True)

    build_native(
        cpp_source_names = tuple(args.cpp_source_names),
        hpp_header_names = tuple(args.hpp_header_names),
        define_names     = tuple(args.define_names),
        module_name      = args.module_name,
        verbose          = True,
        show_commands    = args.show_commands,
        return_output_path = args.return_output_path,
    )


if __name__ == "__main__":
    main()
