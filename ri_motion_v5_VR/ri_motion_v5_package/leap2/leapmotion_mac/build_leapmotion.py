#!/usr/bin/env python3
import subprocess
import sys
import pathlib

TIMEOUT_PER_EXAMPLE = 3

def run(cmd, **kwargs):
    print(f">>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)

def install_requirements():
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def build_leapc_cffi():
    run([sys.executable, "-m", "build", "leapc-cffi"])

def install_leapc_cffi():
    dist_dir = pathlib.Path("leapc-cffi/dist")
    pkgs = list(dist_dir.glob("*.tar.gz"))
    if not pkgs:
        print("Error: cannot find any zip files in leapc-cffi/dist..")
        sys.exit(1)
    latest = max(pkgs, key=lambda p: p.stat().st_mtime)
    run([sys.executable, "-m", "pip", "install", str(latest)])

def install_python_api():
    run([sys.executable, "-m", "pip", "install", "-e", "leapc-python-api"])

def test_examples(timeout=TIMEOUT_PER_EXAMPLE):
    ex_dir = pathlib.Path("examples")
    py_files = sorted(ex_dir.glob("*.py"))
    if not py_files:
        print("Warning: cannot find any .py files in examples/")
        return

    results = []
    print(f"\n▶ Running example scripts (timeout={timeout}s each):")
    for py in py_files:
        cmd = [sys.executable, str(py)]
        name = py.name
        try:
            print(f" Testing {name:30} ... ", end="", flush=True)
            subprocess.run(cmd,
                           check=True,
                           timeout=timeout,
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            print("TIMEOUT (OK)")
            results.append((name, True, ""))
        except subprocess.CalledProcessError as e:
            print("FAIL")
            results.append((name, False, f"Exited with code {e.returncode}"))
        else:
            print("OK")
            results.append((name, True, ""))

    print("\n===== Test Summary =====")
    passed = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]
    print(f"✅ Passed:   {len(passed)}/{len(results)}")
    if failed:
        print(f"❌ Failed:   {len(failed)}/{len(results)}")
        for name, _, reason in failed:
            print(f"  - {name}: {reason}")
        sys.exit(1)
    else:
        print("🎉 All examples ran successfully!")

def main():
    install_requirements()
    build_leapc_cffi()
    install_leapc_cffi()
    install_python_api()
    test_examples()

if __name__ == "__main__":
    main()
