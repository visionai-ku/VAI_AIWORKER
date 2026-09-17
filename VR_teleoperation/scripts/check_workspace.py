"""Verify the two deployed notebooks offline; never execute their live loop."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "sim_notebook/sim_intent_teleop.ipynb",
    "real_notebook/real_intent_teleop.ipynb",
)


def check_workspace():
    previous_cwd = Path.cwd()
    report = {"root": str(ROOT), "scope": "offline notebook setup; no viewer, Quest or robot transport",
              "notebooks": [], "module_origins": {}, "worker_bundle": {}}
    try:
        for relative in NOTEBOOKS:
            notebook = json.loads((ROOT / relative).read_text())
            cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
            for index, cell in enumerate(cells):
                assert not cell.get("outputs") and cell.get("execution_count") is None, relative
                ast.parse("".join(cell["source"]), filename=relative + ":code:" + str(index))
            # Only bootstrap, explicit settings/profile and model/FK/IK preparation.
            assert "runpy.run_path" in "".join(cells[0]["source"])
            assert "settings = IntentTeleopSettings(" in "".join(cells[1]["source"])
            assert "models = build_models(settings)" in "".join(cells[2]["source"])
            for cell in cells[:3]:
                tree = ast.parse("".join(cell["source"]))
                assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                               and node.func.id == "run_intent_teleop" for node in ast.walk(tree))

            # Test lookup from both a notebook folder and the repository root.
            os.chdir((ROOT / relative).parent)
            namespace = {"__name__": "__main__"}
            exec(compile("".join(cells[0]["source"]), relative, "exec"), namespace)
            assert Path(namespace["PROJECT_DIR"]).resolve() == ROOT
            os.chdir(ROOT.parent)
            namespace = {"__name__": "__main__"}
            for cell in cells[:3]:
                exec(compile("".join(cell["source"]), relative, "exec"), namespace)
            assert Path(namespace["PROJECT_DIR"]).resolve() == ROOT
            backend = namespace["settings"].backend
            if backend == "real":
                assert namespace["settings"].DRY_RUN is True
            generated = ROOT / ".cache/fixed_quest" / ("intent_" + backend + "_scene.robot.xml")
            meshes = [Path(node.get("file")).resolve()
                      for node in ET.parse(generated).getroot().iter("mesh")]
            assert meshes and all(p.is_file() and p.is_relative_to(ROOT / "xml") for p in meshes)
            report["notebooks"].append({"path": relative, "setup": "passed",
                "notebook_directory_lookup": "passed", "repository_root_lookup": "passed",
                "local_mesh_references": len(meshes), "profile_id": namespace["profile"]["profile_id"]})

        for name, module in list(sys.modules.items()):
            if any(name == prefix or name.startswith(prefix + ".")
                   for prefix in ("package", "ri_motion_v5_package", "utils")):
                filename = getattr(module, "__file__", None)
                if filename:
                    path = Path(filename).resolve()
                    assert path.is_relative_to(ROOT), (name, path)
                    report["module_origins"][name] = str(path.relative_to(ROOT))
        for relative in ("quest_client/index.html", "quest_client/client.js"):
            assert (ROOT / relative).is_file(), relative
        bundle = ROOT / "deploy/20260917-home-gripper-worker"
        for filename, expected in json.loads((bundle / "SHA256.json").read_text()).items():
            local = hashlib.sha256((ROOT / "package/worker" / filename).read_bytes()).hexdigest()
            packed = hashlib.sha256((bundle / filename).read_bytes()).hexdigest()
            assert local == packed == expected, filename
            report["worker_bundle"][filename] = expected
        destination = ROOT / ".cache/validation/workspace.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print("PASS: both notebook setups, profiles, FK/IK, local mesh/module isolation and worker hashes")
        print("Report:", destination)
        return report
    finally:
        os.chdir(previous_cwd)


if __name__ == "__main__":
    check_workspace()
