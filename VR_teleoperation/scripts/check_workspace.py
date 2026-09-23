"""Verify deployed notebook setup offline; never execute live capture or teleop."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_NOTEBOOK = "calibration_notebook/calibrate_intent_axes.ipynb"
NOTEBOOKS = (
    "sim_notebook/sim_vr_teleop.ipynb",
    "real_notebook/real_vr_teleop.ipynb",
    CALIBRATION_NOTEBOOK,
)


def check_calibration_setup(relative, cells):
    """Execute only bootstrap/imports and fresh-capture configuration."""
    setup = ["".join(cell["source"]) for cell in cells[:2]]
    assert len(setup) == 2 and "runpy.run_path" in setup[0], relative
    for source in setup:
        tree = ast.parse(source)
        assert not any(isinstance(node, ast.Call)
                       and ((isinstance(node.func, ast.Name)
                             and node.func.id in ("run_intent_capture", "run_intent_teleop"))
                            or (isinstance(node.func, ast.Attribute)
                                and node.func.attr in ("run_intent_capture", "run_intent_teleop")))
                       for node in ast.walk(tree)), relative

    lookups = (
        ("notebook_directory_lookup", (ROOT / relative).parent),
        ("project_root_lookup", ROOT),
        ("repository_root_lookup", ROOT.parent),
    )
    result = {"path": relative, "setup": "passed", "executed_code_cells": 2}
    for label, directory in lookups:
        os.chdir(directory)
        namespace = {"__name__": "__main__"}
        exec(compile(setup[0], relative + ":code:0", "exec"), namespace)
        assert Path(namespace["PROJECT_DIR"]).resolve() == ROOT
        capture = namespace["run_intent_capture"]
        origin = Path(sys.modules[capture.__module__].__file__).resolve()
        assert origin == ROOT / "package/intent_capture_ui.py", origin
        assert callable(namespace["get_ip_address"])
        # LAN discovery uses a UDP socket; configuration validation stays offline.
        namespace["get_ip_address"] = lambda: "192.0.2.1"
        exec(compile(setup[1], relative + ":code:1", "exec"), namespace)
        expected = {"PC_HOST": "192.0.2.1", "QUEST_PORT": 8443,
                    "PREPARE_S": 3., "CAPTURE_S": 6., "MAX_WRIST_ROTATION_DEG": 30.}
        for name, value in expected.items():
            assert namespace[name] == value, (relative, name, namespace[name])
        assert namespace["RESUME_CAPTURE_PATH"] is None, relative
        result[label] = "passed"
    result.update(fresh_capture_default=True, module_origin=str(origin.relative_to(ROOT)),
                  lan_discovery="stubbed; no network")
    return result


def check_workspace():
    previous_cwd = Path.cwd()
    report = {"root": str(ROOT), "scope": "offline notebook setup; no viewer, Quest or robot transport",
              "notebooks": [], "module_origins": {}}
    try:
        for relative in NOTEBOOKS:
            notebook = json.loads((ROOT / relative).read_text())
            cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
            saved_execution_cells = [index for index, cell in enumerate(cells)
                                     if cell.get("outputs") or cell.get("execution_count") is not None]
            for index, cell in enumerate(cells):
                ast.parse("".join(cell["source"]), filename=relative + ":code:" + str(index))
            if relative == CALIBRATION_NOTEBOOK:
                assert not saved_execution_cells, relative
                report["notebooks"].append(check_calibration_setup(relative, cells))
                continue
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
                "saved_execution_code_cells": saved_execution_cells,
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
        destination = ROOT / ".cache/validation/workspace.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print("PASS: calibration setup, SIM/REAL profiles, FK/IK, local mesh/module isolation")
        print("Report:", destination)
        return report
    finally:
        os.chdir(previous_cwd)


if __name__ == "__main__":
    check_workspace()
