"""Build a self-contained Windows x64 directory with PyInstaller."""

import ast
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Indica el directorio de salida.")
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--onedir",
        "--name", "HeadControlAI", "--distpath", str(output),
        "--workpath", str(output / "_pyinstaller_work"), "--specpath", str(output),
        "--collect-binaries", "mediapipe",
    ]
    for name in ("cv2", "numpy", "mediapipe", "onnxruntime", "pyautogui", "joblib",
                 "PySide6.QtWidgets", "PySide6.QtGui", "PySide6.QtCore", "csv", "ctypes",
                 "hashlib", "inspect", "logging", "uuid", "json", "dataclasses"):
        cmd += ["--hidden-import", name]
    for name in ("scipy", "torch", "pandas", "sklearn", "pytest", "tkinter",
                 "PySide6.QtWebEngineCore", "IPython", "IPython.core", "jupyter"):
        cmd += ["--exclude-module", name]
    stdlib = set()
    for folder in ("product/mvp", "product/training", "product/experiments", "prueba_ojos", "v2/assets"):
        for file in sorted((ROOT / folder).rglob("*")):
            if not file.is_file() or file.name.startswith("verify_") or file.suffix not in (".py", ".onnx", ".json", ".task"):
                continue
            if file.suffix == ".py":
                for node in ast.walk(ast.parse(file.read_text(encoding="utf-8-sig"))):
                    if isinstance(node, ast.Import):
                        stdlib.update(alias.name.split(".")[0] for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                        stdlib.add(node.module.split(".")[0])
            dest = file.parent.relative_to(ROOT).as_posix()
            cmd += ["--add-data", f"{file}{os.pathsep}{dest}"]
    for name in sorted(stdlib & sys.stdlib_module_names - {"tkinter"}):
        cmd += ["--hidden-import", name]
    cmd.append(str(ROOT / "packaging/windows_entry.py"))
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
