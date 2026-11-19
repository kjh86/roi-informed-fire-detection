#!/usr/bin/env python3
# roi-informed-fire-detection — Copyright (C) 2026 Jinho Kim and Gye-Young Kim
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE).
"""Register FEPF and MDDM in an installed Ultralytics package (tested with ultralytics==8.4.17).

Ultralytics builds models from YAML through `parse_model` in `ultralytics/nn/tasks.py`, which only
scales the channel arguments of modules listed in its `base_modules` set. This script
  1) copies models/fire_modules.py into ultralytics/nn/modules/fire_modules.py, and
  2) adds FEPF and MDDM to the imports and to `base_modules` in ultralytics/nn/tasks.py.
It is idempotent and writes a backup (tasks.py.bak) before the first modification.

Usage:
    pip install ultralytics==8.4.17
    python tools/install_modules.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import ultralytics

MARK = "# [roi-informed-fire-detection] FEPF/MDDM"
ROOT = Path(ultralytics.__file__).resolve().parent
SRC = Path(__file__).resolve().parents[1] / "models" / "fire_modules.py"


def main() -> None:
    dst = ROOT / "nn" / "modules" / "fire_modules.py"
    shutil.copyfile(SRC, dst)
    print(f"copied {SRC.name} -> {dst}")

    tasks = ROOT / "nn" / "tasks.py"
    backup = tasks.with_suffix(".py.bak")
    # Always start from the pristine file so that the script is idempotent and repairs earlier runs.
    if backup.exists():
        text = backup.read_text(encoding="utf-8")
    else:
        text = tasks.read_text(encoding="utf-8")
        if MARK in text:
            print("tasks.py already patched")
            return
        backup.write_text(text, encoding="utf-8")

    anchor = "base_modules = frozenset(\n        {\n"
    if anchor not in text:
        raise SystemExit("Could not find `base_modules` in tasks.py; please patch it manually (see README).")
    text = text.replace(anchor, anchor + "            FEPF,\n            MDDM,\n", 1)

    # Append the import at the end of the module. parse_model() resolves module names and builds
    # base_modules at call time, so a module-level import placed at the end is sufficient and
    # avoids breaking multi-line import blocks.
    text = text.rstrip("\n") + f"\n\n\nfrom ultralytics.nn.modules.fire_modules import FEPF, MDDM  {MARK}\n"
    tasks.write_text(text, encoding="utf-8")
    print(f"patched {tasks} (backup: {backup.name})")

if __name__ == "__main__":
    main()
