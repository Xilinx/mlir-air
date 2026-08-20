#!/usr/bin/env python3
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Repoint a staged LLVM distro wheel's diaguids.lib at the local DIA SDK.

The wheel is built on a machine with its own Visual Studio, and
LLVMExports.cmake bakes in that machine's absolute path to diaguids.lib -- the
DIA SDK import library LLVMDebugInfoPDB links against. Every host whose Visual
Studio edition, version or install location differs then fails at link time
with a bare "missing and no known rule to make it", naming a path that has
nothing to do with the local install.

Rewrites the recorded path to this machine's. Idempotent, and a no-op off
Windows.

    python utils/fixup-llvm-diaguids.py my_install/mlir
"""

import os
import pathlib
import re
import subprocess
import sys

# Matches the absolute path only, leaving the rest of the ;-separated
# INTERFACE_LINK_LIBRARIES list alone.
DIAGUIDS_RE = re.compile(r"[A-Za-z]:[^\";]*diaguids\.lib", re.IGNORECASE)


def visual_studio_dirs():
    """Every installed Visual Studio, newest first.

    -products * is required: vswhere's default product filter covers
    Community/Professional/Enterprise and silently excludes Build Tools, which
    is one of the two installs this repository's Windows guide recommends.
    """
    vswhere = pathlib.Path(
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        "Microsoft Visual Studio",
        "Installer",
        "vswhere.exe",
    )
    if not vswhere.is_file():
        return []
    out = subprocess.run(
        [str(vswhere), "-all", "-products", "*", "-property", "installationPath"],
        capture_output=True,
        text=True,
    ).stdout
    return [pathlib.Path(line.strip()) for line in out.splitlines() if line.strip()]


def find_diaguids():
    """The first installed diaguids.lib, or None."""
    for vs in visual_studio_dirs():
        lib = vs / "DIA SDK" / "lib" / "amd64" / "diaguids.lib"
        if lib.is_file():
            return lib
    return None


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    if os.name != "nt":
        return 0

    exports = pathlib.Path(argv[1]) / "lib" / "cmake" / "llvm" / "LLVMExports.cmake"
    if not exports.is_file():
        print(f"error: {exports} not found; is that the staged mlir directory?")
        return 1

    text = exports.read_text(encoding="utf-8")
    recorded = DIAGUIDS_RE.findall(text)
    if not recorded:
        print("No diaguids.lib path recorded; nothing to do.")
        return 0
    if all(pathlib.Path(p).is_file() for p in recorded):
        print(f"diaguids.lib already resolves: {recorded[0]}")
        return 0

    local = find_diaguids()
    if local is None:
        print(
            "error: no diaguids.lib found in any Visual Studio install.\n"
            "The DIA SDK ships with Visual Studio; install it, or the C++\n"
            "workload if Visual Studio is present without it."
        )
        return 1

    exports.write_text(DIAGUIDS_RE.sub(local.as_posix(), text), encoding="utf-8")
    print(f"diaguids.lib: {recorded[0]}\n          ->  {local.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
