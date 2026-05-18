# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Console-script shims for the native air-* binaries bundled in the wheel.

Pip's ``[project.scripts]`` entry points must reference Python callables.
These helpers exec the matching native binary out of the wheel's ``bin/``
directory so commands like ``air-opt`` resolve transparently after a
``pip install mlir-air``.
"""

from __future__ import annotations

import os
import sys
from typing import NoReturn

__all__ = [
    "MLIR_AIR_BIN_DIR",
    "air_opt",
    "air_translate",
]


# Wheel layout: site-packages/mlir_air/python/air/tools/__init__.py
# Binaries:     site-packages/mlir_air/bin/<tool>
MLIR_AIR_BIN_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "bin")
)
assert os.path.isdir(
    MLIR_AIR_BIN_DIR
), f"MLIR-AIR tools directory does not exist: {MLIR_AIR_BIN_DIR}"


def _exec(name: str, *args: str) -> NoReturn:
    exe = os.path.join(MLIR_AIR_BIN_DIR, name)
    if sys.platform.startswith("win"):
        import subprocess

        raise SystemExit(subprocess.call([exe, *args], close_fds=False))
    os.execl(exe, exe, *args)


def air_opt() -> NoReturn:
    _exec("air-opt", *sys.argv[1:])


def air_translate() -> NoReturn:
    _exec("air-translate", *sys.argv[1:])
