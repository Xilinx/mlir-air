# ./python/air/tools.py -*- Python -*-
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Console entry points for MLIR-AIR native tools.

The wheel installs native tool binaries under ``mlir_air/bin``. These wrappers
make the pip-generated console scripts dispatch to those bundled binaries so a
plain virtual-environment install can run ``air-opt``, ``air-translate``, and
``aircc``, and ``air-runner`` without requiring users to manually edit PATH first.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _package_root() -> Path:
    """Return the installed ``mlir_air`` directory that owns this Python package."""
    # Installed wheel layout: mlir_air/python/air/tools.py
    return Path(__file__).resolve().parents[2]


def _tool_executable_name(tool_name: str) -> str:
    """Return the platform-specific executable name for an MLIR-AIR tool."""
    return f"{tool_name}.exe" if os.name == "nt" else tool_name


def bundled_tool_path(tool_name: str) -> Path:
    """Return the expected path to a bundled native tool in the installed wheel."""
    return _package_root() / "bin" / _tool_executable_name(tool_name)


def resolve_tool(tool_name: str) -> Path:
    """Resolve an MLIR-AIR native tool, preferring the tool bundled in the wheel.

    Installed wheels place native executables under ``mlir_air/bin``. Prefer that
    package-local binary so Python APIs work from IDEs, notebooks, and other
    processes that may not have the virtual environment's script directory on
    PATH. Fall back to PATH for source-tree and developer builds.
    """
    bundled_tool = bundled_tool_path(tool_name)
    if bundled_tool.is_file():
        return bundled_tool

    executable_name = _tool_executable_name(tool_name)
    path_tool = shutil.which(executable_name)
    if path_tool:
        return Path(path_tool)

    raise RuntimeError(
        f"MLIR-AIR tool '{tool_name}' was not found at {bundled_tool} or on PATH"
    )


def _run_tool(tool_name: str, argv: Sequence[str] | None = None) -> int:
    """Run an MLIR-AIR native tool and return its process exit code."""
    args = list(sys.argv[1:] if argv is None else argv)
    completed = subprocess.run([str(resolve_tool(tool_name)), *args], check=False)
    return completed.returncode


def air_opt() -> int:
    """Entry point for the bundled ``air-opt`` binary."""
    return _run_tool("air-opt")


def air_translate() -> int:
    """Entry point for the bundled ``air-translate`` binary."""
    return _run_tool("air-translate")


def aircc() -> int:
    """Entry point for the bundled ``aircc`` binary."""
    return _run_tool("aircc")


def air_runner() -> int:
    """Entry point for the bundled ``air-runner`` binary."""
    return _run_tool("air-runner")
