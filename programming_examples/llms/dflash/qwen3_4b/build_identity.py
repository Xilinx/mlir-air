# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Reject stale build directories after compiler, dependency or source changes."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parents[2]


def package_version(name):
    # An MLIR wheel unpacked rather than pip-installed has no metadata. The
    # tool digests still pin the toolchain, so record the version as unknown.
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build_identity(kind):
    if kind == "prepass":
        sources = [
            HERE / name
            for name in (
                "build_bfp_prepass.py",
                "build_identity.py",
                "kernels/context_prepass_bfp.cc",
                "kernels/context_prepass.cc",
            )
        ]
    else:
        sources = [
            HERE / name
            for name in (
                "build_target.py",
                "build_identity.py",
                "target_config.py",
                "batch_graph.py",
                "fused_attention_graph.py",
                "selected_config.py",
            )
        ]
        sources += list((HERE / "kernels").glob("*"))
        fused = EXAMPLES / "fused_decode"
        sources += (
            list(fused.glob("*.py"))
            + list((fused / "kernels").glob("*"))
            + list((fused / "models").glob("*"))
        )
        if kind == "baseline":
            sources = [HERE / "build_baseline.py", HERE / "build_identity.py"] + [
                p for p in sources if fused in p.parents
            ]
    tools = {
        "air-opt": Path(shutil.which("air-opt")),
        "clang++": Path(os.environ["PEANO_INSTALL_DIR"]) / "bin/clang++",
        "llvm-link": Path(shutil.which("llvm-link")),
    }
    return dict(
        sources={
            str(p.relative_to(EXAMPLES)): digest(p)
            for p in sorted(sources)
            if p.is_file()
        },
        tools={name: digest(path) for name, path in tools.items()},
        dependencies={
            name: package_version(name) for name in ("mlir_aie", "llvm-aie", "mlir")
        },
    )


def check_existing(directory, identity):
    path = Path(directory) / "manifest.json"
    if not path.exists():
        raise ValueError(f"Incomplete build at {directory}; choose a fresh BUILD_DIR")
    manifest = json.loads(path.read_text())
    if (
        manifest.get("status") != "compiled"
        or manifest.get("build_identity") != identity
    ):
        raise ValueError(
            f"Stale or incomplete build at {directory}; choose a fresh BUILD_DIR"
        )
    if any(
        digest(Path(directory) / name) != want
        for name, want in manifest["artifacts"].items()
    ):
        raise ValueError(
            f"Changed build artifact at {directory}; choose a fresh BUILD_DIR"
        )
    print(f"Build is current: {directory}", flush=True)
