# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Reject stale build directories after compiler, dependency or source changes."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil

COMMON = Path(__file__).resolve().parent
EXAMPLES = COMMON.parents[2]


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


def build_identity(kind, model_dir):
    model_dir = Path(model_dir).resolve()
    if kind == "prepass":
        sources = [
            model_dir / "build_bfp_prepass.py",
            COMMON / "build_prepass.py",
            COMMON / "kernels/context_prepass_bfp.cc",
            COMMON / "kernels/context_prepass.cc",
        ]
    else:
        sources = [
            model_dir / name
            for name in (
                "build_target.py",
                "target_config.py",
                "batch_graph.py",
                "owner_graph.py",
                "selected_config.py",
            )
        ]
        sources += list((COMMON / "kernels").glob("*"))
        fused = EXAMPLES / "fused_decode"
        sources += (
            list(fused.glob("*.py"))
            + list((fused / "kernels").glob("*"))
            + list((fused / "models").glob("*"))
        )
        if kind == "baseline":
            sources = [
                model_dir / "build_baseline.py",
                COMMON / "build_identity.py",
            ] + [p for p in sources if fused in p.parents]
    sources += [COMMON / "build_identity.py", model_dir / "paths.py"]
    if kind == "decoder":
        sources += [
            COMMON / name for name in ("fused_attention_graph.py", "buffers.py")
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
