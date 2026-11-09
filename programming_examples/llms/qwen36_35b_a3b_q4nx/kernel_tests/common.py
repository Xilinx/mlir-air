# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Shared bits of the Qwen3.6 kernel tests: the Peano compile of one kernel.
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parent
KERNELS = EXAMPLE / "kernels"
sys.path.insert(0, str(EXAMPLE))

PEANO_FLAGS = [
    "-O2",
    "-std=c++20",
    "--target=aie2p-none-unknown-elf",
    "-Wno-parentheses",
    "-Wno-attributes",
    "-Wno-macro-redefined",
    "-Wno-deprecated-declarations",
    "-DNDEBUG",
    "-D__AIE_API_AIE_ADF_HPP__",
]


GDN_ENGINE = EXAMPLE.parents[1] / "fused_decode_gdn"


def compile_kernel(name, extra=(), engine=False):
    """kernels/<name>.cc -> ./<name>.o with Peano. engine=True compiles a
    kernel of fused_decode_gdn's (attention, rope), or one of ours that
    includes it, with that engine's flags and headers: -O1 for attention,
    -O2 for the rest."""
    peano = os.environ["PEANO_INSTALL_DIR"]
    aie_inc = Path(os.environ["MLIR_AIE_INSTALL_DIR"]) / "include"
    # an engine kernel, unless this model has its own file of that name
    local = KERNELS / f"{name}.cc"
    src = local if local.exists() or not engine else GDN_ENGINE / "kernels" / local.name
    if engine:
        extra = [
            "-O1" if name.startswith("attn") else "-O2",
            "-Wno-empty-body",
            "-DMODEL_TYPE=QWEN3_6_35B_A3B",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            "-DDECODE_ROUND_NEAREST",
            f"-I{GDN_ENGINE / 'kernels'}",
            f"-I{GDN_ENGINE / 'models'}",
            *extra,
        ]
    cmd = [
        f"{peano}/bin/clang++",
        *PEANO_FLAGS,
        *extra,
        f"-I{aie_inc}",
        f"-I{KERNELS}",
        "-c",
        str(src),
        "-o",
        f"{name}.o",
    ]
    subprocess.run(cmd, check=True)


def compare(got, want):
    """(cosine, relative L2 error, max error over max |want|) of got against
    want; the L2 error bounds the scale that the cosine does not see."""
    import numpy as np

    g = np.asarray(got, np.float64).ravel()
    w = np.asarray(want, np.float64).ravel()
    return (
        float(g @ w / np.linalg.norm(g) / np.linalg.norm(w)),
        float(np.linalg.norm(g - w) / np.linalg.norm(w)),
        float(np.abs(g - w).max() / np.abs(w).max()),
    )
