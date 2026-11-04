# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Shared bits of the GPT-OSS kernel tests: the Peano compile of one kernel and
# the reference weights.
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
    "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
]


FUSED_DECODE = EXAMPLE.parents[1] / "fused_decode"
MOE_KERNELS = EXAMPLE.parent / "shared" / "moe" / "kernels"
# the shared MoE kernels, for this model's experts
MOE_FLAGS = ["-DMOE_N_EXPERTS=32", "-DMOE_TOP_K=4"]


def compile_kernel(name, extra=(), shared=False, model=False, moe=False):
    """kernels/<name>.cc -> ./<name>.o with Peano. shared=True builds the dense
    kernel fused_decode/kernels/<name>.cc that the other LLMs use, for the
    gpt-oss-20b model header; model=True builds one of ours against that
    header; moe=True builds shared/moe/kernels/<name>.cc."""
    peano = os.environ["PEANO_INSTALL_DIR"]
    aie_inc = Path(os.environ["MLIR_AIE_INSTALL_DIR"]) / "include"
    fused = FUSED_DECODE / "kernels"
    src = (fused if shared else MOE_KERNELS if moe else KERNELS) / f"{name}.cc"
    if moe:
        extra = [*MOE_FLAGS, *extra]
    if shared or model:
        extra = ["-DMODEL_TYPE=GPT_OSS_20B", f"-I{FUSED_DECODE / 'models'}", *extra]
    cmd = [
        f"{peano}/bin/clang++",
        *PEANO_FLAGS,
        *extra,
        f"-I{aie_inc}",
        f"-I{KERNELS}",
        f"-I{MOE_KERNELS}",
        f"-I{fused}",
        "-c",
        str(src),
        "-o",
        f"{name}.o",
    ]
    subprocess.run(cmd, check=True)
    return f"{name}.o"
