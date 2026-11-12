# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Shared bits of the Qwen3.8 kernel tests: the Peano compile of one kernel,
# and the element-wise gate every test passes or fails on.
#
# A kernel comes from this example's kernels/ when it has one there, else from
# Qwen3.6-35B-A3B's (the same layer kinds), built for this model's sizes.
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXAMPLE = HERE.parent
KERNELS = EXAMPLE / "kernels"
Q36_KERNELS = EXAMPLE.parent / "qwen36_35b_a3b_q4nx" / "kernels"
# this model's sizes for the kernels shared with Qwen3.6
MODEL_FLAGS = ["-DGDN_NV=48", "-DLM_DIM=5120"]
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
    if not local.exists() and (Q36_KERNELS / local.name).exists():
        local = Q36_KERNELS / local.name
    src = local if local.exists() or not engine else GDN_ENGINE / "kernels" / local.name
    if engine:
        extra = [
            "-O1" if name.startswith("attn") else "-O2",
            "-Wno-empty-body",
            "-DMODEL_TYPE=QWEN3_8_27B",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            "-DDECODE_ROUND_NEAREST",
            f"-I{GDN_ENGINE / 'kernels'}",
            f"-I{GDN_ENGINE / 'models'}",
            *extra,
        ]
    cmd = [
        f"{peano}/bin/clang++",
        *PEANO_FLAGS,
        *MODEL_FLAGS,
        *extra,
        f"-I{aie_inc}",
        f"-I{KERNELS}",
        f"-I{Q36_KERNELS}",
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


# the canonical bf16 rtol (kernel_registry/README.md)
RTOL = 1.6e-2


def check(name, got, want, atol, rtol=RTOL):
    """The gate: every element of got within atol + rtol |want| of want, atol
    a fraction of max |want| (the inputs are random), and every element
    finite. The cosine and relative L2 error are printed, not gated on."""
    import numpy as np

    g = np.asarray(got, np.float64).ravel()
    w = np.asarray(want, np.float64).ravel()
    finite = bool(np.isfinite(g).all())
    scale = np.abs(w).max()
    # the atol, over max |want|, that the worst element needs at this rtol
    need = max(float((np.abs(g - w) - rtol * np.abs(w)).max()), 0.0) / scale
    good = finite and need <= atol
    c, l2, _ = compare(g, w) if finite else (float("nan"),) * 3
    print(
        f"{name}: needs atol {need:.2e} max|want| at rtol {rtol:g}"
        f" (limit {atol:.1e}); corr {c:.7f} rel L2 {l2:.2e}"
        f"{'' if finite else ' NON-FINITE'} {'ok' if good else 'BAD'}"
    )
    return good
