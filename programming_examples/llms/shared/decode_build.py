# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build model decode kernels with AIR's shared dependency/flag tracking rules."""

from pathlib import Path
import shlex


def compile_kernels(out, source, flags, source_dirs, run, env):
    """Preserve the accepted O2 objects / O1 inline-attention split."""
    out, source = Path(out).resolve(), Path(source).resolve()
    makefile = out / "decode-kernels.mk"
    makefile.write_text(
        "# Generated from the model build configuration.\n"
        f"KERNEL_DIR := {out}\n"
        f"KERNEL_SRC_DIRS := {' '.join(map(str, source_dirs))}\n"
        f"KERNEL_FLAGS := {shlex.join(list(map(str, flags)))}\n"
        "O2_KERNELS := proj_qmm rms_residual glu rope\n"
        "LL_KERNELS := attn_qk attn_kv\n"
        "LL_OPT := O1\n"
        f"include {source / 'decode_kernels.mk'}\n"
    )
    return run(
        ["make", "-f", makefile, "decode-kernels"], out, out / "decode-kernels.log", env
    )
