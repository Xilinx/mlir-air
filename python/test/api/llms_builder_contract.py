# ./python/test/api/llms_builder_contract.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s %air_src_root | FileCheck %s

"""The builders under programming_examples/llms/ import must return a module.

An air.api example normally ends `build_module` with `return launch`, handing
back the LaunchContext and leaving `.build()` to its own `main`. That is fine
for an example nobody imports, and it is what ~78 of them do. It is wrong for
the handful that llms/ imports: those hand the result straight to
`KernelCache.compile_and_cache`, which writes `str(mlir_module)` into air.mlir.
A LaunchContext stringifies to `<air.api._compile.LaunchContext object at 0x..>`
and aircc rejects it as `air.mlir:1:1: expected operation name in quotes` -- a
diagnostic that names neither the importer nor the builder.

That is how #1927 reddened the nightly for lfm2_1_2b_q4nx. It could not be
caught on the PR: llms/ runs under `check-programming-examples-llms-*`, which
only nightlyPerfBenchmark invokes, and #1927 touched no llms/ file. So the
check lives here instead, in a PR-gated suite, and it needs no NPU -- building
the module is enough, because the failure is in the return type and not in the
IR.

Adding an import to llms/ therefore means adding a row below. The list is the
contract, and it is short on purpose.
"""

import sys

sys.path[:0] = [
    sys.argv[1] + "/programming_examples",
    sys.argv[1] + "/programming_examples/llms",
]

import air.ir
from ml_dtypes import bfloat16

# Each row: (module path, builder, kwargs, the llms/ file that imports it).
# Configs are the smallest legal ones, not the ones llms/ passes -- the return
# type does not depend on the shape, and a prefill-scale build is slow.
BUILDERS = [
    (
        "conv1d_depthwise.conv1d_depthwise",
        "build_module",
        dict(seq=128, channels=256, tile_s=8, np_dtype_in=bfloat16, herd_x=8, herd_y=1),
        "lfm2_1_2b_q4nx/lfm2_1_2b_q4nx_prefill.py",
    ),
    (
        "layer_norm.layer_norm",
        "build_module",
        dict(M=8, N=128),
        "shared/builders (smolvla)",
    ),
    (
        "weighted_rms_norm.weighted_rms_norm",
        "build_module",
        dict(M=8, N=128),
        "shared/infra",
    ),
    (
        "gelu_and_mul.gelu_and_mul",
        "build_module_2d",
        dict(rows=8, cols=256, tile_n=32),
        "shared/infra",
    ),
    (
        "silu_and_mul.silu_and_mul",
        "build_module_2d",
        dict(rows=8, cols=256, tile_n=32),
        "shared/infra",
    ),
    (
        "flash_attention.kernel_fusion_based.attn_npu2",
        "build_module",
        dict(lk=512, lkp=64, lq=512, lqp=256, dk=64, dv=64),
        "shared/infra/fa_headfirst.py",
    ),
]


def check(modpath, fnname, kwargs, importer):
    mod = __import__(modpath, fromlist=[fnname])
    built = getattr(mod, fnname)(**kwargs)
    text = str(built)
    # Assert it is non-empty before asking whether it looks like MLIR: the
    # failure mode this guards is a one-line object repr, and "does not contain
    # air.launch" would also be true of the empty string.
    assert text.strip(), f"{modpath}.{fnname} stringified to nothing"
    # Reparse rather than pattern-match: this is exactly what aircc does with
    # the file, so a text that survives it is a text aircc will accept. Pinning
    # a substring instead would be a guess about which air ops the builder is
    # expected to emit, which is not what this test is about -- layer_norm has
    # no air.launch at all.
    ok = isinstance(built, air.ir.Module)
    if ok:
        with air.ir.Context(), air.ir.Location.unknown():
            air.ir.Module.parse(text)
    print(
        f"{modpath}.{fnname}: {'MLIR' if ok else 'NOT MLIR'} "
        f"({type(built).__name__}, {len(text.splitlines())} lines) <- {importer}"
    )
    return ok


# CHECK: conv1d_depthwise.conv1d_depthwise.build_module: MLIR
# CHECK: layer_norm.layer_norm.build_module: MLIR
# CHECK: weighted_rms_norm.weighted_rms_norm.build_module: MLIR
# CHECK: gelu_and_mul.gelu_and_mul.build_module_2d: MLIR
# CHECK: silu_and_mul.silu_and_mul.build_module_2d: MLIR
# CHECK: flash_attention.kernel_fusion_based.attn_npu2.build_module: MLIR
# CHECK: 6/6 builders return a module
bad = [row for row in BUILDERS if not check(*row)]
assert not bad, f"not a module: {[f'{m}.{f}' for m, f, _, _ in bad]}"
print(f"{len(BUILDERS)}/{len(BUILDERS)} builders return a module")
