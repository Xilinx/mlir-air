# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The decode's LM head: final RMSNorm and the BF16 tied head in one ELF."""

import os
from pathlib import Path
import re
import shutil
import time
from .common import ROOT, save_json, file_hash


def row_module(m, object_path, width=2048):
    from air import api as air
    from air.api import ops

    # Single-row form has a 1D ABI for the fused final norm/head.
    shape = [width] if m == 1 else [m, width]
    x = air.tensor(shape, air.bf16)
    w = air.tensor([width], air.bf16)
    y = air.tensor(shape, air.bf16)
    fn = air.extern(
        "rms_norm_aie",
        link_with=str(Path(object_path).resolve()),
        scalars=[air.i32],
    )
    cores = min(m, 8)
    with air.launch(name="openelm_norm") as launch:

        @launch.body
        def _():
            with air.herd([range(cores)], shape=(cores,)) as herd:

                @herd.body
                def _(tx):
                    a = air.alloc([width], air.bf16, scope=herd.private())
                    b = air.alloc([width], air.bf16, scope=herd.private())
                    out = air.alloc([width], air.bf16, scope=herd.private())
                    ops.load(b, w[:])
                    for i in air.sequential(m // cores):
                        row = tx * (m // cores) + i
                        ops.load(a, x[:] if m == 1 else x[row : row + 1, :])
                        fn(out, a, b, 0)
                        ops.store(out, y[:] if m == 1 else y[row : row + 1, :])

    return launch.build(target="npu2")


def compile_norm_objects(out):
    from .build import run

    src = ROOT / "programming_examples/fused_decode"
    local = Path(__file__).resolve().parent / "kernels"
    base = [
        Path(os.environ["PEANO_INSTALL_DIR"]) / "bin/clang++",
        "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        "-O2",
        "-c",
        "-DNDEBUG",
        "-DMODEL_TYPE=OPENELM_1B",
        "-D__AIE_API_AIE_ADF_HPP__",
        "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        "-I",
        Path(os.environ["MLIR_AIE_INSTALL_DIR"]) / "include",
        "-I",
        src / "kernels",
        "-I",
        src / "models",
    ]
    path = local / "rms_residual.cc"
    run(base + [path, "-o", out / "norm.o"], out, out / "norm.log")


def compile_one(cache, out, name, module, **kw):
    (symbol,) = re.findall(r"func\.func @(\w+)\(", str(module))
    kw.setdefault("instance_name", symbol)
    (out / f"{name}.mlir").write_text(str(module))
    cache.compile_and_cache(
        name,
        module,
        dict(verbose=True, output_format="elf", omit_while_true_loop=False, **kw),
    )
    cache._save_manifest()
    shutil.copytree(out / "air_project", out / f"ir_{name}", dirs_exist_ok=True)


def compile_head(out):
    from ..shared.infra.cache import KernelCache
    from ..shared.infra.stitching import (
        stitch_elf,
        KernelSlice,
        FuncArg,
    )
    from ..shared.builders.lm_head_gemv_multi import (
        build_lm_head_gemv_module,
    )

    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    start = time.perf_counter()
    compile_norm_objects(out)
    cache = KernelCache(cache_dir=str(out / "cache"), verbose=True)
    norm = str(row_module(1, out / "norm.o"))
    head = str(build_lm_head_gemv_module(2048, n_partitions=2))
    args = [
        FuncArg(f"%arg{i}", t)
        for i, t in enumerate(
            [
                "memref<2048xbf16>",
                "memref<2048xbf16>",
                "memref<16384x2048xbf16>",
                "memref<16384xbf16>",
                "memref<16384x2048xbf16>",
                "memref<16384xbf16>",
                "memref<2048xbf16>",
            ]
        )
    ]
    mod = stitch_elf(
        "openelm_head",
        args,
        [
            KernelSlice(
                norm, "norm", {0: 0, 1: 1, 2: 6}, extern_syms={"@rms_norm_aie"}
            ),
            KernelSlice(
                head,
                "head",
                {0: 6, 1: 2, 2: 3, 3: 4, 4: 5},
                extern_syms={"@matvec_vectorized_bf16_bf16", "@linalg_fill_bf16"},
            ),
        ],
    )
    compile_one(
        cache,
        out,
        "head",
        mod,
        runtime_loop_tiling_sizes=[16, 16],
        use_lock_race_condition_fix=False,
    )
    save_json(
        out / "build.json",
        dict(
            seconds=time.perf_counter() - start,
            vocab_rows=32000,
            padded_rows=32768,
            native_submissions=1,
            launches=3,
            weights="original BF16 tied embedding",
            artifacts={
                str(f.relative_to(out)): file_hash(f)
                for f in (out / "cache").glob("*.elf")
            },
        ),
    )
