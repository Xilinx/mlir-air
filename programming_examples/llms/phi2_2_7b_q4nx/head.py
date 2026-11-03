# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The decode's LM head: final biased LayerNorm and the full-vocabulary head
in one ELF."""

import os
from pathlib import Path
import re
import shutil
import time
from .common import ROOT, HERE, save_json, file_hash


def norm_module(out, width=3072):
    """One row's biased LayerNorm (phi_norm)."""
    from air import api as air
    from air.api import ops

    x = air.tensor([width], air.bf16)
    w = air.tensor([width], air.bf16)
    b = air.tensor([width], air.bf16)
    y = air.tensor([width], air.bf16)
    fn = air.extern(
        "phi_norm",
        link_with=str((out / "rms_residual.o").resolve()),
        scalars=[air.i32],
    )
    with air.launch(name="phi_norm_rows") as launch:

        @launch.body
        def _():
            with air.herd([range(1)], shape=(1,)) as herd:

                @herd.body
                def _(tx):
                    a = air.alloc([width], air.bf16, scope=herd.private())
                    z = air.alloc([width], air.bf16, scope=herd.private())
                    ps = [
                        air.alloc([width], air.bf16, scope=herd.private())
                        for _ in range(2)
                    ]
                    for dst, src in zip(ps, (w, b)):
                        ops.load(dst, src[:])
                    for i in air.sequential(1):
                        ops.load(a, x[:])
                        fn(z, a, *ps, 0)
                        ops.store(z, y[:])

    return launch.build(target="npu2")


def compile_objects(out):
    from .build import run

    src = ROOT / "programming_examples/fused_decode"
    base = [
        Path(os.environ["PEANO_INSTALL_DIR"]) / "bin/clang++",
        "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        "-O2",
        "-c",
        "-DNDEBUG",
        "-include",
        HERE / "kernels/phi2.h",
        "-D__AIE_API_AIE_ADF_HPP__",
        "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        "-I",
        Path(os.environ["MLIR_AIE_INSTALL_DIR"]) / "include",
        "-I",
        src / "kernels",
        "-I",
        src / "models",
    ]
    run(
        base + [HERE / "kernels/rms_residual.cc", "-o", out / "rms_residual.o"],
        out,
        out / "rms_residual.log",
    )


def compile_one(cache, out, name, module, **kw):
    if name in cache.artifacts and (out / f"ir_{name}").is_dir():
        print("REUSE verified build identity", name, flush=True)
        return
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
    compile_objects(out)
    cache = KernelCache(cache_dir=str(out / "cache"), verbose=True)
    norm = str(norm_module(out))
    head = str(build_lm_head_gemv_module(3072, n_partitions=4, n_part=12800))
    types = ["memref<3072xbf16>"] * 3
    for i in range(4):
        types += ["memref<12800x3072xbf16>", "memref<12800xbf16>"]
    types += ["memref<3072xbf16>"]
    args = [FuncArg(f"%arg{i}", t) for i, t in enumerate(types)]
    mapping = {0: 11}
    for i in range(4):
        mapping[1 + 2 * i] = 3 + 2 * i
        mapping[2 + 2 * i] = 4 + 2 * i
    mod = stitch_elf(
        "phi_head",
        args,
        [
            KernelSlice(
                norm, "norm", {0: 0, 1: 1, 2: 2, 3: 11}, extern_syms={"@phi_norm"}
            ),
            KernelSlice(
                head,
                "head",
                mapping,
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
            vocab_rows=51200,
            native_submissions=1,
            artifacts={
                str(f.relative_to(out)): file_hash(f)
                for f in (out / "cache").glob("*.elf")
            },
        ),
    )
