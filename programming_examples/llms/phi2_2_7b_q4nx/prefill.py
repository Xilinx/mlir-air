# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Full-input organization adapted from the declared OpenELM/OLMo2 examples.
"""Phi full-P prefill and independent full-vocabulary NPU head."""

import os
from pathlib import Path
import re
import shutil
import time
import numpy as np
from ml_dtypes import bfloat16
from .common import ROOT, HERE, save_json, file_hash


def row_module(m, kind, out):
    from air import api as air
    from air.api import ops

    width = {"norm": 3072, "merge": 3072, "rope": 4096, "affine": 4096, "gelu": 10752}[
        kind
    ]
    shape = [width] if m == 1 else [m, width]
    x = air.tensor(shape, air.bf16)
    params = []
    if kind == "norm":
        params = (
            [air.tensor([width], air.bf16), air.tensor([width], air.bf16)]
            if m == 1
            else [air.tensor([2 * width], air.bf16)]
        )
    if kind == "rope":
        params = [air.tensor([m, 32], air.bf16)]
    if kind == "merge":
        params = [air.tensor([m, 2 * width], air.bf16)]
    y = air.tensor(shape, air.bf16)
    symbol = {
        "norm": "phi_norm",
        "merge": "phi_merge",
        "rope": "phi_rope_row",
        "affine": "phi_attn_affine",
        "gelu": "phi_gelu_row",
    }[kind]
    if kind == "norm" and m > 1:
        symbol = "phi_norm_packed"
    if kind == "merge":
        symbol = "phi_merge_packed"
    obj = {
        "norm": "rms_residual",
        "merge": "rms_residual",
        "rope": "rope",
        "affine": "glu",
        "gelu": "glu",
    }[kind]
    fn = air.extern(
        symbol,
        link_with=str((out / f"{obj}.o").resolve()),
        scalars=[air.i32] if kind == "norm" else [],
    )
    cores = min(m, 8)
    with air.launch(name="phi_" + kind + "_rows") as launch:

        @launch.body
        def _():
            with air.herd([range(cores)], shape=(cores,)) as herd:

                @herd.body
                def _(tx):
                    a = air.alloc([width], air.bf16, scope=herd.private())
                    z = air.alloc([width], air.bf16, scope=herd.private())
                    ps = [
                        air.alloc(
                            [
                                (
                                    32
                                    if kind == "rope"
                                    else (
                                        2 * width
                                        if (kind == "norm" and m > 1) or kind == "merge"
                                        else width
                                    )
                                )
                            ],
                            air.bf16,
                            scope=herd.private(),
                        )
                        for _ in params
                    ]
                    if kind == "norm":
                        for dst, src in zip(ps, params):
                            ops.load(dst, src[:])
                    for i in air.sequential(m // cores):
                        row = tx * (m // cores) + i
                        ops.load(a, x[:] if m == 1 else x[row : row + 1, :])
                        if kind in ("rope", "merge"):
                            for dst, src in zip(ps, params):
                                ops.load(dst, src[row : row + 1, :])
                        if kind == "norm":
                            fn(z, a, *ps, 0)
                        else:
                            fn(z, a, *ps)
                        ops.store(z, y[:] if m == 1 else y[row : row + 1, :])

    return launch.build(target="npu2")


def gemm_spec(p, k, n):
    from ..shared.builders.gemm_builder import (
        gemm_registry_config,
        _spec_with_tiles,
    )

    try:
        return gemm_registry_config(p, k, n, "bf16", "high")
    except (ValueError, KeyError, RuntimeError) as e:
        return dict(
            _spec_with_tiles(
                "drain", dict(tile_m=32, tile_k_l2=64, tile_k_l1=32, tile_n=128)
            ),
            registry_miss=str(e),
        )


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
    for name in ("rms_residual", "rope", "glu"):
        run(
            base + [HERE / "kernels" / f"{name}.cc", "-o", out / f"{name}.o"],
            out,
            out / f"{name}.log",
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


def compile_prefill(out, p, reuse=None):
    from ..shared.infra.cache import KernelCache
    from ..shared.infra.external_kernels import (
        compile_gemm_mm,
        compile_attn_npu2,
    )
    from ..shared.builders.gemm_builder import _build_gemm_module
    from air_examples.flash_attention.kernel_fusion_based.attn_npu2_seqfirst import (
        build_module as attention,
    )

    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    start = time.perf_counter()
    from .session import load_cache

    cache = (
        load_cache(out / "cache")
        if (out / "cache/manifest.json").exists()
        else KernelCache(cache_dir=str(out / "cache"), verbose=True)
    )
    if reuse is not None:
        import json

        info = json.loads((reuse / "cache/manifest.json").read_text())
        info["entries"] = {
            k: v for k, v in info["entries"].items() if not k.startswith("gemm_")
        }
        (out / "cache").mkdir(exist_ok=True)
        for name, entry in info["entries"].items():
            old = reuse / "cache" / Path(entry["output_binary"]).name
            new = out / "cache" / old.name
            shutil.copy2(old, new)
            entry["output_binary"] = str(new)
            shutil.copy2(reuse / f"{name}.mlir", out / f"{name}.mlir")
        save_json(out / "cache/manifest.json", info)
        cache = load_cache(out / "cache")
    else:
        compile_objects(out)
    compile_gemm_mm(
        tile_m=32,
        tile_n=128,
        tile_k_l1=32,
        sym_suffix="_m32",
        out_name="mm_m32.o",
        bfp16=True,
    )
    if reuse is None:
        compile_attn_npu2(
            head_dim=128,
            lkp=64,
            lqp_tile=64,
            force=True,
            bfp16=True,
            attention_multiplier=1 / np.sqrt(80.0),
        )
    specs = {}
    for k, n in ((3072, 12288), (4096, 3072), (3072, 10752), (10752, 3072)):
        spec = gemm_spec(p, k, n)
        name = f"gemm_{k}_{n}"
        specs[name] = spec
        if spec["tile_m"] != 32:
            raise ValueError("uncompiled registry GEMM variant")
        mod = _build_gemm_module(
            p,
            k,
            n,
            spec["tile_m"],
            spec["tile_k_l2"],
            spec["tile_k_l1"],
            spec["tile_n"],
            8,
            4,
            **spec["build_kwargs"],
        )
        compile_one(cache, out, name, mod, runtime_loop_tiling_sizes=[2, 2])
    if reuse is None:
        for kind in ("norm", "rope", "gelu", "affine", "merge"):
            compile_one(cache, out, kind, row_module(p, kind, out))
        compile_one(
            cache,
            out,
            "attention",
            attention(
                lk=p,
                lkp=64,
                lq=p,
                lqp=256,
                dk=128,
                dv=128,
                num_q_tiles=4,
                num_cascade_stages=4,
                num_heads=32,
                num_kv_heads=32,
                causal=True,
            ),
            omit_pingpong="all",
            runtime_loop_tiling_sizes=[1, 1, 1],
            instance_name="attention_bf16",
        )
    save_json(
        out / "build.json",
        dict(
            sequence_length=p,
            seconds=time.perf_counter() - start,
            gemm_specs=specs,
            gemm_bfp16=True,
            attention_bfp16=True,
            reused_non_gemm_from=str(reuse) if reuse else None,
            gemm_macro="AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            artifacts={
                str(f.relative_to(out)): file_hash(f)
                for f in (out / "cache").glob("*.elf")
            },
        ),
    )


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
    norm = str(row_module(1, "norm", out))
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


class Prefill:
    def __init__(self, cache, weights, p, lut):
        self.cache, self.weights, self.p, self.lut = (
            cache,
            weights,
            p,
            np.ascontiguousarray(lut[:p]),
        )
        self.zeros = {n: np.zeros((p, n), bfloat16) for n in (3072, 4096, 10752, 12288)}
        self.submissions = 0

    def call(self, name, *args, key=None, static=()):
        kw = {"output_format": "elf"}
        if name == "attention":
            kw.update(
                instance_name="attention_bf16",
                omit_pingpong="all",
                runtime_loop_tiling_sizes=[1, 1, 1],
            )
        last = len(args) - 1
        ret = self.cache.load_and_run(
            name,
            kw,
            *args,
            bo_key=key,
            static_input_indices=set(static),
            intermediate_indices={last},
            output_indices=[last],
            shared_nonstatic=True,
        )
        self.submissions += 1
        return ret[last].reshape(args[last].shape).copy()

    def gemm(self, x, w, key):
        k, n = w.shape
        return self.call(f"gemm_{k}_{n}", x, w, self.zeros[n], key=key, static=(1,))

    def run(self, x):
        keys = np.empty((32, self.p, 4096), bfloat16)
        values = np.empty_like(keys)
        for l, w in enumerate(self.weights):
            nx = self.call(
                "norm",
                x,
                w["norms"].reshape(-1),
                self.zeros[3072],
                key=f"L{l}_norm",
                static=(1,),
            )
            qkv = self.gemm(nx, w["qkv"], f"L{l}_qkv")
            q = self.call(
                "rope",
                np.ascontiguousarray(qkv[:, :4096]),
                self.lut,
                self.zeros[4096],
                static=(1,),
            )
            k = self.call(
                "rope",
                np.ascontiguousarray(qkv[:, 4096:8192]),
                self.lut,
                self.zeros[4096],
                static=(1,),
            )
            v = np.ascontiguousarray(qkv[:, 8192:])
            keys[l], values[l] = k, v
            a = self.call("attention", q, k, v, self.zeros[4096])
            a = self.call("affine", a, self.zeros[4096])
            attn = self.gemm(a, w["o"], f"L{l}_o")
            f = self.gemm(nx, w["fc1"], f"L{l}_fc1")
            f = self.call("gelu", f, self.zeros[10752])
            f = self.gemm(f, w["down"], f"L{l}_down")
            x = self.call(
                "merge", x, np.concatenate([attn, f], axis=1), self.zeros[3072]
            )
        self.last_keys, self.last_values = keys, values
        return x, keys, values
