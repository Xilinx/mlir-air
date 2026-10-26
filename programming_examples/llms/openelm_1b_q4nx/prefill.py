# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Full-input organization adapted from the declared OLMo2 example.
"""OpenELM full-input NPU prefill and original BF16 tied head."""

import os
from pathlib import Path
import re
import shutil
import time
import numpy as np
from ml_dtypes import bfloat16
from .common import ROOT, save_json, file_hash


def row_module(m, object_path, width=2048, qk=False):
    from air import api as air
    from air.api import ops

    # Single-row form has a 1D ABI for the fused final norm/head.
    shape = [width] if m == 1 else [m, width]
    x = air.tensor(shape, air.bf16)
    if qk:
        w = air.tensor([m, 128], air.bf16)
    else:
        w = air.tensor([width], air.bf16)
    y = air.tensor(shape, air.bf16)
    fn = air.extern(
        "openelm_qk_row" if qk else "rms_norm_aie",
        link_with=str(Path(object_path).resolve()),
        scalars=[air.i32],
    )
    cores = min(m, 8)
    with air.launch(name="openelm_qk" if qk else "openelm_norm") as launch:

        @launch.body
        def _():
            with air.herd([range(cores)], shape=(cores,)) as herd:

                @herd.body
                def _(tx):
                    a = air.alloc([width], air.bf16, scope=herd.private())
                    b = air.alloc(
                        [128 if qk else width], air.bf16, scope=herd.private()
                    )
                    out = air.alloc([width], air.bf16, scope=herd.private())
                    if not qk:
                        ops.load(b, w[:])
                    for i in air.sequential(m // cores):
                        row = tx * (m // cores) + i
                        ops.load(a, x[:] if m == 1 else x[row : row + 1, :])
                        if qk:
                            ops.load(b, w[row : row + 1, :])
                            fn(out, a, b, width // 64)
                        else:
                            fn(out, a, b, 0)
                        ops.store(out, y[:] if m == 1 else y[row : row + 1, :])

    return launch.build(target="npu2")


def gemm_spec(p, k, n):
    from air_examples.llms.shared.builders.gemm_builder import gemm_registry_config

    return gemm_registry_config(p, k, n, "bf16", "high")


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
    for name, path in [
        ("norm", local / "rms_residual.cc"),
        ("qk", local / "prefill_qk.cc"),
    ]:
        run(base + [path, "-o", out / f"{name}.o"], out, out / f"{name}.log")


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


def compile_prefill(out, p):
    from air_examples.llms.shared.infra.cache import KernelCache
    from air_examples.llms.shared.infra.external_kernels import (
        compile_gemm_mm,
        compile_attn_npu2,
    )
    from air_examples.llms.shared.builders.gemm_builder import _build_gemm_module
    from air_examples.llms.shared.builders.o_ffn_multi import _build_add_2d_to_2d
    from air_examples.silu_and_mul.silu_and_mul import build_module_2d
    from air_examples.flash_attention.kernel_fusion_based.attn_npu2_seqfirst import (
        build_module as attention,
    )

    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    start = time.perf_counter()
    cache = KernelCache(cache_dir=str(out / "cache"), verbose=True)
    compile_norm_objects(out)
    for tm in (32, 64):
        compile_gemm_mm(
            tile_m=tm,
            tile_n=128,
            tile_k_l1=32,
            sym_suffix=f"_m{tm}",
            out_name=f"mm_m{tm}.o",
            bfp16=True,
        )
    compile_attn_npu2(
        head_dim=64,
        lkp=64,
        lqp_tile=64,
        force=True,
        bfp16=True,
        attention_multiplier=0.125,
    )
    specs = {}
    for k, n in ((2048, 2048), (2048, 512), (2048, 8192), (8192, 2048)):
        spec = gemm_spec(p, k, n)
        name = f"gemm_{k}_{n}"
        specs[name] = spec
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
    compile_one(cache, out, "norm", row_module(p, out / "norm.o"))
    for width in (2048, 512):
        compile_one(cache, out, f"qk_{width}", row_module(p, out / "qk.o", width, True))
    compile_one(cache, out, "add", _build_add_2d_to_2d(p, 2048, bfloat16))
    compile_one(cache, out, "swiglu", build_module_2d(p, 8192, 4096, bfloat16, 8, 1))
    compile_one(
        cache,
        out,
        "attention",
        attention(
            lk=p,
            lkp=64,
            lq=p,
            lqp=256,
            dk=64,
            dv=64,
            num_q_tiles=4,
            num_cascade_stages=4,
            num_heads=32,
            num_kv_heads=8,
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
            artifacts={
                str(f.relative_to(out)): file_hash(f)
                for f in (out / "cache").glob("*.elf")
            },
        ),
    )


def compile_head(out):
    from air_examples.llms.shared.infra.cache import KernelCache
    from air_examples.llms.shared.infra.stitching import (
        stitch_elf,
        KernelSlice,
        FuncArg,
    )
    from air_examples.llms.shared.builders.lm_head_gemv_multi import (
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


class Prefill:
    def __init__(self, cache, weights, p, lut):
        self.cache, self.weights, self.p, self.lut = (
            cache,
            weights,
            p,
            np.ascontiguousarray(lut[:p]),
        )
        self.zeros = {n: np.zeros((p, n), bfloat16) for n in (512, 2048, 8192)}
        self.scratch = {n: np.zeros((p, n), np.float32) for n in (512, 2048, 8192)}
        self.submissions = 0

    def call(self, name, *args, key=None, static=(), outputs=None):
        kw = {"output_format": "elf"}
        if name == "attention":
            kw.update(
                instance_name="attention_bf16",
                omit_pingpong="all",
                runtime_loop_tiling_sizes=[1, 1, 1],
            )
        outs = outputs if outputs is not None else [len(args) - 1]
        ret = self.cache.load_and_run(
            name,
            kw,
            *args,
            bo_key=key,
            static_input_indices=set(static),
            intermediate_indices=set(outs),
            output_indices=outs,
            shared_nonstatic=True,
        )
        self.submissions += 1
        shaped = [ret[i].reshape(args[i].shape).copy() for i in outs]
        return shaped if outputs is not None else shaped[0]

    def gemm(self, x, w, key):
        k, n = w.shape
        spec = gemm_spec(self.p, k, n)
        args = [x, w]
        if spec["needs_f32_scratch"]:
            args.append(self.scratch[n])
        args.append(self.zeros[n])
        return self.call(f"gemm_{k}_{n}", *args, key=key, static=(1,))

    def norm(self, x, w, key):
        return self.call("norm", x, w, self.zeros[2048], key=key, static=(1,))

    def qk(self, x, w, key):
        n = x.shape[1]
        params = np.concatenate([self.lut, np.broadcast_to(w, (self.p, 64))], axis=1)
        return self.call(f"qk_{n}", x, params, self.zeros[n], key=key, static=(1,))

    def run(self, x):
        keys = np.empty((28, self.p, 512), bfloat16)
        values = np.empty_like(keys)
        for l, w in enumerate(self.weights):
            nx = self.norm(x, w["norms"][0], f"L{l}_an")
            q = self.qk(self.gemm(nx, w["q"], f"L{l}_q"), w["qknorms"][0], f"L{l}_qn")
            k = self.qk(self.gemm(nx, w["k"], f"L{l}_k"), w["qknorms"][1], f"L{l}_kn")
            v = self.gemm(nx, w["v"], f"L{l}_v")
            keys[l], values[l] = k, v
            a = self.call("attention", q, k, v, self.zeros[2048])
            branch = self.gemm(a, w["o"], f"L{l}_o")
            x = self.call("add", x, branch, self.zeros[2048])
            nx = self.norm(x, w["norms"][1], f"L{l}_fn")
            gate = self.gemm(nx, w["gate"], f"L{l}_gate")
            up = self.gemm(nx, w["up"], f"L{l}_up")
            activated = self.call("swiglu", gate, up, self.zeros[8192])
            branch = self.gemm(activated, w["down"], f"L{l}_down")
            x = self.call("add", x, branch, self.zeros[2048])
        self.last_keys, self.last_values = keys, values
        return x, keys, values
