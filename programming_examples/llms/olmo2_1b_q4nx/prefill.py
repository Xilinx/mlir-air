# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Full-input OLMo2 prefill: all block arithmetic runs on NPU.

GEMMs use the shared exact-shape registry and BFP16 multiply/FP32
accumulation. The new ordering is explicit here; no Llama block is invoked.
"""

import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from ml_dtypes import bfloat16
from .common import ROOT, save_json, file_hash


def row_module(m, object_path, rope=False):
    from air import api as air
    from air.api import ops

    x = air.tensor([m, 2048], air.bf16)
    w = air.tensor([m, 128] if rope else [2048], air.bf16)
    y = air.tensor([m, 2048], air.bf16)
    fn = air.extern(
        "olmo_rope_row" if rope else "rms_norm_aie",
        link_with=str(Path(object_path).resolve()),
        scalars=[] if rope else [air.i32],
    )
    with air.launch(name="olmo_rope" if rope else "olmo_norm") as launch:

        @launch.body
        def _():
            with air.herd([range(8)], shape=(8,)) as herd:

                @herd.body
                def _(tx):
                    a = air.alloc([2048], air.bf16, scope=herd.private())
                    b = air.alloc(
                        [128 if rope else 2048], air.bf16, scope=herd.private()
                    )
                    out = air.alloc([2048], air.bf16, scope=herd.private())
                    if not rope:
                        ops.load(b, w[:])
                    for i in air.sequential(m // 8):
                        row = tx * (m // 8) + i
                        ops.load(a, x[row : row + 1, :])
                        if rope:
                            ops.load(b, w[row : row + 1, :])
                            fn(out, a, b)
                        else:
                            fn(out, a, b, 0)
                        ops.store(out, y[row : row + 1, :])

    return launch.build(target="npu2")


def gemm_spec(p, k, n):
    from ..shared.builders.gemm_builder import gemm_registry_config

    return gemm_registry_config(p, k, n, "bf16", "high")


def compile_prefill(out, p, reuse_non_gemm_from=None):
    from .build import run
    from ..shared.infra.cache import KernelCache
    from ..shared.infra.external_kernels import (
        compile_gemm_mm,
        compile_attn_npu2,
    )
    from ..shared.builders.gemm_builder import _build_gemm_module
    from ..shared.builders.o_ffn_multi import _build_add_2d_to_2d
    from air_examples.silu_and_mul.silu_and_mul import build_module_2d
    from air_examples.flash_attention.kernel_fusion_based.attn_npu2_seqfirst import (
        build_module as attention,
    )

    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    start = time.perf_counter()
    cache = KernelCache(cache_dir=str(out / "cache"), verbose=True)
    src = ROOT / "programming_examples/fused_decode"
    cpp = Path(__file__).resolve().parent / "kernels/prefill_rope.cc"
    cxx = [
        Path(os.environ["PEANO_INSTALL_DIR"]) / "bin/clang++",
        "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        "-O2",
        "-c",
        "-DNDEBUG",
        "-DMODEL_TYPE=OLMO2_1B",
        "-D__AIE_API_AIE_ADF_HPP__",
        "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        "-I",
        Path(os.environ["MLIR_AIE_INSTALL_DIR"]) / "include",
        "-I",
        src / "kernels",
        "-I",
        src / "models",
    ]
    if reuse_non_gemm_from is None:
        for name, path in (
            ("olmo_rms", src / "kernels/rms_residual.cc"),
            ("olmo_rope", cpp),
        ):
            run(cxx + [path, "-o", out / f"{name}.o"], out, out / f"{name}.log")
    for tm in (32, 64):
        compile_gemm_mm(
            tile_m=tm,
            tile_n=128,
            tile_k_l1=32,
            sym_suffix=f"_m{tm}",
            out_name=f"mm_m{tm}.o",
            bfp16=True,
        )
    if reuse_non_gemm_from is None:
        compile_attn_npu2(
            head_dim=128,
            lkp=64,
            lqp_tile=64,
            force=True,
            bfp16=True,
            attention_multiplier=128**-0.5,
        )
    records = {}

    def compile_one(name, module, **kw):
        import re

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

    for k, n in ((2048, 2048), (2048, 8192), (8192, 2048)):
        spec = gemm_spec(p, k, n)
        name = f"gemm_{k}_{n}"
        module = _build_gemm_module(
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
        records[name] = spec
        compile_one(name, module, runtime_loop_tiling_sizes=[2, 2])
    if reuse_non_gemm_from is None:
        compile_one("norm", row_module(p, out / "olmo_rms.o"))
        compile_one("rope", row_module(p, out / "olmo_rope.o", True))
        compile_one("add", _build_add_2d_to_2d(p, 2048, bfloat16))
        compile_one("swiglu", build_module_2d(p, 8192, 4096, bfloat16, 8, 1))
        compile_one(
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
                num_heads=16,
                num_kv_heads=16,
                causal=True,
            ),
            omit_pingpong="all",
            runtime_loop_tiling_sizes=[1, 1, 1],
            instance_name="attention_bf16",
        )
    else:
        old = Path(reuse_non_gemm_from).resolve()
        old_build = json.loads((old / "build.json").read_text())
        assert old_build["sequence_length"] == p
        previous = json.loads((old / "cache" / cache.MANIFEST_FILE).read_text())
        manifest_path = out / "cache" / cache.MANIFEST_FILE
        current = json.loads(manifest_path.read_text())
        assert previous["_toolchain"] == current["_toolchain"]
        for name in ("norm", "rope", "add", "swiglu", "attention"):
            entry = dict(previous["entries"][name])
            for field in ("output_binary", "insts"):
                if entry[field]:
                    filename = Path(entry[field]).name
                    shutil.copy2(old / "cache" / filename, out / "cache" / filename)
                    entry[field] = str(out / "cache" / filename)
            shutil.copy2(old / f"{name}.mlir", out / f"{name}.mlir")
            current["entries"][name] = entry
        save_json(manifest_path, current)
    save_json(
        out / "build.json",
        dict(
            sequence_length=p,
            seconds=time.perf_counter() - start,
            gemm_specs=records,
            gemm_bfp16=True,
            attention_bfp16=True,
            gemm_compiler_macro="AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
            reuse_non_gemm_from=(
                str(reuse_non_gemm_from) if reuse_non_gemm_from else None
            ),
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
        self.zeros = {
            2048: np.zeros((p, 2048), bfloat16),
            8192: np.zeros((p, 8192), bfloat16),
        }
        self.scratch = {
            2048: np.zeros((p, 2048), np.float32),
            8192: np.zeros((p, 8192), np.float32),
        }
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

    def rope(self, x):
        return self.call("rope", x, self.lut, self.zeros[2048], static=(1,))

    def run(self, x):
        keys = np.empty((16, self.p, 2048), bfloat16)
        values = np.empty_like(keys)
        for l, w in enumerate(self.weights):
            q = self.rope(
                self.norm(self.gemm(x, w["q"], f"L{l}_q"), w["norms"][0], f"L{l}_qn")
            )
            k = self.rope(
                self.norm(self.gemm(x, w["k"], f"L{l}_k"), w["norms"][1], f"L{l}_kn")
            )
            v = self.gemm(x, w["v"], f"L{l}_v")
            keys[l], values[l] = k, v
            a = self.call("attention", q, k, v, self.zeros[2048])
            branch = self.norm(
                self.gemm(a, w["o"], f"L{l}_o"), w["norms"][2], f"L{l}_an"
            )
            x = self.call("add", x, branch, self.zeros[2048])
            gate = self.gemm(x, w["gate"], f"L{l}_gate")
            up = self.gemm(x, w["up"], f"L{l}_up")
            activated = self.call("swiglu", gate, up, self.zeros[8192])
            branch = self.norm(
                self.gemm(activated, w["down"], f"L{l}_down"), w["norms"][3], f"L{l}_fn"
            )
            x = self.call("add", x, branch, self.zeros[2048])
        self.last_keys, self.last_values = keys, values
        return x, keys, values
