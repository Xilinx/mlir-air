# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Reproducible isolated build, with raw subprocess logs and artifact hashes."""

import json
import os
from pathlib import Path
import time

from .common import ROOT, save_json, file_hash
from ..shared.decode_build import run, build_templates

DECODE_ENV = dict(
    AIR_TARGET_DEVICE="npu2",
    DECODE_MODEL="granite-3.3-2b",
    DECODE_UNI_DEC="40",
    VOCAB_CHUNK_I2="10",
    UNI_LM="5",
    LM_HEAD="0",
    NLAYERS="1",
    DECODE_GOLDEN="1",
    W_DUAL_CHAN="1",
    PROJ_RC_CACHE="1",
    DECODE_GOLDEN_L="1024",
    DECODE_WGROUP="0",
    UNI_WAVE_LO="0",
    UNI_WAVE_HI="45",
    DECODE_DYNSEQ="0",
)


def decode_module():
    os.environ.update(DECODE_ENV)
    from air_examples.fused_decode import fused_decode

    if fused_decode.MODEL_NAME != "granite-3.3-2b":
        raise RuntimeError("another fused model is already imported")
    return fused_decode


def build_decode(out, capacity=1024):
    return build_templates(
        out, capacity, build_decode_kernels, ROOT / "programming_examples/fused_decode"
    )


def build_decode_kernels(out, capacity=1024):
    out.mkdir(parents=True, exist_ok=True)
    src = ROOT / "programming_examples/fused_decode"
    settings = dict(DECODE_ENV, DECODE_GOLDEN_L=str(capacity))
    env = dict(os.environ, **settings)
    records = [
        run(
            ["make", "-f", src / "Makefile", "preflight-llvm-link"],
            out,
            out / "preflight.log",
            env,
        )
    ]
    base = [
        Path(os.environ["PEANO_INSTALL_DIR"]) / "bin/clang++",
        "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        "-Wno-parentheses",
        "-Wno-attributes",
        "-Wno-macro-redefined",
        "-Wno-empty-body",
        "-Wno-deprecated-declarations",
        "-DNDEBUG",
        "-DMODEL_TYPE=GRANITE_3_3_2B",
        "-D__AIE_API_AIE_ADF_HPP__",
        "-DGLU_SLICE_EXPECTED=1024",
        "-I",
        Path(os.environ["MLIR_AIE_INSTALL_DIR"]) / "include",
        "-I",
        src / "kernels",
        "-I",
        src / "models",
        "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
    ]
    from ..shared.decode_build import compile_kernels

    records.append(compile_kernels(out, src, base[1:], [src / "kernels"], run, env))
    return settings, env, records


def validate_prefill_reuse(source, sequence_length):
    """Reject an attention ELF compiled for a different full input length."""
    source = Path(source).resolve()
    metadata = json.loads((source / "build.json").read_text())
    if metadata.get("sequence_length") != sequence_length:
        raise ValueError(
            f"reused prefill sequence_length {metadata.get('sequence_length')} "
            f"does not match requested {sequence_length}: {source}"
        )
    return source


def build_prefill(out, sequence_length=512, rms_impl="seed", reuse_prefill=None):
    if reuse_prefill is not None:
        reuse_prefill = validate_prefill_reuse(reuse_prefill, sequence_length)
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    from .common import Config
    from ..shared.infra.external_kernels import (
        compile_attn_npu2,
        _compile_kernel,
    )
    from ..shared.infra.cache import KernelCache
    from llama32_1b_prefill import compile_all_kernels

    os.environ["TEMPORAL_CAUSAL_SKIP"] = "0"
    os.environ["AIR_TARGET_DEVICE"] = "npu2"
    # prepare_air_project reuses this object in the isolated cwd. Force the
    # geometry and Granite softmax scalar here before any composed build.
    if reuse_prefill is None:
        compile_attn_npu2(
            head_dim=64,
            lkp=64,
            lqp_tile=64,
            force=True,
            attention_multiplier=Config.attention_multiplier,
        )
        _compile_kernel(
            Path(__file__).with_name("scaled_residual.cc"),
            "scaled_residual.o",
            force=True,
        )
    else:
        # Prefill GEMM variant: retain the accepted non-GEMM objects and the
        # entire attention ELF. The two mm variants are compiled afresh below.
        import shutil

        reuse_prefill = Path(reuse_prefill).resolve()
        for obj in reuse_prefill.glob("*.o"):
            if not obj.name.startswith("mm"):
                shutil.copy2(obj, out / obj.name)
    cache = KernelCache(cache_dir=str(out / "cache"), verbose=True)
    start = time.perf_counter()
    config = Config()
    config.gemm_bfp16 = True
    if rms_impl == "decode":
        import functools
        import shutil
        from .prefill_rms import build_module

        source = out.parent / "decode/rms_residual.o"
        shutil.copy2(source, out / "granite_rms.o")
        config.rms_builder = functools.partial(
            build_module, object_path=out / "granite_rms.o"
        )
    # cpu_attn here controls compilation only. The reused flash_attn ELF is
    # installed in the manifest; production still calls cpu_attn=False.
    compile_all_kernels(
        cache, config, sequence_length, cpu_attn=reuse_prefill is not None
    )
    if reuse_prefill is not None:
        manifest = out / "cache/manifest.json"
        current = json.loads(manifest.read_text())
        accepted = json.loads((reuse_prefill / "cache/manifest.json").read_text())
        assert current["_toolchain"] == accepted["_toolchain"]
        entry = dict(accepted["entries"]["flash_attn"])
        for key in ("output_binary", "insts"):
            if entry.get(key):
                source = reuse_prefill / "cache" / Path(entry[key]).name
                target = out / "cache" / source.name
                target.symlink_to(source.resolve())
                entry[key] = str(target)
        current["entries"]["flash_attn"] = entry
        save_json(manifest, current)
    save_json(
        out / "build.json",
        {
            "seconds": time.perf_counter() - start,
            "sequence_length": sequence_length,
            "attention_multiplier": 0.015625,
            "residual_multiplier": 0.22,
            "gemm_bfp16": config.gemm_bfp16,
            "rms_implementation": rms_impl,
            "gemm_macro": (
                "AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16"
                if config.gemm_bfp16
                else None
            ),
            "reused_prefill": str(reuse_prefill) if reuse_prefill else None,
            "gemm_objects": {p.name: file_hash(p) for p in out.glob("mm_*.o")},
            "artifacts": {
                str(p.relative_to(out)): file_hash(p) for p in out.rglob("*.elf")
            },
        },
    )


def build_head(out):
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    os.environ["AIR_TARGET_DEVICE"] = "npu2"
    from ..shared.infra.cache import KernelCache
    from ..shared.infra.backend_presets import LM_GEMV_BACKEND
    from ..shared.builders.lm_head_gemv_multi import (
        build_lm_head_gemv_module,
    )

    cache = KernelCache(cache_dir=str(out / "cache"), verbose=True)
    start = time.perf_counter()
    cache.compile_and_cache(
        "lm_head_gemv",
        build_lm_head_gemv_module(2048, n_partitions=4),
        dict(LM_GEMV_BACKEND, verbose=True, target_device="npu2"),
    )
    cache._save_manifest()
    save_json(
        out / "build.json",
        {
            "seconds": time.perf_counter() - start,
            "artifacts": {
                str(p.relative_to(out)): file_hash(p) for p in out.rglob("*.elf")
            },
        },
    )
