# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Adapted from the declared Granite Q4NX deployment example.
"""Reproducible isolated build, with raw subprocess logs and artifact hashes."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import ROOT, HERE, save_json, file_hash

DECODE_ENV = dict(
    AIR_TARGET_DEVICE="npu2",
    DECODE_MODEL="mistral7b-v03",
    DECODE_UNI_DEC="32",
    VOCAB_CHUNK_I2="16",
    UNI_LM="2",
    LM_HEAD="0",
    NLAYERS="1",
    DECODE_GOLDEN="1",
    W_DUAL_CHAN="1",
    PROJ_RC_CACHE="1",
    DECODE_GOLDEN_L="1024",
    DECODE_WGROUP="8",
    DECODE_STACK="8064",
    UNIFIED="1",
    UNI_WAVE_LO="0",
    UNI_WAVE_HI="34",
    DECODE_DYNSEQ="0",
)


def decode_module():
    os.environ.update(DECODE_ENV)
    from air_examples.fused_decode import fused_decode

    if fused_decode.MODEL_NAME != "mistral7b-v03":
        raise RuntimeError("another fused model is already imported")
    return fused_decode


def run(argv, cwd, log, env=None):
    print("RUN", " ".join(map(str, argv)), flush=True)
    start = time.perf_counter()
    with open(log, "w") as out:
        out.write("COMMAND " + repr(list(map(str, argv))) + "\n")
        out.flush()
        result = subprocess.run(
            list(map(str, argv)), cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT
        )
    rec = {
        "argv": list(map(str, argv)),
        "cwd": str(cwd),
        "log": str(log),
        "seconds": time.perf_counter() - start,
        "returncode": result.returncode,
    }
    save_json(str(log) + ".json", rec)
    if result.returncode:
        raise RuntimeError(f"build failed: see {log}")
    return rec


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
        "-include",
        Path(__file__).resolve().parent / "kernels/mistral.h",
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
    from air_examples.llms.shared.decode_build import compile_kernels

    records.append(compile_kernels(out, src, base[1:], [src / "kernels"], run, env))
    return settings, env, records


def build_decode(out, capacity=1024):
    settings, env, records = build_decode_kernels(out, capacity)
    src = ROOT / "programming_examples/fused_decode"
    for length in (capacity, capacity - 1):
        env["DECODE_GOLDEN_L"] = str(length)
        records.append(
            run(
                [sys.executable, src / "fused_decode.py"],
                out,
                out / f"decode_L{length}.log",
                env,
            )
        )
        for ext in ("xclbin", "insts.bin"):
            (out / f"decode.{ext}").replace(out / f"decode_L{length}.{ext}")
        # Retain candidate-bound compiler IR for each calibration template.
        import shutil

        shutil.copytree(out / "air_project", out / f"ir_L{length}", dirs_exist_ok=True)
    (out / ".decode_windows").write_text(f"{capacity}\n")
    save_json(
        out / "build.json",
        {
            "environment": settings,
            "capacity": capacity,
            "commands": records,
            "artifacts": {
                p.name: file_hash(p)
                for p in out.glob("decode_L*.*")
                if p.suffix != ".log"
            },
        },
    )


def build_prefill(
    out, sequence_length=512, rms_impl="decode", *, gemm_bfp16=True, reuse_prefill=None
):
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    from .common import Config
    from air_examples.llms.shared.infra.external_kernels import (
        compile_attn_npu2,
        _compile_kernel,
    )
    from air_examples.llms.shared.infra.cache import KernelCache
    from llama32_1b_prefill import compile_all_kernels

    os.environ["TEMPORAL_CAUSAL_SKIP"] = "0"
    os.environ["AIR_TARGET_DEVICE"] = "npu2"
    # prepare_air_project reuses this object in the isolated cwd. Force the
    # geometry and Mistral softmax scalar here before any composed build.
    reused = {}
    if reuse_prefill is not None:
        import json
        import shutil

        old = json.loads((reuse_prefill / "build.json").read_text())
        assert old["sequence_length"] == sequence_length
        assert old["attention_bfp16"] == (sequence_length < 4096)
        assert old["attention_single_accumulator"] == (sequence_length == 4096)
        assert old["rms_implementation"] == rms_impl
        # Reuse unchanged support objects, including the original attention.
        # Both GEMM variants are recompiled below with the selected arithmetic.
        for source in reuse_prefill.glob("*.o"):
            if not source.name.startswith("mm"):
                shutil.copy2(source, out / source.name)
                reused[source.name] = {"path": str(source), "sha256": file_hash(source)}
    else:
        compile_attn_npu2(
            head_dim=128,
            lkp=64,
            lqp_tile=64,
            force=True,
            bfp16=sequence_length < 4096,
            single_accumulator=sequence_length == 4096,
            attention_multiplier=Config.attention_multiplier,
        )
    cache = KernelCache(cache_dir=str(out / "cache"), verbose=True)
    start = time.perf_counter()
    config = Config()
    from .recipes import install

    install()
    config.gemm_bfp16 = gemm_bfp16
    if rms_impl == "decode":
        import functools
        import shutil
        from .prefill_rms import build_module

        source = out.parent / "decode/rms_residual.o"
        shutil.copy2(source, out / "mistral_rms.o")
        config.rms_builder = functools.partial(
            build_module, object_path=out / "mistral_rms.o"
        )
    # cpu_attn here only skips building the unchanged attention ELF. Runtime
    # still loads and executes the accepted NPU attention artifact below.
    compile_all_kernels(
        cache, config, sequence_length, cpu_attn=reuse_prefill is not None
    )
    if reuse_prefill is not None:
        from air.backend.xrt import XRTCompileArtifact

        info = json.loads((reuse_prefill / "cache/manifest.json").read_text())
        assert info["_toolchain"] == cache._toolchain_id()
        entry = info["entries"]["flash_attn"]
        assert entry["insts"] is None
        source = reuse_prefill / "cache/flash_attn.elf"
        destination = out / "cache/flash_attn.elf"
        shutil.copy2(source, destination)
        cache.artifacts["flash_attn"] = XRTCompileArtifact(
            str(destination), entry["kernel"], None
        )
        cache._save_manifest()
        reused["flash_attn.elf"] = {"path": str(source), "sha256": file_hash(source)}
    save_json(
        out / "build.json",
        {
            "seconds": time.perf_counter() - start,
            "sequence_length": sequence_length,
            "attention_multiplier": Config.attention_multiplier,
            "residual_multiplier": 1.0,
            "gemm_bfp16": config.gemm_bfp16,
            "gemm_compiler_macro": (
                "AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16" if gemm_bfp16 else None
            ),
            "reused": reused,
            "gemm_objects": {p.name: file_hash(p) for p in sorted(out.glob("mm*.o"))},
            "rms_implementation": rms_impl,
            "attention_bfp16": sequence_length < 4096,
            "attention_single_accumulator": sequence_length == 4096,
            "artifacts": {
                str(p.relative_to(out)): file_hash(p) for p in out.rglob("*.elf")
            },
        },
    )


def build_head(out):
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)
    os.environ["AIR_TARGET_DEVICE"] = "npu2"
    from air_examples.llms.shared.infra.cache import KernelCache
    from air_examples.llms.shared.infra.backend_presets import LM_GEMV_BACKEND
    from air_examples.llms.shared.builders.lm_head_gemv_multi import (
        build_lm_head_gemv_module,
    )

    cache = KernelCache(cache_dir=str(out / "cache"), verbose=True)
    start = time.perf_counter()
    from air_examples.llms.shared.infra.external_kernels import compile_mv

    compile_mv(tile_m=4)
    cache.compile_and_cache(
        "lm_head_gemv",
        build_lm_head_gemv_module(
            4096, n_partitions=2, n_part=16384, herd_m=8, tile_m=4, m_input=2
        ),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["decode", "prefill", "head"])
    ap.add_argument("--out", type=Path, default=HERE / "build/p512")
    ap.add_argument(
        "--sequence-length", type=int, choices=(512, 1024, 2048, 4096), default=512
    )
    ap.add_argument(
        "--capacity", type=int, choices=(1024, 2048, 4096, 8192), default=1024
    )
    ap.add_argument("--rms-impl", choices=("seed", "decode"), default="decode")
    ap.add_argument("--gemm-bfp16", action="store_true", default=True)
    ap.add_argument("--reuse-prefill", type=Path)
    args = ap.parse_args()
    out = args.out.resolve() / args.mode
    if args.mode == "decode":
        build_decode(out, args.capacity)
    elif args.mode == "prefill":
        build_prefill(
            out,
            args.sequence_length,
            args.rms_impl,
            gemm_bfp16=args.gemm_bfp16,
            reuse_prefill=args.reuse_prefill.resolve() if args.reuse_prefill else None,
        )
    else:
        build_head(out)


if __name__ == "__main__":
    main()
