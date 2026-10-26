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
    DECODE_MODEL="openelm-1b",
    DECODE_UNI_DEC="28",
    VOCAB_CHUNK_I2="16",
    UNI_LM="2",
    LM_HEAD="0",
    NLAYERS="1",
    DECODE_GOLDEN="1",
    W_DUAL_CHAN="1",
    PROJ_RC_CACHE="1",
    DECODE_GOLDEN_L="1024",
    DECODE_WGROUP="0",
    UNI_WAVE_LO="0",
    UNI_WAVE_HI="28",
    DECODE_DYNSEQ="0",
)


def decode_module():
    os.environ.update(DECODE_ENV)
    from air_examples.fused_decode import fused_decode

    if fused_decode.MODEL_NAME != "openelm-1b":
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
        "-DMODEL_TYPE=OPENELM_1B",
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


def build_prefill(out, sequence_length=512, rms_impl="decode"):
    from .prefill import compile_prefill

    compile_prefill(out, sequence_length)


def build_head(out):
    from .prefill import compile_head

    compile_head(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["decode", "prefill", "head"])
    ap.add_argument("--out", type=Path, default=HERE / "build/p512")
    ap.add_argument("--sequence-length", type=int, choices=(512, 1024), default=512)
    ap.add_argument("--capacity", type=int, choices=(1024, 2048), default=1024)
    ap.add_argument("--rms-impl", choices=("seed", "decode"), default="decode")
    args = ap.parse_args()
    out = args.out.resolve() / args.mode
    if args.mode == "decode":
        build_decode(out, args.capacity)
    elif args.mode == "prefill":
        build_prefill(out, args.sequence_length, args.rms_impl)
    else:
        build_head(out)


if __name__ == "__main__":
    main()
