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


import subprocess
import sys
import time
from .deployment import save_json, file_hash


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


def build_templates(out, capacity, kernel_builder, source):
    settings, env, records = kernel_builder(out, capacity)
    src = Path(source)
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
