#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Build the Gemma4-E4B eight-token decoder: kernels, then the graph
(b8_graph.py, attention_graph.py) through aircc. With --check-existing an
existing build of the same sources and tools is kept.

  python3 build_target.py --output DIR [--layers N] [--check-existing]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import paths  # noqa: F401
import b8_config
from common.build_identity import build_identity, check_existing, digest

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parents[2]
FUSED = EXAMPLES / "fused_decode"
COMMON = HERE.parent / "common"
KERNELS = {
    "proj_qmm.o": COMMON / "kernels/proj_batch.cc",
    "rms_b8.o": HERE / "kernels/rms_b8.cc",
    "cache_b8.o": HERE / "kernels/cache_b8.cc",
    "glu.o": HERE / "kernels/glu_b8.cc",
    "rope_b8.o": HERE / "kernels/rope_b8.cc",
    "attn_b8.o": HERE / "kernels/attn_b8.cc",
}
ARTIFACTS = ("decode.xclbin", "decode.txn.h")


def identity():
    """The shared decoder identity (fused_decode, common kernels, tools) and
    this example's graph and kernel sources."""
    ident = build_identity("decoder", HERE)
    own = ["b8_config.py", "b8_graph.py", "attention_graph.py", "build_target.py"]
    own += [str(p.relative_to(HERE)) for p in sorted((HERE / "kernels").glob("*"))]
    ident["sources"].update(
        {str((HERE / n).relative_to(EXAMPLES)): digest(HERE / n) for n in own}
    )
    return ident


def compile_kernels(out, fd):
    clang = Path(os.environ["PEANO_INSTALL_DIR"]) / "bin/clang++"
    flags = [
        "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        "-Wno-parentheses",
        "-Wno-attributes",
        "-Wno-macro-redefined",
        "-Wno-empty-body",
        "-Wno-deprecated-declarations",
        "-DNDEBUG",
        "-DMODEL_TYPE=GEMMA4_E4B",
        "-D__AIE_API_AIE_ADF_HPP__",
        "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        f"-DGLU_SLICE_EXPECTED={fd.GLU_SLICE}",
        "-DPROJ_MM_BATCH=8",
        # GLU slices arrive [round][token][PAYLOAD]
        "-DGLU_BATCH_WAVE_LAYOUT=1",
        "-DGLU_BATCH=8",
        "-DDECODE_ROUND_NEAREST",
        "-I" + str(Path(os.environ["MLIR_AIE_INSTALL_DIR"]) / "include"),
        "-I" + str(FUSED / "kernels"),
        "-I" + str(FUSED / "models"),
        "-I" + str(EXAMPLES),
        "-I" + str(COMMON / "kernels"),
        "-O2",
    ]
    cmds = []
    for obj, src in KERNELS.items():
        cmd = [str(clang), *flags, "-c", str(src), "-o", str(out / obj)]
        print("compiling", obj, flush=True)
        subprocess.run(cmd, check=True)
        cmds.append(cmd)
    return cmds


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--layers", type=int, default=b8_config.NUM_LAYERS)
    ap.add_argument("--capacity", type=int, default=4096)
    ap.add_argument("--check-existing", action="store_true")
    a = ap.parse_args()
    cfg = b8_config.Config(capacity=a.capacity, layers=a.layers)
    out = a.output.resolve()
    ident = identity()
    if out.exists() and a.check_existing:
        check_existing(out, ident)
        print("up to date", out)
        return
    work = out / f"compile_L{cfg.build_length}"
    work.mkdir(parents=True, exist_ok=False)
    layout = b8_config.load_layout(cfg)
    cmds = compile_kernels(work, layout)

    import b8_graph
    from attention_graph import GemmaAttention

    module = b8_graph.build_module(layout, GemmaAttention)
    assert module.operation.verify()
    (work / "input.mlir").write_text(str(module))
    from air.backend.xrt import XRTBackend

    old = Path.cwd()
    try:
        os.chdir(work)
        art = XRTBackend(
            omit_while_true_loop=False,
            output_format="xclbin",
            kernel_name="MLIR_AIE",
            stack_size=cfg.stack_bytes,
            omit_pingpong="L2",
            use_lock_race_condition_fix_v2=True,
            coalesce_shim_dma=bool(layout.COALESCE),
            emit_txn_cpp=True,
            target_device="npu2",
        ).compile(module, output_binary_name="decode", insts="decode.insts.bin")
        txn = art.txn_header
    finally:
        os.chdir(old)
    shutil.copy2(work / "decode.xclbin", out / "decode.xclbin")
    shutil.copy2(work / txn, out / "decode.txn.h")
    (out / "manifest.json").write_text(
        json.dumps(
            dict(
                status="compiled",
                build_identity=ident,
                artifacts={n: digest(out / n) for n in ARTIFACTS},
                config=layout.ABI["config"],
                abi=layout.ABI,
                commands=cmds,
            ),
            indent=2,
        )
        + "\n"
    )
    print("built", out)


if __name__ == "__main__":
    main()
