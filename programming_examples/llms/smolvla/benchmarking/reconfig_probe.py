# SPDX-License-Identifier: MIT
"""Reconfiguration-cost probe: a do-nothing herd whose core program size is a knob.

Each core loads 16 bf16 values, calls pad_touch() and stores them back.
pad_touch carries PAD bytes of never-executed padding behind a runtime-false
branch, so the program grows while the work stays fixed. Sweeps program size
x core count x launches per ELF and prints device time plus the ELF's control
code size, to split a launch's cost into per-dispatch, per-launch, per-core and
per-KB-of-program parts.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

_HERE = Path(__file__).resolve().parent
_PROG = _HERE.parent.parent.parent
for p in (str(_PROG), str(_HERE.parent.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)


def pad_src(stores):
    """Peano has no inline asm, so the padding is `stores` distinct volatile
    stores behind a branch that is never taken at runtime."""
    body = "\n".join(
        f"    pad_sink_[{i % 64}] = {0x10000 + i * 7919};" for i in range(stores)
    )
    return (
        'extern "C" {\n'
        "volatile int pad_never_ = 0;\n"
        "volatile int pad_sink_[64];\n"
        "void pad_touch(unsigned short *buf) {\n"
        "  if (pad_never_) {\n" + body + "\n  }\n"
        "  buf[0] ^= 1;\n"
        "}\n}\n"
    )


def compile_pad(pad):
    from shared.infra.external_kernels import _PEANO_FLAGS, _get_peano_clang

    src = Path("pad_probe.cc")
    src.write_text(pad_src(pad))
    out = f"pad_{pad}.o"
    cmd = [_get_peano_clang()] + _PEANO_FLAGS + ["-c", str(src), "-o", out]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    size = (
        subprocess.run(
            [os.path.expanduser("~/peano_pinned/llvm-aie/bin/llvm-size"), out],
            capture_output=True,
            text=True,
        )
        .stdout.splitlines()[-1]
        .split()[0]
    )
    return out, int(size)


def build_one(pad, cols, rows):
    from air import api as air
    from air.api import ops
    from air.api.types import bf16

    n = cols * rows
    IN = air.tensor([n, 16], bf16)
    OUT = air.tensor([n, 16], bf16)
    padk = air.extern("pad_touch", link_with=f"pad_{pad}.o")
    with air.launch(name="padprobe") as launch:

        @launch.body
        def _():
            with air.segment(name="pad_seg") as seg:

                @seg.body
                def _():
                    with air.herd(
                        [range(cols), range(rows)], name="pad_herd", shape=(cols, rows)
                    ) as h:

                        @h.body
                        def _(tx, ty):
                            buf = air.alloc([16], bf16, scope=h.private())
                            r = tx * rows + ty
                            ops.load(buf, IN[r, 0:16])
                            padk(buf)
                            ops.store(buf, OUT[r, 0:16])

    return str(launch.build(target="npu2"))


def build_module(pad, cols, rows, launches):
    from shared.infra.stitching import FuncArg, KernelSlice, stitch_elf

    ir = build_one(pad, cols, rows)
    t = f"memref<{cols * rows}x16xbf16>"
    slices = [
        KernelSlice(
            ir, f"p{i}", {0: 0, 1: 1}, extern_syms={"@pad_touch"}, private_from=(i == 0)
        )
        for i in range(launches)
    ]
    return stitch_elf("padprobe", [FuncArg("%arg0", t), FuncArg("%arg1", t)], slices)


def ctrl_kb(build_dir):
    readelf = os.path.expanduser("~/peano_pinned/llvm-aie/bin/llvm-readelf")
    elfs = sorted(Path(build_dir).rglob("*.elf"), key=lambda p: p.stat().st_mtime)
    if not elfs:
        return float("nan")
    out = subprocess.run(
        [readelf, "-SW", str(elfs[-1])], capture_output=True, text=True
    ).stdout
    tot = 0
    for line in out.splitlines():
        if ".ctrltext" in line:
            f = line.split("]", 1)[1].split()
            tot += int(f[4], 16)
    return tot / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--configs", default="0:8:1:1", help="space-separated pad:cols:rows:launches"
    )
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    from shared.infra.cache import KernelCache, Profiler

    for cfg in args.configs.split():
        pad, cols, rows, launches = (int(v) for v in cfg.split(":"))
        tag = f"pad{pad}_c{cols}x{rows}_L{launches}"
        bdir = _HERE / "build" / f"reconfig_{tag}"
        bdir.mkdir(parents=True, exist_ok=True)
        cwd = os.getcwd()
        os.chdir(bdir)
        try:
            _, text = compile_pad(pad)
            cache = KernelCache(
                str(bdir), verbose=False, profiler=Profiler(enabled=True)
            )
            backend = {
                "verbose": False,
                "omit_while_true_loop": False,
                "output_format": "elf",
                "instance_name": "padprobe",
            }
            cache.compile_and_cache(
                "padprobe", build_module(pad, cols, rows, launches), backend
            )
            n = cols * rows
            x = np.arange(n * 16, dtype=np.float32).reshape(n, 16).astype(bfloat16)
            y = np.zeros((n, 16), bfloat16)

            def run():
                return cache.load_and_run(
                    "padprobe", backend, x, y, output_indices=[1], bo_key="p"
                )[1]

            out = np.asarray(run()).reshape(n, 16).view(np.uint16)
            exp = x.view(np.uint16).copy()
            exp[:, 0] ^= 1
            ok = np.array_equal(out, exp)
            for _ in range(10):
                run()
            cache.profiler.kernel_breakdowns.clear()
            for _ in range(args.iters):
                run()
            dev = sorted(
                e["kernel_ms"] for e in cache.profiler.kernel_breakdowns["padprobe"]
            )
            kb = ctrl_kb(bdir)
        finally:
            os.chdir(cwd)
        print(
            f"{tag:26s} ok={ok} pad.o text {text:6d} B  ctrl {kb:7.1f} KB  device median {dev[len(dev)//2]*1e3:7.1f} us "
            f"p10 {dev[len(dev)//10]*1e3:7.1f}  min {dev[0]*1e3:7.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
