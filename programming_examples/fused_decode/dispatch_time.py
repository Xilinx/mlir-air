#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""How long does ONE decode dispatch take, per batch, on real NPU2?

WHY THIS AND NOT llama32_1b_q4nx/decode_cost.py. That script decomposes a
dispatch into fixed + layers*per_layer + lm_head + ctx*attn, which needs four
template FAMILIES per batch and a model driver that can allocate batched BOs.
llama-3.2-1b has both; qwen3-4b's driver (`qwen3_4b_q4nx_inference.py`) is
batch-1 only, which is why every batch>1 number in docs/DFlashFeasibility.md
used to be borrowed from llama's curve.

This asks the smaller question that IS answerable today: what does the whole
decode body cost at batch B, for a template `build_template.sh` already
produces? Those templates are the full UNI_DEC-layer stack (36 for qwen3-4b, 5
for qwen3-4b-draft) with the LM head dropped, and `batch_equiv.py` already
knows how to size and fill their BOs at any batch -- so this borrows the gate's
own geometry and fills, sets them up ONCE, and puts a stopwatch around the
kernel call. What it gives up against decode_cost.py is the DECOMPOSITION (it
cannot say how much is attention); what it gains is that it runs on qwen3-4b at
batch 8 today.

CAVEATS, both real:
  - build_template.sh skips the Peano pin preflight, so these are dataflow
    timings. A stale Peano changes instruction scheduling and therefore the
    absolute ms. The batch RATIO is the durable number: same compiler, same
    kernels, same design, one env var apart.
  - the templates carry golden/dummy weights and a synthetic KV cache. Time
    depends on how many bytes move and how many keys the attention loop walks,
    not on what is in them -- but this is never a correctness gate. That is
    `batch_equiv.py` and the per-model `make verify`.

    python3 dispatch_time.py                             # qwen3-4b, batches 1 and 8, L 128
    python3 dispatch_time.py --model qwen3-4b-draft
    python3 dispatch_time.py --batches 1 8 --L 128 --iters 50
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_equiv as BE


def time_one(model, vocab_i2, batch, L, iters, warmup, wait, xrt, prefix="decode"):
    """Median / p10 / p90 of the KERNEL CALL alone, for one template.

    NOT `batch_equiv.dispatch` in a loop. That function opens the device,
    registers the xclbin and re-fills every BO -- including the whole weight
    BO -- on each call, and at qwen3-4b's size that setup is ~275 ms against a
    device time of a few. Timing it measures numpy and XRT, and reports a
    batch-8 dispatch as 1.37x a batch-1 one, which is the setup ratio and not
    the device's. So: set up once here, time only `kern(...).wait()`.

    The buffers are filled exactly as the gate fills them (BE.weight_bo /
    BE.rms_bo / the tiled row), so what runs is what `batch_equiv.py` runs.
    """
    g = BE.geom(model, vocab_i2, L, batch)
    K = BE.geom(model, vocab_i2, L, 1)["k"]
    rng = np.random.default_rng(0)
    row = BE.bf16(rng.uniform(-1.0, 1.0, size=K))

    xclbin, insts = BE.template(prefix, batch, L)
    if not Path(xclbin).exists():
        return None, f"missing {Path(xclbin).name}"

    dev = xrt.device(0)
    xb = xrt.xclbin(str(xclbin))
    dev.register_xclbin(xb)
    ctx = xrt.hw_context(dev, xb.get_uuid())
    kn = [k for k in xb.get_kernels() if "MLIR_AIE" in k.get_name()][0]
    kern = xrt.kernel(ctx, kn.get_name())

    ib = np.fromfile(str(insts), dtype=np.uint32)
    i_bo = xrt.bo(dev, ib.nbytes, xrt.bo.cacheable, kern.group_id(1))
    i_bo.write(ib, 0)
    i_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    names = ("x", "w", "r", "y", "kv")
    sizes = (g["k"], g["w_elems"], g["rms_size"], g["ny"], g["kv_elems"])
    bos = {}
    for i, (name, n) in enumerate(zip(names, sizes), start=3):
        bos[name] = xrt.bo(dev, n * 2, xrt.bo.host_only, kern.group_id(i))
    fills = {
        "w": BE.weight_bo(g["w_elems"], 1),
        "r": BE.rms_bo(g, batch, 2),
        "kv": np.zeros(g["kv_elems"], np.int16),
        "y": np.zeros(g["ny"], np.int16),
        "x": np.tile(row, batch)[: g["k"]],
    }
    for name, buf in fills.items():
        bos[name].write(buf, 0)
        bos[name].sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    def fire():
        return kern(
            3, i_bo, ib.size, bos["x"], bos["w"], bos["r"], bos["y"], bos["kv"]
        ).wait(wait)

    try:
        for _ in range(warmup):
            st = fire()
            if not str(st).endswith("COMPLETED"):
                return None, f"dispatch state={st}"
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fire()
            ts.append((time.perf_counter() - t0) * 1e3)
    finally:
        # Same teardown order batch_equiv.dispatch uses: pyxrt segfaults at
        # exit if the BOs outlive the context and the device.
        bos.clear()
        del i_bo, kern, ctx, xb, dev
    ts.sort()
    return (
        statistics.median(ts),
        ts[max(0, int(0.1 * len(ts)) - 1)],
        ts[min(len(ts) - 1, int(0.9 * len(ts)))],
    ), None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--vocab-chunk-i2", type=int, default=30)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--wait", type=int, default=60000)
    ap.add_argument(
        "--prefix", default="decode", help="template stem: <prefix>_b<B>_L<N>.xclbin"
    )
    args = ap.parse_args()

    import pyxrt as xrt

    print(
        f"\ndispatch time  [{args.model}, L {args.L}, "
        f"{args.iters} iters after {args.warmup} warmup]"
    )
    print(
        f"  {'batch':>5} {'median ms':>10} {'p10':>9} {'p90':>9} {'vs batch 1':>11} "
        f"{'per token':>10}"
    )
    base = None
    for b in args.batches:
        r, err = time_one(
            args.model,
            args.vocab_chunk_i2,
            b,
            args.L,
            args.iters,
            args.warmup,
            args.wait,
            xrt,
            args.prefix,
        )
        if err:
            print(f"  {b:>5}  {err}")
            continue
        med, p10, p90 = r
        if base is None:
            base = med
        print(
            f"  {b:>5} {med:>10.3f} {p10:>9.3f} {p90:>9.3f} "
            f"{med/base:>10.3f}x {med/b:>10.3f}"
        )
    print(
        "\n  'per token' is median/batch -- the number a speculative block "
        "pays per\n  candidate. A batch that scaled for free would hold it flat."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
