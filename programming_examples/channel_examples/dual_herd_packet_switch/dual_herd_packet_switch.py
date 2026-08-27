# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Two herds sharing the shim, on air.api.

One segment holding two independent herds -- an element-wise add and an
element-wise multiply, on separate inputs and separate outputs:

    c_add = a0 + b0        add_herd
    c_mul = a1 * b1        mul_herd

Six L3 tensors and two herds means more shim DMA traffic than there are
dedicated channels, which is the point: `air-dma-to-channel` infers the channels
from the transfers and the shim shares them, packet-switched. Nothing here
declares a channel, and nothing needs to -- the two herds are written exactly as
they would be alone, and the sharing is the compiler's business.

Each core takes one tile: `tx * tile_size` is ordinary Python arithmetic on the
coordinate where the predecessor built an `arith.muli` per transfer, three times
per herd.

The compute is `tile_c[:] = tile_a[:] + tile_b[:]` and `* tile_b[:]`, replacing
`linalg.elemwise_binary` with an explicit BinaryFn and a cast attribute. Same
values, and the DSL vectorises rather than handing the pipeline a named op to
scalarise.
"""

import argparse

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api import ops
from air.api.types import bf16
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

HERD_SIZE = 8
INOUT_DATATYPE = bfloat16


def build_module(tile_size, dtype=bf16):
    total = tile_size * HERD_SIZE

    # Inputs first, then outputs: the XRT invocation passes them in that order.
    a0 = air.tensor([total], dtype)
    b0 = air.tensor([total], dtype)
    a1 = air.tensor([total], dtype)
    b1 = air.tensor([total], dtype)
    c_add = air.tensor([total], dtype)
    c_mul = air.tensor([total], dtype)

    with air.launch(name="dual_herd_elemwise") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():

                    def elementwise(name, x, y, out, combine):
                        """One herd: every core does one tile of one operator."""
                        with air.herd(
                            [range(HERD_SIZE)], name=name, shape=(HERD_SIZE,)
                        ) as herd:

                            @herd.body
                            def _(tx):
                                tile_x = air.alloc(
                                    [tile_size], dtype, scope=herd.private()
                                )
                                tile_y = air.alloc(
                                    [tile_size], dtype, scope=herd.private()
                                )
                                tile_out = air.alloc(
                                    [tile_size], dtype, scope=herd.private()
                                )

                                lo = tx * tile_size
                                ops.load(tile_x, x[lo : lo + tile_size])
                                ops.load(tile_y, y[lo : lo + tile_size])

                                tile_out[:] = combine(tile_x[:], tile_y[:])

                                ops.store(tile_out, out[lo : lo + tile_size])

                    elementwise("add_herd", a0, b0, c_add, lambda p, q: p + q)
                    elementwise("mul_herd", a1, b1, c_mul, lambda p, q: p * q)

    return launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="dual_herd_packet_switch.py",
        description="Dual 8x1 herd elemwise add/mul with packet-switched shim DMA",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="elf",
        dest="output_format",
        help="Output format for the compiled binary (default: elf)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Elements per tile per herd (default: 1024)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of warmup iterations before measurement (default: 0)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of measurement iterations (default: 1)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="auto",
        help="NPU generation to build for: auto (default, detects the installed "
        "device), npu1 or npu2",
    )
    args = parser.parse_args()

    tile_size = args.tile_size
    total_size = tile_size * HERD_SIZE

    launch = build_module(tile_size)
    # build() resolves --target auto, so it runs before launch.target is read.
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    # Golden reference computation using small integers (exact in bf16)
    np.random.seed(42)
    a0 = np.random.randint(-8, 9, size=total_size).astype(bfloat16)
    b0 = np.random.randint(-8, 9, size=total_size).astype(bfloat16)
    a1 = np.random.randint(-4, 5, size=total_size).astype(bfloat16)
    b1 = np.random.randint(-4, 5, size=total_size).astype(bfloat16)

    c_add_ref = (a0.astype(np.float32) + b0.astype(np.float32)).astype(bfloat16)
    c_mul_ref = (a1.astype(np.float32) * b1.astype(np.float32)).astype(bfloat16)

    if args.iterations > 1 or args.warmup > 0:
        # Profiling mode: compile once, run many times
        import filelock, tempfile

        backend = XRTBackend(
            verbose=args.verbose,
            output_format=args.output_format,
            instance_name="dual_herd_elemwise",
            target_device=launch.target,
        )

        compiled_module = backend.compile(mlir_module)

        with filelock.FileLock(tempfile.gettempdir() + "/npu.lock"):
            module_function = backend.load(compiled_module)

            c_add_out = np.zeros(total_size, dtype=bfloat16)
            c_mul_out = np.zeros(total_size, dtype=bfloat16)

            # Warmup
            for _ in range(args.warmup):
                module_function(a0, b0, a1, b1, c_add_out, c_mul_out)

            # Measurement
            times = []
            for _ in range(args.iterations):
                t0 = time.perf_counter()
                module_function(a0, b0, a1, b1, c_add_out, c_mul_out)
                t1 = time.perf_counter()
                times.append(t1 - t0)

            # Verify last run
            results = module_function(a0, b0, a1, b1, c_add_out, c_mul_out)
            c_add_actual = results[4]
            c_mul_actual = results[5]

        backend.unload()

        if not np.array_equal(c_add_actual, c_add_ref):
            print("FAIL: add_herd output mismatch")
            exit(1)
        if not np.array_equal(c_mul_actual, c_mul_ref):
            print("FAIL: mul_herd output mismatch")
            exit(1)

        # Stats
        # Data moved: 4 inputs + 2 outputs, each total_size * 2 bytes (bf16)
        data_bytes = 6 * total_size * 2
        times_us = [t * 1e6 for t in times]
        avg_us = np.mean(times_us)
        min_us = np.min(times_us)
        max_us = np.max(times_us)
        avg_gbps = data_bytes / (np.mean(times) * 1e9)
        peak_gbps = data_bytes / (min(times) * 1e9)

        print(f"PASS!")
        print(f"Problem size: {total_size} elements x 6 buffers = {data_bytes} bytes")
        print(f"Iterations:   {args.iterations} (warmup: {args.warmup})")
        print(f"Avg time:     {avg_us:.1f} us")
        print(f"Min time:     {min_us:.1f} us")
        print(f"Max time:     {max_us:.1f} us")
        print(f"Avg BW:       {avg_gbps:.2f} GB/s")
        print(f"Peak BW:      {peak_gbps:.2f} GB/s")
    else:
        # Single-run correctness mode
        runner = XRTRunner(
            verbose=args.verbose,
            output_format=args.output_format,
            instance_name="dual_herd_elemwise",
            target_device=launch.target,
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[a0, b0, a1, b1],
                expected_outputs=[c_add_ref, c_mul_ref],
            )
        )
