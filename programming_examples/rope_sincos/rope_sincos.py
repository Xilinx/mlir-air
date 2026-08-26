# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""RoPE (Rotary Position Embeddings) with on-chip sin/cos, on air.api.

Applies rotary position embeddings to Q and K vectors (V is passed through)::

  Q_out[2i]   = Q[2i] * cos(pos * freq_i) - Q[2i+1] * sin(pos * freq_i)
  Q_out[2i+1] = Q[2i] * sin(pos * freq_i) + Q[2i+1] * cos(pos * freq_i)

where ``freq_i = 1 / (theta ** (2i / head_size))`` and ``theta = 10000``.

Unlike the LUT-based variant, sin/cos are computed on-chip by Chebyshev
polynomial approximation, with the frequency table hardcoded in the kernel
(head_size=48), so no sin/cos input is needed. The arithmetic all lives in
``rope.cc``; what this file does is move one head into L1, call five kernels on
it, and move it back.

    air.ops.load(l1_in.reshape(3 * head_size), IN[o : o + 3 * head_size])
    vector_copy(l1_in, l1_out)
    freq_pos(1, l1_freq_pos)
    sinf(l1_freq_pos, l1_sin)
    cosf(l1_freq_pos, l1_cos)
    shuffle_apply_rope(0,         l1_cos, l1_sin, l1_out)
    shuffle_apply_rope(head_size, l1_cos, l1_sin, l1_out)
    air.ops.store(l1_out.reshape(3 * head_size), OUT[o : o + 3 * head_size])

Five ``air.extern``s replace five hand-built ``FuncOp`` declarations plus the
loop that stamped ``link_with`` and ``llvm.emit_c_interface`` on each. The
declaration is derived from the call: buffer types come from the buffers, and
only the scalar element types are stated, because a Python ``1`` cannot say
whether the kernel wants ``i32`` or ``index``.

**The reshape on the way in is the point of the shapes here.** The L1 tiles are
``[3, head_size]`` because that is what ``rope.cc`` is compiled against -- Q, K
and V as three rows -- while the L3 tensor is flat, one head's 3*head_size
elements contiguous. ``ops.store`` has always drained a view; filling one is new
(see ops.load), and it is what lets the buffer keep the kernel's shape while the
transfer keeps the tensor's.

Input format: ``[num_heads * 3 * head_size]``, Q/K/V concatenated per head.

Two differences from the predecessor, both shared with the rest of the
converted tree:

* The herd is [herd_n, 1] rather than [1, herd_n]. A 1-D air.api herd is laid
  out along x, which is the orientation that places on both generations.
* The strip-mine is the DSL's: the iteration space is one point per head and
  ``shape=`` pins the core count, where the predecessor wrote the outer
  ``scf.for`` and the ``AffineMap`` that combined it with the core coordinate by
  hand. Every head is still processed exactly once by exactly one core.

**The buffers are released explicitly**, all five at the end. Automatic
placement frees each one after its last observed use, which for this kernel is
wrong on hardware -- see the comment at the deallocs.

``--target`` is new and defaults to detecting the installed part.
"""

import argparse
from math import cos, sin

import numpy as np
from ml_dtypes import bfloat16

from air import api as air
from air.api.types import bf16, i32
from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner

# The object the Makefile compiles rope.cc into, in the build directory.
EXTERN_OBJECT = "rope.o"

# Padded to 32 elements (the next multiple of 16) so AIE2P can use native-width
# vectors (n=16) for the sin/cos polynomial. On AIE2 the extra 8 elements are
# computed and unused. This is the predecessor's constant, kept.
SINCOS_BUF_SIZE = 32


def build_module(num_heads, head_size, herd_n):
    # The kernels are specialised for head_size=48: the frequency table is
    # hardcoded and the symbol names carry the size.
    assert head_size == 48, f"Only head_size=48 is supported, got {head_size}"
    assert (num_heads * head_size) % herd_n == 0

    head_stride = 3 * head_size

    cosf = air.extern("cosf_bf16", link_with=EXTERN_OBJECT)
    sinf = air.extern("sinf_bf16", link_with=EXTERN_OBJECT)
    freq_pos = air.extern("freq_pos_bf16", link_with=EXTERN_OBJECT, scalars=[i32])
    shuffle_apply_rope = air.extern(
        "shuffle_apply_rope_bf16_48", link_with=EXTERN_OBJECT, scalars=[i32]
    )
    vector_copy = air.extern("vector_copy_bf16_144_16", link_with=EXTERN_OBJECT)

    # [3 * num_heads, head_size] rather than one flat run: the same bytes,
    # but a whole head is now a [3, head_size] region, which is exactly the
    # L1 tile's shape, so both transfers are plain shape-matching ones.
    IN = air.tensor([num_heads * head_stride], bf16)
    OUT = air.tensor([num_heads * head_stride], bf16)

    with air.launch(name="rope") as launch:

        @launch.body
        def _():
            # One point per head; shape= pins the core count to what the
            # predecessor asked for and the DSL strip-mines the rest.
            with air.herd(
                [range(0, num_heads * head_stride, head_stride)],
                name="herd_0",
                shape=(herd_n,),
                link_with=EXTERN_OBJECT,
            ) as h:

                @h.body
                def _(tx):
                    o = tx * head_stride

                    # [3, head_size] is rope.cc's shape -- Q, K and V as three
                    # rows -- while the L3 region is flat, so the transfers go
                    # through a reshaped view.
                    l1_in = air.alloc([3, head_size], bf16, scope=h.private())
                    l1_out = air.alloc([3, head_size], bf16, scope=h.private())
                    l1_freq_pos = air.alloc([SINCOS_BUF_SIZE], bf16, scope=h.private())
                    l1_sin = air.alloc([SINCOS_BUF_SIZE], bf16, scope=h.private())
                    l1_cos = air.alloc([SINCOS_BUF_SIZE], bf16, scope=h.private())

                    air.ops.load(l1_in.reshape(head_stride), IN[o : o + head_stride])

                    # V passes through unchanged, so the whole head is copied
                    # and only Q and K are then rotated in place.
                    vector_copy(l1_in, l1_out)

                    # position = 1, as in the predecessor and its reference.
                    freq_pos(1, l1_freq_pos)
                    sinf(l1_freq_pos, l1_sin)
                    cosf(l1_freq_pos, l1_cos)

                    # Q lives at offset 0 and K at head_size; V is left alone.
                    shuffle_apply_rope(0, l1_cos, l1_sin, l1_out)
                    shuffle_apply_rope(head_size, l1_cos, l1_sin, l1_out)

                    air.ops.store(l1_out.reshape(head_stride), OUT[o : o + head_stride])

                    # Released explicitly, all five together at the end, which
                    # is where the predecessor's DeallocOps were. This is not
                    # tidiness: left to place them itself the tracer frees each
                    # buffer after the last use it observes, so l1_in goes right
                    # after vector_copy while four more kernels are still
                    # running on the others -- and the result is wrong on
                    # hardware, a handful of Q/K rotation pairs coming back as
                    # huge values or NaN. Measured, and measured both ways: the
                    # only difference between the passing and failing runs is
                    # these five lines. See the PR for the isolation.
                    for buf in (l1_sin, l1_cos, l1_freq_pos, l1_in, l1_out):
                        air.dealloc(buf)

    return launch


if __name__ == "__main__":
    HEAD_SIZE = 48
    NUM_HEADS = 8
    HERD_N = 4
    INPUT_DATATYPE = bfloat16

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the RoPE (on-chip sin/cos) example",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument("--head-size", type=int, default=HEAD_SIZE, help="Head size")
    parser.add_argument(
        "--num-heads", type=int, default=NUM_HEADS, help="Number of heads"
    )
    parser.add_argument(
        "--herd-n",
        type=int,
        default=HERD_N,
        help="Number of L1 tiles along the N dimension",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        choices=["compile-only", "compile-and-run"],
        dest="compile_mode",
        default="compile-and-run",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="auto",
        help="NPU generation to build for: auto (default, detects the installed "
        "device), npu1 or npu2",
    )
    args = parser.parse_args()

    launch = build_module(args.num_heads, args.head_size, args.herd_n)
    mlir_module = launch.build(target=args.target)
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    np.random.seed(0)
    num_tiles = args.num_heads
    inputs = np.random.randn(num_tiles, 3 * args.head_size).astype(INPUT_DATATYPE)
    outputs = inputs.copy()

    # Reference: apply rotation to Q and K, leave V unchanged
    for i in range(num_tiles):
        for s in range(0, args.head_size, 2):
            freq = 1.0 / pow(10000.0, float(s) / float(args.head_size))
            val = 1 * freq  # position = 1

            fcr = cos(val)
            fci = sin(val)

            # Rotate Q
            v0 = outputs[i][s]
            v1 = outputs[i][s + 1]
            outputs[i][s] = v0 * fcr - v1 * fci
            outputs[i][s + 1] = v0 * fci + v1 * fcr

            # Rotate K
            v0 = outputs[i][s + args.head_size]
            v1 = outputs[i][s + args.head_size + 1]
            outputs[i][s + args.head_size] = v0 * fcr - v1 * fci
            outputs[i][s + args.head_size + 1] = v0 * fci + v1 * fcr

    if args.compile_mode == "compile-and-run":
        runner = XRTRunner(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            instance_name="rope",
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
        )
        exit(
            runner.run_test(
                mlir_module,
                inputs=[inputs],
                expected_outputs=[outputs],
                rtol=1e2,
            )
        )

    elif args.compile_mode == "compile-only":
        backend = XRTBackend(
            verbose=args.verbose,
            omit_while_true_loop=False,
            output_format=args.output_format,
            target_device=launch.target,
            runtime_loop_tiling_sizes=[4, 4],
        )
        module_function = backend.compile(mlir_module)
        backend.unload()
