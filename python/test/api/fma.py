# ./python/test/api/fma.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""air.api lowers ops.fma to a single vector.fma.

The point of this op is that it is *not* a spelling of ``a * b + c``. Two arith
ops round the product before adding it; one vector.fma does not. So most of
what these tests pin is a negative -- no arith.mulf, no arith.addf -- because
that is the property a future emitter change could silently lose while every
numerical test still passed within tolerance.

The matched pair is deliberate: vector_muladd emits the two-op form and
vector_fma the fused one, and the last test here holds them side by side so
that the difference is pinned rather than described in a comment.
"""

from air import api as air
from air.api.types import bf16, f32


def build(body, dtype=bf16, N=65536, tile=1024, vector=16, herd_shape=(2,)):
    b = air.tensor([N], dtype)
    c = air.tensor([N], dtype)
    out = air.tensor([N], dtype)

    with air.launch(name="fma") as launch:

        @launch.body
        def _():
            with air.herd(range(0, N, tile), shape=herd_shape) as h:

                @h.body
                def _(tx):
                    (tn,) = h.tile_sizes
                    col = tx * tn
                    b_buf = air.alloc([tn], dtype, scope=h.private(), vector=vector)
                    c_buf = air.alloc([tn], dtype, scope=h.private(), vector=vector)
                    o_buf = air.alloc([tn], dtype, scope=h.private(), vector=vector)
                    air.ops.load(b_buf, b[col : col + tn])
                    air.ops.load(c_buf, c[col : col + tn])
                    o_buf[:] = body(b_buf, c_buf)
                    air.ops.store(o_buf, out[col : col + tn])

    return launch


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: fma_is_a_single_op
# alpha * b + c with the multiply fused into the add. The two CHECK-NOTs are
# the whole test: an emitter that expanded this back into a mul and an add
# would still produce numerically close output and would still pass every
# tolerance-based check in the tree.
# CHECK-NOT: arith.mulf
# CHECK-NOT: arith.addf
# CHECK: vector.broadcast {{.*}} : bf16 to vector<16xbf16>
# CHECK: vector.fma {{.*}} : vector<16xbf16>
@run
def fma_is_a_single_op():
    print(build(lambda b, c: air.ops.fma(2.0, b[:], c[:])).mlir())


# CHECK-LABEL: TEST: fma_of_three_buffer_operands
# No broadcast when nothing is a scalar: three transfer_reads feed one fma.
# CHECK: vector.transfer_read
# CHECK: vector.transfer_read
# CHECK: vector.fma {{.*}} : vector<16xbf16>
@run
def fma_of_three_buffer_operands():
    print(build(lambda b, c: air.ops.fma(b[:], c[:], b[:])).mlir())


# CHECK-LABEL: TEST: fma_reads_a_repeated_buffer_once
# b appears twice above and again here. The emitter's per-iteration read cache
# is keyed on the buffer, so a buffer named more than once in one tree is read
# once -- two reads for the two distinct buffers, not three for the three
# mentions. Pinned because the fma node was the first ternary node whose
# arguments are all value-typed, and a fresh traversal would be easy to write
# without threading the cache through.
# CHECK-COUNT-2: vector.transfer_read
# CHECK-NOT: vector.transfer_read
# CHECK: vector.fma
@run
def fma_reads_a_repeated_buffer_once():
    print(build(lambda b, c: air.ops.fma(b[:], b[:], c[:])).mlir())


# CHECK-LABEL: TEST: fma_composes_with_the_operators
# An fma node is value-typed, so it nests inside an ordinary expression tree
# and the whole thing still lowers as one loop. Here the fma feeds a multiply.
# CHECK: vector.fma {{.*}} : vector<16xbf16>
# CHECK: arith.mulf {{.*}} : vector<16xbf16>
# CHECK: vector.transfer_write
@run
def fma_composes_with_the_operators():
    print(build(lambda b, c: air.ops.fma(2.0, b[:], c[:]) * 3.0).mlir())


# CHECK-LABEL: TEST: fma_nests_inside_itself
# fma(a, b, fma(...)) -- the addend is itself an fma, which is the shape a
# polynomial evaluated by Horner's rule takes.
# CHECK-COUNT-2: vector.fma {{.*}} : vector<16xbf16>
@run
def fma_nests_inside_itself():
    print(
        build(lambda b, c: air.ops.fma(2.0, b[:], air.ops.fma(3.0, c[:], b[:]))).mlir()
    )


# CHECK-LABEL: TEST: fma_at_vector_width_32
# The npu2 lits run VECTOR_SIZE=32. Both widths compile; this pins that the
# width reaches the fma rather than only the transfer_read.
# CHECK: vector.fma {{.*}} : vector<32xbf16>
@run
def fma_at_vector_width_32():
    print(build(lambda b, c: air.ops.fma(2.0, b[:], c[:]), vector=32).mlir())


# CHECK-LABEL: TEST: f32_fma_is_emitted_but_needs_bf16_emulation
# The emitter has no reason to refuse f32 -- vector.fma over f32 is valid MLIR
# and is what --bf16-emulation consumes. What rejects it is the aievec
# conversion, which marks a native f32 vector.fma explicitly illegal ("failed
# to legalize operation 'vector.fma'"). That is a backend decision the DSL
# cannot see, and the error names the op, so it is left to the backend and
# pinned here instead of guessed at.
# CHECK: vector.fma {{.*}} : vector<16xf32>
@run
def f32_fma_is_emitted_but_needs_bf16_emulation():
    print(build(lambda b, c: air.ops.fma(2.0, b[:], c[:]), dtype=f32).mlir())


# CHECK-LABEL: TEST: the_unfused_pair_still_emits_two_ops
# The other half of the matched pair, in the same file so the two cannot drift
# apart unnoticed. vector_muladd depends on `alpha * b[:] + c[:]` continuing to
# emit a separate mul and add; if some future rewrite fused them, that example
# would become a duplicate of vector_fma and the comparison both exist for
# would quietly stop being a comparison.
# CHECK-NOT: vector.fma
# CHECK: arith.mulf {{.*}} : vector<16xbf16>
# CHECK: arith.addf {{.*}} : vector<16xbf16>
@run
def the_unfused_pair_still_emits_two_ops():
    print(build(lambda b, c: 2.0 * b[:] + c[:]).mlir())
