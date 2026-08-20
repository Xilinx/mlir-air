# ./python/test/api/errors.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""Misuse of air.api raises, rather than emitting a silently-wrong kernel.

This is the property that separates this package from the tracer prototype it
replaces: there, an unsupported construct was absorbed and the program still
"ran", returning zeros. Every case below must produce an exception.
"""

from itertools import product

from air import api as air
from air.api import ops  # noqa: F401
from air.api.types import bf16, f32


def expect(exc_types, label):
    """Run a body and print the exception it raised."""

    def decorator(fn):
        print("\nTEST:", label)
        try:
            fn()
        except exc_types as e:
            print(f"{type(e).__name__}: {e}")
        else:
            print("ERROR: no exception raised")
        return fn

    return decorator


def _trace(body, tensors=3, shape=(2, 2), grid=(128, 128, 64)):
    """Build a launch whose herd body is ``body(h, tx, ty, tensors...)``."""
    M, N, tile = grid
    ts = [air.tensor([M, N], bf16) for _ in range(tensors)]

    with air.launch(name="k") as launch:

        @launch.body
        def _():
            with air.herd(
                product(range(0, M, tile), range(0, N, tile)), shape=shape
            ) as h:

                @h.body
                def _(tx, ty):
                    body(h, tx, ty, *ts)

    return launch.mlir()


# CHECK-LABEL: TEST: shape_mismatch
# CHECK: ValueError: shape mismatch in elementwise assignment
@expect(ValueError, "shape_mismatch")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64, 64], bf16, scope=h.private())
        b = air.alloc([32, 32], bf16, scope=h.private())
        a[:] = a[:] + b[:]

    _trace(body)


# CHECK-LABEL: TEST: dtype_mismatch
# CHECK: ValueError: dtype mismatch in elementwise assignment
@expect(ValueError, "dtype_mismatch")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64, 64], bf16, scope=h.private())
        b = air.alloc([64, 64], f32, scope=h.private())
        a[:] = a[:] + b[:]

    _trace(body)


# CHECK-LABEL: TEST: transfer_shape_mismatch
# CHECK: ValueError: transfer shape mismatch
@expect(ValueError, "transfer_shape_mismatch")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([32, 32], bf16, scope=h.private())
        air.ops.load(a, A[0:64, 0:64])

    _trace(body)


# CHECK-LABEL: TEST: partial_buffer_write
# CHECK: NotImplementedError: partial assignment into a buffer is not supported
@expect(NotImplementedError, "partial_buffer_write")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64, 64], bf16, scope=h.private())
        a[0:8, :] = a[:] + a[:]

    _trace(body)


# CHECK-LABEL: TEST: dynamic_tile_size
# CHECK: ValueError: slice size along dim 0 is not a compile-time constant
@expect(ValueError, "dynamic_tile_size")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64, 64], bf16, scope=h.private())
        # The stop bound moves with the tile coordinate, so the extent is not
        # a constant and the DMA would need a dynamic size.
        air.ops.load(a, A[tx : tx * 2 + 64, 0:64])

    _trace(body)


# CHECK-LABEL: TEST: herd_arity
# CHECK: TypeError: herd body takes 1 coordinate argument(s) but the iteration space is 2-D
@expect(TypeError, "herd_arity")
def _():
    M = N = 128
    A = air.tensor([M, N], bf16)

    with air.launch(name="k") as launch:

        @launch.body
        def _():
            with air.herd(product(range(0, M, 64), range(0, N, 64))) as h:

                @h.body
                def _(tx):
                    pass

    launch.mlir()


# CHECK-LABEL: TEST: no_output
# CHECK: RuntimeError: kernel writes no output
@expect(RuntimeError, "no_output")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64, 64], bf16, scope=h.private())
        air.ops.load(a, A[0:64, 0:64])

    _trace(body)


# CHECK-LABEL: TEST: output_before_input
# CHECK: RuntimeError: output tensors must be declared after all input tensors
@expect(RuntimeError, "output_before_input")
def _():
    M = N = 128
    OUT = air.tensor([M, N], bf16)
    IN = air.tensor([M, N], bf16)

    with air.launch(name="k") as launch:

        @launch.body
        def _():
            with air.herd(product(range(0, M, 64), range(0, N, 64)), shape=(2, 2)) as h:

                @h.body
                def _(tx, ty):
                    a = air.alloc([64, 64], bf16, scope=h.private())
                    air.ops.load(a, IN[0:64, 0:64])
                    air.ops.store(a, OUT[0:64, 0:64])

    launch.mlir()


# CHECK-LABEL: TEST: alloc_without_scope
# air.alloc has two scopes to choose between now, so the message names both.
# CHECK: ValueError: air.alloc requires scope=<herd>.private() (L1) or scope=<segment>.private() (L2)
@expect(ValueError, "alloc_without_scope")
def _():
    air.alloc([8, 8], bf16, scope=None)


# CHECK-LABEL: TEST: herd_outside_launch
# CHECK: RuntimeError: air.herd(...) must be used inside a launch body
@expect(RuntimeError, "herd_outside_launch")
def _():
    h = air.herd(range(0, 128, 64), shape=(2,))

    @h.body
    def _(tx):
        pass


# CHECK-LABEL: TEST: bad_herd_shape
# CHECK: ValueError: herd shape (3,{{.*}}) does not evenly divide the logical grid
@expect(ValueError, "bad_herd_shape")
def _():
    def body(h, tx, ty, A, B, C):
        pass

    _trace(body, shape=(3, 2))


def _dot_shapes(a_shape, b_shape, acc_shape):
    """Allocate three L1 tiles with the given shapes and call ops.dot on them."""
    from air.api.types import bf16, f32

    t = air.tensor([64], bf16)
    with air.launch(name="dotchk") as launch:

        @launch.body
        def _():
            with air.herd(range(0, 64, 16), shape=(4,)) as h:

                @h.body
                def _(tx):
                    a = air.alloc(a_shape, bf16, scope=h.private())
                    b = air.alloc(b_shape, bf16, scope=h.private())
                    acc = air.alloc(acc_shape, f32, scope=h.private())
                    air.ops.dot(a, b, acc=acc)

    launch.mlir()


# CHECK-LABEL: TEST: unimplemented_op
# CHECK: NotImplementedError: air.api.ops.reduce is not implemented yet
@expect(NotImplementedError, "unimplemented_op")
def _():
    air.ops.reduce(None, 0, "add")


# ops.dot is implemented, but it is a statement over three L1 buffers, so a
# missing or non-buffer operand is a user error rather than something to default.
# CHECK-LABEL: TEST: dot_requires_buffers
# CHECK: TypeError: air.api.ops.dot requires a=
@expect(TypeError, "dot_requires_buffers")
def _():
    air.ops.dot(None, None)


# CHECK-LABEL: TEST: dot_rank_too_high
# There is no named linalg op past one batch dimension, and a batch axis belongs
# in the herd grid or in air.sequential, not hidden inside the contraction.
# CHECK: NotImplementedError: air.api.ops.dot contracts 1-D and 2-D tiles; got ranks (3, 3)
@expect(NotImplementedError, "dot_rank_too_high")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([2, 8, 8], bf16, scope=h.private())
        acc = air.alloc([2, 8, 8], f32, scope=h.private())
        ops.dot(a, a, acc=acc)

    _trace(body)


# CHECK-LABEL: TEST: dot_vecmat_acc_shape
# The rank rule is uniform: a's last axis contracts against b's first, and acc
# keeps what is left of each -- here (k,) @ (k,n) leaves (n,).
# CHECK: ValueError: air.api.ops.dot shape mismatch for (k,) @ (k, n) -> (n,)
@expect(ValueError, "dot_vecmat_acc_shape")
def _():
    def body(h, tx, ty, A, B, C):
        v = air.alloc([64], bf16, scope=h.private())
        m = air.alloc([64, 32], bf16, scope=h.private())
        acc = air.alloc([64], f32, scope=h.private())  # should be [32]
        ops.dot(v, m, acc=acc)

    _trace(body)


# CHECK-LABEL: TEST: dot_transpose_b_on_a_vector
# CHECK: ValueError: air.api.ops.dot(transpose_b=True) is meaningless for (m, k) @ (k,) -> (m,)
@expect(ValueError, "dot_transpose_b_on_a_vector")
def _():
    def body(h, tx, ty, A, B, C):
        m = air.alloc([32, 64], bf16, scope=h.private())
        v = air.alloc([64], bf16, scope=h.private())
        acc = air.alloc([32], f32, scope=h.private())
        ops.dot(m, v, acc=acc, transpose_b=True)

    _trace(body)


# CHECK-LABEL: TEST: dot_shape_mismatch
# CHECK: ValueError: air.api.ops.dot shape mismatch
@expect(ValueError, "dot_shape_mismatch")
def _():
    _dot_shapes((16, 32), (64, 16), (16, 16))


# CHECK-LABEL: TEST: nonaffine_index
# CHECK: TypeError: non-affine index expression
@expect(TypeError, "nonaffine_index")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64, 64], bf16, scope=h.private())
        air.ops.load(a, A[tx * ty : tx * ty + 64, 0:64])

    _trace(body)


# CHECK-LABEL: TEST: bad_grid_type
# CHECK: TypeError: cannot use list as a herd iteration space
@expect(TypeError, "bad_grid_type")
def _():
    air.herd([0, 1, 2, 3])


# A range that does not tile its extent exactly would put the last tile past
# the end of the tensor.
# CHECK-LABEL: TEST: partial_tile_range
# CHECK: ValueError: iteration range range(0, 3000, 1024) does not tile its extent exactly
@expect(ValueError, "partial_tile_range")
def _():
    air.herd(range(0, 3000, 1024))


# CHECK-LABEL: TEST: nonzero_start_range
# CHECK: NotImplementedError: air.api requires a herd iteration space starting at 0
@expect(NotImplementedError, "nonzero_start_range")
def _():
    air.herd(range(1024, 3072, 1024))


# The product path keeps only the offsets, so the start is still checked there.
# CHECK-LABEL: TEST: nonzero_start_product
# CHECK: NotImplementedError: air.api requires a herd iteration space starting at 0
@expect(NotImplementedError, "nonzero_start_product")
def _():
    air.herd(product(range(0, 128, 64), range(64, 192, 64)))


# CHECK-LABEL: TEST: bad_dependency
# CHECK: TypeError: dependency= expects a Token
@expect(TypeError, "bad_dependency")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64, 64], bf16, scope=h.private())
        air.ops.load(a, A[0:64, 0:64], dependency="not a token")

    _trace(body)


# CHECK-LABEL: TEST: symbol_hint_is_a_list
# CHECK: TypeError: air.symbol(hint=...) takes a single integer{{.*}}choices=
@expect(TypeError, "symbol_hint_is_a_list")
def _():
    air.symbol(hint=[32, 64])


# CHECK-LABEL: TEST: symbol_choices_is_a_scalar
# CHECK: TypeError: air.symbol(choices=...) takes a sequence of integers
@expect(TypeError, "symbol_choices_is_a_scalar")
def _():
    air.symbol(choices=32)


# CHECK-LABEL: TEST: symbol_choices_empty
# CHECK: ValueError: air.symbol(choices=[]) has no candidates
@expect(ValueError, "symbol_choices_empty")
def _():
    air.symbol(choices=[])


# CHECK-LABEL: TEST: symbol_hint_not_in_choices
# CHECK: ValueError: air.symbol(hint=64) is not one of choices=[16, 32]
@expect(ValueError, "symbol_hint_not_in_choices")
def _():
    air.symbol(choices=[16, 32], hint=64)


# CHECK-LABEL: TEST: l1_budget_exceeded
# CHECK: ValueError: L1 budget exceeded: this herd body has 96.0 KB of live buffers
@expect(ValueError, "l1_budget_exceeded")
def _():
    def body(h, tx, ty, A, B, C):
        for _ in range(3):
            air.alloc([128, 128], bf16, scope=h.private())

    _trace(body, grid=(256, 256, 128), shape=(2, 2))


# CHECK-LABEL: TEST: single_alloc_larger_than_l1
# CHECK: ValueError: air.alloc([256, 256], air.api.bf16) needs 128.0 KB
@expect(ValueError, "single_alloc_larger_than_l1")
def _():
    def body(h, tx, ty, A, B, C):
        air.alloc([256, 256], bf16, scope=h.private())

    _trace(body, grid=(512, 512, 256), shape=(2, 2))


# A caller's own "tile must be a multiple of the vector width" guard does not
# catch this: Python's modulo is 0 for any divisor of the tile, sign included,
# so -4 passes it and then quietly selects the scalar path.
# CHECK-LABEL: TEST: negative_vector_width
# CHECK: ValueError: air.alloc vector width must be >= 0, got -4
@expect(ValueError, "negative_vector_width")
def _():
    def body(h, tx, ty, A, B, C):
        air.alloc([64, 64], bf16, scope=h.private(), vector=-4)

    _trace(body)


# ---------------------------------------------------------------------------
# air.sequential and the K reduction
# ---------------------------------------------------------------------------


# CHECK-LABEL: TEST: product_single_tile_axis
# itertools.product materialises its inputs, so a one-tile axis arrives as (0,)
# with its step gone. Guessing 1 would silently compute on 1x1 tiles.
# CHECK: ValueError: air.herd cannot recover the tile size of a single-tile axis
@expect(ValueError, "product_single_tile_axis")
def _():
    air.herd(product(range(0, 64, 64), range(0, 128, 32)))


# CHECK-LABEL: TEST: sequential_partial_trip
# CHECK: ValueError: air.sequential(0, 100, 32) does not tile its extent exactly
@expect(ValueError, "sequential_partial_trip")
def _():
    def body(h, tx, ty, A, B, C):
        for _ in air.sequential(0, 100, 32):
            pass

    _trace(body)


# CHECK-LABEL: TEST: sequential_negative_step
# CHECK: ValueError: air.sequential needs a positive step, got -1
@expect(ValueError, "sequential_negative_step")
def _():
    def body(h, tx, ty, A, B, C):
        for _ in air.sequential(0, 64, -1):
            pass

    _trace(body)


# CHECK-LABEL: TEST: sequential_non_integer_bound
# CHECK: TypeError: air.sequential(stop=...) takes a Python integer
@expect(TypeError, "sequential_non_integer_bound")
def _():
    def body(h, tx, ty, A, B, C):
        for _ in air.sequential(0, 64.0, 32):
            pass

    _trace(body)


# CHECK-LABEL: TEST: sequential_bound_from_tile_coordinate
# A loop bound is resolved at trace time; a tile coordinate is an SSA value and
# cannot be one.
# CHECK: TypeError: air.sequential(stop=...) takes a Python integer
@expect(TypeError, "sequential_bound_from_tile_coordinate")
def _():
    def body(h, tx, ty, A, B, C):
        for _ in air.sequential(0, tx, 1):
            pass

    _trace(body)


# CHECK-LABEL: TEST: alloc_inside_sequential
# The herd frees its buffers after the body, which is outside the loop, so the
# dealloc would not be dominated by its alloc.
# CHECK: NotImplementedError: air.alloc inside an air.sequential body is not supported
@expect(NotImplementedError, "alloc_inside_sequential")
def _():
    def body(h, tx, ty, A, B, C):
        for _ in air.sequential(0, 64, 32):
            air.alloc([32, 32], bf16, scope=h.private())

    _trace(body)


# CHECK-LABEL: TEST: break_out_of_sequential
# An air.sequential body is traced once and stands for every trip, so breaking out
# truncates all of them rather than shortening the loop.
# CHECK: RuntimeError: a body left an air.sequential loop early
@expect(RuntimeError, "break_out_of_sequential")
def _():
    def body(h, tx, ty, A, B, C):
        buf = air.alloc([32, 32], bf16, scope=h.private())
        for _ in air.sequential(0, 64, 32):
            buf[:] = buf[:] + 1.0
            break

    _trace(body)


# CHECK-LABEL: TEST: dot_alpha_unimplemented
# CHECK: NotImplementedError: air.api.ops.dot(alpha=...) is not implemented
@expect(NotImplementedError, "dot_alpha_unimplemented")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([32, 32], bf16, scope=h.private())
        b = air.alloc([32, 32], bf16, scope=h.private())
        acc = air.alloc([32, 32], f32, scope=h.private())
        ops.dot(a, b, acc=acc, alpha=2.0)

    _trace(body)


# CHECK-LABEL: TEST: parallel_unimplemented
# The unordered counterpart of air.sequential. Declared so that reaching for it
# says what it would be, rather than raising AttributeError.
# CHECK: NotImplementedError: air.api.parallel is not implemented yet
@expect(NotImplementedError, "parallel_unimplemented")
def _():
    air.parallel(0, 64, 16)


# ---------------------------------------------------------------------------
# air.segment and L2 staging
# ---------------------------------------------------------------------------


def _staged(body, l2_shape=(64, 64), dtype=bf16):
    """Build a launch whose *segment* body is ``body(seg, A, C)``."""
    A = air.tensor(list(l2_shape), dtype)
    C = air.tensor(list(l2_shape), dtype)

    with air.launch(name="k") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    body(seg, A, C)

    return launch.mlir()


# CHECK-LABEL: TEST: elementwise_on_l2
# A memtile has DMA engines but no compute core, so an L2 buffer can only be a
# transfer endpoint.
# CHECK: TypeError: cannot read an L2 buffer elementwise
@expect(TypeError, "elementwise_on_l2")
def _():
    def body(seg, A, C):
        staged = air.alloc([64, 64], bf16, scope=seg.private())
        staged[:] = staged[:] + 1.0

    _staged(body)


# CHECK-LABEL: TEST: l2_alloc_outside_segment
# CHECK: RuntimeError: this operation must be used inside a segment body
@expect(RuntimeError, "l2_alloc_outside_segment")
def _():
    seg = air.segment(name="detached")

    def body(h, tx, ty, A, B, C):
        air.alloc([32, 32], bf16, scope=seg.private())

    _trace(body)


# CHECK-LABEL: TEST: l1_alloc_outside_herd
# CHECK: RuntimeError: this operation must be used inside a herd body
@expect(RuntimeError, "l1_alloc_outside_herd")
def _():
    def body(seg, A, C):
        h = air.herd(range(0, 64, 32), shape=(2,))
        air.alloc([32, 32], bf16, scope=h.private())

    _staged(body)


# 512 KB per memtile and one memtile per column, so the budget is the whole
# device's L2 -- 2 MB on npu1, 4 MB on npu2. Deliberately a "certainly
# impossible" test, not a placement prediction: the herds do not exist when the
# L2 allocs run, so the number of columns the segment will span is not yet
# known. A per-memtile cap would refuse matrix_multiplication at herd 4x4 with
# an f32 output, which stages 608 KB across four memtiles and runs.
#
# The allocation has to exceed the *largest* device budget, not just npu1's.
# This test has no explicit target, so it resolves against whatever part is
# installed; at exactly 4 MB it raises on npu1 and passes silently on npu2,
# which would leave the case untested on half the CI fleet. The CHECK likewise
# stops before the capacity, which is device-dependent.
# CHECK-LABEL: TEST: l2_budget_exceeded
# CHECK: ValueError: air.alloc([4096, 1024], air.api.bf16) needs 8192.0 KB
@expect(ValueError, "l2_budget_exceeded")
def _():
    def body(seg, A, C):
        air.alloc([4096, 1024], bf16, scope=seg.private())

    _staged(body)


# A segment grid is the launch iteration space, and air.launch is 2-D.
# CHECK-LABEL: TEST: segment_grid_too_deep
# CHECK: NotImplementedError: air.launch is 2-D, so a segment grid is 1-D or 2-D
@expect(NotImplementedError, "segment_grid_too_deep")
def _():
    air.segment(product(range(0, 128, 64), range(0, 128, 64), range(0, 128, 64)))


# CHECK-LABEL: TEST: segment_body_takes_no_args
# CHECK: TypeError: segment body takes 1 coordinate argument(s) but the launch iteration space is 0-D
@expect(TypeError, "segment_body_takes_no_args")
def _():
    air.tensor([64, 64], bf16)
    air.tensor([64, 64], bf16)

    with air.launch(name="k") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _(sx):
                    pass

    launch.mlir()


# CHECK-LABEL: TEST: buffer_slice_in_expression
# A partial subscript names a DMA region, not a value.
# CHECK: TypeError: cannot use BufferSlice
@expect(TypeError, "buffer_slice_in_expression")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([32, 32], bf16, scope=h.private())
        b = air.alloc([32, 32], bf16, scope=h.private())
        b[:] = a[0:16, :]

    _trace(body)


# CHECK-LABEL: TEST: partial_assignment_into_buffer
# CHECK: NotImplementedError: partial assignment into a buffer is not supported
@expect(NotImplementedError, "partial_assignment_into_buffer")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([32, 32], bf16, scope=h.private())
        a[0:16, :] = 1.0

    _trace(body)


# CHECK-LABEL: TEST: staged_transfer_shape_mismatch
# Only *leading* unit dimensions are squeezed, so a genuine mismatch still
# fails rather than being reshaped into something plausible.
# CHECK: ValueError: transfer shape mismatch in air.api.ops.load
@expect(ValueError, "staged_transfer_shape_mismatch")
def _():
    def body(seg, A, C):
        staged = air.alloc([64, 64], bf16, scope=seg.private())

        with air.herd(range(0, 2, 1), shape=(2,)) as h:

            @h.body
            def _(tx):
                l1 = air.alloc([16], bf16, scope=h.private())
                ops.load(l1, staged[tx, 0:32])

    _staged(body)


# CHECK-LABEL: TEST: load_first_argument_must_be_a_buffer
# CHECK: TypeError: air.api.ops.load fills a buffer
@expect(TypeError, "load_first_argument_must_be_a_buffer")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([32, 32], bf16, scope=h.private())
        ops.load(A[0:32, 0:32], a)

    _trace(body)


# ---------------------------------------------------------------------------
# Micro-tiled (packed) layouts
#
# Every one of these is a case where guessing would produce a kernel that runs
# and is silently wrong -- a mismatched micro-tile contracts the wrong elements,
# and a whole-buffer drain copies a tile still in micro-tile order.
# ---------------------------------------------------------------------------


def _mm():
    return air.micro_tile(m=4, k=8, n=4)


# CHECK-LABEL: TEST: micro_tile_does_not_divide
# CHECK: ValueError: operand A's M extent 30 is not a multiple of the micro-tile m=4
@expect(ValueError, "micro_tile_does_not_divide")
def _():
    _mm().a(30, 32)


# CHECK-LABEL: TEST: packed_buffer_no_whole_drain
# A whole-buffer store emits `[] [] []`, a contiguous read, which would copy the
# tile still micro-tiled. The unpack *is* the access pattern.
# CHECK: TypeError: air.api.ops.store cannot drain a micro-tiled buffer whole
@expect(TypeError, "packed_buffer_no_whole_drain")
def _():
    def body(seg, A, C):
        mm = _mm()
        l2 = air.alloc([1, 1, 32, 32], bf16, scope=seg.private())
        acc = air.alloc(mm.c(32, 32), bf16, scope=seg.shared())
        ops.store(acc, l2[0, 0, :, :])

    _staged(body)


# CHECK-LABEL: TEST: packed_load_needs_a_region
# CHECK: TypeError: air.api.ops.load into a micro-tiled buffer needs a source *region*
@expect(TypeError, "packed_load_needs_a_region")
def _():
    def body(h, tx, ty, A, B, C):
        l1 = air.alloc(_mm().a(32, 16), bf16, scope=h.private())
        other = air.alloc([1, 1, 32, 32], bf16, scope=h.private())
        ops.load(l1, other)

    _trace(body)


# CHECK-LABEL: TEST: packed_operands_must_share_a_micro_tile
# CHECK: ValueError: air.api.ops.dot needs one micro-tile across all three operands
@expect(ValueError, "packed_operands_must_share_a_micro_tile")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc(air.micro_tile(4, 8, 4).a(32, 16), bf16, scope=h.private())
        b = air.alloc(air.micro_tile(4, 8, 8).b(16, 32), bf16, scope=h.private())
        c = air.alloc(air.micro_tile(4, 8, 4).c(32, 32), bf16, scope=h.private())
        ops.dot(a, b, acc=c)

    _trace(body)


# CHECK-LABEL: TEST: packed_operand_roles_must_match
# CHECK: ValueError: air.api.ops.dot expects b to be a micro-tiled B operand
@expect(ValueError, "packed_operand_roles_must_match")
def _():
    def body(h, tx, ty, A, B, C):
        mm = _mm()
        a = air.alloc(mm.a(32, 16), bf16, scope=h.private())
        b = air.alloc(mm.a(32, 16), bf16, scope=h.private())
        c = air.alloc(mm.c(32, 32), bf16, scope=h.private())
        ops.dot(a, b, acc=c)

    _trace(body)


# CHECK-LABEL: TEST: packed_contraction_extents_must_agree
# CHECK: ValueError: air.api.ops.dot shape mismatch: a is 32x16 and b is 32x32
@expect(ValueError, "packed_contraction_extents_must_agree")
def _():
    def body(h, tx, ty, A, B, C):
        mm = _mm()
        a = air.alloc(mm.a(32, 16), bf16, scope=h.private())
        b = air.alloc(mm.b(32, 32), bf16, scope=h.private())
        c = air.alloc(mm.c(32, 32), bf16, scope=h.private())
        ops.dot(a, b, acc=c)

    _trace(body)


# CHECK-LABEL: TEST: unpacked_operand_in_packed_dot
# CHECK: TypeError: air.api.ops.dot got rank-6 operands
@expect(TypeError, "unpacked_operand_in_packed_dot")
def _():
    def body(h, tx, ty, A, B, C):
        mm = _mm()
        a = air.alloc(mm.a(32, 16), bf16, scope=h.private())
        b = air.alloc([1, 1, 8, 2, 8, 4], bf16, scope=h.private())
        c = air.alloc(mm.c(32, 32), bf16, scope=h.private())
        ops.dot(a, b, acc=c)

    _trace(body)


# CHECK-LABEL: TEST: no_elementwise_expression_on_packed
# The elements are not in row-major order, so an elementwise expression over a
# packed buffer would not mean what it reads like.
# CHECK: NotImplementedError: only a scalar fill is supported on a micro-tiled buffer
@expect(NotImplementedError, "no_elementwise_expression_on_packed")
def _():
    def body(h, tx, ty, A, B, C):
        mm = _mm()
        c = air.alloc(mm.c(32, 32), bf16, scope=h.private())
        d = air.alloc(mm.c(32, 32), bf16, scope=h.private())
        c[:] = d[:]

    _trace(body)


# CHECK-LABEL: TEST: no_herd_shared_scope
# CHECK: NotImplementedError: air.api has no <herd>.shared()
@expect(NotImplementedError, "no_herd_shared_scope")
def _():
    def body(h, tx, ty, A, B, C):
        air.alloc([32, 32], bf16, scope=h.shared())

    _trace(body)


# CHECK-LABEL: TEST: dot_on_an_l2_buffer
# A contraction runs on a core; a memtile has DMA engines and none. The
# operand check said "L1 buffer" long before it checked for one.
# CHECK: TypeError: air.api.ops.dot expects a to be an L1 buffer, but it is in L2
@expect(TypeError, "dot_on_an_l2_buffer")
def _():
    def body(seg, A, C):
        staged = air.alloc([32, 32], bf16, scope=seg.private())
        ops.dot(staged, staged, acc=staged)

    _staged(body)


# CHECK-LABEL: TEST: shared_alloc_needs_a_packed_shape
# The per-core L1 charge depends on knowing which leading dimensions are the
# herd; a plain shape does not say, and guessing either way misreports the
# budget.
# CHECK: NotImplementedError: <segment>.shared() currently requires a micro-tiled shape
@expect(NotImplementedError, "shared_alloc_needs_a_packed_shape")
def _():
    def body(seg, A, C):
        air.alloc([1, 1, 32, 32], bf16, scope=seg.shared())

    _staged(body)


# CHECK-LABEL: TEST: dot_kernel_must_be_a_name
# CHECK: TypeError: air.api.ops.dot(kernel=...) takes the external function's symbol name
@expect(TypeError, "dot_kernel_must_be_a_name")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([32, 32], bf16, scope=h.private())
        acc = air.alloc([32, 32], f32, scope=h.private())
        ops.dot(a, a, acc=acc, kernel=42)

    _trace(body)


# CHECK-LABEL: TEST: dot_kernel_is_keyword_only
# kernel= sits after dependency and is keyword-only, so adding it did not shift
# any existing positional binding. Passing it positionally is refused rather
# than silently rebinding whatever was in that slot.
# CHECK: TypeError: dot() takes from 2 to 6 positional arguments but 7 were given
@expect(TypeError, "dot_kernel_is_keyword_only")
def _():
    ops.dot(None, None, None, 1.0, False, None, "matmul_bf16")


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------


# CHECK-LABEL: TEST: channel_broadcast_needs_size
# broadcast_shape describes a fan-out relative to the channel's own extents, so
# it is meaningless without them.
# CHECK: ValueError: air.channel(broadcast_shape=...) also needs size=
@expect(ValueError, "channel_broadcast_needs_size")
def _():
    air.channel("C", broadcast_shape=[1, 3])


# CHECK-LABEL: TEST: channel_broadcast_rank
# CHECK: ValueError: air.channel: broadcast_shape [1, 3, 3] has rank 3 but size [1, 1] has rank 2
@expect(ValueError, "channel_broadcast_rank")
def _():
    air.channel("C", size=[1, 1], broadcast_shape=[1, 3, 3])


# CHECK-LABEL: TEST: channel_broadcast_not_multiple
# A fan-out has to be a whole number of destinations per source.
# CHECK: ValueError: air.channel: broadcast_shape [1, 3] is not a whole multiple of size [1, 2]
@expect(ValueError, "channel_broadcast_not_multiple")
def _():
    air.channel("C", size=[1, 2], broadcast_shape=[1, 3])


# CHECK-LABEL: TEST: channel_size_scalar
# size=[2] is a 1-D array of two channels; size=2 is a mistake worth naming,
# because it would otherwise iterate an int and fail somewhere less obvious.
# CHECK: TypeError: air.channel(size=...) takes a list of extents
@expect(TypeError, "channel_size_scalar")
def _():
    air.channel("C", size=2)


# CHECK-LABEL: TEST: channel_type_unsupported
# Accepting channel_type and ignoring it would compile a cascade request as a
# DMA stream -- the silent-wrongness this package exists to avoid.
# CHECK: NotImplementedError: air.api does not implement channel_type=
@expect(NotImplementedError, "channel_type_unsupported")
def _():
    air.channel("C", channel_type="npu_cascade")


# CHECK-LABEL: TEST: channel_indices_without_size
# CHECK: ValueError: air.channel 'C' was declared without size=
@expect(ValueError, "channel_indices_without_size")
def _():
    ch = air.channel("C")

    def body(h, tx, ty, A, B, C):
        buf = air.alloc([64], bf16, scope=h.private())
        ch.get(buf, indices=[0])

    _trace(body)


# CHECK-LABEL: TEST: channel_indices_rank
# CHECK: ValueError: air.channel 'C' has size [2, 2], so it takes 2 index/indices; got 1
@expect(ValueError, "channel_indices_rank")
def _():
    ch = air.channel("C", size=[2, 2])

    def body(h, tx, ty, A, B, C):
        buf = air.alloc([64], bf16, scope=h.private())
        ch.get(buf, indices=[0])

    _trace(body)


# CHECK-LABEL: TEST: channel_index_out_of_range
# Only a constant index can be checked here; a herd coordinate is bounded by the
# herd shape instead.
# CHECK: ValueError: air.channel 'C' index 3 is out of range on axis 0
@expect(ValueError, "channel_index_out_of_range")
def _():
    ch = air.channel("C", size=[2, 2])

    def body(h, tx, ty, A, B, C):
        buf = air.alloc([64], bf16, scope=h.private())
        ch.get(buf, indices=[3, 0])

    _trace(body)


# CHECK-LABEL: TEST: channel_l3_needs_segment
# Reaching L3 needs a shim DMA allocation, which only a segment brings. Measured
# on npu1: the same design with its put hoisted out of the segment fails in
# air-to-aie with "failed to link to any shim dma allocation", so this raises at
# the call site instead, naming the fix.
# CHECK: RuntimeError: air.channel.put on an L3 tensor has to be inside an air.segment
@expect(RuntimeError, "channel_l3_needs_segment")
def _():
    ch = air.channel("C")

    def body(h, tx, ty, A, B, C):
        ch.put(A)

    _trace(body)


# CHECK-LABEL: TEST: channel_bad_endpoint
# CHECK: TypeError: air.api.ops.channel.get expects its argument to be a buffer
@expect(TypeError, "channel_bad_endpoint")
def _():
    ch = air.channel("C")

    def body(h, tx, ty, A, B, C):
        ch.get([1, 2, 3])

    _trace(body)
