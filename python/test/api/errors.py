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
from air.api.types import bf16, f16, f32, i8, i16, i32, ui8


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


# CHECK-LABEL: TEST: dot_transpose_b_wrong_axis
# With transpose_b=True, B is [n, k] -- so the contracting axis is its *last*,
# and passing an ordinary [k, n] operand is caught rather than contracted along
# the wrong axis. The message says which convention is in force, because the
# two spellings differ only in a keyword.
# CHECK: ValueError: air.api.ops.dot shape mismatch for (m, k) @ (k, n) -> (m, n)
# CHECK: with transpose_b=True, b is [n, k]
@expect(ValueError, "dot_transpose_b_wrong_axis")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([32, 64], bf16, scope=h.private())
        b = air.alloc([64, 32], bf16, scope=h.private())  # [k, n], not [n, k]
        acc = air.alloc([32, 32], f32, scope=h.private())
        ops.dot(a, b, acc=acc, transpose_b=True)

    _trace(body)


# CHECK-LABEL: TEST: dot_transpose_b_acc_shape
# CHECK: ValueError: air.api.ops.dot shape mismatch for (m, k) @ (k, n) -> (m, n): a . b is (32, 16) but acc is (32, 32)
@expect(ValueError, "dot_transpose_b_acc_shape")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([32, 64], bf16, scope=h.private())
        b = air.alloc([16, 64], bf16, scope=h.private())  # n = 16
        acc = air.alloc([32, 32], f32, scope=h.private())
        ops.dot(a, b, acc=acc, transpose_b=True)

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


# CHECK-LABEL: TEST: nonaffine_divisor
# `x // k` and `x % k` are affine for a constant k, and only for a constant k:
# affine.floordiv and affine.mod both require a literal right-hand side. A
# coordinate divisor has to say so, rather than reporting the operation itself
# as unsupported -- it is the divisor that is the problem.
# CHECK: TypeError: non-affine index expression: cannot take floordiv of {{.*}} by {{.*}} (the divisor must be a constant, not a tile coordinate)
@expect(TypeError, "nonaffine_divisor")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64, 64], bf16, scope=h.private())
        row = tx // ty
        air.ops.load(a, A[row : row + 64, 0:64])

    _trace(body)


# CHECK-LABEL: TEST: nonpositive_divisor
# CHECK: ValueError: index mod by 0: the divisor must be a positive constant
@expect(ValueError, "nonpositive_divisor")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64, 64], bf16, scope=h.private())
        row = tx % 0
        air.ops.load(a, A[row : row + 64, 0:64])

    _trace(body)


# CHECK-LABEL: TEST: use_after_dealloc
# air.dealloc ends a buffer's life, so a later read is a use of something the
# program has said is gone. It is reported when the body finishes rather than
# at the air.dealloc call: whether a later use exists is not knowable until the
# rest of the body has been traced.
# CHECK: ValueError: air.dealloc released this buffer before its last use
@expect(ValueError, "use_after_dealloc")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64, 64], bf16, scope=h.private())
        b = air.alloc([64, 64], bf16, scope=h.private())
        air.ops.load(a, A[0:64, 0:64])
        air.dealloc(a)
        b[:] = a[:] + 1.0

    _trace(body)


# CHECK-LABEL: TEST: double_dealloc
# CHECK: ValueError: air.dealloc: this buffer has already been released
@expect(ValueError, "double_dealloc")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64, 64], bf16, scope=h.private())
        air.ops.load(a, A[0:64, 0:64])
        air.dealloc(a)
        air.dealloc(a)

    _trace(body)


# CHECK-LABEL: TEST: dealloc_not_a_buffer
# CHECK: TypeError: air.dealloc takes a buffer from air.alloc, got Tensor
@expect(TypeError, "dealloc_not_a_buffer")
def _():
    def body(h, tx, ty, A, B, C):
        air.dealloc(A)

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
# CHECK: NotImplementedError: air.alloc inside an air.sequential or ops.branch body
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


# CHECK-LABEL: TEST: a_parallel_loop_must_tile_its_extent
# The same rule air.sequential applies, for the same reason: there are no
# partial trips, so a step that does not divide the extent would send the last
# one off the end of whatever it indexes. Worth pinning separately because the
# trips of a parallel loop are slots of a spatial fan-out -- overrunning is a
# put addressed to a destination that does not exist, not a short read.
# CHECK: ValueError: air.parallel(0, 64, 24) does not tile its extent exactly
@expect(ValueError, "a_parallel_loop_must_tile_its_extent")
def _():
    # Generators are lazy: the bounds are checked when the first trip is
    # requested, so the loop has to actually be entered.
    for _ in air.parallel(0, 64, 24):
        pass


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


# The segment's iteration space is now emitted rather than refused, so the
# positive cases live in hierarchy.py. What stays here is its arity rule, which
# is the same one air.launch and air.herd follow: a body sees exactly as many
# coordinates as the grid it was given, and a gridless segment sees none. The
# message has to keep pointing at air.launch for the outer tiling, because
# reaching for air.segment to spell that is the original conflation.
# CHECK-LABEL: TEST: segment_body_arity
# CHECK: TypeError: segment body takes 0 coordinate argument(s) but the segment iteration space is 1-D
@expect(TypeError, "segment_body_arity")
def _():
    A = air.tensor([128], bf16)
    B = air.tensor([128], bf16)

    with air.launch(name="k") as launch:

        @launch.body
        def _():
            with air.segment([range(2)], name="seg") as seg:

                @seg.body
                def _():
                    pass

    launch.mlir()


# CHECK-LABEL: TEST: gridless_segment_body_arity
# CHECK: TypeError: segment body takes 1 coordinate argument(s) but the segment iteration space is 0-D
# CHECK-SAME: Outer tiling belongs on air.launch
@expect(TypeError, "gridless_segment_body_arity")
def _():
    A = air.tensor([128], bf16)
    B = air.tensor([128], bf16)

    with air.launch(name="k") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _(u0):
                    pass

    launch.mlir()


# CHECK-LABEL: TEST: segment_grid_too_deep
# CHECK: NotImplementedError: an air.segment iteration space is 1-D or 2-D; got 3-D
@expect(NotImplementedError, "segment_grid_too_deep")
def _():
    air.segment(product(range(0, 128, 64), range(0, 128, 64), range(0, 128, 64)))


# CHECK-LABEL: TEST: launch_grid_too_deep
# CHECK: NotImplementedError: air.launch is 1-D or 2-D; got 3-D
@expect(NotImplementedError, "launch_grid_too_deep")
def _():
    air.launch(product(range(0, 128, 64), range(0, 128, 64), range(0, 128, 64)))


# CHECK-LABEL: TEST: launch_body_arity
# The launch's coordinates arrive in the launch body, so its arity has to match
# the grid it was given -- the same rule air.herd and air.segment follow.
# CHECK: TypeError: launch body takes 0 coordinate argument(s) but the launch iteration space is 1-D
@expect(TypeError, "launch_body_arity")
def _():
    air.tensor([64, 64], bf16)
    air.tensor([64, 64], bf16)

    with air.launch([range(0, 128, 64)], name="k") as launch:

        @launch.body
        def _():
            pass

    launch.mlir()


# CHECK-LABEL: TEST: segment_body_takes_no_args
# CHECK: TypeError: segment body takes 1 coordinate argument(s) but the segment iteration space is 0-D
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


# CHECK-LABEL: TEST: trailing_squeeze_does_not_excuse_a_real_extent
# A trailing unit dimension is squeezed so a reduction's [m, 1] tile can reach
# a rank-1 [m] slice. That is about *unit* axes only: a trailing axis with a
# real extent still has to match, or the transfer would silently move a
# different number of elements.
# CHECK: ValueError: transfer shape mismatch in air.api.ops.store
@expect(ValueError, "trailing_squeeze_does_not_excuse_a_real_extent")
def _():
    def body(h, tx, ty, A, B, C):
        o = air.alloc([16, 4], bf16, scope=h.private())
        ops.store(o, C[0:16, 0:1])

    _trace(body)


# CHECK-LABEL: TEST: staged_transfer_shape_mismatch
# Only *unit* dimensions are squeezed, and only from the ends, so a genuine
# mismatch still fails rather than being reshaped into something plausible.
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
# Blocked layouts
#
# A tile is laid out block-first by reshaping and transposing the region being
# moved, so the failures here are numpy's failures: a reshape with no view, a
# permutation that is not one, a transfer whose two ends hold different numbers
# of elements. Every one of them would otherwise produce a kernel that runs and
# is silently wrong -- a mismatched block contracts the wrong elements, and a
# whole-buffer drain copies a tile still in block order.
# ---------------------------------------------------------------------------


# The AIE2 bf16 block is 4x8 by 8x4. Written out rather than generated, so the
# shapes below say what they are.
def _a(M, K, m=4, k=8):
    return [1, 1, K // k, M // m, m, k]


def _b(K, N, k=8, n=4):
    return [1, 1, N // n, K // k, k, n]


def _c(M, N, m=4, n=4, lead=(1, 1)):
    return list(lead) + [N // n, M // m, m, n]


# CHECK-LABEL: TEST: reshape_element_count_must_match
# numpy's rule: a reshape rearranges, it never adds or drops elements.
# CHECK: ValueError: cannot reshape a (30, 32) view (960 elements)
@expect(ValueError, "reshape_element_count_must_match")
def _():
    def body(seg, A, C):
        l2 = air.alloc([32, 32], bf16, scope=seg.private())
        l2[0:30, 0:32].reshape(1, 1, 8, 4, 4, 8)

    _staged(body)


# CHECK-LABEL: TEST: reshape_must_be_a_view
# A reshape that no stride can describe would need a copy, and a hidden copy
# here is a hidden L2 transfer. numpy copies silently; this refuses.
# CHECK: ValueError: cannot reshape a (4, 8) view with strides
@expect(ValueError, "reshape_must_be_a_view")
def _():
    def body(seg, A, C):
        l2 = air.alloc([32, 32], bf16, scope=seg.private())
        # Rows 0:4 of a 32-wide buffer are not contiguous with each other, so
        # the 32 elements they hold cannot be walked as one axis.
        l2[0:4, 0:8].reshape(32)

    _staged(body)


# CHECK-LABEL: TEST: transpose_takes_a_full_permutation
# As in numpy: transpose reorders every axis, so a partial list is an error
# rather than an implied identity on the rest.
# CHECK: ValueError: transpose(1, 0) is not a permutation of a rank-4 view
@expect(ValueError, "transpose_takes_a_full_permutation")
def _():
    def body(seg, A, C):
        l2 = air.alloc([32, 32], bf16, scope=seg.private())
        l2[0:32, 0:32].reshape(8, 4, 4, 8).transpose(1, 0)

    _staged(body)


# CHECK-LABEL: TEST: blocked_buffer_no_whole_drain
# A whole-buffer store emits `[] [] []`, a contiguous read, which would copy the
# tile still in block order. The unpack *is* the access pattern, so the source
# has to be subscripted and permuted for one to exist.
# CHECK: ValueError: transfer shape mismatch in air.api.ops.store
@expect(ValueError, "blocked_buffer_no_whole_drain")
def _():
    def body(seg, A, C):
        l2 = air.alloc([1, 1, 32, 32], bf16, scope=seg.private())
        acc = air.alloc(_c(32, 32), bf16, scope=seg.shared())
        ops.store(acc, l2[0, 0, :, :])

    _staged(body)


# CHECK-LABEL: TEST: blocked_load_shape_must_match
# Filling a blocked tile from an unpermuted region moves the right number of
# elements in the wrong order, so the shapes are what catch it.
# CHECK: ValueError: transfer shape mismatch in air.api.ops.load
@expect(ValueError, "blocked_load_shape_must_match")
def _():
    def body(h, tx, ty, A, B, C):
        l1 = air.alloc(_a(32, 16), bf16, scope=h.private())
        other = air.alloc([1, 1, 32, 32], bf16, scope=h.private())
        ops.load(l1, other)

    _trace(body)


# CHECK-LABEL: TEST: blocked_operands_must_share_a_block
# CHECK: ValueError: air.api.ops.dot block mismatch: a x b is 4x8 per block
@expect(ValueError, "blocked_operands_must_share_a_block")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc(_a(32, 16), bf16, scope=h.private())
        b = air.alloc(_b(16, 32, n=8), bf16, scope=h.private())
        c = air.alloc(_c(32, 32), bf16, scope=h.private())
        ops.dot(a, b, acc=c)

    _trace(body)


# CHECK-LABEL: TEST: blocked_operands_must_agree_on_k
# Passing an A-shaped tile where B belongs: its last two axes read as (k, n),
# so the k it offers is not the k a offers.
# CHECK: ValueError: air.api.ops.dot block mismatch: a's trailing axes are 4x8, so it offers k=8
@expect(ValueError, "blocked_operands_must_agree_on_k")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc(_a(32, 16), bf16, scope=h.private())
        b = air.alloc(_a(32, 16), bf16, scope=h.private())
        c = air.alloc(_c(32, 32), bf16, scope=h.private())
        ops.dot(a, b, acc=c)

    _trace(body)


# CHECK-LABEL: TEST: blocked_contraction_extents_must_agree
# CHECK: ValueError: air.api.ops.dot shape mismatch: a is 32x16 and b is 32x32
@expect(ValueError, "blocked_contraction_extents_must_agree")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc(_a(32, 16), bf16, scope=h.private())
        b = air.alloc(_b(32, 32), bf16, scope=h.private())
        c = air.alloc(_c(32, 32), bf16, scope=h.private())
        ops.dot(a, b, acc=c)

    _trace(body)


# CHECK-LABEL: TEST: fill_takes_a_scalar
# ops.fill exists so that zeroing an accumulator is one linalg.fill rather than
# a six-deep scalar loop nest; an expression belongs in an assignment.
# CHECK: TypeError: air.api.ops.fill takes a scalar
@expect(TypeError, "fill_takes_a_scalar")
def _():
    def body(h, tx, ty, A, B, C):
        c = air.alloc(_c(32, 32), bf16, scope=h.private())
        ops.fill(c, [0.0])

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


# CHECK-LABEL: TEST: shared_alloc_leaves_room_for_a_tile
# A shared buffer's leading dimensions are the cores, one per herd axis. The
# check waits for a herd because nothing at segment scope knows how many that
# is -- and here the 2-D herd would claim both of a rank-2 buffer's axes,
# leaving each core a slab of nothing.
# CHECK: ValueError: air.alloc([4, 4], air.api.bf16) is herd-shared and the herd
@expect(ValueError, "shared_alloc_leaves_room_for_a_tile")
def _():
    def body(seg, A, C):
        air.alloc([4, 4], bf16, scope=seg.shared())
        with air.herd([range(2), range(2)], name="h") as h:

            @h.body
            def _(tx, ty):
                pass

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
# npu_cascade is implemented now, because it is the one type this package can
# gate on hardware. The rest are still refused: accepting channel_type and
# ignoring it would compile an mmio request as a DMA stream, which is the
# silent-wrongness this package exists to avoid, and each of them has its own
# lowering and verifier rules.
# CHECK: NotImplementedError: air.channel(channel_type='npu_mmio') is not implemented
@expect(NotImplementedError, "channel_type_unsupported")
def _():
    air.channel("C", channel_type="npu_mmio")


# CHECK-LABEL: TEST: channel_type_with_broadcast
# A cascade is a point-to-point link between neighbouring cores, so there is
# nothing for a broadcast shape to describe and asking for both is a mistake
# about what the channel is rather than a combination to resolve.
# CHECK: ValueError: air.channel takes broadcast_shape= or channel_type=, not both
@expect(ValueError, "channel_type_with_broadcast")
def _():
    air.channel("C", size=[2], broadcast_shape=[4], channel_type="npu_cascade")


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


# CHECK-LABEL: TEST: channel_l3_in_a_herd_needs_segment
# Where an L3 endpoint may sit is three separate facts, and this rule used to
# state only one of them -- "it needs a segment" -- which was both too weak and
# too strong. Measured:
#
#   * outside air.launch entirely: "failed to link to any shim dma allocation";
#   * at launch scope with no segment: fine, and data_transfer_transpose/channel
#     is written that way and passes on npu1;
#   * inside a herd with no segment: aircc does not diagnose it, it *crashes*,
#     on a dependencyGraph index assertion in air-dependency.
#
# This pins the third. Crashing the compiler is the worst of the three failure
# modes to inherit, so it raises at the call site with both ways out.
# CHECK: RuntimeError: air.channel.put on an L3 tensor inside a herd body needs an air.segment
# CHECK-SAME: dependencyGraph index assertion
@expect(RuntimeError, "channel_l3_in_a_herd_needs_segment")
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


# CHECK-LABEL: TEST: coord_scalar_into_float
# A herd coordinate broadcasts into an integer expression with an index_cast.
# Into a float one it would need a conversion as well, which nothing has
# required, so it raises rather than guessing a rounding mode.
# CHECK: NotImplementedError: a herd coordinate or loop variable can be broadcast into an integer elementwise expression but not a floating-point one
@expect(NotImplementedError, "coord_scalar_into_float")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64], bf16, scope=h.private())
        a[:] = a[:] + ty

    _trace(body)


# CHECK-LABEL: TEST: whole_tensor_shape_checked
# The bare-tensor spelling is a convenience, not an escape from the shape check.
# CHECK: ValueError: transfer shape mismatch in air.api.ops.load
@expect(ValueError, "whole_tensor_shape_checked")
def _():
    def body(h, tx, ty, A, B, C):
        small = air.alloc([2, 8], bf16, scope=h.private())
        ops.load(small, A)

    _trace(body)


# CHECK-LABEL: TEST: broadcast_index_bounded_by_broadcast_shape
# A broadcast channel is indexed over its destinations, so the bound is
# broadcast_shape, not size -- and the message has to say which, or it reads as
# a contradiction ("index 1 out of range, size is [1, 1]") to someone whose
# 3-destination fan-out is working exactly as declared.
# CHECK: ValueError: air.channel 'BC' index 3 is out of range on axis 0: broadcast_shape is [3, 1], so that axis admits 0..2
@expect(ValueError, "broadcast_index_bounded_by_broadcast_shape")
def _():
    ch = air.channel("BC", size=[1, 1], broadcast_shape=[3, 1])

    def body(h, tx, ty, A, B, C):
        buf = air.alloc([16, 8], bf16, scope=h.private())
        ch.get(buf, indices=[3, 0])

    _trace(body)


# CHECK-LABEL: TEST: broadcast_put_still_bounded_by_size
# The two ends of a broadcast are indexed over different grids. A *get* names a
# destination, so broadcast_shape bounds it (above). A *put* names a slot in the
# source bundle, which is `size` -- widening the bound for puts too would accept
# an out-of-range source and emit an invalid bundle index.
# CHECK: ValueError: air.channel 'BP' index 2 is out of range on axis 0: size is [1, 2], so that axis admits 0..0
@expect(ValueError, "broadcast_put_still_bounded_by_size")
def _():
    ch = air.channel("BP", size=[1, 2], broadcast_shape=[3, 2])

    def body(h, tx, ty, A, B, C):
        buf = air.alloc([16, 8], bf16, scope=h.private())
        ch.put(buf, indices=[2, 0])

    _trace(body)


# CHECK-LABEL: TEST: transpose_axes_must_be_a_permutation
# A repeated axis would visit some elements twice and others never, which is
# not a view of anything -- numpy rejects it for the same reason.
# CHECK: ValueError: transpose(0, 0, 2, 3) is not a permutation of a rank-4 view
@expect(ValueError, "transpose_axes_must_be_a_permutation")
def _():
    ch = air.channel("T")

    def body(h, tx, ty, A, B, C):
        buf = air.alloc([16, 8], bf16, scope=h.private())
        ch.put(buf[0:16, 0:8].reshape(2, 8, 1, 8).transpose(0, 0, 2, 3))

    _trace(body)


# CHECK-LABEL: TEST: channel_put_block_must_divide
# The region's extents have to be whole blocks: 20 is not a multiple of the
# k=16 it would be split into, so the split has no view.
# CHECK: ValueError: cannot reshape a (20, 8) view (160 elements)
@expect(ValueError, "channel_put_block_must_divide")
def _():
    ch = air.channel("T")

    def body(h, tx, ty, A, B, C):
        buf = air.alloc([20, 8], bf16, scope=h.private())
        ch.put(buf[0:20, 0:8].reshape(20 // 16, 16, 1, 8).transpose(2, 0, 1, 3))

    _trace(body)


# CHECK-LABEL: TEST: unsigned_elementwise_operator
# An unsigned tile can be copied but not computed on: every arith op is
# constrained to signless integer operands, so `a[:] + b[:]` on ui8 would build
# an arith.addi that does not verify. Refused at the call site, naming i8.
# CHECK: NotImplementedError: an elementwise operator or broadcast scalar (a plain copy, dst[:] = src[:], is) is not supported for air.api.ui8
# CHECK-SAME: declare it air.api.i8 instead
@expect(NotImplementedError, "unsigned_elementwise_operator")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64], ui8, scope=h.private())
        b = air.alloc([64], ui8, scope=h.private())
        a[:] = a[:] + b[:]

    _trace(body)


# CHECK-LABEL: TEST: unsigned_fill
# A fill is not a copy either: the broadcast scalar is an arith.constant, which
# has no signful form -- `arith.constant 0 : ui8` fails with "integer return
# type must be signless".
# CHECK: NotImplementedError: an elementwise operator or broadcast scalar (a plain copy, dst[:] = src[:], is) is not supported for air.api.ui8
@expect(NotImplementedError, "unsigned_fill")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64], ui8, scope=h.private())
        a[:] = 0

    _trace(body)


# CHECK-LABEL: TEST: unsigned_dot
# A named linalg contraction builds its region out of arith ops -- arith.extsi
# then arith.muli for an integer operand -- so the verifier failure would land
# inside the op rather than at this call.
# CHECK: NotImplementedError: air.api.ops.dot's a operand is not supported for air.api.ui8
@expect(NotImplementedError, "unsigned_dot")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([16, 16], ui8, scope=h.private())
        b = air.alloc([16, 16], ui8, scope=h.private())
        acc = air.alloc([16, 16], i32, scope=h.private())
        ops.dot(a, b, acc=acc)

    _trace(body)


# CHECK-LABEL: TEST: unsigned_extern_scalar
# The *buffer* arguments of an extern kernel may be unsigned -- that is the
# whole point of the type -- but a scalar argument is materialised by
# arith.constant, so it may not be. Caught at the declaration rather than at the
# first call, which is where the constant would actually be built.
# CHECK: NotImplementedError: an air.extern scalar argument is not supported for air.api.ui8
@expect(NotImplementedError, "unsigned_extern_scalar")
def _():
    air.extern("k", link_with="k.o", scalars=[ui8])


# CHECK-LABEL: TEST: select_on_a_bool
# The trap this message exists for. `==` and `!=` are NOT overloaded on buffer
# expressions -- overloading __eq__ would make every expression unhashable and
# would change what `expr == expr` means for ordinary Python -- so `a[:] ==
# b[:]` is an identity comparison that evaluates to a plain bool long before
# select is called. Refusing it by name is the only way that difference does not
# pass silently as `select(False, ...)`.
# CHECK: TypeError: air.api.ops.select got a plain bool as its condition.
@expect(TypeError, "select_on_a_bool")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64], f32, scope=h.private())
        b = air.alloc([64], f32, scope=h.private())
        ops.select(a[:] == b[:], a[:], b[:])

    _trace(body)


# CHECK-LABEL: TEST: select_on_a_value
# A value expression is not a predicate. Its result type is the element type,
# not i1, so arith.select would fail to verify well downstream of the mistake.
# CHECK: TypeError: air.api.ops.select expects a comparison as its condition
@expect(TypeError, "select_on_a_value")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64], f32, scope=h.private())
        b = air.alloc([64], f32, scope=h.private())
        ops.select(a[:] + b[:], a[:], b[:])

    _trace(body)


# CHECK-LABEL: TEST: comparison_of_two_scalars
# A comparison between two scalars has no shape for the emitter to give it, and
# it is a Python-level constant the caller should have folded themselves.
# CHECK: ValueError: air.api.ops.equal needs at least one buffer operand
@expect(ValueError, "comparison_of_two_scalars")
def _():
    ops.equal(1.0, 2.0)


# CHECK-LABEL: TEST: unsigned_select
# Comparisons and select go through arith like every other operator, so an
# unsigned buffer is refused for the same reason it is refused for `+`:
# arith.cmpi takes signless operands.
# CHECK: NotImplementedError: an elementwise operator or broadcast scalar (a plain copy, dst[:] = src[:], is) is not supported for air.api.ui8
@expect(NotImplementedError, "unsigned_select")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64], ui8, scope=h.private())
        a[:] = ops.select(a[:] >= 1, a[:], 1)

    _trace(body)


# CHECK-LABEL: TEST: bitwise_on_a_float_buffer
# The bitwise operators are the DSL's first integer-only ones. MLIR has
# arith.andi and no floating-point counterpart, so this cannot be coerced --
# and the message says which operator and why rather than the generic
# "not supported for dtype float", because & on a float buffer is usually a
# dtype mistake upstream rather than a wrong choice of operator.
# CHECK: NotImplementedError: the bitwise operator 'and' is integer-only
@expect(NotImplementedError, "bitwise_on_a_float_buffer")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64], f32, scope=h.private())
        b = air.alloc([64], f32, scope=h.private())
        a[:] = a[:] & b[:]

    _trace(body)


# CHECK-LABEL: TEST: bitwise_on_an_unsigned_buffer
# Unsigned is refused earlier and for a different reason: arith takes signless
# operands, so no operator at all reaches the emitter for a ui8 buffer.
# CHECK: NotImplementedError: an elementwise operator or broadcast scalar (a plain copy, dst[:] = src[:], is) is not supported for air.api.ui8
@expect(NotImplementedError, "bitwise_on_an_unsigned_buffer")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64], ui8, scope=h.private())
        b = air.alloc([64], ui8, scope=h.private())
        a[:] = a[:] ^ b[:]

    _trace(body)


# CHECK-LABEL: TEST: cast_narrowing_int
# Refused on evidence, not on principle: measured on npu1, the vectorised
# arith.trunci saturates while the scalar one wraps, and the emitter chooses
# between them from the tile size. Accepting this would make the same source
# compute two different things depending on a tile shape.
# CHECK: NotImplementedError: air.api.ops.cast will not narrow air.api.i32 to air.api.i16
@expect(NotImplementedError, "cast_narrowing_int")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([16, 16], i32, scope=h.private())
        c = air.alloc([16, 16], i16, scope=h.private())
        c[:] = ops.cast(a[:], i16)

    _trace(body)


# CHECK-LABEL: TEST: cast_narrowing_int_clamped_to_wider_bounds
# The clamped exception is about the bounds, not the presence of a clamp. These
# bounds do not fit i8, so values the two trunci paths disagree about can still
# reach the cast and the refusal stands.
# CHECK: NotImplementedError: air.api.ops.cast will not narrow air.api.i32 to air.api.i8
@expect(NotImplementedError, "cast_narrowing_int_clamped_to_wider_bounds")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([16, 16], i32, scope=h.private())
        c = air.alloc([16, 16], i8, scope=h.private())
        c[:] = ops.cast(ops.minimum(ops.maximum(a[:], -200), 127), i8)

    _trace(body)


# CHECK-LABEL: TEST: cast_narrowing_int_clamped_on_one_side
# A single-sided clamp leaves the other side unbounded, so it proves nothing.
# CHECK: NotImplementedError: air.api.ops.cast will not narrow air.api.i32 to air.api.i8
@expect(NotImplementedError, "cast_narrowing_int_clamped_on_one_side")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([16, 16], i32, scope=h.private())
        c = air.alloc([16, 16], i8, scope=h.private())
        c[:] = ops.cast(ops.minimum(a[:], 127), i8)

    _trace(body)


# CHECK-LABEL: TEST: shift_on_a_float_buffer
# Integer-only for the same reason as the bitwise operators, and named the same
# way. The message also says what to write instead, because scaling a float by a
# power of two is a thing people reach for << to express.
# CHECK: NotImplementedError: the shift operator '>>' is integer-only
@expect(NotImplementedError, "shift_on_a_float_buffer")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64], f32, scope=h.private())
        a[:] = a[:] >> 2

    _trace(body)


# CHECK-LABEL: TEST: shift_by_a_negative_count
# Python raises ValueError here. MLIR would make it poison instead, which is
# silent and survives into the backend as a wrong answer rather than a
# diagnostic, so this is refused where it is written.
# CHECK: ValueError: negative shift count -1 in '>>'
@expect(ValueError, "shift_by_a_negative_count")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64], i32, scope=h.private())
        a[:] = a[:] >> -1

    _trace(body)


# CHECK-LABEL: TEST: shift_by_the_operand_width
# The boundary, not an arbitrary large number: 32 is the first count an i32
# cannot take. Python would give 0 or -1; LLVM says poison.
# CHECK: ValueError: shift count 32 is not less than the width of air.api.i32
@expect(ValueError, "shift_by_the_operand_width")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64], i32, scope=h.private())
        a[:] = a[:] >> 32

    _trace(body)


# CHECK-LABEL: TEST: shift_by_a_folded_index_expression
# "Constant" has to mean "folds to a constant", not "was typed as an int
# literal". Index arithmetic over herd coordinates is ordinary in a kernel
# body, and `tx - tx + 32` reaches the emitter as a literal arith.constant 32 --
# by then indistinguishable from having been written as 32, and just as much
# poison. Reading it back needs as_const(): IndexExpr implements neither
# equality nor int conversion against a Python int, so an isinstance test alone
# classifies every index expression as runtime and lets the folded ones past.
# CHECK: ValueError: shift count 32 is not less than the width of air.api.i32
@expect(ValueError, "shift_by_a_folded_index_expression")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64], i32, scope=h.private())
        a[:] = a[:] >> (tx - tx + 32)

    _trace(body)


# CHECK-LABEL: TEST: shift_width_is_per_dtype
# i8 runs out eight times sooner, so the check reads the operand's own width
# rather than assuming 32.
# CHECK: ValueError: shift count 8 is not less than the width of air.api.i8
@expect(ValueError, "shift_width_is_per_dtype")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64], i8, scope=h.private())
        a[:] = a[:] << 8

    _trace(body)


# CHECK-LABEL: TEST: cast_same_width_float
# bf16 and f16 are both two bytes, so the conversion is neither a widening nor
# a narrowing and arith has no op for it.
# CHECK: NotImplementedError: air.api.ops.cast cannot convert air.api.bf16 to air.api.f16 directly
@expect(NotImplementedError, "cast_same_width_float")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([16, 16], bf16, scope=h.private())
        c = air.alloc([16, 16], f16, scope=h.private())
        c[:] = ops.cast(a[:], f16)

    _trace(body)


# CHECK-LABEL: TEST: cast_unsigned
# A conversion is an arith op like any other, so the signless rule that governs
# the operators governs it too.
# CHECK: NotImplementedError: air.api.ops.cast is not supported for air.api.ui8
@expect(NotImplementedError, "cast_unsigned")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([16, 16], ui8, scope=h.private())
        c = air.alloc([16, 16], i32, scope=h.private())
        c[:] = ops.cast(a[:], i32)

    _trace(body)


# CHECK-LABEL: TEST: cast_of_a_bare_scalar
# There is no buffer under it, so there is no source type to convert from --
# and the constant would have been built in the target type anyway.
# CHECK: TypeError: air.api.ops.cast needs an expression containing at least one buffer
@expect(TypeError, "cast_of_a_bare_scalar")
def _():
    ops.cast(1.0, i32)


# CHECK-LABEL: TEST: cast_to_a_non_dtype
# The second argument is an element type, not a string or a numpy dtype. Named
# here rather than left to fail later inside the emitter.
# CHECK: TypeError: air.api.ops.cast needs an air.api element type as its second argument
@expect(TypeError, "cast_to_a_non_dtype")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([16, 16], f32, scope=h.private())
        ops.cast(a[:], "i32")

    _trace(body)


# CHECK-LABEL: TEST: dtype_mismatch_without_a_cast_still_raises
# ops.cast is the *only* way to change element type in an expression; a bare
# mismatch is still the error it always was, rather than an implicit
# conversion. The message names the region's type, which without a cast in the
# tree is the destination's.
# CHECK: ValueError: dtype mismatch in elementwise assignment: destination is air.api.i32 but operand is air.api.f32
@expect(ValueError, "dtype_mismatch_without_a_cast_still_raises")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([16, 16], f32, scope=h.private())
        c = air.alloc([16, 16], i32, scope=h.private())
        c[:] = a[:]

    _trace(body)


# CHECK-LABEL: TEST: dtype_mismatch_below_a_cast
# The leaves under a cast are checked against the cast's *source* type, not
# against the destination of the assignment. Here the cast converts from f32,
# so an i32 buffer sitting beside it in the same region is the mismatch.
# CHECK: ValueError: dtype mismatch in elementwise assignment: the cast converts from air.api.f32 but operand is air.api.i32
@expect(ValueError, "dtype_mismatch_below_a_cast")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([16, 16], f32, scope=h.private())
        b = air.alloc([16, 16], i32, scope=h.private())
        c = air.alloc([16, 16], i32, scope=h.private())
        c[:] = ops.cast(a[:] + b[:], i32)

    _trace(body)


# CHECK-LABEL: TEST: cast_of_a_comparison
# A comparison evaluates to i1, not to an element type, so there is nothing to
# convert from -- and a conversion op applied to a vector<Wxi1> would fail the
# verifier well downstream of the mistake. This is the one interaction between
# ops.select and ops.cast that needs naming.
# CHECK: TypeError: air.api.ops.cast got a comparison, which is a predicate rather than a value
@expect(TypeError, "cast_of_a_comparison")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64], f32, scope=h.private())
        b = air.alloc([64], f32, scope=h.private())
        c = air.alloc([64], i32, scope=h.private())
        c[:] = ops.cast(a[:] > b[:], i32)

    _trace(body)


# CHECK-LABEL: TEST: cast_of_a_predicate_only_expression
# ops.select allows every buffer to sit in the *predicate*, choosing between
# two scalars -- so the expression has buffers but no element type of its own,
# and takes one from whatever surrounds it. That is a different situation from
# a bare scalar and gets a different message, because the fix is different:
# cast the operands rather than the select.
# CHECK: TypeError: air.api.ops.cast cannot convert this expression: its buffers appear only in a comparison
@expect(TypeError, "cast_of_a_predicate_only_expression")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64], f32, scope=h.private())
        c = air.alloc([64], i32, scope=h.private())
        c[:] = ops.cast(ops.select(a[:] > 0.0, 1, 2), i32)

    _trace(body)


# CHECK-LABEL: TEST: fma_on_an_integer_buffer
# fma exists to avoid the intermediate rounding, and integer multiply-add has
# none to avoid -- so this is not a gap to be filled later, and the message
# says so and names the spelling that does work rather than implying that
# an arith.fmai might arrive one day.
# CHECK: NotImplementedError: air.api.ops.fma is float-only
@expect(NotImplementedError, "fma_on_an_integer_buffer")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64], i32, scope=h.private())
        b = air.alloc([64], i32, scope=h.private())
        a[:] = ops.fma(2, a[:], b[:])

    _trace(body)


# CHECK-LABEL: TEST: exp_on_an_integer_buffer
# exp and rsqrt are float-only: there is no integer math.exp, and an integer
# buffer reaching one is a mistake rather than something to coerce.
# CHECK: NotImplementedError: elementwise operator 'exp' is not supported for integer buffers
@expect(NotImplementedError, "exp_on_an_integer_buffer")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64], i32, scope=h.private())
        c = air.alloc([64], i32, scope=h.private())
        c[:] = ops.exp(a[:])

    _trace(body)


# CHECK-LABEL: TEST: fma_has_no_scalar_form
# The emitter's usual fallback -- drop to a scalar loop when the innermost
# dimension is not a multiple of the vector width -- is the *unsafe* direction
# here, exactly as it is for math.tanh: AIE2 has no scalar fma instruction, so
# math.fma reaches the backend and fails to legalize. Unlike tanh, which is
# emitted anyway and fails hours later with an LLVM virtual register in the
# message, this is refused at the point where the tile shape can be named.
# CHECK: NotImplementedError: air.api.ops.fma has no scalar form on AIE2
@expect(NotImplementedError, "fma_has_no_scalar_form")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([60], bf16, scope=h.private(), vector=16)
        b = air.alloc([60], bf16, scope=h.private(), vector=16)
        a[:] = ops.fma(2.0, a[:], b[:])

    _trace(body)


# CHECK-LABEL: TEST: fma_of_a_comparison
# A comparison is i1, not a value, so it can be selected between but not
# multiplied. Caught at the call site: letting it through would build an
# arith op over a predicate and fail in the MLIR verifier, which names an SSA
# value rather than the argument at fault.
# CHECK: TypeError: air.api.ops.fma got a comparison as its second argument
@expect(TypeError, "fma_of_a_comparison")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64], bf16, scope=h.private())
        b = air.alloc([64], bf16, scope=h.private())
        a[:] = ops.fma(2.0, a[:] > b[:], b[:])

    _trace(body)


# CHECK-LABEL: TEST: reduce_of_a_scalar
# Nothing supplies an axis to reduce over. Caught by the operand type check
# rather than the leaf check -- a bare float never becomes a BufferExpr at all.
# CHECK: TypeError: air.api.ops.reduce_add expects a buffer slice, got float
@expect(TypeError, "reduce_of_a_scalar")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64], bf16, scope=h.private())
        a[:] = ops.reduce_add(2.0)

    _trace(body)


# CHECK-LABEL: TEST: reduce_of_a_comparison
# A comparison is i1, a predicate rather than a value, so there is nothing
# meaningful to sum or maximise.
# CHECK: TypeError: air.api.ops.reduce_add got a comparison
@expect(TypeError, "reduce_of_a_comparison")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64, 16], bf16, scope=h.private())
        b = air.alloc([64], bf16, scope=h.private())
        b[:] = ops.reduce_add(a[:] > 0.0)

    _trace(body)


# CHECK-LABEL: TEST: fma_of_only_scalars
# Nothing supplies a shape, so there is no loop to build.
# CHECK: ValueError: air.api.ops.fma needs at least one buffer operand
@expect(ValueError, "fma_of_only_scalars")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64], bf16, scope=h.private())
        a[:] = ops.fma(2.0, 3.0, 4.0)

    _trace(body)


# CHECK-LABEL: TEST: reduce_of_a_reduce
# The inner reduction has already collapsed the innermost axis to 1, so the
# outer one would reduce a single element. Reducing a second axis is a
# different feature, and the message says so rather than silently no-opping.
# CHECK: NotImplementedError: air.api.ops.reduce_add cannot reduce a reduction
@expect(NotImplementedError, "reduce_of_a_reduce")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64, 16], bf16, scope=h.private())
        b = air.alloc([64], bf16, scope=h.private())
        b[:] = ops.reduce_add(ops.reduce_add(a[:]))

    _trace(body)


# CHECK-LABEL: TEST: fma_of_a_tensor_slice
# An L3 tensor slice names a DMA region, not a value. The message names the
# argument position, which matters more for a ternary op than a binary one.
# CHECK: TypeError: air.api.ops.fma expects a buffer slice or a numeric scalar as its third argument
@expect(TypeError, "fma_of_a_tensor_slice")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64], bf16, scope=h.private())
        a[:] = ops.fma(2.0, a[:], A[0:64, 0])

    _trace(body)


# CHECK-LABEL: TEST: reduce_into_the_wrong_shape
# The destination must be the operand with its innermost axis collapsed --
# either kept as 1 or dropped. Anything else is named with both alternatives,
# because which one the caller wanted is not inferable.
# CHECK: ValueError: shape mismatch in a reduction
@expect(ValueError, "reduce_into_the_wrong_shape")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64, 16], bf16, scope=h.private())
        b = air.alloc([64, 16], bf16, scope=h.private())
        b[:] = ops.reduce_add(a[:])

    _trace(body)


# CHECK-LABEL: TEST: reduce_on_an_unsigned_buffer
# Same rule as every other arith-building path: vector.reduction's combining
# kinds are signed, and a ui8 operand does not verify.
# CHECK: NotImplementedError: air.api.ops.reduce_add / reduce_max is not supported for air.api.ui8
@expect(NotImplementedError, "reduce_on_an_unsigned_buffer")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64, 16], ui8, scope=h.private())
        b = air.alloc([64], ui8, scope=h.private())
        b[:] = ops.reduce_add(a[:])

    _trace(body)


# CHECK-LABEL: TEST: rsqrt_of_a_scalar
# A unary op needs something to be shaped by. A bare Python float has no shape,
# and the emitter cannot invent one.
# CHECK: TypeError: air.api.ops.rsqrt expects a buffer slice, got float
@expect(TypeError, "rsqrt_of_a_scalar")
def _():
    ops.rsqrt(2.0)


# CHECK-LABEL: TEST: herd_link_with_is_not_a_string
# link_with= names a compiled object file. Anything else would reach the IR as a
# StringAttr conversion failure with no mention of which herd.
# CHECK: TypeError: air.herd(link_with=...) takes the name of a compiled object file
@expect(TypeError, "herd_link_with_is_not_a_string")
def _():
    air.herd(range(0, 128, 64), link_with=17)


# CHECK-LABEL: TEST: herd_link_with_is_empty
# An empty string is the shape of a mistake that would otherwise emit
# link_with = "" and fail much later, in the linker.
# CHECK: TypeError: air.herd(link_with=...) takes the name of a compiled object file
@expect(TypeError, "herd_link_with_is_empty")
def _():
    air.herd(range(0, 128, 64), link_with="")


# CHECK-LABEL: TEST: segment_body_never_registered
# `with air.segment(...)` is pure bookkeeping -- every op comes from the body
# decorator -- so omitting it emits nothing at all for that scope and traces the
# herd straight into the launch. That is the worst available failure: the IR is
# structurally different, still builds, and on a small grid still runs and still
# passes, so neither a hardware test nor an op-count diff catches it. It did
# reach review once, in the data_transfer_transpose conversion.
# CHECK: RuntimeError: air.segment was opened but its body was never registered
@expect(RuntimeError, "segment_body_never_registered")
def _():
    A = air.tensor([64], bf16)
    C = air.tensor([64], bf16)

    with air.launch(name="k") as launch:

        @launch.body
        def _():
            with air.segment(name="s") as seg:
                with air.herd(range(0, 64, 64), shape=(1,)) as h:

                    @h.body
                    def _(tx):
                        t = air.alloc([64], bf16, scope=h.private())
                        ops.load(t, A[:])
                        ops.store(t, C[:])

    launch.mlir()


# CHECK-LABEL: TEST: herd_body_never_registered
# The same hole, and it is a real one rather than a theoretical twin: a lone
# body-less herd happens to trip "kernel writes no output", but that check is
# satisfied by any *other* herd that stores. Two herds with the second one's
# body forgotten built cleanly and dropped it silently.
# CHECK: RuntimeError: air.herd was opened but its body was never registered
@expect(RuntimeError, "herd_body_never_registered")
def _():
    A = air.tensor([64], bf16)
    C = air.tensor([64], bf16)

    with air.launch(name="k") as launch:

        @launch.body
        def _():
            with air.herd(range(0, 64, 64), name="h1", shape=(1,)) as h:

                @h.body
                def _(tx):
                    t = air.alloc([64], bf16, scope=h.private())
                    ops.load(t, A[:])
                    ops.store(t, C[:])

            with air.herd(range(0, 64, 64), name="h2", shape=(1,)):
                pass

    launch.mlir()


# CHECK-LABEL: TEST: a_failing_body_is_not_masked
# The guard must stay quiet while another exception is propagating. A body that
# raised is far more interesting than a body that is absent, and reporting the
# absence here would bury the real error -- which, at that point, is the only
# reason the body never registered.
# CHECK: ValueError: the body's own problem
@expect(ValueError, "a_failing_body_is_not_masked")
def _():
    air.tensor([64], bf16)

    with air.launch(name="k") as launch:

        @launch.body
        def _():
            with air.segment(name="s"):
                raise ValueError("the body's own problem")

    launch.mlir()


# CHECK-LABEL: TEST: broadcast_needs_an_extent_of_one
# Broadcasting stretches an axis of extent 1 and nothing else. A [64, 32]
# operand against a [64, 64] destination is not "half of each row", it is a
# mistake, and numpy refuses it for the same reason.
# CHECK: ValueError: shape mismatch in elementwise assignment
# CHECK-SAME: does not broadcast to it
@expect(ValueError, "broadcast_needs_an_extent_of_one")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64, 64], bf16, scope=h.private())
        b = air.alloc([64, 32], bf16, scope=h.private())
        a[:] = a[:] + b[:]

    _trace(body)


# CHECK-LABEL: TEST: an_operand_cannot_be_wider_than_the_destination
# The rule is one-sided: operands broadcast *to* the destination, which already
# exists. Stretching the other way would mean writing 16 values into 1, and
# numpy's `out=` refuses this too rather than picking one of them.
# CHECK: ValueError: shape mismatch in elementwise assignment
# CHECK-SAME: does not broadcast to it
@expect(ValueError, "an_operand_cannot_be_wider_than_the_destination")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64, 1], bf16, scope=h.private())
        b = air.alloc([64, 16], bf16, scope=h.private())
        a[:] = b[:]

    _trace(body)


# CHECK-LABEL: TEST: an_operand_cannot_have_more_axes_than_the_destination
# Right-aligning a rank-2 operand against a rank-1 destination leaves an axis
# with nowhere to go. Only the *operand* may be short of axes.
# CHECK: ValueError: shape mismatch in elementwise assignment
# CHECK-SAME: does not broadcast to it
@expect(ValueError, "an_operand_cannot_have_more_axes_than_the_destination")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64], bf16, scope=h.private())
        b = air.alloc([4, 64], bf16, scope=h.private())
        a[:] = b[:]

    _trace(body)


# CHECK-LABEL: TEST: a_reduction_does_not_broadcast_its_operands
# Everywhere else an extent of 1 is stretched, but the innermost extent of a
# reduction is the thing being collapsed: stretching it would decide how many
# terms the sum has, which is the reduction's meaning rather than a fit.
# CHECK: ValueError: shape mismatch inside a reduction
# CHECK-SAME: does not broadcast its operands
@expect(ValueError, "a_reduction_does_not_broadcast_its_operands")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64, 16], bf16, scope=h.private())
        s = air.alloc([64, 1], bf16, scope=h.private())
        out = air.alloc([64], bf16, scope=h.private())
        out[:] = ops.reduce_add(a[:] * s[:])

    _trace(body)


# CHECK-LABEL: TEST: a_condition_has_no_truth_value
# `if tx == 0:` is the trap ops.branch exists to close. A herd body is traced
# once for the whole herd, so Python has to pick one branch for every core --
# and would, silently, if Condition let itself be coerced.
# CHECK: TypeError: the truth of (t0 == 0) is not known at trace time
# CHECK-SAME: with ops.branch(t0 == 0)
@expect(TypeError, "a_condition_has_no_truth_value")
def _():
    def body(h, tx, ty, A, B, C):
        if tx == 0:
            pass

    _trace(body)


# CHECK-LABEL: TEST: conditions_have_no_and
# `and` cannot be reached at all -- it coerces through __bool__ -- so the
# bitwise operator is the one worth naming a replacement for.
# CHECK: NotImplementedError: air.api has no `&` on a condition
# CHECK-SAME: Conjunction is nesting
@expect(NotImplementedError, "conditions_have_no_and")
def _():
    def body(h, tx, ty, A, B, C):
        with ops.branch((tx == 0) & (ty == 0)):
            pass

    _trace(body)


# CHECK-LABEL: TEST: when_takes_a_comparison_not_a_bool
# A Python bool decides at trace time whether the region exists at all, which
# is what a plain `if` already does; accepting one here would make the two
# spellings look interchangeable when they are not.
# CHECK: TypeError: ops.branch takes a comparison between index expressions
@expect(TypeError, "when_takes_a_comparison_not_a_bool")
def _():
    def body(h, tx, ty, A, B, C):
        with ops.branch(True):
            pass

    _trace(body)


# CHECK-LABEL: TEST: elsewhere_before_the_where_body
# otherwise() names the else of a region that has been opened. Reaching for it
# first is a sign the two `with` blocks were written the wrong way round.
# CHECK: RuntimeError: otherwise() on an ops.branch whose region was never opened
@expect(RuntimeError, "elsewhere_before_the_where_body")
def _():
    def body(h, tx, ty, A, B, C):
        branch = ops.branch(tx == 0)
        with branch.otherwise():
            pass

    _trace(body)


# CHECK-LABEL: TEST: two_elsewhere_regions
# An scf.if has one else. A second would silently discard the first.
# CHECK: RuntimeError: ops.branch(t0 == 0) already has an otherwise() region
@expect(RuntimeError, "two_elsewhere_regions")
def _():
    def body(h, tx, ty, A, B, C):
        with ops.branch(tx == 0) as branch:
            pass
        with branch.otherwise():
            pass
        with branch.otherwise():
            pass

    _trace(body)


# CHECK-LABEL: TEST: alloc_inside_a_branch
# The herd frees its buffers after the body, outside the region, so the dealloc
# would not be dominated by the alloc. It is also the wrong instinct: L1 is
# charged per core whether or not that core's branch runs.
# CHECK: NotImplementedError: air.alloc inside an air.sequential or ops.branch body
@expect(NotImplementedError, "alloc_inside_a_branch")
def _():
    def body(h, tx, ty, A, B, C):
        with ops.branch(tx == 0):
            air.alloc([64], bf16, scope=h.private())

    _trace(body)


# CHECK-LABEL: TEST: select_with_a_branch_condition
# The two conditionals cannot be told apart by name, so the one place a caller
# can be told which they wanted is the moment they have picked. select decides
# per element; a tile coordinate is the same for every element the core touches.
# CHECK: TypeError: air.api.ops.select got (t0 == 0), a comparison between *index* expressions
# CHECK-SAME: that is ops.branch's condition
# CHECK-SAME: with ops.branch(t0 == 0)
@expect(TypeError, "select_with_a_branch_condition")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64], bf16, scope=h.private())
        b = air.alloc([64], bf16, scope=h.private())
        out = air.alloc([64], bf16, scope=h.private())
        out[:] = ops.select(tx == 0, a[:], b[:])

    _trace(body)


# CHECK-LABEL: TEST: branch_with_a_select_condition
# And the reverse. A branch is taken once per core, so it cannot depend on what
# is in the buffer -- different elements would need different branches.
# CHECK: TypeError: ops.branch takes a comparison between index expressions
# CHECK-SAME: elementwise comparison on buffer *data*
# CHECK-SAME: ops.select(cond, a[:], b[:])
@expect(TypeError, "branch_with_a_select_condition")
def _():
    def body(h, tx, ty, A, B, C):
        a = air.alloc([64], bf16, scope=h.private())
        b = air.alloc([64], bf16, scope=h.private())
        with ops.branch(a[:] >= b[:]):
            pass

    _trace(body)


# CHECK-LABEL: TEST: compare_a_coordinate_against_a_bool
# `tx == True` would otherwise be answered by Python: coerce_index rejects bool,
# the comparison returns NotImplemented, and the fallback makes it a silent
# False -- a trace-time branch decision the program never asked for, which is
# the exact thing Condition.__bool__ exists to stop one line later.
# CHECK: TypeError: cannot compare a tile coordinate against the bool True
# CHECK-SAME: Compare against an integer instead
@expect(TypeError, "compare_a_coordinate_against_a_bool")
def _():
    def body(h, tx, ty, A, B, C):
        with air.ops.branch(tx == True):
            pass

    _trace(body)


# CHECK-LABEL: TEST: break_out_of_a_branch
# ops.branch shares the region bookkeeping with air.sequential, so an abandoned
# branch used to be reported as "left an air.sequential loop early" -- which
# sends the reader to the wrong line, and offers loop-bound advice for something
# that is not a loop.
# CHECK: RuntimeError: a body left an ops.branch region early
# CHECK-SAME: Let the `with` block run to its end
@expect(RuntimeError, "break_out_of_a_branch")
def _():
    def body(h, tx, ty, A, B, C):
        buf = air.alloc([64], bf16, scope=h.private())
        try:
            with air.ops.branch(tx == 0):
                raise ValueError("swallowed by the body")
        except ValueError:
            pass

    _trace(body)
