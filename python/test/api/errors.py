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
from air.api.types import bf16, f16, f32, i16, i32, ui8


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


# CHECK-LABEL: TEST: channel_pack_not_a_packed_shape
# pack= takes a shape from air.micro_tile(...).a/.b, not a plain list: the
# micro-tile is what the walk is derived from, and a bare list carries none.
# CHECK: TypeError: air.channel.put(pack=...) takes a packed shape from air.micro_tile
@expect(TypeError, "channel_pack_not_a_packed_shape")
def _():
    ch = air.channel("P")

    def body(h, tx, ty, A, B, C):
        buf = air.alloc([16, 8], bf16, scope=h.private())
        ch.put(buf[0:16, 0:8], pack=[2, 2, 8, 8])

    _trace(body)


# CHECK-LABEL: TEST: channel_pack_needs_a_region
# The pack reorders a *slice* of a flat staging buffer, so there has to be one
# to reorder; a whole buffer carries no pattern to rewrite.
# CHECK: TypeError: air.channel.put(pack=...) needs a *region* to walk
@expect(TypeError, "channel_pack_needs_a_region")
def _():
    ch = air.channel("Q")
    mm = air.micro_tile(1, 16, 8)

    def body(h, tx, ty, A, B, C):
        buf = air.alloc([16, 8], bf16, scope=h.private())
        ch.put(buf, pack=mm.b(16, 8, lead=()))

    _trace(body)


# CHECK-LABEL: TEST: channel_pack_c_operand
# A C accumulator unpacks the other way round -- the pattern belongs on the
# packed buffer, not on the channel -- so asking a channel to pack one raises
# instead of emitting a walk that would drain it in the wrong order.
# CHECK: NotImplementedError: air.channel.put(pack=...) packs an A or B operand
@expect(NotImplementedError, "channel_pack_c_operand")
def _():
    ch = air.channel("R")
    mm = air.micro_tile(1, 16, 8)

    def body(h, tx, ty, A, B, C):
        buf = air.alloc([16, 8], bf16, scope=h.private())
        ch.put(buf[0:16, 0:8], pack=mm.c(16, 8, lead=()))

    _trace(body)


# CHECK-LABEL: TEST: channel_pack_wrong_rank
# The region has to end in the operand's two logical axes; a rank-1 slice has
# only one, so there is nothing to split into micro-tiles.
# CHECK: ValueError: air.channel.put(pack=...) needs a region of rank 2 for a B operand
@expect(ValueError, "channel_pack_wrong_rank")
def _():
    ch = air.channel("S")
    mm = air.micro_tile(1, 16, 8)

    def body(h, tx, ty, A, B, C):
        buf = air.alloc([128], bf16, scope=h.private())
        ch.put(buf[0:128], pack=mm.b(16, 8, lead=()))

    _trace(body)


# CHECK-LABEL: TEST: channel_pack_indivisible
# The region's extents must be whole micro-tiles: 20 is not a multiple of the
# k=16 the buffer would be packed with.
# CHECK: ValueError: operand B's K extent 20 is not a multiple of the micro-tile k=16
@expect(ValueError, "channel_pack_indivisible")
def _():
    ch = air.channel("T")
    mm = air.micro_tile(1, 16, 8)

    def body(h, tx, ty, A, B, C):
        buf = air.alloc([20, 8], bf16, scope=h.private())
        ch.put(buf[0:20, 0:8], pack=mm.b(20, 8, lead=()))

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
    air.extern("k", object="k.o", scalars=[ui8])


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
