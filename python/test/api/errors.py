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
# CHECK: NotImplementedError: partial write of an L1 buffer
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


# CHECK-LABEL: TEST: alloc_outside_herd
# CHECK: RuntimeError: this operation must be used inside a herd body
@expect(RuntimeError, "alloc_outside_herd")
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


# CHECK-LABEL: TEST: unimplemented_op
# CHECK: NotImplementedError: air.api.ops.dot is not implemented yet
@expect(NotImplementedError, "unimplemented_op")
def _():
    air.ops.dot(None, None)


# CHECK-LABEL: TEST: unimplemented_segment
# CHECK: NotImplementedError: air.api.segment is not implemented yet
@expect(NotImplementedError, "unimplemented_segment")
def _():
    air.segment(range(2))


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
