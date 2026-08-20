# ./python/test/api/activations.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""air.api lowers tanh and the activations composed from it.

`tanh` is the DSL's first *unary* node -- every other expression node is binary
-- so these also pin that a unary op composes with the binary operators in one
tree and lowers as a single loop.

The three compositions are written to emit exactly what the hand-written kernels
emitted, op for op. Each is sigmoid/silu/gelu via tanh rather than exp, which
avoids a division that bf16 cannot vectorise.
"""

from air import api as air
from air.api.types import bf16


def build(N, tile, body, herd_shape=None, vector=None):
    x = air.tensor([N], bf16)
    out = air.tensor([N], bf16)

    with air.launch(name="act") as launch:

        @launch.body
        def _():
            with air.herd(range(0, N, tile), shape=herd_shape) as h:

                @h.body
                def _(tx):
                    (tn,) = h.tile_sizes
                    col = tx * tn
                    x_buf = air.alloc([tn], bf16, scope=h.private(), vector=vector)
                    o_buf = air.alloc([tn], bf16, scope=h.private(), vector=vector)
                    air.ops.load(x_buf, x[col : col + tn])
                    o_buf[:] = body(x_buf)
                    air.ops.store(o_buf, out[col : col + tn])

    return launch


def run(f):
    print("\nTEST:", f.__name__)
    f()
    return f


# CHECK-LABEL: TEST: tanh_unary
# A bare unary node: read, tanh, write. No binary operator involved.
# CHECK: vector.transfer_read {{.*}} vector<16xbf16>
# CHECK: math.tanh {{.*}} : vector<16xbf16>
# CHECK: vector.transfer_write {{.*}} vector<16xbf16>
@run
def tanh_unary():
    print(build(65536, 1024, lambda b: air.ops.tanh(b[:]), herd_shape=(4,)).mlir())


# CHECK-LABEL: TEST: sigmoid_is_tanh_based
# 0.5 * (tanh(x/2) + 1): one mul in, tanh, add one, one mul out -- the same four
# ops the hand-written kernel emitted, and no division.
# CHECK-NOT: arith.divf
# CHECK: arith.mulf {{.*}} : vector<16xbf16>
# CHECK: math.tanh {{.*}} : vector<16xbf16>
# CHECK: arith.addf {{.*}} : vector<16xbf16>
# CHECK: arith.mulf {{.*}} : vector<16xbf16>
@run
def sigmoid_is_tanh_based():
    print(build(65536, 1024, lambda b: air.ops.sigmoid(b[:]), herd_shape=(4,)).mlir())


# CHECK-LABEL: TEST: silu_multiplies_by_x
# x * sigmoid(x): the sigmoid chain, then one more mul against the original x.
# CHECK-NOT: arith.divf
# CHECK: math.tanh {{.*}} : vector<16xbf16>
# CHECK: arith.addf {{.*}} : vector<16xbf16>
# CHECK: arith.mulf {{.*}} : vector<16xbf16>
# CHECK: arith.mulf {{.*}} : vector<16xbf16>
@run
def silu_multiplies_by_x():
    print(build(65536, 1024, lambda b: air.ops.silu(b[:]), herd_shape=(4,)).mlir())


# CHECK-LABEL: TEST: gelu_cubic_then_tanh
# 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))): the cubic is three muls
# before the tanh, and the constants are the Hendrycks & Gimpel ones.
# CHECK-DAG: arith.constant 7.968750e-01 : bf16
# CHECK-DAG: arith.constant 4.467770e-02 : bf16
# CHECK: math.tanh {{.*}} : vector<16xbf16>
# CHECK: arith.addf {{.*}} : vector<16xbf16>
@run
def gelu_cubic_then_tanh():
    print(build(65536, 1024, lambda b: air.ops.gelu(b[:]), herd_shape=(4,)).mlir())


# CHECK-LABEL: TEST: scalar_fallback_is_emitted_but_will_not_compile
# The emitter's usual safety net is a hazard for tanh specifically. A tile
# narrower than the vector width still lowers the whole tree scalar, unary node
# included -- and scalar bf16 tanh does not legalize on either generation
# ("unable to legalize instruction: s16 G_FTANH"), so this IR is valid MLIR that
# the backend cannot build.
#
# The IR is pinned here so the hazard is visible rather than folklore. The
# examples do not rely on the emitter noticing: sigmoid/silu/gelu reject
# --vector-size 0 and reject a tile that is not a multiple of the vector width,
# which are the two routes into this path.
# CHECK-NOT: vector.transfer_read
# CHECK: memref.load
# CHECK: math.tanh {{.*}} : bf16
# CHECK: memref.store
@run
def scalar_fallback_is_emitted_but_will_not_compile():
    print(build(48, 12, lambda b: air.ops.tanh(b[:]), herd_shape=(4,)).mlir())
