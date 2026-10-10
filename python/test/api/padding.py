# ./python/test/api/padding.py -*- Python -*-

# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s

"""pad_before/pad_after on ops.load and ops.store.

The counts are per axis of the source region and become the pad attributes of
air.dma_memcpy_nd. The destination is checked against the padded extent.
"""

from air import api as air
from air.api import ops
from air.api.types import bf16

ROWS, COLS, PADDED = 16, 120, 128


def run(f):
    print("\nTEST:", f.__name__)
    try:
        f()
    except (ValueError, TypeError) as e:
        print(f"{type(e).__name__}: {e}")
    return f


def build(body, l1_shape=(ROWS, PADDED)):
    A = air.tensor([ROWS, COLS], bf16)
    C = air.tensor([ROWS, PADDED], bf16)
    with air.launch(name="pad") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    l2 = air.alloc([ROWS, COLS], bf16, scope=seg.private())
                    ops.load(l2, A)
                    with air.herd([range(1)], name="herd_0", shape=(1,)) as h:

                        @h.body
                        def _(tx):
                            l1 = air.alloc(list(l1_shape), bf16, scope=h.private())
                            body(A, C, l2, l1)

    print(launch.build(target="npu2"))


# CHECK-LABEL: TEST: load_pads_the_l2_source
# CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[0, 0] [16, 120] [120, 1])
# CHECK-SAME: {pad_after = array<i32: 0, 8>, pad_before = array<i32: 0, 0>}
# CHECK-SAME: (memref<16x128xbf16, 2 : i32>, memref<16x120xbf16, 1 : i32>)
@run
def load_pads_the_l2_source():
    build(lambda A, C, l2, l1: ops.load(l1, l2, pad_after=[0, PADDED - COLS]))


# CHECK-LABEL: TEST: pad_before_alone
# CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[0, 0] [16, 120] [120, 1])
# CHECK-SAME: {pad_after = array<i32: 0, 0>, pad_before = array<i32: 0, 8>}
@run
def pad_before_alone():
    build(lambda A, C, l2, l1: ops.load(l1, l2, pad_before=[0, PADDED - COLS]))


# CHECK-LABEL: TEST: load_pads_a_region
# CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[0, 64] [16, 56] [120, 1])
# CHECK-SAME: {pad_after = array<i32: 0, 8>, pad_before = array<i32: 0, 0>}
@run
def load_pads_a_region():
    build(
        lambda A, C, l2, l1: ops.load(l1, l2[:, 64:120], pad_after=[0, 8]),
        l1_shape=(ROWS, 64),
    )


# A reshaped view: the pad is counted in the view's axes, here 8 more rows of
# 8 columns, and only the element count is compared.
# CHECK-LABEL: TEST: load_pads_a_view
# CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[0, 0, 0] [16, 15, 8] [120, 8, 1])
# CHECK-SAME: {pad_after = array<i32: 0, 1, 0>, pad_before = array<i32: 0, 0, 0>}
@run
def load_pads_a_view():
    build(
        lambda A, C, l2, l1: ops.load(
            l1, l2.reshape(ROWS, COLS // 8, 8), pad_after=[0, 1, 0]
        )
    )


# CHECK-LABEL: TEST: store_pads_the_l2_source
# CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[0, 0] [16, 120] [120, 1])
# CHECK-SAME: {pad_after = array<i32: 0, 8>, pad_before = array<i32: 0, 0>}
# CHECK-SAME: (memref<16x128xbf16>, memref<16x120xbf16, 1 : i32>)
@run
def store_pads_the_l2_source():
    A = air.tensor([ROWS, COLS], bf16)
    C = air.tensor([ROWS, PADDED], bf16)
    with air.launch(name="pad") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    l2 = air.alloc([ROWS, COLS], bf16, scope=seg.private())
                    ops.load(l2, A)
                    ops.store(l2, C, pad_after=[0, PADDED - COLS])

    print(launch.build(target="npu2"))


# CHECK-LABEL: TEST: padded_extent_must_match
# CHECK: ValueError: transfer shape mismatch in air.api.ops.load
# CHECK-SAME: destination buffer is (16, 128) but the source padded buffer is (16, 124)
@run
def padded_extent_must_match():
    build(lambda A, C, l2, l1: ops.load(l1, l2, pad_after=[0, 4]))


# CHECK-LABEL: TEST: l3_source_cannot_pad
# CHECK: ValueError: air.api.ops.load: padding needs an L2 source
# CHECK-SAME: the source is tensor slice in L3
@run
def l3_source_cannot_pad():
    build(lambda A, C, l2, l1: ops.load(l1, A[:, :], pad_after=[0, 8]))


# CHECK-LABEL: TEST: l1_source_cannot_pad
# CHECK: ValueError: air.api.ops.store: padding needs an L2 source
# CHECK-SAME: the source is buffer in L1
@run
def l1_source_cannot_pad():
    build(lambda A, C, l2, l1: ops.store(l1, C, pad_after=[0, 0]))


# CHECK-LABEL: TEST: one_count_per_axis
# CHECK: ValueError: air.api.ops.load: pad_after has 1 entries but the source region has 2 axes
@run
def one_count_per_axis():
    build(lambda A, C, l2, l1: ops.load(l1, l2, pad_after=[8]))


# CHECK-LABEL: TEST: counts_are_non_negative_ints
# CHECK: ValueError: air.api.ops.load: pad_before entries must be integers in [0, 65535], got [0, -8]
@run
def counts_are_non_negative_ints():
    build(lambda A, C, l2, l1: ops.load(l1, l2, pad_before=[0, -8]))


# The region starts at a core's coordinate. The offset stays symbolic and the
# padding goes with it.
# CHECK-LABEL: TEST: pads_a_region_at_a_core_offset
# CHECK: air.herd
# CHECK: %[[OFF:.*]] = affine.apply
# CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[0, %[[OFF]]] [16, 56] [120, 1])
# CHECK-SAME: {pad_after = array<i32: 0, 8>, pad_before = array<i32: 0, 0>}
@run
def pads_a_region_at_a_core_offset():
    A = air.tensor([ROWS, COLS], bf16)
    C = air.tensor([ROWS, 128], bf16)
    with air.launch(name="pad") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    l2 = air.alloc([ROWS, COLS], bf16, scope=seg.private())
                    ops.load(l2, A)
                    with air.herd([range(2)], name="herd_0", shape=(2,)) as h:

                        @h.body
                        def _(tx):
                            l1 = air.alloc([ROWS, 64], bf16, scope=h.private())
                            k0 = tx * 56
                            ops.load(l1, l2[:, k0 : k0 + 56], pad_after=[0, 8])
                            ops.store(l1, C[:, tx * 64 : tx * 64 + 64])

    print(launch.build(target="npu2"))


# A transposed destination: the padded source walks [8, 16] blocks of 8 and the
# L1 side writes them back as [16, 8] blocks. Only the counts are compared.
# CHECK-LABEL: TEST: pads_into_a_transposed_destination
# CHECK: air.dma_memcpy_nd (%{{.*}}[0, 0, 0] [16, 16, 8] [8, 128, 1], %{{.*}}[0, 0, 0] [16, 15, 8] [120, 8, 1])
# CHECK-SAME: {pad_after = array<i32: 0, 1, 0>, pad_before = array<i32: 0, 0, 0>}
@run
def pads_into_a_transposed_destination():
    build(
        lambda A, C, l2, l1: ops.load(
            l1.reshape(16, 16, 8).transpose(1, 0, 2),
            l2.reshape(ROWS, COLS // 8, 8),
            pad_after=[0, 1, 0],
        ),
        l1_shape=(16, 128),
    )


# CHECK-LABEL: TEST: no_padding_no_attributes
# CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[] [] []) : (memref<16x120xbf16, 2 : i32>, memref<16x120xbf16, 1 : i32>)
@run
def no_padding_no_attributes():
    build(lambda A, C, l2, l1: ops.load(l1, l2), l1_shape=(ROWS, COLS))


# CHECK-LABEL: TEST: counts_are_not_bools
# CHECK: ValueError: air.api.ops.load: pad_after entries must be integers in [0, 65535], got [0, True]
@run
def counts_are_not_bools():
    build(lambda A, C, l2, l1: ops.load(l1, l2, pad_after=[0, True]))


# CHECK-LABEL: TEST: counts_fit_the_attribute
# CHECK: ValueError: air.api.ops.load: pad_after entries must be integers in [0, 65535], got [0, 65536]
@run
def counts_fit_the_attribute():
    build(lambda A, C, l2, l1: ops.load(l1, l2, pad_after=[0, 65536]))


# CHECK-LABEL: TEST: too_many_counts
# CHECK: ValueError: air.api.ops.store: pad_before has 3 entries but the source region has 2 axes
@run
def too_many_counts():
    A = air.tensor([ROWS, COLS], bf16)
    C = air.tensor([ROWS, PADDED], bf16)
    with air.launch(name="pad") as launch:

        @launch.body
        def _():
            with air.segment(name="seg") as seg:

                @seg.body
                def _():
                    l2 = air.alloc([ROWS, COLS], bf16, scope=seg.private())
                    ops.load(l2, A)
                    ops.store(l2, C, pad_before=[0, 0, 8])

    print(launch.build(target="npu2"))
