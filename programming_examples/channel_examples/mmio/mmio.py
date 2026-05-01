# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""mmio_add: end-to-end demonstration of `channel_type="mmio"`.

A single AIE tile computes `out[i] = a[i] + c[i]` where:
  * `a` is a host input vector delivered to L1 via shim DMA;
  * `c` is a compile-time constant (dense<42>) delivered to L1 via
    host-issued MMIO writes (`aiex.npu.blockwrite`) — no DMA channel,
    no aie.flow, no shim allocation reserved for the constant payload;
  * `out` is the L1 result drained back to the host via shim DMA.

This is the AIR-Python equivalent of the hand-written
`/tmp/mmio_bench/aie_q_n*_w*.mlir` benchmark variants. It exercises the
new `channel_type="mmio"` lowering on real NPU2 hardware.
"""

import argparse

import numpy as np

from air.ir import *
from air.dialects.air import *
from air.dialects import arith
from air.dialects.func import FuncOp
from air.dialects.memref import AllocOp, DeallocOp, GlobalOp, get_global, load, store
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner

N = 64  # vector length
CONST_VALUE = 42


@module_builder
def build_module():
    i32 = T.i32()
    l3_ty = MemRefType.get([N], i32)
    l1_ty = MemRefType.get(
        shape=[N],
        element_type=i32,
        memory_space=IntegerAttr.get(T.i32(), MemorySpace.L1),
    )

    # memref.global "private" @const_data : memref<N x i32> = dense<42>
    # Source for the MMIO blockwrite payload. Must live at module scope so
    # it is visible from the L3 control func and survives air-to-aie
    # outlining. The initial_value attribute is built from a tensor type
    # (DenseElementsAttr is tensor-typed), then assigned to a memref-typed
    # global.
    const_tensor_ty = RankedTensorType.get([N], i32)
    const_attr = DenseElementsAttr.get(
        np.full([N], CONST_VALUE, dtype=np.int32),
        type=const_tensor_ty,
    )
    GlobalOp(
        sym_name="const_data",
        type_=TypeAttr.get(l3_ty),
        sym_visibility="private",
        initial_value=const_attr,
    )

    # Channels: `mmio` for the constant, ordinary dma_stream for input/out.
    channel("a_in", size=[1])  # default: dma_stream
    channel("c_mmio", size=[1], channel_type="mmio")
    channel("o_out", size=[1])

    @FuncOp.from_py_func(l3_ty, l3_ty)
    def mmio_add(arg_a, arg_o):

        @launch(operands=[arg_a, arg_o], sizes=[1, 1])
        def launch_body(_lx, _ly, _sx, _sy, l3_a, l3_o):
            # Host-side puts: input via DMA, constant via MMIO.
            ChannelPut("a_in", l3_a)

            # Source for MMIO must be a memref.get_global of the constant.
            const_l3 = get_global(l3_ty, "const_data")
            ChannelPut("c_mmio", const_l3)

            @segment(name="seg")
            def segment_body():

                @herd(name="h", sizes=[1, 1])
                def herd_body(_tx, _ty, _sx2, _sy2):
                    a_l1 = AllocOp(l1_ty, [], [])
                    c_l1 = AllocOp(l1_ty, [], [])
                    o_l1 = AllocOp(l1_ty, [], [])

                    ChannelGet("a_in", a_l1)
                    ChannelGet("c_mmio", c_l1)

                    # out_l1[i] = a_l1[i] + c_l1[i] (scalar loop, i32)
                    for i in for_(N):
                        va = load(a_l1, [i])
                        vc = load(c_l1, [i])
                        vo = arith.addi(va, vc)
                        store(vo, o_l1, [i])
                        yield_([])

                    ChannelPut("o_out", o_l1)
                    DeallocOp(a_l1)
                    DeallocOp(c_l1)
                    DeallocOp(o_l1)

            ChannelGet("o_out", l3_o)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="mmio_add")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=["compile-only", "compile-and-run"],
        default="compile-and-run",
    )
    parser.add_argument(
        "--output-format",
        choices=["xclbin", "elf"],
        default="xclbin",
    )
    args = parser.parse_args()

    mlir_module = build_module()
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    rng = np.random.default_rng(0)
    input_a = rng.integers(low=0, high=100, size=N, dtype=np.int32)
    expected_out = (input_a + CONST_VALUE).astype(np.int32)

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="mmio_add",
    )
    exit(
        runner.run_test(
            mlir_module,
            inputs=[input_a],
            expected_outputs=[expected_out],
        )
    )
