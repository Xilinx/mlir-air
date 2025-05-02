# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
import argparse
from math import cos, sin

from air.ir import *
from air.dialects.air import *
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, load, store
from air.dialects.func import FuncOp, CallOp
from air.dialects import scf
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper
from air.backend.xrt import XRTBackend

range_ = for_


@module_builder
def build_module(n, np_dtype_in, np_dtype_out, param):

    xrt_dtype_in = type_mapper(np_dtype_in)
    xrt_dtype_out = type_mapper(np_dtype_out)

    # L3 MemRefTypes
    memrefTyIn = MemRefType.get([n], xrt_dtype_in)
    memrefTyOut = MemRefType.get([n], xrt_dtype_in)

    @FuncOp.from_py_func(memrefTyIn, memrefTyOut)
    def conditional_branch(arg0, arg1):

        launch_size = [1, 1]

        @launch(operands=[arg0, arg1], sizes=launch_size)
        def launch_body(
            launch_ivx,
            launch_ivy,
            launch_sizex,
            launch_sizey,
            l3_in_data,
            l3_out_data,
        ):

            @segment(name="segment_0", operands=[l3_in_data, l3_out_data])
            def segment_body(
                _l3_in_data,
                _l3_out_data,
            ):
                # L2 MemRefTypes
                l2_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L2)
                l2MemrefTyIn = MemRefType.get(
                    shape=[n],
                    element_type=xrt_dtype_in,
                    memory_space=l2_mem_space,
                )
                l2MemrefTyOut = MemRefType.get(
                    shape=[n],
                    element_type=xrt_dtype_in,
                    memory_space=l2_mem_space,
                )
                l2_in_data = AllocOp(l2MemrefTyIn, [], [])
                l2_out_data = AllocOp(l2MemrefTyOut, [], [])
                dma_memcpy_nd(
                    l2_in_data,
                    _l3_in_data,
                )

                param_arg = arith.ConstantOp.create_index(param)

                @herd(
                    name="herd_0",
                    sizes=[1, 1],
                    operands=[l2_in_data, l2_out_data, param_arg],
                )
                def herd_body_0(
                    _tx, _ty, _sx, _sy, _l2_in_data, _l2_out_data, _param_arg
                ):

                    l1_mem_space = IntegerAttr.get(T.i32(), MemorySpace.L1)
                    l1MemrefTyIn = MemRefType.get(
                        shape=[n],
                        element_type=xrt_dtype_in,
                        memory_space=l1_mem_space,
                    )
                    l1MemrefTyOut = MemRefType.get(
                        shape=[n],
                        element_type=xrt_dtype_in,
                        memory_space=l1_mem_space,
                    )

                    l1_in_data = AllocOp(l1MemrefTyIn, [], [])
                    dma_memcpy_nd(
                        l1_in_data,
                        _l2_in_data,
                    )

                    l1_buf = AllocOp(l1MemrefTyIn, [], [])

                    # condition
                    bool = IntegerType.get_signless(1)
                    param_arg = arith.index_cast(bool, _param_arg)
                    if_op = scf.IfOp(param_arg, hasElse=True)
                    with InsertionPoint(if_op.then_block):
                        for i in range_(0, n):
                            inval = load(l1_in_data, [i])
                            add100 = arith.addi(
                                inval, ConstantOp(IntegerAttr.get(T.i32(), 100), None)
                            )
                            store(add100, l1_buf, [i])
                            yield_([])
                        yield_([])
                    with InsertionPoint(if_op.else_block):
                        for i in range_(0, n):
                            inval = load(l1_in_data, [i])
                            mul100 = arith.muli(
                                inval, ConstantOp(IntegerAttr.get(T.i32(), 100), None)
                            )
                            store(mul100, l1_buf, [i])
                            yield_([])
                        yield_([])

                    dma_memcpy_nd(
                        _l2_out_data,
                        l1_buf,
                    )
                    DeallocOp(l1_in_data)

                dma_memcpy_nd(
                    _l3_out_data,
                    l2_out_data,
                )

                DeallocOp(l2_in_data)
                DeallocOp(l2_out_data)


if __name__ == "__main__":
    # Default values.
    N = 48
    param = 0
    INPUT_DATATYPE = np.int32
    OUTPUT_DATATYPE = np.int32

    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Builds, runs, and tests the passthrough_dma example",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--print-module-only",
        action="store_true",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=N,
        help="N dimension size in a (1xK) * (KxN) matmul",
    )
    parser.add_argument(
        "--param",
        type=int,
        default=param,
        help="Runtime variable",
    )
    args = parser.parse_args()

    mlir_module = build_module(
        args.n,
        INPUT_DATATYPE,
        OUTPUT_DATATYPE,
        args.param,
    )
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    inputs = np.arange(0, args.n, dtype=INPUT_DATATYPE).reshape(args.n)
    outputs = np.zeros(shape=(args.n), dtype=OUTPUT_DATATYPE)
    if args.param == 1:
        outputs = inputs + 100
    else:
        outputs = inputs * 100

    ###### Compile and test
    runner = XRTRunner(verbose=args.verbose)
    exit(
        runner.run_test(
            mlir_module,
            inputs=[inputs],
            expected_outputs=[outputs],
        )
    )
