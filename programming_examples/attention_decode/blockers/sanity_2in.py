# Sanity 2: 2 inputs (a, b), 1 output. out = a*b.
import argparse, numpy as np
from ml_dtypes import bfloat16
from air.ir import *
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.vector import transfer_read, transfer_write
from air.dialects import arith as arith_dialect
from air.dialects.func import FuncOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner, type_mapper

range_ = for_


@module_builder
def build_module(n, tile_n, vl):
    bf16_t = type_mapper(bfloat16)
    vec_t = VectorType.get([vl], bf16_t)
    id_map = AffineMapAttr.get(AffineMap.get_identity(1))
    l3_t = MemRefType.get([n], bf16_t)
    l1_mem = IntegerAttr.get(T.i32(), MemorySpace.L1)
    l1_t = MemRefType.get([tile_n], bf16_t, memory_space=l1_mem)

    @FuncOp.from_py_func(l3_t, l3_t, l3_t)
    def k(a_in, b_in, xout):
        @herd(name="herd_0", sizes=[1, 1], operands=[a_in, b_in, xout])
        def _(_tx, _ty, _sx, _sy, la, lb, lout):
            l1_a = AllocOp(l1_t, [], [])
            l1_b = AllocOp(l1_t, [], [])
            l1_o = AllocOp(l1_t, [], [])
            for ivx in range_(0, n, tile_n):
                dma_memcpy_nd(
                    l1_a, la, src_offsets=[ivx], src_sizes=[tile_n], src_strides=[1]
                )
                dma_memcpy_nd(
                    l1_b, lb, src_offsets=[ivx], src_sizes=[tile_n], src_strides=[1]
                )
                c0 = ConstantOp(IndexType.get(), 0)
                cTile = ConstantOp(IndexType.get(), tile_n)
                cVl = ConstantOp(IndexType.get(), vl)
                cst0 = arith.ConstantOp(bf16_t, 0.0)
                for j in range_(c0, cTile, cVl):
                    sub_a = subview(l1_a.result, [j], [vl], [1])
                    sub_b = subview(l1_b.result, [j], [vl], [1])
                    sub_o = subview(l1_o.result, [j], [vl], [1])
                    va = transfer_read(vec_t, sub_a, [c0], id_map, cst0, [True])
                    vb = transfer_read(vec_t, sub_b, [c0], id_map, cst0, [True])
                    out = arith_dialect.mulf(va, vb)
                    transfer_write(None, out, sub_o, [c0], id_map, [True])
                    yield_([])
                dma_memcpy_nd(
                    lout, l1_o, dst_offsets=[ivx], dst_sizes=[tile_n], dst_strides=[1]
                )
                yield_([])
            DeallocOp(l1_a)
            DeallocOp(l1_b)
            DeallocOp(l1_o)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--tile-n", type=int, default=64)
    p.add_argument("--vl", type=int, default=16)
    args = p.parse_args()
    rng = np.random.default_rng(0)
    a = rng.uniform(-1.0, 1.0, args.n).astype(bfloat16)
    b = rng.uniform(-1.0, 1.0, args.n).astype(bfloat16)
    ref = (a.astype(np.float32) * b.astype(np.float32)).astype(bfloat16)
    mod = build_module(args.n, args.tile_n, args.vl)
    runner = XRTRunner(
        verbose=False, omit_while_true_loop=False, output_format="xclbin"
    )
    exit(
        runner.run_test(
            mod, inputs=[a, b], expected_outputs=[ref], rtol=5e-2, atol=5e-2
        )
    )
