# Sanity: 1 input, 1 output, copy. If this fails, the harness is broken.
import argparse, numpy as np
from ml_dtypes import bfloat16
from air.ir import *
from air.dialects.air import *
from air.dialects import arith
from air.dialects.arith import ConstantOp
from air.dialects.memref import AllocOp, DeallocOp, subview
from air.dialects.vector import transfer_read, transfer_write
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

    @FuncOp.from_py_func(l3_t, l3_t)
    def k(xin, xout):
        @herd(name="herd_0", sizes=[1, 1], operands=[xin, xout])
        def _(_tx, _ty, _sx, _sy, lin, lout):
            l1_in = AllocOp(l1_t, [], [])
            l1_out = AllocOp(l1_t, [], [])
            for ivx in range_(0, n, tile_n):
                dma_memcpy_nd(
                    l1_in, lin, src_offsets=[ivx], src_sizes=[tile_n], src_strides=[1]
                )
                c0 = ConstantOp(IndexType.get(), 0)
                cTile = ConstantOp(IndexType.get(), tile_n)
                cVl = ConstantOp(IndexType.get(), vl)
                cst0 = arith.ConstantOp(bf16_t, 0.0)
                for j in range_(c0, cTile, cVl):
                    sub_i = subview(l1_in.result, [j], [vl], [1])
                    sub_o = subview(l1_out.result, [j], [vl], [1])
                    v = transfer_read(vec_t, sub_i, [c0], id_map, cst0, [True])
                    transfer_write(None, v, sub_o, [c0], id_map, [True])
                    yield_([])
                dma_memcpy_nd(
                    lout, l1_out, dst_offsets=[ivx], dst_sizes=[tile_n], dst_strides=[1]
                )
                yield_([])
            DeallocOp(l1_in)
            DeallocOp(l1_out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--tile-n", type=int, default=64)
    p.add_argument("--vl", type=int, default=16)
    args = p.parse_args()
    rng = np.random.default_rng(0)
    x = rng.uniform(-1.0, 1.0, args.n).astype(bfloat16)
    mod = build_module(args.n, args.tile_n, args.vl)
    runner = XRTRunner(
        verbose=False, omit_while_true_loop=False, output_format="xclbin"
    )
    exit(runner.run_test(mod, inputs=[x], expected_outputs=[x], rtol=1e-3, atol=1e-3))
