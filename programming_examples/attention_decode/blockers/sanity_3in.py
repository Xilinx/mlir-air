# Sanity 3: 3 distinct L3 input memrefs. Demonstrates the 2-S2MM/tile DMA
# channel limit — this test FAILS with wrong values, even though the
# arithmetic is just out = a*b*c. Pack inputs into a single L3 buffer to
# stay under the limit.
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

    @FuncOp.from_py_func(l3_t, l3_t, l3_t, l3_t)
    def k(a_in, b_in, c_in, xout):
        @herd(name="herd_0", sizes=[1, 1], operands=[a_in, b_in, c_in, xout])
        def _(_tx, _ty, _sx, _sy, la, lb, lc, lout):
            l1_a = AllocOp(l1_t, [], [])
            l1_b = AllocOp(l1_t, [], [])
            l1_c = AllocOp(l1_t, [], [])
            l1_o = AllocOp(l1_t, [], [])
            for ivx in range_(0, n, tile_n):
                dma_memcpy_nd(
                    l1_a, la, src_offsets=[ivx], src_sizes=[tile_n], src_strides=[1]
                )
                dma_memcpy_nd(
                    l1_b, lb, src_offsets=[ivx], src_sizes=[tile_n], src_strides=[1]
                )
                dma_memcpy_nd(
                    l1_c, lc, src_offsets=[ivx], src_sizes=[tile_n], src_strides=[1]
                )
                c0 = ConstantOp(IndexType.get(), 0)
                cTile = ConstantOp(IndexType.get(), tile_n)
                cVl = ConstantOp(IndexType.get(), vl)
                cst0 = arith.ConstantOp(bf16_t, 0.0)
                for j in range_(c0, cTile, cVl):
                    sub_a = subview(l1_a.result, [j], [vl], [1])
                    sub_b = subview(l1_b.result, [j], [vl], [1])
                    sub_c = subview(l1_c.result, [j], [vl], [1])
                    sub_o = subview(l1_o.result, [j], [vl], [1])
                    va = transfer_read(vec_t, sub_a, [c0], id_map, cst0, [True])
                    vb = transfer_read(vec_t, sub_b, [c0], id_map, cst0, [True])
                    vc = transfer_read(vec_t, sub_c, [c0], id_map, cst0, [True])
                    out = arith_dialect.mulf(arith_dialect.mulf(va, vb), vc)
                    transfer_write(None, out, sub_o, [c0], id_map, [True])
                    yield_([])
                dma_memcpy_nd(
                    lout, l1_o, dst_offsets=[ivx], dst_sizes=[tile_n], dst_strides=[1]
                )
                yield_([])
            DeallocOp(l1_a)
            DeallocOp(l1_b)
            DeallocOp(l1_c)
            DeallocOp(l1_o)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = rng.uniform(-1.0, 1.0, 1024).astype(bfloat16)
    b = rng.uniform(-1.0, 1.0, 1024).astype(bfloat16)
    c = rng.uniform(-1.0, 1.0, 1024).astype(bfloat16)
    ref = (a.astype(np.float32) * b.astype(np.float32) * c.astype(np.float32)).astype(
        bfloat16
    )
    mod = build_module(1024, 64, 16)
    runner = XRTRunner(
        verbose=False,
        omit_while_true_loop=False,
        omit_pingpong=True,
        output_format="xclbin",
    )
    exit(
        runner.run_test(
            mod, inputs=[a, b, c], expected_outputs=[ref], rtol=5e-2, atol=5e-2
        )
    )
