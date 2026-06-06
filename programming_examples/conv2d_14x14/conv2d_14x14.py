# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""mlir-air port of upstream mlir-aie programming_examples/ml/conv2d_14x14.

Full-size 8x4 herd (32 cores) executing 9 oc-groups x 16 y-rows x 4 xb-blocks
of 14x14 stride-14 conv on 896x896 input, 4 in-channels, 1152 out-channels,
uint8 act / int8 wts / int8 out.

Design highlights:
  * L2 act buffer holds a full 14-row image-band per memtile (50176 B each,
    4 memtiles); producer L3->L2 uses a 4D scatter wrap matching upstream's
    [<14,56>,<64,784>,<56,1>], and consumer L2->L1 gathers via the same
    [<2,6272>,<98,8>,<8,784>,<8,1>] window the kernel expects.
  * L1 weights are streamed direct L3->L1 with a 4D shim BD per col.
  * L2->L3 output drain is a flat 65536-B/col S2MM per g iter; host
    reorders the kernel's [oc=2,nt=2,nt8=8,oc8=8] byte order into
    upstream-host-major [nt=2,nt8=8,oc=2,oc8=8] before comparing.
"""

import argparse
import numpy as np
import torch

from air.ir import (
    AffineConstantExpr,
    AffineExpr,
    AffineMap,
    AffineSymbolExpr,
    IntegerAttr,
    MemRefType,
    StringAttr,
    UnitAttr,
)
from air.dialects.affine import apply as affine_apply
from air.dialects.air import (
    MemorySpace,
    T,
    dma_memcpy_nd,
    herd,
    launch,
    module_builder,
    segment,
)
from air.dialects.arith import ConstantOp
from air.dialects.func import CallOp, FuncOp
from air.dialects.memref import AllocOp, DeallocOp
from air.dialects.scf import for_, yield_
from air.backend.xrt_runner import XRTRunner

range_ = for_


def _mul_const_map(c):
    """affine_map<(s0) -> (s0 * c)> for affine_apply of `iv * c`."""
    return AffineMap.get(
        0,
        1,
        [AffineExpr.get_mul(AffineSymbolExpr.get(0), AffineConstantExpr.get(c))],
    )


def _add_mul_map(c):
    """affine_map<(s0, s1) -> (s0 * c + s1)> for `iv0 * c + iv1`."""
    return AffineMap.get(
        0,
        2,
        [
            AffineExpr.get_add(
                AffineExpr.get_mul(AffineSymbolExpr.get(0), AffineConstantExpr.get(c)),
                AffineSymbolExpr.get(1),
            )
        ],
    )


@module_builder
def build_module():
    i8 = T.i8()
    i32 = T.i32()
    l2_attr = IntegerAttr.get(i32, MemorySpace.L2)
    l1_attr = IntegerAttr.get(i32, MemorySpace.L1)

    # L3 (host) memref types
    memrefTyI = MemRefType.get([4, 802816], i8)
    memrefTyW = MemRefType.get([8, 112896], i8)
    memrefTyO = MemRefType.get([8, 589824], i8)

    # L2 (memtile)
    l2ActTy = MemRefType.get([4, 14, 3584], i8, memory_space=l2_attr)
    l2OutTy = MemRefType.get([8, 4, 64, 256], i8, memory_space=l2_attr)

    # L1 (compute tile)
    l1InTy = MemRefType.get([12544], i8, memory_space=l1_attr)
    l1WtsTy = MemRefType.get([12544], i8, memory_space=l1_attr)
    l1OutTy = MemRefType.get([256], i8, memory_space=l1_attr)

    # External kernel
    conv_func = FuncOp(
        "conv2dk14_i8",
        ([l1InTy, l1WtsTy, l1OutTy, i32, i32, i32, i32, i32], []),
        visibility="private",
    )
    conv_func.attributes["link_with"] = StringAttr.get("conv2dk14.o")
    conv_func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

    @FuncOp.from_py_func(memrefTyI, memrefTyW, memrefTyO)
    def conv2dk14_test(I, W, O):
        @launch(operands=[I, W, O])
        def launch_body(l3_I, l3_W, l3_O):
            @segment(name="conv2dk14_seg", operands=[l3_I, l3_W, l3_O])
            def segment_body(l3_I_s, l3_W_s, l3_O_s):
                l2_act = AllocOp(l2ActTy, [], [])
                l2_out = AllocOp(l2OutTy, [], [])

                # L3->L2 act fill (9 oc-groups x 16 y-rows of the 4D scatter
                # wrap matching upstream's [<14,56>,<64,784>,<56,1>]).
                mul14 = _mul_const_map(14)
                for _gp in range_(0, 9):
                    for yp in range_(0, 16):
                        y_off = affine_apply(mul14, [yp])
                        dma_memcpy_nd(
                            l2_act,
                            l3_I_s,
                            src_offsets=[0, y_off, 0, 0],
                            src_sizes=[4, 14, 64, 56],
                            src_strides=[802816, 3584, 56, 1],
                            dst_offsets=[0, 0, 0, 0],
                            dst_sizes=[4, 14, 64, 56],
                            dst_strides=[50176, 56, 784, 1],
                        )
                        yield_([])
                    yield_([])

                @herd(
                    name="conv2dk14_herd",
                    sizes=[8, 4],
                    operands=[l3_W_s, l2_act, l2_out],
                )
                def herd_body(tx, ty, _sx, _sy, _l3_W, _l2_act, _l2_out):
                    l1_in = AllocOp(l1InTy, [], [])
                    l1_wts = AllocOp(l1WtsTy, [], [])
                    l1_out = AllocOp(l1OutTy, [], [])

                    c224 = ConstantOp(IntegerAttr.get(i32, 224), None)
                    c4_i32 = ConstantOp(IntegerAttr.get(i32, 4), None)
                    c16_i32 = ConstantOp(IntegerAttr.get(i32, 16), None)
                    c14_i32 = ConstantOp(IntegerAttr.get(i32, 14), None)

                    mul12544 = _mul_const_map(12544)
                    y4_plus_xb = _add_mul_map(4)

                    for g in range_(0, 9):
                        g_off = affine_apply(mul12544, [g])

                        # L3->L1 wts (per oc-group, per col=tx).
                        dma_memcpy_nd(
                            l1_wts,
                            _l3_W,
                            src_offsets=[tx, g_off],
                            src_sizes=[1, 12544],
                            src_strides=[112896, 1],
                        )

                        for y in range_(0, 16):
                            for xb in range_(0, 4):
                                # L2->L1 act gather (4D inner wrap
                                # [<2,6272>,<98,8>,<8,784>,<8,1>] selected
                                # per (tx_row=ty, xb_col=xb) image block).
                                dma_memcpy_nd(
                                    l1_in,
                                    _l2_act,
                                    src_offsets=[ty, xb, 0, 0, 0, 0],
                                    src_sizes=[1, 1, 2, 98, 8, 8],
                                    src_strides=[
                                        50176,
                                        12544,
                                        6272,
                                        8,
                                        784,
                                        1,
                                    ],
                                )

                                CallOp(
                                    conv_func,
                                    [
                                        l1_in,
                                        l1_wts,
                                        l1_out,
                                        c224,
                                        c4_i32,
                                        c16_i32,
                                        c14_i32,
                                        c14_i32,
                                    ],
                                )

                                # L1->L2 out: drop the 256-B kernel result
                                # at (col=tx, row=ty, y_xb_slot=i_yx).
                                i_yx = affine_apply(y4_plus_xb, [y, xb])
                                dma_memcpy_nd(
                                    _l2_out,
                                    l1_out,
                                    dst_offsets=[tx, ty, i_yx, 0],
                                    dst_sizes=[1, 1, 1, 256],
                                    dst_strides=[65536, 16384, 256, 1],
                                )
                                yield_([])
                            yield_([])
                        yield_([])

                    DeallocOp(l1_in)
                    DeallocOp(l1_wts)
                    DeallocOp(l1_out)

                herd_body.attributes["link_with"] = StringAttr.get("conv2dk14.o")

                # L2->L3 out drain (flat 65536 B per (col, g iter)). Host
                # reorders the per-call [oc, nt, nt8, oc8] byte order into
                # spatial [nt, nt8, oc, oc8] before comparing.
                mul65536 = _mul_const_map(65536)
                for g in range_(0, 9):
                    gg_off = affine_apply(mul65536, [g])
                    dma_memcpy_nd(
                        l3_O_s,
                        l2_out,
                        dst_offsets=[0, gg_off],
                        dst_sizes=[8, 65536],
                        dst_strides=[589824, 1],
                        src_offsets=[0, 0, 0, 0],
                        src_sizes=[8, 4, 64, 256],
                        src_strides=[65536, 16384, 256, 1],
                    )
                    yield_([])

                DeallocOp(l2_act)
                DeallocOp(l2_out)


# Problem dims (fixed to match the IR string)
WIDTH = 896
HEIGHT = 896
CI = 4
CO = 1152
KSZ = 14
WIDTH_OUT = WIDTH // KSZ  # 64
HEIGHT_OUT = HEIGHT // KSZ  # 64
NUM_G = 9  # oc-groups
CLIP_MIN, CLIP_MAX = -128, 127


def build_inputs_and_golden():
    """Generate the inputs and the byte-permuted golden expected by the NPU
    output layout.

    The NPU writes 4,718,592 raw bytes laid out as
        [col=8][g=9][row=4][yxb=64][oc=2][nt=2][nt8=8][oc8=8]
    where each 256-B call's inner [oc, nt, nt8, oc8] is the kernel-write
    order. The host-expected layout (which matches PyTorch's (CO, HOUT,
    WOUT)) is [nt, nt8, oc, oc8]; we permute the golden's bytes here so
    XRTRunner's bit-exact compare succeeds end-to-end.
    """
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)

    int_inp = torch.randint(0, 255, (1, CI, HEIGHT, WIDTH)).type(torch.FloatTensor)
    int_weight = torch.randint(2, 20, (CO, CI, KSZ, KSZ)).type(torch.FloatTensor)

    # Golden via pure integer arithmetic so it matches the kernel's
    # `(sum + sign*8192) >> 14` rounding (round half away from zero). Using
    # torch.round() introduces banker's rounding mismatches on ties.
    inp_t = int_inp.to(torch.int32).float()
    wt_t = int_weight.float()
    sum_int = (
        torch.nn.functional.conv2d(inp_t, wt_t, stride=KSZ, padding=0)
        .squeeze(0)
        .numpy()
        .astype(np.int32)
    )
    bias = np.where(sum_int >= 0, 8192, -8192).astype(np.int32)
    quantized = (sum_int + bias) // 16384
    golden_int = np.clip(quantized, CLIP_MIN, CLIP_MAX).astype(np.int8)
    # golden_int shape: (CO, HEIGHT_OUT, WIDTH_OUT) = (1152, 64, 64)

    # Host act buffer must be YXC-packed per col so the per-pixel (CI=4)
    # bytes are contiguous - which is what the L3->L2 scatter wrap and the
    # kernel's gather wrap both assume. The raw torch tensor is (CI, H, W);
    # transpose to (H, W, CI) then flatten gives the YXC stream.
    int_inp_np = int_inp.squeeze().data.numpy().astype(np.uint8)  # (CI, H, W)
    in1 = (
        np.transpose(int_inp_np, (1, 2, 0)).reshape(CI, HEIGHT * WIDTH).astype(np.uint8)
    )
    # Host expects (CI, HEIGHT * WIDTH) = (4, 802816); air.func arg shape
    # matches.

    # Weights need to be reshaped to OIYX -> OYXIO8 grouped form so the
    # 8x112896 buffer matches what the kernel reads. Upstream uses
    # DataShaper.reorder_mat("OYXIO8", "OIYX"). We replicate by computing
    # the expected (8, 112896) byte buffer via numpy: each col holds CO//8
    # = 144 oc-groups of 12544 B each. 8 cols * 9 oc_per_col_group * 12544
    # = 903168 B total = 8 * 112896.
    wts_np = int_weight.data.numpy().astype(np.int8)  # (CO, CI, KSZ, KSZ)
    # OYXIO8 layout: split CO into (CO//8, 8). Source label "OIYX" means
    # axes (CO, CI, Y, X) which is what we have. Target label "OYXIO8"
    # means (CO//8, Y, X, CI, 8 (= oc inner)). Reorder:
    #   OIYX -> OYXIO8:
    #   step 1: split O into (O_outer = CO//8, O_inner = 8): (O_outer, 8,
    #           CI, Y, X)
    #   step 2: permute to (O_outer, Y, X, CI, 8):
    co_outer = CO // 8
    wts_split = wts_np.reshape(co_outer, 8, CI, KSZ, KSZ)
    wts_oyxio8 = np.transpose(wts_split, (0, 3, 4, 2, 1))  # (co_outer, Y, X, CI, 8)
    # Pack into (8 cols, 112896 bytes per col). 8 cols * 144 oc-groups per
    # col * 12544 B per oc-group. Each oc-group = (Y*X*CI*8) = 14*14*4*8 =
    # 6272... wait, an oc-group spans 8 output channels. Volume per oc-
    # group = KSZ*KSZ*CI*8 = 14*14*4*8 = 6272 bytes (= int8 elements).
    # But the kernel reads 12544 B per oc-group. The 12544 B includes
    # something else (likely 2x for stride or duplication). Upstream test
    # uses a single concatenated wts buffer total 8 * 112896 = 903168 B
    # which is exactly CO * CI * KSZ * KSZ * 1 byte (since int8) = 1152 *
    # 4 * 14 * 14 = 903168. So the (8, 112896) layout simply packs the
    # entire weight set into the 8-col shim arrangement.
    in2 = wts_oyxio8.tobytes()
    in2_arr = np.frombuffer(in2, dtype=np.int8).reshape(8, 112896)

    # Construct the byte-permuted golden expected at the L3 output.
    # NPU raw bytes (4,718,592) reshape as
    #   (col=8, g=9, row=4, yxb=64, 2, 2, 8, 8) - last 4 dims are kernel
    #   write order [oc, nt, nt8, oc8].
    # diff_v21fold_correct.py transposes to spatial order and reshapes;
    # we invert it to permute the golden into NPU-byte order.
    #
    # Spatial layout from golden: (CO=1152, HOUT=64, WOUT=64).
    # Co-group decomposition: CO = 8 cols * 9 g * 16 oc-per-group =
    #   col*9*16 + g*16 + oc_in_group, where oc_in_group = oc*8 + oc8.
    # Output decomposition: HOUT = 4 rows * 16 y = row*16 + y. WOUT =
    #   4 xb * 16 (nt*8 + nt8) = xb*16 + nt*8 + nt8.
    g_arr = golden_int.reshape(8, 9, 2, 8, 4, 16, 4, 2, 8)
    # axes: (col=8, g=9, oc=2, oc8=8, row=4, y=16, xb=4, nt=2, nt8=8)
    # Target raw byte order: (col, g, row, yxb=y*4+xb, oc, nt, nt8, oc8)
    # where the inner 4 dims (oc, nt, nt8, oc8) are the kernel write
    # order. yxb interleaves (y, xb).
    g_arr = np.transpose(g_arr, (0, 1, 4, 5, 6, 2, 7, 8, 3))
    # now: (col=8, g=9, row=4, y=16, xb=4, oc=2, nt=2, nt8=8, oc8=8)
    g_arr = g_arr.reshape(8, 9, 4, 16 * 4, 2, 2, 8, 8)
    # (col, g, row, yxb=64, oc, nt, nt8, oc8) - matches NPU byte layout.
    expected_out_npu = g_arr.reshape(8, 589824).astype(np.int8)

    return in1, in2_arr, expected_out_npu


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="conv2d_14x14.py")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-p", "--print-module-only", action="store_true")
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["xclbin", "elf"],
        default="xclbin",
        dest="output_format",
    )
    args = parser.parse_args()

    mlir_module = build_module()
    if args.print_module_only:
        print(mlir_module)
        exit(0)

    in1, in2, expected_out = build_inputs_and_golden()

    runner = XRTRunner(
        verbose=args.verbose,
        omit_while_true_loop=False,
        output_format=args.output_format,
        instance_name="conv2dk14_test",
        lower_linalg_to_func="conv2dk14.o",
        target_device="npu2",
    )
    exit(
        runner.run_test(
            mlir_module,
            inputs=[in1, in2],
            expected_outputs=[expected_out],
        )
    )
