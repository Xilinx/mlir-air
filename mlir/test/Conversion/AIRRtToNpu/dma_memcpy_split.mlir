//===- dma_memcpy_split.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//


// RUN: air-opt -airrt-to-npu --split-input-file %s | FileCheck %s


// CHECK-LABEL: aie.device(npu1_4col)
// CHECK: aie.shim_dma_allocation @[[ID0:airMemcpyId29]](S2MM, 0, 0)
// CHECK: memref.global "public" @[[ID0]] : memref<128x128xf32, 1>
// CHECK: aie.shim_dma_allocation @[[ID1:airMemcpyId4]](MM2S, 0, 0)
// CHECK: memref.global "public" @[[ID1]] : memref<128x256xbf16, 1>
// CHECK: aie.shim_dma_allocation @[[ID2:airMemcpyId10]](MM2S, 1, 0)
// CHECK: memref.global "public" @[[ID2]] : memref<32x8x8x16xbf16, 1>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 4, 128, 128][0, 128, 512, 1]) {id = 0 : i64, metadata = @[[ID1]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][32, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 65536][32, 8, 8, 8][2048, 32, 256, 1]) {id = 2 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 131072][32, 8, 8, 8][2048, 32, 256, 1]) {id = 3 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 196608][32, 8, 8, 8][2048, 32, 256, 1]) {id = 4 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 128, 128][0, 0, 512, 1]) {id = 5 : i64, metadata = @[[ID0]]} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 4, 128, 128][0, 128, 512, 1]) {id = 0 : i64, metadata = @[[ID1]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 8][32, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 65544][32, 8, 8, 8][2048, 32, 256, 1]) {id = 2 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 131080][32, 8, 8, 8][2048, 32, 256, 1]) {id = 3 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 196616][32, 8, 8, 8][2048, 32, 256, 1]) {id = 4 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 128][1, 1, 128, 128][0, 0, 512, 1]) {id = 5 : i64, metadata = @[[ID0]]} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 4, 128, 128][0, 128, 512, 1]) {id = 0 : i64, metadata = @[[ID1]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 16][32, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 65552][32, 8, 8, 8][2048, 32, 256, 1]) {id = 2 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 131088][32, 8, 8, 8][2048, 32, 256, 1]) {id = 3 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 196624][32, 8, 8, 8][2048, 32, 256, 1]) {id = 4 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 256][1, 1, 128, 128][0, 0, 512, 1]) {id = 5 : i64, metadata = @[[ID0]]} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 4, 128, 128][0, 128, 512, 1]) {id = 0 : i64, metadata = @[[ID1]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 24][32, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 65560][32, 8, 8, 8][2048, 32, 256, 1]) {id = 2 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 131096][32, 8, 8, 8][2048, 32, 256, 1]) {id = 3 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 196632][32, 8, 8, 8][2048, 32, 256, 1]) {id = 4 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 384][1, 1, 128, 128][0, 0, 512, 1]) {id = 5 : i64, metadata = @[[ID0]]} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 65536][1, 4, 128, 128][0, 128, 512, 1]) {id = 0 : i64, metadata = @[[ID1]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][32, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 65536][32, 8, 8, 8][2048, 32, 256, 1]) {id = 2 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 131072][32, 8, 8, 8][2048, 32, 256, 1]) {id = 3 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 196608][32, 8, 8, 8][2048, 32, 256, 1]) {id = 4 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 128, 0][1, 1, 128, 128][0, 0, 512, 1]) {id = 5 : i64, metadata = @[[ID0]]} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 65536][1, 4, 128, 128][0, 128, 512, 1]) {id = 0 : i64, metadata = @[[ID1]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 8][32, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 65544][32, 8, 8, 8][2048, 32, 256, 1]) {id = 2 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 131080][32, 8, 8, 8][2048, 32, 256, 1]) {id = 3 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 196616][32, 8, 8, 8][2048, 32, 256, 1]) {id = 4 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 128, 128][1, 1, 128, 128][0, 0, 512, 1]) {id = 5 : i64, metadata = @[[ID0]]} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 65536][1, 4, 128, 128][0, 128, 512, 1]) {id = 0 : i64, metadata = @[[ID1]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 16][32, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 65552][32, 8, 8, 8][2048, 32, 256, 1]) {id = 2 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 131088][32, 8, 8, 8][2048, 32, 256, 1]) {id = 3 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 196624][32, 8, 8, 8][2048, 32, 256, 1]) {id = 4 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 128, 256][1, 1, 128, 128][0, 0, 512, 1]) {id = 5 : i64, metadata = @[[ID0]]} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 65536][1, 4, 128, 128][0, 128, 512, 1]) {id = 0 : i64, metadata = @[[ID1]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 24][32, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 65560][32, 8, 8, 8][2048, 32, 256, 1]) {id = 2 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 131096][32, 8, 8, 8][2048, 32, 256, 1]) {id = 3 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 196632][32, 8, 8, 8][2048, 32, 256, 1]) {id = 4 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 128, 384][1, 1, 128, 128][0, 0, 512, 1]) {id = 5 : i64, metadata = @[[ID0]]} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 131072][1, 4, 128, 128][0, 128, 512, 1]) {id = 0 : i64, metadata = @[[ID1]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][32, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 65536][32, 8, 8, 8][2048, 32, 256, 1]) {id = 2 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 131072][32, 8, 8, 8][2048, 32, 256, 1]) {id = 3 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 196608][32, 8, 8, 8][2048, 32, 256, 1]) {id = 4 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 256, 0][1, 1, 128, 128][0, 0, 512, 1]) {id = 5 : i64, metadata = @[[ID0]]} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 131072][1, 4, 128, 128][0, 128, 512, 1]) {id = 0 : i64, metadata = @[[ID1]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 8][32, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 65544][32, 8, 8, 8][2048, 32, 256, 1]) {id = 2 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 131080][32, 8, 8, 8][2048, 32, 256, 1]) {id = 3 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 196616][32, 8, 8, 8][2048, 32, 256, 1]) {id = 4 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 256, 128][1, 1, 128, 128][0, 0, 512, 1]) {id = 5 : i64, metadata = @[[ID0]]} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 131072][1, 4, 128, 128][0, 128, 512, 1]) {id = 0 : i64, metadata = @[[ID1]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 16][32, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 65552][32, 8, 8, 8][2048, 32, 256, 1]) {id = 2 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 131088][32, 8, 8, 8][2048, 32, 256, 1]) {id = 3 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 196624][32, 8, 8, 8][2048, 32, 256, 1]) {id = 4 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 256, 256][1, 1, 128, 128][0, 0, 512, 1]) {id = 5 : i64, metadata = @[[ID0]]} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 131072][1, 4, 128, 128][0, 128, 512, 1]) {id = 0 : i64, metadata = @[[ID1]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 24][32, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 65560][32, 8, 8, 8][2048, 32, 256, 1]) {id = 2 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 131096][32, 8, 8, 8][2048, 32, 256, 1]) {id = 3 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 196632][32, 8, 8, 8][2048, 32, 256, 1]) {id = 4 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 256, 384][1, 1, 128, 128][0, 0, 512, 1]) {id = 5 : i64, metadata = @[[ID0]]} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 196608][1, 4, 128, 128][0, 128, 512, 1]) {id = 0 : i64, metadata = @[[ID1]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][32, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 65536][32, 8, 8, 8][2048, 32, 256, 1]) {id = 2 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 131072][32, 8, 8, 8][2048, 32, 256, 1]) {id = 3 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 196608][32, 8, 8, 8][2048, 32, 256, 1]) {id = 4 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 384, 0][1, 1, 128, 128][0, 0, 512, 1]) {id = 5 : i64, metadata = @[[ID0]]} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 196608][1, 4, 128, 128][0, 128, 512, 1]) {id = 0 : i64, metadata = @[[ID1]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 8][32, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 65544][32, 8, 8, 8][2048, 32, 256, 1]) {id = 2 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 131080][32, 8, 8, 8][2048, 32, 256, 1]) {id = 3 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 196616][32, 8, 8, 8][2048, 32, 256, 1]) {id = 4 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 384, 128][1, 1, 128, 128][0, 0, 512, 1]) {id = 5 : i64, metadata = @[[ID0]]} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 196608][1, 4, 128, 128][0, 128, 512, 1]) {id = 0 : i64, metadata = @[[ID1]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 16][32, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 65552][32, 8, 8, 8][2048, 32, 256, 1]) {id = 2 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 131088][32, 8, 8, 8][2048, 32, 256, 1]) {id = 3 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 196624][32, 8, 8, 8][2048, 32, 256, 1]) {id = 4 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 384, 256][1, 1, 128, 128][0, 0, 512, 1]) {id = 5 : i64, metadata = @[[ID0]]} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 196608][1, 4, 128, 128][0, 128, 512, 1]) {id = 0 : i64, metadata = @[[ID1]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 24][32, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 65560][32, 8, 8, 8][2048, 32, 256, 1]) {id = 2 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 131096][32, 8, 8, 8][2048, 32, 256, 1]) {id = 3 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 196632][32, 8, 8, 8][2048, 32, 256, 1]) {id = 4 : i64, metadata = @[[ID2]]} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 384, 384][1, 1, 128, 128][0, 0, 512, 1]) {id = 5 : i64, metadata = @[[ID0]]} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}

module {
  aie.device(npu1_4col) {
    aie.shim_dma_allocation @airMemcpyId29(S2MM, 0, 0)
    memref.global "public" @airMemcpyId29 : memref<128x128xf32, 1>
    aie.shim_dma_allocation @airMemcpyId4(MM2S, 0, 0)
    memref.global "public" @airMemcpyId4 : memref<128x256xbf16, 1>
    aie.shim_dma_allocation @airMemcpyId10(MM2S, 1, 0)
    memref.global "public" @airMemcpyId10 : memref<32x8x8x16xbf16, 1>
  } {sym_name = "forward_0"}
  airrt.module_metadata{
    airrt.segment_metadata attributes {sym_name = "forward_0"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 0 : i64, sym_name = "herd_0"}
    }
  }
  func.func @forward(%arg0: memref<512x1024xbf16>, %arg1: memref<128x8x8x64xbf16>, %arg2: memref<512x512xf32>) -> memref<512x512xf32> {
    %c384_i64 = arith.constant 384 : i64
    %c48_i64 = arith.constant 48 : i64
    %c3_i64 = arith.constant 3 : i64
    %c32_i64 = arith.constant 32 : i64
    %c2_i64 = arith.constant 2 : i64
    %c0 = arith.constant 0 : index
    %c16_i64 = arith.constant 16 : i64
    %c8_i64 = arith.constant 8 : i64
    %c512_i64 = arith.constant 512 : i64
    %c64_i64 = arith.constant 64 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c128_i64 = arith.constant 128 : i64
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i64 = arith.constant 0 : i64
    %c29_i32 = arith.constant 29 : i32
    %c10_i32 = arith.constant 10 : i32
    %c4_i32 = arith.constant 4 : i32
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    %c512 = arith.constant 512 : index
    %c64 = arith.constant 64 : index
    %p = airrt.segment_load "forward_0" : i64
    %0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c256_i64], [%c0_i64, %c256_i64, %c1024_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x1024xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c10_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c128_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId10} : (i32, i64, i64, memref<128x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %2 = airrt.dma_memcpy_nd(%c29_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId29} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
      %48 = airrt.wait_all : !airrt.event
      %49 = airrt.wait_all : !airrt.event
      %50:4 = scf.for %arg4 = %c0 to %c1024 step %c512 iter_args(%arg5 = %48, %arg6 = %49, %arg7 = %49, %arg8 = %49) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %51 = airrt.wait_all : !airrt.event
      %52 = airrt.wait_all : !airrt.event
      %53:4 = scf.for %arg4 = %c0 to %c128 step %c64 iter_args(%arg5 = %51, %arg6 = %52, %arg7 = %52, %arg8 = %52) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %54 = airrt.wait_all %50#1, %53#1 : !airrt.event
    }
    %p_0 = airrt.segment_load "forward_0" : i64
    %3 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c1_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c256_i64], [%c0_i64, %c256_i64, %c1024_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x1024xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %4 = airrt.dma_memcpy_nd(%c10_i32, %c0_i64, %c1_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c16_i64], [%c128_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId10} : (i32, i64, i64, memref<128x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %5 = airrt.dma_memcpy_nd(%c29_i32, %c0_i64, %c1_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c128_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId29} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
      %48 = airrt.wait_all : !airrt.event
      %49 = airrt.wait_all : !airrt.event
      %50:4 = scf.for %arg4 = %c0 to %c1024 step %c512 iter_args(%arg5 = %48, %arg6 = %49, %arg7 = %49, %arg8 = %49) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %51 = airrt.wait_all : !airrt.event
      %52 = airrt.wait_all : !airrt.event
      %53:4 = scf.for %arg4 = %c0 to %c128 step %c64 iter_args(%arg5 = %51, %arg6 = %52, %arg7 = %52, %arg8 = %52) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %54 = airrt.wait_all %50#1, %53#1 : !airrt.event
    }
    %p_1 = airrt.segment_load "forward_0" : i64
    %6 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c2_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c256_i64], [%c0_i64, %c256_i64, %c1024_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x1024xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %7 = airrt.dma_memcpy_nd(%c10_i32, %c0_i64, %c2_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c32_i64], [%c128_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId10} : (i32, i64, i64, memref<128x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %8 = airrt.dma_memcpy_nd(%c29_i32, %c0_i64, %c2_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c256_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId29} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
      %48 = airrt.wait_all : !airrt.event
      %49 = airrt.wait_all : !airrt.event
      %50:4 = scf.for %arg4 = %c0 to %c1024 step %c512 iter_args(%arg5 = %48, %arg6 = %49, %arg7 = %49, %arg8 = %49) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %51 = airrt.wait_all : !airrt.event
      %52 = airrt.wait_all : !airrt.event
      %53:4 = scf.for %arg4 = %c0 to %c128 step %c64 iter_args(%arg5 = %51, %arg6 = %52, %arg7 = %52, %arg8 = %52) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %54 = airrt.wait_all %50#1, %53#1 : !airrt.event
    }
    %p_2 = airrt.segment_load "forward_0" : i64
    %9 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c3_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c256_i64], [%c0_i64, %c256_i64, %c1024_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x1024xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %10 = airrt.dma_memcpy_nd(%c10_i32, %c0_i64, %c3_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c48_i64], [%c128_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId10} : (i32, i64, i64, memref<128x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %11 = airrt.dma_memcpy_nd(%c29_i32, %c0_i64, %c3_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c384_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId29} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
      %48 = airrt.wait_all : !airrt.event
      %49 = airrt.wait_all : !airrt.event
      %50:4 = scf.for %arg4 = %c0 to %c1024 step %c512 iter_args(%arg5 = %48, %arg6 = %49, %arg7 = %49, %arg8 = %49) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %51 = airrt.wait_all : !airrt.event
      %52 = airrt.wait_all : !airrt.event
      %53:4 = scf.for %arg4 = %c0 to %c128 step %c64 iter_args(%arg5 = %51, %arg6 = %52, %arg7 = %52, %arg8 = %52) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %54 = airrt.wait_all %50#1, %53#1 : !airrt.event
    }
    %p_3 = airrt.segment_load "forward_0" : i64
    %12 = airrt.dma_memcpy_nd(%c4_i32, %c1_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c128_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c256_i64], [%c0_i64, %c256_i64, %c1024_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x1024xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %13 = airrt.dma_memcpy_nd(%c10_i32, %c1_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c128_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId10} : (i32, i64, i64, memref<128x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %14 = airrt.dma_memcpy_nd(%c29_i32, %c1_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c128_i64, %c0_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId29} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
      %48 = airrt.wait_all : !airrt.event
      %49 = airrt.wait_all : !airrt.event
      %50:4 = scf.for %arg4 = %c0 to %c1024 step %c512 iter_args(%arg5 = %48, %arg6 = %49, %arg7 = %49, %arg8 = %49) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %51 = airrt.wait_all : !airrt.event
      %52 = airrt.wait_all : !airrt.event
      %53:4 = scf.for %arg4 = %c0 to %c128 step %c64 iter_args(%arg5 = %51, %arg6 = %52, %arg7 = %52, %arg8 = %52) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %54 = airrt.wait_all %50#1, %53#1 : !airrt.event
    }
    %p_4 = airrt.segment_load "forward_0" : i64
    %15 = airrt.dma_memcpy_nd(%c4_i32, %c1_i64, %c1_i64, %arg0[%c0_i64, %c0_i64, %c128_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c256_i64], [%c0_i64, %c256_i64, %c1024_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x1024xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %16 = airrt.dma_memcpy_nd(%c10_i32, %c1_i64, %c1_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c16_i64], [%c128_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId10} : (i32, i64, i64, memref<128x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %17 = airrt.dma_memcpy_nd(%c29_i32, %c1_i64, %c1_i64, %arg2[%c0_i64, %c0_i64, %c128_i64, %c128_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId29} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
      %48 = airrt.wait_all : !airrt.event
      %49 = airrt.wait_all : !airrt.event
      %50:4 = scf.for %arg4 = %c0 to %c1024 step %c512 iter_args(%arg5 = %48, %arg6 = %49, %arg7 = %49, %arg8 = %49) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %51 = airrt.wait_all : !airrt.event
      %52 = airrt.wait_all : !airrt.event
      %53:4 = scf.for %arg4 = %c0 to %c128 step %c64 iter_args(%arg5 = %51, %arg6 = %52, %arg7 = %52, %arg8 = %52) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %54 = airrt.wait_all %50#1, %53#1 : !airrt.event
    }
    %p_5 = airrt.segment_load "forward_0" : i64
    %18 = airrt.dma_memcpy_nd(%c4_i32, %c1_i64, %c2_i64, %arg0[%c0_i64, %c0_i64, %c128_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c256_i64], [%c0_i64, %c256_i64, %c1024_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x1024xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %19 = airrt.dma_memcpy_nd(%c10_i32, %c1_i64, %c2_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c32_i64], [%c128_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId10} : (i32, i64, i64, memref<128x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %20 = airrt.dma_memcpy_nd(%c29_i32, %c1_i64, %c2_i64, %arg2[%c0_i64, %c0_i64, %c128_i64, %c256_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId29} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
      %48 = airrt.wait_all : !airrt.event
      %49 = airrt.wait_all : !airrt.event
      %50:4 = scf.for %arg4 = %c0 to %c1024 step %c512 iter_args(%arg5 = %48, %arg6 = %49, %arg7 = %49, %arg8 = %49) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %51 = airrt.wait_all : !airrt.event
      %52 = airrt.wait_all : !airrt.event
      %53:4 = scf.for %arg4 = %c0 to %c128 step %c64 iter_args(%arg5 = %51, %arg6 = %52, %arg7 = %52, %arg8 = %52) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %54 = airrt.wait_all %50#1, %53#1 : !airrt.event
    }
    %p_6 = airrt.segment_load "forward_0" : i64
    %21 = airrt.dma_memcpy_nd(%c4_i32, %c1_i64, %c3_i64, %arg0[%c0_i64, %c0_i64, %c128_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c256_i64], [%c0_i64, %c256_i64, %c1024_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x1024xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %22 = airrt.dma_memcpy_nd(%c10_i32, %c1_i64, %c3_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c48_i64], [%c128_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId10} : (i32, i64, i64, memref<128x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %23 = airrt.dma_memcpy_nd(%c29_i32, %c1_i64, %c3_i64, %arg2[%c0_i64, %c0_i64, %c128_i64, %c384_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId29} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
      %48 = airrt.wait_all : !airrt.event
      %49 = airrt.wait_all : !airrt.event
      %50:4 = scf.for %arg4 = %c0 to %c1024 step %c512 iter_args(%arg5 = %48, %arg6 = %49, %arg7 = %49, %arg8 = %49) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %51 = airrt.wait_all : !airrt.event
      %52 = airrt.wait_all : !airrt.event
      %53:4 = scf.for %arg4 = %c0 to %c128 step %c64 iter_args(%arg5 = %51, %arg6 = %52, %arg7 = %52, %arg8 = %52) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %54 = airrt.wait_all %50#1, %53#1 : !airrt.event
    }
    %p_7 = airrt.segment_load "forward_0" : i64
    %24 = airrt.dma_memcpy_nd(%c4_i32, %c2_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c256_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c256_i64], [%c0_i64, %c256_i64, %c1024_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x1024xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %25 = airrt.dma_memcpy_nd(%c10_i32, %c2_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c128_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId10} : (i32, i64, i64, memref<128x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %26 = airrt.dma_memcpy_nd(%c29_i32, %c2_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c256_i64, %c0_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId29} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
      %48 = airrt.wait_all : !airrt.event
      %49 = airrt.wait_all : !airrt.event
      %50:4 = scf.for %arg4 = %c0 to %c1024 step %c512 iter_args(%arg5 = %48, %arg6 = %49, %arg7 = %49, %arg8 = %49) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %51 = airrt.wait_all : !airrt.event
      %52 = airrt.wait_all : !airrt.event
      %53:4 = scf.for %arg4 = %c0 to %c128 step %c64 iter_args(%arg5 = %51, %arg6 = %52, %arg7 = %52, %arg8 = %52) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %54 = airrt.wait_all %50#1, %53#1 : !airrt.event
    }
    %p_8 = airrt.segment_load "forward_0" : i64
    %27 = airrt.dma_memcpy_nd(%c4_i32, %c2_i64, %c1_i64, %arg0[%c0_i64, %c0_i64, %c256_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c256_i64], [%c0_i64, %c256_i64, %c1024_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x1024xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %28 = airrt.dma_memcpy_nd(%c10_i32, %c2_i64, %c1_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c16_i64], [%c128_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId10} : (i32, i64, i64, memref<128x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %29 = airrt.dma_memcpy_nd(%c29_i32, %c2_i64, %c1_i64, %arg2[%c0_i64, %c0_i64, %c256_i64, %c128_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId29} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
      %48 = airrt.wait_all : !airrt.event
      %49 = airrt.wait_all : !airrt.event
      %50:4 = scf.for %arg4 = %c0 to %c1024 step %c512 iter_args(%arg5 = %48, %arg6 = %49, %arg7 = %49, %arg8 = %49) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %51 = airrt.wait_all : !airrt.event
      %52 = airrt.wait_all : !airrt.event
      %53:4 = scf.for %arg4 = %c0 to %c128 step %c64 iter_args(%arg5 = %51, %arg6 = %52, %arg7 = %52, %arg8 = %52) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %54 = airrt.wait_all %50#1, %53#1 : !airrt.event
    }
    %p_9 = airrt.segment_load "forward_0" : i64
    %30 = airrt.dma_memcpy_nd(%c4_i32, %c2_i64, %c2_i64, %arg0[%c0_i64, %c0_i64, %c256_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c256_i64], [%c0_i64, %c256_i64, %c1024_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x1024xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %31 = airrt.dma_memcpy_nd(%c10_i32, %c2_i64, %c2_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c32_i64], [%c128_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId10} : (i32, i64, i64, memref<128x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %32 = airrt.dma_memcpy_nd(%c29_i32, %c2_i64, %c2_i64, %arg2[%c0_i64, %c0_i64, %c256_i64, %c256_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId29} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
      %48 = airrt.wait_all : !airrt.event
      %49 = airrt.wait_all : !airrt.event
      %50:4 = scf.for %arg4 = %c0 to %c1024 step %c512 iter_args(%arg5 = %48, %arg6 = %49, %arg7 = %49, %arg8 = %49) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %51 = airrt.wait_all : !airrt.event
      %52 = airrt.wait_all : !airrt.event
      %53:4 = scf.for %arg4 = %c0 to %c128 step %c64 iter_args(%arg5 = %51, %arg6 = %52, %arg7 = %52, %arg8 = %52) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %54 = airrt.wait_all %50#1, %53#1 : !airrt.event
    }
    %p_10 = airrt.segment_load "forward_0" : i64
    %33 = airrt.dma_memcpy_nd(%c4_i32, %c2_i64, %c3_i64, %arg0[%c0_i64, %c0_i64, %c256_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c256_i64], [%c0_i64, %c256_i64, %c1024_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x1024xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %34 = airrt.dma_memcpy_nd(%c10_i32, %c2_i64, %c3_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c48_i64], [%c128_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId10} : (i32, i64, i64, memref<128x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %35 = airrt.dma_memcpy_nd(%c29_i32, %c2_i64, %c3_i64, %arg2[%c0_i64, %c0_i64, %c256_i64, %c384_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId29} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
      %48 = airrt.wait_all : !airrt.event
      %49 = airrt.wait_all : !airrt.event
      %50:4 = scf.for %arg4 = %c0 to %c1024 step %c512 iter_args(%arg5 = %48, %arg6 = %49, %arg7 = %49, %arg8 = %49) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %51 = airrt.wait_all : !airrt.event
      %52 = airrt.wait_all : !airrt.event
      %53:4 = scf.for %arg4 = %c0 to %c128 step %c64 iter_args(%arg5 = %51, %arg6 = %52, %arg7 = %52, %arg8 = %52) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %54 = airrt.wait_all %50#1, %53#1 : !airrt.event
    }
    %p_11 = airrt.segment_load "forward_0" : i64
    %36 = airrt.dma_memcpy_nd(%c4_i32, %c3_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c384_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c256_i64], [%c0_i64, %c256_i64, %c1024_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x1024xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %37 = airrt.dma_memcpy_nd(%c10_i32, %c3_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c128_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId10} : (i32, i64, i64, memref<128x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %38 = airrt.dma_memcpy_nd(%c29_i32, %c3_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c384_i64, %c0_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId29} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
      %48 = airrt.wait_all : !airrt.event
      %49 = airrt.wait_all : !airrt.event
      %50:4 = scf.for %arg4 = %c0 to %c1024 step %c512 iter_args(%arg5 = %48, %arg6 = %49, %arg7 = %49, %arg8 = %49) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %51 = airrt.wait_all : !airrt.event
      %52 = airrt.wait_all : !airrt.event
      %53:4 = scf.for %arg4 = %c0 to %c128 step %c64 iter_args(%arg5 = %51, %arg6 = %52, %arg7 = %52, %arg8 = %52) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %54 = airrt.wait_all %50#1, %53#1 : !airrt.event
    }
    %p_12 = airrt.segment_load "forward_0" : i64
    %39 = airrt.dma_memcpy_nd(%c4_i32, %c3_i64, %c1_i64, %arg0[%c0_i64, %c0_i64, %c384_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c256_i64], [%c0_i64, %c256_i64, %c1024_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x1024xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %40 = airrt.dma_memcpy_nd(%c10_i32, %c3_i64, %c1_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c16_i64], [%c128_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId10} : (i32, i64, i64, memref<128x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %41 = airrt.dma_memcpy_nd(%c29_i32, %c3_i64, %c1_i64, %arg2[%c0_i64, %c0_i64, %c384_i64, %c128_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId29} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
      %48 = airrt.wait_all : !airrt.event
      %49 = airrt.wait_all : !airrt.event
      %50:4 = scf.for %arg4 = %c0 to %c1024 step %c512 iter_args(%arg5 = %48, %arg6 = %49, %arg7 = %49, %arg8 = %49) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %51 = airrt.wait_all : !airrt.event
      %52 = airrt.wait_all : !airrt.event
      %53:4 = scf.for %arg4 = %c0 to %c128 step %c64 iter_args(%arg5 = %51, %arg6 = %52, %arg7 = %52, %arg8 = %52) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %54 = airrt.wait_all %50#1, %53#1 : !airrt.event
    }
    %p_13 = airrt.segment_load "forward_0" : i64
    %42 = airrt.dma_memcpy_nd(%c4_i32, %c3_i64, %c2_i64, %arg0[%c0_i64, %c0_i64, %c384_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c256_i64], [%c0_i64, %c256_i64, %c1024_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x1024xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %43 = airrt.dma_memcpy_nd(%c10_i32, %c3_i64, %c2_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c32_i64], [%c128_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId10} : (i32, i64, i64, memref<128x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %44 = airrt.dma_memcpy_nd(%c29_i32, %c3_i64, %c2_i64, %arg2[%c0_i64, %c0_i64, %c384_i64, %c256_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId29} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
      %48 = airrt.wait_all : !airrt.event
      %49 = airrt.wait_all : !airrt.event
      %50:4 = scf.for %arg4 = %c0 to %c1024 step %c512 iter_args(%arg5 = %48, %arg6 = %49, %arg7 = %49, %arg8 = %49) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %51 = airrt.wait_all : !airrt.event
      %52 = airrt.wait_all : !airrt.event
      %53:4 = scf.for %arg4 = %c0 to %c128 step %c64 iter_args(%arg5 = %51, %arg6 = %52, %arg7 = %52, %arg8 = %52) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %54 = airrt.wait_all %50#1, %53#1 : !airrt.event
    }
    %p_14 = airrt.segment_load "forward_0" : i64
    %45 = airrt.dma_memcpy_nd(%c4_i32, %c3_i64, %c3_i64, %arg0[%c0_i64, %c0_i64, %c384_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c256_i64], [%c0_i64, %c256_i64, %c1024_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x1024xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %46 = airrt.dma_memcpy_nd(%c10_i32, %c3_i64, %c3_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c48_i64], [%c128_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId10} : (i32, i64, i64, memref<128x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %47 = airrt.dma_memcpy_nd(%c29_i32, %c3_i64, %c3_i64, %arg2[%c0_i64, %c0_i64, %c384_i64, %c384_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId29} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
      %48 = airrt.wait_all : !airrt.event
      %49 = airrt.wait_all : !airrt.event
      %50:4 = scf.for %arg4 = %c0 to %c1024 step %c512 iter_args(%arg5 = %48, %arg6 = %49, %arg7 = %49, %arg8 = %49) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %51 = airrt.wait_all : !airrt.event
      %52 = airrt.wait_all : !airrt.event
      %53:4 = scf.for %arg4 = %c0 to %c128 step %c64 iter_args(%arg5 = %51, %arg6 = %52, %arg7 = %52, %arg8 = %52) -> (!airrt.event, !airrt.event, !airrt.event, !airrt.event) {
        %55 = airrt.wait_all %arg8, %arg5 : !airrt.event
        %56 = airrt.wait_all %arg7 : !airrt.event
        %57 = airrt.wait_all %arg7 : !airrt.event
        airrt.wait_all %arg8, %arg5
        %58 = airrt.wait_all : !airrt.event
        %59 = airrt.wait_all %arg6 : !airrt.event
        airrt.wait_all %arg6
        %60 = airrt.wait_all : !airrt.event
        scf.yield %58, %60, %60, %59 : !airrt.event, !airrt.event, !airrt.event, !airrt.event
      }
      %54 = airrt.wait_all %50#1, %53#1 : !airrt.event
    }
    return %arg2 : memref<512x512xf32>
  }
}
