// RUN: air-opt %s -air-copy-to-dma | FileCheck %s

// CHECK-LABEL: func.func @pack_2d
// CHECK-NOT: linalg.pack
// CHECK: air.dma_memcpy_nd
func.func @pack_2d(%src: memref<256x64xbf16>, %dst: memref<4x1x64x64xbf16, 1>) {
  linalg.pack %src outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %dst : memref<256x64xbf16> -> memref<4x1x64x64xbf16, 1>
  return
}

// CHECK-LABEL: func.func @pack_transposed
// CHECK-NOT: linalg.pack
// CHECK: air.dma_memcpy_nd
func.func @pack_transposed(%src: memref<64x256xbf16>, %dst: memref<4x1x64x64xbf16, 1>) {
  linalg.pack %src outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %dst : memref<64x256xbf16> -> memref<4x1x64x64xbf16, 1>
  return
}

// CHECK-LABEL: func.func @pack_4d
// CHECK-NOT: linalg.pack
// CHECK: air.dma_memcpy_nd
func.func @pack_4d(%src: memref<1x1x64x64xbf16, 1>, %dst: memref<1x1x8x8x8x8xbf16, 2>) {
  linalg.pack %src outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 8] into %dst : memref<1x1x64x64xbf16, 1> -> memref<1x1x8x8x8x8xbf16, 2>
  return
}
