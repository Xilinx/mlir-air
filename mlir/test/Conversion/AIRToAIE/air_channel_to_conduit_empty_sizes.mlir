// RUN: air-opt --air-channel-to-conduit %s | FileCheck %s
//
// PassB-empty-sizes regression test: when air.channel.put/get has no
// offsets/sizes/strides (nsizes==0), num_elems should be inferred from the
// memref type's total element count, not default to 1.
//
// A memref<16x32xi32> has 512 elements, so num_elems must be 512.
// Phase 2b also infers num_elems from the memref type.

// CHECK-LABEL: module
//
// CHECK:   conduit.create @chan
// CHECK-SAME: depth = 0
// CHECK-SAME: element_type = memref<16x32xi32>
//
// --- put: num_elems = 16*32 = 512 (NOT 1) ---
// CHECK:   %[[TOK0:.*]] = conduit.put_memref_async
// CHECK-SAME: name = @chan
// CHECK-SAME: num_elems = 512
// CHECK-SAME: : !conduit.dma.token
//
// --- get: num_elems = 512 ---
// CHECK:   %[[TOK1:.*]] = conduit.get_memref_async
// CHECK-SAME: name = @chan
// CHECK-SAME: num_elems = 512
// CHECK-SAME: : !conduit.dma.token
//
// CHECK-NOT: air.channel

module {
  aie.device(xcve2802) {
    %tile_0_3 = aie.tile(0, 3)

    "air.channel"() {sym_name = "chan", size = [1, 1]} : () -> ()

    aie.core(%tile_0_3) {
      %src = memref.alloca() : memref<16x32xi32>
      %dst = memref.alloca() : memref<16x32xi32>

      // air.channel.put with NO offsets/sizes/strides — full buffer transfer.
      %tok0 = "air.channel.put"(%src)
          {chan_name = @chan,
           operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>}
          : (memref<16x32xi32>)
          -> !air.async.token

      // air.channel.get with NO offsets/sizes/strides — full buffer transfer.
      %tok1 = "air.channel.get"(%dst)
          {chan_name = @chan,
           operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>}
          : (memref<16x32xi32>)
          -> !air.async.token

      aie.end
    }
  }
}
