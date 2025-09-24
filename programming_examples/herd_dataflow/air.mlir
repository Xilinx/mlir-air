
#map = affine_map<()[s0] -> (s0 * 96)>
module {
  // External function for the final compute stage
  func.func private @add_3_bf16(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "extern_func.o", llvm.emit_c_interface}

  // AIR channels model hardware FIFOs for inter-stage communication
  air.channel @L2ToL1Chan1 [4, 1]         // L2 to L1, input A; default channel_type is "dma_stream", representing data movement performed using DMA streaming interconnects
  air.channel @L2ToL1Chan2 [4, 1]         // L2 to L1, input B
  air.channel @L1ToL1Chan1 [4, 1]         // Between herd_0 and herd_1
  air.channel @L1ToL1Chan2 [4, 1] {channel_type = "cascade"} // Between herd_1 and herd_2; channel_type="cascade" means this channel is expected to map to cascade connections (peer-to-peer communication between compute tiles)
  air.channel @L1ToL2Chan1 [4, 1]         // Output from herd_2 to L2

  // Top-level function: runtime dispatch over a 4x1 iteration space (not necessarily hardware parallelism)
  func.func @func1(%arg0: memref<256x256xbf16>, %arg1: memref<256x256xbf16>, %arg2: memref<256x256xbf16>) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    // air.launch: runtime dispatch over a 4x1 iteration space (may be sequential or parallel depending on runtime)
    air.launch (%arg5, %arg6) in (%arg7=%c4, %arg8=%c1) args(%arg9=%arg0, %arg10=%arg1, %arg11=%arg2) : memref<256x256xbf16>, memref<256x256xbf16>, memref<256x256xbf16> {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c4_0 = arith.constant 4 : index
      %c64 = arith.constant 64 : index
      %c256 = arith.constant 256 : index
      // Each segment is a program mapped to a hardware tile of resources, including the L1 and L2 memories and compute cores
      air.segment @seg args(%arg51=%arg5, %arg61=%arg6, %arg91=%arg9, %arg101=%arg10, %arg111=%arg11) : index, index, memref<256x256xbf16>, memref<256x256xbf16>, memref<256x256xbf16> {
        // Allocate L2 buffers for tile-local computation (memory space 1 = L2)
        %alloc_1 = memref.alloc() : memref<64x256xbf16, 1 : i32>
        %alloc_2 = memref.alloc() : memref<64x256xbf16, 1 : i32>
        %alloc_3 = memref.alloc() : memref<64x256xbf16, 1 : i32>

        %c0_2 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c2_1 = arith.constant 2 : index
        %c3_1 = arith.constant 3 : index
        %c4_1 = arith.constant 4 : index
        %c64_1 = arith.constant 64 : index
        %c256_1 = arith.constant 256 : index

        %pid_x_offset = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%arg51]
        // Stage 1: DMA from L2 to L1 for both inputs, in parallel for each sub-tile
        // This parallel loop partitions the DMA transfers so that each iteration moves a unique tile's data from L2 to its corresponding L1 buffer.
        // air.dma_memcpy_nd represents a full memcpy, expected to be mapped to hardware DMAs, specifying both source and destination.
        scf.parallel (%par1) = (%c0_2) to (%c4_1) step (%c1_1) {
          %apply = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%par1]
          air.dma_memcpy_nd (%alloc_1[%c0_2, %apply] [%c64_1, %c64_1] [%c256_1, %c1_1], %arg91[%pid_x_offset, %apply] [%c64_1, %c64_1] [%c256_1, %c1_1]) : (memref<64x256xbf16, 1 : i32>, memref<256x256xbf16>)
          air.dma_memcpy_nd (%alloc_2[%c0_2, %apply] [%c64_1, %c64_1] [%c256_1, %c1_1], %arg101[%pid_x_offset, %apply] [%c64_1, %c64_1] [%c256_1, %c1_1]) : (memref<64x256xbf16, 1 : i32>, memref<256x256xbf16>)
        }
        // Stage 2: Send L1 buffers to next stage via AIR channels
        // This parallel loop partitions the channel put operations, so each iteration sends a specific tile's L1 buffer to its corresponding downstream channel.
        // air.channel.put represents a "half DMA" operation, expected to be mapped to hardware DMAs but only specifies the source (put) end.
        scf.parallel (%par1) = (%c0_2) to (%c4_1) step (%c1_1) {
          %apply = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%par1]
          air.channel.put  @L2ToL1Chan1[%par1, %c0_2] (%alloc_1[%c0_2, %apply] [%c64_1, %c64_1] [%c256_1, %c1_1]) : (memref<64x256xbf16, 1 : i32>)
          air.channel.put  @L2ToL1Chan2[%par1, %c0_2] (%alloc_2[%c0_2, %apply] [%c64_1, %c64_1] [%c256_1, %c1_1]) : (memref<64x256xbf16, 1 : i32>)
        }
        
        // Stage 3: First compute herd (herd_0): 4x1 shape. Each of the 4 tiles asynchronously receives two input tiles via channels, performs vector addition, and sends the result to the next stage.
        air.herd @herd_0  tile (%arg22, %arg23) in (%arg24=%c4_1, %arg25=%c1_1) {
          %alloc_a = memref.alloc() : memref<64x64xbf16, 2 : i32> // L1 memory (memory space 2)
          %alloc_b = memref.alloc() : memref<64x64xbf16, 2 : i32> // L1 memory (memory space 2)
          %alloc_c = memref.alloc() : memref<64x64xbf16, 2 : i32> // L1 memory (memory space 2)
          // Receive input tiles from previous stage via channels
          air.channel.get  @L2ToL1Chan1[%arg22, %arg23] (%alloc_a[] [] []) : (memref<64x64xbf16, 2 : i32>)
          air.channel.get  @L2ToL1Chan2[%arg22, %arg23] (%alloc_b[] [] []) : (memref<64x64xbf16, 2 : i32>)
          %c0_3 = arith.constant 0 : index
          %c1_3 = arith.constant 1 : index
          %c64_3 = arith.constant 64 : index
          %c16_3 = arith.constant 16 : index
          
          // Vectorized tile-wise addition: C = A + B
          // This is the first form of kernel coding in this example: MLIR vector dialect is used to represent vector loads, stores, and computation.
          // The vectorized code here maps directly to hardware vector intrinsics.
          scf.for %arg29 = %c0_3 to %c64_3 step %c1_3 {
            %subview = memref.subview %alloc_a[%arg29, 0] [1, 64] [1, 1] : memref<64x64xbf16, 2 : i32> to memref<1x64xbf16, strided<[64, 1], offset: ?>, 2 : i32>
            %subview_10 = memref.subview %alloc_b[%arg29, 0] [1, 64] [1, 1] : memref<64x64xbf16, 2 : i32> to memref<1x64xbf16, strided<[64, 1], offset: ?>, 2 : i32>
            %subview_11 = memref.subview %alloc_c[%arg29, 0] [1, 64] [1, 1] : memref<64x64xbf16, 2 : i32> to memref<1x64xbf16, strided<[64, 1], offset: ?>, 2 : i32>
            scf.for %arg30 = %c0_3 to %c64_3 step %c16_3 {
              %subview_12 = memref.subview %subview[0, %arg30] [1, 16] [1, 1] : memref<1x64xbf16, strided<[64, 1], offset: ?>, 2 : i32> to memref<1x16xbf16, strided<[64, 1], offset: ?>, 2 : i32>
              %subview_13 = memref.subview %subview_10[0, %arg30] [1, 16] [1, 1] : memref<1x64xbf16, strided<[64, 1], offset: ?>, 2 : i32> to memref<1x16xbf16, strided<[64, 1], offset: ?>, 2 : i32>
              %subview_14 = memref.subview %subview_11[0, %arg30] [1, 16] [1, 1] : memref<1x64xbf16, strided<[64, 1], offset: ?>, 2 : i32> to memref<1x16xbf16, strided<[64, 1], offset: ?>, 2 : i32>
              %collapse_shape_a = memref.collapse_shape %subview_12 [[0, 1]] : memref<1x16xbf16, strided<[64, 1], offset: ?>, 2 : i32> into memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
              %collapse_shape_b = memref.collapse_shape %subview_13 [[0, 1]] : memref<1x16xbf16, strided<[64, 1], offset: ?>, 2 : i32> into memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
              %collapse_shape_c = memref.collapse_shape %subview_14 [[0, 1]] : memref<1x16xbf16, strided<[64, 1], offset: ?>, 2 : i32> into memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
              %poison = ub.poison : bf16
              %3 = vector.transfer_read %collapse_shape_a[%c0_3], %poison {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
              %4 = vector.transfer_read %collapse_shape_b[%c0_3], %poison {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2 : i32>, vector<16xbf16>
              %5 = arith.addf %3, %4 : vector<16xbf16>
              vector.transfer_write %5, %collapse_shape_c[%c0_3] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, strided<[1], offset: ?>, 2 : i32>
            }
          }
          // Send result to next herd via channel
          air.channel.put  @L1ToL1Chan1[%arg22, %arg23] (%alloc_c[] [] []) : (memref<64x64xbf16, 2 : i32>)
        }
        // Stage 4: Second herd (herd_1): 4x1 shape. Each of the 4 tiles asynchronously receives a tile via channel, copies its contents, and sends it to the next stage.
        air.herd @herd_1  tile (%arg22, %arg23) in (%arg24=%c4_1, %arg25=%c1_1) {
          %alloc_a = memref.alloc() : memref<64x64xbf16, 2 : i32> // L1 memory (memory space 2)
          %alloc_c = memref.alloc() : memref<64x64xbf16, 2 : i32> // L1 memory (memory space 2)
          // Receive from previous herd via channel
          air.channel.get  @L1ToL1Chan1[%arg22, %arg23] (%alloc_a[] [] []) : (memref<64x64xbf16, 2 : i32>)
          %c0_3 = arith.constant 0 : index
          %c1_3 = arith.constant 1 : index
          %c64_3 = arith.constant 64 : index
          // Second form of kernel writing: using the memref dialect to represent scalar load and store of individual data.
          // This maps to scalar memory operations on the hardware.
          scf.for %arg29 = %c0_3 to %c64_3 step %c1_3 {
            scf.for %arg30 = %c0_3 to %c64_3 step %c1_3 {
              %0 = memref.load %alloc_a[%arg29, %arg30] : memref<64x64xbf16, 2 : i32>
              memref.store %0, %alloc_c[%arg29, %arg30] : memref<64x64xbf16, 2 : i32>
            }
          }
          // Send to next herd via channel (cascade type)
          // This channel.put sends the output of herd_1 to herd_2 using the cascade interconnect, enabling direct peer-to-peer transfer between adjacent tiles.
          air.channel.put  @L1ToL1Chan2[%arg22, %arg23] (%alloc_c[] [] []) : (memref<64x64xbf16, 2 : i32>)
        }
        // Stage 5: Third herd (herd_2): 4x1 shape. Each of the 4 tiles asynchronously receives a tile via channel, calls an external function for further computation, and sends the result to the output channel.
        air.herd @herd_2  tile (%arg22, %arg23) in (%arg24=%c4_1, %arg25=%c1_1) attributes {link_with = "extern_func.o"} {
          %alloc_a = memref.alloc() : memref<64x64xbf16, 2 : i32> // L1 memory (memory space 2)
          %alloc_c = memref.alloc() : memref<64x64xbf16, 2 : i32> // L1 memory (memory space 2)
          // Receive from previous herd via channel
          // This channel.get receives input from herd_1 via the cascade interconnect, providing low-latency data movement into herd_2 from its neighbor tile.
          air.channel.get  @L1ToL1Chan2[%arg22, %arg23] (%alloc_a[] [] []) : (memref<64x64xbf16, 2 : i32>)
          %c0_3 = arith.constant 0 : index
          %c1_3 = arith.constant 1 : index
          %c64_3 = arith.constant 64 : index
          // Third form of kernel coding: function call to an external kernel.
          // The link_with attribute in the herd specifies which object file to link, containing the implementation of the function.
          func.call @add_3_bf16(%alloc_a, %alloc_c) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          // Send result back to L2 via channel
          air.channel.put  @L1ToL2Chan1[%arg22, %arg23] (%alloc_c[] [] []) : (memref<64x64xbf16, 2 : i32>)
        }
        // Stage 6: Gather results from all tiles and DMA back to L2
        // This parallel loop partitions the result collection and DMA, so each iteration gathers a tile's output from its channel and writes it back to its region in L2.
        scf.parallel (%par1) = (%c0_2) to (%c4_1) step (%c1_1) {
          %apply = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%par1]
          // air.channel.get represents a "half DMA" operation, expected to be mapped to hardware DMAs but only specifies the destination (get) end.
          air.channel.get  @L1ToL2Chan1[%par1, %c0_2] (%alloc_3[%c0_2, %apply] [%c64_1, %c64_1] [%c256_1, %c1_1]) : (memref<64x256xbf16, 1 : i32>) // L2 memory (memory space 1)
          air.dma_memcpy_nd (%arg111[%pid_x_offset, %apply] [%c64_1, %c64_1] [%c256_1, %c1_1], %alloc_3[%c0_2, %apply] [%c64_1, %c64_1] [%c256_1, %c1_1]) : (memref<256x256xbf16>, memref<64x256xbf16, 1 : i32>)
        }
      }
    }
    return
  }
}
