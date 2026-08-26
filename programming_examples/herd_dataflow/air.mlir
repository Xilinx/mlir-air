#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0, s1] -> (s0 * 256 + s1 * 64)>
#map2 = affine_map<()[s0] -> (s0)>
module {
  func.func private @add_3_bf16(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "extern_func.o", llvm.emit_c_interface}
  air.channel @L2ToL1Chan1 [4, 1]
  air.channel @L2ToL1Chan2 [4, 1]
  air.channel @L1ToL1Chan1 [4, 1]
  air.channel @L1ToL1Chan2 [4, 1] {channel_type = "npu_cascade"}
  air.channel @L1ToL2Chan1 [4, 1]
  func.func @func1(%arg0: memref<256x256xbf16>, %arg1: memref<256x256xbf16>, %arg2: memref<256x256xbf16>) {
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    air.launch (%arg3, %arg4) in (%arg5=%c4, %arg6=%c1) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<256x256xbf16>, memref<256x256xbf16>, memref<256x256xbf16> {
      air.segment @segment_0  args(%arg10=%arg3, %arg11=%arg4, %arg12=%arg7, %arg13=%arg8, %arg14=%arg9) : index, index, memref<256x256xbf16>, memref<256x256xbf16>, memref<256x256xbf16> {
        %alloc = memref.alloc() : memref<64x256xbf16, 1 : i32>
        %alloc_0 = memref.alloc() : memref<64x256xbf16, 1 : i32>
        %alloc_1 = memref.alloc() : memref<64x256xbf16, 1 : i32>
        scf.forall (%arg15) in (4) {
          %0 = affine.apply #map()[%arg15]
          %1 = affine.apply #map()[%arg10]
          %2 = affine.apply #map1()[%arg11, %arg15]
          air.dma_memcpy_nd (%alloc[0, %0] [64, 64] [256, 1], %arg12[%1, %2] [64, 64] [256, 1]) : (memref<64x256xbf16, 1 : i32>, memref<256x256xbf16>)
          %3 = affine.apply #map()[%arg15]
          %4 = affine.apply #map()[%arg10]
          %5 = affine.apply #map1()[%arg11, %arg15]
          air.dma_memcpy_nd (%alloc_0[0, %3] [64, 64] [256, 1], %arg13[%4, %5] [64, 64] [256, 1]) : (memref<64x256xbf16, 1 : i32>, memref<256x256xbf16>)
        }
        scf.forall (%arg15) in (4) {
          %0 = affine.apply #map()[%arg15]
          %1 = affine.apply #map2()[%arg15]
          %c0 = arith.constant 0 : index
          air.channel.put  @L2ToL1Chan1[%1, %c0] (%alloc[0, %0] [64, 64] [256, 1]) : (memref<64x256xbf16, 1 : i32>)
          %2 = affine.apply #map()[%arg15]
          %3 = affine.apply #map2()[%arg15]
          %c0_8 = arith.constant 0 : index
          air.channel.put  @L2ToL1Chan2[%3, %c0_8] (%alloc_0[0, %2] [64, 64] [256, 1]) : (memref<64x256xbf16, 1 : i32>)
        }
        %c4_2 = arith.constant 4 : index
        %c1_3 = arith.constant 1 : index
        air.herd @herd_0  tile (%arg15, %arg16) in (%arg17=%c4_2, %arg18=%c1_3) args(%arg19=%arg10, %arg20=%arg11, %arg21=%arg12, %arg22=%arg13, %arg23=%arg14, %arg24=%alloc, %arg25=%alloc_0, %arg26=%alloc_1) : index, index, memref<256x256xbf16>, memref<256x256xbf16>, memref<256x256xbf16>, memref<64x256xbf16, 1 : i32>, memref<64x256xbf16, 1 : i32>, memref<64x256xbf16, 1 : i32> {
          %alloc_8 = memref.alloc() : memref<64x64xbf16, 2 : i32>
          %alloc_9 = memref.alloc() : memref<64x64xbf16, 2 : i32>
          %alloc_10 = memref.alloc() : memref<64x64xbf16, 2 : i32>
          %0 = affine.apply #map2()[%arg15]
          %c0 = arith.constant 0 : index
          air.channel.get  @L2ToL1Chan1[%0, %c0] (%alloc_8[] [] []) : (memref<64x64xbf16, 2 : i32>)
          %1 = affine.apply #map2()[%arg15]
          %c0_11 = arith.constant 0 : index
          air.channel.get  @L2ToL1Chan2[%1, %c0_11] (%alloc_9[] [] []) : (memref<64x64xbf16, 2 : i32>)
          %cst = arith.constant 0.000000e+00 : bf16
          %c0_12 = arith.constant 0 : index
          %c64 = arith.constant 64 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg27 = %c0_12 to %c64 step %c1_13 {
            %c0_15 = arith.constant 0 : index
            %c64_16 = arith.constant 64 : index
            %c16 = arith.constant 16 : index
            scf.for %arg28 = %c0_15 to %c64_16 step %c16 {
              %3 = vector.transfer_read %alloc_8[%arg27, %arg28], %cst {in_bounds = [true]} : memref<64x64xbf16, 2 : i32>, vector<16xbf16>
              %4 = vector.transfer_read %alloc_9[%arg27, %arg28], %cst {in_bounds = [true]} : memref<64x64xbf16, 2 : i32>, vector<16xbf16>
              %5 = arith.addf %3, %4 : vector<16xbf16>
              vector.transfer_write %5, %alloc_10[%arg27, %arg28] {in_bounds = [true]} : vector<16xbf16>, memref<64x64xbf16, 2 : i32>
            }
          }
          memref.dealloc %alloc_9 : memref<64x64xbf16, 2 : i32>
          memref.dealloc %alloc_8 : memref<64x64xbf16, 2 : i32>
          %2 = affine.apply #map2()[%arg15]
          %c0_14 = arith.constant 0 : index
          air.channel.put  @L1ToL1Chan1[%2, %c0_14] (%alloc_10[] [] []) : (memref<64x64xbf16, 2 : i32>)
          memref.dealloc %alloc_10 : memref<64x64xbf16, 2 : i32>
        }
        %c4_4 = arith.constant 4 : index
        %c1_5 = arith.constant 1 : index
        air.herd @herd_1  tile (%arg15, %arg16) in (%arg17=%c4_4, %arg18=%c1_5) args(%arg19=%arg10, %arg20=%arg11, %arg21=%arg12, %arg22=%arg13, %arg23=%arg14, %arg24=%alloc, %arg25=%alloc_0, %arg26=%alloc_1) : index, index, memref<256x256xbf16>, memref<256x256xbf16>, memref<256x256xbf16>, memref<64x256xbf16, 1 : i32>, memref<64x256xbf16, 1 : i32>, memref<64x256xbf16, 1 : i32> {
          %alloc_8 = memref.alloc() : memref<64x64xbf16, 2 : i32>
          %alloc_9 = memref.alloc() : memref<64x64xbf16, 2 : i32>
          %0 = affine.apply #map2()[%arg15]
          %c0 = arith.constant 0 : index
          air.channel.get  @L1ToL1Chan1[%0, %c0] (%alloc_8[] [] []) : (memref<64x64xbf16, 2 : i32>)
          %cst = arith.constant 0.000000e+00 : bf16
          %c0_10 = arith.constant 0 : index
          %c64 = arith.constant 64 : index
          %c1_11 = arith.constant 1 : index
          scf.for %arg27 = %c0_10 to %c64 step %c1_11 {
            %c0_13 = arith.constant 0 : index
            %c64_14 = arith.constant 64 : index
            %c16 = arith.constant 16 : index
            scf.for %arg28 = %c0_13 to %c64_14 step %c16 {
              %2 = vector.transfer_read %alloc_8[%arg27, %arg28], %cst {in_bounds = [true]} : memref<64x64xbf16, 2 : i32>, vector<16xbf16>
              vector.transfer_write %2, %alloc_9[%arg27, %arg28] {in_bounds = [true]} : vector<16xbf16>, memref<64x64xbf16, 2 : i32>
            }
          }
          memref.dealloc %alloc_8 : memref<64x64xbf16, 2 : i32>
          %1 = affine.apply #map2()[%arg15]
          %c0_12 = arith.constant 0 : index
          air.channel.put  @L1ToL1Chan2[%1, %c0_12] (%alloc_9[] [] []) : (memref<64x64xbf16, 2 : i32>)
          memref.dealloc %alloc_9 : memref<64x64xbf16, 2 : i32>
        }
        %c4_6 = arith.constant 4 : index
        %c1_7 = arith.constant 1 : index
        air.herd @herd_2  tile (%arg15, %arg16) in (%arg17=%c4_6, %arg18=%c1_7) args(%arg19=%arg10, %arg20=%arg11, %arg21=%arg12, %arg22=%arg13, %arg23=%arg14, %arg24=%alloc, %arg25=%alloc_0, %arg26=%alloc_1) : index, index, memref<256x256xbf16>, memref<256x256xbf16>, memref<256x256xbf16>, memref<64x256xbf16, 1 : i32>, memref<64x256xbf16, 1 : i32>, memref<64x256xbf16, 1 : i32> attributes {link_with = "extern_func.o"} {
          %alloc_8 = memref.alloc() : memref<64x64xbf16, 2 : i32>
          %alloc_9 = memref.alloc() : memref<64x64xbf16, 2 : i32>
          %0 = affine.apply #map2()[%arg15]
          %c0 = arith.constant 0 : index
          air.channel.get  @L1ToL1Chan2[%0, %c0] (%alloc_8[] [] []) : (memref<64x64xbf16, 2 : i32>)
          func.call @add_3_bf16(%alloc_8, %alloc_9) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          memref.dealloc %alloc_8 : memref<64x64xbf16, 2 : i32>
          %1 = affine.apply #map2()[%arg15]
          %c0_10 = arith.constant 0 : index
          air.channel.put  @L1ToL2Chan1[%1, %c0_10] (%alloc_9[] [] []) : (memref<64x64xbf16, 2 : i32>)
          memref.dealloc %alloc_9 : memref<64x64xbf16, 2 : i32>
        }
        memref.dealloc %alloc_0 : memref<64x256xbf16, 1 : i32>
        memref.dealloc %alloc : memref<64x256xbf16, 1 : i32>
        scf.forall (%arg15) in (4) {
          %0 = affine.apply #map()[%arg15]
          %1 = affine.apply #map2()[%arg15]
          %c0 = arith.constant 0 : index
          air.channel.get  @L1ToL2Chan1[%1, %c0] (%alloc_1[0, %0] [64, 64] [256, 1]) : (memref<64x256xbf16, 1 : i32>)
          %2 = affine.apply #map()[%arg15]
          %3 = affine.apply #map()[%arg10]
          %4 = affine.apply #map1()[%arg11, %arg15]
          air.dma_memcpy_nd (%arg14[%3, %4] [64, 64] [256, 1], %alloc_1[0, %2] [64, 64] [256, 1]) : (memref<256x256xbf16>, memref<64x256xbf16, 1 : i32>)
        }
        memref.dealloc %alloc_1 : memref<64x256xbf16, 1 : i32>
      }
    }
    return
  }
}

