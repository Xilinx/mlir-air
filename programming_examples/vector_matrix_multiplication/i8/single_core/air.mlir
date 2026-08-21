#map = affine_map<()[s0] -> (s0 * 48)>
#map1 = affine_map<()[s0] -> (s0 * 96)>
module {
  func.func private @vecmat_i8_i32(memref<6x16xi8, 2 : i32>, memref<6x6x16x8xi8, 2 : i32>, memref<6x8xi32, 2 : i32>) attributes {link_with = "vm.o", llvm.emit_c_interface}
  func.func private @linalg_fill_i32_view16x8xi32as2(memref<6x8xi32, 2 : i32>) attributes {link_with = "vm.o", llvm.emit_c_interface}
  air.channel @aL3ToL2 []
  air.channel @bL3ToL2 []
  air.channel @aL2ToL1 []
  air.channel @bL2ToL1 []
  air.channel @cL1ToL2 []
  air.channel @cL2ToL3 []
  func.func @vecmat_i8(%arg0: memref<288xi8>, %arg1: memref<288x48xi8>, %arg2: memref<48xi32>) {
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    air.launch (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1_0) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<288xi8>, memref<288x48xi8>, memref<48xi32> {
      air.segment @vecmat_i8_0  args(%arg10=%arg3, %arg11=%arg7, %arg12=%arg8, %arg13=%arg9) : index, memref<288xi8>, memref<288x48xi8>, memref<48xi32> {
        %alloc = memref.alloc() : memref<288xi8, 1 : i32>
        %alloc_1 = memref.alloc() : memref<288x48xi8, 1 : i32>
        %alloc_2 = memref.alloc() : memref<48xi32, 1 : i32>
        air.channel.put  @aL3ToL2[] (%arg11[] [] []) : (memref<288xi8>)
        %0 = affine.apply #map()[%arg10]
        air.channel.put  @bL3ToL2[] (%arg12[0, %0] [288, 48] [48, 1]) : (memref<288x48xi8>)
        air.channel.get  @aL3ToL2[] (%alloc[] [] []) : (memref<288xi8, 1 : i32>)
        air.channel.get  @bL3ToL2[] (%alloc_1[] [] []) : (memref<288x48xi8, 1 : i32>)
        %c0 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg14 = %c0 to %c3 step %c1_3 {
          %2 = affine.apply #map1()[%arg14]
          air.channel.put  @aL2ToL1[] (%alloc[%2] [96] [1]) : (memref<288xi8, 1 : i32>)
        }
        %c0_4 = arith.constant 0 : index
        %c3_5 = arith.constant 3 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg14 = %c0_4 to %c3_5 step %c1_6 {
          %2 = affine.apply #map1()[%arg14]
          %3 = affine.apply #map1()[%arg14]
          air.channel.put  @bL2ToL1[] (%alloc_1[0, 0, %3, 0] [6, 6, 16, 8] [8, 768, 48, 1]) : (memref<288x48xi8, 1 : i32>)
        }
        %c1_7 = arith.constant 1 : index
        %c1_8 = arith.constant 1 : index
        air.herd @herd_0  tile (%arg14, %arg15) in (%arg16=%c1_7, %arg17=%c1_8) args(%arg18=%arg11, %arg19=%arg12, %arg20=%arg13, %arg21=%alloc, %arg22=%alloc_1, %arg23=%alloc_2) : memref<288xi8>, memref<288x48xi8>, memref<48xi32>, memref<288xi8, 1 : i32>, memref<288x48xi8, 1 : i32>, memref<48xi32, 1 : i32> attributes {link_with = "vm.o"} {
          %alloc_9 = memref.alloc() : memref<6x16xi8, 2 : i32>
          %alloc_10 = memref.alloc() : memref<6x6x16x8xi8, 2 : i32>
          %alloc_11 = memref.alloc() : memref<6x8xi32, 2 : i32>
          func.call @linalg_fill_i32_view16x8xi32as2(%alloc_11) : (memref<6x8xi32, 2 : i32>) -> ()
          %c0_12 = arith.constant 0 : index
          %c288 = arith.constant 288 : index
          %c96 = arith.constant 96 : index
          scf.for %arg24 = %c0_12 to %c288 step %c96 {
            air.channel.get  @aL2ToL1[] (%alloc_9[] [] []) : (memref<6x16xi8, 2 : i32>)
            air.channel.get  @bL2ToL1[] (%alloc_10[] [] []) : (memref<6x6x16x8xi8, 2 : i32>)
            func.call @vecmat_i8_i32(%alloc_9, %alloc_10, %alloc_11) : (memref<6x16xi8, 2 : i32>, memref<6x6x16x8xi8, 2 : i32>, memref<6x8xi32, 2 : i32>) -> ()
          }
          air.channel.put  @cL1ToL2[] (%alloc_11[] [] []) : (memref<6x8xi32, 2 : i32>)
          memref.dealloc %alloc_9 : memref<6x16xi8, 2 : i32>
          memref.dealloc %alloc_10 : memref<6x6x16x8xi8, 2 : i32>
          memref.dealloc %alloc_11 : memref<6x8xi32, 2 : i32>
        }
        air.channel.get  @cL1ToL2[] (%alloc_2[] [] []) : (memref<48xi32, 1 : i32>)
        air.channel.put  @cL2ToL3[] (%alloc_2[] [] []) : (memref<48xi32, 1 : i32>)
        %1 = affine.apply #map()[%arg10]
        air.channel.get  @cL2ToL3[] (%arg13[%1] [48] [1]) : (memref<48xi32>)
        memref.dealloc %alloc : memref<288xi8, 1 : i32>
        memref.dealloc %alloc_1 : memref<288x48xi8, 1 : i32>
        memref.dealloc %alloc_2 : memref<48xi32, 1 : i32>
      }
    }
    return
  }
}
