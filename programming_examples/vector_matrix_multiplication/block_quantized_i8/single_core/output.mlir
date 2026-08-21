python3 /home/erweiw/mlir-air/programming_examples/vector_matrix_multiplication/block_quantized_i8/single_core/single_core.py -p
#map = affine_map<()[s0] -> (s0 * 48)>
#map1 = affine_map<()[s0] -> (s0 * 96)>
#map2 = affine_map<()[s0] -> (s0 * 3)>
module {
  air.channel @aL3ToL2 []
  air.channel @bL3ToL2 []
  air.channel @aL2ToL1 []
  air.channel @bL2ToL1 []
  air.channel @cL1ToL2 []
  air.channel @cL2ToL3 []
  func.func private @linalg_fill_i32_view16x8xi32as2(f32, memref<6x8xf32, 2 : i32>) attributes {link_with = "vm.o", llvm.emit_c_interface}
  func.func private @vecmat_i8_f32_i32_32(memref<6x16xi8, 2 : i32>, memref<3xf32, 2 : i32>, memref<6x6x16x8xi8, 2 : i32>, memref<6x3x8xf32, 2 : i32>, memref<6x8xf32, 2 : i32>) attributes {link_with = "vm.o", llvm.emit_c_interface}
  func.func @vecmat_i8(%arg0: memref<288xi8>, %arg1: memref<9xf32>, %arg2: memref<288x48xi8>, %arg3: memref<9x48xf32>, %arg4: memref<48xf32>) {
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    air.launch (%arg5, %arg6) in (%arg7=%c1, %arg8=%c1_0) args(%arg9=%arg0, %arg10=%arg1, %arg11=%arg2, %arg12=%arg3, %arg13=%arg4) : memref<288xi8>, memref<9xf32>, memref<288x48xi8>, memref<9x48xf32>, memref<48xf32> {
      air.channel.put  @aL3ToL2[] (%arg9[] [] []) : (memref<288xi8>)
      air.channel.put  @aL3ToL2[] (%arg10[] [] []) : (memref<9xf32>)
      %0 = affine.apply #map()[%arg6]
      %c0 = arith.constant 0 : index
      %c288 = arith.constant 288 : index
      %c48 = arith.constant 48 : index
      %c48_1 = arith.constant 48 : index
      %c1_2 = arith.constant 1 : index
      air.channel.put  @bL3ToL2[] (%arg11[%c0, %0] [%c288, %c48] [%c48_1, %c1_2]) : (memref<288x48xi8>)
      %c0_3 = arith.constant 0 : index
      %c9 = arith.constant 9 : index
      %c48_4 = arith.constant 48 : index
      %c48_5 = arith.constant 48 : index
      %c1_6 = arith.constant 1 : index
      air.channel.put  @bL3ToL2[] (%arg12[%c0_3, %0] [%c9, %c48_4] [%c48_5, %c1_6]) : (memref<9x48xf32>)
      %c48_7 = arith.constant 48 : index
      %c1_8 = arith.constant 1 : index
      air.channel.get  @cL2ToL3[] (%arg13[%0] [%c48_7] [%c1_8]) : (memref<48xf32>)
      air.segment @vecmat_i8_0  {
        %alloc = memref.alloc() : memref<288xi8, 1 : i32>
        %alloc_9 = memref.alloc() : memref<9xf32, 1 : i32>
        air.channel.get  @aL3ToL2[] (%alloc[] [] []) : (memref<288xi8, 1 : i32>)
        air.channel.get  @aL3ToL2[] (%alloc_9[] [] []) : (memref<9xf32, 1 : i32>)
        %alloc_10 = memref.alloc() : memref<288x48xi8, 1 : i32>
        %alloc_11 = memref.alloc() : memref<9x48xf32, 1 : i32>
        air.channel.get  @bL3ToL2[] (%alloc_10[] [] []) : (memref<288x48xi8, 1 : i32>)
        air.channel.get  @bL3ToL2[] (%alloc_11[] [] []) : (memref<9x48xf32, 1 : i32>)
        %c0_12 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_13 = arith.constant 1 : index
        scf.for %arg14 = %c0_12 to %c3 step %c1_13 {
          %1 = affine.apply #map1()[%arg14]
          %c96 = arith.constant 96 : index
          %c1_20 = arith.constant 1 : index
          air.channel.put  @aL2ToL1[] (%alloc[%1] [%c96] [%c1_20]) : (memref<288xi8, 1 : i32>)
          %2 = affine.apply #map2()[%arg14]
          %c3_21 = arith.constant 3 : index
          %c1_22 = arith.constant 1 : index
          air.channel.put  @aL2ToL1[] (%alloc_9[%2] [%c3_21] [%c1_22]) : (memref<9xf32, 1 : i32>)
        }
        %c0_14 = arith.constant 0 : index
        %c3_15 = arith.constant 3 : index
        %c1_16 = arith.constant 1 : index
        scf.for %arg14 = %c0_14 to %c3_15 step %c1_16 {
          %1 = affine.apply #map1()[%arg14]
          %c0_20 = arith.constant 0 : index
          %c0_21 = arith.constant 0 : index
          %c6 = arith.constant 6 : index
          %c96 = arith.constant 96 : index
          %c8 = arith.constant 8 : index
          %c8_22 = arith.constant 8 : index
          %c48_23 = arith.constant 48 : index
          %c1_24 = arith.constant 1 : index
          air.channel.put  @bL2ToL1[] (%alloc_10[%c0_20, %1, %c0_21] [%c6, %c96, %c8] [%c8_22, %c48_23, %c1_24]) : (memref<288x48xi8, 1 : i32>)
          %2 = affine.apply #map2()[%arg14]
          %c0_25 = arith.constant 0 : index
          %c0_26 = arith.constant 0 : index
          %c6_27 = arith.constant 6 : index
          %c3_28 = arith.constant 3 : index
          %c8_29 = arith.constant 8 : index
          %c8_30 = arith.constant 8 : index
          %c48_31 = arith.constant 48 : index
          %c1_32 = arith.constant 1 : index
          air.channel.put  @bL2ToL1[] (%alloc_11[%c0_25, %2, %c0_26] [%c6_27, %c3_28, %c8_29] [%c8_30, %c48_31, %c1_32]) : (memref<9x48xf32, 1 : i32>)
        }
        %alloc_17 = memref.alloc() : memref<48xf32, 1 : i32>
        air.channel.get  @cL1ToL2[] (%alloc_17[] [] []) : (memref<48xf32, 1 : i32>)
        %c1_18 = arith.constant 1 : index
        %c1_19 = arith.constant 1 : index
        air.herd @herd_0  tile (%arg14, %arg15) in (%arg16=%c1_18, %arg17=%c1_19) attributes {link_with = "vm.o"} {
          %alloc_20 = memref.alloc() : memref<6x8xf32, 2 : i32>
          %cst = arith.constant 0.000000e+00 : f32
          func.call @linalg_fill_i32_view16x8xi32as2(%cst, %alloc_20) : (f32, memref<6x8xf32, 2 : i32>) -> ()
          %c0_21 = arith.constant 0 : index
          %c288_22 = arith.constant 288 : index
          %c96 = arith.constant 96 : index
          scf.for %arg18 = %c0_21 to %c288_22 step %c96 {
            %alloc_23 = memref.alloc() : memref<6x16xi8, 2 : i32>
            %alloc_24 = memref.alloc() : memref<3xf32, 2 : i32>
            air.channel.get  @aL2ToL1[] (%alloc_23[] [] []) : (memref<6x16xi8, 2 : i32>)
            air.channel.get  @aL2ToL1[] (%alloc_24[] [] []) : (memref<3xf32, 2 : i32>)
            %alloc_25 = memref.alloc() : memref<6x6x16x8xi8, 2 : i32>
            %alloc_26 = memref.alloc() : memref<6x3x8xf32, 2 : i32>
            air.channel.get  @bL2ToL1[] (%alloc_25[] [] []) : (memref<6x6x16x8xi8, 2 : i32>)
            air.channel.get  @bL2ToL1[] (%alloc_26[] [] []) : (memref<6x3x8xf32, 2 : i32>)
            func.call @vecmat_i8_f32_i32_32(%alloc_23, %alloc_24, %alloc_25, %alloc_26, %alloc_20) : (memref<6x16xi8, 2 : i32>, memref<3xf32, 2 : i32>, memref<6x6x16x8xi8, 2 : i32>, memref<6x3x8xf32, 2 : i32>, memref<6x8xf32, 2 : i32>) -> ()
            memref.dealloc %alloc_23 : memref<6x16xi8, 2 : i32>
            memref.dealloc %alloc_24 : memref<3xf32, 2 : i32>
            memref.dealloc %alloc_25 : memref<6x6x16x8xi8, 2 : i32>
            memref.dealloc %alloc_26 : memref<6x3x8xf32, 2 : i32>
          }
          air.channel.put  @cL1ToL2[] (%alloc_20[] [] []) : (memref<6x8xf32, 2 : i32>)
          memref.dealloc %alloc_20 : memref<6x8xf32, 2 : i32>
        }
        air.channel.put  @cL2ToL3[] (%alloc_17[] [] []) : (memref<48xf32, 1 : i32>)
        memref.dealloc %alloc : memref<288xi8, 1 : i32>
        memref.dealloc %alloc_9 : memref<9xf32, 1 : i32>
        memref.dealloc %alloc_10 : memref<288x48xi8, 1 : i32>
        memref.dealloc %alloc_11 : memref<9x48xf32, 1 : i32>
        memref.dealloc %alloc_17 : memref<48xf32, 1 : i32>
      }
    }
    return
  }
}

