#map = affine_map<()[s0] -> (s0 * 48)>
#map1 = affine_map<()[s0] -> (s0 * 96)>
module {
  air.channel @aL3ToL2 []
  air.channel @bL3ToL2 []
  air.channel @aL2ToL1 []
  air.channel @bL2ToL1 []
  air.channel @cL1ToL2 []
  air.channel @cL2ToL3 []
  func.func private @linalg_fill_bf16(bf16, memref<48xbf16, 2 : i32>) attributes {link_with = "vm.o", llvm.emit_c_interface}
  func.func private @vecmat_bf16_bf16(memref<96xbf16, 2 : i32>, memref<96x48xbf16, 2 : i32>, memref<48xbf16, 2 : i32>) attributes {link_with = "vm.o", llvm.emit_c_interface}
  func.func @vecmat_bf16(%arg0: memref<288xbf16>, %arg1: memref<288x48xbf16>, %arg2: memref<48xbf16>) {
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    air.launch (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1_0) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<288xbf16>, memref<288x48xbf16>, memref<48xbf16> {
      air.channel.put  @aL3ToL2[] (%arg7[] [] []) : (memref<288xbf16>)
      %0 = affine.apply #map()[%arg4]
      %c0 = arith.constant 0 : index
      %c288 = arith.constant 288 : index
      %c48 = arith.constant 48 : index
      %c48_1 = arith.constant 48 : index
      %c1_2 = arith.constant 1 : index
      air.channel.put  @bL3ToL2[] (%arg8[%c0, %0] [%c288, %c48] [%c48_1, %c1_2]) : (memref<288x48xbf16>)
      %c48_3 = arith.constant 48 : index
      %c1_4 = arith.constant 1 : index
      air.channel.get  @cL2ToL3[] (%arg9[%0] [%c48_3] [%c1_4]) : (memref<48xbf16>)
      air.segment @vecmat_i8_0  {
        %alloc = memref.alloc() : memref<288xbf16, 1 : i32>
        air.channel.get  @aL3ToL2[] (%alloc[] [] []) : (memref<288xbf16, 1 : i32>)
        %alloc_5 = memref.alloc() : memref<288x48xbf16, 1 : i32>
        air.channel.get  @bL3ToL2[] (%alloc_5[] [] []) : (memref<288x48xbf16, 1 : i32>)
        %c0_6 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_7 = arith.constant 1 : index
        scf.for %arg10 = %c0_6 to %c3 step %c1_7 {
          %1 = affine.apply #map1()[%arg10]
          air.channel.put  @aL2ToL1[] (%alloc[] [] []) : (memref<288xbf16, 1 : i32>)
        }
        %c0_8 = arith.constant 0 : index
        %c3_9 = arith.constant 3 : index
        %c1_10 = arith.constant 1 : index
        scf.for %arg10 = %c0_8 to %c3_9 step %c1_10 {
          %1 = affine.apply #map1()[%arg10]
          air.channel.put  @bL2ToL1[] (%alloc_5[] [] []) : (memref<288x48xbf16, 1 : i32>)
        }
        %alloc_11 = memref.alloc() : memref<48xbf16, 1 : i32>
        air.channel.get  @cL1ToL2[] (%alloc_11[] [] []) : (memref<48xbf16, 1 : i32>)
        %c1_12 = arith.constant 1 : index
        %c1_13 = arith.constant 1 : index
        air.herd @herd_0  tile (%arg10, %arg11) in (%arg12=%c1_12, %arg13=%c1_13) attributes {link_with = "vm.o"} {
          %alloc_14 = memref.alloc() : memref<48xbf16, 2 : i32>
          %cst = arith.constant 0.000000e+00 : bf16
          func.call @linalg_fill_bf16(%cst, %alloc_14) : (bf16, memref<48xbf16, 2 : i32>) -> ()
          %c0_15 = arith.constant 0 : index
          %c288_16 = arith.constant 288 : index
          %c96 = arith.constant 96 : index
          scf.for %arg14 = %c0_15 to %c288_16 step %c96 {
            %alloc_17 = memref.alloc() : memref<96xbf16, 2 : i32>
            air.channel.get  @aL2ToL1[] (%alloc_17[] [] []) : (memref<96xbf16, 2 : i32>)
            %alloc_18 = memref.alloc() : memref<96x48xbf16, 2 : i32>
            air.channel.get  @bL2ToL1[] (%alloc_18[] [] []) : (memref<96x48xbf16, 2 : i32>)
            func.call @vecmat_bf16_bf16(%alloc_17, %alloc_18, %alloc_14) : (memref<96xbf16, 2 : i32>, memref<96x48xbf16, 2 : i32>, memref<48xbf16, 2 : i32>) -> ()
            memref.dealloc %alloc_17 : memref<96xbf16, 2 : i32>
            memref.dealloc %alloc_18 : memref<96x48xbf16, 2 : i32>
          }
          air.channel.put  @cL1ToL2[] (%alloc_14[] [] []) : (memref<48xbf16, 2 : i32>)
          memref.dealloc %alloc_14 : memref<48xbf16, 2 : i32>
        }
        air.channel.put  @cL2ToL3[] (%alloc_11[] [] []) : (memref<48xbf16, 1 : i32>)
        memref.dealloc %alloc : memref<288xbf16, 1 : i32>
        memref.dealloc %alloc_5 : memref<288x48xbf16, 1 : i32>
        memref.dealloc %alloc_11 : memref<48xbf16, 1 : i32>
      }
    }
    return
  }
}

