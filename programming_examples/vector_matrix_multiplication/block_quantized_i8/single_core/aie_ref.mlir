python3 /home/erweiw/mlir-air/programming_examples/vector_matrix_multiplication/block_quantized_i8/single_core/single_core.py -p
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
      air.channel.put  @bL3ToL2[] (%arg11[] [] []) : (memref<288x48xi8>)
      air.channel.put  @bL3ToL2[] (%arg12[] [] []) : (memref<9x48xf32>)
      air.channel.get  @cL2ToL3[] (%arg13[] [] []) : (memref<48xf32>)
      air.segment @vecmat_i8_0  {
        %alloc = memref.alloc() : memref<288xi8, 1 : i32>
        %alloc_1 = memref.alloc() : memref<9xf32, 1 : i32>
        air.channel.get  @aL3ToL2[] (%alloc[] [] []) : (memref<288xi8, 1 : i32>)
        air.channel.get  @aL3ToL2[] (%alloc_1[] [] []) : (memref<9xf32, 1 : i32>)
        %alloc_2 = memref.alloc() : memref<288x48xi8, 1 : i32>
        %alloc_3 = memref.alloc() : memref<9x48xf32, 1 : i32>
        air.channel.get  @bL3ToL2[] (%alloc_2[] [] []) : (memref<288x48xi8, 1 : i32>)
        air.channel.get  @bL3ToL2[] (%alloc_3[] [] []) : (memref<9x48xf32, 1 : i32>)
        %c0 = arith.constant 0 : index
        %c96 = arith.constant 96 : index
        %c1_4 = arith.constant 1 : index
        air.channel.put  @aL2ToL1[] (%alloc[%c0] [%c96] [%c1_4]) : (memref<288xi8, 1 : i32>)
        %c0_5 = arith.constant 0 : index
        %c3 = arith.constant 3 : index
        %c1_6 = arith.constant 1 : index
        air.channel.put  @aL2ToL1[] (%alloc_1[%c0_5] [%c3] [%c1_6]) : (memref<9xf32, 1 : i32>)
        %c96_7 = arith.constant 96 : index
        %c96_8 = arith.constant 96 : index
        %c1_9 = arith.constant 1 : index
        air.channel.put  @aL2ToL1[] (%alloc[%c96_7] [%c96_8] [%c1_9]) : (memref<288xi8, 1 : i32>)
        %c3_10 = arith.constant 3 : index
        %c3_11 = arith.constant 3 : index
        %c1_12 = arith.constant 1 : index
        air.channel.put  @aL2ToL1[] (%alloc_1[%c3_10] [%c3_11] [%c1_12]) : (memref<9xf32, 1 : i32>)
        %c192 = arith.constant 192 : index
        %c96_13 = arith.constant 96 : index
        %c1_14 = arith.constant 1 : index
        air.channel.put  @aL2ToL1[] (%alloc[%c192] [%c96_13] [%c1_14]) : (memref<288xi8, 1 : i32>)
        %c6 = arith.constant 6 : index
        %c3_15 = arith.constant 3 : index
        %c1_16 = arith.constant 1 : index
        air.channel.put  @aL2ToL1[] (%alloc_1[%c6] [%c3_15] [%c1_16]) : (memref<9xf32, 1 : i32>)
        %c0_17 = arith.constant 0 : index
        %c0_18 = arith.constant 0 : index
        %c0_19 = arith.constant 0 : index
        %c6_20 = arith.constant 6 : index
        %c96_21 = arith.constant 96 : index
        %c8 = arith.constant 8 : index
        %c8_22 = arith.constant 8 : index
        %c48 = arith.constant 48 : index
        %c1_23 = arith.constant 1 : index
        air.channel.put  @bL2ToL1[] (%alloc_2[%c0_17, %c0_18, %c0_19] [%c6_20, %c96_21, %c8] [%c8_22, %c48, %c1_23]) : (memref<288x48xi8, 1 : i32>)
        %c0_24 = arith.constant 0 : index
        %c0_25 = arith.constant 0 : index
        %c0_26 = arith.constant 0 : index
        %c6_27 = arith.constant 6 : index
        %c3_28 = arith.constant 3 : index
        %c8_29 = arith.constant 8 : index
        %c8_30 = arith.constant 8 : index
        %c48_31 = arith.constant 48 : index
        %c1_32 = arith.constant 1 : index
        air.channel.put  @bL2ToL1[] (%alloc_3[%c0_24, %c0_25, %c0_26] [%c6_27, %c3_28, %c8_29] [%c8_30, %c48_31, %c1_32]) : (memref<9x48xf32, 1 : i32>)
        %c0_33 = arith.constant 0 : index
        %c96_34 = arith.constant 96 : index
        %c0_35 = arith.constant 0 : index
        %c6_36 = arith.constant 6 : index
        %c96_37 = arith.constant 96 : index
        %c8_38 = arith.constant 8 : index
        %c8_39 = arith.constant 8 : index
        %c48_40 = arith.constant 48 : index
        %c1_41 = arith.constant 1 : index
        air.channel.put  @bL2ToL1[] (%alloc_2[%c0_33, %c96_34, %c0_35] [%c6_36, %c96_37, %c8_38] [%c8_39, %c48_40, %c1_41]) : (memref<288x48xi8, 1 : i32>)
        %c0_42 = arith.constant 0 : index
        %c3_43 = arith.constant 3 : index
        %c0_44 = arith.constant 0 : index
        %c6_45 = arith.constant 6 : index
        %c3_46 = arith.constant 3 : index
        %c8_47 = arith.constant 8 : index
        %c8_48 = arith.constant 8 : index
        %c48_49 = arith.constant 48 : index
        %c1_50 = arith.constant 1 : index
        air.channel.put  @bL2ToL1[] (%alloc_3[%c0_42, %c3_43, %c0_44] [%c6_45, %c3_46, %c8_47] [%c8_48, %c48_49, %c1_50]) : (memref<9x48xf32, 1 : i32>)
        %c0_51 = arith.constant 0 : index
        %c192_52 = arith.constant 192 : index
        %c0_53 = arith.constant 0 : index
        %c6_54 = arith.constant 6 : index
        %c96_55 = arith.constant 96 : index
        %c8_56 = arith.constant 8 : index
        %c8_57 = arith.constant 8 : index
        %c48_58 = arith.constant 48 : index
        %c1_59 = arith.constant 1 : index
        air.channel.put  @bL2ToL1[] (%alloc_2[%c0_51, %c192_52, %c0_53] [%c6_54, %c96_55, %c8_56] [%c8_57, %c48_58, %c1_59]) : (memref<288x48xi8, 1 : i32>)
        %c0_60 = arith.constant 0 : index
        %c6_61 = arith.constant 6 : index
        %c0_62 = arith.constant 0 : index
        %c6_63 = arith.constant 6 : index
        %c3_64 = arith.constant 3 : index
        %c8_65 = arith.constant 8 : index
        %c8_66 = arith.constant 8 : index
        %c48_67 = arith.constant 48 : index
        %c1_68 = arith.constant 1 : index
        air.channel.put  @bL2ToL1[] (%alloc_3[%c0_60, %c6_61, %c0_62] [%c6_63, %c3_64, %c8_65] [%c8_66, %c48_67, %c1_68]) : (memref<9x48xf32, 1 : i32>)
        %alloc_69 = memref.alloc() : memref<48xf32, 1 : i32>
        air.channel.get  @cL1ToL2[] (%alloc_69[] [] []) : (memref<48xf32, 1 : i32>)
        %c1_70 = arith.constant 1 : index
        %c1_71 = arith.constant 1 : index
        air.herd @herd_0  tile (%arg14, %arg15) in (%arg16=%c1_70, %arg17=%c1_71) attributes {link_with = "vm.o"} {
          %alloc_72 = memref.alloc() : memref<6x8xf32, 2 : i32>
          %cst = arith.constant 0.000000e+00 : f32
          func.call @linalg_fill_i32_view16x8xi32as2(%cst, %alloc_72) : (f32, memref<6x8xf32, 2 : i32>) -> ()
          %c0_73 = arith.constant 0 : index
          %c288 = arith.constant 288 : index
          %c96_74 = arith.constant 96 : index
          scf.for %arg18 = %c0_73 to %c288 step %c96_74 {
            %alloc_75 = memref.alloc() : memref<6x16xi8, 2 : i32>
            %alloc_76 = memref.alloc() : memref<3xf32, 2 : i32>
            air.channel.get  @aL2ToL1[] (%alloc_75[] [] []) : (memref<6x16xi8, 2 : i32>)
            air.channel.get  @aL2ToL1[] (%alloc_76[] [] []) : (memref<3xf32, 2 : i32>)
            %alloc_77 = memref.alloc() : memref<6x6x16x8xi8, 2 : i32>
            %alloc_78 = memref.alloc() : memref<6x3x8xf32, 2 : i32>
            air.channel.get  @bL2ToL1[] (%alloc_77[] [] []) : (memref<6x6x16x8xi8, 2 : i32>)
            air.channel.get  @bL2ToL1[] (%alloc_78[] [] []) : (memref<6x3x8xf32, 2 : i32>)
            func.call @vecmat_i8_f32_i32_32(%alloc_75, %alloc_76, %alloc_77, %alloc_78, %alloc_72) : (memref<6x16xi8, 2 : i32>, memref<3xf32, 2 : i32>, memref<6x6x16x8xi8, 2 : i32>, memref<6x3x8xf32, 2 : i32>, memref<6x8xf32, 2 : i32>) -> ()
            memref.dealloc %alloc_75 : memref<6x16xi8, 2 : i32>
            memref.dealloc %alloc_76 : memref<3xf32, 2 : i32>
            memref.dealloc %alloc_77 : memref<6x6x16x8xi8, 2 : i32>
            memref.dealloc %alloc_78 : memref<6x3x8xf32, 2 : i32>
          }
          air.channel.put  @cL1ToL2[] (%alloc_72[] [] []) : (memref<6x8xf32, 2 : i32>)
          memref.dealloc %alloc_72 : memref<6x8xf32, 2 : i32>
        }
        air.channel.put  @cL2ToL3[] (%alloc_69[] [] []) : (memref<48xf32, 1 : i32>)
        memref.dealloc %alloc : memref<288xi8, 1 : i32>
        memref.dealloc %alloc_1 : memref<9xf32, 1 : i32>
        memref.dealloc %alloc_2 : memref<288x48xi8, 1 : i32>
        memref.dealloc %alloc_3 : memref<9x48xf32, 1 : i32>
        memref.dealloc %alloc_69 : memref<48xf32, 1 : i32>
      }
    }
    return
  }
}

