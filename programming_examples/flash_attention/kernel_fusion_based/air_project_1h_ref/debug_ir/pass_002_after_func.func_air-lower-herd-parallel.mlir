#map = affine_map<()[s0] -> (s0 * 16384)>
#map1 = affine_map<()[s0] -> (s0 * 64)>
#set = affine_set<()[s0, s1] : (s0 >= 0, s1 == 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 1 == 0)>
#set2 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 2 == 0)>
#set3 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 3 == 0)>
#set4 = affine_set<()[s0, s1] : (s1 - 1 >= 0, -s1 + 2 >= 0, s0 >= 0, -s0 + 3 >= 0)>
module {
  func.func private @zero_fill_g_bf16(memref<4096xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @zero_fill_gp_bf16(memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @zero_fill_sp_bf16(memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @neg_inf_fill_up_bf16(memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @matmul_a_b_bf16(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @matmul_g_b_bf16(memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @fused_softmax(memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @maximum_up_u_bf16(memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @exp_up_minus_u(memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @mul_r_gp(memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @accum_sp_r_s(memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @vector_copy_32elems(i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @copy_tile(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @div_gp_sp(memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  func.func private @add_gp_g(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn.o", llvm.emit_c_interface}
  air.channel @QK2L1 [1, 4] {broadcast_shape = [4 : index, 4 : index]}
  air.channel @V2L1_0 [1, 1] {broadcast_shape = [4 : index, 1 : index]}
  air.channel @VIn_0 [1]
  air.channel @V2L1_1 [1, 1] {broadcast_shape = [4 : index, 1 : index]}
  air.channel @VIn_1 [1]
  air.channel @V2L1_2 [1, 1] {broadcast_shape = [4 : index, 1 : index]}
  air.channel @VIn_2 [1]
  air.channel @V2L1_3 [1, 1] {broadcast_shape = [4 : index, 1 : index]}
  air.channel @VIn_3 [1]
  air.channel @cascade_gp [4, 3] {channel_type = "cascade"}
  air.channel @cascade_up [4, 3] {channel_type = "cascade"}
  air.channel @cascade_sp [4, 3] {channel_type = "cascade"}
  air.channel @Gp2L2 [4, 1]
  air.channel @GpOut [1]
  func.func @attention_bf16(%arg0: memref<512x64xbf16>, %arg1: memref<512x64xbf16>, %arg2: memref<512x64xbf16>, %arg3: memref<512x64xbf16>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    air.launch (%arg4, %arg5) in (%arg6=%c2, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<512x64xbf16>, memref<512x64xbf16>, memref<512x64xbf16>, memref<512x64xbf16> {
      %c24576 = arith.constant 24576 : index
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c3 = arith.constant 3 : index
      %c2_0 = arith.constant 2 : index
      %c1_1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %c4096 = arith.constant 4096 : index
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %0 = affine.apply #map()[%arg4]
      air.channel.put  @QK2L1[%c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %0] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) : (memref<512x64xbf16>)
      air.channel.put  @QK2L1[%c0, %c1_1] (%arg8[%c0, %c0, %c0, %c0, %0] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) : (memref<512x64xbf16>)
      air.channel.put  @QK2L1[%c0, %c2_0] (%arg8[%c0, %c0, %c0, %c0, %0] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) : (memref<512x64xbf16>)
      air.channel.put  @QK2L1[%c0, %c3] (%arg8[%c0, %c0, %c0, %c0, %0] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) : (memref<512x64xbf16>)
      air.channel.put  @QK2L1[%c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %c0] [%c2_0, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) : (memref<512x64xbf16>)
      air.channel.put  @QK2L1[%c0, %c1_1] (%arg9[%c0, %c0, %c0, %c0, %c8192] [%c2_0, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) : (memref<512x64xbf16>)
      air.channel.put  @QK2L1[%c0, %c2_0] (%arg9[%c0, %c0, %c0, %c0, %c16384] [%c2_0, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) : (memref<512x64xbf16>)
      air.channel.put  @QK2L1[%c0, %c3] (%arg9[%c0, %c0, %c0, %c0, %c24576] [%c2_0, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) : (memref<512x64xbf16>)
      air.channel.put  @VIn_0[%c0] (%arg10[%c0, %c0, %c0] [%c2_0, %c64, %c64] [%c4096, %c64, %c1_1]) : (memref<512x64xbf16>)
      air.channel.put  @VIn_1[%c0] (%arg10[%c0, %c0, %c8192] [%c2_0, %c64, %c64] [%c4096, %c64, %c1_1]) : (memref<512x64xbf16>)
      air.channel.put  @VIn_2[%c0] (%arg10[%c0, %c0, %c16384] [%c2_0, %c64, %c64] [%c4096, %c64, %c1_1]) : (memref<512x64xbf16>)
      air.channel.put  @VIn_3[%c0] (%arg10[%c0, %c0, %c24576] [%c2_0, %c64, %c64] [%c4096, %c64, %c1_1]) : (memref<512x64xbf16>)
      air.channel.get  @GpOut[%c0] (%arg11[%0] [%c16384] [%c1_1]) : (memref<512x64xbf16>)
      air.segment @attn_seg  unroll(%arg12, %arg13) in (%arg14=%c1_1, %arg15=%c1_1) {
        %c64_2 = arith.constant 64 : index
        %c512_3 = arith.constant 512 : index
        %c8_4 = arith.constant 8 : index
        %c1_5 = arith.constant 1 : index
        %c2_6 = arith.constant 2 : index
        %c0_7 = arith.constant 0 : index
        %c4_8 = arith.constant 4 : index
        %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_9 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_10 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_11 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_12 = memref.alloc() : memref<256x64xbf16, 1 : i32>
        %alloc_13 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_14 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_15 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_16 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_17 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_18 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        %alloc_19 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        scf.for %arg16 = %c0_7 to %c2_6 step %c1_5 {
          air.channel.get  @VIn_0[%c0_7] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @V2L1_0[%c0_7, %c0_7] (%alloc[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_3, %c64_2, %c1_5]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_7 to %c2_6 step %c1_5 {
          air.channel.get  @VIn_1[%c0_7] (%alloc_9[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @V2L1_1[%c0_7, %c0_7] (%alloc_9[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_3, %c64_2, %c1_5]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_7 to %c2_6 step %c1_5 {
          air.channel.get  @VIn_2[%c0_7] (%alloc_10[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @V2L1_2[%c0_7, %c0_7] (%alloc_10[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_3, %c64_2, %c1_5]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_7 to %c2_6 step %c1_5 {
          air.channel.get  @VIn_3[%c0_7] (%alloc_11[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @V2L1_3[%c0_7, %c0_7] (%alloc_11[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_3, %c64_2, %c1_5]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.forall (%arg16) in (4) {
          %1 = affine.apply #map1()[%arg16]
          air.channel.get  @Gp2L2[%arg16, %c0_7] (%alloc_12[%1, %c0_7] [%c64_2, %c64_2] [%c64_2, %c1_5]) : (memref<256x64xbf16, 1 : i32>)
        }
        air.channel.put  @GpOut[%c0_7] (%alloc_12[] [] []) : (memref<256x64xbf16, 1 : i32>)
        air.herd @herd_0  tile (%arg16, %arg17) in (%arg18=%c4_8, %arg19=%c4_8) args(%arg20=%alloc_13, %arg21=%alloc_14, %arg22=%alloc_15, %arg23=%alloc_16, %arg24=%alloc_17, %arg25=%alloc_18, %arg26=%alloc_19) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32> attributes {link_with = "attn.o"} {
          %c512_20 = arith.constant 512 : index
          %c64_21 = arith.constant 64 : index
          %c8_22 = arith.constant 8 : index
          %c1_23 = arith.constant 1 : index
          %c0_24 = arith.constant 0 : index
          %c2_25 = arith.constant 2 : index
          %c3_i32 = arith.constant 3 : i32
          %c2_i32 = arith.constant 2 : i32
          %c1_i32 = arith.constant 1 : i32
          %c0_i32 = arith.constant 0 : i32
          func.call @zero_fill_gp_bf16(%arg24) : (memref<64x64xbf16, 2 : i32>) -> ()
          func.call @zero_fill_sp_bf16(%arg26) : (memref<64x1xbf16, 2 : i32>) -> ()
          func.call @neg_inf_fill_up_bf16(%arg25) : (memref<64x1xbf16, 2 : i32>) -> ()
          air.channel.get  @QK2L1[%arg16, %arg17] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          %1 = arith.index_cast %arg16 : index to i32
          %2 = arith.cmpi eq, %1, %c0_i32 : i32
          scf.if %2 {
            func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          air.channel.get  @QK2L1[%arg16, %arg17] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          %3 = arith.index_cast %arg16 : index to i32
          %4 = arith.cmpi eq, %3, %c1_i32 : i32
          scf.if %4 {
            func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          air.channel.get  @QK2L1[%arg16, %arg17] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          %5 = arith.index_cast %arg16 : index to i32
          %6 = arith.cmpi eq, %5, %c2_i32 : i32
          scf.if %6 {
            func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          air.channel.get  @QK2L1[%arg16, %arg17] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          %7 = arith.index_cast %arg16 : index to i32
          %8 = arith.cmpi eq, %7, %c3_i32 : i32
          scf.if %8 {
            func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          scf.for %arg27 = %c0_24 to %c2_25 step %c1_23 {
            %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            air.channel.get  @QK2L1[%arg16, %arg17] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
            affine.if #set()[%arg16, %arg17] {
              air.channel.get  @V2L1_0[%arg16, %arg17] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set1()[%arg16, %arg17] {
              air.channel.get  @V2L1_1[%arg16, %arg17] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set2()[%arg16, %arg17] {
              air.channel.get  @V2L1_2[%arg16, %arg17] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            affine.if #set3()[%arg16, %arg17] {
              air.channel.get  @V2L1_3[%arg16, %arg17] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
            }
            func.call @matmul_a_b_bf16(%arg20, %arg21, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            %alloc_26 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            %alloc_27 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            func.call @fused_softmax(%collapse_shape, %arg25, %alloc_26, %alloc_27) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @mul_r_gp(%alloc_27, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            func.call @matmul_g_b_bf16(%collapse_shape, %arg22, %arg24) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            func.call @accum_sp_r_s(%arg26, %alloc_27, %alloc_26) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @vector_copy_32elems(%c0_i32, %alloc_26, %arg26) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            memref.dealloc %alloc_26 : memref<64x1xbf16, 2 : i32>
            memref.dealloc %alloc_27 : memref<64x1xbf16, 2 : i32>
          }
          affine.if #set3()[%arg16, %arg17] {
            %9 = arith.subi %arg17, %c1_23 : index
            air.channel.put  @cascade_gp[%arg16, %9] (%arg24[] [] []) : (memref<64x64xbf16, 2 : i32>)
            air.channel.put  @cascade_up[%arg16, %9] (%arg25[] [] []) : (memref<64x1xbf16, 2 : i32>)
            air.channel.put  @cascade_sp[%arg16, %9] (%arg26[] [] []) : (memref<64x1xbf16, 2 : i32>)
          } else {
            affine.if #set4()[%arg16, %arg17] {
              %alloc_26 = memref.alloc() : memref<64x64xbf16, 2 : i32>
              %alloc_27 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %alloc_28 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.channel.get  @cascade_gp[%arg16, %arg17] (%alloc_26[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.get  @cascade_up[%arg16, %arg17] (%alloc_27[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.get  @cascade_sp[%arg16, %arg17] (%alloc_28[] [] []) : (memref<64x1xbf16, 2 : i32>)
              %alloc_29 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @vector_copy_32elems(%c0_i32, %arg25, %alloc_29) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @maximum_up_u_bf16(%alloc_27, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_30 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_27, %arg25, %alloc_30) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_31 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_29, %arg25, %alloc_31) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_30, %alloc_26) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_31, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @add_gp_g(%arg24, %alloc_26) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %alloc_32 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @zero_fill_sp_bf16(%alloc_32) : (memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%alloc_28, %alloc_30, %alloc_32) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%arg26, %alloc_31, %alloc_32) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @vector_copy_32elems(%c0_i32, %alloc_32, %alloc_28) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %9 = arith.subi %arg17, %c1_23 : index
              air.channel.put  @cascade_gp[%arg16, %9] (%alloc_26[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.put  @cascade_up[%arg16, %9] (%arg25[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.put  @cascade_sp[%arg16, %9] (%alloc_28[] [] []) : (memref<64x1xbf16, 2 : i32>)
              memref.dealloc %alloc_26 : memref<64x64xbf16, 2 : i32>
              memref.dealloc %alloc_27 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_28 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_29 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_30 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_31 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_32 : memref<64x1xbf16, 2 : i32>
            } else {
              %alloc_26 = memref.alloc() : memref<64x64xbf16, 2 : i32>
              %alloc_27 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              %alloc_28 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.channel.get  @cascade_gp[%arg16, %arg17] (%alloc_26[] [] []) : (memref<64x64xbf16, 2 : i32>)
              air.channel.get  @cascade_up[%arg16, %arg17] (%alloc_27[] [] []) : (memref<64x1xbf16, 2 : i32>)
              air.channel.get  @cascade_sp[%arg16, %arg17] (%alloc_28[] [] []) : (memref<64x1xbf16, 2 : i32>)
              %alloc_29 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @vector_copy_32elems(%c0_i32, %arg25, %alloc_29) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @maximum_up_u_bf16(%alloc_27, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_30 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_27, %arg25, %alloc_30) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              %alloc_31 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @exp_up_minus_u(%alloc_29, %arg25, %alloc_31) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_30, %alloc_26) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @mul_r_gp(%alloc_31, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              func.call @add_gp_g(%arg24, %alloc_26) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              %alloc_32 = memref.alloc() : memref<64x1xbf16, 2 : i32>
              func.call @zero_fill_sp_bf16(%alloc_32) : (memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%alloc_28, %alloc_30, %alloc_32) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @accum_sp_r_s(%arg26, %alloc_31, %alloc_32) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @vector_copy_32elems(%c0_i32, %alloc_32, %alloc_28) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              func.call @div_gp_sp(%alloc_28, %alloc_26) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              air.channel.put  @Gp2L2[%arg16, %c0_24] (%alloc_26[%c0_24, %c0_24, %c0_24, %c0_24] [%c8_22, %c8_22, %c8_22, %c8_22] [%c64_21, %c8_22, %c512_20, %c1_23]) : (memref<64x64xbf16, 2 : i32>)
              memref.dealloc %alloc_26 : memref<64x64xbf16, 2 : i32>
              memref.dealloc %alloc_27 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_28 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_29 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_30 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_31 : memref<64x1xbf16, 2 : i32>
              memref.dealloc %alloc_32 : memref<64x1xbf16, 2 : i32>
            }
          }
        }
        memref.dealloc %alloc_13 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_14 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_15 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_16 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_17 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_18 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc_19 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_9 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_10 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_11 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_12 : memref<256x64xbf16, 1 : i32>
      }
    }
    return
  }
}
