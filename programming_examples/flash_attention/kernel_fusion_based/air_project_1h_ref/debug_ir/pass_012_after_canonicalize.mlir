#map = affine_map<()[s0] -> (s0 * 16384)>
#map1 = affine_map<()[s0] -> (s0 * 64)>
#set = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 == 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 1 == 0)>
#set2 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 2 == 0)>
#set3 = affine_set<()[s0, s1] : (s0 >= 0, s1 == 0)>
#set4 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 1 == 0)>
#set5 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 2 == 0)>
#set6 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 3 == 0)>
#set7 = affine_set<()[s0, s1] : (s1 - 1 >= 0, -s1 + 2 >= 0, s0 >= 0, -s0 + 3 >= 0)>
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
  air.channel @QK2L1_0 [1, 1] {broadcast_shape = [4 : index, 1]}
  air.channel @QK2L1_1 [1, 1] {broadcast_shape = [4 : index, 1]}
  air.channel @QK2L1_2 [1, 1] {broadcast_shape = [4 : index, 1]}
  air.channel @QK2L1_3 [1, 1] {broadcast_shape = [4 : index, 1]}
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
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c2, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<512x64xbf16>, memref<512x64xbf16>, memref<512x64xbf16>, memref<512x64xbf16> attributes {id = 1 : i32} {
      %c24576 = arith.constant 24576 : index
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c2_0 = arith.constant 2 : index
      %c1_1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %c4096 = arith.constant 4096 : index
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %1 = affine.apply #map()[%arg4]
      %2 = air.channel.put async  @QK2L1_0[%c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %1] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 1 : i32} : (memref<512x64xbf16>)
      %3 = air.channel.put async  @QK2L1_1[%c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %1] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 2 : i32} : (memref<512x64xbf16>)
      %4 = air.channel.put async  @QK2L1_2[%c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %1] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 3 : i32} : (memref<512x64xbf16>)
      %5 = air.channel.put async  @QK2L1_3[%c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %1] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 4 : i32} : (memref<512x64xbf16>)
      %6 = air.channel.put async  @QK2L1_0[%c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %c0] [%c2_0, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 5 : i32} : (memref<512x64xbf16>)
      %7 = air.channel.put async  @QK2L1_1[%c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %c8192] [%c2_0, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 6 : i32} : (memref<512x64xbf16>)
      %8 = air.channel.put async  @QK2L1_2[%c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %c16384] [%c2_0, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 7 : i32} : (memref<512x64xbf16>)
      %9 = air.channel.put async  @QK2L1_3[%c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %c24576] [%c2_0, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_1]) {id = 8 : i32} : (memref<512x64xbf16>)
      %10 = air.channel.put async  @VIn_0[%c0] (%arg10[%c0, %c0, %c0] [%c2_0, %c64, %c64] [%c4096, %c64, %c1_1]) {id = 9 : i32} : (memref<512x64xbf16>)
      %11 = air.channel.put async  @VIn_1[%c0] (%arg10[%c0, %c0, %c8192] [%c2_0, %c64, %c64] [%c4096, %c64, %c1_1]) {id = 10 : i32} : (memref<512x64xbf16>)
      %12 = air.channel.put async  @VIn_2[%c0] (%arg10[%c0, %c0, %c16384] [%c2_0, %c64, %c64] [%c4096, %c64, %c1_1]) {id = 11 : i32} : (memref<512x64xbf16>)
      %13 = air.channel.put async  @VIn_3[%c0] (%arg10[%c0, %c0, %c24576] [%c2_0, %c64, %c64] [%c4096, %c64, %c1_1]) {id = 12 : i32} : (memref<512x64xbf16>)
      %14 = air.channel.get async  @GpOut[%c0] (%arg11[%1] [%c16384] [%c1_1]) {id = 13 : i32} : (memref<512x64xbf16>)
      %15 = air.segment @attn_seg async  unroll(%arg12, %arg13) in (%arg14=%c1_1, %arg15=%c1_1) attributes {id = 2 : i32} {
        %c64_2 = arith.constant 64 : index
        %c512_3 = arith.constant 512 : index
        %c8_4 = arith.constant 8 : index
        %c1_5 = arith.constant 1 : index
        %c2_6 = arith.constant 2 : index
        %c0_7 = arith.constant 0 : index
        %c4_8 = arith.constant 4 : index
        %async_token, %results = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_9, %results_10 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_11, %results_12 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_13, %results_14 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_15, %results_16 = air.execute -> (memref<256x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<256x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<256x64xbf16, 1 : i32>
        }
        %async_token_17, %results_18 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_19, %results_20 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_21, %results_22 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_23, %results_24 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_25, %results_26 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_27, %results_28 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        }
        %async_token_29, %results_30 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        }
        %16 = scf.for %arg16 = %c0_7 to %c2_6 step %c1_5 iter_args(%arg17 = %async_token) -> (!air.async.token) {
          %23 = air.channel.get async [%arg17]  @VIn_0[%c0_7] (%results[] [] []) {id = 14 : i32} : (memref<64x64xbf16, 1 : i32>)
          %24 = air.channel.put async [%23]  @V2L1_0[%c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_3, %c64_2, %c1_5]) {id = 15 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %24 : !air.async.token
        }
        %17 = scf.for %arg16 = %c0_7 to %c2_6 step %c1_5 iter_args(%arg17 = %async_token_9) -> (!air.async.token) {
          %23 = air.channel.get async [%arg17]  @VIn_1[%c0_7] (%results_10[] [] []) {id = 16 : i32} : (memref<64x64xbf16, 1 : i32>)
          %24 = air.channel.put async [%23]  @V2L1_1[%c0_7, %c0_7] (%results_10[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_3, %c64_2, %c1_5]) {id = 17 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %24 : !air.async.token
        }
        %18 = scf.for %arg16 = %c0_7 to %c2_6 step %c1_5 iter_args(%arg17 = %async_token_11) -> (!air.async.token) {
          %23 = air.channel.get async [%arg17]  @VIn_2[%c0_7] (%results_12[] [] []) {id = 18 : i32} : (memref<64x64xbf16, 1 : i32>)
          %24 = air.channel.put async [%23]  @V2L1_2[%c0_7, %c0_7] (%results_12[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_3, %c64_2, %c1_5]) {id = 19 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %24 : !air.async.token
        }
        %19 = scf.for %arg16 = %c0_7 to %c2_6 step %c1_5 iter_args(%arg17 = %async_token_13) -> (!air.async.token) {
          %23 = air.channel.get async [%arg17]  @VIn_3[%c0_7] (%results_14[] [] []) {id = 20 : i32} : (memref<64x64xbf16, 1 : i32>)
          %24 = air.channel.put async [%23]  @V2L1_3[%c0_7, %c0_7] (%results_14[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_3, %c64_2, %c1_5]) {id = 21 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %24 : !air.async.token
        }
        %20 = scf.parallel (%arg16) = (%c0_7) to (%c4_8) step (%c1_5) init (%async_token_15) -> !air.async.token {
          %23 = affine.apply #map1()[%arg16]
          %24 = air.channel.get async [%async_token_15]  @Gp2L2[%arg16, %c0_7] (%results_16[%23, %c0_7] [%c64_2, %c64_2] [%c64_2, %c1_5]) {id = 22 : i32} : (memref<256x64xbf16, 1 : i32>)
          scf.reduce(%24 : !air.async.token) {
          ^bb0(%arg17: !air.async.token, %arg18: !air.async.token):
            %25 = air.wait_all async [%arg17, %arg18] 
            scf.reduce.return %25 : !air.async.token
          }
        }
        %21 = air.channel.put async [%20]  @GpOut[%c0_7] (%results_16[] [] []) {id = 23 : i32} : (memref<256x64xbf16, 1 : i32>)
        %22 = air.herd @herd_0 async [%async_token_17, %async_token_19, %async_token_21, %async_token_23, %async_token_25, %async_token_27, %async_token_29]  tile (%arg16, %arg17) in (%arg18=%c4_8, %arg19=%c4_8) args(%arg20=%results_18, %arg21=%results_20, %arg22=%results_22, %arg23=%results_24, %arg24=%results_26, %arg25=%results_28, %arg26=%results_30) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32> attributes {id = 3 : i32, link_with = "attn.o"} {
          %c512_43 = arith.constant 512 : index
          %c64_44 = arith.constant 64 : index
          %c8_45 = arith.constant 8 : index
          %c1_46 = arith.constant 1 : index
          %c0_47 = arith.constant 0 : index
          %c2_48 = arith.constant 2 : index
          %c3_i32 = arith.constant 3 : i32
          %c2_i32 = arith.constant 2 : i32
          %c1_i32 = arith.constant 1 : i32
          %c0_i32 = arith.constant 0 : i32
          %async_token_49 = air.execute {
            func.call @zero_fill_gp_bf16(%arg24) : (memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_50 = air.execute {
            func.call @zero_fill_sp_bf16(%arg26) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_51 = air.execute {
            func.call @neg_inf_fill_up_bf16(%arg25) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %23 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %35 = air.channel.get async  @QK2L1_0[%arg16, %c0_47] (%arg21[] [] []) {id = 24 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %35 : !air.async.token
          } else {
            %35 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %36 = air.channel.get async  @QK2L1_1[%arg16, %c0_47] (%arg21[] [] []) {id = 25 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %36 : !air.async.token
            } else {
              %36 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %37 = air.channel.get async  @QK2L1_2[%arg16, %c0_47] (%arg21[] [] []) {id = 26 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %37 : !air.async.token
              } else {
                %37 = air.channel.get async  @QK2L1_3[%arg16, %c0_47] (%arg21[] [] []) {id = 27 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %37 : !air.async.token
              }
              affine.yield %36 : !air.async.token
            }
            affine.yield %35 : !air.async.token
          }
          %24 = arith.index_cast %arg16 : index to i32
          %25 = arith.cmpi eq, %24, %c0_i32 : i32
          scf.if %25 {
            %async_token_52 = air.execute [%23] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %26 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %35 = air.channel.get async  @QK2L1_0[%arg16, %c0_47] (%arg21[] [] []) {id = 28 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %35 : !air.async.token
          } else {
            %35 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %36 = air.channel.get async  @QK2L1_1[%arg16, %c0_47] (%arg21[] [] []) {id = 29 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %36 : !air.async.token
            } else {
              %36 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %37 = air.channel.get async  @QK2L1_2[%arg16, %c0_47] (%arg21[] [] []) {id = 30 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %37 : !air.async.token
              } else {
                %37 = air.channel.get async  @QK2L1_3[%arg16, %c0_47] (%arg21[] [] []) {id = 31 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %37 : !air.async.token
              }
              affine.yield %36 : !air.async.token
            }
            affine.yield %35 : !air.async.token
          }
          %27 = arith.cmpi eq, %24, %c1_i32 : i32
          scf.if %27 {
            %async_token_52 = air.execute [%26] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %28 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %35 = air.channel.get async  @QK2L1_0[%arg16, %c0_47] (%arg21[] [] []) {id = 32 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %35 : !air.async.token
          } else {
            %35 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %36 = air.channel.get async  @QK2L1_1[%arg16, %c0_47] (%arg21[] [] []) {id = 33 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %36 : !air.async.token
            } else {
              %36 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %37 = air.channel.get async  @QK2L1_2[%arg16, %c0_47] (%arg21[] [] []) {id = 34 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %37 : !air.async.token
              } else {
                %37 = air.channel.get async  @QK2L1_3[%arg16, %c0_47] (%arg21[] [] []) {id = 35 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %37 : !air.async.token
              }
              affine.yield %36 : !air.async.token
            }
            affine.yield %35 : !air.async.token
          }
          %29 = arith.cmpi eq, %24, %c2_i32 : i32
          scf.if %29 {
            %async_token_52 = air.execute [%28] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %30 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %35 = air.channel.get async  @QK2L1_0[%arg16, %c0_47] (%arg21[] [] []) {id = 36 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %35 : !air.async.token
          } else {
            %35 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %36 = air.channel.get async  @QK2L1_1[%arg16, %c0_47] (%arg21[] [] []) {id = 37 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %36 : !air.async.token
            } else {
              %36 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %37 = air.channel.get async  @QK2L1_2[%arg16, %c0_47] (%arg21[] [] []) {id = 38 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %37 : !air.async.token
              } else {
                %37 = air.channel.get async  @QK2L1_3[%arg16, %c0_47] (%arg21[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %37 : !air.async.token
              }
              affine.yield %36 : !air.async.token
            }
            affine.yield %35 : !air.async.token
          }
          %31 = arith.cmpi eq, %24, %c3_i32 : i32
          scf.if %31 {
            %async_token_52 = air.execute [%30] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %32 = air.wait_all async [%async_token_49, %async_token_50, %async_token_51] 
          %33 = scf.for %arg27 = %c0_47 to %c2_48 step %c1_46 iter_args(%arg28 = %32) -> (!air.async.token) {
            %async_token_52 = air.execute [%arg28] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            }
            %35 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %41 = air.channel.get async [%arg28]  @QK2L1_0[%arg16, %c0_47] (%arg21[] [] []) {id = 40 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %41 : !air.async.token
            } else {
              %41 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %42 = air.channel.get async [%arg28]  @QK2L1_1[%arg16, %c0_47] (%arg21[] [] []) {id = 41 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %42 : !air.async.token
              } else {
                %42 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %43 = air.channel.get async [%arg28]  @QK2L1_2[%arg16, %c0_47] (%arg21[] [] []) {id = 42 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %43 : !air.async.token
                } else {
                  %43 = air.channel.get async [%arg28]  @QK2L1_3[%arg16, %c0_47] (%arg21[] [] []) {id = 43 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %43 : !air.async.token
                }
                affine.yield %42 : !air.async.token
              }
              affine.yield %41 : !air.async.token
            }
            %36 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
              %41 = air.channel.get async  @V2L1_0[%arg16, %arg17] (%arg22[] [] []) {id = 44 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %41 : !air.async.token
            } else {
              %41 = air.wait_all async 
              affine.yield %41 : !air.async.token
            }
            %37 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
              %41 = air.channel.get async [%36, %arg28]  @V2L1_1[%arg16, %arg17] (%arg22[] [] []) {id = 45 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %41 : !air.async.token
            } else {
              %41 = air.wait_all async 
              affine.yield %41 : !air.async.token
            }
            %38 = affine.if #set5()[%arg16, %arg17] -> !air.async.token {
              %41 = air.channel.get async [%37]  @V2L1_2[%arg16, %arg17] (%arg22[] [] []) {id = 46 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %41 : !air.async.token
            } else {
              %41 = air.wait_all async 
              affine.yield %41 : !air.async.token
            }
            %39 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
              %41 = air.channel.get async [%38]  @V2L1_3[%arg16, %arg17] (%arg22[] [] []) {id = 47 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %41 : !air.async.token
            } else {
              %41 = air.wait_all async 
              affine.yield %41 : !air.async.token
            }
            %async_token_53 = air.execute [%async_token_52, %35] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_a_b_bf16(%arg20, %arg21, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            }
            %async_token_54, %results_55 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_56, %results_57 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_58 = air.execute [%async_token_56, %async_token_54, %async_token_53] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @fused_softmax(%collapse_shape, %arg25, %results_55, %results_57) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_59 = air.execute [%async_token_58] {
              func.call @mul_r_gp(%results_57, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_60 = air.execute [%39, %async_token_59] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_g_b_bf16(%collapse_shape, %arg22, %arg24) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_61 = air.execute [%async_token_59] {
              func.call @accum_sp_r_s(%arg26, %results_57, %results_55) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_62 = air.execute [%async_token_61] {
              func.call @vector_copy_32elems(%c0_i32, %results_55, %arg26) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_63 = air.execute [%async_token_62] {
              memref.dealloc %results_55 : memref<64x1xbf16, 2 : i32>
            }
            %async_token_64 = air.execute [%async_token_61] {
              memref.dealloc %results_57 : memref<64x1xbf16, 2 : i32>
            }
            %40 = air.wait_all async [%async_token_60, %async_token_62] 
            scf.yield %40 : !air.async.token
          }
          %34 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
            %35 = arith.subi %arg17, %c1_46 : index
            %36 = air.channel.put async [%33]  @cascade_gp[%arg16, %35] (%arg24[] [] []) {id = 48 : i32} : (memref<64x64xbf16, 2 : i32>)
            %37 = air.channel.put async [%33]  @cascade_up[%arg16, %35] (%arg25[] [] []) {id = 49 : i32} : (memref<64x1xbf16, 2 : i32>)
            %38 = air.channel.put async [%33]  @cascade_sp[%arg16, %35] (%arg26[] [] []) {id = 50 : i32} : (memref<64x1xbf16, 2 : i32>)
            %39 = air.wait_all async [%36, %37, %38] 
            affine.yield %39 : !air.async.token
          } else {
            %35 = affine.if #set7()[%arg16, %arg17] -> !air.async.token {
              %async_token_52, %results_53 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_54, %results_55 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_56, %results_57 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %36 = air.channel.get async [%async_token_52]  @cascade_gp[%arg16, %arg17] (%results_53[] [] []) {id = 51 : i32} : (memref<64x64xbf16, 2 : i32>)
              %37 = air.channel.get async [%async_token_54]  @cascade_up[%arg16, %arg17] (%results_55[] [] []) {id = 52 : i32} : (memref<64x1xbf16, 2 : i32>)
              %38 = air.channel.get async [%async_token_56]  @cascade_sp[%arg16, %arg17] (%results_57[] [] []) {id = 53 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_58, %results_59 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_60 = air.execute [%async_token_58, %33] {
                func.call @vector_copy_32elems(%c0_i32, %arg25, %results_59) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_61 = air.execute [%async_token_60, %37] {
                func.call @maximum_up_u_bf16(%results_55, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_62, %results_63 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_64 = air.execute [%async_token_62, %async_token_61] {
                func.call @exp_up_minus_u(%results_55, %arg25, %results_63) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_65, %results_66 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_67 = air.execute [%async_token_65, %async_token_64] {
                func.call @exp_up_minus_u(%results_59, %arg25, %results_66) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_68 = air.execute [%async_token_64, %36] {
                func.call @mul_r_gp(%results_63, %results_53) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_69 = air.execute [%async_token_67] {
                func.call @mul_r_gp(%results_66, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_70 = air.execute [%async_token_69, %async_token_68] {
                func.call @add_gp_g(%arg24, %results_53) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_71, %results_72 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_73 = air.execute [%async_token_71] {
                func.call @zero_fill_sp_bf16(%results_72) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_74 = air.execute [%async_token_73, %async_token_68, %38] {
                func.call @accum_sp_r_s(%results_57, %results_63, %results_72) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_75 = air.execute [%async_token_74, %async_token_69] {
                func.call @accum_sp_r_s(%arg26, %results_66, %results_72) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_76 = air.execute [%async_token_75] {
                func.call @vector_copy_32elems(%c0_i32, %results_72, %results_57) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %39 = arith.subi %arg17, %c1_46 : index
              %40 = air.channel.put async [%async_token_70]  @cascade_gp[%arg16, %39] (%results_53[] [] []) {id = 54 : i32} : (memref<64x64xbf16, 2 : i32>)
              %41 = air.channel.put async [%async_token_67]  @cascade_up[%arg16, %39] (%arg25[] [] []) {id = 55 : i32} : (memref<64x1xbf16, 2 : i32>)
              %42 = air.channel.put async [%async_token_76]  @cascade_sp[%arg16, %39] (%results_57[] [] []) {id = 56 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_77 = air.execute [%40] {
                memref.dealloc %results_53 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_78 = air.execute [%async_token_64] {
                memref.dealloc %results_55 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_79 = air.execute [%42] {
                memref.dealloc %results_57 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_80 = air.execute [%async_token_67] {
                memref.dealloc %results_59 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_81 = air.execute [%async_token_74] {
                memref.dealloc %results_63 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_82 = air.execute [%async_token_75] {
                memref.dealloc %results_66 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_83 = air.execute [%async_token_76] {
                memref.dealloc %results_72 : memref<64x1xbf16, 2 : i32>
              }
              %43 = air.wait_all async [%40, %41, %42] 
              affine.yield %43 : !air.async.token
            } else {
              %async_token_52, %results_53 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_54, %results_55 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_56, %results_57 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %36 = air.channel.get async [%async_token_52]  @cascade_gp[%arg16, %arg17] (%results_53[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 2 : i32>)
              %37 = air.channel.get async [%async_token_54]  @cascade_up[%arg16, %arg17] (%results_55[] [] []) {id = 58 : i32} : (memref<64x1xbf16, 2 : i32>)
              %38 = air.channel.get async [%async_token_56]  @cascade_sp[%arg16, %arg17] (%results_57[] [] []) {id = 59 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_58, %results_59 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_60 = air.execute [%async_token_58, %33] {
                func.call @vector_copy_32elems(%c0_i32, %arg25, %results_59) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_61 = air.execute [%async_token_60, %37] {
                func.call @maximum_up_u_bf16(%results_55, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_62, %results_63 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_64 = air.execute [%async_token_62, %async_token_61] {
                func.call @exp_up_minus_u(%results_55, %arg25, %results_63) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_65, %results_66 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_67 = air.execute [%async_token_65, %async_token_64] {
                func.call @exp_up_minus_u(%results_59, %arg25, %results_66) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_68 = air.execute [%async_token_64, %36] {
                func.call @mul_r_gp(%results_63, %results_53) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_69 = air.execute [%async_token_67] {
                func.call @mul_r_gp(%results_66, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_70 = air.execute [%async_token_69, %async_token_68] {
                func.call @add_gp_g(%arg24, %results_53) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_71, %results_72 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_73 = air.execute [%async_token_71] {
                func.call @zero_fill_sp_bf16(%results_72) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_74 = air.execute [%async_token_73, %async_token_68, %38] {
                func.call @accum_sp_r_s(%results_57, %results_63, %results_72) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_75 = air.execute [%async_token_74, %async_token_69] {
                func.call @accum_sp_r_s(%arg26, %results_66, %results_72) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_76 = air.execute [%async_token_75] {
                func.call @vector_copy_32elems(%c0_i32, %results_72, %results_57) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_77 = air.execute [%async_token_76, %async_token_70] {
                func.call @div_gp_sp(%results_57, %results_53) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %39 = air.channel.put async [%async_token_77]  @Gp2L2[%arg16, %c0_47] (%results_53[%c0_47, %c0_47, %c0_47, %c0_47] [%c8_45, %c8_45, %c8_45, %c8_45] [%c64_44, %c8_45, %c512_43, %c1_46]) {id = 60 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_78 = air.execute [%39] {
                memref.dealloc %results_53 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_79 = air.execute [%async_token_64] {
                memref.dealloc %results_55 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_80 = air.execute [%async_token_77] {
                memref.dealloc %results_57 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_81 = air.execute [%async_token_67] {
                memref.dealloc %results_59 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_82 = air.execute [%async_token_74] {
                memref.dealloc %results_63 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_83 = air.execute [%async_token_75] {
                memref.dealloc %results_66 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_84 = air.execute [%async_token_76] {
                memref.dealloc %results_72 : memref<64x1xbf16, 2 : i32>
              }
              affine.yield %39 : !air.async.token
            }
            affine.yield %33 : !air.async.token
          }
        }
        %async_token_31 = air.execute [%22] {
          memref.dealloc %results_18 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_32 = air.execute [%22] {
          memref.dealloc %results_20 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_33 = air.execute [%22] {
          memref.dealloc %results_22 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_34 = air.execute [%22] {
          memref.dealloc %results_24 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_35 = air.execute [%22] {
          memref.dealloc %results_26 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_36 = air.execute [%22] {
          memref.dealloc %results_28 : memref<64x1xbf16, 2 : i32>
        }
        %async_token_37 = air.execute [%22] {
          memref.dealloc %results_30 : memref<64x1xbf16, 2 : i32>
        }
        %async_token_38 = air.execute [%16] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        }
        %async_token_39 = air.execute [%17] {
          memref.dealloc %results_10 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_40 = air.execute [%18] {
          memref.dealloc %results_12 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_41 = air.execute [%19] {
          memref.dealloc %results_14 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_42 = air.execute [%21] {
          memref.dealloc %results_16 : memref<256x64xbf16, 1 : i32>
        }
      }
    }
    return
  }
}
