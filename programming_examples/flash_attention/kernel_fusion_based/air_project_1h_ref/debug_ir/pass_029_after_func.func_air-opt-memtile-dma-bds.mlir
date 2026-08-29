#map = affine_map<()[s0] -> (s0 * 16384)>
#set = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 == 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 1 == 0)>
#set2 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 2 == 0)>
#set3 = affine_set<()[s0, s1] : (s0 >= 0, s1 == 0)>
#set4 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 1 == 0)>
#set5 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 2 == 0)>
#set6 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 3 == 0)>
#set7 = affine_set<()[s0, s1] : (s1 - 1 >= 0, -s1 + 2 >= 0, s0 >= 0, -s0 + 3 >= 0)>
module {
  air.channel @channel_0 [4, 1]
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
  func.func @attention_bf16(%arg0: memref<512x64xbf16>, %arg1: memref<512x64xbf16>, %arg2: memref<512x64xbf16>, %arg3: memref<512x64xbf16>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c2, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<512x64xbf16>, memref<512x64xbf16>, memref<512x64xbf16>, memref<512x64xbf16> attributes {id = 1 : i32} {
      %c192 = arith.constant 192 : index
      %c128 = arith.constant 128 : index
      %c3 = arith.constant 3 : index
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
      %14 = air.channel.get async  @channel_0[%c0, %c0] (%arg11[%c0, %1] [%c64, %c64] [%c64, %c1_1]) {id = 13 : i32} : (memref<512x64xbf16>)
      %15 = air.channel.get async  @channel_0[%c1_1, %c0] (%arg11[%c64, %1] [%c64, %c64] [%c64, %c1_1]) {id = 14 : i32} : (memref<512x64xbf16>)
      %16 = air.channel.get async  @channel_0[%c2_0, %c0] (%arg11[%c128, %1] [%c64, %c64] [%c64, %c1_1]) {id = 15 : i32} : (memref<512x64xbf16>)
      %17 = air.channel.get async  @channel_0[%c3, %c0] (%arg11[%c192, %1] [%c64, %c64] [%c64, %c1_1]) {id = 16 : i32} : (memref<512x64xbf16>)
      %18 = air.segment @attn_seg async  unroll(%arg12, %arg13) in (%arg14=%c1_1, %arg15=%c1_1) attributes {id = 2 : i32} {
        %c3_2 = arith.constant 3 : index
        %c64_3 = arith.constant 64 : index
        %c8_4 = arith.constant 8 : index
        %c1_5 = arith.constant 1 : index
        %c2_6 = arith.constant 2 : index
        %c0_7 = arith.constant 0 : index
        %c4_8 = arith.constant 4 : index
        %19 = air.wait_all async 
        %20 = air.wait_all async 
        %21 = air.wait_all async 
        %22 = air.wait_all async 
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
        %23 = scf.for %arg16 = %c0_7 to %c2_6 step %c1_5 iter_args(%arg17 = %19) -> (!air.async.token) {
          %async_token_19, %results_20 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          } {hoist_alloc = true}
          %36 = air.channel.get async [%async_token_19, %arg17]  @VIn_0[%c0_7] (%results_20[] [] []) {id = 17 : i32} : (memref<64x64xbf16, 1 : i32>)
          %37 = air.channel.put async [%async_token_19, %36]  @V2L1_0[%c0_7, %c0_7] (%results_20[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 18 : i32} : (memref<64x64xbf16, 1 : i32>)
          %async_token_21 = air.execute [%37] {
            memref.dealloc %results_20 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %37 : !air.async.token
        }
        %24 = scf.for %arg16 = %c0_7 to %c2_6 step %c1_5 iter_args(%arg17 = %20) -> (!air.async.token) {
          %async_token_19, %results_20 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          } {hoist_alloc = true}
          %36 = air.channel.get async [%async_token_19, %arg17]  @VIn_1[%c0_7] (%results_20[] [] []) {id = 19 : i32} : (memref<64x64xbf16, 1 : i32>)
          %37 = air.channel.put async [%async_token_19, %36]  @V2L1_1[%c0_7, %c0_7] (%results_20[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 20 : i32} : (memref<64x64xbf16, 1 : i32>)
          %async_token_21 = air.execute [%37] {
            memref.dealloc %results_20 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %37 : !air.async.token
        }
        %25 = scf.for %arg16 = %c0_7 to %c2_6 step %c1_5 iter_args(%arg17 = %21) -> (!air.async.token) {
          %async_token_19, %results_20 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          } {hoist_alloc = true}
          %36 = air.channel.get async [%async_token_19, %arg17]  @VIn_2[%c0_7] (%results_20[] [] []) {id = 21 : i32} : (memref<64x64xbf16, 1 : i32>)
          %37 = air.channel.put async [%async_token_19, %36]  @V2L1_2[%c0_7, %c0_7] (%results_20[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 22 : i32} : (memref<64x64xbf16, 1 : i32>)
          %async_token_21 = air.execute [%37] {
            memref.dealloc %results_20 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %37 : !air.async.token
        }
        %26 = scf.for %arg16 = %c0_7 to %c2_6 step %c1_5 iter_args(%arg17 = %22) -> (!air.async.token) {
          %async_token_19, %results_20 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          } {hoist_alloc = true}
          %36 = air.channel.get async [%async_token_19, %arg17]  @VIn_3[%c0_7] (%results_20[] [] []) {id = 23 : i32} : (memref<64x64xbf16, 1 : i32>)
          %37 = air.channel.put async [%async_token_19, %36]  @V2L1_3[%c0_7, %c0_7] (%results_20[%c0_7, %c0_7, %c0_7] [%c8_4, %c64_3, %c8_4] [%c8_4, %c64_3, %c1_5]) {id = 24 : i32} : (memref<64x64xbf16, 1 : i32>)
          %async_token_21 = air.execute [%37] {
            memref.dealloc %results_20 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %37 : !air.async.token
        }
        %27 = air.channel.get async [%async_token]  @Gp2L2[%c0_7, %c0_7] (%results[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %28 = air.channel.get async [%async_token_9]  @Gp2L2[%c1_5, %c0_7] (%results_10[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %29 = air.channel.get async [%async_token_11]  @Gp2L2[%c2_6, %c0_7] (%results_12[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %30 = air.channel.get async [%async_token_13]  @Gp2L2[%c3_2, %c0_7] (%results_14[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %31 = air.channel.put async [%27]  @channel_0[%c0_7, %c0_7] (%results[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %32 = air.channel.put async [%28]  @channel_0[%c1_5, %c0_7] (%results_10[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %33 = air.channel.put async [%29]  @channel_0[%c2_6, %c0_7] (%results_12[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %34 = air.channel.put async [%30]  @channel_0[%c3_2, %c0_7] (%results_14[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %35 = air.herd @herd_0 async  tile (%arg16, %arg17) in (%arg18=%c4_8, %arg19=%c4_8) attributes {id = 3 : i32, link_with = "attn.o"} {
          %c64_19 = arith.constant 64 : index
          %c0_i32 = arith.constant 0 : i32
          %c1_i32 = arith.constant 1 : i32
          %c2_i32 = arith.constant 2 : i32
          %c3_i32 = arith.constant 3 : i32
          %c2_20 = arith.constant 2 : index
          %c0_21 = arith.constant 0 : index
          %c1_22 = arith.constant 1 : index
          %c8_23 = arith.constant 8 : index
          %c512_24 = arith.constant 512 : index
          %async_token_25, %results_26 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_27, %results_28 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_29, %results_30 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_31, %results_32 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_33, %results_34 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_35 = air.execute [%async_token_29] {
            func.call @zero_fill_gp_bf16(%results_30) : (memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_36 = air.execute [%async_token_25] {
            func.call @zero_fill_sp_bf16(%results_26) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_37 = air.execute [%async_token_27] {
            func.call @neg_inf_fill_up_bf16(%results_28) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %36 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %48 = air.channel.get async [%async_token_31]  @QK2L1_0[%arg16, %c0_21] (%results_32[] [] []) {id = 33 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %48 : !air.async.token
          } else {
            %48 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %49 = air.channel.get async [%async_token_31]  @QK2L1_1[%arg16, %c0_21] (%results_32[] [] []) {id = 34 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %49 : !air.async.token
            } else {
              %49 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %50 = air.channel.get async [%async_token_31]  @QK2L1_2[%arg16, %c0_21] (%results_32[] [] []) {id = 35 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %50 : !air.async.token
              } else {
                %50 = air.channel.get async [%async_token_31]  @QK2L1_3[%arg16, %c0_21] (%results_32[] [] []) {id = 36 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %50 : !air.async.token
              }
              affine.yield %49 : !air.async.token
            }
            affine.yield %48 : !air.async.token
          }
          %37 = arith.index_cast %arg16 : index to i32
          %38 = arith.cmpi eq, %37, %c0_i32 : i32
          scf.if %38 {
            %async_token_43 = air.execute [%async_token_31, %async_token_33, %36] {
              func.call @copy_tile(%results_32, %results_34) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %39 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %48 = air.channel.get async [%async_token_31]  @QK2L1_0[%arg16, %c0_21] (%results_32[] [] []) {id = 37 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %48 : !air.async.token
          } else {
            %48 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %49 = air.channel.get async [%async_token_31]  @QK2L1_1[%arg16, %c0_21] (%results_32[] [] []) {id = 38 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %49 : !air.async.token
            } else {
              %49 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %50 = air.channel.get async [%async_token_31]  @QK2L1_2[%arg16, %c0_21] (%results_32[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %50 : !air.async.token
              } else {
                %50 = air.channel.get async [%async_token_31]  @QK2L1_3[%arg16, %c0_21] (%results_32[] [] []) {id = 40 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %50 : !air.async.token
              }
              affine.yield %49 : !air.async.token
            }
            affine.yield %48 : !air.async.token
          }
          %40 = arith.cmpi eq, %37, %c1_i32 : i32
          scf.if %40 {
            %async_token_43 = air.execute [%async_token_31, %async_token_33, %39] {
              func.call @copy_tile(%results_32, %results_34) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %41 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %48 = air.channel.get async [%async_token_31]  @QK2L1_0[%arg16, %c0_21] (%results_32[] [] []) {id = 41 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %48 : !air.async.token
          } else {
            %48 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %49 = air.channel.get async [%async_token_31]  @QK2L1_1[%arg16, %c0_21] (%results_32[] [] []) {id = 42 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %49 : !air.async.token
            } else {
              %49 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %50 = air.channel.get async [%async_token_31]  @QK2L1_2[%arg16, %c0_21] (%results_32[] [] []) {id = 43 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %50 : !air.async.token
              } else {
                %50 = air.channel.get async [%async_token_31]  @QK2L1_3[%arg16, %c0_21] (%results_32[] [] []) {id = 44 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %50 : !air.async.token
              }
              affine.yield %49 : !air.async.token
            }
            affine.yield %48 : !air.async.token
          }
          %42 = arith.cmpi eq, %37, %c2_i32 : i32
          scf.if %42 {
            %async_token_43 = air.execute [%async_token_31, %async_token_33, %41] {
              func.call @copy_tile(%results_32, %results_34) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %43 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %48 = air.channel.get async [%async_token_31]  @QK2L1_0[%arg16, %c0_21] (%results_32[] [] []) {id = 45 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %48 : !air.async.token
          } else {
            %48 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %49 = air.channel.get async [%async_token_31]  @QK2L1_1[%arg16, %c0_21] (%results_32[] [] []) {id = 46 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %49 : !air.async.token
            } else {
              %49 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %50 = air.channel.get async [%async_token_31]  @QK2L1_2[%arg16, %c0_21] (%results_32[] [] []) {id = 47 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %50 : !air.async.token
              } else {
                %50 = air.channel.get async [%async_token_31]  @QK2L1_3[%arg16, %c0_21] (%results_32[] [] []) {id = 48 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %50 : !air.async.token
              }
              affine.yield %49 : !air.async.token
            }
            affine.yield %48 : !air.async.token
          }
          %44 = arith.cmpi eq, %37, %c3_i32 : i32
          scf.if %44 {
            %async_token_43 = air.execute [%async_token_31, %async_token_33, %43] {
              func.call @copy_tile(%results_32, %results_34) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %45 = air.wait_all async [%async_token_31, %async_token_33, %async_token_35, %async_token_36, %async_token_37] 
          %46 = scf.for %arg20 = %c0_21 to %c2_20 step %c1_22 iter_args(%arg21 = %45) -> (!air.async.token) {
            %async_token_43, %results_44 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
            }
            %async_token_45, %results_46 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
            }
            %async_token_47 = air.execute [%async_token_45, %arg21] {
              %collapse_shape = memref.collapse_shape %results_46 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            }
            %48 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %54 = air.channel.get async [%arg21]  @QK2L1_0[%arg16, %c0_21] (%results_32[] [] []) {id = 49 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %54 : !air.async.token
            } else {
              %54 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %55 = air.channel.get async [%arg21]  @QK2L1_1[%arg16, %c0_21] (%results_32[] [] []) {id = 50 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %55 : !air.async.token
              } else {
                %55 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %56 = air.channel.get async [%arg21]  @QK2L1_2[%arg16, %c0_21] (%results_32[] [] []) {id = 51 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %56 : !air.async.token
                } else {
                  %56 = air.channel.get async [%arg21]  @QK2L1_3[%arg16, %c0_21] (%results_32[] [] []) {id = 52 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %56 : !air.async.token
                }
                affine.yield %55 : !air.async.token
              }
              affine.yield %54 : !air.async.token
            }
            %49 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
              %54 = air.channel.get async [%async_token_43]  @V2L1_0[%arg16, %arg17] (%results_44[] [] []) {id = 53 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %54 : !air.async.token
            } else {
              %54 = air.wait_all async 
              affine.yield %54 : !air.async.token
            }
            %50 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
              %54 = air.channel.get async [%async_token_43, %49, %arg21]  @V2L1_1[%arg16, %arg17] (%results_44[] [] []) {id = 54 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %54 : !air.async.token
            } else {
              %54 = air.wait_all async 
              affine.yield %54 : !air.async.token
            }
            %51 = affine.if #set5()[%arg16, %arg17] -> !air.async.token {
              %54 = air.channel.get async [%async_token_43, %50]  @V2L1_2[%arg16, %arg17] (%results_44[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %54 : !air.async.token
            } else {
              %54 = air.wait_all async 
              affine.yield %54 : !air.async.token
            }
            %52 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
              %54 = air.channel.get async [%async_token_43, %51]  @V2L1_3[%arg16, %arg17] (%results_44[] [] []) {id = 56 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %54 : !air.async.token
            } else {
              %54 = air.wait_all async 
              affine.yield %54 : !air.async.token
            }
            %async_token_48 = air.execute [%48, %async_token_47] {
              %collapse_shape = memref.collapse_shape %results_46 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_a_b_bf16(%results_34, %results_32, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            }
            %async_token_49, %results_50 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_51, %results_52 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_53 = air.execute [%async_token_48, %async_token_49, %async_token_51] {
              %collapse_shape = memref.collapse_shape %results_46 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @fused_softmax(%collapse_shape, %results_28, %results_50, %results_52) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_54 = air.execute [%async_token_53] {
              func.call @mul_r_gp(%results_52, %results_30) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_55 = air.execute [%async_token_54, %52, %async_token_43, %async_token_45] {
              %collapse_shape = memref.collapse_shape %results_46 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_g_b_bf16(%collapse_shape, %results_44, %results_30) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_56 = air.execute [%async_token_54] {
              func.call @accum_sp_r_s(%results_26, %results_52, %results_50) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_57 = air.execute [%async_token_56] {
              func.call @vector_copy_32elems(%c0_i32, %results_50, %results_26) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_58 = air.execute [%async_token_57] {
              memref.dealloc %results_50 : memref<64x1xbf16, 2 : i32>
            }
            %async_token_59 = air.execute [%async_token_56] {
              memref.dealloc %results_52 : memref<64x1xbf16, 2 : i32>
            }
            %53 = air.wait_all async [%async_token_55, %async_token_57] 
            %async_token_60 = air.execute [%async_token_53, %async_token_55] {
              memref.dealloc %results_46 : memref<64x64xbf16, 2 : i32>
            }
            %async_token_61 = air.execute [%49, %50, %51, %async_token_55] {
              memref.dealloc %results_44 : memref<64x64xbf16, 2 : i32>
            }
            scf.yield %53 : !air.async.token
          }
          %47 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
            %48 = arith.subi %arg17, %c1_22 : index
            %49 = air.channel.put async [%async_token_29, %46]  @cascade_gp[%arg16, %48] (%results_30[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 2 : i32>)
            %50 = air.channel.put async [%async_token_27, %46]  @cascade_up[%arg16, %48] (%results_28[] [] []) {id = 58 : i32} : (memref<64x1xbf16, 2 : i32>)
            %51 = air.channel.put async [%async_token_25, %46]  @cascade_sp[%arg16, %48] (%results_26[] [] []) {id = 59 : i32} : (memref<64x1xbf16, 2 : i32>)
            %52 = air.wait_all async [%49, %50, %51] 
            affine.yield %52 : !air.async.token
          } else {
            %48 = affine.if #set7()[%arg16, %arg17] -> !air.async.token {
              %async_token_43, %results_44 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_45, %results_46 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_47, %results_48 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %49 = air.channel.get async [%async_token_43]  @cascade_gp[%arg16, %arg17] (%results_44[] [] []) {id = 60 : i32} : (memref<64x64xbf16, 2 : i32>)
              %50 = air.channel.get async [%async_token_45]  @cascade_up[%arg16, %arg17] (%results_46[] [] []) {id = 61 : i32} : (memref<64x1xbf16, 2 : i32>)
              %51 = air.channel.get async [%async_token_47]  @cascade_sp[%arg16, %arg17] (%results_48[] [] []) {id = 62 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_49, %results_50 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_51 = air.execute [%async_token_27, %async_token_49, %46] {
                func.call @vector_copy_32elems(%c0_i32, %results_28, %results_50) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_52 = air.execute [%50, %async_token_51] {
                func.call @maximum_up_u_bf16(%results_46, %results_28) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_53, %results_54 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_55 = air.execute [%async_token_52, %async_token_53] {
                func.call @exp_up_minus_u(%results_46, %results_28, %results_54) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_56, %results_57 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_58 = air.execute [%async_token_55, %async_token_56] {
                func.call @exp_up_minus_u(%results_50, %results_28, %results_57) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_59 = air.execute [%async_token_55, %49] {
                func.call @mul_r_gp(%results_54, %results_44) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_60 = air.execute [%async_token_29, %async_token_58] {
                func.call @mul_r_gp(%results_57, %results_30) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_61 = air.execute [%async_token_59, %async_token_60] {
                func.call @add_gp_g(%results_30, %results_44) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_62, %results_63 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_64 = air.execute [%async_token_62] {
                func.call @zero_fill_sp_bf16(%results_63) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_65 = air.execute [%async_token_64, %async_token_59, %51] {
                func.call @accum_sp_r_s(%results_48, %results_54, %results_63) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_66 = air.execute [%async_token_25, %async_token_65, %async_token_60] {
                func.call @accum_sp_r_s(%results_26, %results_57, %results_63) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_67 = air.execute [%async_token_66] {
                func.call @vector_copy_32elems(%c0_i32, %results_63, %results_48) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %52 = arith.subi %arg17, %c1_22 : index
              %53 = air.channel.put async [%async_token_61]  @cascade_gp[%arg16, %52] (%results_44[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 2 : i32>)
              %54 = air.channel.put async [%async_token_27, %async_token_58]  @cascade_up[%arg16, %52] (%results_28[] [] []) {id = 64 : i32} : (memref<64x1xbf16, 2 : i32>)
              %55 = air.channel.put async [%async_token_67]  @cascade_sp[%arg16, %52] (%results_48[] [] []) {id = 65 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_68 = air.execute [%53] {
                memref.dealloc %results_44 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_69 = air.execute [%async_token_55] {
                memref.dealloc %results_46 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_70 = air.execute [%55] {
                memref.dealloc %results_48 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_71 = air.execute [%async_token_58] {
                memref.dealloc %results_50 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_72 = air.execute [%async_token_65] {
                memref.dealloc %results_54 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_73 = air.execute [%async_token_66] {
                memref.dealloc %results_57 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_74 = air.execute [%async_token_67] {
                memref.dealloc %results_63 : memref<64x1xbf16, 2 : i32>
              }
              %56 = air.wait_all async [%53, %54, %55] 
              affine.yield %56 : !air.async.token
            } else {
              %async_token_43, %results_44 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_45, %results_46 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_47, %results_48 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %49 = air.channel.get async [%async_token_43]  @cascade_gp[%arg16, %arg17] (%results_44[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 2 : i32>)
              %50 = air.channel.get async [%async_token_45]  @cascade_up[%arg16, %arg17] (%results_46[] [] []) {id = 67 : i32} : (memref<64x1xbf16, 2 : i32>)
              %51 = air.channel.get async [%async_token_47]  @cascade_sp[%arg16, %arg17] (%results_48[] [] []) {id = 68 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_49, %results_50 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_51 = air.execute [%async_token_27, %async_token_49, %46] {
                func.call @vector_copy_32elems(%c0_i32, %results_28, %results_50) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_52 = air.execute [%50, %async_token_51] {
                func.call @maximum_up_u_bf16(%results_46, %results_28) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_53, %results_54 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_55 = air.execute [%async_token_52, %async_token_53] {
                func.call @exp_up_minus_u(%results_46, %results_28, %results_54) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_56, %results_57 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_58 = air.execute [%async_token_55, %async_token_56] {
                func.call @exp_up_minus_u(%results_50, %results_28, %results_57) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_59 = air.execute [%async_token_55, %49] {
                func.call @mul_r_gp(%results_54, %results_44) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_60 = air.execute [%async_token_29, %async_token_58] {
                func.call @mul_r_gp(%results_57, %results_30) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_61 = air.execute [%async_token_59, %async_token_60] {
                func.call @add_gp_g(%results_30, %results_44) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_62, %results_63 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_64 = air.execute [%async_token_62] {
                func.call @zero_fill_sp_bf16(%results_63) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_65 = air.execute [%async_token_64, %async_token_59, %51] {
                func.call @accum_sp_r_s(%results_48, %results_54, %results_63) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_66 = air.execute [%async_token_25, %async_token_65, %async_token_60] {
                func.call @accum_sp_r_s(%results_26, %results_57, %results_63) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_67 = air.execute [%async_token_66] {
                func.call @vector_copy_32elems(%c0_i32, %results_63, %results_48) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_68 = air.execute [%async_token_67, %async_token_61] {
                func.call @div_gp_sp(%results_48, %results_44) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %52 = air.channel.put async [%async_token_68]  @Gp2L2[%arg16, %c0_21] (%results_44[%c0_21, %c0_21, %c0_21] [%c64_19, %c8_23, %c8_23] [%c8_23, %c512_24, %c1_22]) {id = 69 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_69 = air.execute [%52] {
                memref.dealloc %results_44 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_70 = air.execute [%async_token_55] {
                memref.dealloc %results_46 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_71 = air.execute [%async_token_68] {
                memref.dealloc %results_48 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_72 = air.execute [%async_token_58] {
                memref.dealloc %results_50 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_73 = air.execute [%async_token_65] {
                memref.dealloc %results_54 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_74 = air.execute [%async_token_66] {
                memref.dealloc %results_57 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_75 = air.execute [%async_token_67] {
                memref.dealloc %results_63 : memref<64x1xbf16, 2 : i32>
              }
              affine.yield %52 : !air.async.token
            }
            affine.yield %46 : !air.async.token
          }
          %async_token_38 = air.execute [%46] {
            memref.dealloc %results_34 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_39 = air.execute [%46, %43, %41, %39, %36] {
            memref.dealloc %results_32 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_40 = air.execute [%47, %46, %async_token_35] {
            memref.dealloc %results_30 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_41 = air.execute [%47, %46, %async_token_37] {
            memref.dealloc %results_28 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_42 = air.execute [%47, %46, %async_token_36] {
            memref.dealloc %results_26 : memref<64x1xbf16, 2 : i32>
          }
        }
        %async_token_15 = air.execute [%34] {
          memref.dealloc %results_14 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_16 = air.execute [%33] {
          memref.dealloc %results_12 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_17 = air.execute [%32] {
          memref.dealloc %results_10 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_18 = air.execute [%31] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        }
        air.wait_all [%23, %24, %25, %26, %35, %async_token_15, %async_token_16, %async_token_17, %async_token_18]  {air.segment_end}
      }
    }
    return
  }
}
