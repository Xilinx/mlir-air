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
  func.func @attention_bf16(%arg0: memref<256x64xbf16>, %arg1: memref<512x64xbf16>, %arg2: memref<512x64xbf16>, %arg3: memref<256x64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<256x64xbf16>, memref<512x64xbf16>, memref<512x64xbf16>, memref<256x64xbf16> attributes {id = 1 : i32} {
      %c192 = arith.constant 192 : index
      %c128 = arith.constant 128 : index
      %c3 = arith.constant 3 : index
      %c24576 = arith.constant 24576 : index
      %c16384 = arith.constant 16384 : index
      %c8192 = arith.constant 8192 : index
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c512 = arith.constant 512 : index
      %c4096 = arith.constant 4096 : index
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %1 = affine.apply #map()[%arg4]
      %2 = air.channel.put async  @QK2L1_0[%c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %1] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 1 : i32} : (memref<256x64xbf16>)
      %3 = air.channel.put async  @QK2L1_1[%c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %1] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 2 : i32} : (memref<256x64xbf16>)
      %4 = air.channel.put async  @QK2L1_2[%c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %1] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 3 : i32} : (memref<256x64xbf16>)
      %5 = air.channel.put async  @QK2L1_3[%c0, %c0] (%arg8[%c0, %c0, %c0, %c0, %1] [%c4, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 4 : i32} : (memref<256x64xbf16>)
      %6 = air.channel.put async  @QK2L1_0[%c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %c0] [%c2, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 5 : i32} : (memref<512x64xbf16>)
      %7 = air.channel.put async  @QK2L1_1[%c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %c8192] [%c2, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 6 : i32} : (memref<512x64xbf16>)
      %8 = air.channel.put async  @QK2L1_2[%c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %c16384] [%c2, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 7 : i32} : (memref<512x64xbf16>)
      %9 = air.channel.put async  @QK2L1_3[%c0, %c0] (%arg9[%c0, %c0, %c0, %c0, %c24576] [%c2, %c8, %c8, %c8, %c8] [%c4096, %c8, %c512, %c64, %c1_0]) {id = 8 : i32} : (memref<512x64xbf16>)
      %10 = air.channel.put async  @VIn_0[%c0] (%arg10[%c0, %c0, %c0] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 9 : i32} : (memref<512x64xbf16>)
      %11 = air.channel.put async  @VIn_1[%c0] (%arg10[%c0, %c0, %c8192] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 10 : i32} : (memref<512x64xbf16>)
      %12 = air.channel.put async  @VIn_2[%c0] (%arg10[%c0, %c0, %c16384] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 11 : i32} : (memref<512x64xbf16>)
      %13 = air.channel.put async  @VIn_3[%c0] (%arg10[%c0, %c0, %c24576] [%c2, %c64, %c64] [%c4096, %c64, %c1_0]) {id = 12 : i32} : (memref<512x64xbf16>)
      %14 = air.channel.get async  @channel_0[%c0, %c0] (%arg11[%c0, %c0] [%c64, %c64] [%c64, %c1_0]) {id = 13 : i32} : (memref<256x64xbf16>)
      %15 = air.channel.get async  @channel_0[%c1_0, %c0] (%arg11[%c64, %c0] [%c64, %c64] [%c64, %c1_0]) {id = 14 : i32} : (memref<256x64xbf16>)
      %16 = air.channel.get async  @channel_0[%c2, %c0] (%arg11[%c128, %c0] [%c64, %c64] [%c64, %c1_0]) {id = 15 : i32} : (memref<256x64xbf16>)
      %17 = air.channel.get async  @channel_0[%c3, %c0] (%arg11[%c192, %c0] [%c64, %c64] [%c64, %c1_0]) {id = 16 : i32} : (memref<256x64xbf16>)
      %18 = air.segment @attn_seg async  unroll(%arg12, %arg13) in (%arg14=%c1_0, %arg15=%c1_0) attributes {id = 2 : i32} {
        %c3_1 = arith.constant 3 : index
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
        %async_token_15, %results_16 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_17, %results_18 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_19, %results_20 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_21, %results_22 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_23, %results_24 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_25, %results_26 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_27, %results_28 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_29, %results_30 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_31, %results_32 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_33, %results_34 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        }
        %async_token_35, %results_36 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        }
        %19 = scf.for %arg16 = %c0_7 to %c2_6 step %c1_5 iter_args(%arg17 = %async_token) -> (!air.async.token) {
          %32 = air.channel.get async [%arg17]  @VIn_0[%c0_7] (%results[] [] []) {id = 17 : i32} : (memref<64x64xbf16, 1 : i32>)
          %33 = air.channel.put async [%32]  @V2L1_0[%c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_3, %c64_2, %c1_5]) {id = 18 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %33 : !air.async.token
        }
        %20 = scf.for %arg16 = %c0_7 to %c2_6 step %c1_5 iter_args(%arg17 = %async_token_9) -> (!air.async.token) {
          %32 = air.channel.get async [%arg17]  @VIn_1[%c0_7] (%results_10[] [] []) {id = 19 : i32} : (memref<64x64xbf16, 1 : i32>)
          %33 = air.channel.put async [%32]  @V2L1_1[%c0_7, %c0_7] (%results_10[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_3, %c64_2, %c1_5]) {id = 20 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %33 : !air.async.token
        }
        %21 = scf.for %arg16 = %c0_7 to %c2_6 step %c1_5 iter_args(%arg17 = %async_token_11) -> (!air.async.token) {
          %32 = air.channel.get async [%arg17]  @VIn_2[%c0_7] (%results_12[] [] []) {id = 21 : i32} : (memref<64x64xbf16, 1 : i32>)
          %33 = air.channel.put async [%32]  @V2L1_2[%c0_7, %c0_7] (%results_12[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_3, %c64_2, %c1_5]) {id = 22 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %33 : !air.async.token
        }
        %22 = scf.for %arg16 = %c0_7 to %c2_6 step %c1_5 iter_args(%arg17 = %async_token_13) -> (!air.async.token) {
          %32 = air.channel.get async [%arg17]  @VIn_3[%c0_7] (%results_14[] [] []) {id = 23 : i32} : (memref<64x64xbf16, 1 : i32>)
          %33 = air.channel.put async [%32]  @V2L1_3[%c0_7, %c0_7] (%results_14[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_3, %c64_2, %c1_5]) {id = 24 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %33 : !air.async.token
        }
        %23 = air.channel.get async [%async_token_15]  @Gp2L2[%c0_7, %c0_7] (%results_16[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %24 = air.channel.get async [%async_token_17]  @Gp2L2[%c1_5, %c0_7] (%results_18[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %25 = air.channel.get async [%async_token_19]  @Gp2L2[%c2_6, %c0_7] (%results_20[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %26 = air.channel.get async [%async_token_21]  @Gp2L2[%c3_1, %c0_7] (%results_22[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %27 = air.channel.put async [%23]  @channel_0[%c0_7, %c0_7] (%results_16[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %28 = air.channel.put async [%24]  @channel_0[%c1_5, %c0_7] (%results_18[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %29 = air.channel.put async [%25]  @channel_0[%c2_6, %c0_7] (%results_20[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %30 = air.channel.put async [%26]  @channel_0[%c3_1, %c0_7] (%results_22[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %31 = air.herd @herd_0 async [%async_token_23, %async_token_25, %async_token_27, %async_token_29, %async_token_31, %async_token_33, %async_token_35]  tile (%arg16, %arg17) in (%arg18=%c4_8, %arg19=%c4_8) args(%arg20=%results_24, %arg21=%results_26, %arg22=%results_28, %arg23=%results_30, %arg24=%results_32, %arg25=%results_34, %arg26=%results_36) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32> attributes {id = 3 : i32, link_with = "attn.o"} {
          %c512_52 = arith.constant 512 : index
          %c64_53 = arith.constant 64 : index
          %c8_54 = arith.constant 8 : index
          %c1_55 = arith.constant 1 : index
          %c0_56 = arith.constant 0 : index
          %c2_57 = arith.constant 2 : index
          %c3_i32 = arith.constant 3 : i32
          %c2_i32 = arith.constant 2 : i32
          %c1_i32 = arith.constant 1 : i32
          %c0_i32 = arith.constant 0 : i32
          %async_token_58 = air.execute {
            func.call @zero_fill_gp_bf16(%arg24) : (memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_59 = air.execute {
            func.call @zero_fill_sp_bf16(%arg26) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_60 = air.execute {
            func.call @neg_inf_fill_up_bf16(%arg25) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %32 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %44 = air.channel.get async  @QK2L1_0[%arg16, %arg17] (%arg21[] [] []) {id = 33 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %44 : !air.async.token
          } else {
            %44 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %45 = air.channel.get async  @QK2L1_1[%arg16, %arg17] (%arg21[] [] []) {id = 34 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %45 : !air.async.token
            } else {
              %45 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %46 = air.channel.get async  @QK2L1_2[%arg16, %arg17] (%arg21[] [] []) {id = 35 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %46 : !air.async.token
              } else {
                %46 = air.channel.get async  @QK2L1_3[%arg16, %arg17] (%arg21[] [] []) {id = 36 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %46 : !air.async.token
              }
              affine.yield %45 : !air.async.token
            }
            affine.yield %44 : !air.async.token
          }
          %33 = arith.index_cast %arg16 : index to i32
          %34 = arith.cmpi eq, %33, %c0_i32 : i32
          scf.if %34 {
            %async_token_61 = air.execute [%32] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %35 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %44 = air.channel.get async  @QK2L1_0[%arg16, %arg17] (%arg21[] [] []) {id = 37 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %44 : !air.async.token
          } else {
            %44 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %45 = air.channel.get async  @QK2L1_1[%arg16, %arg17] (%arg21[] [] []) {id = 38 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %45 : !air.async.token
            } else {
              %45 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %46 = air.channel.get async  @QK2L1_2[%arg16, %arg17] (%arg21[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %46 : !air.async.token
              } else {
                %46 = air.channel.get async  @QK2L1_3[%arg16, %arg17] (%arg21[] [] []) {id = 40 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %46 : !air.async.token
              }
              affine.yield %45 : !air.async.token
            }
            affine.yield %44 : !air.async.token
          }
          %36 = arith.cmpi eq, %33, %c1_i32 : i32
          scf.if %36 {
            %async_token_61 = air.execute [%35] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %37 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %44 = air.channel.get async  @QK2L1_0[%arg16, %arg17] (%arg21[] [] []) {id = 41 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %44 : !air.async.token
          } else {
            %44 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %45 = air.channel.get async  @QK2L1_1[%arg16, %arg17] (%arg21[] [] []) {id = 42 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %45 : !air.async.token
            } else {
              %45 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %46 = air.channel.get async  @QK2L1_2[%arg16, %arg17] (%arg21[] [] []) {id = 43 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %46 : !air.async.token
              } else {
                %46 = air.channel.get async  @QK2L1_3[%arg16, %arg17] (%arg21[] [] []) {id = 44 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %46 : !air.async.token
              }
              affine.yield %45 : !air.async.token
            }
            affine.yield %44 : !air.async.token
          }
          %38 = arith.cmpi eq, %33, %c2_i32 : i32
          scf.if %38 {
            %async_token_61 = air.execute [%37] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %39 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %44 = air.channel.get async  @QK2L1_0[%arg16, %arg17] (%arg21[] [] []) {id = 45 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %44 : !air.async.token
          } else {
            %44 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %45 = air.channel.get async  @QK2L1_1[%arg16, %arg17] (%arg21[] [] []) {id = 46 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %45 : !air.async.token
            } else {
              %45 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %46 = air.channel.get async  @QK2L1_2[%arg16, %arg17] (%arg21[] [] []) {id = 47 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %46 : !air.async.token
              } else {
                %46 = air.channel.get async  @QK2L1_3[%arg16, %arg17] (%arg21[] [] []) {id = 48 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %46 : !air.async.token
              }
              affine.yield %45 : !air.async.token
            }
            affine.yield %44 : !air.async.token
          }
          %40 = arith.cmpi eq, %33, %c3_i32 : i32
          scf.if %40 {
            %async_token_61 = air.execute [%39] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %41 = air.wait_all async [%async_token_58, %async_token_59, %async_token_60] 
          %42 = scf.for %arg27 = %c0_56 to %c2_57 step %c1_55 iter_args(%arg28 = %41) -> (!air.async.token) {
            %async_token_61 = air.execute [%arg28] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            }
            %44 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %50 = air.channel.get async [%arg28]  @QK2L1_0[%arg16, %arg17] (%arg21[] [] []) {id = 49 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %50 : !air.async.token
            } else {
              %50 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %51 = air.channel.get async [%arg28]  @QK2L1_1[%arg16, %arg17] (%arg21[] [] []) {id = 50 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %51 : !air.async.token
              } else {
                %51 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %52 = air.channel.get async [%arg28]  @QK2L1_2[%arg16, %arg17] (%arg21[] [] []) {id = 51 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %52 : !air.async.token
                } else {
                  %52 = air.channel.get async [%arg28]  @QK2L1_3[%arg16, %arg17] (%arg21[] [] []) {id = 52 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %52 : !air.async.token
                }
                affine.yield %51 : !air.async.token
              }
              affine.yield %50 : !air.async.token
            }
            %45 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
              %50 = air.channel.get async  @V2L1_0[%arg16, %arg17] (%arg22[] [] []) {id = 53 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %50 : !air.async.token
            } else {
              %50 = air.wait_all async 
              affine.yield %50 : !air.async.token
            }
            %46 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
              %50 = air.channel.get async [%45, %arg28]  @V2L1_1[%arg16, %arg17] (%arg22[] [] []) {id = 54 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %50 : !air.async.token
            } else {
              %50 = air.wait_all async 
              affine.yield %50 : !air.async.token
            }
            %47 = affine.if #set5()[%arg16, %arg17] -> !air.async.token {
              %50 = air.channel.get async [%46]  @V2L1_2[%arg16, %arg17] (%arg22[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %50 : !air.async.token
            } else {
              %50 = air.wait_all async 
              affine.yield %50 : !air.async.token
            }
            %48 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
              %50 = air.channel.get async [%47]  @V2L1_3[%arg16, %arg17] (%arg22[] [] []) {id = 56 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %50 : !air.async.token
            } else {
              %50 = air.wait_all async 
              affine.yield %50 : !air.async.token
            }
            %async_token_62 = air.execute [%async_token_61, %44] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_a_b_bf16(%arg20, %arg21, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            }
            %async_token_63, %results_64 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_65, %results_66 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_67 = air.execute [%async_token_65, %async_token_63, %async_token_62] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @fused_softmax(%collapse_shape, %arg25, %results_64, %results_66) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_68 = air.execute [%async_token_67] {
              func.call @mul_r_gp(%results_66, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_69 = air.execute [%48, %async_token_68] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_g_b_bf16(%collapse_shape, %arg22, %arg24) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_70 = air.execute [%async_token_68] {
              func.call @accum_sp_r_s(%arg26, %results_66, %results_64) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_71 = air.execute [%async_token_70] {
              func.call @vector_copy_32elems(%c0_i32, %results_64, %arg26) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_72 = air.execute [%async_token_71] {
              memref.dealloc %results_64 : memref<64x1xbf16, 2 : i32>
            }
            %async_token_73 = air.execute [%async_token_70] {
              memref.dealloc %results_66 : memref<64x1xbf16, 2 : i32>
            }
            %49 = air.wait_all async [%async_token_69, %async_token_71] 
            scf.yield %49 : !air.async.token
          }
          %43 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
            %44 = arith.subi %arg17, %c1_55 : index
            %45 = air.channel.put async [%42]  @cascade_gp[%arg16, %44] (%arg24[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 2 : i32>)
            %46 = air.channel.put async [%42]  @cascade_up[%arg16, %44] (%arg25[] [] []) {id = 58 : i32} : (memref<64x1xbf16, 2 : i32>)
            %47 = air.channel.put async [%42]  @cascade_sp[%arg16, %44] (%arg26[] [] []) {id = 59 : i32} : (memref<64x1xbf16, 2 : i32>)
            %48 = air.wait_all async [%45, %46, %47] 
            affine.yield %48 : !air.async.token
          } else {
            %44 = affine.if #set7()[%arg16, %arg17] -> !air.async.token {
              %async_token_61, %results_62 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_63, %results_64 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_65, %results_66 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %45 = air.channel.get async [%async_token_61]  @cascade_gp[%arg16, %arg17] (%results_62[] [] []) {id = 60 : i32} : (memref<64x64xbf16, 2 : i32>)
              %46 = air.channel.get async [%async_token_63]  @cascade_up[%arg16, %arg17] (%results_64[] [] []) {id = 61 : i32} : (memref<64x1xbf16, 2 : i32>)
              %47 = air.channel.get async [%async_token_65]  @cascade_sp[%arg16, %arg17] (%results_66[] [] []) {id = 62 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_67, %results_68 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_69 = air.execute [%async_token_67, %42] {
                func.call @vector_copy_32elems(%c0_i32, %arg25, %results_68) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_70 = air.execute [%async_token_69, %46] {
                func.call @maximum_up_u_bf16(%results_64, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_71, %results_72 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_73 = air.execute [%async_token_71, %async_token_70] {
                func.call @exp_up_minus_u(%results_64, %arg25, %results_72) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_74, %results_75 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_76 = air.execute [%async_token_74, %async_token_73] {
                func.call @exp_up_minus_u(%results_68, %arg25, %results_75) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_77 = air.execute [%async_token_73, %45] {
                func.call @mul_r_gp(%results_72, %results_62) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_78 = air.execute [%async_token_76] {
                func.call @mul_r_gp(%results_75, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_79 = air.execute [%async_token_78, %async_token_77] {
                func.call @add_gp_g(%arg24, %results_62) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_80, %results_81 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_82 = air.execute [%async_token_80] {
                func.call @zero_fill_sp_bf16(%results_81) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_83 = air.execute [%async_token_82, %async_token_77, %47] {
                func.call @accum_sp_r_s(%results_66, %results_72, %results_81) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_84 = air.execute [%async_token_83, %async_token_78] {
                func.call @accum_sp_r_s(%arg26, %results_75, %results_81) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_85 = air.execute [%async_token_84] {
                func.call @vector_copy_32elems(%c0_i32, %results_81, %results_66) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %48 = arith.subi %arg17, %c1_55 : index
              %49 = air.channel.put async [%async_token_79]  @cascade_gp[%arg16, %48] (%results_62[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 2 : i32>)
              %50 = air.channel.put async [%async_token_76]  @cascade_up[%arg16, %48] (%arg25[] [] []) {id = 64 : i32} : (memref<64x1xbf16, 2 : i32>)
              %51 = air.channel.put async [%async_token_85]  @cascade_sp[%arg16, %48] (%results_66[] [] []) {id = 65 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_86 = air.execute [%49] {
                memref.dealloc %results_62 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_87 = air.execute [%async_token_73] {
                memref.dealloc %results_64 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_88 = air.execute [%51] {
                memref.dealloc %results_66 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_89 = air.execute [%async_token_76] {
                memref.dealloc %results_68 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_90 = air.execute [%async_token_83] {
                memref.dealloc %results_72 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_91 = air.execute [%async_token_84] {
                memref.dealloc %results_75 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_92 = air.execute [%async_token_85] {
                memref.dealloc %results_81 : memref<64x1xbf16, 2 : i32>
              }
              %52 = air.wait_all async [%49, %50, %51] 
              affine.yield %52 : !air.async.token
            } else {
              %async_token_61, %results_62 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_63, %results_64 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_65, %results_66 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %45 = air.channel.get async [%async_token_61]  @cascade_gp[%arg16, %arg17] (%results_62[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 2 : i32>)
              %46 = air.channel.get async [%async_token_63]  @cascade_up[%arg16, %arg17] (%results_64[] [] []) {id = 67 : i32} : (memref<64x1xbf16, 2 : i32>)
              %47 = air.channel.get async [%async_token_65]  @cascade_sp[%arg16, %arg17] (%results_66[] [] []) {id = 68 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_67, %results_68 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_69 = air.execute [%async_token_67, %42] {
                func.call @vector_copy_32elems(%c0_i32, %arg25, %results_68) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_70 = air.execute [%async_token_69, %46] {
                func.call @maximum_up_u_bf16(%results_64, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_71, %results_72 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_73 = air.execute [%async_token_71, %async_token_70] {
                func.call @exp_up_minus_u(%results_64, %arg25, %results_72) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_74, %results_75 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_76 = air.execute [%async_token_74, %async_token_73] {
                func.call @exp_up_minus_u(%results_68, %arg25, %results_75) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_77 = air.execute [%async_token_73, %45] {
                func.call @mul_r_gp(%results_72, %results_62) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_78 = air.execute [%async_token_76] {
                func.call @mul_r_gp(%results_75, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_79 = air.execute [%async_token_78, %async_token_77] {
                func.call @add_gp_g(%arg24, %results_62) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_80, %results_81 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_82 = air.execute [%async_token_80] {
                func.call @zero_fill_sp_bf16(%results_81) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_83 = air.execute [%async_token_82, %async_token_77, %47] {
                func.call @accum_sp_r_s(%results_66, %results_72, %results_81) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_84 = air.execute [%async_token_83, %async_token_78] {
                func.call @accum_sp_r_s(%arg26, %results_75, %results_81) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_85 = air.execute [%async_token_84] {
                func.call @vector_copy_32elems(%c0_i32, %results_81, %results_66) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_86 = air.execute [%async_token_85, %async_token_79] {
                func.call @div_gp_sp(%results_66, %results_62) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %48 = air.channel.put async [%async_token_86]  @Gp2L2[%arg16, %c0_56] (%results_62[%c0_56, %c0_56, %c0_56, %c0_56] [%c8_54, %c8_54, %c8_54, %c8_54] [%c64_53, %c8_54, %c512_52, %c1_55]) {id = 69 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_87 = air.execute [%48] {
                memref.dealloc %results_62 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_88 = air.execute [%async_token_73] {
                memref.dealloc %results_64 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_89 = air.execute [%async_token_86] {
                memref.dealloc %results_66 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_90 = air.execute [%async_token_76] {
                memref.dealloc %results_68 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_91 = air.execute [%async_token_83] {
                memref.dealloc %results_72 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_92 = air.execute [%async_token_84] {
                memref.dealloc %results_75 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_93 = air.execute [%async_token_85] {
                memref.dealloc %results_81 : memref<64x1xbf16, 2 : i32>
              }
              affine.yield %48 : !air.async.token
            }
            affine.yield %42 : !air.async.token
          }
        }
        %async_token_37 = air.execute [%31] {
          memref.dealloc %results_24 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_38 = air.execute [%31] {
          memref.dealloc %results_26 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_39 = air.execute [%31] {
          memref.dealloc %results_28 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_40 = air.execute [%31] {
          memref.dealloc %results_30 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_41 = air.execute [%31] {
          memref.dealloc %results_32 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_42 = air.execute [%31] {
          memref.dealloc %results_34 : memref<64x1xbf16, 2 : i32>
        }
        %async_token_43 = air.execute [%31] {
          memref.dealloc %results_36 : memref<64x1xbf16, 2 : i32>
        }
        %async_token_44 = air.execute [%19] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        }
        %async_token_45 = air.execute [%20] {
          memref.dealloc %results_10 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_46 = air.execute [%21] {
          memref.dealloc %results_12 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_47 = air.execute [%22] {
          memref.dealloc %results_14 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_48 = air.execute [%30] {
          memref.dealloc %results_22 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_49 = air.execute [%29] {
          memref.dealloc %results_20 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_50 = air.execute [%28] {
          memref.dealloc %results_18 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_51 = air.execute [%27] {
          memref.dealloc %results_16 : memref<64x64xbf16, 1 : i32>
        }
      }
    }
    return
  }
}
