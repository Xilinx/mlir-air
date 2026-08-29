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
        %c512_4 = arith.constant 512 : index
        %c8_5 = arith.constant 8 : index
        %c1_6 = arith.constant 1 : index
        %c2_7 = arith.constant 2 : index
        %c0_8 = arith.constant 0 : index
        %c4_9 = arith.constant 4 : index
        %async_token, %results = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_10, %results_11 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_12, %results_13 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_14, %results_15 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_16, %results_17 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_18, %results_19 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_20, %results_21 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_22, %results_23 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %async_token_24, %results_25 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_26, %results_27 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_28, %results_29 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_30, %results_31 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_32, %results_33 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_34, %results_35 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        }
        %async_token_36, %results_37 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        }
        %19 = scf.for %arg16 = %c0_8 to %c2_7 step %c1_6 iter_args(%arg17 = %async_token) -> (!air.async.token) {
          %32 = air.channel.get async [%arg17]  @VIn_0[%c0_8] (%results[] [] []) {id = 17 : i32} : (memref<64x64xbf16, 1 : i32>)
          %33 = air.channel.put async [%32]  @V2L1_0[%c0_8, %c0_8] (%results[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 18 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %33 : !air.async.token
        }
        %20 = scf.for %arg16 = %c0_8 to %c2_7 step %c1_6 iter_args(%arg17 = %async_token_10) -> (!air.async.token) {
          %32 = air.channel.get async [%arg17]  @VIn_1[%c0_8] (%results_11[] [] []) {id = 19 : i32} : (memref<64x64xbf16, 1 : i32>)
          %33 = air.channel.put async [%32]  @V2L1_1[%c0_8, %c0_8] (%results_11[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 20 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %33 : !air.async.token
        }
        %21 = scf.for %arg16 = %c0_8 to %c2_7 step %c1_6 iter_args(%arg17 = %async_token_12) -> (!air.async.token) {
          %32 = air.channel.get async [%arg17]  @VIn_2[%c0_8] (%results_13[] [] []) {id = 21 : i32} : (memref<64x64xbf16, 1 : i32>)
          %33 = air.channel.put async [%32]  @V2L1_2[%c0_8, %c0_8] (%results_13[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 22 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %33 : !air.async.token
        }
        %22 = scf.for %arg16 = %c0_8 to %c2_7 step %c1_6 iter_args(%arg17 = %async_token_14) -> (!air.async.token) {
          %32 = air.channel.get async [%arg17]  @VIn_3[%c0_8] (%results_15[] [] []) {id = 23 : i32} : (memref<64x64xbf16, 1 : i32>)
          %33 = air.channel.put async [%32]  @V2L1_3[%c0_8, %c0_8] (%results_15[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 24 : i32} : (memref<64x64xbf16, 1 : i32>)
          scf.yield %33 : !air.async.token
        }
        %23 = air.channel.get async [%async_token_16]  @Gp2L2[%c0_8, %c0_8] (%results_17[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %24 = air.channel.get async [%async_token_18]  @Gp2L2[%c1_6, %c0_8] (%results_19[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %25 = air.channel.get async [%async_token_20]  @Gp2L2[%c2_7, %c0_8] (%results_21[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %26 = air.channel.get async [%async_token_22]  @Gp2L2[%c3_2, %c0_8] (%results_23[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %27 = air.channel.put async [%23]  @channel_0[%c0_8, %c0_8] (%results_17[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %28 = air.channel.put async [%24]  @channel_0[%c1_6, %c0_8] (%results_19[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %29 = air.channel.put async [%25]  @channel_0[%c2_7, %c0_8] (%results_21[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %30 = air.channel.put async [%26]  @channel_0[%c3_2, %c0_8] (%results_23[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %31 = air.herd @herd_0 async [%async_token_24, %async_token_26, %async_token_28, %async_token_30, %async_token_32, %async_token_34, %async_token_36]  tile (%arg16, %arg17) in (%arg18=%c4_9, %arg19=%c4_9) args(%arg20=%results_25, %arg21=%results_27, %arg22=%results_29, %arg23=%results_31, %arg24=%results_33, %arg25=%results_35, %arg26=%results_37) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32> attributes {id = 3 : i32, link_with = "attn.o"} {
          %c512_53 = arith.constant 512 : index
          %c64_54 = arith.constant 64 : index
          %c8_55 = arith.constant 8 : index
          %c1_56 = arith.constant 1 : index
          %c0_57 = arith.constant 0 : index
          %c2_58 = arith.constant 2 : index
          %c3_i32 = arith.constant 3 : i32
          %c2_i32 = arith.constant 2 : i32
          %c1_i32 = arith.constant 1 : i32
          %c0_i32 = arith.constant 0 : i32
          %async_token_59 = air.execute {
            func.call @zero_fill_gp_bf16(%arg24) : (memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_60 = air.execute {
            func.call @zero_fill_sp_bf16(%arg26) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_61 = air.execute {
            func.call @neg_inf_fill_up_bf16(%arg25) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %32 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %44 = air.channel.get async  @QK2L1_0[%arg16, %c0_57] (%arg21[] [] []) {id = 33 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %44 : !air.async.token
          } else {
            %44 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %45 = air.channel.get async  @QK2L1_1[%arg16, %c0_57] (%arg21[] [] []) {id = 34 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %45 : !air.async.token
            } else {
              %45 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %46 = air.channel.get async  @QK2L1_2[%arg16, %c0_57] (%arg21[] [] []) {id = 35 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %46 : !air.async.token
              } else {
                %46 = air.channel.get async  @QK2L1_3[%arg16, %c0_57] (%arg21[] [] []) {id = 36 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %46 : !air.async.token
              }
              affine.yield %45 : !air.async.token
            }
            affine.yield %44 : !air.async.token
          }
          %33 = arith.index_cast %arg16 : index to i32
          %34 = arith.cmpi eq, %33, %c0_i32 : i32
          scf.if %34 {
            %async_token_62 = air.execute [%32] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %35 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %44 = air.channel.get async  @QK2L1_0[%arg16, %c0_57] (%arg21[] [] []) {id = 37 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %44 : !air.async.token
          } else {
            %44 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %45 = air.channel.get async  @QK2L1_1[%arg16, %c0_57] (%arg21[] [] []) {id = 38 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %45 : !air.async.token
            } else {
              %45 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %46 = air.channel.get async  @QK2L1_2[%arg16, %c0_57] (%arg21[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %46 : !air.async.token
              } else {
                %46 = air.channel.get async  @QK2L1_3[%arg16, %c0_57] (%arg21[] [] []) {id = 40 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %46 : !air.async.token
              }
              affine.yield %45 : !air.async.token
            }
            affine.yield %44 : !air.async.token
          }
          %36 = arith.cmpi eq, %33, %c1_i32 : i32
          scf.if %36 {
            %async_token_62 = air.execute [%35] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %37 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %44 = air.channel.get async  @QK2L1_0[%arg16, %c0_57] (%arg21[] [] []) {id = 41 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %44 : !air.async.token
          } else {
            %44 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %45 = air.channel.get async  @QK2L1_1[%arg16, %c0_57] (%arg21[] [] []) {id = 42 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %45 : !air.async.token
            } else {
              %45 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %46 = air.channel.get async  @QK2L1_2[%arg16, %c0_57] (%arg21[] [] []) {id = 43 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %46 : !air.async.token
              } else {
                %46 = air.channel.get async  @QK2L1_3[%arg16, %c0_57] (%arg21[] [] []) {id = 44 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %46 : !air.async.token
              }
              affine.yield %45 : !air.async.token
            }
            affine.yield %44 : !air.async.token
          }
          %38 = arith.cmpi eq, %33, %c2_i32 : i32
          scf.if %38 {
            %async_token_62 = air.execute [%37] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %39 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %44 = air.channel.get async  @QK2L1_0[%arg16, %c0_57] (%arg21[] [] []) {id = 45 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %44 : !air.async.token
          } else {
            %44 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %45 = air.channel.get async  @QK2L1_1[%arg16, %c0_57] (%arg21[] [] []) {id = 46 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %45 : !air.async.token
            } else {
              %45 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %46 = air.channel.get async  @QK2L1_2[%arg16, %c0_57] (%arg21[] [] []) {id = 47 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %46 : !air.async.token
              } else {
                %46 = air.channel.get async  @QK2L1_3[%arg16, %c0_57] (%arg21[] [] []) {id = 48 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %46 : !air.async.token
              }
              affine.yield %45 : !air.async.token
            }
            affine.yield %44 : !air.async.token
          }
          %40 = arith.cmpi eq, %33, %c3_i32 : i32
          scf.if %40 {
            %async_token_62 = air.execute [%39] {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %41 = air.wait_all async [%async_token_59, %async_token_60, %async_token_61] 
          %42 = scf.for %arg27 = %c0_57 to %c2_58 step %c1_56 iter_args(%arg28 = %41) -> (!air.async.token) {
            %async_token_62 = air.execute [%arg28] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            }
            %44 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %50 = air.channel.get async [%arg28]  @QK2L1_0[%arg16, %c0_57] (%arg21[] [] []) {id = 49 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %50 : !air.async.token
            } else {
              %50 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %51 = air.channel.get async [%arg28]  @QK2L1_1[%arg16, %c0_57] (%arg21[] [] []) {id = 50 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %51 : !air.async.token
              } else {
                %51 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %52 = air.channel.get async [%arg28]  @QK2L1_2[%arg16, %c0_57] (%arg21[] [] []) {id = 51 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %52 : !air.async.token
                } else {
                  %52 = air.channel.get async [%arg28]  @QK2L1_3[%arg16, %c0_57] (%arg21[] [] []) {id = 52 : i32} : (memref<64x64xbf16, 2 : i32>)
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
            %async_token_63 = air.execute [%async_token_62, %44] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_a_b_bf16(%arg20, %arg21, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            }
            %async_token_64, %results_65 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_66, %results_67 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_68 = air.execute [%async_token_66, %async_token_64, %async_token_63] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @fused_softmax(%collapse_shape, %arg25, %results_65, %results_67) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_69 = air.execute [%async_token_68] {
              func.call @mul_r_gp(%results_67, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_70 = air.execute [%48, %async_token_69] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_g_b_bf16(%collapse_shape, %arg22, %arg24) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_71 = air.execute [%async_token_69] {
              func.call @accum_sp_r_s(%arg26, %results_67, %results_65) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_72 = air.execute [%async_token_71] {
              func.call @vector_copy_32elems(%c0_i32, %results_65, %arg26) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_73 = air.execute [%async_token_72] {
              memref.dealloc %results_65 : memref<64x1xbf16, 2 : i32>
            }
            %async_token_74 = air.execute [%async_token_71] {
              memref.dealloc %results_67 : memref<64x1xbf16, 2 : i32>
            }
            %49 = air.wait_all async [%async_token_70, %async_token_72] 
            scf.yield %49 : !air.async.token
          }
          %43 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
            %44 = arith.subi %arg17, %c1_56 : index
            %45 = air.channel.put async [%42]  @cascade_gp[%arg16, %44] (%arg24[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 2 : i32>)
            %46 = air.channel.put async [%42]  @cascade_up[%arg16, %44] (%arg25[] [] []) {id = 58 : i32} : (memref<64x1xbf16, 2 : i32>)
            %47 = air.channel.put async [%42]  @cascade_sp[%arg16, %44] (%arg26[] [] []) {id = 59 : i32} : (memref<64x1xbf16, 2 : i32>)
            %48 = air.wait_all async [%45, %46, %47] 
            affine.yield %48 : !air.async.token
          } else {
            %44 = affine.if #set7()[%arg16, %arg17] -> !air.async.token {
              %async_token_62, %results_63 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_64, %results_65 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_66, %results_67 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %45 = air.channel.get async [%async_token_62]  @cascade_gp[%arg16, %arg17] (%results_63[] [] []) {id = 60 : i32} : (memref<64x64xbf16, 2 : i32>)
              %46 = air.channel.get async [%async_token_64]  @cascade_up[%arg16, %arg17] (%results_65[] [] []) {id = 61 : i32} : (memref<64x1xbf16, 2 : i32>)
              %47 = air.channel.get async [%async_token_66]  @cascade_sp[%arg16, %arg17] (%results_67[] [] []) {id = 62 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_68, %results_69 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_70 = air.execute [%async_token_68, %42] {
                func.call @vector_copy_32elems(%c0_i32, %arg25, %results_69) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_71 = air.execute [%async_token_70, %46] {
                func.call @maximum_up_u_bf16(%results_65, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_72, %results_73 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_74 = air.execute [%async_token_72, %async_token_71] {
                func.call @exp_up_minus_u(%results_65, %arg25, %results_73) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_75, %results_76 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_77 = air.execute [%async_token_75, %async_token_74] {
                func.call @exp_up_minus_u(%results_69, %arg25, %results_76) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_78 = air.execute [%async_token_74, %45] {
                func.call @mul_r_gp(%results_73, %results_63) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_79 = air.execute [%async_token_77] {
                func.call @mul_r_gp(%results_76, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_80 = air.execute [%async_token_79, %async_token_78] {
                func.call @add_gp_g(%arg24, %results_63) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_81, %results_82 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_83 = air.execute [%async_token_81] {
                func.call @zero_fill_sp_bf16(%results_82) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_84 = air.execute [%async_token_83, %async_token_78, %47] {
                func.call @accum_sp_r_s(%results_67, %results_73, %results_82) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_85 = air.execute [%async_token_84, %async_token_79] {
                func.call @accum_sp_r_s(%arg26, %results_76, %results_82) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_86 = air.execute [%async_token_85] {
                func.call @vector_copy_32elems(%c0_i32, %results_82, %results_67) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %48 = arith.subi %arg17, %c1_56 : index
              %49 = air.channel.put async [%async_token_80]  @cascade_gp[%arg16, %48] (%results_63[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 2 : i32>)
              %50 = air.channel.put async [%async_token_77]  @cascade_up[%arg16, %48] (%arg25[] [] []) {id = 64 : i32} : (memref<64x1xbf16, 2 : i32>)
              %51 = air.channel.put async [%async_token_86]  @cascade_sp[%arg16, %48] (%results_67[] [] []) {id = 65 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_87 = air.execute [%49] {
                memref.dealloc %results_63 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_88 = air.execute [%async_token_74] {
                memref.dealloc %results_65 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_89 = air.execute [%51] {
                memref.dealloc %results_67 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_90 = air.execute [%async_token_77] {
                memref.dealloc %results_69 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_91 = air.execute [%async_token_84] {
                memref.dealloc %results_73 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_92 = air.execute [%async_token_85] {
                memref.dealloc %results_76 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_93 = air.execute [%async_token_86] {
                memref.dealloc %results_82 : memref<64x1xbf16, 2 : i32>
              }
              %52 = air.wait_all async [%49, %50, %51] 
              affine.yield %52 : !air.async.token
            } else {
              %async_token_62, %results_63 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_64, %results_65 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_66, %results_67 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %45 = air.channel.get async [%async_token_62]  @cascade_gp[%arg16, %arg17] (%results_63[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 2 : i32>)
              %46 = air.channel.get async [%async_token_64]  @cascade_up[%arg16, %arg17] (%results_65[] [] []) {id = 67 : i32} : (memref<64x1xbf16, 2 : i32>)
              %47 = air.channel.get async [%async_token_66]  @cascade_sp[%arg16, %arg17] (%results_67[] [] []) {id = 68 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_68, %results_69 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_70 = air.execute [%async_token_68, %42] {
                func.call @vector_copy_32elems(%c0_i32, %arg25, %results_69) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_71 = air.execute [%async_token_70, %46] {
                func.call @maximum_up_u_bf16(%results_65, %arg25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_72, %results_73 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_74 = air.execute [%async_token_72, %async_token_71] {
                func.call @exp_up_minus_u(%results_65, %arg25, %results_73) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_75, %results_76 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_77 = air.execute [%async_token_75, %async_token_74] {
                func.call @exp_up_minus_u(%results_69, %arg25, %results_76) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_78 = air.execute [%async_token_74, %45] {
                func.call @mul_r_gp(%results_73, %results_63) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_79 = air.execute [%async_token_77] {
                func.call @mul_r_gp(%results_76, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_80 = air.execute [%async_token_79, %async_token_78] {
                func.call @add_gp_g(%arg24, %results_63) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_81, %results_82 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_83 = air.execute [%async_token_81] {
                func.call @zero_fill_sp_bf16(%results_82) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_84 = air.execute [%async_token_83, %async_token_78, %47] {
                func.call @accum_sp_r_s(%results_67, %results_73, %results_82) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_85 = air.execute [%async_token_84, %async_token_79] {
                func.call @accum_sp_r_s(%arg26, %results_76, %results_82) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_86 = air.execute [%async_token_85] {
                func.call @vector_copy_32elems(%c0_i32, %results_82, %results_67) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_87 = air.execute [%async_token_86, %async_token_80] {
                func.call @div_gp_sp(%results_67, %results_63) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %48 = air.channel.put async [%async_token_87]  @Gp2L2[%arg16, %c0_57] (%results_63[%c0_57, %c0_57, %c0_57, %c0_57] [%c8_55, %c8_55, %c8_55, %c8_55] [%c64_54, %c8_55, %c512_53, %c1_56]) {id = 69 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_88 = air.execute [%48] {
                memref.dealloc %results_63 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_89 = air.execute [%async_token_74] {
                memref.dealloc %results_65 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_90 = air.execute [%async_token_87] {
                memref.dealloc %results_67 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_91 = air.execute [%async_token_77] {
                memref.dealloc %results_69 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_92 = air.execute [%async_token_84] {
                memref.dealloc %results_73 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_93 = air.execute [%async_token_85] {
                memref.dealloc %results_76 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_94 = air.execute [%async_token_86] {
                memref.dealloc %results_82 : memref<64x1xbf16, 2 : i32>
              }
              affine.yield %48 : !air.async.token
            }
            affine.yield %42 : !air.async.token
          }
        }
        %async_token_38 = air.execute [%31] {
          memref.dealloc %results_25 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_39 = air.execute [%31] {
          memref.dealloc %results_27 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_40 = air.execute [%31] {
          memref.dealloc %results_29 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_41 = air.execute [%31] {
          memref.dealloc %results_31 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_42 = air.execute [%31] {
          memref.dealloc %results_33 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_43 = air.execute [%31] {
          memref.dealloc %results_35 : memref<64x1xbf16, 2 : i32>
        }
        %async_token_44 = air.execute [%31] {
          memref.dealloc %results_37 : memref<64x1xbf16, 2 : i32>
        }
        %async_token_45 = air.execute [%19] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        }
        %async_token_46 = air.execute [%20] {
          memref.dealloc %results_11 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_47 = air.execute [%21] {
          memref.dealloc %results_13 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_48 = air.execute [%22] {
          memref.dealloc %results_15 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_49 = air.execute [%30] {
          memref.dealloc %results_23 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_50 = air.execute [%29] {
          memref.dealloc %results_21 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_51 = air.execute [%28] {
          memref.dealloc %results_19 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_52 = air.execute [%27] {
          memref.dealloc %results_17 : memref<64x64xbf16, 1 : i32>
        }
      }
    }
    return
  }
}
