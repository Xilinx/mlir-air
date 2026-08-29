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
        %23 = air.wait_all async 
        %24 = air.wait_all async 
        %25 = air.wait_all async 
        %26 = air.wait_all async 
        %27 = air.wait_all async 
        %28 = air.wait_all async 
        %29 = air.wait_all async 
        %30 = scf.for %arg16 = %c0_7 to %c2_6 step %c1_5 iter_args(%arg17 = %19) -> (!air.async.token) {
          %async_token_19, %results_20 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          }
          %54 = air.channel.get async [%async_token_19, %arg17]  @VIn_0[%c0_7] (%results_20[] [] []) {id = 17 : i32} : (memref<64x64xbf16, 1 : i32>)
          %55 = air.channel.put async [%async_token_19, %54]  @V2L1_0[%c0_7, %c0_7] (%results_20[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_3, %c64_2, %c1_5]) {id = 18 : i32} : (memref<64x64xbf16, 1 : i32>)
          %async_token_21 = air.execute [%55, %54] {
            memref.dealloc %results_20 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %55 : !air.async.token
        }
        %31 = scf.for %arg16 = %c0_7 to %c2_6 step %c1_5 iter_args(%arg17 = %20) -> (!air.async.token) {
          %async_token_19, %results_20 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          }
          %54 = air.channel.get async [%async_token_19, %arg17]  @VIn_1[%c0_7] (%results_20[] [] []) {id = 19 : i32} : (memref<64x64xbf16, 1 : i32>)
          %55 = air.channel.put async [%async_token_19, %54]  @V2L1_1[%c0_7, %c0_7] (%results_20[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_3, %c64_2, %c1_5]) {id = 20 : i32} : (memref<64x64xbf16, 1 : i32>)
          %async_token_21 = air.execute [%55, %54] {
            memref.dealloc %results_20 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %55 : !air.async.token
        }
        %32 = scf.for %arg16 = %c0_7 to %c2_6 step %c1_5 iter_args(%arg17 = %21) -> (!air.async.token) {
          %async_token_19, %results_20 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          }
          %54 = air.channel.get async [%async_token_19, %arg17]  @VIn_2[%c0_7] (%results_20[] [] []) {id = 21 : i32} : (memref<64x64xbf16, 1 : i32>)
          %55 = air.channel.put async [%async_token_19, %54]  @V2L1_2[%c0_7, %c0_7] (%results_20[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_3, %c64_2, %c1_5]) {id = 22 : i32} : (memref<64x64xbf16, 1 : i32>)
          %async_token_21 = air.execute [%55, %54] {
            memref.dealloc %results_20 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %55 : !air.async.token
        }
        %33 = scf.for %arg16 = %c0_7 to %c2_6 step %c1_5 iter_args(%arg17 = %22) -> (!air.async.token) {
          %async_token_19, %results_20 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          }
          %54 = air.channel.get async [%async_token_19, %arg17]  @VIn_3[%c0_7] (%results_20[] [] []) {id = 23 : i32} : (memref<64x64xbf16, 1 : i32>)
          %55 = air.channel.put async [%async_token_19, %54]  @V2L1_3[%c0_7, %c0_7] (%results_20[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_4, %c8_4, %c8_4, %c8_4] [%c8_4, %c512_3, %c64_2, %c1_5]) {id = 24 : i32} : (memref<64x64xbf16, 1 : i32>)
          %async_token_21 = air.execute [%55, %54] {
            memref.dealloc %results_20 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %55 : !air.async.token
        }
        %34 = air.channel.get async [%async_token]  @Gp2L2[%c0_7, %c0_7] (%results[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %35 = air.channel.get async [%async_token_9]  @Gp2L2[%c1_5, %c0_7] (%results_10[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %36 = air.channel.get async [%async_token_11]  @Gp2L2[%c2_6, %c0_7] (%results_12[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %37 = air.channel.get async [%async_token_13]  @Gp2L2[%c3_1, %c0_7] (%results_14[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %38 = air.channel.put async [%34]  @channel_0[%c0_7, %c0_7] (%results[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %39 = air.channel.put async [%35]  @channel_0[%c1_5, %c0_7] (%results_10[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %40 = air.channel.put async [%36]  @channel_0[%c2_6, %c0_7] (%results_12[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %41 = air.channel.put async [%37]  @channel_0[%c3_1, %c0_7] (%results_14[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %42 = air.herd @herd_0 async  tile (%arg16, %arg17) in (%arg18=%c4_8, %arg19=%c4_8) attributes {id = 3 : i32, link_with = "attn.o"} {
          %c0_i32 = arith.constant 0 : i32
          %c1_i32 = arith.constant 1 : i32
          %c2_i32 = arith.constant 2 : i32
          %c3_i32 = arith.constant 3 : i32
          %c2_19 = arith.constant 2 : index
          %c0_20 = arith.constant 0 : index
          %c1_21 = arith.constant 1 : index
          %c8_22 = arith.constant 8 : index
          %c64_23 = arith.constant 64 : index
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
          %54 = air.wait_all async 
          %55 = air.wait_all async 
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
          %56 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %70 = air.channel.get async [%async_token_31]  @QK2L1_0[%arg16, %arg17] (%results_32[] [] []) {id = 33 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %70 : !air.async.token
          } else {
            %70 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %71 = air.channel.get async [%async_token_31]  @QK2L1_1[%arg16, %arg17] (%results_32[] [] []) {id = 34 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %71 : !air.async.token
            } else {
              %71 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %72 = air.channel.get async [%async_token_31]  @QK2L1_2[%arg16, %arg17] (%results_32[] [] []) {id = 35 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              } else {
                %72 = air.channel.get async [%async_token_31]  @QK2L1_3[%arg16, %arg17] (%results_32[] [] []) {id = 36 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              }
              affine.yield %71 : !air.async.token
            }
            affine.yield %70 : !air.async.token
          }
          %57 = arith.index_cast %arg16 : index to i32
          %58 = arith.cmpi eq, %57, %c0_i32 : i32
          scf.if %58 {
            %async_token_43 = air.execute [%async_token_31, %async_token_33, %56] {
              func.call @copy_tile(%results_32, %results_34) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %59 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %70 = air.channel.get async [%async_token_31]  @QK2L1_0[%arg16, %arg17] (%results_32[] [] []) {id = 37 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %70 : !air.async.token
          } else {
            %70 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %71 = air.channel.get async [%async_token_31]  @QK2L1_1[%arg16, %arg17] (%results_32[] [] []) {id = 38 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %71 : !air.async.token
            } else {
              %71 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %72 = air.channel.get async [%async_token_31]  @QK2L1_2[%arg16, %arg17] (%results_32[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              } else {
                %72 = air.channel.get async [%async_token_31]  @QK2L1_3[%arg16, %arg17] (%results_32[] [] []) {id = 40 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              }
              affine.yield %71 : !air.async.token
            }
            affine.yield %70 : !air.async.token
          }
          %60 = arith.cmpi eq, %57, %c1_i32 : i32
          scf.if %60 {
            %async_token_43 = air.execute [%async_token_31, %async_token_33, %59] {
              func.call @copy_tile(%results_32, %results_34) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %61 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %70 = air.channel.get async [%async_token_31]  @QK2L1_0[%arg16, %arg17] (%results_32[] [] []) {id = 41 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %70 : !air.async.token
          } else {
            %70 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %71 = air.channel.get async [%async_token_31]  @QK2L1_1[%arg16, %arg17] (%results_32[] [] []) {id = 42 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %71 : !air.async.token
            } else {
              %71 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %72 = air.channel.get async [%async_token_31]  @QK2L1_2[%arg16, %arg17] (%results_32[] [] []) {id = 43 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              } else {
                %72 = air.channel.get async [%async_token_31]  @QK2L1_3[%arg16, %arg17] (%results_32[] [] []) {id = 44 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              }
              affine.yield %71 : !air.async.token
            }
            affine.yield %70 : !air.async.token
          }
          %62 = arith.cmpi eq, %57, %c2_i32 : i32
          scf.if %62 {
            %async_token_43 = air.execute [%async_token_31, %async_token_33, %61] {
              func.call @copy_tile(%results_32, %results_34) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %63 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %70 = air.channel.get async [%async_token_31]  @QK2L1_0[%arg16, %arg17] (%results_32[] [] []) {id = 45 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %70 : !air.async.token
          } else {
            %70 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %71 = air.channel.get async [%async_token_31]  @QK2L1_1[%arg16, %arg17] (%results_32[] [] []) {id = 46 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %71 : !air.async.token
            } else {
              %71 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %72 = air.channel.get async [%async_token_31]  @QK2L1_2[%arg16, %arg17] (%results_32[] [] []) {id = 47 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              } else {
                %72 = air.channel.get async [%async_token_31]  @QK2L1_3[%arg16, %arg17] (%results_32[] [] []) {id = 48 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              }
              affine.yield %71 : !air.async.token
            }
            affine.yield %70 : !air.async.token
          }
          %64 = arith.cmpi eq, %57, %c3_i32 : i32
          scf.if %64 {
            %async_token_43 = air.execute [%async_token_31, %async_token_33, %63] {
              func.call @copy_tile(%results_32, %results_34) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %65 = air.wait_all async [%async_token_35, %async_token_36, %async_token_37] 
          %66 = scf.for %arg20 = %c0_20 to %c2_19 step %c1_21 iter_args(%arg21 = %65) -> (!air.async.token) {
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
            %70 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %76 = air.channel.get async [%async_token_31, %arg21]  @QK2L1_0[%arg16, %arg17] (%results_32[] [] []) {id = 49 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %76 : !air.async.token
            } else {
              %76 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %77 = air.channel.get async [%async_token_31, %arg21]  @QK2L1_1[%arg16, %arg17] (%results_32[] [] []) {id = 50 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %77 : !air.async.token
              } else {
                %77 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %78 = air.channel.get async [%async_token_31, %arg21]  @QK2L1_2[%arg16, %arg17] (%results_32[] [] []) {id = 51 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %78 : !air.async.token
                } else {
                  %78 = air.channel.get async [%async_token_31, %arg21]  @QK2L1_3[%arg16, %arg17] (%results_32[] [] []) {id = 52 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %78 : !air.async.token
                }
                affine.yield %77 : !air.async.token
              }
              affine.yield %76 : !air.async.token
            }
            %71 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
              %76 = air.channel.get async [%async_token_43]  @V2L1_0[%arg16, %arg17] (%results_44[] [] []) {id = 53 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %76 : !air.async.token
            } else {
              %76 = air.wait_all async 
              affine.yield %76 : !air.async.token
            }
            %72 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
              %76 = air.channel.get async [%async_token_43, %71, %arg21]  @V2L1_1[%arg16, %arg17] (%results_44[] [] []) {id = 54 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %76 : !air.async.token
            } else {
              %76 = air.wait_all async 
              affine.yield %76 : !air.async.token
            }
            %73 = affine.if #set5()[%arg16, %arg17] -> !air.async.token {
              %76 = air.channel.get async [%async_token_43, %72]  @V2L1_2[%arg16, %arg17] (%results_44[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %76 : !air.async.token
            } else {
              %76 = air.wait_all async 
              affine.yield %76 : !air.async.token
            }
            %74 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
              %76 = air.channel.get async [%async_token_43, %73]  @V2L1_3[%arg16, %arg17] (%results_44[] [] []) {id = 56 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %76 : !air.async.token
            } else {
              %76 = air.wait_all async 
              affine.yield %76 : !air.async.token
            }
            %async_token_48 = air.execute [%async_token_45, %async_token_31, %async_token_33, %async_token_47, %70] {
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
            %async_token_53 = air.execute [%async_token_27, %async_token_45, %async_token_51, %async_token_49, %async_token_48] {
              %collapse_shape = memref.collapse_shape %results_46 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @fused_softmax(%collapse_shape, %results_28, %results_50, %results_52) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_54 = air.execute [%async_token_29, %async_token_53] {
              func.call @mul_r_gp(%results_52, %results_30) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_55 = air.execute [%async_token_29, %async_token_45, %async_token_43, %74, %async_token_54] {
              %collapse_shape = memref.collapse_shape %results_46 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_g_b_bf16(%collapse_shape, %results_44, %results_30) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_56 = air.execute [%async_token_25, %async_token_54] {
              func.call @accum_sp_r_s(%results_26, %results_52, %results_50) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_57 = air.execute [%async_token_25, %async_token_56] {
              func.call @vector_copy_32elems(%c0_i32, %results_50, %results_26) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_58 = air.execute [%async_token_57] {
              memref.dealloc %results_50 : memref<64x1xbf16, 2 : i32>
            }
            %async_token_59 = air.execute [%async_token_56] {
              memref.dealloc %results_52 : memref<64x1xbf16, 2 : i32>
            }
            %75 = air.wait_all async [%async_token_55, %async_token_57] 
            %async_token_60 = air.execute [%async_token_55, %async_token_53, %async_token_48, %async_token_47] {
              memref.dealloc %results_46 : memref<64x64xbf16, 2 : i32>
            }
            %async_token_61 = air.execute [%async_token_55, %74, %73, %72, %71] {
              memref.dealloc %results_44 : memref<64x64xbf16, 2 : i32>
            }
            scf.yield %75 : !air.async.token
          }
          %67 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
            %70 = arith.subi %arg17, %c1_21 : index
            %71 = air.channel.put async [%async_token_29, %66]  @cascade_gp[%arg16, %70] (%results_30[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 2 : i32>)
            %72 = air.channel.put async [%async_token_27, %66]  @cascade_up[%arg16, %70] (%results_28[] [] []) {id = 58 : i32} : (memref<64x1xbf16, 2 : i32>)
            %73 = air.channel.put async [%async_token_25, %66]  @cascade_sp[%arg16, %70] (%results_26[] [] []) {id = 59 : i32} : (memref<64x1xbf16, 2 : i32>)
            %74 = air.wait_all async [%71, %72, %73] 
            affine.yield %74 : !air.async.token
          } else {
            %70 = affine.if #set7()[%arg16, %arg17] -> !air.async.token {
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
              %71 = air.channel.get async [%async_token_43]  @cascade_gp[%arg16, %arg17] (%results_44[] [] []) {id = 60 : i32} : (memref<64x64xbf16, 2 : i32>)
              %72 = air.channel.get async [%async_token_45]  @cascade_up[%arg16, %arg17] (%results_46[] [] []) {id = 61 : i32} : (memref<64x1xbf16, 2 : i32>)
              %73 = air.channel.get async [%async_token_47]  @cascade_sp[%arg16, %arg17] (%results_48[] [] []) {id = 62 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_49, %results_50 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_51 = air.execute [%async_token_27, %async_token_49, %66] {
                func.call @vector_copy_32elems(%c0_i32, %results_28, %results_50) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_52 = air.execute [%async_token_27, %async_token_51, %72] {
                func.call @maximum_up_u_bf16(%results_46, %results_28) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_53, %results_54 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_55 = air.execute [%async_token_27, %async_token_53, %async_token_52] {
                func.call @exp_up_minus_u(%results_46, %results_28, %results_54) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_56, %results_57 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_58 = air.execute [%async_token_27, %async_token_56, %async_token_55] {
                func.call @exp_up_minus_u(%results_50, %results_28, %results_57) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_59 = air.execute [%async_token_55, %71] {
                func.call @mul_r_gp(%results_54, %results_44) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_60 = air.execute [%async_token_29, %async_token_58] {
                func.call @mul_r_gp(%results_57, %results_30) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_61 = air.execute [%async_token_29, %async_token_60, %async_token_59] {
                func.call @add_gp_g(%results_30, %results_44) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_62, %results_63 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_64 = air.execute [%async_token_62] {
                func.call @zero_fill_sp_bf16(%results_63) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_65 = air.execute [%async_token_64, %async_token_59, %73] {
                func.call @accum_sp_r_s(%results_48, %results_54, %results_63) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_66 = air.execute [%async_token_25, %async_token_65, %async_token_60] {
                func.call @accum_sp_r_s(%results_26, %results_57, %results_63) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_67 = air.execute [%async_token_66] {
                func.call @vector_copy_32elems(%c0_i32, %results_63, %results_48) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %74 = arith.subi %arg17, %c1_21 : index
              %75 = air.channel.put async [%async_token_61]  @cascade_gp[%arg16, %74] (%results_44[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 2 : i32>)
              %76 = air.channel.put async [%async_token_27, %async_token_58]  @cascade_up[%arg16, %74] (%results_28[] [] []) {id = 64 : i32} : (memref<64x1xbf16, 2 : i32>)
              %77 = air.channel.put async [%async_token_67]  @cascade_sp[%arg16, %74] (%results_48[] [] []) {id = 65 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_68 = air.execute [%75] {
                memref.dealloc %results_44 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_69 = air.execute [%async_token_55] {
                memref.dealloc %results_46 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_70 = air.execute [%77] {
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
              %78 = air.wait_all async [%75, %76, %77] 
              affine.yield %78 : !air.async.token
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
              %71 = air.channel.get async [%async_token_43]  @cascade_gp[%arg16, %arg17] (%results_44[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 2 : i32>)
              %72 = air.channel.get async [%async_token_45]  @cascade_up[%arg16, %arg17] (%results_46[] [] []) {id = 67 : i32} : (memref<64x1xbf16, 2 : i32>)
              %73 = air.channel.get async [%async_token_47]  @cascade_sp[%arg16, %arg17] (%results_48[] [] []) {id = 68 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_49, %results_50 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_51 = air.execute [%async_token_27, %async_token_49, %66] {
                func.call @vector_copy_32elems(%c0_i32, %results_28, %results_50) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_52 = air.execute [%async_token_27, %async_token_51, %72] {
                func.call @maximum_up_u_bf16(%results_46, %results_28) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_53, %results_54 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_55 = air.execute [%async_token_27, %async_token_53, %async_token_52] {
                func.call @exp_up_minus_u(%results_46, %results_28, %results_54) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_56, %results_57 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_58 = air.execute [%async_token_27, %async_token_56, %async_token_55] {
                func.call @exp_up_minus_u(%results_50, %results_28, %results_57) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_59 = air.execute [%async_token_55, %71] {
                func.call @mul_r_gp(%results_54, %results_44) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_60 = air.execute [%async_token_29, %async_token_58] {
                func.call @mul_r_gp(%results_57, %results_30) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_61 = air.execute [%async_token_29, %async_token_60, %async_token_59] {
                func.call @add_gp_g(%results_30, %results_44) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_62, %results_63 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_64 = air.execute [%async_token_62] {
                func.call @zero_fill_sp_bf16(%results_63) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_65 = air.execute [%async_token_64, %async_token_59, %73] {
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
              %74 = air.channel.put async [%async_token_68]  @Gp2L2[%arg16, %c0_20] (%results_44[%c0_20, %c0_20, %c0_20, %c0_20] [%c8_22, %c8_22, %c8_22, %c8_22] [%c64_23, %c8_22, %c512_24, %c1_21]) {id = 69 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_69 = air.execute [%74] {
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
              affine.yield %74 : !air.async.token
            }
            affine.yield %66 : !air.async.token
          }
          %async_token_38 = air.execute [%66] {
            memref.dealloc %results_34 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_39 = air.execute [%66, %63, %61, %59, %56] {
            memref.dealloc %results_32 : memref<64x64xbf16, 2 : i32>
          }
          %68 = air.wait_all async 
          %69 = air.wait_all async 
          %async_token_40 = air.execute [%67, %66, %async_token_35] {
            memref.dealloc %results_30 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_41 = air.execute [%67, %66, %async_token_37] {
            memref.dealloc %results_28 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_42 = air.execute [%67, %66, %async_token_36] {
            memref.dealloc %results_26 : memref<64x1xbf16, 2 : i32>
          }
        }
        %43 = air.wait_all async 
        %44 = air.wait_all async 
        %45 = air.wait_all async 
        %46 = air.wait_all async 
        %47 = air.wait_all async 
        %48 = air.wait_all async 
        %49 = air.wait_all async 
        %50 = air.wait_all async 
        %51 = air.wait_all async 
        %52 = air.wait_all async 
        %53 = air.wait_all async 
        %async_token_15 = air.execute [%41] {
          memref.dealloc %results_14 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_16 = air.execute [%40] {
          memref.dealloc %results_12 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_17 = air.execute [%39] {
          memref.dealloc %results_10 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_18 = air.execute [%38] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        }
      }
    }
    return
  }
}
