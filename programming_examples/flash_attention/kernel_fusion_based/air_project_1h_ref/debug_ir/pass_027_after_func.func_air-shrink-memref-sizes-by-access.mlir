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
        %19 = air.wait_all async 
        %20 = air.wait_all async 
        %21 = air.wait_all async 
        %22 = air.wait_all async 
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
        %23 = air.wait_all async 
        %24 = air.wait_all async 
        %25 = air.wait_all async 
        %26 = air.wait_all async 
        %27 = air.wait_all async 
        %28 = air.wait_all async 
        %29 = air.wait_all async 
        %30 = scf.for %arg16 = %c0_8 to %c2_7 step %c1_6 iter_args(%arg17 = %19) -> (!air.async.token) {
          %async_token_20, %results_21 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          }
          %54 = air.channel.get async [%async_token_20, %arg17]  @VIn_0[%c0_8] (%results_21[] [] []) {id = 17 : i32} : (memref<64x64xbf16, 1 : i32>)
          %55 = air.channel.put async [%async_token_20, %54]  @V2L1_0[%c0_8, %c0_8] (%results_21[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 18 : i32} : (memref<64x64xbf16, 1 : i32>)
          %async_token_22 = air.execute [%55, %54] {
            memref.dealloc %results_21 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %55 : !air.async.token
        }
        %31 = scf.for %arg16 = %c0_8 to %c2_7 step %c1_6 iter_args(%arg17 = %20) -> (!air.async.token) {
          %async_token_20, %results_21 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          }
          %54 = air.channel.get async [%async_token_20, %arg17]  @VIn_1[%c0_8] (%results_21[] [] []) {id = 19 : i32} : (memref<64x64xbf16, 1 : i32>)
          %55 = air.channel.put async [%async_token_20, %54]  @V2L1_1[%c0_8, %c0_8] (%results_21[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 20 : i32} : (memref<64x64xbf16, 1 : i32>)
          %async_token_22 = air.execute [%55, %54] {
            memref.dealloc %results_21 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %55 : !air.async.token
        }
        %32 = scf.for %arg16 = %c0_8 to %c2_7 step %c1_6 iter_args(%arg17 = %21) -> (!air.async.token) {
          %async_token_20, %results_21 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          }
          %54 = air.channel.get async [%async_token_20, %arg17]  @VIn_2[%c0_8] (%results_21[] [] []) {id = 21 : i32} : (memref<64x64xbf16, 1 : i32>)
          %55 = air.channel.put async [%async_token_20, %54]  @V2L1_2[%c0_8, %c0_8] (%results_21[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 22 : i32} : (memref<64x64xbf16, 1 : i32>)
          %async_token_22 = air.execute [%55, %54] {
            memref.dealloc %results_21 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %55 : !air.async.token
        }
        %33 = scf.for %arg16 = %c0_8 to %c2_7 step %c1_6 iter_args(%arg17 = %22) -> (!air.async.token) {
          %async_token_20, %results_21 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          }
          %54 = air.channel.get async [%async_token_20, %arg17]  @VIn_3[%c0_8] (%results_21[] [] []) {id = 23 : i32} : (memref<64x64xbf16, 1 : i32>)
          %55 = air.channel.put async [%async_token_20, %54]  @V2L1_3[%c0_8, %c0_8] (%results_21[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_5, %c8_5, %c8_5, %c8_5] [%c8_5, %c512_4, %c64_3, %c1_6]) {id = 24 : i32} : (memref<64x64xbf16, 1 : i32>)
          %async_token_22 = air.execute [%55, %54] {
            memref.dealloc %results_21 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %55 : !air.async.token
        }
        %34 = air.channel.get async [%async_token]  @Gp2L2[%c0_8, %c0_8] (%results[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %35 = air.channel.get async [%async_token_10]  @Gp2L2[%c1_6, %c0_8] (%results_11[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %36 = air.channel.get async [%async_token_12]  @Gp2L2[%c2_7, %c0_8] (%results_13[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %37 = air.channel.get async [%async_token_14]  @Gp2L2[%c3_2, %c0_8] (%results_15[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %38 = air.channel.put async [%34]  @channel_0[%c0_8, %c0_8] (%results[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %39 = air.channel.put async [%35]  @channel_0[%c1_6, %c0_8] (%results_11[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %40 = air.channel.put async [%36]  @channel_0[%c2_7, %c0_8] (%results_13[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %41 = air.channel.put async [%37]  @channel_0[%c3_2, %c0_8] (%results_15[] [] []) : (memref<64x64xbf16, 1 : i32>)
        %42 = air.herd @herd_0 async  tile (%arg16, %arg17) in (%arg18=%c4_9, %arg19=%c4_9) attributes {id = 3 : i32, link_with = "attn.o"} {
          %c0_i32 = arith.constant 0 : i32
          %c1_i32 = arith.constant 1 : i32
          %c2_i32 = arith.constant 2 : i32
          %c3_i32 = arith.constant 3 : i32
          %c2_20 = arith.constant 2 : index
          %c0_21 = arith.constant 0 : index
          %c1_22 = arith.constant 1 : index
          %c8_23 = arith.constant 8 : index
          %c64_24 = arith.constant 64 : index
          %c512_25 = arith.constant 512 : index
          %async_token_26, %results_27 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_28, %results_29 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_30, %results_31 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %54 = air.wait_all async 
          %55 = air.wait_all async 
          %async_token_32, %results_33 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_34, %results_35 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
          }
          %async_token_36 = air.execute [%async_token_30] {
            func.call @zero_fill_gp_bf16(%results_31) : (memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_37 = air.execute [%async_token_26] {
            func.call @zero_fill_sp_bf16(%results_27) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_38 = air.execute [%async_token_28] {
            func.call @neg_inf_fill_up_bf16(%results_29) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %56 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %70 = air.channel.get async [%async_token_32]  @QK2L1_0[%arg16, %c0_21] (%results_33[] [] []) {id = 33 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %70 : !air.async.token
          } else {
            %70 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %71 = air.channel.get async [%async_token_32]  @QK2L1_1[%arg16, %c0_21] (%results_33[] [] []) {id = 34 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %71 : !air.async.token
            } else {
              %71 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %72 = air.channel.get async [%async_token_32]  @QK2L1_2[%arg16, %c0_21] (%results_33[] [] []) {id = 35 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              } else {
                %72 = air.channel.get async [%async_token_32]  @QK2L1_3[%arg16, %c0_21] (%results_33[] [] []) {id = 36 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              }
              affine.yield %71 : !air.async.token
            }
            affine.yield %70 : !air.async.token
          }
          %57 = arith.index_cast %arg16 : index to i32
          %58 = arith.cmpi eq, %57, %c0_i32 : i32
          scf.if %58 {
            %async_token_44 = air.execute [%async_token_32, %async_token_34, %56] {
              func.call @copy_tile(%results_33, %results_35) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %59 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %70 = air.channel.get async [%async_token_32]  @QK2L1_0[%arg16, %c0_21] (%results_33[] [] []) {id = 37 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %70 : !air.async.token
          } else {
            %70 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %71 = air.channel.get async [%async_token_32]  @QK2L1_1[%arg16, %c0_21] (%results_33[] [] []) {id = 38 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %71 : !air.async.token
            } else {
              %71 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %72 = air.channel.get async [%async_token_32]  @QK2L1_2[%arg16, %c0_21] (%results_33[] [] []) {id = 39 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              } else {
                %72 = air.channel.get async [%async_token_32]  @QK2L1_3[%arg16, %c0_21] (%results_33[] [] []) {id = 40 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              }
              affine.yield %71 : !air.async.token
            }
            affine.yield %70 : !air.async.token
          }
          %60 = arith.cmpi eq, %57, %c1_i32 : i32
          scf.if %60 {
            %async_token_44 = air.execute [%async_token_32, %async_token_34, %59] {
              func.call @copy_tile(%results_33, %results_35) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %61 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %70 = air.channel.get async [%async_token_32]  @QK2L1_0[%arg16, %c0_21] (%results_33[] [] []) {id = 41 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %70 : !air.async.token
          } else {
            %70 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %71 = air.channel.get async [%async_token_32]  @QK2L1_1[%arg16, %c0_21] (%results_33[] [] []) {id = 42 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %71 : !air.async.token
            } else {
              %71 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %72 = air.channel.get async [%async_token_32]  @QK2L1_2[%arg16, %c0_21] (%results_33[] [] []) {id = 43 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              } else {
                %72 = air.channel.get async [%async_token_32]  @QK2L1_3[%arg16, %c0_21] (%results_33[] [] []) {id = 44 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              }
              affine.yield %71 : !air.async.token
            }
            affine.yield %70 : !air.async.token
          }
          %62 = arith.cmpi eq, %57, %c2_i32 : i32
          scf.if %62 {
            %async_token_44 = air.execute [%async_token_32, %async_token_34, %61] {
              func.call @copy_tile(%results_33, %results_35) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %63 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
            %70 = air.channel.get async [%async_token_32]  @QK2L1_0[%arg16, %c0_21] (%results_33[] [] []) {id = 45 : i32} : (memref<64x64xbf16, 2 : i32>)
            affine.yield %70 : !air.async.token
          } else {
            %70 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
              %71 = air.channel.get async [%async_token_32]  @QK2L1_1[%arg16, %c0_21] (%results_33[] [] []) {id = 46 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %71 : !air.async.token
            } else {
              %71 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                %72 = air.channel.get async [%async_token_32]  @QK2L1_2[%arg16, %c0_21] (%results_33[] [] []) {id = 47 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              } else {
                %72 = air.channel.get async [%async_token_32]  @QK2L1_3[%arg16, %c0_21] (%results_33[] [] []) {id = 48 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %72 : !air.async.token
              }
              affine.yield %71 : !air.async.token
            }
            affine.yield %70 : !air.async.token
          }
          %64 = arith.cmpi eq, %57, %c3_i32 : i32
          scf.if %64 {
            %async_token_44 = air.execute [%async_token_32, %async_token_34, %63] {
              func.call @copy_tile(%results_33, %results_35) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %65 = air.wait_all async [%async_token_36, %async_token_37, %async_token_38] 
          %66 = scf.for %arg20 = %c0_21 to %c2_20 step %c1_22 iter_args(%arg21 = %65) -> (!air.async.token) {
            %async_token_44, %results_45 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
            }
            %async_token_46, %results_47 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
            }
            %async_token_48 = air.execute [%async_token_46, %arg21] {
              %collapse_shape = memref.collapse_shape %results_47 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            }
            %70 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %76 = air.channel.get async [%async_token_32, %arg21]  @QK2L1_0[%arg16, %c0_21] (%results_33[] [] []) {id = 49 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %76 : !air.async.token
            } else {
              %76 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %77 = air.channel.get async [%async_token_32, %arg21]  @QK2L1_1[%arg16, %c0_21] (%results_33[] [] []) {id = 50 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %77 : !air.async.token
              } else {
                %77 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %78 = air.channel.get async [%async_token_32, %arg21]  @QK2L1_2[%arg16, %c0_21] (%results_33[] [] []) {id = 51 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %78 : !air.async.token
                } else {
                  %78 = air.channel.get async [%async_token_32, %arg21]  @QK2L1_3[%arg16, %c0_21] (%results_33[] [] []) {id = 52 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %78 : !air.async.token
                }
                affine.yield %77 : !air.async.token
              }
              affine.yield %76 : !air.async.token
            }
            %71 = affine.if #set3()[%arg16, %arg17] -> !air.async.token {
              %76 = air.channel.get async [%async_token_44]  @V2L1_0[%arg16, %arg17] (%results_45[] [] []) {id = 53 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %76 : !air.async.token
            } else {
              %76 = air.wait_all async 
              affine.yield %76 : !air.async.token
            }
            %72 = affine.if #set4()[%arg16, %arg17] -> !air.async.token {
              %76 = air.channel.get async [%async_token_44, %71, %arg21]  @V2L1_1[%arg16, %arg17] (%results_45[] [] []) {id = 54 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %76 : !air.async.token
            } else {
              %76 = air.wait_all async 
              affine.yield %76 : !air.async.token
            }
            %73 = affine.if #set5()[%arg16, %arg17] -> !air.async.token {
              %76 = air.channel.get async [%async_token_44, %72]  @V2L1_2[%arg16, %arg17] (%results_45[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %76 : !air.async.token
            } else {
              %76 = air.wait_all async 
              affine.yield %76 : !air.async.token
            }
            %74 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
              %76 = air.channel.get async [%async_token_44, %73]  @V2L1_3[%arg16, %arg17] (%results_45[] [] []) {id = 56 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %76 : !air.async.token
            } else {
              %76 = air.wait_all async 
              affine.yield %76 : !air.async.token
            }
            %async_token_49 = air.execute [%async_token_46, %async_token_32, %async_token_34, %async_token_48, %70] {
              %collapse_shape = memref.collapse_shape %results_47 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_a_b_bf16(%results_35, %results_33, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            }
            %async_token_50, %results_51 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_52, %results_53 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_54 = air.execute [%async_token_28, %async_token_46, %async_token_52, %async_token_50, %async_token_49] {
              %collapse_shape = memref.collapse_shape %results_47 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @fused_softmax(%collapse_shape, %results_29, %results_51, %results_53) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_55 = air.execute [%async_token_30, %async_token_54] {
              func.call @mul_r_gp(%results_53, %results_31) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_56 = air.execute [%async_token_30, %async_token_46, %async_token_44, %74, %async_token_55] {
              %collapse_shape = memref.collapse_shape %results_47 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_g_b_bf16(%collapse_shape, %results_45, %results_31) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_57 = air.execute [%async_token_26, %async_token_55] {
              func.call @accum_sp_r_s(%results_27, %results_53, %results_51) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_58 = air.execute [%async_token_26, %async_token_57] {
              func.call @vector_copy_32elems(%c0_i32, %results_51, %results_27) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_59 = air.execute [%async_token_58] {
              memref.dealloc %results_51 : memref<64x1xbf16, 2 : i32>
            }
            %async_token_60 = air.execute [%async_token_57] {
              memref.dealloc %results_53 : memref<64x1xbf16, 2 : i32>
            }
            %75 = air.wait_all async [%async_token_56, %async_token_58] 
            %async_token_61 = air.execute [%async_token_56, %async_token_54, %async_token_49, %async_token_48] {
              memref.dealloc %results_47 : memref<64x64xbf16, 2 : i32>
            }
            %async_token_62 = air.execute [%async_token_56, %74, %73, %72, %71] {
              memref.dealloc %results_45 : memref<64x64xbf16, 2 : i32>
            }
            scf.yield %75 : !air.async.token
          }
          %67 = affine.if #set6()[%arg16, %arg17] -> !air.async.token {
            %70 = arith.subi %arg17, %c1_22 : index
            %71 = air.channel.put async [%async_token_30, %66]  @cascade_gp[%arg16, %70] (%results_31[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 2 : i32>)
            %72 = air.channel.put async [%async_token_28, %66]  @cascade_up[%arg16, %70] (%results_29[] [] []) {id = 58 : i32} : (memref<64x1xbf16, 2 : i32>)
            %73 = air.channel.put async [%async_token_26, %66]  @cascade_sp[%arg16, %70] (%results_27[] [] []) {id = 59 : i32} : (memref<64x1xbf16, 2 : i32>)
            %74 = air.wait_all async [%71, %72, %73] 
            affine.yield %74 : !air.async.token
          } else {
            %70 = affine.if #set7()[%arg16, %arg17] -> !air.async.token {
              %async_token_44, %results_45 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_46, %results_47 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_48, %results_49 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %71 = air.channel.get async [%async_token_44]  @cascade_gp[%arg16, %arg17] (%results_45[] [] []) {id = 60 : i32} : (memref<64x64xbf16, 2 : i32>)
              %72 = air.channel.get async [%async_token_46]  @cascade_up[%arg16, %arg17] (%results_47[] [] []) {id = 61 : i32} : (memref<64x1xbf16, 2 : i32>)
              %73 = air.channel.get async [%async_token_48]  @cascade_sp[%arg16, %arg17] (%results_49[] [] []) {id = 62 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_50, %results_51 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_52 = air.execute [%async_token_28, %async_token_50, %66] {
                func.call @vector_copy_32elems(%c0_i32, %results_29, %results_51) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_53 = air.execute [%async_token_28, %async_token_52, %72] {
                func.call @maximum_up_u_bf16(%results_47, %results_29) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_54, %results_55 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_56 = air.execute [%async_token_28, %async_token_54, %async_token_53] {
                func.call @exp_up_minus_u(%results_47, %results_29, %results_55) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_57, %results_58 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_59 = air.execute [%async_token_28, %async_token_57, %async_token_56] {
                func.call @exp_up_minus_u(%results_51, %results_29, %results_58) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_60 = air.execute [%async_token_56, %71] {
                func.call @mul_r_gp(%results_55, %results_45) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_61 = air.execute [%async_token_30, %async_token_59] {
                func.call @mul_r_gp(%results_58, %results_31) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_62 = air.execute [%async_token_30, %async_token_61, %async_token_60] {
                func.call @add_gp_g(%results_31, %results_45) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_63, %results_64 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_65 = air.execute [%async_token_63] {
                func.call @zero_fill_sp_bf16(%results_64) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_66 = air.execute [%async_token_65, %async_token_60, %73] {
                func.call @accum_sp_r_s(%results_49, %results_55, %results_64) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_67 = air.execute [%async_token_26, %async_token_66, %async_token_61] {
                func.call @accum_sp_r_s(%results_27, %results_58, %results_64) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_68 = air.execute [%async_token_67] {
                func.call @vector_copy_32elems(%c0_i32, %results_64, %results_49) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %74 = arith.subi %arg17, %c1_22 : index
              %75 = air.channel.put async [%async_token_62]  @cascade_gp[%arg16, %74] (%results_45[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 2 : i32>)
              %76 = air.channel.put async [%async_token_28, %async_token_59]  @cascade_up[%arg16, %74] (%results_29[] [] []) {id = 64 : i32} : (memref<64x1xbf16, 2 : i32>)
              %77 = air.channel.put async [%async_token_68]  @cascade_sp[%arg16, %74] (%results_49[] [] []) {id = 65 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_69 = air.execute [%75] {
                memref.dealloc %results_45 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_70 = air.execute [%async_token_56] {
                memref.dealloc %results_47 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_71 = air.execute [%77] {
                memref.dealloc %results_49 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_72 = air.execute [%async_token_59] {
                memref.dealloc %results_51 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_73 = air.execute [%async_token_66] {
                memref.dealloc %results_55 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_74 = air.execute [%async_token_67] {
                memref.dealloc %results_58 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_75 = air.execute [%async_token_68] {
                memref.dealloc %results_64 : memref<64x1xbf16, 2 : i32>
              }
              %78 = air.wait_all async [%75, %76, %77] 
              affine.yield %78 : !air.async.token
            } else {
              %async_token_44, %results_45 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
              }
              %async_token_46, %results_47 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_48, %results_49 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %71 = air.channel.get async [%async_token_44]  @cascade_gp[%arg16, %arg17] (%results_45[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 2 : i32>)
              %72 = air.channel.get async [%async_token_46]  @cascade_up[%arg16, %arg17] (%results_47[] [] []) {id = 67 : i32} : (memref<64x1xbf16, 2 : i32>)
              %73 = air.channel.get async [%async_token_48]  @cascade_sp[%arg16, %arg17] (%results_49[] [] []) {id = 68 : i32} : (memref<64x1xbf16, 2 : i32>)
              %async_token_50, %results_51 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_52 = air.execute [%async_token_28, %async_token_50, %66] {
                func.call @vector_copy_32elems(%c0_i32, %results_29, %results_51) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_53 = air.execute [%async_token_28, %async_token_52, %72] {
                func.call @maximum_up_u_bf16(%results_47, %results_29) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_54, %results_55 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_56 = air.execute [%async_token_28, %async_token_54, %async_token_53] {
                func.call @exp_up_minus_u(%results_47, %results_29, %results_55) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_57, %results_58 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_59 = air.execute [%async_token_28, %async_token_57, %async_token_56] {
                func.call @exp_up_minus_u(%results_51, %results_29, %results_58) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_60 = air.execute [%async_token_56, %71] {
                func.call @mul_r_gp(%results_55, %results_45) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_61 = air.execute [%async_token_30, %async_token_59] {
                func.call @mul_r_gp(%results_58, %results_31) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_62 = air.execute [%async_token_30, %async_token_61, %async_token_60] {
                func.call @add_gp_g(%results_31, %results_45) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %async_token_63, %results_64 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
                %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
                air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
              }
              %async_token_65 = air.execute [%async_token_63] {
                func.call @zero_fill_sp_bf16(%results_64) : (memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_66 = air.execute [%async_token_65, %async_token_60, %73] {
                func.call @accum_sp_r_s(%results_49, %results_55, %results_64) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_67 = air.execute [%async_token_26, %async_token_66, %async_token_61] {
                func.call @accum_sp_r_s(%results_27, %results_58, %results_64) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_68 = air.execute [%async_token_67] {
                func.call @vector_copy_32elems(%c0_i32, %results_64, %results_49) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
              }
              %async_token_69 = air.execute [%async_token_68, %async_token_62] {
                func.call @div_gp_sp(%results_49, %results_45) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
              }
              %74 = air.channel.put async [%async_token_69]  @Gp2L2[%arg16, %c0_21] (%results_45[%c0_21, %c0_21, %c0_21, %c0_21] [%c8_23, %c8_23, %c8_23, %c8_23] [%c64_24, %c8_23, %c512_25, %c1_22]) {id = 69 : i32} : (memref<64x64xbf16, 2 : i32>)
              %async_token_70 = air.execute [%74] {
                memref.dealloc %results_45 : memref<64x64xbf16, 2 : i32>
              }
              %async_token_71 = air.execute [%async_token_56] {
                memref.dealloc %results_47 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_72 = air.execute [%async_token_69] {
                memref.dealloc %results_49 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_73 = air.execute [%async_token_59] {
                memref.dealloc %results_51 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_74 = air.execute [%async_token_66] {
                memref.dealloc %results_55 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_75 = air.execute [%async_token_67] {
                memref.dealloc %results_58 : memref<64x1xbf16, 2 : i32>
              }
              %async_token_76 = air.execute [%async_token_68] {
                memref.dealloc %results_64 : memref<64x1xbf16, 2 : i32>
              }
              affine.yield %74 : !air.async.token
            }
            affine.yield %66 : !air.async.token
          }
          %async_token_39 = air.execute [%66] {
            memref.dealloc %results_35 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_40 = air.execute [%66, %63, %61, %59, %56] {
            memref.dealloc %results_33 : memref<64x64xbf16, 2 : i32>
          }
          %68 = air.wait_all async 
          %69 = air.wait_all async 
          %async_token_41 = air.execute [%67, %66, %async_token_36] {
            memref.dealloc %results_31 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_42 = air.execute [%67, %66, %async_token_38] {
            memref.dealloc %results_29 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_43 = air.execute [%67, %66, %async_token_37] {
            memref.dealloc %results_27 : memref<64x1xbf16, 2 : i32>
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
        %async_token_16 = air.execute [%41] {
          memref.dealloc %results_15 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_17 = air.execute [%40] {
          memref.dealloc %results_13 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_18 = air.execute [%39] {
          memref.dealloc %results_11 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_19 = air.execute [%38] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        }
      }
    }
    return
  }
}
