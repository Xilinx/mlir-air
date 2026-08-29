#map = affine_map<()[s0] -> (s0 * 256)>
#map1 = affine_map<()[s0] -> (s0 * 128)>
#map2 = affine_map<()[s0] -> (s0 * 512)>
#map3 = affine_map<()[s0] -> (s0 * 512 + 64)>
#map4 = affine_map<()[s0] -> (s0 * 512 + 128)>
#map5 = affine_map<()[s0] -> (s0 * 512 + 192)>
#map6 = affine_map<()[s0] -> (s0 * 128 + 64)>
#map7 = affine_map<()[s0] -> (s0 * 512 + 256)>
#map8 = affine_map<()[s0] -> (s0 * 512 + 320)>
#map9 = affine_map<()[s0] -> (s0 * 512 + 384)>
#map10 = affine_map<()[s0] -> (s0 * 512 + 448)>
#set = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 == 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 1 == 0)>
#set2 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 2 == 0)>
module {
  func.func private @zero_fill_g_bf16(memref<4096xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @zero_fill_gp_bf16(memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @zero_fill_sp_bf16(memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @neg_inf_fill_up_bf16(memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @matmul_a_b_bf16(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @matmul_g_b_bf16(memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @fused_softmax(memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @maximum_up_u_bf16(memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @exp_up_minus_u(memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @mul_r_gp(memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @accum_sp_r_s(memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @vector_copy_32elems(i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @copy_tile(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @div_gp_sp(memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @add_gp_g(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  func.func private @apply_causal_mask(memref<64x64xbf16, 2 : i32>, i32, i32) attributes {link_with = "attn_npu2.o", llvm.emit_c_interface}
  air.channel @Q2L1_0_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @Q2L1_0_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @Q2L1_0_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @Q2L1_0_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @Q2L1_1_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @Q2L1_1_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @Q2L1_1_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @Q2L1_1_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @QIn [2]
  air.channel @K2L1_0_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @K2L1_0_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @K2L1_0_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @K2L1_0_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @K2L1_1_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @K2L1_1_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @K2L1_1_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @K2L1_1_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @KIn [2]
  air.channel @V2L1_0_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @V2L1_0_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @V2L1_0_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @V2L1_0_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @V2L1_1_0 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @V2L1_1_1 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @V2L1_1_2 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @V2L1_1_3 [1, 1, 1] {broadcast_shape = [1, 1, 4 : index]}
  air.channel @VIn [2]
  air.channel @Gp2L2 [4, 4]
  air.channel @GpOut [2]
  func.func @attention_bf16(%arg0: memref<512x512xbf16>, %arg1: memref<512x128xbf16>, %arg2: memref<512x128xbf16>, %arg3: memref<512x512xbf16>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c2, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<512x512xbf16>, memref<512x128xbf16>, memref<512x128xbf16>, memref<512x512xbf16> attributes {id = 1 : i32} {
      %c256 = arith.constant 256 : index
      %c2_0 = arith.constant 2 : index
      %c512 = arith.constant 512 : index
      %c32768 = arith.constant 32768 : index
      %c4 = arith.constant 4 : index
      %c128 = arith.constant 128 : index
      %c8192 = arith.constant 8192 : index
      %c64 = arith.constant 64 : index
      %c1_1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c0 = arith.constant 0 : index
      %1 = affine.apply #map()[%arg4]
      %2 = affine.apply #map1()[%arg5]
      %3 = air.channel.put async  @KIn[%c0] (%arg9[%c0, %2] [%c8, %c1_1, %c64, %c64] [%c8192, %c64, %c128, %c1_1]) {id = 1 : i32} : (memref<512x128xbf16>)
      %4 = air.channel.put async  @VIn[%c0] (%arg10[%c0, %2] [%c8, %c64, %c64] [%c8192, %c128, %c1_1]) {id = 2 : i32} : (memref<512x128xbf16>)
      %5 = affine.apply #map2()[%arg5]
      %6 = air.channel.put async  @QIn[%c0] (%arg8[%1, %5] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) {id = 3 : i32} : (memref<512x512xbf16>)
      %7 = affine.apply #map3()[%arg5]
      %8 = air.channel.put async  @QIn[%c0] (%arg8[%1, %7] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) {id = 4 : i32} : (memref<512x512xbf16>)
      %9 = affine.apply #map4()[%arg5]
      %10 = air.channel.put async  @QIn[%c0] (%arg8[%1, %9] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) {id = 5 : i32} : (memref<512x512xbf16>)
      %11 = affine.apply #map5()[%arg5]
      %12 = air.channel.put async  @QIn[%c0] (%arg8[%1, %11] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) {id = 6 : i32} : (memref<512x512xbf16>)
      %13 = affine.apply #map6()[%arg5]
      %14 = air.channel.put async  @KIn[%c1_1] (%arg9[%c0, %13] [%c8, %c1_1, %c64, %c64] [%c8192, %c64, %c128, %c1_1]) {id = 7 : i32} : (memref<512x128xbf16>)
      %15 = air.channel.put async  @VIn[%c1_1] (%arg10[%c0, %13] [%c8, %c64, %c64] [%c8192, %c128, %c1_1]) {id = 8 : i32} : (memref<512x128xbf16>)
      %16 = affine.apply #map7()[%arg5]
      %17 = air.channel.put async  @QIn[%c1_1] (%arg8[%1, %16] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) {id = 9 : i32} : (memref<512x512xbf16>)
      %18 = affine.apply #map8()[%arg5]
      %19 = air.channel.put async  @QIn[%c1_1] (%arg8[%1, %18] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) {id = 10 : i32} : (memref<512x512xbf16>)
      %20 = affine.apply #map9()[%arg5]
      %21 = air.channel.put async  @QIn[%c1_1] (%arg8[%1, %20] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) {id = 11 : i32} : (memref<512x512xbf16>)
      %22 = affine.apply #map10()[%arg5]
      %23 = air.channel.put async  @QIn[%c1_1] (%arg8[%1, %22] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) {id = 12 : i32} : (memref<512x512xbf16>)
      %24 = air.segment @attn_seg async  unroll(%arg12, %arg13) in (%arg14=%c2_0, %arg15=%c1_1) attributes {id = 2 : i32} {
        %c192 = arith.constant 192 : index
        %c128_2 = arith.constant 128 : index
        %c3 = arith.constant 3 : index
        %c2_3 = arith.constant 2 : index
        %c64_4 = arith.constant 64 : index
        %c512_5 = arith.constant 512 : index
        %c1_6 = arith.constant 1 : index
        %c8_7 = arith.constant 8 : index
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
        %async_token_12, %results_13 = air.execute -> (memref<256x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<256x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<256x64xbf16, 1 : i32>
        }
        %async_token_14, %results_15 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_16, %results_17 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_18, %results_19 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_20, %results_21 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_22, %results_23 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_24, %results_25 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        }
        %async_token_26, %results_27 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        }
        %async_token_28, %results_29 = air.execute -> (memref<3xi32, 2 : i32>) {
          %alloc = memref.alloc() : memref<3xi32, 2 : i32>
          air.execute_terminator %alloc : memref<3xi32, 2 : i32>
        }
        %33 = scf.for %arg16 = %c0_8 to %c4_9 step %c1_6 iter_args(%arg17 = %async_token) -> (!air.async.token) {
          %84 = air.channel.get async [%arg17]  @QIn[%arg12] (%results[] [] []) {id = 13 : i32} : (memref<64x64xbf16, 1 : i32>)
          %85 = arith.cmpi eq, %arg12, %c0_8 : index
          %86 = scf.if %85 -> (!air.async.token) {
            %87 = air.channel.put async [%84]  @Q2L1_0_0[%c0_8, %c0_8, %c0_8] (%results[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_7, %c8_7, %c8_7, %c8_7] [%c8_7, %c512_5, %c64_4, %c1_6]) {id = 14 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %87 : !air.async.token
          } else {
            %87 = air.channel.put async [%84]  @Q2L1_1_0[%c0_8, %c0_8, %c0_8] (%results[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_7, %c8_7, %c8_7, %c8_7] [%c8_7, %c512_5, %c64_4, %c1_6]) {id = 15 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %87 : !air.async.token
          }
          scf.yield %86 : !air.async.token
        }
        %34 = scf.for %arg16 = %c0_8 to %c4_9 step %c1_6 iter_args(%arg17 = %33) -> (!air.async.token) {
          %84 = air.channel.get async [%arg17]  @QIn[%arg12] (%results[] [] []) {id = 16 : i32} : (memref<64x64xbf16, 1 : i32>)
          %85 = arith.cmpi eq, %arg12, %c0_8 : index
          %86 = scf.if %85 -> (!air.async.token) {
            %87 = air.channel.put async [%84]  @Q2L1_0_1[%c0_8, %c0_8, %c0_8] (%results[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_7, %c8_7, %c8_7, %c8_7] [%c8_7, %c512_5, %c64_4, %c1_6]) {id = 17 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %87 : !air.async.token
          } else {
            %87 = air.channel.put async [%84]  @Q2L1_1_1[%c0_8, %c0_8, %c0_8] (%results[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_7, %c8_7, %c8_7, %c8_7] [%c8_7, %c512_5, %c64_4, %c1_6]) {id = 18 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %87 : !air.async.token
          }
          scf.yield %86 : !air.async.token
        }
        %35 = scf.for %arg16 = %c0_8 to %c4_9 step %c1_6 iter_args(%arg17 = %34) -> (!air.async.token) {
          %84 = air.channel.get async [%arg17]  @QIn[%arg12] (%results[] [] []) {id = 19 : i32} : (memref<64x64xbf16, 1 : i32>)
          %85 = arith.cmpi eq, %arg12, %c0_8 : index
          %86 = scf.if %85 -> (!air.async.token) {
            %87 = air.channel.put async [%84]  @Q2L1_0_2[%c0_8, %c0_8, %c0_8] (%results[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_7, %c8_7, %c8_7, %c8_7] [%c8_7, %c512_5, %c64_4, %c1_6]) {id = 20 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %87 : !air.async.token
          } else {
            %87 = air.channel.put async [%84]  @Q2L1_1_2[%c0_8, %c0_8, %c0_8] (%results[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_7, %c8_7, %c8_7, %c8_7] [%c8_7, %c512_5, %c64_4, %c1_6]) {id = 21 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %87 : !air.async.token
          }
          scf.yield %86 : !air.async.token
        }
        %36 = scf.for %arg16 = %c0_8 to %c4_9 step %c1_6 iter_args(%arg17 = %35) -> (!air.async.token) {
          %84 = air.channel.get async [%arg17]  @QIn[%arg12] (%results[] [] []) {id = 22 : i32} : (memref<64x64xbf16, 1 : i32>)
          %85 = arith.cmpi eq, %arg12, %c0_8 : index
          %86 = scf.if %85 -> (!air.async.token) {
            %87 = air.channel.put async [%84]  @Q2L1_0_3[%c0_8, %c0_8, %c0_8] (%results[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_7, %c8_7, %c8_7, %c8_7] [%c8_7, %c512_5, %c64_4, %c1_6]) {id = 23 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %87 : !air.async.token
          } else {
            %87 = air.channel.put async [%84]  @Q2L1_1_3[%c0_8, %c0_8, %c0_8] (%results[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_7, %c8_7, %c8_7, %c8_7] [%c8_7, %c512_5, %c64_4, %c1_6]) {id = 24 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %87 : !air.async.token
          }
          scf.yield %86 : !air.async.token
        }
        %37 = scf.for %arg16 = %c0_8 to %c8_7 step %c1_6 iter_args(%arg17 = %36) -> (!air.async.token) {
          %84 = air.channel.get async [%arg17]  @KIn[%arg12] (%results[] [] []) {id = 25 : i32} : (memref<64x64xbf16, 1 : i32>)
          %85 = arith.cmpi eq, %arg12, %c0_8 : index
          %86:4 = scf.if %85 -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
            %88 = air.channel.put async [%84]  @K2L1_0_0[%c0_8, %c0_8, %c0_8] (%results[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_7, %c8_7, %c8_7, %c8_7] [%c8_7, %c512_5, %c64_4, %c1_6]) {id = 26 : i32} : (memref<64x64xbf16, 1 : i32>)
            %89 = air.channel.put async [%84]  @K2L1_0_1[%c0_8, %c0_8, %c0_8] (%results[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_7, %c8_7, %c8_7, %c8_7] [%c8_7, %c512_5, %c64_4, %c1_6]) {id = 27 : i32} : (memref<64x64xbf16, 1 : i32>)
            %90 = air.channel.put async [%84]  @K2L1_0_2[%c0_8, %c0_8, %c0_8] (%results[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_7, %c8_7, %c8_7, %c8_7] [%c8_7, %c512_5, %c64_4, %c1_6]) {id = 28 : i32} : (memref<64x64xbf16, 1 : i32>)
            %91 = air.channel.put async [%84]  @K2L1_0_3[%c0_8, %c0_8, %c0_8] (%results[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_7, %c8_7, %c8_7, %c8_7] [%c8_7, %c512_5, %c64_4, %c1_6]) {id = 29 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %88, %89, %90, %91 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
          } else {
            %88 = air.channel.put async [%84]  @K2L1_1_0[%c0_8, %c0_8, %c0_8] (%results[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_7, %c8_7, %c8_7, %c8_7] [%c8_7, %c512_5, %c64_4, %c1_6]) {id = 30 : i32} : (memref<64x64xbf16, 1 : i32>)
            %89 = air.channel.put async [%84]  @K2L1_1_1[%c0_8, %c0_8, %c0_8] (%results[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_7, %c8_7, %c8_7, %c8_7] [%c8_7, %c512_5, %c64_4, %c1_6]) {id = 31 : i32} : (memref<64x64xbf16, 1 : i32>)
            %90 = air.channel.put async [%84]  @K2L1_1_2[%c0_8, %c0_8, %c0_8] (%results[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_7, %c8_7, %c8_7, %c8_7] [%c8_7, %c512_5, %c64_4, %c1_6]) {id = 32 : i32} : (memref<64x64xbf16, 1 : i32>)
            %91 = air.channel.put async [%84]  @K2L1_1_3[%c0_8, %c0_8, %c0_8] (%results[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_7, %c8_7, %c8_7, %c8_7] [%c8_7, %c512_5, %c64_4, %c1_6]) {id = 33 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %88, %89, %90, %91 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
          }
          %87 = air.wait_all async [%86#0, %86#1, %86#2, %86#3] 
          scf.yield %87 : !air.async.token
        }
        %38 = scf.for %arg16 = %c0_8 to %c8_7 step %c1_6 iter_args(%arg17 = %async_token_10) -> (!air.async.token) {
          %84 = air.channel.get async [%arg17]  @VIn[%arg12] (%results_11[] [] []) {id = 34 : i32} : (memref<64x64xbf16, 1 : i32>)
          %85 = arith.cmpi eq, %arg12, %c0_8 : index
          %86:4 = scf.if %85 -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
            %88 = air.channel.put async [%84]  @V2L1_0_0[%c0_8, %c0_8, %c0_8] (%results_11[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_7, %c8_7, %c8_7, %c8_7] [%c8_7, %c512_5, %c64_4, %c1_6]) {id = 35 : i32} : (memref<64x64xbf16, 1 : i32>)
            %89 = air.channel.put async [%84]  @V2L1_0_1[%c0_8, %c0_8, %c0_8] (%results_11[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_7, %c8_7, %c8_7, %c8_7] [%c8_7, %c512_5, %c64_4, %c1_6]) {id = 36 : i32} : (memref<64x64xbf16, 1 : i32>)
            %90 = air.channel.put async [%84]  @V2L1_0_2[%c0_8, %c0_8, %c0_8] (%results_11[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_7, %c8_7, %c8_7, %c8_7] [%c8_7, %c512_5, %c64_4, %c1_6]) {id = 37 : i32} : (memref<64x64xbf16, 1 : i32>)
            %91 = air.channel.put async [%84]  @V2L1_0_3[%c0_8, %c0_8, %c0_8] (%results_11[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_7, %c8_7, %c8_7, %c8_7] [%c8_7, %c512_5, %c64_4, %c1_6]) {id = 38 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %88, %89, %90, %91 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
          } else {
            %88 = air.channel.put async [%84]  @V2L1_1_0[%c0_8, %c0_8, %c0_8] (%results_11[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_7, %c8_7, %c8_7, %c8_7] [%c8_7, %c512_5, %c64_4, %c1_6]) {id = 39 : i32} : (memref<64x64xbf16, 1 : i32>)
            %89 = air.channel.put async [%84]  @V2L1_1_1[%c0_8, %c0_8, %c0_8] (%results_11[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_7, %c8_7, %c8_7, %c8_7] [%c8_7, %c512_5, %c64_4, %c1_6]) {id = 40 : i32} : (memref<64x64xbf16, 1 : i32>)
            %90 = air.channel.put async [%84]  @V2L1_1_2[%c0_8, %c0_8, %c0_8] (%results_11[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_7, %c8_7, %c8_7, %c8_7] [%c8_7, %c512_5, %c64_4, %c1_6]) {id = 41 : i32} : (memref<64x64xbf16, 1 : i32>)
            %91 = air.channel.put async [%84]  @V2L1_1_3[%c0_8, %c0_8, %c0_8] (%results_11[%c0_8, %c0_8, %c0_8, %c0_8] [%c8_7, %c8_7, %c8_7, %c8_7] [%c8_7, %c512_5, %c64_4, %c1_6]) {id = 42 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %88, %89, %90, %91 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
          }
          %87 = air.wait_all async [%86#0, %86#1, %86#2, %86#3] 
          scf.yield %87 : !air.async.token
        }
        %39 = air.herd @herd_0 async [%async_token_14, %async_token_16, %async_token_18, %async_token_20, %async_token_22, %async_token_24, %async_token_26, %async_token_28]  tile (%arg16, %arg17) in (%arg18=%c4_9, %arg19=%c4_9) args(%arg20=%results_15, %arg21=%results_17, %arg22=%results_19, %arg23=%results_21, %arg24=%results_23, %arg25=%results_25, %arg26=%results_27, %arg27=%arg12, %arg28=%results_29) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, index, memref<3xi32, 2 : i32> attributes {id = 3 : i32, link_with = "attn_npu2.o"} {
          %c4_i32 = arith.constant 4 : i32
          %c512_41 = arith.constant 512 : index
          %c64_42 = arith.constant 64 : index
          %c8_43 = arith.constant 8 : index
          %c3_i32 = arith.constant 3 : i32
          %c2_i32 = arith.constant 2 : i32
          %c1_i32 = arith.constant 1 : i32
          %c0_i32 = arith.constant 0 : i32
          %c2_44 = arith.constant 2 : index
          %c1_45 = arith.constant 1 : index
          %c0_46 = arith.constant 0 : index
          %async_token_47 = air.execute {
            func.call @zero_fill_gp_bf16(%arg24) : (memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_48 = air.execute {
            func.call @zero_fill_sp_bf16(%arg26) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_49 = air.execute {
            func.call @neg_inf_fill_up_bf16(%arg25) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_50, %results_51 = air.execute -> (i32) {
            %97 = memref.load %arg28[%c1_45] : memref<3xi32, 2 : i32>
            air.execute_terminator %97 : i32
          }
          %84 = arith.cmpi eq, %results_51, %c0_i32 : i32
          scf.if %84 {
            %async_token_55 = air.execute [%async_token_50] {
              memref.store %c0_i32, %arg28[%c0_46] : memref<3xi32, 2 : i32>
            }
            %async_token_56 = air.execute [%async_token_55] {
              memref.store %c1_i32, %arg28[%c1_45] : memref<3xi32, 2 : i32>
            }
            %async_token_57 = air.execute [%async_token_56] {
              memref.store %c0_i32, %arg28[%c2_44] : memref<3xi32, 2 : i32>
            }
          }
          %85 = arith.cmpi eq, %arg27, %c0_46 : index
          scf.if %85 {
            %97 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %98 = air.channel.get async  @Q2L1_0_0[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 43 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %98 : !air.async.token
            } else {
              %98 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %99 = air.channel.get async  @Q2L1_0_1[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 44 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %99 : !air.async.token
              } else {
                %99 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %100 = air.channel.get async  @Q2L1_0_2[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 45 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %100 : !air.async.token
                } else {
                  %100 = air.channel.get async  @Q2L1_0_3[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 46 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %100 : !air.async.token
                }
                affine.yield %99 : !air.async.token
              }
              affine.yield %98 : !air.async.token
            }
          } else {
            %97 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %98 = air.channel.get async  @Q2L1_1_0[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 47 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %98 : !air.async.token
            } else {
              %98 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %99 = air.channel.get async  @Q2L1_1_1[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 48 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %99 : !air.async.token
              } else {
                %99 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %100 = air.channel.get async  @Q2L1_1_2[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 49 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %100 : !air.async.token
                } else {
                  %100 = air.channel.get async  @Q2L1_1_3[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 50 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %100 : !air.async.token
                }
                affine.yield %99 : !air.async.token
              }
              affine.yield %98 : !air.async.token
            }
          }
          %86 = arith.index_cast %arg16 : index to i32
          %87 = arith.cmpi eq, %86, %c0_i32 : i32
          scf.if %87 {
            %async_token_55 = air.execute {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %85 {
            %97 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %98 = air.channel.get async  @Q2L1_0_0[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 51 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %98 : !air.async.token
            } else {
              %98 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %99 = air.channel.get async  @Q2L1_0_1[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 52 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %99 : !air.async.token
              } else {
                %99 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %100 = air.channel.get async  @Q2L1_0_2[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 53 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %100 : !air.async.token
                } else {
                  %100 = air.channel.get async  @Q2L1_0_3[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 54 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %100 : !air.async.token
                }
                affine.yield %99 : !air.async.token
              }
              affine.yield %98 : !air.async.token
            }
          } else {
            %97 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %98 = air.channel.get async  @Q2L1_1_0[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %98 : !air.async.token
            } else {
              %98 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %99 = air.channel.get async  @Q2L1_1_1[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 56 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %99 : !air.async.token
              } else {
                %99 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %100 = air.channel.get async  @Q2L1_1_2[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %100 : !air.async.token
                } else {
                  %100 = air.channel.get async  @Q2L1_1_3[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 58 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %100 : !air.async.token
                }
                affine.yield %99 : !air.async.token
              }
              affine.yield %98 : !air.async.token
            }
          }
          %88 = arith.cmpi eq, %86, %c1_i32 : i32
          scf.if %88 {
            %async_token_55 = air.execute {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %85 {
            %97 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %98 = air.channel.get async  @Q2L1_0_0[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 59 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %98 : !air.async.token
            } else {
              %98 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %99 = air.channel.get async  @Q2L1_0_1[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 60 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %99 : !air.async.token
              } else {
                %99 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %100 = air.channel.get async  @Q2L1_0_2[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 61 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %100 : !air.async.token
                } else {
                  %100 = air.channel.get async  @Q2L1_0_3[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 62 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %100 : !air.async.token
                }
                affine.yield %99 : !air.async.token
              }
              affine.yield %98 : !air.async.token
            }
          } else {
            %97 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %98 = air.channel.get async  @Q2L1_1_0[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %98 : !air.async.token
            } else {
              %98 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %99 = air.channel.get async  @Q2L1_1_1[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 64 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %99 : !air.async.token
              } else {
                %99 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %100 = air.channel.get async  @Q2L1_1_2[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 65 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %100 : !air.async.token
                } else {
                  %100 = air.channel.get async  @Q2L1_1_3[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %100 : !air.async.token
                }
                affine.yield %99 : !air.async.token
              }
              affine.yield %98 : !air.async.token
            }
          }
          %89 = arith.cmpi eq, %86, %c2_i32 : i32
          scf.if %89 {
            %async_token_55 = air.execute {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %85 {
            %97 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %98 = air.channel.get async  @Q2L1_0_0[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 67 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %98 : !air.async.token
            } else {
              %98 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %99 = air.channel.get async  @Q2L1_0_1[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 68 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %99 : !air.async.token
              } else {
                %99 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %100 = air.channel.get async  @Q2L1_0_2[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 69 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %100 : !air.async.token
                } else {
                  %100 = air.channel.get async  @Q2L1_0_3[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 70 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %100 : !air.async.token
                }
                affine.yield %99 : !air.async.token
              }
              affine.yield %98 : !air.async.token
            }
          } else {
            %97 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %98 = air.channel.get async  @Q2L1_1_0[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 71 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %98 : !air.async.token
            } else {
              %98 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %99 = air.channel.get async  @Q2L1_1_1[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 72 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %99 : !air.async.token
              } else {
                %99 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %100 = air.channel.get async  @Q2L1_1_2[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 73 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %100 : !air.async.token
                } else {
                  %100 = air.channel.get async  @Q2L1_1_3[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 74 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %100 : !air.async.token
                }
                affine.yield %99 : !air.async.token
              }
              affine.yield %98 : !air.async.token
            }
          }
          %90 = arith.cmpi eq, %86, %c3_i32 : i32
          scf.if %90 {
            %async_token_55 = air.execute {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %91 = air.wait_all async [%async_token_47, %async_token_48, %async_token_49] 
          %92 = scf.for %arg29 = %c0_46 to %c8_43 step %c1_45 iter_args(%arg30 = %91) -> (!air.async.token) {
            %async_token_55 = air.execute [%arg30] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            }
            scf.if %85 {
              %100 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
                %101 = air.channel.get async [%arg30]  @K2L1_0_0[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 75 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %101 : !air.async.token
              } else {
                %101 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                  %102 = air.channel.get async [%arg30]  @K2L1_0_1[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 76 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %102 : !air.async.token
                } else {
                  %102 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                    %103 = air.channel.get async [%arg30]  @K2L1_0_2[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 77 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %103 : !air.async.token
                  } else {
                    %103 = air.channel.get async [%arg30]  @K2L1_0_3[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 78 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %103 : !air.async.token
                  }
                  affine.yield %102 : !air.async.token
                }
                affine.yield %101 : !air.async.token
              }
            } else {
              %100 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
                %101 = air.channel.get async [%arg30]  @K2L1_1_0[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 79 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %101 : !air.async.token
              } else {
                %101 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                  %102 = air.channel.get async [%arg30]  @K2L1_1_1[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 80 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %102 : !air.async.token
                } else {
                  %102 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                    %103 = air.channel.get async [%arg30]  @K2L1_1_2[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 81 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %103 : !air.async.token
                  } else {
                    %103 = air.channel.get async [%arg30]  @K2L1_1_3[%c0_46, %c0_46, %arg16] (%arg21[] [] []) {id = 82 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %103 : !air.async.token
                  }
                  affine.yield %102 : !air.async.token
                }
                affine.yield %101 : !air.async.token
              }
            }
            %async_token_56 = air.execute [%async_token_55] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_a_b_bf16(%arg20, %arg21, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            }
            scf.if %85 {
              %100 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
                %101 = air.channel.get async [%arg30]  @V2L1_0_0[%c0_46, %c0_46, %arg16] (%arg22[] [] []) {id = 83 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %101 : !air.async.token
              } else {
                %101 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                  %102 = air.channel.get async [%arg30]  @V2L1_0_1[%c0_46, %c0_46, %arg16] (%arg22[] [] []) {id = 84 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %102 : !air.async.token
                } else {
                  %102 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                    %103 = air.channel.get async [%arg30]  @V2L1_0_2[%c0_46, %c0_46, %arg16] (%arg22[] [] []) {id = 85 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %103 : !air.async.token
                  } else {
                    %103 = air.channel.get async [%arg30]  @V2L1_0_3[%c0_46, %c0_46, %arg16] (%arg22[] [] []) {id = 86 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %103 : !air.async.token
                  }
                  affine.yield %102 : !air.async.token
                }
                affine.yield %101 : !air.async.token
              }
            } else {
              %100 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
                %101 = air.channel.get async [%arg30]  @V2L1_1_0[%c0_46, %c0_46, %arg16] (%arg22[] [] []) {id = 87 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %101 : !air.async.token
              } else {
                %101 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                  %102 = air.channel.get async [%arg30]  @V2L1_1_1[%c0_46, %c0_46, %arg16] (%arg22[] [] []) {id = 88 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %102 : !air.async.token
                } else {
                  %102 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                    %103 = air.channel.get async [%arg30]  @V2L1_1_2[%c0_46, %c0_46, %arg16] (%arg22[] [] []) {id = 89 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %103 : !air.async.token
                  } else {
                    %103 = air.channel.get async [%arg30]  @V2L1_1_3[%c0_46, %c0_46, %arg16] (%arg22[] [] []) {id = 90 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %103 : !air.async.token
                  }
                  affine.yield %102 : !air.async.token
                }
                affine.yield %101 : !air.async.token
              }
            }
            %97 = arith.index_cast %arg29 : index to i32
            %async_token_57, %results_58 = air.execute [%arg30] -> (i32) {
              %100 = memref.load %arg28[%c0_46] : memref<3xi32, 2 : i32>
              air.execute_terminator %100 : i32
            }
            %98 = arith.addi %results_58, %86 : i32
            %async_token_59 = air.execute [%async_token_56] {
              func.call @apply_causal_mask(%arg23, %98, %97) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
            }
            %async_token_60, %results_61 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_62, %results_63 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_64 = air.execute [%async_token_62, %async_token_60, %async_token_59] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @fused_softmax(%collapse_shape, %arg25, %results_61, %results_63) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_65 = air.execute [%async_token_64] {
              func.call @mul_r_gp(%results_63, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_66 = air.execute [%async_token_65] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_g_b_bf16(%collapse_shape, %arg22, %arg24) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_67 = air.execute [%async_token_65] {
              func.call @accum_sp_r_s(%arg26, %results_63, %results_61) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_68 = air.execute [%async_token_67] {
              func.call @vector_copy_32elems(%c0_i32, %results_61, %arg26) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_69 = air.execute [%async_token_68] {
              memref.dealloc %results_61 : memref<64x1xbf16, 2 : i32>
            }
            %async_token_70 = air.execute [%async_token_67] {
              memref.dealloc %results_63 : memref<64x1xbf16, 2 : i32>
            }
            %99 = air.wait_all async [%async_token_57, %async_token_66, %async_token_68] 
            scf.yield %99 : !air.async.token
          }
          %async_token_52 = air.execute [%92] {
            func.call @div_gp_sp(%arg26, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %93 = air.channel.put async [%async_token_52]  @Gp2L2[%arg17, %arg16] (%arg24[%c0_46, %c0_46, %c0_46, %c0_46] [%c8_43, %c8_43, %c8_43, %c8_43] [%c64_42, %c8_43, %c512_41, %c1_45]) {id = 91 : i32} : (memref<64x64xbf16, 2 : i32>)
          %async_token_53, %results_54 = air.execute [%92] -> (i32) {
            %97 = memref.load %arg28[%c2_44] : memref<3xi32, 2 : i32>
            air.execute_terminator %97 : i32
          }
          %94 = arith.addi %results_54, %c1_i32 : i32
          %95 = arith.cmpi sge, %94, %c1_i32 : i32
          scf.if %95 {
            %async_token_55, %results_56 = air.execute [%async_token_53] -> (i32) {
              %98 = memref.load %arg28[%c0_46] : memref<3xi32, 2 : i32>
              air.execute_terminator %98 : i32
            }
            %97 = arith.addi %results_56, %c4_i32 : i32
            %async_token_57 = air.execute [%async_token_55] {
              memref.store %97, %arg28[%c0_46] : memref<3xi32, 2 : i32>
            }
            %async_token_58 = air.execute [%async_token_57] {
              memref.store %c0_i32, %arg28[%c2_44] : memref<3xi32, 2 : i32>
            }
          }
          %96 = arith.cmpi slt, %94, %c1_i32 : i32
          scf.if %96 {
            %async_token_55 = air.execute {
              memref.store %94, %arg28[%c2_44] : memref<3xi32, 2 : i32>
            }
          }
        }
        %40 = air.channel.get async [%async_token_12]  @Gp2L2[%c0_8, %c0_8] (%results_13[%c0_8, %c0_8] [%c64_4, %c64_4] [%c64_4, %c1_6]) {id = 92 : i32} : (memref<256x64xbf16, 1 : i32>)
        %41 = air.wait_all async [%40] 
        %42 = air.channel.get async [%async_token_12]  @Gp2L2[%c0_8, %c1_6] (%results_13[%c64_4, %c0_8] [%c64_4, %c64_4] [%c64_4, %c1_6]) {id = 93 : i32} : (memref<256x64xbf16, 1 : i32>)
        %43 = air.wait_all async [%42] 
        %44 = air.channel.get async [%async_token_12]  @Gp2L2[%c0_8, %c2_3] (%results_13[%c128_2, %c0_8] [%c64_4, %c64_4] [%c64_4, %c1_6]) {id = 94 : i32} : (memref<256x64xbf16, 1 : i32>)
        %45 = air.wait_all async [%44] 
        %46 = air.channel.get async [%async_token_12]  @Gp2L2[%c0_8, %c3] (%results_13[%c192, %c0_8] [%c64_4, %c64_4] [%c64_4, %c1_6]) {id = 95 : i32} : (memref<256x64xbf16, 1 : i32>)
        %47 = air.wait_all async [%46] 
        %48 = air.wait_all async [%41, %43, %45, %47] 
        %49 = air.wait_all async 
        %50 = air.channel.put async [%48]  @GpOut[%arg12] (%results_13[] [] []) {id = 96 : i32} : (memref<256x64xbf16, 1 : i32>)
        %51 = air.channel.get async [%50]  @Gp2L2[%c1_6, %c0_8] (%results_13[%c0_8, %c0_8] [%c64_4, %c64_4] [%c64_4, %c1_6]) {id = 97 : i32} : (memref<256x64xbf16, 1 : i32>)
        %52 = air.wait_all async [%51] 
        %53 = air.channel.get async [%50]  @Gp2L2[%c1_6, %c1_6] (%results_13[%c64_4, %c0_8] [%c64_4, %c64_4] [%c64_4, %c1_6]) {id = 98 : i32} : (memref<256x64xbf16, 1 : i32>)
        %54 = air.wait_all async [%53] 
        %55 = air.channel.get async [%50]  @Gp2L2[%c1_6, %c2_3] (%results_13[%c128_2, %c0_8] [%c64_4, %c64_4] [%c64_4, %c1_6]) {id = 99 : i32} : (memref<256x64xbf16, 1 : i32>)
        %56 = air.wait_all async [%55] 
        %57 = air.channel.get async [%50]  @Gp2L2[%c1_6, %c3] (%results_13[%c192, %c0_8] [%c64_4, %c64_4] [%c64_4, %c1_6]) {id = 100 : i32} : (memref<256x64xbf16, 1 : i32>)
        %58 = air.wait_all async [%57] 
        %59 = air.wait_all async [%52, %54, %56, %58] 
        %60 = air.wait_all async 
        %61 = air.channel.put async [%59]  @GpOut[%arg12] (%results_13[] [] []) {id = 101 : i32} : (memref<256x64xbf16, 1 : i32>)
        %62 = air.channel.get async [%61]  @Gp2L2[%c2_3, %c0_8] (%results_13[%c0_8, %c0_8] [%c64_4, %c64_4] [%c64_4, %c1_6]) {id = 102 : i32} : (memref<256x64xbf16, 1 : i32>)
        %63 = air.wait_all async [%62] 
        %64 = air.channel.get async [%61]  @Gp2L2[%c2_3, %c1_6] (%results_13[%c64_4, %c0_8] [%c64_4, %c64_4] [%c64_4, %c1_6]) {id = 103 : i32} : (memref<256x64xbf16, 1 : i32>)
        %65 = air.wait_all async [%64] 
        %66 = air.channel.get async [%61]  @Gp2L2[%c2_3, %c2_3] (%results_13[%c128_2, %c0_8] [%c64_4, %c64_4] [%c64_4, %c1_6]) {id = 104 : i32} : (memref<256x64xbf16, 1 : i32>)
        %67 = air.wait_all async [%66] 
        %68 = air.channel.get async [%61]  @Gp2L2[%c2_3, %c3] (%results_13[%c192, %c0_8] [%c64_4, %c64_4] [%c64_4, %c1_6]) {id = 105 : i32} : (memref<256x64xbf16, 1 : i32>)
        %69 = air.wait_all async [%68] 
        %70 = air.wait_all async [%63, %65, %67, %69] 
        %71 = air.wait_all async 
        %72 = air.channel.put async [%70]  @GpOut[%arg12] (%results_13[] [] []) {id = 106 : i32} : (memref<256x64xbf16, 1 : i32>)
        %73 = air.channel.get async [%72]  @Gp2L2[%c3, %c0_8] (%results_13[%c0_8, %c0_8] [%c64_4, %c64_4] [%c64_4, %c1_6]) {id = 107 : i32} : (memref<256x64xbf16, 1 : i32>)
        %74 = air.wait_all async [%73] 
        %75 = air.channel.get async [%72]  @Gp2L2[%c3, %c1_6] (%results_13[%c64_4, %c0_8] [%c64_4, %c64_4] [%c64_4, %c1_6]) {id = 108 : i32} : (memref<256x64xbf16, 1 : i32>)
        %76 = air.wait_all async [%75] 
        %77 = air.channel.get async [%72]  @Gp2L2[%c3, %c2_3] (%results_13[%c128_2, %c0_8] [%c64_4, %c64_4] [%c64_4, %c1_6]) {id = 109 : i32} : (memref<256x64xbf16, 1 : i32>)
        %78 = air.wait_all async [%77] 
        %79 = air.channel.get async [%72]  @Gp2L2[%c3, %c3] (%results_13[%c192, %c0_8] [%c64_4, %c64_4] [%c64_4, %c1_6]) {id = 110 : i32} : (memref<256x64xbf16, 1 : i32>)
        %80 = air.wait_all async [%79] 
        %81 = air.wait_all async [%74, %76, %78, %80] 
        %82 = air.wait_all async 
        %83 = air.channel.put async [%81]  @GpOut[%arg12] (%results_13[] [] []) {id = 111 : i32} : (memref<256x64xbf16, 1 : i32>)
        %async_token_30 = air.execute [%39] {
          memref.dealloc %results_15 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_31 = air.execute [%39] {
          memref.dealloc %results_17 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_32 = air.execute [%39] {
          memref.dealloc %results_19 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_33 = air.execute [%39] {
          memref.dealloc %results_21 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_34 = air.execute [%39] {
          memref.dealloc %results_23 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_35 = air.execute [%39] {
          memref.dealloc %results_25 : memref<64x1xbf16, 2 : i32>
        }
        %async_token_36 = air.execute [%39] {
          memref.dealloc %results_27 : memref<64x1xbf16, 2 : i32>
        }
        %async_token_37 = air.execute [%38] {
          memref.dealloc %results_11 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_38 = air.execute [%37] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        }
        %async_token_39 = air.execute [%83] {
          memref.dealloc %results_13 : memref<256x64xbf16, 1 : i32>
        }
        %async_token_40 = air.execute [%39] {
          memref.dealloc %results_29 : memref<3xi32, 2 : i32>
        }
      }
      %25 = air.channel.get async [%24]  @GpOut[%c0] (%arg11[%1, %5] [%c256, %c64] [%c512, %c1_1]) {id = 112 : i32} : (memref<512x512xbf16>)
      %26 = air.channel.get async [%24]  @GpOut[%c0] (%arg11[%1, %7] [%c256, %c64] [%c512, %c1_1]) {id = 113 : i32} : (memref<512x512xbf16>)
      %27 = air.channel.get async [%24]  @GpOut[%c0] (%arg11[%1, %9] [%c256, %c64] [%c512, %c1_1]) {id = 114 : i32} : (memref<512x512xbf16>)
      %28 = air.channel.get async [%24]  @GpOut[%c0] (%arg11[%1, %11] [%c256, %c64] [%c512, %c1_1]) {id = 115 : i32} : (memref<512x512xbf16>)
      %29 = air.channel.get async [%24]  @GpOut[%c1_1] (%arg11[%1, %16] [%c256, %c64] [%c512, %c1_1]) {id = 116 : i32} : (memref<512x512xbf16>)
      %30 = air.channel.get async [%24]  @GpOut[%c1_1] (%arg11[%1, %18] [%c256, %c64] [%c512, %c1_1]) {id = 117 : i32} : (memref<512x512xbf16>)
      %31 = air.channel.get async [%24]  @GpOut[%c1_1] (%arg11[%1, %20] [%c256, %c64] [%c512, %c1_1]) {id = 118 : i32} : (memref<512x512xbf16>)
      %32 = air.channel.get async [%24]  @GpOut[%c1_1] (%arg11[%1, %22] [%c256, %c64] [%c512, %c1_1]) {id = 119 : i32} : (memref<512x512xbf16>)
    }
    return
  }
}
