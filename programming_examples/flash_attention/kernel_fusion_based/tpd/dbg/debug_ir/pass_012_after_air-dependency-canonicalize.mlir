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
#map11 = affine_map<()[s0] -> (s0 * 64)>
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
        %c3 = arith.constant 3 : index
        %c2_2 = arith.constant 2 : index
        %c64_3 = arith.constant 64 : index
        %c512_4 = arith.constant 512 : index
        %c1_5 = arith.constant 1 : index
        %c8_6 = arith.constant 8 : index
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
        %async_token_11, %results_12 = air.execute -> (memref<256x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<256x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<256x64xbf16, 1 : i32>
        }
        %async_token_13, %results_14 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
        }
        %async_token_15, %results_16 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
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
        %async_token_23, %results_24 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        }
        %async_token_25, %results_26 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
        }
        %async_token_27, %results_28 = air.execute -> (memref<3xi32, 2 : i32>) {
          %alloc = memref.alloc() : memref<3xi32, 2 : i32>
          air.execute_terminator %alloc : memref<3xi32, 2 : i32>
        }
        %33 = scf.for %arg16 = %c0_7 to %c4_8 step %c1_5 iter_args(%arg17 = %async_token) -> (!air.async.token) {
          %48 = air.channel.get async [%arg17]  @QIn[%arg12] (%results[] [] []) {id = 13 : i32} : (memref<64x64xbf16, 1 : i32>)
          %49 = arith.cmpi eq, %arg12, %c0_7 : index
          %50 = scf.if %49 -> (!air.async.token) {
            %51 = air.channel.put async [%48]  @Q2L1_0_0[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 14 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %51 : !air.async.token
          } else {
            %51 = air.channel.put async [%48]  @Q2L1_1_0[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 15 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %51 : !air.async.token
          }
          scf.yield %50 : !air.async.token
        }
        %34 = scf.for %arg16 = %c0_7 to %c4_8 step %c1_5 iter_args(%arg17 = %33) -> (!air.async.token) {
          %48 = air.channel.get async [%arg17]  @QIn[%arg12] (%results[] [] []) {id = 16 : i32} : (memref<64x64xbf16, 1 : i32>)
          %49 = arith.cmpi eq, %arg12, %c0_7 : index
          %50 = scf.if %49 -> (!air.async.token) {
            %51 = air.channel.put async [%48]  @Q2L1_0_1[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 17 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %51 : !air.async.token
          } else {
            %51 = air.channel.put async [%48]  @Q2L1_1_1[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 18 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %51 : !air.async.token
          }
          scf.yield %50 : !air.async.token
        }
        %35 = scf.for %arg16 = %c0_7 to %c4_8 step %c1_5 iter_args(%arg17 = %34) -> (!air.async.token) {
          %48 = air.channel.get async [%arg17]  @QIn[%arg12] (%results[] [] []) {id = 19 : i32} : (memref<64x64xbf16, 1 : i32>)
          %49 = arith.cmpi eq, %arg12, %c0_7 : index
          %50 = scf.if %49 -> (!air.async.token) {
            %51 = air.channel.put async [%48]  @Q2L1_0_2[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 20 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %51 : !air.async.token
          } else {
            %51 = air.channel.put async [%48]  @Q2L1_1_2[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 21 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %51 : !air.async.token
          }
          scf.yield %50 : !air.async.token
        }
        %36 = scf.for %arg16 = %c0_7 to %c4_8 step %c1_5 iter_args(%arg17 = %35) -> (!air.async.token) {
          %48 = air.channel.get async [%arg17]  @QIn[%arg12] (%results[] [] []) {id = 22 : i32} : (memref<64x64xbf16, 1 : i32>)
          %49 = arith.cmpi eq, %arg12, %c0_7 : index
          %50 = scf.if %49 -> (!air.async.token) {
            %51 = air.channel.put async [%48]  @Q2L1_0_3[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 23 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %51 : !air.async.token
          } else {
            %51 = air.channel.put async [%48]  @Q2L1_1_3[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 24 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %51 : !air.async.token
          }
          scf.yield %50 : !air.async.token
        }
        %37 = scf.for %arg16 = %c0_7 to %c8_6 step %c1_5 iter_args(%arg17 = %36) -> (!air.async.token) {
          %48 = air.channel.get async [%arg17]  @KIn[%arg12] (%results[] [] []) {id = 25 : i32} : (memref<64x64xbf16, 1 : i32>)
          %49 = arith.cmpi eq, %arg12, %c0_7 : index
          %50 = scf.if %49 -> (!air.async.token) {
            %55 = air.channel.put async [%48]  @K2L1_0_0[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 26 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %55 : !air.async.token
          } else {
            %55 = air.channel.put async [%48]  @K2L1_1_0[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 27 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %55 : !air.async.token
          }
          %51 = scf.if %49 -> (!air.async.token) {
            %55 = air.channel.put async [%48]  @K2L1_0_1[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 28 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %55 : !air.async.token
          } else {
            %55 = air.channel.put async [%48]  @K2L1_1_1[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 29 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %55 : !air.async.token
          }
          %52 = scf.if %49 -> (!air.async.token) {
            %55 = air.channel.put async [%48]  @K2L1_0_2[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 30 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %55 : !air.async.token
          } else {
            %55 = air.channel.put async [%48]  @K2L1_1_2[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 31 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %55 : !air.async.token
          }
          %53 = scf.if %49 -> (!air.async.token) {
            %55 = air.channel.put async [%48]  @K2L1_0_3[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 32 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %55 : !air.async.token
          } else {
            %55 = air.channel.put async [%48]  @K2L1_1_3[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 33 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %55 : !air.async.token
          }
          %54 = air.wait_all async [%50, %50, %51, %51, %52, %52, %53, %53] 
          scf.yield %54 : !air.async.token
        }
        %38 = scf.for %arg16 = %c0_7 to %c8_6 step %c1_5 iter_args(%arg17 = %async_token_9) -> (!air.async.token) {
          %48 = air.channel.get async [%arg17]  @VIn[%arg12] (%results_10[] [] []) {id = 34 : i32} : (memref<64x64xbf16, 1 : i32>)
          %49 = arith.cmpi eq, %arg12, %c0_7 : index
          %50 = scf.if %49 -> (!air.async.token) {
            %55 = air.channel.put async [%48]  @V2L1_0_0[%c0_7, %c0_7, %c0_7] (%results_10[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 35 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %55 : !air.async.token
          } else {
            %55 = air.channel.put async [%48]  @V2L1_1_0[%c0_7, %c0_7, %c0_7] (%results_10[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 36 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %55 : !air.async.token
          }
          %51 = scf.if %49 -> (!air.async.token) {
            %55 = air.channel.put async [%48]  @V2L1_0_1[%c0_7, %c0_7, %c0_7] (%results_10[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 37 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %55 : !air.async.token
          } else {
            %55 = air.channel.put async [%48]  @V2L1_1_1[%c0_7, %c0_7, %c0_7] (%results_10[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 38 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %55 : !air.async.token
          }
          %52 = scf.if %49 -> (!air.async.token) {
            %55 = air.channel.put async [%48]  @V2L1_0_2[%c0_7, %c0_7, %c0_7] (%results_10[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 39 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %55 : !air.async.token
          } else {
            %55 = air.channel.put async [%48]  @V2L1_1_2[%c0_7, %c0_7, %c0_7] (%results_10[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 40 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %55 : !air.async.token
          }
          %53 = scf.if %49 -> (!air.async.token) {
            %55 = air.channel.put async [%48]  @V2L1_0_3[%c0_7, %c0_7, %c0_7] (%results_10[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 41 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %55 : !air.async.token
          } else {
            %55 = air.channel.put async [%48]  @V2L1_1_3[%c0_7, %c0_7, %c0_7] (%results_10[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) {id = 42 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %55 : !air.async.token
          }
          %54 = air.wait_all async [%50, %50, %51, %51, %52, %52, %53, %53] 
          scf.yield %54 : !air.async.token
        }
        %39 = air.herd @herd_0 async [%async_token_13, %async_token_15, %async_token_17, %async_token_19, %async_token_21, %async_token_23, %async_token_25, %async_token_27]  tile (%arg16, %arg17) in (%arg18=%c4_8, %arg19=%c4_8) args(%arg20=%results_14, %arg21=%results_16, %arg22=%results_18, %arg23=%results_20, %arg24=%results_22, %arg25=%results_24, %arg26=%results_26, %arg27=%arg12, %arg28=%results_28) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, index, memref<3xi32, 2 : i32> attributes {id = 3 : i32, link_with = "attn_npu2.o"} {
          %c4_i32 = arith.constant 4 : i32
          %c512_40 = arith.constant 512 : index
          %c64_41 = arith.constant 64 : index
          %c8_42 = arith.constant 8 : index
          %c3_i32 = arith.constant 3 : i32
          %c2_i32 = arith.constant 2 : i32
          %c1_i32 = arith.constant 1 : i32
          %c0_i32 = arith.constant 0 : i32
          %c2_43 = arith.constant 2 : index
          %c1_44 = arith.constant 1 : index
          %c0_45 = arith.constant 0 : index
          %async_token_46 = air.execute {
            func.call @zero_fill_gp_bf16(%arg24) : (memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_47 = air.execute {
            func.call @zero_fill_sp_bf16(%arg26) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_48 = air.execute {
            func.call @neg_inf_fill_up_bf16(%arg25) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_49, %results_50 = air.execute -> (i32) {
            %65 = memref.load %arg28[%c1_44] : memref<3xi32, 2 : i32>
            air.execute_terminator %65 : i32
          }
          %48 = arith.cmpi eq, %results_50, %c0_i32 : i32
          scf.if %48 {
            %async_token_54 = air.execute [%async_token_49] {
              memref.store %c0_i32, %arg28[%c0_45] : memref<3xi32, 2 : i32>
            }
            %async_token_55 = air.execute [%async_token_54] {
              memref.store %c1_i32, %arg28[%c1_44] : memref<3xi32, 2 : i32>
            }
            %async_token_56 = air.execute [%async_token_55] {
              memref.store %c0_i32, %arg28[%c2_43] : memref<3xi32, 2 : i32>
            }
          }
          %49 = arith.cmpi eq, %arg27, %c0_45 : index
          %50 = scf.if %49 -> (!air.async.token) {
            %65 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %66 = air.channel.get async  @Q2L1_0_0[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 43 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %66 : !air.async.token
            } else {
              %66 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %67 = air.channel.get async  @Q2L1_0_1[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 44 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              } else {
                %67 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %68 = air.channel.get async  @Q2L1_0_2[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 45 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %68 : !air.async.token
                } else {
                  %68 = air.channel.get async  @Q2L1_0_3[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 46 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %68 : !air.async.token
                }
                affine.yield %67 : !air.async.token
              }
              affine.yield %66 : !air.async.token
            }
            scf.yield %65 : !air.async.token
          } else {
            %65 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %66 = air.channel.get async  @Q2L1_1_0[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 47 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %66 : !air.async.token
            } else {
              %66 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %67 = air.channel.get async  @Q2L1_1_1[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 48 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              } else {
                %67 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %68 = air.channel.get async  @Q2L1_1_2[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 49 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %68 : !air.async.token
                } else {
                  %68 = air.channel.get async  @Q2L1_1_3[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 50 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %68 : !air.async.token
                }
                affine.yield %67 : !air.async.token
              }
              affine.yield %66 : !air.async.token
            }
            scf.yield %65 : !air.async.token
          }
          %51 = arith.index_cast %arg16 : index to i32
          %52 = arith.cmpi eq, %51, %c0_i32 : i32
          scf.if %52 {
            %async_token_54 = air.execute {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %53 = scf.if %49 -> (!air.async.token) {
            %65 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %66 = air.channel.get async  @Q2L1_0_0[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 51 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %66 : !air.async.token
            } else {
              %66 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %67 = air.channel.get async  @Q2L1_0_1[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 52 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              } else {
                %67 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %68 = air.channel.get async  @Q2L1_0_2[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 53 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %68 : !air.async.token
                } else {
                  %68 = air.channel.get async  @Q2L1_0_3[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 54 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %68 : !air.async.token
                }
                affine.yield %67 : !air.async.token
              }
              affine.yield %66 : !air.async.token
            }
            scf.yield %65 : !air.async.token
          } else {
            %65 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %66 = air.channel.get async  @Q2L1_1_0[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %66 : !air.async.token
            } else {
              %66 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %67 = air.channel.get async  @Q2L1_1_1[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 56 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              } else {
                %67 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %68 = air.channel.get async  @Q2L1_1_2[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %68 : !air.async.token
                } else {
                  %68 = air.channel.get async  @Q2L1_1_3[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 58 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %68 : !air.async.token
                }
                affine.yield %67 : !air.async.token
              }
              affine.yield %66 : !air.async.token
            }
            scf.yield %65 : !air.async.token
          }
          %54 = arith.cmpi eq, %51, %c1_i32 : i32
          scf.if %54 {
            %async_token_54 = air.execute {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %55 = scf.if %49 -> (!air.async.token) {
            %65 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %66 = air.channel.get async  @Q2L1_0_0[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 59 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %66 : !air.async.token
            } else {
              %66 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %67 = air.channel.get async  @Q2L1_0_1[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 60 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              } else {
                %67 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %68 = air.channel.get async  @Q2L1_0_2[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 61 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %68 : !air.async.token
                } else {
                  %68 = air.channel.get async  @Q2L1_0_3[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 62 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %68 : !air.async.token
                }
                affine.yield %67 : !air.async.token
              }
              affine.yield %66 : !air.async.token
            }
            scf.yield %65 : !air.async.token
          } else {
            %65 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %66 = air.channel.get async  @Q2L1_1_0[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %66 : !air.async.token
            } else {
              %66 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %67 = air.channel.get async  @Q2L1_1_1[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 64 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              } else {
                %67 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %68 = air.channel.get async  @Q2L1_1_2[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 65 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %68 : !air.async.token
                } else {
                  %68 = air.channel.get async  @Q2L1_1_3[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %68 : !air.async.token
                }
                affine.yield %67 : !air.async.token
              }
              affine.yield %66 : !air.async.token
            }
            scf.yield %65 : !air.async.token
          }
          %56 = arith.cmpi eq, %51, %c2_i32 : i32
          scf.if %56 {
            %async_token_54 = air.execute {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %57 = scf.if %49 -> (!air.async.token) {
            %65 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %66 = air.channel.get async  @Q2L1_0_0[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 67 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %66 : !air.async.token
            } else {
              %66 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %67 = air.channel.get async  @Q2L1_0_1[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 68 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              } else {
                %67 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %68 = air.channel.get async  @Q2L1_0_2[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 69 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %68 : !air.async.token
                } else {
                  %68 = air.channel.get async  @Q2L1_0_3[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 70 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %68 : !air.async.token
                }
                affine.yield %67 : !air.async.token
              }
              affine.yield %66 : !air.async.token
            }
            scf.yield %65 : !air.async.token
          } else {
            %65 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %66 = air.channel.get async  @Q2L1_1_0[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 71 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %66 : !air.async.token
            } else {
              %66 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %67 = air.channel.get async  @Q2L1_1_1[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 72 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %67 : !air.async.token
              } else {
                %67 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %68 = air.channel.get async  @Q2L1_1_2[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 73 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %68 : !air.async.token
                } else {
                  %68 = air.channel.get async  @Q2L1_1_3[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 74 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %68 : !air.async.token
                }
                affine.yield %67 : !air.async.token
              }
              affine.yield %66 : !air.async.token
            }
            scf.yield %65 : !air.async.token
          }
          %58 = arith.cmpi eq, %51, %c3_i32 : i32
          scf.if %58 {
            %async_token_54 = air.execute {
              func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %59 = air.wait_all async [%async_token_46, %async_token_47, %async_token_48] 
          %60 = scf.for %arg29 = %c0_45 to %c8_42 step %c1_44 iter_args(%arg30 = %59) -> (!air.async.token) {
            %async_token_54 = air.execute [%arg30] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            }
            %65 = scf.if %49 -> (!air.async.token) {
              %70 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
                %71 = air.channel.get async [%arg30]  @K2L1_0_0[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 75 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %71 : !air.async.token
              } else {
                %71 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                  %72 = air.channel.get async [%arg30]  @K2L1_0_1[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 76 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %72 : !air.async.token
                } else {
                  %72 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                    %73 = air.channel.get async [%arg30]  @K2L1_0_2[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 77 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %73 : !air.async.token
                  } else {
                    %73 = air.channel.get async [%arg30]  @K2L1_0_3[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 78 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %73 : !air.async.token
                  }
                  affine.yield %72 : !air.async.token
                }
                affine.yield %71 : !air.async.token
              }
              scf.yield %70 : !air.async.token
            } else {
              %70 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
                %71 = air.channel.get async [%arg30]  @K2L1_1_0[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 79 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %71 : !air.async.token
              } else {
                %71 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                  %72 = air.channel.get async [%arg30]  @K2L1_1_1[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 80 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %72 : !air.async.token
                } else {
                  %72 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                    %73 = air.channel.get async [%arg30]  @K2L1_1_2[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 81 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %73 : !air.async.token
                  } else {
                    %73 = air.channel.get async [%arg30]  @K2L1_1_3[%c0_45, %c0_45, %arg16] (%arg21[] [] []) {id = 82 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %73 : !air.async.token
                  }
                  affine.yield %72 : !air.async.token
                }
                affine.yield %71 : !air.async.token
              }
              scf.yield %70 : !air.async.token
            }
            %async_token_55 = air.execute [%async_token_54] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_a_b_bf16(%arg20, %arg21, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            }
            %66 = scf.if %49 -> (!air.async.token) {
              %70 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
                %71 = air.channel.get async [%arg30]  @V2L1_0_0[%c0_45, %c0_45, %arg16] (%arg22[] [] []) {id = 83 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %71 : !air.async.token
              } else {
                %71 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                  %72 = air.channel.get async [%arg30]  @V2L1_0_1[%c0_45, %c0_45, %arg16] (%arg22[] [] []) {id = 84 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %72 : !air.async.token
                } else {
                  %72 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                    %73 = air.channel.get async [%arg30]  @V2L1_0_2[%c0_45, %c0_45, %arg16] (%arg22[] [] []) {id = 85 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %73 : !air.async.token
                  } else {
                    %73 = air.channel.get async [%arg30]  @V2L1_0_3[%c0_45, %c0_45, %arg16] (%arg22[] [] []) {id = 86 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %73 : !air.async.token
                  }
                  affine.yield %72 : !air.async.token
                }
                affine.yield %71 : !air.async.token
              }
              scf.yield %70 : !air.async.token
            } else {
              %70 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
                %71 = air.channel.get async [%arg30]  @V2L1_1_0[%c0_45, %c0_45, %arg16] (%arg22[] [] []) {id = 87 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %71 : !air.async.token
              } else {
                %71 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                  %72 = air.channel.get async [%arg30]  @V2L1_1_1[%c0_45, %c0_45, %arg16] (%arg22[] [] []) {id = 88 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %72 : !air.async.token
                } else {
                  %72 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                    %73 = air.channel.get async [%arg30]  @V2L1_1_2[%c0_45, %c0_45, %arg16] (%arg22[] [] []) {id = 89 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %73 : !air.async.token
                  } else {
                    %73 = air.channel.get async [%arg30]  @V2L1_1_3[%c0_45, %c0_45, %arg16] (%arg22[] [] []) {id = 90 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %73 : !air.async.token
                  }
                  affine.yield %72 : !air.async.token
                }
                affine.yield %71 : !air.async.token
              }
              scf.yield %70 : !air.async.token
            }
            %67 = arith.index_cast %arg29 : index to i32
            %async_token_56, %results_57 = air.execute [%arg30] -> (i32) {
              %70 = memref.load %arg28[%c0_45] : memref<3xi32, 2 : i32>
              air.execute_terminator %70 : i32
            }
            %68 = arith.addi %results_57, %51 : i32
            %async_token_58 = air.execute [%async_token_55] {
              func.call @apply_causal_mask(%arg23, %68, %67) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
            }
            %async_token_59, %results_60 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_61, %results_62 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_63 = air.execute [%async_token_61, %async_token_59, %async_token_58] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @fused_softmax(%collapse_shape, %arg25, %results_60, %results_62) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_64 = air.execute [%async_token_63] {
              func.call @mul_r_gp(%results_62, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_65 = air.execute [%async_token_64] {
              %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_g_b_bf16(%collapse_shape, %arg22, %arg24) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_66 = air.execute [%async_token_64] {
              func.call @accum_sp_r_s(%arg26, %results_62, %results_60) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_67 = air.execute [%async_token_66] {
              func.call @vector_copy_32elems(%c0_i32, %results_60, %arg26) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_68 = air.execute [%async_token_67] {
              memref.dealloc %results_60 : memref<64x1xbf16, 2 : i32>
            }
            %async_token_69 = air.execute [%async_token_66] {
              memref.dealloc %results_62 : memref<64x1xbf16, 2 : i32>
            }
            %69 = air.wait_all async [%async_token_56, %async_token_65, %async_token_67] 
            scf.yield %69 : !air.async.token
          }
          %async_token_51 = air.execute [%60] {
            func.call @div_gp_sp(%arg26, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %61 = air.channel.put async [%async_token_51]  @Gp2L2[%arg17, %arg16] (%arg24[%c0_45, %c0_45, %c0_45, %c0_45] [%c8_42, %c8_42, %c8_42, %c8_42] [%c64_41, %c8_42, %c512_40, %c1_44]) {id = 91 : i32} : (memref<64x64xbf16, 2 : i32>)
          %async_token_52, %results_53 = air.execute [%60] -> (i32) {
            %65 = memref.load %arg28[%c2_43] : memref<3xi32, 2 : i32>
            air.execute_terminator %65 : i32
          }
          %62 = arith.addi %results_53, %c1_i32 : i32
          %63 = arith.cmpi sge, %62, %c1_i32 : i32
          scf.if %63 {
            %async_token_54, %results_55 = air.execute [%async_token_52] -> (i32) {
              %66 = memref.load %arg28[%c0_45] : memref<3xi32, 2 : i32>
              air.execute_terminator %66 : i32
            }
            %65 = arith.addi %results_55, %c4_i32 : i32
            %async_token_56 = air.execute [%async_token_54] {
              memref.store %65, %arg28[%c0_45] : memref<3xi32, 2 : i32>
            }
            %async_token_57 = air.execute [%async_token_56] {
              memref.store %c0_i32, %arg28[%c2_43] : memref<3xi32, 2 : i32>
            }
          }
          %64 = arith.cmpi slt, %62, %c1_i32 : i32
          scf.if %64 {
            %async_token_54 = air.execute {
              memref.store %62, %arg28[%c2_43] : memref<3xi32, 2 : i32>
            }
          }
        }
        %40 = scf.parallel (%arg16) = (%c0_7) to (%c4_8) step (%c1_5) init (%async_token_11) -> !air.async.token {
          %48 = affine.apply #map11()[%arg16]
          %49 = air.channel.get async [%async_token_11]  @Gp2L2[%c0_7, %arg16] (%results_12[%48, %c0_7] [%c64_3, %c64_3] [%c64_3, %c1_5]) {id = 92 : i32} : (memref<256x64xbf16, 1 : i32>)
          scf.reduce(%49 : !air.async.token) {
          ^bb0(%arg17: !air.async.token, %arg18: !air.async.token):
            %50 = air.wait_all async [%arg17, %arg18] 
            scf.reduce.return %50 : !air.async.token
          }
        }
        %41 = air.channel.put async [%40]  @GpOut[%arg12] (%results_12[] [] []) {id = 93 : i32} : (memref<256x64xbf16, 1 : i32>)
        %42 = scf.parallel (%arg16) = (%c0_7) to (%c4_8) step (%c1_5) init (%41) -> !air.async.token {
          %48 = affine.apply #map11()[%arg16]
          %49 = air.channel.get async [%41]  @Gp2L2[%c1_5, %arg16] (%results_12[%48, %c0_7] [%c64_3, %c64_3] [%c64_3, %c1_5]) {id = 94 : i32} : (memref<256x64xbf16, 1 : i32>)
          scf.reduce(%49 : !air.async.token) {
          ^bb0(%arg17: !air.async.token, %arg18: !air.async.token):
            %50 = air.wait_all async [%arg17, %arg18] 
            scf.reduce.return %50 : !air.async.token
          }
        }
        %43 = air.channel.put async [%42]  @GpOut[%arg12] (%results_12[] [] []) {id = 95 : i32} : (memref<256x64xbf16, 1 : i32>)
        %44 = scf.parallel (%arg16) = (%c0_7) to (%c4_8) step (%c1_5) init (%43) -> !air.async.token {
          %48 = affine.apply #map11()[%arg16]
          %49 = air.channel.get async [%43]  @Gp2L2[%c2_2, %arg16] (%results_12[%48, %c0_7] [%c64_3, %c64_3] [%c64_3, %c1_5]) {id = 96 : i32} : (memref<256x64xbf16, 1 : i32>)
          scf.reduce(%49 : !air.async.token) {
          ^bb0(%arg17: !air.async.token, %arg18: !air.async.token):
            %50 = air.wait_all async [%arg17, %arg18] 
            scf.reduce.return %50 : !air.async.token
          }
        }
        %45 = air.channel.put async [%44]  @GpOut[%arg12] (%results_12[] [] []) {id = 97 : i32} : (memref<256x64xbf16, 1 : i32>)
        %46 = scf.parallel (%arg16) = (%c0_7) to (%c4_8) step (%c1_5) init (%45) -> !air.async.token {
          %48 = affine.apply #map11()[%arg16]
          %49 = air.channel.get async [%45]  @Gp2L2[%c3, %arg16] (%results_12[%48, %c0_7] [%c64_3, %c64_3] [%c64_3, %c1_5]) {id = 98 : i32} : (memref<256x64xbf16, 1 : i32>)
          scf.reduce(%49 : !air.async.token) {
          ^bb0(%arg17: !air.async.token, %arg18: !air.async.token):
            %50 = air.wait_all async [%arg17, %arg18] 
            scf.reduce.return %50 : !air.async.token
          }
        }
        %47 = air.channel.put async [%46]  @GpOut[%arg12] (%results_12[] [] []) {id = 99 : i32} : (memref<256x64xbf16, 1 : i32>)
        %async_token_29 = air.execute [%39] {
          memref.dealloc %results_14 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_30 = air.execute [%39] {
          memref.dealloc %results_16 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_31 = air.execute [%39] {
          memref.dealloc %results_18 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_32 = air.execute [%39] {
          memref.dealloc %results_20 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_33 = air.execute [%39] {
          memref.dealloc %results_22 : memref<64x64xbf16, 2 : i32>
        }
        %async_token_34 = air.execute [%39] {
          memref.dealloc %results_24 : memref<64x1xbf16, 2 : i32>
        }
        %async_token_35 = air.execute [%39] {
          memref.dealloc %results_26 : memref<64x1xbf16, 2 : i32>
        }
        %async_token_36 = air.execute [%38] {
          memref.dealloc %results_10 : memref<64x64xbf16, 1 : i32>
        }
        %async_token_37 = air.execute [%37] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        }
        %async_token_38 = air.execute [%47] {
          memref.dealloc %results_12 : memref<256x64xbf16, 1 : i32>
        }
        %async_token_39 = air.execute [%39] {
          memref.dealloc %results_28 : memref<3xi32, 2 : i32>
        }
      }
      %25 = air.channel.get async [%24]  @GpOut[%c0] (%arg11[%1, %5] [%c256, %c64] [%c512, %c1_1]) {id = 100 : i32} : (memref<512x512xbf16>)
      %26 = air.channel.get async [%24]  @GpOut[%c0] (%arg11[%1, %7] [%c256, %c64] [%c512, %c1_1]) {id = 101 : i32} : (memref<512x512xbf16>)
      %27 = air.channel.get async [%24]  @GpOut[%c0] (%arg11[%1, %9] [%c256, %c64] [%c512, %c1_1]) {id = 102 : i32} : (memref<512x512xbf16>)
      %28 = air.channel.get async [%24]  @GpOut[%c0] (%arg11[%1, %11] [%c256, %c64] [%c512, %c1_1]) {id = 103 : i32} : (memref<512x512xbf16>)
      %29 = air.channel.get async [%24]  @GpOut[%c1_1] (%arg11[%1, %16] [%c256, %c64] [%c512, %c1_1]) {id = 104 : i32} : (memref<512x512xbf16>)
      %30 = air.channel.get async [%24]  @GpOut[%c1_1] (%arg11[%1, %18] [%c256, %c64] [%c512, %c1_1]) {id = 105 : i32} : (memref<512x512xbf16>)
      %31 = air.channel.get async [%24]  @GpOut[%c1_1] (%arg11[%1, %20] [%c256, %c64] [%c512, %c1_1]) {id = 106 : i32} : (memref<512x512xbf16>)
      %32 = air.channel.get async [%24]  @GpOut[%c1_1] (%arg11[%1, %22] [%c256, %c64] [%c512, %c1_1]) {id = 107 : i32} : (memref<512x512xbf16>)
    }
    return
  }
}
