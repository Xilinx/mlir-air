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
      %8 = air.channel.put async [%6]  @QIn[%c0] (%arg8[%1, %7] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) {id = 4 : i32} : (memref<512x512xbf16>)
      %9 = affine.apply #map4()[%arg5]
      %10 = air.channel.put async [%8]  @QIn[%c0] (%arg8[%1, %9] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) {id = 5 : i32} : (memref<512x512xbf16>)
      %11 = affine.apply #map5()[%arg5]
      %12 = air.channel.put async [%10]  @QIn[%c0] (%arg8[%1, %11] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) {id = 6 : i32} : (memref<512x512xbf16>)
      %13 = affine.apply #map6()[%arg5]
      %14 = air.channel.put async  @KIn[%c1_1] (%arg9[%c0, %13] [%c8, %c1_1, %c64, %c64] [%c8192, %c64, %c128, %c1_1]) {id = 7 : i32} : (memref<512x128xbf16>)
      %15 = air.channel.put async  @VIn[%c1_1] (%arg10[%c0, %13] [%c8, %c64, %c64] [%c8192, %c128, %c1_1]) {id = 8 : i32} : (memref<512x128xbf16>)
      %16 = affine.apply #map7()[%arg5]
      %17 = air.channel.put async  @QIn[%c1_1] (%arg8[%1, %16] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) {id = 9 : i32} : (memref<512x512xbf16>)
      %18 = affine.apply #map8()[%arg5]
      %19 = air.channel.put async [%17]  @QIn[%c1_1] (%arg8[%1, %18] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) {id = 10 : i32} : (memref<512x512xbf16>)
      %20 = affine.apply #map9()[%arg5]
      %21 = air.channel.put async [%19]  @QIn[%c1_1] (%arg8[%1, %20] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) {id = 11 : i32} : (memref<512x512xbf16>)
      %22 = affine.apply #map10()[%arg5]
      %23 = air.channel.put async [%21]  @QIn[%c1_1] (%arg8[%1, %22] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) {id = 12 : i32} : (memref<512x512xbf16>)
      %24 = air.segment @attn_seg async  unroll(%arg12, %arg13) in (%arg14=%c2_0, %arg15=%c1_1) attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 8 : i64, y_loc = 2 : i64, y_size = 6 : i64} {
        %c12288 = arith.constant 12288 : index
        %c8192_2 = arith.constant 8192 : index
        %c4096 = arith.constant 4096 : index
        %c3 = arith.constant 3 : index
        %c2_3 = arith.constant 2 : index
        %c64_4 = arith.constant 64 : index
        %c1_5 = arith.constant 1 : index
        %c8_6 = arith.constant 8 : index
        %c0_7 = arith.constant 0 : index
        %c4_8 = arith.constant 4 : index
        %async_token, %results = air.execute -> (memref<64x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
        }
        %33 = air.wait_all async 
        %async_token_9, %results_10 = air.execute -> (memref<256x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<256x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<256x64xbf16, 1 : i32>
        }
        %34 = scf.for %arg16 = %c0_7 to %c4_8 step %c1_5 iter_args(%arg17 = %async_token) -> (!air.async.token) {
          %61 = air.channel.get async [%arg17]  @QIn[%arg12] (%results[] [] []) {id = 13 : i32} : (memref<64x64xbf16, 1 : i32>)
          %62 = arith.cmpi eq, %arg12, %c0_7 : index
          %63 = scf.if %62 -> (!air.async.token) {
            %64 = air.channel.put async [%61]  @Q2L1_0_0[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 14 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          } else {
            %64 = air.channel.put async [%61]  @Q2L1_1_0[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 15 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          }
          scf.yield %63 : !air.async.token
        }
        %35 = scf.for %arg16 = %c0_7 to %c4_8 step %c1_5 iter_args(%arg17 = %34) -> (!air.async.token) {
          %61 = air.channel.get async [%arg17]  @QIn[%arg12] (%results[] [] []) {id = 16 : i32} : (memref<64x64xbf16, 1 : i32>)
          %62 = arith.cmpi eq, %arg12, %c0_7 : index
          %63 = scf.if %62 -> (!air.async.token) {
            %64 = air.channel.put async [%61]  @Q2L1_0_1[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 17 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          } else {
            %64 = air.channel.put async [%61]  @Q2L1_1_1[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 18 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          }
          scf.yield %63 : !air.async.token
        }
        %36 = scf.for %arg16 = %c0_7 to %c4_8 step %c1_5 iter_args(%arg17 = %35) -> (!air.async.token) {
          %61 = air.channel.get async [%arg17]  @QIn[%arg12] (%results[] [] []) {id = 19 : i32} : (memref<64x64xbf16, 1 : i32>)
          %62 = arith.cmpi eq, %arg12, %c0_7 : index
          %63 = scf.if %62 -> (!air.async.token) {
            %64 = air.channel.put async [%61]  @Q2L1_0_2[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 20 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          } else {
            %64 = air.channel.put async [%61]  @Q2L1_1_2[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 21 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          }
          scf.yield %63 : !air.async.token
        }
        %37 = scf.for %arg16 = %c0_7 to %c4_8 step %c1_5 iter_args(%arg17 = %36) -> (!air.async.token) {
          %61 = air.channel.get async [%arg17]  @QIn[%arg12] (%results[] [] []) {id = 22 : i32} : (memref<64x64xbf16, 1 : i32>)
          %62 = arith.cmpi eq, %arg12, %c0_7 : index
          %63 = scf.if %62 -> (!air.async.token) {
            %64 = air.channel.put async [%61]  @Q2L1_0_3[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 23 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          } else {
            %64 = air.channel.put async [%61]  @Q2L1_1_3[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 24 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %64 : !air.async.token
          }
          scf.yield %63 : !air.async.token
        }
        %38 = scf.for %arg16 = %c0_7 to %c8_6 step %c1_5 iter_args(%arg17 = %37) -> (!air.async.token) {
          %61 = air.channel.get async [%arg17]  @KIn[%arg12] (%results[] [] []) {id = 25 : i32} : (memref<64x64xbf16, 1 : i32>)
          %62 = arith.cmpi eq, %arg12, %c0_7 : index
          %63:4 = scf.if %62 -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
            %65 = air.channel.put async [%61]  @K2L1_0_0[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 26 : i32} : (memref<64x64xbf16, 1 : i32>)
            %66 = air.channel.put async [%61]  @K2L1_0_1[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 27 : i32} : (memref<64x64xbf16, 1 : i32>)
            %67 = air.channel.put async [%61]  @K2L1_0_2[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 28 : i32} : (memref<64x64xbf16, 1 : i32>)
            %68 = air.channel.put async [%61]  @K2L1_0_3[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 29 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %65, %66, %67, %68 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
          } else {
            %65 = air.channel.put async [%61]  @K2L1_1_0[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 30 : i32} : (memref<64x64xbf16, 1 : i32>)
            %66 = air.channel.put async [%61]  @K2L1_1_1[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 31 : i32} : (memref<64x64xbf16, 1 : i32>)
            %67 = air.channel.put async [%61]  @K2L1_1_2[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 32 : i32} : (memref<64x64xbf16, 1 : i32>)
            %68 = air.channel.put async [%61]  @K2L1_1_3[%c0_7, %c0_7, %c0_7] (%results[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 33 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %65, %66, %67, %68 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
          }
          %64 = air.wait_all async [%63#0, %63#1, %63#2, %63#3] 
          scf.yield %64 : !air.async.token
        }
        %39 = scf.for %arg16 = %c0_7 to %c8_6 step %c1_5 iter_args(%arg17 = %33) -> (!air.async.token) {
          %async_token_13, %results_14 = air.execute -> (memref<64x64xbf16, 1 : i32>) {
            %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
            air.execute_terminator %alloc : memref<64x64xbf16, 1 : i32>
          }
          %61 = air.channel.get async [%async_token_13, %arg17]  @VIn[%arg12] (%results_14[] [] []) {id = 34 : i32} : (memref<64x64xbf16, 1 : i32>)
          %62 = arith.cmpi eq, %arg12, %c0_7 : index
          %63:4 = scf.if %62 -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
            %65 = air.channel.put async [%61]  @V2L1_0_0[%c0_7, %c0_7, %c0_7] (%results_14[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 35 : i32} : (memref<64x64xbf16, 1 : i32>)
            %66 = air.channel.put async [%61]  @V2L1_0_1[%c0_7, %c0_7, %c0_7] (%results_14[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 36 : i32} : (memref<64x64xbf16, 1 : i32>)
            %67 = air.channel.put async [%61]  @V2L1_0_2[%c0_7, %c0_7, %c0_7] (%results_14[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 37 : i32} : (memref<64x64xbf16, 1 : i32>)
            %68 = air.channel.put async [%61]  @V2L1_0_3[%c0_7, %c0_7, %c0_7] (%results_14[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 38 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %65, %66, %67, %68 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
          } else {
            %65 = air.channel.put async [%61]  @V2L1_1_0[%c0_7, %c0_7, %c0_7] (%results_14[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 39 : i32} : (memref<64x64xbf16, 1 : i32>)
            %66 = air.channel.put async [%61]  @V2L1_1_1[%c0_7, %c0_7, %c0_7] (%results_14[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 40 : i32} : (memref<64x64xbf16, 1 : i32>)
            %67 = air.channel.put async [%61]  @V2L1_1_2[%c0_7, %c0_7, %c0_7] (%results_14[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 41 : i32} : (memref<64x64xbf16, 1 : i32>)
            %68 = air.channel.put async [%61]  @V2L1_1_3[%c0_7, %c0_7, %c0_7] (%results_14[%c0_7, %c0_7, %c0_7] [%c8_6, %c64_4, %c8_6] [%c8_6, %c64_4, %c1_5]) {id = 42 : i32} : (memref<64x64xbf16, 1 : i32>)
            scf.yield %65, %66, %67, %68 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
          }
          %64 = air.wait_all async [%63#0, %63#1, %63#2, %63#3] 
          %async_token_15 = air.execute [%63#0, %61] {
            memref.dealloc %results_14 : memref<64x64xbf16, 1 : i32>
          }
          scf.yield %64 : !air.async.token
        }
        %40 = air.herd @herd_0 async  tile (%arg16, %arg17) in (%arg18=%c4_8, %arg19=%c4_8) args(%arg20=%arg12) : index attributes {id = 3 : i32, link_with = "attn_npu2.o", x_loc = 0 : i64, y_loc = 2 : i64} {
          %c64_13 = arith.constant 64 : index
          %c8_i32 = arith.constant 8 : i32
          %c0_14 = arith.constant 0 : index
          %c1_15 = arith.constant 1 : index
          %c2_16 = arith.constant 2 : index
          %c0_i32 = arith.constant 0 : i32
          %c1_i32 = arith.constant 1 : i32
          %c2_i32 = arith.constant 2 : i32
          %c3_i32 = arith.constant 3 : i32
          %c8_17 = arith.constant 8 : index
          %c512_18 = arith.constant 512 : index
          %c4_i32 = arith.constant 4 : i32
          %async_token_19, %results_20 = air.execute -> (memref<3xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<3xi32, 2 : i32>
            air.execute_terminator %alloc : memref<3xi32, 2 : i32>
          }
          %async_token_21, %results_22 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
          }
          %async_token_23, %results_24 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
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
          %async_token_31 = air.execute [%async_token_25] {
            func.call @zero_fill_gp_bf16(%results_26) : (memref<64x64xbf16, 2 : i32>) -> ()
          }
          %async_token_32 = air.execute [%async_token_21] {
            func.call @zero_fill_sp_bf16(%results_22) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_33 = air.execute [%async_token_23] {
            func.call @neg_inf_fill_up_bf16(%results_24) : (memref<64x1xbf16, 2 : i32>) -> ()
          }
          %async_token_34, %results_35 = air.execute [%async_token_19] -> (i32) {
            %74 = memref.load %results_20[%c1_15] : memref<3xi32, 2 : i32>
            air.execute_terminator %74 : i32
          }
          %61 = arith.cmpi eq, %results_35, %c0_i32 : i32
          scf.if %61 {
            %async_token_45 = air.execute [%async_token_34] {
              memref.store %c0_i32, %results_20[%c0_14] : memref<3xi32, 2 : i32>
            }
            %async_token_46 = air.execute [%async_token_45] {
              memref.store %c1_i32, %results_20[%c1_15] : memref<3xi32, 2 : i32>
            }
            %async_token_47 = air.execute [%async_token_46] {
              memref.store %c0_i32, %results_20[%c2_16] : memref<3xi32, 2 : i32>
            }
          }
          %62 = arith.cmpi eq, %arg20, %c0_14 : index
          scf.if %62 {
            %74 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %75 = air.channel.get async [%async_token_27]  @Q2L1_0_0[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 43 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %75 : !air.async.token
            } else {
              %75 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %76 = air.channel.get async [%async_token_27]  @Q2L1_0_1[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 44 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %76 : !air.async.token
              } else {
                %76 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_0_2[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 45 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                } else {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_0_3[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 46 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                }
                affine.yield %76 : !air.async.token
              }
              affine.yield %75 : !air.async.token
            }
          } else {
            %74 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %75 = air.channel.get async [%async_token_27]  @Q2L1_1_0[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 47 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %75 : !air.async.token
            } else {
              %75 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %76 = air.channel.get async [%async_token_27]  @Q2L1_1_1[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 48 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %76 : !air.async.token
              } else {
                %76 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_1_2[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 49 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                } else {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_1_3[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 50 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                }
                affine.yield %76 : !air.async.token
              }
              affine.yield %75 : !air.async.token
            }
          }
          %63 = arith.index_cast %arg16 : index to i32
          %64 = arith.cmpi eq, %63, %c0_i32 : i32
          scf.if %64 {
            %async_token_45 = air.execute [%async_token_27, %async_token_29] {
              func.call @copy_tile(%results_28, %results_30) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %62 {
            %74 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %75 = air.channel.get async [%async_token_27]  @Q2L1_0_0[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 51 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %75 : !air.async.token
            } else {
              %75 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %76 = air.channel.get async [%async_token_27]  @Q2L1_0_1[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 52 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %76 : !air.async.token
              } else {
                %76 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_0_2[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 53 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                } else {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_0_3[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 54 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                }
                affine.yield %76 : !air.async.token
              }
              affine.yield %75 : !air.async.token
            }
          } else {
            %74 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %75 = air.channel.get async [%async_token_27]  @Q2L1_1_0[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 55 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %75 : !air.async.token
            } else {
              %75 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %76 = air.channel.get async [%async_token_27]  @Q2L1_1_1[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 56 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %76 : !air.async.token
              } else {
                %76 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_1_2[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 57 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                } else {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_1_3[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 58 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                }
                affine.yield %76 : !air.async.token
              }
              affine.yield %75 : !air.async.token
            }
          }
          %65 = arith.cmpi eq, %63, %c1_i32 : i32
          scf.if %65 {
            %async_token_45 = air.execute [%async_token_27, %async_token_29] {
              func.call @copy_tile(%results_28, %results_30) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %62 {
            %74 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %75 = air.channel.get async [%async_token_27]  @Q2L1_0_0[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 59 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %75 : !air.async.token
            } else {
              %75 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %76 = air.channel.get async [%async_token_27]  @Q2L1_0_1[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 60 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %76 : !air.async.token
              } else {
                %76 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_0_2[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 61 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                } else {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_0_3[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 62 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                }
                affine.yield %76 : !air.async.token
              }
              affine.yield %75 : !air.async.token
            }
          } else {
            %74 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %75 = air.channel.get async [%async_token_27]  @Q2L1_1_0[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 63 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %75 : !air.async.token
            } else {
              %75 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %76 = air.channel.get async [%async_token_27]  @Q2L1_1_1[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 64 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %76 : !air.async.token
              } else {
                %76 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_1_2[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 65 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                } else {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_1_3[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 66 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                }
                affine.yield %76 : !air.async.token
              }
              affine.yield %75 : !air.async.token
            }
          }
          %66 = arith.cmpi eq, %63, %c2_i32 : i32
          scf.if %66 {
            %async_token_45 = air.execute [%async_token_27, %async_token_29] {
              func.call @copy_tile(%results_28, %results_30) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          scf.if %62 {
            %74 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %75 = air.channel.get async [%async_token_27]  @Q2L1_0_0[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 67 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %75 : !air.async.token
            } else {
              %75 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %76 = air.channel.get async [%async_token_27]  @Q2L1_0_1[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 68 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %76 : !air.async.token
              } else {
                %76 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_0_2[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 69 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                } else {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_0_3[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 70 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                }
                affine.yield %76 : !air.async.token
              }
              affine.yield %75 : !air.async.token
            }
          } else {
            %74 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
              %75 = air.channel.get async [%async_token_27]  @Q2L1_1_0[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 71 : i32} : (memref<64x64xbf16, 2 : i32>)
              affine.yield %75 : !air.async.token
            } else {
              %75 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                %76 = air.channel.get async [%async_token_27]  @Q2L1_1_1[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 72 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %76 : !air.async.token
              } else {
                %76 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_1_2[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 73 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                } else {
                  %77 = air.channel.get async [%async_token_27]  @Q2L1_1_3[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 74 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %77 : !air.async.token
                }
                affine.yield %76 : !air.async.token
              }
              affine.yield %75 : !air.async.token
            }
          }
          %67 = arith.cmpi eq, %63, %c3_i32 : i32
          scf.if %67 {
            %async_token_45 = air.execute [%async_token_27, %async_token_29] {
              func.call @copy_tile(%results_28, %results_30) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
          }
          %68 = air.wait_all async [%async_token_19, %async_token_27, %async_token_29, %async_token_31, %async_token_32, %async_token_33] 
          %69 = scf.for %arg21 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg22 = %68) -> (!air.async.token)  : i32 {
            %async_token_45, %results_46 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
            }
            %async_token_47, %results_48 = air.execute -> (memref<64x64xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x64xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x64xbf16, 2 : i32>
            }
            %async_token_49 = air.execute [%async_token_47, %arg22] {
              %collapse_shape = memref.collapse_shape %results_48 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            }
            scf.if %62 {
              %76 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
                %77 = air.channel.get async [%arg22]  @K2L1_0_0[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 75 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %77 : !air.async.token
              } else {
                %77 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                  %78 = air.channel.get async [%arg22]  @K2L1_0_1[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 76 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %78 : !air.async.token
                } else {
                  %78 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                    %79 = air.channel.get async [%arg22]  @K2L1_0_2[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 77 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %79 : !air.async.token
                  } else {
                    %79 = air.channel.get async [%arg22]  @K2L1_0_3[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 78 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %79 : !air.async.token
                  }
                  affine.yield %78 : !air.async.token
                }
                affine.yield %77 : !air.async.token
              }
            } else {
              %76 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
                %77 = air.channel.get async [%arg22]  @K2L1_1_0[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 79 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %77 : !air.async.token
              } else {
                %77 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                  %78 = air.channel.get async [%arg22]  @K2L1_1_1[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 80 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %78 : !air.async.token
                } else {
                  %78 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                    %79 = air.channel.get async [%arg22]  @K2L1_1_2[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 81 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %79 : !air.async.token
                  } else {
                    %79 = air.channel.get async [%arg22]  @K2L1_1_3[%c0_14, %c0_14, %arg16] (%results_28[] [] []) {id = 82 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %79 : !air.async.token
                  }
                  affine.yield %78 : !air.async.token
                }
                affine.yield %77 : !air.async.token
              }
            }
            %async_token_50 = air.execute [%async_token_49] {
              %collapse_shape = memref.collapse_shape %results_48 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_a_b_bf16(%results_30, %results_28, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            }
            scf.if %62 {
              %76 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
                %77 = air.channel.get async [%async_token_45, %arg22]  @V2L1_0_0[%c0_14, %c0_14, %arg16] (%results_46[] [] []) {id = 83 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %77 : !air.async.token
              } else {
                %77 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                  %78 = air.channel.get async [%async_token_45, %arg22]  @V2L1_0_1[%c0_14, %c0_14, %arg16] (%results_46[] [] []) {id = 84 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %78 : !air.async.token
                } else {
                  %78 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                    %79 = air.channel.get async [%async_token_45, %arg22]  @V2L1_0_2[%c0_14, %c0_14, %arg16] (%results_46[] [] []) {id = 85 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %79 : !air.async.token
                  } else {
                    %79 = air.channel.get async [%async_token_45, %arg22]  @V2L1_0_3[%c0_14, %c0_14, %arg16] (%results_46[] [] []) {id = 86 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %79 : !air.async.token
                  }
                  affine.yield %78 : !air.async.token
                }
                affine.yield %77 : !air.async.token
              }
            } else {
              %76 = affine.if #set()[%arg16, %arg17] -> !air.async.token {
                %77 = air.channel.get async [%async_token_45, %arg22]  @V2L1_1_0[%c0_14, %c0_14, %arg16] (%results_46[] [] []) {id = 87 : i32} : (memref<64x64xbf16, 2 : i32>)
                affine.yield %77 : !air.async.token
              } else {
                %77 = affine.if #set1()[%arg16, %arg17] -> !air.async.token {
                  %78 = air.channel.get async [%async_token_45, %arg22]  @V2L1_1_1[%c0_14, %c0_14, %arg16] (%results_46[] [] []) {id = 88 : i32} : (memref<64x64xbf16, 2 : i32>)
                  affine.yield %78 : !air.async.token
                } else {
                  %78 = affine.if #set2()[%arg16, %arg17] -> !air.async.token {
                    %79 = air.channel.get async [%async_token_45, %arg22]  @V2L1_1_2[%c0_14, %c0_14, %arg16] (%results_46[] [] []) {id = 89 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %79 : !air.async.token
                  } else {
                    %79 = air.channel.get async [%async_token_45, %arg22]  @V2L1_1_3[%c0_14, %c0_14, %arg16] (%results_46[] [] []) {id = 90 : i32} : (memref<64x64xbf16, 2 : i32>)
                    affine.yield %79 : !air.async.token
                  }
                  affine.yield %78 : !air.async.token
                }
                affine.yield %77 : !air.async.token
              }
            }
            %async_token_51, %results_52 = air.execute [%arg22] -> (i32) {
              %76 = memref.load %results_20[%c0_14] : memref<3xi32, 2 : i32>
              air.execute_terminator %76 : i32
            }
            %74 = arith.addi %results_52, %63 : i32
            %async_token_53 = air.execute [%async_token_50] {
              func.call @apply_causal_mask(%results_48, %74, %arg21) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
            }
            %async_token_54, %results_55 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_56, %results_57 = air.execute -> (memref<64x1xbf16, 2 : i32>) {
              %alloc = memref.alloc() : memref<64x1xbf16, 2 : i32>
              air.execute_terminator %alloc : memref<64x1xbf16, 2 : i32>
            }
            %async_token_58 = air.execute [%async_token_53, %async_token_54, %async_token_56, %arg22] {
              %collapse_shape = memref.collapse_shape %results_48 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @fused_softmax(%collapse_shape, %results_24, %results_55, %results_57) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_59 = air.execute [%async_token_58] {
              func.call @mul_r_gp(%results_57, %results_26) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_60 = air.execute [%async_token_59, %async_token_45, %async_token_47] {
              %collapse_shape = memref.collapse_shape %results_48 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
              func.call @matmul_g_b_bf16(%collapse_shape, %results_46, %results_26) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            }
            %async_token_61 = air.execute [%async_token_59] {
              func.call @accum_sp_r_s(%results_22, %results_57, %results_55) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_62 = air.execute [%async_token_61] {
              func.call @vector_copy_32elems(%c0_i32, %results_55, %results_22) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            }
            %async_token_63 = air.execute [%async_token_62] {
              memref.dealloc %results_55 : memref<64x1xbf16, 2 : i32>
            }
            %async_token_64 = air.execute [%async_token_61] {
              memref.dealloc %results_57 : memref<64x1xbf16, 2 : i32>
            }
            %75 = air.wait_all async [%async_token_51, %async_token_60, %async_token_62] 
            %async_token_65 = air.execute [%async_token_58, %async_token_60] {
              memref.dealloc %results_48 : memref<64x64xbf16, 2 : i32>
            }
            %async_token_66 = air.execute [%async_token_60] {
              memref.dealloc %results_46 : memref<64x64xbf16, 2 : i32>
            }
            scf.yield %75 : !air.async.token
          }
          %async_token_36 = air.execute [%async_token_21, %async_token_25, %69] {
            func.call @div_gp_sp(%results_22, %results_26) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          %70 = air.channel.put async [%async_token_36]  @Gp2L2[%arg17, %arg16] (%results_26[%c0_14, %c0_14, %c0_14] [%c64_13, %c8_17, %c8_17] [%c8_17, %c512_18, %c1_15]) {id = 91 : i32} : (memref<64x64xbf16, 2 : i32>)
          %async_token_37, %results_38 = air.execute [%async_token_19, %69] -> (i32) {
            %74 = memref.load %results_20[%c2_16] : memref<3xi32, 2 : i32>
            air.execute_terminator %74 : i32
          }
          %71 = arith.addi %results_38, %c1_i32 : i32
          %72 = arith.cmpi sge, %71, %c1_i32 : i32
          scf.if %72 {
            %async_token_45, %results_46 = air.execute [%async_token_37] -> (i32) {
              %75 = memref.load %results_20[%c0_14] : memref<3xi32, 2 : i32>
              air.execute_terminator %75 : i32
            }
            %74 = arith.addi %results_46, %c4_i32 : i32
            %async_token_47 = air.execute [%async_token_45] {
              memref.store %74, %results_20[%c0_14] : memref<3xi32, 2 : i32>
            }
            %async_token_48 = air.execute [%async_token_47] {
              memref.store %c0_i32, %results_20[%c2_16] : memref<3xi32, 2 : i32>
            }
          }
          %73 = arith.cmpi slt, %71, %c1_i32 : i32
          scf.if %73 {
            %async_token_45 = air.execute [%async_token_19] {
              memref.store %71, %results_20[%c2_16] : memref<3xi32, 2 : i32>
            }
          }
          %async_token_39 = air.execute [%69] {
            memref.dealloc %results_30 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_40 = air.execute [%69] {
            memref.dealloc %results_28 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_41 = air.execute [%async_token_31, %70] {
            memref.dealloc %results_26 : memref<64x64xbf16, 2 : i32>
          }
          %async_token_42 = air.execute [%69, %async_token_33] {
            memref.dealloc %results_24 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_43 = air.execute [%async_token_32, %async_token_36] {
            memref.dealloc %results_22 : memref<64x1xbf16, 2 : i32>
          }
          %async_token_44 = air.execute [%async_token_34, %async_token_37] {
            memref.dealloc %results_20 : memref<3xi32, 2 : i32>
          }
        }
        %41 = air.channel.get async [%async_token_9]  @Gp2L2[%c0_7, %c0_7] (%results_10[%c0_7] [%c4096] [%c1_5]) {id = 92 : i32} : (memref<256x64xbf16, 1 : i32>)
        %42 = air.channel.get async [%async_token_9]  @Gp2L2[%c0_7, %c1_5] (%results_10[%c4096] [%c4096] [%c1_5]) {id = 93 : i32} : (memref<256x64xbf16, 1 : i32>)
        %43 = air.channel.get async [%async_token_9]  @Gp2L2[%c0_7, %c2_3] (%results_10[%c8192_2] [%c4096] [%c1_5]) {id = 94 : i32} : (memref<256x64xbf16, 1 : i32>)
        %44 = air.channel.get async [%async_token_9]  @Gp2L2[%c0_7, %c3] (%results_10[%c12288] [%c4096] [%c1_5]) {id = 95 : i32} : (memref<256x64xbf16, 1 : i32>)
        %45 = air.channel.put async [%41, %42, %43, %44]  @GpOut[%arg12] (%results_10[] [] []) {id = 96 : i32} : (memref<256x64xbf16, 1 : i32>)
        %46 = air.channel.get async [%45]  @Gp2L2[%c1_5, %c0_7] (%results_10[%c0_7] [%c4096] [%c1_5]) {id = 97 : i32} : (memref<256x64xbf16, 1 : i32>)
        %47 = air.channel.get async [%45]  @Gp2L2[%c1_5, %c1_5] (%results_10[%c4096] [%c4096] [%c1_5]) {id = 98 : i32} : (memref<256x64xbf16, 1 : i32>)
        %48 = air.channel.get async [%45]  @Gp2L2[%c1_5, %c2_3] (%results_10[%c8192_2] [%c4096] [%c1_5]) {id = 99 : i32} : (memref<256x64xbf16, 1 : i32>)
        %49 = air.channel.get async [%45]  @Gp2L2[%c1_5, %c3] (%results_10[%c12288] [%c4096] [%c1_5]) {id = 100 : i32} : (memref<256x64xbf16, 1 : i32>)
        %50 = air.channel.put async [%45, %46, %47, %48, %49]  @GpOut[%arg12] (%results_10[] [] []) {id = 101 : i32} : (memref<256x64xbf16, 1 : i32>)
        %51 = air.channel.get async [%50]  @Gp2L2[%c2_3, %c0_7] (%results_10[%c0_7] [%c4096] [%c1_5]) {id = 102 : i32} : (memref<256x64xbf16, 1 : i32>)
        %52 = air.channel.get async [%50]  @Gp2L2[%c2_3, %c1_5] (%results_10[%c4096] [%c4096] [%c1_5]) {id = 103 : i32} : (memref<256x64xbf16, 1 : i32>)
        %53 = air.channel.get async [%50]  @Gp2L2[%c2_3, %c2_3] (%results_10[%c8192_2] [%c4096] [%c1_5]) {id = 104 : i32} : (memref<256x64xbf16, 1 : i32>)
        %54 = air.channel.get async [%50]  @Gp2L2[%c2_3, %c3] (%results_10[%c12288] [%c4096] [%c1_5]) {id = 105 : i32} : (memref<256x64xbf16, 1 : i32>)
        %55 = air.channel.put async [%50, %51, %52, %53, %54]  @GpOut[%arg12] (%results_10[] [] []) {id = 106 : i32} : (memref<256x64xbf16, 1 : i32>)
        %56 = air.channel.get async [%55]  @Gp2L2[%c3, %c0_7] (%results_10[%c0_7] [%c4096] [%c1_5]) {id = 107 : i32} : (memref<256x64xbf16, 1 : i32>)
        %57 = air.channel.get async [%55]  @Gp2L2[%c3, %c1_5] (%results_10[%c4096] [%c4096] [%c1_5]) {id = 108 : i32} : (memref<256x64xbf16, 1 : i32>)
        %58 = air.channel.get async [%55]  @Gp2L2[%c3, %c2_3] (%results_10[%c8192_2] [%c4096] [%c1_5]) {id = 109 : i32} : (memref<256x64xbf16, 1 : i32>)
        %59 = air.channel.get async [%55]  @Gp2L2[%c3, %c3] (%results_10[%c12288] [%c4096] [%c1_5]) {id = 110 : i32} : (memref<256x64xbf16, 1 : i32>)
        %60 = air.channel.put async [%55, %56, %57, %58, %59]  @GpOut[%arg12] (%results_10[] [] []) {id = 111 : i32} : (memref<256x64xbf16, 1 : i32>)
        %async_token_11 = air.execute [%38] {
          memref.dealloc %results : memref<64x64xbf16, 1 : i32>
        }
        %async_token_12 = air.execute [%60] {
          memref.dealloc %results_10 : memref<256x64xbf16, 1 : i32>
        }
        air.wait_all [%39, %40, %async_token_11, %async_token_12]  {air.segment_end}
      }
      %25 = air.channel.get async [%24]  @GpOut[%c0] (%arg11[%1, %5] [%c256, %c64] [%c512, %c1_1]) {id = 112 : i32} : (memref<512x512xbf16>)
      %26 = air.channel.get async [%24, %25]  @GpOut[%c0] (%arg11[%1, %7] [%c256, %c64] [%c512, %c1_1]) {id = 113 : i32} : (memref<512x512xbf16>)
      %27 = air.channel.get async [%24, %26]  @GpOut[%c0] (%arg11[%1, %9] [%c256, %c64] [%c512, %c1_1]) {id = 114 : i32} : (memref<512x512xbf16>)
      %28 = air.channel.get async [%24, %27]  @GpOut[%c0] (%arg11[%1, %11] [%c256, %c64] [%c512, %c1_1]) {id = 115 : i32} : (memref<512x512xbf16>)
      %29 = air.channel.get async [%24]  @GpOut[%c1_1] (%arg11[%1, %16] [%c256, %c64] [%c512, %c1_1]) {id = 116 : i32} : (memref<512x512xbf16>)
      %30 = air.channel.get async [%24, %29]  @GpOut[%c1_1] (%arg11[%1, %18] [%c256, %c64] [%c512, %c1_1]) {id = 117 : i32} : (memref<512x512xbf16>)
      %31 = air.channel.get async [%24, %30]  @GpOut[%c1_1] (%arg11[%1, %20] [%c256, %c64] [%c512, %c1_1]) {id = 118 : i32} : (memref<512x512xbf16>)
      %32 = air.channel.get async [%24, %31]  @GpOut[%c1_1] (%arg11[%1, %22] [%c256, %c64] [%c512, %c1_1]) {id = 119 : i32} : (memref<512x512xbf16>)
    }
    return
  }
}
