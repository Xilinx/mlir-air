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
  air.channel @Q2L1 [2, 4, 1] {broadcast_shape = [2 : index, 4 : index, 4 : index]}
  air.channel @QIn [2]
  air.channel @K2L1 [2, 4, 1] {broadcast_shape = [2 : index, 4 : index, 4 : index]}
  air.channel @KIn [2]
  air.channel @V2L1 [2, 4, 1] {broadcast_shape = [2 : index, 4 : index, 4 : index]}
  air.channel @VIn [2]
  air.channel @Gp2L2 [4, 4]
  air.channel @GpOut [2]
  func.func @attention_bf16(%arg0: memref<512x512xbf16>, %arg1: memref<512x128xbf16>, %arg2: memref<512x128xbf16>, %arg3: memref<512x512xbf16>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    air.launch (%arg4, %arg5) in (%arg6=%c2, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<512x512xbf16>, memref<512x128xbf16>, memref<512x128xbf16>, memref<512x512xbf16> {
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
      %0 = affine.apply #map()[%arg4]
      %1 = affine.apply #map1()[%arg5]
      %2 = affine.apply #map1()[%arg5]
      air.channel.put  @KIn[%c0] (%arg9[%c0, %1] [%c8, %c1_1, %c64, %c64] [%c8192, %c64, %c128, %c1_1]) : (memref<512x128xbf16>)
      air.channel.put  @VIn[%c0] (%arg10[%c0, %2] [%c8, %c64, %c64] [%c8192, %c128, %c1_1]) : (memref<512x128xbf16>)
      %3 = affine.apply #map2()[%arg5]
      air.channel.put  @QIn[%c0] (%arg8[%0, %3] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) : (memref<512x512xbf16>)
      %4 = affine.apply #map3()[%arg5]
      air.channel.put  @QIn[%c0] (%arg8[%0, %4] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) : (memref<512x512xbf16>)
      %5 = affine.apply #map4()[%arg5]
      air.channel.put  @QIn[%c0] (%arg8[%0, %5] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) : (memref<512x512xbf16>)
      %6 = affine.apply #map5()[%arg5]
      air.channel.put  @QIn[%c0] (%arg8[%0, %6] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) : (memref<512x512xbf16>)
      %7 = affine.apply #map6()[%arg5]
      %8 = affine.apply #map6()[%arg5]
      air.channel.put  @KIn[%c1_1] (%arg9[%c0, %7] [%c8, %c1_1, %c64, %c64] [%c8192, %c64, %c128, %c1_1]) : (memref<512x128xbf16>)
      air.channel.put  @VIn[%c1_1] (%arg10[%c0, %8] [%c8, %c64, %c64] [%c8192, %c128, %c1_1]) : (memref<512x128xbf16>)
      %9 = affine.apply #map7()[%arg5]
      air.channel.put  @QIn[%c1_1] (%arg8[%0, %9] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) : (memref<512x512xbf16>)
      %10 = affine.apply #map8()[%arg5]
      air.channel.put  @QIn[%c1_1] (%arg8[%0, %10] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) : (memref<512x512xbf16>)
      %11 = affine.apply #map9()[%arg5]
      air.channel.put  @QIn[%c1_1] (%arg8[%0, %11] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) : (memref<512x512xbf16>)
      %12 = affine.apply #map10()[%arg5]
      air.channel.put  @QIn[%c1_1] (%arg8[%0, %12] [%c4, %c1_1, %c64, %c64] [%c32768, %c64, %c512, %c1_1]) : (memref<512x512xbf16>)
      air.segment @attn_seg  unroll(%arg12, %arg13) in (%arg14=%c2_0, %arg15=%c1_1) {
        %c3 = arith.constant 3 : index
        %c2_2 = arith.constant 2 : index
        %c64_3 = arith.constant 64 : index
        %c512_4 = arith.constant 512 : index
        %c1_5 = arith.constant 1 : index
        %c8_6 = arith.constant 8 : index
        %c0_7 = arith.constant 0 : index
        %c4_8 = arith.constant 4 : index
        %alloc = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_9 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_10 = memref.alloc() : memref<256x64xbf16, 1 : i32>
        %alloc_11 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_12 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_13 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_14 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_15 = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %alloc_16 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        %alloc_17 = memref.alloc() : memref<64x1xbf16, 2 : i32>
        %alloc_18 = memref.alloc() : memref<3xi32, 2 : i32>
        scf.for %arg16 = %c0_7 to %c4_8 step %c1_5 {
          air.channel.get  @QIn[%arg12] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @Q2L1[%arg12, %c0_7, %c0_7] (%alloc[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_7 to %c4_8 step %c1_5 {
          air.channel.get  @QIn[%arg12] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @Q2L1[%arg12, %c1_5, %c0_7] (%alloc[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_7 to %c4_8 step %c1_5 {
          air.channel.get  @QIn[%arg12] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @Q2L1[%arg12, %c2_2, %c0_7] (%alloc[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_7 to %c4_8 step %c1_5 {
          air.channel.get  @QIn[%arg12] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @Q2L1[%arg12, %c3, %c0_7] (%alloc[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_7 to %c8_6 step %c1_5 {
          air.channel.get  @KIn[%arg12] (%alloc[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @K2L1[%arg12, %c0_7, %c0_7] (%alloc[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @K2L1[%arg12, %c1_5, %c0_7] (%alloc[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @K2L1[%arg12, %c2_2, %c0_7] (%alloc[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @K2L1[%arg12, %c3, %c0_7] (%alloc[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) : (memref<64x64xbf16, 1 : i32>)
        }
        scf.for %arg16 = %c0_7 to %c8_6 step %c1_5 {
          air.channel.get  @VIn[%arg12] (%alloc_9[] [] []) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @V2L1[%arg12, %c0_7, %c0_7] (%alloc_9[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @V2L1[%arg12, %c1_5, %c0_7] (%alloc_9[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @V2L1[%arg12, %c2_2, %c0_7] (%alloc_9[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) : (memref<64x64xbf16, 1 : i32>)
          air.channel.put  @V2L1[%arg12, %c3, %c0_7] (%alloc_9[%c0_7, %c0_7, %c0_7, %c0_7] [%c8_6, %c8_6, %c8_6, %c8_6] [%c8_6, %c512_4, %c64_3, %c1_5]) : (memref<64x64xbf16, 1 : i32>)
        }
        air.herd @herd_0  tile (%arg16, %arg17) in (%arg18=%c4_8, %arg19=%c4_8) args(%arg20=%alloc_11, %arg21=%alloc_12, %arg22=%alloc_13, %arg23=%alloc_14, %arg24=%alloc_15, %arg25=%alloc_16, %arg26=%alloc_17, %arg27=%arg12, %arg28=%alloc_18) : memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, index, memref<3xi32, 2 : i32> attributes {link_with = "attn_npu2.o"} {
          %c4_i32 = arith.constant 4 : i32
          %c512_19 = arith.constant 512 : index
          %c64_20 = arith.constant 64 : index
          %c8_21 = arith.constant 8 : index
          %c3_i32 = arith.constant 3 : i32
          %c2_i32 = arith.constant 2 : i32
          %c1_i32 = arith.constant 1 : i32
          %c0_i32 = arith.constant 0 : i32
          %c2_22 = arith.constant 2 : index
          %c1_23 = arith.constant 1 : index
          %c0_24 = arith.constant 0 : index
          func.call @zero_fill_gp_bf16(%arg24) : (memref<64x64xbf16, 2 : i32>) -> ()
          func.call @zero_fill_sp_bf16(%arg26) : (memref<64x1xbf16, 2 : i32>) -> ()
          func.call @neg_inf_fill_up_bf16(%arg25) : (memref<64x1xbf16, 2 : i32>) -> ()
          %21 = memref.load %arg28[%c1_23] : memref<3xi32, 2 : i32>
          %22 = arith.cmpi eq, %21, %c0_i32 : i32
          scf.if %22 {
            memref.store %c0_i32, %arg28[%c0_24] : memref<3xi32, 2 : i32>
            memref.store %c1_i32, %arg28[%c1_23] : memref<3xi32, 2 : i32>
            memref.store %c0_i32, %arg28[%c2_22] : memref<3xi32, 2 : i32>
          }
          air.channel.get  @Q2L1[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          %23 = arith.index_cast %arg16 : index to i32
          %24 = arith.cmpi eq, %23, %c0_i32 : i32
          scf.if %24 {
            func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          air.channel.get  @Q2L1[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          %25 = arith.index_cast %arg16 : index to i32
          %26 = arith.cmpi eq, %25, %c1_i32 : i32
          scf.if %26 {
            func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          air.channel.get  @Q2L1[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          %27 = arith.index_cast %arg16 : index to i32
          %28 = arith.cmpi eq, %27, %c2_i32 : i32
          scf.if %28 {
            func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          air.channel.get  @Q2L1[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
          %29 = arith.index_cast %arg16 : index to i32
          %30 = arith.cmpi eq, %29, %c3_i32 : i32
          scf.if %30 {
            func.call @copy_tile(%arg21, %arg20) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          }
          scf.for %arg29 = %c0_24 to %c8_21 step %c1_23 {
            %collapse_shape = memref.collapse_shape %arg23 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
            func.call @zero_fill_g_bf16(%collapse_shape) : (memref<4096xbf16, 2 : i32>) -> ()
            air.channel.get  @K2L1[%arg27, %arg17, %arg16] (%arg21[] [] []) : (memref<64x64xbf16, 2 : i32>)
            func.call @matmul_a_b_bf16(%arg20, %arg21, %collapse_shape) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
            air.channel.get  @V2L1[%arg27, %arg17, %arg16] (%arg22[] [] []) : (memref<64x64xbf16, 2 : i32>)
            %35 = arith.index_cast %arg29 : index to i32
            %36 = memref.load %arg28[%c0_24] : memref<3xi32, 2 : i32>
            %37 = arith.index_cast %arg16 : index to i32
            %38 = arith.addi %36, %37 : i32
            func.call @apply_causal_mask(%arg23, %38, %35) : (memref<64x64xbf16, 2 : i32>, i32, i32) -> ()
            %alloc_25 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            %alloc_26 = memref.alloc() : memref<64x1xbf16, 2 : i32>
            func.call @fused_softmax(%collapse_shape, %arg25, %alloc_25, %alloc_26) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @mul_r_gp(%alloc_26, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            func.call @matmul_g_b_bf16(%collapse_shape, %arg22, %arg24) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
            func.call @accum_sp_r_s(%arg26, %alloc_26, %alloc_25) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            func.call @vector_copy_32elems(%c0_i32, %alloc_25, %arg26) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
            memref.dealloc %alloc_25 : memref<64x1xbf16, 2 : i32>
            memref.dealloc %alloc_26 : memref<64x1xbf16, 2 : i32>
          }
          func.call @div_gp_sp(%arg26, %arg24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
          air.channel.put  @Gp2L2[%arg17, %arg16] (%arg24[%c0_24, %c0_24, %c0_24, %c0_24] [%c8_21, %c8_21, %c8_21, %c8_21] [%c64_20, %c8_21, %c512_19, %c1_23]) : (memref<64x64xbf16, 2 : i32>)
          %31 = memref.load %arg28[%c2_22] : memref<3xi32, 2 : i32>
          %32 = arith.addi %31, %c1_i32 : i32
          %33 = arith.cmpi sge, %32, %c1_i32 : i32
          scf.if %33 {
            %35 = memref.load %arg28[%c0_24] : memref<3xi32, 2 : i32>
            %36 = arith.addi %35, %c4_i32 : i32
            memref.store %36, %arg28[%c0_24] : memref<3xi32, 2 : i32>
            memref.store %c0_i32, %arg28[%c2_22] : memref<3xi32, 2 : i32>
          }
          %34 = arith.cmpi slt, %32, %c1_i32 : i32
          scf.if %34 {
            memref.store %32, %arg28[%c2_22] : memref<3xi32, 2 : i32>
          }
        }
        scf.forall (%arg16) in (4) {
          %21 = affine.apply #map11()[%arg16]
          air.channel.get  @Gp2L2[%c0_7, %arg16] (%alloc_10[%21, %c0_7] [%c64_3, %c64_3] [%c64_3, %c1_5]) : (memref<256x64xbf16, 1 : i32>)
        }
        air.channel.put  @GpOut[%arg12] (%alloc_10[] [] []) : (memref<256x64xbf16, 1 : i32>)
        scf.forall (%arg16) in (4) {
          %21 = affine.apply #map11()[%arg16]
          air.channel.get  @Gp2L2[%c1_5, %arg16] (%alloc_10[%21, %c0_7] [%c64_3, %c64_3] [%c64_3, %c1_5]) : (memref<256x64xbf16, 1 : i32>)
        }
        air.channel.put  @GpOut[%arg12] (%alloc_10[] [] []) : (memref<256x64xbf16, 1 : i32>)
        scf.forall (%arg16) in (4) {
          %21 = affine.apply #map11()[%arg16]
          air.channel.get  @Gp2L2[%c2_2, %arg16] (%alloc_10[%21, %c0_7] [%c64_3, %c64_3] [%c64_3, %c1_5]) : (memref<256x64xbf16, 1 : i32>)
        }
        air.channel.put  @GpOut[%arg12] (%alloc_10[] [] []) : (memref<256x64xbf16, 1 : i32>)
        scf.forall (%arg16) in (4) {
          %21 = affine.apply #map11()[%arg16]
          air.channel.get  @Gp2L2[%c3, %arg16] (%alloc_10[%21, %c0_7] [%c64_3, %c64_3] [%c64_3, %c1_5]) : (memref<256x64xbf16, 1 : i32>)
        }
        air.channel.put  @GpOut[%arg12] (%alloc_10[] [] []) : (memref<256x64xbf16, 1 : i32>)
        memref.dealloc %alloc_11 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_12 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_13 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_14 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_15 : memref<64x64xbf16, 2 : i32>
        memref.dealloc %alloc_16 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc_17 : memref<64x1xbf16, 2 : i32>
        memref.dealloc %alloc_9 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_10 : memref<256x64xbf16, 1 : i32>
        memref.dealloc %alloc_18 : memref<3xi32, 2 : i32>
      }
      %13 = affine.apply #map2()[%arg5]
      air.channel.get  @GpOut[%c0] (%arg11[%0, %13] [%c256, %c64] [%c512, %c1_1]) : (memref<512x512xbf16>)
      %14 = affine.apply #map3()[%arg5]
      air.channel.get  @GpOut[%c0] (%arg11[%0, %14] [%c256, %c64] [%c512, %c1_1]) : (memref<512x512xbf16>)
      %15 = affine.apply #map4()[%arg5]
      air.channel.get  @GpOut[%c0] (%arg11[%0, %15] [%c256, %c64] [%c512, %c1_1]) : (memref<512x512xbf16>)
      %16 = affine.apply #map5()[%arg5]
      air.channel.get  @GpOut[%c0] (%arg11[%0, %16] [%c256, %c64] [%c512, %c1_1]) : (memref<512x512xbf16>)
      %17 = affine.apply #map7()[%arg5]
      air.channel.get  @GpOut[%c1_1] (%arg11[%0, %17] [%c256, %c64] [%c512, %c1_1]) : (memref<512x512xbf16>)
      %18 = affine.apply #map8()[%arg5]
      air.channel.get  @GpOut[%c1_1] (%arg11[%0, %18] [%c256, %c64] [%c512, %c1_1]) : (memref<512x512xbf16>)
      %19 = affine.apply #map9()[%arg5]
      air.channel.get  @GpOut[%c1_1] (%arg11[%0, %19] [%c256, %c64] [%c512, %c1_1]) : (memref<512x512xbf16>)
      %20 = affine.apply #map10()[%arg5]
      air.channel.get  @GpOut[%c1_1] (%arg11[%0, %20] [%c256, %c64] [%c512, %c1_1]) : (memref<512x512xbf16>)
    }
    return
  }
}
