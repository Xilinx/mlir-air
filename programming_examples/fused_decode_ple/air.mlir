module {
  func.func private @proj_qmm_zero(memref<32xf32, 2 : i32>, i32) attributes {link_with = "proj_qmm.o"}
  func.func private @proj_qmm_acc256(memref<256xbf16, 2 : i32>, memref<2560xbf16, 2 : i32>, memref<32xf32, 2 : i32>) attributes {link_with = "proj_qmm.o"}
  func.func private @proj_qmm_acc256_c(memref<256xbf16, 2 : i32>, memref<2560xbf16, 2 : i32>, memref<32xf32, 2 : i32>, memref<384xbf16, 2 : i32>, i32, i32) attributes {link_with = "proj_qmm.o"}
  func.func private @proj_qmm_rc_arm(memref<384xbf16, 2 : i32>, i32) attributes {link_with = "proj_qmm.o"}
  func.func private @proj_qmm_flush_row(memref<32xf32, 2 : i32>, memref<48xbf16, 2 : i32>, i32) attributes {link_with = "proj_qmm.o"}
  func.func private @rms_norm_aie(memref<1536xbf16, 2 : i32>, memref<1536xbf16, 2 : i32>, memref<1536xbf16, 2 : i32>, i32) attributes {link_with = "rms_residual.o"}
  func.func private @rms_norm_lo_aie(memref<1536xbf16, 2 : i32>, memref<1536xbf16, 2 : i32>, memref<3072xbf16, 2 : i32>, i32) attributes {link_with = "rms_residual.o"}
  func.func private @rms_norm_hi_aie(memref<1536xbf16, 2 : i32>, memref<1536xbf16, 2 : i32>, memref<3072xbf16, 2 : i32>, i32) attributes {link_with = "rms_residual.o"}
  func.func private @residual_add_aie(memref<1536xbf16, 2 : i32>, memref<1536xbf16, 2 : i32>, memref<1536xbf16, 2 : i32>) attributes {link_with = "rms_residual.o"}
  func.func private @glu_aie(memref<512xbf16, 2 : i32>, memref<1024xbf16, 2 : i32>, i32) attributes {link_with = "glu.o"}
  func.func private @rope_compute(memref<4096xbf16, 2 : i32>, memref<1024xbf16, 2 : i32>, memref<1024xbf16, 2 : i32>, memref<6144xbf16, 2 : i32>, memref<1536xbf16, 2 : i32>, i32) attributes {link_with = "rope.o"}
  func.func private @attn_qk_blk(memref<2048xbf16, 2 : i32>, memref<8192xbf16, 2 : i32>, memref<16xbf16, 2 : i32>, memref<8xf32, 2 : i32>, memref<128xbf16, 2 : i32>, i32, i32) attributes {link_with = "attn_qk.ll", link_with_mode = "merge"}
  func.func private @attn_kv_blk(memref<128xbf16, 2 : i32>, memref<8192xbf16, 2 : i32>, memref<2048xf32, 2 : i32>, memref<16xf32, 2 : i32>, i32, i32) attributes {link_with = "attn_kv.ll", link_with_mode = "merge"}
  func.func private @attn_kv_fin(memref<2048xf32, 2 : i32>, memref<16xf32, 2 : i32>, memref<2048xbf16, 2 : i32>) attributes {link_with = "attn_kv.ll", link_with_mode = "merge"}
  air.channel @rmsX [1] {channel_type = "npu_dma_packet"}
  air.channel @rmsW [1]
  air.channel @rmsW2 [1]
  air.channel @xnorm [1] {channel_type = "npu_dma_packet"}
  air.channel @inX [1, 1] {air.shared_resident_ring, broadcast_shape = [4 : index, 4 : index]}
  air.channel @ropeLUT [1]
  air.channel @ropeQ [1]
  air.channel @toAttnQ [2]
  air.channel @toAttnKV [1] {channel_type = "npu_dma_packet"}
  air.channel @toK [2]
  air.channel @toV [2]
  air.channel @appendK [1] {channel_type = "npu_dma_packet"}
  air.channel @appendV [1] {channel_type = "npu_dma_packet"}
  air.channel @inKV_K [1]
  air.channel @inKV_V [1]
  air.channel @attnO [2]
  air.channel @inW0c0 [1]
  air.channel @inW1c0 [1]
  air.channel @inW0c1 [1]
  air.channel @inW1c1 [1]
  air.channel @inW0c2 [1]
  air.channel @inW1c2 [1]
  air.channel @inW0c3 [1]
  air.channel @inW1c3 [1]
  air.channel @wL2ToL1 [4, 4] {air.shared_resident_ring}
  air.channel @outA [4, 4] {channel_type = "npu_dma_packet"}
  air.channel @toMain [4] {channel_type = "npu_dma_packet"}
  air.channel @outY [1, 1] {broadcast_shape = [1 : index, 3 : index], channel_type = "npu_dma_packet"}
  air.channel @toShim [3]
  air.channel @layerOut [1]
  air.channel @gluOut [1]
  func.func @q4nx_decode(%arg0: memref<1536xbf16>, %arg1: memref<193904640xbf16>, %arg2: memref<29184xbf16>, %arg3: memref<270336xbf16>, %arg4: memref<98304xbf16>) {
    %c0 = arith.constant 0 : index
    %c22 = arith.constant 22 : index
    %c1 = arith.constant 1 : index
    scf.for %arg5 = %c0 to %c22 step %c1 {
      %c1_0 = arith.constant 1 : index
      %c1_1 = arith.constant 1 : index
      air.launch (%arg6, %arg7) in (%arg8=%c1_0, %arg9=%c1_1) args(%arg10=%arg0, %arg11=%arg1, %arg12=%arg2, %arg13=%arg3, %arg14=%arg4, %arg15=%arg5) : memref<1536xbf16>, memref<193904640xbf16>, memref<29184xbf16>, memref<270336xbf16>, memref<98304xbf16>, index attributes {air.preserve_shim_dma_order} {
        %c22609920 = arith.constant 22609920 : index
        %0 = arith.muli %arg15, %c22609920 : index
        %c7680 = arith.constant 7680 : index
        %1 = arith.muli %arg15, %c7680 : index
        %c32768 = arith.constant 32768 : index
        %2 = arith.muli %arg15, %c32768 : index
        %c0_2 = arith.constant 0 : index
        %3 = arith.muli %arg15, %c0_2 : index
        %c1_i32 = arith.constant 1 : i32
        %c0_i32 = arith.constant 0 : i32
        %c3 = arith.constant 3 : index
        %4 = arith.cmpi slt, %arg15, %c3 : index
        %5 = arith.select %4, %c1_i32, %c0_i32 : i32
        %6 = arith.index_cast %5 : i32 to index
        scf.index_switch %6 
        case 0 {
          %c67829760 = arith.constant 67829760 : index
          %c3_3 = arith.constant 3 : index
          %7 = arith.subi %arg15, %c3_3 : index
          %c6635520 = arith.constant 6635520 : index
          %8 = arith.muli %7, %c6635520 : index
          %9 = arith.addi %c67829760, %8 : index
          %c7680_4 = arith.constant 7680 : index
          %c3_5 = arith.constant 3 : index
          %10 = arith.subi %arg15, %c3_5 : index
          %c13824 = arith.constant 13824 : index
          %11 = arith.muli %10, %c13824 : index
          %12 = arith.addi %c7680_4, %11 : index
          air.channel.put  @rmsX[] (%arg10[0] [1536] [1]) : (memref<1536xbf16>)
          air.channel.put  @rmsW[] (%arg12[26112] [3072] [1]) : (memref<29184xbf16>)
          air.channel.put  @rmsW2[] (%arg12[0] [3072] [1]) : (memref<29184xbf16>)
          %c0_6 = arith.constant 0 : index
          %13 = arith.addi %9, %c0_6 : index
          %c0_7 = arith.constant 0 : index
          air.channel.put  @inW0c0[%c0_7] (%arg11[%13] [414720] [1]) : (memref<193904640xbf16>)
          %c414720 = arith.constant 414720 : index
          %14 = arith.addi %13, %c414720 : index
          air.channel.put  @inW0c0[%c0_7] (%arg11[%14] [414720] [1]) : (memref<193904640xbf16>)
          %c829440 = arith.constant 829440 : index
          %15 = arith.addi %13, %c829440 : index
          %c0_8 = arith.constant 0 : index
          air.channel.put  @inW1c0[%c0_8] (%arg11[%15] [414720] [1]) : (memref<193904640xbf16>)
          %c414720_9 = arith.constant 414720 : index
          %16 = arith.addi %15, %c414720_9 : index
          air.channel.put  @inW1c0[%c0_8] (%arg11[%16] [414720] [1]) : (memref<193904640xbf16>)
          %c1658880 = arith.constant 1658880 : index
          %17 = arith.addi %9, %c1658880 : index
          %c0_10 = arith.constant 0 : index
          air.channel.put  @inW0c1[%c0_10] (%arg11[%17] [414720] [1]) : (memref<193904640xbf16>)
          %c414720_11 = arith.constant 414720 : index
          %18 = arith.addi %17, %c414720_11 : index
          air.channel.put  @inW0c1[%c0_10] (%arg11[%18] [414720] [1]) : (memref<193904640xbf16>)
          %c829440_12 = arith.constant 829440 : index
          %19 = arith.addi %17, %c829440_12 : index
          %c0_13 = arith.constant 0 : index
          air.channel.put  @inW1c1[%c0_13] (%arg11[%19] [414720] [1]) : (memref<193904640xbf16>)
          %c414720_14 = arith.constant 414720 : index
          %20 = arith.addi %19, %c414720_14 : index
          air.channel.put  @inW1c1[%c0_13] (%arg11[%20] [414720] [1]) : (memref<193904640xbf16>)
          %c3317760 = arith.constant 3317760 : index
          %21 = arith.addi %9, %c3317760 : index
          %c0_15 = arith.constant 0 : index
          air.channel.put  @inW0c2[%c0_15] (%arg11[%21] [414720] [1]) : (memref<193904640xbf16>)
          %c414720_16 = arith.constant 414720 : index
          %22 = arith.addi %21, %c414720_16 : index
          air.channel.put  @inW0c2[%c0_15] (%arg11[%22] [414720] [1]) : (memref<193904640xbf16>)
          %c829440_17 = arith.constant 829440 : index
          %23 = arith.addi %21, %c829440_17 : index
          %c0_18 = arith.constant 0 : index
          air.channel.put  @inW1c2[%c0_18] (%arg11[%23] [414720] [1]) : (memref<193904640xbf16>)
          %c414720_19 = arith.constant 414720 : index
          %24 = arith.addi %23, %c414720_19 : index
          air.channel.put  @inW1c2[%c0_18] (%arg11[%24] [414720] [1]) : (memref<193904640xbf16>)
          %c4976640 = arith.constant 4976640 : index
          %25 = arith.addi %9, %c4976640 : index
          %c0_20 = arith.constant 0 : index
          air.channel.put  @inW0c3[%c0_20] (%arg11[%25] [414720] [1]) : (memref<193904640xbf16>)
          %c414720_21 = arith.constant 414720 : index
          %26 = arith.addi %25, %c414720_21 : index
          air.channel.put  @inW0c3[%c0_20] (%arg11[%26] [414720] [1]) : (memref<193904640xbf16>)
          %c829440_22 = arith.constant 829440 : index
          %27 = arith.addi %25, %c829440_22 : index
          %c0_23 = arith.constant 0 : index
          air.channel.put  @inW1c3[%c0_23] (%arg11[%27] [414720] [1]) : (memref<193904640xbf16>)
          %c414720_24 = arith.constant 414720 : index
          %28 = arith.addi %27, %c414720_24 : index
          air.channel.put  @inW1c3[%c0_23] (%arg11[%28] [414720] [1]) : (memref<193904640xbf16>)
          %c0_25 = arith.constant 0 : index
          air.channel.get  @layerOut[%c0_25] (%arg13[%12] [27, 512] [512, 1]) : (memref<270336xbf16>)
          scf.yield
        }
        default {
          air.channel.put  @rmsX[] (%arg10[0] [1536] [1]) : (memref<1536xbf16>)
          air.channel.put  @rmsW[] (%arg12[%1] [3072] [1]) : (memref<29184xbf16>)
          %c3072 = arith.constant 3072 : index
          %7 = arith.addi %1, %c3072 : index
          air.channel.put  @rmsW2[] (%arg12[%7] [3072] [1]) : (memref<29184xbf16>)
          %c1536 = arith.constant 1536 : index
          %8 = arith.muli %arg15, %c1536 : index
          %c23040 = arith.constant 23040 : index
          %9 = arith.addi %8, %c23040 : index
          air.channel.put  @ropeLUT[] (%arg12[%9] [1536] [1]) : (memref<29184xbf16>)
          %c0_3 = arith.constant 0 : index
          %10 = arith.addi %0, %c0_3 : index
          %c0_4 = arith.constant 0 : index
          air.channel.put  @inW0c0[%c0_4] (%arg11[%10] [184320] [1]) : (memref<193904640xbf16>)
          %c184320 = arith.constant 184320 : index
          %11 = arith.addi %10, %c184320 : index
          air.channel.put  @inW0c0[%c0_4] (%arg11[%11] [184320] [1]) : (memref<193904640xbf16>)
          %c368640 = arith.constant 368640 : index
          %12 = arith.addi %10, %c368640 : index
          %c0_5 = arith.constant 0 : index
          air.channel.put  @inW1c0[%c0_5] (%arg11[%12] [184320] [1]) : (memref<193904640xbf16>)
          %c184320_6 = arith.constant 184320 : index
          %13 = arith.addi %12, %c184320_6 : index
          air.channel.put  @inW1c0[%c0_5] (%arg11[%13] [184320] [1]) : (memref<193904640xbf16>)
          %c737280 = arith.constant 737280 : index
          %14 = arith.addi %0, %c737280 : index
          %c0_7 = arith.constant 0 : index
          air.channel.put  @inW0c1[%c0_7] (%arg11[%14] [184320] [1]) : (memref<193904640xbf16>)
          %c184320_8 = arith.constant 184320 : index
          %15 = arith.addi %14, %c184320_8 : index
          air.channel.put  @inW0c1[%c0_7] (%arg11[%15] [184320] [1]) : (memref<193904640xbf16>)
          %c368640_9 = arith.constant 368640 : index
          %16 = arith.addi %14, %c368640_9 : index
          %c0_10 = arith.constant 0 : index
          air.channel.put  @inW1c1[%c0_10] (%arg11[%16] [184320] [1]) : (memref<193904640xbf16>)
          %c184320_11 = arith.constant 184320 : index
          %17 = arith.addi %16, %c184320_11 : index
          air.channel.put  @inW1c1[%c0_10] (%arg11[%17] [184320] [1]) : (memref<193904640xbf16>)
          %c1474560 = arith.constant 1474560 : index
          %18 = arith.addi %0, %c1474560 : index
          %c0_12 = arith.constant 0 : index
          air.channel.put  @inW0c2[%c0_12] (%arg11[%18] [184320] [1]) : (memref<193904640xbf16>)
          %c184320_13 = arith.constant 184320 : index
          %19 = arith.addi %18, %c184320_13 : index
          air.channel.put  @inW0c2[%c0_12] (%arg11[%19] [184320] [1]) : (memref<193904640xbf16>)
          %c368640_14 = arith.constant 368640 : index
          %20 = arith.addi %18, %c368640_14 : index
          %c0_15 = arith.constant 0 : index
          air.channel.put  @inW1c2[%c0_15] (%arg11[%20] [184320] [1]) : (memref<193904640xbf16>)
          %c184320_16 = arith.constant 184320 : index
          %21 = arith.addi %20, %c184320_16 : index
          air.channel.put  @inW1c2[%c0_15] (%arg11[%21] [184320] [1]) : (memref<193904640xbf16>)
          %c2211840 = arith.constant 2211840 : index
          %22 = arith.addi %0, %c2211840 : index
          %c0_17 = arith.constant 0 : index
          air.channel.put  @inW0c3[%c0_17] (%arg11[%22] [184320] [1]) : (memref<193904640xbf16>)
          %c184320_18 = arith.constant 184320 : index
          %23 = arith.addi %22, %c184320_18 : index
          air.channel.put  @inW0c3[%c0_17] (%arg11[%23] [184320] [1]) : (memref<193904640xbf16>)
          %c368640_19 = arith.constant 368640 : index
          %24 = arith.addi %22, %c368640_19 : index
          %c0_20 = arith.constant 0 : index
          air.channel.put  @inW1c3[%c0_20] (%arg11[%24] [184320] [1]) : (memref<193904640xbf16>)
          %c184320_21 = arith.constant 184320 : index
          %25 = arith.addi %24, %c184320_21 : index
          air.channel.put  @inW1c3[%c0_20] (%arg11[%25] [184320] [1]) : (memref<193904640xbf16>)
          %c0_22 = arith.constant 0 : index
          %c15360 = arith.constant 15360 : index
          %26 = arith.addi %2, %c15360 : index
          %c1_23 = arith.constant 1 : index
          %c1024 = arith.constant 1024 : index
          %c16384 = arith.constant 16384 : index
          %c1_24 = arith.constant 1 : index
          air.channel.get  @appendK[%c0_22] (%arg14[%26] [%c1_23, %c1024] [%c16384, %c1_24]) : (memref<98304xbf16>)
          %c0_25 = arith.constant 0 : index
          %c31744 = arith.constant 31744 : index
          %27 = arith.addi %2, %c31744 : index
          %c1_26 = arith.constant 1 : index
          %c1024_27 = arith.constant 1024 : index
          %c16384_28 = arith.constant 16384 : index
          %c1_29 = arith.constant 1 : index
          air.channel.get  @appendV[%c0_25] (%arg14[%27] [%c1_26, %c1024_27] [%c16384_28, %c1_29]) : (memref<98304xbf16>)
          %c0_30 = arith.constant 0 : index
          %c1_31 = arith.constant 1 : index
          %c16 = arith.constant 16 : index
          %c1024_32 = arith.constant 1024 : index
          %c16384_33 = arith.constant 16384 : index
          %c1024_34 = arith.constant 1024 : index
          %c1_35 = arith.constant 1 : index
          air.channel.put  @inKV_K[%c0_30] (%arg14[%2] [%c1_31, %c16, %c1024_32] [%c16384_33, %c1024_34, %c1_35]) : (memref<98304xbf16>)
          %c0_36 = arith.constant 0 : index
          %c16384_37 = arith.constant 16384 : index
          %28 = arith.addi %2, %c16384_37 : index
          %c1_38 = arith.constant 1 : index
          %c16_39 = arith.constant 16 : index
          %c1024_40 = arith.constant 1024 : index
          %c16384_41 = arith.constant 16384 : index
          %c1024_42 = arith.constant 1024 : index
          %c1_43 = arith.constant 1 : index
          air.channel.put  @inKV_V[%c0_36] (%arg14[%28] [%c1_38, %c16_39, %c1024_40] [%c16384_41, %c1024_42, %c1_43]) : (memref<98304xbf16>)
          %c2949120 = arith.constant 2949120 : index
          %29 = arith.addi %0, %c2949120 : index
          %c0_44 = arith.constant 0 : index
          %30 = arith.addi %29, %c0_44 : index
          %c0_45 = arith.constant 0 : index
          air.channel.put  @inW0c0[%c0_45] (%arg11[%30] [122880] [1]) : (memref<193904640xbf16>)
          %c122880 = arith.constant 122880 : index
          %31 = arith.addi %30, %c122880 : index
          air.channel.put  @inW0c0[%c0_45] (%arg11[%31] [122880] [1]) : (memref<193904640xbf16>)
          %c245760 = arith.constant 245760 : index
          %32 = arith.addi %30, %c245760 : index
          %c0_46 = arith.constant 0 : index
          air.channel.put  @inW1c0[%c0_46] (%arg11[%32] [122880] [1]) : (memref<193904640xbf16>)
          %c122880_47 = arith.constant 122880 : index
          %33 = arith.addi %32, %c122880_47 : index
          air.channel.put  @inW1c0[%c0_46] (%arg11[%33] [122880] [1]) : (memref<193904640xbf16>)
          %c491520 = arith.constant 491520 : index
          %34 = arith.addi %29, %c491520 : index
          %c0_48 = arith.constant 0 : index
          air.channel.put  @inW0c1[%c0_48] (%arg11[%34] [122880] [1]) : (memref<193904640xbf16>)
          %c122880_49 = arith.constant 122880 : index
          %35 = arith.addi %34, %c122880_49 : index
          air.channel.put  @inW0c1[%c0_48] (%arg11[%35] [122880] [1]) : (memref<193904640xbf16>)
          %c245760_50 = arith.constant 245760 : index
          %36 = arith.addi %34, %c245760_50 : index
          %c0_51 = arith.constant 0 : index
          air.channel.put  @inW1c1[%c0_51] (%arg11[%36] [122880] [1]) : (memref<193904640xbf16>)
          %c122880_52 = arith.constant 122880 : index
          %37 = arith.addi %36, %c122880_52 : index
          air.channel.put  @inW1c1[%c0_51] (%arg11[%37] [122880] [1]) : (memref<193904640xbf16>)
          %c983040 = arith.constant 983040 : index
          %38 = arith.addi %29, %c983040 : index
          %c0_53 = arith.constant 0 : index
          air.channel.put  @inW0c2[%c0_53] (%arg11[%38] [122880] [1]) : (memref<193904640xbf16>)
          %c122880_54 = arith.constant 122880 : index
          %39 = arith.addi %38, %c122880_54 : index
          air.channel.put  @inW0c2[%c0_53] (%arg11[%39] [122880] [1]) : (memref<193904640xbf16>)
          %c245760_55 = arith.constant 245760 : index
          %40 = arith.addi %38, %c245760_55 : index
          %c0_56 = arith.constant 0 : index
          air.channel.put  @inW1c2[%c0_56] (%arg11[%40] [122880] [1]) : (memref<193904640xbf16>)
          %c122880_57 = arith.constant 122880 : index
          %41 = arith.addi %40, %c122880_57 : index
          air.channel.put  @inW1c2[%c0_56] (%arg11[%41] [122880] [1]) : (memref<193904640xbf16>)
          %c1474560_58 = arith.constant 1474560 : index
          %42 = arith.addi %29, %c1474560_58 : index
          %c0_59 = arith.constant 0 : index
          air.channel.put  @inW0c3[%c0_59] (%arg11[%42] [122880] [1]) : (memref<193904640xbf16>)
          %c122880_60 = arith.constant 122880 : index
          %43 = arith.addi %42, %c122880_60 : index
          air.channel.put  @inW0c3[%c0_59] (%arg11[%43] [122880] [1]) : (memref<193904640xbf16>)
          %c245760_61 = arith.constant 245760 : index
          %44 = arith.addi %42, %c245760_61 : index
          %c0_62 = arith.constant 0 : index
          air.channel.put  @inW1c3[%c0_62] (%arg11[%44] [122880] [1]) : (memref<193904640xbf16>)
          %c122880_63 = arith.constant 122880 : index
          %45 = arith.addi %44, %c122880_63 : index
          air.channel.put  @inW1c3[%c0_62] (%arg11[%45] [122880] [1]) : (memref<193904640xbf16>)
          %c4915200 = arith.constant 4915200 : index
          %46 = arith.addi %0, %c4915200 : index
          %c0_64 = arith.constant 0 : index
          %47 = arith.addi %46, %c0_64 : index
          %c0_65 = arith.constant 0 : index
          air.channel.put  @inW0c0[%c0_65] (%arg11[%47] [737280] [1]) : (memref<193904640xbf16>)
          %c737280_66 = arith.constant 737280 : index
          %48 = arith.addi %47, %c737280_66 : index
          air.channel.put  @inW0c0[%c0_65] (%arg11[%48] [737280] [1]) : (memref<193904640xbf16>)
          %c1474560_67 = arith.constant 1474560 : index
          %49 = arith.addi %47, %c1474560_67 : index
          %c0_68 = arith.constant 0 : index
          air.channel.put  @inW1c0[%c0_68] (%arg11[%49] [737280] [1]) : (memref<193904640xbf16>)
          %c737280_69 = arith.constant 737280 : index
          %50 = arith.addi %49, %c737280_69 : index
          air.channel.put  @inW1c0[%c0_68] (%arg11[%50] [737280] [1]) : (memref<193904640xbf16>)
          %c2949120_70 = arith.constant 2949120 : index
          %51 = arith.addi %46, %c2949120_70 : index
          %c0_71 = arith.constant 0 : index
          air.channel.put  @inW0c1[%c0_71] (%arg11[%51] [737280] [1]) : (memref<193904640xbf16>)
          %c737280_72 = arith.constant 737280 : index
          %52 = arith.addi %51, %c737280_72 : index
          air.channel.put  @inW0c1[%c0_71] (%arg11[%52] [737280] [1]) : (memref<193904640xbf16>)
          %c1474560_73 = arith.constant 1474560 : index
          %53 = arith.addi %51, %c1474560_73 : index
          %c0_74 = arith.constant 0 : index
          air.channel.put  @inW1c1[%c0_74] (%arg11[%53] [737280] [1]) : (memref<193904640xbf16>)
          %c737280_75 = arith.constant 737280 : index
          %54 = arith.addi %53, %c737280_75 : index
          air.channel.put  @inW1c1[%c0_74] (%arg11[%54] [737280] [1]) : (memref<193904640xbf16>)
          %c5898240 = arith.constant 5898240 : index
          %55 = arith.addi %46, %c5898240 : index
          %c0_76 = arith.constant 0 : index
          air.channel.put  @inW0c2[%c0_76] (%arg11[%55] [737280] [1]) : (memref<193904640xbf16>)
          %c737280_77 = arith.constant 737280 : index
          %56 = arith.addi %55, %c737280_77 : index
          air.channel.put  @inW0c2[%c0_76] (%arg11[%56] [737280] [1]) : (memref<193904640xbf16>)
          %c1474560_78 = arith.constant 1474560 : index
          %57 = arith.addi %55, %c1474560_78 : index
          %c0_79 = arith.constant 0 : index
          air.channel.put  @inW1c2[%c0_79] (%arg11[%57] [737280] [1]) : (memref<193904640xbf16>)
          %c737280_80 = arith.constant 737280 : index
          %58 = arith.addi %57, %c737280_80 : index
          air.channel.put  @inW1c2[%c0_79] (%arg11[%58] [737280] [1]) : (memref<193904640xbf16>)
          %c8847360 = arith.constant 8847360 : index
          %59 = arith.addi %46, %c8847360 : index
          %c0_81 = arith.constant 0 : index
          air.channel.put  @inW0c3[%c0_81] (%arg11[%59] [737280] [1]) : (memref<193904640xbf16>)
          %c737280_82 = arith.constant 737280 : index
          %60 = arith.addi %59, %c737280_82 : index
          air.channel.put  @inW0c3[%c0_81] (%arg11[%60] [737280] [1]) : (memref<193904640xbf16>)
          %c1474560_83 = arith.constant 1474560 : index
          %61 = arith.addi %59, %c1474560_83 : index
          %c0_84 = arith.constant 0 : index
          air.channel.put  @inW1c3[%c0_84] (%arg11[%61] [737280] [1]) : (memref<193904640xbf16>)
          %c737280_85 = arith.constant 737280 : index
          %62 = arith.addi %61, %c737280_85 : index
          air.channel.put  @inW1c3[%c0_84] (%arg11[%62] [737280] [1]) : (memref<193904640xbf16>)
          %c16711680 = arith.constant 16711680 : index
          %63 = arith.addi %0, %c16711680 : index
          %c0_86 = arith.constant 0 : index
          %64 = arith.addi %63, %c0_86 : index
          %c0_87 = arith.constant 0 : index
          air.channel.put  @inW0c0[%c0_87] (%arg11[%64] [368640] [1]) : (memref<193904640xbf16>)
          %c368640_88 = arith.constant 368640 : index
          %65 = arith.addi %64, %c368640_88 : index
          air.channel.put  @inW0c0[%c0_87] (%arg11[%65] [368640] [1]) : (memref<193904640xbf16>)
          %c737280_89 = arith.constant 737280 : index
          %66 = arith.addi %64, %c737280_89 : index
          %c0_90 = arith.constant 0 : index
          air.channel.put  @inW1c0[%c0_90] (%arg11[%66] [368640] [1]) : (memref<193904640xbf16>)
          %c368640_91 = arith.constant 368640 : index
          %67 = arith.addi %66, %c368640_91 : index
          air.channel.put  @inW1c0[%c0_90] (%arg11[%67] [368640] [1]) : (memref<193904640xbf16>)
          %c1474560_92 = arith.constant 1474560 : index
          %68 = arith.addi %63, %c1474560_92 : index
          %c0_93 = arith.constant 0 : index
          air.channel.put  @inW0c1[%c0_93] (%arg11[%68] [368640] [1]) : (memref<193904640xbf16>)
          %c368640_94 = arith.constant 368640 : index
          %69 = arith.addi %68, %c368640_94 : index
          air.channel.put  @inW0c1[%c0_93] (%arg11[%69] [368640] [1]) : (memref<193904640xbf16>)
          %c737280_95 = arith.constant 737280 : index
          %70 = arith.addi %68, %c737280_95 : index
          %c0_96 = arith.constant 0 : index
          air.channel.put  @inW1c1[%c0_96] (%arg11[%70] [368640] [1]) : (memref<193904640xbf16>)
          %c368640_97 = arith.constant 368640 : index
          %71 = arith.addi %70, %c368640_97 : index
          air.channel.put  @inW1c1[%c0_96] (%arg11[%71] [368640] [1]) : (memref<193904640xbf16>)
          %c2949120_98 = arith.constant 2949120 : index
          %72 = arith.addi %63, %c2949120_98 : index
          %c0_99 = arith.constant 0 : index
          air.channel.put  @inW0c2[%c0_99] (%arg11[%72] [368640] [1]) : (memref<193904640xbf16>)
          %c368640_100 = arith.constant 368640 : index
          %73 = arith.addi %72, %c368640_100 : index
          air.channel.put  @inW0c2[%c0_99] (%arg11[%73] [368640] [1]) : (memref<193904640xbf16>)
          %c737280_101 = arith.constant 737280 : index
          %74 = arith.addi %72, %c737280_101 : index
          %c0_102 = arith.constant 0 : index
          air.channel.put  @inW1c2[%c0_102] (%arg11[%74] [368640] [1]) : (memref<193904640xbf16>)
          %c368640_103 = arith.constant 368640 : index
          %75 = arith.addi %74, %c368640_103 : index
          air.channel.put  @inW1c2[%c0_102] (%arg11[%75] [368640] [1]) : (memref<193904640xbf16>)
          %c4423680 = arith.constant 4423680 : index
          %76 = arith.addi %63, %c4423680 : index
          %c0_104 = arith.constant 0 : index
          air.channel.put  @inW0c3[%c0_104] (%arg11[%76] [368640] [1]) : (memref<193904640xbf16>)
          %c368640_105 = arith.constant 368640 : index
          %77 = arith.addi %76, %c368640_105 : index
          air.channel.put  @inW0c3[%c0_104] (%arg11[%77] [368640] [1]) : (memref<193904640xbf16>)
          %c737280_106 = arith.constant 737280 : index
          %78 = arith.addi %76, %c737280_106 : index
          %c0_107 = arith.constant 0 : index
          air.channel.put  @inW1c3[%c0_107] (%arg11[%78] [368640] [1]) : (memref<193904640xbf16>)
          %c368640_108 = arith.constant 368640 : index
          %79 = arith.addi %78, %c368640_108 : index
          air.channel.put  @inW1c3[%c0_107] (%arg11[%79] [368640] [1]) : (memref<193904640xbf16>)
          %c0_109 = arith.constant 0 : index
          air.channel.get  @layerOut[%c0_109] (%arg10[0] [1536] [1]) : (memref<1536xbf16>)
        }
        air.segment @seg  args(%arg16=%arg15) : index {
          %c3_3 = arith.constant 3 : index
          %7 = arith.cmpi slt, %arg16, %c3_3 : index
          %c1_i32_4 = arith.constant 1 : i32
          %c0_i32_5 = arith.constant 0 : i32
          %8 = arith.select %7, %c1_i32_4, %c0_i32_5 : i32
          %9 = arith.index_cast %8 : i32 to index
          scf.index_switch %9 
          case 0 {
            %c0_47 = arith.constant 0 : index
            %c81 = arith.constant 81 : index
            %c1_48 = arith.constant 1 : index
            scf.for %arg17 = %c0_47 to %c81 step %c1_48 {
              %alloc_49 = memref.alloc() {air.memtile_col = 2 : i32} : memref<512xbf16, 1 : i32>
              air.channel.get  @xnorm[] (%alloc_49[0] [512] [1]) : (memref<512xbf16, 1 : i32>)
              %c0_50 = arith.constant 0 : index
              %c2_51 = arith.constant 2 : index
              %c1_52 = arith.constant 1 : index
              scf.for %arg18 = %c0_50 to %c2_51 step %c1_52 {
                %c256 = arith.constant 256 : index
                %10 = arith.muli %arg18, %c256 : index
                air.channel.put  @inX[] (%alloc_49[%10] [256] [1]) : (memref<512xbf16, 1 : i32>)
              }
              memref.dealloc %alloc_49 : memref<512xbf16, 1 : i32>
            }
            scf.yield
          }
          default {
            %c0_47 = arith.constant 0 : index
            %c261 = arith.constant 261 : index
            %c1_48 = arith.constant 1 : index
            scf.for %arg17 = %c0_47 to %c261 step %c1_48 {
              %alloc_49 = memref.alloc() {air.memtile_col = 2 : i32} : memref<512xbf16, 1 : i32>
              air.channel.get  @xnorm[] (%alloc_49[0] [512] [1]) : (memref<512xbf16, 1 : i32>)
              %c0_50 = arith.constant 0 : index
              %c2_51 = arith.constant 2 : index
              %c1_52 = arith.constant 1 : index
              scf.for %arg18 = %c0_50 to %c2_51 step %c1_52 {
                %c256 = arith.constant 256 : index
                %10 = arith.muli %arg18, %c256 : index
                air.channel.put  @inX[] (%alloc_49[%10] [256] [1]) : (memref<512xbf16, 1 : i32>)
              }
              memref.dealloc %alloc_49 : memref<512xbf16, 1 : i32>
            }
          }
          %c0_6 = arith.constant 0 : index
          %c552 = arith.constant 552 : index
          %c1_7 = arith.constant 1 : index
          scf.for %arg17 = %c0_6 to %c552 step %c1_7 {
            %alloc_47 = memref.alloc() {air.memtile_col = 0 : i32} : memref<5120xbf16, 1 : i32>
            %c0_48 = arith.constant 0 : index
            air.channel.get  @inW0c0[%c0_48] (%alloc_47[] [] []) : (memref<5120xbf16, 1 : i32>)
            %c0_49 = arith.constant 0 : index
            %c0_50 = arith.constant 0 : index
            air.channel.put  @wL2ToL1[%c0_49, %c0_50] (%alloc_47[0] [2560] [1]) : (memref<5120xbf16, 1 : i32>)
            %c0_51 = arith.constant 0 : index
            %c1_52 = arith.constant 1 : index
            air.channel.put  @wL2ToL1[%c0_51, %c1_52] (%alloc_47[2560] [2560] [1]) : (memref<5120xbf16, 1 : i32>)
            memref.dealloc %alloc_47 : memref<5120xbf16, 1 : i32>
          }
          %c0_8 = arith.constant 0 : index
          %c552_9 = arith.constant 552 : index
          %c1_10 = arith.constant 1 : index
          scf.for %arg17 = %c0_8 to %c552_9 step %c1_10 {
            %alloc_47 = memref.alloc() {air.memtile_col = 0 : i32} : memref<5120xbf16, 1 : i32>
            %c0_48 = arith.constant 0 : index
            air.channel.get  @inW1c0[%c0_48] (%alloc_47[] [] []) : (memref<5120xbf16, 1 : i32>)
            %c0_49 = arith.constant 0 : index
            %c2_50 = arith.constant 2 : index
            air.channel.put  @wL2ToL1[%c0_49, %c2_50] (%alloc_47[0] [2560] [1]) : (memref<5120xbf16, 1 : i32>)
            %c0_51 = arith.constant 0 : index
            %c3_52 = arith.constant 3 : index
            air.channel.put  @wL2ToL1[%c0_51, %c3_52] (%alloc_47[2560] [2560] [1]) : (memref<5120xbf16, 1 : i32>)
            memref.dealloc %alloc_47 : memref<5120xbf16, 1 : i32>
          }
          %c0_11 = arith.constant 0 : index
          %c552_12 = arith.constant 552 : index
          %c1_13 = arith.constant 1 : index
          scf.for %arg17 = %c0_11 to %c552_12 step %c1_13 {
            %alloc_47 = memref.alloc() {air.memtile_col = 1 : i32} : memref<5120xbf16, 1 : i32>
            %c0_48 = arith.constant 0 : index
            air.channel.get  @inW0c1[%c0_48] (%alloc_47[] [] []) : (memref<5120xbf16, 1 : i32>)
            %c1_49 = arith.constant 1 : index
            %c0_50 = arith.constant 0 : index
            air.channel.put  @wL2ToL1[%c1_49, %c0_50] (%alloc_47[0] [2560] [1]) : (memref<5120xbf16, 1 : i32>)
            %c1_51 = arith.constant 1 : index
            %c1_52 = arith.constant 1 : index
            air.channel.put  @wL2ToL1[%c1_51, %c1_52] (%alloc_47[2560] [2560] [1]) : (memref<5120xbf16, 1 : i32>)
            memref.dealloc %alloc_47 : memref<5120xbf16, 1 : i32>
          }
          %c0_14 = arith.constant 0 : index
          %c552_15 = arith.constant 552 : index
          %c1_16 = arith.constant 1 : index
          scf.for %arg17 = %c0_14 to %c552_15 step %c1_16 {
            %alloc_47 = memref.alloc() {air.memtile_col = 1 : i32} : memref<5120xbf16, 1 : i32>
            %c0_48 = arith.constant 0 : index
            air.channel.get  @inW1c1[%c0_48] (%alloc_47[] [] []) : (memref<5120xbf16, 1 : i32>)
            %c1_49 = arith.constant 1 : index
            %c2_50 = arith.constant 2 : index
            air.channel.put  @wL2ToL1[%c1_49, %c2_50] (%alloc_47[0] [2560] [1]) : (memref<5120xbf16, 1 : i32>)
            %c1_51 = arith.constant 1 : index
            %c3_52 = arith.constant 3 : index
            air.channel.put  @wL2ToL1[%c1_51, %c3_52] (%alloc_47[2560] [2560] [1]) : (memref<5120xbf16, 1 : i32>)
            memref.dealloc %alloc_47 : memref<5120xbf16, 1 : i32>
          }
          %c0_17 = arith.constant 0 : index
          %c552_18 = arith.constant 552 : index
          %c1_19 = arith.constant 1 : index
          scf.for %arg17 = %c0_17 to %c552_18 step %c1_19 {
            %alloc_47 = memref.alloc() {air.memtile_col = 6 : i32} : memref<5120xbf16, 1 : i32>
            %c0_48 = arith.constant 0 : index
            air.channel.get  @inW0c2[%c0_48] (%alloc_47[] [] []) : (memref<5120xbf16, 1 : i32>)
            %c2_49 = arith.constant 2 : index
            %c0_50 = arith.constant 0 : index
            air.channel.put  @wL2ToL1[%c2_49, %c0_50] (%alloc_47[0] [2560] [1]) : (memref<5120xbf16, 1 : i32>)
            %c2_51 = arith.constant 2 : index
            %c1_52 = arith.constant 1 : index
            air.channel.put  @wL2ToL1[%c2_51, %c1_52] (%alloc_47[2560] [2560] [1]) : (memref<5120xbf16, 1 : i32>)
            memref.dealloc %alloc_47 : memref<5120xbf16, 1 : i32>
          }
          %c0_20 = arith.constant 0 : index
          %c552_21 = arith.constant 552 : index
          %c1_22 = arith.constant 1 : index
          scf.for %arg17 = %c0_20 to %c552_21 step %c1_22 {
            %alloc_47 = memref.alloc() {air.memtile_col = 6 : i32} : memref<5120xbf16, 1 : i32>
            %c0_48 = arith.constant 0 : index
            air.channel.get  @inW1c2[%c0_48] (%alloc_47[] [] []) : (memref<5120xbf16, 1 : i32>)
            %c2_49 = arith.constant 2 : index
            %c2_50 = arith.constant 2 : index
            air.channel.put  @wL2ToL1[%c2_49, %c2_50] (%alloc_47[0] [2560] [1]) : (memref<5120xbf16, 1 : i32>)
            %c2_51 = arith.constant 2 : index
            %c3_52 = arith.constant 3 : index
            air.channel.put  @wL2ToL1[%c2_51, %c3_52] (%alloc_47[2560] [2560] [1]) : (memref<5120xbf16, 1 : i32>)
            memref.dealloc %alloc_47 : memref<5120xbf16, 1 : i32>
          }
          %c0_23 = arith.constant 0 : index
          %c552_24 = arith.constant 552 : index
          %c1_25 = arith.constant 1 : index
          scf.for %arg17 = %c0_23 to %c552_24 step %c1_25 {
            %alloc_47 = memref.alloc() {air.memtile_col = 7 : i32} : memref<5120xbf16, 1 : i32>
            %c0_48 = arith.constant 0 : index
            air.channel.get  @inW0c3[%c0_48] (%alloc_47[] [] []) : (memref<5120xbf16, 1 : i32>)
            %c3_49 = arith.constant 3 : index
            %c0_50 = arith.constant 0 : index
            air.channel.put  @wL2ToL1[%c3_49, %c0_50] (%alloc_47[0] [2560] [1]) : (memref<5120xbf16, 1 : i32>)
            %c3_51 = arith.constant 3 : index
            %c1_52 = arith.constant 1 : index
            air.channel.put  @wL2ToL1[%c3_51, %c1_52] (%alloc_47[2560] [2560] [1]) : (memref<5120xbf16, 1 : i32>)
            memref.dealloc %alloc_47 : memref<5120xbf16, 1 : i32>
          }
          %c0_26 = arith.constant 0 : index
          %c552_27 = arith.constant 552 : index
          %c1_28 = arith.constant 1 : index
          scf.for %arg17 = %c0_26 to %c552_27 step %c1_28 {
            %alloc_47 = memref.alloc() {air.memtile_col = 7 : i32} : memref<5120xbf16, 1 : i32>
            %c0_48 = arith.constant 0 : index
            air.channel.get  @inW1c3[%c0_48] (%alloc_47[] [] []) : (memref<5120xbf16, 1 : i32>)
            %c3_49 = arith.constant 3 : index
            %c2_50 = arith.constant 2 : index
            air.channel.put  @wL2ToL1[%c3_49, %c2_50] (%alloc_47[0] [2560] [1]) : (memref<5120xbf16, 1 : i32>)
            %c3_51 = arith.constant 3 : index
            %c3_52 = arith.constant 3 : index
            air.channel.put  @wL2ToL1[%c3_51, %c3_52] (%alloc_47[2560] [2560] [1]) : (memref<5120xbf16, 1 : i32>)
            memref.dealloc %alloc_47 : memref<5120xbf16, 1 : i32>
          }
          scf.index_switch %9 
          case 0 {
            %c0_47 = arith.constant 0 : index
            %c27 = arith.constant 27 : index
            %c1_48 = arith.constant 1 : index
            scf.for %arg17 = %c0_47 to %c27 step %c1_48 {
              %alloc_49 = memref.alloc() {air.memtile_col = 0 : i32} : memref<130xbf16, 1 : i32>
              %c0_50 = arith.constant 0 : index
              %c0_51 = arith.constant 0 : index
              air.channel.get  @outA[%c0_50, %c0_51] (%alloc_49[0] [34] [1]) : (memref<130xbf16, 1 : i32>)
              %c0_52 = arith.constant 0 : index
              %c1_53 = arith.constant 1 : index
              air.channel.get  @outA[%c0_52, %c1_53] (%alloc_49[34] [32] [1]) : (memref<130xbf16, 1 : i32>)
              %c0_54 = arith.constant 0 : index
              %c2_55 = arith.constant 2 : index
              air.channel.get  @outA[%c0_54, %c2_55] (%alloc_49[66] [32] [1]) : (memref<130xbf16, 1 : i32>)
              %c0_56 = arith.constant 0 : index
              %c3_57 = arith.constant 3 : index
              air.channel.get  @outA[%c0_56, %c3_57] (%alloc_49[98] [32] [1]) : (memref<130xbf16, 1 : i32>)
              %c0_58 = arith.constant 0 : index
              air.channel.put  @toMain[%c0_58] (%alloc_49[0] [130] [1]) : (memref<130xbf16, 1 : i32>)
              memref.dealloc %alloc_49 : memref<130xbf16, 1 : i32>
              %alloc_59 = memref.alloc() {air.memtile_col = 1 : i32} : memref<130xbf16, 1 : i32>
              %c1_60 = arith.constant 1 : index
              %c0_61 = arith.constant 0 : index
              air.channel.get  @outA[%c1_60, %c0_61] (%alloc_59[0] [34] [1]) : (memref<130xbf16, 1 : i32>)
              %c1_62 = arith.constant 1 : index
              %c1_63 = arith.constant 1 : index
              air.channel.get  @outA[%c1_62, %c1_63] (%alloc_59[34] [32] [1]) : (memref<130xbf16, 1 : i32>)
              %c1_64 = arith.constant 1 : index
              %c2_65 = arith.constant 2 : index
              air.channel.get  @outA[%c1_64, %c2_65] (%alloc_59[66] [32] [1]) : (memref<130xbf16, 1 : i32>)
              %c1_66 = arith.constant 1 : index
              %c3_67 = arith.constant 3 : index
              air.channel.get  @outA[%c1_66, %c3_67] (%alloc_59[98] [32] [1]) : (memref<130xbf16, 1 : i32>)
              %c1_68 = arith.constant 1 : index
              air.channel.put  @toMain[%c1_68] (%alloc_59[0] [130] [1]) : (memref<130xbf16, 1 : i32>)
              memref.dealloc %alloc_59 : memref<130xbf16, 1 : i32>
              %alloc_69 = memref.alloc() {air.memtile_col = 6 : i32} : memref<130xbf16, 1 : i32>
              %c2_70 = arith.constant 2 : index
              %c0_71 = arith.constant 0 : index
              air.channel.get  @outA[%c2_70, %c0_71] (%alloc_69[0] [34] [1]) : (memref<130xbf16, 1 : i32>)
              %c2_72 = arith.constant 2 : index
              %c1_73 = arith.constant 1 : index
              air.channel.get  @outA[%c2_72, %c1_73] (%alloc_69[34] [32] [1]) : (memref<130xbf16, 1 : i32>)
              %c2_74 = arith.constant 2 : index
              %c2_75 = arith.constant 2 : index
              air.channel.get  @outA[%c2_74, %c2_75] (%alloc_69[66] [32] [1]) : (memref<130xbf16, 1 : i32>)
              %c2_76 = arith.constant 2 : index
              %c3_77 = arith.constant 3 : index
              air.channel.get  @outA[%c2_76, %c3_77] (%alloc_69[98] [32] [1]) : (memref<130xbf16, 1 : i32>)
              %c2_78 = arith.constant 2 : index
              air.channel.put  @toMain[%c2_78] (%alloc_69[0] [130] [1]) : (memref<130xbf16, 1 : i32>)
              memref.dealloc %alloc_69 : memref<130xbf16, 1 : i32>
              %alloc_79 = memref.alloc() {air.memtile_col = 7 : i32} : memref<130xbf16, 1 : i32>
              %c3_80 = arith.constant 3 : index
              %c0_81 = arith.constant 0 : index
              air.channel.get  @outA[%c3_80, %c0_81] (%alloc_79[0] [34] [1]) : (memref<130xbf16, 1 : i32>)
              %c3_82 = arith.constant 3 : index
              %c1_83 = arith.constant 1 : index
              air.channel.get  @outA[%c3_82, %c1_83] (%alloc_79[34] [32] [1]) : (memref<130xbf16, 1 : i32>)
              %c3_84 = arith.constant 3 : index
              %c2_85 = arith.constant 2 : index
              air.channel.get  @outA[%c3_84, %c2_85] (%alloc_79[66] [32] [1]) : (memref<130xbf16, 1 : i32>)
              %c3_86 = arith.constant 3 : index
              %c3_87 = arith.constant 3 : index
              air.channel.get  @outA[%c3_86, %c3_87] (%alloc_79[98] [32] [1]) : (memref<130xbf16, 1 : i32>)
              %c3_88 = arith.constant 3 : index
              air.channel.put  @toMain[%c3_88] (%alloc_79[0] [130] [1]) : (memref<130xbf16, 1 : i32>)
              memref.dealloc %alloc_79 : memref<130xbf16, 1 : i32>
              %alloc_89 = memref.alloc() {air.memtile_col = 2 : i32} : memref<514xbf16, 1 : i32>
              %c0_90 = arith.constant 0 : index
              air.channel.get  @toMain[%c0_90] (%alloc_89[0] [130] [1]) : (memref<514xbf16, 1 : i32>)
              %c1_91 = arith.constant 1 : index
              air.channel.get  @toMain[%c1_91] (%alloc_89[130] [128] [1]) : (memref<514xbf16, 1 : i32>)
              %c2_92 = arith.constant 2 : index
              air.channel.get  @toMain[%c2_92] (%alloc_89[258] [128] [1]) : (memref<514xbf16, 1 : i32>)
              %c3_93 = arith.constant 3 : index
              air.channel.get  @toMain[%c3_93] (%alloc_89[386] [128] [1]) : (memref<514xbf16, 1 : i32>)
              %c0_94 = arith.constant 0 : index
              %c0_95 = arith.constant 0 : index
              air.channel.put  @outY[%c0_94, %c0_95] (%alloc_89[0] [514] [1]) : (memref<514xbf16, 1 : i32>)
              memref.dealloc %alloc_89 : memref<514xbf16, 1 : i32>
            }
            scf.yield
          }
          default {
            %c0_47 = arith.constant 0 : index
            %c66 = arith.constant 66 : index
            %c1_48 = arith.constant 1 : index
            scf.for %arg17 = %c0_47 to %c66 step %c1_48 {
              %alloc_49 = memref.alloc() {air.memtile_col = 0 : i32} : memref<130xbf16, 1 : i32>
              %c0_50 = arith.constant 0 : index
              %c0_51 = arith.constant 0 : index
              air.channel.get  @outA[%c0_50, %c0_51] (%alloc_49[0] [34] [1]) : (memref<130xbf16, 1 : i32>)
              %c0_52 = arith.constant 0 : index
              %c1_53 = arith.constant 1 : index
              air.channel.get  @outA[%c0_52, %c1_53] (%alloc_49[34] [32] [1]) : (memref<130xbf16, 1 : i32>)
              %c0_54 = arith.constant 0 : index
              %c2_55 = arith.constant 2 : index
              air.channel.get  @outA[%c0_54, %c2_55] (%alloc_49[66] [32] [1]) : (memref<130xbf16, 1 : i32>)
              %c0_56 = arith.constant 0 : index
              %c3_57 = arith.constant 3 : index
              air.channel.get  @outA[%c0_56, %c3_57] (%alloc_49[98] [32] [1]) : (memref<130xbf16, 1 : i32>)
              %c0_58 = arith.constant 0 : index
              air.channel.put  @toMain[%c0_58] (%alloc_49[0] [130] [1]) : (memref<130xbf16, 1 : i32>)
              memref.dealloc %alloc_49 : memref<130xbf16, 1 : i32>
              %alloc_59 = memref.alloc() {air.memtile_col = 1 : i32} : memref<130xbf16, 1 : i32>
              %c1_60 = arith.constant 1 : index
              %c0_61 = arith.constant 0 : index
              air.channel.get  @outA[%c1_60, %c0_61] (%alloc_59[0] [34] [1]) : (memref<130xbf16, 1 : i32>)
              %c1_62 = arith.constant 1 : index
              %c1_63 = arith.constant 1 : index
              air.channel.get  @outA[%c1_62, %c1_63] (%alloc_59[34] [32] [1]) : (memref<130xbf16, 1 : i32>)
              %c1_64 = arith.constant 1 : index
              %c2_65 = arith.constant 2 : index
              air.channel.get  @outA[%c1_64, %c2_65] (%alloc_59[66] [32] [1]) : (memref<130xbf16, 1 : i32>)
              %c1_66 = arith.constant 1 : index
              %c3_67 = arith.constant 3 : index
              air.channel.get  @outA[%c1_66, %c3_67] (%alloc_59[98] [32] [1]) : (memref<130xbf16, 1 : i32>)
              %c1_68 = arith.constant 1 : index
              air.channel.put  @toMain[%c1_68] (%alloc_59[0] [130] [1]) : (memref<130xbf16, 1 : i32>)
              memref.dealloc %alloc_59 : memref<130xbf16, 1 : i32>
              %alloc_69 = memref.alloc() {air.memtile_col = 6 : i32} : memref<130xbf16, 1 : i32>
              %c2_70 = arith.constant 2 : index
              %c0_71 = arith.constant 0 : index
              air.channel.get  @outA[%c2_70, %c0_71] (%alloc_69[0] [34] [1]) : (memref<130xbf16, 1 : i32>)
              %c2_72 = arith.constant 2 : index
              %c1_73 = arith.constant 1 : index
              air.channel.get  @outA[%c2_72, %c1_73] (%alloc_69[34] [32] [1]) : (memref<130xbf16, 1 : i32>)
              %c2_74 = arith.constant 2 : index
              %c2_75 = arith.constant 2 : index
              air.channel.get  @outA[%c2_74, %c2_75] (%alloc_69[66] [32] [1]) : (memref<130xbf16, 1 : i32>)
              %c2_76 = arith.constant 2 : index
              %c3_77 = arith.constant 3 : index
              air.channel.get  @outA[%c2_76, %c3_77] (%alloc_69[98] [32] [1]) : (memref<130xbf16, 1 : i32>)
              %c2_78 = arith.constant 2 : index
              air.channel.put  @toMain[%c2_78] (%alloc_69[0] [130] [1]) : (memref<130xbf16, 1 : i32>)
              memref.dealloc %alloc_69 : memref<130xbf16, 1 : i32>
              %alloc_79 = memref.alloc() {air.memtile_col = 7 : i32} : memref<130xbf16, 1 : i32>
              %c3_80 = arith.constant 3 : index
              %c0_81 = arith.constant 0 : index
              air.channel.get  @outA[%c3_80, %c0_81] (%alloc_79[0] [34] [1]) : (memref<130xbf16, 1 : i32>)
              %c3_82 = arith.constant 3 : index
              %c1_83 = arith.constant 1 : index
              air.channel.get  @outA[%c3_82, %c1_83] (%alloc_79[34] [32] [1]) : (memref<130xbf16, 1 : i32>)
              %c3_84 = arith.constant 3 : index
              %c2_85 = arith.constant 2 : index
              air.channel.get  @outA[%c3_84, %c2_85] (%alloc_79[66] [32] [1]) : (memref<130xbf16, 1 : i32>)
              %c3_86 = arith.constant 3 : index
              %c3_87 = arith.constant 3 : index
              air.channel.get  @outA[%c3_86, %c3_87] (%alloc_79[98] [32] [1]) : (memref<130xbf16, 1 : i32>)
              %c3_88 = arith.constant 3 : index
              air.channel.put  @toMain[%c3_88] (%alloc_79[0] [130] [1]) : (memref<130xbf16, 1 : i32>)
              memref.dealloc %alloc_79 : memref<130xbf16, 1 : i32>
              %alloc_89 = memref.alloc() {air.memtile_col = 2 : i32} : memref<514xbf16, 1 : i32>
              %c0_90 = arith.constant 0 : index
              air.channel.get  @toMain[%c0_90] (%alloc_89[0] [130] [1]) : (memref<514xbf16, 1 : i32>)
              %c1_91 = arith.constant 1 : index
              air.channel.get  @toMain[%c1_91] (%alloc_89[130] [128] [1]) : (memref<514xbf16, 1 : i32>)
              %c2_92 = arith.constant 2 : index
              air.channel.get  @toMain[%c2_92] (%alloc_89[258] [128] [1]) : (memref<514xbf16, 1 : i32>)
              %c3_93 = arith.constant 3 : index
              air.channel.get  @toMain[%c3_93] (%alloc_89[386] [128] [1]) : (memref<514xbf16, 1 : i32>)
              %c0_94 = arith.constant 0 : index
              %c0_95 = arith.constant 0 : index
              air.channel.put  @outY[%c0_94, %c0_95] (%alloc_89[0] [514] [1]) : (memref<514xbf16, 1 : i32>)
              memref.dealloc %alloc_89 : memref<514xbf16, 1 : i32>
            }
          }
          %c1_29 = arith.constant 1 : index
          %c1_30 = arith.constant 1 : index
          air.herd @rope  tile (%arg17, %arg18) in (%arg19=%c1_29, %arg20=%c1_30) args(%arg21=%8) : i32 attributes {link_with = "rope.o", x_loc = 2 : i64, y_loc = 3 : i64} {
            %10 = arith.index_cast %arg21 : i32 to index
            scf.index_switch %10 
            case 0 {
              scf.yield
            }
            default {
              %alloc_47 = memref.alloc() : memref<6144xbf16, 2 : i32>
              %c0_48 = arith.constant 0 : index
              %c0_49 = arith.constant 0 : index
              %c0_50 = arith.constant 0 : index
              %c512 = arith.constant 512 : index
              %c1_51 = arith.constant 1 : index
              air.channel.get  @outY[%c0_48, %c0_49] (%alloc_47[%c0_50] [%c512] [%c1_51]) : (memref<6144xbf16, 2 : i32>)
              %c0_52 = arith.constant 0 : index
              %c0_53 = arith.constant 0 : index
              %c512_54 = arith.constant 512 : index
              %c512_55 = arith.constant 512 : index
              %c1_56 = arith.constant 1 : index
              air.channel.get  @outY[%c0_52, %c0_53] (%alloc_47[%c512_54] [%c512_55] [%c1_56]) : (memref<6144xbf16, 2 : i32>)
              %c0_57 = arith.constant 0 : index
              %c0_58 = arith.constant 0 : index
              %c1024 = arith.constant 1024 : index
              %c512_59 = arith.constant 512 : index
              %c1_60 = arith.constant 1 : index
              air.channel.get  @outY[%c0_57, %c0_58] (%alloc_47[%c1024] [%c512_59] [%c1_60]) : (memref<6144xbf16, 2 : i32>)
              %c0_61 = arith.constant 0 : index
              %c0_62 = arith.constant 0 : index
              %c1536 = arith.constant 1536 : index
              %c512_63 = arith.constant 512 : index
              %c1_64 = arith.constant 1 : index
              air.channel.get  @outY[%c0_61, %c0_62] (%alloc_47[%c1536] [%c512_63] [%c1_64]) : (memref<6144xbf16, 2 : i32>)
              %c0_65 = arith.constant 0 : index
              %c0_66 = arith.constant 0 : index
              %c2048 = arith.constant 2048 : index
              %c512_67 = arith.constant 512 : index
              %c1_68 = arith.constant 1 : index
              air.channel.get  @outY[%c0_65, %c0_66] (%alloc_47[%c2048] [%c512_67] [%c1_68]) : (memref<6144xbf16, 2 : i32>)
              %c0_69 = arith.constant 0 : index
              %c0_70 = arith.constant 0 : index
              %c2560 = arith.constant 2560 : index
              %c512_71 = arith.constant 512 : index
              %c1_72 = arith.constant 1 : index
              air.channel.get  @outY[%c0_69, %c0_70] (%alloc_47[%c2560] [%c512_71] [%c1_72]) : (memref<6144xbf16, 2 : i32>)
              %c0_73 = arith.constant 0 : index
              %c0_74 = arith.constant 0 : index
              %c3072 = arith.constant 3072 : index
              %c512_75 = arith.constant 512 : index
              %c1_76 = arith.constant 1 : index
              air.channel.get  @outY[%c0_73, %c0_74] (%alloc_47[%c3072] [%c512_75] [%c1_76]) : (memref<6144xbf16, 2 : i32>)
              %c0_77 = arith.constant 0 : index
              %c0_78 = arith.constant 0 : index
              %c3584 = arith.constant 3584 : index
              %c512_79 = arith.constant 512 : index
              %c1_80 = arith.constant 1 : index
              air.channel.get  @outY[%c0_77, %c0_78] (%alloc_47[%c3584] [%c512_79] [%c1_80]) : (memref<6144xbf16, 2 : i32>)
              %c0_81 = arith.constant 0 : index
              %c0_82 = arith.constant 0 : index
              %c4096 = arith.constant 4096 : index
              %c512_83 = arith.constant 512 : index
              %c1_84 = arith.constant 1 : index
              air.channel.get  @outY[%c0_81, %c0_82] (%alloc_47[%c4096] [%c512_83] [%c1_84]) : (memref<6144xbf16, 2 : i32>)
              %c0_85 = arith.constant 0 : index
              %c0_86 = arith.constant 0 : index
              %c4608 = arith.constant 4608 : index
              %c512_87 = arith.constant 512 : index
              %c1_88 = arith.constant 1 : index
              air.channel.get  @outY[%c0_85, %c0_86] (%alloc_47[%c4608] [%c512_87] [%c1_88]) : (memref<6144xbf16, 2 : i32>)
              %c0_89 = arith.constant 0 : index
              %c0_90 = arith.constant 0 : index
              %c5120 = arith.constant 5120 : index
              %c512_91 = arith.constant 512 : index
              %c1_92 = arith.constant 1 : index
              air.channel.get  @outY[%c0_89, %c0_90] (%alloc_47[%c5120] [%c512_91] [%c1_92]) : (memref<6144xbf16, 2 : i32>)
              %c0_93 = arith.constant 0 : index
              %c0_94 = arith.constant 0 : index
              %c5632 = arith.constant 5632 : index
              %c512_95 = arith.constant 512 : index
              %c1_96 = arith.constant 1 : index
              air.channel.get  @outY[%c0_93, %c0_94] (%alloc_47[%c5632] [%c512_95] [%c1_96]) : (memref<6144xbf16, 2 : i32>)
              %alloc_97 = memref.alloc() : memref<1536xbf16, 2 : i32>
              %c0_98 = arith.constant 0 : index
              air.channel.get  @ropeLUT[%c0_98] (%alloc_97[] [] []) : (memref<1536xbf16, 2 : i32>)
              %alloc_99 = memref.alloc() : memref<4096xbf16, 2 : i32>
              %alloc_100 = memref.alloc() : memref<1024xbf16, 2 : i32>
              %alloc_101 = memref.alloc() : memref<1024xbf16, 2 : i32>
              func.call @rope_compute(%alloc_99, %alloc_100, %alloc_101, %alloc_47, %alloc_97, %arg21) : (memref<4096xbf16, 2 : i32>, memref<1024xbf16, 2 : i32>, memref<1024xbf16, 2 : i32>, memref<6144xbf16, 2 : i32>, memref<1536xbf16, 2 : i32>, i32) -> ()
              %c0_102 = arith.constant 0 : index
              %c0_103 = arith.constant 0 : index
              %c4096_104 = arith.constant 4096 : index
              %c1_105 = arith.constant 1 : index
              air.channel.put  @ropeQ[%c0_102] (%alloc_99[%c0_103] [%c4096_104] [%c1_105]) : (memref<4096xbf16, 2 : i32>)
              %c0_106 = arith.constant 0 : index
              %c0_107 = arith.constant 0 : index
              %c1024_108 = arith.constant 1024 : index
              %c1_109 = arith.constant 1 : index
              air.channel.put  @appendK[%c0_106] (%alloc_100[%c0_107] [%c1024_108] [%c1_109]) : (memref<1024xbf16, 2 : i32>)
              %c0_110 = arith.constant 0 : index
              %c0_111 = arith.constant 0 : index
              %c1024_112 = arith.constant 1024 : index
              %c1_113 = arith.constant 1 : index
              air.channel.put  @appendV[%c0_110] (%alloc_101[%c0_111] [%c1024_112] [%c1_113]) : (memref<1024xbf16, 2 : i32>)
              memref.dealloc %alloc_47 : memref<6144xbf16, 2 : i32>
              memref.dealloc %alloc_97 : memref<1536xbf16, 2 : i32>
              memref.dealloc %alloc_99 : memref<4096xbf16, 2 : i32>
              memref.dealloc %alloc_100 : memref<1024xbf16, 2 : i32>
              memref.dealloc %alloc_101 : memref<1024xbf16, 2 : i32>
            }
          }
          scf.index_switch %9 
          case 0 {
            scf.yield
          }
          default {
            %alloc_47 = memref.alloc() {air.memtile_col = 5 : i32, air.no_split} : memref<4096xbf16, 1 : i32>
            %c0_48 = arith.constant 0 : index
            air.channel.get  @ropeQ[%c0_48] (%alloc_47[] [] []) : (memref<4096xbf16, 1 : i32>)
            %c0_49 = arith.constant 0 : index
            %c0_50 = arith.constant 0 : index
            %c0_51 = arith.constant 0 : index
            %c0_52 = arith.constant 0 : index
            %c64 = arith.constant 64 : index
            %c4_53 = arith.constant 4 : index
            %c8 = arith.constant 8 : index
            %c8_54 = arith.constant 8 : index
            %c512 = arith.constant 512 : index
            %c1_55 = arith.constant 1 : index
            air.channel.put  @toAttnQ[%c0_49] (%alloc_47[%c0_50, %c0_51, %c0_52] [%c64, %c4_53, %c8] [%c8_54, %c512, %c1_55]) : (memref<4096xbf16, 1 : i32>)
            %c1_56 = arith.constant 1 : index
            %c0_57 = arith.constant 0 : index
            %c4_58 = arith.constant 4 : index
            %c0_59 = arith.constant 0 : index
            %c64_60 = arith.constant 64 : index
            %c4_61 = arith.constant 4 : index
            %c8_62 = arith.constant 8 : index
            %c8_63 = arith.constant 8 : index
            %c512_64 = arith.constant 512 : index
            %c1_65 = arith.constant 1 : index
            air.channel.put  @toAttnQ[%c1_56] (%alloc_47[%c0_57, %c4_58, %c0_59] [%c64_60, %c4_61, %c8_62] [%c8_63, %c512_64, %c1_65]) : (memref<4096xbf16, 1 : i32>)
            memref.dealloc %alloc_47 : memref<4096xbf16, 1 : i32>
          }
          %alloc = memref.alloc() : memref<128xbf16, 2 : i32>
          %c16_i32 = arith.constant 16 : i32
          scf.index_switch %9 
          case 0 {
            scf.yield
          }
          default {
            %c0_47 = arith.constant 0 : index
            %c1_48 = arith.constant 1 : index
            %c1_49 = arith.constant 1 : index
            scf.for %arg17 = %c0_47 to %c1_48 step %c1_49 {
              %alloc_50 = memref.alloc() {air.memtile_col = 4 : i32} : memref<16384xbf16, 1 : i32>
              %alloc_51 = memref.alloc() {air.memtile_col = 4 : i32} : memref<16384xbf16, 1 : i32>
              %c0_52 = arith.constant 0 : index
              air.channel.get  @inKV_K[%c0_52] (%alloc_50[] [] []) : (memref<16384xbf16, 1 : i32>)
              %c0_53 = arith.constant 0 : index
              air.channel.get  @inKV_V[%c0_53] (%alloc_51[] [] []) : (memref<16384xbf16, 1 : i32>)
              %c0_54 = arith.constant 0 : index
              %c0_55 = arith.constant 0 : index
              %c0_56 = arith.constant 0 : index
              %c0_57 = arith.constant 0 : index
              %c64 = arith.constant 64 : index
              %c16 = arith.constant 16 : index
              %c8 = arith.constant 8 : index
              %c8_58 = arith.constant 8 : index
              %c1024 = arith.constant 1024 : index
              %c1_59 = arith.constant 1 : index
              air.channel.put  @toK[%c0_54] (%alloc_50[%c0_55, %c0_56, %c0_57] [%c64, %c16, %c8] [%c8_58, %c1024, %c1_59]) : (memref<16384xbf16, 1 : i32>)
              %c0_60 = arith.constant 0 : index
              %c0_61 = arith.constant 0 : index
              %c0_62 = arith.constant 0 : index
              %c0_63 = arith.constant 0 : index
              %c0_64 = arith.constant 0 : index
              %c2_65 = arith.constant 2 : index
              %c64_66 = arith.constant 64 : index
              %c8_67 = arith.constant 8 : index
              %c8_68 = arith.constant 8 : index
              %c8192 = arith.constant 8192 : index
              %c8_69 = arith.constant 8 : index
              %c1024_70 = arith.constant 1024 : index
              %c1_71 = arith.constant 1 : index
              air.channel.put  @toV[%c0_60] (%alloc_51[%c0_61, %c0_62, %c0_63, %c0_64] [%c2_65, %c64_66, %c8_67, %c8_68] [%c8192, %c8_69, %c1024_70, %c1_71]) : (memref<16384xbf16, 1 : i32>)
              %c1_72 = arith.constant 1 : index
              %c0_73 = arith.constant 0 : index
              %c0_74 = arith.constant 0 : index
              %c512 = arith.constant 512 : index
              %c64_75 = arith.constant 64 : index
              %c16_76 = arith.constant 16 : index
              %c8_77 = arith.constant 8 : index
              %c8_78 = arith.constant 8 : index
              %c1024_79 = arith.constant 1024 : index
              %c1_80 = arith.constant 1 : index
              air.channel.put  @toK[%c1_72] (%alloc_50[%c0_73, %c0_74, %c512] [%c64_75, %c16_76, %c8_77] [%c8_78, %c1024_79, %c1_80]) : (memref<16384xbf16, 1 : i32>)
              %c1_81 = arith.constant 1 : index
              %c0_82 = arith.constant 0 : index
              %c0_83 = arith.constant 0 : index
              %c0_84 = arith.constant 0 : index
              %c512_85 = arith.constant 512 : index
              %c2_86 = arith.constant 2 : index
              %c64_87 = arith.constant 64 : index
              %c8_88 = arith.constant 8 : index
              %c8_89 = arith.constant 8 : index
              %c8192_90 = arith.constant 8192 : index
              %c8_91 = arith.constant 8 : index
              %c1024_92 = arith.constant 1024 : index
              %c1_93 = arith.constant 1 : index
              air.channel.put  @toV[%c1_81] (%alloc_51[%c0_82, %c0_83, %c0_84, %c512_85] [%c2_86, %c64_87, %c8_88, %c8_89] [%c8192_90, %c8_91, %c1024_92, %c1_93]) : (memref<16384xbf16, 1 : i32>)
              memref.dealloc %alloc_50 : memref<16384xbf16, 1 : i32>
              memref.dealloc %alloc_51 : memref<16384xbf16, 1 : i32>
            }
          }
          %alloc_31 = memref.alloc() : memref<128xbf16, 2 : i32>
          %c16_i32_32 = arith.constant 16 : i32
          scf.index_switch %9 
          case 0 {
            scf.yield
          }
          default {
          }
          %c1_33 = arith.constant 1 : index
          %c4 = arith.constant 4 : index
          air.herd @attn_blk  tile (%arg17, %arg18) in (%arg19=%c1_33, %arg20=%c4) args(%arg21=%alloc, %arg22=%alloc_31, %arg23=%c16_i32, %arg24=%8) : memref<128xbf16, 2 : i32>, memref<128xbf16, 2 : i32>, i32, i32 attributes {x_loc = 4 : i64, y_loc = 2 : i64} {
            %10 = arith.index_cast %arg24 : i32 to index
            scf.index_switch %10 
            case 0 {
              scf.yield
            }
            default {
              %c2_47 = arith.constant 2 : index
              %11 = arith.cmpi slt, %arg18, %c2_47 : index
              scf.if %11 {
                %c0_48 = arith.constant 0 : index
                %12 = arith.cmpi eq, %arg18, %c0_48 : index
                scf.if %12 {
                  %alloc_49 = memref.alloc() : memref<2048xbf16, 2 : i32>
                  %c0_50 = arith.constant 0 : index
                  air.channel.get  @toAttnQ[%c0_50] (%alloc_49[] [] []) : (memref<2048xbf16, 2 : i32>)
                  %alloc_51 = memref.alloc() : memref<16xbf16, 2 : i32>
                  %alloc_52 = memref.alloc() : memref<8xf32, 2 : i32>
                  %c1_53 = arith.constant 1 : index
                  %c0_54 = arith.constant 0 : index
                  %c1_55 = arith.constant 1 : index
                  scf.for %arg25 = %c0_54 to %c1_53 step %c1_55 {
                    %alloc_56 = memref.alloc() : memref<8192xbf16, 2 : i32>
                    %c0_57 = arith.constant 0 : index
                    air.channel.get  @toK[%c0_57] (%alloc_56[] [] []) : (memref<8192xbf16, 2 : i32>)
                    %13 = arith.index_cast %arg25 : index to i32
                    func.call @attn_qk_blk(%alloc_49, %alloc_56, %alloc_51, %alloc_52, %arg21, %13, %arg23) : (memref<2048xbf16, 2 : i32>, memref<8192xbf16, 2 : i32>, memref<16xbf16, 2 : i32>, memref<8xf32, 2 : i32>, memref<128xbf16, 2 : i32>, i32, i32) -> ()
                    memref.dealloc %alloc_56 : memref<8192xbf16, 2 : i32>
                  }
                  memref.dealloc %alloc_49 : memref<2048xbf16, 2 : i32>
                  memref.dealloc %alloc_51 : memref<16xbf16, 2 : i32>
                  memref.dealloc %alloc_52 : memref<8xf32, 2 : i32>
                } else {
                  %alloc_49 = memref.alloc() : memref<2048xf32, 2 : i32>
                  %alloc_50 = memref.alloc() : memref<16xf32, 2 : i32>
                  %alloc_51 = memref.alloc() : memref<2048xbf16, 2 : i32>
                  %c1_52 = arith.constant 1 : index
                  %c0_53 = arith.constant 0 : index
                  %c1_54 = arith.constant 1 : index
                  scf.for %arg25 = %c0_53 to %c1_52 step %c1_54 {
                    %alloc_62 = memref.alloc() : memref<8192xbf16, 2 : i32>
                    %c0_63 = arith.constant 0 : index
                    air.channel.get  @toV[%c0_63] (%alloc_62[] [] []) : (memref<8192xbf16, 2 : i32>)
                    %13 = arith.index_cast %arg25 : index to i32
                    func.call @attn_kv_blk(%arg21, %alloc_62, %alloc_49, %alloc_50, %13, %arg23) : (memref<128xbf16, 2 : i32>, memref<8192xbf16, 2 : i32>, memref<2048xf32, 2 : i32>, memref<16xf32, 2 : i32>, i32, i32) -> ()
                    memref.dealloc %alloc_62 : memref<8192xbf16, 2 : i32>
                  }
                  func.call @attn_kv_fin(%alloc_49, %alloc_50, %alloc_51) : (memref<2048xf32, 2 : i32>, memref<16xf32, 2 : i32>, memref<2048xbf16, 2 : i32>) -> ()
                  %c0_55 = arith.constant 0 : index
                  %c0_56 = arith.constant 0 : index
                  %c0_57 = arith.constant 0 : index
                  %c0_58 = arith.constant 0 : index
                  %c4_59 = arith.constant 4 : index
                  %c64 = arith.constant 64 : index
                  %c8 = arith.constant 8 : index
                  %c8_60 = arith.constant 8 : index
                  %c32 = arith.constant 32 : index
                  %c1_61 = arith.constant 1 : index
                  air.channel.put  @attnO[%c0_55] (%alloc_51[%c0_56, %c0_57, %c0_58] [%c4_59, %c64, %c8] [%c8_60, %c32, %c1_61]) : (memref<2048xbf16, 2 : i32>)
                  memref.dealloc %alloc_51 : memref<2048xbf16, 2 : i32>
                  memref.dealloc %alloc_49 : memref<2048xf32, 2 : i32>
                  memref.dealloc %alloc_50 : memref<16xf32, 2 : i32>
                }
              } else {
                %c2_48 = arith.constant 2 : index
                %12 = arith.cmpi eq, %arg18, %c2_48 : index
                scf.if %12 {
                  %alloc_49 = memref.alloc() : memref<2048xbf16, 2 : i32>
                  %c1_50 = arith.constant 1 : index
                  air.channel.get  @toAttnQ[%c1_50] (%alloc_49[] [] []) : (memref<2048xbf16, 2 : i32>)
                  %alloc_51 = memref.alloc() : memref<16xbf16, 2 : i32>
                  %alloc_52 = memref.alloc() : memref<8xf32, 2 : i32>
                  %c1_53 = arith.constant 1 : index
                  %c0_54 = arith.constant 0 : index
                  %c1_55 = arith.constant 1 : index
                  scf.for %arg25 = %c0_54 to %c1_53 step %c1_55 {
                    %alloc_56 = memref.alloc() : memref<8192xbf16, 2 : i32>
                    %c1_57 = arith.constant 1 : index
                    air.channel.get  @toK[%c1_57] (%alloc_56[] [] []) : (memref<8192xbf16, 2 : i32>)
                    %13 = arith.index_cast %arg25 : index to i32
                    func.call @attn_qk_blk(%alloc_49, %alloc_56, %alloc_51, %alloc_52, %arg22, %13, %arg23) : (memref<2048xbf16, 2 : i32>, memref<8192xbf16, 2 : i32>, memref<16xbf16, 2 : i32>, memref<8xf32, 2 : i32>, memref<128xbf16, 2 : i32>, i32, i32) -> ()
                    memref.dealloc %alloc_56 : memref<8192xbf16, 2 : i32>
                  }
                  memref.dealloc %alloc_49 : memref<2048xbf16, 2 : i32>
                  memref.dealloc %alloc_51 : memref<16xbf16, 2 : i32>
                  memref.dealloc %alloc_52 : memref<8xf32, 2 : i32>
                } else {
                  %alloc_49 = memref.alloc() : memref<2048xf32, 2 : i32>
                  %alloc_50 = memref.alloc() : memref<16xf32, 2 : i32>
                  %alloc_51 = memref.alloc() : memref<2048xbf16, 2 : i32>
                  %c1_52 = arith.constant 1 : index
                  %c0_53 = arith.constant 0 : index
                  %c1_54 = arith.constant 1 : index
                  scf.for %arg25 = %c0_53 to %c1_52 step %c1_54 {
                    %alloc_62 = memref.alloc() : memref<8192xbf16, 2 : i32>
                    %c1_63 = arith.constant 1 : index
                    air.channel.get  @toV[%c1_63] (%alloc_62[] [] []) : (memref<8192xbf16, 2 : i32>)
                    %13 = arith.index_cast %arg25 : index to i32
                    func.call @attn_kv_blk(%arg22, %alloc_62, %alloc_49, %alloc_50, %13, %arg23) : (memref<128xbf16, 2 : i32>, memref<8192xbf16, 2 : i32>, memref<2048xf32, 2 : i32>, memref<16xf32, 2 : i32>, i32, i32) -> ()
                    memref.dealloc %alloc_62 : memref<8192xbf16, 2 : i32>
                  }
                  func.call @attn_kv_fin(%alloc_49, %alloc_50, %alloc_51) : (memref<2048xf32, 2 : i32>, memref<16xf32, 2 : i32>, memref<2048xbf16, 2 : i32>) -> ()
                  %c1_55 = arith.constant 1 : index
                  %c0_56 = arith.constant 0 : index
                  %c0_57 = arith.constant 0 : index
                  %c0_58 = arith.constant 0 : index
                  %c4_59 = arith.constant 4 : index
                  %c64 = arith.constant 64 : index
                  %c8 = arith.constant 8 : index
                  %c8_60 = arith.constant 8 : index
                  %c32 = arith.constant 32 : index
                  %c1_61 = arith.constant 1 : index
                  air.channel.put  @attnO[%c1_55] (%alloc_51[%c0_56, %c0_57, %c0_58] [%c4_59, %c64, %c8] [%c8_60, %c32, %c1_61]) : (memref<2048xbf16, 2 : i32>)
                  memref.dealloc %alloc_51 : memref<2048xbf16, 2 : i32>
                  memref.dealloc %alloc_49 : memref<2048xf32, 2 : i32>
                  memref.dealloc %alloc_50 : memref<16xf32, 2 : i32>
                }
              }
            }
          }
          scf.index_switch %9 
          case 0 {
            scf.yield
          }
          default {
            %alloc_47 = memref.alloc() {air.memtile_col = 5 : i32} : memref<4096xbf16, 1 : i32>
            %c0_48 = arith.constant 0 : index
            %c0_49 = arith.constant 0 : index
            %c2048 = arith.constant 2048 : index
            %c1_50 = arith.constant 1 : index
            air.channel.get  @attnO[%c0_48] (%alloc_47[%c0_49] [%c2048] [%c1_50]) : (memref<4096xbf16, 1 : i32>)
            %c1_51 = arith.constant 1 : index
            %c2048_52 = arith.constant 2048 : index
            %c2048_53 = arith.constant 2048 : index
            %c1_54 = arith.constant 1 : index
            air.channel.get  @attnO[%c1_51] (%alloc_47[%c2048_52] [%c2048_53] [%c1_54]) : (memref<4096xbf16, 1 : i32>)
            %c0_55 = arith.constant 0 : index
            %c3_56 = arith.constant 3 : index
            %c1_57 = arith.constant 1 : index
            scf.for %arg17 = %c0_55 to %c3_56 step %c1_57 {
              %c0_58 = arith.constant 0 : index
              %c0_59 = arith.constant 0 : index
              %c4096 = arith.constant 4096 : index
              %c1_60 = arith.constant 1 : index
              air.channel.put  @xnorm[%c0_58] (%alloc_47[%c0_59] [%c4096] [%c1_60]) : (memref<4096xbf16, 1 : i32>)
            }
            memref.dealloc %alloc_47 : memref<4096xbf16, 1 : i32>
          }
          %c1_34 = arith.constant 1 : index
          %c1_35 = arith.constant 1 : index
          air.herd @glu  tile (%arg17, %arg18) in (%arg19=%c1_34, %arg20=%c1_35) args(%arg21=%8) : i32 attributes {link_with = "glu.o", x_loc = 5 : i64, y_loc = 3 : i64} {
            %10 = arith.index_cast %arg21 : i32 to index
            scf.index_switch %10 
            case 0 {
              scf.yield
            }
            default {
              %c0_47 = arith.constant 0 : index
              %c12 = arith.constant 12 : index
              %c1_48 = arith.constant 1 : index
              scf.for %arg22 = %c0_47 to %c12 step %c1_48 {
                %alloc_49 = memref.alloc() : memref<1024xbf16, 2 : i32>
                %c0_50 = arith.constant 0 : index
                %c2_51 = arith.constant 2 : index
                %c0_52 = arith.constant 0 : index
                %c1024 = arith.constant 1024 : index
                %c1_53 = arith.constant 1 : index
                air.channel.get  @outY[%c0_50, %c2_51] (%alloc_49[%c0_52] [%c1024] [%c1_53]) : (memref<1024xbf16, 2 : i32>)
                %alloc_54 = memref.alloc() : memref<512xbf16, 2 : i32>
                func.call @glu_aie(%alloc_54, %alloc_49, %arg21) : (memref<512xbf16, 2 : i32>, memref<1024xbf16, 2 : i32>, i32) -> ()
                %c0_55 = arith.constant 0 : index
                %c512 = arith.constant 512 : index
                %c1_56 = arith.constant 1 : index
                air.channel.put  @gluOut[] (%alloc_54[%c0_55] [%c512] [%c1_56]) : (memref<512xbf16, 2 : i32>)
                memref.dealloc %alloc_49 : memref<1024xbf16, 2 : i32>
                memref.dealloc %alloc_54 : memref<512xbf16, 2 : i32>
                %alloc_57 = memref.alloc() : memref<1024xbf16, 2 : i32>
                %c0_58 = arith.constant 0 : index
                %c2_59 = arith.constant 2 : index
                %c0_60 = arith.constant 0 : index
                %c1024_61 = arith.constant 1024 : index
                %c1_62 = arith.constant 1 : index
                air.channel.get  @outY[%c0_58, %c2_59] (%alloc_57[%c0_60] [%c1024_61] [%c1_62]) : (memref<1024xbf16, 2 : i32>)
                %alloc_63 = memref.alloc() : memref<512xbf16, 2 : i32>
                func.call @glu_aie(%alloc_63, %alloc_57, %arg21) : (memref<512xbf16, 2 : i32>, memref<1024xbf16, 2 : i32>, i32) -> ()
                %c0_64 = arith.constant 0 : index
                %c512_65 = arith.constant 512 : index
                %c1_66 = arith.constant 1 : index
                air.channel.put  @gluOut[] (%alloc_63[%c0_64] [%c512_65] [%c1_66]) : (memref<512xbf16, 2 : i32>)
                memref.dealloc %alloc_57 : memref<1024xbf16, 2 : i32>
                memref.dealloc %alloc_63 : memref<512xbf16, 2 : i32>
              }
            }
          }
          %alloc_36 = memref.alloc() {air.memtile_col = 3 : i32} : memref<12288xbf16, 1 : i32>
          %c0_37 = arith.constant 0 : index
          %c24 = arith.constant 24 : index
          %c1_38 = arith.constant 1 : index
          scf.for %arg17 = %c0_37 to %c24 step %c1_38 {
            %c512 = arith.constant 512 : index
            %10 = arith.muli %arg17, %c512 : index
            %c512_47 = arith.constant 512 : index
            %c1_48 = arith.constant 1 : index
            air.channel.get  @gluOut[] (%alloc_36[%10] [%c512_47] [%c1_48]) : (memref<12288xbf16, 1 : i32>)
          }
          %c0_39 = arith.constant 0 : index
          %c3_40 = arith.constant 3 : index
          %c1_41 = arith.constant 1 : index
          scf.for %arg17 = %c0_39 to %c3_40 step %c1_41 {
            air.channel.put  @xnorm[] (%alloc_36[0] [12288] [1]) : (memref<12288xbf16, 1 : i32>)
          }
          memref.dealloc %alloc_36 : memref<12288xbf16, 1 : i32>
          %c2 = arith.constant 2 : index
          %c4_42 = arith.constant 4 : index
          air.herd @proj_blk0  tile (%arg17, %arg18) in (%arg19=%c2, %arg20=%c4_42) args(%arg21=%8) : i32 attributes {link_with = "proj_qmm.o", x_loc = 0 : i64, y_loc = 2 : i64} {
            %c0_47 = arith.constant 0 : index
            %10 = arith.addi %c0_47, %arg17 : index
            %c12 = arith.constant 12 : index
            %c3_48 = arith.constant 3 : index
            %c48 = arith.constant 48 : index
            %c3_49 = arith.constant 3 : index
            %c3_50 = arith.constant 3 : index
            %c8 = arith.constant 8 : index
            %c3_51 = arith.constant 3 : index
            %c24_52 = arith.constant 24 : index
            %c0_53 = arith.constant 0 : index
            %c1_54 = arith.constant 1 : index
            %c2_55 = arith.constant 2 : index
            %c1_56 = arith.constant 1 : index
            %c2_57 = arith.constant 2 : index
            %c0_i32_58 = arith.constant 0 : i32
            %11 = arith.index_cast %arg21 : i32 to index
            %c1_59 = arith.constant 1 : index
            %c1_60 = arith.constant 1 : index
            %12 = scf.index_switch %11 -> index 
            case 0 {
              scf.yield %c1_60 : index
            }
            default {
              %c4_63 = arith.constant 4 : index
              scf.yield %c4_63 : index
            }
            %c0_61 = arith.constant 0 : index
            %c1_62 = arith.constant 1 : index
            scf.for %arg22 = %c0_61 to %12 step %c1_62 {
              %c27 = arith.constant 27 : index
              %13 = scf.index_switch %11 -> index 
              case 0 {
                scf.yield %c27 : index
              }
              default {
                %16 = scf.index_switch %arg22 -> index 
                case 0 {
                  scf.yield %c12 : index
                }
                case 1 {
                  scf.yield %c3_48 : index
                }
                case 2 {
                  scf.yield %c48 : index
                }
                default {
                  scf.yield %c3_49 : index
                }
                scf.yield %16 : index
              }
              %c3_63 = arith.constant 3 : index
              %14 = scf.index_switch %11 -> index 
              case 0 {
                scf.yield %c3_63 : index
              }
              default {
                %16 = scf.index_switch %arg22 -> index 
                case 0 {
                  scf.yield %c3_50 : index
                }
                case 1 {
                  scf.yield %c8 : index
                }
                case 2 {
                  scf.yield %c3_51 : index
                }
                default {
                  scf.yield %c24_52 : index
                }
                scf.yield %16 : index
              }
              %15 = scf.index_switch %11 -> index 
              case 0 {
                scf.yield %c1_59 : index
              }
              default {
                %16 = scf.index_switch %arg22 -> index 
                case 0 {
                  scf.yield %c0_53 : index
                }
                case 1 {
                  scf.yield %c1_54 : index
                }
                case 2 {
                  scf.yield %c2_55 : index
                }
                default {
                  scf.yield %c1_56 : index
                }
                scf.yield %16 : index
              }
              %c0_64 = arith.constant 0 : index
              %c1_65 = arith.constant 1 : index
              scf.for %arg23 = %c0_64 to %13 step %c1_65 {
                %16 = arith.muli %14, %c2_57 : index
                %alloc_66 = memref.alloc() : memref<32xf32, 2 : i32>
                func.call @proj_qmm_zero(%alloc_66, %arg21) : (memref<32xf32, 2 : i32>, i32) -> ()
                %c0_67 = arith.constant 0 : index
                %c1_68 = arith.constant 1 : index
                scf.for %arg24 = %c0_67 to %16 step %c1_68 {
                  %alloc_71 = memref.alloc() : memref<256xbf16, 2 : i32>
                  air.channel.get  @inX[%10, %arg18] (%alloc_71[] [] []) : (memref<256xbf16, 2 : i32>)
                  %alloc_72 = memref.alloc() : memref<2560xbf16, 2 : i32>
                  air.channel.get  @wL2ToL1[%10, %arg18] (%alloc_72[] [] []) : (memref<2560xbf16, 2 : i32>)
                  func.call @proj_qmm_acc256(%alloc_71, %alloc_72, %alloc_66) : (memref<256xbf16, 2 : i32>, memref<2560xbf16, 2 : i32>, memref<32xf32, 2 : i32>) -> ()
                  memref.dealloc %alloc_71 : memref<256xbf16, 2 : i32>
                  memref.dealloc %alloc_72 : memref<2560xbf16, 2 : i32>
                }
                %alloc_69 = memref.alloc() : memref<48xbf16, 2 : i32>
                func.call @proj_qmm_flush_row(%alloc_66, %alloc_69, %c0_i32_58) : (memref<32xf32, 2 : i32>, memref<48xbf16, 2 : i32>, i32) -> ()
                %c14 = arith.constant 14 : index
                %c34 = arith.constant 34 : index
                %c1_70 = arith.constant 1 : index
                air.channel.put  @outA[%10, %arg18] (%alloc_69[%c14] [%c34] [%c1_70]) dest(%15) : (memref<48xbf16, 2 : i32>)
                memref.dealloc %alloc_69 : memref<48xbf16, 2 : i32>
                memref.dealloc %alloc_66 : memref<32xf32, 2 : i32>
              }
            }
          }
          %c2_43 = arith.constant 2 : index
          %c4_44 = arith.constant 4 : index
          air.herd @proj_blk1  tile (%arg17, %arg18) in (%arg19=%c2_43, %arg20=%c4_44) args(%arg21=%8) : i32 attributes {link_with = "proj_qmm.o", x_loc = 6 : i64, y_loc = 2 : i64} {
            %c2_47 = arith.constant 2 : index
            %10 = arith.addi %c2_47, %arg17 : index
            %c12 = arith.constant 12 : index
            %c3_48 = arith.constant 3 : index
            %c48 = arith.constant 48 : index
            %c3_49 = arith.constant 3 : index
            %c3_50 = arith.constant 3 : index
            %c8 = arith.constant 8 : index
            %c3_51 = arith.constant 3 : index
            %c24_52 = arith.constant 24 : index
            %c0_53 = arith.constant 0 : index
            %c1_54 = arith.constant 1 : index
            %c2_55 = arith.constant 2 : index
            %c1_56 = arith.constant 1 : index
            %c2_57 = arith.constant 2 : index
            %c0_i32_58 = arith.constant 0 : i32
            %11 = arith.index_cast %arg21 : i32 to index
            %c1_59 = arith.constant 1 : index
            %c1_60 = arith.constant 1 : index
            %12 = scf.index_switch %11 -> index 
            case 0 {
              scf.yield %c1_60 : index
            }
            default {
              %c4_63 = arith.constant 4 : index
              scf.yield %c4_63 : index
            }
            %c0_61 = arith.constant 0 : index
            %c1_62 = arith.constant 1 : index
            scf.for %arg22 = %c0_61 to %12 step %c1_62 {
              %c27 = arith.constant 27 : index
              %13 = scf.index_switch %11 -> index 
              case 0 {
                scf.yield %c27 : index
              }
              default {
                %16 = scf.index_switch %arg22 -> index 
                case 0 {
                  scf.yield %c12 : index
                }
                case 1 {
                  scf.yield %c3_48 : index
                }
                case 2 {
                  scf.yield %c48 : index
                }
                default {
                  scf.yield %c3_49 : index
                }
                scf.yield %16 : index
              }
              %c3_63 = arith.constant 3 : index
              %14 = scf.index_switch %11 -> index 
              case 0 {
                scf.yield %c3_63 : index
              }
              default {
                %16 = scf.index_switch %arg22 -> index 
                case 0 {
                  scf.yield %c3_50 : index
                }
                case 1 {
                  scf.yield %c8 : index
                }
                case 2 {
                  scf.yield %c3_51 : index
                }
                default {
                  scf.yield %c24_52 : index
                }
                scf.yield %16 : index
              }
              %15 = scf.index_switch %11 -> index 
              case 0 {
                scf.yield %c1_59 : index
              }
              default {
                %16 = scf.index_switch %arg22 -> index 
                case 0 {
                  scf.yield %c0_53 : index
                }
                case 1 {
                  scf.yield %c1_54 : index
                }
                case 2 {
                  scf.yield %c2_55 : index
                }
                default {
                  scf.yield %c1_56 : index
                }
                scf.yield %16 : index
              }
              %c0_64 = arith.constant 0 : index
              %c1_65 = arith.constant 1 : index
              scf.for %arg23 = %c0_64 to %13 step %c1_65 {
                %16 = arith.muli %14, %c2_57 : index
                %alloc_66 = memref.alloc() : memref<32xf32, 2 : i32>
                func.call @proj_qmm_zero(%alloc_66, %arg21) : (memref<32xf32, 2 : i32>, i32) -> ()
                %c0_67 = arith.constant 0 : index
                %c1_68 = arith.constant 1 : index
                scf.for %arg24 = %c0_67 to %16 step %c1_68 {
                  %alloc_71 = memref.alloc() : memref<256xbf16, 2 : i32>
                  air.channel.get  @inX[%10, %arg18] (%alloc_71[] [] []) : (memref<256xbf16, 2 : i32>)
                  %alloc_72 = memref.alloc() : memref<2560xbf16, 2 : i32>
                  air.channel.get  @wL2ToL1[%10, %arg18] (%alloc_72[] [] []) : (memref<2560xbf16, 2 : i32>)
                  func.call @proj_qmm_acc256(%alloc_71, %alloc_72, %alloc_66) : (memref<256xbf16, 2 : i32>, memref<2560xbf16, 2 : i32>, memref<32xf32, 2 : i32>) -> ()
                  memref.dealloc %alloc_71 : memref<256xbf16, 2 : i32>
                  memref.dealloc %alloc_72 : memref<2560xbf16, 2 : i32>
                }
                %alloc_69 = memref.alloc() : memref<48xbf16, 2 : i32>
                func.call @proj_qmm_flush_row(%alloc_66, %alloc_69, %c0_i32_58) : (memref<32xf32, 2 : i32>, memref<48xbf16, 2 : i32>, i32) -> ()
                %c14 = arith.constant 14 : index
                %c34 = arith.constant 34 : index
                %c1_70 = arith.constant 1 : index
                air.channel.put  @outA[%10, %arg18] (%alloc_69[%c14] [%c34] [%c1_70]) dest(%15) : (memref<48xbf16, 2 : i32>)
                memref.dealloc %alloc_69 : memref<48xbf16, 2 : i32>
                memref.dealloc %alloc_66 : memref<32xf32, 2 : i32>
              }
            }
          }
          %c1_45 = arith.constant 1 : index
          %c1_46 = arith.constant 1 : index
          air.herd @rms  tile (%arg17, %arg18) in (%arg19=%c1_45, %arg20=%c1_46) args(%arg21=%8) : i32 attributes {link_with = "rms_residual.o", x_loc = 2 : i64, y_loc = 2 : i64} {
            %10 = arith.index_cast %arg21 : i32 to index
            scf.index_switch %10 
            case 0 {
              %alloc_47 = memref.alloc() : memref<1536xbf16, 2 : i32>
              %c0_48 = arith.constant 0 : index
              air.channel.get  @rmsX[%c0_48] (%alloc_47[] [] []) : (memref<1536xbf16, 2 : i32>)
              %alloc_49 = memref.alloc() : memref<3072xbf16, 2 : i32>
              %c0_50 = arith.constant 0 : index
              air.channel.get  @rmsW[%c0_50] (%alloc_49[] [] []) : (memref<3072xbf16, 2 : i32>)
              %alloc_51 = memref.alloc() : memref<3072xbf16, 2 : i32>
              %c0_52 = arith.constant 0 : index
              air.channel.get  @rmsW2[%c0_52] (%alloc_51[] [] []) : (memref<3072xbf16, 2 : i32>)
              memref.dealloc %alloc_51 : memref<3072xbf16, 2 : i32>
              %alloc_53 = memref.alloc() : memref<1536xbf16, 2 : i32>
              %alloc_54 = memref.alloc() : memref<1536xbf16, 2 : i32>
              func.call @rms_norm_hi_aie(%alloc_53, %alloc_47, %alloc_49, %arg21) : (memref<1536xbf16, 2 : i32>, memref<1536xbf16, 2 : i32>, memref<3072xbf16, 2 : i32>, i32) -> ()
              %c0_55 = arith.constant 0 : index
              %c9 = arith.constant 9 : index
              %c1_56 = arith.constant 1 : index
              scf.for %arg22 = %c0_55 to %c9 step %c1_56 {
                %c0_57 = arith.constant 0 : index
                %c3_58 = arith.constant 3 : index
                %c1_59 = arith.constant 1 : index
                scf.for %arg23 = %c0_57 to %c3_58 step %c1_59 {
                  air.channel.put  @xnorm[] (%alloc_53[0] [1536] [1]) : (memref<1536xbf16, 2 : i32>)
                }
                %c0_60 = arith.constant 0 : index
                %c1_61 = arith.constant 1 : index
                %c0_62 = arith.constant 0 : index
                %c1536 = arith.constant 1536 : index
                %c1_63 = arith.constant 1 : index
                air.channel.get  @outY[%c0_60, %c1_61] (%alloc_54[%c0_62] [%c1536] [%c1_63]) : (memref<1536xbf16, 2 : i32>)
                %c0_64 = arith.constant 0 : index
                %c1536_65 = arith.constant 1536 : index
                %c1_66 = arith.constant 1 : index
                air.channel.put  @layerOut[] (%alloc_54[%c0_64] [%c1536_65] [%c1_66]) : (memref<1536xbf16, 2 : i32>)
              }
              memref.dealloc %alloc_47 : memref<1536xbf16, 2 : i32>
              memref.dealloc %alloc_49 : memref<3072xbf16, 2 : i32>
              memref.dealloc %alloc_53 : memref<1536xbf16, 2 : i32>
              memref.dealloc %alloc_54 : memref<1536xbf16, 2 : i32>
              scf.yield
            }
            default {
              %alloc_47 = memref.alloc() : memref<1536xbf16, 2 : i32>
              %c0_48 = arith.constant 0 : index
              air.channel.get  @rmsX[%c0_48] (%alloc_47[] [] []) : (memref<1536xbf16, 2 : i32>)
              %alloc_49 = memref.alloc() : memref<3072xbf16, 2 : i32>
              %c0_50 = arith.constant 0 : index
              air.channel.get  @rmsW[%c0_50] (%alloc_49[] [] []) : (memref<3072xbf16, 2 : i32>)
              %alloc_51 = memref.alloc() : memref<3072xbf16, 2 : i32>
              %c0_52 = arith.constant 0 : index
              air.channel.get  @rmsW2[%c0_52] (%alloc_51[] [] []) : (memref<3072xbf16, 2 : i32>)
              %alloc_53 = memref.alloc() : memref<1536xbf16, 2 : i32>
              %alloc_54 = memref.alloc() : memref<1536xbf16, 2 : i32>
              %alloc_55 = memref.alloc() : memref<1536xbf16, 2 : i32>
              %alloc_56 = memref.alloc() : memref<1536xbf16, 2 : i32>
              func.call @rms_norm_lo_aie(%alloc_53, %alloc_47, %alloc_49, %arg21) : (memref<1536xbf16, 2 : i32>, memref<1536xbf16, 2 : i32>, memref<3072xbf16, 2 : i32>, i32) -> ()
              %c0_57 = arith.constant 0 : index
              %c12 = arith.constant 12 : index
              %c1_58 = arith.constant 1 : index
              scf.for %arg22 = %c0_57 to %c12 step %c1_58 {
                air.channel.put  @xnorm[] (%alloc_53[0] [1536] [1]) : (memref<1536xbf16, 2 : i32>)
              }
              %c0_59 = arith.constant 0 : index
              %c1_60 = arith.constant 1 : index
              %c0_61 = arith.constant 0 : index
              %c1536 = arith.constant 1536 : index
              %c1_62 = arith.constant 1 : index
              air.channel.get  @outY[%c0_59, %c1_60] (%alloc_54[%c0_61] [%c1536] [%c1_62]) : (memref<1536xbf16, 2 : i32>)
              func.call @rms_norm_hi_aie(%alloc_55, %alloc_54, %alloc_49, %arg21) : (memref<1536xbf16, 2 : i32>, memref<1536xbf16, 2 : i32>, memref<3072xbf16, 2 : i32>, i32) -> ()
              func.call @residual_add_aie(%alloc_56, %alloc_47, %alloc_55) : (memref<1536xbf16, 2 : i32>, memref<1536xbf16, 2 : i32>, memref<1536xbf16, 2 : i32>) -> ()
              memref.dealloc %alloc_49 : memref<3072xbf16, 2 : i32>
              func.call @rms_norm_lo_aie(%alloc_53, %alloc_56, %alloc_51, %arg21) : (memref<1536xbf16, 2 : i32>, memref<1536xbf16, 2 : i32>, memref<3072xbf16, 2 : i32>, i32) -> ()
              %c0_63 = arith.constant 0 : index
              %c48 = arith.constant 48 : index
              %c1_64 = arith.constant 1 : index
              scf.for %arg22 = %c0_63 to %c48 step %c1_64 {
                air.channel.put  @xnorm[] (%alloc_53[0] [1536] [1]) : (memref<1536xbf16, 2 : i32>)
              }
              memref.dealloc %alloc_53 : memref<1536xbf16, 2 : i32>
              %c0_65 = arith.constant 0 : index
              %c1_66 = arith.constant 1 : index
              %c0_67 = arith.constant 0 : index
              %c1536_68 = arith.constant 1536 : index
              %c1_69 = arith.constant 1 : index
              air.channel.get  @outY[%c0_65, %c1_66] (%alloc_54[%c0_67] [%c1536_68] [%c1_69]) : (memref<1536xbf16, 2 : i32>)
              func.call @rms_norm_hi_aie(%alloc_55, %alloc_54, %alloc_51, %arg21) : (memref<1536xbf16, 2 : i32>, memref<1536xbf16, 2 : i32>, memref<3072xbf16, 2 : i32>, i32) -> ()
              func.call @residual_add_aie(%alloc_47, %alloc_56, %alloc_55) : (memref<1536xbf16, 2 : i32>, memref<1536xbf16, 2 : i32>, memref<1536xbf16, 2 : i32>) -> ()
              memref.dealloc %alloc_56 : memref<1536xbf16, 2 : i32>
              memref.dealloc %alloc_54 : memref<1536xbf16, 2 : i32>
              memref.dealloc %alloc_55 : memref<1536xbf16, 2 : i32>
              memref.dealloc %alloc_51 : memref<3072xbf16, 2 : i32>
              %c0_70 = arith.constant 0 : index
              %c1536_71 = arith.constant 1536 : index
              %c1_72 = arith.constant 1 : index
              air.channel.put  @layerOut[] (%alloc_47[%c0_70] [%c1536_71] [%c1_72]) : (memref<1536xbf16, 2 : i32>)
              memref.dealloc %alloc_47 : memref<1536xbf16, 2 : i32>
            }
          }
        }
      }
    }
    return
  }
}
