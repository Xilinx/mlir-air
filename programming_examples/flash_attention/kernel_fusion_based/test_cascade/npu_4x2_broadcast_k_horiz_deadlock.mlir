#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module {
  aie.device(npu2) @s {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %shim_noc_tile_3_0 = aie.tile(3, 0)
    %shim_noc_tile_4_0 = aie.tile(4, 0)
    %shim_noc_tile_5_0 = aie.tile(5, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %mem_tile_2_1 = aie.tile(2, 1)
    %mem_tile_3_1 = aie.tile(3, 1)
    %mem_tile_4_1 = aie.tile(4, 1)
    %mem_tile_5_1 = aie.tile(5, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_3 = aie.tile(1, 3)
    %tile_2_3 = aie.tile(2, 3)
    %tile_3_3 = aie.tile(3, 3)
    %lock_5_1 = aie.lock(%mem_tile_5_1, 1) {init = 1 : i32}
    %lock_5_1_0 = aie.lock(%mem_tile_5_1, 0) {init = 0 : i32}
    %lock_4_1 = aie.lock(%mem_tile_4_1, 1) {init = 1 : i32}
    %lock_4_1_1 = aie.lock(%mem_tile_4_1, 0) {init = 0 : i32}
    %lock_3_1 = aie.lock(%mem_tile_3_1, 1) {init = 1 : i32}
    %lock_3_1_2 = aie.lock(%mem_tile_3_1, 0) {init = 0 : i32}
    %lock_2_1 = aie.lock(%mem_tile_2_1, 1) {init = 1 : i32}
    %lock_2_1_3 = aie.lock(%mem_tile_2_1, 0) {init = 0 : i32}
    %lock_1_1 = aie.lock(%mem_tile_1_1, 1) {init = 1 : i32}
    %lock_1_1_4 = aie.lock(%mem_tile_1_1, 0) {init = 0 : i32}
    %lock_0_1 = aie.lock(%mem_tile_0_1, 1) {init = 1 : i32}
    %lock_0_1_5 = aie.lock(%mem_tile_0_1, 0) {init = 0 : i32}
    %lock_0_2 = aie.lock(%tile_0_2, 7) {init = 1 : i32}
    %lock_0_2_6 = aie.lock(%tile_0_2, 6) {init = 0 : i32}
    %lock_0_2_7 = aie.lock(%tile_0_2, 5) {init = 1 : i32}
    %lock_0_2_8 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_9 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_10 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_11 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_12 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %lock_1_2 = aie.lock(%tile_1_2, 7) {init = 1 : i32}
    %lock_1_2_13 = aie.lock(%tile_1_2, 6) {init = 0 : i32}
    %lock_1_2_14 = aie.lock(%tile_1_2, 5) {init = 1 : i32}
    %lock_1_2_15 = aie.lock(%tile_1_2, 4) {init = 0 : i32}
    %lock_1_2_16 = aie.lock(%tile_1_2, 3) {init = 1 : i32}
    %lock_1_2_17 = aie.lock(%tile_1_2, 2) {init = 0 : i32}
    %lock_1_2_18 = aie.lock(%tile_1_2, 1) {init = 1 : i32}
    %lock_1_2_19 = aie.lock(%tile_1_2, 0) {init = 0 : i32}
    %lock_2_2 = aie.lock(%tile_2_2, 7) {init = 1 : i32}
    %lock_2_2_20 = aie.lock(%tile_2_2, 6) {init = 0 : i32}
    %lock_2_2_21 = aie.lock(%tile_2_2, 5) {init = 1 : i32}
    %lock_2_2_22 = aie.lock(%tile_2_2, 4) {init = 0 : i32}
    %lock_2_2_23 = aie.lock(%tile_2_2, 3) {init = 1 : i32}
    %lock_2_2_24 = aie.lock(%tile_2_2, 2) {init = 0 : i32}
    %lock_2_2_25 = aie.lock(%tile_2_2, 1) {init = 1 : i32}
    %lock_2_2_26 = aie.lock(%tile_2_2, 0) {init = 0 : i32}
    %lock_3_2 = aie.lock(%tile_3_2, 7) {init = 1 : i32}
    %lock_3_2_27 = aie.lock(%tile_3_2, 6) {init = 0 : i32}
    %lock_3_2_28 = aie.lock(%tile_3_2, 5) {init = 1 : i32}
    %lock_3_2_29 = aie.lock(%tile_3_2, 4) {init = 0 : i32}
    %lock_3_2_30 = aie.lock(%tile_3_2, 3) {init = 1 : i32}
    %lock_3_2_31 = aie.lock(%tile_3_2, 2) {init = 0 : i32}
    %lock_3_2_32 = aie.lock(%tile_3_2, 1) {init = 1 : i32}
    %lock_3_2_33 = aie.lock(%tile_3_2, 0) {init = 0 : i32}
    %lock_0_3 = aie.lock(%tile_0_3, 5) {init = 1 : i32}
    %lock_0_3_34 = aie.lock(%tile_0_3, 4) {init = 0 : i32}
    %lock_0_3_35 = aie.lock(%tile_0_3, 3) {init = 1 : i32}
    %lock_0_3_36 = aie.lock(%tile_0_3, 2) {init = 0 : i32}
    %lock_0_3_37 = aie.lock(%tile_0_3, 1) {init = 1 : i32}
    %lock_0_3_38 = aie.lock(%tile_0_3, 0) {init = 0 : i32}
    %lock_1_3 = aie.lock(%tile_1_3, 5) {init = 1 : i32}
    %lock_1_3_39 = aie.lock(%tile_1_3, 4) {init = 0 : i32}
    %lock_1_3_40 = aie.lock(%tile_1_3, 3) {init = 1 : i32}
    %lock_1_3_41 = aie.lock(%tile_1_3, 2) {init = 0 : i32}
    %lock_1_3_42 = aie.lock(%tile_1_3, 1) {init = 1 : i32}
    %lock_1_3_43 = aie.lock(%tile_1_3, 0) {init = 0 : i32}
    %lock_2_3 = aie.lock(%tile_2_3, 5) {init = 1 : i32}
    %lock_2_3_44 = aie.lock(%tile_2_3, 4) {init = 0 : i32}
    %lock_2_3_45 = aie.lock(%tile_2_3, 3) {init = 1 : i32}
    %lock_2_3_46 = aie.lock(%tile_2_3, 2) {init = 0 : i32}
    %lock_2_3_47 = aie.lock(%tile_2_3, 1) {init = 1 : i32}
    %lock_2_3_48 = aie.lock(%tile_2_3, 0) {init = 0 : i32}
    %lock_3_3 = aie.lock(%tile_3_3, 5) {init = 1 : i32}
    %lock_3_3_49 = aie.lock(%tile_3_3, 4) {init = 0 : i32}
    %lock_3_3_50 = aie.lock(%tile_3_3, 3) {init = 1 : i32}
    %lock_3_3_51 = aie.lock(%tile_3_3, 2) {init = 0 : i32}
    %lock_3_3_52 = aie.lock(%tile_3_3, 1) {init = 1 : i32}
    %lock_3_3_53 = aie.lock(%tile_3_3, 0) {init = 0 : i32}
    %buf77 = aie.buffer(%mem_tile_0_1) {sym_name = "buf77"} : memref<64x64xbf16, 1 : i32> 
    %buf76 = aie.buffer(%mem_tile_1_1) {sym_name = "buf76"} : memref<64x64xbf16, 1 : i32> 
    %buf75 = aie.buffer(%mem_tile_2_1) {sym_name = "buf75"} : memref<64x64xbf16, 1 : i32> 
    %buf74 = aie.buffer(%mem_tile_3_1) {sym_name = "buf74"} : memref<64x64xbf16, 1 : i32> 
    %buf73 = aie.buffer(%mem_tile_4_1) {sym_name = "buf73"} : memref<64x64xbf16, 1 : i32> 
    %buf72 = aie.buffer(%mem_tile_5_1) {sym_name = "buf72"} : memref<64x64xbf16, 1 : i32> 
    %buf71 = aie.buffer(%tile_3_3) {sym_name = "buf71"} : memref<64x1xbf16, 2 : i32> 
    %buf70 = aie.buffer(%tile_3_3) {sym_name = "buf70"} : memref<64x1xbf16, 2 : i32> 
    %buf69 = aie.buffer(%tile_3_3) {sym_name = "buf69"} : memref<64x64xbf16, 2 : i32> 
    %buf68 = aie.buffer(%tile_3_3) {sym_name = "buf68"} : memref<64x64xbf16, 2 : i32> 
    %buf67 = aie.buffer(%tile_3_3) {sym_name = "buf67"} : memref<64x64xbf16, 2 : i32> 
    %buf66 = aie.buffer(%tile_3_3) {sym_name = "buf66"} : memref<64x64xbf16, 2 : i32> 
    %buf65 = aie.buffer(%tile_3_3) {sym_name = "buf65"} : memref<64x64xbf16, 2 : i32> 
    %buf64 = aie.buffer(%tile_3_3) {sym_name = "buf64"} : memref<64x1xbf16, 2 : i32> 
    %buf63 = aie.buffer(%tile_3_3) {sym_name = "buf63"} : memref<64x1xbf16, 2 : i32> 
    %buf62 = aie.buffer(%tile_2_3) {sym_name = "buf62"} : memref<64x1xbf16, 2 : i32> 
    %buf61 = aie.buffer(%tile_2_3) {sym_name = "buf61"} : memref<64x1xbf16, 2 : i32> 
    %buf60 = aie.buffer(%tile_2_3) {sym_name = "buf60"} : memref<64x64xbf16, 2 : i32> 
    %buf59 = aie.buffer(%tile_2_3) {sym_name = "buf59"} : memref<64x64xbf16, 2 : i32> 
    %buf58 = aie.buffer(%tile_2_3) {sym_name = "buf58"} : memref<64x64xbf16, 2 : i32> 
    %buf57 = aie.buffer(%tile_2_3) {sym_name = "buf57"} : memref<64x64xbf16, 2 : i32> 
    %buf56 = aie.buffer(%tile_2_3) {sym_name = "buf56"} : memref<64x64xbf16, 2 : i32> 
    %buf55 = aie.buffer(%tile_2_3) {sym_name = "buf55"} : memref<64x1xbf16, 2 : i32> 
    %buf54 = aie.buffer(%tile_2_3) {sym_name = "buf54"} : memref<64x1xbf16, 2 : i32> 
    %buf53 = aie.buffer(%tile_1_3) {sym_name = "buf53"} : memref<64x1xbf16, 2 : i32> 
    %buf52 = aie.buffer(%tile_1_3) {sym_name = "buf52"} : memref<64x1xbf16, 2 : i32> 
    %buf51 = aie.buffer(%tile_1_3) {sym_name = "buf51"} : memref<64x64xbf16, 2 : i32> 
    %buf50 = aie.buffer(%tile_1_3) {sym_name = "buf50"} : memref<64x64xbf16, 2 : i32> 
    %buf49 = aie.buffer(%tile_1_3) {sym_name = "buf49"} : memref<64x64xbf16, 2 : i32> 
    %buf48 = aie.buffer(%tile_1_3) {sym_name = "buf48"} : memref<64x64xbf16, 2 : i32> 
    %buf47 = aie.buffer(%tile_1_3) {sym_name = "buf47"} : memref<64x64xbf16, 2 : i32> 
    %buf46 = aie.buffer(%tile_1_3) {sym_name = "buf46"} : memref<64x1xbf16, 2 : i32> 
    %buf45 = aie.buffer(%tile_1_3) {sym_name = "buf45"} : memref<64x1xbf16, 2 : i32> 
    %buf44 = aie.buffer(%tile_0_3) {sym_name = "buf44"} : memref<64x1xbf16, 2 : i32> 
    %buf43 = aie.buffer(%tile_0_3) {sym_name = "buf43"} : memref<64x1xbf16, 2 : i32> 
    %buf42 = aie.buffer(%tile_0_3) {sym_name = "buf42"} : memref<64x64xbf16, 2 : i32> 
    %buf41 = aie.buffer(%tile_0_3) {sym_name = "buf41"} : memref<64x64xbf16, 2 : i32> 
    %buf40 = aie.buffer(%tile_0_3) {sym_name = "buf40"} : memref<64x64xbf16, 2 : i32> 
    %buf39 = aie.buffer(%tile_0_3) {sym_name = "buf39"} : memref<64x64xbf16, 2 : i32> 
    %buf38 = aie.buffer(%tile_0_3) {sym_name = "buf38"} : memref<64x64xbf16, 2 : i32> 
    %buf37 = aie.buffer(%tile_0_3) {sym_name = "buf37"} : memref<64x1xbf16, 2 : i32> 
    %buf36 = aie.buffer(%tile_0_3) {sym_name = "buf36"} : memref<64x1xbf16, 2 : i32> 
    %buf35 = aie.buffer(%tile_3_2) {sym_name = "buf35"} : memref<64x1xbf16, 2 : i32> 
    %buf34 = aie.buffer(%tile_3_2) {sym_name = "buf34"} : memref<64x1xbf16, 2 : i32> 
    %buf33 = aie.buffer(%tile_3_2) {sym_name = "buf33"} : memref<64x64xbf16, 2 : i32> 
    %buf32 = aie.buffer(%tile_3_2) {sym_name = "buf32"} : memref<64x64xbf16, 2 : i32> 
    %buf31 = aie.buffer(%tile_3_2) {sym_name = "buf31"} : memref<64x64xbf16, 2 : i32> 
    %buf30 = aie.buffer(%tile_3_2) {sym_name = "buf30"} : memref<64x64xbf16, 2 : i32> 
    %buf29 = aie.buffer(%tile_3_2) {sym_name = "buf29"} : memref<64x64xbf16, 2 : i32> 
    %buf28 = aie.buffer(%tile_3_2) {sym_name = "buf28"} : memref<64x1xbf16, 2 : i32> 
    %buf27 = aie.buffer(%tile_3_2) {sym_name = "buf27"} : memref<64x1xbf16, 2 : i32> 
    %buf26 = aie.buffer(%tile_2_2) {sym_name = "buf26"} : memref<64x1xbf16, 2 : i32> 
    %buf25 = aie.buffer(%tile_2_2) {sym_name = "buf25"} : memref<64x1xbf16, 2 : i32> 
    %buf24 = aie.buffer(%tile_2_2) {sym_name = "buf24"} : memref<64x64xbf16, 2 : i32> 
    %buf23 = aie.buffer(%tile_2_2) {sym_name = "buf23"} : memref<64x64xbf16, 2 : i32> 
    %buf22 = aie.buffer(%tile_2_2) {sym_name = "buf22"} : memref<64x64xbf16, 2 : i32> 
    %buf21 = aie.buffer(%tile_2_2) {sym_name = "buf21"} : memref<64x64xbf16, 2 : i32> 
    %buf20 = aie.buffer(%tile_2_2) {sym_name = "buf20"} : memref<64x64xbf16, 2 : i32> 
    %buf19 = aie.buffer(%tile_2_2) {sym_name = "buf19"} : memref<64x1xbf16, 2 : i32> 
    %buf18 = aie.buffer(%tile_2_2) {sym_name = "buf18"} : memref<64x1xbf16, 2 : i32> 
    %buf17 = aie.buffer(%tile_1_2) {sym_name = "buf17"} : memref<64x1xbf16, 2 : i32> 
    %buf16 = aie.buffer(%tile_1_2) {sym_name = "buf16"} : memref<64x1xbf16, 2 : i32> 
    %buf15 = aie.buffer(%tile_1_2) {sym_name = "buf15"} : memref<64x64xbf16, 2 : i32> 
    %buf14 = aie.buffer(%tile_1_2) {sym_name = "buf14"} : memref<64x64xbf16, 2 : i32> 
    %buf13 = aie.buffer(%tile_1_2) {sym_name = "buf13"} : memref<64x64xbf16, 2 : i32> 
    %buf12 = aie.buffer(%tile_1_2) {sym_name = "buf12"} : memref<64x64xbf16, 2 : i32> 
    %buf11 = aie.buffer(%tile_1_2) {sym_name = "buf11"} : memref<64x64xbf16, 2 : i32> 
    %buf10 = aie.buffer(%tile_1_2) {sym_name = "buf10"} : memref<64x1xbf16, 2 : i32> 
    %buf9 = aie.buffer(%tile_1_2) {sym_name = "buf9"} : memref<64x1xbf16, 2 : i32> 
    %buf8 = aie.buffer(%tile_0_2) {sym_name = "buf8"} : memref<64x1xbf16, 2 : i32> 
    %buf7 = aie.buffer(%tile_0_2) {sym_name = "buf7"} : memref<64x1xbf16, 2 : i32> 
    %buf6 = aie.buffer(%tile_0_2) {sym_name = "buf6"} : memref<64x64xbf16, 2 : i32> 
    %buf5 = aie.buffer(%tile_0_2) {sym_name = "buf5"} : memref<64x64xbf16, 2 : i32> 
    %buf4 = aie.buffer(%tile_0_2) {sym_name = "buf4"} : memref<64x64xbf16, 2 : i32> 
    %buf3 = aie.buffer(%tile_0_2) {sym_name = "buf3"} : memref<64x64xbf16, 2 : i32> 
    %buf2 = aie.buffer(%tile_0_2) {sym_name = "buf2"} : memref<64x64xbf16, 2 : i32> 
    %buf1 = aie.buffer(%tile_0_2) {sym_name = "buf1"} : memref<64x1xbf16, 2 : i32> 
    %buf0 = aie.buffer(%tile_0_2) {sym_name = "buf0"} : memref<64x1xbf16, 2 : i32> 
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<256x64xbf16>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<256x64xbf16>
    %__air_external_buffer_2 = aie.external_buffer {sym_name = "__air_external_buffer_2"} : memref<256x64xbf16>
    %__air_external_buffer_3 = aie.external_buffer {sym_name = "__air_external_buffer_3"} : memref<256x64xbf16>
    %mem_3_3 = aie.mem(%tile_3_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_3_52, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf66 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_53, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf68 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_3_49, Release, 1)
      aie.next_bd ^bb4
    }
    %core_3_3 = aie.core(%tile_3_3) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf69) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf71) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf70) : (memref<64x1xbf16, 2 : i32>) -> ()
      // Q: receive once, copy to saved buffer (outside chunk loop)
      aie.use_lock(%lock_3_3_53, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf66, %buf67) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_3_52, Release, 1)
      // K: per chunk
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_56 = memref.collapse_shape %buf65 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_56) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3_53, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_3_51, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_3_49, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf67, %buf66, %collapse_shape_56) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        func.call @fused_softmax(%collapse_shape_56, %buf70, %buf64, %buf63) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf63, %buf69) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_56, %buf68, %buf69) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf71, %buf63, %buf64) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf64, %buf71) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_3_50, Release, 1)
        aie.use_lock(%lock_3_3_52, Release, 1)
        aie.use_lock(%lock_3_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf69 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_54 = memref.collapse_shape %buf70 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_54[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_55 = memref.collapse_shape %buf71 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_55[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_3 = aie.mem(%tile_2_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_3_47, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf57 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_48, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf59 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_3_44, Release, 1)
      aie.next_bd ^bb4
    }
    %core_2_3 = aie.core(%tile_2_3) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf60) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf62) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf61) : (memref<64x1xbf16, 2 : i32>) -> ()
      // Q: receive once, copy to saved buffer
      aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf57, %buf58) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_3_47, Release, 1)
      // K: per chunk
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_56 = memref.collapse_shape %buf56 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_56) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3_48, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_3_46, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_3_44, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf58, %buf57, %collapse_shape_56) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        func.call @fused_softmax(%collapse_shape_56, %buf61, %buf55, %buf54) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf54, %buf60) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_56, %buf59, %buf60) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf62, %buf54, %buf55) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf55, %buf62) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_3_45, Release, 1)
        aie.use_lock(%lock_2_3_47, Release, 1)
        aie.use_lock(%lock_2_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf60 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_54 = memref.collapse_shape %buf61 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_54[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_55 = memref.collapse_shape %buf62 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_55[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_3 = aie.mem(%tile_1_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_3_42, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf48 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_43, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf50 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_3_39, Release, 1)
      aie.next_bd ^bb4
    }
    %core_1_3 = aie.core(%tile_1_3) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf51) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf53) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf52) : (memref<64x1xbf16, 2 : i32>) -> ()
      // Q: receive once, copy to saved buffer
      aie.use_lock(%lock_1_3_43, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf48, %buf49) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_3_42, Release, 1)
      // K: per chunk
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_56 = memref.collapse_shape %buf47 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_56) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3_43, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_3_41, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_3_39, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf49, %buf48, %collapse_shape_56) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        func.call @fused_softmax(%collapse_shape_56, %buf52, %buf46, %buf45) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf45, %buf51) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_56, %buf50, %buf51) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf53, %buf45, %buf46) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf46, %buf53) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_3_40, Release, 1)
        aie.use_lock(%lock_1_3_42, Release, 1)
        aie.use_lock(%lock_1_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf51 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_54 = memref.collapse_shape %buf52 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_54[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_55 = memref.collapse_shape %buf53 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_55[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_37, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf39 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_38, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf41 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_3_34, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %c64 = arith.constant 64 : index
      %cst = arith.constant 0.000000e+00 : bf16
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      func.call @zero_fill_gp_bf16(%buf42) : (memref<64x64xbf16, 2 : i32>) -> ()
      func.call @zero_fill_sp_bf16(%buf44) : (memref<64x1xbf16, 2 : i32>) -> ()
      func.call @neg_inf_fill_up_bf16(%buf43) : (memref<64x1xbf16, 2 : i32>) -> ()
      // Q: receive once, copy to saved buffer
      aie.use_lock(%lock_0_3_38, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf39, %buf40) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_3_37, Release, 1)
      // K: per chunk
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_56 = memref.collapse_shape %buf38 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_56) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3_38, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_3_36, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_3_34, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf40, %buf39, %collapse_shape_56) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        func.call @fused_softmax(%collapse_shape_56, %buf43, %buf37, %buf36) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf36, %buf42) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_56, %buf41, %buf42) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf44, %buf36, %buf37) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf37, %buf44) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_3_35, Release, 1)
        aie.use_lock(%lock_0_3_37, Release, 1)
        aie.use_lock(%lock_0_3, Release, 1)
      } {loop_annotation = #loop_annotation}
      %collapse_shape = memref.collapse_shape %buf42 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_54 = memref.collapse_shape %buf43 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_54[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      %collapse_shape_55 = memref.collapse_shape %buf44 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_55[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = vector.transfer_read %subview[%c0], %cst {in_bounds = [true]} : memref<32xbf16, strided<[1], offset: ?>, 2 : i32>, vector<32xbf16>
        aie.put_cascade(%0 : vector<32xbf16>)
      }
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_3_2 = aie.mem(%tile_3_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_33, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf33 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_32, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_2_30, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf30 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_31, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_3_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf32 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_2_27, Release, 1)
      aie.next_bd ^bb6
    }
    %core_3_2 = aie.core(%tile_3_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_2_32, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf33 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_54 = memref.collapse_shape %buf34 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_54[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_55 = memref.collapse_shape %buf35 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_55[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      // Q: receive once, copy to saved buffer (after cascade receive)
      aie.use_lock(%lock_3_2_31, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf30, %buf31) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_30, Release, 1)
      // K: per chunk
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_56 = memref.collapse_shape %buf29 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_56) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_31, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_2_29, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_3_2_27, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf31, %buf30, %collapse_shape_56) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        func.call @fused_softmax(%collapse_shape_56, %buf34, %buf28, %buf27) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf27, %buf33) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_56, %buf32, %buf33) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf35, %buf27, %buf28) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf28, %buf35) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_3_2_28, Release, 1)
        aie.use_lock(%lock_3_2_30, Release, 1)
        aie.use_lock(%lock_3_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf35, %buf33) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_3_2_33, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_2_2 = aie.mem(%tile_2_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_26, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf24 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_25, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_2_23, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf21 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_24, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_2_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf23 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_2_20, Release, 1)
      aie.next_bd ^bb6
    }
    %core_2_2 = aie.core(%tile_2_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_2_25, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf24 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_54 = memref.collapse_shape %buf25 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_54[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_55 = memref.collapse_shape %buf26 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_55[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      // Q: receive once, copy to saved buffer
      aie.use_lock(%lock_2_2_24, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf21, %buf22) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_23, Release, 1)
      // K: per chunk
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_56 = memref.collapse_shape %buf20 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_56) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_24, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_2_22, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_2_2_20, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf22, %buf21, %collapse_shape_56) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        func.call @fused_softmax(%collapse_shape_56, %buf25, %buf19, %buf18) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf18, %buf24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_56, %buf23, %buf24) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf26, %buf18, %buf19) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf19, %buf26) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_2_2_21, Release, 1)
        aie.use_lock(%lock_2_2_23, Release, 1)
        aie.use_lock(%lock_2_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf26, %buf24) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_2_2_26, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_1_2 = aie.mem(%tile_1_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_19, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf15 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_18, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_2_16, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf12 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_17, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf14 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_2_13, Release, 1)
      aie.next_bd ^bb6
    }
    %core_1_2 = aie.core(%tile_1_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_2_18, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf15 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_54 = memref.collapse_shape %buf16 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_54[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_55 = memref.collapse_shape %buf17 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_55[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      // Q: receive once, copy to saved buffer
      aie.use_lock(%lock_1_2_17, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf12, %buf13) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_16, Release, 1)
      // K: per chunk
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_56 = memref.collapse_shape %buf11 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_56) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_17, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_2_15, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_1_2_13, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf13, %buf12, %collapse_shape_56) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        func.call @fused_softmax(%collapse_shape_56, %buf16, %buf10, %buf9) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf9, %buf15) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_56, %buf14, %buf15) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf17, %buf9, %buf10) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf10, %buf17) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_1_2_14, Release, 1)
        aie.use_lock(%lock_1_2_16, Release, 1)
        aie.use_lock(%lock_1_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf17, %buf15) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_1_2_19, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_12, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf6 : memref<64x64xbf16, 2 : i32>, 0, 4096, [<size = 64, stride = 8>, <size = 8, stride = 512>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_11, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb5
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb5)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_9, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf3 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_10, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb3
      %2 = aie.dma_start(S2MM, 1, ^bb6, ^bb2)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf5 : memref<64x64xbf16, 2 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_2_6, Release, 1)
      aie.next_bd ^bb6
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c32 = arith.constant 32 : index
      %c4096 = arith.constant 4096 : index
      %c0_i32 = arith.constant 0 : i32
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_11, AcquireGreaterEqual, 1)
      %collapse_shape = memref.collapse_shape %buf6 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c4096 step %c32 {
        %subview = memref.subview %collapse_shape[%arg0] [32] [1] : memref<4096xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_54 = memref.collapse_shape %buf7 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_54[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      %collapse_shape_55 = memref.collapse_shape %buf8 [[0, 1]] : memref<64x1xbf16, 2 : i32> into memref<64xbf16, 2 : i32>
      scf.for %arg0 = %c0 to %c64 step %c32 {
        %subview = memref.subview %collapse_shape_55[%arg0] [32] [1] : memref<64xbf16, 2 : i32> to memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
        %0 = aie.get_cascade() : vector<32xbf16>
        vector.transfer_write %0, %subview[%c0] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, strided<[1], offset: ?>, 2 : i32>
      }
      // Q: receive once, copy to saved buffer
      aie.use_lock(%lock_0_2_10, AcquireGreaterEqual, 1)
      func.call @copy_tile(%buf3, %buf4) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_9, Release, 1)
      // K: per chunk
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %collapse_shape_56 = memref.collapse_shape %buf2 [[0, 1]] : memref<64x64xbf16, 2 : i32> into memref<4096xbf16, 2 : i32>
        func.call @zero_fill_g_bf16(%collapse_shape_56) : (memref<4096xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_10, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_2_8, AcquireGreaterEqual, 1)
        aie.use_lock(%lock_0_2_6, AcquireGreaterEqual, 1)
        func.call @matmul_a_b_bf16(%buf4, %buf3, %collapse_shape_56) : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>) -> ()
        func.call @fused_softmax(%collapse_shape_56, %buf7, %buf1, %buf0) : (memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @mul_r_gp(%buf0, %buf6) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @matmul_g_b_bf16(%collapse_shape_56, %buf5, %buf6) : (memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
        func.call @accum_sp_r_s(%buf8, %buf0, %buf1) : (memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        func.call @vector_copy_32elems(%c0_i32, %buf1, %buf8) : (i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>) -> ()
        aie.use_lock(%lock_0_2_7, Release, 1)
        aie.use_lock(%lock_0_2_9, Release, 1)
        aie.use_lock(%lock_0_2, Release, 1)
      } {loop_annotation = #loop_annotation}
      func.call @div_gp_sp(%buf8, %buf6) : (memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>) -> ()
      aie.use_lock(%lock_0_2_12, Release, 1)
      cf.br ^bb1
    } {link_with = "attn.o"}
    func.func private @zero_fill_gp_bf16(memref<64x64xbf16, 2 : i32>)
    func.func private @zero_fill_sp_bf16(memref<64x1xbf16, 2 : i32>)
    func.func private @neg_inf_fill_up_bf16(memref<64x1xbf16, 2 : i32>)
    func.func private @zero_fill_g_bf16(memref<4096xbf16, 2 : i32>)
    func.func private @copy_tile(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>)
    func.func private @matmul_a_b_bf16(memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<4096xbf16, 2 : i32>)
    func.func private @fused_softmax(memref<4096xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>)
    func.func private @mul_r_gp(memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>)
    func.func private @matmul_g_b_bf16(memref<4096xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>)
    func.func private @accum_sp_r_s(memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>)
    func.func private @vector_copy_32elems(i32, memref<64x1xbf16, 2 : i32>, memref<64x1xbf16, 2 : i32>)
    func.func private @div_gp_sp(memref<64x1xbf16, 2 : i32>, memref<64x64xbf16, 2 : i32>)
    aie.packet_flow(0) {
      aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, DMA : 0>
      aie.packet_dest<%tile_0_3, DMA : 0>
    }
    aie.packet_flow(1) {
      aie.packet_source<%shim_noc_tile_1_0, DMA : 0>
      aie.packet_dest<%tile_1_2, DMA : 0>
      aie.packet_dest<%tile_1_3, DMA : 0>
    }
    aie.packet_flow(2) {
      aie.packet_source<%shim_noc_tile_2_0, DMA : 0>
      aie.packet_dest<%tile_2_2, DMA : 0>
      aie.packet_dest<%tile_2_3, DMA : 0>
    }
    aie.packet_flow(3) {
      aie.packet_source<%shim_noc_tile_3_0, DMA : 0>
      aie.packet_dest<%tile_3_2, DMA : 0>
      aie.packet_dest<%tile_3_3, DMA : 0>
    }
    aie.packet_flow(6) {
      aie.packet_source<%shim_noc_tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, DMA : 0>
      aie.packet_dest<%tile_1_2, DMA : 0>
      aie.packet_dest<%tile_2_2, DMA : 0>
      aie.packet_dest<%tile_3_2, DMA : 0>
    }
    aie.packet_flow(7) {
      aie.packet_source<%shim_noc_tile_1_0, DMA : 0>
      aie.packet_dest<%tile_0_3, DMA : 0>
      aie.packet_dest<%tile_1_3, DMA : 0>
      aie.packet_dest<%tile_2_3, DMA : 0>
      aie.packet_dest<%tile_3_3, DMA : 0>
    }
    aie.flow(%shim_noc_tile_4_0, DMA : 0, %mem_tile_4_1, DMA : 0)
    aie.flow(%shim_noc_tile_5_0, DMA : 0, %mem_tile_5_1, DMA : 0)
    aie.flow(%mem_tile_0_1, DMA : 0, %shim_noc_tile_0_0, DMA : 0)
    aie.flow(%mem_tile_1_1, DMA : 0, %shim_noc_tile_1_0, DMA : 0)
    aie.flow(%mem_tile_2_1, DMA : 0, %shim_noc_tile_2_0, DMA : 0)
    aie.flow(%mem_tile_3_1, DMA : 0, %shim_noc_tile_3_0, DMA : 0)
    aie.flow(%mem_tile_4_1, DMA : 0, %tile_0_2, DMA : 1)
    aie.flow(%mem_tile_4_1, DMA : 0, %tile_1_2, DMA : 1)
    aie.flow(%mem_tile_4_1, DMA : 0, %tile_2_2, DMA : 1)
    aie.flow(%mem_tile_4_1, DMA : 0, %tile_3_2, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_0_3, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_1_3, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_2_3, DMA : 1)
    aie.flow(%mem_tile_5_1, DMA : 0, %tile_3_3, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 0)
    aie.flow(%tile_1_2, DMA : 0, %mem_tile_1_1, DMA : 0)
    aie.flow(%tile_2_2, DMA : 0, %mem_tile_2_1, DMA : 0)
    aie.flow(%tile_3_2, DMA : 0, %mem_tile_3_1, DMA : 0)
    aie.cascade_flow(%tile_3_3, %tile_3_2)
    aie.cascade_flow(%tile_2_3, %tile_2_2)
    aie.cascade_flow(%tile_1_3, %tile_1_2)
    aie.cascade_flow(%tile_0_3, %tile_0_2)
    %memtile_dma_0_1 = aie.memtile_dma(%mem_tile_0_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf77 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf77 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_0_1_5, Release, 1)
      aie.next_bd ^bb4
    }
    %memtile_dma_1_1 = aie.memtile_dma(%mem_tile_1_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_1_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf76 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf76 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_1_1_4, Release, 1)
      aie.next_bd ^bb4
    }
    %memtile_dma_2_1 = aie.memtile_dma(%mem_tile_2_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_2_1_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf75 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_2_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf75 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_2_1_3, Release, 1)
      aie.next_bd ^bb4
    }
    %memtile_dma_3_1 = aie.memtile_dma(%mem_tile_3_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_3_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf74 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_3_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf74 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_3_1_2, Release, 1)
      aie.next_bd ^bb4
    }
    %memtile_dma_4_1 = aie.memtile_dma(%mem_tile_4_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_4_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf73 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_4_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf73 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_4_1_1, Release, 1)
      aie.next_bd ^bb4
    }
    %memtile_dma_5_1 = aie.memtile_dma(%mem_tile_5_1) {
      %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_5_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf72 : memref<64x64xbf16, 1 : i32>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 0, ^bb4, ^bb2)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_5_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf72 : memref<64x64xbf16, 1 : i32>, 0, 4096) {task_id = 0 : i32}
      aie.use_lock(%lock_5_1_0, Release, 1)
      aie.next_bd ^bb4
    }
    aie.shim_dma_allocation @air_channel_0_0(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_1(%shim_noc_tile_1_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_2(%shim_noc_tile_2_0, S2MM, 0)
    aie.shim_dma_allocation @air_channel_0_3(%shim_noc_tile_3_0, S2MM, 0)
    aie.shim_dma_allocation @air_Q2L1_0(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @air_Q2L1_1(%shim_noc_tile_1_0, MM2S, 0)
    aie.shim_dma_allocation @air_Q2L1_2(%shim_noc_tile_2_0, MM2S, 0)
    aie.shim_dma_allocation @air_Q2L1_3(%shim_noc_tile_3_0, MM2S, 0)
    aie.shim_dma_allocation @air_VIn_0(%shim_noc_tile_4_0, MM2S, 0)
    aie.shim_dma_allocation @air_VIn_1(%shim_noc_tile_5_0, MM2S, 0)
    aie.runtime_sequence @s_sequence(%arg0: memref<256x64xbf16>, %arg1: memref<256x64xbf16>, %arg2: memref<256x64xbf16>, %arg3: memref<256x64xbf16>) {
      // === Phase 1: Q (once per column, vertical broadcast to both stages) ===
      %q0 = aiex.dma_configure_task_for @air_Q2L1_0 {
        aie.dma_bd(%arg0 : memref<256x64xbf16>, 0, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%q0)
      %q1 = aiex.dma_configure_task_for @air_Q2L1_1 {
        aie.dma_bd(%arg0 : memref<256x64xbf16>, 4096, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%q1)
      %q2 = aiex.dma_configure_task_for @air_Q2L1_2 {
        aie.dma_bd(%arg0 : memref<256x64xbf16>, 8192, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 2>}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%q2)
      %q3 = aiex.dma_configure_task_for @air_Q2L1_3 {
        aie.dma_bd(%arg0 : memref<256x64xbf16>, 12288, 4096, [<size = 8, stride = 8>, <size = 64, stride = 64>, <size = 8, stride = 1>]) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%q3)
      // === Barrier: wait for all Q to complete, free BDs, before sending K ===
      aiex.dma_await_task(%q0)
      aiex.dma_free_task(%q0)
      aiex.dma_await_task(%q1)
      aiex.dma_free_task(%q1)
      aiex.dma_await_task(%q2)
      aiex.dma_free_task(%q2)
      aiex.dma_await_task(%q3)
      aiex.dma_free_task(%q3)
      // === Phase 2: K (per-stage horizontal broadcast, reusing Q channels) ===
      // K chunk 0 stage 0: shim 0 MM2S:0, pkt_id=6, horiz broadcast to row 2
      %k0_c0 = aiex.dma_configure_task_for @air_Q2L1_0 {
        aie.dma_bd(%arg1 : memref<256x64xbf16>, 0, 512, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%k0_c0)
      // K chunk 0 stage 1: shim 1 MM2S:0, pkt_id=7, horiz broadcast to row 3
      %k1_c0 = aiex.dma_configure_task_for @air_Q2L1_1 {
        aie.dma_bd(%arg1 : memref<256x64xbf16>, 8192, 512, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 7>}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%k1_c0)
      // K chunk 1 stage 0: shim 0 MM2S:0, pkt_id=6
      %k0_c1 = aiex.dma_configure_task_for @air_Q2L1_0 {
        aie.dma_bd(%arg1 : memref<256x64xbf16>, 4096, 512, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%k0_c1)
      // K chunk 1 stage 1: shim 1 MM2S:0, pkt_id=7
      %k1_c1 = aiex.dma_configure_task_for @air_Q2L1_1 {
        aie.dma_bd(%arg1 : memref<256x64xbf16>, 12288, 512, [<size = 8, stride = 512>, <size = 8, stride = 8>, <size = 8, stride = 64>, <size = 8, stride = 1>]) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 7>}
        aie.end
      } {repeat_count = 7 : i32}
      aiex.dma_start_task(%k1_c1)
      // === Phase 3: V (per-stage via memtile) ===
      %v0 = aiex.dma_configure_task_for @air_VIn_0 {
        aie.dma_bd(%arg2 : memref<256x64xbf16>, 0, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%v0)
      %v1 = aiex.dma_configure_task_for @air_VIn_1 {
        aie.dma_bd(%arg2 : memref<256x64xbf16>, 8192, 8192, [<size = 16, stride = 512>, <size = 512, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%v1)
      // === Phase 4: Output (Gp from row 2 tiles via memtile) ===
      %out0 = aiex.dma_configure_task_for @air_channel_0_0 {
        aie.dma_bd(%arg3 : memref<256x64xbf16>, 0, 4096, [<size = 8, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%out0)
      %out1 = aiex.dma_configure_task_for @air_channel_0_1 {
        aie.dma_bd(%arg3 : memref<256x64xbf16>, 4096, 4096, [<size = 8, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%out1)
      %out2 = aiex.dma_configure_task_for @air_channel_0_2 {
        aie.dma_bd(%arg3 : memref<256x64xbf16>, 8192, 4096, [<size = 8, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%out2)
      %out3 = aiex.dma_configure_task_for @air_channel_0_3 {
        aie.dma_bd(%arg3 : memref<256x64xbf16>, 12288, 4096, [<size = 8, stride = 512>, <size = 512, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%out3)
      // === Cleanup: await output, free all tasks ===
      aiex.dma_await_task(%out0)
      aiex.dma_await_task(%out1)
      aiex.dma_await_task(%out2)
      aiex.dma_await_task(%out3)
      aiex.dma_free_task(%k0_c0)
      aiex.dma_free_task(%k1_c0)
      aiex.dma_free_task(%k0_c1)
      aiex.dma_free_task(%k1_c1)
      aiex.dma_free_task(%v0)
      aiex.dma_free_task(%v1)
    }
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  aie.device(npu2) {
    aie.runtime_sequence @full_4x2_direct(%arg0: memref<256x64xbf16>, %arg1: memref<256x64xbf16>, %arg2: memref<256x64xbf16>, %arg3: memref<256x64xbf16>) {
      aiex.configure @s {
        aiex.run @s_sequence(%arg0, %arg1, %arg2, %arg3) : (memref<256x64xbf16>, memref<256x64xbf16>, memref<256x64xbf16>, memref<256x64xbf16>)
      }
    }
  }
}
