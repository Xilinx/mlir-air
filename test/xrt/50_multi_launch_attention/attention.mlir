#map = affine_map<()[s0, s1] -> (s0 + s1 * 8)>
#map1 = affine_map<()[s0, s1] -> (s0 + s1)>
#map2 = affine_map<()[s0] -> (s0 * 64)>
#map3 = affine_map<()[s0, s1] -> (s0 * 8 + s1 + 1)>
#map4 = affine_map<(d0) -> (d0 * 64)>
#map5 = affine_map<(d0) -> (d0 * 512)>
#map6 = affine_map<()[s0] -> (s0 * 512 + 512)>
#map7 = affine_map<()[s0] -> (s0 * 64 + 64)>
#map8 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
#map9 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map10 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#map11 = affine_map<()[s0] -> (s0 * 8)>
module {
  func.func @attention(%Q: memref<256x256xbf16>, %K_T: memref<256x256xbf16>, %V: memref<256x256xbf16>, %S: memref<256x256xbf16>, %P: memref<256x256xbf16>, %O: memref<256x256xbf16>) {
    %c0_100 = arith.constant 0 : index
    %c1_100 = arith.constant 1 : index
    %c64_100 = arith.constant 64 : index

    // Launch 1: Matmul on Q @ K_T -> S
    air.launch (%arg9, %arg10, %arg11) in (%arg12=%c1_100, %arg13=%c1_100, %arg14=%c1_100) args(%arg15=%Q, %arg16=%K_T, %arg17=%S) : memref<256x256xbf16>, memref<256x256xbf16>, memref<256x256xbf16> attributes {id = 1 : i32} {
      air.segment @npu_mm_exact_0  args(%arg18=%arg9, %arg19=%arg10, %arg20=%arg15, %arg21=%arg16, %arg22=%arg17) : index, index, memref<256x256xbf16>, memref<256x256xbf16>, memref<256x256xbf16> {
        %c1_0 = arith.constant 1 : index
        %c65536 = arith.constant 65536 : index
        %c256 = arith.constant 256 : index
        %c4 = arith.constant 4 : index
        %c0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %0 = arith.muli %arg19, %c256 : index
        %1 = arith.muli %arg18, %c65536 : index
        %alloc = memref.alloc() : memref<256x256xbf16, 1 : i32>
        %alloc_1 = memref.alloc() : memref<256x256xbf16, 1 : i32>
        %2 = arith.addi %1, %0 : index
        %alloc_2 = memref.alloc() : memref<256x256xbf16, 1>
        %alloc_3 = memref.alloc() : memref<32x32x8x8xbf16, 2>
        air.herd @herd_0  tile (%arg23, %arg24) in (%arg25=%c4, %arg26=%c4) args(%arg27=%alloc_3) : memref<32x32x8x8xbf16, 2> {
          %cst = arith.constant dense<0.000000e+00> : vector<1x1x8x8xbf16>
          %c0_4 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1_5 = arith.constant 1 : index
          scf.for %arg28 = %c0_4 to %c8 step %c1_5 {
            scf.for %arg29 = %c0_4 to %c8 step %c1_5 {
              %3 = affine.apply #map()[%arg29, %arg24]
              %4 = affine.apply #map()[%arg28, %arg23]
              vector.transfer_write %cst, %arg27[%3, %4, %c0_4, %c0_4] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xbf16>, memref<32x32x8x8xbf16, 2>
            }
          }
        }
        scf.for %arg23 = %c0 to %c256 step %c64 {
          %3 = affine.apply #map1()[%1, %arg23]
          air.dma_memcpy_nd (%alloc[%c0, %arg23] [%c256, %c64] [%c256, %c1_0], %arg20[%c0, %3] [%c256, %c64] [%c256, %c1_0]) : (memref<256x256xbf16, 1 : i32>, memref<256x256xbf16>)
          air.dma_memcpy_nd (%alloc_1[%arg23, %c0] [%c64, %c256] [%c256, %c1_0], %arg21[%arg23, %0] [%c64, %c256] [%c256, %c1_0]) : (memref<256x256xbf16, 1 : i32>, memref<256x256xbf16>)
          air.herd @herd_0  tile (%arg24, %arg25) in (%arg26=%c4, %arg27=%c4) args(%arg28=%alloc, %arg29=%arg23, %arg30=%alloc_1, %arg31=%alloc_3) : memref<256x256xbf16, 1 : i32>, index, memref<256x256xbf16, 1 : i32>, memref<32x32x8x8xbf16, 2> {
            %c64_4 = arith.constant 64 : index
            %c512 = arith.constant 512 : index
            %4 = ub.poison : bf16
            %c0_5 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c2048 = arith.constant 2048 : index
            %c256_6 = arith.constant 256 : index
            %c1_7 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %5 = affine.apply #map2()[%arg24]
            %alloc_8 = memref.alloc() : memref<8x8x8x8xbf16, 2>
            air.dma_memcpy_nd (%alloc_8[] [] [], %arg28[%c0_5, %c0_5, %5, %arg29] [%c8, %c8, %c8, %c8] [%c8, %c2048, %c256_6, %c1_7]) : (memref<8x8x8x8xbf16, 2>, memref<256x256xbf16, 1 : i32>)
            %6 = affine.apply #map2()[%arg25]
            %alloc_9 = memref.alloc() : memref<8x8x8x8xbf16, 2>
            air.dma_memcpy_nd (%alloc_9[] [] [], %arg30[%c0_5, %c0_5, %arg29, %6] [%c8, %c8, %c8, %c8] [%c8, %c2048, %c256_6, %c1_7]) : (memref<8x8x8x8xbf16, 2>, memref<256x256xbf16, 1 : i32>)
            scf.for %arg32 = %c0_5 to %c8 step %c2 {
              scf.for %arg33 = %c0_5 to %c8 step %c2 {
                %7 = affine.apply #map()[%arg33, %arg25]
                %8 = affine.apply #map()[%arg32, %arg24]
                %9 = vector.transfer_read %arg31[%7, %8, %c0_5, %c0_5], %4 {in_bounds = [true, true, true, true]} : memref<32x32x8x8xbf16, 2>, vector<1x1x8x8xbf16>
                %10 = affine.apply #map3()[%arg25, %arg33]
                %11 = vector.transfer_read %arg31[%10, %8, %c0_5, %c0_5], %4 {in_bounds = [true, true, true, true]} : memref<32x32x8x8xbf16, 2>, vector<1x1x8x8xbf16>
                %12 = affine.apply #map3()[%arg24, %arg32]
                %13 = vector.transfer_read %arg31[%7, %12, %c0_5, %c0_5], %4 {in_bounds = [true, true, true, true]} : memref<32x32x8x8xbf16, 2>, vector<1x1x8x8xbf16>
                %14 = vector.transfer_read %arg31[%10, %12, %c0_5, %c0_5], %4 {in_bounds = [true, true, true, true]} : memref<32x32x8x8xbf16, 2>, vector<1x1x8x8xbf16>
                %15 = arith.extf %9 : vector<1x1x8x8xbf16> to vector<1x1x8x8xf32>
                %16 = arith.extf %11 : vector<1x1x8x8xbf16> to vector<1x1x8x8xf32>
                %17 = arith.extf %13 : vector<1x1x8x8xbf16> to vector<1x1x8x8xf32>
                %18 = arith.extf %14 : vector<1x1x8x8xbf16> to vector<1x1x8x8xf32>
                %19 = vector.shape_cast %15 : vector<1x1x8x8xf32> to vector<64xf32>
                %20 = vector.shape_cast %16 : vector<1x1x8x8xf32> to vector<64xf32>
                %21 = vector.shape_cast %17 : vector<1x1x8x8xf32> to vector<64xf32>
                %22 = vector.shape_cast %18 : vector<1x1x8x8xf32> to vector<64xf32>
                %collapse_shape = memref.collapse_shape %alloc_8 [[0, 1, 2, 3]] : memref<8x8x8x8xbf16, 2> into memref<4096xbf16, 2>
                %23 = affine.apply #map4(%arg32)
                %collapse_shape_10 = memref.collapse_shape %alloc_9 [[0, 1, 2, 3]] : memref<8x8x8x8xbf16, 2> into memref<4096xbf16, 2>
                %24 = affine.apply #map5(%arg33)
                %25 = affine.apply #map6()[%arg33]
                %26 = affine.apply #map7()[%arg32]
                %27:8 = scf.for %arg34 = %c0_5 to %c8 step %c1_7 iter_args(%arg35 = %19, %arg36 = %20, %arg37 = %21, %arg38 = %22, %arg39 = %23, %arg40 = %24, %arg41 = %25, %arg42 = %26) -> (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, index, index, index, index) {
                  %36 = vector.shape_cast %arg35 : vector<64xf32> to vector<1x1x8x8xf32>
                  %37 = vector.shape_cast %arg36 : vector<64xf32> to vector<1x1x8x8xf32>
                  %38 = vector.shape_cast %arg37 : vector<64xf32> to vector<1x1x8x8xf32>
                  %39 = vector.shape_cast %arg38 : vector<64xf32> to vector<1x1x8x8xf32>
                  %40 = vector.transfer_read %collapse_shape[%arg39], %4 {in_bounds = [true]} : memref<4096xbf16, 2>, vector<64xbf16>
                  %41 = vector.shape_cast %40 : vector<64xbf16> to vector<1x1x8x8xbf16>
                  %42 = arith.addi %arg39, %c512 : index
                  %43 = vector.transfer_read %collapse_shape_10[%arg40], %4 {in_bounds = [true]} : memref<4096xbf16, 2>, vector<64xbf16>
                  %44 = vector.shape_cast %43 : vector<64xbf16> to vector<1x1x8x8xbf16>
                  %45 = arith.addi %arg40, %c64_4 : index
                  %46 = vector.contract {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %41, %44, %36 : vector<1x1x8x8xbf16>, vector<1x1x8x8xbf16> into vector<1x1x8x8xf32>
                  %47 = vector.transfer_read %collapse_shape_10[%arg41], %4 {in_bounds = [true]} : memref<4096xbf16, 2>, vector<64xbf16>
                  %48 = vector.shape_cast %47 : vector<64xbf16> to vector<1x1x8x8xbf16>
                  %49 = arith.addi %arg41, %c64_4 : index
                  %50 = vector.contract {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %41, %48, %37 : vector<1x1x8x8xbf16>, vector<1x1x8x8xbf16> into vector<1x1x8x8xf32>
                  %51 = vector.transfer_read %collapse_shape[%arg42], %4 {in_bounds = [true]} : memref<4096xbf16, 2>, vector<64xbf16>
                  %52 = vector.shape_cast %51 : vector<64xbf16> to vector<1x1x8x8xbf16>
                  %53 = arith.addi %arg42, %c512 : index
                  %54 = vector.contract {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %52, %44, %38 : vector<1x1x8x8xbf16>, vector<1x1x8x8xbf16> into vector<1x1x8x8xf32>
                  %55 = vector.contract {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %52, %48, %39 : vector<1x1x8x8xbf16>, vector<1x1x8x8xbf16> into vector<1x1x8x8xf32>
                  %56 = vector.shape_cast %46 : vector<1x1x8x8xf32> to vector<64xf32>
                  %57 = vector.shape_cast %50 : vector<1x1x8x8xf32> to vector<64xf32>
                  %58 = vector.shape_cast %54 : vector<1x1x8x8xf32> to vector<64xf32>
                  %59 = vector.shape_cast %55 : vector<1x1x8x8xf32> to vector<64xf32>
                  scf.yield %56, %57, %58, %59, %42, %45, %49, %53 : vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, index, index, index, index
                }
                %28 = vector.shape_cast %27#0 : vector<64xf32> to vector<1x1x8x8xf32>
                %29 = vector.shape_cast %27#1 : vector<64xf32> to vector<1x1x8x8xf32>
                %30 = vector.shape_cast %27#2 : vector<64xf32> to vector<1x1x8x8xf32>
                %31 = vector.shape_cast %27#3 : vector<64xf32> to vector<1x1x8x8xf32>
                %32 = arith.truncf %31 : vector<1x1x8x8xf32> to vector<1x1x8x8xbf16>
                %33 = arith.truncf %30 : vector<1x1x8x8xf32> to vector<1x1x8x8xbf16>
                %34 = arith.truncf %29 : vector<1x1x8x8xf32> to vector<1x1x8x8xbf16>
                %35 = arith.truncf %28 : vector<1x1x8x8xf32> to vector<1x1x8x8xbf16>
                vector.transfer_write %32, %arg31[%10, %12, %c0_5, %c0_5] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xbf16>, memref<32x32x8x8xbf16, 2>
                vector.transfer_write %33, %arg31[%7, %12, %c0_5, %c0_5] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xbf16>, memref<32x32x8x8xbf16, 2>
                vector.transfer_write %34, %arg31[%10, %8, %c0_5, %c0_5] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xbf16>, memref<32x32x8x8xbf16, 2>
                vector.transfer_write %35, %arg31[%7, %8, %c0_5, %c0_5] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xbf16>, memref<32x32x8x8xbf16, 2>
              }
            }
            memref.dealloc %alloc_8 : memref<8x8x8x8xbf16, 2>
            memref.dealloc %alloc_9 : memref<8x8x8x8xbf16, 2>
          }
        }
        air.herd @herd_0  tile (%arg23, %arg24) in (%arg25=%c4, %arg26=%c4) args(%arg27=%alloc_2, %arg28=%alloc_3) : memref<256x256xbf16, 1>, memref<32x32x8x8xbf16, 2> {
          %c64_4 = arith.constant 64 : index
          %c256_5 = arith.constant 256 : index
          %c1_6 = arith.constant 1 : index
          %c0_7 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c2048 = arith.constant 2048 : index
          %3 = affine.apply #map2()[%arg23]
          %4 = affine.apply #map2()[%arg24]
          %5 = affine.apply #map11()[%arg23]
          %6 = affine.apply #map11()[%arg24]
          air.dma_memcpy_nd (%arg27[%3, %4] [%c64_4, %c64_4] [%c256_5, %c1_6], %arg28[%5, %c0_7, %6, %c0_7] [%c8, %c8, %c8, %c8] [%c64_4, %c8, %c2048, %c1_6]) : (memref<256x256xbf16, 1>, memref<32x32x8x8xbf16, 2>)
        }
        air.dma_memcpy_nd (%arg22[%c0, %2] [%c256, %c256] [%c256, %c1_0], %alloc_2[] [] []) {id = 3 : i32} : (memref<256x256xbf16>, memref<256x256xbf16, 1>)
        memref.dealloc %alloc_2 : memref<256x256xbf16, 1>
        memref.dealloc %alloc_3 : memref<32x32x8x8xbf16, 2>
      }
    }

    // Launch 2: Softmax on S -> P
    air.launch (%lx1, %ly1, %lz1) in (%sx1=%c64_100, %sy1=%c1_100, %sz1=%c1_100) args(%arg_S=%S, %arg_P=%P) : memref<256x256xbf16>, memref<256x256xbf16> attributes {id = 4 : i32} {
      air.segment @softmax args(%seg_idx=%lx1, %seg_S=%arg_S, %seg_P=%arg_P) : index, memref<256x256xbf16>, memref<256x256xbf16> attributes {id = 5 : i32} {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c4_s = arith.constant 4 : index
        %c256_s = arith.constant 256 : index
        
        %row_offset = arith.muli %seg_idx, %c4_s : index
        
        %l2_in = memref.alloc() : memref<4x256xbf16, 1>
        %l2_out = memref.alloc() : memref<4x256xbf16, 1>
        
        air.dma_memcpy_nd (%l2_in[] [] [], %seg_S[%row_offset, %c0_s] [%c4_s, %c256_s] [%c256_s, %c1_s]) : (memref<4x256xbf16, 1>, memref<256x256xbf16>)
        
        air.herd @softmax_herd tile (%tx, %ty) in (%sx=%c4_s, %sy=%c1_s) args(%herd_in=%l2_in, %herd_out=%l2_out) : memref<4x256xbf16, 1>, memref<4x256xbf16, 1> attributes {id = 6 : i32} {
          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c32_h = arith.constant 32 : index
          %c256_h = arith.constant 256 : index
          %cst_zero = arith.constant 0.000000e+00 : f32
          %cst_neg_inf = arith.constant 0xFF800000 : f32
          %poison = ub.poison : bf16
          
          %l1_row = memref.alloc() : memref<1x256xbf16, 2>
          %l1_out = memref.alloc() : memref<1x256xbf16, 2>
          
          air.dma_memcpy_nd (%l1_row[] [] [], %herd_in[%tx, %c0_h] [%c1_h, %c256_h] [%c256_h, %c1_h]) : (memref<1x256xbf16, 2>, memref<4x256xbf16, 1>)
          
          %max_val = memref.alloc() : memref<1xf32, 2>
          memref.store %cst_neg_inf, %max_val[%c0_h] : memref<1xf32, 2>
          scf.for %j = %c0_h to %c256_h step %c32_h {
            %subview = memref.subview %l1_row[0, %j] [1, 32] [1, 1] : memref<1x256xbf16, 2> to memref<1x32xbf16, strided<[256, 1], offset: ?>, 2>
            %vec = vector.transfer_read %subview[%c0_h, %c0_h], %poison {in_bounds = [true]} : memref<1x32xbf16, strided<[256, 1], offset: ?>, 2>, vector<32xbf16>
            %curr = memref.load %max_val[%c0_h] : memref<1xf32, 2>
            %curr_bf16 = arith.truncf %curr : f32 to bf16
            %chunk_max = vector.reduction <maxnumf>, %vec, %curr_bf16 : vector<32xbf16> into bf16
            %chunk_max_f32 = arith.extf %chunk_max : bf16 to f32
            memref.store %chunk_max_f32, %max_val[%c0_h] : memref<1xf32, 2>
          }
          
          %sum_val = memref.alloc() : memref<1xf32, 2>
          memref.store %cst_zero, %sum_val[%c0_h] : memref<1xf32, 2>
          scf.for %j = %c0_h to %c256_h step %c32_h {
            %subview = memref.subview %l1_row[0, %j] [1, 32] [1, 1] : memref<1x256xbf16, 2> to memref<1x32xbf16, strided<[256, 1], offset: ?>, 2>
            %vec = vector.transfer_read %subview[%c0_h, %c0_h], %poison {in_bounds = [true]} : memref<1x32xbf16, strided<[256, 1], offset: ?>, 2>, vector<32xbf16>
            %max_f32 = memref.load %max_val[%c0_h] : memref<1xf32, 2>
            %curr_sum = memref.load %sum_val[%c0_h] : memref<1xf32, 2>
            %vec_f32 = arith.extf %vec : vector<32xbf16> to vector<32xf32>
            %max_bc = vector.broadcast %max_f32 : f32 to vector<32xf32>
            %shifted = arith.subf %vec_f32, %max_bc : vector<32xf32>
            %shifted_bf16 = arith.truncf %shifted : vector<32xf32> to vector<32xbf16>
            %exp_vec = math.exp %shifted_bf16 : vector<32xbf16>
            %curr_sum_bf16 = arith.truncf %curr_sum : f32 to bf16
            %chunk_sum = vector.reduction <add>, %exp_vec, %curr_sum_bf16 : vector<32xbf16> into bf16
            %chunk_sum_f32 = arith.extf %chunk_sum : bf16 to f32
            memref.store %chunk_sum_f32, %sum_val[%c0_h] : memref<1xf32, 2>
          }
          
          scf.for %j = %c0_h to %c256_h step %c32_h {
            %subview_in = memref.subview %l1_row[0, %j] [1, 32] [1, 1] : memref<1x256xbf16, 2> to memref<1x32xbf16, strided<[256, 1], offset: ?>, 2>
            %subview_out = memref.subview %l1_out[0, %j] [1, 32] [1, 1] : memref<1x256xbf16, 2> to memref<1x32xbf16, strided<[256, 1], offset: ?>, 2>
            %vec = vector.transfer_read %subview_in[%c0_h, %c0_h], %poison {in_bounds = [true]} : memref<1x32xbf16, strided<[256, 1], offset: ?>, 2>, vector<32xbf16>
            %max_f32 = memref.load %max_val[%c0_h] : memref<1xf32, 2>
            %sum_f32 = memref.load %sum_val[%c0_h] : memref<1xf32, 2>
            %vec_f32 = arith.extf %vec : vector<32xbf16> to vector<32xf32>
            %max_bc = vector.broadcast %max_f32 : f32 to vector<32xf32>
            %shifted = arith.subf %vec_f32, %max_bc : vector<32xf32>
            %shifted_bf16 = arith.truncf %shifted : vector<32xf32> to vector<32xbf16>
            %exp_vec = math.exp %shifted_bf16 : vector<32xbf16>
            %exp_f32 = arith.extf %exp_vec : vector<32xbf16> to vector<32xf32>
            %sum_bc = vector.broadcast %sum_f32 : f32 to vector<32xf32>
            %normalized = arith.divf %exp_f32, %sum_bc : vector<32xf32>
            %normalized_bf16 = arith.truncf %normalized : vector<32xf32> to vector<32xbf16>
            vector.transfer_write %normalized_bf16, %subview_out[%c0_h, %c0_h] {in_bounds = [true]} : vector<32xbf16>, memref<1x32xbf16, strided<[256, 1], offset: ?>, 2>
          }
          
          air.dma_memcpy_nd (%herd_out[%tx, %c0_h] [%c1_h, %c256_h] [%c256_h, %c1_h], %l1_out[] [] []) : (memref<4x256xbf16, 1>, memref<1x256xbf16, 2>)
          
          memref.dealloc %max_val : memref<1xf32, 2>
          memref.dealloc %sum_val : memref<1xf32, 2>
          memref.dealloc %l1_row : memref<1x256xbf16, 2>
          memref.dealloc %l1_out : memref<1x256xbf16, 2>
        }
        
        air.dma_memcpy_nd (%seg_P[%row_offset, %c0_s] [%c4_s, %c256_s] [%c256_s, %c1_s], %l2_out[] [] []) : (memref<256x256xbf16>, memref<4x256xbf16, 1>)
        
        memref.dealloc %l2_in : memref<4x256xbf16, 1>
        memref.dealloc %l2_out : memref<4x256xbf16, 1>
      }
    }

    // Launch 3: Matmul P @ V -> O
    air.launch (%lx2, %ly2, %lz2) in (%sx2=%c1_100, %sy2=%c1_100, %sz2=%c1_100) args(%arg_P=%P, %arg_V=%V, %arg_O=%O) : memref<256x256xbf16>, memref<256x256xbf16>, memref<256x256xbf16> attributes {id = 7 : i32} {
      air.segment @npu_mm_exact_1  args(%arg18=%lx2, %arg19=%ly2, %arg20=%arg_P, %arg21=%arg_V, %arg22=%arg_O) : index, index, memref<256x256xbf16>, memref<256x256xbf16>, memref<256x256xbf16> {
        %c1_0 = arith.constant 1 : index
        %c65536 = arith.constant 65536 : index
        %c256 = arith.constant 256 : index
        %c4 = arith.constant 4 : index
        %c0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %0 = arith.muli %arg19, %c256 : index
        %1 = arith.muli %arg18, %c65536 : index
        %alloc = memref.alloc() : memref<256x256xbf16, 1 : i32>
        %alloc_1 = memref.alloc() : memref<256x256xbf16, 1 : i32>
        %2 = arith.addi %1, %0 : index
        %alloc_2 = memref.alloc() : memref<256x256xbf16, 1>
        %alloc_3 = memref.alloc() : memref<32x32x8x8xbf16, 2>
        air.herd @herd_0  tile (%arg23, %arg24) in (%arg25=%c4, %arg26=%c4) args(%arg27=%alloc_3) : memref<32x32x8x8xbf16, 2> {
          %cst = arith.constant dense<0.000000e+00> : vector<1x1x8x8xbf16>
          %c0_4 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1_5 = arith.constant 1 : index
          scf.for %arg28 = %c0_4 to %c8 step %c1_5 {
            scf.for %arg29 = %c0_4 to %c8 step %c1_5 {
              %3 = affine.apply #map()[%arg29, %arg24]
              %4 = affine.apply #map()[%arg28, %arg23]
              vector.transfer_write %cst, %arg27[%3, %4, %c0_4, %c0_4] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xbf16>, memref<32x32x8x8xbf16, 2>
            }
          }
        }
        scf.for %arg23 = %c0 to %c256 step %c64 {
          %3 = affine.apply #map1()[%1, %arg23]
          air.dma_memcpy_nd (%alloc[%c0, %arg23] [%c256, %c64] [%c256, %c1_0], %arg20[%c0, %3] [%c256, %c64] [%c256, %c1_0]) : (memref<256x256xbf16, 1 : i32>, memref<256x256xbf16>)
          air.dma_memcpy_nd (%alloc_1[%arg23, %c0] [%c64, %c256] [%c256, %c1_0], %arg21[%arg23, %0] [%c64, %c256] [%c256, %c1_0]) : (memref<256x256xbf16, 1 : i32>, memref<256x256xbf16>)
          air.herd @herd_0  tile (%arg24, %arg25) in (%arg26=%c4, %arg27=%c4) args(%arg28=%alloc, %arg29=%arg23, %arg30=%alloc_1, %arg31=%alloc_3) : memref<256x256xbf16, 1 : i32>, index, memref<256x256xbf16, 1 : i32>, memref<32x32x8x8xbf16, 2> {
            %c64_4 = arith.constant 64 : index
            %c512 = arith.constant 512 : index
            %4 = ub.poison : bf16
            %c0_5 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %c2048 = arith.constant 2048 : index
            %c256_6 = arith.constant 256 : index
            %c1_7 = arith.constant 1 : index
            %c2 = arith.constant 2 : index
            %5 = affine.apply #map2()[%arg24]
            %alloc_8 = memref.alloc() : memref<8x8x8x8xbf16, 2>
            air.dma_memcpy_nd (%alloc_8[] [] [], %arg28[%c0_5, %c0_5, %5, %arg29] [%c8, %c8, %c8, %c8] [%c8, %c2048, %c256_6, %c1_7]) : (memref<8x8x8x8xbf16, 2>, memref<256x256xbf16, 1 : i32>)
            %6 = affine.apply #map2()[%arg25]
            %alloc_9 = memref.alloc() : memref<8x8x8x8xbf16, 2>
            air.dma_memcpy_nd (%alloc_9[] [] [], %arg30[%c0_5, %c0_5, %arg29, %6] [%c8, %c8, %c8, %c8] [%c8, %c2048, %c256_6, %c1_7]) : (memref<8x8x8x8xbf16, 2>, memref<256x256xbf16, 1 : i32>)
            scf.for %arg32 = %c0_5 to %c8 step %c2 {
              scf.for %arg33 = %c0_5 to %c8 step %c2 {
                %7 = affine.apply #map()[%arg33, %arg25]
                %8 = affine.apply #map()[%arg32, %arg24]
                %9 = vector.transfer_read %arg31[%7, %8, %c0_5, %c0_5], %4 {in_bounds = [true, true, true, true]} : memref<32x32x8x8xbf16, 2>, vector<1x1x8x8xbf16>
                %10 = affine.apply #map3()[%arg25, %arg33]
                %11 = vector.transfer_read %arg31[%10, %8, %c0_5, %c0_5], %4 {in_bounds = [true, true, true, true]} : memref<32x32x8x8xbf16, 2>, vector<1x1x8x8xbf16>
                %12 = affine.apply #map3()[%arg24, %arg32]
                %13 = vector.transfer_read %arg31[%7, %12, %c0_5, %c0_5], %4 {in_bounds = [true, true, true, true]} : memref<32x32x8x8xbf16, 2>, vector<1x1x8x8xbf16>
                %14 = vector.transfer_read %arg31[%10, %12, %c0_5, %c0_5], %4 {in_bounds = [true, true, true, true]} : memref<32x32x8x8xbf16, 2>, vector<1x1x8x8xbf16>
                %15 = arith.extf %9 : vector<1x1x8x8xbf16> to vector<1x1x8x8xf32>
                %16 = arith.extf %11 : vector<1x1x8x8xbf16> to vector<1x1x8x8xf32>
                %17 = arith.extf %13 : vector<1x1x8x8xbf16> to vector<1x1x8x8xf32>
                %18 = arith.extf %14 : vector<1x1x8x8xbf16> to vector<1x1x8x8xf32>
                %19 = vector.shape_cast %15 : vector<1x1x8x8xf32> to vector<64xf32>
                %20 = vector.shape_cast %16 : vector<1x1x8x8xf32> to vector<64xf32>
                %21 = vector.shape_cast %17 : vector<1x1x8x8xf32> to vector<64xf32>
                %22 = vector.shape_cast %18 : vector<1x1x8x8xf32> to vector<64xf32>
                %collapse_shape = memref.collapse_shape %alloc_8 [[0, 1, 2, 3]] : memref<8x8x8x8xbf16, 2> into memref<4096xbf16, 2>
                %23 = affine.apply #map4(%arg32)
                %collapse_shape_10 = memref.collapse_shape %alloc_9 [[0, 1, 2, 3]] : memref<8x8x8x8xbf16, 2> into memref<4096xbf16, 2>
                %24 = affine.apply #map5(%arg33)
                %25 = affine.apply #map6()[%arg33]
                %26 = affine.apply #map7()[%arg32]
                %27:8 = scf.for %arg34 = %c0_5 to %c8 step %c1_7 iter_args(%arg35 = %19, %arg36 = %20, %arg37 = %21, %arg38 = %22, %arg39 = %23, %arg40 = %24, %arg41 = %25, %arg42 = %26) -> (vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, index, index, index, index) {
                  %36 = vector.shape_cast %arg35 : vector<64xf32> to vector<1x1x8x8xf32>
                  %37 = vector.shape_cast %arg36 : vector<64xf32> to vector<1x1x8x8xf32>
                  %38 = vector.shape_cast %arg37 : vector<64xf32> to vector<1x1x8x8xf32>
                  %39 = vector.shape_cast %arg38 : vector<64xf32> to vector<1x1x8x8xf32>
                  %40 = vector.transfer_read %collapse_shape[%arg39], %4 {in_bounds = [true]} : memref<4096xbf16, 2>, vector<64xbf16>
                  %41 = vector.shape_cast %40 : vector<64xbf16> to vector<1x1x8x8xbf16>
                  %42 = arith.addi %arg39, %c512 : index
                  %43 = vector.transfer_read %collapse_shape_10[%arg40], %4 {in_bounds = [true]} : memref<4096xbf16, 2>, vector<64xbf16>
                  %44 = vector.shape_cast %43 : vector<64xbf16> to vector<1x1x8x8xbf16>
                  %45 = arith.addi %arg40, %c64_4 : index
                  %46 = vector.contract {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %41, %44, %36 : vector<1x1x8x8xbf16>, vector<1x1x8x8xbf16> into vector<1x1x8x8xf32>
                  %47 = vector.transfer_read %collapse_shape_10[%arg41], %4 {in_bounds = [true]} : memref<4096xbf16, 2>, vector<64xbf16>
                  %48 = vector.shape_cast %47 : vector<64xbf16> to vector<1x1x8x8xbf16>
                  %49 = arith.addi %arg41, %c64_4 : index
                  %50 = vector.contract {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %41, %48, %37 : vector<1x1x8x8xbf16>, vector<1x1x8x8xbf16> into vector<1x1x8x8xf32>
                  %51 = vector.transfer_read %collapse_shape[%arg42], %4 {in_bounds = [true]} : memref<4096xbf16, 2>, vector<64xbf16>
                  %52 = vector.shape_cast %51 : vector<64xbf16> to vector<1x1x8x8xbf16>
                  %53 = arith.addi %arg42, %c512 : index
                  %54 = vector.contract {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %52, %44, %38 : vector<1x1x8x8xbf16>, vector<1x1x8x8xbf16> into vector<1x1x8x8xf32>
                  %55 = vector.contract {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %52, %48, %39 : vector<1x1x8x8xbf16>, vector<1x1x8x8xbf16> into vector<1x1x8x8xf32>
                  %56 = vector.shape_cast %46 : vector<1x1x8x8xf32> to vector<64xf32>
                  %57 = vector.shape_cast %50 : vector<1x1x8x8xf32> to vector<64xf32>
                  %58 = vector.shape_cast %54 : vector<1x1x8x8xf32> to vector<64xf32>
                  %59 = vector.shape_cast %55 : vector<1x1x8x8xf32> to vector<64xf32>
                  scf.yield %56, %57, %58, %59, %42, %45, %49, %53 : vector<64xf32>, vector<64xf32>, vector<64xf32>, vector<64xf32>, index, index, index, index
                }
                %28 = vector.shape_cast %27#0 : vector<64xf32> to vector<1x1x8x8xf32>
                %29 = vector.shape_cast %27#1 : vector<64xf32> to vector<1x1x8x8xf32>
                %30 = vector.shape_cast %27#2 : vector<64xf32> to vector<1x1x8x8xf32>
                %31 = vector.shape_cast %27#3 : vector<64xf32> to vector<1x1x8x8xf32>
                %32 = arith.truncf %31 : vector<1x1x8x8xf32> to vector<1x1x8x8xbf16>
                %33 = arith.truncf %30 : vector<1x1x8x8xf32> to vector<1x1x8x8xbf16>
                %34 = arith.truncf %29 : vector<1x1x8x8xf32> to vector<1x1x8x8xbf16>
                %35 = arith.truncf %28 : vector<1x1x8x8xf32> to vector<1x1x8x8xbf16>
                vector.transfer_write %32, %arg31[%10, %12, %c0_5, %c0_5] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xbf16>, memref<32x32x8x8xbf16, 2>
                vector.transfer_write %33, %arg31[%7, %12, %c0_5, %c0_5] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xbf16>, memref<32x32x8x8xbf16, 2>
                vector.transfer_write %34, %arg31[%10, %8, %c0_5, %c0_5] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xbf16>, memref<32x32x8x8xbf16, 2>
                vector.transfer_write %35, %arg31[%7, %8, %c0_5, %c0_5] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xbf16>, memref<32x32x8x8xbf16, 2>
              }
            }
            memref.dealloc %alloc_8 : memref<8x8x8x8xbf16, 2>
            memref.dealloc %alloc_9 : memref<8x8x8x8xbf16, 2>
          }
        }
        air.herd @herd_0  tile (%arg23, %arg24) in (%arg25=%c4, %arg26=%c4) args(%arg27=%alloc_2, %arg28=%alloc_3) : memref<256x256xbf16, 1>, memref<32x32x8x8xbf16, 2> {
          %c64_4 = arith.constant 64 : index
          %c256_5 = arith.constant 256 : index
          %c1_6 = arith.constant 1 : index
          %c0_7 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c2048 = arith.constant 2048 : index
          %3 = affine.apply #map2()[%arg23]
          %4 = affine.apply #map2()[%arg24]
          %5 = affine.apply #map11()[%arg23]
          %6 = affine.apply #map11()[%arg24]
          air.dma_memcpy_nd (%arg27[%3, %4] [%c64_4, %c64_4] [%c256_5, %c1_6], %arg28[%5, %c0_7, %6, %c0_7] [%c8, %c8, %c8, %c8] [%c64_4, %c8, %c2048, %c1_6]) : (memref<256x256xbf16, 1>, memref<32x32x8x8xbf16, 2>)
        }
        air.dma_memcpy_nd (%arg22[%c0, %2] [%c256, %c256] [%c256, %c1_0], %alloc_2[] [] []) {id = 3 : i32} : (memref<256x256xbf16>, memref<256x256xbf16, 1>)
        memref.dealloc %alloc_2 : memref<256x256xbf16, 1>
        memref.dealloc %alloc_3 : memref<32x32x8x8xbf16, 2>
      }
    }

    return
  }
}