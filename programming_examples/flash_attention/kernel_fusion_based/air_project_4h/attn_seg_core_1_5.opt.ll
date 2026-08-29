; ModuleID = 'air_project_4h/attn_seg_core_1_5.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@buf201_unroll_0 = external global [64 x [1 x bfloat]]
@buf202_unroll_0 = external global [64 x [1 x bfloat]]
@buf203_unroll_0 = external global [64 x [64 x bfloat]]
@buf204_unroll_0 = external global [64 x [64 x bfloat]]
@buf205_unroll_0 = external global [64 x [64 x bfloat]]
@buf206_unroll_0 = external global [64 x [64 x bfloat]]
@buf207_unroll_0 = external global [64 x [64 x bfloat]]
@buf208_unroll_0 = external global [64 x [1 x bfloat]]
@buf209_unroll_0 = external global [64 x [1 x bfloat]]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.aie2p.mcd.write.vec(<16 x i32>, i32) #0

; Function Attrs: nounwind
declare void @llvm.aie2p.acquire(i32, i32) #1

; Function Attrs: nounwind
declare void @llvm.aie2p.release(i32, i32) #1

declare void @zero_fill_gp_bf16(ptr) local_unnamed_addr

declare void @zero_fill_sp_bf16(ptr) local_unnamed_addr

declare void @neg_inf_fill_up_bf16(ptr) local_unnamed_addr

declare void @copy_tile(ptr, ptr) local_unnamed_addr

declare void @zero_fill_g_bf16(ptr) local_unnamed_addr

declare void @matmul_a_b_bf16(ptr, ptr, ptr) local_unnamed_addr

declare void @fused_softmax(ptr, ptr, ptr, ptr) local_unnamed_addr

declare void @mul_r_gp(ptr, ptr) local_unnamed_addr

declare void @matmul_g_b_bf16(ptr, ptr, ptr) local_unnamed_addr

declare void @accum_sp_r_s(ptr, ptr, ptr) local_unnamed_addr

declare void @vector_copy_32elems(i32, ptr, ptr) local_unnamed_addr

; Function Attrs: noreturn
define void @core_1_5() local_unnamed_addr #2 {
  br label %.loopexit

.loopexit:                                        ; preds = %.preheader, %0
  tail call void @zero_fill_gp_bf16(ptr nonnull @buf207_unroll_0)
  tail call void @zero_fill_sp_bf16(ptr nonnull @buf209_unroll_0)
  tail call void @neg_inf_fill_up_bf16(ptr nonnull @buf208_unroll_0)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @copy_tile(ptr nonnull @buf206_unroll_0, ptr nonnull @buf205_unroll_0)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @zero_fill_g_bf16(ptr nonnull @buf203_unroll_0)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @matmul_a_b_bf16(ptr nonnull @buf205_unroll_0, ptr nonnull @buf206_unroll_0, ptr nonnull @buf203_unroll_0)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @fused_softmax(ptr nonnull @buf203_unroll_0, ptr nonnull @buf208_unroll_0, ptr nonnull @buf202_unroll_0, ptr nonnull @buf201_unroll_0)
  tail call void @mul_r_gp(ptr nonnull @buf201_unroll_0, ptr nonnull @buf207_unroll_0)
  tail call void @matmul_g_b_bf16(ptr nonnull @buf203_unroll_0, ptr nonnull @buf204_unroll_0, ptr nonnull @buf207_unroll_0)
  tail call void @accum_sp_r_s(ptr nonnull @buf209_unroll_0, ptr nonnull @buf201_unroll_0, ptr nonnull @buf202_unroll_0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf202_unroll_0, ptr nonnull @buf209_unroll_0)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @zero_fill_g_bf16(ptr nonnull @buf203_unroll_0)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @matmul_a_b_bf16(ptr nonnull @buf205_unroll_0, ptr nonnull @buf206_unroll_0, ptr nonnull @buf203_unroll_0)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @fused_softmax(ptr nonnull @buf203_unroll_0, ptr nonnull @buf208_unroll_0, ptr nonnull @buf202_unroll_0, ptr nonnull @buf201_unroll_0)
  tail call void @mul_r_gp(ptr nonnull @buf201_unroll_0, ptr nonnull @buf207_unroll_0)
  tail call void @matmul_g_b_bf16(ptr nonnull @buf203_unroll_0, ptr nonnull @buf204_unroll_0, ptr nonnull @buf207_unroll_0)
  tail call void @accum_sp_r_s(ptr nonnull @buf209_unroll_0, ptr nonnull @buf201_unroll_0, ptr nonnull @buf202_unroll_0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf202_unroll_0, ptr nonnull @buf209_unroll_0)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @zero_fill_g_bf16(ptr nonnull @buf203_unroll_0)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @matmul_a_b_bf16(ptr nonnull @buf205_unroll_0, ptr nonnull @buf206_unroll_0, ptr nonnull @buf203_unroll_0)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @fused_softmax(ptr nonnull @buf203_unroll_0, ptr nonnull @buf208_unroll_0, ptr nonnull @buf202_unroll_0, ptr nonnull @buf201_unroll_0)
  tail call void @mul_r_gp(ptr nonnull @buf201_unroll_0, ptr nonnull @buf207_unroll_0)
  tail call void @matmul_g_b_bf16(ptr nonnull @buf203_unroll_0, ptr nonnull @buf204_unroll_0, ptr nonnull @buf207_unroll_0)
  tail call void @accum_sp_r_s(ptr nonnull @buf209_unroll_0, ptr nonnull @buf201_unroll_0, ptr nonnull @buf202_unroll_0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf202_unroll_0, ptr nonnull @buf209_unroll_0)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @zero_fill_g_bf16(ptr nonnull @buf203_unroll_0)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @matmul_a_b_bf16(ptr nonnull @buf205_unroll_0, ptr nonnull @buf206_unroll_0, ptr nonnull @buf203_unroll_0)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @fused_softmax(ptr nonnull @buf203_unroll_0, ptr nonnull @buf208_unroll_0, ptr nonnull @buf202_unroll_0, ptr nonnull @buf201_unroll_0)
  tail call void @mul_r_gp(ptr nonnull @buf201_unroll_0, ptr nonnull @buf207_unroll_0)
  tail call void @matmul_g_b_bf16(ptr nonnull @buf203_unroll_0, ptr nonnull @buf204_unroll_0, ptr nonnull @buf207_unroll_0)
  tail call void @accum_sp_r_s(ptr nonnull @buf209_unroll_0, ptr nonnull @buf201_unroll_0, ptr nonnull @buf202_unroll_0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf202_unroll_0, ptr nonnull @buf209_unroll_0)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @zero_fill_g_bf16(ptr nonnull @buf203_unroll_0)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @matmul_a_b_bf16(ptr nonnull @buf205_unroll_0, ptr nonnull @buf206_unroll_0, ptr nonnull @buf203_unroll_0)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @fused_softmax(ptr nonnull @buf203_unroll_0, ptr nonnull @buf208_unroll_0, ptr nonnull @buf202_unroll_0, ptr nonnull @buf201_unroll_0)
  tail call void @mul_r_gp(ptr nonnull @buf201_unroll_0, ptr nonnull @buf207_unroll_0)
  tail call void @matmul_g_b_bf16(ptr nonnull @buf203_unroll_0, ptr nonnull @buf204_unroll_0, ptr nonnull @buf207_unroll_0)
  tail call void @accum_sp_r_s(ptr nonnull @buf209_unroll_0, ptr nonnull @buf201_unroll_0, ptr nonnull @buf202_unroll_0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf202_unroll_0, ptr nonnull @buf209_unroll_0)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @zero_fill_g_bf16(ptr nonnull @buf203_unroll_0)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @matmul_a_b_bf16(ptr nonnull @buf205_unroll_0, ptr nonnull @buf206_unroll_0, ptr nonnull @buf203_unroll_0)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @fused_softmax(ptr nonnull @buf203_unroll_0, ptr nonnull @buf208_unroll_0, ptr nonnull @buf202_unroll_0, ptr nonnull @buf201_unroll_0)
  tail call void @mul_r_gp(ptr nonnull @buf201_unroll_0, ptr nonnull @buf207_unroll_0)
  tail call void @matmul_g_b_bf16(ptr nonnull @buf203_unroll_0, ptr nonnull @buf204_unroll_0, ptr nonnull @buf207_unroll_0)
  tail call void @accum_sp_r_s(ptr nonnull @buf209_unroll_0, ptr nonnull @buf201_unroll_0, ptr nonnull @buf202_unroll_0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf202_unroll_0, ptr nonnull @buf209_unroll_0)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @zero_fill_g_bf16(ptr nonnull @buf203_unroll_0)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @matmul_a_b_bf16(ptr nonnull @buf205_unroll_0, ptr nonnull @buf206_unroll_0, ptr nonnull @buf203_unroll_0)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @fused_softmax(ptr nonnull @buf203_unroll_0, ptr nonnull @buf208_unroll_0, ptr nonnull @buf202_unroll_0, ptr nonnull @buf201_unroll_0)
  tail call void @mul_r_gp(ptr nonnull @buf201_unroll_0, ptr nonnull @buf207_unroll_0)
  tail call void @matmul_g_b_bf16(ptr nonnull @buf203_unroll_0, ptr nonnull @buf204_unroll_0, ptr nonnull @buf207_unroll_0)
  tail call void @accum_sp_r_s(ptr nonnull @buf209_unroll_0, ptr nonnull @buf201_unroll_0, ptr nonnull @buf202_unroll_0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf202_unroll_0, ptr nonnull @buf209_unroll_0)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @zero_fill_g_bf16(ptr nonnull @buf203_unroll_0)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @matmul_a_b_bf16(ptr nonnull @buf205_unroll_0, ptr nonnull @buf206_unroll_0, ptr nonnull @buf203_unroll_0)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @fused_softmax(ptr nonnull @buf203_unroll_0, ptr nonnull @buf208_unroll_0, ptr nonnull @buf202_unroll_0, ptr nonnull @buf201_unroll_0)
  tail call void @mul_r_gp(ptr nonnull @buf201_unroll_0, ptr nonnull @buf207_unroll_0)
  tail call void @matmul_g_b_bf16(ptr nonnull @buf203_unroll_0, ptr nonnull @buf204_unroll_0, ptr nonnull @buf207_unroll_0)
  tail call void @accum_sp_r_s(ptr nonnull @buf209_unroll_0, ptr nonnull @buf201_unroll_0, ptr nonnull @buf202_unroll_0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf202_unroll_0, ptr nonnull @buf209_unroll_0)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  br label %.preheader5

.preheader5:                                      ; preds = %.loopexit, %.preheader5
  %1 = phi i32 [ %5, %.preheader5 ], [ 0, %.loopexit ]
  %2 = trunc nuw i32 %1 to i20
  %3 = getelementptr bfloat, ptr @buf207_unroll_0, i20 %2
  %4 = load <16 x i32>, ptr %3, align 64
  tail call void @llvm.aie2p.mcd.write.vec(<16 x i32> %4, i32 1)
  %5 = add nuw nsw i32 %1, 32
  %6 = icmp ult i32 %1, 4064
  br i1 %6, label %.preheader5, label %.preheader4

.preheader4:                                      ; preds = %.preheader5, %.preheader4
  %7 = phi i32 [ %11, %.preheader4 ], [ 0, %.preheader5 ]
  %8 = trunc nuw i32 %7 to i20
  %9 = getelementptr bfloat, ptr @buf208_unroll_0, i20 %8
  %10 = load <16 x i32>, ptr %9, align 64
  tail call void @llvm.aie2p.mcd.write.vec(<16 x i32> %10, i32 1)
  %11 = add nuw nsw i32 %7, 32
  %12 = icmp eq i32 %7, 0
  br i1 %12, label %.preheader4, label %.preheader

.preheader:                                       ; preds = %.preheader4, %.preheader
  %13 = phi i32 [ %17, %.preheader ], [ 0, %.preheader4 ]
  %14 = trunc nuw i32 %13 to i20
  %15 = getelementptr bfloat, ptr @buf209_unroll_0, i20 %14
  %16 = load <16 x i32>, ptr %15, align 64
  tail call void @llvm.aie2p.mcd.write.vec(<16 x i32> %16, i32 1)
  %17 = add nuw nsw i32 %13, 32
  %18 = icmp eq i32 %13, 0
  br i1 %18, label %.preheader, label %.loopexit
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #1 = { nounwind }
attributes #2 = { noreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
