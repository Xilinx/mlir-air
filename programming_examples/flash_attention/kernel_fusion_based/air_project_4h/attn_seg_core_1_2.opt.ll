; ModuleID = 'air_project_4h/attn_seg_core_1_2.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@buf16_unroll_0 = external global [64 x [1 x bfloat]]
@buf17_unroll_0 = external global [64 x [1 x bfloat]]
@buf18_unroll_0 = external global [64 x [1 x bfloat]]
@buf19_unroll_0 = external global [64 x [1 x bfloat]]
@buf20_unroll_0 = external global [64 x [1 x bfloat]]
@buf21_unroll_0 = external global [64 x [1 x bfloat]]
@buf22_unroll_0 = external global [64 x [64 x bfloat]]
@buf23_unroll_0 = external global [64 x [1 x bfloat]]
@buf24_unroll_0 = external global [64 x [1 x bfloat]]
@buf25_unroll_0 = external global [64 x [64 x bfloat]]
@buf26_unroll_0 = external global [64 x [64 x bfloat]]
@buf27_unroll_0 = external global [64 x [64 x bfloat]]
@buf28_unroll_0 = external global [64 x [64 x bfloat]]
@buf29_unroll_0 = external global [64 x [64 x bfloat]]
@buf30_unroll_0 = external global [64 x [1 x bfloat]]
@buf31_unroll_0 = external global [64 x [1 x bfloat]]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare <16 x i32> @llvm.aie2p.scd.read.vec(i32) #0

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

declare void @maximum_up_u_bf16(ptr, ptr) local_unnamed_addr

declare void @exp_up_minus_u(ptr, ptr, ptr) local_unnamed_addr

declare void @add_gp_g(ptr, ptr) local_unnamed_addr

declare void @div_gp_sp(ptr, ptr) local_unnamed_addr

; Function Attrs: noreturn
define void @core_1_2() local_unnamed_addr #2 {
  br label %.preheader5.preheader

.preheader5.preheader:                            ; preds = %19, %0
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @zero_fill_gp_bf16(ptr nonnull @buf29_unroll_0)
  tail call void @zero_fill_sp_bf16(ptr nonnull @buf31_unroll_0)
  tail call void @neg_inf_fill_up_bf16(ptr nonnull @buf30_unroll_0)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @copy_tile(ptr nonnull @buf28_unroll_0, ptr nonnull @buf27_unroll_0)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @zero_fill_g_bf16(ptr nonnull @buf25_unroll_0)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @matmul_a_b_bf16(ptr nonnull @buf27_unroll_0, ptr nonnull @buf28_unroll_0, ptr nonnull @buf25_unroll_0)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @fused_softmax(ptr nonnull @buf25_unroll_0, ptr nonnull @buf30_unroll_0, ptr nonnull @buf24_unroll_0, ptr nonnull @buf23_unroll_0)
  tail call void @mul_r_gp(ptr nonnull @buf23_unroll_0, ptr nonnull @buf29_unroll_0)
  tail call void @matmul_g_b_bf16(ptr nonnull @buf25_unroll_0, ptr nonnull @buf26_unroll_0, ptr nonnull @buf29_unroll_0)
  tail call void @accum_sp_r_s(ptr nonnull @buf31_unroll_0, ptr nonnull @buf23_unroll_0, ptr nonnull @buf24_unroll_0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf24_unroll_0, ptr nonnull @buf31_unroll_0)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @zero_fill_g_bf16(ptr nonnull @buf25_unroll_0)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @matmul_a_b_bf16(ptr nonnull @buf27_unroll_0, ptr nonnull @buf28_unroll_0, ptr nonnull @buf25_unroll_0)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @fused_softmax(ptr nonnull @buf25_unroll_0, ptr nonnull @buf30_unroll_0, ptr nonnull @buf24_unroll_0, ptr nonnull @buf23_unroll_0)
  tail call void @mul_r_gp(ptr nonnull @buf23_unroll_0, ptr nonnull @buf29_unroll_0)
  tail call void @matmul_g_b_bf16(ptr nonnull @buf25_unroll_0, ptr nonnull @buf26_unroll_0, ptr nonnull @buf29_unroll_0)
  tail call void @accum_sp_r_s(ptr nonnull @buf31_unroll_0, ptr nonnull @buf23_unroll_0, ptr nonnull @buf24_unroll_0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf24_unroll_0, ptr nonnull @buf31_unroll_0)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @zero_fill_g_bf16(ptr nonnull @buf25_unroll_0)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @matmul_a_b_bf16(ptr nonnull @buf27_unroll_0, ptr nonnull @buf28_unroll_0, ptr nonnull @buf25_unroll_0)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @fused_softmax(ptr nonnull @buf25_unroll_0, ptr nonnull @buf30_unroll_0, ptr nonnull @buf24_unroll_0, ptr nonnull @buf23_unroll_0)
  tail call void @mul_r_gp(ptr nonnull @buf23_unroll_0, ptr nonnull @buf29_unroll_0)
  tail call void @matmul_g_b_bf16(ptr nonnull @buf25_unroll_0, ptr nonnull @buf26_unroll_0, ptr nonnull @buf29_unroll_0)
  tail call void @accum_sp_r_s(ptr nonnull @buf31_unroll_0, ptr nonnull @buf23_unroll_0, ptr nonnull @buf24_unroll_0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf24_unroll_0, ptr nonnull @buf31_unroll_0)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @zero_fill_g_bf16(ptr nonnull @buf25_unroll_0)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @matmul_a_b_bf16(ptr nonnull @buf27_unroll_0, ptr nonnull @buf28_unroll_0, ptr nonnull @buf25_unroll_0)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @fused_softmax(ptr nonnull @buf25_unroll_0, ptr nonnull @buf30_unroll_0, ptr nonnull @buf24_unroll_0, ptr nonnull @buf23_unroll_0)
  tail call void @mul_r_gp(ptr nonnull @buf23_unroll_0, ptr nonnull @buf29_unroll_0)
  tail call void @matmul_g_b_bf16(ptr nonnull @buf25_unroll_0, ptr nonnull @buf26_unroll_0, ptr nonnull @buf29_unroll_0)
  tail call void @accum_sp_r_s(ptr nonnull @buf31_unroll_0, ptr nonnull @buf23_unroll_0, ptr nonnull @buf24_unroll_0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf24_unroll_0, ptr nonnull @buf31_unroll_0)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @zero_fill_g_bf16(ptr nonnull @buf25_unroll_0)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @matmul_a_b_bf16(ptr nonnull @buf27_unroll_0, ptr nonnull @buf28_unroll_0, ptr nonnull @buf25_unroll_0)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @fused_softmax(ptr nonnull @buf25_unroll_0, ptr nonnull @buf30_unroll_0, ptr nonnull @buf24_unroll_0, ptr nonnull @buf23_unroll_0)
  tail call void @mul_r_gp(ptr nonnull @buf23_unroll_0, ptr nonnull @buf29_unroll_0)
  tail call void @matmul_g_b_bf16(ptr nonnull @buf25_unroll_0, ptr nonnull @buf26_unroll_0, ptr nonnull @buf29_unroll_0)
  tail call void @accum_sp_r_s(ptr nonnull @buf31_unroll_0, ptr nonnull @buf23_unroll_0, ptr nonnull @buf24_unroll_0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf24_unroll_0, ptr nonnull @buf31_unroll_0)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @zero_fill_g_bf16(ptr nonnull @buf25_unroll_0)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @matmul_a_b_bf16(ptr nonnull @buf27_unroll_0, ptr nonnull @buf28_unroll_0, ptr nonnull @buf25_unroll_0)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @fused_softmax(ptr nonnull @buf25_unroll_0, ptr nonnull @buf30_unroll_0, ptr nonnull @buf24_unroll_0, ptr nonnull @buf23_unroll_0)
  tail call void @mul_r_gp(ptr nonnull @buf23_unroll_0, ptr nonnull @buf29_unroll_0)
  tail call void @matmul_g_b_bf16(ptr nonnull @buf25_unroll_0, ptr nonnull @buf26_unroll_0, ptr nonnull @buf29_unroll_0)
  tail call void @accum_sp_r_s(ptr nonnull @buf31_unroll_0, ptr nonnull @buf23_unroll_0, ptr nonnull @buf24_unroll_0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf24_unroll_0, ptr nonnull @buf31_unroll_0)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @zero_fill_g_bf16(ptr nonnull @buf25_unroll_0)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @matmul_a_b_bf16(ptr nonnull @buf27_unroll_0, ptr nonnull @buf28_unroll_0, ptr nonnull @buf25_unroll_0)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @fused_softmax(ptr nonnull @buf25_unroll_0, ptr nonnull @buf30_unroll_0, ptr nonnull @buf24_unroll_0, ptr nonnull @buf23_unroll_0)
  tail call void @mul_r_gp(ptr nonnull @buf23_unroll_0, ptr nonnull @buf29_unroll_0)
  tail call void @matmul_g_b_bf16(ptr nonnull @buf25_unroll_0, ptr nonnull @buf26_unroll_0, ptr nonnull @buf29_unroll_0)
  tail call void @accum_sp_r_s(ptr nonnull @buf31_unroll_0, ptr nonnull @buf23_unroll_0, ptr nonnull @buf24_unroll_0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf24_unroll_0, ptr nonnull @buf31_unroll_0)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @zero_fill_g_bf16(ptr nonnull @buf25_unroll_0)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @matmul_a_b_bf16(ptr nonnull @buf27_unroll_0, ptr nonnull @buf28_unroll_0, ptr nonnull @buf25_unroll_0)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @fused_softmax(ptr nonnull @buf25_unroll_0, ptr nonnull @buf30_unroll_0, ptr nonnull @buf24_unroll_0, ptr nonnull @buf23_unroll_0)
  tail call void @mul_r_gp(ptr nonnull @buf23_unroll_0, ptr nonnull @buf29_unroll_0)
  tail call void @matmul_g_b_bf16(ptr nonnull @buf25_unroll_0, ptr nonnull @buf26_unroll_0, ptr nonnull @buf29_unroll_0)
  tail call void @accum_sp_r_s(ptr nonnull @buf31_unroll_0, ptr nonnull @buf23_unroll_0, ptr nonnull @buf24_unroll_0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf24_unroll_0, ptr nonnull @buf31_unroll_0)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  br label %.preheader5

.preheader5:                                      ; preds = %.preheader5.preheader, %.preheader5
  %1 = phi i32 [ %5, %.preheader5 ], [ 0, %.preheader5.preheader ]
  %2 = tail call <16 x i32> @llvm.aie2p.scd.read.vec(i32 1)
  %3 = trunc nuw i32 %1 to i20
  %4 = getelementptr bfloat, ptr @buf22_unroll_0, i20 %3
  store <16 x i32> %2, ptr %4, align 64
  %5 = add nuw nsw i32 %1, 32
  %6 = icmp ult i32 %1, 4064
  br i1 %6, label %.preheader5, label %.preheader4

.preheader4:                                      ; preds = %.preheader5, %.preheader4
  %7 = phi i32 [ %11, %.preheader4 ], [ 0, %.preheader5 ]
  %8 = tail call <16 x i32> @llvm.aie2p.scd.read.vec(i32 1)
  %9 = trunc nuw i32 %7 to i20
  %10 = getelementptr bfloat, ptr @buf21_unroll_0, i20 %9
  store <16 x i32> %8, ptr %10, align 64
  %11 = add nuw nsw i32 %7, 32
  %12 = icmp eq i32 %7, 0
  br i1 %12, label %.preheader4, label %.preheader

.preheader:                                       ; preds = %.preheader4, %.preheader
  %13 = phi i32 [ %17, %.preheader ], [ 0, %.preheader4 ]
  %14 = tail call <16 x i32> @llvm.aie2p.scd.read.vec(i32 1)
  %15 = trunc nuw i32 %13 to i20
  %16 = getelementptr bfloat, ptr @buf20_unroll_0, i20 %15
  store <16 x i32> %14, ptr %16, align 64
  %17 = add nuw nsw i32 %13, 32
  %18 = icmp eq i32 %13, 0
  br i1 %18, label %.preheader, label %19

19:                                               ; preds = %.preheader
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf30_unroll_0, ptr nonnull @buf19_unroll_0)
  tail call void @maximum_up_u_bf16(ptr nonnull @buf21_unroll_0, ptr nonnull @buf30_unroll_0)
  tail call void @exp_up_minus_u(ptr nonnull @buf21_unroll_0, ptr nonnull @buf30_unroll_0, ptr nonnull @buf18_unroll_0)
  tail call void @exp_up_minus_u(ptr nonnull @buf19_unroll_0, ptr nonnull @buf30_unroll_0, ptr nonnull @buf17_unroll_0)
  tail call void @mul_r_gp(ptr nonnull @buf18_unroll_0, ptr nonnull @buf22_unroll_0)
  tail call void @mul_r_gp(ptr nonnull @buf17_unroll_0, ptr nonnull @buf29_unroll_0)
  tail call void @add_gp_g(ptr nonnull @buf29_unroll_0, ptr nonnull @buf22_unroll_0)
  tail call void @zero_fill_sp_bf16(ptr nonnull @buf16_unroll_0)
  tail call void @accum_sp_r_s(ptr nonnull @buf20_unroll_0, ptr nonnull @buf18_unroll_0, ptr nonnull @buf16_unroll_0)
  tail call void @accum_sp_r_s(ptr nonnull @buf31_unroll_0, ptr nonnull @buf17_unroll_0, ptr nonnull @buf16_unroll_0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf16_unroll_0, ptr nonnull @buf20_unroll_0)
  tail call void @div_gp_sp(ptr nonnull @buf20_unroll_0, ptr nonnull @buf22_unroll_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  br label %.preheader5.preheader
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #1 = { nounwind }
attributes #2 = { noreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
