; ModuleID = 'air_project_2h_ref/attn_seg_core_2_2.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@buf32_unroll_0 = external global [64 x [1 x bfloat]]
@buf33_unroll_0 = external global [64 x [1 x bfloat]]
@buf34_unroll_0 = external global [64 x [1 x bfloat]]
@buf35_unroll_0 = external global [64 x [1 x bfloat]]
@buf36_unroll_0 = external global [64 x [1 x bfloat]]
@buf37_unroll_0 = external global [64 x [1 x bfloat]]
@buf38_unroll_0 = external global [64 x [64 x bfloat]]
@buf39_unroll_0 = external global [64 x [1 x bfloat]]
@buf40_unroll_0 = external global [64 x [1 x bfloat]]
@buf41_unroll_0 = external global [64 x [64 x bfloat]]
@buf42_unroll_0 = external global [64 x [64 x bfloat]]
@buf43_unroll_0 = external global [64 x [64 x bfloat]]
@buf44_unroll_0 = external global [64 x [64 x bfloat]]
@buf45_unroll_0 = external global [64 x [64 x bfloat]]
@buf46_unroll_0 = external global [64 x [1 x bfloat]]
@buf47_unroll_0 = external global [64 x [1 x bfloat]]

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
define void @core_2_2() local_unnamed_addr #2 {
  br label %.preheader5.preheader

.preheader5.preheader:                            ; preds = %19, %0
  tail call void @llvm.aie2p.acquire(i32 49, i32 -1)
  tail call void @zero_fill_gp_bf16(ptr nonnull @buf45_unroll_0)
  tail call void @zero_fill_sp_bf16(ptr nonnull @buf47_unroll_0)
  tail call void @neg_inf_fill_up_bf16(ptr nonnull @buf46_unroll_0)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @copy_tile(ptr nonnull @buf44_unroll_0, ptr nonnull @buf43_unroll_0)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @zero_fill_g_bf16(ptr nonnull @buf41_unroll_0)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @matmul_a_b_bf16(ptr nonnull @buf43_unroll_0, ptr nonnull @buf44_unroll_0, ptr nonnull @buf41_unroll_0)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @fused_softmax(ptr nonnull @buf41_unroll_0, ptr nonnull @buf46_unroll_0, ptr nonnull @buf40_unroll_0, ptr nonnull @buf39_unroll_0)
  tail call void @mul_r_gp(ptr nonnull @buf39_unroll_0, ptr nonnull @buf45_unroll_0)
  tail call void @matmul_g_b_bf16(ptr nonnull @buf41_unroll_0, ptr nonnull @buf42_unroll_0, ptr nonnull @buf45_unroll_0)
  tail call void @accum_sp_r_s(ptr nonnull @buf47_unroll_0, ptr nonnull @buf39_unroll_0, ptr nonnull @buf40_unroll_0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf40_unroll_0, ptr nonnull @buf47_unroll_0)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  tail call void @zero_fill_g_bf16(ptr nonnull @buf41_unroll_0)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 52, i32 -1)
  tail call void @matmul_a_b_bf16(ptr nonnull @buf43_unroll_0, ptr nonnull @buf44_unroll_0, ptr nonnull @buf41_unroll_0)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @fused_softmax(ptr nonnull @buf41_unroll_0, ptr nonnull @buf46_unroll_0, ptr nonnull @buf40_unroll_0, ptr nonnull @buf39_unroll_0)
  tail call void @mul_r_gp(ptr nonnull @buf39_unroll_0, ptr nonnull @buf45_unroll_0)
  tail call void @matmul_g_b_bf16(ptr nonnull @buf41_unroll_0, ptr nonnull @buf42_unroll_0, ptr nonnull @buf45_unroll_0)
  tail call void @accum_sp_r_s(ptr nonnull @buf47_unroll_0, ptr nonnull @buf39_unroll_0, ptr nonnull @buf40_unroll_0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf40_unroll_0, ptr nonnull @buf47_unroll_0)
  tail call void @llvm.aie2p.release(i32 53, i32 1)
  br label %.preheader5

.preheader5:                                      ; preds = %.preheader5.preheader, %.preheader5
  %1 = phi i32 [ %5, %.preheader5 ], [ 0, %.preheader5.preheader ]
  %2 = tail call <16 x i32> @llvm.aie2p.scd.read.vec(i32 1)
  %3 = trunc nuw i32 %1 to i20
  %4 = getelementptr bfloat, ptr @buf38_unroll_0, i20 %3
  store <16 x i32> %2, ptr %4, align 64
  %5 = add nuw nsw i32 %1, 32
  %6 = icmp ult i32 %1, 4064
  br i1 %6, label %.preheader5, label %.preheader4

.preheader4:                                      ; preds = %.preheader5, %.preheader4
  %7 = phi i32 [ %11, %.preheader4 ], [ 0, %.preheader5 ]
  %8 = tail call <16 x i32> @llvm.aie2p.scd.read.vec(i32 1)
  %9 = trunc nuw i32 %7 to i20
  %10 = getelementptr bfloat, ptr @buf37_unroll_0, i20 %9
  store <16 x i32> %8, ptr %10, align 64
  %11 = add nuw nsw i32 %7, 32
  %12 = icmp eq i32 %7, 0
  br i1 %12, label %.preheader4, label %.preheader

.preheader:                                       ; preds = %.preheader4, %.preheader
  %13 = phi i32 [ %17, %.preheader ], [ 0, %.preheader4 ]
  %14 = tail call <16 x i32> @llvm.aie2p.scd.read.vec(i32 1)
  %15 = trunc nuw i32 %13 to i20
  %16 = getelementptr bfloat, ptr @buf36_unroll_0, i20 %15
  store <16 x i32> %14, ptr %16, align 64
  %17 = add nuw nsw i32 %13, 32
  %18 = icmp eq i32 %13, 0
  br i1 %18, label %.preheader, label %19

19:                                               ; preds = %.preheader
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf46_unroll_0, ptr nonnull @buf35_unroll_0)
  tail call void @maximum_up_u_bf16(ptr nonnull @buf37_unroll_0, ptr nonnull @buf46_unroll_0)
  tail call void @exp_up_minus_u(ptr nonnull @buf37_unroll_0, ptr nonnull @buf46_unroll_0, ptr nonnull @buf34_unroll_0)
  tail call void @exp_up_minus_u(ptr nonnull @buf35_unroll_0, ptr nonnull @buf46_unroll_0, ptr nonnull @buf33_unroll_0)
  tail call void @mul_r_gp(ptr nonnull @buf34_unroll_0, ptr nonnull @buf38_unroll_0)
  tail call void @mul_r_gp(ptr nonnull @buf33_unroll_0, ptr nonnull @buf45_unroll_0)
  tail call void @add_gp_g(ptr nonnull @buf45_unroll_0, ptr nonnull @buf38_unroll_0)
  tail call void @zero_fill_sp_bf16(ptr nonnull @buf32_unroll_0)
  tail call void @accum_sp_r_s(ptr nonnull @buf36_unroll_0, ptr nonnull @buf34_unroll_0, ptr nonnull @buf32_unroll_0)
  tail call void @accum_sp_r_s(ptr nonnull @buf47_unroll_0, ptr nonnull @buf33_unroll_0, ptr nonnull @buf32_unroll_0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf32_unroll_0, ptr nonnull @buf36_unroll_0)
  tail call void @div_gp_sp(ptr nonnull @buf36_unroll_0, ptr nonnull @buf38_unroll_0)
  tail call void @llvm.aie2p.release(i32 48, i32 1)
  br label %.preheader5.preheader
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #1 = { nounwind }
attributes #2 = { noreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
