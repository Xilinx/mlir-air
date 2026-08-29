; ModuleID = 'air_project_1h_ref/attn_seg_core_2_3.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@buf96 = external global [64 x [1 x bfloat]]
@buf97 = external global [64 x [1 x bfloat]]
@buf98 = external global [64 x [1 x bfloat]]
@buf99 = external global [64 x [1 x bfloat]]
@buf100 = external global [64 x [1 x bfloat]]
@buf101 = external global [64 x [1 x bfloat]]
@buf102 = external global [64 x [64 x bfloat]]
@buf103 = external global [64 x [1 x bfloat]]
@buf104 = external global [64 x [1 x bfloat]]
@buf105 = external global [64 x [64 x bfloat]]
@buf106 = external global [64 x [64 x bfloat]]
@buf107 = external global [64 x [64 x bfloat]]
@buf108 = external global [64 x [64 x bfloat]]
@buf109 = external global [64 x [64 x bfloat]]
@buf110 = external global [64 x [1 x bfloat]]
@buf111 = external global [64 x [1 x bfloat]]

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.aie2p.mcd.write.vec(<16 x i32>, i32) #0

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

; Function Attrs: noreturn
define void @core_2_3() local_unnamed_addr #2 {
  br label %.loopexit

.loopexit:                                        ; preds = %.preheader, %0
  tail call void @zero_fill_gp_bf16(ptr nonnull @buf109)
  tail call void @zero_fill_sp_bf16(ptr nonnull @buf111)
  tail call void @neg_inf_fill_up_bf16(ptr nonnull @buf110)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @copy_tile(ptr nonnull @buf108, ptr nonnull @buf107)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @zero_fill_g_bf16(ptr nonnull @buf105)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @matmul_a_b_bf16(ptr nonnull @buf107, ptr nonnull @buf108, ptr nonnull @buf105)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @fused_softmax(ptr nonnull @buf105, ptr nonnull @buf110, ptr nonnull @buf104, ptr nonnull @buf103)
  tail call void @mul_r_gp(ptr nonnull @buf103, ptr nonnull @buf109)
  tail call void @matmul_g_b_bf16(ptr nonnull @buf105, ptr nonnull @buf106, ptr nonnull @buf109)
  tail call void @accum_sp_r_s(ptr nonnull @buf111, ptr nonnull @buf103, ptr nonnull @buf104)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf104, ptr nonnull @buf111)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @zero_fill_g_bf16(ptr nonnull @buf105)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @matmul_a_b_bf16(ptr nonnull @buf107, ptr nonnull @buf108, ptr nonnull @buf105)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @fused_softmax(ptr nonnull @buf105, ptr nonnull @buf110, ptr nonnull @buf104, ptr nonnull @buf103)
  tail call void @mul_r_gp(ptr nonnull @buf103, ptr nonnull @buf109)
  tail call void @matmul_g_b_bf16(ptr nonnull @buf105, ptr nonnull @buf106, ptr nonnull @buf109)
  tail call void @accum_sp_r_s(ptr nonnull @buf111, ptr nonnull @buf103, ptr nonnull @buf104)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf104, ptr nonnull @buf111)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  br label %.preheader10

.preheader10:                                     ; preds = %.loopexit, %.preheader10
  %1 = phi i32 [ %5, %.preheader10 ], [ 0, %.loopexit ]
  %2 = tail call <16 x i32> @llvm.aie2p.scd.read.vec(i32 1)
  %3 = trunc nuw i32 %1 to i20
  %4 = getelementptr bfloat, ptr @buf102, i20 %3
  store <16 x i32> %2, ptr %4, align 64
  %5 = add nuw nsw i32 %1, 32
  %6 = icmp ult i32 %1, 4064
  br i1 %6, label %.preheader10, label %.preheader9

.preheader9:                                      ; preds = %.preheader10, %.preheader9
  %7 = phi i32 [ %11, %.preheader9 ], [ 0, %.preheader10 ]
  %8 = tail call <16 x i32> @llvm.aie2p.scd.read.vec(i32 1)
  %9 = trunc nuw i32 %7 to i20
  %10 = getelementptr bfloat, ptr @buf101, i20 %9
  store <16 x i32> %8, ptr %10, align 64
  %11 = add nuw nsw i32 %7, 32
  %12 = icmp eq i32 %7, 0
  br i1 %12, label %.preheader9, label %.preheader8

.preheader8:                                      ; preds = %.preheader9, %.preheader8
  %13 = phi i32 [ %17, %.preheader8 ], [ 0, %.preheader9 ]
  %14 = tail call <16 x i32> @llvm.aie2p.scd.read.vec(i32 1)
  %15 = trunc nuw i32 %13 to i20
  %16 = getelementptr bfloat, ptr @buf100, i20 %15
  store <16 x i32> %14, ptr %16, align 64
  %17 = add nuw nsw i32 %13, 32
  %18 = icmp eq i32 %13, 0
  br i1 %18, label %.preheader8, label %19

19:                                               ; preds = %.preheader8
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf110, ptr nonnull @buf99)
  tail call void @maximum_up_u_bf16(ptr nonnull @buf101, ptr nonnull @buf110)
  tail call void @exp_up_minus_u(ptr nonnull @buf101, ptr nonnull @buf110, ptr nonnull @buf98)
  tail call void @exp_up_minus_u(ptr nonnull @buf99, ptr nonnull @buf110, ptr nonnull @buf97)
  tail call void @mul_r_gp(ptr nonnull @buf98, ptr nonnull @buf102)
  tail call void @mul_r_gp(ptr nonnull @buf97, ptr nonnull @buf109)
  tail call void @add_gp_g(ptr nonnull @buf109, ptr nonnull @buf102)
  tail call void @zero_fill_sp_bf16(ptr nonnull @buf96)
  tail call void @accum_sp_r_s(ptr nonnull @buf100, ptr nonnull @buf98, ptr nonnull @buf96)
  tail call void @accum_sp_r_s(ptr nonnull @buf111, ptr nonnull @buf97, ptr nonnull @buf96)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf96, ptr nonnull @buf100)
  br label %20

20:                                               ; preds = %19, %20
  %21 = phi i32 [ 0, %19 ], [ %25, %20 ]
  %22 = trunc nuw i32 %21 to i20
  %23 = getelementptr bfloat, ptr @buf102, i20 %22
  %24 = load <16 x i32>, ptr %23, align 64
  tail call void @llvm.aie2p.mcd.write.vec(<16 x i32> %24, i32 1)
  %25 = add nuw nsw i32 %21, 32
  %26 = icmp ult i32 %21, 4064
  br i1 %26, label %20, label %.preheader7

.preheader7:                                      ; preds = %20, %.preheader7
  %27 = phi i32 [ %31, %.preheader7 ], [ 0, %20 ]
  %28 = trunc nuw i32 %27 to i20
  %29 = getelementptr bfloat, ptr @buf110, i20 %28
  %30 = load <16 x i32>, ptr %29, align 64
  tail call void @llvm.aie2p.mcd.write.vec(<16 x i32> %30, i32 1)
  %31 = add nuw nsw i32 %27, 32
  %32 = icmp eq i32 %27, 0
  br i1 %32, label %.preheader7, label %.preheader

.preheader:                                       ; preds = %.preheader7, %.preheader
  %33 = phi i32 [ %37, %.preheader ], [ 0, %.preheader7 ]
  %34 = trunc nuw i32 %33 to i20
  %35 = getelementptr bfloat, ptr @buf100, i20 %34
  %36 = load <16 x i32>, ptr %35, align 64
  tail call void @llvm.aie2p.mcd.write.vec(<16 x i32> %36, i32 1)
  %37 = add nuw nsw i32 %33, 32
  %38 = icmp eq i32 %33, 0
  br i1 %38, label %.preheader, label %.loopexit
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
attributes #1 = { nounwind }
attributes #2 = { noreturn }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
