; ModuleID = 'air_project_2h_ref/attn_seg_core_1_4.peanohack.ll'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p:20:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-f32:32:32-i64:32-f64:32-a:0:32-n32"
target triple = "aie2p"

@buf144_unroll_0 = external global [64 x [1 x bfloat]]
@buf145_unroll_0 = external global [64 x [1 x bfloat]]
@buf146_unroll_0 = external global [64 x [1 x bfloat]]
@buf147_unroll_0 = external global [64 x [1 x bfloat]]
@buf148_unroll_0 = external global [64 x [1 x bfloat]]
@buf149_unroll_0 = external global [64 x [1 x bfloat]]
@buf150_unroll_0 = external global [64 x [64 x bfloat]]
@buf151_unroll_0 = external global [64 x [1 x bfloat]]
@buf152_unroll_0 = external global [64 x [1 x bfloat]]
@buf153_unroll_0 = external global [64 x [64 x bfloat]]
@buf154_unroll_0 = external global [64 x [64 x bfloat]]
@buf155_unroll_0 = external global [64 x [64 x bfloat]]
@buf156_unroll_0 = external global [64 x [64 x bfloat]]
@buf157_unroll_0 = external global [64 x [64 x bfloat]]
@buf158_unroll_0 = external global [64 x [1 x bfloat]]
@buf159_unroll_0 = external global [64 x [1 x bfloat]]

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
define void @core_1_4() local_unnamed_addr #2 {
  br label %.loopexit

.loopexit:                                        ; preds = %.preheader, %0
  tail call void @zero_fill_gp_bf16(ptr nonnull @buf157_unroll_0)
  tail call void @zero_fill_sp_bf16(ptr nonnull @buf159_unroll_0)
  tail call void @neg_inf_fill_up_bf16(ptr nonnull @buf158_unroll_0)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @copy_tile(ptr nonnull @buf156_unroll_0, ptr nonnull @buf155_unroll_0)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @zero_fill_g_bf16(ptr nonnull @buf153_unroll_0)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @matmul_a_b_bf16(ptr nonnull @buf155_unroll_0, ptr nonnull @buf156_unroll_0, ptr nonnull @buf153_unroll_0)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @fused_softmax(ptr nonnull @buf153_unroll_0, ptr nonnull @buf158_unroll_0, ptr nonnull @buf152_unroll_0, ptr nonnull @buf151_unroll_0)
  tail call void @mul_r_gp(ptr nonnull @buf151_unroll_0, ptr nonnull @buf157_unroll_0)
  tail call void @matmul_g_b_bf16(ptr nonnull @buf153_unroll_0, ptr nonnull @buf154_unroll_0, ptr nonnull @buf157_unroll_0)
  tail call void @accum_sp_r_s(ptr nonnull @buf159_unroll_0, ptr nonnull @buf151_unroll_0, ptr nonnull @buf152_unroll_0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf152_unroll_0, ptr nonnull @buf159_unroll_0)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  tail call void @zero_fill_g_bf16(ptr nonnull @buf153_unroll_0)
  tail call void @llvm.aie2p.acquire(i32 48, i32 -1)
  tail call void @llvm.aie2p.acquire(i32 50, i32 -1)
  tail call void @matmul_a_b_bf16(ptr nonnull @buf155_unroll_0, ptr nonnull @buf156_unroll_0, ptr nonnull @buf153_unroll_0)
  tail call void @llvm.aie2p.release(i32 49, i32 1)
  tail call void @fused_softmax(ptr nonnull @buf153_unroll_0, ptr nonnull @buf158_unroll_0, ptr nonnull @buf152_unroll_0, ptr nonnull @buf151_unroll_0)
  tail call void @mul_r_gp(ptr nonnull @buf151_unroll_0, ptr nonnull @buf157_unroll_0)
  tail call void @matmul_g_b_bf16(ptr nonnull @buf153_unroll_0, ptr nonnull @buf154_unroll_0, ptr nonnull @buf157_unroll_0)
  tail call void @accum_sp_r_s(ptr nonnull @buf159_unroll_0, ptr nonnull @buf151_unroll_0, ptr nonnull @buf152_unroll_0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf152_unroll_0, ptr nonnull @buf159_unroll_0)
  tail call void @llvm.aie2p.release(i32 51, i32 1)
  br label %.preheader10

.preheader10:                                     ; preds = %.loopexit, %.preheader10
  %1 = phi i32 [ %5, %.preheader10 ], [ 0, %.loopexit ]
  %2 = tail call <16 x i32> @llvm.aie2p.scd.read.vec(i32 1)
  %3 = trunc nuw i32 %1 to i20
  %4 = getelementptr bfloat, ptr @buf150_unroll_0, i20 %3
  store <16 x i32> %2, ptr %4, align 64
  %5 = add nuw nsw i32 %1, 32
  %6 = icmp ult i32 %1, 4064
  br i1 %6, label %.preheader10, label %.preheader9

.preheader9:                                      ; preds = %.preheader10, %.preheader9
  %7 = phi i32 [ %11, %.preheader9 ], [ 0, %.preheader10 ]
  %8 = tail call <16 x i32> @llvm.aie2p.scd.read.vec(i32 1)
  %9 = trunc nuw i32 %7 to i20
  %10 = getelementptr bfloat, ptr @buf149_unroll_0, i20 %9
  store <16 x i32> %8, ptr %10, align 64
  %11 = add nuw nsw i32 %7, 32
  %12 = icmp eq i32 %7, 0
  br i1 %12, label %.preheader9, label %.preheader8

.preheader8:                                      ; preds = %.preheader9, %.preheader8
  %13 = phi i32 [ %17, %.preheader8 ], [ 0, %.preheader9 ]
  %14 = tail call <16 x i32> @llvm.aie2p.scd.read.vec(i32 1)
  %15 = trunc nuw i32 %13 to i20
  %16 = getelementptr bfloat, ptr @buf148_unroll_0, i20 %15
  store <16 x i32> %14, ptr %16, align 64
  %17 = add nuw nsw i32 %13, 32
  %18 = icmp eq i32 %13, 0
  br i1 %18, label %.preheader8, label %19

19:                                               ; preds = %.preheader8
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf158_unroll_0, ptr nonnull @buf147_unroll_0)
  tail call void @maximum_up_u_bf16(ptr nonnull @buf149_unroll_0, ptr nonnull @buf158_unroll_0)
  tail call void @exp_up_minus_u(ptr nonnull @buf149_unroll_0, ptr nonnull @buf158_unroll_0, ptr nonnull @buf146_unroll_0)
  tail call void @exp_up_minus_u(ptr nonnull @buf147_unroll_0, ptr nonnull @buf158_unroll_0, ptr nonnull @buf145_unroll_0)
  tail call void @mul_r_gp(ptr nonnull @buf146_unroll_0, ptr nonnull @buf150_unroll_0)
  tail call void @mul_r_gp(ptr nonnull @buf145_unroll_0, ptr nonnull @buf157_unroll_0)
  tail call void @add_gp_g(ptr nonnull @buf157_unroll_0, ptr nonnull @buf150_unroll_0)
  tail call void @zero_fill_sp_bf16(ptr nonnull @buf144_unroll_0)
  tail call void @accum_sp_r_s(ptr nonnull @buf148_unroll_0, ptr nonnull @buf146_unroll_0, ptr nonnull @buf144_unroll_0)
  tail call void @accum_sp_r_s(ptr nonnull @buf159_unroll_0, ptr nonnull @buf145_unroll_0, ptr nonnull @buf144_unroll_0)
  tail call void @vector_copy_32elems(i32 0, ptr nonnull @buf144_unroll_0, ptr nonnull @buf148_unroll_0)
  br label %20

20:                                               ; preds = %19, %20
  %21 = phi i32 [ 0, %19 ], [ %25, %20 ]
  %22 = trunc nuw i32 %21 to i20
  %23 = getelementptr bfloat, ptr @buf150_unroll_0, i20 %22
  %24 = load <16 x i32>, ptr %23, align 64
  tail call void @llvm.aie2p.mcd.write.vec(<16 x i32> %24, i32 1)
  %25 = add nuw nsw i32 %21, 32
  %26 = icmp ult i32 %21, 4064
  br i1 %26, label %20, label %.preheader7

.preheader7:                                      ; preds = %20, %.preheader7
  %27 = phi i32 [ %31, %.preheader7 ], [ 0, %20 ]
  %28 = trunc nuw i32 %27 to i20
  %29 = getelementptr bfloat, ptr @buf158_unroll_0, i20 %28
  %30 = load <16 x i32>, ptr %29, align 64
  tail call void @llvm.aie2p.mcd.write.vec(<16 x i32> %30, i32 1)
  %31 = add nuw nsw i32 %27, 32
  %32 = icmp eq i32 %27, 0
  br i1 %32, label %.preheader7, label %.preheader

.preheader:                                       ; preds = %.preheader7, %.preheader
  %33 = phi i32 [ %37, %.preheader ], [ 0, %.preheader7 ]
  %34 = trunc nuw i32 %33 to i20
  %35 = getelementptr bfloat, ptr @buf148_unroll_0, i20 %34
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
