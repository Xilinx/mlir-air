//===- conv.cc --------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define __AIENGINE__ 2
#define NOCPP
#define __AIEARCH__ 20

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

#include "zero.cc"

template <typename T_in, typename T_out, unsigned rowA, unsigned colA,
          unsigned colB, unsigned r, unsigned s, unsigned t>
void conv_vectorized(const T_in *__restrict pA, const T_in *__restrict pB,
                     T_out *__restrict pC) {
  using MMUL = aie::mmul<r, s, t, T_in, T_in>;

  static_assert(r == 4);
  static_assert(s == 8);
  static_assert(t == 8);
  static_assert(MMUL::size_A == 32);
  static_assert(MMUL::size_B == 64);

  event0();

  aie::vector<T_out, MMUL::size_C> acc_C00 = aie::load_v<MMUL::size_C>(pC);
  MMUL C00(acc_C00);
  const T_in *__restrict pA1 = pA;
  const T_in *__restrict pB1 = pB;
  for (unsigned i = 0; i < colA; i += 1)
    chess_prepare_for_pipelining chess_loop_range(18, ) {
      aie::vector<T_in, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1);
      aie::vector<T_in, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1);
      C00.mac(A0, B0);
      pA1 += MMUL::size_A;
      pB1 += MMUL::size_B;
    }
  aie::store_v(pC, C00.template to_vector<T_out>());

  event1();
}

template <unsigned m, unsigned k, unsigned n>
void conv_vectorized_4x8x8_i8_i32(const int8 *__restrict pA,
                                  const int8 *__restrict pB,
                                  int *__restrict pC) {
  constexpr int r = 4;
  constexpr int s = 8;
  constexpr int t = 8;
  static_assert(m % r == 0);
  static_assert(k % s == 0);
  static_assert(n % t == 0);
  return conv_vectorized<int8, int, m / r, k / s, n / t, r, s, t>(pA, pB, pC);
}

extern "C" {

void linalg_conv_1d_nwc_wcf_view1x4x8xi8as2_view1x8x8xi8as2_view1x4x8xi32as2(
    int8 *a_in, int8 *b_in, int *c_out) {
  conv_vectorized_4x8x8_i8_i32<4, 8, 8>(a_in, b_in, c_out);
}
void linalg_fill_i32_view1x1x4x8xi32as2(int *c_out) {
  zero_vectorized<int, 4, 8, 32>(c_out);
}

} // extern "C"

//   aie::vector<T_out, MMUL::size_C> acc_C00 =
//       aie::load_v<MMUL::size_C>(pC);
//   MMUL C00(acc_C00);
//   aie::vector<T_in, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA);
//   aie::vector<T_in, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB);
//   C00.mac(A0, B0);
//   aie::store_v(pC, C00.template to_vector<T_out>());

//     vector<T_in, 32> vec_Xbuff0, vec_Xbuff1, vec_Xbuff2, vec_Xbuff3;
//     using MMUL = aie::mmul<r, s, t, T_in, T_in>;

//     vector<T_in, 64> Ybuff0, Ybuff1;

//     #define LOAD_WEIGHTS(wts,wbuff,wts_incr) \
//     {\
//         wbuff.insert(0, load_v<32>(wts));wts+=32;\
//         wbuff.insert(1, load_v<32>(wts));wts+=wts_incr;\
//     }

//     Vecs4_ui8 Xbuff;

//     for (unsigned j=0; j<conv2d_params.outer_g; j++)
//     chess_prepare_for_pipelining
//     chess_loop_range(2,)
//     {
//         if constexpr(batch_size < 4) {
//             v64int8 Sbuff;
//             int pdel;
//             pdel = (long)p_in;
//             frac = (pdel & 31) + 33;
//         }
//         mmul Cbuff0( load_v<32>( p_init16 ), acc_init_shift ); p_init16 =
//         byte_incr(p_init16, inc_ST_0-chnd_T); mmul Cbuff1( load_v<32>(
//         p_init16 ), acc_init_shift ); p_init16 = byte_incr(p_init16,
//         inc_ST_1-chnd_T); mmul Cbuff2( load_v<32>( p_init16 ), acc_init_shift
//         ); p_init16 = byte_incr(p_init16, inc_ST_0-chnd_T); mmul Cbuff3(
//         load_v<32>( p_init16 ), acc_init_shift ); p_init16 =
//         byte_incr(p_init16, inc_ST_2-chnd_T); mmul Cbuff4( load_v<32>(
//         p_init16 ), acc_init_shift ); p_init16 = byte_incr(p_init16,
//         inc_ST_0-chnd_T); mmul Cbuff5( load_v<32>( p_init16 ), acc_init_shift
//         ); p_init16 = byte_incr(p_init16, inc_ST_1-chnd_T); mmul Cbuff6(
//         load_v<32>( p_init16 ), acc_init_shift ); p_init16 =
//         byte_incr(p_init16, inc_ST_0-chnd_T); mmul Cbuff7( load_v<32>(
//         p_init16 ), acc_init_shift ); p_init16 = byte_incr(p_init16,
//         inc_ST_3-chnd_T);

//         for (int i=0; i<conv2d_params.inner_g; i++)
//         chess_prepare_for_pipelining
//         chess_loop_range(4,)
//         {
//             if constexpr(batch_size < 4) {
//                 vector<uint8_t, 64>  Sbuff0,
//                 Sbuff1,Xbuff0,Xbuff1,Xbuff2,Xbuff3;

//                 // Load IFM data, two 4x8 matrices plus additional adjacent
//                 data needed for shift alignment Sbuff0.insert( 0,
//                 load_v<32>((uint8 IFM_DM_BANK *) p_in));  p_in =
//                 byte_incr(p_in, conv2d_params.inc_A); Sbuff0.insert( 1,
//                 load_v<32>((uint8 IFM_DM_BANK *) p_in));  p_in =
//                 byte_incr(p_in, conv2d_params.inc_A); Sbuff1.insert( 0,
//                 load_v<32>((uint8 IFM_DM_BANK *) p_in));  p_in =
//                 byte_incr(p_in, conv2d_params.inc_A); Sbuff1.insert( 1,
//                 load_v<32>((uint8 IFM_DM_BANK *) p_in));

//                 Xbuff.x0 = shiftx(Xbuff.x0, Sbuff0, conv2d_params.step_align,
//                 frac); Xbuff.x2 = shiftx(Xbuff.x2, Sbuff1,
//                 conv2d_params.step_align, frac);

//     int frac;
//     int chnd   = 32;
//     int chnd_T = 64;

//     if constexpr(batch_size < 4) {
//         p_in += 31 ;
//     }

//     int    inc_ST_0       = (conv2d_params.psum_in ==  0) ? chnd_T :
//     2*chnd_T; int    inc_ST_1       = (conv2d_params.psum_in ==  0) ? chnd_T
//     : conv2d_params.inc_ST_0; int    inc_ST_2       = (conv2d_params.psum_in
//     ==  0) ? conv2d_params.inc_acc_1+chnd_T      : conv2d_params.inc_ST_1;
//     int    inc_ST_3       = (conv2d_params.psum_in ==  0) ?
//     conv2d_params.inc_acc_2+chnd_T      : chnd_T;

//     DECL_ITR3D(iterator_psum)
//     SETUP_ITR3D(iterator_psum,
//                 0,
//                 (conv2d_params.psum_in ==  0) ? conv2d_params.num_X : 0,
//                 (conv2d_params.psum_in ==  0) ? conv2d_params.inc_acc_d1 :
//                 conv2d_params.inc_Xo_T-chnd_T, (conv2d_params.psum_in ==  0)
//                 ? conv2d_params.num_Co : conv2d_params.num_X,
//                 (conv2d_params.psum_in ==  0) ? conv2d_params.inc_acc_rev :
//                 conv2d_params.inc_Co_T-chnd_T)

//     int    acc_init_shift = (conv2d_params.psum_in ==  0) ? (int)
//     conv2d_params.shift_bias_init : (int) conv2d_params.shift_psum_in;

//     DECL_ITR2D(iterator_pout)
//     if constexpr(out_mode == PARTIAL){
//         SETUP_ITR2D(iterator_pout,conv2d_params.inc_Xo_T-chnd_T,
//         conv2d_params.num_X,conv2d_params.inc_Co_T-chnd_T);
//     }else{
//         SETUP_ITR2D(iterator_pout,conv2d_params.inc_Xo-chnd,conv2d_params.num_X,conv2d_params.inc_Co-chnd)
//         ;
//     }
//     /*
//     SETUP_ITR2D(iterator_pout,
//                (conv2d_params.psum_out == 0)? conv2d_params.inc_Xo-chnd :
//                conv2d_params.inc_Xo_T-chnd_T, conv2d_params.num_X,
//                (conv2d_params.psum_out == 0)? conv2d_params.inc_Co-chnd :
//                conv2d_params.inc_Co_T-chnd_T)
//     */

//     for (unsigned j=0; j<conv2d_params.outer_g; j++)
//     chess_prepare_for_pipelining
//     chess_loop_range(2,)
//     {
//         if constexpr(batch_size < 4) {
//             v64int8 Sbuff;
//             int pdel;
//             pdel = (long)p_in;
//             frac = (pdel & 31) + 33;
//         }
//         mmul Cbuff0( load_v<32>( p_init16 ), acc_init_shift ); p_init16 =
//         byte_incr(p_init16, inc_ST_0-chnd_T); mmul Cbuff1( load_v<32>(
//         p_init16 ), acc_init_shift ); p_init16 = byte_incr(p_init16,
//         inc_ST_1-chnd_T); mmul Cbuff2( load_v<32>( p_init16 ), acc_init_shift
//         ); p_init16 = byte_incr(p_init16, inc_ST_0-chnd_T); mmul Cbuff3(
//         load_v<32>( p_init16 ), acc_init_shift ); p_init16 =
//         byte_incr(p_init16, inc_ST_2-chnd_T); mmul Cbuff4( load_v<32>(
//         p_init16 ), acc_init_shift ); p_init16 = byte_incr(p_init16,
//         inc_ST_0-chnd_T); mmul Cbuff5( load_v<32>( p_init16 ), acc_init_shift
//         ); p_init16 = byte_incr(p_init16, inc_ST_1-chnd_T); mmul Cbuff6(
//         load_v<32>( p_init16 ), acc_init_shift ); p_init16 =
//         byte_incr(p_init16, inc_ST_0-chnd_T); mmul Cbuff7( load_v<32>(
//         p_init16 ), acc_init_shift ); p_init16 = byte_incr(p_init16,
//         inc_ST_3-chnd_T);

//         for (int i=0; i<conv2d_params.inner_g; i++)
//         chess_prepare_for_pipelining
//         chess_loop_range(4,)
//         {
//             if constexpr(batch_size < 4) {
//                 vector<uint8_t, 64>  Sbuff0,
//                 Sbuff1,Xbuff0,Xbuff1,Xbuff2,Xbuff3;

//                 // Load IFM data, two 4x8 matrices plus additional adjacent
//                 data needed for shift alignment Sbuff0.insert( 0,
//                 load_v<32>((uint8 IFM_DM_BANK *) p_in));  p_in =
//                 byte_incr(p_in, conv2d_params.inc_A); Sbuff0.insert( 1,
//                 load_v<32>((uint8 IFM_DM_BANK *) p_in));  p_in =
//                 byte_incr(p_in, conv2d_params.inc_A); Sbuff1.insert( 0,
//                 load_v<32>((uint8 IFM_DM_BANK *) p_in));  p_in =
//                 byte_incr(p_in, conv2d_params.inc_A); Sbuff1.insert( 1,
//                 load_v<32>((uint8 IFM_DM_BANK *) p_in));

//                 Xbuff.x0 = shiftx(Xbuff.x0, Sbuff0, conv2d_params.step_align,
//                 frac); Xbuff.x2 = shiftx(Xbuff.x2, Sbuff1,
//                 conv2d_params.step_align, frac);

//                 // TODO: CR-1103776
//                 p_in = INCR_ITR3D_PARAM(p_in, iterator_inner,conv2d_params)
//                 //
//                 conv2d_params.iterator_inner_incr0,conv2d_params.iterator_inner_wrap0,conv2d_params.iterator_inner_cnt0
//                 //
//                 conv2d_params.iterator_inner_incr1,conv2d_params.iterator_inner_wrap1,conv2d_params.iterator_inner_cnt1
//                 // conv2d_params.iterator_inner_incr2
//                 Xbuff0 = interleave(Xbuff.x0, Xbuff.x2,
//                 conv2d_params.shfl_0); Xbuff1 = interleave(Xbuff0,   unu8,
//                 T256_2x2_hi); Xbuff2 = interleave(Xbuff.x0, Xbuff.x2,
//                 conv2d_params.shfl_1); Xbuff3 = interleave(Xbuff2,   unu8,
//                 T256_2x2_hi);

//                 vec_Xbuff0 = Xbuff0.cast_to<dtype_ifm>(). template
//                 extract<32>(0); vec_Xbuff1 = Xbuff1.cast_to<dtype_ifm>().
//                 template extract<32>(0); vec_Xbuff2 =
//                 Xbuff2.cast_to<dtype_ifm>(). template extract<32>(0);
//                 vec_Xbuff3 = Xbuff3.cast_to<dtype_ifm>(). template
//                 extract<32>(0);
//             }
//             else {
//                 vec_Xbuff0 = load_v<32>((dtype_ifm IFM_DM_BANK *) p_in); p_in
//                 = byte_incr(p_in, conv2d_params.inc_A); vec_Xbuff1 =
//                 load_v<32>((dtype_ifm IFM_DM_BANK *) p_in); p_in =
//                 byte_incr(p_in, conv2d_params.inc_A); vec_Xbuff2 =
//                 load_v<32>((dtype_ifm IFM_DM_BANK *) p_in); p_in =
//                 byte_incr(p_in, conv2d_params.inc_A); vec_Xbuff3 =
//                 load_v<32>((dtype_ifm IFM_DM_BANK *) p_in);

//                 p_in = INCR_ITR3D_PARAM(p_in, iterator_inner,conv2d_params)
//             }

//             // Load weights -- two matrices 8*8
//             LOAD_WEIGHTS(p_w,Ybuff0,32) ;
//             LOAD_WEIGHTS(p_w,Ybuff1,32) ;

//             #ifdef USE_RTP_SIGN
//             int sign_x=conv2d_params.ifm_sign;
//             Cbuff0.mac(op_sign(vec_Xbuff0,sign_x), op_sign(Ybuff0,1));
//             Cbuff1.mac(op_sign(vec_Xbuff1,sign_x), op_sign(Ybuff0,1));
//             Cbuff2.mac(op_sign(vec_Xbuff2,sign_x), op_sign(Ybuff0,1));
//             Cbuff3.mac(op_sign(vec_Xbuff3,sign_x), op_sign(Ybuff0,1));
//             Cbuff4.mac(op_sign(vec_Xbuff0,sign_x), op_sign(Ybuff1,1));
//             Cbuff5.mac(op_sign(vec_Xbuff1,sign_x), op_sign(Ybuff1,1));
//             Cbuff6.mac(op_sign(vec_Xbuff2,sign_x), op_sign(Ybuff1,1));
//             Cbuff7.mac(op_sign(vec_Xbuff3,sign_x), op_sign(Ybuff1,1));
//             #else
//             Cbuff0.mac(vec_Xbuff0, Ybuff0);
//             Cbuff1.mac(vec_Xbuff1, Ybuff0);
//             Cbuff2.mac(vec_Xbuff2, Ybuff0);
//             Cbuff3.mac(vec_Xbuff3, Ybuff0);
//             Cbuff4.mac(vec_Xbuff0, Ybuff1);
//             Cbuff5.mac(vec_Xbuff1, Ybuff1);
//             Cbuff6.mac(vec_Xbuff2, Ybuff1);
//             Cbuff7.mac(vec_Xbuff3, Ybuff1);
//             #endif
//             if constexpr(batch_size < 4) {
//                 int pdel = (long)p_in;
//                 frac = (pdel & 31) + 33;
//             }
//         }

//         p_in     = INCR_ITR3D_PARAM(p_in, iterator_outer,conv2d_params)
//         p_w      = INCR_ITR3D_PARAM(p_w, iterator_weights,conv2d_params)
//         p_init16 = INCR_ITR3D(p_init16, iterator_psum)

//         // Write out the final outputs
//         if constexpr(out_mode == FULL){
//             if constexpr(act_type==LINEAR || act_type==RELU){
//                  #ifdef USE_RTP_ACT
//                  int out_sign=conv2d_params.out_sign;
//                  #else
//                  int out_sign = 1-act_type; // Note enum for act types are:
//                  0,1,2 -> linear, relu, lrelu #endif store_v(p_out,
//                  Cbuff0.template
//                  to_vector_sign<dtype_ofm>(out_sign,conv2d_params.shift_out));
//                  p_out = byte_incr(p_out, 32); store_v(p_out, Cbuff1.template
//                  to_vector_sign<dtype_ofm>(out_sign,conv2d_params.shift_out));
//                  p_out = byte_incr(p_out, conv2d_params.inc_S_0-chnd);
//                  store_v(p_out, Cbuff2.template
//                  to_vector_sign<dtype_ofm>(out_sign,conv2d_params.shift_out));
//                  p_out = byte_incr(p_out, 32); store_v(p_out, Cbuff3.template
//                  to_vector_sign<dtype_ofm>(out_sign,conv2d_params.shift_out));
//                  p_out = byte_incr(p_out, conv2d_params.inc_S_1-chnd);
//                  store_v(p_out, Cbuff4.template
//                  to_vector_sign<dtype_ofm>(out_sign,conv2d_params.shift_out));
//                  p_out = byte_incr(p_out, 32); store_v(p_out, Cbuff5.template
//                  to_vector_sign<dtype_ofm>(out_sign,conv2d_params.shift_out));
//                  p_out = byte_incr(p_out, conv2d_params.inc_S_0-chnd);
//                  store_v(p_out, Cbuff6.template
//                  to_vector_sign<dtype_ofm>(out_sign,conv2d_params.shift_out));
//                  p_out = byte_incr(p_out, 32); store_v(p_out, Cbuff7.template
//                  to_vector_sign<dtype_ofm>(out_sign,conv2d_params.shift_out));

//             } else {
//                 // Code/Macro for Leaky reLU

//                 LEAKY_RELU_COMPUTE(Cbuff0,conv2d_params.shift_lrelu_input,conv2d_params.lrelu_alpha,conv2d_params.shift_lrelu_alpha,conv2d_params.shift_lrelu_out,p_out)
//                 p_out = byte_incr(p_out, 32);

//                 LEAKY_RELU_COMPUTE(Cbuff1,conv2d_params.shift_lrelu_input,conv2d_params.lrelu_alpha,conv2d_params.shift_lrelu_alpha,conv2d_params.shift_lrelu_out,p_out)
//                 p_out = byte_incr(p_out, conv2d_params.inc_S_0-chnd);

//                 LEAKY_RELU_COMPUTE(Cbuff2,conv2d_params.shift_lrelu_input,conv2d_params.lrelu_alpha,conv2d_params.shift_lrelu_alpha,conv2d_params.shift_lrelu_out,p_out)
//                 p_out = byte_incr(p_out, 32);

//                 LEAKY_RELU_COMPUTE(Cbuff3,conv2d_params.shift_lrelu_input,conv2d_params.lrelu_alpha,conv2d_params.shift_lrelu_alpha,conv2d_params.shift_lrelu_out,p_out)
//                 p_out = byte_incr(p_out, conv2d_params.inc_S_1-chnd);

//                 LEAKY_RELU_COMPUTE(Cbuff4,conv2d_params.shift_lrelu_input,conv2d_params.lrelu_alpha,conv2d_params.shift_lrelu_alpha,conv2d_params.shift_lrelu_out,p_out)
//                 p_out = byte_incr(p_out, 32);

//                 LEAKY_RELU_COMPUTE(Cbuff5,conv2d_params.shift_lrelu_input,conv2d_params.lrelu_alpha,conv2d_params.shift_lrelu_alpha,conv2d_params.shift_lrelu_out,p_out)
//                 p_out = byte_incr(p_out, conv2d_params.inc_S_0-chnd);

//                 LEAKY_RELU_COMPUTE(Cbuff6,conv2d_params.shift_lrelu_input,conv2d_params.lrelu_alpha,conv2d_params.shift_lrelu_alpha,conv2d_params.shift_lrelu_out,p_out)
//                 p_out = byte_incr(p_out, 32);

//                 LEAKY_RELU_COMPUTE(Cbuff7,conv2d_params.shift_lrelu_input,conv2d_params.lrelu_alpha,conv2d_params.shift_lrelu_alpha,conv2d_params.shift_lrelu_out,p_out)

//             }
//         }
//         else {
//         // Store partial sum
//             store_v((int16 OFM_DM_BANK *) p_out, Cbuff0.template
//             to_vector<int16>(conv2d_params.shift_psum_out)); p_out =
//             (dtype_ofm OFM_DM_BANK *)byte_incr((int16*)p_out, 64);
//             store_v((int16 OFM_DM_BANK *) p_out, Cbuff1.template
//             to_vector<int16>(conv2d_params.shift_psum_out)); p_out =
//             (dtype_ofm OFM_DM_BANK *)byte_incr((int16*)p_out,
//             conv2d_params.inc_ST_0-chnd_T); store_v((int16 OFM_DM_BANK *)
//             p_out, Cbuff2.template
//             to_vector<int16>(conv2d_params.shift_psum_out)); p_out =
//             (dtype_ofm OFM_DM_BANK *)byte_incr((int16*)p_out, 64);
//             store_v((int16 OFM_DM_BANK *) p_out, Cbuff3.template
//             to_vector<int16>(conv2d_params.shift_psum_out)); p_out =
//             (dtype_ofm OFM_DM_BANK *)byte_incr((int16*)p_out,
//             conv2d_params.inc_ST_1-chnd_T); store_v((int16 OFM_DM_BANK *)
//             p_out, Cbuff4.template
//             to_vector<int16>(conv2d_params.shift_psum_out)); p_out =
//             (dtype_ofm OFM_DM_BANK *)byte_incr((int16*)p_out, 64);
//             store_v((int16 OFM_DM_BANK *) p_out, Cbuff5.template
//             to_vector<int16>(conv2d_params.shift_psum_out)); p_out =
//             (dtype_ofm OFM_DM_BANK *)byte_incr((int16*)p_out,
//             conv2d_params.inc_ST_0-chnd_T); store_v((int16 OFM_DM_BANK *)
//             p_out, Cbuff6.template
//             to_vector<int16>(conv2d_params.shift_psum_out)); p_out =
//             (dtype_ofm OFM_DM_BANK *)byte_incr((int16*)p_out, 64);
//             store_v((int16 OFM_DM_BANK *) p_out, Cbuff7.template
//             to_vector<int16>(conv2d_params.shift_psum_out));
//         }
//         p_out = INCR_ITR2D(p_out,iterator_pout)
//     }

//     if constexpr(out_mode == FULL){
//          #if RELU_INT8_ENABLE == 1
//             #ifdef USE_RTP_ACT
//             	relu_int8_post_process(output,conv2d_params.out_sign,conv2d_params.ofm_len);
//             #else
//                 int act = 1 - act_type;
//                 relu_int8_post_process(output,act,conv2d_params.ofm_len);
//             #endif
//          #endif
//     }
// }