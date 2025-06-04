//===- vm.cc ----------------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc.
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

template <typename T_in, typename T_out, unsigned k, unsigned n, unsigned s,
          unsigned t>
void vecmat_vectorized(T_in *__restrict a, T_in *__restrict b,
                       T_out *__restrict c) {
  static_assert(n % t == 0 && k % 2 == 0);
  static_assert(s == 8); // s is fixed to 8 because that is the number of
                         // column vectors (a_vec_0_0..a_vec_3_1) we create
  static_assert(k % s == 0);
  static_assert(std::is_same<T_in, bfloat16>::value ||
                std::is_same<T_in, int16_t>::value);

  event0();
  T_in *__restrict a_ptr = a;
  T_in *__restrict b_ptr = b;

  T_out *__restrict c_ptr = c; // reset to the first row of C output on
  // each outer loop tieration
  for (int col = 0; col < n; col += t) {

    const T_in *__restrict a_ptr1 = a_ptr;
    const T_in *__restrict b_ptr1 = b_ptr;
    for (int row = 0; row < k; row += 8)
      chess_loop_range(k / 8, ) {
        aie::vector<T_in, 8> a_vec = aie::load_v<8>(a_ptr1);
        a_ptr1 += 8;
        aie::accum<accfloat, t> c_acc_in;
        c_acc_in.from_vector(aie::load_v<t>(c_ptr));

        for (int i = 0; i < 8; i++) {
          const aie::vector<T_in, t> b_vec = aie::load_v<t>(b_ptr1);
          b_ptr1 += n;
          const aie::vector<T_in, t> s0 = aie::broadcast<T_in, t>(a_vec[i]);
          c_acc_in = mac(c_acc_in, s0, b_vec);
        }
        aie::store_v(c_ptr, c_acc_in.template to_vector<T_out>());
      }
    b_ptr += t;
    c_ptr += t;
  }
  event1();
}

extern "C" {

#ifndef DIM_N
#define DIM_N 48
#endif

#ifndef DIM_K
#define DIM_K 96
#endif

#define combos(X) X(bfloat16, bf16, bfloat16, bf16)

#define vecmat_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,            \
                                 mlir_type_out)                                \
  void vecmat_##mlir_type_in##_##mlir_type_out(ctype_in *a_in, ctype_in *b_in, \
                                               ctype_out *c_out) {             \
    vecmat_vectorized<ctype_in, ctype_out, DIM_K, DIM_N, 8, 16>(a_in, b_in,    \
                                                                c_out);        \
  }

#define zero_vectorized_c_func(ctype_in, mlir_type_in, ctype_out,              \
                               mlir_type_out)                                  \
  void linalg_fill_##mlir_type_out(ctype_out *c_out) {                         \
    zero_vectorized<ctype_out, 1, DIM_N, 32>(c_out);                           \
  }

combos(vecmat_vectorized_c_func) combos(zero_vectorized_c_func)

} // extern "C"
