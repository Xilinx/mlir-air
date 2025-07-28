//===- cascade_mm.cc --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include <aie_api/aie.hpp>

template <typename T_in, int num_elem>
void cascade_put_impl(const T_in *__restrict a) {
  event0();
  constexpr int vec_size = 16;
  for (int i = 0; i < num_elem; i += vec_size) {
    aie::vector<T_in, vec_size> v16 = aie::load_v<vec_size>(a + i);
    put_mcd(v16);
  }
  event1();
}

template <typename T_in, int num_elem>
void cascade_get_impl(T_in *__restrict a) {
  event0();
  constexpr int vec_size = 16;
  for (int i = 0; i < num_elem; i += vec_size) {
    aie::vector<T_in, vec_size> v16 = get_scd_v16int32();
    aie::store_v(a + i, v16);
  }
  event1();
}

extern "C" {

void cascade_put(const int32_t *__restrict a) {
  cascade_put_impl<int32_t, 2048>(a);
}

void cascade_get(int32_t *__restrict a) { cascade_get_impl<int32_t, 2048>(a); }

} // extern "C"
