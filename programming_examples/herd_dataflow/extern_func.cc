//===- extern_func.cc -------------------------------------------*- C++ -*-===//
//
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc.
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

extern "C" {

void add_3_bf16(bfloat16 *a, bfloat16 *c) {
  // c := a + 3
  bfloat16 *__restrict pa = a;
  bfloat16 *__restrict pc = c;
  aie::vector<bfloat16, 16> const_bcast =
      aie::broadcast<bfloat16, 16>((bfloat16)3.0);
  for (int i = 0; i < 64 * 64; i += 16) {
    aie::vector<bfloat16, 16> aTemp = aie::load_v<16>(pa);
    aie::vector<bfloat16, 16> cTemp = aie::add(aTemp, const_bcast);
    aie::store_v(pc, cTemp);
    pa += 16;
    pc += 16;
  }
}

} // extern "C"
