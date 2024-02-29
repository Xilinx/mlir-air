//===----------------------------------------------------------------------===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#define __AIENGINE__ 2
#define NOCPP
#define __AIEARCH__ 20

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include <aie_api/aie.hpp>

#define N (128*128)

extern "C" {
void add_bf16(bfloat16 *a, bfloat16 *b, bfloat16 *c) {
  for (int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
}
}