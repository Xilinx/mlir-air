//===- main.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

#include "air_tensor.h"

extern "C" {
void _mlir_ciface_forward(void *, void *, void *);
}

#define M_SIZE 16

template <typename T> void _c_reference(tensor_t<T, 2> *a, tensor_t<T, 2> *b) {
  for (size_t i = 0; i < 8; i++) {
    for (size_t j = 0; j < 8; j++) {
      size_t src_idx = i * 16 + j;
      size_t dst_idx = (i + 8) * 16 + (j + 8);
      b->data[dst_idx] = a->data[src_idx];
    }
  }
}

int main(int argc, char *argv[]) {

  tensor_t<int32_t, 2> input;
  tensor_t<int32_t, 2> output0;
  tensor_t<int32_t, 2> output1;
  tensor_t<int32_t, 2> golden;

  input.shape[0] = input.shape[1] = M_SIZE;
  input.alloc = input.data =
      (int32_t *)malloc(sizeof(int32_t) * input.shape[0] * input.shape[1]);

  output0.shape[0] = output0.shape[1] = M_SIZE;
  output0.alloc = output0.data =
      (int32_t *)malloc(sizeof(int32_t) * output0.shape[0] * output0.shape[1]);

  output1.shape[0] = output1.shape[1] = M_SIZE;
  output1.alloc = output1.data =
      (int32_t *)malloc(sizeof(int32_t) * output1.shape[0] * output1.shape[1]);

  golden.shape[0] = golden.shape[1] = M_SIZE;
  golden.alloc = golden.data =
      (int32_t *)malloc(sizeof(int32_t) * golden.shape[0] * golden.shape[1]);

  for (unsigned int i = 0; i < input.shape[0] * input.shape[1]; i++) {
    input.data[i] = ((int32_t)i) % 1024;
    output0.data[i] = 0;
    output1.data[i] = 0;
    golden.data[i] = 0;
  }

  _mlir_ciface_forward((void *)&input, (void *)&output0, (void *)&output1);
  _c_reference<int32_t>(&input, &golden);

  int errors = 0;
  auto output_size = output0.shape[0] * output0.shape[1];
  for (unsigned int i = 0; i < output_size; i++) {
    auto d0 = output0.data[i];
    auto d1 = output1.data[i];
    auto ref = golden.data[i];
    if (d0 != ref || d1 != ref) {
      errors++;
      if (errors < 10)
        printf("%04X: mismatch %d != %d || %d != %d\n", i, d0, ref, d1, ref);
    }
  }
  if (!errors) {
    printf("PASS!\n");
  } else {
    printf("fail %ld/%ld.\n", (output_size - errors), output_size);
  }

  free(input.alloc);
  free(output0.alloc);
  free(output1.alloc);
  free(golden.alloc);

  return 0;
}
