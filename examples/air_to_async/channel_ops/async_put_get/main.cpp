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
void _mlir_ciface_forward(void *);
}

template <typename T>
void ref(tensor_t<T, 3> *r) {
  for (int i = 0; i < 32; i++) {
    for (int j = 0; j < 32; j++) {
      for (int k = 0; k < 32; k++) {
        r->data[i * 32 * 32 + j * 32 + k] = i;
      }
    }
  }
  
}

#define INPUT_SIZE 32

int main(int argc, char *argv[]) {

  tensor_t<uint32_t, 3> output;
  tensor_t<uint32_t, 3> output_ref;

// All matrices are size (M_SIZE, M_SIZE)
#define M_SIZE 32

  output.shape[0] = output.shape[1] = output.shape[2] = M_SIZE;
  output.alloc = output.data =
      (uint32_t *)malloc(sizeof(uint32_t) * output.shape[0] * output.shape[1] * output.shape[2]);

  output_ref.shape[0] = output_ref.shape[1] = output_ref.shape[2] = M_SIZE;
  output_ref.alloc = output_ref.data =
      (uint32_t *)malloc(sizeof(uint32_t) * output_ref.shape[0] * output_ref.shape[1] * output_ref.shape[2]);

  for (int i = 0; i < output.shape[0] * output.shape[1] * output.shape[2]; i++) {
    output.data[i] = 0;
    output_ref.data[i] = 0;
  }

  ref(&output_ref);

  void *o = &output;

  _mlir_ciface_forward((void *)&output);

  int errors = 0;
  auto output_size = output.shape[0] * output.shape[1] * output.shape[2];
  for (int i = 0; i < output_size; i++) {
    auto d = output.data[i];
    auto ref = output_ref.data[i];
    if (d != ref) {
      errors++;
      if (errors < 10)
        printf("%04X: mismatch %d != %d\n", i, d, ref);
    }
  }
  if (!errors) {
    printf("PASS!\n");
  } else {
    printf("fail %ld/%ld.\n", (output_size - errors), output_size);
  }

  free(output.alloc);
  free(output_ref.alloc);

  return errors;
}
