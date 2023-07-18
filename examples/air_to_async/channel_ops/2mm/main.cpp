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
void _mlir_ciface_forward(void *, void *, void *, void *);
}

template <typename T>
void mm_out(tensor_t<T, 2> *a, tensor_t<T, 2> *b, tensor_t<T, 2> *r) {
  size_t a_h = a->shape[0];
  size_t a_w = a->shape[1];
  size_t b_h = b->shape[0];
  size_t b_w = b->shape[1];

  for (size_t i = 0; i < a_h; i++) {
    for (size_t j = 0; j < b_w; j++) {
      size_t idx = i * b_w + j;
      r->data[idx] = (T)(0);
      for (size_t k = 0, ke = a_w; k < a_w; k++) {
        T _a = a->data[i * a_w + k];
        T _b = b->data[k * b_w + j];
        r->data[idx] += _a * _b;
      }
    }
  }
}

#define INPUT_SIZE 32

int main(int argc, char *argv[]) {

  tensor_t<uint32_t, 2> input_A;
  tensor_t<uint32_t, 2> input_B;
  tensor_t<uint32_t, 2> input_C;
  tensor_t<uint32_t, 2> output;
  tensor_t<uint32_t, 2> output_ref0;
  tensor_t<uint32_t, 2> output_ref1;

// All matrices are size (M_SIZE, M_SIZE)
#define M_SIZE 32

  input_A.shape[0] = input_A.shape[1] = M_SIZE;
  input_A.alloc = input_A.data = (uint32_t *)malloc(
      sizeof(uint32_t) * input_A.shape[0] * input_A.shape[1]);

  input_B.shape[0] = input_B.shape[1] = M_SIZE;
  input_B.alloc = input_B.data = (uint32_t *)malloc(
      sizeof(uint32_t) * input_B.shape[0] * input_B.shape[1]);

  input_C.shape[0] = input_C.shape[1] = M_SIZE;
  input_C.alloc = input_C.data = (uint32_t *)malloc(
      sizeof(uint32_t) * input_C.shape[0] * input_C.shape[1]);

  output.shape[0] = output.shape[1] = M_SIZE;
  output.alloc = output.data =
      (uint32_t *)malloc(sizeof(uint32_t) * output.shape[0] * output.shape[1]);

  output_ref0.shape[0] = output_ref0.shape[1] = M_SIZE;
  output_ref0.alloc = output_ref0.data = (uint32_t *)malloc(
      sizeof(uint32_t) * output_ref0.shape[0] * output_ref0.shape[1]);

  output_ref1.shape[0] = output_ref1.shape[1] = M_SIZE;
  output_ref1.alloc = output_ref1.data = (uint32_t *)malloc(
      sizeof(uint32_t) * output_ref1.shape[0] * output_ref1.shape[1]);

  for (int i = 0; i < input_A.shape[0] * input_A.shape[1]; i++) {
    input_A.data[i] = ((uint32_t)i) % 3;
    input_B.data[i] = ((uint32_t)i + 1) % 5;
    input_C.data[i] = ((uint32_t)i + 2) % 7;
    output.data[i] = 0;
    output_ref0.data[i] = 0;
    output_ref1.data[i] = 0;
  }

  mm_out(&input_A, &input_B, &output_ref0);
  mm_out(&output_ref0, &input_C, &output_ref1);

  void *a, *b, *c, *o;
  a = &input_A;
  b = &input_B;
  c = &input_C;
  o = &output;

  _mlir_ciface_forward((void *)&input_A, (void *)&input_B, (void *)&input_C,
                       (void *)&output);

  int errors = 0;
  auto output_size = output.shape[0] * output.shape[1];
  for (int i = 0; i < output_size; i++) {
    auto d = output.data[i];
    auto ref = output_ref1.data[i];
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

  free(input_A.alloc);
  free(input_B.alloc);
  free(input_C.alloc);
  free(output.alloc);
  free(output_ref0.alloc);
  free(output_ref1.alloc);

  return errors;
}
