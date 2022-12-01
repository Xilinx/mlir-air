//===- main.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "air_tensor.h"

extern "C" {
void _mlir_ciface_forward(void *, void *, void *);
void dump_graph(char *);
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

#define INPUT_SIZE 1024

int main(int argc, char *argv[]) {
  tensor_t<int32_t, 2> input0;
  tensor_t<int32_t, 2> input1;
  tensor_t<int32_t, 2> output;
  tensor_t<int32_t, 2> output_ref;

  input0.shape[0] = input0.shape[1] = INPUT_SIZE;
  input0.stride[1] = 1;
  input0.stride[0] = 1024;
  input0.alloc = input0.data =
      (int32_t *)malloc(sizeof(int32_t) * input0.shape[0] * input0.shape[1]);

  input1.shape[0] = input1.shape[1] = INPUT_SIZE;
  input1.stride[1] = 1;
  input1.stride[0] = 1024;
  input1.alloc = input1.data =
      (int32_t *)malloc(sizeof(int32_t) * input1.shape[0] * input1.shape[1]);

  output.shape[0] = output.shape[1] = INPUT_SIZE;
  output.stride[1] = 1;
  output.stride[0] = 1024;
  output.alloc = output.data =
      (int32_t *)malloc(sizeof(int32_t) * output.shape[0] * output.shape[1]);

  output_ref.shape[0] = output_ref.shape[1] = INPUT_SIZE;
  output_ref.alloc = output_ref.data = (int32_t *)malloc(
      sizeof(int32_t) * output_ref.shape[0] * output_ref.shape[1]);

  for (int i = 0; i < input0.shape[0] * input0.shape[1]; i++) {
    input0.data[i] = ((int32_t)i % 3) + 1;
    input1.data[i] = ((int32_t)i + 1) % 4 + 1;
    output.data[i] = -1;
    output_ref.data[i] = -1;
  }
  mm_out(&input0, &input1, &output_ref);

  _mlir_ciface_forward((void *)&input0, (void *)&input1, (void *)&output);

  int errors = 0;
  auto output_size = output.shape[0] * output.shape[1];
  for (int i = 0; i < output_size; i++) {
    auto d = output.data[i];
    auto ref = output_ref.data[i];
    if (d != ref) {
      errors++;
      // if (errors < 10)
      //   printf("%04X: mismatch %d != %d (output != ref)\n", i, d, ref);
    }
  }
  // errors are expected because the mlir only computes part of the output
  //if (!errors) {
  if ((output_size - errors) == 8192) {
    printf("PASS!\n");
  } else {
    printf("fail %ld/%ld.\n", (output_size - errors), output_size);
  }

  char dotfile[] = "out.dot";
  dump_graph(dotfile);

  return 0;
}