// (c) Copyright 2022 Xilinx Inc. All Rights Reserved.

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
      r->d[idx] = (T)(0);
      for (size_t k = 0, ke = a_w; k < a_w; k++) {
        T _a = a->d[i * a_w + k];
        T _b = b->d[k * b_w + j];
        r->d[idx] += _a * _b;
      }
    }
  }
}

#define INPUT_SIZE 1024

//int test_cmd_proc();

int main(int argc, char *argv[]) {
  // test_cmd_proc();
  // return 0;
  tensor_t<int32_t, 2> input0;
  tensor_t<int32_t, 2> input1;
  tensor_t<int32_t, 2> output;
  tensor_t<int32_t, 2> output_ref;

  input0.shape[0] = input0.shape[1] = INPUT_SIZE;
  input0.stride[1] = 1;
  input0.stride[0] = 1024;
  input0.d = input0.aligned =
      (int32_t *)malloc(sizeof(int32_t) * input0.shape[0] * input0.shape[1]);

  input1.shape[0] = input1.shape[1] = INPUT_SIZE;
  input1.stride[1] = 1;
  input1.stride[0] = 1024;
  input1.d = input1.aligned =
      (int32_t *)malloc(sizeof(int32_t) * input1.shape[0] * input1.shape[1]);

  output.shape[0] = output.shape[1] = INPUT_SIZE;
  output.stride[1] = 1;
  output.stride[0] = 1024;
  output.d = output.aligned =
      (int32_t *)malloc(sizeof(int32_t) * output.shape[0] * output.shape[1]);

  output_ref.shape[0] = output_ref.shape[1] = INPUT_SIZE;
  output_ref.d = output_ref.aligned = (int32_t *)malloc(
      sizeof(int32_t) * output_ref.shape[0] * output_ref.shape[1]);

  for (int i = 0; i < input0.shape[0] * input0.shape[1]; i++) {
    input0.d[i] = ((int32_t)i % 3) + 1;
    input1.d[i] = ((int32_t)i + 1) % 4 + 1;
    output.d[i] = -1;
    output_ref.d[i] = -1;
  }
  mm_out(&input0, &input1, &output_ref);

  _mlir_ciface_forward((void *)&input0, (void *)&input1, (void *)&output);

  int errors = 0;
  auto output_size = output.shape[0] * output.shape[1];
  for (int i = 0; i < output_size; i++) {
    auto d = output.d[i];
    auto ref = output_ref.d[i];
    if (d != ref) {
      errors++;
      if (errors < 10)
        printf("%04X: mismatch %d != %d (output != ref)\n", i, d, ref);
    }
  }
  if (!errors) {
    printf("PASS!\n");
  } else {
    printf("fail %ld/%ld.\n", (output_size - errors), output_size);
  }

  dump_graph("out.dot");

  return 0;
}