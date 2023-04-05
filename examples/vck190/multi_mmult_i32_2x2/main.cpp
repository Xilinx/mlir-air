//===- herd.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air_host.h"
#include "test_library.h"

#include <cassert>
#include <cstdio>
#include <dlfcn.h>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>

#define VERBOSE 1
#define PROFILE 1

namespace {

template <typename T>
void mm_out(tensor_t<T, 2> *a, tensor_t<T, 2> *b, tensor_t<T, 2> *r) {
  size_t a_h = a->shape[0];
  size_t a_w = a->shape[1];
  size_t b_h = b->shape[0];
  size_t b_w = b->shape[1];
  assert(a_w == b_h);

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

} // namespace

namespace air::segments::segment_1 {
int32_t mlir_aie_read_buffer_buf48(aie_libxaie_ctx_t *ctx, int index);
int32_t mlir_aie_read_buffer_buf49(aie_libxaie_ctx_t *ctx, int index);
int32_t mlir_aie_read_buffer_buf50(aie_libxaie_ctx_t *ctx, int index);
} // namespace air::segments::segment_1

int main(int argc, char *argv[]) {
  auto init_ret = air_init();
  assert(init_ret == HSA_STATUS_SUCCESS);

  std::vector<air_agent_t> agents;
  auto get_agents_ret = air_iterate_agents(
      [](air_agent_t a, void *d) {
        auto *v = static_cast<std::vector<air_agent_t> *>(d);
        v->push_back(a);
        return HSA_STATUS_SUCCESS;
      },
      (void *)&agents);
  assert(get_agents_ret == HSA_STATUS_SUCCESS && agents.size() &&
         "failed to get agents!");

  queue_t *q = nullptr;
  auto create_queue_ret =
      air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q,
                       agents[0].handle, 0 /* device_id (optional) */);
  assert(q && create_queue_ret == 0 && "failed to create queue!");

  tensor_t<uint32_t, 2> input_A;
  tensor_t<uint32_t, 2> input_B;
  tensor_t<uint32_t, 2> input_C;
  tensor_t<uint32_t, 2> output;
  tensor_t<uint32_t, 2> output_ref0;
  tensor_t<uint32_t, 2> output_ref1;

// All matrices are size (M_SIZE, M_SIZE)
#define M_SIZE 128

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

  auto handle = air_module_load_from_file(nullptr, q);
  assert(handle && "failed to open air module");

  auto herd_fn = (void (*)(void *, void *, void *, void *))dlsym(
      (void *)handle, "_mlir_ciface_forward");
  assert(herd_fn && "failed to locate _mlir_ciface_forward in .so");

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
  struct timeval before, after;
  long diff_s, diff_us;
  gettimeofday(&before, NULL);

  // run it
  herd_fn(a, b, c, o);

  gettimeofday(&after, NULL);
  diff_s = after.tv_sec - before.tv_sec;
  diff_us = after.tv_usec - before.tv_usec;

  if (diff_s)
    diff_us += 10000000;

  if (PROFILE) {
    printf("before %ld.%06ld\n", before.tv_sec, before.tv_usec);
    printf("after  %ld.%06ld\n", after.tv_sec, after.tv_usec);
    printf("diff   %ld.%06ld\n", diff_s, diff_us);
  }

  if (VERBOSE) {
    uint64_t row = 2;
    uint64_t col = 11;
    aie_libxaie_ctx_t *xaie = (aie_libxaie_ctx_t *)air_get_libxaie_ctx();
    mlir_aie_print_tile_status(xaie, col, row);
    for (int i = 0; i < 16; i++) {
      printf("%d\n",
             air::segments::segment_1::mlir_aie_read_buffer_buf48(xaie, i));
      printf("%d\n",
             air::segments::segment_1::mlir_aie_read_buffer_buf49(xaie, i));
      printf("%d\n",
             air::segments::segment_1::mlir_aie_read_buffer_buf50(xaie, i));
    }
  }

  int errors = 0;
  auto output_size = output.shape[0] * output.shape[1];
  for (int i = 0; i < output_size; i++) {
    auto d = output.data[i];
    auto ref = output_ref1.data[i];
    if (d != ref) {
      errors++;
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
}
