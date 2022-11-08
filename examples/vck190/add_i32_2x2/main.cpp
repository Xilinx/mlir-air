//===- main.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdio>
#include <dlfcn.h>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>

#include "air_host.h"

#define PROFILE 0

namespace {

template <typename T>
void add_out(tensor_t<T, 3> *a, tensor_t<T, 3> *b, tensor_t<T, 3> *r) {
  size_t a_h = a->shape[0];
  size_t a_w = a->shape[1];
  size_t a_x = a->shape[2];
  size_t b_h = b->shape[0];
  size_t b_w = b->shape[1];
  assert(a_h == b_h);
  assert(a_w == b_w);

  for (size_t i = 0; i < a_h * a_w * a_x; i++) {
    T _a = a->data[i];
    T _b = b->data[i];
    r->data[i] = _a + _b;
  }
}

} // namespace

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

  tensor_t<uint32_t, 3> input_A;
  tensor_t<uint32_t, 3> input_B;
  tensor_t<uint32_t, 3> output;
  tensor_t<uint32_t, 3> output_ref;

  input_A.shape[0] = 64;
  input_A.shape[1] = 64;
  input_A.shape[2] = 32;
  input_A.alloc = input_A.data =
      (uint32_t *)malloc(sizeof(uint32_t) * input_A.shape[0] *
                         input_A.shape[1] * input_A.shape[2]);

  input_B.shape[0] = 64;
  input_B.shape[1] = 64;
  input_B.shape[2] = 32;
  input_B.alloc = input_B.data =
      (uint32_t *)malloc(sizeof(uint32_t) * input_B.shape[0] *
                         input_B.shape[1] * input_B.shape[2]);

  output.shape[0] = 64;
  output.shape[1] = 64;
  output.shape[2] = 32;
  output.alloc = output.data = (uint32_t *)(size_t)malloc(
      sizeof(uint32_t) * output.shape[0] * output.shape[1] * output.shape[2]);

  output_ref.shape[0] = 64;
  output_ref.shape[1] = 64;
  output_ref.shape[2] = 32;
  output_ref.alloc = output_ref.data =
      (uint32_t *)malloc(sizeof(uint32_t) * output_ref.shape[0] *
                         output_ref.shape[1] * output_ref.shape[2]);

  auto handle = air_module_load_from_file(nullptr, q);
  assert(handle && "failed to open linked air module");

  auto herd_fn = (void (*)(void *, void *, void *))dlsym(
      (void *)handle, "_mlir_ciface_forward");
  assert(herd_fn && "failed to locate _mlir_ciface_forward in .so");

  for (int i = 0; i < input_A.shape[0] * input_A.shape[1] * input_A.shape[2];
       i++) {
    input_A.data[i] = (rand() % 1024) + 1;
    input_B.data[i] = (rand() % 1024) + 1;
    output.data[i] = 0;
    output_ref.data[i] = 0;
  }

  add_out(&input_A, &input_B, &output_ref);

  void *a, *b, *o;
  a = &input_A;
  b = &input_B;
  o = &output;
  struct timeval before, after;
  long diff_s, diff_us;
  gettimeofday(&before, NULL);

  // run it
  herd_fn(a, b, o);

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

  int errors = 0;
  auto output_size = output.shape[0] * output.shape[1];
  for (int i = 0; i < output_size; i++) {
    auto d = output.data[i];
    auto ref = output_ref.data[i];
    if (d != ref) {
      errors++;
      if (errors < 100)
        printf("%04X: mismatch %d != %d\n", i, d, ref);
    }
  }
  if (!errors) {
    printf("PASS!\n");
  } else {
    printf("fail %ld/%ld.\n", (output_size - errors), output_size);
  }

  air_shut_down();
}
