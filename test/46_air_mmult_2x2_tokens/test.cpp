//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdio>
#include <dlfcn.h>
#include <fcntl.h>
#include <iostream>
#include <stdlib.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "air.hpp"
#include "test_library.h"

#define VERBOSE 0
#define PROFILE 0

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

int main(int argc, char *argv[]) {
  uint64_t col = 5;
  uint64_t row = 3;

  std::vector<air_agent_t> agents;
  auto get_agents_ret = air_get_agents(agents);
  assert(get_agents_ret == HSA_STATUS_SUCCESS && "failed to get agents!");

  if (agents.empty()) {
    std::cout << "fail." << std::endl;
    return -1;
  }

  if (VERBOSE)
    std::cout << "Found " << agents.size() << " agents" << std::endl;

  std::vector<queue_t *> queues;
  for (auto agent : agents) {
    // create the queue
    queue_t *q = nullptr;
    auto create_queue_ret = air_queue_create(
        MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, agent.handle);
    assert(create_queue_ret == 0 && "failed to create queue!");
    queues.push_back(q);
  }

  aie_libxaie_ctx_t *xaie = (aie_libxaie_ctx_t *)air_init_libxaie();

  queue_t *q = queues[0];

  if (VERBOSE)
    mlir_aie_print_tile_status(xaie, col, row);

  tensor_t<uint32_t, 2> input_A;
  tensor_t<uint32_t, 2> input_B;
  tensor_t<uint32_t, 2> output;
  tensor_t<uint32_t, 2> output_ref0;

#define M_SIZE 64

  input_A.shape[0] = input_A.shape[1] = M_SIZE;
  input_A.alloc = input_A.data = (uint32_t *)malloc(
      sizeof(uint32_t) * input_A.shape[0] * input_A.shape[1]);

  input_B.shape[0] = input_B.shape[1] = M_SIZE;
  input_B.alloc = input_B.data = (uint32_t *)malloc(
      sizeof(uint32_t) * input_B.shape[0] * input_B.shape[1]);

  output.shape[0] = output.shape[1] = M_SIZE;
  output.alloc = output.data =
      (uint32_t *)malloc(sizeof(uint32_t) * output.shape[0] * output.shape[1]);

  output_ref0.shape[0] = output_ref0.shape[1] = M_SIZE;
  output_ref0.alloc = output_ref0.data = (uint32_t *)malloc(
      sizeof(uint32_t) * output_ref0.shape[0] * output_ref0.shape[1]);

  auto handle = air_module_load_from_file(nullptr, q);
  assert(handle && "failed to open linked air module");

  auto herd_fn = (void (*)(void *, void *, void *))dlsym(
      (void *)handle, "_mlir_ciface_forward");
  assert(herd_fn && "failed to locate _mlir_ciface_forward in .so");

  for (int i = 0; i < input_A.shape[0] * input_A.shape[1]; i++) {
    input_A.data[i] = (rand() % 1024) + 1;
    input_B.data[i] = (rand() % 1024) + 1;
    output.data[i] = 0;
    output_ref0.data[i] = 0;
  }

  mm_out(&input_A, &input_B, &output_ref0);

  void *a, *b, *o;
  a = &input_A;
  b = &input_B;
  o = &output;

  // run it
  herd_fn(a, b, o);

  int errors = 0;
  auto output_size = output.shape[0] * output.shape[1];
  for (int i = 0; i < output_size; i++) {
    auto d = output.data[i];
    auto ref = output_ref0.data[i];
    if (d != ref) {
      errors++;
      if (errors < 100)
        printf("%04X: mismatch %d != %d\n", i, d, ref);
    }
  }

  free(input_A.alloc);
  free(input_B.alloc);
  free(output.alloc);
  free(output_ref0.alloc);

  if (!errors) {
    printf("PASS!\n");
  } else {
    printf("fail %ld/%ld.\n", (output_size - errors), output_size);
    return -1;
  }
  return 0;
}
