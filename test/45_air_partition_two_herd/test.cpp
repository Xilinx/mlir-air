//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
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

#define DATA_SIZE 10240

int main(int argc, char *argv[]) {

  hsa_status_t init_status = air_init();

  if (init_status != HSA_STATUS_SUCCESS) {
    std::cout << "air_init() failed. Exiting" << std::endl;
    return -1;
  }

  std::vector<air_agent_t> agents;
  auto get_agents_ret = air_get_agents(agents);
  assert(get_agents_ret == HSA_STATUS_SUCCESS && "failed to get agents!");

  if (agents.empty()) {
    std::cout << "fail." << std::endl;
    return -1;
  }

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

  aie_libxaie_ctx_t *xaie = (aie_libxaie_ctx_t *)air_get_libxaie_ctx();

  queue_t *q = queues[0];

  tensor_t<uint32_t, 1> input_A;
  tensor_t<uint32_t, 1> input_B;
  tensor_t<uint32_t, 1> input_C;
  tensor_t<uint32_t, 1> output;
  tensor_t<uint32_t, 1> output_ref;

  input_A.shape[0] = DATA_SIZE;
  input_A.alloc = input_A.data =
      (uint32_t *)malloc(sizeof(uint32_t) * input_A.shape[0]);

  input_B.shape[0] = DATA_SIZE;
  input_B.alloc = input_B.data =
      (uint32_t *)malloc(sizeof(uint32_t) * input_B.shape[0]);

  input_C.shape[0] = DATA_SIZE;
  input_C.alloc = input_C.data =
      (uint32_t *)malloc(sizeof(uint32_t) * input_C.shape[0]);

  output.shape[0] = DATA_SIZE;
  output.alloc = output.data =
      (uint32_t *)malloc(sizeof(uint32_t) * output.shape[0]);

  output_ref.shape[0] = DATA_SIZE;
  output_ref.alloc = output_ref.data =
      (uint32_t *)malloc(sizeof(uint32_t) * output_ref.shape[0]);

  auto handle = air_module_load_from_file(nullptr, q);
  assert(handle && "failed to open linked air module");

  auto herd_fn = (void (*)(void *, void *, void *, void *))dlsym(
      (void *)handle, "_mlir_ciface_forward");
  assert(herd_fn && "failed to locate _mlir_ciface_forward in .so");

  for (int i = 0; i < input_A.shape[0]; i++) {
    input_A.data[i] = (rand() % 1024) + 1;
    input_B.data[i] = (rand() % 1024) + 1;
    input_C.data[i] = (rand() % 1024) + 1;
    output.data[i] = 0xdeadcafe;
    output_ref.data[i] = input_A.data[i] * (input_B.data[i] + input_C.data[i]);
  }

  void *a, *b, *c, *o;
  a = &input_A;
  b = &input_B;
  c = &input_C;
  o = &output;

  // run it
  herd_fn(a, b, c, o);

  int errors = 0;
  auto output_size = output.shape[0];
  for (int i = 0; i < output_size; i++) {
    auto d = output.data[i];
    auto ref = output_ref.data[i];
    if (d != ref) {
      errors++;
      if (errors < 100)
        printf("%04X: mismatch %d != %d\n", i, d, ref);
    }
  }

  free(input_A.alloc);
  free(input_B.alloc);
  free(input_C.alloc);
  free(output.alloc);
  free(output_ref.alloc);

  if (!errors) {
    printf("PASS!\n");
  } else {
    printf("fail %ld/%ld.\n", (output_size - errors), output_size);
    return -1;
  }
  return 0;
}
