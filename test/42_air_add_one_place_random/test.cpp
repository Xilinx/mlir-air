//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
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

namespace air::partitions::partition_0 {
int32_t mlir_aie_read_buffer_scratch_0_0(aie_libxaie_ctx_t*, int);
int32_t mlir_aie_read_buffer_scratch_copy_0_0(aie_libxaie_ctx_t*, int);
void mlir_aie_write_buffer_scratch_0_0(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_copy_0_0(aie_libxaie_ctx_t*, int, int32_t);
}; // namespace air::partitions::partition_0
using namespace air::partitions::partition_0;

#define DMA_COUNT 16

int
main(int argc, char *argv[])
{
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

  aie_libxaie_ctx_t *xaie = (aie_libxaie_ctx_t *)air_init_libxaie();

  queue_t *q = queues[0];

  for (int i=0; i<DMA_COUNT; i++) {
    mlir_aie_write_buffer_scratch_0_0(xaie, i, 0xfadefade);
    mlir_aie_write_buffer_scratch_copy_0_0(xaie, i, 0xfadefade);
  }

  printf("loading air module\n");
  auto handle = air_module_load_from_file(nullptr,q);
  assert(handle && "failed to open air module");

  auto graph_fn = (void (*)(void*,void *))dlsym((void*)handle, "_mlir_ciface_graph");
  assert(graph_fn && "failed to locate _mlir_ciface_graph in .so");

  tensor_t<uint32_t,1> input;
  tensor_t<uint32_t,1> output;

  input.shape[0] = DMA_COUNT;
  input.alloc = input.data =
      (uint32_t *)malloc(sizeof(uint32_t) * input.shape[0]);

  output.shape[0] = DMA_COUNT;
  output.alloc = output.data =
      (uint32_t *)malloc(sizeof(uint32_t) * output.shape[0]);

  for (int i=0; i<DMA_COUNT; i++) {
    input.data[i] = i + 0x1;
    output.data[i] = 0x00defaced;
  }

  void *i, *o;
  i = &input;
  o = &output;
  graph_fn(i, o);

  int errors = 0;
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d0 = mlir_aie_read_buffer_scratch_0_0(xaie, i);
    uint32_t d1 = mlir_aie_read_buffer_scratch_copy_0_0(xaie, i);
    if (d0+1 != d1) {
      printf("mismatch tile %x != %x\n", d0, d1);
      errors++;
    }
  }

  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = output.data[i];
    if (d != (i+2)) {
      errors++;
      printf("mismatch %x != 2 + %x\n", d, i);
    }
  }

  free(input.alloc);
  free(output.alloc);

  if (!errors) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail %d/%d.\n", errors, 2*DMA_COUNT);
    return -1;
  }

}
