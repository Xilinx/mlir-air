//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
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

#include "air_host.h"
#include "air_tensor.h"

namespace air::partitions::partition_0 {
void mlir_aie_write_buffer_buf0(aie_libxaie_ctx_t*, int, int32_t);
uint32_t mlir_aie_read_buffer_buf0(aie_libxaie_ctx_t*, int);
}; // namespace air::partitions::partition_0
using namespace air::partitions::partition_0;

int
main(int argc, char *argv[])
{
  uint64_t row = 2;
  uint64_t col = 7;

  std::vector<air_agent_t> agents;
  auto get_agents_ret = air_get_agents(&agents);
  assert(get_agents_ret == 0 && "failed to get agents!");

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

  aie_libxaie_ctx_t *xaie = air_init_libxaie();

  queue_t *q = queues[0];

  printf("loading aie_ctrl.so\n");
  auto handle = air_module_load_from_file(nullptr,q);
  assert(handle && "failed to open aie_ctrl.so");

  auto graph_fn = (void (*)(void*))dlsym((void*)handle, "_mlir_ciface_graph");
  assert(graph_fn && "failed to locate _mlir_ciface_graph in aie_ctrl.so");

  tensor_t<uint32_t,1> input;
  input.shape[0] = 256;
  input.alloc = input.data = (uint32_t *)malloc(sizeof(uint32_t) * 256);
  for (int i=0; i<input.shape[0]; i++) {
    input.data[i] = i;
  }

  input.data[24] = 0xacdc;

  auto i = &input;
  graph_fn(i);

  mlir_aie_print_dma_status(xaie, 7, 2);

  uint32_t d = mlir_aie_read_buffer_buf0(xaie, 24);
  printf("ID %x\n", d);

  free(input.alloc);

  if (d == 0xacdc) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail.\n");
    return -1;
  }

}
