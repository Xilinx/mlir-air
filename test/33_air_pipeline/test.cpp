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

#include <xaiengine.h>

#include "air_host.h"
#include "air_tensor.h"

int main(int argc, char *argv[]) {
  auto col = 3;
  auto row = 3;

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

#define DATA_LENGTH 1024
#define DATA_TYPE int

  tensor_t<DATA_TYPE, 1> input_a, input_b;
  tensor_t<DATA_TYPE, 1> output;
  input_a.shape[0] = DATA_LENGTH;
  input_a.alloc = input_a.data =
      (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * input_a.shape[0]);

  input_b.shape[0] = DATA_LENGTH;
  input_b.alloc = input_b.data =
      (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * input_b.shape[0]);

  output.shape[0] = input_a.shape[0];
  output.alloc = output.data =
      (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * output.shape[0]);
  DATA_TYPE d = 1;
  for (int i = 0; i < input_a.shape[0]; i++) {
    input_a.data[i] = d;
    input_b.data[i] = ((DATA_TYPE)DATA_LENGTH) + d;
    output.data[i] = -1;
    d += 1;
  }

  printf("loading aie_ctrl.so\n");
  auto handle = air_module_load_from_file(nullptr, q);
  assert(handle && "failed to open aie_ctrl.so");

  auto launch = (void (*)(void *, void *, void *))dlsym((void *)handle,
                                                        "_mlir_ciface_launch");
  assert(launch && "failed to locate _mlir_ciface_launch in .so");

  launch((void *)&input_a, (void *)&input_b, (void *)&output);

  int errors = 0;

  for (int i = 0; i < DATA_LENGTH; i++) {
    DATA_TYPE ref = (input_a.data[i] * input_b.data[i]) + (DATA_TYPE)1 +
                    (DATA_TYPE)2 + (DATA_TYPE)3;
    if (output.data[i] != ref) {
      printf("output[%d] = %d (expected %d)\n", i, output.data[i], ref);
      errors++;
    }
  }

  free(output.alloc);
  free(input_a.alloc);
  free(input_b.alloc);

  if (!errors) {
    printf("PASS!\n");
    return 0;
  } else {
    printf("fail %d/%d.\n", (DATA_LENGTH - errors), DATA_LENGTH);
    return -1;
  }
}
