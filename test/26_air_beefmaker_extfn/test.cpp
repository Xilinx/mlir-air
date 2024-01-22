//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdio>
#include <dlfcn.h>
#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

#include "air.hpp"
#include "air_host.h"
#include "test_library.h"

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

#define VERBOSE 1
#define PROFILE 0

namespace air::segments::segment_0 {
int32_t mlir_aie_read_buffer_beef_0_0(aie_libxaie_ctx_t *, int);
};
using namespace air::segments::segment_0;

int main(int argc, char *argv[]) {

  std::vector<hsa_queue_t *> queues;
  uint32_t aie_max_queue_size(0);

  hsa_status_t init_status = air_init();

  if (init_status != HSA_STATUS_SUCCESS) {
    std::cout << "air_init() failed. Exiting" << std::endl;
    return -1;
  }

  std::vector<hsa_agent_t> agents;
  auto get_agents_ret = air_get_agents(agents);
  assert(get_agents_ret == HSA_STATUS_SUCCESS && "failed to get agents!");

  if (agents.empty()) {
    std::cout << "No agents found. Exiting." << std::endl;
    return -1;
  }

  std::cout << "Found " << agents.size() << " agents" << std::endl;

  hsa_agent_get_info(agents[0], HSA_AGENT_INFO_QUEUE_MAX_SIZE,
                     &aie_max_queue_size);

  std::cout << "Max AIE queue size: " << aie_max_queue_size << std::endl;

  hsa_queue_t *q = NULL;

  // Creating a queue
  auto queue_create_status =
      hsa_queue_create(agents[0], aie_max_queue_size, HSA_QUEUE_TYPE_SINGLE,
                       nullptr, nullptr, 0, 0, &q);

  if (queue_create_status != HSA_STATUS_SUCCESS) {
    std::cout << "hsa_queue_create failed" << std::endl;
  }

  // Adding to our vector of queues
  queues.push_back(q);
  assert(queues.size() > 0 && "No queues were sucesfully created!");

  auto handle = air_module_load_from_file(nullptr, &agents[0], q);
  assert(handle && "failed to open air module");

  auto cow_fn = (void (*)())dlsym((void *)handle, "_mlir_ciface_moo");
  assert(cow_fn && "failed to locate _mlir_ciface_foo in .so");

#if PROFILE
  struct timeval before, after;
  long diff_s, diff_us;
  gettimeofday(&before, NULL);
#endif

  // run it
  cow_fn();

#if PROFILE
  gettimeofday(&after, NULL);
  diff_s = after.tv_sec - before.tv_sec;
  diff_us = after.tv_usec - before.tv_usec;

  if (diff_s)
    diff_us += 10000000;

  printf("before %ld.%06ld\n", before.tv_sec, before.tv_usec);
  printf("after  %ld.%06ld\n", after.tv_sec, after.tv_usec);
  printf("diff   %ld.%06ld\n", diff_s, diff_us);
#endif

  uint32_t reference_data[4] = {0xdeadbeef, 0xcafecafe, 0x000decaf, 0x5a1ad000};

  uint32_t output_data[4];
  aie_libxaie_ctx_t *xaie = (aie_libxaie_ctx_t *)air_get_libxaie_ctx();

  for (int i = 0; i < 4; i++)
    output_data[i] = mlir_aie_read_buffer_beef_0_0(xaie, i);

  unsigned errors = 0;
  for (int i = 0; i < 4; i++) {
    if (VERBOSE)
      printf("data[%d] = %x\n", i, output_data[i]);
    if (output_data[i] != reference_data[i])
      errors++;
  }

  // Clean up
  air_module_unload(handle);
  hsa_queue_destroy(queues[0]);

  hsa_status_t shut_down_ret = air_shut_down();
  if (shut_down_ret != HSA_STATUS_SUCCESS) {
    printf("[ERROR] air_shut_down() failed\n");
    errors++;
  }

  if (!errors) {
    printf("PASS!\n");
  } else {
    printf("fail %d/%d.\n", 4 - errors, 4);
    return -1;
  }

  return 0;
}
