//===- test.cpp -------------------------------------------------*- C++ -*-===//
//
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
#define PROFILE 0

namespace air::segments::segment_0 {
int32_t mlir_aie_read_buffer_beef_0_0(aie_libxaie_ctx_t *, int);
};
using namespace air::segments::segment_0;

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

  auto handle = air_module_load_from_file(nullptr, q);
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

  if (!errors) {
    printf("PASS!\n");
  } else {
    printf("fail %d/%d.\n", 4 - errors, 4);
    return -1;
  }

  return 0;
}
