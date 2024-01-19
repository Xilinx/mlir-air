//===- air_test.cpp ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <vector>

#include "air.hpp"
#include "test_library.h"

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

#define IMAGE_WIDTH 256  
#define IMAGE_HEIGHT 16
#define IMAGE_SIZE  (IMAGE_WIDTH * IMAGE_HEIGHT)

#define TILE_WIDTH 8
#define TILE_HEIGHT 8
#define TILE_SIZE  (TILE_WIDTH * TILE_HEIGHT)

#define NUM_3D (IMAGE_WIDTH / TILE_WIDTH)
#define NUM_4D (IMAGE_HEIGHT / TILE_HEIGHT)

namespace air::segments::segment_0 {
void mlir_aie_write_buffer_scratch_0_0(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_0_1(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_0_2(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_0_3(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_1_0(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_1_1(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_1_2(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_1_3(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_2_0(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_2_1(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_2_2(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_2_3(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_3_0(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_3_1(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_3_2(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_3_3(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_4_0(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_4_1(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_4_2(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_4_3(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_5_0(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_5_1(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_5_2(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_5_3(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_6_0(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_6_1(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_6_2(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_6_3(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_7_0(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_7_1(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_7_2(aie_libxaie_ctx_t*, int, int32_t);
void mlir_aie_write_buffer_scratch_7_3(aie_libxaie_ctx_t*, int, int32_t);
}; // namespace air::segments::segment_0
using namespace air::segments::segment_0;

int
main(int argc, char *argv[])
{

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

  aie_libxaie_ctx_t *xaie = (aie_libxaie_ctx_t *)air_get_libxaie_ctx();

  for (int i=0; i<TILE_SIZE; i++) {
    mlir_aie_write_buffer_scratch_0_0(xaie,i,0xfade0000);
    mlir_aie_write_buffer_scratch_0_1(xaie,i,0xfade0000);
    mlir_aie_write_buffer_scratch_0_2(xaie,i,0xfade0000);
    mlir_aie_write_buffer_scratch_0_3(xaie,i,0xfade0000);
    mlir_aie_write_buffer_scratch_1_0(xaie,i,0xfade0001);
    mlir_aie_write_buffer_scratch_1_1(xaie,i,0xfade0001);
    mlir_aie_write_buffer_scratch_1_2(xaie,i,0xfade0001);
    mlir_aie_write_buffer_scratch_1_3(xaie,i,0xfade0001);
    mlir_aie_write_buffer_scratch_2_0(xaie,i,0xfade0002);
    mlir_aie_write_buffer_scratch_2_1(xaie,i,0xfade0002);
    mlir_aie_write_buffer_scratch_2_2(xaie,i,0xfade0002);
    mlir_aie_write_buffer_scratch_2_3(xaie,i,0xfade0002);
    mlir_aie_write_buffer_scratch_3_0(xaie,i,0xfade0003);
    mlir_aie_write_buffer_scratch_3_1(xaie,i,0xfade0003);
    mlir_aie_write_buffer_scratch_3_2(xaie,i,0xfade0003);
    mlir_aie_write_buffer_scratch_3_3(xaie,i,0xfade0003);
    mlir_aie_write_buffer_scratch_4_0(xaie,i,0xfade0004);
    mlir_aie_write_buffer_scratch_4_1(xaie,i,0xfade0004);
    mlir_aie_write_buffer_scratch_4_2(xaie,i,0xfade0004);
    mlir_aie_write_buffer_scratch_4_3(xaie,i,0xfade0004);
    mlir_aie_write_buffer_scratch_5_0(xaie,i,0xfade0005);
    mlir_aie_write_buffer_scratch_5_1(xaie,i,0xfade0005);
    mlir_aie_write_buffer_scratch_5_2(xaie,i,0xfade0005);
    mlir_aie_write_buffer_scratch_5_3(xaie,i,0xfade0005);
    mlir_aie_write_buffer_scratch_6_0(xaie,i,0xfade0006);
    mlir_aie_write_buffer_scratch_6_1(xaie,i,0xfade0006);
    mlir_aie_write_buffer_scratch_6_2(xaie,i,0xfade0006);
    mlir_aie_write_buffer_scratch_6_3(xaie,i,0xfade0006);
    mlir_aie_write_buffer_scratch_7_0(xaie,i,0xfade0007);
    mlir_aie_write_buffer_scratch_7_1(xaie,i,0xfade0007);
    mlir_aie_write_buffer_scratch_7_2(xaie,i,0xfade0007);
    mlir_aie_write_buffer_scratch_7_3(xaie,i,0xfade0007);
  }

  printf("loading aie_ctrl.so\n");
  auto handle = air_module_load_from_file(nullptr, &agents[0], q);
  assert(handle && "failed to open aie_ctrl.so");

  auto graph_fn = (void (*)(void*,void *))dlsym((void*)handle, "_mlir_ciface_graph");
  assert(graph_fn && "failed to locate _mlir_ciface_graph in .so");

  tensor_t<uint32_t,2> input;
  tensor_t<uint32_t,2> output;

  input.shape[1] = IMAGE_WIDTH;
  input.shape[0] = IMAGE_HEIGHT;
  input.alloc = input.data =
      (uint32_t *)malloc(sizeof(uint32_t) * input.shape[0] * input.shape[1]);

  output.shape[1] = IMAGE_WIDTH;
  output.shape[0] = IMAGE_HEIGHT;
  output.alloc = output.data =
      (uint32_t *)malloc(sizeof(uint32_t) * output.shape[0] * output.shape[1]);

  for (int i=0; i<IMAGE_SIZE; i++) {
    input.data[i] = i + 0x1000;
    output.data[i] = 0xdecaf;
  }

  void *i, *o;
  i = &input;
  o = &output;
  graph_fn(i, o);

  uint32_t errs = 0;
  // Check the memory we updated
  for (int i=0; i<IMAGE_SIZE; i++) {
    uint32_t d = output.data[i];
    u32 r = i / IMAGE_WIDTH;
    u32 c = i % IMAGE_WIDTH;
    uint32_t id =
        (r >= TILE_HEIGHT) ? input.data[i - IMAGE_WIDTH * TILE_HEIGHT] : 0;
    if ((r >= TILE_HEIGHT)) {
      if (d != (id)) {
        printf("ERROR: copy idx %d Expected %08X, got %08X\n", i, id, d);
        errs++;
      }
    } 
    if ((r < TILE_HEIGHT)) {
      if (d != 0xdecaf) {
        printf("ERROR: output idx %d Expected %08X, got %08X\n", i, 0xdecaf, d);
        errs++;
      }
    }
  }

  // Clean up
  free(input.alloc);
  free(output.alloc);
  air_module_unload(handle);
  hsa_queue_destroy(queues[0]);

  hsa_status_t shut_down_ret = air_shut_down();
  if (shut_down_ret != HSA_STATUS_SUCCESS) {
    printf("[ERROR] air_shut_down() failed\n");
    errs++;
  }

  if (errs == 0) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail.\n");
    return -1;
  }

}
