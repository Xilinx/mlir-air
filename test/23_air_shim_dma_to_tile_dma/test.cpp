// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <dlfcn.h>

#include "air_host.h"
#include "air_tensor.h"

namespace air::herds::herd_0 {
void mlir_aie_write_buffer_buf0(aie_libxaie_ctx_t*, int, int32_t);
uint32_t mlir_aie_read_buffer_buf0(aie_libxaie_ctx_t*, int);
};
using namespace air::herds::herd_0;

int
main(int argc, char *argv[])
{
  uint64_t row = 2;
  uint64_t col = 7;

  aie_libxaie_ctx_t *xaie = air_init_libxaie1();

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  printf("loading aie_ctrl.so\n");
  auto handle = air_module_load_from_file(nullptr,q);
  assert(handle && "failed to open aie_ctrl.so");

  auto graph_fn = (void (*)(void*))dlsym((void*)handle, "_mlir_ciface_graph");
  assert(graph_fn && "failed to locate _mlir_ciface_graph in aie_ctrl.so");

  tensor_t<uint32_t,1> input;
  input.shape[0] = 256;
  input.d = (uint32_t*)malloc(sizeof(uint32_t)*256);
  for (int i=0; i<input.shape[0]; i++) {
    input.d[i] = i;
  }

  input.d[24] = 0xacdc;

  auto i = &input;
  graph_fn(i);

  mlir_aie_print_dma_status(xaie, 7, 2);

  uint32_t d = mlir_aie_read_buffer_buf0(xaie, 24);
  printf("ID %x\n", d);

  if (d == 0xacdc) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail.\n");
    return -1;
  }

}
