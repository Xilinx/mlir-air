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

#define DMA_COUNT 256

namespace air::herds::herd_0 {
void mlir_aie_write_buffer_buf0(aie_libxaie_ctx_t*, int, int32_t);
};
using namespace air::herds::herd_0;

int
main(int argc, char *argv[])
{
  uint64_t row = 4;
  uint64_t col = 5;
  
  aie_libxaie_ctx_t *xaie = air_init_libxaie1();

  for (int i=0; i<DMA_COUNT; i++)
    mlir_aie_write_buffer_buf0(xaie, i, i+0x10);

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  printf("loading aie_ctrl.so\n");
  auto handle = air_module_load_from_file(nullptr,q);
  assert(handle && "failed to open aie_ctrl.so");

  auto graph_fn = (void (*)(void*))dlsym((void*)handle, "_mlir_ciface_graph");
  assert(graph_fn && "failed to locate _mlir_ciface_graph in aie_ctrl.so");

  tensor_t<uint32_t,1> output;
  output.shape[0] = DMA_COUNT;
  XAieLib_MemInst *mem_o = XAieLib_MemAllocate(sizeof(uint32_t)*output.shape[0], 0);
  output.d = output.aligned = (uint32_t*)XAieLib_MemGetPaddr(mem_o);
  uint32_t *out = (uint32_t*)XAieLib_MemGetVaddr(mem_o);
  for (int i=0; i<output.shape[0]; i++) {
    out[i] = 0xfacefeed;
  }

  XAieLib_MemSyncForDev(mem_o);

  auto o = &output;
  graph_fn(o);

  int errors = 0;
  for (int i=0; i<DMA_COUNT; i++) {
    uint32_t d = out[i];
    if (d != (i+0x10)) {
      errors++;
      printf("mismatch %x != 0x10 + %x\n", d, i);
    }
  }

  XAieLib_MemFree(mem_o);

  if (!errors) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail %d/%d.\n", (DMA_COUNT-errors), DMA_COUNT);
    return -1;
  }

}
