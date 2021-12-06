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

#include <xaiengine.h>

#include "air_host.h"
#include "air_tensor.h"

#include "aie_inc.cpp"

#define VERBOSE 1

#define TENSOR_1D 16
#define TENSOR_2D 4
#define TENSOR_3D 2
#define TENSOR_4D 3
#define TENSOR_SIZE  (TENSOR_1D * TENSOR_2D * TENSOR_3D * TENSOR_4D)

#define TILE_1D 4
#define TILE_2D 2
#define TILE_3D 2
#define TILE_4D 2
#define TILE_SIZE  (TILE_1D * TILE_2D * TILE_3D * TILE_4D)

int
main(int argc, char *argv[])
{
  auto shim_col = 2;

  aie_libxaie_ctx_t *xaie = air_init_libxaie1();
  mlir_aie_init_device(xaie);

  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);
  mlir_aie_configure_dmas(xaie);

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  for (int i=0; i<TILE_SIZE; i++)
    mlir_aie_write_buffer_buf0(xaie, i, 0xfadefade);

  printf("loading aie_ctrl.so\n");
  auto handle = air_module_load_from_file("./aie_ctrl.so",q);
  assert(handle && "failed to open aie_ctrl.so");

  auto graph_fn = (void (*)(void*))dlsym((void*)handle, "_mlir_ciface_graph");
  assert(graph_fn && "failed to locate _mlir_ciface_graph in .so");

  tensor_t<uint32_t,4> input;

  input.shape[0] = TENSOR_1D; input.shape[1] = TENSOR_2D;
  input.shape[2] = TENSOR_3D; input.shape[3] = TENSOR_4D;
  input.d = input.aligned = (uint32_t*)malloc(sizeof(uint32_t)*input.shape[0]*
                                              input.shape[1]*input.shape[2]*input.shape[3]);

  for (int i=0; i<TENSOR_SIZE; i++) {
    input.d[i] = i;
  }

  mlir_aie_start_cores(xaie);

  void *i;
  i = &input;
  graph_fn(i);

  int errors = 0;

  // Now look at the image, should have the bottom left filled in
  for (int i=0;i<TILE_SIZE;i++) {
    uint32_t rb = mlir_aie_read_buffer_buf0(xaie, i);
    // An = Aoffset * ((n / Aincrement) % Awrap)
    // Aoffset = add for each increment
    // Awrap = how many increments before wrapping
    // Aincrement = how many streams before increment
    uint32_t xn = 1*((i/1)%4);
    uint32_t yn = 16*((i/4)%2);
    uint32_t zn = 64*((i/8)%2);
    uint32_t wn = 256*((i/16)%2);
    uint32_t a = xn + yn + zn + wn;
    uint32_t vb = input.d[a];
    if (!(rb == vb)) {
      printf("Tile Mem %d should be %08X, is %08X\n", i, vb, rb);
      errors++;
    }
  }

  if (!errors) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail %d/%d.\n", (TILE_SIZE+TENSOR_SIZE-errors), TILE_SIZE+TENSOR_SIZE);
    return -1;
  }

}
