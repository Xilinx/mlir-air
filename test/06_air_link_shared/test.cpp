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

#define IMAGE_WIDTH 32
#define IMAGE_HEIGHT 16
#define IMAGE_SIZE  (IMAGE_WIDTH * IMAGE_HEIGHT)

#define TILE_WIDTH 16
#define TILE_HEIGHT 8
#define TILE_SIZE  (TILE_WIDTH * TILE_HEIGHT)

int
main(int argc, char *argv[])
{
  auto shim_col = 2;

  aie_libxaie_ctx_t *xaie = air_init_libxaie1();

  for (int i=0; i<TILE_SIZE; i++)
    mlir_aie_write_buffer_scratch_0_0(xaie, i, 0xfadefade);

  printf("loading aie_ctrl.so\n");
  auto handle = air_module_load_from_file("./aie_ctrl.so");
  assert(handle && "failed to load aie_ctrl.so");

  auto graph_fn = (void (*)(void*,void *))dlsym((void*)handle, "_mlir_ciface_graph");
  assert(graph_fn && "failed to locate _mlir_ciface_graph in .so");

  tensor_t<uint32_t,2> input;
  tensor_t<uint32_t,2> output;

  input.shape[0] = 32; input.shape[1] = 16;
  input.alloc = input.data =
      (uint32_t *)malloc(sizeof(uint32_t) * input.shape[0] * input.shape[1]);

  output.shape[0] = 32; output.shape[1] = 16;
  output.alloc = output.data =
      (uint32_t *)malloc(sizeof(uint32_t) * output.shape[0] * output.shape[1]);

  for (int i=0; i<IMAGE_SIZE; i++) {
    input.data[i] = i + 0x1000;
    output.data[i] = 0x00defaced;
  }

  void *i, *o;
  i = &input;
  o = &output;
  graph_fn(i, o);

  int errors = 0;

  // Go look at the scratch buffer
  for (int i=0;i<TILE_SIZE;i++) {
    u32 rb = mlir_aie_read_buffer_scratch_0_0(xaie, i);
    u32 row = i / TILE_WIDTH;
    u32 col = i % TILE_WIDTH;
    u32 orig_index = row * IMAGE_WIDTH + col;
    if (!(rb == orig_index+0x1000)) {
      printf("SB %d [%d, %d] should be %08X, is %08X\n", i, col, row, orig_index, rb);
      errors++;
    }
  }

  // Now look at the image, should have the top left filled in
  for (int i=0;i<IMAGE_SIZE;i++) {
    u32 rb = output.data[i];

    u32 row = i / IMAGE_WIDTH;
    u32 col = i % IMAGE_WIDTH;

    if ((row < TILE_HEIGHT) && (col < TILE_WIDTH)) {
      if (!(rb == 0x1000+i)) {
        printf("IM %d [%d, %d] should be %08X, is %08X\n", i, col, row, i, rb);
        errors++;
      }
    }
    else {
      if (rb != 0x00defaced) {
        printf("IM %d [%d, %d] should be 0xdefaced, is %08X\n", i, col, row, rb);
        errors++;
      }
    }
  }

  free(input.alloc);
  free(output.alloc);

  if (!errors) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail %d/%d.\n", (TILE_SIZE+IMAGE_SIZE-errors), TILE_SIZE+IMAGE_SIZE);
    return -1;
  }

}
