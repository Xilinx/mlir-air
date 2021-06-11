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
#include "test_library.h"

#define XAIE_NUM_ROWS            8
#define XAIE_NUM_COLS           50
#define XAIE_ADDR_ARRAY_OFF     0x800

#define HIGH_ADDR(addr)	((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr)	(addr & 0x00000000ffffffff)

#define VERBOSE 1

#define IMAGE_WIDTH 32
#define IMAGE_HEIGHT 16
#define IMAGE_SIZE  (IMAGE_WIDTH * IMAGE_HEIGHT)

#define TILE_WIDTH 16
#define TILE_HEIGHT 8
#define TILE_SIZE  (TILE_WIDTH * TILE_HEIGHT)

namespace {

// global libxaie state
air_libxaie1_ctx_t *xaie;

#define TileInst (xaie->TileInst)
#define TileDMAInst (xaie->TileDMAInst)
#include "aie.1.inc"
#undef TileInst
#undef TileDMAInst

queue_t *q = nullptr;
uint32_t *bram_ptr = nullptr;

}

int
main(int argc, char *argv[])
{
  auto shim_col = 2;

  xaie = air_init_libxaie1();

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd != -1) {
    bram_ptr = (uint32_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, (0x4000+AIR_VCK190_SHMEM_BASE));
    for (int i=0; i<IMAGE_SIZE; i++) {
      bram_ptr[i] = 0xFFFF1111;
      bram_ptr[i+IMAGE_SIZE] = 0xeeee2222;
    }
  }

  for (int i=0; i<TILE_SIZE; i++)
    mlir_write_buffer_scratch_0_0(i,0xfadefade);

  // Fire up the AIE array

  mlir_configure_cores();
  mlir_configure_dmas();
  mlir_initialize_locks();
  mlir_configure_switchboxes();
  mlir_start_cores();

  printf("loading aie_ctrl.so\n");
  auto handle = air_module_load_from_file("./aie_ctrl.so");
  assert(handle && "failed to open aie_ctrl.so");

  auto graph_fn = (void (*)(void*,void *))dlsym((void*)handle, "_mlir_ciface_graph");
  assert(graph_fn && "failed to locate _mlir_ciface_graph in .so");

  tensor_t<uint32_t,2> input;
  tensor_t<uint32_t,2> output;

  input.shape[0] = 32; input.shape[1] = 16;
  input.d = input.aligned = (uint32_t*)malloc(sizeof(uint32_t)*input.shape[0]*input.shape[1]);;

  output.shape[0] = 32; output.shape[1] = 16;
  output.d = output.aligned = (uint32_t*)malloc(sizeof(uint32_t)*output.shape[0]*output.shape[1]);

  for (int i=0; i<IMAGE_SIZE; i++) {
    input.d[i] = i+0x1000;
    output.d[i] = 0x00defaced;
  }

  void *i, *o;
  i = &input;
  o = &output;
  graph_fn(i, o);

  // Go look at the scratch buffer
  int errors = 0;
  for (int i=0;i<TILE_SIZE;i++) {
    u32 rb = mlir_read_buffer_scratch_0_0(i);
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
    u32 rb = output.d[i];

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

  // Now clean up

  for (int bd=0;bd<16;bd++) {
    // Take no prisoners.  No regerts
    // Overwrites the DMA_BDX_Control registers
    u32 rb = XAieGbl_Read32(xaie->TileInst[shim_col][0].TileAddr + 0x0001D008+(bd*0x14));
    printf("Before : bd%x control is %08X\n", bd, rb);
    XAieGbl_Write32(xaie->TileInst[shim_col][0].TileAddr + 0x0001D008+(bd*0x14), 0x0);
  }

  if (!errors) {
    printf("PASS!\n");
  }
  else {
    printf("fail %d/%d.\n", (TILE_SIZE+IMAGE_SIZE-errors), TILE_SIZE+IMAGE_SIZE);
  }

}
