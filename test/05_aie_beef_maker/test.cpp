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
#include <xaiengine.h>

#include "air_host.h"
#include "test_library.h"

#define SCRATCH_AREA 8

namespace {

// global libxaie state
air_libxaie1_ctx_t *xaie;

#define TileInst (xaie->TileInst)
#define TileDMAInst (xaie->TileDMAInst)
#include "aie_inc.cpp"
#undef TileInst
#undef TileDMAInst

}

int
main(int argc, char *argv[])
{
  auto col = 7;
  auto row = 2;

  xaie = air_init_libxaie1();

  ACDC_print_tile_status(xaie->TileInst[col][row]);

  mlir_configure_cores();
  mlir_configure_switchboxes();
  mlir_initialize_locks();
  mlir_configure_dmas();
  mlir_start_cores();

  ACDC_print_tile_status(xaie->TileInst[col][row]);

  // We first write an ascending pattern into the area the AIE will write into
  for (int i=0; i<SCRATCH_AREA; i++) {
    uint32_t d = i+1;
    mlir_write_buffer_buffer(i, d);
  }

  // We wrote data, so lets toggle the job lock 0
  XAieTile_LockRelease(&(xaie->TileInst[col][row]), 0, 0x1, 0);

  ACDC_print_tile_status(xaie->TileInst[col][row]);

  auto count = 0;
  while (!XAieTile_LockAcquire(&(xaie->TileInst[col][row]), 0, 0, 1000)) {
    count++;
    if (!(count % 1000)) {
      printf("%d seconds\n",count/1000);
      if (count == 2000) break;
    }
  }

  int errors = 0;
  ACDC_check("Check Result 0:", mlir_read_buffer_buffer(0), 0xdeadbeef,errors);
  ACDC_check("Check Result 1:", mlir_read_buffer_buffer(1), 0xcafecafe,errors);
  ACDC_check("Check Result 2:", mlir_read_buffer_buffer(2), 0x000decaf,errors);
  ACDC_check("Check Result:", mlir_read_buffer_buffer(3), 0x5a1ad000,errors);

  for (int i=4; i<SCRATCH_AREA; i++)
    ACDC_check("Check Result 3:", mlir_read_buffer_buffer(i), i+1,errors);

  if (!errors) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail %d/%d.\n", (SCRATCH_AREA-errors), SCRATCH_AREA);
    return -1;
  }
}
