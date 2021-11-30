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

#define SCRATCH_AREA 8

#include "aie_inc.cpp"

int
main(int argc, char *argv[])
{
  auto col = 7;
  auto row = 2;

  aie_libxaie_ctx_t *xaie = mlir_aie_init_libxaie();
  mlir_aie_init_device(xaie);

  mlir_aie_print_tile_status(xaie, col, row);

  mlir_aie_configure_cores(xaie);
  mlir_aie_configure_switchboxes(xaie);
  mlir_aie_initialize_locks(xaie);
  mlir_aie_configure_dmas(xaie);
  mlir_aie_start_cores(xaie);

  mlir_aie_print_tile_status(xaie, col, row);

  // We first write an ascending pattern into the area the AIE will write into
  for (int i=0; i<SCRATCH_AREA; i++) {
    uint32_t d = i+1;
    mlir_aie_write_buffer_buffer(xaie, i, d);
  }

  // We wrote data, so lets toggle the job lock 0
  XAieTile_LockRelease(&(xaie->TileInst[col][row]), 0, 0x1, 0);

  mlir_aie_print_tile_status(xaie, col, row);

  auto count = 0;
  while (!XAieTile_LockAcquire(&(xaie->TileInst[col][row]), 0, 0, 1000)) {
    count++;
    if (!(count % 1000)) {
      printf("%d seconds\n",count/1000);
      if (count == 2000) break;
    }
  }

  int errors = 0;
  mlir_aie_check("Check Result 0:", mlir_aie_read_buffer_buffer(xaie, 0), 0xdeadbeef,errors);
  mlir_aie_check("Check Result 1:", mlir_aie_read_buffer_buffer(xaie, 1), 0xcafecafe,errors);
  mlir_aie_check("Check Result 2:", mlir_aie_read_buffer_buffer(xaie, 2), 0x000decaf,errors);
  mlir_aie_check("Check Result:", mlir_aie_read_buffer_buffer(xaie, 3), 0x5a1ad000,errors);

  for (int i=4; i<SCRATCH_AREA; i++)
    mlir_aie_check("Check Result 3:", mlir_aie_read_buffer_buffer(xaie, i), i+1,errors);

  if (!errors) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail %d/%d.\n", (SCRATCH_AREA-errors), SCRATCH_AREA);
    return -1;
  }
}
