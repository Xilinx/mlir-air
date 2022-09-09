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

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  // reserve a packet in the queue
  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;

  // herd_setup packet
  dispatch_packet_t *herd_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_herd_init(herd_pkt, 0, col, 1, row, 1);
  air_queue_dispatch_and_wait(q, wr_idx, herd_pkt);

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

  uint32_t herd_id = 0;
  uint32_t lock_id = 0;

  // We wrote data, so lets tell the MicroBlaze to toggle the job lock 0
  // reserve another packet in the queue
  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;

  // lock packet
  dispatch_packet_t *lock_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_aie_lock(lock_pkt, herd_id, lock_id, /*acq_rel*/1, /*value*/1, 0, 0);
  air_queue_dispatch_and_wait(q, wr_idx, lock_pkt);

  auto count = 0;
  while (!mlir_aie_acquire_lock(xaie, col, 2, 0, 0, 1000)) {
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
  mlir_aie_check("Check Result 3:", mlir_aie_read_buffer_buffer(xaie, 3), 0x5a1ad000,errors);

  for (int i=4; i<SCRATCH_AREA; i++)
    mlir_aie_check("Check Result:", mlir_aie_read_buffer_buffer(xaie, i), i+1,errors);

  if (!errors) {
    printf("PASS!\n");
    return 0;
      }
  else {
    printf("fail %d/%d.\n", (SCRATCH_AREA-errors), SCRATCH_AREA);
    return -1;
      }
}
