// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

#include <cstdio>
#include <cassert>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>

#include <xaiengine.h>

#include "air_host.h"
#include "acdc_queue.h"
#include "hsa_defs.h"

#define SHMEM_BASE 0x020100000000LL

#define XAIE_NUM_ROWS            8
#define XAIE_NUM_COLS           50
#define XAIE_ADDR_ARRAY_OFF     0x800

#define HIGH_ADDR(addr)	((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr)	(addr & 0x00000000ffffffff)

namespace {

// global libxaie state
air_libxaie1_ctx_t *xaie;

#define TileInst (xaie->TileInst)
#define TileDMAInst (xaie->TileDMAInst)
#include "aie_inc.cpp"
#undef TileInst
#undef TileDMAInst

}

int main(int argc, char *argv[])
{
  auto col = 7;
  auto row = 2;

  xaie = air_init_libxaie1();

  mlir_configure_cores();
  mlir_configure_switchboxes();
  mlir_initialize_locks();
  mlir_configure_dmas();
  //mlir_start_cores();

  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  // reserve a packet in the queue
  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;

  auto herd_id = 0;
  auto num_rows = 4;
  auto num_cols = 2;
  auto lock_id = 0;

  // herd_setup packet
  // Set up a 2x4 herd starting 7,2
  dispatch_packet_t *herd_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_herd_init(herd_pkt, 0, col, 2, row, 4);
  //air_queue_dispatch_and_wait(q, wr_idx, herd_pkt);

  // reserve another packet in the queue
  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;

  // lock packet
  dispatch_packet_t *lock_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_aie_lock_range(lock_pkt, herd_id, lock_id, /*acq_rel*/0,
                            /*value*/0, 0, num_cols, 0, num_rows);
  air_queue_dispatch_and_wait(q, wr_idx, lock_pkt);

  u32 errors = 0;
  for (int c = 6; c < 10; c++)
    for (int r = 1; r < 7; r++) {
      u32 locks = XAieGbl_Read32(xaie->TileInst[c][r].TileAddr + 0x0001EF00);
      if ((c >= 7) && (c <= 8) && (r >=2) && (r <= 5)) {
        if (locks != 0x1) errors++;
      } else {
        if (locks != 0x0) errors++;
      }
    }

  if (errors) {
    printf("%d errors\n", errors);
    for (int c = 6; c < 10 ; c++)
      for (int r = 1; r < 7; r++) {
        u32 locks = XAieGbl_Read32(xaie->TileInst[c][r].TileAddr + 0x0001EF00);
        printf("C[%d][%d] %08X\n", c, r, locks);
      }
  }
  else {
    // Release the herd locks!
    wr_idx = queue_add_write_index(q, 1);
    packet_id = wr_idx % q->size;
  
    dispatch_packet_t *release_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
    air_packet_aie_lock_range(release_pkt, herd_id, lock_id, /*acq_rel*/1,
                            /*value*/1, 0, num_cols, 0, num_rows);
    air_queue_dispatch_and_wait(q, wr_idx, release_pkt);

    for (int c = 6; c < 10 ; c++)
      for (int r = 1; r < 7; r++) {
        u32 locks = XAieGbl_Read32(xaie->TileInst[c][r].TileAddr + 0x0001EF00);
        if ((c >= 7) && (c <= 8) && (r >=2) && (r <= 5)) {
          if (locks != 0x2) errors++;
	      }
        else {
      	  if (locks != 0x0) errors++;
        }
      }

    if (errors) {
      for (int col = 6; col < 10 ; col++)
        for (int row = 1; row < 7; row++) {
          u32 locks = XAieGbl_Read32(xaie->TileInst[col][row].TileAddr + 0x0001EF00);
          printf("C[%d][%d] %08X\n", col, row, locks);
        }
    }
  }

  if (errors == 0) {
    printf("PASS!\n");
    return 0;
  }
  else {
    printf("fail.\n");
    return -1;
  }
}
