#include <cstdio>
#include <cassert>

#include "air_host.h"
#include "acdc_queue.h"
#include "hsa_defs.h"

int main(int argc, char *argv[])
{
  // create the queue
  queue_t *q = nullptr;
  auto ret = air_queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q, AIR_VCK190_SHMEM_BASE);
  assert(ret == 0 && "failed to create queue!");

  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;

  auto row = 4;
  auto col = 13;
  auto num_rows = 1;
  auto num_cols = 1;

  dispatch_packet_t *herd_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  air_packet_herd_init(herd_pkt, 0, col, num_cols, row, num_rows);
  air_queue_dispatch_and_wait(q, wr_idx, herd_pkt);

  printf("PASS!\n");
}
