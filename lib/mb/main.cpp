
#include <cstdint>
#include <cstring>

extern "C" {
#include "xil_printf.h"
#include "mb_interface.h"

#include "acdc_queue.h"
#include "hsa_defs.h"
}

#include "platform.h"

#define MB_QUEUE_SIZE 128

namespace {

const uint64_t base_address = 0x020100000000UL;

int queue_create(uint32_t size, queue_t **queue)
{
  uint64_t queue_address[1] = {base_address + sizeof(dispatch_packet_t)};
  uint64_t queue_base_address[1] = {queue_address[0] + sizeof(dispatch_packet_t)};

  xil_printf("setup_queue, %x bytes + %d 64 byte packets\n", sizeof(queue_t), size);

  // The address of the queue_t is stored @ base_address
  memcpy((void*)base_address, (void*)queue_address, sizeof(uint64_t));

  // Initialize the queue_t
  queue_t q;
  q.type = HSA_QUEUE_TYPE_SINGLE;
  q.features = HSA_QUEUE_FEATURE_AGENT_DISPATCH;

  memcpy((void*)&q.base_address, (void*)queue_base_address, sizeof(uint64_t));
  q.doorbell = 0xffffffffffffffffUL;
  q.size = size;
  q.reserved0 = 0;
  q.id = 0xacdc;

  q.read_index = 0;
  q.write_index = 0;
  q.last_doorbell = 0;

  memcpy((void*)queue_address[0], (void*)&q, sizeof(queue_t));

  // Invalidate the packets in the queue
  for (uint32_t idx=0; idx<size; idx++) {
    dispatch_packet_t *pkt = &((dispatch_packet_t*)queue_base_address[0])[idx];
    pkt->header = HSA_PACKET_TYPE_INVALID;
  }

  memcpy((void*)queue, (void*)queue_address, sizeof(uint64_t));
  return 0;
}

uint64_t queue_load_read_index(queue_t *q)
{
  return q->read_index;
}

uint64_t queue_add_write_index(queue_t *q)
{
  return q->write_index++;
}

uint64_t queue_load_write_index(queue_t *q)
{
  return q->write_index;
}

bool packet_get_active(dispatch_packet_t *pkt)
{
  return pkt->reserved1 & 0x1;
}

void packet_set_active(dispatch_packet_t *pkt, bool b)
{
  pkt->reserved1 = (pkt->reserved1 & ~0x1) | b;
}

void handle_agent_dispatch(dispatch_packet_t *pkt)
{
  packet_set_active(pkt, true);
  uint32_t pkt_idx;
  memcpy(&pkt_idx, &pkt, sizeof(pkt_idx));
  pkt_idx = ((pkt_idx & 0x3fff) >> 6) - 2;
  xil_printf("handle agent dispatch pkt %x\n", pkt_idx);
}

} // namespace

int main()
{
  init_platform();

  queue_t *q = nullptr;
  queue_create(MB_QUEUE_SIZE, &q);
  xil_printf("Created queue @ 0x%llx\n", (size_t)q);
  bool done = false;
  int cnt = 0;
  while (!done) {
    if (!(cnt++ % 0x00100000))
      xil_printf("No Ding Dong 0x%llx\n", q->doorbell);

    if (q->doorbell+1 > q->last_doorbell) {
      xil_printf("Ding Dong 0x%llx\n", q->doorbell);

      q->last_doorbell = q->doorbell+1;

      auto rd_idx = queue_load_read_index(q);
      auto wr_idx = queue_load_write_index(q);
      for (; rd_idx < wr_idx; rd_idx++) {
        xil_printf("Handle pkts\n");
        dispatch_packet_t *pkt = &((dispatch_packet_t*)q->base_address)[rd_idx % q->size];
        switch (pkt->header & 0xff) {
          default:
          case HSA_PACKET_TYPE_INVALID:
            break;
          case HSA_PACKET_TYPE_AGENT_DISPATCH:
            handle_agent_dispatch(pkt);
            break;
        }
      }

    }
  }

  cleanup_platform();
  return 0;
}
