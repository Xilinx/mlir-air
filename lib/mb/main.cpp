
#include <cstdint>
#include <cstring>

extern "C" {
#include "xil_printf.h"

#include "xaiengine.h"
#include "mb_interface.h"

#include "acdc_queue.h"
#include "hsa_defs.h"
}

#include "platform.h"

#define XAIE_NUM_ROWS            8
#define XAIE_NUM_COLS           50
#define XAIE_ADDR_ARRAY_OFF     0x800

#define HIGH_ADDR(addr)	((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr)	(addr & 0x00000000ffffffff)

namespace {

XAieGbl_Config *AieConfigPtr;	                          /**< AIE configuration pointer */
XAieGbl AieInst;	                                      /**< AIE global instance */
XAieGbl_HwCfg AieConfig;                                /**< AIE HW configuration instance */
XAieGbl_Tile TileInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];  /**< Instantiates AIE array of [XAIE_NUM_COLS] x [XAIE_NUM_ROWS] */
XAieDma_Tile TileDMAInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];

int xaie_lock_release(u32 col, u32 row, u32 lock_id, u32 val)
{
  //XAieTile_LockRelease(&(TileInst[col][row]), lock_id, val, 0);
  return 1;
}

int xaie_lock_acquire_nb(u32 col, u32 row, u32 lock_id, u32 val)
{
  u8 lock_ret = 0;
  u32 loop = 0;
  while ((!lock_ret) && (loop < 512)) {
    //lock_ret = XAieTile_LockAcquire(&(TileInst[col][row]), lock_id, val, 10000);
    loop++;
  }
  if (loop == 512) {
    //xil_printf("Acquire of lock [%d, %d, %d] with value %d timed out\n", col, row, lock_id, val);
    return 0;
  }
  return 1;
}

void xaie_init()
{
  size_t aie_base = XAIE_ADDR_ARRAY_OFF << 14;
  // XAIEGBL_HWCFG_SET_CONFIG((&AieConfig), XAIE_NUM_ROWS, XAIE_NUM_COLS, XAIE_ADDR_ARRAY_OFF);
  // XAieGbl_HwInit(&AieConfig);
  // AieConfigPtr = XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
  // XAieGbl_CfgInitialize(&AieInst, &TileInst[0][0], AieConfigPtr);
}

}

namespace {

const uint64_t base_address = 0x020100000000UL;

int queue_create(uint32_t size, queue_t **queue)
{
  uint64_t queue_address[1] = {base_address + sizeof(dispatch_packet_t)};
  uint64_t queue_base_address[1] = {queue_address[0] + sizeof(dispatch_packet_t)};

  xil_printf("setup_queue, %x bytes + %d 64 byte packets\n\r", sizeof(queue_t), size);

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

void complete_agent_dispatch_packet(dispatch_packet_t *pkt)
{
  // completion phase
  packet_set_active(pkt, false);
  signal_subtract_acq_rel((signal_t*)&pkt->completion_signal, 1);
}

void handle_agent_dispatch_packet(dispatch_packet_t *pkt)
{
  // get the index
  uint32_t pkt_idx;
  memcpy(&pkt_idx, &pkt, sizeof(pkt_idx));
  pkt_idx = ((pkt_idx & 0x3fff) >> 6) - 2;
  xil_printf("handle agent dispatch pkt %x @ 0x%llx\n\r", pkt_idx, (size_t)pkt);

  // packet is in active phase
  packet_set_active(pkt, true);
}

} // namespace

int main()
{
  init_platform();
  xaie_init();

  queue_t *q = nullptr;
  queue_create(MB_QUEUE_SIZE, &q);
  xil_printf("Created queue @ 0x%llx\n\r", (size_t)q);
  bool done = false;
  int cnt = 0;
  while (!done) {
    if (!(cnt++ % 0x00100000))
      xil_printf("No Ding Dong 0x%llx\n\r", q->doorbell);

    if (q->doorbell+1 > q->last_doorbell) {
      xil_printf("Ding Dong 0x%llx\n\r", q->doorbell);

      q->last_doorbell = q->doorbell+1;

      auto rd_idx = queue_load_read_index(q);
      xil_printf("Handle pkt read_index=%d\n\r", rd_idx);

      dispatch_packet_t *pkt = &((dispatch_packet_t*)q->base_address)[rd_idx % q->size];
      switch (pkt->type) {
        default:
        case HSA_PACKET_TYPE_INVALID:
          break;
        case HSA_PACKET_TYPE_AGENT_DISPATCH:
          handle_agent_dispatch_packet(pkt);
          complete_agent_dispatch_packet(pkt);
          queue_add_read_index(q, 1);
          break;
      }
    }
  }

  cleanup_platform();
  return 0;
}
