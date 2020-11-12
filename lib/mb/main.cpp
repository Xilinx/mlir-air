
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

namespace xaie {

XAieGbl_Config *AieConfigPtr;	                          /**< AIE configuration pointer */
//XAieGbl AieInst;	                                      /**< AIE global instance */
//XAieGbl_HwCfg AieConfig;                                /**< AIE HW configuration instance */
//XAieGbl_Tile TileInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];  /**< Instantiates AIE array of [XAIE_NUM_COLS] x [XAIE_NUM_ROWS] */
XAieGbl_Tile TileInst;
//XAieDma_Tile TileDMAInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];

XAieGbl_Config XAieGbl_ConfigTable[XPAR_AIE_NUM_INSTANCES] =
{
	{
		XPAR_AIE_DEVICE_ID,
		XPAR_AIE_ARRAY_OFFSET,
		XPAR_AIE_NUM_ROWS,
		XPAR_AIE_NUM_COLUMNS
	}
};

void XAieGbl_HwInit(XAieGbl_HwCfg *CfgPtr)
{
  xaie::XAieGbl_ConfigTable->NumRows = CfgPtr->NumRows;
  xaie::XAieGbl_ConfigTable->NumCols = CfgPtr->NumCols;
  xaie::XAieGbl_ConfigTable->ArrOffset = CfgPtr->ArrayOff;
}

void XAieGbl_CfgInitialize_Tile(XAieGbl *InstancePtr,
                                XAieGbl_Tile *TilePtr,
                                u16 ColIdx,
                                u16 RowIdx,
                                XAieGbl_Config *ConfigPtr)
{
  u64 TileAddr;

  // XAie_AssertNonvoid(InstancePtr != XAIE_NULL);
  XAie_AssertNonvoid(ConfigPtr != XAIE_NULL);

  // if(InstancePtr->IsReady != XAIE_COMPONENT_IS_READY) {
  //   InstancePtr->IsReady = 0U;
  //   InstancePtr->Config = ConfigPtr;
  //   InstancePtr->IsReady = XAIE_COMPONENT_IS_READY;
  //   //XAieLib_InitDev();
  //   InstancePtr->Tiles = TilePtr;

  #ifdef XAIE_BASE_ARRAY_ADDR_OFFSET
    ConfigPtr->ArrOffset = XAIE_BASE_ARRAY_ADDR_OFFSET;
  #endif

    if (RowIdx != 0) {
      /* Row index starts with 1 as row-0 is for shim */
      //for(RowIdx = 1; RowIdx <= ConfigPtr->NumRows; RowIdx++) {
      //for(ColIdx=0; ColIdx < ConfigPtr->NumCols; ColIdx++) {
      // TilePtr = (XAieGbl_Tile *)((char *)TileInstPtr +
      //                                 ((ColIdx * (ConfigPtr->NumRows + 1)) *
      //                                 sizeof(XAieGbl_Tile)) +
      //                                 (RowIdx * sizeof(XAieGbl_Tile)));

      TilePtr->RowId = RowIdx; /* Row index */
      TilePtr->ColId = ColIdx; /* Column index */

      /*
        * Tile address format:
        * --------------------------------------------
        * |                7 bits  5 bits   18 bits  |
        * --------------------------------------------
        * | Array offset | Column | Row | Tile addr  |
        * --------------------------------------------
        */
      TileAddr = (u64)(((u64)ConfigPtr->ArrOffset <<
          XAIEGBL_TILE_ADDR_ARR_SHIFT) |
        (ColIdx << XAIEGBL_TILE_ADDR_COL_SHIFT) |
        (RowIdx << XAIEGBL_TILE_ADDR_ROW_SHIFT));

      TilePtr->TileAddr = TileAddr;

      /* Set memory module base address for tile */
      TilePtr->MemModAddr = TileAddr +
          XAIEGBL_TILE_ADDR_MEMMODOFF;

      /* Set core module base address for tile */
      TilePtr->CoreModAddr = TileAddr +
          XAIEGBL_TILE_ADDR_COREMODOFF;

      TilePtr->NocModAddr = 0U;
      TilePtr->PlModAddr = 0U;

      /* Set locks base address in memory module */
      TilePtr->LockAddr = TilePtr->MemModAddr+
          XAIEGBL_TILE_ADDR_MEMLOCKOFF;

      /* Set Stream SW base address in core module */
      TilePtr->StrmSwAddr =
          TilePtr->CoreModAddr +
          XAIEGBL_TILE_ADDR_CORESTRMOFF;

      TilePtr->TileType = XAIEGBL_TILE_TYPE_AIETILE;

      TilePtr->IsReady = XAIE_COMPONENT_IS_READY;
      //XAieLib_InitTile(TilePtr);

      // XAie_print("Tile addr:%016lx, Row idx:%d, "
      // 	"Col idx:%d, Memmodaddr:%016lx, "
      // 	"Coremodaddr:%016lx\n",TileAddr,RowIdx,
      // 	ColIdx, TilePtr->MemModAddr,
      // 	TilePtr->CoreModAddr);
    }
    else { // (RowIdx == 0)

      TilePtr->RowId = 0U; /* Row index */
      TilePtr->ColId = ColIdx; /* Column index */

      TileAddr = (u64)(((u64)ConfigPtr->ArrOffset <<
          XAIEGBL_TILE_ADDR_ARR_SHIFT) |
          (ColIdx << XAIEGBL_TILE_ADDR_COL_SHIFT));

      TilePtr->TileAddr = TileAddr;
      TilePtr->MemModAddr = 0U;
      TilePtr->CoreModAddr = 0U;

      /* Set Noc module base address for tile */
      TilePtr->NocModAddr = TileAddr +
            XAIEGBL_TILE_ADDR_NOCMODOFF;
      /* Set PL module base address for tile */
      TilePtr->PlModAddr = TileAddr +
            XAIEGBL_TILE_ADDR_PLMODOFF;

      /* Set locks base address in NoC module */
      TilePtr->LockAddr = TilePtr->NocModAddr +
            XAIEGBL_TILE_ADDR_NOCLOCKOFF;

      /* Set Stream SW base address in PL module */
      TilePtr->StrmSwAddr = TilePtr->PlModAddr +
            XAIEGBL_TILE_ADDR_PLSTRMOFF;

      switch (ColIdx % 4) {
      case 0:
      case 1:
        TilePtr->TileType = XAIEGBL_TILE_TYPE_SHIMPL;
        break;
      default:
        TilePtr->TileType = XAIEGBL_TILE_TYPE_SHIMNOC;
        break;
      }

      TilePtr->IsReady = XAIE_COMPONENT_IS_READY;

      // XAie_print("Tile addr:%016lx, Row idx:%d, Col idx:%d, "
      // 	"Nocmodaddr:%016lx, Plmodaddr:%016lx\n",
      // 	TileAddr, 0U, ColIdx, TilePtr->NocModAddr,
      // 	TilePtr->PlModAddr);
    }
  //}
}

XAieGbl_Config *XAieGbl_LookupConfig(u16 DeviceId)
{
  XAieGbl_Config *CfgPtr = (XAieGbl_Config *)XAIE_NULL;
  u32 Index;

  for (Index=0U; Index < (u32)XPAR_AIE_NUM_INSTANCES; Index++) {
    if (xaie::XAieGbl_ConfigTable[Index].DeviceId == DeviceId) {
      CfgPtr = &xaie::XAieGbl_ConfigTable[Index];
      break;
    }
  }

  return (XAieGbl_Config *)CfgPtr;
}

} // namespace xaie

int xaie_lock_release(XAieGbl_Tile *tile, u32 lock_id, u32 val)
{
  XAieTile_LockRelease(tile, lock_id, val, 0);
  return 1;
}

int xaie_lock_acquire_nb(XAieGbl_Tile *tile, u32 lock_id, u32 val)
{
  u8 lock_ret = 0;
  u32 loop = 0;
  while ((!lock_ret) && (loop < 512)) {
    lock_ret = XAieTile_LockAcquire(tile, lock_id, val, 10000);
    loop++;
  }
  if (loop == 512) {
    xil_printf("Acquire [%d, %d, %d] value %d time-out\n\r", tile->ColId, tile->RowId, lock_id, val);
    return 0;
  }
  return 1;
}

void xaie_init()
{
  size_t aie_base = XAIE_ADDR_ARRAY_OFF << 14;
  XAieGbl_HwCfg AieConfig;                                /**< AIE HW configuration instance */
  XAIEGBL_HWCFG_SET_CONFIG((&AieConfig), XAIE_NUM_ROWS, XAIE_NUM_COLS, XAIE_ADDR_ARRAY_OFF);
  xaie::XAieGbl_HwInit(&AieConfig);
  xaie::AieConfigPtr = xaie::XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
  xaie::XAieGbl_CfgInitialize_Tile(0, &xaie::TileInst, 7, 2, xaie::AieConfigPtr);
}

} // namespace

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
  //xil_printf("handle agent dispatch pkt %x @ 0x%llx\n\r", pkt_idx, (size_t)pkt);

  xil_printf("handle dispatch packet, args: 0x%x 0x%x 0x%x 0x%x\n\r",
             pkt->arg[0], pkt->arg[1], pkt->arg[2], pkt->arg[3]);
  auto op = pkt->arg[0];
  if (op == 0x00beef00) {
    u32 lock_id = pkt->arg[1];
    u32 acqrel = pkt->arg[2];
    u32 val = pkt->arg[3];
    if (acqrel == 0)
      xaie_lock_acquire_nb(&xaie::TileInst, lock_id, val);
    else
      xaie_lock_release(&xaie::TileInst, lock_id, val);
  }
  // packet is in active phase
  packet_set_active(pkt, true);
}

} // namespace

int main()
{
  xil_printf("Hello, world!\n\r");
  init_platform();
  xaie_init();

  queue_t *q = nullptr;
  queue_create(MB_QUEUE_SIZE, &q);
  xil_printf("Created queue @ 0x%llx\n\r", (size_t)q);
  bool done = false;
  int cnt = 0;
  while (!done) {
    // if (!(cnt++ % 0x00100000))
    //   xil_printf("No Ding Dong 0x%llx\n\r", q->doorbell);

    if (q->doorbell+1 > q->last_doorbell) {
      xil_printf("Ding Dong 0x%llx\n\r", q->doorbell);

      q->last_doorbell = q->doorbell+1;

      auto rd_idx = queue_load_read_index(q);
      //xil_printf("Handle pkt read_index=%d\n\r", rd_idx);

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
