
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

struct HerdConfig {
  uint32_t row_start;
  uint32_t num_rows;
  uint32_t col_start;
  uint32_t num_cols;
};

HerdConfig HerdCfgInst;

namespace xaie {

XAieGbl_Config *AieConfigPtr;	                          /**< AIE configuration pointer */
//XAieGbl AieInst;	                                      /**< AIE global instance */
//XAieGbl_HwCfg AieConfig;                                /**< AIE HW configuration instance */
//XAieGbl_Tile TileInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];  /**< Instantiates AIE array of [XAIE_NUM_COLS] x [XAIE_NUM_ROWS] */
XAieGbl_Tile TileInst[4][4];   // Needs to be dynamic, and have a pool of these
XAieGbl_Tile ShimTileInst[XAIE_NUM_COLS];
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

int xaie_shim_dma_s2mm(XAieGbl_Tile *tile, int channel, uint64_t addr, uint32_t len)
{
  uint32_t shimDMAchannel = channel + XAIEDMA_SHIM_CHNUM_S2MM0;
  xil_printf("Shim S2MM start chanel %d\n\r", shimDMAchannel);

  XAieDma_Shim dma;
  XAieDma_ShimSoftInitialize(tile, &dma);  // We don't want to reset ...
  // Status print out for debug
  uint32_t s2mm_status = XAieGbl_Read32(tile->TileAddr + 0x0001D160);
  uint32_t mm2s_status = XAieGbl_Read32(tile->TileAddr + 0x0001D164);

  xil_printf("s2mm status : %08X\n\r", s2mm_status);
  xil_printf("mm2s status : %08X\n\r", mm2s_status);

  uint8_t start_bd = 8 + 4*channel;
  uint32_t outstanding = XAieDma_ShimPendingBdCount(&dma, shimDMAchannel);
  // If outstanding >=4, we're in trouble!!!!
  if (outstanding >=4) {
    xil_printf("\n\r *** BD OVERFLOW in s2mm channel %d *** \n\r",channel);
    while (XAieDma_ShimPendingBdCount(&dma, shimDMAchannel) > 3) {}
  }
  xil_printf("Outstanding pre : %d\n\r", outstanding);
  uint8_t bd = start_bd+outstanding;
  XAieDma_ShimBdSetAddr(&dma, bd, HIGH_ADDR((u64)addr), LOW_ADDR((u64)addr), len);
  XAieDma_ShimBdSetAxi(&dma, bd, 0, 4, 0, 0, XAIE_ENABLE);
  XAieDma_ShimBdWrite(&dma, bd);
  XAieDma_ShimSetStartBd(&dma, shimDMAchannel, bd);
  XAieDma_ShimChControl(&dma, shimDMAchannel, XAIE_DISABLE, XAIE_DISABLE, XAIE_ENABLE);

  outstanding = XAieDma_ShimPendingBdCount(&dma, shimDMAchannel);
  xil_printf("Outstanding post: %d\n\r", outstanding);
  //while (XAieDma_ShimPendingBdCount(&ShimDmaInst, channel)) {}
  xil_printf("s2mm bd pushed as bd %d\n\r",bd);
    // Status print out for debug
  s2mm_status = XAieGbl_Read32(tile->TileAddr + 0x0001D160);
  mm2s_status = XAieGbl_Read32(tile->TileAddr + 0x0001D164);

  xil_printf("s2mm status : %08X\n\r", s2mm_status);
  xil_printf("mm2s status : %08X\n\r", mm2s_status);

}

int xaie_shim_dma_mm2s(XAieGbl_Tile *tile, int channel, uint64_t addr, uint32_t len)
{
  uint32_t shimDMAchannel = channel + XAIEDMA_SHIM_CHNUM_MM2S0;
  xil_printf("Shim MM2S start channel %d\n\r", shimDMAchannel);

  XAieDma_Shim dma;
  XAieDma_ShimSoftInitialize(tile, &dma);  // We don't want to reset ...

  // Status print out for debug
  uint32_t s2mm_status = XAieGbl_Read32(tile->TileAddr + 0x0001D160);
  uint32_t mm2s_status = XAieGbl_Read32(tile->TileAddr + 0x0001D164);

  xil_printf("s2mm status : %08X\n\r", s2mm_status);
  xil_printf("mm2s status : %08X\n\r", mm2s_status);

  uint8_t start_bd = 0 + 4*channel;
  uint32_t outstanding = XAieDma_ShimPendingBdCount(&dma, shimDMAchannel);
  // If outstanding >=4, we're in trouble!!!!
  if (outstanding >=4) {
    xil_printf("\n\r *** BD OVERFLOW in mm2s channel %d *** \n\r",channel);
    while (XAieDma_ShimPendingBdCount(&dma, shimDMAchannel) > 3) {}
  }
  xil_printf("Outstanding pre : %d\n\r", outstanding);
  uint8_t bd = start_bd + outstanding;
  XAieDma_ShimBdSetAddr(&dma, bd, HIGH_ADDR(addr), LOW_ADDR(addr), len);
  XAieDma_ShimBdSetAxi(&dma, bd, 0, 4, 0, 0, XAIE_ENABLE);
  XAieDma_ShimBdWrite(&dma, bd);
  XAieDma_ShimSetStartBd(&dma, shimDMAchannel, bd);
  XAieDma_ShimChControl(&dma, shimDMAchannel, XAIE_DISABLE, XAIE_DISABLE, XAIE_ENABLE);

  outstanding = XAieDma_ShimPendingBdCount(&dma, shimDMAchannel);
  xil_printf("Outstanding post: %d\n\r", outstanding);

  //while (XAieDma_ShimPendingBdCount(&ShimDmaInst, channel)) {}
  xil_printf("mm2s bd pushed as bd %d\n\r",bd);

  // Status print out for debug
  s2mm_status = XAieGbl_Read32(tile->TileAddr + 0x0001D160);
  mm2s_status = XAieGbl_Read32(tile->TileAddr + 0x0001D164);

  xil_printf("s2mm status : %08X\n\r", s2mm_status);
  xil_printf("mm2s status : %08X\n\r", mm2s_status);


}

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

// Initialize the structures for the shim DMA at the bottom of the device
void xaie_device_init(int num_cols)
{
  size_t aie_base = XAIE_ADDR_ARRAY_OFF << 14;
  XAieGbl_HwCfg AieConfig;                                /**< AIE HW configuration instance */
  XAIEGBL_HWCFG_SET_CONFIG((&AieConfig), XAIE_NUM_ROWS, XAIE_NUM_COLS, XAIE_ADDR_ARRAY_OFF);
  xaie::XAieGbl_HwInit(&AieConfig);
  xaie::AieConfigPtr = xaie::XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);

  for (int col=0; col<num_cols; col++) {
      xil_printf("init physical dma col %d\n\r", col);
      xaie::XAieGbl_CfgInitialize_Tile(0, &xaie::ShimTileInst[col],
                                       col, 0, xaie::AieConfigPtr);
      // I'd love to do this, but it overflows the program memory, because, yeah
      /*
      XAieDma_Shim dma;
      XAieDma_ShimInitialize(&xaie::ShimTileInst[col], &dma);  // Reset!!!
      */
    // Invalidate all BDs by writing to their buffer control register
    for (int ch=0;ch<4;ch++) {
      XAieGbl_Write32(xaie::ShimTileInst[col].TileAddr + 0x0001D40 + 0x8*ch, 0x00); // Disable all channels
    }
    for (int bd=0;bd<16;bd++) {
      XAieGbl_Write32(xaie::ShimTileInst[col].TileAddr + 0x0001D008 + 0x15*bd, 0);
    }
    // Enable all 4 channels
    for (int ch=0;ch<4;ch++) {
      XAieGbl_Write32(xaie::ShimTileInst[col].TileAddr + 0x0001D40 + 0x8*ch, 0x01); // Enable channel
    }

      //XAieDma_ShimInitialize(&xaie::ShimTileInst[col], &xaie::ShimDMAInst[col]);  // We might want to reset ...
  }
}

// Initialize one herd with lower left corner at (col_start, row_start)
void xaie_herd_init(int col_start, int num_cols, int row_start, int num_rows)
{
  size_t aie_base = XAIE_ADDR_ARRAY_OFF << 14;
  XAieGbl_HwCfg AieConfig;                                /**< AIE HW configuration instance */
  XAIEGBL_HWCFG_SET_CONFIG((&AieConfig), XAIE_NUM_ROWS, XAIE_NUM_COLS, XAIE_ADDR_ARRAY_OFF);
  xaie::XAieGbl_HwInit(&AieConfig);
  xaie::AieConfigPtr = xaie::XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);

  HerdCfgInst.col_start = col_start;
  HerdCfgInst.num_cols = num_cols;
  HerdCfgInst.row_start = row_start;
  HerdCfgInst.num_rows = num_rows;

  for (int col=0; col<num_cols; col++) {
    for (int row=0; row<num_rows; row++) {
      xil_printf("init physical col %d, row %d as herd col %d row %d\n\r", col+col_start, row+row_start, col, row);
      xaie::XAieGbl_CfgInitialize_Tile(0, &xaie::TileInst[col][row],
                                       col+col_start, row+row_start, xaie::AieConfigPtr);
    }
  }
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
  pkt->type = HSA_PACKET_TYPE_INVALID;
  signal_subtract_acq_rel((signal_t*)&pkt->completion_signal, 1);
}

void handle_packet_device_initialize(dispatch_packet_t *pkt) {
  packet_set_active(pkt, true);
  // Address mode here is absolute range
  if (((pkt->arg[0] >> 48) & 0xf) == AIR_ADDRESS_ABSOLUTE_RANGE) {
    u32 num_cols  = (pkt->arg[0] >> 40) & 0xff;
    xaie_device_init(num_cols);
    xil_printf("Initialized shim DMA of size %d\r\n",num_cols);
    // herd_id is ignored - current restriction is 1 herd -> 1 controller
  }
  else {
    xil_printf("Unsupported address type 0x%04X for device initialize\r\n",(pkt->arg[0] >> 48) & 0xf);
  }
}

void handle_packet_herd_initialize(dispatch_packet_t *pkt) {
  packet_set_active(pkt, true);
  // Address mode here is absolute range
  if (((pkt->arg[0] >> 48) & 0xf) == AIR_ADDRESS_ABSOLUTE_RANGE) {
    u32 start_row = (pkt->arg[0] >> 16) & 0xff;
    u32 num_rows  = (pkt->arg[0] >> 24) & 0xff;
    u32 start_col = (pkt->arg[0] >> 32) & 0xff;
    u32 num_cols  = (pkt->arg[0] >> 40) & 0xff;
  
    u32 herd_id = pkt->arg[1] & 0xffff;
    xaie_herd_init(start_col, num_cols, start_row, num_rows);
    xil_printf("Initialized herd %d at (%d, %d) of size (%d,%d)\r\n",herd_id, start_col, start_row, num_cols, num_rows);
    // herd_id is ignored - current restriction is 1 herd -> 1 controller
  }
  else {
    xil_printf("Unsupported address type 0x%04X for herd initialize\r\n",(pkt->arg[0] >> 48) & 0xf);
  }
}


void handle_packet_xaie_lock(dispatch_packet_t *pkt)
{
  // packet is in active phase
  packet_set_active(pkt, true);

  u32 num_cols = (((pkt->arg[0] >> 48) & 0xf) == AIR_ADDRESS_HERD_RELATIVE_RANGE) ? ((pkt->arg[0] >> 40) & 0xff) : 1;
  u32 num_rows = (((pkt->arg[0] >> 48) & 0xf) == AIR_ADDRESS_HERD_RELATIVE_RANGE) ? ((pkt->arg[0] >> 24) & 0xff) : 1;
  u32 start_col = (pkt->arg[0] >> 32) & 0xff;
  u32 start_row = (pkt->arg[0] >> 16) & 0xff;
  u32 lock_id = pkt->arg[1];
  u32 acqrel = pkt->arg[2];
  u32 val = pkt->arg[3];
  for (u32 col = 0; col < num_cols; col++) {
    for (u32 row = 0; row < num_rows; row++) {
      if (acqrel == 0)
	xaie_lock_acquire_nb(&xaie::TileInst[start_col+col][start_row+row], lock_id, val);
      else
	xaie_lock_release(&xaie::TileInst[start_col+col][start_row+row], lock_id, val);
    }
  }
}


void handle_packet_put_stream(dispatch_packet_t *pkt)
{
  // packet is in active phase
  packet_set_active(pkt, true);

  uint64_t which_stream = pkt->arg[1];
  uint64_t data = pkt->arg[2];

  register uint32_t d0 = data & 0xffffffff;
  register uint32_t d1 = data >> 32;

  switch (which_stream) {
  case 0:
    putfsl(d0, 0);
    cputfsl(d1, 0);
    break;
  case 1:
    putfsl(d0, 1);
    cputfsl(d1, 1);
    break;
  case 2:
    putfsl(d0, 2);
    cputfsl(d1, 2);
    break;
  case 3:
    putfsl(d0, 3);
    cputfsl(d1, 3);
    break;
  default:
    break;
  }
}

void handle_packet_get_stream(dispatch_packet_t *pkt)
{
  // packet is in active phase
  packet_set_active(pkt, true);

  uint64_t which_stream = pkt->arg[1];
  register uint32_t d;

  switch (which_stream) {
  case 0:
    getfsl_interruptible(d, 0);
    break;
  case 1:
    getfsl_interruptible(d, 1);
    break;
  case 2:
    getfsl_interruptible(d, 2);
    break;
  case 3:
    getfsl_interruptible(d, 3);
    break;
  default:
    break;
  }

  // BUG
  pkt->return_address = d;

}


void handle_packet_hello(dispatch_packet_t *pkt)
{
  // packet is in active phase
  packet_set_active(pkt, true);

  uint64_t say_what = pkt->arg[1];
  xil_printf("HELLO %08X\n\r",(uint32_t)say_what);
}

void handle_packet_shim_memcpy(dispatch_packet_t *pkt)
{
  xil_printf("handle_packet_shim_memory\n\r");
  packet_set_active(pkt, true);

  uint16_t col = (pkt->arg[0] >> 32) & 0xffff;
  uint16_t flags = (pkt->arg[0] >> 48) & 0xffff;
  bool start = flags & 0x1;
  uint32_t burst_len = pkt->arg[1] & 0xffffffff;
  uint16_t direction = (pkt->arg[1] >> 32) & 0xffff;
  uint16_t channel = (pkt->arg[1] >> 48) & 0xffff;
  uint64_t paddr = pkt->arg[2];
  uint64_t bytes = pkt->arg[3];

  //XAieGbl_Tile tile;
  //xaie::XAieGbl_CfgInitialize_Tile(0, &(aie::TileInst[0][0]), col, row, xaie::AieConfigPtr);

  xil_printf("shim_memcpy: col %d direction %d channel %d paddr %llx bytes %d\n\r",col, direction, channel, paddr, bytes);

  if (direction == 0)
    xaie_shim_dma_s2mm(&xaie::ShimTileInst[col], channel, paddr, bytes);
  else
    xaie_shim_dma_mm2s(&xaie::ShimTileInst[col], channel, paddr, bytes);
}

} // namespace

struct dma_cmd_t {
  uint8_t select;
  uint16_t length;
  uint16_t uram_addr;
  uint8_t id;
};

struct dma_rsp_t {
	uint8_t id;
};

void put_dma_cmd(dma_cmd_t *cmd, int stream)
{
  static dispatch_packet_t pkt;

  pkt.arg[1] = stream;
  pkt.arg[2] = 0;
  pkt.arg[2] |= ((uint64_t)cmd->select) << 32;
  pkt.arg[2] |= cmd->length << 18;
  pkt.arg[2] |= cmd->uram_addr << 5;
  pkt.arg[2] |= cmd->id;

  handle_packet_put_stream(&pkt);
}

void get_dma_rsp(dma_rsp_t *rsp, int stream)
{
  static dispatch_packet_t pkt;
  pkt.arg[1] = stream;
  handle_packet_get_stream(&pkt);
  rsp->id = pkt.return_address;
}

// void test_stream()
// {
//   xil_printf("Test stream..");
//   static dma_cmd_t cmd;

//   cmd.select = 2;
//   cmd.length = 1;
//   cmd.uram_addr = 0;
//   cmd.id = 3;

//   put_dma_cmd(&cmd, 0);

//   xil_printf("..");

//   static dma_rsp_t rsp;
//   rsp.id = -1;
//   get_dma_rsp(&rsp, 0);


//   if (rsp.id == cmd.id)
//     xil_printf("PASS!\n\r");
//   else
//     xil_printf("fail, cmd=%d, rsp=%d\n\r", cmd.id, rsp.id);
// }

void handle_agent_dispatch_packet(dispatch_packet_t *pkt)
{
  // get the index
  uint32_t pkt_idx;
  memcpy(&pkt_idx, &pkt, sizeof(pkt_idx));
  pkt_idx = ((pkt_idx & 0x3fff) >> 6) - 2;
  //xil_printf("handle agent dispatch pkt %x @ 0x%llx\n\r", pkt_idx, (size_t)pkt);

  xil_printf("handle dispatch packet, args: 0x%llx 0x%llx 0x%llx 0x%llx\n\r",pkt->arg[0], pkt->arg[1], pkt->arg[2], pkt->arg[3]);
  auto op = pkt->arg[0] & 0xffff;
  xil_printf("Op is %04X\n\r",op);
  switch (op) {
    case AIR_PKT_TYPE_INVALID:
    default:
      break;

    case AIR_PKT_TYPE_DEVICE_INITIALIZE:
      handle_packet_device_initialize(pkt);
      break;
    case AIR_PKT_TYPE_HERD_INITIALIZE:
      handle_packet_herd_initialize(pkt);
      break;

    // case AIR_PKT_TYPE_READ_MEMORY_32:
    //   handle_packet_read_memory(pkt);
    //   break;
    // case AIR_PKT_TYPE_WRITE_MEMORY_32:
    //   handle_packet_write_memory(pkt);
    //   break;

    // case AIR_PKT_TYPE_MEMCPY:
    //   handle_packet_memcpy(pkt);
    //   break;

    case AIR_PKT_TYPE_HELLO:
      handle_packet_hello(pkt);
      break;
    case AIR_PKT_TYPE_PUT_STREAM:
      handle_packet_put_stream(pkt);
      break;
    case AIR_PKT_TYPE_GET_STREAM:
      handle_packet_get_stream(pkt);
      break;

    case AIR_PKT_TYPE_XAIE_LOCK:
      handle_packet_xaie_lock(pkt);
      break;

    case AIR_PKT_TYPE_SHIM_DMA_MEMCPY:
      handle_packet_shim_memcpy(pkt);
      break;
  }

}

int main()
{
  xil_printf("\n\nMB firmware created on %s at %s GMT\n\r",__DATE__, __TIME__); 
  init_platform();

  //test_stream();

  queue_t *q = nullptr;
  queue_create(MB_QUEUE_SIZE, &q);
  xil_printf("Created queue @ 0x%llx\n\r", (size_t)q);
  bool done = false;
  int cnt = 0;
  while (!done) {
    // if (!(cnt++ % 0x00100000))
    //   xil_printf("No Ding Dong 0x%llx\n\r", q->doorbell);

    if (q->doorbell+1 > q->last_doorbell) {
      xil_printf("Ding Dong 0x%llx\n\r", q->doorbell+1);

      q->last_doorbell = q->doorbell+1;

      // process packets until we hit an invalid packet
      bool invalid = false;
      while (!invalid) {
        uint64_t rd_idx = queue_load_read_index(q);
        xil_printf("Handle pkt read_index=%d\n\r", rd_idx);

        dispatch_packet_t *pkt = &((dispatch_packet_t*)q->base_address)[rd_idx % q->size];
        switch (pkt->type) {
          default:
          case HSA_PACKET_TYPE_INVALID:
            invalid = true;
            break;
          case HSA_PACKET_TYPE_AGENT_DISPATCH:
            handle_agent_dispatch_packet(pkt);
            complete_agent_dispatch_packet(pkt);
            queue_add_read_index(q, 1);
            break;
        }
      }
    }
  }

  cleanup_platform();
  return 0;
}
