#include <cstdint>
#include <cstring>

extern "C" {
#include "xil_printf.h"
#include "pvr.h"

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

#define LOGICAL_HERD_DMAS      16

#define SHIM_DMA_S2MM 0 
#define SHIM_DMA_MM2S 1 

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
uint16_t logicalToPhysicalS2MMshimDMAcolMap[LOGICAL_HERD_DMAS];                 // The hardwired 16 will never bite us
uint16_t logicalToPhysicalS2MMshimDMAchannelMap[LOGICAL_HERD_DMAS];             // The hardwired 16 will never bite us
uint16_t logicalToPhysicalMM2SshimDMAcolMap[LOGICAL_HERD_DMAS];                 // The hardwired 16 will never bite us
uint16_t logicalToPhysicalMM2SshimDMAchannelMap[LOGICAL_HERD_DMAS];             // The hardwired 16 will never bite us

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

int xaie_shim_dma_push_bd(XAieGbl_Tile *tile, int direction, int channel, uint64_t addr, uint32_t len)
{
  uint32_t shimDMAchannel = channel;
  uint32_t status_register_offset;
  uint32_t status_mask_shift;
  uint32_t control_register_offset;
  uint32_t start_queue_register_offset;
  uint32_t start_queue_size_mask_shift;

  if (direction == SHIM_DMA_S2MM) {
    shimDMAchannel += XAIEDMA_SHIM_CHNUM_S2MM0;
    xil_printf("\n\r  S2MM Shim DMA start channel %d\n\r", shimDMAchannel);
    status_register_offset = 0x1d160;
    if (channel == 0) {
      status_mask_shift = 0;
      control_register_offset = 0x1d140;
      start_queue_register_offset = 0x1d144;
      start_queue_size_mask_shift = 6;
    }
    else {
      status_mask_shift = 2;
      control_register_offset = 0x1d148;
      start_queue_register_offset = 0x1d14c;
      start_queue_size_mask_shift = 9;

    }
  }
  else {
    shimDMAchannel += XAIEDMA_SHIM_CHNUM_MM2S0;
    xil_printf("\n\r  MM2S Shim DMA start channel %d\n\r", shimDMAchannel);
    status_register_offset = 0x1d164;
    if (channel == 0) {
      status_mask_shift = 0;
      control_register_offset = 0x1d150;
      start_queue_register_offset = 0x1d154;
      start_queue_size_mask_shift = 6;
    }
    else {
      status_mask_shift = 2;
      control_register_offset = 0x1d158;
      start_queue_register_offset = 0x1d15c;
      start_queue_size_mask_shift = 9;
    }
  }

  uint32_t s2mm_status = XAieGbl_Read32(tile->TileAddr + 0x0001D160);
  uint32_t mm2s_status = XAieGbl_Read32(tile->TileAddr + 0x0001D164);
  uint32_t s2mm0_ctrl  = XAieGbl_Read32(tile->TileAddr + 0x0001D140);
  uint32_t mm2s0_ctrl  = XAieGbl_Read32(tile->TileAddr + 0x0001D150);

  xil_printf("s2mm status : %08X\n\r", s2mm_status);
  xil_printf("mm2s status : %08X\n\r", mm2s_status);
  xil_printf("s2mm0 ctrl  : %08X\n\r", s2mm0_ctrl);
  xil_printf("mm2s0 ctrl  : %08X\n\r", mm2s0_ctrl);

  uint32_t start_bd = 4*shimDMAchannel;
  uint32_t outstanding = (XAieGbl_Read32(tile->TileAddr + status_register_offset) >> start_queue_size_mask_shift) & 0b111;
  // If outstanding >=4, we're in trouble!!!!
  if (outstanding >=4) {
    xil_printf("\n\r *** BD OVERFLOW in shimDMA channel %d *** \n\r",shimDMAchannel);
    bool waiting = true;
    while (waiting) {
      outstanding = (XAieGbl_Read32(tile->TileAddr + status_register_offset) >> start_queue_size_mask_shift) & 0b111;
      waiting = (outstanding > 3);
    }
  }
  xil_printf("Outstanding pre : %d\n\r", outstanding);
  uint32_t bd = start_bd+outstanding + 0; //hack
  XAieGbl_Write32(tile->TileAddr + 0x0001D008+(bd*0x14), 0x0);           // Mark the BD as invalid
  uint32_t chk = XAieGbl_Read32(tile->TileAddr + 0x0001D008+(bd*0x14));
  xil_printf("bd %d %08X : %08X\n\r",bd, 0x0001D008+(bd*0x14), chk);
  xil_printf("bd %d HAD : %08X\n\r",bd, HIGH_ADDR((u64)addr));
  xil_printf("bd %d LAD : %08X\n\r",bd, LOW_ADDR((u64)addr));
  xil_printf("bd %d LEN : %08X\n\r",bd, len);

  // Set the registers directly ...
  uint32_t base_address =  0x1d000 + bd * 0x14;
  XAieGbl_Write32(tile->TileAddr + base_address + 0x00, LOW_ADDR((u64)addr));
  XAieGbl_Write32(tile->TileAddr + base_address + 0x04, len >> 2); // We pass in bytes, but the shim DMA can ony deal with 32 bits
  u32 control = (HIGH_ADDR((u64)addr) << 16) | 1;
  XAieGbl_Write32(tile->TileAddr + base_address + 0x08, control);
  XAieGbl_Write32(tile->TileAddr + base_address + 0x0C, 0x10); // Secure bit
  XAieGbl_Write32(tile->TileAddr + base_address + 0x10, 0x0);


  // Check if the channel is running or not
  uint32_t precheck_status = (XAieGbl_Read32(tile->TileAddr + status_register_offset) >> status_mask_shift) & 0b11;
  if (precheck_status == 0b00) {
    XAieGbl_Write32(tile->TileAddr + control_register_offset, 0xb001); // Stream traffic can run, we can issue AXI-MM, and the channel is enabled
  }
  // Now push into the queue
  XAieGbl_Write32(tile->TileAddr + start_queue_register_offset, bd);

  for (int i=0;i<0x14;i+=4) {
    uint32_t rb = XAieGbl_Read32(tile->TileAddr + 0x0001D000+(bd*0x14)+i);
    xil_printf("bd %d %08X : %08X\n\r",bd, 0x0001D000+(bd*0x14)+i, rb);
  }

  outstanding = (XAieGbl_Read32(tile->TileAddr + status_register_offset) >> start_queue_size_mask_shift) & 0b111;
  xil_printf("Outstanding post: %d\n\r", outstanding);
  xil_printf("bd pushed as bd %d\n\r",bd);
    // Status print out for debug
  s2mm_status = XAieGbl_Read32(tile->TileAddr + 0x0001D160);
  mm2s_status = XAieGbl_Read32(tile->TileAddr + 0x0001D164);
  s2mm0_ctrl  = XAieGbl_Read32(tile->TileAddr + 0x0001D140);
  mm2s0_ctrl  = XAieGbl_Read32(tile->TileAddr + 0x0001D150);

  xil_printf("s2mm status : %08X\n\r", s2mm_status);
  xil_printf("mm2s status : %08X\n\r", mm2s_status);
  xil_printf("s2mm0 ctrl  : %08X\n\r", s2mm0_ctrl);
  xil_printf("mm2s0 ctrl  : %08X\n\r", mm2s0_ctrl);

  if (direction == SHIM_DMA_S2MM) {
    xil_printf("  End of S2MM Shim DMA start channel %d\n\r", shimDMAchannel);
  }
  else {
    xil_printf("  End of MM2S Shim DMA start channel %d\n\r", shimDMAchannel);
  }
  return 1;
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
      //xil_printf("init physical dma col %d\n\r", col);
      xaie::XAieGbl_CfgInitialize_Tile(0, &xaie::ShimTileInst[col],
                                       col, 0, xaie::AieConfigPtr);
      // I'd love to do this, but it overflows the program memory, because, yeah
      /*
      XAieDma_Shim dma;
      XAieDma_ShimInitialize(&xaie::ShimTileInst[col], &dma);  // Reset!!!
      */
    // Invalidate all BDs by writing to their buffer control register
    for (int ch=0;ch<4;ch++) {
      XAieGbl_Write32(xaie::ShimTileInst[col].TileAddr + 0x0001D140 + 0x8*ch, 0x00); // Disable all channels
    }
    for (int bd=0;bd<16;bd++) {
      XAieGbl_Write32(xaie::ShimTileInst[col].TileAddr + 0x0001D008 + 0x14*bd, 0);
    }
    /*
    // Enable all 4 channels
    for (int ch=0;ch<4;ch++) {
      XAieGbl_Write32(xaie::ShimTileInst[col].TileAddr + 0x0001D140 + 0x8*ch, 0x01); // Enable channel
    }
    */
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

uint64_t shmem_base = 0x020100000000UL;
uint64_t uart_lock_base = shmem_base + 0x20;
uint64_t base_address;

u32 last_before_hi, last_before_lo, last_after_hi, last_after_lo;
u32 phase;
bool setup;
u32 dispatch_before_lo, call_before_lo, invalidate_before_lo;

void lock_uart(uint32_t id) {
  bool is_locked = false;

  while (!is_locked) {
    uint32_t status = *(uint32_t *)uart_lock_base;
    if (status != 1) {
      *(uint32_t *)(uart_lock_base + 0x4) = id;
      *(uint32_t *)uart_lock_base = 1;
      // See if they stuck
      uint32_t status = *(uint32_t *)uart_lock_base;
      uint32_t lockee = *(uint32_t *)(uart_lock_base + 0x04);
      if ((status == 1) && (lockee == id)) {
        xil_printf("MB %02d: ", id);
        is_locked = true;
      }
    }
  }
}

  // This looks unsafe, but its okay as long as we always aquire
  // the lock first
void unlock_uart() {
  *(uint32_t *)(uart_lock_base + 0x4) = 0;
  *(uint32_t *)uart_lock_base = 0;
}

int queue_create(uint32_t size, queue_t **queue, uint32_t mb_id)
{
  uint64_t queue_address[1] = {base_address + sizeof(dispatch_packet_t)};
  uint64_t queue_base_address[1] = {queue_address[0] + sizeof(dispatch_packet_t)};
  lock_uart(mb_id);
  xil_printf("setup_queue 0x%llx, %x bytes + %d 64 byte packets\n\r", (void *)queue_address, sizeof(queue_t), size);
  xil_printf("base address 0x%llx\n\r", base_address);
  unlock_uart();

  // The address of the queue_t is stored @ shmem_base[mb_id]
  memcpy((void*)(((uint64_t*)shmem_base)+mb_id), (void*)queue_address, sizeof(uint64_t));

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
  phase = 0;
  setup = true;
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

void handle_packet_get_capabilities(dispatch_packet_t *pkt, uint32_t mb_id)
{
  // packet is in active phase
  packet_set_active(pkt, true);
  uint64_t *addr = (uint64_t *)(pkt->arg[1]);

  lock_uart(mb_id); xil_printf("Writing to 0x%llx\n\r",(uint64_t)addr); unlock_uart();
  // We now write a capabilities structure to the address we were just passed
  // We've already done this once - should we just cache the results?
  pvr_t pvr;
  microblaze_get_pvr(&pvr);
  int user1 = MICROBLAZE_PVR_USER1(pvr);
  int user2 = MICROBLAZE_PVR_USER2(pvr);
  //int mb_id = user2 & 0xff;    // We'll use the one passed in
  int maj   = (user2 >> 24) & 0xff;
  int min   = (user2 >> 16) & 0xff;
  int ver   = (user2 >> 8) & 0xff;

  addr[0] = (uint64_t)mb_id;           // region id
  addr[1] = (uint64_t)user1;           // num regions
  addr[2] = (uint64_t)(user2 >> 8);    // region controller firmware version
  addr[3] = 16L;                       // cores per region
  addr[4] = 32768L;                    // Total L1 data memory per core
  addr[5] = 8L;                        // Number of L1 data memory banks
  addr[6] = 16384L;                    // L1 program memory per core
  addr[7] = 0L;                        // L2 data memory per region
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


void handle_packet_hello(dispatch_packet_t *pkt, uint32_t mb_id)
{
  // packet is in active phase
  packet_set_active(pkt, true);

  uint64_t say_what = pkt->arg[1];
  lock_uart(mb_id); xil_printf("HELLO %08X\n\r",(uint32_t)say_what); unlock_uart();
  /*
  //u32 before_hi = XAieGbl_Read32(xaie::TileInst[0][0].TileAddr + 0x000340FC); // Timer high
  u32 before_lo = XAieGbl_Read32(xaie::TileInst[0][0].TileAddr + 0x000340F8); // Timer low
  //u32 after_hi  = XAieGbl_Read32(xaie::TileInst[0][0].TileAddr + 0x000340FC); // Timer high
  u32 after_lo  = XAieGbl_Read32(xaie::TileInst[0][0].TileAddr + 0x000340F8); // Timer low

  if (phase == 1) {
    xil_printf("last before %08X\n\r",last_before_lo);
    xil_printf("last after  %08X\n\r",last_after_lo);
    xil_printf("this before %08X\n\r",before_lo);
    xil_printf("this after  %08X\n\r",after_lo);
    xil_printf("turnaround time %d\n\r",before_lo - last_after_lo);
    phase = 0;
  }
  else {
    last_before_lo = before_lo;
    last_after_lo = after_lo;
    phase = 1;
  }
  setup= true;
  */
}

void handle_packet_allocate_herd_shim_dmas(dispatch_packet_t *pkt)
{
  xil_printf("handle_packet_allocate_herd_shim_dmas\n\r");
  packet_set_active(pkt, true);

  uint16_t physical_col   = (pkt->arg[0] >> 32) & 0x00ff;
  uint16_t logical_col    = (pkt->arg[0] >> 40) & 0x00ff;
  uint32_t herd_id        =  pkt->arg[1]        & 0x00ff;
  uint16_t channel   =       pkt->arg[2]        & 0x00ff;
  uint16_t direction =       pkt->arg[3]        & 0x00ff;

  if (direction == 0) {
    xaie::logicalToPhysicalS2MMshimDMAcolMap[logical_col] = physical_col;
    xaie::logicalToPhysicalS2MMshimDMAchannelMap[logical_col] = channel;
  }
  else {
    xaie::logicalToPhysicalMM2SshimDMAcolMap[logical_col] = physical_col;
    xaie::logicalToPhysicalMM2SshimDMAchannelMap[logical_col] = channel;
  }
  xil_printf("shim_dma_allocate: herd %d allocated physical shim dma in col %d channel %d as logical dma %d for direction %d\n\r",herd_id, physical_col, channel, logical_col, direction);
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
  xaie_shim_dma_push_bd(&xaie::ShimTileInst[col], direction, channel, paddr, bytes);
}

void handle_packet_herd_shim_memcpy(dispatch_packet_t *pkt)
{
  xil_printf("handle_packet_herd_shim_memory\n\r");
  packet_set_active(pkt, true);

  uint16_t col = (pkt->arg[0] >> 32) & 0xffff;
  uint16_t herd_id = (pkt->arg[0] >> 52) & 0xff;
  uint32_t burst_len = pkt->arg[1] & 0xffffffff;
  uint16_t direction = (pkt->arg[1] >> 32) & 0xffff;
  uint64_t paddr = pkt->arg[2];
  uint64_t bytes = pkt->arg[3];

  //XAieGbl_Tile tile;
  //xaie::XAieGbl_CfgInitialize_Tile(0, &(aie::TileInst[0][0]), col, row, xaie::AieConfigPtr);

  //xil_printf("herd_shim_memcpy: herd_d relative col %d direction %d paddr %llx bytes %d\n\r",col, direction, paddr, bytes);
  uint16_t mapped_col = (direction ==0) ? xaie::logicalToPhysicalS2MMshimDMAcolMap[col] : xaie::logicalToPhysicalMM2SshimDMAcolMap[col];  // todo: a 2d array
  uint16_t mapped_channel = (direction == 0) ? xaie::logicalToPhysicalS2MMshimDMAchannelMap[col] : xaie::logicalToPhysicalMM2SshimDMAchannelMap[col];
  //xil_printf("Unmaps to col %d chan %d\n\r",mapped_col, mapped_channel);
  //xil_printf("Somethings a happenin over in col %d\n\r", mapped_col);
/*
  for (int off=0x1D160; off <= 0x1d164; off += 4) {
    xil_printf("Offset 0x%05X: %08X\n\r",off, XAieGbl_Read32(xaie::ShimTileInst[mapped_col].TileAddr + off));
  }
  */
  xaie_shim_dma_push_bd(&xaie::ShimTileInst[mapped_col], direction, mapped_channel, paddr, bytes);
}

void handle_packet_herd_shim_1d_strided_memcpy(dispatch_packet_t *pkt)
{
  xil_printf("handle_packet_herd_shim_1d_strided_memcpy\n\r");
  packet_set_active(pkt, true);

  uint16_t start_col = (pkt->arg[0] >> 32) & 0x00ff;
  uint16_t num_cols = (pkt->arg[0] >> 40) & 0x00ff;
  uint16_t herd_id = (pkt->arg[0] >> 52) & 0xff;
  uint32_t burst_len = pkt->arg[1] & 0xffffffff;
  uint16_t direction = (pkt->arg[1] >> 32) & 0xffff;
  uint64_t paddr = pkt->arg[2];
  uint64_t bytes = pkt->arg[3] & 0xffffffff;
  uint64_t stride = (pkt->arg[3]>>32) & 0xffffffff;

  //XAieGbl_Tile tile;
  //xaie::XAieGbl_CfgInitialize_Tile(0, &(aie::TileInst[0][0]), col, row, xaie::AieConfigPtr);

  xil_printf("herd_shim_1d_strided_memcpy: herd direction %d paddr %llx bytes %d stride %d\n\r",direction, paddr, bytes, stride);

  for (int col=start_col; col<(start_col+num_cols); col++) {
    uint16_t mapped_col = (direction ==0) ? xaie::logicalToPhysicalS2MMshimDMAcolMap[col] : xaie::logicalToPhysicalMM2SshimDMAcolMap[col];  // todo: a 2d array
    uint16_t mapped_channel = (direction == 0) ? xaie::logicalToPhysicalS2MMshimDMAchannelMap[col] : xaie::logicalToPhysicalMM2SshimDMAchannelMap[col];
    xil_printf("Unmaps to col %d chan %d\n\r",mapped_col, mapped_channel);
  xil_printf("Somethings a happenin over in col %d\n\r", mapped_col);

  for (int off=0x1D160; off <= 0x1d164; off += 4) {
    xil_printf("Offset 0x%05X: %08X\n\r",off, XAieGbl_Read32(xaie::ShimTileInst[mapped_col].TileAddr + off));
  }
    xaie_shim_dma_push_bd(&xaie::ShimTileInst[mapped_col], direction, mapped_channel, paddr, bytes);
    paddr += stride; // Because multiplication is nothing more than repeated addition ...
  }
}

void handle_packet_nd_memcpy(dispatch_packet_t *pkt)
{
  xil_printf("handle_packet_nd_memcpy\n\r");
  packet_set_active(pkt, true);

  uint16_t memory_space = (pkt->arg[0] >> 16) & 0xffff;
  uint16_t channel      = (pkt->arg[0] >> 24) & 0x00ff;
  uint16_t col          = (pkt->arg[0] >> 32) & 0x00ff;
  uint16_t direction    = (pkt->arg[0] >> 60) & 0x000f;
  uint32_t burst_len    = (pkt->arg[0] >> 52) & 0x00ff;
  uint64_t paddr        =  pkt->arg[1];
  uint32_t length_1d    = (pkt->arg[2] >>  0) & 0xffffffff;
  uint32_t length_2d    = (pkt->arg[2] >> 32) & 0x0000ffff;
  uint32_t stride_2d    = (pkt->arg[2] >> 48) & 0x0000ffff;
  uint32_t length_3d    = (pkt->arg[3] >>  0) & 0x0000ffff;
  uint32_t stride_3d    = (pkt->arg[3] >> 16) & 0x0000ffff;
  uint32_t length_4d    = (pkt->arg[3] >> 32) & 0x0000ffff;
  uint32_t stride_4d    = (pkt->arg[3] >> 48) & 0x0000ffff;

  if (memory_space == 2) {
    // This is the shim DMA
    xil_printf("shim DMA %d dir %d chan %d paddr %llx 4d %d stride %d length 3d %d stride %d length, 2d %d stride %d length, 1d %d length\n\r",col, direction, channel, paddr, stride_4d, length_4d,
       stride_3d, length_3d, stride_2d, length_2d, length_1d);
    uint64_t paddr_4d = paddr;
    uint64_t paddr_3d = paddr;
    uint64_t paddr_2d = paddr;
    uint64_t paddr_1d = paddr;
    for (uint32_t index_4d=0;index_4d<length_4d;index_4d++) {
      paddr_2d = paddr_3d;
      for (uint32_t index_3d=0;index_3d<length_3d;index_3d++) {
        paddr_1d = paddr_2d;
        for (uint32_t index_2d=0;index_2d<length_2d;index_2d++) {
          xil_printf("shim DMA %d %d [%d][%d][%d] paddr %llx \n\r",direction, channel, index_4d, index_3d, index_2d, paddr_1d);
          xaie_shim_dma_push_bd(&xaie::ShimTileInst[col], direction, channel, paddr_1d, length_1d);
          paddr_1d += stride_2d;
        }
        paddr_2d += stride_3d;
      }
      paddr_3d += stride_4d;
    }
  }
  else {
    xil_printf("NOT SUPPORTED: Cannot program memory space %d DMAs\n\r",memory_space);
  }
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

void handle_agent_dispatch_packet(dispatch_packet_t *pkt, uint32_t mb_id)
{
  // get the index
  uint32_t pkt_idx;
  memcpy(&pkt_idx, &pkt, sizeof(pkt_idx));
  pkt_idx = ((pkt_idx & 0x3fff) >> 6) - 2;
  //xil_printf("handle agent dispatch pkt %x @ 0x%llx\n\r", pkt_idx, (size_t)pkt);

  //xil_printf("handle dispatch packet, args: 0x%llx 0x%llx 0x%llx 0x%llx\n\r",pkt->arg[0], pkt->arg[1], pkt->arg[2], pkt->arg[3]);
  auto op = pkt->arg[0] & 0xffff;
  //xil_printf("Op is %04X\n\r",op);
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
      handle_packet_hello(pkt, mb_id);
      break;
    case AIR_PKT_TYPE_GET_CAPABILITIES:
      handle_packet_get_capabilities(pkt, mb_id);
      break;

    case AIR_PKT_TYPE_ALLOCATE_HERD_SHIM_DMAS:
      handle_packet_allocate_herd_shim_dmas(pkt);
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
    case AIR_PKT_TYPE_HERD_SHIM_DMA_MEMCPY:
      handle_packet_herd_shim_memcpy(pkt);
      break;
    case AIR_PKT_TYPE_HERD_SHIM_DMA_1D_STRIDED_MEMCPY:
      handle_packet_herd_shim_1d_strided_memcpy(pkt);
      break;
    case AIR_PKT_TYPE_ND_MEMCPY:
      handle_packet_nd_memcpy(pkt);
      break;

  }

}

int main()
{
  pvr_t pvr;
  init_platform();
  microblaze_get_pvr(&pvr);
  int user1 = MICROBLAZE_PVR_USER1(pvr);
  int user2 = MICROBLAZE_PVR_USER2(pvr);
  int mb_id = user2 & 0xff;  
  int maj   = (user2 >> 24) & 0xff;
  int min   = (user2 >> 16) & 0xff;
  int ver   = (user2 >> 8) & 0xff;

  // Skip over the system wide shmem area, then find your own
  base_address = shmem_base + (1+mb_id)*MB_SHMEM_SEGMENT_SIZE;

  lock_uart(mb_id);
  xil_printf("MB %d firmware %d.%d.%d created on %s at %s GMT\n\r",mb_id,maj,min,ver,__DATE__, __TIME__); 
  xil_printf("(c) Copyright 2020-2021 Xilinx, Inc. All rights reserved.\n\r");
  unlock_uart();

  setup = false;
  //test_stream();
  queue_t *q = nullptr;
  queue_create(MB_QUEUE_SIZE, &q, mb_id);
  lock_uart(mb_id); xil_printf("Created queue @ 0x%llx\n\r", (size_t)q); unlock_uart();

  bool done = false;
  int cnt = 0;
  while (!done) {
    // if (!(cnt++ % 0x00100000))
    //   xil_printf("No Ding Dong 0x%llx\n\r", q->doorbell);
    if (q->doorbell+1 > q->last_doorbell) {
      lock_uart(mb_id); xil_printf("Ding Dong 0x%llx\n\r", q->doorbell+1); unlock_uart();

      q->last_doorbell = q->doorbell+1;

      // process packets until we hit an invalid packet
      bool invalid = false;
      while (!invalid) {
        uint64_t rd_idx = queue_load_read_index(q);
        //xil_printf("Handle pkt read_index=%d\n\r", rd_idx);

        dispatch_packet_t *pkt = &((dispatch_packet_t*)q->base_address)[rd_idx % q->size];
        switch (pkt->type) {
          default:
          case HSA_PACKET_TYPE_INVALID:
            if (setup) {
              lock_uart(mb_id); xil_printf("Waiting\n\r"); unlock_uart();
              setup = false;
            }
            invalid = true;
            break;
          case HSA_PACKET_TYPE_AGENT_DISPATCH:
            handle_agent_dispatch_packet(pkt, mb_id);
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
