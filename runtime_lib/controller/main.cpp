// (c) Copyright 2022 Xilinx Inc. All Rights Reserved.

#include <cstdint>
#include <cstring>
#include "unistd.h"

extern "C" {
#include "xil_printf.h"
#include "pvr.h"

//#include "xaiengine.h"
//#include "mb_interface.h"

#include "acdc_queue.h"
#include "hsa_defs.h"
}

#include "platform.h"

#define XAIE_NUM_ROWS            8
#define XAIE_NUM_COLS           50
#define XAIE_ADDR_ARRAY_OFF     0x800

#define XAIEGBL_TILE_ADDR_ARR_SHIFT         30U
#define XAIEGBL_TILE_ADDR_ROW_SHIFT         18U
#define XAIEGBL_TILE_ADDR_COL_SHIFT         23U

#define XAIEDMA_SHIM_CHNUM_S2MM0      0U
#define XAIEDMA_SHIM_CHNUM_S2MM1      1U
#define XAIEDMA_SHIM_CHNUM_MM2S0      2U
#define XAIEDMA_SHIM_CHNUM_MM2S1      3U

#define HIGH_ADDR(addr)	((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr)	(addr & 0x00000000ffffffff)

#define LOGICAL_HERD_DMAS      16

#define SHIM_DMA_S2MM 0 
#define SHIM_DMA_MM2S 1 

#define NUM_SHIM_DMAS 16
#define NUM_COL_DMAS 4
int shim_dma_cols[NUM_SHIM_DMAS] = {2, 3, 6, 7, 10, 11, 18, 19, 26, 27, 34, 35, 42, 43, 46, 47};
int col_dma_cols[NUM_COL_DMAS] = {7, 8, 9, 10};
#define NUM_DMAS (NUM_SHIM_DMAS + NUM_COL_DMAS)

#define CHATTY 0 

#define air_printf(fmt, ...) \
	                    do { if (CHATTY) xil_printf(fmt, ##__VA_ARGS__); } while (0)

uint64_t mymod(uint64_t a) {
  uint64_t result = a;
  while (result >= MB_QUEUE_SIZE) {
    result -= MB_QUEUE_SIZE;
  }
  return result;
}

namespace {

struct HerdConfig {
  uint32_t row_start;
  uint32_t num_rows;
  uint32_t col_start;
  uint32_t num_cols;
};

HerdConfig HerdCfgInst;

namespace xaie {

u64 getTileAddr(u16 ColIdx, u16 RowIdx) 
{
  u64 TileAddr = 0;
  u64 ArrOffset = XAIE_ADDR_ARRAY_OFF;

  #ifdef XAIE_BASE_ARRAY_ADDR_OFFSET
    ArrOffset = XAIE_BASE_ARRAY_ADDR_OFFSET;
  #endif

  /*
    * Tile address format:
    * --------------------------------------------
    * |                7 bits  5 bits   18 bits  |
    * --------------------------------------------
    * | Array offset | Column | Row | Tile addr  |
    * --------------------------------------------
    */
  TileAddr = (u64)((ArrOffset <<
      XAIEGBL_TILE_ADDR_ARR_SHIFT) |
    (ColIdx << XAIEGBL_TILE_ADDR_COL_SHIFT) |
    (RowIdx << XAIEGBL_TILE_ADDR_ROW_SHIFT));

  return TileAddr;
}

static inline u32 in32(u64 Addr)
{
  /* read 32 bit value from specified address */
  return *(volatile u32 *) Addr;
}

static inline void out32(u64 Addr, u32 Value)
{
  /* write 32 bit value to specified address */
  volatile u32 *LocalAddr = (volatile u32 *)Addr;
  *LocalAddr = Value;
}

u32 maskpoll32(u64 Addr, u32 Mask, u32 Value, u32 TimeOut)
{
  u32 Ret = 1;

  u32 Count = 10 + TimeOut;

  while (Count > 0U) {
    if ((in32(Addr) & Mask) == Value) {
      Ret = 0;
      break;
    }
    Count--;
  }

  return Ret;
}

} // namespace xaie


void xaie_shim_dma_wait_idle(uint64_t TileAddr, int direction, int channel) {
  uint32_t shimDMAchannel = channel;
  uint32_t status_register_offset;
  uint32_t status_mask_shift;
  if (channel == 0) {
    status_mask_shift = 0;
  }
  else {
    status_mask_shift = 2;
  }
  if (direction == SHIM_DMA_S2MM) {
    shimDMAchannel += XAIEDMA_SHIM_CHNUM_S2MM0;
    status_register_offset = 0x1d160;
  }
  else {
    shimDMAchannel += XAIEDMA_SHIM_CHNUM_MM2S0;
    status_register_offset = 0x1d164;
  }
  while ((xaie::in32(TileAddr + status_register_offset) >> status_mask_shift) & 0b11);
}

uint32_t xaie_shim_dma_get_outstanding(uint64_t TileAddr, int direction, int channel) {
  uint32_t shimDMAchannel = channel;
  uint32_t status_register_offset;
  uint32_t start_queue_size_mask_shift;
  if (channel == 0) {
    start_queue_size_mask_shift = 6;
  }
  else {
    start_queue_size_mask_shift = 9;
  }
  if (direction == SHIM_DMA_S2MM) {
    shimDMAchannel += XAIEDMA_SHIM_CHNUM_S2MM0;
    status_register_offset = 0x1d160;
  }
  else {
    shimDMAchannel += XAIEDMA_SHIM_CHNUM_MM2S0;
    status_register_offset = 0x1d164;
  }
  uint32_t outstanding = (xaie::in32(TileAddr + status_register_offset) >> start_queue_size_mask_shift) & 0b111;
  return outstanding;
}

//// GLOBAL for shim DMAs mapped to the controller
//uint16_t mappedShimDMA[2] = {0};
//// GLOBAL for round-robin bd locations
//uint32_t last_bd[4][2] = {0};
uint32_t last_bd[8] = {0};

int xaie_shim_dma_push_bd(uint64_t TileAddr, int direction, int channel, int col, uint64_t addr, uint32_t len)
{
  uint32_t shimDMAchannel = channel; // Need
  uint32_t status_register_offset;
  uint32_t status_mask_shift;
  uint32_t control_register_offset;
  uint32_t start_queue_register_offset;
  uint32_t start_queue_size_mask_shift;

  if (direction == SHIM_DMA_S2MM) {
    shimDMAchannel += XAIEDMA_SHIM_CHNUM_S2MM0;
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
    air_printf("\n\r  S2MM Shim DMA %d start channel %d\n\r", col, shimDMAchannel);
    //air_printf("\n\r  S2MM Shim DMA %d start channel %d\n\r", mappedShimDMA[dma], shimDMAchannel);
  }
  else {
    shimDMAchannel += XAIEDMA_SHIM_CHNUM_MM2S0;
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
    air_printf("\n\r  MM2S Shim DMA %d start channel %d\n\r", col, shimDMAchannel);
    //air_printf("\n\r  MM2S Shim DMA %d start channel %d\n\r", mappedShimDMA[dma], shimDMAchannel);
  }

  uint32_t start_bd = 4*shimDMAchannel; // shimDMAchannel<<2;
  uint32_t outstanding = (xaie::in32(TileAddr + status_register_offset) >> start_queue_size_mask_shift) & 0b111;
  // If outstanding >=4, we're in trouble!!!!
  // Theoretically this should never occur due to check in do_packet_nd_memcpy 
  if (outstanding >=4) { // NOTE had this at 3? // What is proper 'stalled' threshold? 
    //if (outstanding >=4)
      air_printf("\n\r *** BD OVERFLOW in shimDMA channel %d *** \n\r",shimDMAchannel);
    bool waiting = true;
    while (waiting) {
      outstanding = (xaie::in32(TileAddr + status_register_offset) >> start_queue_size_mask_shift) & 0b111;
      waiting = (outstanding > 3); // NOTE maybe >= 3
      air_printf("*** Stalled in shimDMA channel %d outstanding = %d *** \n\r",shimDMAchannel,outstanding+1);
    } // WARNING this can lead to an endless loop 
  }
  air_printf("Outstanding pre : %d\n\r", outstanding);
  //uint32_t bd = start_bd+outstanding;// + 0; // HACK
  int slot = channel;
  slot += ((col%2)==1)?4:0;
  if (direction == SHIM_DMA_S2MM) 
    slot += XAIEDMA_SHIM_CHNUM_S2MM0;
  else 
    slot += XAIEDMA_SHIM_CHNUM_MM2S0;
  uint32_t bd = start_bd+last_bd[slot];
  last_bd[slot] = (last_bd[slot]==3)?0:last_bd[slot]+1;
  //uint32_t bd = start_bd+last_bd[shimDMAchannel][dma];
  //last_bd[shimDMAchannel][dma] = (last_bd[shimDMAchannel][dma]==3)?0:last_bd[shimDMAchannel][dma]+1;
  uint32_t bd_offset = bd*0x14;
  xaie::out32(TileAddr + 0x0001D008+(bd_offset), 0x0);           // Mark the BD as invalid

  // Set the registers directly ...
  uint32_t base_address =  0x1d000 + bd_offset;
  xaie::out32(TileAddr + base_address + 0x00, LOW_ADDR((u64)addr));
  xaie::out32(TileAddr + base_address + 0x04, len >> 2); // We pass in bytes, but the shim DMA can ony deal with 32 bits
  u32 control = (HIGH_ADDR((u64)addr) << 16) | 1;
  xaie::out32(TileAddr + base_address + 0x08, control);
  xaie::out32(TileAddr + base_address + 0x0C, 0x410); // Burst len [10:9] = 2 (16)
                                                                // QoS [8:5] = 0 (best effort)
                                                                // Secure bit [4] = 1 (set)
  xaie::out32(TileAddr + base_address + 0x10, 0x0);


  // Check if the channel is running or not
  uint32_t precheck_status = (xaie::in32(TileAddr + status_register_offset) >> status_mask_shift) & 0b11;
  if (precheck_status == 0b00) {
    xaie::out32(TileAddr + control_register_offset, 0xb001); // Stream traffic can run, we can issue AXI-MM, and the channel is enabled
  }
  // Now push into the queue
  xaie::out32(TileAddr + start_queue_register_offset, bd);

#if CHATTY
  outstanding = (xaie::in32(TileAddr + status_register_offset) >> start_queue_size_mask_shift) & 0b111;
  air_printf("Outstanding post: %d\n\r", outstanding);
  air_printf("bd pushed as bd %d\n\r",bd);
 
  if (direction == SHIM_DMA_S2MM) {
    air_printf("  End of S2MM Shim DMA %d start channel %d\n\r", col, shimDMAchannel);
    //air_printf("  End of S2MM Shim DMA %d start channel %d\n\r", mappedShimDMA[dma], shimDMAchannel);
  }
  else {
    air_printf("  End of MM2S Shim DMA %d start channel %d\n\r", col, shimDMAchannel);
    //air_printf("  End of MM2S Shim DMA %d start channel %d\n\r", mappedShimDMA[dma], shimDMAchannel);
  }
#endif
  return 1;
}

int xaie_lock_release(u16 col, u16 row, u32 lock_id, u32 val)
{
  u64 Addr = xaie::getTileAddr(col,row);
  u64 LockOfst = 0x0001E020;
  if (row != 0) 
    LockOfst = 0x0001E020 + 0x10*(val&0x1);
  else { 
    switch (col % 4) {
    case 0:
    case 1:
      LockOfst = 0x00014020 + 0x10*(val&0x1);
      break;
    default:
      LockOfst = 0x00014020 + 0x10*(val&0x1);
      break;
    }
  }
  xaie::maskpoll32(Addr + LockOfst + 0x80*lock_id, 0x1, 0x1, 0); 
  //XAieTile_LockRelease(tile, lock_id, val, 0);
  return 1;
}

int xaie_lock_acquire_nb(u16 col, u16 row, u32 lock_id, u32 val)
{
  u64 Addr = xaie::getTileAddr(col,row);
  u64 LockOfst = 0x0001E060;
  if (row != 0) 
    LockOfst = 0x0001E060 + 0x10*(val&0x1);
  else { 
    switch (col % 4) {
    case 0:
    case 1:
      LockOfst = 0x00014060 + 0x10*(val&0x1);
      break;
    default:
      LockOfst = 0x00014060 + 0x10*(val&0x1);
      break;
    }
  }
  u8 lock_ret = 0;
  u32 loop = 0;
  while ((!lock_ret) && (loop < 512)) {
    lock_ret = xaie::maskpoll32(Addr + LockOfst + 0x80*lock_id, 0x1, 0x1, 100);
    //lock_ret = XAieTile_LockAcquire(tile, lock_id, val, 10000);
    loop++;
  }
  if (loop == 512) {
    air_printf("Acquire [%d, %d, %d] value %d time-out\n\r", col, row, lock_id, val);
    return 0;
  }
  return 1;
}

void xaie_shim_dma_init(int col)
{
  // Invalidate all BDs by writing to their buffer control register
  for (int ch=0;ch<4;ch++) {
    xaie::out32(xaie::getTileAddr(col,0) + 0x0001D140 + 0x8*ch, 0x00); // Disable all channels
  }
  for (int bd=0;bd<16;bd++) {
    xaie::out32(xaie::getTileAddr(col,0) + 0x0001D008 + 0x14*bd, 0);
  }
}

void xaie_device_init(int num_cols)
{
  if (num_cols > NUM_SHIM_DMAS) {
    air_printf("WARN: attempt to initialize more shim DMAs than device has available!\n\r");
    num_cols = NUM_SHIM_DMAS;
  }

  for (int c=0; c<num_cols; c++) {
    xaie_shim_dma_init(shim_dma_cols[c]);
  }
}

// Initialize one herd with lower left corner at (col_start, row_start)
void xaie_herd_init(int col_start, int num_cols, int row_start, int num_rows)
{
  HerdCfgInst.col_start = col_start;
  HerdCfgInst.num_cols = num_cols;
  HerdCfgInst.row_start = row_start;
  HerdCfgInst.num_rows = num_rows;
}

} // namespace

namespace {

uint64_t shmem_base = 0x020100000000UL;
uint64_t uart_lock_offset = 0x200;
uint64_t base_address;

bool setup;

void lock_uart(uint32_t id) {
  bool is_locked = false;
  volatile uint32_t *ulb = (volatile uint32_t *)(shmem_base+uart_lock_offset);

  while (!is_locked) {
    uint32_t status = ulb[0];
    if (status != 1) {
      ulb[1] = id;
      ulb[0] = 1;
      // See if they stuck
      uint32_t status = ulb[0];
      uint32_t lockee = ulb[1];
      if ((status == 1) && (lockee == id)) {
        //air_printf("ULock @ %lx MB %02d: ",ulb, id);
        is_locked = true;
      }
    }
  }
}

  // This looks unsafe, but its okay as long as we always aquire
  // the lock first
void unlock_uart() {
  volatile uint32_t *ulb = (volatile uint32_t *)(shmem_base+uart_lock_offset);
  ulb[1] = 0; 
  ulb[0] = 0;
}

int queue_create(uint32_t size, queue_t **queue, uint32_t mb_id)
{
  uint64_t queue_address[1] = {base_address + sizeof(dispatch_packet_t)};
  uint64_t queue_base_address[1] = {queue_address[0] + sizeof(dispatch_packet_t)};
  lock_uart(mb_id);
  air_printf("setup_queue 0x%llx, %x bytes + %d 64 byte packets\n\r", (void *)queue_address, sizeof(queue_t), size);
  air_printf("base address 0x%llx\n\r", base_address);
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
  pkt->header = HSA_PACKET_TYPE_INVALID;
  pkt->type = AIR_PKT_TYPE_INVALID;
  signal_subtract_acq_rel((signal_t*)&pkt->completion_signal, 1);
}

void complete_barrier_packet(void *pkt)
{
  barrier_and_packet_t *p = (barrier_and_packet_t *)(pkt);
  // completion phase
  p->header = HSA_PACKET_TYPE_INVALID;
  signal_subtract_acq_rel((signal_t*)&p->completion_signal, 1);
}

void handle_packet_device_initialize(dispatch_packet_t *pkt) { 
  packet_set_active(pkt, true);
  air_printf("Called depricated function: handle_packet_device_initialize...\r\n");
  air_printf("...but will still invalidate shim DMA bds\r\n");
  xaie_device_init(NUM_SHIM_DMAS);
}

void handle_packet_herd_initialize(dispatch_packet_t *pkt) {
  setup = true;
  packet_set_active(pkt, true);
  // Address mode here is absolute range
  if (((pkt->arg[0] >> 48) & 0xf) == AIR_ADDRESS_ABSOLUTE_RANGE) {
    u32 start_row = (pkt->arg[0] >> 16) & 0xff;
    u32 num_rows  = (pkt->arg[0] >> 24) & 0xff;
    u32 start_col = (pkt->arg[0] >> 32) & 0xff;
    u32 num_cols  = (pkt->arg[0] >> 40) & 0xff;

    u32 herd_id  = pkt->arg[1] & 0xffff;
    u32 shimDMA0 = (pkt->arg[1] >> 16) & 0xff;
    u32 shimDMA1 = (pkt->arg[1] >> 24) & 0xff;
    xaie_herd_init(start_col, num_cols, start_row, num_rows);
    air_printf("Initialized herd %d at (%d, %d) of size (%d,%d)\r\n",
                      herd_id, start_col, start_row, num_cols, num_rows);
    // herd_id is ignored - current restriction is 1 herd -> 1 controller
    xaie_device_init(NUM_SHIM_DMAS);
    //mappedShimDMA[0] = shimDMA0;
    //mappedShimDMA[1] = shimDMA1;
    //xaie_shim_dma_init(shimDMA0);
    //air_printf("Initialized shim DMA physical idx %d to logical idx %d\r\n",shimDMA0,0);
    //xaie_shim_dma_init(shimDMA1);
    //air_printf("Initialized shim DMA physical idx %d to logical idx %d\r\n",shimDMA1,1);
  }
  else {
    air_printf("Unsupported address type 0x%04X for herd initialize\r\n",(pkt->arg[0] >> 48) & 0xf);
  }
}

void handle_packet_get_capabilities(dispatch_packet_t *pkt, uint32_t mb_id)
{
  // packet is in active phase
  packet_set_active(pkt, true);
  uint64_t *addr = (uint64_t *)(pkt->return_address);

  lock_uart(mb_id); air_printf("Writing to 0x%llx\n\r",(uint64_t)addr); unlock_uart();
  // We now write a capabilities structure to the address we were just passed
  // We've already done this once - should we just cache the results?
  pvr_t pvr;
  microblaze_get_pvr(&pvr);
  int user1 = MICROBLAZE_PVR_USER1(pvr);
  int user2 = MICROBLAZE_PVR_USER2(pvr);

  addr[0] = (uint64_t)mb_id;           // region id
  addr[1] = (uint64_t)user1;           // num regions
  addr[2] = (uint64_t)(user2 >> 8);    // region controller firmware version
  addr[3] = 16L;                       // cores per region
  addr[4] = 32768L;                    // Total L1 data memory per core
  addr[5] = 8L;                        // Number of L1 data memory banks
  addr[6] = 16384L;                    // L1 program memory per core
  addr[7] = 0L;                        // L2 data memory per region
}

void handle_packet_get_info(dispatch_packet_t *pkt, uint32_t mb_id)
{
  // packet is in active phase
  packet_set_active(pkt, true);
  uint64_t attribute = (pkt->arg[0]);
  uint64_t *addr = (uint64_t *)(&pkt->return_address); // FIXME when we can use a VA

  pvr_t pvr;
  microblaze_get_pvr(&pvr);
  int user1 = MICROBLAZE_PVR_USER1(pvr);
  int user2 = MICROBLAZE_PVR_USER2(pvr);
  char name[] = {'A','C','D','C','\0'};
  char vend[] = {'A','M','D','\0'};

  // TODO change this to use pkt->return_address
  switch(attribute) {
    case AIR_AGENT_INFO_NAME:
      memcpy(addr,name,8); 
      break;
    case AIR_AGENT_INFO_VENDOR_NAME:
      memcpy(addr,vend,8); 
      break;
    case AIR_AGENT_INFO_CONTROLLER_ID:
      *addr = (uint64_t)mb_id;           // region id
      break;
    case AIR_AGENT_INFO_FIRMWARE_VER:
      *addr = (uint64_t)(user2 >> 8);    // region controller firmware version
      break;
    case AIR_AGENT_INFO_NUM_REGIONS: 
      *addr = (uint64_t)user1;           // num regions
      break;
    case AIR_AGENT_INFO_HERD_SIZE:       // cores per region 
      *addr = HerdCfgInst.num_cols*HerdCfgInst.num_rows;
      break;
    case AIR_AGENT_INFO_HERD_ROWS:
      *addr = HerdCfgInst.num_rows;      // rows of cores
      break;
    case AIR_AGENT_INFO_HERD_COLS:
      *addr = HerdCfgInst.num_cols;      // cols of cores
      break;
    case AIR_AGENT_INFO_TILE_DATA_MEM_SIZE:
      *addr = 32768L;                    // total L1 data memory per core
      break;
    case AIR_AGENT_INFO_TILE_PROG_MEM_SIZE:
      *addr = 16384L;                    // L1 program memory per core
      break;
    case AIR_AGENT_INFO_L2_MEM_SIZE:     // L2 memory per region (cols * 256k)
      *addr = 262144L * HerdCfgInst.num_cols;
      break;
    default:
      *addr = 0;
      break;
  }
}

//uint64_t cdma_base = 0x0202C0000000UL;
//uint64_t cdma_base1 = 0x020340000000UL;
uint64_t cfg_cdma_base = 0x000044A00000UL;

void handle_packet_sg_cdma(dispatch_packet_t *pkt)
{
  //volatile uint32_t *timerCtrl = (volatile uint32_t *)(xaie::getTileAddr(7,1) + 0x00034000);
  //volatile uint32_t *timer = (volatile uint32_t *)(xaie::getTileAddr(7,1) + 0x000340F8);
  // packet is in active phase
  packet_set_active(pkt, true);
  volatile uint32_t *cdmab = (volatile uint32_t *)(cfg_cdma_base);
  u32 start_row = (pkt->arg[3] >>  0) & 0xff;
  u32 num_rows  = (pkt->arg[3] >>  8) & 0xff;
  u32 start_col = (pkt->arg[3] >> 16) & 0xff;
  u32 num_cols  = (pkt->arg[3] >> 24) & 0xff;
  for (uint c=start_col; c<start_col+num_cols; c++) {
    for (uint r=start_row; r<start_row+num_rows; r++) {
      //int st = xaie::in32(xaie::getTileAddr(c,r) + 0x00032004);
      //if ((0x3&st) != 0x2) {
        xaie::out32(xaie::getTileAddr(c,r) + 0x00032000, 0x2);
        air_printf("Done resetting col %d row %d.\n\r",c,r);
      //}
    }
  air_printf("Resetting column %d.\n\r",c);
    xaie::out32(xaie::getTileAddr(c,0) + 0x00036048, !!1); // 1 == ResetEnable
    xaie::out32(xaie::getTileAddr(c,0) + 0x00036048, !!0); // 0 == ResetDisable
  air_printf("Done resetting column %d.\n\r",c);
  }
  air_printf("CDMA reset.\n\r");
  cdmab[0] |= 0x4;
  cdmab[0] &= 0x4;
  while (cdmab[0]&0x4);
  //uint32_t status = cdmab[1];
  //air_printf("CMDA raw %x idle %x\n\r",status,status&2);
  uint64_t daddr = (pkt->arg[0]);
  uint64_t saddr = (pkt->arg[1]);
  uint32_t bytes = (pkt->arg[2]);
  //uint32_t before = 0;
  cdmab[0] = 0x0; // unset SG mode 
  if (bytes >= 0xffffff) { // SG
    cdmab[0] = 0x8; // set SG mode 
    cdmab[2] = saddr&0xffffffff; 
    cdmab[3] = saddr>>32; 
    //timerCtrl[0] = 1<<31;
    cdmab[5] = daddr>>32;  
    //before = timer[0];
    cdmab[4] = daddr&0xffffffff; 
  } else {
    cdmab[6] = saddr&0xffffffff; 
    cdmab[7] = saddr>>32; 
    cdmab[8] = daddr&0xffffffff; 
    //timerCtrl[0] = 1<<31;
    cdmab[9] = daddr>>32;  
    //before = timer[0];
    cdmab[10] = bytes;
  }
  int cnt = 100;
  while (!(cdmab[1]&2)&&cnt--) air_printf("SG CDMA wait... %x\n\r",cdmab[1]);
  for (uint c=start_col; c<start_col+num_cols; c++) {
    for (uint r=start_row; r<=start_row+num_rows; r++) {
      for (int l=0; l<16; l++)
        xaie::maskpoll32(xaie::getTileAddr(c,r) + 0x0001E020 + 0x80*l, 0x1, 0x1, 0); 
      xaie::out32(xaie::getTileAddr(c,r) + 0x00032000, 0x1); 
    }
  }
  //uint32_t after = timer[0];
  //xil_printf("CDMA usec B %4u A %6u A-B %6u\n\r",before, after, (after - before));
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
        xaie_lock_acquire_nb(HerdCfgInst.col_start+start_col+col, HerdCfgInst.row_start+start_row+row, lock_id, val);
      else
        xaie_lock_release(HerdCfgInst.col_start+start_col+col, HerdCfgInst.row_start+start_row+row, lock_id, val);
    }
  }
}

void handle_packet_hello(dispatch_packet_t *pkt, uint32_t mb_id)
{
  packet_set_active(pkt, true);

  uint64_t say_what = pkt->arg[0];
  lock_uart(mb_id); xil_printf("MB %d : HELLO %08X\n\r",mb_id,(uint32_t)say_what); unlock_uart();
}

typedef struct staged_nd_memcpy_s {
  uint32_t valid;
	dispatch_packet_t *pkt;
	uint64_t paddr[3];
	uint32_t index[3];
} staged_nd_memcpy_t; // about 48B therefore @ 64 slots ~3kB

int get_slot(int col, int space) {
  if (space == 2) {
    for (int i=0; i<NUM_SHIM_DMAS; i++) {
      if (col == shim_dma_cols[i]) {
        return i*4;
      }
    }
  } else if (space == 1) {
    for (int i=0; i<NUM_COL_DMAS; i++) {
      if (col == col_dma_cols[i]) {
        return i*4 + NUM_SHIM_DMAS*4;
      }
    }
  }
  return 0;
}

// GLOBAL storage for 'in progress' ND memcpy work
// NOTE 4 slots per shim DMA 
staged_nd_memcpy_t staged_nd_slot[NUM_DMAS*4]; 

void nd_dma_put_checkpoint(dispatch_packet_t **pkt, uint32_t slot, 
			    uint32_t idx_4d, uint32_t idx_3d, uint32_t idx_2d,
    			uint64_t pad_3d, uint64_t pad_2d, uint64_t pad_1d) 
{
  staged_nd_slot[slot].pkt = *pkt;
  staged_nd_slot[slot].paddr[0] = pad_1d;
  staged_nd_slot[slot].paddr[1] = pad_2d;
  staged_nd_slot[slot].paddr[2] = pad_3d;
  staged_nd_slot[slot].index[0] = idx_2d;
  staged_nd_slot[slot].index[1] = idx_3d;
  staged_nd_slot[slot].index[2] = idx_4d;
}

void nd_dma_get_checkpoint(dispatch_packet_t **pkt, uint32_t slot,
			    uint32_t& idx_4d, uint32_t& idx_3d, uint32_t& idx_2d,
    			uint64_t& pad_3d, uint64_t& pad_2d, uint64_t& pad_1d) 
{
  *pkt = staged_nd_slot[slot].pkt;
  pad_1d = staged_nd_slot[slot].paddr[0];
  pad_2d = staged_nd_slot[slot].paddr[1];
  pad_3d = staged_nd_slot[slot].paddr[2];
  idx_2d = staged_nd_slot[slot].index[0];
  idx_3d = staged_nd_slot[slot].index[1];
  idx_4d = staged_nd_slot[slot].index[2];
}

int do_packet_nd_memcpy(uint32_t slot)
{
  dispatch_packet_t* a_pkt;
  uint64_t paddr_3d;
  uint64_t paddr_2d;
  uint64_t paddr_1d;
  uint32_t index_4d;
  uint32_t index_3d;
  uint32_t index_2d;
  nd_dma_get_checkpoint(&a_pkt,slot,index_4d,index_3d,index_2d,paddr_3d,paddr_2d,paddr_1d);

  uint16_t channel      = (a_pkt->arg[0] >> 24) & 0x00ff;
  uint16_t col          = (a_pkt->arg[0] >> 32) & 0x00ff;
  //uint16_t logical_col  = (a_pkt->arg[0] >> 32) & 0x00ff;
  uint16_t direction    = (a_pkt->arg[0] >> 60) & 0x000f;
  uint32_t length_1d    = (a_pkt->arg[2] >>  0) & 0xffffffff;
  uint32_t length_2d    = (a_pkt->arg[2] >> 32) & 0x0000ffff;
  uint32_t stride_2d    = (a_pkt->arg[2] >> 48) & 0x0000ffff;
  uint32_t length_3d    = (a_pkt->arg[3] >>  0) & 0x0000ffff;
  uint32_t stride_3d    = (a_pkt->arg[3] >> 16) & 0x0000ffff;
  uint32_t length_4d    = (a_pkt->arg[3] >> 32) & 0x0000ffff;
  uint32_t stride_4d    = (a_pkt->arg[3] >> 48) & 0x0000ffff;
  //uint16_t col          = mappedShimDMA[logical_col];
  uint32_t outstanding = 0;

  air_printf("Do ND shim DMA %d dir %d chan %d paddr %llx 4d %d stride %d length 3d %d stride %d length, 2d %d stride %d length, 1d %d length\n\r",col, direction, channel, paddr_1d, stride_4d, length_4d,
       stride_3d, length_3d, stride_2d, length_2d, length_1d);

  for (;index_4d<length_4d;index_4d++) {
    for (;index_3d<length_3d;index_3d++) {
      for (;index_2d<length_2d;index_2d++) {
        outstanding = xaie_shim_dma_get_outstanding(xaie::getTileAddr(col,0),direction,channel);
        air_printf("\n\rND start shim DMA %d %d [%d][%d][%d] paddr %llx \n\r",
      		  direction, channel, index_4d, index_3d, index_2d, paddr_1d);
        if (outstanding >= 4) { // NOTE What is proper 'stalled' threshold? 
          nd_dma_put_checkpoint(&a_pkt,slot,index_4d,index_3d,index_2d,paddr_3d,paddr_2d,paddr_1d);
	        return 1;
        } else { 
          xaie_shim_dma_push_bd(xaie::getTileAddr(col,0), direction, channel, col, paddr_1d, length_1d);
          //xaie_shim_dma_push_bd(&xaie::ShimTileInst[col], direction, channel, logical_col, paddr_1d, length_1d);
        }
        paddr_1d += stride_2d;
      }
      index_2d = 0;
      paddr_2d += stride_3d;
      if (index_3d+1<length_3d) paddr_1d = paddr_2d;
      else paddr_1d = paddr_3d + stride_4d;
    }
    index_3d = 0;
    paddr_3d += stride_4d;
    paddr_2d = paddr_3d;
  }

  // Wait check idle
  xaie_shim_dma_wait_idle(xaie::getTileAddr(col,0),direction,channel);

  return 0;
}

int do_packet_memcpy(uint32_t slot) {
  if (slot >= NUM_SHIM_DMAS*4) {
    return 0;
  } else {
    return do_packet_nd_memcpy(slot);
  }
}

int stage_packet_nd_memcpy(dispatch_packet_t *pkt, uint32_t slot, uint32_t memory_space)
{
  air_printf("stage_packet_nd_memcpy %d\n\r",slot);
  if (staged_nd_slot[slot].valid) {
    air_printf("STALL: ND Memcpy Slot %d Busy!\n\r",slot);
    return 2;
  }
  packet_set_active(pkt, true);

  uint64_t paddr        =  pkt->arg[1];

  if (memory_space == 2) {
    nd_dma_put_checkpoint(&pkt,slot,0,0,0,paddr,paddr,paddr);
    staged_nd_slot[slot].valid = 1; 
    return 0;
  }
  else {
    air_printf("NOT SUPPORTED: Cannot program memory space %d DMAs\n\r",memory_space);
    return 1;
  }
}

} // namespace

void handle_agent_dispatch_packet(queue_t *q, uint32_t mb_id)
{
  uint64_t rd_idx = queue_load_read_index(q);
  dispatch_packet_t *pkt = &((dispatch_packet_t*)q->base_address)[mymod(rd_idx)];
  int last_slot = 0;
  int max_slot = 4*NUM_DMAS-1;

  int num_active_packets = 1;
  int packets_processed = 0;
  do {
    // Looped back because ND memcpy failed to finish on the first try.
    // No other packet type will not finish on first try.  
    if (num_active_packets > 1) {
      // NOTE assume we are coming from a stall, that's why we RR. 
      
      // INFO: 
      // 1)  check for valid staged packets that aren't the previous 
      // 2a)  FOUND process packet here
      // 2b) !FOUND get next packet && check invalid
      // 3b) goto packet_op
      int slot = last_slot;
      bool stalled = true;
      bool active = false; 
      do {
        slot = (slot==max_slot)?0:slot+1; // TODO better heuristic 
        if (slot == last_slot) break;
        air_printf("RR check slot: %d\n\r",slot);
        if (staged_nd_slot[slot].valid) {
          dispatch_packet_t* a_pkt = staged_nd_slot[slot].pkt;
          uint16_t channel      = (a_pkt->arg[0] >> 24) & 0x00ff;
          uint16_t col          = (a_pkt->arg[0] >> 32) & 0x00ff;
          //uint16_t logical_col  = (a_pkt->arg[0] >> 32) & 0x00ff;
          uint16_t direction    = (a_pkt->arg[0] >> 60) & 0x000f;
          //uint16_t col          = mappedShimDMA[logical_col];
          stalled = (xaie_shim_dma_get_outstanding(xaie::getTileAddr(col,0),direction,channel) >= 4); 
          active = packet_get_active(a_pkt);
        } else {
          stalled = true;
          active = false;
        }
        air_printf("RR slot: %d - valid %d stalled %d active %d\n\r",slot,staged_nd_slot[slot].valid,stalled,active);
      } while (!staged_nd_slot[slot].valid || stalled || !active); 

      if (slot==last_slot) { // Begin get next packet
        rd_idx++; 
        pkt = &((dispatch_packet_t*)q->base_address)[mymod(rd_idx)];
        air_printf("HELLO NEW PACKET IN FLIGHT!\n\r");
        if (((pkt->header)&0xF) != HSA_PACKET_TYPE_AGENT_DISPATCH) { 
          rd_idx--;
          pkt = &((dispatch_packet_t*)q->base_address)[mymod(rd_idx)];
          air_printf("WARN: Found invalid HSA packet inside peek loop!\n\r");
          // TRICKY weird state where we didn't find a new packet but RR won't let us retry. So advance last_slot.
          last_slot = (slot==max_slot)?0:slot+1; // TODO better heuristic 
          continue;  
        } else goto packet_op;
      } // End get next packet

      // FOUND ND packet process here 
found:
      last_slot = slot; 
      int ret = do_packet_memcpy(slot);  
      if (ret) continue;
      else {
        num_active_packets--;
        staged_nd_slot[slot].valid = 0; 
        complete_agent_dispatch_packet(staged_nd_slot[slot].pkt);  
        packets_processed++;
        continue;
      }
    }

packet_op:
    auto op = pkt->type & 0xffff;
    //air_printf("Op is %04X\n\r",op);
    switch (op) {
      case AIR_PKT_TYPE_INVALID:
      default:
        air_printf("WARN: invalid air pkt type\n\r");
        complete_agent_dispatch_packet(pkt);
        packets_processed++;
        break;

      case AIR_PKT_TYPE_DEVICE_INITIALIZE:
        handle_packet_device_initialize(pkt);
        complete_agent_dispatch_packet(pkt);
        packets_processed++;
        break;
      case AIR_PKT_TYPE_HERD_INITIALIZE:
        handle_packet_herd_initialize(pkt);
        complete_agent_dispatch_packet(pkt);
        packets_processed++;
        break;

      case AIR_PKT_TYPE_CONFIGURE:
        handle_packet_sg_cdma(pkt);
        complete_agent_dispatch_packet(pkt);
        packets_processed++;
        break;

      case AIR_PKT_TYPE_HELLO:
        handle_packet_hello(pkt, mb_id);
        complete_agent_dispatch_packet(pkt);
        packets_processed++;
        break;
      case AIR_PKT_TYPE_GET_CAPABILITIES:
        handle_packet_get_capabilities(pkt, mb_id);
        complete_agent_dispatch_packet(pkt);
        packets_processed++;
        break;
      case AIR_PKT_TYPE_GET_INFO:
        handle_packet_get_info(pkt, mb_id);
        complete_agent_dispatch_packet(pkt);
        packets_processed++;
        break;

      case AIR_PKT_TYPE_XAIE_LOCK:
        handle_packet_xaie_lock(pkt);
        complete_agent_dispatch_packet(pkt);
        packets_processed++;
        break;

      case AIR_PKT_TYPE_ND_MEMCPY: // Only arrive here the first try. 
        uint16_t memory_space = (pkt->arg[0] >> 16) & 0x00ff;
        uint16_t channel      = (pkt->arg[0] >> 24) & 0x00ff;
        uint16_t direction    = (pkt->arg[0] >> 60) & 0x000f;
        uint16_t col  = (pkt->arg[0] >> 32) & 0x00ff;
        int slot = channel;
        slot += get_slot(col,memory_space); 
        if (direction == SHIM_DMA_S2MM) 
          slot += XAIEDMA_SHIM_CHNUM_S2MM0;
        else 
          slot += XAIEDMA_SHIM_CHNUM_MM2S0;
        int ret = stage_packet_nd_memcpy(pkt,slot,memory_space);
        if (ret == 0) { 
          last_slot = slot;
	        if (do_packet_memcpy(slot)) {
	          num_active_packets++;
	          break;
	        } // else completed the packet in the first try
        } else if (ret == 2) break; // slot busy, retry.
        staged_nd_slot[slot].valid = 0; 
        complete_agent_dispatch_packet(pkt); // this is correct for the first try or invalid stage 
        packets_processed++;
        break;

    } //switch
  } while (num_active_packets > 1);
  lock_uart(mb_id); air_printf("Completing: %d packets processed.\n\r",packets_processed); unlock_uart();
  queue_add_read_index(q,packets_processed);
}

inline signal_value_t signal_wait(volatile signal_t *signal,
                                  signal_value_t compare_value,
                                  uint64_t timeout_hint,
                                  signal_value_t default_value)
{
  if (signal->handle == 0) return default_value;
  signal_value_t ret = 0;
  uint64_t timeout = timeout_hint;
  do {
    ret = signal->handle;
    if (ret == compare_value)
      return compare_value;
  } while (timeout--);
  return ret;
}

void handle_barrier_and_packet(queue_t *q, uint32_t mb_id)
{
  uint64_t rd_idx = queue_load_read_index(q);
  barrier_and_packet_t *pkt = &((barrier_and_packet_t*)q->base_address)[mymod(rd_idx)];

  // TODO complete functionality with VAs
  signal_t *s0 = (signal_t *)pkt->dep_signal[0]; 
  signal_t *s1 = (signal_t *)pkt->dep_signal[1]; 
  signal_t *s2 = (signal_t *)pkt->dep_signal[2]; 
  signal_t *s3 = (signal_t *)pkt->dep_signal[3]; 
  signal_t *s4 = (signal_t *)pkt->dep_signal[4]; 

  //lock_uart(mb_id);
  //for (int i = 0; i < 5; i++)
  //  air_printf("MB %d : dep_signal[%d] @ %p\n\r",mb_id,i,(uint64_t *)(pkt->dep_signal[i]));
  //unlock_uart();

  while ((signal_wait(s0, 0, 0x80000, 0) != 0) ||
         (signal_wait(s1, 0, 0x80000, 0) != 0) ||
         (signal_wait(s2, 0, 0x80000, 0) != 0) ||
         (signal_wait(s3, 0, 0x80000, 0) != 0) ||
         (signal_wait(s4, 0, 0x80000, 0) != 0))
  {
    lock_uart(mb_id);
    air_printf("MB %d : barrier AND packet completion signal timeout!\n\r",mb_id);
    for (int i = 0; i < 5; i++)
      air_printf("MB %d : dep_signal[%d] = %d\n\r",mb_id,i,*((uint32_t *)(pkt->dep_signal[i])));
    unlock_uart();
  }

  complete_barrier_packet(pkt);
  queue_add_read_index(q,1);
}
  
void handle_barrier_or_packet(queue_t *q, uint32_t mb_id)
{
  uint64_t rd_idx = queue_load_read_index(q);
  barrier_or_packet_t *pkt = &((barrier_or_packet_t*)q->base_address)[mymod(rd_idx)];

  // TODO complete functionality with VAs
  signal_t *s0 = (signal_t *)pkt->dep_signal[0]; 
  signal_t *s1 = (signal_t *)pkt->dep_signal[1]; 
  signal_t *s2 = (signal_t *)pkt->dep_signal[2]; 
  signal_t *s3 = (signal_t *)pkt->dep_signal[3]; 
  signal_t *s4 = (signal_t *)pkt->dep_signal[4]; 

  //lock_uart(mb_id);
  //for (int i = 0; i < 5; i++)
  //  air_printf("MB %d : dep_signal[%d] @ %p\n\r",mb_id,i,(uint64_t *)(pkt->dep_signal[i]));
  //unlock_uart();

  while ((signal_wait(s0, 0, 0x80000, 1) != 0) &&
         (signal_wait(s1, 0, 0x80000, 1) != 0) &&
         (signal_wait(s2, 0, 0x80000, 1) != 0) &&
         (signal_wait(s3, 0, 0x80000, 1) != 0) &&
         (signal_wait(s4, 0, 0x80000, 1) != 0))
  {
    lock_uart(mb_id);
    air_printf("MB %d : barrier OR packet completion signal timeout!\n\r",mb_id);
    for (int i = 0; i < 5; i++)
      air_printf("MB %d : dep_signal[%d] = %d\n\r",mb_id,i,*((uint32_t *)(pkt->dep_signal[i])));
    unlock_uart();
  }

  complete_barrier_packet(pkt);
  queue_add_read_index(q,1);
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
  uint32_t *num_mbs = (uint32_t *)(shmem_base+0x208);
  num_mbs[0] = user1;

  if (mb_id==0) unlock_uart(); // NOTE: Initialize uart lock only from 'first' MB

  lock_uart(mb_id);
  xil_printf("MB %d of %d firmware %d.%d.%d created on %s at %s GMT\n\r",mb_id+1,*num_mbs,maj,min,ver,__DATE__, __TIME__); 
  xil_printf("(c) Copyright 2020-2021 Xilinx, Inc. All rights reserved.\n\r");
  unlock_uart();

  setup = false;
  queue_t *q = nullptr;
  queue_create(MB_QUEUE_SIZE, &q, mb_id);
  lock_uart(mb_id); xil_printf("Created queue @ 0x%llx\n\r", (size_t)q); unlock_uart();

  volatile bool done = false;
  while (!done) {
    if (q->doorbell+1 > q->last_doorbell) {
      lock_uart(mb_id); air_printf("Ding Dong 0x%llx\n\r", q->doorbell+1); unlock_uart();

      q->last_doorbell = q->doorbell+1;

      // process packets until we hit an invalid packet
      bool invalid = false;
      while (!invalid) {
        uint64_t rd_idx = queue_load_read_index(q);
        
        //air_printf("Handle pkt read_index=%d\n\r", rd_idx);

        dispatch_packet_t *pkt = &((dispatch_packet_t*)q->base_address)[mymod(rd_idx)];
        uint8_t type = ((pkt->header) & (0xF));
        //uint8_t type = ((pkt->header >> HSA_PACKET_HEADER_TYPE) &
        //                ((1 << HSA_PACKET_HEADER_WIDTH_TYPE) - 1));
        switch (type) {
          default:
          case HSA_PACKET_TYPE_INVALID:
            if (setup) {
              lock_uart(mb_id); air_printf("Waiting\n\r"); unlock_uart();
              setup = false;
            }
            invalid = true;
            break;
          case HSA_PACKET_TYPE_AGENT_DISPATCH:
	          handle_agent_dispatch_packet(q, mb_id);
            break;
          case HSA_PACKET_TYPE_BARRIER_AND:
            handle_barrier_and_packet(q, mb_id); 
            break;
          case HSA_PACKET_TYPE_BARRIER_OR:
            handle_barrier_or_packet(q, mb_id); 
            break;
        }
      }
    }
  }

  cleanup_platform();
  return 0;
}
