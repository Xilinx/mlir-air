
#include <cstdio>
#include <cassert>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>

#include <xaiengine.h>

//#include "acdc_agent.h"
#include "acdc_queue.h"
#include "hsa_defs.h"

#define SHMEM_BASE 0x020100000000LL

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

#include "aie_inc.cpp"

}

hsa_status_t queue_create(uint32_t size, uint32_t type, queue_t **queue)
{
  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1)
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;

  uint64_t *bram_ptr = (uint64_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, SHMEM_BASE);
  // I have no idea if this does anything
  __clear_cache((void*)bram_ptr, (void*)(bram_ptr+0x1000));
  //for (int i=0; i<20; i++)
  //  printf("%p %llx\n", &bram_ptr[i], bram_ptr[i]);

  printf("Opened shared memory paddr: %p vaddr: %p\n", SHMEM_BASE, bram_ptr);
  uint64_t q_paddr = bram_ptr[0];
  uint64_t q_offset = q_paddr - SHMEM_BASE;
  queue_t *q = (queue_t*)( ((size_t)bram_ptr) + q_offset );
  printf("Queue location at paddr: %p vaddr: %p\n", bram_ptr[0], q);

  if (q->id !=  0xacdc) {
    printf("%s error invalid id %x\n", __func__, q->id);
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
  }

  if (q->size != size) {
    printf("%s error size mismatch %d\n", __func__, q->size);
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
  }

  if (q->type != type) {
    printf("%s error type mismatch %d\n", __func__, q->type);
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
  }

  uint64_t base_address_offset = q->base_address - SHMEM_BASE;
  q->base_address_vaddr = ((size_t)bram_ptr) + base_address_offset;

  *queue = q;
  return HSA_STATUS_SUCCESS;
}

void printDMAStatus(int col, int row) {


  u32 dma_mm2s_status = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001DF10);
  u32 dma_s2mm_status = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001DF00);
  u32 dma_mm2s_control = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001DE10);
  u32 dma_s2mm_control = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001DE00);
  u32 dma_bd0_a       = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001D000); 
  u32 dma_bd0_control = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001D018);

  u32 s2mm_ch0_running = dma_s2mm_status & 0x3;
  u32 s2mm_ch1_running = (dma_s2mm_status >> 2) & 0x3;
  u32 mm2s_ch0_running = dma_mm2s_status & 0x3;
  u32 mm2s_ch1_running = (dma_mm2s_status >> 2) & 0x3;

  printf("DMA [%d, %d] mm2s_status/ctrl is %08X %08X, s2mm_status is %08X %08X, BD0_Addr_A is %08X, BD0_control is %08X\n",col, row, dma_mm2s_status, dma_mm2s_control, dma_s2mm_status, dma_s2mm_control, dma_bd0_a, dma_bd0_control);
  for (int bd=0;bd<8;bd++) {
      u32 dma_bd_addr_a        = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001D000 + (0x20*bd));
      u32 dma_bd_control       = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001D018 + (0x20*bd));
    if (dma_bd_control & 0x80000000) {
      printf("BD %d valid\n",bd);
      int current_s2mm_ch0 = (dma_s2mm_status >> 16) & 0xf;  
      int current_s2mm_ch1 = (dma_s2mm_status >> 20) & 0xf;  
      int current_mm2s_ch0 = (dma_mm2s_status >> 16) & 0xf;  
      int current_mm2s_ch1 = (dma_mm2s_status >> 20) & 0xf;  

      if (s2mm_ch0_running && bd == current_s2mm_ch0) {
        printf(" * Current BD for s2mm channel 0\n");
      }
      if (s2mm_ch1_running && bd == current_s2mm_ch1) {
        printf(" * Current BD for s2mm channel 1\n");
      }
      if (mm2s_ch0_running && bd == current_mm2s_ch0) {
        printf(" * Current BD for mm2s channel 0\n");
      }
      if (mm2s_ch1_running && bd == current_mm2s_ch1) {
        printf(" * Current BD for mm2s channel 1\n");
      }

      if (dma_bd_control & 0x08000000) {
        u32 dma_packet = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001D010 + (0x20*bd));
        printf("   Packet mode: %02X\n",dma_packet & 0x1F);
      }
      int words_to_transfer = 1+(dma_bd_control & 0x1FFF);
      int base_address = dma_bd_addr_a  & 0x1FFF;
      printf("   Transfering %d 32 bit words to/from %05X\n",words_to_transfer, base_address);

      printf("   ");
      for (int w=0;w<4; w++) {
        printf("%08X ",XAieTile_DmReadWord(&(TileInst[col][row]), (base_address+w) * 4));
      }
      printf("\n");
      if (dma_bd_addr_a & 0x40000) {
        u32 lock_id = (dma_bd_addr_a >> 22) & 0xf;
        printf("   Acquires lock %d ",lock_id);
        if (dma_bd_addr_a & 0x10000) 
          printf("with value %d ",(dma_bd_addr_a >> 17) & 0x1);

        printf("currently ");
        u32 locks = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001EF00);
        u32 two_bits = (locks >> (lock_id*2)) & 0x3;
        if (two_bits) {
          u32 acquired = two_bits & 0x1;
          u32 value = two_bits & 0x2;
          if (acquired)
            printf("Acquired ");
          printf(value?"1":"0");
        }
        else printf("0");
        printf("\n");

      }
      if (dma_bd_control & 0x30000000) { // FIFO MODE
        int FIFO = (dma_bd_control >> 28) & 0x3;
          u32 dma_fifo_counter = XAieGbl_Read32(TileInst[col][row].TileAddr + 0x0001DF20);				
        printf("   Using FIFO Cnt%d : %08X\n",FIFO, dma_fifo_counter);
      }
    }

  }

}

void setup_aie(void)
{
  auto col = 7;

  size_t aie_base = XAIE_ADDR_ARRAY_OFF << 14;
  XAIEGBL_HWCFG_SET_CONFIG((&AieConfig), XAIE_NUM_ROWS, XAIE_NUM_COLS, XAIE_ADDR_ARRAY_OFF);
  XAieGbl_HwInit(&AieConfig);
  AieConfigPtr = XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
  XAieGbl_CfgInitialize(&AieInst, &TileInst[0][0], AieConfigPtr);

  // reset cores and locks
  for (int i = 1; i <= XAIE_NUM_ROWS; i++) {
    for (int j = 0; j < XAIE_NUM_COLS; j++) {
      XAieTile_CoreControl(&(TileInst[j][i]), XAIE_DISABLE, XAIE_ENABLE);
      for (int l=0; l<16; l++)
        XAieTile_LockRelease(&(TileInst[j][i]), l, 0x0, 0);
    }
  }
  // cores
  //

  //  mlir_initialize_cores();
  XAieTile_CoreControl(&(TileInst[col][1]), XAIE_DISABLE, XAIE_DISABLE);
  XAieTile_CoreControl(&(TileInst[col][2]), XAIE_DISABLE, XAIE_DISABLE);

  XAieTile_ShimColumnReset(&(TileInst[col][0]), XAIE_RESETENABLE);
  XAieTile_ShimColumnReset(&(TileInst[col][0]), XAIE_RESETDISABLE);

  // configure switchboxes
  //
  mlir_configure_switchboxes();
  XAieTile_ShimStrmMuxConfig(&(TileInst[col][0]), XAIETILE_SHIM_STRM_MUX_SOUTH7, XAIETILE_SHIM_STRM_MUX_DMA);

  // locks
  //
  //mlir_initialize_locks();

  // dmas
  //

  XAieTile_LockRelease(&(TileInst[col][2]), 0, 1, 0);
  u8 lock_ret = XAieTile_LockAcquire(&(TileInst[col][2]), 0, 1, 10000);
  assert(lock_ret);

  mlir_configure_dmas();
}

int main(int argc, char *argv[])
{
  auto col = 7;

  // setup the aie array
  setup_aie();

  XAieTile_DmWriteWord(&(TileInst[col][2]), 24*4, -1);

  // create the queue
  queue_t *q = nullptr;
  auto ret = queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q);
  assert(ret == 0 && "failed to create queue!");

  // setup the shim dma descriptors
  XAieDma_Shim ShimDmaInst1;
  uint32_t *bram_ptr;

  #define DMA_COUNT 256

  auto burstlen = 4;
  XAieDma_ShimInitialize(&(TileInst[col][0]), &ShimDmaInst1);
  XAieDma_ShimBdSetAddr(&ShimDmaInst1, 1, HIGH_ADDR((u64)SHMEM_BASE), LOW_ADDR((u64)SHMEM_BASE), sizeof(u32) * DMA_COUNT);
  XAieDma_ShimBdSetAxi(&ShimDmaInst1, 1 , 0, burstlen, 0, 0, XAIE_ENABLE);
  XAieDma_ShimBdWrite(&ShimDmaInst1, 1);
  XAieDma_ShimSetStartBd((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_MM2S1, 1);

  auto cnt = XAieDma_ShimPendingBdCount(&ShimDmaInst1, XAIEDMA_SHIM_CHNUM_MM2S1);
  if (cnt)
    printf("%s %d Warn %d\n", __FUNCTION__, __LINE__, cnt);

  XAieDma_ShimChControl((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_MM2S1, XAIE_DISABLE, XAIE_DISABLE, XAIE_ENABLE);

  // reserve a packet in the queue
  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;

  // setup packet
  dispatch_packet_t *pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  initialize_packet(pkt);
  pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;

  // release lock 0 with value 0
  pkt->arg[0] = 0x00beef00;
  pkt->arg[1] = 0;
  pkt->arg[2] = 1;
  pkt->arg[3] = 0;

  // dispatch packet
  signal_create(1, 0, NULL, (signal_t*)&pkt->completion_signal);
  signal_create(0, 0, NULL, (signal_t*)&q->doorbell);
  signal_store_release((signal_t*)&q->doorbell, wr_idx);

  // wait for packet completion
  while (signal_wait_aquire((signal_t*)&pkt->completion_signal, HSA_SIGNAL_CONDITION_EQ, 0, 0x80000, HSA_WAIT_STATE_ACTIVE) != 0) {
    printf("packet completion signal timeout!\n");
    printf("%x\n", pkt->header);
    printf("%x\n", pkt->type);
    printf("%x\n", pkt->completion_signal);
  }

  //XAieTile_LockRelease(&(TileInst[col][2]), 0, 0, 0);

  // wait for shim dma to finish
  auto count = 0;
  while (XAieDma_ShimPendingBdCount(&ShimDmaInst1, XAIEDMA_SHIM_CHNUM_MM2S1)) {
    XAieLib_usleep(1000);
    count++;
    if (!(count % 1000)) {
      printf("%d seconds\n",count/1000);
      if (count == 2000) break;
    }
  }

  // we copied the start of the shared bram into tile memory,
  // fish out the queue id and check it
  uint32_t d = XAieTile_DmReadWord(&(TileInst[col][2]), 24*4);
  printf("ID %x\n", d);

  if (d == 0xacdc)
    printf("PASS!\n");
  else
    printf("fail.\n");
}