//===- main.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "unistd.h"
#include <cstdint>
#include <cstring>

#ifdef __aarch64__
#define ARM_CONTROLLER 1
#endif

extern "C" {
#include "xil_printf.h"
#ifdef ARM_CONTROLLER
#include "xaiengine.h"
#include "xil_cache.h"
#else
#include "pvr.h"
#endif

// #include "mb_interface.h"

#include "air_queue.h"
#include "hsa_defs.h"
}

#include "platform.h"
#include "shell.h"

#define XAIE_NUM_ROWS 8
#define XAIE_NUM_COLS 50

#ifdef ARM_CONTROLLER
#define XAIE_ADDR_ARRAY_OFF 0
#else
#define XAIE_ADDR_ARRAY_OFF 0x800ULL
#endif // ARM_CONTROLLER

#define NUM_SHIM_DMA_S2MM_CHANNELS 2
#define NUM_SHIM_DMA_MM2S_CHANNELS 2
#define XAIEDMA_SHIM_CHNUM_S2MM0 0U
#define XAIEDMA_SHIM_CHNUM_S2MM1 1U
#define XAIEDMA_SHIM_CHNUM_MM2S0 2U
#define XAIEDMA_SHIM_CHNUM_MM2S1 3U

#define HIGH_ADDR(addr) ((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr) (addr & 0x00000000ffffffff)

#define ALIGN(_x, _size) (((_x) + ((_size)-1)) & ~((_size)-1))

#define LOGICAL_HERD_DMAS 16

// direction
#define SHIM_DMA_S2MM 0
#define SHIM_DMA_MM2S 1

#define NUM_SHIM_DMAS 16
#define NUM_COL_DMAS 4
int shim_dma_cols[NUM_SHIM_DMAS] = {2,  3,  6,  7,  10, 11, 18, 19,
                                    26, 27, 34, 35, 42, 43, 46, 47};
int col_dma_cols[NUM_COL_DMAS] = {7, 8, 9, 10};
#define NUM_DMAS (NUM_SHIM_DMAS + NUM_COL_DMAS)

/*
 * Tile address format:
 * --------------------------------------------
 * |                7 bits  5 bits   18 bits  |
 * --------------------------------------------
 * | Array offset | Column | Row | Tile addr  |
 * --------------------------------------------
 */
#define AIE_TILE_WIDTH 18
#define AIE_ROW_WIDTH 5
#define AIE_COLUMN_WIDTH 7

#define AIE_ROW_SHIFT (AIE_TILE_WIDTH)
#define AIE_COLUMN_SHIFT (AIE_TILE_WIDTH + AIE_ROW_WIDTH)
#define AIE_ARRAY_SHIFT (AIE_TILE_WIDTH + AIE_ROW_WIDTH + AIE_COLUMN_WIDTH)
#define AIE_TILE_MASK ((1 << AIE_TILE_WIDTH) - 1)
#define AIE_ROW_MASK ((1 << AIE_ROW_WIDTH) - 1)
#define AIE_COLUMN_MASK ((1 << AIE_COLUMN_WIDTH) - 1)

#define GET_COLUMN(_addr) (((_addr) >> AIE_COLUMN_SHIFT) & AIE_COLUMN_MASK)
#define GET_ROW(_addr) (((_addr) >> AIE_ROW_SHIFT) & AIE_ROW_MASK)
#define GET_TILE(_addr) ((_addr) & AIE_TILE_MASK)

#define SHIM_DMA_NUM_BDS 16

// AIE (ME) registers
#define REG_AIE_DMA_BD_ADDR_A(_idx) (0x1D000 + (0x20 * _idx))
#define REG_AIE_DMA_BD_ADDR_B(_idx) (0x1D004 + (0x20 * _idx))
#define AIE_DMA_BD_ADDR_LOCK (0xFUL << 22)
#define AIE_DMA_BD_ADDR_ENA_REL (1UL << 21)
#define AIE_DMA_BD_ADDR_REL_VAL (1UL << 20)
#define AIE_DMA_BD_ADDR_USE_REL_VAL (1UL << 19)
#define AIE_DMA_BD_ADDR_ENA_ACQ (1UL << 18)
#define AIE_DMA_BD_ADDR_ACQ_VAL (1UL << 17)
#define AIE_DMA_BD_ADDR_USE_ACQ_VAL (1UL << 16)
#define AIE_DMA_BD_ADDR_BASE (0x1FFFUL << 0)

#define REG_AIE_DMA_BD_2D_X(_idx) (0x1D008 + (0x20 * _idx))
#define REG_AIE_DMA_BD_2D_Y(_idx) (0x1D00C + (0x20 * _idx))
#define REG_AIE_DMA_BD_PKT(_idx) (0x1D010 + (0x20 * _idx))
#define AIE_DMA_BD_PKT_TYPE (0x3UL << 12)
#define AIE_DMA_BD_PKT_ID (0x1FUL << 0)

#define REG_AIE_DMA_BD_IS(_idx) (0x1D014 + (0x20 * _idx))
#define REG_AIE_DMA_BD_CTL(_idx) (0x1D018 + (0x20 * _idx))
#define AIE_DMA_BD_CTL_VALID (1UL << 31)
#define AIE_DMA_BD_CTL_ENA_AB (1UL << 30)
#define AIE_DMA_BD_CTL_ENA_FIFO (3UL << 28)
#define AIE_DMA_BD_CTL_ENA_PKT (1UL << 27)
#define AIE_DMA_BD_CTL_ENA_ILV (1UL << 26)
#define AIE_DMA_BD_CTL_ILV_CNT (0xFFUL << 18)
#define AIE_DMA_BD_CTL_USE_NEXT (1UL << 17)
#define AIE_DMA_BD_CTL_NEXT (0xFUL << 13)
#define AIE_DMA_BD_CTL_LEN (0x1FFFUL << 0)

#define REG_AIE_LOCK_RELEASE_0(_idx) (0x1E020 + (0x80 * _idx))
#define REG_AIE_CORE_CTL 0x00032000
#define REG_AIE_CORE_STATUS 0x00032004

// NoC (shim) registers
#define REG_SHIM_DMA_BD_ADDR(_idx) (0x1D000 + (0x14 * _idx))
#define REG_SHIM_DMA_BD_BUF_LEN(_idx) (0x1D004 + (0x14 * _idx))
#define REG_SHIM_DMA_BD_CTRL(_idx) (0x1D008 + (0x14 * _idx))
#define SHIM_DMA_BD_CTRL_VALID (1 << 0)

#define REG_SHIM_DMA_BD_AXI_CFG(_idx) (0x1D00C + (0x14 * _idx))
#define REG_SHIM_DMA_BD_PKT(_idx) (0x1D010 + (0x14 * _idx))
#define REG_SHIM_DMA_CTRL(_chan) (0x1D140 + (0x8 *_chan))
#define REG_SHIM_DMA_START_QUEUE(_chan) (0x1D144 + (0x8 *_chan))

#define REG_SHIM_DMA_S2MM_STATUS (0x1D160)
#define SHIM_DMA_CURR_BD_SHIFT 16
#define SHIM_DMA_CURR_BD_WIDTH 4
#define SHIM_DMA_CURR_BD_MASK ((1 << SHIM_DMA_CURR_BD_WIDTH) - 1)
#define SHIM_DMA_QUEUE_SIZE_SHIFT 6
#define SHIM_DMA_QUEUE_SIZE_WIDTH 3
#define SHIM_DMA_QUEUE_SIZE_MASK ((1 << SHIM_DMA_QUEUE_SIZE_WIDTH) - 1)
#define SHIM_DMA_STATUS_SHIFT 0
#define SHIM_DMA_STATUS_WIDTH 2
#define SHIM_DMA_STATUS_MASK ((1 << SHIM_DMA_STATUS_WIDTH) - 1)
#define SHIM_DMA_STALLED_SHIFT 4
#define SHIM_DMA_STALLED_WIDTH 1
#define SHIM_DMA_STALLED_MASK 1
#define GET_SHIM_DMA(_field, _reg, _ch) ((_reg) >> (SHIM_DMA_##_field##_SHIFT + (SHIM_DMA_##_field##_WIDTH * (_ch))) & SHIM_DMA_##_field##_MASK)

#define REG_SHIM_DMA_MM2S_STATUS (0x1D164)

#define REG_AIE_COL_RESET 0x00036048
#define REG_SHIM_RESET_ENA 0x0003604C

#define REG_AIE_CORE_CTL_RESET (1U << 1)
#define REG_AIE_CORE_CTL_ENABLE (1U << 0)

#define CHATTY 0

#define air_printf(fmt, ...)                                                   \
  do {                                                                         \
    if (CHATTY)                                                                \
      xil_printf(fmt, ##__VA_ARGS__);                                          \
  } while (0)

inline uint64_t mymod(uint64_t a) {
  uint64_t result = a;
  while (result >= MB_QUEUE_SIZE) {
    result -= MB_QUEUE_SIZE;
  }
  return result;
}

constexpr uint32_t NUM_BD = 16;

#ifdef ARM_CONTROLLER
// The NPI registers we use to reset the array
constexpr auto NPI_MASK_REG = 0x0;
constexpr auto NPI_VAL_REG = 0x4;
constexpr auto NPI_LOCK_REG = 0xC;
#endif // ARM_CONTROLLER

struct HerdConfig {
  uint32_t row_start;
  uint32_t num_rows;
  uint32_t col_start;
  uint32_t num_cols;
};

HerdConfig HerdCfgInst;

#ifdef ARM_CONTROLLER
struct aie_libxaie_ctx_t {
  XAie_Config AieConfigPtr;
  XAie_DevInst DevInst;
};

aie_libxaie_ctx_t *_xaie;

/*
  This is the cheap way to share addresses between the host userspace and the
  device. Addresses are translated across 4 different address spaces, making it
  quite difficult to share addresses transparently. If we assume that regions
  are physically contiguous (i.e. no paging) we can use the offset from some
  known base address.

  Address spaces:
  device | PCIe | host kernel | userspace

  This follows the same rules as the driver.
*/
uint64_t offset_to_phys(uint64_t offset) {
  return (uint64_t) (DRAM_1_BASE | offset);
}

/*
  read 32 bit value from specified address
*/
static inline u32 in32(u64 Addr) {
  return *(volatile u32 *)((XAIE_ADDR_ARRAY_OFF << AIE_ARRAY_SHIFT) | Addr);
}

/*
  write 32 bit value to specified address
*/
static inline void out32(u64 Addr, u32 Value) {
  volatile u32 *LocalAddr = (volatile u32 *)((XAIE_ADDR_ARRAY_OFF << AIE_ARRAY_SHIFT) | Addr);
  *LocalAddr = Value;
}

u32 maskpoll32(u64 Addr, u32 Mask, u32 Value, u32 TimeOut) {
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

/*
  Calculate the address of an AIE tile
*/
u64 getTileAddr(u16 ColIdx, u16 RowIdx) {
#ifdef ARM_CONTROLLER
  u64 aie_ta = _XAie_GetTileAddr(&(_xaie->DevInst), RowIdx, ColIdx);
  xil_printf("AIE tile address=0x%lx\r\n", aie_ta);
  u64 my_ta = (u64)((XAIE_ADDR_ARRAY_OFF << AIE_ARRAY_SHIFT) |
                   (ColIdx << AIE_COLUMN_SHIFT) |
                   (RowIdx << AIE_ROW_SHIFT));
  xil_printf("MY tile address=0x%lx\r\n", my_ta);
  return aie_ta;
#else
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
  TileAddr = (u64)((ArrOffset << XAIEGBL_TILE_ADDR_ARR_SHIFT) |
                   (ColIdx << XAIEGBL_TILE_ADDR_COL_SHIFT) |
                   (RowIdx << XAIEGBL_TILE_ADDR_ROW_SHIFT));

  return TileAddr;
#endif
}

static const char *decode_dma_state(uint32_t state) {
   switch (state) {
   case 0:
      return "idle";
   case 1:
      return "starting";
   case 2:
      return "running";
   }
   return "unknown";
}


void mlir_aie_init_libxaie(aie_libxaie_ctx_t *ctx) {
  if (!ctx)
    return;

  ctx->AieConfigPtr.AieGen = XAIE_DEV_GEN_AIE;
  ctx->AieConfigPtr.BaseAddr = 0x20000000000; // XAIE_BASE_ADDR;
  ctx->AieConfigPtr.ColShift = AIE_COLUMN_SHIFT;
  ctx->AieConfigPtr.RowShift = AIE_ROW_SHIFT;
  ctx->AieConfigPtr.NumRows = XAIE_NUM_ROWS + 1;
  ctx->AieConfigPtr.NumCols = XAIE_NUM_COLS;
  ctx->AieConfigPtr.ShimRowNum = 0;      // XAIE_SHIM_ROW;
  ctx->AieConfigPtr.MemTileRowStart = 0; // XAIE_RES_TILE_ROW_START;
  ctx->AieConfigPtr.MemTileNumRows = 0;  // XAIE_RES_TILE_NUM_ROWS;
  ctx->AieConfigPtr.AieTileRowStart = 1; // XAIE_AIE_TILE_ROW_START;
  ctx->AieConfigPtr.AieTileNumRows = XAIE_NUM_ROWS;
  ctx->AieConfigPtr.PartProp = {0};
  ctx->DevInst = {0};
}

int mlir_aie_init_device(aie_libxaie_ctx_t *ctx) {
  AieRC RC = XAIE_OK;

  RC = XAie_CfgInitialize(&(ctx->DevInst), &(ctx->AieConfigPtr));
  if (RC != XAIE_OK) {
    xil_printf("Driver initialization failed.\n\r");
    return -1;
  }

  RC = XAie_PmRequestTiles(&(ctx->DevInst), NULL, 0);
  if (RC != XAIE_OK) {
    xil_printf("Failed to request tiles.\n\r");
    return -1;
  }

  // TODO Extra code to really teardown the segments
  RC = XAie_Finish(&(ctx->DevInst));
  if (RC != XAIE_OK) {
    xil_printf("Failed to finish tiles.\n\r");
    return -1;
  }
  RC = XAie_CfgInitialize(&(ctx->DevInst), &(ctx->AieConfigPtr));
  if (RC != XAIE_OK) {
    xil_printf("Driver initialization failed.\n\r");
    return -1;
  }
  RC = XAie_PmRequestTiles(&(ctx->DevInst), NULL, 0);
  if (RC != XAIE_OK) {
    xil_printf("Failed to request tiles.\n\r");
    return -1;
  }

  return 0;
}

int mlir_aie_reinit_device(aie_libxaie_ctx_t *ctx) {
  AieRC RC = XAIE_OK;

  RC = XAie_Finish(&(ctx->DevInst));
  if (RC != XAIE_OK) {
    xil_printf("Failed to finish tiles.\n\r");
    return -1;
  }
  RC = XAie_CfgInitialize(&(ctx->DevInst), &(ctx->AieConfigPtr));
  if (RC != XAIE_OK) {
    xil_printf("Driver initialization failed.\n\r");
    return -1;
  }
  RC = XAie_PmRequestTiles(&(ctx->DevInst), NULL, 0);
  if (RC != XAIE_OK) {
    xil_printf("Failed to request tiles.\n\r");
    return -1;
  }

  return 0;
}

void mlir_aie_print_dma_status(aie_libxaie_ctx_t *ctx, int col, int row) {
  u64 tileAddr = _XAie_GetTileAddr(&(ctx->DevInst), row, col);

  u32 dma_s2mm0_control = in32(tileAddr + 0x0001DE00);
  u32 dma_s2mm1_control = in32(tileAddr + 0x0001DE08);
  u32 dma_mm2s0_control = in32(tileAddr + 0x0001DE10);
  u32 dma_mm2s1_control = in32(tileAddr + 0x0001DE18);
  u32 dma_s2mm_status = in32(tileAddr + 0x0001DF00);
  u32 dma_mm2s_status = in32(tileAddr + 0x0001DF10);

  u32 s2mm_ch0_running = dma_s2mm_status & 0x3;
  u32 s2mm_ch1_running = (dma_s2mm_status >> 2) & 0x3;
  u32 mm2s_ch0_running = dma_mm2s_status & 0x3;
  u32 mm2s_ch1_running = (dma_mm2s_status >> 2) & 0x3;

  xil_printf("DMA [%d, %d] tile addr=0x%lx\r\n", col, row, tileAddr);
  xil_printf("  mm2s (0=%s 1=%s) status=%08X ctrl0=%02X ctrl1=%02X\r\n",
             decode_dma_state(mm2s_ch0_running), decode_dma_state(mm2s_ch1_running),
               dma_mm2s_status, dma_mm2s0_control, dma_mm2s1_control);
  xil_printf("  s2mm (0=%s 1=%s) status=%08X ctrl0=%02X ctrl1=%02X\r\n",
             decode_dma_state(s2mm_ch0_running), decode_dma_state(s2mm_ch1_running),
               dma_s2mm_status, dma_s2mm0_control, dma_s2mm1_control);

  xil_printf("Descriptors:\r\n");
  for (uint32_t bd = 0; bd < NUM_BD; bd++) {
    u32 dma_bd_addr_a = in32(tileAddr + REG_AIE_DMA_BD_ADDR_A(bd));
    u32 dma_bd_control = in32(tileAddr + REG_AIE_DMA_BD_CTL(bd));
    if (dma_bd_control & AIE_DMA_BD_CTL_VALID) {
      xil_printf("BD %d valid\n\r", bd);
      u32 current_s2mm_ch0 = (dma_s2mm_status >> 16) & 0xf;
      u32 current_s2mm_ch1 = (dma_s2mm_status >> 20) & 0xf;
      u32 current_mm2s_ch0 = (dma_mm2s_status >> 16) & 0xf;
      u32 current_mm2s_ch1 = (dma_mm2s_status >> 20) & 0xf;

      if (s2mm_ch0_running && bd == current_s2mm_ch0) {
        xil_printf(" * Current BD for s2mm channel 0\n\r");
      }
      if (s2mm_ch1_running && bd == current_s2mm_ch1) {
        xil_printf(" * Current BD for s2mm channel 1\n\r");
      }
      if (mm2s_ch0_running && bd == current_mm2s_ch0) {
        xil_printf(" * Current BD for mm2s channel 0\n\r");
      }
      if (mm2s_ch1_running && bd == current_mm2s_ch1) {
        xil_printf(" * Current BD for mm2s channel 1\n\r");
      }

      if (dma_bd_control & AIE_DMA_BD_CTL_ENA_PKT) {
        u32 dma_packet = in32(tileAddr + REG_AIE_DMA_BD_PKT(bd));
        xil_printf("   Packet mode: %02X\n\r", dma_packet & AIE_DMA_BD_PKT_ID);
      }
      int words_to_transfer = 1 + (dma_bd_control & AIE_DMA_BD_CTL_LEN);
      int base_address = dma_bd_addr_a & AIE_DMA_BD_ADDR_BASE;
      xil_printf("   Transfering %d 32 bit words to/from %06X\n\r",
                 words_to_transfer, base_address);

      xil_printf("   ");
      for (int w = 0; w < 7; w++) {
        u32 tmpd = in32(tileAddr + (base_address << 2) + (w * 4));
        xil_printf("%08X ", tmpd);
      }
      xil_printf("\n\r");
      if (dma_bd_addr_a & AIE_DMA_BD_ADDR_ENA_ACQ) {
        u32 lock_id = (dma_bd_addr_a >> 22) & 0xf;
        xil_printf("   Acquires lock %d ", lock_id);
        if (dma_bd_addr_a & 0x10000)
          xil_printf("with value %d ", (dma_bd_addr_a >> 17) & 0x1);

        xil_printf("currently ");
        u32 locks = in32(tileAddr + 0x0001EF00);
        u32 two_bits = (locks >> (lock_id * 2)) & 0x3;
        if (two_bits) {
          u32 acquired = two_bits & 0x1;
          u32 value = two_bits & 0x2;
          if (acquired)
            xil_printf("Acquired ");
          xil_printf(value ? "1" : "0");
        } else
          xil_printf("0");
        xil_printf("\n\r");
      }
      if (dma_bd_control & 0x30000000) { // FIFO MODE
        int FIFO = (dma_bd_control >> 28) & 0x3;
        u32 dma_fifo_counter = in32(tileAddr + 0x0001DF20);
        xil_printf("   Using FIFO Cnt%d : %08X\n\r", FIFO, dma_fifo_counter);
      }
      u32 nextBd = ((dma_bd_control >> 13) & 0xF);
      u32 useNextBd = ((dma_bd_control >> 17) & 0x1);
      xil_printf("   Next BD: %d %s\r\n", nextBd, (useNextBd == 0) ? "(unused)" : "(used)");
    }
  }
}

void mlir_aie_print_shimdma_status(aie_libxaie_ctx_t *ctx, int col, int row) {
  uint64_t tileAddr = _XAie_GetTileAddr(&(ctx->DevInst), 0, col);
  uint32_t s2mm_status = in32(tileAddr + REG_SHIM_DMA_S2MM_STATUS);
  uint32_t mm2s_status = in32(tileAddr + REG_SHIM_DMA_MM2S_STATUS);

  xil_printf("Shim DMA [%u]\r\n", col);
  xil_printf("S2MM\r\n");
  for (uint8_t channel=0; channel < NUM_SHIM_DMA_S2MM_CHANNELS; channel++) {
    xil_printf("   [channel %u] start_bd=%u queue_size=%u curr_bd=%u status=%s stalled=%s\r\n",
                channel,
                in32(tileAddr + REG_SHIM_DMA_START_QUEUE(channel)),
                GET_SHIM_DMA(QUEUE_SIZE, s2mm_status, channel),
                GET_SHIM_DMA(CURR_BD, s2mm_status, channel),
                GET_SHIM_DMA(STATUS, s2mm_status, channel),
                GET_SHIM_DMA(STALLED, s2mm_status, channel));
  }
  xil_printf("MM2S\r\n");
  for (uint8_t channel=0; channel < NUM_SHIM_DMA_MM2S_CHANNELS; channel++) {
    xil_printf("   [channel %u] start_bd=%u queue_size=%u curr_bd=%u status=%s stalled=%s\r\n",
                channel,
                in32(tileAddr + REG_SHIM_DMA_START_QUEUE(channel)),
                GET_SHIM_DMA(QUEUE_SIZE, mm2s_status, channel),
                GET_SHIM_DMA(CURR_BD, mm2s_status, channel),
                GET_SHIM_DMA(STATUS, mm2s_status, channel),
                GET_SHIM_DMA(STALLED, mm2s_status, channel));
  }

  xil_printf("Descriptors:\r\n");
  for (int bd = 0; bd < 16; bd++) {
    u64 bd_addr_a = in32(tileAddr + REG_SHIM_DMA_BD_ADDR(bd));
    u32 dma_bd_buffer_length = in32(tileAddr + REG_SHIM_DMA_BD_BUF_LEN(bd));
    u32 dma_bd_control = in32(tileAddr + REG_SHIM_DMA_BD_CTRL(bd));

    xil_printf("[%02d] ", bd);
    if (dma_bd_control & SHIM_DMA_BD_CTRL_VALID)
      xil_printf("valid ");

    int words_to_transfer = dma_bd_buffer_length;
    u64 base_address =
        (u64)bd_addr_a + ((u64)((dma_bd_control >> 16) & 0xFFFF) << 32);
    xil_printf("   Transferring %d 32 bit words to/from %08lX\n\r",
               words_to_transfer, base_address);

    int use_next_bd = ((dma_bd_control >> 15) & 0x1);
    int next_bd = ((dma_bd_control >> 11) & 0xF);
    int lockID = ((dma_bd_control >> 7) & 0xF);
    int enable_lock_release = ((dma_bd_control >> 6) & 0x1);
    int lock_release_val = ((dma_bd_control >> 5) & 0x1);
    int use_release_val = ((dma_bd_control >> 4) & 0x1);
    int enable_lock_acquire = ((dma_bd_control >> 3) & 0x1);
    int lock_acquire_val = ((dma_bd_control >> 2) & 0x1);
    int use_acquire_val = ((dma_bd_control >> 1) & 0x1);

    xil_printf("next=%d, use_next=%d ", next_bd, use_next_bd);
    xil_printf(
        "lock: %d, acq(en: %d, val: %d, use: %d), rel(en: %d, val: %d, "
        "use: %d)\r\n",
        lockID, enable_lock_acquire, lock_acquire_val, use_acquire_val,
        enable_lock_release, lock_release_val, use_release_val);
  }
}

/// Print the status of a core represented by the given tile, at the given
/// coordinates.
void mlir_aie_print_tile_status(aie_libxaie_ctx_t *ctx, int col, int row) {
  u32 trace_status;
  u32 status, coreTimerLow, PC, LR, SP, locks, R0, R4;
  u64 tileAddr = getTileAddr(col, row);

  status = in32(tileAddr + REG_AIE_CORE_STATUS);
  coreTimerLow = in32(tileAddr + 0x0340F8);
  PC = in32(tileAddr + 0x00030280);
  LR = in32(tileAddr + 0x000302B0);
  SP = in32(tileAddr + 0x000302A0);
  locks = in32(tileAddr + 0x0001EF00);
  trace_status = in32(tileAddr + 0x000140D8);
  R0 = in32(tileAddr + 0x00030000);
  R4 = in32(tileAddr + 0x00030040);

  xil_printf("Core [%d, %d] addr is 0x%08lX\n\r", col, row, tileAddr);
  xil_printf("Core [%d, %d] status is 0x%08X, timer is %u, PC is 0x%08X, locks are "
             "%08X, LR is %08X, SP is %08X, R0 is %08X,R4 is %08X\n\r",
             col, row, status, coreTimerLow, PC, locks, LR, SP, R0, R4);
  xil_printf("Core [%d, %d] trace status is %08X\n\r", col, row, trace_status);

  for (int lock = 0; lock < 16; lock++) {
    u32 two_bits = (locks >> (lock * 2)) & 0x3;
    if (two_bits) {
      xil_printf("Lock %d: ", lock);
      u32 acquired = two_bits & 0x1;
      u32 value = two_bits & 0x2;
      if (acquired)
        xil_printf("Acquired ");
      xil_printf(value ? "1" : "0");
      xil_printf("\n\r");
    }
  }

  const char *core_status_strings[] = {"Enabled",
                                       "In Reset",
                                       "Memory Stall S",
                                       "Memory Stall W",
                                       "Memory Stall N",
                                       "Memory Stall E",
                                       "Lock Stall S",
                                       "Lock Stall W",
                                       "Lock Stall N",
                                       "Lock Stall E",
                                       "Stream Stall S",
                                       "Stream Stall W",
                                       "Stream Stall N",
                                       "Stream Stall E",
                                       "Cascade Stall Master",
                                       "Cascade Stall Slave",
                                       "Debug Halt",
                                       "ECC Error",
                                       "ECC Scrubbing",
                                       "Error Halt",
                                       "Core Done"};
  xil_printf("Core Status: ");
  for (int i = 0; i <= 20; i++) {
    if ((status >> i) & 0x1)
      xil_printf("%s ", core_status_strings[i]);
  }
  xil_printf("\r\n");
}

#endif

int xaie_shim_dma_wait_idle(uint64_t TileAddr, int direction, int channel) {
  uint32_t shimDMAchannel = channel;
  uint32_t status_register_offset;
  uint32_t status_mask_shift;
  if (channel == 0) {
    status_mask_shift = 0;
  } else {
    status_mask_shift = 2;
  }
  if (direction == SHIM_DMA_S2MM) {
    shimDMAchannel += XAIEDMA_SHIM_CHNUM_S2MM0;
    status_register_offset = 0x1d160;
  } else {
    shimDMAchannel += XAIEDMA_SHIM_CHNUM_MM2S0;
    status_register_offset = 0x1d164;
  }

  // Will timeout if shim is busy
  uint32_t timeout_count = 0;
  uint32_t timeout_val = 100;
  while ((in32(TileAddr + status_register_offset) >> status_mask_shift) &
         0b11) {
    if (timeout_count >= timeout_val) {
      air_printf("[WARNING] xaie_shim_dma_wait_idle timed out\r\n");
      return 1;
    }
    timeout_count++;
  }

  return 0;
}

uint32_t xaie_shim_dma_get_outstanding(uint64_t TileAddr, int direction,
                                       int channel) {
  uint32_t shimDMAchannel = channel;
  uint32_t status_register_offset;
  uint32_t start_queue_size_mask_shift;
  if (channel == 0) {
    start_queue_size_mask_shift = 6;
  } else {
    start_queue_size_mask_shift = 9;
  }
  if (direction == SHIM_DMA_S2MM) {
    shimDMAchannel += XAIEDMA_SHIM_CHNUM_S2MM0;
    status_register_offset = 0x1d160;
  } else {
    shimDMAchannel += XAIEDMA_SHIM_CHNUM_MM2S0;
    status_register_offset = 0x1d164;
  }
  uint32_t outstanding = (in32(TileAddr + status_register_offset) >>
                          start_queue_size_mask_shift) &
                         0b111;
  return outstanding;
}

//// GLOBAL for shim DMAs mapped to the controller
// uint16_t mappedShimDMA[2] = {0};
//// GLOBAL for round-robin bd locations
// uint32_t last_bd[4][2] = {0};
uint32_t last_bd[8] = {0};

int xaie_shim_dma_push_bd(uint64_t TileAddr, int direction, int channel,
                          int col, uint64_t addr, uint32_t len) {
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
    } else {
      status_mask_shift = 2;
      control_register_offset = 0x1d148;
      start_queue_register_offset = 0x1d14c;
      start_queue_size_mask_shift = 9;
    }
    air_printf("\n\r  S2MM Shim DMA %d start channel %d\n\r", col,
               shimDMAchannel);
    // air_printf("\n\r  S2MM Shim DMA %d start channel %d\n\r",
    // mappedShimDMA[dma], shimDMAchannel);
  } else {
    shimDMAchannel += XAIEDMA_SHIM_CHNUM_MM2S0;
    status_register_offset = 0x1d164;
    if (channel == 0) {
      status_mask_shift = 0;
      control_register_offset = 0x1d150;
      start_queue_register_offset = 0x1d154;
      start_queue_size_mask_shift = 6;
    } else {
      status_mask_shift = 2;
      control_register_offset = 0x1d158;
      start_queue_register_offset = 0x1d15c;
      start_queue_size_mask_shift = 9;
    }
    air_printf("\n\r  MM2S Shim DMA %d start channel %d\n\r", col,
               shimDMAchannel);
    // air_printf("\n\r  MM2S Shim DMA %d start channel %d\n\r",
    // mappedShimDMA[dma], shimDMAchannel);
  }

  uint32_t start_bd = 4 * shimDMAchannel; // shimDMAchannel<<2;
  uint32_t outstanding = (in32(TileAddr + status_register_offset) >>
                          start_queue_size_mask_shift) &
                         0b111;
  // If outstanding >=4, we're in trouble!!!!
  // Theoretically this should never occur due to check in do_packet_nd_memcpy
  if (outstanding >= 4) { // NOTE had this at 3? // What is proper 'stalled'
                          // threshold? if (outstanding >=4)
    air_printf("\n\r *** BD OVERFLOW in shimDMA channel %d *** \n\r",
               shimDMAchannel);
    bool waiting = true;
    while (waiting) {
      outstanding = (in32(TileAddr + status_register_offset) >>
                     start_queue_size_mask_shift) &
                    0b111;
      waiting = (outstanding > 3); // NOTE maybe >= 3
      air_printf("*** Stalled in shimDMA channel %d outstanding = %d *** \n\r",
                 shimDMAchannel, outstanding + 1);
    } // WARNING this can lead to an endless loop
  }
  air_printf("Outstanding pre : %d\n\r", outstanding);
  // uint32_t bd = start_bd+outstanding;// + 0; // HACK
  int slot = channel;
  slot += ((col % 2) == 1) ? 4 : 0;
  if (direction == SHIM_DMA_S2MM)
    slot += XAIEDMA_SHIM_CHNUM_S2MM0;
  else
    slot += XAIEDMA_SHIM_CHNUM_MM2S0;
  uint32_t bd = start_bd + last_bd[slot];
  last_bd[slot] = (last_bd[slot] == 3) ? 0 : last_bd[slot] + 1;

  // Mark the BD as invalid
  out32(TileAddr + REG_SHIM_DMA_BD_CTRL(bd), 0);

  // Set the registers directly ...
  out32(TileAddr + REG_SHIM_DMA_BD_ADDR(bd), LOW_ADDR(addr));

  // change length in bytes to 32 bit words
  out32(TileAddr + REG_SHIM_DMA_BD_BUF_LEN(bd), len >> 2);

  u32 control = (HIGH_ADDR(addr) << 16) | SHIM_DMA_BD_CTRL_VALID;
  out32(TileAddr + REG_SHIM_DMA_BD_CTRL(bd), control);
  out32(TileAddr + REG_SHIM_DMA_BD_AXI_CFG(bd),
              0x410); // Burst len [10:9] = 2 (16)
                      // QoS [8:5] = 0 (best effort)
                      // Secure bit [4] = 1 (set)

  out32(TileAddr + REG_SHIM_DMA_BD_PKT(bd), 0);

  // Check if the channel is running or not
  uint32_t precheck_status =
      (in32(TileAddr + status_register_offset) >> status_mask_shift) &
      0b11;
  if (precheck_status == 0b00) {
    // Stream traffic can run, we can issue AXI-MM, and the channel is enabled
    xil_printf("Enabling shim DMA [%u] channel %u\r\n", col, channel);
    out32(TileAddr + control_register_offset, 0x1);
  }

  // Now push into the queue
  xil_printf("Pushing bd %u into 0x%x\r\n", bd, TileAddr + start_queue_register_offset);
  out32(TileAddr + start_queue_register_offset, bd);

#if CHATTY
  outstanding = (in32(TileAddr + status_register_offset) >>
                 start_queue_size_mask_shift) &
                0b111;
  air_printf("Outstanding post: %d\n\r", outstanding);
  air_printf("bd pushed as bd %u\r\n", bd);

  if (direction == SHIM_DMA_S2MM) {
    air_printf("  End of S2MM Shim DMA %d start channel %d\n\r", col,
               shimDMAchannel);
  } else {
    air_printf("  End of MM2S Shim DMA %d start channel %d\n\r", col,
               shimDMAchannel);
  }
#endif

  return 1;
}

int xaie_lock_release(u16 col, u16 row, u32 lock_id, u32 val) {
  u64 Addr = getTileAddr(col, row);
  u64 LockOfst = 0x0001E020;
  if (row != 0)
    LockOfst = 0x0001E020 + 0x10 * (val & 0x1);
  else {
    switch (col % 4) {
    case 0:
    case 1:
      LockOfst = 0x00014020 + 0x10 * (val & 0x1);
      break;
    default:
      LockOfst = 0x00014020 + 0x10 * (val & 0x1);
      break;
    }
  }
  maskpoll32(Addr + LockOfst + 0x80 * lock_id, 0x1, 0x1, 0);
  // XAieTile_LockRelease(tile, lock_id, val, 0);
  return 1;
}

int xaie_lock_acquire_nb(u16 col, u16 row, u32 lock_id, u32 val) {
  u64 Addr = getTileAddr(col, row);
  u64 LockOfst = 0x0001E060;
  if (row != 0)
    LockOfst = 0x0001E060 + 0x10 * (val & 0x1);
  else {
    switch (col % 4) {
    case 0:
    case 1:
      LockOfst = 0x00014060 + 0x10 * (val & 0x1);
      break;
    default:
      LockOfst = 0x00014060 + 0x10 * (val & 0x1);
      break;
    }
  }
  u8 lock_ret = 0;
  u32 loop = 0;
  while ((!lock_ret) && (loop < 512)) {
    lock_ret =
        maskpoll32(Addr + LockOfst + 0x80 * lock_id, 0x1, 0x1, 100);
    // lock_ret = XAieTile_LockAcquire(tile, lock_id, val, 10000);
    loop++;
  }
  if (loop == 512) {
    air_printf("Acquire [%d, %d, %d] value %d time-out\n\r", col, row, lock_id,
               val);
    return 0;
  }
  return 1;
}

#ifdef ARM_CONTROLLER

void xaie_array_reset() {

  // Getting a pointer to NPI
  auto *npib = (volatile uint32_t *)(NPI_BASE);

  // Performing array reset sequence
  air_printf("Starting array reset\r\n");

  // Unlocking NPI
  npib[NPI_LOCK_REG >> 2] = 0xF9E8D7C6;

  // Performing reset
  npib[NPI_MASK_REG >> 2] = 0x04000000;
  npib[NPI_VAL_REG >> 2] = 0x040381B1;
  npib[NPI_MASK_REG >> 2] = 0x04000000;
  npib[NPI_VAL_REG >> 2] = 0x000381B1;

  // Locking NPI
  npib[NPI_LOCK_REG >> 2] = 0x12341234;
  air_printf("Done with array reset\r\n");
}

// This should be called after enabling the proper
// shims to be reset via the mask
void xaie_strobe_shim_reset() {

  // Getting a pointer to NPI
  auto *npib = (volatile uint32_t *)(NPI_BASE);

  air_printf("Starting shim reset\r\n");

  // Unlocking NPI
  npib[NPI_LOCK_REG >> 2] = 0xF9E8D7C6;

  // Performing reset
  npib[NPI_MASK_REG >> 2] = 0x08000000;
  npib[NPI_VAL_REG >> 2] = 0x080381B1;
  npib[NPI_MASK_REG >> 2] = 0x08000000;
  npib[NPI_VAL_REG >> 2] = 0x000381B1;

  // Locking NPI
  npib[NPI_LOCK_REG >> 2] = 0x12341234;
  air_printf("Done with shim reset\r\n");
}

#endif

/*
  Reset all of the ME tiles in the specified column
*/
static void aie_reset_column(uint16_t col_idx)
{
    printf("Resetting column %u\r\n", col_idx);
    out32(getTileAddr(col_idx, 0) + REG_AIE_COL_RESET, 1); // 1 == ResetEnable
    out32(getTileAddr(col_idx, 0) + REG_AIE_COL_RESET, 0); // 0 == ResetDisable
}

/*
  Invalidate all BDs by writing to their buffer control register
*/
void xaie_shim_dma_init(uint16_t col) {
  uint64_t tileAddr = getTileAddr(col, 0);
  // Disable all channels
  for (uint8_t ch = 0; ch < 4; ch++) {
    out32(tileAddr + REG_SHIM_DMA_CTRL(ch), 0);
  }
  for (uint8_t bd = 0; bd < SHIM_DMA_NUM_BDS; bd++) {
    out32(tileAddr + REG_SHIM_DMA_BD_CTRL(bd), 0);
  }
}

void xaie_device_init(int num_cols) {

  air_printf("Initializing device...\r\n");

  // First, resetting the entire device
  xaie_array_reset();

#ifdef ARM_CONTROLLER
  int err = mlir_aie_reinit_device(_xaie);
  if (err)
    xil_printf("ERROR initializing device.\n\r");
#endif

  if (num_cols > NUM_SHIM_DMAS) {
    air_printf("WARN: attempt to initialize more shim DMAs than device has "
               "available!\n\r");
    num_cols = NUM_SHIM_DMAS;
  }

  for (int c = 0; c < num_cols; c++) {
    xaie_shim_dma_init(shim_dma_cols[c]);
  }

  // Turning the shim_reset_enable bit low for every column so they don't get
  // reset when we perform a global shim reset
  for (int col = 0; col < XAIE_NUM_COLS; col++) {
    out32(getTileAddr(col, 0) + 0x0003604C, 0);
  }
}

// Initialize one segment with lower left corner at (col_start, row_start)
void xaie_segment_init(int start_col, int num_cols, int start_row,
                       int num_rows) {
  HerdCfgInst.col_start = start_col;
  HerdCfgInst.num_cols = num_cols;
  HerdCfgInst.row_start = start_row;
  HerdCfgInst.num_rows = num_rows;
#ifdef ARM_CONTROLLER

  // Performing the shim reset
  air_printf("Performing shim reset; start_col=%u num_cols=%u\r\n", start_col, num_cols);
  for (uint16_t c = start_col; c < start_col + num_cols; c++) {
    out32(getTileAddr(c, 0) + REG_SHIM_RESET_ENA, 1);
  }

  xaie_strobe_shim_reset();

  for (uint16_t c = start_col; c < start_col + num_cols; c++) {
    out32(getTileAddr(c, 0) + REG_SHIM_RESET_ENA, 0);
  }

  // Performing the column reset
  air_printf("Performing col reset\r\n");
  for (uint16_t c = start_col; c < start_col + num_cols; c++)
    aie_reset_column(c);

#endif
}

/*
  Put a tile into reset
*/
void aie_tile_reset(int col, int row) {
  u64 tileAddr = getTileAddr(col, row);
  out32(tileAddr + REG_AIE_CORE_CTL, REG_AIE_CORE_CTL_RESET);
}

/*
  Take a tile out of reset
*/
void aie_tile_enable(int col, int row) {
  u64 tileAddr = getTileAddr(col, row);
  out32(tileAddr + REG_AIE_CORE_CTL, REG_AIE_CORE_CTL_ENABLE);
}

uint64_t shmem_base = 0x020100000000UL;
uint64_t uart_lock_offset = 0x200;
uint64_t base_address;

bool setup;

uint64_t get_base_address(void) { return shmem_base; }

void lock_uart(uint32_t id) {
  bool is_locked = false;
  volatile uint32_t *ulb = (volatile uint32_t *)(shmem_base + uart_lock_offset);

  while (!is_locked) {
    uint32_t status = ulb[0];
    if (status != 1) {
      ulb[1] = id;
      ulb[0] = 1;
      // See if they stuck
      uint32_t status = ulb[0];
      uint32_t lockee = ulb[1];
      if ((status == 1) && (lockee == id)) {
        // air_printf("ULock @ %lx MB %02d: ",ulb, id);
        is_locked = true;
      }
    }
  }
}

// This looks unsafe, but its okay as long as we always aquire
// the lock first
void unlock_uart() {
  volatile uint32_t *ulb = (volatile uint32_t *)(shmem_base + uart_lock_offset);
  ulb[1] = 0;
  ulb[0] = 0;
}

int queue_create(uint32_t size, queue_t **queue, uint32_t mb_id) {
  uint64_t queue_address[1] = {base_address + sizeof(dispatch_packet_t)};
  uint64_t queue_base_address[1] = {
      ALIGN(queue_address[0] + sizeof(queue_t), sizeof(dispatch_packet_t))};
  lock_uart(mb_id);
  air_printf("setup_queue 0x%llx, %x bytes + %d 64 byte packets\n\r",
             (void *)queue_address, sizeof(queue_t), size);
  air_printf("base address 0x%llx\n\r", base_address);
  unlock_uart();

  // The address of the queue_t is stored @ shmem_base[mb_id]
  memcpy((void *)(((uint64_t *)shmem_base) + mb_id), (void *)queue_address,
         sizeof(uint64_t));

  // Initialize the queue_t
  queue_t q;
  q.type = HSA_QUEUE_TYPE_SINGLE;
  q.features = HSA_QUEUE_FEATURE_AGENT_DISPATCH;

  memcpy((void *)&q.base_address, (void *)queue_base_address, sizeof(uint64_t));
  q.doorbell = 0xffffffffffffffffUL;
  q.size = size;
  q.reserved0 = 0;
  q.id = 0xacdc;

  q.read_index = 0;
  q.write_index = 0;
  q.last_doorbell = 0;

  memcpy((void *)queue_address[0], (void *)&q, sizeof(queue_t));

  // Invalidate the packets in the queue
  for (uint32_t idx = 0; idx < size; idx++) {
    dispatch_packet_t *pkt = &((dispatch_packet_t *)queue_base_address[0])[idx];
    pkt->header = HSA_PACKET_TYPE_INVALID;
  }

  memcpy((void *)queue, (void *)queue_address, sizeof(uint64_t));
  return 0;
}

void complete_agent_dispatch_packet(dispatch_packet_t *pkt) {
  // completion phase
  packet_set_active(pkt, false);
  pkt->header = HSA_PACKET_TYPE_INVALID;
  pkt->type = AIR_PKT_TYPE_INVALID;
  signal_subtract_acq_rel((signal_t *)&pkt->completion_signal, 1);
}

void complete_barrier_packet(void *pkt) {
  barrier_and_packet_t *p = (barrier_and_packet_t *)(pkt);
  // completion phase
  p->header = HSA_PACKET_TYPE_INVALID;
  signal_subtract_acq_rel((signal_t *)&p->completion_signal, 1);
}

void handle_packet_device_initialize(dispatch_packet_t *pkt) {
  packet_set_active(pkt, true);
  xaie_device_init(NUM_SHIM_DMAS);
}

void handle_packet_segment_initialize(dispatch_packet_t *pkt) {
  setup = true;
  packet_set_active(pkt, true);

  // Address mode here is absolute range
  if (((pkt->arg[0] >> 48) & 0xf) == AIR_ADDRESS_ABSOLUTE_RANGE) {
    u32 start_row = (pkt->arg[0] >> 16) & 0xff;
    u32 num_rows = (pkt->arg[0] >> 24) & 0xff;
    u32 start_col = (pkt->arg[0] >> 32) & 0xff;
    u32 num_cols = (pkt->arg[0] >> 40) & 0xff;

    u32 segment_id = pkt->arg[1] & 0xffff;

    // TODO more checks on segment dimensions
    if (start_row == 0)
      start_row++;
    xaie_segment_init(start_col, num_cols, start_row, num_rows);
    air_printf("Initialized segment %d at (%d, %d) of size (%d,%d)\r\n",
               segment_id, start_col, start_row, num_cols, num_rows);
    // segment_id is ignored - current restriction is 1 segment -> 1 controller
    // mappedShimDMA[0] = shimDMA0;
    // mappedShimDMA[1] = shimDMA1;
    // xaie_shim_dma_init(shimDMA0);
    // air_printf("Initialized shim DMA physical idx %d to logical idx
    // %d\r\n",shimDMA0,0); xaie_shim_dma_init(shimDMA1);
    // air_printf("Initialized shim DMA physical idx %d to logical idx
    // %d\r\n",shimDMA1,1);
  } else {
    air_printf("Unsupported address type 0x%04X for segment initialize\r\n",
               (pkt->arg[0] >> 48) & 0xf);
  }
}

void handle_packet_get_capabilities(dispatch_packet_t *pkt, uint32_t mb_id) {
  // packet is in active phase
  packet_set_active(pkt, true);
  uint64_t *addr = (uint64_t *)(pkt->return_address);

  lock_uart(mb_id);
  air_printf("Writing to 0x%llx\n\r", (uint64_t)addr);
  unlock_uart();
  // We now write a capabilities structure to the address we were just passed
  // We've already done this once - should we just cache the results?
#if ARM_CONTROLLER
  int user1 = 1;
  int user2 = 0;
#else
  pvr_t pvr;
  microblaze_get_pvr(&pvr);
  int user1 = MICROBLAZE_PVR_USER1(pvr);
  int user2 = MICROBLAZE_PVR_USER2(pvr);
#endif
  addr[0] = (uint64_t)mb_id;        // region id
  addr[1] = (uint64_t)user1;        // num regions
  addr[2] = (uint64_t)(user2 >> 8); // region controller firmware version
  addr[3] = 16L;                    // cores per region
  addr[4] = 32768L;                 // Total L1 data memory per core
  addr[5] = 8L;                     // Number of L1 data memory banks
  addr[6] = 16384L;                 // L1 program memory per core
  addr[7] = 0L;                     // L2 data memory per region
}

void handle_packet_get_info(dispatch_packet_t *pkt, uint32_t mb_id) {
  // packet is in active phase
  packet_set_active(pkt, true);
  uint64_t attribute = (pkt->arg[0]);
  uint64_t *addr =
      (uint64_t *)(&pkt->return_address); // FIXME when we can use a VA

#if ARM_CONTROLLER
  int user1 = 1;
  int user2 = 0;
#else
  pvr_t pvr;
  microblaze_get_pvr(&pvr);
  int user1 = MICROBLAZE_PVR_USER1(pvr);
  int user2 = MICROBLAZE_PVR_USER2(pvr);
#endif
  char name[] = {'A', 'C', 'D', 'C', '\0'};
  char vend[] = {'A', 'M', 'D', '\0'};

  // TODO change this to use pkt->return_address
  switch (attribute) {
  case AIR_AGENT_INFO_NAME:
    memcpy(addr, name, 8);
    break;
  case AIR_AGENT_INFO_VENDOR_NAME:
    memcpy(addr, vend, 8);
    break;
  case AIR_AGENT_INFO_CONTROLLER_ID:
    *addr = (uint64_t)mb_id; // region id
    break;
  case AIR_AGENT_INFO_FIRMWARE_VER:
    *addr = (uint64_t)(user2 >> 8); // region controller firmware version
    break;
  case AIR_AGENT_INFO_NUM_REGIONS:
    *addr = (uint64_t)user1; // num regions
    break;
  case AIR_AGENT_INFO_HERD_SIZE: // cores per region
    *addr = HerdCfgInst.num_cols * HerdCfgInst.num_rows;
    break;
  case AIR_AGENT_INFO_HERD_ROWS:
    *addr = HerdCfgInst.num_rows; // rows of cores
    break;
  case AIR_AGENT_INFO_HERD_COLS:
    *addr = HerdCfgInst.num_cols; // cols of cores
    break;
  case AIR_AGENT_INFO_TILE_DATA_MEM_SIZE:
    *addr = 32768L; // total L1 data memory per core
    break;
  case AIR_AGENT_INFO_TILE_PROG_MEM_SIZE:
    *addr = 16384L; // L1 program memory per core
    break;
  case AIR_AGENT_INFO_L2_MEM_SIZE: // L2 memory per region (cols * 256k)
    *addr = 262144L * HerdCfgInst.num_cols;
    break;
  default:
    *addr = 0;
    break;
  }
}

#define AIE_BASE 0x020000000000
#define AIE_CSR_SIZE 0x000100000000

void handle_packet_read_write_32(dispatch_packet_t *pkt) {

  packet_set_active(pkt, true);

  uint64_t *return_addr =
      (uint64_t *)(&pkt->return_address); // FIXME when we can use a VA

  uint64_t address = pkt->arg[0];
  uint32_t value = pkt->arg[1] & 0x0FFFFFFFF;
  bool is_write = (pkt->arg[1] >> 32) & 0x1;

  volatile uint32_t *aie_csr = (volatile uint32_t *)AIE_BASE;

  if (address > AIE_CSR_SIZE) {
    printf("[ERROR] read32/write32 packets provided address of size 0x%lx. "
           "Window is only 4GB\n",
           address);
  }

  if (is_write) {
    aie_csr[address >> 2] = value;
  } else {
    *return_addr = aie_csr[address >> 2];
  }
}


void handle_packet_sg_cdma(dispatch_packet_t *pkt) {
  // packet is in active phase
  packet_set_active(pkt, true);
  volatile uint32_t *cdmab = (volatile uint32_t *)(CDMA_BASE);
  u32 start_row = (pkt->arg[3] >> 0) & 0xff;
  u32 num_rows = (pkt->arg[3] >> 8) & 0xff;
  u32 start_col = (pkt->arg[3] >> 16) & 0xff;
  u32 num_cols = (pkt->arg[3] >> 24) & 0xff;
  for (uint c = start_col; c < start_col + num_cols; c++) {
    for (uint r = start_row; r < start_row + num_rows; r++) {
      out32(getTileAddr(c, r) + 0x00032000, 0x2);
      air_printf("Done resetting col %d row %d.\n\r", c, r);
    }
    air_printf("Resetting column %d.\n\r", c);
    aie_reset_column(c);
  }
  air_printf("CDMA reset.\n\r");
  cdmab[0] |= 0x4;
  cdmab[0] &= 0x4;
  while (cdmab[0] & 0x4)
    ;
  air_printf("CDMA start.\n\r");
  uint64_t daddr = (pkt->arg[0]);
  uint64_t saddr = (pkt->arg[1]);
  uint32_t bytes = (pkt->arg[2]);
  air_printf("CMDA daddr 0x%016lx saddr 0x%016lx\n\r", daddr, saddr);
  cdmab[0] = 0x0;          // unset SG mode
  if (bytes >= 0xffffff) { // SG
    cdmab[0] = 0x8;        // set SG mode
    cdmab[2] = saddr & 0xffffffff;
    cdmab[3] = saddr >> 32;
    cdmab[5] = daddr >> 32;
    cdmab[4] = daddr & 0xffffffff;
  } else {
    cdmab[6] = saddr & 0xffffffff;
    cdmab[7] = saddr >> 32;
    cdmab[8] = daddr & 0xffffffff;
    cdmab[9] = daddr >> 32;
    cdmab[10] = bytes;
  }
  int cnt = 100;
  while (!(cdmab[1] & 2) && cnt--)
    air_printf("SG CDMA wait... %x\n\r", cdmab[1]);
  for (uint c = start_col; c < start_col + num_cols; c++) {
    for (uint r = start_row; r <= start_row + num_rows; r++) {
      for (int l = 0; l < 16; l++)
        maskpoll32(getTileAddr(c, r) + REG_AIE_LOCK_RELEASE_0(l), 0x1, 0x1, 0);
      out32(getTileAddr(c, r) + REG_AIE_CORE_CTL, REG_AIE_CORE_CTL_ENABLE);
    }
  }
  air_printf("CDMA done!\n\r");
}

void handle_packet_cdma(dispatch_packet_t *pkt) {
  // packet is in active phase
  packet_set_active(pkt, true);
  u32 start_row = (pkt->arg[3] >> 0) & 0xff;
  u32 num_rows = (pkt->arg[3] >> 8) & 0xff;
  u32 start_col = (pkt->arg[3] >> 16) & 0xff;
  u32 num_cols = (pkt->arg[3] >> 24) & 0xff;
  u32 op = (pkt->arg[3] >> 32) & 0xff;
  if (op == 2) {
    for (uint c = start_col; c < start_col + num_cols; c++) {
      for (uint r = start_row; r < start_row + num_rows; r++) {
        int st = in32(getTileAddr(c, r) + REG_AIE_CORE_STATUS);
        air_printf("Status col %d row %d. 0x%x\n\r", c, r, st & 0x3);
        if ((0x3 & st) != 0x2) {
          out32(getTileAddr(c, r) + REG_AIE_CORE_CTL, REG_AIE_CORE_CTL_RESET);
          air_printf("Done resetting col %d row %d.\n\r", c, r);
        }
      }
    }
  }
  if (op == 1) {
    for (uint c = start_col; c < start_col + num_cols; c++) {
      air_printf("Resetting column %d.\n\r", c);
      aie_reset_column(c);
      air_printf("Done resetting column %d.\n\r", c);
    }
  }
  volatile uint32_t *cdmab = (volatile uint32_t *)(CDMA_BASE);
  uint32_t status = cdmab[1];
  air_printf("CMDA raw %x idle %x\n\r", status, status & 2);
  uint64_t daddr = (pkt->arg[0]);
  uint64_t saddr = (pkt->arg[1]);
  uint32_t bytes = (pkt->arg[2]);
  air_printf("CMDA dst %lx src %lx\n\r", daddr, saddr);
  cdmab[0] = 0x0; // unset SG mode
  cdmab[6] = saddr & 0xffffffff;
  cdmab[7] = saddr >> 32;
  cdmab[8] = daddr & 0xffffffff;
  cdmab[9] = daddr >> 32;
  cdmab[10] = bytes;
  while (!(cdmab[1] & 2))
    air_printf("CMDA wait...\n\r");
  if (op == 2) {
    for (uint c = start_col; c < start_col + num_cols; c++) {
      for (uint r = start_row; r <= start_row + num_rows; r++) {
        for (int l = 0; l < 16; l++)
          maskpoll32(getTileAddr(c, r) + REG_AIE_LOCK_RELEASE_0(l), 0x1, 0x1, 0);
        out32(getTileAddr(c, r) + REG_AIE_CORE_CTL, REG_AIE_CORE_CTL_ENABLE);
      }
    }
  }
}

void handle_packet_xaie_lock(dispatch_packet_t *pkt) {
  // packet is in active phase
  packet_set_active(pkt, true);

  u32 num_cols =
      (((pkt->arg[0] >> 48) & 0xf) == AIR_ADDRESS_HERD_RELATIVE_RANGE)
          ? ((pkt->arg[0] >> 40) & 0xff)
          : 1;
  u32 num_rows =
      (((pkt->arg[0] >> 48) & 0xf) == AIR_ADDRESS_HERD_RELATIVE_RANGE)
          ? ((pkt->arg[0] >> 24) & 0xff)
          : 1;
  u32 start_col = (pkt->arg[0] >> 32) & 0xff;
  u32 start_row = (pkt->arg[0] >> 16) & 0xff;
  u32 lock_id = pkt->arg[1];
  u32 acqrel = pkt->arg[2];
  u32 val = pkt->arg[3];
  for (u32 col = 0; col < num_cols; col++) {
    for (u32 row = 0; row < num_rows; row++) {
      if (acqrel == 0)
        xaie_lock_acquire_nb(HerdCfgInst.col_start + start_col + col,
                             HerdCfgInst.row_start + start_row + row, lock_id,
                             val);
      else
        xaie_lock_release(HerdCfgInst.col_start + start_col + col,
                          HerdCfgInst.row_start + start_row + row, lock_id,
                          val);
    }
  }
}

#ifdef ARM_CONTROLLER
void handle_packet_xaie_status(dispatch_packet_t *pkt, u32 type) {
  xil_printf("Reading status! %d %d %d\n\r", type, pkt->arg[0], pkt->arg[1]);
  if (type == 1) {
    mlir_aie_print_shimdma_status(_xaie, pkt->arg[0], 0);
  } else if (type == 2) {
    mlir_aie_print_dma_status(_xaie, pkt->arg[0], pkt->arg[1]);
  } else if (type == 3) {
    mlir_aie_print_tile_status(_xaie, pkt->arg[0], pkt->arg[1]);
  }
}
#endif

void handle_packet_hello(dispatch_packet_t *pkt, uint32_t mb_id) {
  packet_set_active(pkt, true);

  uint64_t say_what = pkt->arg[0];
  lock_uart(mb_id);
  xil_printf("MB %d : HELLO %08X\n\r", mb_id, (uint32_t)say_what);
  unlock_uart();
}

typedef struct staged_nd_memcpy_s {
  uint32_t valid;
  dispatch_packet_t *pkt;
  uint64_t paddr[3];
  uint32_t index[3];
} staged_nd_memcpy_t; // about 48B therefore @ 64 slots ~3kB

int get_slot(int col, int space) {
  if (space == 2) {
    for (int i = 0; i < NUM_SHIM_DMAS; i++) {
      if (col == shim_dma_cols[i]) {
        return i * 4;
      }
    }
  } else if (space == 1) {
    for (int i = 0; i < NUM_COL_DMAS; i++) {
      if (col == col_dma_cols[i]) {
        return i * 4 + NUM_SHIM_DMAS * 4;
      }
    }
  }
  return 0;
}

// GLOBAL storage for 'in progress' ND memcpy work
// NOTE 4 slots per shim DMA
staged_nd_memcpy_t staged_nd_slot[NUM_DMAS * 4];

void nd_dma_put_checkpoint(dispatch_packet_t **pkt, uint32_t slot,
                           uint32_t idx_4d, uint32_t idx_3d, uint32_t idx_2d,
                           uint64_t pad_3d, uint64_t pad_2d, uint64_t pad_1d) {
  staged_nd_slot[slot].pkt = *pkt;
  staged_nd_slot[slot].paddr[0] = pad_1d;
  staged_nd_slot[slot].paddr[1] = pad_2d;
  staged_nd_slot[slot].paddr[2] = pad_3d;
  staged_nd_slot[slot].index[0] = idx_2d;
  staged_nd_slot[slot].index[1] = idx_3d;
  staged_nd_slot[slot].index[2] = idx_4d;
}

void nd_dma_get_checkpoint(dispatch_packet_t **pkt, uint32_t slot,
                           uint32_t &idx_4d, uint32_t &idx_3d, uint32_t &idx_2d,
                           uint64_t &pad_3d, uint64_t &pad_2d,
                           uint64_t &pad_1d) {
  *pkt = staged_nd_slot[slot].pkt;
  pad_1d = staged_nd_slot[slot].paddr[0];
  pad_2d = staged_nd_slot[slot].paddr[1];
  pad_3d = staged_nd_slot[slot].paddr[2];
  idx_2d = staged_nd_slot[slot].index[0];
  idx_3d = staged_nd_slot[slot].index[1];
  idx_4d = staged_nd_slot[slot].index[2];
}

int do_packet_nd_memcpy(uint32_t slot) {
  dispatch_packet_t *a_pkt;
  uint64_t paddr_3d;
  uint64_t paddr_2d;
  uint64_t paddr_1d;
  uint32_t index_4d;
  uint32_t index_3d;
  uint32_t index_2d;
  nd_dma_get_checkpoint(&a_pkt, slot, index_4d, index_3d, index_2d, paddr_3d,
                        paddr_2d, paddr_1d);

  uint16_t channel = (a_pkt->arg[0] >> 24) & 0x00ff;
  uint16_t col = (a_pkt->arg[0] >> 32) & 0x00ff;
  // uint16_t logical_col  = (a_pkt->arg[0] >> 32) & 0x00ff;
  uint16_t direction = (a_pkt->arg[0] >> 60) & 0x000f;
  uint32_t length_1d = (a_pkt->arg[2] >> 0) & 0xffffffff;
  uint32_t length_2d = (a_pkt->arg[2] >> 32) & 0x0000ffff;
  uint32_t stride_2d = (a_pkt->arg[2] >> 48) & 0x0000ffff;
  uint32_t length_3d = (a_pkt->arg[3] >> 0) & 0x0000ffff;
  uint32_t stride_3d = (a_pkt->arg[3] >> 16) & 0x0000ffff;
  uint32_t length_4d = (a_pkt->arg[3] >> 32) & 0x0000ffff;
  uint32_t stride_4d = (a_pkt->arg[3] >> 48) & 0x0000ffff;
  uint32_t outstanding = 0;

  air_printf("%s: col=%u dir=%u chan=%u paddr=0x%llx 4d stride=%u length=%u\r\n",
      __func__, col, direction, channel, paddr_1d, stride_4d, length_4d);
  air_printf("  3d stride=%u length=%u, 2d stride=%u length=%u, 1d length=%u\r\n",
      stride_3d, length_3d, stride_2d, length_2d, length_1d);

  for (; index_4d < length_4d; index_4d++) {
    for (; index_3d < length_3d; index_3d++) {
      for (; index_2d < length_2d; index_2d++) {
        outstanding = xaie_shim_dma_get_outstanding(getTileAddr(col, 0),
                                                    direction, channel);
        air_printf("\n\rND start shim DMA %u %u [%u][%u][%u] paddr=0x%llx\r\n",
                   direction, channel, index_4d, index_3d, index_2d, paddr_1d);
        if (outstanding >= 4) { // NOTE What is proper 'stalled' threshold?
          nd_dma_put_checkpoint(&a_pkt, slot, index_4d, index_3d, index_2d,
                                paddr_3d, paddr_2d, paddr_1d);
          return 1;
        } else {
          xaie_shim_dma_push_bd(getTileAddr(col, 0), direction, channel,
                                col, paddr_1d, length_1d);
        }
        paddr_1d += stride_2d;
      }
      index_2d = 0;
      paddr_2d += stride_3d;
      if (index_3d + 1 < length_3d)
        paddr_1d = paddr_2d;
      else
        paddr_1d = paddr_3d + stride_4d;
    }
    index_3d = 0;
    paddr_3d += stride_4d;
    paddr_2d = paddr_3d;
  }

  // Wait check idle
  int wait_idle_ret =
      xaie_shim_dma_wait_idle(getTileAddr(col, 0), direction, channel);

  // If return 1 we timed out, BDs waiting on other BDs. Put checkpoint and
  // return 1
  if (wait_idle_ret) {
    nd_dma_put_checkpoint(&a_pkt, slot, index_4d, index_3d, index_2d, paddr_3d,
                          paddr_2d, paddr_1d);
  }

  return wait_idle_ret;
}

int do_packet_memcpy(uint32_t slot) {
  if (slot >= NUM_SHIM_DMAS * 4) {
    return 0;
  } else {
    return do_packet_nd_memcpy(slot);
  }
}

int stage_packet_nd_memcpy(dispatch_packet_t *pkt, uint32_t slot,
                           uint32_t memory_space) {
  air_printf("stage_packet_nd_memcpy %d\n\r", slot);
  if (staged_nd_slot[slot].valid) {
    air_printf("STALL: ND Memcpy Slot %d Busy!\n\r", slot);
    return 2;
  }
  packet_set_active(pkt, true);

  uint64_t paddr = pkt->arg[1];

  if (memory_space == 2) {
    nd_dma_put_checkpoint(&pkt, slot, 0, 0, 0, paddr, paddr, paddr);
    staged_nd_slot[slot].valid = 1;
    return 0;
  } else {
    air_printf("NOT SUPPORTED: Cannot program memory space %d DMAs\n\r",
               memory_space);
    return 1;
  }
}

void handle_agent_dispatch_packet(queue_t *q, uint32_t mb_id) {
  uint64_t rd_idx = queue_load_read_index(q);
  dispatch_packet_t *pkt =
      &((dispatch_packet_t *)q->base_address)[mymod(rd_idx)];
  int last_slot = 0;
  int max_slot = 4 * NUM_DMAS - 1;

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
        slot = (slot == max_slot) ? 0 : slot + 1; // TODO better heuristic
        if (slot == last_slot)
          break;
        air_printf("RR check slot: %d\n\r", slot);
        if (staged_nd_slot[slot].valid) {
          dispatch_packet_t *a_pkt = staged_nd_slot[slot].pkt;
          uint16_t channel = (a_pkt->arg[0] >> 24) & 0x00ff;
          uint16_t col = (a_pkt->arg[0] >> 32) & 0x00ff;
          // uint16_t logical_col  = (a_pkt->arg[0] >> 32) & 0x00ff;
          uint16_t direction = (a_pkt->arg[0] >> 60) & 0x000f;
          // uint16_t col          = mappedShimDMA[logical_col];
          stalled = (xaie_shim_dma_get_outstanding(getTileAddr(col, 0),
                                                   direction, channel) >= 4);
          active = packet_get_active(a_pkt);
        } else {
          stalled = true;
          active = false;
        }
        air_printf("RR slot: %d - valid %d stalled %d active %d\n\r", slot,
                   staged_nd_slot[slot].valid, stalled, active);
      } while (!staged_nd_slot[slot].valid || stalled || !active);

      if (slot == last_slot) { // Begin get next packet
        rd_idx++;
        pkt = &((dispatch_packet_t *)q->base_address)[mymod(rd_idx)];
        air_printf("HELLO NEW PACKET IN FLIGHT!\n\r");
        if (((pkt->header) & 0xF) != HSA_PACKET_TYPE_AGENT_DISPATCH) {
          rd_idx--;
          pkt = &((dispatch_packet_t *)q->base_address)[mymod(rd_idx)];
          air_printf("WARN: Found invalid HSA packet inside peek loop!\n\r");
          // TRICKY weird state where we didn't find a new packet but RR won't
          // let us retry. So advance last_slot.
          last_slot =
              (slot == max_slot) ? 0 : slot + 1; // TODO better heuristic
          continue;
        } else
          goto packet_op;
      } // End get next packet

      // FOUND ND packet process here
      last_slot = slot;
      int ret = do_packet_memcpy(slot);
      if (ret)
        continue;
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
    // air_printf("Op is %04X\n\r",op);
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
    case AIR_PKT_TYPE_SEGMENT_INITIALIZE:
      handle_packet_segment_initialize(pkt);
      complete_agent_dispatch_packet(pkt);
      packets_processed++;
      break;

    case AIR_PKT_TYPE_CONFIGURE:
      handle_packet_sg_cdma(pkt);
      complete_agent_dispatch_packet(pkt);
      packets_processed++;
      break;

    case AIR_PKT_TYPE_RW32:
      handle_packet_read_write_32(pkt);
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
      uint16_t channel = (pkt->arg[0] >> 24) & 0x00ff;
      uint16_t direction = (pkt->arg[0] >> 60) & 0x000f;
      uint16_t col = (pkt->arg[0] >> 32) & 0x00ff;
      int slot = channel;
      slot += get_slot(col, memory_space);
      if (direction == SHIM_DMA_S2MM)
        slot += XAIEDMA_SHIM_CHNUM_S2MM0;
      else
        slot += XAIEDMA_SHIM_CHNUM_MM2S0;
      int ret = stage_packet_nd_memcpy(pkt, slot, memory_space);
      if (ret == 0) {
        last_slot = slot;
        if (do_packet_memcpy(slot)) {
          num_active_packets++;
          break;
        } // else completed the packet in the first try
      } else if (ret == 2)
        break; // slot busy, retry.
      staged_nd_slot[slot].valid = 0;
      complete_agent_dispatch_packet(
          pkt); // this is correct for the first try or invalid stage
      packets_processed++;
      break;

    } // switch
  } while (num_active_packets > 1);
  lock_uart(mb_id);
  air_printf("Completing: %d packets processed.\n\r", packets_processed);
  unlock_uart();
  queue_add_read_index(q, packets_processed);
  q->read_index = mymod(q->read_index);
}

inline signal_value_t signal_wait(volatile signal_t *signal,
                                  signal_value_t compare_value,
                                  uint64_t timeout_hint,
                                  signal_value_t default_value) {
  if (signal->handle == 0)
    return default_value;
  signal_value_t ret = 0;
  uint64_t timeout = timeout_hint;
  do {
    ret = signal->handle;
    if (ret == compare_value)
      return compare_value;
  } while (timeout--);
  return ret;
}

void handle_barrier_and_packet(queue_t *q, uint32_t mb_id) {
  uint64_t rd_idx = queue_load_read_index(q);
  barrier_and_packet_t *pkt =
      &((barrier_and_packet_t *)q->base_address)[mymod(rd_idx)];

  // TODO complete functionality with VAs
  signal_t *s0 = (signal_t *)pkt->dep_signal[0];
  signal_t *s1 = (signal_t *)pkt->dep_signal[1];
  signal_t *s2 = (signal_t *)pkt->dep_signal[2];
  signal_t *s3 = (signal_t *)pkt->dep_signal[3];
  signal_t *s4 = (signal_t *)pkt->dep_signal[4];

  // lock_uart(mb_id);
  // for (int i = 0; i < 5; i++)
  //  air_printf("MB %d : dep_signal[%d] @ %p\n\r",mb_id,i,(uint64_t
  //  *)(pkt->dep_signal[i]));
  // unlock_uart();

  while ((signal_wait(s0, 0, 0x80000, 0) != 0) ||
         (signal_wait(s1, 0, 0x80000, 0) != 0) ||
         (signal_wait(s2, 0, 0x80000, 0) != 0) ||
         (signal_wait(s3, 0, 0x80000, 0) != 0) ||
         (signal_wait(s4, 0, 0x80000, 0) != 0)) {
    lock_uart(mb_id);
    air_printf("MB %d : barrier AND packet completion signal timeout!\n\r",
               mb_id);
    for (int i = 0; i < 5; i++)
      air_printf("MB %d : dep_signal[%d] = %d\n\r", mb_id, i,
                 *((uint32_t *)(pkt->dep_signal[i])));
    unlock_uart();
  }

  complete_barrier_packet(pkt);
  queue_add_read_index(q, 1);
  q->read_index = mymod(q->read_index);
}

void handle_barrier_or_packet(queue_t *q, uint32_t mb_id) {
  uint64_t rd_idx = queue_load_read_index(q);
  barrier_or_packet_t *pkt =
      &((barrier_or_packet_t *)q->base_address)[mymod(rd_idx)];

  // TODO complete functionality with VAs
  signal_t *s0 = (signal_t *)pkt->dep_signal[0];
  signal_t *s1 = (signal_t *)pkt->dep_signal[1];
  signal_t *s2 = (signal_t *)pkt->dep_signal[2];
  signal_t *s3 = (signal_t *)pkt->dep_signal[3];
  signal_t *s4 = (signal_t *)pkt->dep_signal[4];

  // lock_uart(mb_id);
  // for (int i = 0; i < 5; i++)
  //  air_printf("MB %d : dep_signal[%d] @ %p\n\r",mb_id,i,(uint64_t
  //  *)(pkt->dep_signal[i]));
  // unlock_uart();

  while ((signal_wait(s0, 0, 0x80000, 1) != 0) &&
         (signal_wait(s1, 0, 0x80000, 1) != 0) &&
         (signal_wait(s2, 0, 0x80000, 1) != 0) &&
         (signal_wait(s3, 0, 0x80000, 1) != 0) &&
         (signal_wait(s4, 0, 0x80000, 1) != 0)) {
    lock_uart(mb_id);
    air_printf("MB %d : barrier OR packet completion signal timeout!\n\r",
               mb_id);
    for (int i = 0; i < 5; i++)
      air_printf("MB %d : dep_signal[%d] = %d\n\r", mb_id, i,
                 *((uint32_t *)(pkt->dep_signal[i])));
    unlock_uart();
  }

  complete_barrier_packet(pkt);
  queue_add_read_index(q, 1);
  q->read_index = mymod(q->read_index);
}

int main() {
  init_platform();
#ifdef ARM_CONTROLLER
  Xil_DCacheDisable();

  aie_libxaie_ctx_t ctx;
  _xaie = &ctx;
  mlir_aie_init_libxaie(_xaie);
  int err = mlir_aie_init_device(_xaie);
  if (err)
    xil_printf("ERROR initializing device.\n\r");
  int user1 = 1;
  int user2 = 0;
#else
  pvr_t pvr;
  microblaze_get_pvr(&pvr);
  int user1 = MICROBLAZE_PVR_USER1(pvr);
  int user2 = MICROBLAZE_PVR_USER2(pvr);
#endif
  int mb_id = user2 & 0xff;
  int maj = (user2 >> 24) & 0xff;
  int min = (user2 >> 16) & 0xff;
  int ver = (user2 >> 8) & 0xff;

  // Skip over the system wide shmem area, then find your own
  base_address = shmem_base + (1 + mb_id) * MB_SHMEM_SEGMENT_SIZE;
  uint32_t *num_mbs = (uint32_t *)(shmem_base + 0x208);
  num_mbs[0] = user1;

  if (mb_id == 0) {
    unlock_uart(); // NOTE: Initialize uart lock only from 'first' MB
    // initialize shared signals
    uint64_t *s = (uint64_t *)(shmem_base + MB_SHMEM_SIGNAL_OFFSET);
    for (uint64_t i = 0; i < (MB_SHMEM_SIGNAL_SIZE) / sizeof(uint64_t); i++)
      s[i] = 0;
  }

  lock_uart(mb_id);
#ifdef ARM_CONTROLLER
  xil_printf("ARM %d of %d firmware %d.%d.%d created on %s at %s GMT\n\r",
             mb_id + 1, *num_mbs, maj, min, ver, __DATE__, __TIME__);
#else
  xil_printf("MB %d of %d firmware %d.%d.%d created on %s at %s GMT\n\r",
             mb_id + 1, *num_mbs, maj, min, ver, __DATE__, __TIME__);
#endif
  xil_printf("(c) Copyright 2020-2022 AMD, Inc. All rights reserved.\n\r");
  unlock_uart();

  setup = false;
  queue_t *q = nullptr;
  queue_create(MB_QUEUE_SIZE, &q, mb_id);
  lock_uart(mb_id);
  xil_printf("Created queue @ 0x%llx\n\r", (size_t)q);
  unlock_uart();

  volatile bool done = false;
  while (!done) {
    if (q->doorbell + 1 > q->last_doorbell) {
      lock_uart(mb_id);
      air_printf("Ding Dong 0x%llx\n\r", q->doorbell + 1);
      unlock_uart();

      q->last_doorbell = q->doorbell + 1;

      // process packets until we hit an invalid packet
      bool invalid = false;
      while (!invalid) {
        uint64_t rd_idx = queue_load_read_index(q);

        // air_printf("Handle pkt read_index=%d\n\r", rd_idx);

        dispatch_packet_t *pkt =
            &((dispatch_packet_t *)q->base_address)[mymod(rd_idx)];
        uint8_t type = ((pkt->header) & (0xF));
        // uint8_t type = ((pkt->header >> HSA_PACKET_HEADER_TYPE) &
        //                ((1 << HSA_PACKET_HEADER_WIDTH_TYPE) - 1));
        switch (type) {
        default:
        case HSA_PACKET_TYPE_INVALID:
          if (setup) {
            lock_uart(mb_id);
            air_printf("Waiting\n\r");
            unlock_uart();
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
    shell();
  }

  cleanup_platform();
  return 0;
}
