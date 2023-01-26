//===- plat_vck190.cpp ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "platform.h"
extern "C" {
#include "pvr.h"
#include "xil_cache.h"
}

#define VCK190_CDMA_BASE 0x000044A00000UL
#define XAIE_ADDR_ARRAY_OFF 0x800

const char vck190_platform_name[] = "vck190";

u32 in32(u64 Addr)
{
  return *(volatile u32 *)Addr;
}

void out32(u64 Addr, u32 Value)
{
  volatile u32 *LocalAddr = (volatile u32 *)Addr;
  *LocalAddr = Value;
}

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
  TileAddr = (u64)((ArrOffset << XAIEGBL_TILE_ADDR_ARR_SHIFT) |
                   (ColIdx << XAIEGBL_TILE_ADDR_COL_SHIFT) |
                   (RowIdx << XAIEGBL_TILE_ADDR_ROW_SHIFT));

  return TileAddr;
}

int plat_device_init(void)
{
	return 0;
}

int init_platform(struct platform_cfg *cfg)
{
#ifdef XPAR_MICROBLAZE_USE_ICACHE
    Xil_ICacheEnable();
#endif
#ifdef XPAR_MICROBLAZE_USE_DCACHE
    Xil_DCacheEnable();
#endif


#ifdef STDOUT_IS_16550
    XUartNs550_SetBaud(STDOUT_BASEADDR, XPAR_XUARTNS550_CLOCK_HZ, UART_BAUD);
    XUartNs550_SetLineControlReg(STDOUT_BASEADDR, XUN_LCR_8_DATA_BITS);
#endif

  pvr_t pvr;
  microblaze_get_pvr(&pvr);
  cfg->mb_count = MICROBLAZE_PVR_USER1(pvr);
  cfg->version = MICROBLAZE_PVR_USER2(pvr);
	cfg->cdma_base = VCK190_CDMA_BASE;
	cfg->platform_name = vck190_platform_name;

	return 0;
}
