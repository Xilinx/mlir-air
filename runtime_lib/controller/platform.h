/******************************************************************************
*
* Copyright (C) 2008 - 2014 Xilinx, Inc.  All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* Use of the Software is limited solely to applications:
* (a) running on a Xilinx device, or
* (b) that interact with a Xilinx device through a bus or interconnect.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* XILINX  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF
* OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*
* Except as contained in this notice, the name of the Xilinx shall not be used
* in advertising or otherwise to promote the sale, use or other dealings in
* this Software without prior written authorization from Xilinx.
*
******************************************************************************/

#ifndef __PLATFORM_H_
#define __PLATFORM_H_

extern "C" {
#include "xil_types.h"
}

#define XAIEGBL_TILE_ADDR_ARR_SHIFT 30U
#define XAIEGBL_TILE_ADDR_ROW_SHIFT 18U
#define XAIEGBL_TILE_ADDR_COL_SHIFT 23U

#define ENCODE_VERSION(_major, _minor, _build) ((_major & 0xFF) << 24 | (_minor & 0xFF) << 16 | (_build & 0xFF) << 8)
#define GET_VERSION_MAJOR(_x) ((_x >> 24) & 0xFF)
#define GET_VERSION_MINOR(_x) ((_x >> 16) & 0xFF)
#define GET_VERSION_BUILD(_x) ((_x >> 8) & 0xFF)

struct platform_cfg
{
	int mb_count;			// Number of microblaze controllers
	int version;			// encoded version number (see GET_VERSION_ macros)
	uint64_t cdma_base;	// base address for CDMA ?
	const char *platform_name; // name of the platform in C string form
};

/*
	Platform-specific initialization

	Must be called once at startup. Fills in the platform_cfg
	with parameters for this platform.
*/
int init_platform(struct platform_cfg *cfg);

/*
	read 32 bit value from specified address
*/
u32 in32(u64 Addr);

void out32(u64 Addr, u32 Value);

u64 getTileAddr(u16 ColIdx, u16 RowIdx);

/*
	Platform-specific initialization of the device

	Returns 0 on success, non-zero on error
*/
int plat_device_init(void);

#endif
