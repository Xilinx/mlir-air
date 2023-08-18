//===- platform.h -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef __PLATFORM_H_
#define __PLATFORM_H_

#include "platform_config.h"
#include <cstdint>

extern "C" {
#ifdef ARM_CONTROLLER
#include "xaiengine.h"
#endif // ARM_CONTROLLER
}

#define NPI_BASE 0xF70A0000UL

#define DRAM_1_BASE 0x000800000000ULL
#define DRAM_1_SIZE 0x000380000000ULL

#ifdef ARM_CONTROLLER
#define CDMA_BASE 0x0000A4000000UL
#else
#define CDMA_BASE 0x000044000000UL
#endif // ARM_CONTROLLER

#define AIE_BASE 0x020000000000ULL
#define AIE_CSR_SIZE 0x000100000000ULL

#define IO_READ32(addr) *((volatile uint32_t *)(addr))
#define IO_WRITE32(addr, val) *((volatile uint32_t *)(addr)) = val

struct aie_libxaie_ctx_t {
  XAie_Config AieConfigPtr;
  XAie_DevInst DevInst;
};

void init_platform();
void cleanup_platform();

void mlir_aie_print_dma_status(int col, int row);
void mlir_aie_print_shimdma_status(uint16_t col);
void mlir_aie_print_tile_status(int col, int row);

void aie_tile_reset(int col, int row);
void aie_tile_enable(int col, int row);

void xaie_device_init(void);
void xaie_array_reset(void);

/*
        Return the base address of the interface data structures
*/
uint64_t get_base_address(void);

#endif
