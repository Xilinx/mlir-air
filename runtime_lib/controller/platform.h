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

#define NPI_BASE 0xF70A0000UL

#define DRAM_1_BASE 0x000800000000ULL
#define DRAM_1_SIZE 0x000380000000ULL

#ifdef ARM_CONTROLLER
#define CDMA_BASE 0x0000A4000000UL
#else
#define CDMA_BASE 0x000044000000UL
#endif // ARM_CONTROLLER

#define IO_READ32(addr) *((volatile uint32_t *)(addr))
#define IO_WRITE32(addr, val) *((volatile uint32_t *)(addr)) = val

void init_platform();
void cleanup_platform();

/*
        Return the base address of the interface data structures
*/
uint64_t get_base_address(void);

#endif
