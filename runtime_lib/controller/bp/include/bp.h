// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef BP_H
#define BP_H

// This header defines the address space of BlackParrot, as viewed from the BP core

// BlackParrot address space (must be kept up to date with BlackParrot RTL)
// Refer to the BlackParrot Platform Guide for more details
//
// Local address map works as follows:
// address = 0x00_0(nnnN)(D)(A_AAAA)
// A_AAAA: 20-bit address space per device
// D: 4-bits to select device
// nnn_N: 7-bits to select tile (i.e., core) (= 0 for unicore designs)

// per-core Configuration Device
#define BP_CFG_OFFSET             0x200000LL
// only need first 4KiB, even though config technically has a 16MiB address space
#define BP_CFG_SIZE                 0x1000LL
#define BP_CFG_FREEZE_OFFSET           0x8LL
#define BP_CFG_NPC_OFFSET             0x10LL
#define BP_CFG_CORE_ID_OFFSET         0x18LL
#define BP_CFG_DID_OFFSET             0x20LL
#define BP_CFG_CORD_OFFSET            0x28LL
#define BP_CFG_HOST_DID_OFFSET        0x30LL
#define BP_CFG_HIO_MASK_OFFSET        0x38LL
#define BP_CFG_ICACHE_ID_OFFSET      0x200LL
#define BP_CFG_ICACHE_MODE_OFFSET    0x208LL
#define BP_CFG_DCACHE_ID_OFFSET      0x400LL
#define BP_CFG_DCACHE_MODE_OFFSET    0x408LL
#define BP_CFG_CCE_ID_OFFSET         0x600LL
#define BP_CFG_CCE_MODE_OFFSET       0x608LL

// BP Freeze - active high
#define BP_CFG_FREEZE_ON  0x1LL
#define BP_CFG_FREEZE_OFF 0x0LL

// BP cache modes
#define BP_CFG_CACHE_UNCACHED 0x0LL
#define BP_CFG_CACHE_NORMAL 0x1LL
#define BP_CFG_CACHE_NONSPEC 0x2LL

// BP cce mode
#define BP_CFG_CCE_UNCACHED 0x0LL
#define BP_CFG_CCE_NORMAL 0x1LL

// High I/O space mask
#define BP_CFG_HIO_MASK_ALL 0xFFFFFFFFLL

// BP CLINT (Core Local Interrupt Controller)
#define BP_CLINT_OFFSET 0x300000LL
#define BP_CLINT_SIZE 0x10000LL
#define BP_CLINT_MIPI_OFFSET 0x0LL
#define BP_CLINT_MTIMECMP_OFFSET 0x4000LL
#define BP_CLINT_MTIMESEL_OFFSET 0x8000LL
#define BP_CLINT_MTIME_OFFSET 0xBFF8LL
#define BP_CLINT_PLIC_OFFSET 0xB000LL
#define BP_CLINT_DEBUG_OFFSET 0xC000LL

// Local ID device (provides a global ID for the core)
// This device is external to BP, but mapped to a local I/O address
// that will route to the I/O network
#define BP_ID_OFFSET 0x40000000LL

// DRAM starts at 2GiB
// maximum cacheable DRAM is 2 GiB
// protoype design has 512 KiB "DRAM" that is backed by URAM blocks
#define BP_DRAM_OFFSET 0x80000000LL
#define BP_DRAM_SIZE    0x80000LL
#define BP_DRAM_HIGH (BP_DRAM_OFFSET+BP_DRAM_SIZE)

#endif // BP_H
