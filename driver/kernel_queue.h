//===- kernel_queue.h ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef KERNEL_QUEUE_H
#define KERNEL_QUEUE_H

#include "hsa_defs.h"

/* Offsets into BRAM BAR */
#define HERD_CONTROLLER_BASE_ADDR(_base, _x)                                   \
	((((uint64_t)ioread32(_base + (_x * sizeof(uint64_t) + 4))) << 32) |   \
	 ioread32(_base + (_x * sizeof(uint64_t) + 0)))
#define REG_HERD_CONTROLLER_COUNT 0x208
#define BRAM_PADDR 0x20100000000

// How a queue is defined
#define QUEUE_TYPE_OFFSET 0x00
#define QUEUE_DOORBELL_OFFSET 0x10
#define QUEUE_SIZE_OFFSET 0x18
#define QUEUE_ID_OFFSET 0x20
#define QUEUE_RD_PTR_OFFSET 0x28
#define QUEUE_WR_PTR_OFFSET 0x30
#define QUEUE_RING_OFFSET 0x80
#define QUEUE_TIMEOUT_VAL 5000000

// How a packet is defined
#define PKT_SIZE 0x40
#define PKT_HEADER_TYPE_OFFSET 0x00
#define PKT_RET_ADDR_OFFSET 0x08
#define PKT_ARG_ADDR_OFFSET 0x10
#define PKT_COMPL_OFFSET 0x38
#define NUM_PKTS 0x30

// Define the number of HSA packets we can have in a queue
#define MB_QUEUE_SIZE 48

// Define the amount of shared memory accessible to each controller
// This includes the queues, events, doorbells etc
#define MB_SHMEM_SEGMENT_SIZE 0x1000

// A small area of memory that can be used for signals.
// A controller will initialize these to zero.
#define MB_SHMEM_SIGNAL_OFFSET 0x0300
#define MB_SHMEM_SIGNAL_SIZE 0x0100

// See
// https://confluence.xilinx.com/display/XRLABS/AIR+Controller+HSA+Packet+Formats
// All defined as longs, so we can shift them into 64 registers

#define AIR_PKT_TYPE_INVALID 0x0000L
#define AIR_PKT_TYPE_PUT_STREAM 0x0001L
#define AIR_PKT_TYPE_GET_STREAM 0x0002L
#define AIR_PKT_TYPE_SDMA_STATUS 0x0003L
#define AIR_PKT_TYPE_TDMA_STATUS 0x0004L
#define AIR_PKT_TYPE_CORE_STATUS 0x0005L

#define AIR_PKT_TYPE_DEVICE_INITIALIZE 0x0010L
#define AIR_PKT_TYPE_SEGMENT_INITIALIZE 0x0011L
#define AIR_PKT_TYPE_HELLO 0x0012L
#define AIR_PKT_TYPE_ALLOCATE_HERD_SHIM_DMAS 0x0013L
#define AIR_PKT_TYPE_GET_CAPABILITIES 0x0014L
#define AIR_PKT_TYPE_GET_INFO 0x0015L

#define AIR_PKT_TYPE_XAIE_LOCK 0x0020L

#define AIR_PKT_TYPE_CDMA 0x030L
#define AIR_PKT_TYPE_CONFIGURE 0x031L

#define AIR_PKT_TYPE_POST_RDMA_WQE 0x040L
#define AIR_PKT_TYPE_POST_RDMA_RECV 0x041L

#define AIR_PKT_TYPE_RW32 0x50L

#define AIR_PKT_TYPE_SHIM_DMA_MEMCPY 0x0100L
#define AIR_PKT_TYPE_HERD_SHIM_DMA_MEMCPY 0x0101L
#define AIR_PKT_TYPE_HERD_SHIM_DMA_1D_STRIDED_MEMCPY 0x0102L
#define AIR_PKT_TYPE_ND_MEMCPY 0x0103L

#define AIR_ADDRESS_ABSOLUTE 0x0L
#define AIR_ADDRESS_ABSOLUTE_RANGE 0x1L
#define AIR_ADDRESS_HERD_RELATIVE 0x2L
#define AIR_ADDRESS_HERD_RELATIVE_RANGE 0x3L

typedef enum {
	AIR_AGENT_INFO_NAME = 0, // NUL-terminated char[8]
	AIR_AGENT_INFO_VENDOR_NAME = 1, // NUL-terminated char[8]
	AIR_AGENT_INFO_CONTROLLER_ID = 2,
	AIR_AGENT_INFO_FIRMWARE_VER = 3,
	AIR_AGENT_INFO_NUM_REGIONS = 4,
	AIR_AGENT_INFO_HERD_SIZE = 5,
	AIR_AGENT_INFO_HERD_ROWS = 6,
	AIR_AGENT_INFO_HERD_COLS = 7,
	AIR_AGENT_INFO_TILE_DATA_MEM_SIZE = 8,
	AIR_AGENT_INFO_TILE_PROG_MEM_SIZE = 9,
	AIR_AGENT_INFO_L2_MEM_SIZE = 10 // Per region
} air_agent_info_t;

// Note below that "__attribute__((packed))" also asserts that the whole
// structure is unaligned in some compilers.  This helps to silence errors from
// -waddress-of-packed-struct

typedef struct dispatch_packet_s {
	// HSA-like interface
	volatile uint16_t header;
	volatile uint16_t type;
	uint32_t reserved0;
	uint64_t return_address;
	uint64_t arg[4];
	uint64_t reserved1;
	uint64_t completion_signal;

} __attribute__((packed, aligned(__alignof__(uint64_t)))) dispatch_packet_t;

typedef struct barrier_and_packet_s {
	// HSA-like interface
	volatile uint16_t header;
	uint16_t reserved0;
	uint32_t reserved1;
	uint64_t dep_signal[5];
	uint64_t reserved2;
	uint64_t completion_signal;

} __attribute__((packed, aligned(__alignof__(uint64_t)))) barrier_and_packet_t;

typedef struct barrier_or_packet_s {
	// HSA-like interface
	volatile uint16_t header;
	uint16_t reserved0;
	uint32_t reserved1;
	uint64_t dep_signal[5];
	uint64_t reserved2;
	uint64_t completion_signal;

} __attribute__((packed, aligned(__alignof__(uint64_t)))) barrier_or_packet_t;

typedef struct queue_s {
	// HSA-like interface
	uint32_t type;
	uint32_t features;
	uint64_t base_address;
	volatile uint64_t doorbell;
	uint32_t size;
	uint32_t reserved0;
	uint64_t id;

	// implementation detail
	uint64_t read_index;
	uint64_t write_index;
	uint64_t last_doorbell;

	uint64_t base_address_paddr;
	uint64_t base_address_vaddr;

} __attribute__((packed, aligned(__alignof__(uint64_t)))) queue_t;

typedef struct signal_s {
	uint64_t handle;
} signal_t;

typedef uint64_t signal_value_t;

// Function declarations
hsa_status_t air_queue_create(uint32_t size, uint32_t type,
			      struct vck5000_device *dev);

#endif
