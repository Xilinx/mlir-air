//===- air_queue.h ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef ACDC_QUEUE_H
#define ACDC_QUEUE_H

#include "hsa/hsa.h"

#include <stdint.h>

// Default BRAM layout is as follows:
// 1 system page
// 1 queue descriptor page, with 16 queue descriptors
// N (=9 for now, 1 for each BP and 1 for ARM) pages for queue ring buffers
// Remaining pages for signals, doorbells, etc

// Define the number of HSA packets we can have in a queue
// as well as the number of pages used for different purposes in BRAM
#define MB_QUEUE_SIZE 64
#define NUM_SYSTEM_PAGES 1
#define NUM_QUEUE_STRUCT_PAGES 1
#define NUM_QUEUE_BUFFER_PAGES 9
#define NUM_SIGNAL_PAGES 16
#define NUM_DOORBELL_PAGES 1

// Should be no need to change below this line
#define MB_PAGE_SIZE 0x1000

#define MB_SHMEM_QUEUE_STRUCT_OFFSET NUM_SYSTEM_PAGES *MB_PAGE_SIZE
#define MB_SHMEM_QUEUE_STRUCT_SIZE NUM_QUEUE_STRUCT_PAGES *MB_PAGE_SIZE

#define MB_SHMEM_QUEUE_BUFFER_OFFSET                                           \
  MB_SHMEM_QUEUE_STRUCT_OFFSET + MB_SHMEM_QUEUE_STRUCT_SIZE
#define MB_SHMEM_QUEUE_BUFFER_SIZE NUM_QUEUE_BUFFER_PAGES *MB_PAGE_SIZE

// Area of memory that can be used for signals.
// A controller will initialize these to zero.
#define MB_SHMEM_SIGNAL_OFFSET                                                 \
  MB_SHMEM_QUEUE_BUFFER_OFFSET + MB_SHMEM_QUEUE_BUFFER_SIZE
#define MB_SHMEM_SIGNAL_SIZE NUM_SIGNAL_PAGES *MB_PAGE_SIZE

#define MB_SHMEM_DOORBELL_OFFSET MB_SHMEM_SIGNAL_OFFSET + MB_SHMEM_SIGNAL_SIZE
#define MB_SHMEM_DOORBELL_SIZE NUM_DOORBELL_PAGES *MB_PAGE_SIZE

#endif
