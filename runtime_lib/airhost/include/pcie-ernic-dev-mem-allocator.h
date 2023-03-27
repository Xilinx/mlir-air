//===- pcie-ernic-dev-mem-allocator.h --------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef PCIE_ERNIC_DEV_MEM_ALLOCATOR_H
#define PCIE_ERNIC_DEV_MEM_ALLOCATOR_H

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <termios.h>
#include <unistd.h>

// #include "pcie-bdf.h"

// Defining our memory allocator. Right now going
// to implement it as a stack, where we can just
// make arbitrary allocations to it, and it grows
// until it runs out of space. I guess that is the
// easiest for now
struct pcie_ernic_dev_mem_allocator {
  void *dev_mem;                    // Pointing to device BAR
  const char *dev_mem_bar_filename; // BAR which is backed by device memory
  uint64_t dev_mem_ptr;             // Points to the top of the device memory
  uint64_t dev_mem_size; // The total size of the device memory so we can report
                         // errors when too much is requested
  uint64_t segment_offset; // Need an offset in case multiple processes are
                           // using device memory
  uint64_t global_offset; // This is the offset in the hardware memory map so we
                          // can directly address device memory
};

struct pcie_ernic_dev_mem_allocator *init_dev_mem_allocator(
    const char *dev_mem_bar_filename, uint64_t dev_mem_bar_size,
    uint64_t dev_mem_global_offset, uint64_t dev_mem_segment_offset);
void free_dev_mem_allocator(struct pcie_ernic_dev_mem_allocator *allocator);
void *dev_mem_alloc(struct pcie_ernic_dev_mem_allocator *allocator,
                    uint32_t size, uint64_t *pa);

#endif
