//===- pcie-ernic-dev-mem-allocator.h ----------------------------*- C++
//-*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
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
  uint64_t partition_offset; // Need an offset in case multiple processes are
                             // using device memory
  uint64_t global_offset; // This is the offset in the hardware memory map so we
                          // can directly address device memory
};

struct pcie_ernic_dev_mem_allocator *init_dev_mem_allocator(
    const char *dev_mem_bar_filename, uint64_t dev_mem_bar_size,
    uint64_t dev_mem_global_offset, uint64_t dev_mem_partition_offset);
void free_dev_mem_allocator(struct pcie_ernic_dev_mem_allocator *allocator);
void *dev_mem_alloc(struct pcie_ernic_dev_mem_allocator *allocator,
                    uint32_t size, uint64_t *pa);

#endif
