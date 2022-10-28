//===- pcie-ernic-dev-mem-allocator.cpp ---------------------------*- C++
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

#include "include/pcie-ernic-dev-mem-allocator.h"

struct pcie_ernic_dev_mem_allocator *
init_dev_mem_allocator(const char *dev_mem_bar_filename, uint64_t dev_mem_bar_size,
                       uint64_t dev_mem_global_offset,
                       uint64_t dev_mem_partition_offset) {

  // Allocating memory for allocator structure
  struct pcie_ernic_dev_mem_allocator *allocator =
      (struct pcie_ernic_dev_mem_allocator *)malloc(
          sizeof(struct pcie_ernic_dev_mem_allocator));

  // Initialize components of the allocator
  allocator->dev_mem_bar_filename = dev_mem_bar_filename;
  allocator->dev_mem_ptr = 0;
  allocator->dev_mem_size = dev_mem_bar_size;
  allocator->partition_offset = dev_mem_partition_offset;
  allocator->global_offset = dev_mem_global_offset;

  // Map the
  int axib_fd;
  if ((axib_fd = open(dev_mem_bar_filename, O_RDWR | O_SYNC)) == -1) {
    printf("[ERROR] Failed to open device file: %s\n", dev_mem_bar_filename);
    return NULL;
  }

  printf("Opening %s with size %lu\n", dev_mem_bar_filename, dev_mem_bar_size);

  allocator->dev_mem = mmap(NULL,                   // virtual address
                            dev_mem_bar_size,       // length
                            PROT_READ | PROT_WRITE, // prot
                            MAP_SHARED,             // flags
                            axib_fd,                // device fd
                            0);

  printf("[INFO] Device memory mapped into userspace\n");
  printf("\tVA: %p\n", allocator->dev_mem);
  printf("\tPartition Offset: 0x%lx\n", allocator->partition_offset);
  printf("\tGlobal Offset: 0x%lx\n", allocator->global_offset);
  printf("\tSize: %lu\n", allocator->dev_mem_size);

  return allocator;
}

void free_dev_mem_allocator(struct pcie_ernic_dev_mem_allocator *allocator) {

  // Setting device memory pointer to zero
  allocator->dev_mem_ptr = 0;

  // Unmapping the device memory
  if (munmap(allocator->dev_mem, allocator->dev_mem_size) == -1) {
    printf("[ERROR] Failed to unmap device memory\n");
  }

  // Free the entire thing
  free(allocator);

#ifdef VERBOSE_DEBUG
  printf("[INFO] Freeing device memory allocator\n");
#endif
}

// Allocating memory on the device. Since we are treating the memory just like a
// stack, this is pretty straightforward as we are just giving the user the
// next portion of memory equal to the size that they want. Also, if user gives
// a non NULL uint64_t pointer, we will provide the PA which is useful for some
// applications to know -- Note the PA is the physical address in the device
// memory map, not the memory map of the CPU.
void *dev_mem_alloc(struct pcie_ernic_dev_mem_allocator *allocator,
                    uint32_t size, uint64_t *pa) {

  // Making sure we are given a real allocator
  if (allocator == NULL) {
    printf("[ERROR] dev_mem_alloc given NULL allocator\n");
    return NULL;
  }

  // Making sure we have enough space on the device
  if (size + allocator->dev_mem_ptr + allocator->partition_offset >
      allocator->dev_mem_size) {
    printf("[ERROR] Device memory cannot accept this allocation due to lack of "
           "space\n");
    return NULL;
  }

  // If user provided valid pointer, give the physical address
  if (pa != NULL) {
    *pa = allocator->dev_mem_ptr + allocator->partition_offset +
          allocator->global_offset /*DEV_MEM_OFFSET*/;
  }

  // Setting the user pointer equal to the next portion
  // of available memory
  void *user_ptr = (void *)((unsigned char *)allocator->dev_mem + allocator->partition_offset + allocator->dev_mem_ptr);

#ifdef VERBOSE_DEBUG
  printf("Giving user %dB starting at dev_mem[0x%lx]\n", size,
         allocator->dev_mem_ptr + allocator->partition_offset);
#endif

  // Incrementing pointer by the size of memory allocated
  allocator->dev_mem_ptr += size;

  return user_ptr;
}
