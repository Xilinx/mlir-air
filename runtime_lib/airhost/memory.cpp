//===- memory.cpp -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air_host.h"
#include "air_host_impl.h"

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fcntl.h>    /* for open() */
#include <string.h>   /* for memset() */
#include <sys/mman.h> /* for mlock() */
#include <unistd.h>   /* for getpagesize() */
#include <vector>

extern "C" {

extern air_rt_herd_desc_t _air_host_active_herd;
extern aie_libxaie_ctx_t *_air_host_active_libxaie;
extern uint32_t *_air_host_bram_ptr;
extern uint64_t _air_host_bram_paddr;
}

// Global variable
air_dev_mem_allocator_t *dev_mem_allocator = NULL;

#define PAGE_SHIFT 12
#define PAGEMAP_LENGTH 8

/* Used to get the PFN of a virtual address */
unsigned long get_page_frame_number_of_address(void *addr) {
  // Getting the pagemap file for the current process
  FILE *pagemap = fopen("/proc/self/pagemap", "rb");

  // Seek to the page that the buffer is on
  unsigned long offset = (unsigned long)addr / getpagesize() * PAGEMAP_LENGTH;
  if (fseek(pagemap, (unsigned long)offset, SEEK_SET) != 0) {
    printf("[ERROR] Failed to seek pagemap to proper location\n");
    exit(1);
  }

  // The page frame number is in bits 0 - 54 so read the first 7 bytes and clear
  // the 55th bit
  unsigned long page_frame_number = 0;
  fread(&page_frame_number, 1, PAGEMAP_LENGTH - 1, pagemap);
  page_frame_number &= 0x7FFFFFFFFFFFFF;

  fclose(pagemap);
  return page_frame_number;
}

/* This function is used to get the physical address of a buffer. */
uint64_t air_mem_get_paddr(void *buff) {
  // Getting the page frame the buffer is in
  unsigned long page_frame_number = get_page_frame_number_of_address(buff);

  // Getting the offset of the buffer into the page
  unsigned int distance_from_page_boundary =
      (unsigned long)buff % getpagesize();
  uint64_t paddr = (uint64_t)(page_frame_number << PAGE_SHIFT) +
                   (uint64_t)distance_from_page_boundary;

  return paddr;
}

void *air_mem_alloc(size_t size) {
  void *ptr = NULL;

  ptr = (void *)mmap(NULL, size, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);

  if (ptr == MAP_FAILED) {
    perror("mmap fails. ");
    return NULL;
  }

  /* obtain physical memory */
  // printf("obtain physical memory\n");
  memset(ptr, 1, size);

  /* lock the allocated memory in RAM */
  // printf("lock physical memory\n");
  mlock(ptr, size);

  return ptr;
}

int air_mem_free(void *buff, size_t size) { return munmap(buff, size); }

// Initializing the runtime's handle on the device memory allocator
int air_init_dev_mem_allocator(uint64_t dev_mem_size, uint32_t device_id) {

  // If already have a dev_mem_allocator just going to skip
  // initializing
  if (dev_mem_allocator != NULL) {
    return 0;
  }

  dev_mem_allocator =
      (air_dev_mem_allocator_t *)malloc(sizeof(air_dev_mem_allocator_t));
  if (dev_mem_allocator == NULL) {
    printf("[ERROR] Could not allocate dev_mem_allocator_t struct\n");
    return 1;
  }

  // Initializing new struct
  dev_mem_allocator->dev_mem_size = dev_mem_size;
  dev_mem_allocator->dev_mem_ptr = 0;

  // Getting userspace pointers to device memory
#ifdef AIR_PCIE
  int fd = open(air_get_driver_name(), O_RDWR | O_SYNC);
  if (fd < 0) {
    printf("[ERROR] Could not open DDR BAR\n");
    return 1;
  }
  dev_mem_allocator->dev_mem =
      (uint32_t *)mmap(NULL, dev_mem_size /*0x8000*/, PROT_READ | PROT_WRITE,
                       MAP_SHARED, fd, 0x1C0000);
  if (dev_mem_allocator->dev_mem == MAP_FAILED) {
    printf("[ERROR] Could not map DDR BAR\n");
    return 1;
  }
#else

#ifndef __aarch64__
  printf("[ERROR] Attempting to map /dev/mem on x86. Please define AIR_PCIE "
         "when compiling\n");
  return 1;
#endif

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd != -1) {
    dev_mem_allocator->dev_mem =
        (uint32_t *)mmap(NULL, dev_mem_size /*0x8000*/, PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, AIR_BBUFF_BASE);
  } else {
    printf("[ERROR] Could not open /dev/mem\n");
    return 1;
  }
#endif

  return 0;
}

// Freeing the device_memory_allocator
void air_dev_mem_allocator_free() {

  munmap(dev_mem_allocator->dev_mem, dev_mem_allocator->dev_mem_size);
  dev_mem_allocator = NULL;
  free(dev_mem_allocator);
}

// Allocating memory on the device. Since we are treeting the memory just like a
// stack, this is pretty straightforward as we are just giving the user the
// next portion of memory of size that they want.
void *air_dev_mem_alloc(uint32_t size) {

  // Making sure we have a real allocator
  if (dev_mem_allocator == NULL) {
    printf(
        "[ERROR] Attempting to allocate device memory without a valid device "
        "memory allocator. Call air_init_dev_mem_allocator() first\n");
    return NULL;
  }

  // Making sure we have enough space on the device
  if (size + dev_mem_allocator->dev_mem_ptr > dev_mem_allocator->dev_mem_size) {
    printf("[ERROR] Device memory cannot accept this allocation due to lack of "
           "space\n");
    return NULL;
  }

  // Setting the user pointer equal to the next portion
  // of available memory
  void *user_ptr = (void *)((unsigned char *)dev_mem_allocator->dev_mem +
                            dev_mem_allocator->dev_mem_ptr);

  // Incrementing pointer by the size of memory allocated
  dev_mem_allocator->dev_mem_ptr += size;

  return user_ptr;
}

// Used to get the physical address of device allocated through
// the device memory allocator. Due to how memory is allocated
// in both platforms, the offsets of the virtual and physical
// address are the same, and we can directly convert between
// the two.
uint64_t air_dev_mem_get_pa(void *buff_va) {

  // Making sure we have a real allocator
  if (dev_mem_allocator == NULL) {
    printf(
        "[ERROR] Attempting to get a physical address without a valid device "
        "memory allocator. Call air_init_dev_mem_allocator() first\n");
    return 0;
  }

  // Get the virtual address offset
  uint64_t offset = (uint64_t)buff_va - (uint64_t)(dev_mem_allocator->dev_mem);

  // Adding that offset to our base physical address
  return offset + (uint64_t)AIR_BBUFF_BASE;
}

static int64_t shim_location_data(air_herd_shim_desc_t *sd, int i, int j,
                                  int k) {
  return sd->location_data[i * 8 * 8 + j * 8 + k];
}

static int64_t shim_channel_data(air_herd_shim_desc_t *sd, int i, int j,
                                 int k) {
  return sd->channel_data[i * 8 * 8 + j * 8 + k];
}

template <typename T, int R>
static void air_mem_shim_nd_memcpy_queue_impl(
    signal_t *s, uint32_t id, uint64_t x, uint64_t y, tensor_t<T, R> *t,
    uint32_t space, uint64_t offset_3, uint64_t offset_2, uint64_t offset_1,
    uint64_t offset_0, uint64_t length_4d, uint64_t length_3d,
    uint64_t length_2d, uint64_t length_1d, uint64_t stride_4d,
    uint64_t stride_3d, uint64_t stride_2d) {
  assert(_air_host_active_herd.herd_desc &&
         "cannot shim memcpy without active herd");
  assert(_air_host_active_herd.q &&
         "cannot shim memcpy using a queue without active queue");

  auto shim_desc = _air_host_active_herd.herd_desc->shim_desc;
  auto shim_col = shim_location_data(shim_desc, id - 1, x, y);
  auto shim_chan = shim_channel_data(shim_desc, id - 1, x, y);

  // printf("Do transfer %p with id %d on behalf of x=%ld, y=%ld space %d, col
  // %d, dir %d, chan %d, offset [%ld,%ld,%ld,%ld], length [%ld,%ld,%ld,%ld],
  // stride [%ld,%ld,%ld]\n",
  //       t->data, id, x, y, space, shim_col, shim_chan>=2, (shim_chan>=2) ?
  //       shim_chan-2 : shim_chan, offset_3, offset_2, offset_1, offset_0,
  //       length_4d, length_3d, length_2d, length_1d,
  //       stride_4d, stride_3d, stride_2d);

  bool isMM2S = shim_chan >= 2;

  bool uses_pa = (space == 1); // t->uses_pa;
  if (uses_pa) {
    if (isMM2S)
      shim_chan = shim_chan - 2;

    size_t stride = 1;
    size_t offset = 0;
    std::vector<uint64_t> offsets{offset_0, offset_1, offset_2, offset_3};
    for (int i = 0; i < R; i++) {
      offset += offsets[i] * stride * sizeof(T);
      stride *= t->shape[R - i - 1];
    }

    uint64_t wr_idx = queue_add_write_index(_air_host_active_herd.q, 1);
    uint64_t packet_id = wr_idx % _air_host_active_herd.q->size;

    dispatch_packet_t *pkt =
        (dispatch_packet_t *)(_air_host_active_herd.q->base_address_vaddr) +
        packet_id;
    air_packet_nd_memcpy(
        pkt, /*herd_id=*/0, shim_col, /*direction=*/isMM2S, shim_chan,
        /*burst_len=*/4, /*memory_space=*/space, (uint64_t)t->data + offset,
        length_1d * sizeof(T), length_2d, stride_2d * sizeof(T), length_3d,
        stride_3d * sizeof(T), length_4d, stride_4d * sizeof(T));
    if (s) {
      air_queue_dispatch(_air_host_active_herd.q, wr_idx, pkt);
      uint64_t signal_offset = offsetof(dispatch_packet_t, completion_signal);
      s->handle = queue_paddr_from_index(
          _air_host_active_herd.q,
          (packet_id) * sizeof(dispatch_packet_t) + signal_offset);
    } else {
      air_queue_dispatch_and_wait(_air_host_active_herd.q, wr_idx, pkt);
    }
    return;
  } else {
    uint32_t *bounce_buffer = _air_host_bram_ptr;

    size_t stride = 1;
    size_t offset = 0;
    std::vector<uint64_t> offsets{offset_0, offset_1, offset_2, offset_3};
    for (int i = 0; i < R; i++) {
      offset += offsets[i] * stride * sizeof(T);
      stride *= t->shape[R - i - 1];
    }

    uint64_t length = 0;
    for (uint32_t index_4d = 0; index_4d < length_4d; index_4d++)
      for (uint32_t index_3d = 0; index_3d < length_3d; index_3d++)
        for (uint32_t index_2d = 0; index_2d < length_2d; index_2d++)
          length += length_1d;

    size_t p = (size_t)t->data + offset;
    uint64_t paddr_4d = p;
    uint64_t paddr_3d = p;
    uint64_t paddr_2d = p;
    uint64_t paddr_1d = p;

    if (isMM2S) {
      shim_chan = shim_chan - 2;
      for (uint32_t index_4d = 0; index_4d < length_4d; index_4d++) {
        paddr_2d = paddr_3d;
        for (uint32_t index_3d = 0; index_3d < length_3d; index_3d++) {
          paddr_1d = paddr_2d;
          for (uint32_t index_2d = 0; index_2d < length_2d; index_2d++) {
            memcpy((size_t *)bounce_buffer, (size_t *)paddr_1d,
                   length_1d * sizeof(T));
            bounce_buffer += length_1d;
            paddr_1d += stride_2d * sizeof(T);
          }
          paddr_2d += stride_3d * sizeof(T);
        }
        paddr_3d += stride_4d * sizeof(T);
      }
    }

    uint64_t wr_idx = queue_add_write_index(_air_host_active_herd.q, 1);
    uint64_t packet_id = wr_idx % _air_host_active_herd.q->size;

    dispatch_packet_t *pkt =
        (dispatch_packet_t *)(_air_host_active_herd.q->base_address_vaddr) +
        packet_id;
    air_packet_nd_memcpy(pkt, /*herd_id=*/0, shim_col, /*direction=*/isMM2S,
                         shim_chan, /*burst_len=*/4, /*memory_space=*/2,
                         _air_host_bram_paddr, length * sizeof(T), 1, 0, 1, 0,
                         1, 0);
    if (s) {
      // TODO: don't block here
      air_queue_dispatch_and_wait(_air_host_active_herd.q, wr_idx, pkt);
      uint64_t signal_offset = offsetof(dispatch_packet_t, completion_signal);
      s->handle = queue_paddr_from_index(
          _air_host_active_herd.q,
          (packet_id) * sizeof(dispatch_packet_t) + signal_offset);
    } else {
      air_queue_dispatch_and_wait(_air_host_active_herd.q, wr_idx, pkt);
    }
    if (!isMM2S) {
      for (uint32_t index_4d = 0; index_4d < length_4d; index_4d++) {
        paddr_2d = paddr_3d;
        for (uint32_t index_3d = 0; index_3d < length_3d; index_3d++) {
          paddr_1d = paddr_2d;
          for (uint32_t index_2d = 0; index_2d < length_2d; index_2d++) {
            memcpy((size_t *)paddr_1d, (size_t *)bounce_buffer,
                   length_1d * sizeof(T));
            bounce_buffer += length_1d;
            paddr_1d += stride_2d * sizeof(T);
          }
          paddr_2d += stride_3d * sizeof(T);
        }
        paddr_3d += stride_4d * sizeof(T);
      }
    }
  }
}

#define mlir_air_dma_nd_memcpy(mangle, rank, space, type)                      \
  void _mlir_ciface___airrt_dma_nd_memcpy_##mangle(                            \
      signal_t *s, uint32_t id, uint64_t x, uint64_t y, void *t,               \
      uint64_t offset_3, uint64_t offset_2, uint64_t offset_1,                 \
      uint64_t offset_0, uint64_t length_3, uint64_t length_2,                 \
      uint64_t length_1, uint64_t length_0, uint64_t stride_2,                 \
      uint64_t stride_1, uint64_t stride_0) {                                  \
    tensor_t<type, rank> *tt = (tensor_t<type, rank> *)t;                      \
    if (_air_host_active_herd.q) {                                             \
      air_mem_shim_nd_memcpy_queue_impl(                                       \
          s, id, x, y, tt, space, offset_3, offset_2, offset_1, offset_0,      \
          length_3, length_2, length_1, length_0, stride_2, stride_1,          \
          stride_0);                                                           \
    } else {                                                                   \
      printf(                                                                  \
          "WARNING: no queue provided. ND memcpy will not be performed.\n");   \
    }                                                                          \
  }

extern "C" {

mlir_air_dma_nd_memcpy(1d0i32, 1, 2, uint32_t);
mlir_air_dma_nd_memcpy(2d0i32, 2, 2, uint32_t);
mlir_air_dma_nd_memcpy(3d0i32, 3, 2, uint32_t);
mlir_air_dma_nd_memcpy(4d0i32, 4, 2, uint32_t);
mlir_air_dma_nd_memcpy(1d0f32, 1, 2, float);
mlir_air_dma_nd_memcpy(2d0f32, 2, 2, float);
mlir_air_dma_nd_memcpy(3d0f32, 3, 2, float);
mlir_air_dma_nd_memcpy(4d0f32, 4, 2, float);
mlir_air_dma_nd_memcpy(1d1i32, 1, 1, uint32_t);
mlir_air_dma_nd_memcpy(2d1i32, 2, 1, uint32_t);
mlir_air_dma_nd_memcpy(3d1i32, 3, 1, uint32_t);
mlir_air_dma_nd_memcpy(4d1i32, 4, 1, uint32_t);
mlir_air_dma_nd_memcpy(1d1f32, 1, 1, float);
mlir_air_dma_nd_memcpy(2d1f32, 2, 1, float);
mlir_air_dma_nd_memcpy(3d1f32, 3, 1, float);
mlir_air_dma_nd_memcpy(4d1f32, 4, 1, float);

} // extern "C"

#define mlir_air_nd_memcpy(mangle, rank0, space0, type0, rank1, space1, type1) \
  void _mlir_ciface___airrt_nd_memcpy_##mangle(                                \
      void *t0, void *t1, uint64_t offset_3, uint64_t offset_2,                \
      uint64_t offset_1, uint64_t offset_0, uint64_t length_3,                 \
      uint64_t length_2, uint64_t length_1, uint64_t length_0,                 \
      uint64_t stride_2, uint64_t stride_1, uint64_t stride_0) {               \
    tensor_t<type0, rank0> *tt0 = (tensor_t<type0, rank0> *)t0;                \
    tensor_t<type1, rank1> *tt1 = (tensor_t<type1, rank1> *)t1;                \
    if (_air_host_active_herd.q) {                                             \
      printf("WARNING: ND memcpy will not be performed.\n");                   \
    } else {                                                                   \
      printf(                                                                  \
          "WARNING: no queue provided. ND memcpy will not be performed.\n");   \
    }                                                                          \
  }

extern "C" {

mlir_air_nd_memcpy(1d0i32_1d1i32, 1, 2, uint32_t, 1, 1, uint32_t);
mlir_air_nd_memcpy(1d1i32_1d0i32, 1, 1, uint32_t, 1, 2, uint32_t);
mlir_air_nd_memcpy(2d0i32_2d1i32, 2, 2, uint32_t, 2, 1, uint32_t);
mlir_air_nd_memcpy(2d1i32_2d0i32, 2, 1, uint32_t, 2, 2, uint32_t);
mlir_air_nd_memcpy(3d0i32_3d1i32, 3, 2, uint32_t, 3, 1, uint32_t);
mlir_air_nd_memcpy(3d1i32_3d0i32, 3, 1, uint32_t, 3, 2, uint32_t);
mlir_air_nd_memcpy(4d0i32_4d1i32, 4, 2, uint32_t, 4, 1, uint32_t);
mlir_air_nd_memcpy(4d1i32_4d0i32, 4, 1, uint32_t, 4, 2, uint32_t);
mlir_air_nd_memcpy(1d0f32_1d1f32, 1, 2, float, 1, 1, float);
mlir_air_nd_memcpy(1d1f32_1d0f32, 1, 1, float, 1, 2, float);
mlir_air_nd_memcpy(2d0f32_2d1f32, 2, 2, float, 2, 1, float);
mlir_air_nd_memcpy(2d1f32_2d0f32, 2, 1, float, 2, 2, float);
mlir_air_nd_memcpy(3d0f32_3d1f32, 3, 2, float, 3, 1, float);
mlir_air_nd_memcpy(3d1f32_3d0f32, 3, 1, float, 3, 2, float);
mlir_air_nd_memcpy(4d0f32_4d1f32, 4, 2, float, 4, 1, float);
mlir_air_nd_memcpy(4d1f32_4d0f32, 4, 1, float, 4, 2, float);

} // extern "C"
