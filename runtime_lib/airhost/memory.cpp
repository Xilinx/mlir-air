//===- memory.cpp -----------------------------------------------*- C++ -*-===//
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

#include "air_host.h"
#include "air_tensor.h"

#include <cassert>
#include <vector>
#include <cstdio>
#include <cstring>
#include <unistd.h>     /* for getpagesize() */
#include <sys/mman.h>   /* for mlock() */
#include <string.h>     /* for memset() */
#include <fcntl.h>      /* for open() */

//#include <xaiengine.h>

extern "C" {

extern air_rt_herd_desc_t _air_host_active_herd;
extern aie_libxaie_ctx_t *_air_host_active_libxaie1;
extern uint32_t *_air_host_bram_ptr;
extern uint64_t _air_host_bram_paddr;

}


#define PAGE_SHIFT 12
#define PAGEMAP_LENGTH 8

/* Used to get the PFN of a virtual address */
unsigned long get_page_frame_number_of_address(void *addr) {
  // Getting the pagemap file for the current process
  FILE *pagemap = fopen("/proc/self/pagemap", "rb");

  // Seek to the page that the buffer is on
  unsigned long offset = (unsigned long)addr / getpagesize() * PAGEMAP_LENGTH;
  if(fseek(pagemap, (unsigned long)offset, SEEK_SET) != 0) { 
      printf("[ERROR] Failed to seek pagemap to proper location\n");
      exit(1);
  } 

  // The page frame number is in bits 0 - 54 so read the first 7 bytes and clear the 55th bit
  unsigned long page_frame_number = 0;
  fread(&page_frame_number, 1, PAGEMAP_LENGTH-1, pagemap);
  page_frame_number &= 0x7FFFFFFFFFFFFF;

  fclose(pagemap);
  return page_frame_number;
}

/* This function is used to get the physical address of a buffer. */
uint64_t air_mem_get_paddr(void *buff) {
  // Getting the page frame the buffer is in
  unsigned long page_frame_number = get_page_frame_number_of_address(buff);
  
  // Getting the offset of the buffer into the page
  unsigned int distance_from_page_boundary = (unsigned long)buff % getpagesize();
  uint64_t paddr = (uint64_t)(page_frame_number << PAGE_SHIFT) + (uint64_t)distance_from_page_boundary;
  
  return paddr;
}

void* air_mem_alloc(size_t size) {
  void *ptr = NULL;

  ptr = (void*)mmap(NULL,
            size,
            PROT_READ | PROT_WRITE,
            MAP_SHARED | MAP_ANONYMOUS | MAP_HUGETLB,
            -1,
            0);

  if (!ptr) {
    perror("mmap fails. ");
    return NULL;
  }

  /* obtain physical memory */
  //printf("obtain physical memory\n");
  memset(ptr, 1, size);

  /* lock the allocated memory in RAM */
  //printf("lock physical memory\n");
  mlock(ptr, size);

  return ptr;
}

int air_mem_free(void *buff, size_t size) {
  return munmap(buff,size);
}

namespace {

int64_t shim_location_data(air_herd_shim_desc_t *sd, int i, int j, int k) {
  return sd->location_data[i*8*8 + j*8 + k];
}

int64_t shim_channel_data(air_herd_shim_desc_t *sd, int i, int j, int k) {
  return sd->channel_data[i*8*8 + j*8 + k];
}

}

#define HIGH_ADDR(addr)	((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr)	(addr & 0x00000000ffffffff)

namespace {

void air_shim_memcpy4d_queue_impl(uint32_t id, uint64_t x, uint64_t y, void* t,
                                  uint64_t offset_3, uint64_t offset_2, uint64_t offset_1, uint64_t offset_0,
                                  uint64_t length, uint64_t stride, uint64_t elem_per_stride) {

  assert(_air_host_active_herd.herd_desc && "cannot shim memcpy without active herd");
  assert(_air_host_active_herd.q && "cannot shim memcpy using a queue without active queue");

  auto shim_desc = _air_host_active_herd.herd_desc->shim_desc;
  auto shim_col = shim_location_data(shim_desc, id-1, x, y);
  auto shim_chan = shim_channel_data(shim_desc, id-1, x, y);

  tensor_t<uint32_t,4> *tt = (tensor_t<uint32_t,4> *)t;

  // printf("Do queue transfer %p with id %ld of length %ld on behalf of x=%ld, y=%ld shim col %ld channel %ld, offset %ld,%ld,%ld,%ld, stride %ld, elem %ld\n",
  //         tt->data, id, length, x, y, shim_col, shim_chan, offset_3, offset_2, offset_1, offset_0, stride, elem_per_stride);

  uint32_t *bounce_buffer = _air_host_bram_ptr;
  bool isMM2S = shim_chan >= 2;

  uint32_t *data_ptr = tt->data + (offset_3 * tt->shape[3] * tt->shape[2] * tt->shape[1]) +
                                (offset_2 * tt->shape[3] * tt->shape[2]) +
                                (offset_1 * tt->shape[3]) + 
                                offset_0;
  if (isMM2S) {
    shim_chan = shim_chan - 2;
    uint32_t *bounce_ptr = bounce_buffer;
    for (int n=0; n<length; n+=elem_per_stride) {
      // This is the input, so we need to take what is in t and put it into the BRAM
      memcpy((void*)bounce_ptr, data_ptr, elem_per_stride*sizeof(uint32_t));
      data_ptr += stride;
      bounce_ptr += elem_per_stride;
    }
  }

  uint64_t wr_idx = queue_add_write_index(_air_host_active_herd.q, 1);
  uint64_t packet_id = wr_idx % _air_host_active_herd.q->size;

  dispatch_packet_t *pkt = (dispatch_packet_t*)(_air_host_active_herd.q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt, /*herd_id=*/0, shim_col, /*direction=*/isMM2S, shim_chan, /*burst_len=*/4, /*memory_space=*/2,
                       _air_host_bram_paddr, length*sizeof(uint32_t), 1, 0, 1, 0, 1, 0);
  //if (s) {
  //  // TODO: don't block here
  //  air_queue_dispatch_and_wait(_air_host_active_herd.q, wr_idx, pkt);
  //  uint64_t signal_offset = offsetof(dispatch_packet_t, completion_signal);
  //  s->handle = queue_paddr_from_index(_air_host_active_herd.q,
  //                                     (packet_id) * sizeof(dispatch_packet_t) +
  //                                         signal_offset);
  //} else {
    air_queue_dispatch_and_wait(_air_host_active_herd.q, wr_idx, pkt);
  //}

  if (!isMM2S) {
    volatile uint32_t *bounce_ptr = bounce_buffer;
    for (int n=0; n<length; n+=elem_per_stride) {
      // This is the input, so we need to take what is in t and put it into the BRAM
      memcpy(data_ptr, (void*)bounce_ptr, elem_per_stride*sizeof(uint32_t));
      data_ptr += stride;
      bounce_ptr += elem_per_stride;
    }
  }
}

void air_shim_memcpy2d_queue_impl(uint32_t id, uint64_t x, uint64_t y, void* t,
                                  uint64_t offset_y, uint64_t offset_x,
                                  uint64_t length, uint64_t stride, uint64_t elem_per_stride) {

  assert(_air_host_active_herd.herd_desc && "cannot shim memcpy without active herd");
  assert(_air_host_active_herd.q && "cannot shim memcpy using a queue without active queue");

  auto shim_desc = _air_host_active_herd.herd_desc->shim_desc;
  auto shim_col = shim_location_data(shim_desc, id-1, x, y);
  auto shim_chan = shim_channel_data(shim_desc, id-1, x, y);

  tensor_t<uint32_t,2> *tt = (tensor_t<uint32_t,2> *)t;

  // printf("Do queue transfer %p with id %ld of length %ld on behalf of x=%ld, y=%ld shim col %ld channel %ld, offset %ld,%ld, stride %ld, elem %ld\n",
  //         tt->data, id, length, x, y, shim_col, shim_chan, offset_y, offset_x, stride, elem_per_stride);

  uint32_t *bounce_buffer = _air_host_bram_ptr;
  bool isMM2S = shim_chan >= 2;

  if (isMM2S) {
    shim_chan = shim_chan - 2;
    uint32_t *data_ptr = tt->data + (offset_y * tt->shape[1] + offset_x);
    uint32_t *bounce_ptr = bounce_buffer;
    for (int n=0; n<length; n+=elem_per_stride) {
      // This is the input, so we need to take what is in t and put it into the BRAM
      memcpy((void*)bounce_ptr, data_ptr, elem_per_stride*sizeof(uint32_t));
      data_ptr += stride;
      bounce_ptr += elem_per_stride;
    }
  }

  uint64_t wr_idx = queue_add_write_index(_air_host_active_herd.q, 1);
  uint64_t packet_id = wr_idx % _air_host_active_herd.q->size;

  dispatch_packet_t *pkt = (dispatch_packet_t*)(_air_host_active_herd.q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt, /*herd_id=*/0, shim_col, /*direction=*/isMM2S, shim_chan, /*burst_len=*/4, /*memory_space=*/2,
                       _air_host_bram_paddr, length*sizeof(uint32_t), 1, 0, 1, 0, 1, 0);
  //if (s) {
  //  // TODO: don't block here
  //  air_queue_dispatch_and_wait(_air_host_active_herd.q, wr_idx, pkt);
  //  uint64_t signal_offset = offsetof(dispatch_packet_t, completion_signal);
  //  s->handle = queue_paddr_from_index(_air_host_active_herd.q,
  //                                     (packet_id) * sizeof(dispatch_packet_t) +
  //                                         signal_offset);
  //} else {
    air_queue_dispatch_and_wait(_air_host_active_herd.q, wr_idx, pkt);
  //}

  if (!isMM2S) {
    uint32_t *data_ptr = tt->data + (offset_y * tt->shape[1] + offset_x);
    uint32_t *bounce_ptr = bounce_buffer;
    for (int n=0; n<length; n+=elem_per_stride) {
      // This is the input, so we need to take what is in t and put it into the BRAM
      memcpy(data_ptr, (void*)bounce_ptr, elem_per_stride*sizeof(uint32_t));
      data_ptr += stride;
      bounce_ptr += elem_per_stride;
    }
  }
}

void air_shim_memcpy_queue_impl(signal_t *s, uint32_t id, uint64_t x,
                                uint64_t y, void *t, uint64_t offset,
                                uint64_t length) {
  assert(_air_host_active_herd.herd_desc && "cannot shim memcpy without active herd");
  assert(_air_host_active_herd.q && "cannot shim memcpy using a queue without active queue");

  auto shim_desc = _air_host_active_herd.herd_desc->shim_desc;
  auto shim_col = shim_location_data(shim_desc, id-1, x, y);
  auto shim_chan = shim_channel_data(shim_desc, id-1, x, y);

  tensor_t<uint32_t,1> *tt = (tensor_t<uint32_t,1> *)t;
  uint32_t *bounce_buffer = _air_host_bram_ptr;
  bool isMM2S = shim_chan >= 2;

  if (isMM2S) {
    shim_chan = shim_chan - 2;
    uint32_t *data_ptr = tt->data + offset;
    // This is the input, so we need to take what is in t and put it into the BRAM
    memcpy((void*)bounce_buffer, data_ptr, length*sizeof(uint32_t));
  }

  uint64_t wr_idx = queue_add_write_index(_air_host_active_herd.q, 1);
  uint64_t packet_id = wr_idx % _air_host_active_herd.q->size;

  dispatch_packet_t *pkt = (dispatch_packet_t*)(_air_host_active_herd.q->base_address_vaddr) + packet_id;
  air_packet_nd_memcpy(pkt, /*herd_id=*/0, shim_col, /*direction=*/isMM2S, shim_chan, /*burst_len=*/4, /*memory_space=*/2,
                       _air_host_bram_paddr, length*sizeof(uint32_t), 1, 0, 1, 0, 1, 0);
  if (s) {
    // TODO: don't block here
    air_queue_dispatch_and_wait(_air_host_active_herd.q, wr_idx, pkt);
    uint64_t signal_offset = offsetof(dispatch_packet_t, completion_signal);
    s->handle = queue_paddr_from_index(_air_host_active_herd.q,
                                       (packet_id) * sizeof(dispatch_packet_t) +
                                           signal_offset);
  } else {
    air_queue_dispatch_and_wait(_air_host_active_herd.q, wr_idx, pkt);
  }
  if (!isMM2S) {
    uint32_t *data_ptr = tt->data + offset;
    memcpy(data_ptr, (void*)bounce_buffer, length*sizeof(uint32_t));
  }
}

template<typename T0, int R0, typename T1, int R1>
void air_mem_cdma_nd_memcpy_queue_impl(tensor_t<T0, R0>* t0, tensor_t<T1, R1>* t1, uint32_t space0, uint32_t space1, 
                                 uint64_t offset_3, uint64_t offset_2, uint64_t offset_1, uint64_t offset_0,
                                 uint64_t length_4d, uint64_t length_3d, uint64_t length_2d, uint64_t length_1d,
                                 uint64_t stride_4d, uint64_t stride_3d, uint64_t stride_2d) {
  assert(_air_host_active_herd.herd_desc && "cannot cdma memcpy without active herd");
  assert(_air_host_active_herd.q && "cannot cdma memcpy using a queue without active queue");
  assert(sizeof(T0) == sizeof(T1) && "cannot cdma memcpy with mismatched sized types");
  assert(R0 == R1 && "cannot cdma memcpy with mismatched tensor ranks");
  //assert(length_2d<=1 && length_3d<=1 && length_4d<=1 && "ERROR: CDMA memcpy only supports 1D DMAs");

  //printf("Do CDMA transfer dst %p src %p dst space %d, src space %d, offset [%ld,%ld,%ld,%ld], length [%ld,%ld,%ld,%ld], stride [%ld,%ld,%ld]\n",
  //       t0->data, t1->data, space0, space1,
  //       offset_3, offset_2, offset_1, offset_0,
  //       length_4d, length_3d, length_2d, length_1d,
  //       stride_4d, stride_3d, stride_2d);

  uint32_t *bounce_buffer = _air_host_bram_ptr;

  bool isDDR2L2 = space0 < space1;
  tensor_t<T1, R1>* t = (isDDR2L2) ? t1 : t0;
 
  size_t stride = 1;
  size_t offset = 0;
  std::vector<uint64_t> offsets{offset_0, offset_1, offset_2, offset_3};
  for (int i=0; i<R0; i++) {
    offset += offsets[i] * stride * sizeof(T0);
    stride *= t->shape[R0 - i - 1];
  }

  uint64_t length = 0;
  for (uint32_t index_4d=0;index_4d<length_4d;index_4d++)
    for (uint32_t index_3d=0;index_3d<length_3d;index_3d++)
      for (uint32_t index_2d=0;index_2d<length_2d;index_2d++)
        length += length_1d;

  assert(length*sizeof(T0) <= 16*8192 && "error cdma memcpy length out of bounds");

  size_t p = (size_t)t->data + offset;
  uint64_t paddr_4d = p;
  uint64_t paddr_3d = p;
  uint64_t paddr_2d = p;
  uint64_t paddr_1d = p;

  if (isDDR2L2) {
    for (uint32_t index_4d=0;index_4d<length_4d;index_4d++) {
      paddr_2d = paddr_3d;
      for (uint32_t index_3d=0;index_3d<length_3d;index_3d++) {
        paddr_1d = paddr_2d;
        for (uint32_t index_2d=0;index_2d<length_2d;index_2d++) {
          memcpy((size_t*)bounce_buffer, (size_t*)paddr_1d, length_1d*sizeof(T0));
          bounce_buffer += length_1d;
          paddr_1d += stride_2d*sizeof(T0);
        }
        paddr_2d += stride_3d*sizeof(T0);
      }
      paddr_3d += stride_4d*sizeof(T0);
    }
  }

  uint64_t ofs = uint64_t(AIR_VCK190_L2_DMA_BASE);

  uint64_t wr_idx = queue_add_write_index(_air_host_active_herd.q, 1);
  uint64_t packet_id = wr_idx % _air_host_active_herd.q->size;
  dispatch_packet_t *pkt = (dispatch_packet_t*)(_air_host_active_herd.q->base_address_vaddr) + packet_id;

  if (isDDR2L2)
    air_packet_cdma_memcpy(pkt, uint64_t(t0->data) + ofs,
                           _air_host_bram_paddr, sizeof(T0) * length);
  else
    air_packet_cdma_memcpy(pkt, _air_host_bram_paddr,
                           uint64_t(t1->data) + ofs, sizeof(T0) * length);
  air_queue_dispatch_and_wait(_air_host_active_herd.q, wr_idx, pkt);

  if (!isDDR2L2) {
    for (uint32_t index_4d=0;index_4d<length_4d;index_4d++) {
      paddr_2d = paddr_3d;
      for (uint32_t index_3d=0;index_3d<length_3d;index_3d++) {
        paddr_1d = paddr_2d;
        for (uint32_t index_2d=0;index_2d<length_2d;index_2d++) {
          memcpy((size_t*)paddr_1d, (size_t*)bounce_buffer, length_1d*sizeof(T0));
          bounce_buffer += length_1d;
          paddr_1d += stride_2d*sizeof(T0);
        }
        paddr_2d += stride_3d*sizeof(T0);
      }
      paddr_3d += stride_4d*sizeof(T0);
    }
  }
}

template <typename T, int R>
void air_mem_shim_nd_memcpy_queue_impl(signal_t *s, uint32_t id, uint64_t x,
                                       uint64_t y, tensor_t<T, R> *t,
                                       uint32_t space, uint64_t offset_3,
                                       uint64_t offset_2, uint64_t offset_1,
                                       uint64_t offset_0, uint64_t length_4d,
                                       uint64_t length_3d, uint64_t length_2d,
                                       uint64_t length_1d, uint64_t stride_4d,
                                       uint64_t stride_3d, uint64_t stride_2d) {
  assert(_air_host_active_herd.herd_desc && "cannot shim memcpy without active herd");
  assert(_air_host_active_herd.q && "cannot shim memcpy using a queue without active queue");

  auto shim_desc = _air_host_active_herd.herd_desc->shim_desc;
  auto shim_col = shim_location_data(shim_desc, id-1, x, y);
  auto shim_chan = shim_channel_data(shim_desc, id-1, x, y);

  // printf("Do transfer %p with id %d on behalf of x=%ld, y=%ld space %d, col %d, dir %d, chan %d, offset [%ld,%ld,%ld,%ld], length [%ld,%ld,%ld,%ld], stride [%ld,%ld,%ld]\n",
  //       t->data, id, x, y, space, shim_col, shim_chan>=2, (shim_chan>=2) ? shim_chan-2 : shim_chan,
  //       offset_3, offset_2, offset_1, offset_0,
  //       length_4d, length_3d, length_2d, length_1d,
  //       stride_4d, stride_3d, stride_2d);

  bool isMM2S = shim_chan >= 2;

  bool uses_pa = (space == 1); // t->uses_pa;
  if (uses_pa) {
    if (isMM2S) shim_chan = shim_chan - 2;

    size_t stride = 1;
    size_t offset = 0;
    std::vector<uint64_t> offsets{offset_0, offset_1, offset_2, offset_3};
    for (int i=0; i<R; i++) {
      offset += offsets[i] * stride * sizeof(T);
      stride *= t->shape[R - i - 1];
    }

    uint64_t wr_idx = queue_add_write_index(_air_host_active_herd.q, 1);
    uint64_t packet_id = wr_idx % _air_host_active_herd.q->size;

    dispatch_packet_t *pkt = (dispatch_packet_t*)(_air_host_active_herd.q->base_address_vaddr) + packet_id;
    air_packet_nd_memcpy(pkt, /*herd_id=*/0, shim_col, /*direction=*/isMM2S, shim_chan, /*burst_len=*/4, /*memory_space=*/space,
                         (uint64_t)t->data + offset, length_1d*sizeof(T), length_2d, stride_2d*sizeof(T), length_3d, stride_3d*sizeof(T), length_4d, stride_4d*sizeof(T));
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
    for (int i=0; i<R; i++) {
      offset += offsets[i] * stride * sizeof(T);
      stride *= t->shape[R - i - 1];
    }

    uint64_t length = 0;
    for (uint32_t index_4d=0;index_4d<length_4d;index_4d++)
      for (uint32_t index_3d=0;index_3d<length_3d;index_3d++)
        for (uint32_t index_2d=0;index_2d<length_2d;index_2d++)
          length += length_1d;

    size_t p = (size_t)t->data + offset;
    uint64_t paddr_4d = p;
    uint64_t paddr_3d = p;
    uint64_t paddr_2d = p;
    uint64_t paddr_1d = p;

    if (isMM2S) {
      shim_chan = shim_chan - 2;
      for (uint32_t index_4d=0;index_4d<length_4d;index_4d++) {
        paddr_2d = paddr_3d;
        for (uint32_t index_3d=0;index_3d<length_3d;index_3d++) {
          paddr_1d = paddr_2d;
          for (uint32_t index_2d=0;index_2d<length_2d;index_2d++) {
            memcpy((size_t*)bounce_buffer, (size_t*)paddr_1d, length_1d*sizeof(T));
            bounce_buffer += length_1d;
            paddr_1d += stride_2d*sizeof(T);
          }
          paddr_2d += stride_3d*sizeof(T);
        }
        paddr_3d += stride_4d*sizeof(T);
      }
    }

    uint64_t wr_idx = queue_add_write_index(_air_host_active_herd.q, 1);
    uint64_t packet_id = wr_idx % _air_host_active_herd.q->size;

    dispatch_packet_t *pkt = (dispatch_packet_t*)(_air_host_active_herd.q->base_address_vaddr) + packet_id;
    air_packet_nd_memcpy(pkt, /*herd_id=*/0, shim_col, /*direction=*/isMM2S, shim_chan, /*burst_len=*/4, /*memory_space=*/2,
                         _air_host_bram_paddr, length*sizeof(T), 1, 0, 1, 0, 1, 0);
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
      for (uint32_t index_4d=0;index_4d<length_4d;index_4d++) {
        paddr_2d = paddr_3d;
        for (uint32_t index_3d=0;index_3d<length_3d;index_3d++) {
          paddr_1d = paddr_2d;
          for (uint32_t index_2d=0;index_2d<length_2d;index_2d++) {
            memcpy((size_t*)paddr_1d, (size_t*)bounce_buffer, length_1d*sizeof(T));
            bounce_buffer += length_1d;
            paddr_1d += stride_2d*sizeof(T);
          }
          paddr_2d += stride_3d*sizeof(T);
        }
        paddr_3d += stride_4d*sizeof(T);
      }
    }
  }
}

extern "C" void _mlir_ciface_air_shim_memcpy(signal_t *s, uint32_t id,
                                             uint64_t x, uint64_t y, void *t,
                                             uint64_t offset, uint64_t length);

template<typename T, int R>
void air_mem_shim_nd_memcpy_impl(uint32_t id, uint64_t x, uint64_t y, tensor_t<T, R>* t, uint32_t space,
                                uint64_t offset_3, uint64_t offset_2, uint64_t offset_1, uint64_t offset_0,
                                uint64_t length_4d, uint64_t length_3d, uint64_t length_2d, uint64_t length_1d,
                                uint64_t stride_4d, uint64_t stride_3d, uint64_t stride_2d)
{
  // printf("Do transfer %p with id %d on behalf of x=%ld, y=%ld space %d, offset [%ld,%ld,%ld,%ld], length [%ld,%ld,%ld,%ld], stride [%ld,%ld,%ld]\n",
  //        t->data, id, x, y, space,
  //        offset_3, offset_2, offset_1, offset_0,
  //        length_4d, length_3d, length_2d, length_1d,
  //        stride_4d, stride_3d, stride_2d);

  size_t stride = 1;
  size_t offset = 0;
  std::vector<uint64_t> offsets{offset_0, offset_1, offset_2, offset_3};
  for (int i=0; i<R; i++) {
    offset += offsets[i] * stride;
    stride *= t->shape[R - i - 1];
  }
  // printf("offset %d stride %d\n",offset, stride);
  size_t p = (size_t)t->data + offset;
  size_t paddr_3d = p;
  size_t paddr_2d = p;
  size_t paddr_1d = p;
  uint64_t index_3d=0;
  uint64_t index_2d=0;
  for (uint64_t index_4d=0;index_4d<length_4d;index_4d++) {
    for (;index_3d<length_3d;index_3d++) {
      for (;index_2d<length_2d;index_2d++) {
        _mlir_ciface_air_shim_memcpy(nullptr, id, x, y, t,
                                     paddr_1d - (size_t)t->data, length_1d);
        paddr_1d += stride_2d;
      }
      index_2d = 0;
      paddr_2d += stride_3d;
      if (index_3d+1<length_3d) paddr_1d = paddr_2d;
      else paddr_1d = paddr_3d + stride_4d;
    }
    index_3d = 0;
    paddr_3d += stride_4d;
    paddr_2d = paddr_3d;
  }
}

extern "C"  {

void _mlir_ciface_air_shim_memcpy(signal_t *s, uint32_t id, uint64_t x,
                                  uint64_t y, void *t, uint64_t offset,
                                  uint64_t length) {
  assert(_air_host_active_herd.herd_desc && "cannot shim memcpy without active herd");
  if (_air_host_active_herd.q)
    air_shim_memcpy_queue_impl(s, id, x, y, t, offset, length);
  else
    printf("WARNING: no queue provided. memcpy will not be performed.\n");
}

void _mlir_ciface_air_shim_memcpy2d(signal_t *s, uint32_t id, uint64_t x,
                                    uint64_t y, void *t, uint64_t offset_y,
                                    uint64_t offset_x, uint64_t length,
                                    uint64_t stride, uint64_t elem_per_stride) {
  if (_air_host_active_herd.q)
    air_shim_memcpy2d_queue_impl(id, x, y, t, offset_y, offset_x, length, stride, elem_per_stride);
  else
    printf("WARNING: no queue provided. 2d memcpy will not be performed.\n");
}

void _mlir_ciface_air_shim_memcpy4d(signal_t *s, uint32_t id, uint64_t x,
                                    uint64_t y, void *t, uint64_t offset_3,
                                    uint64_t offset_2, uint64_t offset_1,
                                    uint64_t offset_0, uint64_t length,
                                    uint64_t stride, uint64_t elem_per_stride) {
  if (_air_host_active_herd.q)
    air_shim_memcpy4d_queue_impl(id, x, y, t, offset_3, offset_2, offset_1, offset_0, length, stride, elem_per_stride); 
  else
    printf("WARNING: no queue provided. 4d memcpy will not be performed.\n");
}

#define mlir_air_dma_nd_memcpy(mangle, rank, space, type)                      \
  void _mlir_ciface_air_dma_nd_memcpy_##mangle(                                \
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
      printf("WARNING: no queue provided. ND memcpy will not be performed.\n");\
    }                                                                          \
  }

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

#define mlir_air_nd_memcpy(mangle, rank0, space0, type0, rank1, space1, type1) \
void _mlir_ciface_air_nd_memcpy_##mangle( \
  void* t0, void* t1, \
  uint64_t offset_3, uint64_t offset_2, uint64_t offset_1, uint64_t offset_0, \
  uint64_t length_3, uint64_t length_2, uint64_t length_1, uint64_t length_0, \
  uint64_t stride_2, uint64_t stride_1, uint64_t stride_0) \
{ \
  tensor_t<type0, rank0> *tt0 = (tensor_t<type0, rank0>*)t0; \
  tensor_t<type1, rank1> *tt1 = (tensor_t<type1, rank1>*)t1; \
  if (_air_host_active_herd.q) { \
    air_mem_cdma_nd_memcpy_queue_impl(tt0, tt1, space0, space1, \
                                     offset_3, offset_2, offset_1, offset_0, \
                                     length_3, length_2, length_1, length_0, \
                                     stride_2, stride_1, stride_0); \
  } else {                                                                   \
    printf("WARNING: no queue provided. ND memcpy will not be performed.\n");\
  }                                                                          \
}

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
} // namespace
